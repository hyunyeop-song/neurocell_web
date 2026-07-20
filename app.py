import cv2
import mediapipe as mp
import time
import os
import json
import csv
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify

app = Flask(__name__)

PATIENT_DB = "patients.json"
LOG_FILE = "patient_event_log.csv"

# --- [초기 인프라 파일 세팅] ---
if not os.path.exists(PATIENT_DB):
    with open(PATIENT_DB, "w", encoding="utf-8") as f:
        json.dump([], f)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Patient_ID", "Hour", "Event_Type"])

# --- [데이터베이스 헬퍼 함수] ---
def load_patients():
    with open(PATIENT_DB, "r", encoding="utf-8") as f:
        return json.load(f)

def save_patients(patients):
    with open(PATIENT_DB, "w", encoding="utf-8") as f:
        json.dump(patients, f, ensure_ascii=False, indent=4)

def log_patient_event(patient_id, event_type):
    now = datetime.now()
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([now.strftime("%Y-%m-%d %H:%M:%S"), patient_id, now.hour, event_type])

# --- [AI 시계열 예측 알고리즘] ---
def predict_probabilities(patient_id):
    try:
        with open(LOG_FILE, mode="r", encoding="utf-8") as f:
            rows = list(csv.reader(f))[1:]
        p_rows = [r for r in rows if r[1] == str(patient_id)]
        if len(p_rows) < 3: return {"Pain": 15.0, "Position": 20.0}
        
        current_hour = datetime.now().hour
        target_hours = [current_hour, (current_hour + 1) % 24]
        pain_c = sum(1 for r in p_rows if int(r[2]) in target_hours and r[3] == "2nd_Pain")
        pos_c = sum(1 for r in p_rows if int(r[2]) in target_hours and r[3] == "3rd_Position")
        total = pain_c + pos_c
        if total == 0: return {"Pain": 15.0, "Position": 20.0}
        return {"Pain": round((pain_c/total)*100, 1), "Position": round((pos_c/total)*100, 1)}
    except:
        return {"Pain": 15.0, "Position": 20.0}

# --- [핵심 AI 영상 처리 비동기 제네레이터] ---
def generate_frames(patient_id):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    
    both_was_closed, left_hold, right_hold, both_closed_start = False, None, None, None
    both_blinks = []
    cooldown = 0

    while True:
        success, frame = cap.read()
        if not success: break
        
        current_time = time.time()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        both_blinks = [t for t in both_blinks if current_time - t <= 7.0]
        status_msg, status_color = "Normal Monitoring", (0, 255, 0)

        if results.multi_face_landmarks and current_time > cooldown:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # 3D Head-Pose & 가림 필터 수식 계산
                left_x, right_x, nose_x = landmarks[33].x, landmarks[263].x, landmarks[1].x
                eye_dist = right_x - left_x
                rel_nose = (nose_x - left_x) / eye_dist if eye_dist > 0 else 0.5
                is_side = (rel_nose < 0.35) or (rel_nose > 0.65)

                f_width = abs(landmarks[454].x - landmarks[234].x)
                r_w, l_w = abs(landmarks[133].x - landmarks[33].x), abs(landmarks[263].x - landmarks[362].x)
                is_asym = (r_w/l_w < 0.5 or r_w/l_w > 2.0) if l_w > 0 and r_w > 0 else False

                if is_side or is_asym or (r_w/f_width < 0.08) or (l_w/f_width < 0.08):
                    left_hold = right_hold = both_closed_start = None
                    both_was_closed = False
                    status_msg, status_color = "Side Face / Occluded - Paused", (0, 165, 255)
                    continue

                # 안구 개폐 분석
                r_closed = (landmarks[145].y - landmarks[159].y) < 0.016
                l_closed = (landmarks[374].y - landmarks[386].y) < 0.016

                if r_closed and l_closed:
                    left_hold = right_hold = None
                    if not both_was_closed:
                        both_closed_start = current_time
                        both_was_closed = True
                    if both_closed_start and (current_time - both_closed_start >= 3.0):
                        both_blinks.clear()
                        status_msg, status_color = "Sleeping (Ignored)", (0, 0, 255)
                elif r_closed and not l_closed:
                    both_was_closed = both_closed_start = left_hold = None
                    if right_hold is None: right_hold = current_time
                    if current_time - right_hold >= 3.0:
                        log_patient_event(patient_id, "2nd_Pain")
                        right_hold = None
                        cooldown = current_time + 3.0
                    status_msg = f"Right Holding: {current_time - (right_hold or current_time):.1f}s"
                elif l_closed and not r_closed:
                    both_was_closed = both_closed_start = right_hold = None
                    if left_hold is None: left_hold = current_time
                    if current_time - left_hold >= 3.0:
                        log_patient_event(patient_id, "3rd_Position")
                        left_hold = None
                        cooldown = current_time + 3.0
                    status_msg = f"Left Holding: {current_time - (left_hold or current_time):.1f}s"
                else:
                    if both_was_closed and both_closed_start and (current_time - both_closed_start < 3.0):
                        both_blinks.append(current_time)
                    both_was_closed = both_closed_start = left_hold = right_hold = None

            if len(both_blinks) >= 7:
                log_patient_event(patient_id, "1st_Suction")
                both_blinks.clear()
                cooldown = current_time + 4.0

        cv2.putText(frame, status_msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

# --- [웹 라우팅 로직] ---
@app.route('/')
def index():
    return render_template('index.html', patients=load_patients())

@app.route('/add_patient', methods=['POST'])
def add_patient():
    patients = load_patients()
    new_patient = {
        "id": int(time.time()),
        "room": request.form['room'],
        "name": request.form['name'],
        "age": request.form['age'],
        "diagnosis": request.form['diagnosis']
    }
    patients.append(new_patient)
    save_patients(patients)
    return redirect(url_for('index'))

# 🚨 [추가된 코드] 환자 삭제(퇴원) 라우트
@app.route('/delete_patient/<int:patient_id>')
def delete_patient(patient_id):
    patients = load_patients()
    # 입력받은 ID와 일치하지 않는 환자들만 남겨서 필터링 (즉, 일치하는 환자 제거)
    filtered_patients = [p for p in patients if p['id'] != patient_id]
    save_patients(filtered_patients)
    return redirect(url_for('index'))

@app.route('/monitor/<int:patient_id>')
def monitor(patient_id):
    patient = next((p for p in load_patients() if p['id'] == patient_id), None)
    if not patient: return "환자를 찾을 수 없습니다.", 404
    return render_template('monitor.html', patient=patient)

@app.route('/video_feed/<int:patient_id>')
def video_feed(patient_id):
    return Response(generate_frames(patient_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_predictions/<int:patient_id>')
def get_predictions(patient_id):
    return jsonify(predict_probabilities(patient_id))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)