import cv2
import os
import time
import pickle
import sqlite3
from datetime import date, datetime
from collections import Counter, defaultdict
from ultralytics import YOLO

# ---------------- CAMERA ----------------
camera = cv2.VideoCapture(0)

MODE = "idle"          # idle | register | attendance
STUDENT_NAME = ""
COUNT = 0
MESSAGE = "Waiting..."

# 🔧 FIX: prediction history PER FACE
recent_predictions = defaultdict(list)

recognizer = None
label_map = {}

if os.path.exists("trainer.yml") and os.path.exists("labels.pickle"):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")
    with open("labels.pickle", "rb") as f:
        label_map = pickle.load(f)

# YOLO (person / face detection)
yolo_model = YOLO("yolov8n.pt")


# ---------------- DATABASE ----------------
def mark_present_once(name):
    today = date.today().isoformat()
    now_time = datetime.now().strftime("%H:%M:%S")

    conn = sqlite3.connect("database/attendance.db")
    cur = conn.cursor()

    cur.execute("""
        SELECT 1 FROM attendance
        WHERE name=? AND date=?
    """, (name, today))

    if cur.fetchone() is None:
        cur.execute("""
            INSERT INTO attendance (name, date, time, status)
            VALUES (?, ?, ?, ?)
        """, (name, today, now_time, "PRESENT"))
        conn.commit()

    conn.close()


def mark_absent_remaining():
    today = date.today().isoformat()
    now_time = datetime.now().strftime("%H:%M:%S")

    registered = [
        d for d in os.listdir("dataset")
        if os.path.isdir(os.path.join("dataset", d))
    ]

    conn = sqlite3.connect("database/attendance.db")
    cur = conn.cursor()

    for person in registered:
        cur.execute("""
            SELECT 1 FROM attendance
            WHERE name=? AND date=?
        """, (person, today))

        if cur.fetchone() is None:
            cur.execute("""
                INSERT INTO attendance (name, date, time, status)
                VALUES (?, ?, ?, ?)
            """, (person, today, now_time, "ABSENT"))

    conn.commit()
    conn.close()


# ---------------- CAMERA STREAM ----------------
def gen_frames():
    global COUNT, MESSAGE, recent_predictions

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = yolo_model(frame, conf=0.4, imgsz=640)

        active_faces = set()

        if len(results[0].boxes) == 0:
            MESSAGE = "No person detected"
            recent_predictions.clear()

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Ignore very small detections
            if (x2 - x1) < 80 or (y2 - y1) < 80:
                continue

            face = gray[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # 🔑 FACE ID (location-based tracking)
            face_id = f"{x1//50}_{y1//50}"
            active_faces.add(face_id)

            display_name = ""

            # ---------- REGISTER MODE ----------
            if MODE == "register" and COUNT < 10:
                os.makedirs(f"dataset/{STUDENT_NAME}", exist_ok=True)
                COUNT += 1
                cv2.imwrite(f"dataset/{STUDENT_NAME}/{COUNT}.jpg", face)
                time.sleep(0.8)

                if COUNT == 3:
                    MESSAGE = f"Registered: {STUDENT_NAME}"

            # ---------- ATTENDANCE MODE ----------
            if MODE == "attendance" and recognizer:
                label, conf = recognizer.predict(face)

                if conf > 30 and label in label_map:
                    recent_predictions[face_id].append(label_map[label])

                    if len(recent_predictions[face_id]) > 10:
                        recent_predictions[face_id].pop(0)

                    common = Counter(
                        recent_predictions[face_id]
                    ).most_common(1)

                    if common and common[0][1] >= 6:
                        display_name = common[0][0]
                        mark_present_once(display_name)
                    else:
                        display_name = "Verifying..."
                else:
                    display_name = "Unknown"

            # ---------- DRAW ----------
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if display_name:
                cv2.putText(
                    frame,
                    display_name,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

        # 🔧 CLEANUP disappeared faces
        for fid in list(recent_predictions.keys()):
            if fid not in active_faces:
                del recent_predictions[fid]

        cv2.putText(
            frame,
            MESSAGE,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )