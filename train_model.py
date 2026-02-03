import cv2
import os
import numpy as np
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE_DIR, "dataset")

TRAINER_PATH = os.path.join(BASE_DIR, "trainer.yml")
LABELS_PATH = os.path.join(BASE_DIR, "labels.pickle")

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}
label_id = 0

for name in sorted(os.listdir(DATASET)):
    person_dir = os.path.join(DATASET, name)
    if not os.path.isdir(person_dir):
        continue

    label_map[label_id] = name

    for img_file in os.listdir(person_dir):
        if not img_file.lower().endswith(".jpg"):
            continue

        img_path = os.path.join(person_dir, img_file)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        # 🔧 CRITICAL FIX: resize all faces
        image = cv2.resize(image, (200, 200))

        faces.append(image)
        labels.append(label_id)

    label_id += 1

if len(faces) == 0:
    print("❌ No face images found for training")
    exit()

recognizer.train(faces, np.array(labels))
recognizer.save(TRAINER_PATH)

with open(LABELS_PATH, "wb") as f:
    pickle.dump(label_map, f)

print("✅ Training completed successfully")
print("Saved:", TRAINER_PATH)
print("Saved:", LABELS_PATH)
print("Labels:", label_map)
