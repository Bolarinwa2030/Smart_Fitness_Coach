import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle

model = tf.keras.models.load_model("../models/pose_classifier_model.h5")

with open("../models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])

        landmarks = np.array(landmarks).reshape(1, -1)

        # Predict class
        prediction = model.predict(landmarks)
        predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        confidence = np.max(prediction)

        # Show prediction on video
        cv2.putText(frame, f"{predicted_class} ({confidence:.2f})",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Smart Fitness Coach", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
