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


PERFECT_THRESHOLD = 0.90  

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

        prediction = model.predict(landmarks)
        predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        confidence = np.max(prediction)

        if confidence >= PERFECT_THRESHOLD:
            feedback_text = "PERFECT"
            color = (0, 255, 0)  
        else:
            feedback_text = f"{predicted_class} ({confidence:.2f})"
            color = (0, 255, 255)  

        cv2.putText(frame, feedback_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.imshow("Smart Fitness Coach", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
