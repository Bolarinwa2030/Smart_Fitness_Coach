import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
from tkinter import Tk, filedialog, Button, Label

model = tf.keras.models.load_model("../models/pose_classifier_model.h5")
label_encoder = joblib.load("../models/label_encoder.pkl")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def select_video():
    video_path = filedialog.askopenfilename(
        title="Select Workout Video",
        filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")]
    )
    if video_path:
        process_video(video_path)
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])

            landmarks_array = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(landmarks_array, verbose=0)
            class_id = np.argmax(prediction)
            class_name = label_encoder.inverse_transform([class_id])[0]
            confidence = np.max(prediction) * 100

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.putText(frame, f"{class_name} ({confidence:.1f}%)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Smart Fitness Coach - Video Analysis", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
root = Tk()
root.title("Smart Fitness Coach - Video Test")

label = Label(root, text="Select a workout video to test classification")
label.pack(pady=10)

btn = Button(root, text="Select Video", command=select_video)
btn.pack(pady=20)

root.mainloop()
