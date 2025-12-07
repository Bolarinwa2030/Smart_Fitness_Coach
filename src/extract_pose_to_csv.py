import cv2
import mediapipe as mp
import pandas as pd
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
drawing = mp.solutions.drawing_utils


label = "stretching"
video_path = "../dataset/videos/Leg.mp4"


output_file = f"../dataset/{label}_landmarks.csv"
columns = [f"{joint}_{axis}" for joint in range(33) for axis in ["x", "y", "z", "visibility"]] + ["label"]
data = []

cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        row = [lm.x for lm in landmarks] + \
              [lm.y for lm in landmarks] + \
              [lm.z for lm in landmarks] + \
              [lm.visibility for lm in landmarks]
        row.append(label)
        data.append(row)

        drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    frame_count += 1
    cv2.imshow("Pose Extraction", frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data, columns=columns)
df.to_csv(output_file, index=False)
print(f"[INFO] Saved {len(df)} frames to {output_file}")
