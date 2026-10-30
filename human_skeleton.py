import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

print("All libraries imported successfully!")

# Load YOLO Pose Model
model = YOLO("yolo26n-pose.pt")

print("YOLO Pose Model Loaded Successfully!")

# Load input video
video_path = "video dance.mp4"

cap = cv2.VideoCapture(video_path)

if cap.isOpened():
    print("Video loaded successfully!")
    print("Video width:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print("Video height:", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("Total frames:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
else:
    print("Error: Video not found!")

cap.release()

# Perform Human Pose Estimation
results = model.predict(
    source=video_path,
    conf=0.25,
    save=True,
    show=False,
    project="pose_output",
    name="human_skeleton",
    exist_ok=True
)

print("Human Pose Estimation Completed Successfully!")

# Black Background + White Skeleton
import numpy as np

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    "white_skeleton.mp4",
    fourcc,
    fps,
    (width, height)
)

# YOLO skeleton connections
skeleton = [
    (5, 6), (5, 11), (6, 12), (11, 12),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(frame, conf=0.25, verbose=False)

    black_frame = np.zeros((height, width, 3), dtype=np.uint8)

    for result in results:
        if result.keypoints is None:
            continue

        keypoints = result.keypoints.xy.cpu().numpy()

        for person in keypoints:
            for a, b in skeleton:
                x1, y1 = person[a]
                x2, y2 = person[b]

                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                    cv2.line(
                        black_frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (255, 255, 255),
                        2
                    )

            for x, y in person:
                if x > 0 and y > 0:
                    cv2.circle(
                        black_frame,
                        (int(x), int(y)),
                        4,
                        (255, 255, 255),
                        -1
                    )

    out.write(black_frame)

cap.release()
out.release()

print("Black Background White Skeleton Video Created!")