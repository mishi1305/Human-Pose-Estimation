# Human Pose Estimation and Human Skeleton Detection

## About the Project

This project implements Human Pose Estimation and Human Skeleton Detection using YOLO26 Pose Model and OpenCV.

It detects human body keypoints from a dance video, including shoulders, elbows, wrists, hips, knees, and ankles, and connects them to visualize the human skeleton.

The project generates two types of output:
- Human pose estimation with skeleton overlay on the original video.
- Black background with white skeleton points and connecting lines.

## Technologies Used

- Python
- YOLO26 Pose Model (Ultralytics)
- OpenCV
- NumPy
- Matplotlib

## Project Files

| File | Description |
|---|---|
| `human_skeleton.py` | Main Python source code |
| `video dance.mp4` | Original dance video |
| `video dance.avi` | YOLO skeleton detection output |
| `white_skeleton.mp4` | Black background with white skeleton |

## Installation

Install the required Python libraries:

```bash
pip install ultralytics opencv-python matplotlib numpy
```

## How to Run

Run the following command in the VS Code terminal:

```bash
python human_skeleton.py
```

The YOLO model will detect human poses and generate the output videos.

## Output Videos

### 1. Original Dance Video

[Watch Original Video](https://drive.google.com/file/d/15z2GE9gnAOv_NLRKfyz7Odi4y0XWwpl-/view?usp=drive_link)

### 2. Human Pose Estimation Output

[Watch YOLO Skeleton Output (.avi)](https://drive.google.com/file/d/114i4QsAkTCFsMY-TmPV_65bCw16sHaDx/view?usp=drive_link)

### 3. Black Background + White Skeleton

[Watch Black and White Skeleton Output](https://drive.google.com/file/d/17y8kwu26FrSinWp6KOGxGLorGFOQJ-_T/view?usp=drive_link)

## Model Used

YOLO26 Nano Pose Model (`yolo26n-pose.pt`)

## Author

**Mishika Singh Nayal**
