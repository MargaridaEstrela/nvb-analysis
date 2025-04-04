# 🏃🏽‍♀️‍➡️ Pose Estimation

This repository contains the complete pipeline for human pose estimation.

## 🔍 Extracting Pose Landmarks
Use the [`pose_landmark_extractor.py`](../scripts/pose_landmark_extractor.py) script to extract 2D pose landmarks from a video using [MediaPipe](https://developers.google.com/mediapipe). This is a crucial step if you plan to perform **3D pose reconstruction** using footage from **two synchronized cameras**, as the built-in 3D estimation from MediaPipe is not sufficiently accurate for our purposes.

The script can extract pose landmarks for up to 2 persons in the video. You can change the number of persons to extract by modifying the `num_poses` parameter in the script.

### Usage

```bash
python pose_landmark_extractor.py <video_path>
```

- <*video_path*>: Path to the input video file.
- <*output_csv_path*>: Path where the extracted landmarks CSV will be saved.

> 💡 **TIP:** Check out the [`pose_reconstruction.m`](../scripts/pose_reconstruction.m) script (MATLAB) for triangulating 3D poses from the 2D landmarks. It relies on camera calibration and stereo geometry. For more details, refer to the [`3D_pose_reconstruction.md`](../docs/3D_pose_reconstruction.md) file.

## ⚙️ Pose Tracking Correction

The `pose_tracker.py` script is used to correct the pose tracking by assigning consistent IDs to the landmarks and fix possible tracking errors.

### Usage

```bash
python pose_tracker.py <pose_landmarks_csv_path>
```

- <*pose_landmarks_csv_path*>: Path to the pose landmarks CSV file extracted from the [`pose_landmark_extractor.py`](../scripts/pose_landmark_extractor.py) script.

## 🧍‍♂️📹 Smoothing and Visualization

Once you obtain the 3D pose estimations, you can use the [`pose_estimation.py`](../scripts/pose_estimation.py) script to smooth the trajectory movements and generate a video with the reconstructed skeletons overlaid.

### Usage

```bash
python pose_estimation.py <3d_pose_csv_path>
```

- <*3d_pose_csv_path*>: Path to the 3D pose CSV file extracted from the Matlab triangulation script or just using the 3D pose estimation from MediaPipe.
