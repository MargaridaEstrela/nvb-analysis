# 3D Pose Reconstruction

This repository provides a full pipeline for reconstructing 3D human poses from stereo videos using MediaPipe keypoints and MATLAB triangulation.

---

## 📸 Camera Calibration

1. **Extract frames**  
   Use [`extract_frames.m`](extract_frames.m) to extract all frames from your calibration video

2. **Stereo Calibration**  
   Use MATLAB’s stereoCameraCalibrator app to perform stereo calibration directly between the two cameras. Save the result as stereoParams.mat.

---

## 🧍🏼Pose Reconstruction Pipeline

After calibration, run the main script:

```matlab
pose_reconstruction.m
```

This script performs the full 3D reconstruction pipeline:

-	Loads stereo calibration parameters.
-	Computes projection matrices for both cameras.
-	Loads 2D keypoints detected by MediaPipe from both views.
-	Reassigns pose IDs per frame to ensure consistent tracking (left = ID 0, right = ID 1).
-	Triangulates 3D keypoints for each pose.
-	Rotates the point cloud to align the coordinate system (Y = vertical).
-	Exports 3D pose data to CSV.
-	Visualizes the 3D skeletons over time.

---


## 📦 Output

- Two CSV files:  
  - `3D_pose_reconstruction_0.csv` – Pose 0  
  - `3D_pose_reconstruction_1.csv` – Pose 1  
  
- Each file has the format:  
  
  ``` csv
  frame, pose, keypoint, x, y, z
  ```

---

## 📊 Visualization

The script visualizes the 3D skeletons using MATLAB’s 3D plotting tools. Each frame is animated and displays the skeletons for both individuals:

- Pose 0 → Red
- Pose 1 → Blue

---

## ✅ Requirements

- MATLAB with the Computer Vision Toolbox
- Stereo calibration file: stereoParams.mat
- 2D keypoint CSV files from MediaPipe for both cameras

---

## Acknowledgments

This pipeline uses MATLAB’s Camera Calibration Toolbox for stereo calibration, and MediaPipe for 2D keypoint detection.
