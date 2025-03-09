# Depth - COLMAP

## Overview
This script uses a COLMAP-generated point cloud to project 3D depth information onto a given image. The method enables depth estimation by leveraging camera intrinsics, extrinsics, and 3D reconstruction from COLMAP.


## Input
- COLMAP sparse reconstruction
- Target image
- Point cloud extracted from the COLMAP`reconstruction file used to extract frames for visualization.

## Output
- Depth heatmap overlay visualizing projected 3D depths on the image

## Methods
1. Load COLMAP Reconstruction
   - Reads COLMAP's sparse reconstruction to extract 3D points and camera parameters.

2. Extract Camera Intrinsics & Pose
    - Intrinsic parameters $K$ are derived based on the COLMAP camera model.
    - Camera pose (extrinsics $\Tau$) is extracted for projection.

3. Project 3D Points to Image Space
    - The 3D point cloud is projected into 2D pixel coordinates using:

    $$
    \mathbf{x}_{\text{proj}} = K \cdot (T \cdot X)
    $$

    where:

   - $X$ is the homogeneous 3D point,

   - $K$ is the intrinsic camera matrix,

   - $\Tau$ is the extrinsic transformation matrix.

4. Feature Selection for Depth Scaling
    - The user select two points in the heatmap to define a known real-world distance.

    - A scale factor is computed to convert COLMAP depth units into real-world meters.

5. Generate and Overlay Depth Heatmap

    - Projects depth values onto the image and visualizes them using a color heatmap.


## Acknowledgments
This script leverages **COLMAP 3D reconstruction and camera parameters** to estimate depth, aligning with **structure-from-motion (SfM) techniques** for accurate depth estimation and visualization.
