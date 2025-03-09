# Gaze Analysis

## Overview
This script processes gaze data from OpenFace to determine where participants are looking during a multi-party interaction between two persons and a robot.
It assigns gaze directions based on eye landmarks and generates visualizations.

## Input
- `openface.csv`: Gaze data from OpenFace.
- `video.mp4`: Video file used to extract frames for visualization.

## Output
- `gaze_histogram.png`: Histogram of gaze classifications, showing distribution across different categories ("Looking at the Robot", "Looking at other Person", "Looking Elsewhere").
- `gaze_directions.png`: Overlay visualization of gaze directions on the first video frame.

## Steps to Run
1. Load the dataset and video.
2. Manually mark the robot's position.
3. Process gaze data using eye landmarks.
4. Classify gaze behavior using ray-circle intersection.
5. Generate visualizations.

## Fixing Face IDs in OpenFace Output
The `openface_fix_ids.py` script is available to automatically correct misassigned face IDs in the OpenFace output, ensuring consistent and accurate participant identification throughout the analysis.

## Methods
This analysis follows a structured approach to estimate gaze direction and classify gaze targets:

1. **Eye Landmark Extraction**: We extract eye landmarks, specifically iris landmarks, from OpenFace csv output to compute the approximate gaze origin for each participant. The landmarks used are:

   - Left Eye: [20, 21, 22, 23, 24, 25, 26, 27]

    - Right Eye: [48, 49, 50, 51, 52, 53, 54, 55]
   
   $$\
   \mathbf{eye\_center} = \frac{1}{2} \left( \mathbf{left\_eye} + \mathbf{right\_eye} \right)
   $$
   
   where `left_eye` and `right_eye` are the mean positions of the respective iris landmarks.

2. **Participant Target Estimation**: Using the estimated **gaze center**, the gaze direction is calculated from OpenFace's gaze angles:
   
   $$\mathbf{gaze\_direction} = \left( \tan(\theta_x), \tan(\theta_y) \right)$$
   
   where $\theta_x$ and $\theta_y$ are the horizontal and vertical gaze angles, respectively.
   
3. **Ray Tracing to Targets**: The gaze direction is projected as a ray from the eye center, and an intersection check with a circular target region is performed. The intersection condition for a sphere of radius $r$ at position $\mathbf{p}$ (target position) is given by:
   
   $$
   \left( \mathbf{o} + t \cdot \mathbf{d} - \mathbf{p} \right)^2 = r^2
   $$
   
   where $\mathbf{o}$ is the eye center, $\mathbf{d}$ is the normalized gaze direction, and $t$ is a scalar parameter.
   
4. **Gaze Classification**: If the ray intersects the robot’s marked position, it is classified as `"Looking at the Robot"`; if it intersects the other participant's estimated gaze center, it is classified as `"Looking at other Person"`; otherwise, it is classified as `"Looking Elsewhere"`.


## Acknowledgments
This script was developed for **analyzing human gaze behavior in a multiparty interaction study**, leveraging OpenFace for gaze tracking and MATLAB for visualization and data processing.
