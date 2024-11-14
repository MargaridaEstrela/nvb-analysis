clc; clear; close all;

% Load data
data = readtable('csv/pipeline.csv');

frame_width = 1920;
frame_height = 1080;
data.x = data.x * frame_width;
data.y = data.y * frame_height;

keypoint_names = {
    'Nose', 'Left Eye Inner', 'Left Eye', 'Left Eye Outer', 'Right Eye Inner', ...
    'Right Eye', 'Right Eye Outer', 'Left Ear', 'Right Ear', 'Mouth Left', ...
    'Mouth Right', 'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow', ...
    'Left Wrist', 'Right Wrist', 'Left Pinky', 'Right Pinky', 'Left Index', ...
    'Right Index', 'Left Thumb', 'Right Thumb', 'Left Hip', 'Right Hip', ...
    'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle', 'Left Heel', ...
    'Right Heel', 'Left Foot Index', 'Right Foot Index'
};

max_keypoint_id = 23;

% Call each plotting function
plot_keypoint_trajectories(data, frame_width, frame_height, keypoint_names, max_keypoint_id);
plot_x_coordinate_trend(data, frame_width, keypoint_names, max_keypoint_id);
plot_y_coordinate_trend(data, frame_height, keypoint_names, max_keypoint_id);
plot_depth_trend(data);