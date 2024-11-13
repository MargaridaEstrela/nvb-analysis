close all;
clear;

% Load the CSV files into MATLAB
data1 = readtable('csv/processed.csv');
data2 = readtable('csv/openpose_formatted.csv');

frame_width = 1920;
frame_height = 1080;
data1.x = data1.x * frame_width;
data1.y = data1.y * frame_height;
data1.y = frame_height - data1.y; % invert Y

% Extract unique persons and keypoints
unique_persons = unique(data1.pose_id);
unique_keypoints = unique(data1.landmark_id);

% Loop through each person and create subplots for x and y comparisons
for i = 1:length(unique_persons)
    person_id = unique_persons(i);
    
    for k = 1:length(unique_keypoints)
        keypoint = unique_keypoints(k);
        fig = figure;
        
        % Extract data for the current person and keypoint from both datasets
        data1_person = data1(data1.pose_id == person_id & data1.landmark_id == keypoint, :);
        data2_person = data2(data2.pose_id == person_id & data2.landmark_id == keypoint, :);
        
        % Subplot for x-coordinate over frames
        subplot(2, 1, 1);
        hold on;
        plot(data1_person.frame, data1_person.x, '-', 'DisplayName', 'MediaPipe');
        plot(data2_person.frame, data2_person.x, '-', 'DisplayName', 'OpenPose');
        title(['X Coordinate Comparison for Keypoint ', num2str(keypoint), ' - Person ', num2str(person_id)]);
        xlabel('Frame');
        ylabel('X');
        axis tight;
        legend;
        grid on;
        hold off;
        
        % Subplot for y-coordinate over frames
        subplot(2, 1, 2);
        hold on;
        plot(data1_person.frame, data1_person.y, '-', 'DisplayName', 'MediaPipe');
        plot(data2_person.frame, data2_person.y, '-', 'DisplayName', 'OpenPose');
        title(['Y Coordinate Comparison for Keypoint ', num2str(keypoint), ' - Person ', num2str(person_id)]);
        xlabel('Frame');
        ylabel('Y');
        axis tight;
        legend;
        grid on;
        hold off;

        saveas(gcf, ['figures/openpose_mediapipe/person_', num2str(person_id), '_keypoint_', num2str(keypoint), '.png']);

        waitforbuttonpress;
        close(fig);
    end
end
