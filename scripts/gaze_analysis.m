clear; close all; clc;

%% Load Gaze Data
data = readtable('../csv/openface_0808_fixed.csv');

%% Load Video and Select Positions
videoFile = '../../../pilot_studies/data/08.08.24/piloto.mp4';
video = VideoReader(videoFile);
firstFrame = readFrame(video);

% Screen dimensions
frame_width = 1920;
frame_height = 1080;

target_radius = 200; % Adjust as needed

output_path = '../figures';
if ~isfolder(output_path)
    mkdir(output_path);
end

output_path = '../results';
if ~isfolder(output_path)
    mkdir(output_path);
end

% Display the first frame for marking positions
figure;
imshow(firstFrame);
title('Mark the Robot');
hold on;

%% Robot Position 
% Mark the robot position by hand
robot_marked = ginput(1);
robot_position = [robot_marked(1), robot_marked(2)];
close;

%% Filter rows based on confidence
data = data(data.confidence >= 0.8, :);

%% Gaze Positions 
% Initialize empty arrays to store gaze positions
gaze_positions_0 = NaN(height(data), 2); % [gaze_x, gaze_y] for participant 0
gaze_positions_1 = NaN(height(data), 2); % [gaze_x, gaze_y] for participant 1

% Left eye landmarks:
% left_eye = [20, 21, 22, 23, 24, 25, 26, 27];
% right_eye = [48, 49, 50, 51, 52, 53, 54, 55];
left_eye = [8, 11, 14, 17];
right_eye = [36, 39, 42, 45];

% Loop through each row to assign gaze based on face_id
for i = 1:height(data)
    % Compute mean x and y for left eye
    left_eye_x = mean(table2array(data(i, strcat("eye_lmk_X_", string(left_eye)))), 'omitnan');
    left_eye_y = mean(table2array(data(i, strcat("eye_lmk_Y_", string(left_eye)))), 'omitnan');

    % Compute mean x and y for right eye
    right_eye_x = mean(table2array(data(i, strcat("eye_lmk_X_", string(right_eye)))), 'omitnan');
    right_eye_y = mean(table2array(data(i, strcat("eye_lmk_Y_", string(right_eye)))), 'omitnan');

    % Compute overall eye center as the mean of both eyes
    gaze_x = (left_eye_x + right_eye_x) / 2;
    gaze_y = (left_eye_y + right_eye_y) / 2;

    % Convert from OpenFace centered coordinates to image pixel coordinates
    gaze_x = (frame_width / 2) + gaze_x;
    gaze_y = (frame_height / 2) + gaze_y;

    % Store gaze positions in the respective array
    if data.face_id(i) == 0
        gaze_positions_0(i, :) = [gaze_x, gaze_y]; % Store for participant 0
    elseif data.face_id(i) == 1
        gaze_positions_1(i, :) = [gaze_x, gaze_y]; % Store for participant 1
    end
end

%% Compute Mean Gaze Positions
% Remove missing lines
gaze_positions_0 = rmmissing(gaze_positions_0);
gaze_positions_1 = rmmissing(gaze_positions_1);

mean_gaze_0 = mean(gaze_positions_0, 1); % Mean gaze position for participant 0
mean_gaze_1 = mean(gaze_positions_1, 1); % Mean gaze position for participant 1

%% Visualize Marked Positions
figure;
imshow(firstFrame);
hold on;

% Draw circles around marked positions
viscircles(robot_position, target_radius, 'Color', 'r', 'LineWidth', 2);
viscircles(mean_gaze_0, target_radius, 'Color', 'b', 'LineWidth', 2);
viscircles(mean_gaze_1, target_radius, 'Color', 'g', 'LineWidth', 2);

legend('Robot', 'Participant 0', 'Participant 1');
title('Manually Marked Target Areas');
pause(5);
close;

%% Gaze Classification Using Ray-Sphere Intersection
gaze_direction_0 = NaN(height(gaze_positions_0), 2);
gaze_direction_1 = NaN(height(gaze_positions_1), 2);

% Initialize classification labels
gaze_classification = repmat("Looking Elsewhere", height(data), 1);

gaze_0_height = height(gaze_positions_0);
gaze_1_height = height(gaze_positions_1);
gaze_height = max(gaze_0_height, gaze_1_height);

gaze_angle_0 = [data.gaze_angle_x(data.face_id == 0), data.gaze_angle_y(data.face_id == 0)];
gaze_angle_1 = [data.gaze_angle_x(data.face_id == 1), data.gaze_angle_y(data.face_id == 1)];

% Loop through each row to classify gaze
for i = 1:height(data)
    % Get the correct gaze position
    j = max(1, floor(i / 2));
    if data.face_id(i) == 0 && j <= gaze_0_height
        eye_center = gaze_positions_0(j, :); % Participant 0
        gaze_angle = gaze_angle_0(j, :);
        if j <= gaze_1_height
            other_position = gaze_positions_1(j, :); % Participant 1 target
        else
            % other_position = mean_gaze_1;
            continue;
        end

    elseif data.face_id(i) == 1 && j <= gaze_1_height
        eye_center = gaze_positions_1(j, :); % Participant 1
        gaze_angle = gaze_angle_1(j, :);
        if j <= gaze_0_height
            other_position = gaze_positions_0(j, :); % Participant 0 target
        else
            % other_position = mean_gaze_0;
            continue;
        end

    else
        continue; % Invalid face_id
    end

    % % Compute gaze direction (normalized vector)
    % gaze_direction = [data.gaze_0_x(i), data.gaze_0_y(i)];
    % gaze_direction = gaze_direction / norm(gaze_direction); % Normalize

    % Compute gaze direction using gaze angles in radians
    theta_x = gaze_angle(1); % Horizontal gaze angle (left/right)
    theta_y = gaze_angle(2); % Vertical gaze angle (up/down)
    
    % Convert angles to a 2D direction unit vector
    gaze_direction = [tan(theta_x), tan(theta_y)];
    gaze_direction = gaze_direction / norm(gaze_direction); % Normalize

    if data.face_id(i) == 0
        gaze_direction_0(j, :) = gaze_direction;
    else
        gaze_direction_1(j, :) = gaze_direction;
    end

    % Check intersection with Robot Sphere
    intersects_robot = ray_sphere_intersection(eye_center, gaze_direction, robot_position, target_radius);

    % Check intersection with Other Person Sphere
    intersects_other = ray_sphere_intersection(eye_center, gaze_direction, other_position, target_radius);

    % Assign classifications
    if intersects_robot
        gaze_classification(i) = "Looking at the Robot";
    elseif intersects_other
        gaze_classification(i) = "Looking at other Person";
    end
end

%% Separate Gaze Classification for Each Participant
gaze_classification_0 = gaze_classification(data.face_id == 0); % Participant 0
gaze_classification_1 = gaze_classification(data.face_id == 1); % Participant 1

%% Save Results to a CSV
% Extract Frame Numbers for Each Participant
frame_numbers_0 = data.frame(data.face_id == 0);
frame_numbers_1 = data.frame(data.face_id == 1);

% Ensure valid indices
num_frames_0 = length(frame_numbers_0);
num_frames_1 = length(frame_numbers_1);

% Prepare data for each participant
gaze_data_0 = table(frame_numbers_0, NaN(num_frames_0, 1), NaN(num_frames_0, 1), ...
                     NaN(num_frames_0, 1), NaN(num_frames_0, 1), gaze_classification_0, ...
                     'VariableNames', {'Frame', 'Gaze_Pos_X', 'Gaze_Pos_Y', ...
                                       'Gaze_Dir_X', 'Gaze_Dir_Y', 'Classification'});

gaze_data_1 = table(frame_numbers_1, NaN(num_frames_1, 1), NaN(num_frames_1, 1), ...
                     NaN(num_frames_1, 1), NaN(num_frames_1, 1), gaze_classification_1, ...
                     'VariableNames', {'Frame', 'Gaze_Pos_X', 'Gaze_Pos_Y', ...
                                       'Gaze_Dir_X', 'Gaze_Dir_Y', 'Classification'});

for i = 1:length(gaze_classification_0)
    gaze_data_0.Gaze_Pos_X(i) = gaze_positions_0(i, 1);
    gaze_data_0.Gaze_Pos_Y(i) = gaze_positions_0(i, 2);
    gaze_data_0.Gaze_Dir_X(i) = gaze_direction_0(i, 1);
    gaze_data_0.Gaze_Dir_Y(i) = gaze_direction_0(i, 2);
end

for i = 1:length(gaze_classification_1)
    gaze_data_1.Gaze_Pos_X(i) = gaze_positions_1(i, 1);
    gaze_data_1.Gaze_Pos_Y(i) = gaze_positions_1(i, 2);
    gaze_data_1.Gaze_Dir_X(i) = gaze_direction_1(i, 1);
    gaze_data_1.Gaze_Dir_Y(i) = gaze_direction_1(i, 2);
end

% Extract Facial Action Units
au_columns = {'frame', 'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', ...
              'AU07_r','AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', ...
              'AU17_r','AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r'};

au_0 = data(data.face_id == 0, :);
au_1 = data(data.face_id == 1, :);

% Extract relevant columns
au_0 = au_0(:, au_columns);
au_1 = au_1(:, au_columns);

% Save CSV files per participant
gaze_csv_0 = fullfile(output_path, '../results/gaze_0.csv');
gaze_csv_1 = fullfile(output_path, '../results/gaze_1.csv');
au_csv_0 = fullfile(output_path, '../results/au_0.csv');
au_csv_1 = fullfile(output_path, '../results/au_1.csv');
data_output_csv = fullfile(output_path, '../results/openface.csv');

writetable(gaze_data_0, gaze_csv_0);
writetable(gaze_data_1, gaze_csv_1);
writetable(au_0, au_csv_0);
writetable(au_1, au_csv_1);
writetable(data, data_output_csv);


%% Results
% Count gaze shifts
gaze_target_0 = gaze_classification_0(1);
gaze_shifts_0 = 0;
for i = 2:length(gaze_classification_0)-1
    if gaze_target_0 ~= gaze_classification_0(i)
        gaze_target_0 = gaze_classification_0(i);
        gaze_shifts_0 = gaze_shifts_0 + 1;
    end
end

fprintf('Gaze Shifts 0: %s\n', num2str(gaze_shifts_0));

gaze_target_1 = gaze_classification_1(1);
gaze_shifts_1 = 0;
for i = 2:length(gaze_classification_1)-1
    if gaze_target_1 ~= gaze_classification_1(i)
        gaze_target_1 = gaze_classification_1(i);
        gaze_shifts_1 = gaze_shifts_1 + 1;
    end
end

fprintf('Gaze Shifts 1: %s\n', num2str(gaze_shifts_1));

% Plots Histograms
fig1 = figure;
fig1.Position = [100, 100, 1200, 500]; % Adjust figure size and position
set(fig1, 'Color', 'w'); % Set white background

% Define colors
color_0 = [0, 0.4470, 0.7410]; % Deep Blue
color_1 = [0.4660, 0.6740, 0.1880]; % Soft Green

% Set maximum
max_y = max([sum(gaze_classification_0 == "Looking Elsewhere"), ...
             sum(gaze_classification_0 == "Looking at the Robot"), ...
             sum(gaze_classification_0 == "Looking at other Person"), ...
             sum(gaze_classification_1 == "Looking Elsewhere"), ...
             sum(gaze_classification_1 == "Looking at the Robot"), ...
             sum(gaze_classification_1 == "Looking at other Person")]) + 100;

% Histogram for Participant 0
ax1 = subplot(1, 2, 1);
histogram(ax1, categorical(gaze_classification_0), 'FaceColor', color_0, 'EdgeColor', 'k');
xlabel(ax1, 'Gaze Classification', 'FontSize', 12, 'FontWeight', 'bold');
ylabel(ax1, 'Frequency', 'FontSize', 12, 'FontWeight', 'bold');
title(ax1, 'Participant 0 Gaze Behavior', 'FontSize', 14, 'FontWeight', 'bold');
xtickangle(ax1, 25);
grid(ax1, 'on');
set(ax1, 'FontSize', 12, 'FontWeight', 'bold'); % Ensure uniform font size

% Histogram for Participant 1
ax2 = subplot(1, 2, 2);
histogram(ax2, categorical(gaze_classification_1), 'FaceColor', color_1, 'EdgeColor', 'k');
xlabel(ax2, 'Gaze Classification', 'FontSize', 12, 'FontWeight', 'bold');
ylabel(ax2, 'Frequency', 'FontSize', 12, 'FontWeight', 'bold');
title(ax2, 'Participant 1 Gaze Behavior', 'FontSize', 14, 'FontWeight', 'bold');
xtickangle(ax2, 25);
grid(ax2, 'on');
set(ax2, 'FontSize', 12, 'FontWeight', 'bold'); % Ensure uniform font size

% Apply to histograms
ylim(ax1, [0 max_y]);
ylim(ax2, [0 max_y]);

sgtitle('Gaze Classification Distribution for Each Participant', 'FontSize', 16, 'FontWeight', 'bold');

% Enable Interactive Mouse Hover
dcm = datacursormode(fig1);
set(dcm, 'Enable', 'on', 'DisplayStyle', 'datatip', 'SnapToDataVertex', 'on');

% Save figure
output_file = fullfile(output_path, 'gaze_histogram.png');
saveas(fig1, output_file);




%% Plot Gaze Directions
fig2 = figure;
fig2.Position = [150, 150, 1200, 600]; % Adjust figure size
set(fig2, 'Color', 'w'); % White background

imshow(firstFrame);
hold on;

% Draw circles around marked positions
viscircles(robot_position, target_radius, 'Color', 'r', 'LineWidth', 2);
viscircles(mean_gaze_0, target_radius, 'Color', 'b', 'LineWidth', 2);
viscircles(mean_gaze_1, target_radius, 'Color', 'g', 'LineWidth', 2);

% Define arrow scaling
arrow_scale = 100; % Adjust for visibility
h_gaze = [];

% Loop through a subset of frames for clarity
for i = 1:50:height(data)  % Adjust step size to prevent clutter
    % Get index based on cumulative face_id count
    j = max(1, floor(i / 2));
    if data.face_id(i) == 0
        if j > size(gaze_positions_0, 1), continue; end
        gaze_start = gaze_positions_0(j, :);
        gaze_angle = gaze_angle_0(j, :);
        arrow_color = 'b'; % Blue for Participant 0
    elseif data.face_id(i) == 1
        if j > size(gaze_positions_1, 1), continue; end
        gaze_start = gaze_positions_1(j, :);
        gaze_angle = gaze_angle_1(j, :);
        arrow_color = 'g'; % Green for Participant 1
    else
        continue; % Skip invalid face_id
    end

    % Fix coordinate system for correct projection
    gaze_dir_x = arrow_scale * sin(gaze_angle(1)); % Swap cos/sin
    gaze_dir_y = -arrow_scale * sin(gaze_angle(2)); % Invert Y for image

    % Plot the gaze direction arrow and store the handle for legend
    h_gaze = [h_gaze, quiver(gaze_start(1), gaze_start(2), gaze_dir_x, gaze_dir_y, ...
        'Color', arrow_color, 'LineWidth', 2, 'MaxHeadSize', 3)];
end

% Plot gaze target points
h_target0 = scatter(mean_gaze_0(1), mean_gaze_0(2), 100, 'b', 'filled', 'MarkerEdgeColor', 'k'); % Participant 0
h_target1 = scatter(mean_gaze_1(1), mean_gaze_1(2), 100, 'g', 'filled', 'MarkerEdgeColor', 'k'); % Participant 1
h_robot_target = scatter(robot_position(1), robot_position(2), 120, 'r', 'filled', 'MarkerEdgeColor', 'k'); % Robot

% Fix legend colors by explicitly defining corresponding markers
legend([h_robot_target, h_target0, h_target1, h_gaze(1)], ...
       {'Robot', 'Participant 0', 'Participant 1', 'Gaze Directions'}, ...
       'TextColor', 'k', 'FontSize', 12, 'FontWeight', 'bold');

title('Gaze Directions');

% Save figure
output_file = fullfile(output_path, 'gaze_directions.png');
saveas(fig2, output_file);







%% Auxiliary Functions 
% Function to Check Ray-Sphere Intersection
function intersects = ray_sphere_intersection(ray_origin, ray_dir, sphere_center, sphere_radius)
    % Vector from ray origin to sphere center
    oc = ray_origin - sphere_center;

    a = dot(ray_dir, ray_dir);
    b = 2 * dot(oc, ray_dir);
    c = dot(oc, oc) - sphere_radius^2;
    discriminant = b^2 - 4 * a * c;

    % If discriminant is positive, there is an intersection
    intersects = (discriminant >= 0);
end