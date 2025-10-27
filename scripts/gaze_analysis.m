clear; close all; clc;

%% Load Gaze Data
session_path = '../../../experimental_studies/gaips/30/';
data = readtable(fullfile(session_path, 'results/top_tracked.csv'));

%% Load Video and Select Positions
videoFile = fullfile(session_path, 'videos/top.mp4');
video = VideoReader(videoFile);
firstFrame = readFrame(video);

% Screen dimensions
frame_width = 1920;
frame_height = 1080;

target_radius = 170; % Adjust as needed

figures_path = fullfile(session_path, 'results/figures');
if ~isfolder(figures_path)
    mkdir(figures_path);
end

results_path = fullfile(session_path, 'results/openface');
if ~isfolder(results_path)
    mkdir(results_path);
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
data = data(data.confidence >= 0.5, :);

landmarks = [8, 11, 14, 17, 36, 39, 42, 45];
facepos = zeros(2,2);  % rows = [x,y] for tracked_id = 0 and 1


for personID = 0:1
    sel = data.tracked_id == personID;

    xs = [];
    ys = [];

    for l = landmarks
        x_col = "x_" + string(l);
        y_col = "y_" + string(l);

        x_vals = data.(x_col)(sel);
        y_vals = data.(y_col)(sel);

        xs = [xs, x_vals];
        ys = [ys, y_vals];
    end

    % Mean of valid landmark coordinates
    mean_x = mean(xs, 2, 'omitnan');
    mean_y = mean(ys, 2, 'omitnan');

    % Get final average position in pixel space
    facepos(personID+1,1) = mean(mean_x, 'omitnan');
    facepos(personID+1,2) = mean(mean_y, 'omitnan');
end


%% Gaze Positions 
% Initialize empty arrays to store gaze positions
gaze_positions_0 = NaN(height(data), 2); % [gaze_x, gaze_y] for participant 0
gaze_positions_1 = NaN(height(data), 2); % [gaze_x, gaze_y] for participant 1


left_eye = [8, 11, 14, 17];
right_eye = [36, 39, 42, 45];

% Loop through each row to assign gaze based on tracked_id
for i = 1:height(data)
    % Compute mean x and y for left eye
    left_eye_x = mean(table2array(data(i, strcat("eye_lmk_X_", string(left_eye)))), 'omitnan');
    left_eye_y = mean(table2array(data(i, strcat("eye_lmk_Y_", string(left_eye)))), 'omitnan');

    % Compute mean x and y for right eye
    right_eye_x = mean(table2array(data(i, strcat("eye_lmk_X_", string(right_eye)))), 'omitnan');
    right_eye_y = mean(table2array(data(i, strcat("eye_lmk_Y_", string(right_eye)))), 'omitnan');

    % % Compute overall eye center as the mean of both eyes
    gaze_x = (left_eye_x + right_eye_x) / 2;
    gaze_y = (left_eye_y + right_eye_y) / 2;

    % % Convert from OpenFace centered coordinates to image pixel coordinates
    gaze_x = (frame_width / 2) + gaze_x;
    gaze_y = (frame_height / 2) + gaze_y;

    % Store gaze positions in the respective array
    if data.tracked_id(i) == 0
        gaze_positions_0(i, :) = [gaze_x, gaze_y]; % Store for participant 0
    elseif data.tracked_id(i) == 1
        gaze_positions_1(i, :) = [gaze_x, gaze_y]; % Store for participant 1
    end
end

%% Compute Mean Gaze Positions
% Remove missing lines
gaze_positions_0 = rmmissing(gaze_positions_0);
gaze_positions_1 = rmmissing(gaze_positions_1);

% mean_gaze_0 = mean(gaze_positions_0, 1); % Mean gaze position for participant 0
% mean_gaze_1 = mean(gaze_positions_1, 1); % Mean gaze position for participant 1

%% Visualize Marked Positions
figure;
imshow(firstFrame);
hold on;

% Draw circles around marked positions
viscircles(robot_position, target_radius, 'Color', 'r', 'LineWidth', 2);
viscircles(facepos(1, :), target_radius, 'Color', 'b', 'LineWidth', 2);
viscircles(facepos(2, :), target_radius, 'Color', 'g', 'LineWidth', 2);

legend('Robot', 'Participant 0', 'Participant 1');
title('Manually Marked Target Areas');
pause(5);
close;




%% Gaze Classification Using Ray-Sphere Intersection
gaze_direction_0 = NaN(height(gaze_positions_0), 2);
gaze_direction_1 = NaN(height(gaze_positions_1), 2);


% Initialize classification labels
gaze_classification = repmat("Looking Elsewhere", height(data), 1);

gaze_angle_0 = [data.gaze_angle_x(data.tracked_id == 0), data.gaze_angle_y(data.tracked_id == 0)];
gaze_angle_1 = [data.gaze_angle_x(data.tracked_id == 1), data.gaze_angle_y(data.tracked_id == 1)];

% Loop through each row to classify gaze
for i = 1:height(data)
    if data.tracked_id(i) == 0
        j = sum(data.tracked_id(1:i) == 0);
        if j > size(gaze_positions_0, 1), continue; end
        eye_center = gaze_positions_0(j, :);
        gaze_angle = gaze_angle_0(j, :);
        if j > size(gaze_positions_1, 1), continue; end
        other_position = gaze_positions_1(j, :);
    elseif data.tracked_id(i) == 1
        j = sum(data.tracked_id(1:i) == 1);
        if j > size(gaze_positions_1, 1), continue; end
        eye_center = gaze_positions_1(j, :);
        gaze_angle = gaze_angle_1(j, :);
        if j > size(gaze_positions_0, 1), continue; end
        other_position = gaze_positions_0(j, :);
    else
        continue;
    end

    % Compute gaze direction
    theta_x = gaze_angle(1);
    theta_y = gaze_angle(2);
    gaze_direction = [tan(theta_x), tan(theta_y)];
    % gaze_direction = gaze_direction / norm(gaze_direction);

    if data.tracked_id(i) == 0
        gaze_direction_0(j, :) = gaze_direction;
    else
        gaze_direction_1(j, :) = gaze_direction;
    end

    intersects_robot = ray_circle_intersection(eye_center, gaze_direction, robot_position, target_radius);
    intersects_other = ray_circle_intersection(eye_center, gaze_direction, other_position, target_radius);

    if intersects_robot
        gaze_classification(i) = "Looking at the Robot";
    elseif intersects_other
        gaze_classification(i) = "Looking at other Person";
    end
end

% Separate Gaze Classification for Each Participant
% Initialize with 'Looking Elsewhere' for all frames
max_frames = max(data.frame);
gaze_classification_0_full = repmat("Looking Elsewhere", max_frames, 1);
gaze_classification_1_full = repmat("Looking Elsewhere", max_frames, 1);

% Get frame indices for each participant
frame_idx_0 = data.frame(data.tracked_id == 0);
frame_idx_1 = data.frame(data.tracked_id == 1);

% Get classifications already computed
gaze_classification_0 = gaze_classification(data.tracked_id == 0); % Pose 0
gaze_classification_1 = gaze_classification(data.tracked_id == 1); % Pose 1

% Assign to correct frame positions
gaze_classification_0_full(frame_idx_0) = gaze_classification_0;
gaze_classification_1_full(frame_idx_1) = gaze_classification_1;

% Replace originals with the full-length versions
gaze_classification_0 = gaze_classification_0_full;
gaze_classification_1 = gaze_classification_1_full;




%% Save Results to a CSV
% Extract Frame Numbers for Each Participant
frame_numbers_0 = (1:max_frames)';
frame_numbers_1 = (1:max_frames)';

% Ensure valid indices
num_frames_0 = length(frame_numbers_0);
num_frames_1 = length(frame_numbers_1);

% Prepare data for each participant
gaze_data_0 = table(frame_numbers_0, NaN(max_frames, 1), NaN(max_frames, 1), ...
                     NaN(max_frames, 1), NaN(max_frames, 1), gaze_classification_0, ...
                     'VariableNames', {'Frame', 'Gaze_Pos_X', 'Gaze_Pos_Y', ...
                                       'Gaze_Dir_X', 'Gaze_Dir_Y', 'Classification'});

gaze_data_1 = table(frame_numbers_1, NaN(max_frames, 1), NaN(max_frames, 1), ...
                     NaN(max_frames, 1), NaN(max_frames, 1), gaze_classification_1, ...
                     'VariableNames', {'Frame', 'Gaze_Pos_X', 'Gaze_Pos_Y', ...
                                       'Gaze_Dir_X', 'Gaze_Dir_Y', 'Classification'});


valid_idx_0 = find(~all(isnan(gaze_positions_0), 2));
gaze_positions_0_full = NaN(max_frames, 2);
gaze_direction_0_full = NaN(max_frames, 2);
gaze_positions_0_full(frame_idx_0(valid_idx_0), :) = gaze_positions_0(valid_idx_0, :);
gaze_direction_0_full(frame_idx_0(valid_idx_0), :) = gaze_direction_0(valid_idx_0, :);

valid_idx_1 = find(~all(isnan(gaze_positions_1), 2));
gaze_positions_1_full = NaN(max_frames, 2);
gaze_direction_1_full = NaN(max_frames, 2);
gaze_positions_1_full(frame_idx_1(valid_idx_1), :) = gaze_positions_1(valid_idx_1, :);
gaze_direction_1_full(frame_idx_1(valid_idx_1), :) = gaze_direction_1(valid_idx_1, :);


for i = 1:max_frames
    gaze_data_0.Gaze_Pos_X(i) = gaze_positions_0_full(i, 1);
    gaze_data_0.Gaze_Pos_Y(i) = gaze_positions_0_full(i, 2);
    gaze_data_0.Gaze_Dir_X(i) = gaze_direction_0_full(i, 1);
    gaze_data_0.Gaze_Dir_Y(i) = gaze_direction_0_full(i, 2);
end

for i = 1:max_frames
    gaze_data_1.Gaze_Pos_X(i) = gaze_positions_1_full(i, 1);
    gaze_data_1.Gaze_Pos_Y(i) = gaze_positions_1_full(i, 2);
    gaze_data_1.Gaze_Dir_X(i) = gaze_direction_1_full(i, 1);
    gaze_data_1.Gaze_Dir_Y(i) = gaze_direction_1_full(i, 2);
end

% Extract Facial Action Units
au_columns = {'frame', 'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', ...
              'AU07_r','AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', ...
              'AU17_r','AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r', ...
              'AU01_c','AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c',...
              'AU09_c','AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c',...
              'AU20_c','AU23_c', 'AU25_c', 'AU26_c', 'AU45_c'};

au_0 = data(data.tracked_id == 0, :);
au_1 = data(data.tracked_id == 1, :);

% Extract relevant columns
au_0 = au_0(:, au_columns);
au_1 = au_1(:, au_columns);

% Save CSV files per participant
gaze_csv_0 = fullfile(session_path, '/results/openface/gaze_0.csv');
gaze_csv_1 = fullfile(session_path, '/results/openface/gaze_1.csv');
au_csv_0 = fullfile(session_path, '/results/openface/au_0.csv');
au_csv_1 = fullfile(session_path, '/results/openface/au_1.csv');

writetable(gaze_data_0, gaze_csv_0);
writetable(gaze_data_1, gaze_csv_1);
writetable(au_0, au_csv_0);
writetable(au_1, au_csv_1);


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




%% Plots Histograms
fig1 = figure;
fig1.Position = [100, 100, 1200, 500]; % Adjust figure size and position

% Define colors
color_0 = [0, 0.4470, 0.7410]; % Deep Blue
color_1 = [0.4660, 0.6740, 0.1880]; % Soft Green

gaze_labels = {'Looking at the Robot', 'Looking at other Person', 'Looking Elsewhere'};

% Count and normalize for Participant 0
counts_0 = countcats(categorical(gaze_classification_0, gaze_labels));
proportions_0 = counts_0 / sum(counts_0);

% Count and normalize for Participant 1
counts_1 = countcats(categorical(gaze_classification_1, gaze_labels));
proportions_1 = counts_1 / sum(counts_1);

% Histogram for Participant 0
ax1 = subplot(1, 2, 1);
bar(ax1, proportions_0, 'FaceColor', color_0, 'EdgeColor', 'k');
set(ax1, 'XTickLabel', gaze_labels, 'XTick', 1:3);
ylim(ax1, [0 1]);
for i = 1:length(proportions_0)
    text(ax1, i, proportions_0(i) + 0.02, sprintf('%.1f%%', 100 * proportions_0(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', 'w');
end
xlabel(ax1, 'Gaze Classification', 'FontSize', 12, 'FontWeight', 'bold');
ylabel(ax1, 'Frequency', 'FontSize', 12, 'FontWeight', 'bold');
title(ax1, 'Participant 0 Gaze Behavior', 'FontSize', 14, 'FontWeight', 'bold');
xtickangle(ax1, 25);
grid(ax1, 'on');
set(ax1, 'FontSize', 12, 'FontWeight', 'bold'); % Ensure uniform font size

% Histogram for Participant 1
ax2 = subplot(1, 2, 2);
bar(ax2, proportions_1, 'FaceColor', color_1, 'EdgeColor', 'k');
set(ax2, 'XTickLabel', gaze_labels, 'XTick', 1:3);
ylim(ax2, [0 1]);
for i = 1:length(proportions_1)
    text(ax2, i, proportions_1(i) + 0.02, sprintf('%.1f%%', 100 * proportions_1(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', 'w');
end
xlabel(ax2, 'Gaze Classification', 'FontSize', 12, 'FontWeight', 'bold');
ylabel(ax2, 'Frequency', 'FontSize', 12, 'FontWeight', 'bold');
title(ax2, 'Participant 1 Gaze Behavior', 'FontSize', 14, 'FontWeight', 'bold');
xtickangle(ax2, 25);
grid(ax2, 'on');
set(ax2, 'FontSize', 12, 'FontWeight', 'bold'); % Ensure uniform font size

sgtitle('Gaze Classification Distribution for Each Participant', 'FontSize', 16, 'FontWeight', 'bold');

% Enable Interactive Mouse Hover
dcm = datacursormode(fig1);
set(dcm, 'Enable', 'on', 'DisplayStyle', 'datatip', 'SnapToDataVertex', 'on');

% Save figure
output_file = fullfile(figures_path, 'gaze_histogram.png');
saveas(fig1, output_file);


% Percent values for Participant 0
disp('📊 Participant 0 Gaze Summary (%):');
for i = 1:length(gaze_labels)
    fprintf('  - %s: %.1f%%\n', gaze_labels{i}, 100 * proportions_0(i));
end

% Percent values for Participant 1
disp('📊 Participant 1 Gaze Summary (%):');
for i = 1:length(gaze_labels)
    fprintf('  - %s: %.1f%%\n', gaze_labels{i}, 100 * proportions_1(i));
end





%% Plot Gaze Directions
fig2 = figure;
fig2.Position = [150, 150, 1200, 600]; % Adjust figure size

imshow(firstFrame);
hold on;

% Draw circles around marked positions
viscircles(robot_position, target_radius, 'Color', 'r', 'LineWidth', 2);
viscircles(facepos(1, :), target_radius, 'Color', 'b', 'LineWidth', 2);
viscircles(facepos(2, :), target_radius, 'Color', 'g', 'LineWidth', 2);

% Define arrow scaling
arrow_scale = 100; % Adjust for visibility
h_gaze = [];

% Loop through a subset of frames for clarity
for i = 1:30:height(data)  % Adjust step size to prevent clutter
    % Get index based on cumulative tracked_id count
    j = max(1, floor(i / 2));
    if data.tracked_id(i) == 0
        if j > size(gaze_positions_0, 1), continue; end
        gaze_start = gaze_positions_0(j, :);
        gaze_angle = gaze_angle_0(j, :);
        gaze_dir = gaze_direction_0(j, :);
        arrow_color = 'b'; % Blue for Participant 0
    elseif data.tracked_id(i) == 1
        if j > size(gaze_positions_1, 1), continue; end
        gaze_start = gaze_positions_1(j, :);
        gaze_angle = gaze_angle_1(j, :);
        gaze_dir = gaze_direction_1(j, :);
        arrow_color = 'g'; % Green for Participant 1
    else
        continue; % Skip invalid tracked_id
    end

    % Fix coordinate system for correct projection
    gaze_dir_x = arrow_scale * gaze_dir(1); % Swap cos/sin
    gaze_dir_y = arrow_scale * gaze_dir(2); % Invert Y for image

    % Plot the gaze direction arrow and store the handle for legend
    h_gaze = [h_gaze, quiver(gaze_start(1), gaze_start(2), gaze_dir_x, gaze_dir_y, ...
        'Color', arrow_color, 'LineWidth', 2, 'MaxHeadSize', 3)];
end

% Plot gaze target points
h_target0 = scatter(facepos(1, 1), facepos(1, 2), 100, 'b', 'filled', 'MarkerEdgeColor', 'k'); % Participant 0
h_target1 = scatter(facepos(2, 1), facepos(2, 2), 100, 'g', 'filled', 'MarkerEdgeColor', 'k'); % Participant 1
h_robot_target = scatter(robot_position(1), robot_position(2), 120, 'r', 'filled', 'MarkerEdgeColor', 'k'); % Robot

legend([h_robot_target, h_target0, h_target1], ...
       {'Robot', 'Participant 0', 'Participant 1', ...
        'Gaze Direction P0', 'Gaze Direction P1'}, ...
       'FontSize', 12, 'FontWeight', 'bold', ...
       'Location', 'northeastoutside');

title('Gaze Directions');

% Save figure
output_file = fullfile(figures_path, 'gaze_directions.png');
saveas(fig2, output_file);







%% Auxiliary Functions 
% Function to Check Ray-Sphere Intersection
function intersects = ray_circle_intersection(ray_origin, ray_dir, sphere_center, sphere_radius)
    % Vector from ray origin to sphere center
    oc = ray_origin - sphere_center;

    a = dot(ray_dir, ray_dir);
    b = 2 * dot(oc, ray_dir);
    c = dot(oc, oc) - sphere_radius^2;
    discriminant = b^2 - 4 * a * c;

    % If discriminant is positive, there is an intersection
    intersects = (discriminant >= 0);
end