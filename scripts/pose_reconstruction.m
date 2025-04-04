clear; close all; clc;

% Define path to stereo calibration parameters
path = "../../../experimental_studies/gaips/matlab_calibrations/params/";
stereo_params = load_and_extract(path + "stereoParams.mat");

% Compute projection matrices (P1, P2) and camera translation vectors (T1, T2)
[P1, P2, T1, T2] = compute_camera_matrix(stereo_params);



%% Handle MediaPipe Data
% Set image resolution (used for converting normalized coords to pixel coords)
image_width = 1920;
image_height = 1080;

% Load MediaPipe CSV keypoint data from both camera views
data_cam1 = readmatrix('../../../experimental_studies/gaips/1/results/G1_mediapipe_fixed.csv'); % Change to the right path
data_cam2 = readmatrix('../../../experimental_studies/gaips/1/results/G2_mediapipe_fixed.csv'); % Change to the right path

% Reassign pose IDs based on horizontal location (left = 0, right = 1)
data_cam1 = reassign_pose_ids_by_horizontal_position(data_cam1, image_width, image_height);
data_cam2 = reassign_pose_ids_by_horizontal_position(data_cam2, image_width, image_height);



%% Triangulate keypoints
% Reconstruct 3D keypoints for pose 0 and pose 1
pose0_3D = triangulate_pose(0, data_cam1, data_cam2, P1, P2);
pose1_3D = triangulate_pose(1, data_cam1, data_cam2, P1, P2);

% Export pose reconstruction to csv
all_poses = [pose0_3D; pose1_3D];


%% Rotate
% Extract just the 3D positions
XYZ = all_poses(:, 4:6)';

% Define rotation matrix to align coordinate system
% (Swaps Y and Z axes to make Y = vertical, Z = lateral)
R = [0  0  1;
     0  1  0;
    -1  0  0];

% Apply rotation
XYZ_rot = R * XYZ;

% Update `all_poses` with rotated coordinates
all_poses(:, 4:6) = XYZ_rot';



%% Write reconstructed poses to CSV files
% Export pose 0 keypoints
writetable(array2table(pose0_3D, ...
    'VariableNames', {'frame', 'pose', 'keypoint', 'x', 'y', 'z'}), ...
    '3D_pose_reconstruction_0.csv');

% Export pose 1 keypoints
writetable(array2table(pose1_3D, ...
    'VariableNames', {'frame', 'pose', 'keypoint', 'x', 'y', 'z'}), ...
    '3D_pose_reconstruction_1.csv');



%% Plot results
% Visualize the 3D skeletons over time
plot_skeletons(all_poses);




%% ========================= Auxiliary Functions ==========================

function out = load_and_extract(path)
    tmp = load(path);
    f = fieldnames(tmp);
    out = tmp.(f{1});
end


function [P1, P2, T1, T2] = compute_camera_matrix(stereoParams)
    % Compute projection matrices for stereo triangulation
    % P1, P2: 3x4 projection matrices
    % T1, T2: translation vectors (T1 is zero since camera 1 is reference)

    % Intrinsic matrices
    K1 = stereoParams.CameraParameters1.K;
    K2 = stereoParams.CameraParameters2.K;

    % Extrinsics (camera 1 is at origin)
    R1 = eye(3);
    T1 = [0; 0; 0]; % in mm
    R2 = stereoParams.RotationOfCamera2;
    T2 = stereoParams.TranslationOfCamera2(:) / 1000; % convert to meters

    % Projection matrices
    P1 = K1 * [R1, -R1*T1];
    P2 = K2 * [R2, -R2*T2];
end


function data_out = reassign_pose_ids_by_horizontal_position(data, img_width, img_height)
    % Reassign pose IDs per frame so that leftmost = 0 and rightmost = 1
    % Useful to maintain consistent identity assignment across views

    frames = unique(data(:,1));
    data_out = data;
    
    for i = 1:length(frames)
        frame = frames(i);
        idx_frame = data(:,1) == frame;
        poses_in_frame = unique(data(idx_frame,2));
        
        if numel(poses_in_frame) ~= 2
            continue;  % Skip frames that don't have exactly 2 poses
        end
        
        centroids = zeros(2,1);
        for j = 1:2
            idx_pose = idx_frame & data(:,2) == poses_in_frame(j);
            keypoints = data(idx_pose, 4:5) .* [img_width, img_height];
            centroids(j) = mean(keypoints(:,1));  % X-centroid
        end
        
        % Sort poses left to right and reassign IDs
        [~, order] = sort(centroids);
        original_ids = poses_in_frame(order);
        new_ids = [0, 1];
        
        for j = 1:2
            data_out(idx_frame & data(:,2) == original_ids(j), 2) = new_ids(j);
        end
    end
end


function pose_3D = triangulate_pose(poseID, data_cam1, data_cam2, P1, P2)
    % Triangulate all visible keypoints for a given poseID across frames

    image_width = 1920;
    image_height = 1080;
    keypoints_to_use = 0:22;  % Keypoints 0 to 22

    % Parse data
    frames1 = data_cam1(:,1); poses1 = data_cam1(:,2); landmarks1 = data_cam1(:,3);
    frames2 = data_cam2(:,1); poses2 = data_cam2(:,2); landmarks2 = data_cam2(:,3);

    % Convert normalized coords to pixels
    xy1 = [data_cam1(:,4), data_cam1(:,5)] .* [image_width, image_height];
    xy2 = [data_cam2(:,4), data_cam2(:,5)] .* [image_width, image_height];

    % Flip Y-axis to match image coordinate MediaPipe convention
    xy1(:,2) = image_height - xy1(:,2);
    xy2(:,2) = image_height - xy2(:,2);

    common_frames = intersect(frames1, frames2);
    pose_3D = [];

    for f_idx = 1:length(common_frames)
        frame = common_frames(f_idx);
        idx1 = frames1 == frame;
        idx2 = frames2 == frame;

        for k = keypoints_to_use
            match1 = idx1 & landmarks1 == k & poses1 == poseID;
            match2 = idx2 & landmarks2 == k & poses2 == poseID;

            if any(match1) && any(match2)
                x1 = mean(xy1(match1, :), 1);
                x2 = mean(xy2(match2, :), 1);

                x1_hom = [x1, 1]';
                x2_hom = [x2, 1]';

                X = triangulate_points(x1_hom, x2_hom, P1, P2);
                % X(3) = -X(3); % Invert Z if needed

                pose_3D = [pose_3D; frame, poseID, k, X'];
            end
        end
    end
end


function X = triangulate_points(x1, x2, P1, P2)
    % Linear triangulation using SVD
    % x1, x2: homogeneous 2D points (3x1)
    % P1, P2: projection matrices
    % X: triangulated 3D point in world coordinates

    A = [
        x1(1) * P1(3,:) - P1(1,:);
        x1(2) * P1(3,:) - P1(2,:);
        x2(1) * P2(3,:) - P2(1,:);
        x2(2) * P2(3,:) - P2(2,:);
    ];
    [~, ~, V] = svd(A);
    X_hom = V(:, end);
    X = X_hom(1:3) / X_hom(4);
end


function plot_skeletons(all_poses)
    % Visualize 3D skeletons frame by frame
    % Skeleton is drawn using a predefined MediaPipe-style connectivity map

    pose0 = all_poses(all_poses(:,2) == 0, :);
    pose1 = all_poses(all_poses(:,2) == 1, :);

    frames = unique(all_poses(:,1));
    keypoints = unique(all_poses(:,3));

    figure;
    hold on;
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title('3D Reconstructed Skeletons');
    grid on; axis equal;
    view(45, 20);

    % Define skeleton connections (based on MediaPipe)
    skeleton_connections = [ ...
        0 1; 1 2; 2 3; 3 7;  
        0 4; 4 5; 5 6; 6 8;  
        9 10; 
        11 12; 
        11 13; 13 15; 15 17; 15 19; 17 19; 15 21;  
        12 14; 14 16; 16 18; 16 20; 18 20; 16 22 
    ];

    for f = frames'
        % Clear previous frame
        h = findobj(gca, 'Type', 'Line', '-or', 'Type', 'Scatter');
        delete(h);

        % Draw pose 0
        X0 = nan(3, length(keypoints));
        for k = keypoints'
            idx = pose0(:,1) == f & pose0(:,3) == k;
            if any(idx)
                X0(:,k+1) = pose0(idx, 4:6)';
            end
        end

        if any(~isnan(X0(1,:)))
            scatter3(X0(1,:), X0(2,:), X0(3,:), 50, 'ro', 'filled');
            for i = 1:size(skeleton_connections,1)
                p1 = skeleton_connections(i,1)+1;
                p2 = skeleton_connections(i,2)+1;
                if all(~isnan(X0(:,[p1 p2])), 'all')
                    plot3(X0(1,[p1 p2]), X0(2,[p1 p2]), X0(3,[p1 p2]), 'r-', 'LineWidth', 2);
                end
            end
        end

        % Draw pose 1
        X1 = nan(3, length(keypoints));
        for k = keypoints'
            idx = pose1(:,1) == f & pose1(:,3) == k;
            if any(idx)
                X1(:,k+1) = pose1(idx, 4:6)';
            end
        end

        if any(~isnan(X1(1,:)))
            scatter3(X1(1,:), X1(2,:), X1(3,:), 50, 'bo', 'filled');
            for i = 1:size(skeleton_connections,1)
                p1 = skeleton_connections(i,1)+1;
                p2 = skeleton_connections(i,2)+1;
                if all(~isnan(X1(:,[p1 p2])), 'all')
                    plot3(X1(1,[p1 p2]), X1(2,[p1 p2]), X1(3,[p1 p2]), 'b-', 'LineWidth', 2);
                end
            end
        end

        pause(0.1); % Animate
    end
end