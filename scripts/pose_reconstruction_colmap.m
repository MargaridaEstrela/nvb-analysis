clear; close all; clc;

% Define path to stereo calibration parameters
path_colmap = "../../../experimental_studies/gaips/calibration/colmap/";
colmap_imgs = path_colmap + "images.txt";
allCameraPoses = loadColmapExtrinsics(colmap_imgs);
cameraPoses = selectThreeCameras(allCameraPoses);

path_calibratrions = "../../../experimental_studies/gaips/calibration/cameras/";
cam1_params = load_and_extract(path_calibratrions + "cameraParams_G1");
cam2_params = load_and_extract(path_calibratrions + "cameraParams_G2");
cam3_params = load_and_extract(path_calibratrions + "cameraParams_G3");

P1 = compute_camera_matrix(cameraPoses(1), cam1_params);
P2 = compute_camera_matrix(cameraPoses(2), cam2_params);
P3 = compute_camera_matrix(cameraPoses(3), cam3_params);


%% Handle MediaPipe Data
% Set image resolution (used for converting normalized coords to pixel coords)
image_width = 1920;
image_height = 1080;

% Load MediaPipe CSV keypoint data from both camera views
data_cam1 = readmatrix('../../../experimental_studies/gaips/1/results/mediapipe/left_fixed.csv'); % Change to the right path
data_cam2 = readmatrix('../../../experimental_studies/gaips/1/results/mediapipe/right_fixed.csv'); % Change to the right path
data_cam3 = readmatrix('../../../experimental_studies/gaips/1/results/mediapipe/top_fixed.csv'); % Change to the right path

% Convert normalized coords from MediaPipe to pixels
data_cam1(:,4:5) = data_cam1(:,4:5) .* [image_width, image_height];
data_cam2(:,4:5) = data_cam2(:,4:5) .* [image_width, image_height];
data_cam3(:,4:5) = data_cam3(:,4:5) .* [image_width, image_height];

data_cam1(:,5) = image_height - data_cam1(:,5);
data_cam2(:,5) = image_height - data_cam2(:,5);
data_cam3(:,5) = image_height - data_cam3(:,5);


%% Triangulate keypoints
% Reconstruct 3D keypoints for pose 0 and pose 1
pose0_3D = triangulate_pose_3view(0, data_cam1, data_cam2, data_cam3, P1, P2, P3);
pose1_3D = triangulate_pose_3view(1, data_cam1, data_cam2, data_cam3, P1, P2, P3);


%% Scale skeletons
% Get 3D shoulder positions from reconstructed skeletons
L_SHOULDER = 11;
R_SHOULDER = 12;

% Pose 0
idx_L0 = pose0_3D(:,3) == L_SHOULDER;
idx_R0 = pose0_3D(:,3) == R_SHOULDER;
shoulder0 = vecnorm(pose0_3D(idx_L0, 4:6) - pose0_3D(idx_R0, 4:6), 2, 2);

% Pose 1
idx_L1 = pose1_3D(:,3) == L_SHOULDER;
idx_R1 = pose1_3D(:,3) == R_SHOULDER;
shoulder1 = vecnorm(pose1_3D(idx_L1, 4:6) - pose1_3D(idx_R1, 4:6), 2, 2);

% Remove NaNs just in case
shoulder0 = shoulder0(~isnan(shoulder0));
shoulder1 = shoulder1(~isnan(shoulder1));

% Compute scaling
scale0 = mean(shoulder0);
scale1 = mean(shoulder1);
target_scale = mean([scale0, scale1]);

% Normalize poses using the 3D shoulder distance
pose0_3D(:,4:6) = pose0_3D(:,4:6) * (target_scale / scale0);
pose1_3D(:,4:6) = pose1_3D(:,4:6) * (target_scale / scale1);


%% Write reconstructed poses to CSV files
% Export pose reconstruction to csv
all_poses = [pose0_3D; pose1_3D];

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

function cameraPoses = loadColmapExtrinsics(colmapImagesPath)
    % Extract R and t from COLMAP's images.txt file
    %
    % colmapImagesPath: full path to 'images.txt' file from COLMAP
    %
    % cameraPoses - struct array with fields:
    %     .name - image filename
    %     .R    - 3x3 rotation matrix (world to camera)
    %     .t    - 3x1 translation vector (world to camera)

    % Read file lines
    fid = fopen(colmapImagesPath, 'r');
    lines = textscan(fid, '%s', 'Delimiter', '\n');
    fclose(fid);
    lines = lines{1};

    % Remove comment and empty lines
    lines = lines(~startsWith(lines, '#') & ~cellfun(@isempty, lines));

    % Preallocate
    numCameras = length(lines) / 2;
    cameraPoses(numCameras) = struct('name', '', 'R', [], 't', []);

    % Parse each camera entry
    for i = 1:numCameras
        header = strsplit(lines{2*i - 1});
        
        % Parse quaternion and translation
        qw = str2double(header{2});
        qx = str2double(header{3});
        qy = str2double(header{4});
        qz = str2double(header{5});
        tx = str2double(header{6});
        ty = str2double(header{7});
        tz = str2double(header{8});
        imageName = header{10};

        % Convert to rotation matrix (camera to world)
        q = quaternion([qw, qx, qy, qz]);  % MATLAB: [w x y z]
        R_cw = rotmat(q, 'frame');
        t_cw = [tx; ty; tz];

        % Convert to world to camera
        R_wc = R_cw';
        t_wc = -R_wc * t_cw;

        % Store
        cameraPoses(i).name = imageName;
        cameraPoses(i).R = R_wc;
        cameraPoses(i).t = t_wc;
    end
    
    % Align all cameras to cam3's frame
    ref_R = cameraPoses(3).R;
    ref_t = cameraPoses(3).t;
    
    for i = 1:numCameras
        R_new = cameraPoses(i).R * ref_R';
        t_new = cameraPoses(i).t - R_new * ref_t;
    
        cameraPoses(i).R = R_new;
        cameraPoses(i).t = t_new;
    end
end

function selectedPoses = selectThreeCameras(cameraPoses)
    % Define the target filenames
    orderedNames = {'frame_G1.jpg', 'frame_G2.jpg', 'frame_G3.jpg'};

    % Initialize output
    selectedPoses = repmat(struct('name', '', 'R', [], 't', []), 1, 3);

    % Search and assign in the desired order
    for i = 1:3
        match = find(strcmp({cameraPoses.name}, orderedNames{i}));
        if isempty(match)
            error("Camera %s not found in cameraPoses!", orderedNames{i});
        end
        selectedPoses(i) = cameraPoses(match);
    end
end

function P = compute_camera_matrix(cameraPoses, cam_params)
    % Compute projection matrices for stereo triangulation
    % P1: 3x4 projection matrix

    % Intrinsic matrix
    K = cam_params.K;

    % Extrinsics
    R = cameraPoses.R;
    t = cameraPoses.t;

    % Projection matrices
    P = K * [R, -R*t];
end

function pose_3D = triangulate_pose_3view(poseID, data_cam1, data_cam2, data_cam3, P1, P2, P3)
    % Triangulate all visible keypoints for a given poseID across frames

    min_confidence = 0.01;
    keypoints_to_use = [0, ...                      % Nose
                        1, 2, 3, 4, 5, 6, ...       % Eyes
                        7, 8, ...                   % Ears
                        9, 10, ...                  % Mouth
                        11, 12, ...                 % Shoulders
                        13, 14, ...                 % Elbows
                        15, 16, ...                 % Wrists
                        ];

    % Parse data
    frames1 = data_cam1(:,1); poses1 = data_cam1(:,2); landmarks1 = data_cam1(:,3);
    frames2 = data_cam2(:,1); poses2 = data_cam2(:,2); landmarks2 = data_cam2(:,3);
    frames3 = data_cam3(:,1); poses3 = data_cam3(:,2); landmarks3 = data_cam3(:,3);

    xy1 = [data_cam1(:,4), data_cam1(:,5)];
    xy2 = [data_cam2(:,4), data_cam2(:,5)];
    xy3 = [data_cam3(:,4), data_cam3(:,5)];

    % Get confidence values for each landmark
    conf1 = data_cam1(:,7);
    conf2 = data_cam2(:,7);
    conf3 = data_cam3(:,7);

    common_frames = intersect(intersect(frames1, frames2), frames3);
    pose_3D = [];

    for f_idx = 1:length(common_frames)
        frame = common_frames(f_idx);
        idx1 = frames1 == frame;
        idx2 = frames2 == frame;
        idx3 = frames3 == frame;

        for k = keypoints_to_use
            match1 = idx1 & landmarks1 == k & poses1 == poseID;
            match2 = idx2 & landmarks2 == k & poses2 == poseID;
            match3 = idx3 & landmarks3 == k & poses3 == poseID;

            if any(match1) && any(match2) && any(match3)
                % Check confidence
                c1 = mean(conf1(match1));
                c2 = mean(conf2(match2));
                c3 = mean(conf3(match3));

                if c1 < min_confidence || c2 < min_confidence || c3 < min_confidence
                    continue;
                end

                x1 = mean(xy1(match1, :), 1);
                x2 = mean(xy2(match2, :), 1);
                x3 = mean(xy3(match3, :), 1);

                x1_hom = [x1, 1]';
                x2_hom = [x2, 1]';
                x3_hom = [x3, 1]';

                X = triangulate_points_multi({x1_hom, x2_hom, x3_hom}, {P1, P2, P3});
                pose_3D = [pose_3D; frame, poseID, k, X'];

            end
        end
    end
end

function X = triangulate_points_multi(x_list, P_list)
    % x_list: cell array of 2D homogeneous points {x1, x2, x3}
    % P_list: cell array of 3x4 projection matrices {P1, P2, P3}
    % X: triangulated 3D point in world coordinates

    A = [];

    for i = 1:length(x_list)
        x = x_list{i};
        P = P_list{i};

        A = [A;
             x(1) * P(3,:) - P(1,:);
             x(2) * P(3,:) - P(2,:)];
    end

    [~, ~, V] = svd(A);
    X_hom = V(:, end);
    X = X_hom(1:3) / X_hom(4);
end

function scale = compute_shoulder_scale_2d(cam3_data, poseID)
    % Filter for current pose and keypoints
    idx_L = cam3_data(:,2) == poseID & cam3_data(:,3) == 11;
    idx_R = cam3_data(:,2) == poseID & cam3_data(:,3) == 12;

    % Get frame numbers where each shoulder appears
    frames_L = cam3_data(idx_L, 1);
    frames_R = cam3_data(idx_R, 1);
    common_frames = intersect(frames_L, frames_R);

    distances = [];

    for f = common_frames'
        xL = cam3_data(cam3_data(:,1) == f & cam3_data(:,3) == 11 & cam3_data(:,2) == poseID, 4:5);
        xR = cam3_data(cam3_data(:,1) == f & cam3_data(:,3) == 12 & cam3_data(:,2) == poseID, 4:5);

        if ~isempty(xL) && ~isempty(xR) && all(~isnan(xL)) && all(~isnan(xR))
            d = norm(xL - xR);
            if ~isnan(d)
                distances(end+1) = d;
            end
        end
    end

    % Return median distance if available, otherwise NaN
    if ~isempty(distances)
        scale = median(distances);
    else
        scale = NaN;
    end
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
    view(150, -10);

    % Define skeleton connections (based on MediaPipe)
    skeleton_connections = [ ...
        0 1; 1 2; 2 3; 3 7;  
        0 4; 4 5; 5 6; 6 8;  
        9 10; 
        11 12; 
        11 13; 13 15; % 15 17; 15 19; 17 19; 15 21;  
        12 14; 14 16; % 16 18; 16 20; 18 20; 16 22 
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