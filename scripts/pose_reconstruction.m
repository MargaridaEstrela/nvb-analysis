clear; close all; clc;

% Define path to stereo calibration parameters
path = "../../../experimental_studies/gaips/matlab_calibrations/colmap/";
stereo_params = load_and_extract(path + "stereoParams_G1_G2.mat");

[P1, P2] = compute_camera_matrix(stereo_params);


%% Handle MediaPipe Data
% Set image resolution (used for converting normalized coords to pixel coords)
image_width = 1920;
image_height = 1080;

% Load MediaPipe CSV keypoint data from both camera views
data_cam1 = readmatrix('../../../experimental_studies/gaips/2/results/G1_mediapipe_fixed.csv'); % Change to the right path
data_cam2 = readmatrix('../../../experimental_studies/gaips/2/results/G2_mediapipe_fixed.csv'); % Change to the right path
data_cam3 = readmatrix('../../../experimental_studies/gaips/2/results/G3_mediapipe_fixed.csv'); % Change to the right path

% Convert normalized coords from MediaPipe to pixels
data_cam1(:,4:5) = data_cam1(:,4:5) .* [image_width, image_height];
data_cam2(:,4:5) = data_cam2(:,4:5) .* [image_width, image_height];
data_cam1(:,5) = image_height - data_cam1(:,5);
data_cam2(:,5) = image_height - data_cam2(:,5);


%% Compute Fundamental Matrix
[F, E] = compute_fundamental_and_essential(data_cam1, data_cam2, stereo_params, 0, 8);
disp('Fundamental Matrix:'), disp(F)
disp('Essential Matrix:'), disp(E)


%% Triangulate keypoints
% Reconstruct 3D keypoints for pose 0 and pose 1
pose0_3D = triangulate_pose(0, data_cam1, data_cam2, P1, P2, F);
pose1_3D = triangulate_pose(1, data_cam1, data_cam2, P1, P2, F);


%% Correct skeletons using symmetric joint references
% Get per-keypoint confidence from data
conf0 = extract_confidence(0, data_cam1, data_cam2);
conf1 = extract_confidence(1, data_cam1, data_cam2);

pose0_3D(:,4:6) = correct_low_confidence_symmetric(pose0_3D(:,4:6), conf0, 0.2);
pose1_3D(:,4:6) = correct_low_confidence_symmetric(pose1_3D(:,4:6), conf1, 0.2);


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


function [P1, P2] = compute_camera_matrix(stereoParams)
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


function conf = extract_confidence(poseID, data1, data2)
    N = 33;  % Total number of keypoints
    conf = zeros(N,1);

    for k = 0:N-1
        c1 = mean(data1(data1(:,2)==poseID & data1(:,3)==k, 7), 'omitnan');
        c2 = mean(data2(data2(:,2)==poseID & data2(:,3)==k, 7), 'omitnan');
        conf(k+1) = mean([c1 c2], 'omitnan');  % merge both cameras
    end
end


function [F, E] = compute_fundamental_and_essential(data_cam1, data_cam2, stereoParams, poseID, num_points)
    % Estimate F and E matrices from top-N confident correspondences across frames

    % Convert normalized coordinates to pixels
    data1 = data_cam1;
    data2 = data_cam2;

    % Filter only poseID entries
    data1 = data1(data1(:,2) == poseID, :);
    data2 = data2(data2(:,2) == poseID, :);

    % Accumulate confident matches across frames
    matches = [];
    frames = intersect(unique(data1(:,1)), unique(data2(:,1)));

    for f = frames'
        d1 = data1(data1(:,1) == f, :);
        d2 = data2(data2(:,1) == f, :);
        [common_kpts, ia, ib] = intersect(d1(:,3), d2(:,3));

        if numel(common_kpts) < 1
            continue;
        end

        for i = 1:numel(common_kpts)
            pt1 = d1(ia(i), 4:5);
            pt2 = d2(ib(i), 4:5);
        
            if any(isnan(pt1)) || any(isnan(pt2))
                continue;  % skip invalid points
            end
        
            c = (d1(ia(i), 7) + d2(ib(i), 7)) / 2;
            matches = [matches; c, pt1, pt2];
        end
    end

    % Sort by confidence and take top-N
    matches = sortrows(matches, -1);  % descending
    matches = matches(1:min(num_points, size(matches,1)), :);

    disp("Number of matches used:");
    disp(size(matches,1));
    disp("Matches preview:");
    disp(matches(:,2:5));

    pts1 = matches(:,2:3);
    pts2 = matches(:,4:5);

    % Sanity check before estimating F
    if size(matches,1) < 8
        error("Not enough valid matches to estimate F (got %d)", size(matches,1));
    end
    
    if any(isnan(matches(:)))
        error("Matches contain NaNs. Check landmark filtering.");
    end

    % Estimate Fundamental Matrix
    F = estimateFundamentalMatrix(pts1, pts2, ...
        'Method', 'Norm8Point', 'NumTrials', 2000, 'DistanceThreshold', 1);

    % Compute Essential Matrix: E = K2' * F * K1
    K1 = stereoParams.CameraParameters1.IntrinsicMatrix';
    K2 = stereoParams.CameraParameters2.IntrinsicMatrix';
    E = K2' * F * K1;
end


function pose_3D = triangulate_pose(poseID, data_cam1, data_cam2, P1, P2, F)
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

    xy1 = [data_cam1(:,4), data_cam1(:,5)];
    xy2 = [data_cam2(:,4), data_cam2(:,5)];

    % Get confidence values for each landmark
    conf1 = data_cam1(:,7);
    conf2 = data_cam2(:,7);

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
                % Check confidence
                c1 = mean(conf1(match1));
                c2 = mean(conf2(match2));

                if c1 < min_confidence || c2 < min_confidence
                    continue;
                end

                x1 = mean(xy1(match1, :), 1);
                x2 = mean(xy2(match2, :), 1);

                if c1 < 0.2 && c2 >= 0.5
                    % Refine x1 using epipolar line from x2
                    x1 = refine_point_with_epipolar_line(x2, x1, F');  % Use F' to invert direction
                elseif c2 < 0.2 && c1 >= 0.5
                    % Refine x2 using epipolar line from x1
                    x2 = refine_point_with_epipolar_line(x1, x2, F);
                end

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


function x2_refined = refine_point_with_epipolar_line(x1, x2_noisy, F)
    % x1: [x, y] from camera 1 (high confidence)
    % x2_noisy: [x, y] from camera 2 (low confidence)
    % F: fundamental matrix
    % Output: x2_refined: [x, y] corrected point on the epipolar line

    % Compute epipolar line in image 2 for point in image 1
    x1_hom = [x1(:); 1];  % 3x1
    l2 = F * x1_hom;      % 3x1 line in form [a, b, c]

    % Line: a*x + b*y + c = 0
    a = l2(1); b = l2(2); c = l2(3);

    % Project x2_noisy onto epipolar line
    x0 = x2_noisy(1);
    y0 = x2_noisy(2);

    % Projection of (x0, y0) onto line ax + by + c = 0
    d = (a*x0 + b*y0 + c) / (a^2 + b^2);
    x_proj = x0 - a * d;
    y_proj = y0 - b * d;

    x2_refined = [x_proj, y_proj];
end


function pose_corrected = correct_low_confidence_symmetric(pose, conf, threshold)
% pose : N x 3 matrix with landmark coordinates (x, y, z)
% conf : N x 1 vector of confidence values
% threshold : minimum confidence required to keep the landmark
% pose_corrected : corrected pose (N x 3)

    pose_corrected = pose;

    % Define symmetric limbs: [target, mirror, parent]
    limbs = [
        13, 14, 11;
        14, 13, 12;
        15, 16, 13;
        16, 15, 14;
        17, 18, 15;
        18, 17, 16;
        19, 20, 17;
        20, 19, 18;
        21, 22, 17;
        22, 21, 18;
        23, 24, 11;
        24, 23, 11;
        25, 26, 23;
        26, 25, 24;
        27, 28, 25;
        28, 27, 26;
    ];

    for i = 1:size(limbs,1)
        target = limbs(i,1);
        mirror = limbs(i,2);
        parent = limbs(i,3);

        if conf(target+1) < threshold && ...
           conf(mirror+1) >= threshold && ...
           conf(parent+1) >= threshold

            % Get mirrored vector
            v = pose(mirror+1,:) - pose(parent+1,:);
            v_mirror = -v;

            % Reconstruct target landmark position
            pose_corrected(target+1,:) = pose(parent+1,:) + v_mirror;
        end
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
    view(0, 15);

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