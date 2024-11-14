function plot_y_coordinate_trend(data, frame_height, keypoint_names, max_landmark_id)
    pose_ids = unique(data.pose_id);
    landmark_ids = unique(data.landmark_id);
    landmark_ids = landmark_ids(landmark_ids <= max_landmark_id);
    num_landmarks = length(landmark_ids);
    
    % Determine grid size
    num_cols = ceil(sqrt(num_landmarks));
    num_rows = ceil(num_landmarks / num_cols);

    fig = figure;
    t = tiledlayout(num_rows, num_cols, 'TileSpacing', 'tigh', 'Padding', 'compact');
    sgtitle(t, 'Y Coordinate Trend of Keypoints Over Time', 'FontSize', 12);

    for i = 1:num_landmarks
        nexttile;
        hold on;
        for j = 1:length(pose_ids)
            idx = (data.pose_id == pose_ids(j)) & (data.landmark_id == landmark_ids(i));
            frames = data.frame(idx);
            y = data.y(idx);
            
            if pose_ids(j) == 0
                plot(frames, y, 'b-', 'LineWidth', 1);
            else
                plot(frames, y, 'r-', 'LineWidth', 1);
            end
        end
        hold off;
        title(keypoint_names{landmark_ids(i) + 1}, 'FontSize', 10);
        xlabel('Frame Index', 'FontSize', 8);
        ylabel('Y Coordinate', 'FontSize', 8);
        xlim([0, max(frames)]);
        ylim([0, frame_height]);
        set(gca, 'FontSize', 7);
        grid on;
    end

    % Add a hidden axis for the legend outside the tiled layout
    ax = axes(fig, 'Visible', 'off');
    hold(ax, 'on');
    plot(ax, NaN, NaN, 'b-', 'LineWidth', 1); % Dummy plot for Person 0
    plot(ax, NaN, NaN, 'r-', 'LineWidth', 1); % Dummy plot for Person 1
    lgd = legend(ax, {'Person 0', 'Person 1'}, 'Orientation', 'horizontal', 'FontSize', 10);
    lgd.Position = [0.5, 0.015, 0, 0]; % Position legend centered below the layout

    % Save the figure with high resolution
    print(fig, 'figures/y_coordinate_trend', '-dpng', '-r300'); % Save at 300 DPI
end
