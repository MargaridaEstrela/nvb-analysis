function plot_keypoint_trajectories(data, frame_width, frame_height, keypoint_names, max_landmark_id)
    pose_ids = unique(data.pose_id);
    landmark_ids = unique(data.landmark_id);

    % Filter landmark IDs to include only those up to max_keypoint_id
    landmark_ids = landmark_ids(landmark_ids <= max_landmark_id);
    num_landmarks = length(landmark_ids);

    % Calculate grid size
    num_cols = ceil(sqrt(num_landmarks));
    num_rows = ceil(num_landmarks / num_cols);

    fig = figure;
    t = tiledlayout(num_rows, num_cols, 'TileSpacing', 'compact', 'Padding', 'compact');
    sgtitle(t, 'Keypoint Trajectories in 2D Space for Both Persons', 'FontSize', 12);
    
    for i = 1:num_landmarks
        nexttile;
        hold on;
        for j = 1:length(pose_ids)
            idx = (data.pose_id == pose_ids(j)) & (data.landmark_id == landmark_ids(i));
            x = data.x(idx);
            y = data.y(idx);
            
            if pose_ids(j) == 0
                plot(x, y, 'b-', 'LineWidth', 1);
            else
                plot(x, y, 'r-', 'LineWidth', 1);
            end
        end
        hold off;
        title(keypoint_names{landmark_ids(i) + 1}, 'FontSize', 10);
        xlabel('X Coordinate', 'FontSize', 8);
        ylabel('Y Coordinate', 'FontSize', 8);
        xlim([0, frame_width]);
        ylim([0, frame_height]);
        set(gca, 'YDir', 'reverse'); % Reverse Y-axis direction
        set(gca, 'XAxisLocation', 'top'); % Move X-axis to the top
        set(gca, 'FontSize', 7);
        grid on;
    end
    
    % Add a hidden axis to hold the legend
    ax = axes(fig, 'Visible', 'off');
    ax.Position = [0.1, 0.05, 0.8, 0.1];
    hold(ax, 'on');
    plot(ax, NaN, NaN, 'b-', 'LineWidth', 1); % Dummy plot for Person 0
    plot(ax, NaN, NaN, 'r-', 'LineWidth', 1); % Dummy plot for Person 1
    lgd = legend(ax, {'Person 0', 'Person 1'}, 'Orientation', 'horizontal', 'FontSize', 10);
    lgd.Position = [0.5, 0.02, 0, 0]; % Adjust to place it centered below the tiled layout

    % Save the figure with high resolution
    print(fig, 'figures/keypoint_trajectories', '-dpng', '-r300'); % Save at 300 DPI
end
