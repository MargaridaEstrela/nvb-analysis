function plot_speed_trend(data, frame_rate, keypoint_names, max_landmark_id)
    % Calculate frame interval from frame rate
    frame_interval = 1 / frame_rate;

    pose_ids = unique(data.pose_id);
    landmark_ids = unique(data.landmark_id);
    landmark_ids = landmark_ids(landmark_ids <= max_landmark_id);
    num_landmarks = length(landmark_ids);
    
    % Determine grid size
    num_cols = ceil(sqrt(num_landmarks));
    num_rows = ceil(num_landmarks / num_cols);

    fig = figure;
    t = tiledlayout(num_rows, num_cols, 'TileSpacing', 'compact', 'Padding', 'loose');
    sgtitle(t, 'Speed Trend of Keypoints Over Time', 'FontSize', 14);

    for i = 1:num_landmarks
        nexttile;
        hold on;
        
        max_speed = 0; % Initialize max speed for dynamic ylim

        for j = 1:length(pose_ids)
            % Filter data for the current pose and landmark
            idx = (data.pose_id == pose_ids(j)) & (data.landmark_id == landmark_ids(i));
            frames = data.frame(idx);
            x = data.x(idx);
            y = data.y(idx);
            
            % Calculate Euclidean speed between consecutive (x, y) points
            if length(x) > 1
                speed = sqrt(diff(x).^2 + diff(y).^2) / frame_interval; % Euclidean speed with frame interval
                speed_frames = frames(1:end-1); % Adjust frames for diff operation
                
                % Smooth speed data (optional)
                smooth_speed = movmean(speed, 5); % Apply a moving average for smoothing
                
                % Update max speed for this landmark
                max_speed = max(max_speed, max(smooth_speed));
                
                % Plot the speed trend
                if pose_ids(j) == 0
                    plot(speed_frames, smooth_speed, 'Color', [0, 0, 1, 0.3], 'LineWidth', 1);
                else
                    plot(speed_frames, smooth_speed, 'Color', [1, 0, 0, 0.3], 'LineWidth', 1);
                end
            end
        end

        hold off;
        title(keypoint_names{landmark_ids(i) + 1}, 'FontSize', 10);
        % xlabel('Frame Index', 'FontSize', 8);
        % ylabel('Speed (units/second)', 'FontSize', 8);

        % Set dynamic y-axis limit for each subplot
        ylim([0, max_speed + 10]); % Add small padding to max speed
        yticks(linspace(0, max_speed + 10, 4)); % Use 4 y-ticks for simplicity
        ytickformat('%,.0f'); % Round y-tick values to the nearest integer
        xlim([0, max(frames)]);
        set(gca, 'FontSize', 7);
        grid on;
    end

        % Add a single y-axis label for the entire layout
    ylabel(t, 'Speed (units/second)', 'FontSize', 10);
    xlabel(t, 'Frame Index', 'FontSize', 10);

    % Add a hidden axis for the legend outside the tiled layout
    ax = axes(fig, 'Visible', 'off');
    hold(ax, 'on');
    plot(ax, NaN, NaN, 'b-', 'LineWidth', 1); % Dummy plot for Person 0
    plot(ax, NaN, NaN, 'r-', 'LineWidth', 1); % Dummy plot for Person 1
    lgd = legend(ax, {'Person 0', 'Person 1'}, 'Orientation', 'horizontal', 'FontSize', 10);
    lgd.Position = [0.82, 0.04, 0, 0]; % Position legend centered below the layout

    % Save the figure with high resolution
    print(fig, 'figures/speed_trend', '-dpng', '-r300'); % Save at 300 DPI
end
