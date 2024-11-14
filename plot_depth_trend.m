function plot_depth_trend(data)
    pose_ids = unique(data.pose_id);
    
    fig = figure;
    global_min_depth = inf;
    global_max_depth = -inf;
    hold on;

    % Loop through each person and calculate the average depth per frame
    for j = 1:length(pose_ids)
        % Filter data for the current person
        person_data = data(data.pose_id == pose_ids(j), :);
        
        % Get unique frames and calculate the average depth per frame
        unique_frames = unique(person_data.frame);
        avg_depth = arrayfun(@(f) mean(person_data.z(person_data.frame == f)), unique_frames);
        
        % Interpolate missing frames to create a continuous line
        all_frames = min(unique_frames):max(unique_frames); % Full range of frames
        avg_depth_interp = interp1(unique_frames, avg_depth, all_frames, 'linear', 'extrap');

        % Update the global min and max depth values
        global_min_depth = min(global_min_depth, min(avg_depth_interp));
        global_max_depth = max(global_max_depth, max(avg_depth_interp));
        
        % Plot the interpolated average depth
        if pose_ids(j) == 0
            plot(all_frames, avg_depth_interp, 'b-', 'LineWidth', 1.5);
            plot(all_frames, avg_depth_interp, 'r-', 'LineWidth', 1.5);
        end
    end
    
    hold off;

    % Add title and labels
    title('Depth Trend Over Frames for Each Person', 'FontSize', 12);
    xlabel('Frame Index', 'FontSize', 10);
    ylabel('Average Z Coordinate', 'FontSize', 10);
    
    % Set a consistent Y-axis limit
    ylim([global_min_depth - 0.1, global_max_depth + 0.1]);
    xlim([0, max(all_frames)]);
    
    grid on;
    set(gca, 'FontSize', 9);
    legend('Person 0', 'Person 1', 'Location', 'northeast'); % Position legend inside top-right corner

    % Adjust plot margins to avoid clipping of labels
    set(gca, 'LooseInset', get(gca, 'TightInset') + [0.02, 0.02, 0.02, 0.02]);

    % Save the figure with high resolution
    print(fig, 'figures/depth_trend', '-dpng', '-r300'); % Save at 300 DPI
end
