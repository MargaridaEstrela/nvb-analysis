import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from pose_metrics import PoseMetrics
from pose_tracking import PoseIDTracker
from keypoint_handler import KeypointHandler
from pose_landmark_extractor import PoseLandmarkExtractor


model = "pose_landmarker.task"

def extract_mediapipe_skeletons(data_path, sessionID):
    """
    Extracts mediapipe skeletons from the videos in the specified session directory
    """
    session_path = data_path / sessionID
    videos_path = session_path / "videos"

    expected_videos = {"top.mp4", "left.mp4", "right.mp4"}

    # Check if all expected videos exist
    existing_videos = set(os.listdir(videos_path))
    if not expected_videos.issubset(existing_videos):
        missing = expected_videos - existing_videos
        raise FileNotFoundError(f"Missing videos: {', '.join(missing)}")

    # Check if results folder exists
    results_path = session_path / "results"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Check if mediapipe folder exists
    mediapipe_path = results_path / "mediapipe"
    if not os.path.exists(mediapipe_path):
        os.makedirs(mediapipe_path)

    # Extract landmarks from each video
    for video in os.listdir(videos_path):
        print(video)
        if video in ["top.mp4", "left.mp4", "right.mp4"]:
            landmarks_path = mediapipe_path / f"{video.split('.')[0]}.csv"
            video_path = videos_path / video
            landmarks = PoseLandmarkExtractor(video_path, model, landmarks_path)
            landmarks.setup_csv()
            landmarks.extract_landmarks()

def track_poses(results_path):
    """
    Tracks poses using the extracted mediapipe landmarks and saves the results.
    """
    mediapipe_raw_paths = results_path / "mediapipe"
    tracker = PoseIDTracker()

    for landmarks_file in os.listdir(mediapipe_raw_paths):
        if landmarks_file in ["top.csv", "left.csv", "right.csv"]:
            path = mediapipe_raw_paths / landmarks_file
            tracker.run(path)

def apply_filters(data_path, sessionID):
    """
    Applies filters to the tracked poses and processes the keypoints.
    """
    pose0_3D = data_path / sessionID / "results" / "3D_pose_reconstruction_0.csv"
    pose1_3D = data_path / sessionID / "results" / "3D_pose_reconstruction_1.csv"

    keypoint_handler = KeypointHandler(pose0_3D, pose1_3D)
    keypoint_handler.process_data()
    pose0_3D, pose1_3D = keypoint_handler.get_processed_data()

    # Save processed DataFrames
    output_dir = data_path / sessionID / "results"
    pose0_3D.to_csv(output_dir / "pose0_processed.csv", index=False)
    pose1_3D.to_csv(output_dir / "pose1_processed.csv", index=False)

    return pose0_3D, pose1_3D

def sanitize_and_round(obj, ndigits=4):
    """
    Recursively sanitize and round numerical values.
    Handles NaN and Inf values by converting them to None.
    """
    if isinstance(obj, dict):
        return {k: sanitize_and_round(v, ndigits) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_and_round(v, ndigits) for v in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return round(obj, ndigits)
    elif isinstance(obj, (np.integer, np.floating)):
        return round(float(obj), ndigits)
    return obj

def read_frame_windows(csv_path):
    """
    Reads the frame windows from a CSV file and returns a dictionary mapping round labels to frame ranges.
    """
    df = pd.read_csv(csv_path)
    frame_windows = {}

    for i in range(len(df) - 1):
        label = df.loc[i, 'round']
        start_frame = df.loc[i, 'frame']
        end_frame = df.loc[i + 1, 'frame']
        frame_windows[label] = (start_frame, end_frame)

    # Add last window till end
    label = df.loc[len(df) - 1, 'round']
    frame_windows[label] = (df.loc[len(df) - 1, 'frame'], np.inf)

    return frame_windows

def metrics_per_round(results_path, time_window_csv):
    """
    Computes metrics for each round based on the provided time window CSV file.
    """
    frame_windows = read_frame_windows(time_window_csv)
    pose0 = pd.read_csv(results_path / "pose0_processed.csv")
    pose1 = pd.read_csv(results_path / "pose1_processed.csv")

    all_metrics = {}
    first_start = None

    for label, (start, end) in frame_windows.items():
        print(f"Processing {label} [{start}, {end})")

        if first_start is None:
            first_start = start

        d0 = pose0[(pose0['frame'] >= first_start) & (pose0['frame'] < end)]
        d1 = pose1[(pose1['frame'] >= first_start) & (pose1['frame'] < end)]

        if d0.empty or d1.empty:
            print(f"Skipping {label}: no data in range.")
            continue

        round_path = results_path / "rounds" / label.replace(" ", "_")
        round_path.mkdir(parents=True, exist_ok=True)
        d0.to_csv(round_path / "pose0_processed_com.csv", index=False)
        d1.to_csv(round_path / "pose1_processed_com.csv", index=False)

        metrics_obj = PoseMetrics(d0, d1, round_path)
        metrics_obj.compute_all_metrics()

        all_metrics[label] = {
            'frame_start': start,
            'frame_end': end,
            'shouldersAngle_pose0': metrics_obj.metrics0['shoulders_vector_angle']['mean'],
            'shouldersAngle_pose1': metrics_obj.metrics1['shoulders_vector_angle']['mean'],
            'deltaXMidPointShoulders_pose0': metrics_obj.metrics0['delta_shoulders_midpoint']['delta_x_mean'],
            'deltaXMidPointShoulders_pose1': metrics_obj.metrics1['delta_shoulders_midpoint']['delta_x_mean'],
            'deltaZMidPointShoulders_pose0': metrics_obj.metrics0['delta_shoulders_midpoint']['delta_z_mean'],
            'deltaZMidPointShoulders_pose1': metrics_obj.metrics1['delta_shoulders_midpoint']['delta_z_mean'],
            'speed_pose0': metrics_obj.metrics0['speed']['overall']['mean'],
            'speed_pose1': metrics_obj.metrics1['speed']['overall']['mean'],
            'mean_squared_jerk_pose0': metrics_obj.metrics0['mean_squared_jerk']['overall'],
            'mean_squared_jerk_pose1': metrics_obj.metrics1['mean_squared_jerk']['overall'],
            'distance': metrics_obj.other_metrics['distance']['mean']
        }

    return all_metrics

def metrics(results_path, time_window=None):
    """
    Computes metrics for the poses stored in the results path.
    If a time window is provided, it reads the frame windows from the CSV file.
    """
    pose0_csvPath = results_path / "pose0_processed.csv"
    pose1_csvPath = results_path / "pose1_processed.csv"

    pose0 = pd.read_csv(pose0_csvPath)
    pose1 = pd.read_csv(pose1_csvPath)

    frame_windows = read_frame_windows(time_window) if time_window is not None else None

    metrics = PoseMetrics(pose0, pose1, results_path, window_frames=frame_windows)
    metrics.compute_all_metrics()
    metrics.print_metrics()

    metrics.plot_shoulders_vector_angle()
    metrics.plot_upper_body_frame(frame=1000)
    metrics.plot_participants_delta_distance()
    metrics.plot_delta_displacement_keypoints()
    metrics.plot_speed_over_time()

def plot_metric_step_graph(all_metrics, key, save_path):
    labels = sorted(all_metrics.keys(), key=lambda l: all_metrics[l]['frame_start'])
    steps = [(all_metrics[label]['frame_start'], all_metrics[label][key]) for label in labels]
    ends = [all_metrics[label]['frame_end'] for label in labels]

    xs = []
    ys = []
    for (start, val), end in zip(steps, ends):
        xs.extend([start, end])
        ys.extend([val, val])

    plt.figure(figsize=(10, 5))
    plt.plot(xs, ys, drawstyle='steps-post', color='blue')
    plt.xlabel("Frame")
    plt.ylabel(key.replace('_', ' ').title())
    plt.title(f"{key.replace('_', ' ').title()} Mean per Round")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_metric_step_graph_dual(metrics, key_pose0, key_pose1, save_path, ylabel):
    rounds = list(metrics.keys())
    frame_starts = [metrics[r]['frame_start'] for r in rounds]
    frame_ends = [metrics[r]['frame_end'] for r in rounds]

    values0 = [metrics[r][key_pose0] for r in rounds]
    values1 = [metrics[r][key_pose1] for r in rounds]

    # Prepare X and Y for Pose 0
    xs0, ys0 = [], []
    for start, end, val in zip(frame_starts, frame_ends, values0):
        if not pd.isna(val):
            xs0.extend([start, end])
            ys0.extend([val, val])

    # Prepare X and Y for Pose 1
    xs1, ys1 = [], []
    for start, end, val in zip(frame_starts, frame_ends, values1):
        if not pd.isna(val):
            xs1.extend([start, end])
            ys1.extend([val, val])

    # Plot step graphs
    plt.figure(figsize=(12, 5))
    if xs0:
        plt.plot(xs0, ys0, drawstyle='steps-post', color='royalblue', linewidth=1.5, label='Pose 0')
    if xs1:
        plt.plot(xs1, ys1, drawstyle='steps-post', color='darkorange', linewidth=1.5, label='Pose 1')

    # Aesthetic improvements
    plt.xlabel("Frame", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f"{ylabel} per Round", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def export_rounds_metrics_to_csv(all_metrics, results_path, session_id):
    """
    Exports the computed metrics per round to a CSV file.
    """
    results_path = Path(results_path)
    rows = []
    for round_name, metrics in all_metrics.items():
        for pose in (0, 1):
            rows.append({
                "Session":                 session_id,
                "Round":                   round_name,
                "Participant":             pose,
                "ShouldersAngle":          metrics.get(f"shouldersAngle_pose{pose}"),
                "DeltaXMidPointShoulders": metrics.get(f"deltaXMidPointShoulders_pose{pose}"),
                "DeltaZMidPointShoulders": metrics.get(f"deltaZMidPointShoulders_pose{pose}"),
                "Speed":                   metrics.get(f"speed_pose{pose}"),
                "Jerk":                    metrics.get(f"mean_squared_jerk_pose{pose}"),
                "Distance":                metrics.get("distance"),
            })

    df = pd.DataFrame(rows)
    num_rounds = df["Round"].str.extract(r'(\d+)')
    if not num_rounds.isna().any().any():
        df["round_num"] = num_rounds.astype(int)
        df.sort_values(["round_num", "Participant"], inplace=True)
        df.drop(columns="round_num", inplace=True)

    # Save CSV file
    out_csv = results_path / "metrics_per_round.csv"
    df.to_csv(out_csv, index=False)
    print(f"Exported round metrics to {out_csv}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python pose_estimation.py <experimental_studies_path> <session_ID>")
        return
    
    data_path = Path(sys.argv[1])
    session_id = sys.argv[2]
    results_path = data_path / session_id / "results"

    ## [Step 1]: extract mediapipe skeletons and track poses (uncomment as needed)
    extract_mediapipe_skeletons(data_path, session_id)
    track_poses(results_path)

    ## [Step 2]: Apply filters to tracked poses (after computing depth in MATLAB; uncomment as needed)
    # pose0_3D, pose1_3D = apply_filters(data_path, session_id)

    ## [Step 3]: compute metrics (uncomment as needed)
    # figures_path = data_path / session_id / "results" / "figures"
    # if not figures_path.exists():
    #     figures_path.mkdir(parents=True, exist_ok=True)

    # metrics(results_path)

    ## [Step 4] (optional): compute metrics per round (uncomment as needed)
    # time_window = data_path / session_id / "results" / "frames_by_round.csv"
    # metrics_ = metrics_per_round(results_path, time_window)

    # metrics_ = sanitize_and_round(metrics_)
    # export_rounds_metrics_to_csv(metrics_, results_path, session_id)

    # plot_metric_step_graph_dual(metrics_, 'shouldersAngle_pose0', 'shouldersAngle_pose1', figures_path / "shouldersAngle_step.png", "Yaw (°)")
    # plot_metric_step_graph_dual(metrics_, 'deltaXMidPointShoulders_pose0', 'deltaXMidPointShoulders_pose1', figures_path / "deltaXMidPointShoulders_step.png", "ΔX Shoulders (m)")
    # plot_metric_step_graph_dual(metrics_, 'deltaZMidPointShoulders_pose0', 'deltaZMidPointShoulders_pose1', figures_path / "deltaZMidPointShoulders_step.png", "ΔZ Shoulders (m)")
    # plot_metric_step_graph(metrics_, 'distance', figures_path / "distance.png")

if __name__ == "__main__":
    main()
