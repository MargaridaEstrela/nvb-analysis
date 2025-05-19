import os
import sys

from pathlib import Path
from pose_metrics import PoseMetrics
from pose_tracking import PoseIDTracker
from keypoint_handler import KeypointHandler
from pose_landmark_extractor import PoseLandmarkExtractor


model = "pose_landmarker.task"

def extract_mediapipe_skeletons(data_path, sessionID):
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
    mediapipe_raw_paths = results_path / "mediapipe"
    tracker = PoseIDTracker()

    for landmarks_file in os.listdir(mediapipe_raw_paths):
        if landmarks_file in ["top.csv", "left.csv", "right.csv"]:
            path = mediapipe_raw_paths / landmarks_file
            tracker.run(path)

def apply_filters(data_path, sessionID):
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

def metrics(results_path):
    pose0_csvPath = results_path / "pose0_processed.csv"
    pose1_csvPath = results_path / "pose1_processed.csv"

    metrics = PoseMetrics(pose0_csvPath, pose1_csvPath, results_path)
    metrics.compute_all_metrics()
    metrics.print_metrics()

    metrics.plot_yaw_over_time()
    metrics.plot_upper_body_frame(frame=1000)
    metrics.plot_proximity_over_time()
    metrics.plot_displacement_all_keypoints_initial_point()

def main():
    if len(sys.argv) < 3:
        print("Usage: python pose_estimation.py <experimental_studies_path> <session_ID>")
        return
    
    data_path = Path(sys.argv[1])
    sessionID = sys.argv[2]
    results_path = data_path / sessionID / "results"
    
    # extract_mediapipe_skeletons(data_path, sessionID)
    # track_poses(results_path)

    pose0_3D, pose1_3D = apply_filters(data_path, sessionID)
    metrics(results_path)

if __name__ == "__main__":
    main()
