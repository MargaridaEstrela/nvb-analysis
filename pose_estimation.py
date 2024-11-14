from pose_landmark_extractor import PoseLandmarkExtractor
from pose_metrics import PoseMetrics
from keypoint_handler import KeypointHandler
from keypoint_plotter import KeypointPlotter
import pandas as pd
import sys
import os

model = "pose_landmarker.task"

def main():
    if len(sys.argv) < 2:
        print("Usage: python pose_estimation.py <video_path>")
        return

    video_path = sys.argv[1]
    video_name = os.path.basename(video_path)
    output_csv_path = "csv/mediapipe.csv"
    
    # Step 1: Extract keypoints from video and save to CSV
    extractor = PoseLandmarkExtractor(video_path, model, output_csv_path)
    extractor.setup_csv()
    extractor.extract_landmarks()

    # Step 2: Load keypoints data from CSV and fine-tune it
    data = pd.read_csv(output_csv_path)
    handler = KeypointHandler(data)
    handler.process_data()
    processed_data = handler.get_processed_data()
    processed_data.to_csv("csv/pipeline.csv", index=False)

    # Step 3: Plot and animate keypoints data
    plotter = KeypointPlotter(processed_data)
    plotter.animate_skeletons()

    metrics = PoseMetrics(output_csv_path)
    print(metrics.get_metrics())


if __name__ == "__main__":
    main()
