from pose_landmark_extractor import PoseLandmarkExtractor
from pose_metrics import PoseMetrics
from keypoint_handler import KeypointHandler
from keypoint_plotter import KeypointPlotter
import pandas as pd
import cv2
import sys

model = "pose_landmarker.task"

def main():
    if len(sys.argv) < 3:
        print("Usage: python pose_estimation.py <video_path> <mediapipe_csv_path>")
        return
    
    video_path = sys.argv[1]
    mediapipe_csv_path = sys.argv[2]

    # Load keypoints data from CSV and fine-tune it
    # data = pd.read_csv(mediapipe_csv_path)
    # data = data.dropna(subset=["x", "y", "z"]).copy()
    # handler = KeypointHandler(data)
    # handler.process_data()
    # processed_data = handler.get_processed_data()
    # processed_data.to_csv("pipeline.csv", index=False)

    # lot and animate keypoints data
    data = pd.read_csv(mediapipe_csv_path)
    data = data.dropna(subset=["x", "y", "z"]).copy()
    plotter = KeypointPlotter(data)
    # plotter.animate_skeletons()
    
    cap = cv2.VideoCapture(video_path)
    
    # Get the original video details
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create a video writer to save the output with skeletons
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("../figures/video_with_skeletons.mp4", fourcc, fps, (frame_width, frame_height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw skeletons on the frame
        frame_with_skeletons = plotter.plot_skeletons_on_frame(frame, frame_idx)
        out.write(frame_with_skeletons)
        frame_idx += 1

    cap.release()
    out.release()

    # metrics = PoseMetrics(output_csv_path)
    # print(metrics.get_metrics())


if __name__ == "__main__":
    main()
