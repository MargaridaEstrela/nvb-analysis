import os
import sys
import cv2
import numpy as np
import pandas as pd
import argparse
import rerun as rr
import rerun.blueprint as rrb

from lists import POSE_CONNECTIONS, CONNECTION_COLORS

SCALE_FACTOR = 50

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    return fps, frame_count, width, height, duration

def load_keypoints(skeletons_data):
    keypoints = {}
    for _, row in skeletons_data.iterrows():
        frame, pose_id, landmark_id, x, y, z = row
        if frame not in keypoints:
            keypoints[frame] = {}
        if pose_id not in keypoints[frame]:
            keypoints[frame][pose_id] = []
        keypoints[frame][pose_id].append((x, y, z))
    return keypoints

def visualize(video_path, experiment_name, skeletons, gaze_0, gaze_1, au_0, au_1, args):
    fps, frame_count, width, height, duration = get_video_info(video_path)
    print(f"FPS: {fps}, Total Frames: {frame_count}, Duration: {duration} sec")
    
    keypoints = load_keypoints(skeletons)
    
    cap = cv2.VideoCapture(video_path)
        
    rr.script_setup(
        args,
        application_id=f"human_pose_{experiment_name}",
        default_blueprint=rrb.Blueprint(
            rrb.Vertical(
                rrb.Spatial2DView(origin=f"/image", name="Video"),
                rrb.Horizontal(  # Horizontal row with two sections
                    rrb.Vertical(  # Left section: Gaze classification
                        rrb.Tabs(
                            rrb.TextDocumentView(origin=f"/gaze_0/classification", name="Gaze 0"),
                            rrb.TextDocumentView(origin=f"/gaze_1/classification", name="Gaze 1"),
                        ),
                    ),
                    rrb.Vertical(  # Right section: AU data inside tabs
                        rrb.Tabs(
                            rrb.TimeSeriesView(origin=f"AUs_0", name="AUs_0"),
                            rrb.TimeSeriesView(origin=f"AUs_1", name="AUs_1"),
                        ),
                    ),
                ),
            )
        ),
    )

    for frame_idx in range(frame_count):
        success, frame = cap.read()
        if not success:
            break  # Stop when the video ends
        
        rr.set_time_sequence("frame", frame_idx)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rr.log(f"{experiment_name}/image", rr.Image(frame_rgb).compress(jpeg_quality=50))
        
        # Gaze Data
        if "frame" in gaze_0.columns:
            frame_gaze_0 = gaze_0[gaze_0["frame"] == frame_idx]
        else:
            frame_gaze_0 = gaze_0.iloc[frame_idx % len(gaze_0)]

        if "frame" in gaze_1.columns:
            frame_gaze_1 = gaze_1[gaze_1["frame"] == frame_idx]
        else:
            frame_gaze_1 = gaze_1.iloc[frame_idx % len(gaze_1)]

        # Skeleton Data
        if frame_idx in keypoints:
            for pose_id, landmarks in keypoints[frame_idx].items():
                # Convert landmarks list to a dictionary for fast lookup
                landmark_dict = {i: (x * width, y * height) for i, (x, y, _) in enumerate(landmarks) if i < 22}
                
                rr.log(
                    f"{experiment_name}/pose_{pose_id}", 
                    rr.Points2D(
                        positions=np.array(list(landmark_dict.values())), 
                        colors=np.array([[1, 1, 1]] * len(landmark_dict)),  # White color
                        radii=5.0
                    )
                )
                # Log all keypoint connections
                connection_lines = []
                for idx, (p1, p2) in enumerate(POSE_CONNECTIONS):
                    if p1 in landmark_dict and p2 in landmark_dict and idx < len(CONNECTION_COLORS):
                        rr.log(
                            f"{experiment_name}/connections/pose_{pose_id}/connection_{p1}_{p2}",
                            rr.LineStrips2D(
                                [np.array([landmark_dict[p1], landmark_dict[p2]])],
                                colors=[CONNECTION_COLORS[idx]],  # Green color
                                radii=2.5  
                            )
                        )
                        
        # Gaze vectors
        if not frame_gaze_0.empty:
            gaze_start_0 = frame_gaze_0[["Gaze_Pos_X", "Gaze_Pos_Y"]].values
            gaze_direction_0 = frame_gaze_0[["Gaze_Dir_X", "Gaze_Dir_Y"]].values
            classification_0 = frame_gaze_0["Classification"]
            rr.log(f"{experiment_name}/gaze_0", rr.Arrows2D(origins=gaze_start_0, vectors=gaze_direction_0*SCALE_FACTOR, radii=3.0))
            rr.log("Gaze 0", rr.TextDocument(classification_0, media_type=rr.MediaType.MARKDOWN))
            
        if not frame_gaze_1.empty:
            gaze_start_1 = frame_gaze_1[["Gaze_Pos_X", "Gaze_Pos_Y"]].values
            gaze_direction_1 = frame_gaze_1[["Gaze_Dir_X", "Gaze_Dir_Y"]].values
            classification_1 = frame_gaze_1["Classification"]
            rr.log(f"{experiment_name}/gaze_1", rr.Arrows2D(origins=gaze_start_1, vectors=gaze_direction_1*SCALE_FACTOR, radii=3.0)) 
            rr.log("Gaze 1", rr.TextDocument(classification_1, media_type=rr.MediaType.MARKDOWN))

        # AU Data Logging
        if frame_idx in au_0["frame"].values:
            frame_au_0 = au_0[au_0["frame"] == frame_idx].drop(columns=["frame"])
            for au_name, au_value in frame_au_0.items():
                value = float(au_value.iloc[0])
                rr.log(f"AUs_0/{au_name}", rr.Scalar(value))

        if frame_idx in au_1["frame"].values:
            frame_au_1 = au_1[au_1["frame"] == frame_idx].drop(columns=["frame"])
            for au_name, au_value in frame_au_1.items():
                value = float(au_value.iloc[0])
                rr.log(f"AUs_1/{au_name}", rr.Scalar(value))
            
    cap.release()
    print(f"Pose overlay for {experiment_name} logged to Rerun!")

def main():
    if len(sys.argv) != 2:
        print("Usage: python visualize.py <experiment_id>")
        sys.exit(1)

    experiment_name = sys.argv[1]
    folder = os.path.join("../archive", experiment_name)

    video_path = os.path.join(folder, "piloto.mp4")
    skeletons_csv_path = os.path.join(folder, "pipeline.csv")
    gaze_0_csv_path = os.path.join(folder, "gaze_0.csv")
    gaze_1_csv_path = os.path.join(folder, "gaze_1.csv")
    au_0_csv_path = os.path.join(folder, "au_0.csv")
    au_1_csv_path = os.path.join(folder, "au_1.csv")

    print(f"Checking files in {folder}...")

    missing_files = [f for f in [video_path, skeletons_csv_path, gaze_0_csv_path, 
                                gaze_1_csv_path, au_0_csv_path, au_1_csv_path] if not os.path.exists(f)]

    if missing_files:
        print("Error: The following files are missing:")
        for f in missing_files:
            print(f" - {f}")
        sys.exit(1)

    print("All files found! Loading data...")

    skeletons = pd.read_csv(skeletons_csv_path)
    gaze_0 = pd.read_csv(gaze_0_csv_path)
    gaze_1 = pd.read_csv(gaze_1_csv_path)
    au_0 = pd.read_csv(au_0_csv_path)
    au_1 = pd.read_csv(au_1_csv_path)

    print(f"Loaded...")
    print(f"Skeleton frames: {len(skeletons)} ")
    print(f"gaze_0 entries: {len(gaze_0)}")
    print(f"Gaze_1 entries: {len(gaze_1)}")
    print(f"AU_0 entries: {len(au_0)}")
    print(f"AU_1 entries: {len(au_1)}")

    parser = argparse.ArgumentParser()
    rr.script_add_args(parser)
    rerun_args, _ = parser.parse_known_args()
    
    print("Starting visualization...")
    visualize(video_path, experiment_name, skeletons, gaze_0, gaze_1, au_0, au_1, rerun_args)

    print("Visualization completed!")
    
if __name__ == "__main__":
    main()