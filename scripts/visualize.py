import os
import sys
import cv2
import numpy as np
import pandas as pd
import argparse
import rerun as rr
import rerun.blueprint as rrb

# Define skeleton connections
POSE_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11),
                    (0, 11), (11, 12), (12, 13), (13, 14),
                    (11, 15), (15, 16), (16, 17), (17, 18)]

SCALE_FACTOR = 50

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    return fps, frame_count, duration

def visualize(video_path, experiment_name, skeletons, gaze_0, gaze_1, args):
    fps, frame_count, duration = get_video_info(video_path)
    print(f"FPS: {fps}, Total Frames: {frame_count}, Duration: {duration} sec")
    
    cap = cv2.VideoCapture(video_path)
        
    rr.script_setup(
        args,
        application_id=f"human_pose_{experiment_name}",
        default_blueprint=rrb.Blueprint(
            rrb.Vertical(
                rrb.Spatial2DView(origin=f"/image", name="Video"),
                rrb.Horizontal(
                    rrb.TextDocumentView(origin=f"/gaze_0/classification", name="Gaze 0"),
                    rrb.TextDocumentView(origin=f"/gaze_1/classification", name="Gaze 1"),
                ),
                row_shares=[2, 1]  # Allocate 75% of space to the video and 25% to gaze labels
            )
        ),
    )

    for frame_idx in range(frame_count):  # Iterate through total frame count
        success, frame = cap.read()
        if not success:
            break  # Stop when the video ends
        rr.set_time_sequence("frame", frame_idx)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rr.log(f"{experiment_name}/image", rr.Image(frame_rgb).compress(jpeg_quality=50))

        # Skeleton Data
        if "frame" in skeletons.columns:
            frame_keypoints = skeletons[skeletons["frame"] == frame_idx]
        else:
            frame_keypoints = skeletons.iloc[frame_idx % len(skeletons)]  # Approximate using index

        if not frame_keypoints.empty:
            keypoints_2d = frame_keypoints[["x", "y"]].values
            rr.log(f"{experiment_name}/pose", rr.Points2D(keypoints_2d))
            rr.log(f"{experiment_name}/skeleton", rr.LineStrips2D(np.array(POSE_CONNECTIONS)))

        # Gaze Data
        if "frame" in gaze_0.columns:
            frame_gaze_0 = gaze_0[gaze_0["frame"] == frame_idx]
        else:
            frame_gaze_0 = gaze_0.iloc[frame_idx % len(gaze_0)]

        if "frame" in gaze_1.columns:
            frame_gaze_1 = gaze_1[gaze_1["frame"] == frame_idx]
        else:
            frame_gaze_1 = gaze_1.iloc[frame_idx % len(gaze_1)]

        # Log gaze vectors
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

    print(f"Checking files in {folder}...")

    missing_files = [f for f in [video_path, skeletons_csv_path, gaze_0_csv_path, gaze_1_csv_path] if not os.path.exists(f)]

    if missing_files:
        print("Error: The following files are missing:")
        for f in missing_files:
            print(f" - {f}")
        sys.exit(1)

    print("All files found! Loading data...")

    skeletons = pd.read_csv(skeletons_csv_path)
    gaze_0 = pd.read_csv(gaze_0_csv_path)
    gaze_1 = pd.read_csv(gaze_1_csv_path)

    print(f"Loaded {len(skeletons)} skeleton frames, {len(gaze_0)} gaze_0 entries, {len(gaze_1)} gaze_1 entries.")

    parser = argparse.ArgumentParser()
    rr.script_add_args(parser)
    rerun_args, _ = parser.parse_known_args()
    
    print("Starting visualization...")
    visualize(video_path, experiment_name, skeletons, gaze_0, gaze_1, rerun_args)

    print("Visualization completed!")
    
if __name__ == "__main__":
    main()