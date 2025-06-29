import os
import cv2
import sys
import numpy as np
import pandas as pd
import argparse
import rerun as rr
import rerun.blueprint as rrb

def find_videos(folder):
    # Customize these names if needed
    video_top = os.path.join(folder, "videos/top.mp4")
    video_left = os.path.join(folder, "videos/left.mp4")
    video_right = os.path.join(folder, "videos/right.mp4")

    for path in [video_top, video_left, video_right]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing video: {path}")

    return video_top, video_left, video_right

def get_video_info(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return int(fps), int(frame_count)


def load_pose_data(folder):
    path0 = os.path.join(folder, "results", "pose0_processed.csv")
    path1 = os.path.join(folder, "results", "pose1_processed.csv")

    if not os.path.exists(path0) or not os.path.exists(path1):
        raise FileNotFoundError("Missing pose0 or pose1 CSV in 'results' folder.")

    df0 = pd.read_csv(path0)
    df1 = pd.read_csv(path1)

    return df0, df1

def log_3d_skeletons(pose_df, label, color, frame_idx):
    frame_data = pose_df[pose_df["frame"] == frame_idx]
    if frame_data.empty:
        return

    scale_factor = 100

    # Keypoint positions
    positions = frame_data[["x", "y", "z"]].to_numpy()
    positions = positions * scale_factor

    transform = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0,  1]
    ])
    
    positions = positions @ transform
    
    rr.log(f"/skeletons/{label}/keypoints", rr.Points3D(
        positions,
        colors=color,
        radii=5  # Adjust dot size
    ))

    # Skeleton connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12),
        (11, 13), (13, 15), (15, 17), (15, 19), (17, 19), (15, 21),
        (12, 14), (14, 16), (16, 18), (16, 20), (18, 20), (16, 22),
    ]

    lines = []
    
    for i, j in connections:
        try:
            p1 = frame_data.loc[frame_data["keypoint"] == i, ["x", "y", "z"]].values[0]
            p2 = frame_data.loc[frame_data["keypoint"] == j, ["x", "y", "z"]].values[0]
            p1 = (p1 * scale_factor) @ transform
            p2 = (p2 * scale_factor) @ transform
            lines.append(np.array([p1, p2]))
        except IndexError:
            continue  # One or both points missing

    if lines:
        rr.log(f"/skeletons/{label}/bones", rr.LineStrips3D(
            lines,
            colors=color,
            radii=2  # Thicker lines
        ))

        
def visualize(video_top, video_bottom_left, video_bottom_right, args, pose0_df, pose1_df):
    cap_top = cv2.VideoCapture(video_top)
    cap_left = cv2.VideoCapture(video_bottom_left)
    cap_right = cv2.VideoCapture(video_bottom_right)

    _, n_top = get_video_info(video_top)
    _, n_left = get_video_info(video_bottom_left)
    _, n_right = get_video_info(video_bottom_right)
    max_frames = max(n_top, n_left, n_right)

    rr.script_setup(
        args,
        application_id="video_grid_viewer",
        default_blueprint=rrb.Blueprint(
            rrb.Vertical(
                rrb.Spatial2DView(origin="/top", name="Top Video"),
                rrb.Horizontal(
                    rrb.Spatial2DView(origin="/left", name="Left Video"),
                    rrb.Spatial2DView(origin="/right", name="Right Video"),
                ),
                rrb.Horizontal(  # ⬅ NEW row for two 3D skeleton views
                    rrb.Spatial3DView(origin="/skeletons/pose0", name="Pose 0 (Left)"),
                    rrb.Spatial3DView(origin="/skeletons/pose1", name="Pose 1 (Right)"),
                )
            )
        ),
    )

    for frame_idx in range(max_frames):
        rr.set_time_sequence("frame", frame_idx)

        def read_frame(cap):
            success, frame = cap.read()
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if success else None

        top_frame = read_frame(cap_top)
        left_frame = read_frame(cap_left)
        right_frame = read_frame(cap_right)

        if top_frame is not None:
            rr.log("/top", rr.Image(top_frame).compress(jpeg_quality=75))
        if left_frame is not None:
            rr.log("/left", rr.Image(left_frame).compress(jpeg_quality=75))
        if right_frame is not None:
            rr.log("/right", rr.Image(right_frame).compress(jpeg_quality=75))

        # Log 3D skeletons
        log_3d_skeletons(pose0_df, "pose0", [255, 0, 0], frame_idx)
        log_3d_skeletons(pose1_df, "pose1", [0, 255, 0], frame_idx)

    print("Done logging video frames and 3D skeletons to Rerun.")
    cap_top.release()
    cap_left.release()
    cap_right.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_folder", help="Path to the experiment folder containing the videos and results")
    rr.script_add_args(parser)
    args = parser.parse_args()

    print(f"Loading experiment from: {args.experiment_folder}")

    try:
        video_top, video_left, video_right = find_videos(args.experiment_folder)
        pose0_df, pose1_df = load_pose_data(args.experiment_folder)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    visualize(video_top, video_left, video_right, args, pose0_df, pose1_df)


if __name__ == "__main__":
    main()