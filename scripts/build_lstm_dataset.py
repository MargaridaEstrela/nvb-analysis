import os
import sys
import pandas as pd
import numpy as np

from pathlib import Path
from pose_metrics import PoseMetrics

def get_pose_data(session_path):
    """ Extract the pose data from each participant for a given session. """
    pose0_path = session_path / "results" / "pose0_processed.csv"
    pose1_path = session_path / "results" / "pose1_processed.csv"

    if not os.path.exists(pose0_path) or not os.path.exists(pose1_path):
        print(f"[WARNING] Missing file(s) in {session_path.name}")
        return None, None

    try:
        pose0 = pd.read_csv(pose0_path)
        pose1 = pd.read_csv(pose1_path)
        return pose0, pose1
    except Exception as e:
        print(f"[ERROR] Failed to read files in {session_path.name}: {e}")
        return None, None

def align_frames(pose0, pose1, num_keypoints=17):
    """
    Aligns frames of pose0 and pose1 so that they have the same frame numbers.
    For any missing frame, insert rows for all keypoints with NaN in x, y, z.
    """
    all_frames = sorted(set(pose0['frame'].unique()).union(pose1['frame'].unique()))
    new_rows_0 = []
    new_rows_1 = []

    for frame in all_frames:
        if frame not in pose0['frame'].values:
            for k in range(num_keypoints):
                new_rows_0.append({'frame': frame, 'pose': 0, 'keypoint': k, 'x': np.nan, 'y': np.nan, 'z': np.nan})

        if frame not in pose1['frame'].values:
            for k in range(num_keypoints):
                new_rows_1.append({'frame': frame, 'pose': 1, 'keypoint': k, 'x': np.nan, 'y': np.nan, 'z': np.nan})

    # Append new rows if any
    if new_rows_0:
        pose0 = pd.concat([pose0, pd.DataFrame(new_rows_0)], ignore_index=True)
    if new_rows_1:
        pose1 = pd.concat([pose1, pd.DataFrame(new_rows_1)], ignore_index=True)

    # Sort by frame and keypoint to keep it tidy
    pose0 = pose0.sort_values(by=['frame', 'keypoint']).reset_index(drop=True)
    pose1 = pose1.sort_values(by=['frame', 'keypoint']).reset_index(drop=True)
    
    pose0.replace([np.inf, -np.inf], np.nan, inplace=True)
    pose1.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return pose0, pose1

def find_min_frames(sessions_path, session_ids):
    frame_counts = []
    
    for session_id in session_ids:
        session_path = sessions_path / f"{session_id}"

        pose0_path = session_path / "results" / "pose0.csv"
        pose1_path = session_path / "results" / "pose1.csv"

        if not os.path.exists(pose0_path) or not os.path.exists(pose1_path):
            continue

        try:
            pose0 = pd.read_csv(pose0_path)
            pose1 = pd.read_csv(pose1_path)

            if min(pose0['frame']) != min(pose1['frame']) or max(pose0['frame']) != max(pose1['frame']):
                print(f"[ERROR] Frame mismatch in {session_id}: pose0 has {len(pose0)}, pose1 has {len(pose1)}")
                continue

            frame_counts.append(max(pose0['frame']) - min(pose0['frame']) + 1)

        except Exception as e:
            print(f"[ERROR] Failed to process {session_id}: {e}")
            
    return frame_counts

def truncate_center(data, min_frames):
    """
    Truncates the DataFrame to min_frames (× num_keypoints) rows,
    centered in time.
    """
    total_frames = data['frame'].nunique()
    start_frame = (total_frames - min_frames) // 2
    end_frame = start_frame + min_frames
    selected_frames = sorted(data['frame'].unique())[start_frame:end_frame]
    data = data[data['frame'].isin(selected_frames)].copy()

    # Reset frame numbers to start from 0
    frame_map = {old: new for new, old in enumerate(sorted(data['frame'].unique()))}
    data['frame'] = data['frame'].map(frame_map)

    # Re-sort
    data = data.sort_values(by=["frame", "keypoint"]).reset_index(drop=True)
    return data

def normalize_frames(sessions_path, session_ids, min_frames):
    for session_id in session_ids:
        session_path = sessions_path / f"{session_id}"

        pose0_path = session_path / "results" / "pose0.csv"
        pose1_path = session_path / "results" / "pose1.csv"

        if not os.path.exists(pose0_path) or not os.path.exists(pose1_path):
            continue

        try:
            pose0 = pd.read_csv(pose0_path)
            pose1 = pd.read_csv(pose1_path)
            
            # Truncate center and reset frame numbers
            pose0 = truncate_center(pose0, min_frames)
            pose1 = truncate_center(pose1, min_frames)

            # Save back
            pose0.to_csv(pose0_path, index=False)
            pose1.to_csv(pose1_path, index=False)

        except Exception as e:
            print(f"[ERROR] Failed to normalize frames for {session_id}: {e}")
            
    return pose0, pose1 

def interpolate_keypoints(data):
    all_processed = []
    for kpt in data['keypoint'].unique():
        kpt_data = data[data['keypoint'] == kpt].sort_values('frame').copy()
        kpt_data[['x', 'y', 'z']] = kpt_data[['x', 'y', 'z']].interpolate(
            method='linear', limit_direction='both'
        )
        all_processed.append(kpt_data)
    return pd.concat(all_processed).sort_values(['frame', 'keypoint']).reset_index(drop=True)
    
def compute_displacement_velocity(pose, fps=30, flip_x=False):
    """
    Computes displacement and velocity per keypoint and returns a DataFrame
    with one row per frame, and columns disp_0, vel_0, ..., disp_16, vel_16.
    
    If flip_x=True, the X coordinate is negated before computing displacement.
    Useful for aligning inward/outward movements across symmetrical positions.
    """
    dt = 1.0 / fps
    result_rows = []

    pose = pose.copy()
    if flip_x:
        pose['x'] = -pose['x']  # invert X direction to mirror movement

    for kpt in pose['keypoint'].unique():
        kpt_data = pose[pose['keypoint'] == kpt].copy()
        kpt_data.sort_values('frame', inplace=True)

        displacement = np.sqrt(
            (kpt_data['x'].diff() ** 2) +
            (kpt_data['y'].diff() ** 2) +
            (kpt_data['z'].diff() ** 2)
        ).fillna(0.0)

        velocity = (displacement / dt).fillna(0.0)

        kpt_data[f'disp_{kpt}'] = displacement
        kpt_data[f'vel_{kpt}'] = velocity

        result_rows.append(kpt_data[['frame', f'disp_{kpt}', f'vel_{kpt}']])

    # Merge all keypoint data by frame
    df_merged = result_rows[0][['frame']]  # start from frame column
    for df in result_rows:
        df_merged = df_merged.merge(df, on='frame', how='left')

    return df_merged.sort_values('frame').reset_index(drop=True)


def main():
    if len(sys.argv) != 2:
        print("Usage: python build_lstm_dataset.py <sessions_path>")
        sys.exit(1)

    sessions_p = sys.argv[1]
    sessions_path = Path(sessions_p)
    session_ids = range(1, 32)
    
    for session_id in session_ids:
        if not (sessions_path / f"{session_id}").exists():
            print(f"[WARNING] Session {session_id} does not exist in {sessions_path}")
            continue

        pose0, pose1 = get_pose_data(sessions_path / f"{session_id}")
        pose0, pose1 = align_frames(pose0, pose1)
            
        # Save the new csv
        pose0_out_path = sessions_path / f"{session_id}" / "results" / "pose0.csv"
        pose1_out_path = sessions_path / f"{session_id}" / "results" / "pose1.csv"
        pose0.to_csv(pose0_out_path, index=False)
        pose1.to_csv(pose1_out_path, index=False)
        
    frame_counts = find_min_frames(sessions_path, session_ids)
    print(f"Minimum number of frames across sessions: {min(frame_counts)}")
    
    # Create a folder to save all the results from pose metrics per session
    pose_metrics_path = sessions_path / "pose_metrics /"
    pose_metrics_path.mkdir(parents=True, exist_ok=True)
    
    openface_path = sessions_path / f"{session_id}" / "results" / "openface"
    
    for session_id in session_ids:
        pose0, pose1 = normalize_frames(sessions_path, session_ids, min(frame_counts))
        pose0 = interpolate_keypoints(pose0)
        pose1 = interpolate_keypoints(pose1)
        
        pose0_metrics = compute_displacement_velocity(pose0, flip_x=False)
        pose1_metrics = compute_displacement_velocity(pose1, flip_x=True)
        
        # Load AU and gaze for each person
        au0 = pd.read_csv(openface_path / "au_0.csv")
        au1 = pd.read_csv(openface_path / "au_1.csv")
        # gaze0 = pd.read_csv(openface_path / "gaze_0.csv")
        # gaze1 = pd.read_csv(openface_path / "gaze_1.csv")

        # Merge them all on frame (or drop "frame" after aligning)
        merged0 = pose0_metrics.merge(au0, on="frame", how="left").fillna(0.0)
        merged1 = pose1_metrics.merge(au1, on="frame", how="left").fillna(0.0)
            
        merged0.drop(columns=["frame"]).to_csv(pose_metrics_path / f"session{session_id}_0.csv", index=False)
        merged1.drop(columns=["frame"]).to_csv(pose_metrics_path / f"session{session_id}_1.csv", index=False)
    
if __name__ == "__main__":
    main()