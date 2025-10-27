import re
import os
import sys
import glob
import pandas as pd
import numpy as np

from pathlib import Path
from pose_metrics import PoseMetrics

# MediaPipe Pose (upper body)
UPPER_BODY_KPS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# nose, eyes(1–6), ears(7–8), mouth(9–10), shoulders(11–12), elbows(13–14), wrists(15–16)

# Window config
WINDOW_SIZE  = 30   # 30 frames ~= 1s @ 30fps
WINDOW_STRIDE = 30  # 50% overlap

PORTION = "full"  # "full", "second_half", "last_three_quarters", "last_two_thirds"

####################################
#### Data loading and normalization
####################################
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

def find_min_frames_from_rounds(sessions_path, session_ids, csv_name="frames_by_round.csv"):
    """
    For each session, read the frame_by_rounds.csv file and compute the total
    number of frames spanned by all rounds. Return the minimum total across sessions.
    """
    min_total_frames = None
    start_frame_per_session = {}

    for session_id in session_ids:
        csv_path = sessions_path / f"{session_id}" / "results" / csv_name
        if not csv_path.exists():
            print(f"[WARNING] Missing {csv_name} in session {session_id}")
            continue

        try:
            df = pd.read_csv(csv_path)
            frames = df["frame"].values

            if len(frames) < 16:
                print(f"[WARNING] Less than 16 rounds in session {session_id}")
                continue

            # Total span is from mean of round 1 to End Game
            round1_frame = frames[0]
            round16_frame = frames[15]
            total_frames = round16_frame - round1_frame + 1

            if min_total_frames is None or total_frames < min_total_frames:
                min_total_frames = total_frames

            start_frame_per_session[session_id] = round1_frame

        except Exception as e:
            print(f"[ERROR] Could not read {csv_name} in session {session_id}: {e}")
            continue

    return min_total_frames, start_frame_per_session

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

def truncate_from_start(data, start_frame, total_frames):
    """
    Truncates the data starting from a specific frame and spanning `total_frames` unique frames.
    """
    unique_frames = sorted(data['frame'].unique())
    if start_frame not in unique_frames:
        raise ValueError(f"Start frame {start_frame} not found in data.")

    start_idx = unique_frames.index(start_frame)
    selected_frames = unique_frames[start_idx:start_idx + total_frames]
    
    if len(selected_frames) < total_frames:
        raise ValueError("Not enough frames to truncate after start_frame")

    data = data[data['frame'].isin(selected_frames)].copy()

    # Reset frame numbers to start from 0
    frame_map = {old: new for new, old in enumerate(selected_frames)}
    data['frame'] = data['frame'].map(frame_map)

    # Re-sort
    data = data.sort_values(by=["frame", "keypoint"]).reset_index(drop=True)
    return data, selected_frames[0], selected_frames[-1]

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
    
    print(f"[INFO] Truncated data to [{start_frame}:{end_frame}]:")

    # Reset frame numbers to start from 0
    frame_map = {old: new for new, old in enumerate(sorted(data['frame'].unique()))}
    data['frame'] = data['frame'].map(frame_map)

    # Re-sort
    data = data.sort_values(by=["frame", "keypoint"]).reset_index(drop=True)
    return data

def normalize_frames_from_rounds(sessions_path, session_ids, min_frames, start_frames_dict):
    
    if PORTION == "second_half":
        skip_frames = min_frames // 2
        keep_frames = min_frames // 2
    elif PORTION == "last_three_quarters":
        skip_frames = min_frames // 4  # Skip first 25%
        keep_frames = (min_frames * 3) // 4  # Keep last 75%
    elif PORTION == "last_two_thirds":
        skip_frames = min_frames // 3  # Skip first 33%
        keep_frames = (min_frames * 2) // 3  # Keep last 67%
    elif PORTION == "full":
        skip_frames = 0
        keep_frames = min_frames
    else:
        raise ValueError(f"Unknown portion: {PORTION}")
    
    for session_id in session_ids:
        session_path = sessions_path / f"{session_id}"
        pose0_path = session_path / "results" / "pose0.csv"
        pose1_path = session_path / "results" / "pose1.csv"

        if not os.path.exists(pose0_path) or not os.path.exists(pose1_path):
            continue

        try:
            pose0 = pd.read_csv(pose0_path)
            pose1 = pd.read_csv(pose1_path)

            original_start_frame = start_frames_dict[session_id]
            new_start_frame = original_start_frame + skip_frames

            pose0, start0, end0 = truncate_from_start(pose0, new_start_frame, keep_frames)
            pose1, start1, end1 = truncate_from_start(pose1, new_start_frame, keep_frames)

            print(f"[INFO] Truncated session {session_id} frames {start0}–{end0}")

            # Save back
            pose0.to_csv(pose0_path, index=False)
            pose1.to_csv(pose1_path, index=False)

        except Exception as e:
            print(f"[ERROR] Failed to normalize frames for {session_id}: {e}")

def normalize_frames(sessions_path, session_ids, min_frames):
    for session_id in session_ids:
        print(f"[INFO] Normalizing frames for session {session_id}...")
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

def interpolate_keypoints(data):
    all_processed = []
    for kpt in data['keypoint'].unique():
        kpt_data = data[data['keypoint'] == kpt].sort_values('frame').copy()
        kpt_data[['x', 'y', 'z']] = kpt_data[['x', 'y', 'z']].interpolate(
            method='linear', limit_direction='both'
        )
        all_processed.append(kpt_data)
    return pd.concat(all_processed).sort_values(['frame', 'keypoint']).reset_index(drop=True)

def interpolate_to_wide_format(df):
    """ Converts a DataFrame with columns ['frame', 'keypoint', 'x', 'y', 'z']
        to a wide format DataFrame with columns like ['frame', 'x_0', 'y_0', 'z_0', ..., 'x_16', 'y_16', 'z_16'].
    """
    # Use 'frame' if it exists, otherwise use 'pose'
    time_col = 'frame' if 'frame' in df.columns else 'pose'
    df_coords = df[[time_col, 'keypoint', 'x', 'y', 'z']]

    # Pivot table to wide format: (x, 0) -> x_0
    df_wide = df_coords.pivot(index=time_col, columns='keypoint', values=['x', 'y', 'z'])

    df_wide.columns = [f"{coord}_{kpt}" for coord, kpt in df_wide.columns]
    df_wide = df_wide.reset_index()
    
    return df_wide

####################################
####### Feature computations #######
####################################
def central_diff(arr, dt):
    v = np.empty_like(arr)
    v[1:-1] = (arr[2:] - arr[:-2]) / (2*dt)
    v[0]    = (arr[1]  - arr[0])   / dt
    v[-1]   = (arr[-1] - arr[-2])  / dt
    return v

def compute_centroid_wide(wide_df, kps=(11,12)):
    cols = []
    for k in kps:
        cols += [f"x_{k}", f"y_{k}", f"z_{k}"]
    xyz = wide_df[cols].to_numpy().reshape(len(wide_df), len(kps), 3)
    return xyz.mean(axis=1)  # (T,3)

def shoulder_yaw(wide_df, flip_x=False):
    x11, x12 = wide_df["x_11"].to_numpy(), wide_df["x_12"].to_numpy()
    z11, z12 = wide_df["z_11"].to_numpy(), wide_df["z_12"].to_numpy()
    if flip_x: x11, x12 = -x11, -x12
    dx = x12 - x11; dz = z12 - z11
    return np.arctan2(dz, dx)

def head_velocities(wide_df, fps=30, flip_x=False):
    dt = 1.0/fps
    y = wide_df["y_0"].to_numpy()
    x = wide_df["x_0"].to_numpy();  x = -x if flip_x else x
    vy = np.diff(y, prepend=y[0]) / dt
    vx = np.diff(x, prepend=x[0]) / dt
    return vy, vx

def hand_raise(wide_df, flip_x=False):
    # Y-up → positive means wrist higher than shoulder
    l = wide_df["y_15"] - wide_df["y_11"]
    r = wide_df["y_16"] - wide_df["y_12"]
    # (flip_x has no effect on Y)
    return l, r

def cross_reach(self_wide, other_wide):
    c_other = compute_centroid_wide(other_wide, kps=(11,12))
    lw = np.linalg.norm(
        np.stack([
            self_wide["x_15"].to_numpy() - c_other[:,0],
            self_wide["y_15"].to_numpy() - c_other[:,1],
            self_wide["z_15"].to_numpy() - c_other[:,2],
        ], axis=1), axis=1)
    rw = np.linalg.norm(
        np.stack([
            self_wide["x_16"].to_numpy() - c_other[:,0],
            self_wide["y_16"].to_numpy() - c_other[:,1],
            self_wide["z_16"].to_numpy() - c_other[:,2],
        ], axis=1), axis=1)
    return lw, rw

def interpersonal_distance(p0_wide, p1_wide, kps=(11,12)):
    c0 = compute_centroid_wide(p0_wide, kps)   # (T,3)
    c1 = compute_centroid_wide(p1_wide, kps)   # (T,3)
    delta = c0 - c1
    dcentroid = np.linalg.norm(delta, axis=1)
    return pd.DataFrame({
        "frame": p0_wide["frame"].values,
        "d_centroid": dcentroid,
        "delta_x": delta[:,0], "delta_y": delta[:,1], "delta_z": delta[:,2],
    })

def session_center_lateral_depth(wide_df):
    """
    Remove seat bias: subtract per-session median from X and Z for ALL keypoints.
    (Keeps Y absolute for height; interpersonal computed separately.)
    """
    # find kps present
    kp_ids = sorted({
        int(m.group(1))
        for c in wide_df.columns
        for m in [re.match(r'^[xyz]_(\d+)$', c)]
        if m
    })
    centered = wide_df.copy()
    # medians from centroids (robust), but apply to all keypoints’ X,Z
    centroid = compute_centroid_wide(wide_df, kps=(11,12))
    med_x = np.median(centroid[:,0])
    med_z = np.median(centroid[:,2])
    for k in kp_ids:
        centered[f"x_{k}"] = centered[f"x_{k}"] - med_x
        centered[f"z_{k}"] = centered[f"z_{k}"] - med_z
        # y stays as-is
    return centered

def make_pair_features(feat0, feat1, ip_df):
    """
    Build order-invariant pair features: [mean || absdiff] + interpersonal (deltas).
    Assumes feat0/feat1 have a 'frame' column and same rows/frames.
    """
    f0 = feat0.sort_values("frame").reset_index(drop=True)
    f1 = feat1.sort_values("frame").reset_index(drop=True)
    assert np.all(f0["frame"].values == f1["frame"].values), "Frame mismatch in pair features"

    # Align numeric columns (exclude 'frame')
    num0 = f0.select_dtypes(include=[np.number]).drop(columns=[], errors="ignore")
    num1 = f1.select_dtypes(include=[np.number]).drop(columns=[], errors="ignore")
    # Make sure they share same columns
    shared_cols = sorted(set(num0.columns) & set(num1.columns) - {"frame"})
    num0 = num0[shared_cols]
    num1 = num1[shared_cols]

    mean_df    = 0.5 * (num0.values + num1.values)
    absdiff_df = np.abs(num0.values - num1.values)
    mean_cols    = [f"pair_mean__{c}" for c in shared_cols]
    absdiff_cols = [f"pair_abs__{c}"  for c in shared_cols]

    pair = pd.concat([
        pd.DataFrame({"frame": f0["frame"].values}),
        pd.DataFrame(mean_df, columns=mean_cols),
        pd.DataFrame(absdiff_df, columns=absdiff_cols),
    ], axis=1)

    # add interpersonal (already has frame)
    pair = pair.merge(ip_df, on="frame", how="left")

    # safety
    pair = pair.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pair

def compute_features(wide_df, fps=30, baseline_mode="first", baseline_sec=1.0, flip_x=False, center_lateral_depth=True):
    """
    Per-person features:
        - disp/vel per keypoint (seat-offset neutral if center_lateral_depth=True)
        - drifts (relative to early baseline)
        - raw AUs (absolute)
    """
    df = wide_df.sort_values("frame").reset_index(drop=True).copy()
    if center_lateral_depth:
        df_for_dynamics = session_center_lateral_depth(df)
    else:
        df_for_dynamics = df

    out = {"frame": df["frame"].values}

    # baseline length
    baseline_len = 1 if baseline_mode == "first" else max(1, int(round(baseline_sec * fps)))

    kp_ids = sorted({
        int(m.group(1))
        for c in df.columns
        for m in [re.match(r'^[xyz]_(\d+)$', c)]
        if m
    })

    dt = 1.0 / float(fps)

    for k in kp_ids:
        # dynamics from centered coordinates (neutral to seat)
        x = df_for_dynamics[f"x_{k}"].to_numpy()
        y = df_for_dynamics[f"y_{k}"].to_numpy()
        z = df_for_dynamics[f"z_{k}"].to_numpy()
        if flip_x:
            x = -x

        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        dz = np.diff(z, prepend=z[0])

        disp = np.sqrt(dx*dx + dy*dy + dz*dz)
        vel  = disp / dt

        out[f"disp_{k}"] = disp
        out[f"vel_{k}"]  = vel

        # drifts relative to early baseline (use original df, not centered)
        x0 = df[f"x_{k}"][:baseline_len].mean()
        y0 = df[f"y_{k}"][:baseline_len].mean()
        z0 = df[f"z_{k}"][:baseline_len].mean()
        x_abs = df[f"x_{k}"].to_numpy()
        y_abs = df[f"y_{k}"].to_numpy()
        z_abs = df[f"z_{k}"].to_numpy()
        if flip_x:
            x_abs = -x_abs

        out[f"xdrift_{k}"] = x_abs - x0
        out[f"ydrift_{k}"] = y_abs - y0
        out[f"zdrift_{k}"] = z_abs - z0

    # keep raw AUs (absolute)
    au_cols = [c for c in df.columns if c.startswith("AU") and not c.endswith("_c")]
    for c in au_cols:
        out[c] = df[c].to_numpy()

    feat = pd.DataFrame(out)
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feat

def temporal_dynamics_wide(pose_wide: pd.DataFrame, fps: int = 30, flip_x: bool = False) -> pd.DataFrame:
    """
    Compute displacement and velocity per keypoint from wide-format pose data.
    Input columns: ['frame','x_0','y_0','z_0', ..., 'x_16','y_16','z_16']
    Output: ['frame','disp_0','vel_0', ..., 'disp_16','vel_16']
    """
    if "frame" not in pose_wide.columns:
        raise ValueError("Expected 'frame' column in pose_wide")

    # detect keypoint ids present
    kp_ids = sorted({
        int(m.group(1))
        for c in pose_wide.columns
        for m in [re.match(r"^[xyz]_(\d+)$", c)]
        if m
    })

    dt = 1.0 / float(fps)
    out = {"frame": pose_wide["frame"].values}

    for k in kp_ids:
        x = pose_wide[f"x_{k}"].values
        y = pose_wide[f"y_{k}"].values
        z = pose_wide[f"z_{k}"].values

        if flip_x:
            x = -x

        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        dz = np.diff(z, prepend=z[0])

        disp = np.sqrt(dx*dx + dy*dy + dz*dz)
        vel  = disp / dt

        out[f"disp_{k}"] = disp
        out[f"vel_{k}"]  = vel

    return pd.DataFrame(out)

def temporal_to_wide(dynamics_long: pd.DataFrame) -> pd.DataFrame:
    # columns: frame,keypoint,disp,vel  ->  disp_0,... vel_16
    wide = dynamics_long.pivot(index='frame', columns='keypoint', values=['disp','vel'])
    wide.columns = [f"{m}_{k}" for m, k in wide.columns]
    return wide.reset_index()
    
def compute_au_delta(au_df):
    """
    Computes the delta for each AU column in the DataFrame.
    The delta is the difference between the current and previous frame for each AU.
    """
    au_df = au_df.sort_values("frame").reset_index(drop=True)
    au_cols = [col for col in au_df.columns if col.startswith('AU')]
    au_df[au_cols] = au_df[au_cols].diff().fillna(0.0)
    
    return au_df

def window_aggregate(df, window_size, stride):
    """
    Aggregate per-frame table into windows along 'frame'.
    Returns per-window mean & std for all numeric cols except 'frame', plus window metadata.
    Assumes frames are consecutive; if not, it reindexes and fills with 0.0.
    """
    if "frame" not in df.columns:
        raise ValueError("Expected 'frame' column in dataframe")
    df = df.sort_values("frame").reset_index(drop=True)

    # Ensure consecutive frames
    f = df["frame"].to_numpy()
    if len(f) == 0:
        return pd.DataFrame()
    if not np.all(np.diff(f) == 1):
        full = np.arange(int(f.min()), int(f.max()) + 1)
        df = df.set_index("frame").reindex(full).reset_index().rename(columns={"index": "frame"})
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "frame" in num_cols:
        num_cols.remove("frame")

    rows = []
    T = len(df)
    win_id = 0
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size  # exclusive
        chunk = df.iloc[start:end]
        
        meta = {"window_id": win_id}
        mean_ = chunk[num_cols].mean()
        rows.append(pd.concat([pd.Series(meta), mean_]))
        win_id += 1

    return pd.DataFrame(rows)

def to_partner_frame(self_wide: pd.DataFrame,
                     partner_wide: pd.DataFrame,
                     up_axis=np.array([0.0, 1.0, 0.0])) -> pd.DataFrame:
    """
    Re-express `self_wide` coordinates in a frame rooted at the partner's
    shoulder centroid with axes:
      x̂ = (right_shoulder - left_shoulder) (partner's left→right)
      ŷ = global up (given by up_axis)
      ẑ = ŷ × x̂  (approx. forward/back)

    Returns a copy of `self_wide` with x_*, y_*, z_* projected onto [x̂, ŷ, ẑ].
    Assumes frames are aligned and wide format with columns x_k,y_k,z_k.
    """
    eps = 1e-9
    self_w = self_wide.sort_values("frame").reset_index(drop=True).copy()
    part_w = partner_wide.sort_values("frame").reset_index(drop=True).copy()

    if not np.array_equal(self_w["frame"].values, part_w["frame"].values):
        raise ValueError("Frame mismatch between self and partner.")

    # partner shoulders + centroid
    L = np.stack([part_w["x_11"], part_w["y_11"], part_w["z_11"]], axis=1)
    R = np.stack([part_w["x_12"], part_w["y_12"], part_w["z_12"]], axis=1)
    C = 0.5 * (L + R)                         # (T,3)
    xvec = R - L                              # (T,3)
    xhat = xvec / (np.linalg.norm(xvec, axis=1, keepdims=True) + eps)

    yhat = np.tile(up_axis / (np.linalg.norm(up_axis) + eps), (len(xhat), 1))
    zhat = np.cross(yhat, xhat)
    zhat /= (np.linalg.norm(zhat, axis=1, keepdims=True) + eps)
    # re-orthogonalize y to ensure a right-handed basis
    yhat = np.cross(xhat, zhat)
    yhat /= (np.linalg.norm(yhat, axis=1, keepdims=True) + eps)

    kp_ids = sorted({
        int(m.group(1))
        for c in self_w.columns
        for m in [re.match(r'^[xyz]_(\d+)$', c)]
        if m
    })

    out = self_w.copy()
    for k in kp_ids:
        P = np.stack([self_w[f"x_{k}"], self_w[f"y_{k}"], self_w[f"z_{k}"]], axis=1)  # (T,3)
        rel = P - C
        out[f"x_{k}"] = np.sum(rel * xhat, axis=1)
        out[f"y_{k}"] = np.sum(rel * yhat, axis=1)
        out[f"z_{k}"] = np.sum(rel * zhat, axis=1)
    return out

####################################
####### Main processing loop #######
####################################
def main():
    if len(sys.argv) != 2:
        print("Usage: python build_lstm_dataset.py <sessions_path>")
        sys.exit(1)

    sessions_p = sys.argv[1]
    sessions_path = Path(sessions_p)
    session_ids = [i for i in range(1, 32) if i != 23]
    
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


    min_frames, start_frames_dict = find_min_frames_from_rounds(sessions_path, session_ids)
    print(f"[INFO] Minimum total frames from rounds: {min_frames}")
    normalize_frames_from_rounds(sessions_path, session_ids, min_frames, start_frames_dict)

    # Create a folder to save all the results from pose metrics per session
    pose_metrics_path = sessions_path / f"pose_metrics_truncated_{PORTION}"
    if not pose_metrics_path.exists():
        pose_metrics_path.mkdir(parents=True, exist_ok=True)
        
    # Create folders inside pose_metrics_path
    relative_path = pose_metrics_path / "relative"
    relative_path.mkdir(parents=True, exist_ok=True)
    
    per_frame_path = pose_metrics_path / "per_frame"
    per_frame_path.mkdir(parents=True, exist_ok=True)
    
    for session_id in session_ids:
        pose0_path = sessions_path / f"{session_id}" / "results" / "pose0.csv"
        pose1_path = sessions_path / f"{session_id}" / "results" / "pose1.csv"
        openface_path = sessions_path / f"{session_id}" / "results" / "openface"

        if not pose0_path.exists() or not pose1_path.exists():
            print(f"[WARNING] Missing normalized pose files for session {session_id}")
            continue

        pose0 = pd.read_csv(pose0_path)
        pose1 = pd.read_csv(pose1_path)

        pose0 = interpolate_keypoints(pose0)
        pose1 = interpolate_keypoints(pose1)
        
        pose0 = interpolate_to_wide_format(pose0)
        pose1 = interpolate_to_wide_format(pose1)
        
        # --- partner-centric coordinates (subtract + rotate into partner basis) ---
        pose0_rel = to_partner_frame(pose0, pose1)  # self in partner-0 frame (partner = pose1)
        pose1_rel = to_partner_frame(pose1, pose0)  # self in partner-1 frame (partner = pose0)

        # --- temporal dynamics on partner-relative coords (no mirroring) ---
        dyn0 = temporal_dynamics_wide(pose0_rel, fps=30, flip_x=False)
        dyn1 = temporal_dynamics_wide(pose1_rel, fps=30, flip_x=False)
        pose0_dyn = pose0_rel.merge(dyn0, on='frame', how='left')
        pose1_dyn = pose1_rel.merge(dyn1, on='frame', how='left')

        # --- AUs (absolute, unchanged) ---
        au0 = pd.read_csv(openface_path / "au_0.csv")
        au1 = pd.read_csv(openface_path / "au_1.csv")
        au0 = au0.loc[:, ~au0.columns.str.endswith('_c')].drop_duplicates(subset='frame')
        au1 = au1.loc[:, ~au1.columns.str.endswith('_c')].drop_duplicates(subset='frame')

        merged0 = pose0_dyn.merge(au0, on="frame", how="left").fillna(0.0)
        merged1 = pose1_dyn.merge(au1, on="frame", how="left").fillna(0.0)

        # --- per-person features in partner frame ---
        feat0 = compute_features(merged0, fps=30, baseline_mode="first",
                                 baseline_sec=1.0, flip_x=False, center_lateral_depth=False)
        feat1 = compute_features(merged1, fps=30, baseline_mode="first",
                                 baseline_sec=1.0, flip_x=False, center_lateral_depth=False)

        # Keep ABSOLUTE cues if you still want them (camera-space):
        feat0["shoulder_yaw"] = shoulder_yaw(pose0, flip_x=False)
        feat1["shoulder_yaw"] = shoulder_yaw(pose1, flip_x=True)
        
        vy0, vx0 = head_velocities(pose0, fps=30, flip_x=False)
        vy1, vx1 = head_velocities(pose1, fps=30, flip_x=True)
        feat0["head_vy"] = vy0; feat0["head_vx"] = vx0
        feat1["head_vy"] = vy1; feat1["head_vx"] = vx1
        
        l0, r0 = hand_raise(pose0)
        l1, r1 = hand_raise(pose1)
        feat0["hand_raise_L"] = l0; feat0["hand_raise_R"] = r0
        feat1["hand_raise_L"] = l1; feat1["hand_raise_R"] = r1

        # --- interpersonal distance remains absolute (socially meaningful) ---
        ip = interpersonal_distance(pose0, pose1, kps=(11,12))

        # --- order-invariant pair features (mean || absdiff) + interpersonal ---
        pair = make_pair_features(feat0, feat1, ip)

        # # --- temporal dynamics (per-person) ---
        # dyn0 = temporal_dynamics_wide(pose0, fps=30, flip_x=False)
        # dyn1 = temporal_dynamics_wide(pose1, fps=30, flip_x=True)
        # pose0_dyn = pose0.merge(dyn0, on='frame', how='left')
        # pose1_dyn = pose1.merge(dyn1, on='frame', how='left')

        # # --- AUs (absolute) ---
        # au0 = pd.read_csv(openface_path / "au_0.csv")
        # au1 = pd.read_csv(openface_path / "au_1.csv")
        # au0 = au0.loc[:, ~au0.columns.str.endswith('_c')].drop_duplicates(subset='frame')
        # au1 = au1.loc[:, ~au1.columns.str.endswith('_c')].drop_duplicates(subset='frame')

        # merged0 = pose0_dyn.merge(au0, on="frame", how="left").fillna(0.0)
        # merged1 = pose1_dyn.merge(au1, on="frame", how="left").fillna(0.0)

        # # --- per-person hybrid features (relative dynamics + absolute AUs) ---
        # feat0 = compute_features(merged0, fps=30, baseline_mode="first", baseline_sec=1.0, flip_x=False, center_lateral_depth=True)
        # feat1 = compute_features(merged1, fps=30, baseline_mode="first", baseline_sec=1.0, flip_x=True,  center_lateral_depth=True)

        # feat0["shoulder_yaw"] = shoulder_yaw(pose0, flip_x=False)
        # feat1["shoulder_yaw"] = shoulder_yaw(pose1, flip_x=True)
        
        # vy0, vx0 = head_velocities(pose0, fps=30, flip_x=False)
        # vy1, vx1 = head_velocities(pose1, fps=30, flip_x=True)
        
        # feat0["head_vy"] = vy0; feat0["head_vx"] = vx0
        # feat1["head_vy"] = vy1; feat1["head_vx"] = vx1

        # l0, r0 = hand_raise(pose0); l1, r1 = hand_raise(pose1)
        # feat0["hand_raise_L"] = l0; feat0["hand_raise_R"] = r0
        # feat1["hand_raise_L"] = l1; feat1["hand_raise_R"] = r1

        # # # cross-person reach (kept absolute, good social cue)
        # # lw0, rw0 = cross_reach(pose0, pose1)
        # # lw1, rw1 = cross_reach(pose1, pose0)
        # # feat0["reach_L_to_partner"] = lw0; feat0["reach_R_to_partner"] = rw0
        # # feat1["reach_L_to_partner"] = lw1; feat1["reach_R_to_partner"] = rw1

        # # --- interpersonal (absolute, socially meaningful) ---
        # ip = interpersonal_distance(pose0, pose1, kps=(11,12))

        # # --- order-invariant pair features (mean || absdiff) + interpersonal ---
        # pair = make_pair_features(feat0, feat1, ip)
        
        # # ---- windowed aggregates from per-frame features ----
        # feat0_win = window_aggregate(feat0, WINDOW_SIZE, WINDOW_STRIDE)
        # feat1_win = window_aggregate(feat1, WINDOW_SIZE, WINDOW_STRIDE)
        # pair_win  = window_aggregate(pair,  WINDOW_SIZE, WINDOW_STRIDE)
        
        # Save per-person per-frame
        feat0.to_csv(relative_path / f"session{session_id}_0.csv", index=False)
        feat1.to_csv(relative_path / f"session{session_id}_1.csv", index=False)

        # Save pair (per-frame)
        pair.to_csv(relative_path / f"session{session_id}_pair.csv", index=False)

        # # Save windowed
        # feat0_win.to_csv(windowed_path / f"session{session_id}_0.csv", index=False)
        # feat1_win.to_csv(windowed_path / f"session{session_id}_1.csv", index=False)
        # pair_win.to_csv(windowed_path / f"session{session_id}_pair.csv", index=False)
        
if __name__ == "__main__":
    main()
