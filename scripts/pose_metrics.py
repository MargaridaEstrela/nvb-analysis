import os
import cv2
import numpy as np
import pandas as pd
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

from tqdm import tqdm

class PoseMetrics:
    """
    Compute and export pose comparison metrics for two synchronized participants.

    Args:
        data0: keypoint trajectories for pose 0 (cols: frame, pose, keypoint, x, y, z)
        data1: keypoint trajectories for pose 1
        results_path: base path where figures/CSVs will be saved
        window_frames: named annotation frames (initial and end frame) for each window
        fs: sampling frequency (frames per second)
    """

    def __init__(self, data0, data1, results_path, window_frames=None):
        self.data0 = data0
        self.data1 = data1
        self.results_path = results_path
        self.window_frames = window_frames

        # Convert dataframes to centimeters
        # self.data0[['x', 'y', 'z']] *= 100
        # self.data1[['x', 'y', 'z']] *= 100

        # Metrics initialization
        self.inter_dist = None
        self.yaw_0 = None
        self.yaw_1 = None
        self.deltaX_0 = None
        self.deltaX_1 = None
        self.deltaZ_0 = None
        self.deltaZ_1 = None
        self.speed_0 = None
        self.speed_1 = None
        self.jerk_0 = None
        self.jerk_1 = None
        self.metrics0 = None
        self.metrics1 = None
        self.other_metrics = None
        self.disp0 = None
        self.disp1 = None
    
    @staticmethod
    def keypoints_dict():
        return {
            0: 'nose', 2: 'left_eye', 5: 'right_eye',
            7: 'left_ear', 8: 'right_ear',
            9: 'left_mouth', 10: 'right_mouth',
            11: 'left_shoulder', 12: 'right_shoulder',
            13: 'left_elbow', 14: 'right_elbow',
            15: 'left_wrist', 16: 'right_wrist',
        }

    @staticmethod
    def skeleton_connections():
        return [
            (11, 13), (13, 15),     # Left arm
            (12, 14), (14, 16),     # Right arm
            (11, 12),               # Shoulders
            (0, 2), (0, 5),         # Nose to eyes
            (2, 7), (5, 8),         # Eyes to ears
            (9, 10)                 # Mouth
        ]

    @staticmethod
    def compute_deltas_and_velocities(df, fps=30):
        dt = 1.0 / fps
        df = df.fillna(0.0)

        # Only xyz columns
        pos_cols = [col for col in df.columns if any(axis in col for axis in ['_x', '_y', '_z'])]
        data = df[pos_cols].values

        # Frame-to-frame deltas (one less row)
        deltas = np.diff(data, axis=0)
        velocities = deltas / dt

        # Speed: norm of velocity vector for each keypoint (groups of 3)
        speeds = np.linalg.norm(velocities.reshape(len(velocities), -1, 3), axis=2)

        # Flatten speeds (concat as last column)
        features = np.concatenate([deltas, velocities, speeds], axis=1)

        return features

    def _get_keypoint_trajectory(self, data, keypoint_id):
        return data[data['keypoint'] == keypoint_id].iloc[:, 3:6].values

    def _get_keypoint_by_frame(self, data, keypoint_id, frame):
        point = data[(data['keypoint'] == keypoint_id) & (data['frame'] == frame)].iloc[:, 3:6].values
        return point.flatten() if point.size == 3 else np.array([])

    def get_metric_series(self):
        """
        Returns a DataFrame with metrics per frame.
        Columns:
            - yaw_0, yaw_1: Orientation (degrees)
            - deltaX_0, deltaX_1: Horizontal displacement (cm)
            - deltaZ_0, deltaZ_1: Depth displacement (cm)
            - speed_0, speed_1: Movement speed (cm/s)
            - jerk_0, jerk_1: Rate of change of acceleration (cm/s³)
            - interpersonal_distance: Distance between participants (cm)

        Index: frame number (0-n_frames)
        """

        data_dict = {
            "interpersonal_distance": self.inter_dist,
            "yaw_0": self.yaw_0,
            "yaw_1": self.yaw_1,
            "deltaX_0": self.deltaX_0,
            "deltaX_1": self.deltaX_1,
            "deltaZ_0": self.deltaZ_0,
            "deltaZ_1": self.deltaZ_1,
            "speed_0": self.speed_0,
            "speed_1": self.speed_1,
            "jerk_0": self.jerk_0,
            "jerk_1": self.jerk_1,
        }

        n_frames = len(self.inter_dist)
        return pd.DataFrame(data_dict, index=np.arange(n_frames))

    def _angle_with_front_view(self, vectors, pose_id):
        """
        Computes the signed yaw angle between a set of vectors and the front-facing camera axis (-X).
        The sign is flipped for pose 0 to ensure inward movement is considered positive.
        """
        front_axis = np.array([-1, 0, 0])       # The camera looks along -X
        up_axis = np.array([0, 1, 0])

        v_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        dot = np.clip(np.einsum('ij,j->i', v_norm, front_axis), -1.0, 1.0)
        angles = np.degrees(np.arccos(dot))

        cross = np.cross(front_axis, v_norm)
        signs = np.sign(np.einsum('ij,j->i', cross, up_axis))
        signed_angles = angles * signs

        # Flip sign for Pose 0 to make "turning inside" positive
        signed_angles = angles * signs

        # Compute "yaw inwardness": 180 - |signed_angle|, but keep the sign
        deviation = (180.0 - np.abs(signed_angles)) * np.sign(signed_angles)

        # Flip sign for Pose 0 so that turning toward the center is positive
        if pose_id == 0:
            deviation = -deviation

        return deviation

    # -------------------------------------------------------------------------
    # Displacement & Midpoint Metrics
    # -------------------------------------------------------------------------
    def _compute_delta_keypoints(self, data, pose_id):
        """
        Computes the displacement from the initial position for each keypoint.
        Returns mean and amplitude along each axis (x, y, z).
        """
        summary = {}
        keypoints = self.keypoints_dict()

        for kpt_id, kpt_name in keypoints.items():
            coords = data[data['keypoint'] == kpt_id].iloc[:, 3:6].values
            coords = coords[~np.isnan(coords).any(axis=1)]

            if coords.shape[0] < 2:
                summary[kpt_name] = {
                    'mean_dx': np.nan, 'mean_dy': np.nan, 'mean_dz': np.nan,
                    'amp_dx': np.nan, 'amp_dy': np.nan, 'amp_dz': np.nan
                }
                continue

            initial = coords[0]
            deltas = coords - initial  # shape (n_frames, 3)

            mean_disp = np.mean(deltas, axis=0)
            amp_disp = np.ptp(deltas, axis=0)  # max - min

            summary[kpt_name] = {
                'mean_dx': float(mean_disp[0]),
                'mean_dy': float(mean_disp[1]),
                'mean_dz': float(mean_disp[2]),
                'amp_dx': float(amp_disp[0]),
                'amp_dy': float(amp_disp[1]),
                'amp_dz': float(amp_disp[2]),
            }

        # Compute delta X and Z for shoulders midpoint
        left = self._get_keypoint_trajectory(data, 11)  # left shoulder
        right = self._get_keypoint_trajectory(data, 12)  # right shoulder
        if left.shape[0] > 0 and right.shape[0] > 0:
            midpoints = (left + right) / 2
            delta_x = midpoints[:, 0] - midpoints[0, 0]
            delta_z = midpoints[:, 2] - midpoints[0, 2]

            if pose_id == 0:
                self.deltaX_0 = delta_x
                self.deltaZ_0 = delta_z
            else:
                self.deltaX_1 = delta_x
                self.deltaZ_1 = delta_z

        return summary

    def _compute_delta_shoulders_midpoint(self, disp_dict, pose_id):
        """
        Estimates mean and amplitude along the X and Z axes
        """
        if 'left_shoulder' in disp_dict and 'right_shoulder' in disp_dict:
            left = disp_dict['left_shoulder']
            right = disp_dict['right_shoulder']

            delta_x_mean = (left['mean_dx'] + right['mean_dx']) / 2
            delta_x_amp = (left['amp_dx'] + right['amp_dx']) / 2
            delta_z_mean = (left['mean_dz'] + right['mean_dz']) / 2
            delta_z_amp = (left['amp_dz'] + right['amp_dz']) / 2

            # Flip X for pose 1 to align inward movement with positive direction
            if pose_id == 1:
                delta_x_mean *= -1

            # Normalize Z: toward robot (–Z) = positive movement
            delta_z_mean *= -1

            return {
                "delta_x_mean": delta_x_mean,
                "delta_x_amp": delta_x_amp,
                "delta_z_mean": delta_z_mean,
                "delta_z_amp": delta_z_amp
            }

        return {
            "delta_x_mean": np.nan,
            "delta_x_amp": np.nan,
            "delta_z_mean": np.nan,
            "delta_z_amp": np.nan,
        }

    def _compute_shoulders_vector_angle(self, data, left_id=11, right_id=12, pose_id=0):
        """
        Computes the angle between the vector connecting the left and right shoulders and the front-facing camera axis
        """
        left = self._get_keypoint_trajectory(data, left_id)
        right = self._get_keypoint_trajectory(data, right_id)

        shoulder_vec = left - right
        vec_angles = self._angle_with_front_view(shoulder_vec, pose_id)

        if vec_angles.size == 0:
            return np.array([])

        # initial_angle = vec_angles[0]
        # angle_delta = vec_angles - initial_angle

        return vec_angles

    def _compute_shoulders_midpoint(self, data):
        """
        Returns the 3D midpoint between the shoulders for each frame
        """

        left = self._get_keypoint_trajectory(data, 11)
        right = self._get_keypoint_trajectory(data, 12)
        return (left + right) / 2

    def _compute_vector_angles_metrics(self, yaw):
        """
        Returns basic yaw metrics: mean and range.
        """
        yaw = yaw[~np.isnan(yaw)]
        if yaw.size == 0:
            return {"mean": np.nan, "amp": np.nan}

        return {
            "mean": yaw.mean(),
            "amp": yaw.max() - yaw.min()
        }

    # -------------------------------------------------------------------------
    # Participants Distance & Inward Movement
    # -------------------------------------------------------------------------
    def _compute_participants_distance(self):
        """
        Computes variation in proximity between the left shoulder of pose 0
        and the right shoulder of pose 1
        """

        shoulder_left0, shoulder_right1 = [], []
        common_frames = sorted(set(self.data0['frame']).intersection(set(self.data1['frame'])))

        for frame in common_frames:
            left_shoulder = self._get_keypoint_by_frame(self.data0, 11, frame)
            right_shoulder = self._get_keypoint_by_frame(self.data1, 12, frame)

            if left_shoulder.size == 3 and right_shoulder.size == 3:
                shoulder_left0.append(left_shoulder)
                shoulder_right1.append(right_shoulder)

        shoulder_left0 = np.array(shoulder_left0)
        shoulder_right1 = np.array(shoulder_right1)

        if len(shoulder_left0) == 0 or len(shoulder_right1) == 0:
            return {
                "mean": np.nan,
                "amp": np.nan
            }

        distances = np.linalg.norm(shoulder_left0 - shoulder_right1, axis=1)
        distances = distances[~np.isnan(distances)]

        if len(distances) == 0:
            return {
                "mean": np.nan,
                "amp": np.nan
            }

        initial_distance = distances[0]
        self.inter_dist = initial_distance - distances

        return {
            "mean": np.mean(self.inter_dist),
            "amp": np.ptp(self.inter_dist)
        }

    def _compare_inward_movement_x(self):
        """
        Compares overall inward movement along the X-axis (side-to-side direction)
        for both poses, based on the displacement of the shoulder midpoint.

        Assumes the camera is facing along -X, so inward movement = decreasing X.

        Returns:
            A dictionary with the mean X displacement for each pose,
            and the ID of the pose that moved more inward.
        """
        # Get X coordinate of shoulder midpoint over time
        mid_x0 = self._compute_shoulders_midpoint(self.data0)[:, 0]
        mid_x1 = self._compute_shoulders_midpoint(self.data1)[:, 0]
        
        valid_x0 = mid_x0[~np.isnan(mid_x0)]
        valid_x1 = mid_x1[~np.isnan(mid_x1)]

        if len(valid_x0) == 0 or len(valid_x1) == 0:
            return {
                "mean_disp_x_pose0": np.nan,
                "mean_disp_x_pose1": np.nan,
                "more_inward_pose": np.nan
            }

        delta_x0 = valid_x0 - valid_x0[0]       # inward = +X
        delta_x1 = valid_x1 - valid_x1[0]       # inward = –X

        inward_disp_x0 = delta_x0               # +X = inward
        inward_disp_x1 = -delta_x1              # –X = inward → flip to +

        mean_inward_x0 = np.nanmean(inward_disp_x0)
        mean_inward_x1 = np.nanmean(inward_disp_x1)

        more_inward_pose = 0 if mean_inward_x0 > mean_inward_x1 else 1

        return {
            "mean_disp_x_pose0": mean_inward_x0,
            "mean_disp_x_pose1": mean_inward_x1,
            "more_inward_pose": more_inward_pose
        }

    def _compute_absolute_distance(self):
        """
        Computes the delta distance between the two participants.
        """
        common_frames = sorted(set(self.data0['frame']).intersection(self.data1['frame']))

        start_dist = None
        end_dist = None

        # First valid distance
        for frame in common_frames:
            shoulder0 = self._get_keypoint_by_frame(self.data0, 11, frame)  # left shoulder
            shoulder1 = self._get_keypoint_by_frame(self.data1, 12, frame)  # right shoulder
            if shoulder0.size == 3 and shoulder1.size == 3 and not np.isnan(shoulder0).any() and not np.isnan(shoulder1).any():
                start_dist = np.linalg.norm(shoulder0 - shoulder1)
                break

        # Last valid distance
        for frame in reversed(common_frames):
            shoulder0 = self._get_keypoint_by_frame(self.data0, 11, frame)
            shoulder1 = self._get_keypoint_by_frame(self.data1, 12, frame)
            if shoulder0.size == 3 and shoulder1.size == 3 and not np.isnan(shoulder0).any() and not np.isnan(shoulder1).any():
                end_dist = np.linalg.norm(shoulder0 - shoulder1)
                break

        if start_dist is not None and end_dist is not None:
            delta = start_dist - end_dist
            return delta

        return None

    # -------------------------------------------------------------------------
    # Speed & Jerk Metrics
    # -------------------------------------------------------------------------
    def _compute_speed(self, data, pose_id, fs=30.0):
        """
        Computes mean speed and speed variability (SD).
        """

        results = {}
        all_speeds = []

        # Initialize a matrix to store speed values for each frame
        speed_matrix = []

        for kpt_id, kpt_name in self.keypoints_dict().items():
            coords = data[data['keypoint'] == kpt_id][['x', 'y', 'z']].values
            coords = np.asarray(coords, dtype=np.float64)
            coords = coords[~np.isnan(coords).any(axis=1)]

            if len(coords) < 2:
                results[kpt_name] = {'mean': np.nan, 'std': np.nan}
                continue

            velocity = np.diff(coords, axis=0) * fs
            speed = np.linalg.norm(velocity, axis=1)

            speed = np.insert(speed, 0, 0.0) # insert zero speed for the first frame
            speed_matrix.append(speed)

            results[kpt_name] = {
                'mean': float(np.mean(speed)),
                'std': float(np.std(speed))
            }

            all_speeds.extend(speed)

        results['overall'] = {
            'mean': float(np.mean(all_speeds)) if all_speeds else np.nan,
            'std': float(np.std(all_speeds)) if all_speeds else np.nan
        }

        speed_matrix = np.array(speed_matrix)
        mean_speeds_per_frame = np.nanmean(speed_matrix, axis=0)
        if pose_id == 0:
            self.speed_0 = mean_speeds_per_frame
        elif pose_id == 1:
            self.speed_1 = mean_speeds_per_frame

        return results

    def _compute_mean_jerk(self, data, pose_id, fs=30.0):
        """
        Computes RMS jerk per keypoint and average jerk per frame.
        Returns a dict with per-keypoint RMS jerk and 'overall' mean jerk across frames.
        """
        results = {}
        jerk_series = []  # list of 1D arrays (per-keypoint)

        for kpt_id, kpt_name in self.keypoints_dict().items():
            coords = data[data['keypoint'] == kpt_id][['x', 'y', 'z']].values
            coords = np.asarray(coords, dtype=np.float64)
            coords = coords[~np.isnan(coords).any(axis=1)]

            if coords.shape[0] < 4:
                results[kpt_name] = np.nan
                continue

            jerk = np.diff(coords, n=3, axis=0) * (fs ** 3)     # (n-3, 3)
            jerk_mag = np.linalg.norm(jerk, axis=1)             # (n-3,)

            rms_jerk = float(np.sqrt(np.mean(jerk_mag ** 2)))
            results[kpt_name] = rms_jerk

            # Pad the first 3 frames with NaN so they're ignored in per-frame mean
            jerk_padded = np.concatenate((np.full(3, np.nan), jerk_mag))
            jerk_series.append(jerk_padded)

        if len(jerk_series) == 0:
            mean_jerk_per_frame = np.array([], dtype=float)
        else:
            max_len = max(arr.shape[0] for arr in jerk_series)
            padded = []
            for arr in jerk_series:
                if arr.shape[0] < max_len:
                    arr = np.concatenate((arr, np.full(max_len - arr.shape[0], np.nan)))
                padded.append(arr)
            jerk_matrix = np.vstack(padded)                      # (n_keypoints, max_len)
            mean_jerk_per_frame = np.nanmean(jerk_matrix, axis=0)

        if pose_id == 0:
            self.jerk_0 = mean_jerk_per_frame
        elif pose_id == 1:
            self.jerk_1 = mean_jerk_per_frame

        results['overall'] = float(np.nanmean(mean_jerk_per_frame)) if mean_jerk_per_frame.size > 0 else np.nan
        return results

    # -------------------------------------------------------------------------
    # Compute All Metrics
    # -------------------------------------------------------------------------
    def compute_all_metrics(self):
        self.yaw_0 = self._compute_shoulders_vector_angle(self.data0, pose_id=0)
        self.yaw_1 = self._compute_shoulders_vector_angle(self.data1, pose_id=1)

        self.disp0 = self._compute_delta_keypoints(self.data0, pose_id=0)
        self.disp1 = self._compute_delta_keypoints(self.data1, pose_id=1)

        self.metrics0 = {
            "shoulders_vector_angle": self._compute_vector_angles_metrics(self.yaw_0),
            "delta_shoulders_midpoint": self._compute_delta_shoulders_midpoint(self.disp0, pose_id=0),
            "speed": self._compute_speed(self.data0, pose_id=0),
            "mean_squared_jerk": self._compute_mean_jerk(self.data0, pose_id=0)
        }

        self.metrics1 = {
            "shoulders_vector_angle": self._compute_vector_angles_metrics(self.yaw_1),
            "delta_shoulders_midpoint": self._compute_delta_shoulders_midpoint(self.disp1, pose_id=1),
            "speed": self._compute_speed(self.data1, pose_id=1),
            "mean_squared_jerk": self._compute_mean_jerk(self.data1, pose_id=1)
        }

        self.other_metrics = {"distance": self._compute_participants_distance(),
                            "absolute_final_distance": self._compute_absolute_distance(),
                            "inward_movement": self._compare_inward_movement_x()}

    # -------------------------------------------------------------------------
    # Print Metrics
    # -------------------------------------------------------------------------
    def print_metrics(self):
        metrics0_flat = {
            'shoulders_vector_angle_mean': self.metrics0['shoulders_vector_angle']['mean'],
            'shoulders_vector_angle_amp': self.metrics0['shoulders_vector_angle']['amp'],
            'deltaX_shoulders_midpoint_mean': self.metrics0['delta_shoulders_midpoint']['delta_x_mean'],
            'deltaX_shoulders_midpoint_amp': self.metrics0['delta_shoulders_midpoint']['delta_x_amp'],
            'deltaZ_shoulders_midpoint_mean': self.metrics0['delta_shoulders_midpoint']['delta_z_mean'],
            'deltaZ_shoulders_midpoint_amp': self.metrics0['delta_shoulders_midpoint']['delta_z_amp'],
            'speed_mean': self.metrics0['speed']['overall']['mean'],
            'speed_std': self.metrics0['speed']['overall']['std'],
            'mean_squared_jerk': self.metrics0['mean_squared_jerk']['overall']
        }

        metrics1_flat = {
            'shoulders_vector_angle_mean': self.metrics1['shoulders_vector_angle']['mean'],
            'shoulders_vector_angle_amp': self.metrics1['shoulders_vector_angle']['amp'],
            'deltaX_shoulders_midpoint_mean': self.metrics1['delta_shoulders_midpoint']['delta_x_mean'],
            'deltaX_shoulders_midpoint_amp': self.metrics1['delta_shoulders_midpoint']['delta_x_amp'],
            'deltaZ_shoulders_midpoint_mean': self.metrics1['delta_shoulders_midpoint']['delta_z_mean'],
            'deltaZ_shoulders_midpoint_amp': self.metrics1['delta_shoulders_midpoint']['delta_z_amp'],
            'speed_mean': self.metrics1['speed']['overall']['mean'],
            'speed_std': self.metrics1['speed']['overall']['std'],
            'mean_squared_jerk': self.metrics1['mean_squared_jerk']['overall']
        }

        other_metrics_flat = {
            'participants_distance_mean': self.other_metrics['distance']['mean'],
            'participants_distance_amp': self.other_metrics['distance']['amp'],
            'absolute_final_distance': self.other_metrics['absolute_final_distance'],
            'inward_movement_pose': self.other_metrics['inward_movement']['more_inward_pose'],
        }

        # Create DataFrame
        df = pd.DataFrame({
            'Pose 0': metrics0_flat,
            'Pose 1': metrics1_flat
        })

        df.index = [
            'Shoulders Vector Angle Mean (°)',
            'Shoulders Vector Angle Amplitude (°)',
            'ΔX Shoulders Midpoint Mean (m)',
            'ΔX Shoulders Midpoint Amplitude (m)',
            'ΔZ Shoulders Midpoint Mean (m)',
            'ΔZ Shoulders Midpoint Amplitude (m)',
            'Speed Mean (m/s)',
            'Speed SD (m/s)',
            'Mean Jerk (m/s^3)',
        ]
        df = df.round(4)
        
        df.T.to_csv(f"{self.results_path}/session_metrics.csv")

        print("\n📊 Summary of Pose Metrics:")
        print(df)

        # Other metrics
        _df = pd.DataFrame({
            'Overall': other_metrics_flat
        })

        _df.index = [
            'Participants Distance Mean (m)',
            'Participants Distance Amplitude (m)',
            'ΔDistance (initial - final) (m)',
            'Inward Movement Pose (0 or 1)',
        ]
        _df = _df.round(4)
        print("\n📊 Other Metrics:")
        print(_df)

        df0 = pd.DataFrame(self.disp0).T.rename(columns={
            'mean_dx': 'Mean ΔX (m)', 'mean_dy': 'Mean ΔY (m)', 'mean_dz': 'Mean ΔZ (m)',
            'amp_dx': 'Amp ΔX (m)', 'amp_dy': 'Amp ΔY (m)', 'amp_dz': 'Amp ΔZ (m)'
        }).dropna()

        df1 = pd.DataFrame(self.disp1).T.rename(columns={
            'mean_dx': 'Mean ΔX (m)', 'mean_dy': 'Mean ΔY (m)', 'mean_dz': 'Mean ΔZ (m)',
            'amp_dx': 'Amp ΔX (m)', 'amp_dy': 'Amp ΔY (m)', 'amp_dz': 'Amp ΔZ (m)'
        }).dropna()

        print("\n📍 Displacement per Keypoint — Pose 0:")
        print(df0)

        print("\n📍 Displacement per Keypoint — Pose 1:")
        print(df1)

        keypoints = df0.index
        components = ['Mean ΔX (m)', 'Mean ΔY (m)', 'Mean ΔZ (m)', 'Amp ΔX (m)', 'Amp ΔY (m)', 'Amp ΔZ (m)']
        columns = pd.MultiIndex.from_product([keypoints, components])
        data = []

        for df in [df0, df1]:
            row = []
            for kp in keypoints:
                for comp in components:
                    row.append(df.loc[kp, comp])
            data.append(row)

        df_combined = pd.DataFrame(data, index=['Pose 0', 'Pose 1'], columns=columns)

        df_combined.to_csv(f"{self.results_path}/displacement_vector_by_pose_and_keypoint.csv")

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    def plot_shoulders_vector_angle(self):
        if self.yaw_0 is None or self.yaw_1 is None:
            self.compute_all_metrics()

        plt.figure(figsize=(10, 5))
        plt.plot(self.yaw_0, label="Pose 0", color='blue')
        plt.plot(self.yaw_1, label="Pose 1", color='orange')
        plt.xlabel("Frame")
        plt.ylabel("Angle (°)")
        plt.title("Shoulders Vector Angle Over Time")
        plt.legend()
        plt.grid(True)

        if self.window_frames:
            for label, (start, _) in self.window_frames.items():
                plt.axvline(x=start, color='black', linestyle='--', linewidth=1)
                plt.text(start, plt.ylim()[0], label, rotation=90,
                        verticalalignment='bottom', horizontalalignment='center',
                        fontsize=8, color='black')

        plt.tight_layout()
        # plt.show()
        
        save_path = f"{self.results_path}/figures/shoulders_vector_angle.png"
        plt.savefig(save_path, dpi=300)

    def plot_participants_delta_distance(self):
        total_frames = int(max(self.data0['frame'].max(), self.data1['frame'].max())) + 1
        delta_distances = np.full(total_frames, np.nan)

        common_frames = sorted(set(self.data0['frame']).intersection(set(self.data1['frame'])))
        initial_distance = None
        valid_count = 0

        for frame in common_frames:
            left = self._get_keypoint_by_frame(self.data0, 11, frame)
            right = self._get_keypoint_by_frame(self.data1, 12, frame)

            if left.size == 3 and right.size == 3 and not np.isnan(left).any() and not np.isnan(right).any():
                dist = np.linalg.norm(left - right)
                if initial_distance is None:
                    initial_distance = dist
                delta_distances[frame] = dist - initial_distance
                valid_count += 1
            # else: delta_distances[frame] remains np.nan

        if initial_distance is None:
            print("⚠️ No valid distance data found.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(total_frames), delta_distances, label="Δ Shoulders Distance", color="darkorange")
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)

        if self.window_frames:
            for label, (start, _) in self.window_frames.items():
                plt.axvline(x=start, color='black', linestyle='--', linewidth=1)
                plt.text(start, plt.ylim()[0], label, rotation=90,
                        verticalalignment='bottom', horizontalalignment='center',
                        fontsize=8, color='black')


        plt.xlabel("Frame")
        plt.ylabel("Δ Distance (m)")
        plt.title("Shoulders Distance Over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        # plt.show()

        save_path = f"{self.results_path}/figures/distance_over_time.png"
        plt.savefig(save_path, dpi=300)

    def plot_delta_displacement_keypoints(self):
        keypoints = self.keypoints_dict()

        for pose_label, data, color in [("Pose 0", self.data0, 'blue'), ("Pose 1", self.data1, 'orange')]:
            total_frames = int(data['frame'].max()) + 1
            n_kpts = len(keypoints)
            n_cols = 3
            n_rows = int(np.ceil(n_kpts / n_cols))

            fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 3.5), sharex=True)
            axs = axs.flatten()

            for i, (kp_id, kp_name) in enumerate(keypoints.items()):
                kp_data = data[data['keypoint'] == kp_id].sort_values('frame')
                frames = kp_data['frame'].values
                coords = kp_data[['x', 'y', 'z']].values
                valid_mask = ~np.isnan(coords).any(axis=1)
                valid_frames = frames[valid_mask]
                valid_coords = coords[valid_mask]

                dx = np.full(total_frames, np.nan)
                dy = np.full(total_frames, np.nan)
                dz = np.full(total_frames, np.nan)

                if valid_coords.shape[0] > 0:
                    initial = valid_coords[0]
                    deltas = valid_coords - initial
                    for f, d in zip(valid_frames, deltas):
                        dx[int(f)], dy[int(f)], dz[int(f)] = d

                axs[i].plot(np.arange(total_frames), dx, label='Δx', color='red')
                axs[i].plot(np.arange(total_frames), dy, label='Δy', color='green')
                axs[i].plot(np.arange(total_frames), dz, label='Δz', color='blue')

                if self.window_frames:
                    for label, (start, _) in self.window_frames.items():
                        axs[i].axvline(x=start, color='black', linestyle='--', linewidth=1)
                        axs[i].text(start, axs[i].get_ylim()[0], label, rotation=90,
                                    verticalalignment='bottom', horizontalalignment='center',
                                    fontsize=8, color='black')

                axs[i].set_title(kp_name)
                axs[i].grid(True)
                axs[i].legend()

            for j in range(i + 1, len(axs)):
                fig.delaxes(axs[j])

            fig.suptitle(f"Keypoints (ΔX, ΔY, ΔZ) Over Time — {pose_label}", fontsize=16)
            fig.supxlabel("Frame")
            fig.supylabel("Displacement (m)")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            # plt.show()

            save_path = f"{self.results_path}/figures/displacement_over_time_delta_{pose_label}.png"
            plt.savefig(save_path, dpi=300)

    def plot_speed_over_time(self, fs=30.0):
        """
        Plot speed over time for each keypoint in Pose 0 and Pose 1.
        """
        keypoints = self.keypoints_dict()

        for pose_label, data, color in [("Pose 0", self.data0, 'blue'), ("Pose 1", self.data1, 'orange')]:
            plt.figure(figsize=(12, 6))

            for kpt_id, kpt_name in keypoints.items():
                coords = data[data['keypoint'] == kpt_id][['x', 'y', 'z']].values
                coords = np.asarray(coords, dtype=np.float64)
                coords = coords[~np.isnan(coords).any(axis=1)]

                if len(coords) < 2:
                    continue

                velocity = np.diff(coords, axis=0) * fs
                speed = np.linalg.norm(velocity, axis=1)

                plt.plot(speed, label=kpt_name, alpha=0.6)

            if self.window_frames:
                for label, (start, _) in self.window_frames.items():
                    plt.axvline(x=start, color='black', linestyle='--', linewidth=1)
                    plt.text(start, plt.ylim()[0], label, rotation=90,
                            verticalalignment='bottom', horizontalalignment='center',
                            fontsize=8, color='black')

            plt.title(f"Speed Over Time — {pose_label}")
            plt.xlabel("Frame")
            plt.ylabel("Speed (m/s)")
            plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
            plt.tight_layout()
            plt.grid(True)

            save_path = f"{self.results_path}/figures/speed_over_time_{pose_label.replace(' ', '_')}.png"
            plt.savefig(save_path, dpi=300)
            plt.close()

    def plot_upper_body_frame(self, frame=5000):
        keypoints = self.keypoints_dict()
        connections = self.skeleton_connections()

        fig = go.Figure()

        def plot_skeleton(data, label, color):
            df = data[data['frame'] == frame]
            coords = {k: df[df['keypoint'] == k][['x', 'y', 'z']].values[0]
                    for k in keypoints if not df[df['keypoint'] == k].empty}

            # Plot joints
            x, y, z = zip(*coords.values())
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(size=5, color=color),
                name=label
            ))

            # Plot bones (connections)
            for a, b in connections:
                if a in coords and b in coords:
                    xa, ya, za = coords[a]
                    xb, yb, zb = coords[b]
                    fig.add_trace(go.Scatter3d(
                        x=[xa, xb], y=[ya, yb], z=[za, zb],
                        mode='lines',
                        line=dict(color=color, width=3),
                        showlegend=False
                    ))

        plot_skeleton(self.data0, "Pose 0", "blue")
        plot_skeleton(self.data1, "Pose 1", "orange")

        # Concatenate keypoints for the current frame
        df_all = pd.concat([
            self.data0[self.data0['frame'] == frame][['x', 'y', 'z']],
            self.data1[self.data1['frame'] == frame][['x', 'y', 'z']]
        ]).dropna()

        # Compute margins
        margin_factor = 0.1
        x_min, x_max = df_all['x'].min(), df_all['x'].max()
        y_min, y_max = df_all['y'].min(), df_all['y'].max()
        z_min, z_max = df_all['z'].min(), df_all['z'].max()

        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min

        # Apply padding
        x_pad = x_range * margin_factor
        y_pad = y_range * margin_factor
        z_pad = z_range * margin_factor

        fig.update_layout(
            scene=dict(
                xaxis=dict(title="X", range=[x_max + x_pad, x_min - x_pad], visible=False),  # Invert X
                yaxis=dict(title="Y", range=[y_min - y_pad, y_max + y_pad], visible=False),
                zaxis=dict(title="Z", range=[z_min - z_pad, z_max + z_pad], visible=False),
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=20),
            showlegend=False,
            scene_camera=dict(
                eye=dict(x=0.0, y=0.5, z=-2.5),
                up=dict(x=0, y=1, z=0)  # Set Y as up, instead of default Z
            )
        )

        # Export to HTML
        file_path = f"{self.results_path}/figures/skeleton_3D_frame_{frame}.html"
        fig.write_html(file_path, auto_open=False)
        
    def export_skeleton_frames_and_video(self, out_name="skeletons.mp4",
                                        min_frame=200, max_frame=600,
                                        step=10, fps=30, out_dir=None):
        """
        Export skeleton frames as PNG (Plotly) and merge into a video.
        """

        keypoints = self.keypoints_dict()
        connections = self.skeleton_connections()

        # ✅ Only frames where both skeletons exist in the range
        frames = sorted(set(self.data0['frame']).intersection(set(self.data1['frame'])))
        frames = [f for f in frames if min_frame <= f <= max_frame][::step]
        if not frames:
            print("⚠️ No common frames found in given range!")
            return

        if out_dir is None:
            out_dir = f"{self.results_path}/figures/frames"
        os.makedirs(out_dir, exist_ok=True)

        # --- Global axis limits
        df_all = pd.concat([self.data0[['x','y','z']], self.data1[['x','y','z']]]).dropna()
        margin_factor = 0.1
        x_min, x_max = df_all['x'].min(), df_all['x'].max()
        y_min, y_max = df_all['y'].min(), df_all['y'].max()
        z_min, z_max = df_all['z'].min(), df_all['z'].max()
        x_pad, y_pad, z_pad = (x_max-x_min)*margin_factor, (y_max-y_min)*margin_factor, (z_max-z_min)*margin_factor

        def plot_skeleton(fig, data, frame, color):
            df = data[data['frame'] == frame]
            if df.empty:
                return
            coords = {k: df[df['keypoint'] == k][['x','y','z']].values[0]
                    for k in keypoints if not df[df['keypoint'] == k].empty}
            if not coords:
                return
            # joints
            x, y, z = zip(*coords.values())
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z,
                                    mode="markers",
                                    marker=dict(size=5, color=color)))
            # bones
            for a, b in connections:
                if a in coords and b in coords:
                    xa, ya, za = coords[a]
                    xb, yb, zb = coords[b]
                    fig.add_trace(go.Scatter3d(x=[xa, xb], y=[ya, yb], z=[za, zb],
                                            mode="lines",
                                            line=dict(color=color, width=3),
                                            showlegend=False))

        img_paths = []
        for frame in frames:
            fig = go.Figure()
            plot_skeleton(fig, self.data0, frame, "blue")
            plot_skeleton(fig, self.data1, frame, "orange")

            fig.update_layout(
                scene=dict(
                    xaxis=dict(range=[x_max+x_pad, x_min-x_pad], visible=False),  # invert X
                    yaxis=dict(range=[y_min-y_pad, y_max+y_pad], visible=False),
                    zaxis=dict(range=[z_min-z_pad, z_max+z_pad], visible=False),
                    aspectmode="data"
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                showlegend=False,
                scene_camera=dict(eye=dict(x=0.0, y=0.5, z=-2.5), up=dict(x=0,y=1,z=0))
            )

            img_path = os.path.join(out_dir, f"frame_{frame:05d}.png")
            img_bytes = pio.to_image(fig, format="png", scale=2, engine="kaleido")
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            img_paths.append(img_path)

        # --- Merge into video
        img0 = cv2.imread(img_paths[0])
        h, w, _ = img0.shape
        out_path = os.path.join(self.results_path, "figures", out_name)
        video = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for p in img_paths:
            video.write(cv2.imread(p))
        video.release()

        print(f"✅ Exported {len(img_paths)} frames → {out_path}")