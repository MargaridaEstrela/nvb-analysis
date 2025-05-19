import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class PoseMetrics:
    def __init__(self, pose0_csvPath, pose1_csvPath, results_path):
        self.data0 = pd.read_csv(pose0_csvPath)
        self.data1 = pd.read_csv(pose1_csvPath)
        self.results_path = results_path
        self.yaw0 = None
        self.yaw1 = None
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

    def _get_keypoint_trajectory(self, data, keypoint_id):
        return data[data['keypoint'] == keypoint_id].iloc[:, 3:6].values

    def _get_keypoint_by_frame(self, data, keypoint_id, frame):
        point = data[(data['keypoint'] == keypoint_id) & (data['frame'] == frame)].iloc[:, 3:6].values
        return point.flatten() if point.size == 3 else np.array([])

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

    def _compute_displacement_metrics(self, data):
        """
        Computes frame-to-frame displacement magnitude for each keypoint.
        Returns a dictionary with mean and amplitude for each keypoint.
        """
        keypoints = self.keypoints_dict()
        summary = {}

        for kpt_id, kpt_name in keypoints.items():
            coords = self._get_keypoint_trajectory(data, kpt_id)
            coords = coords[~np.isnan(coords).any(axis=1)]

            if coords.shape[0] < 2:
                summary[kpt_name] = np.array([])
                continue

            # Compute displacement between consecutive valid frames
            displacements = np.linalg.norm(np.diff(coords, axis=0), axis=1)
            summary[kpt_name] = {
                'displacement_mean': float(np.mean(displacements)),
                'displacement_amplitude': float(np.ptp(displacements))
            }

        return summary

    def _compute_displacement_initial(self, data):
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
                summary[kpt_name] = np.array([])
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

        return summary

    def _compute_depth_from_initial(self, disp_dict):
        """
        Estimates mean depth and its variation using the Z-axis displacement from the shoulders midpoint
        """
        if 'left_shoulder' in disp_dict and 'right_shoulder' in disp_dict:
            left = disp_dict['left_shoulder']
            right = disp_dict['right_shoulder']

            mean_depth = (left['mean_dz'] + right['mean_dz']) / 2
            depth_range = (left['amp_dz'] + right['amp_dz']) / 2

            return {
                "mean_depth": mean_depth,
                "depth_range": depth_range
            }

        return {
            "mean_depth": np.nan,
            "depth_range": np.nan
        }

    def _compute_yaw(self, data, left_id=11, right_id=12, pose_id=0):
        """
        Computes yaw angles over time by measuring the orientation of the shoulder line
        """

        left = self._get_keypoint_trajectory(data, left_id)
        right = self._get_keypoint_trajectory(data, right_id)

        shoulder_vec = left - right
        yaw_angles = self._angle_with_front_view(shoulder_vec, pose_id)

        # yaw_angles = yaw_angles[~np.isnan(yaw_angles)]
        if yaw_angles.size == 0:
            return np.array([])

        # initial_yaw = yaw_angles[0]
        # yaw_deltas = yaw_angles - initial_yaw

        return yaw_angles

    def _compute_shoulders_midpoint(self, data):
        """
        Returns the 3D midpoint between the shoulders for each frame
        """

        left = self._get_keypoint_trajectory(data, 11)
        right = self._get_keypoint_trajectory(data, 12)
        return (left + right) / 2

    def _compute_yaw_metrics(self, yaw):
        """
        Returns basic yaw metrics: mean and range.
        """

        yaw = yaw[~np.isnan(yaw)]
        return {
            "mean_yaw": yaw.mean(),
            "yaw_range": yaw.max() - yaw.min()
        }

    def _compute_engagement_features(self, yaw_values):
        """
        Extracts engagement features from yaw data: amplitude, std deviation, number of peaks, and frozen frames
        """

        yaw_values = yaw_values[~np.isnan(yaw_values)]
        if len(yaw_values) < 3:
            return {
                "amplitude": np.nan,
                "std_dev": np.nan,
                "num_peaks": np.nan,
                "frozen_frames": np.nan
            }

        amplitude = np.max(yaw_values) - np.min(yaw_values)
        std_dev = np.std(yaw_values)

        # Count sign changes in first derivative → peaks and troughs
        diffs = np.diff(yaw_values)
        sign_changes = np.diff(np.sign(diffs))
        num_peaks = np.sum(sign_changes != 0)

        # Count frames with very small variation (signal freezing)
        frozen_frames = np.sum(np.abs(diffs) < 0.1)

        return {
            "amplitude": amplitude,
            "std_dev": std_dev,
            "num_peaks": num_peaks,
            "frozen_frames": frozen_frames
        }

    def _compute_yaw_speed_features(self, yaw):
        """
        Computes angular speed metrics from the yaw signal
        """

        yaw = yaw[~np.isnan(yaw)]
        if len(yaw) < 2:
            return {
                "mean_speed": np.nan,
                "max_speed": np.nan,
                "speed_std": np.nan,
                "num_fast_turns": np.nan
            }

        velocity = np.diff(yaw)
        speed = np.abs(velocity)

        fast_threshold = 0.1  # deg/frame

        return {
            "mean_speed": np.mean(speed),
            "max_speed": np.max(speed),
            "speed_std": np.std(speed),
            "num_fast_turns": np.sum(speed > fast_threshold)
        }

    def _compute_depht_metrics(self, data):
        """
        Calculates mean and range of depth (Z-axis) from shoulder midpoint
        """

        depth = self._compute_shoulders_midpoint(data)[:, 2]
        depth = depth[~np.isnan(depth)]
        return {
            "mean_depth": depth.mean(),
            "depth_range": depth.max() - depth.min()
        }

    def _compute_proximity_metrics(self):
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
                "mean_distance": np.nan,
                "distance_range": np.nan
            }

        distances = np.linalg.norm(shoulder_left0 - shoulder_right1, axis=1)
        distances = distances[~np.isnan(distances)]

        if len(distances) == 0:
            return {
                "mean_distance": np.nan,
                "distance_range": np.nan
            }

        initial_distance = distances[0]
        delta_distances = distances - initial_distance

        return {
            "mean_distance": np.mean(delta_distances),
            "distance_range": np.ptp(delta_distances)
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
        
        mid_x0 = mid_x0[~np.isnan(mid_x0)]
        mid_x1 = mid_x1[~np.isnan(mid_x1)]

        mean_disp_x0 = np.nanmean(mid_x0 - mid_x0[0])
        mean_disp_x1 = np.nanmean(mid_x1[0] - mid_x1)

        # Decide which pose moved more inward (i.e., lower mean X)
        if mean_disp_x0 < mean_disp_x1:
            more_inward_pose = 0
        else:
            more_inward_pose = 1

        return {
            "mean_disp_x_pose0": mean_disp_x0,
            "mean_disp_x_pose1": mean_disp_x1,
            "more_inward_pose": more_inward_pose
        }

    def compute_all_metrics(self):
        self.yaw0 = self._compute_yaw(self.data0, pose_id=0)
        self.yaw1 = self._compute_yaw(self.data1, pose_id=1)

        self.disp0 = self._compute_displacement_initial(self.data0)
        self.disp1 = self._compute_displacement_initial(self.data1)

        self.metrics0 = {
            "yaw_metrics": self._compute_yaw_metrics(self.yaw0),
            "engagement_metrics": self._compute_engagement_features(self.yaw0),
            "depth_metrics": self._compute_depth_from_initial_disp(self.disp0)
        }

        self.metrics1 = {
            "yaw_metrics": self._compute_yaw_metrics(self.yaw1),
            "engagement_metrics": self._compute_engagement_features(self.yaw1),
            "depth_metrics": self._compute_depth_from_initial_disp(self.disp1)
        }

        self.other_metrics = {"proximity_metrics": self._compute_proximity_metrics(),
                            "inward_movement": self._compare_inward_movement_x()}

    def print_metrics(self):
        metrics0_flat = {
            'yaw_mean': self.metrics0['yaw_metrics']['mean_yaw'],
            'yaw_amplitude': self.metrics0['engagement_metrics']['amplitude'],
            'yaw_std_dev': self.metrics0['engagement_metrics']['std_dev'],
            'depth_mean': self.metrics0['depth_metrics']['mean_depth'],
            'depth_amplitude': self.metrics0['depth_metrics']['depth_range'],
        }

        metrics1_flat = {
            'yaw_mean': self.metrics1['yaw_metrics']['mean_yaw'],
            'yaw_amplitude': self.metrics1['engagement_metrics']['amplitude'],
            'yaw_std_dev': self.metrics1['engagement_metrics']['std_dev'],
            'depth_mean': self.metrics1['depth_metrics']['mean_depth'],
            'depth_amplitude': self.metrics1['depth_metrics']['depth_range'],
        }

        other_metrics_flat = {
            'mean_proximity': self.other_metrics['proximity_metrics']['mean_distance'],
            'proximity_range': self.other_metrics['proximity_metrics']['distance_range'],
            'inward_movement_pose': self.other_metrics['inward_movement']['more_inward_pose'],
        }

        # Create DataFrame
        df = pd.DataFrame({
            'Pose 0': metrics0_flat,
            'Pose 1': metrics1_flat
        })

        df.index = [
            'Yaw Mean (°)',
            'Yaw Amplitude (°)',
            'Yaw Std Dev (°)',
            'Depth Mean (m)',
            'Depth Amplitude (m)',
        ]
        df = df.round(3)

        print("\n📊 Summary of Pose Metrics:")
        print(df)

        # Other metrics
        _df = pd.DataFrame({
            'Overall': other_metrics_flat
        })

        _df.index = [
            'Mean Proximity (m)',
            'Proximity Range (m)',
            'Inward Movement Pose (0 or 1)',
        ]
        _df = _df.round(3)
        print("\n📊 Other Metrics:")
        print(_df)

        df0 = pd.DataFrame(self.disp0).T.rename(columns={
            'mean_dx': 'Mean ΔX (m)', 'mean_dy': 'Mean ΔY (m)', 'mean_dz': 'Mean ΔZ (m)',
            'amp_dx': 'Amp ΔX (m)', 'amp_dy': 'Amp ΔY (m)', 'amp_dz': 'Amp ΔZ (m)'
        }).dropna().round(4)

        df1 = pd.DataFrame(self.disp1).T.rename(columns={
            'mean_dx': 'Mean ΔX (m)', 'mean_dy': 'Mean ΔY (m)', 'mean_dz': 'Mean ΔZ (m)',
            'amp_dx': 'Amp ΔX (m)', 'amp_dy': 'Amp ΔY (m)', 'amp_dz': 'Amp ΔZ (m)'
        }).dropna().round(4)

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

    def plot_yaw_over_time(self):
        if self.yaw0 is None or self.yaw1 is None:
            self.compute_all_metrics()

        plt.figure(figsize=(10, 5))
        plt.plot(self.yaw0, label="Pose 0", color='blue')
        plt.plot(self.yaw1, label="Pose 1", color='orange')
        plt.xlabel("Frame")
        plt.ylabel("Yaw (°)")
        plt.title("Shoulders Vector Angle Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.show()
        
        save_path = f"{self.results_path}/figures/yaw_over_time.png"
        plt.savefig(save_path, dpi=300)

    def plot_proximity_over_time(self):
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
            print("⚠️ No valid proximity data found.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(total_frames), delta_distances, label="Δ Shoulders Distance", color="darkorange")
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.xlabel("Frame")
        plt.ylabel("Δ Distance (m)")
        plt.title("Shoulders Distance Over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        # plt.show()

        save_path = f"{self.results_path}/figures/distance_over_time.png"
        plt.savefig(save_path, dpi=300)

    def plot_displacement_all_keypoints(self):
        keypoints = self.keypoints_dict()

        for pose_label, data, color in [("Pose 0", self.data0, 'blue'), ("Pose 1", self.data1, 'orange')]:
            total_frames = int(data['frame'].max()) + 1
            n_kpts = len(keypoints)
            n_cols = 4
            n_rows = int(np.ceil(n_kpts / n_cols))

            fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 2.5), sharex=True)
            axs = axs.flatten()

            for i, (kp_id, kp_name) in enumerate(keypoints.items()):
                kp_data = data[data['keypoint'] == kp_id].sort_values('frame')
                frames = kp_data['frame'].values
                coords = kp_data[['x', 'y', 'z']].values
                valid_mask = ~np.isnan(coords).any(axis=1)
                valid_frames = frames[valid_mask]
                valid_coords = coords[valid_mask]
                displacement = np.full(total_frames, np.nan)
                displacements = np.linalg.norm(np.diff(valid_coords, axis=0), axis=1)
                displacement_frames = valid_frames[:-1]
                for f, d in zip(displacement_frames, displacements):
                    displacement[int(f)] = d

                axs[i].plot(np.arange(total_frames), displacement, color=color)
                axs[i].set_title(kp_name)
                axs[i].grid(True)

            for j in range(i + 1, len(axs)):
                fig.delaxes(axs[j])

            fig.suptitle(f"Displacement Over Time — {pose_label}", fontsize=16)
            fig.supxlabel("Frame")
            fig.supylabel("Displacement (m)")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            # plt.show()

            save_path = f"{self.results_path}/figures/displacement_over_time_{pose_label}.png"
            plt.savefig(save_path, dpi=300)


    def plot_displacement_all_keypoints_initial_point(self):
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
                axs[i].set_title(kp_name)
                axs[i].grid(True)
                axs[i].legend()

            for j in range(i + 1, len(axs)):
                fig.delaxes(axs[j])

            fig.suptitle(f"Displacement (x, y, z) Over Time — {pose_label}", fontsize=16)
            fig.supxlabel("Frame")
            fig.supylabel("Displacement (m)")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            # plt.show()

            save_path = f"{self.results_path}/figures/displacement_over_time_delta_{pose_label}.png"
            plt.savefig(save_path, dpi=300)

    def plot_upper_body_frame(self, frame=1000):
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
                xaxis=dict(title="X", range=[x_max + x_pad, x_min - x_pad]),  # Invert X
                yaxis=dict(title="Y", range=[y_min - y_pad, y_max + y_pad]),
                zaxis=dict(title="Z", range=[z_min - z_pad, z_max + z_pad]),
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=20),
            legend=dict(
                x=0.02, y=0.98,
                xanchor='left', yanchor='top',
                bgcolor='rgba(255,255,255,0.6)',
                bordercolor='black',
                borderwidth=1
            ),
            scene_camera=dict(
                eye=dict(x=0.0, y=0.5, z=-2.5),
                up=dict(x=0, y=1, z=0)  # Set Y as up, instead of default Z
            )
        )

        # Export to HTML
        file_path = f"{self.results_path}/figures/skeleton_3D_frame_{frame}.html"
        fig.write_html(file_path, auto_open=False)