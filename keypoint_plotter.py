import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class KeypointPlotter:
    def __init__(self, data):
        self.data = data
        self.keypoint_names = {
            0: "Nose", 1: "Left Eye Inner", 2: "Left Eye", 3: "Left Eye Outer", 4: "Right Eye Inner", 
            5: "Right Eye", 6: "Right Eye Outer", 7: "Left Ear", 8: "Right Ear", 9: "Mouth Left", 
            10: "Mouth Right", 11: "Left Shoulder", 12: "Right Shoulder", 13: "Left Elbow", 
            14: "Right Elbow", 15: "Left Wrist", 16: "Right Wrist", 17: "Left Pinky", 
            18: "Right Pinky", 19: "Left Index", 20: "Right Index", 21: "Left Thumb", 
            22: "Right Thumb", 23: "Left Hip", 24: "Right Hip", 25: "Left Knee", 
            26: "Right Knee", 27: "Left Ankle", 28: "Right Ankle", 29: "Left Heel", 
            30: "Right Heel", 31: "Left Foot Index", 32: "Right Foot Index"
        }
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 7),  # Nose to left eye to left ear
            (0, 4), (4, 5), (5, 6), (6, 8),  # Nose to right eye to right ear
            (9, 10),   # Mouth left to mouth right
            (11, 12),  # Shoulders
            (11, 13), (13, 15), (15, 17), (15, 19), (17, 19), (15, 21),  # Left arm to fingers
            (12, 14), (14, 16), (16, 18), (16, 20), (18, 20), (16, 22),  # Right arm to fingers
        ]
        self.connection_colors = [
            'gold', 'gold', 'gold', 'gold',
            'gold', 'gold', 'gold', 'gold',
            'orange',
            'mediumpurple',
            'coral', 'coral', 'coral', 'coral', 'coral', 'coral',
            'turquoise', 'turquoise', 'turquoise', 'turquoise', 'turquoise', 'turquoise',
        ]

    def plot_skeletons(self, ax, frame):
        ax.clear()
        frame_data = self.data[self.data["frame"] == frame]
        if frame_data.empty:
            print(f"No data found for frame {frame}")
            return

        for pose_id in frame_data["pose_id"].unique():
            pose_data = frame_data[frame_data["pose_id"] == pose_id]
            for idx, connection in enumerate(self.connections):
                point1 = pose_data[pose_data["landmark_id"] == connection[0]]
                point2 = pose_data[pose_data["landmark_id"] == connection[1]]

                if not point1.empty and not point2.empty:
                    x_coords = [point1["x"].values[0], point2["x"].values[0]]
                    y_coords = [point1["y"].values[0], point2["y"].values[0]]
                    color = self.connection_colors[idx % len(self.connection_colors)]
                    ax.plot(x_coords, y_coords, color=color, marker="o", markersize=5, markerfacecolor="white")
                        
            # Plot the landmark IDs as text labels
            connection_keypoints = pose_data[pose_data["landmark_id"].isin([c for conn in self.connections for c in conn])]
            for _, keypoint in connection_keypoints.iterrows():
                ax.text(keypoint["x"], keypoint["y"], str(int(keypoint["landmark_id"])), fontsize=8, color='blue')

        ax.set_title(f"Skeletons for Frame {frame}")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)

    def animate_skeletons(self):
        frames = self.data["frame"].unique()
        fig, ax = plt.subplots(figsize=(8, 8))

        def update(frame):
            self.plot_skeletons(ax, frame)

        ani = animation.FuncAnimation(fig, update, frames=frames, repeat=False)
        ani.save("skeleton_animation.mp4", writer="ffmpeg", fps=50)
        plt.close(fig)
        # plt.show()

    def plot_depth_trend(self):
        frames = self.data["frame"].unique()
        z_person_0 = []
        z_person_1 = []

        for frame in frames:
            frame_data = self.data[self.data["frame"] == frame]
            
            # Person 0 data
            person_0_data = frame_data[frame_data["pose_id"] == 0]
            if not person_0_data.empty:
                avg_z_person_0 = person_0_data["z"].mean()
            else:
                avg_z_person_0 = None
            z_person_0.append(avg_z_person_0)

            # Person 1 data
            person_1_data = frame_data[frame_data["pose_id"] == 1]
            if not person_1_data.empty:
                avg_z_person_1 = person_1_data["z"].mean()
            else:
                avg_z_person_1 = None
            z_person_1.append(avg_z_person_1)

        # Plotting the trends for each person
        plt.figure(figsize=(10, 6))
        plt.plot(
            frames, z_person_0, color="deepskyblue", label="Person 0"
        )
        plt.plot(
            frames, z_person_1, color="salmon", label="Person 1"
        )
        plt.xlabel("Frame Index")
        plt.ylabel("Average Z Coordinate")
        plt.title("Depth Trend Over Frames for Each Person")
        plt.grid(True)
        plt.legend()
        plt.savefig("figures/depth_trend.png")
        # plt.show()
        
    def plot_keypoint_heatmap(self, keypoint_id):
        # Create a heatmap of the keypoint frequency
        keypoint_data = self.data[self.data["landmark_id"] == keypoint_id]
        x_coords = keypoint_data["x"].values
        y_coords = keypoint_data["y"].values

        plt.figure(figsize=(8, 6))
        # Increase bins for higher resolution and add transparency
        plt.hist2d(x_coords, y_coords, bins=100, cmap='hot', alpha=0.8, cmin=1)
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.xlim(0, 1)
        plt.ylim(1, 0)
        plt.xlabel("X Coordinate", fontsize=14)
        plt.ylabel("Y Coordinate", fontsize=14)
        plt.title(f"Heatmap of {self.keypoint_names.get(keypoint_id, 'Keypoint')} Frequency", fontsize=16)
        plt.savefig(f"figures/heatmap_{keypoint_id}.png")
        # plt.show()

    def plot_keypoint_coordinates(self, keypoint_ids):
        num_keypoints = len(keypoint_ids)

        num_cols = 6
        num_rows = math.ceil(num_keypoints / num_cols)

        figsize = (28, 4 * num_rows)

        # Plot X Coordinates
        fig_x, axes_x = plt.subplots(num_rows, num_cols, figsize=figsize)
        fig_x.suptitle("X Coordinate Trend of Keypoints Over Time", fontsize=26)

        # Plot Y Coordinates
        fig_y, axes_y = plt.subplots(num_rows, num_cols, figsize=figsize)
        fig_y.suptitle("Y Coordinate Trend of Keypoints Over Time", fontsize=26)

        frames = self.data["frame"].unique()

        for idx, keypoint_id in enumerate(keypoint_ids):
            row = idx // num_cols
            col = idx % num_cols

            # Data for both persons
            keypoint_data = self.data[self.data["landmark_id"] == keypoint_id]
            person_0_data = keypoint_data[keypoint_data["pose_id"] == 0]
            person_1_data = keypoint_data[keypoint_data["pose_id"] == 1]

            keypoint_name = self.keypoint_names.get(keypoint_id, f"Keypoint {keypoint_id}")

            # X Coordinate Plot
            ax_x = axes_x[row, col] if num_rows > 1 else axes_x[col]
            ax_x.plot(person_0_data["frame"], person_0_data["x"], color='deepskyblue', linewidth=1.5)
            ax_x.plot(person_1_data["frame"], person_1_data["x"], color='salmon', linestyle='--', linewidth=1.5)
            ax_x.set_title(f"{keypoint_name}", fontsize=16)
            ax_x.grid(True, linewidth=0.5)
            ax_x.set_xlim(frames.min(), frames.max())
            ax_x.set_ylim(0, 1)

            # Y Coordinate Plot
            ax_y = axes_y[row, col] if num_rows > 1 else axes_y[col]
            ax_y.plot(person_0_data["frame"], person_0_data["y"], color='deepskyblue', linewidth=1.5)
            ax_y.plot(person_1_data["frame"], person_1_data["y"], color='salmon', linestyle='--', linewidth=1.5)
            ax_y.set_title(f"{keypoint_name}", fontsize=16)
            ax_y.grid(True, linewidth=0.5)
            ax_y.set_xlim(frames.min(), frames.max())
            ax_y.set_ylim(0, 1)

            # Rotate x-axis labels for better readability
            for tick in ax_x.get_xticklabels():
                tick.set_rotation(45)
            for tick in ax_y.get_xticklabels():
                tick.set_rotation(45)

        # Hide any unused subplots
        for idx in range(num_keypoints, num_rows * num_cols):
            fig_x.delaxes(axes_x.flatten()[idx])
            fig_y.delaxes(axes_y.flatten()[idx])

        # Labels for X Coordinates Figure
        fig_x.text(0.5, 0.04, "Frame Index", ha="center", fontsize=20)
        fig_x.text(0.04, 0.5, "X Coordinate", va="center", rotation="vertical", fontsize=20)

        # Labels for Y Coordinates Figure
        fig_y.text(0.5, 0.04, "Frame Index", ha="center", fontsize=20)
        fig_y.text(0.04, 0.5, "Y Coordinate", va="center", rotation="vertical", fontsize=20)

        fig_x.subplots_adjust(hspace=0.4, wspace=0.3)
        fig_y.subplots_adjust(hspace=0.4, wspace=0.3)

        handles = [
            plt.Line2D([0], [0], color='deepskyblue', linewidth=1.5, label='Person 0'),
            plt.Line2D([0], [0], color='salmon', linestyle='--', linewidth=1.5, label='Person 1')
        ]
        fig_x.legend(handles=handles, loc='upper right', fontsize=16)
        fig_y.legend(handles=handles, loc='upper right', fontsize=16)

        fig_x.savefig("figures/keypoint_x_coordinates.png")
        fig_y.savefig("figures/keypoint_y_coordinates.png")

        # plt.show()

    def plot_keypoint_trajectories(self, keypoint_ids):
        num_keypoints = len(keypoint_ids)
        
        num_cols = 6
        num_rows = math.ceil(num_keypoints / num_cols)
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(28, 4 * num_rows))
        fig.suptitle("Keypoint Trajectories in 2D Space for Both Persons", fontsize=16)

        for idx, keypoint_id in enumerate(keypoint_ids):
            row = idx // num_cols
            col = idx % num_cols
            ax = axes[row, col] if num_rows > 1 else axes[col]
            
            keypoint_data = self.data[self.data["landmark_id"] == keypoint_id]
            person_0_data = keypoint_data[keypoint_data["pose_id"] == 0]
            person_1_data = keypoint_data[keypoint_data["pose_id"] == 1]
            
            ax.plot(person_0_data["x"], person_0_data["y"], color="deepskyblue", linewidth=2)
            ax.plot(person_1_data["x"], person_1_data["y"], color="salmon", linewidth=2)

            keypoint_name = self.keypoint_names.get(keypoint_id, f"Keypoint {keypoint_id}")
            ax.set_title(keypoint_name, fontsize=12)

            ax.invert_yaxis()
            ax.grid(True, linewidth=0.3)

            # Adjust x and y limits to be consistent for all plots
            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)

        # Hide any unused subplots
        for idx in range(num_keypoints, num_rows * num_cols):
            fig.delaxes(axes.flatten()[idx])
            
        handles = [
            plt.Line2D([0], [0], color='deepskyblue', linewidth=1.5, label='Person 0'),
            plt.Line2D([0], [0], color='salmon', linestyle='--', linewidth=1.5, label='Person 1')
        ]
        fig.legend(handles=handles, loc='upper right', fontsize=16)

        # Common X and Y axis labels
        fig.text(0.5, 0.04, "X Coordinate", ha='center', fontsize=16)
        fig.text(0.04, 0.5, "Y Coordinate", va='center', rotation='vertical', fontsize=16)

        # Adjust the layout for better spacing
        plt.tight_layout(rect=[0.05, 0.05, 1, 0.96])
        plt.savefig("figures/keypoint_trajectories.png")
        # plt.show()

    def plot_velocity_trend(self, keypoint_ids):
        frames = self.data["frame"].unique()
        velocities = {}

        # Compute velocity for each landmark and each person
        for landmark_id in self.data["landmark_id"].unique():
            velocities[landmark_id] = {'person_0': [], 'person_1': []}

            for frame in range(1, len(frames)):
                frame_data_prev = self.data[self.data["frame"] == frames[frame - 1]]
                frame_data_curr = self.data[self.data["frame"] == frames[frame]]

                for pose_id in [0, 1]:
                    prev_data = frame_data_prev[
                        (frame_data_prev["landmark_id"] == landmark_id) & (frame_data_prev["pose_id"] == pose_id)
                    ]
                    curr_data = frame_data_curr[
                        (frame_data_curr["landmark_id"] == landmark_id) & (frame_data_curr["pose_id"] == pose_id)
                    ]

                    if not prev_data.empty and not curr_data.empty:
                        dx = curr_data["x"].values[0] - prev_data["x"].values[0]
                        dy = curr_data["y"].values[0] - prev_data["y"].values[0]
                        velocity = np.sqrt(dx ** 2 + dy ** 2)
                    else:
                        velocity = None

                    if pose_id == 0:
                        velocities[landmark_id]['person_0'].append(velocity)
                    else:
                        velocities[landmark_id]['person_1'].append(velocity)

        # Plot velocity trends for each keypoint in separate subplots
        num_keypoints = len(keypoint_ids)

        num_cols = 6
        num_rows = math.ceil(num_keypoints / num_cols)
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(28, 4 * num_rows))
        fig.suptitle("Velocity Trend of Keypoints Over Time", fontsize=26, y=1.02)

        for idx, keypoint_id in enumerate(keypoint_ids):
            row = idx // num_cols
            col = idx % num_cols
            ax = axes[row, col] if num_rows > 1 else axes[col]

            # Plot velocity for Person 0 and Person 1
            ax.plot(range(1, len(frames)), velocities[keypoint_id]['person_0'], color='deepskyblue', linewidth=1.5)
            ax.plot(range(1, len(frames)), velocities[keypoint_id]['person_1'], color='salmon', linestyle='--', linewidth=1.5)

            keypoint_name = self.keypoint_names.get(keypoint_id, f"Keypoint {keypoint_id}")
            ax.set_title(keypoint_name, fontsize=16)
            ax.grid(True, linewidth=0.3)

        for idx in range(num_keypoints, num_rows * num_cols):
            fig.delaxes(axes.flatten()[idx])
            
        handles = [
            plt.Line2D([0], [0], color='deepskyblue', linewidth=1.5, label='Person 0'),
            plt.Line2D([0], [0], color='salmon', linestyle='--', linewidth=1.5, label='Person 1')
        ]
        fig.legend(handles=handles, loc='upper right', fontsize=16)

        fig.text(0.5, 0.04, "Frame Index", ha="center", fontsize=18)
        fig.text(0.04, 0.5, "Velocity", va="center", rotation="vertical", fontsize=18)

        plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        plt.savefig("figures/velocity_trend.png")
        # plt.show()