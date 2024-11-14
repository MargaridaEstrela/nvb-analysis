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
        ani.save("skeleton_animation.mp4", writer="ffmpeg", fps=30)
        plt.close(fig)
        # plt.show()