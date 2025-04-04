import cv2
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
        
    def plot_skeletons_on_frame(self, frame_img, frame):
        """
        Draw skeletons directly on the video frame using OpenCV.
        """
        connection_colors_bgr = [
            (0, 215, 255),  # Gold
            (0, 215, 255),  # Gold
            (0, 215, 255),  # Gold
            (0, 215, 255),  # Gold
            (0, 215, 255),  # Gold
            (0, 215, 255),  # Gold
            (0, 215, 255),  # Gold
            (0, 215, 255),  # Gold
            (0, 165, 255),  # Orange
            (147, 112, 219),  # Mediumpurple
            (0, 127, 255),  # Coral
            (0, 127, 255),  # Coral
            (0, 127, 255),  # Coral
            (0, 127, 255),  # Coral
            (0, 127, 255),  # Coral
            (0, 127, 255),  # Coral
            (64, 224, 208),  # Turquoise
            (64, 224, 208),  # Turquoise
            (64, 224, 208),  # Turquoise
            (64, 224, 208),  # Turquoise
            (64, 224, 208),  # Turquoise
            (64, 224, 208),  # Turquoise
        ]
        
        frame_data = self.data[self.data["frame"] == frame]
        if frame_data.empty:
            print(f"No data found for frame {frame}")
            return frame_img

        height, width, _ = frame_img.shape  # Get frame dimensions

        for pose_id in frame_data["pose_id"].unique():
            pose_data = frame_data[frame_data["pose_id"] == pose_id]
            for idx, connection in enumerate(self.connections):
                point1 = pose_data[pose_data["landmark_id"] == connection[0]]
                point2 = pose_data[pose_data["landmark_id"] == connection[1]]

                if not point1.empty and not point2.empty:
                    # Scale normalized coordinates to video resolution
                    x1, y1 = int(point1["x"].values[0] * width), int(point1["y"].values[0] * height)
                    x2, y2 = int(point2["x"].values[0] * width), int(point2["y"].values[0] * height)
                    color = connection_colors_bgr[idx % len(connection_colors_bgr)]
                    
                    # Draw the connection line
                    thickness = 4
                    cv2.line(frame_img, (x1, y1), (x2, y2), color, thickness)
                    cv2.circle(frame_img, (x1, y1), 3, (255, 255, 255), -1)
                    cv2.circle(frame_img, (x2, y2), 3, (255, 255, 255), -1)

            # Draw landmark IDs as text labels
            for _, keypoint in pose_data.iterrows():
                if (keypoint["landmark_id"] < 23):
                    x, y = int(keypoint["x"] * width), int(keypoint["y"] * height)
                    # cv2.circle(frame_img, (x, y), 5, (255, 255, 255), -1)
                    landmark_id = int(keypoint["landmark_id"])
                    cv2.putText(frame_img, str(landmark_id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        return frame_img

    def animate_skeletons(self):
        frames = self.data["frame"].unique()
        fig, ax = plt.subplots(figsize=(8, 8))

        def update(frame):
            self.plot_skeletons(ax, frame)

        ani = animation.FuncAnimation(fig, update, frames=frames, repeat=False)
        ani.save("skeleton_animation.mp4", writer="ffmpeg", fps=30)
        plt.close(fig)
        # plt.show()