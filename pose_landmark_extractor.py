import cv2
import mediapipe as mp
import csv
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.framework.formats import landmark_pb2

class PoseLandmarkExtractor:
    def __init__(self, video_path, model_path, output_csv):
        self.video_path = video_path
        self.output_csv = output_csv
        self.frame_index = 0
        self.prev_keypoints = [None, None]  # To store previous positions for 2 persons

        # Initialize options for Pose Landmarker
        self.options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=2,  # Allows up to 2 poses
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def calculate_distance(self, kp1, kp2):
        return np.sqrt((kp1.x - kp2.x) ** 2 + (kp1.y - kp2.y) ** 2 + (kp1.z - kp2.z) ** 2)

    def assign_closest_person(self, detected_poses):
        new_assignments = [None, None]

        # If both previous keypoints are available, calculate distances
        if self.prev_keypoints[0] and self.prev_keypoints[1]:
            dist_1_0 = self.calculate_distance(detected_poses[0][0], self.prev_keypoints[0][0])
            dist_1_1 = self.calculate_distance(detected_poses[0][0], self.prev_keypoints[1][0])
            dist_2_0 = self.calculate_distance(detected_poses[1][0], self.prev_keypoints[0][0])
            dist_2_1 = self.calculate_distance(detected_poses[1][0], self.prev_keypoints[1][0])

            # Assign closest by comparing distances
            if dist_1_0 + dist_2_1 < dist_1_1 + dist_2_0:
                new_assignments = detected_poses
            else:
                new_assignments = [detected_poses[1], detected_poses[0]]
        else:
            # For the first frame, initialize directly
            new_assignments = detected_poses

        # Update previous keypoints
        self.prev_keypoints = new_assignments
        return new_assignments

    def setup_csv(self):
        # CSV file setup with headers
        with open(self.output_csv, mode="w", newline="") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(["frame", "pose_id", "landmark_id", "x", "y", "z"])

    def extract_landmarks(self):
        # Load Pose Landmarker
        with vision.PoseLandmarker.create_from_options(self.options) as landmarker:
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)  # Get video FPS for timestamp calculation

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert the frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(mp.ImageFormat.SRGB, frame_rgb)

                # Perform pose detection on each frame
                detection_result = landmarker.detect_for_video(
                    mp_image, self.frame_index * (1000 // int(fps))
                )

                # Save pose landmarks to CSV with consistent assignments
                if detection_result.pose_landmarks and len(detection_result.pose_landmarks) >= 2:
                    # Assign closest poses based on previous frame
                    assigned_poses = self.assign_closest_person(detection_result.pose_landmarks[:2])
                    self.save_landmarks_to_csv(assigned_poses)

                self.frame_index += 1

            cap.release()
        print(f"Landmarks saved to {self.output_csv}")

    def save_landmarks_to_csv(self, assigned_poses):
        with open(self.output_csv, mode="a", newline="") as file:
            csv_writer = csv.writer(file)
            for pose_id, pose_landmarks in enumerate(assigned_poses):
                for landmark_id, landmark in enumerate(pose_landmarks):
                    csv_writer.writerow(
                        [
                            self.frame_index,  # Current frame number
                            pose_id,  # Pose ID (for each detected person)
                            landmark_id,  # Landmark ID
                            landmark.x,  # X coordinate
                            landmark.y,  # Y coordinate
                            landmark.z,  # Z coordinate (depth)
                        ]
                    )
