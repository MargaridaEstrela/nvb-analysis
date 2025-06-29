import csv
import cv2
import sys
import numpy as np
import mediapipe as mp

from tqdm import tqdm
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

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
            min_pose_detection_confidence=0.4,
            min_tracking_confidence=0.7,
        )

    def calculate_distance(self, kp1, kp2):
        return np.sqrt((kp1.x - kp2.x) ** 2 + (kp1.y - kp2.y) ** 2 + (kp1.z - kp2.z) ** 2)
    
    def average_distance(self, landmarks1, landmarks2, indices=[0, 11, 12, 23, 24]):
        distances = []
        for i in indices:
            try:
                lm1 = landmarks1[i]
                lm2 = landmarks2[i]
                d = np.sqrt(
                    (lm1.x - lm2.x) ** 2 +
                    (lm1.y - lm2.y) ** 2 +
                    (lm1.z - lm2.z) ** 2
                )
                distances.append(d)
            except IndexError:
                # In case landmarks are incomplete for some reason
                continue
        return np.mean(distances) if distances else float('inf')

    def assign_closest_person(self, detected_poses):
        new_assignments = [None, None]

        # If both previous keypoints are available, calculate distances
        if self.prev_keypoints[0] and self.prev_keypoints[1]:
            dist_1_0 = self.average_distance(detected_poses[0], self.prev_keypoints[0])
            dist_1_1 = self.average_distance(detected_poses[0], self.prev_keypoints[1])
            dist_2_0 = self.average_distance(detected_poses[1], self.prev_keypoints[0])
            dist_2_1 = self.average_distance(detected_poses[1], self.prev_keypoints[1])

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
            csv_writer.writerow(["frame", "pose_id", "landmark_id", "x", "y", "z", "confidence"])

    def extract_landmarks(self):
        # Load Pose Landmarker
        with vision.PoseLandmarker.create_from_options(self.options) as landmarker:
            cap = cv2.VideoCapture(self.video_path)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            with tqdm(total=total_frames, desc="Extracting Poses", unit="frame") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Convert the frame to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(mp.ImageFormat.SRGB, frame_rgb)

                    # Perform pose detection on each frame
                    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    detection_result = landmarker.detect_for_video(mp_image, int(timestamp_ms))

                    # Save pose landmarks to CSV with consistent assignments
                    if detection_result.pose_landmarks:
                        for pose_id, pose_landmarks in enumerate(detection_result.pose_landmarks[:2]):  # up to 2 people
                            self.save_landmarks_to_csv(pose_landmarks, pose_id)

                    self.frame_index += 1
                    pbar.update(1)

            cap.release()
        print(f"Landmarks saved to {self.output_csv}")

    def save_landmarks_to_csv(self, pose_landmarks, pose_id):
        with open(self.output_csv, mode="a", newline="") as file:
            csv_writer = csv.writer(file)# Skip if None
            for landmark_id, landmark in enumerate(pose_landmarks):
                csv_writer.writerow(
                    [
                        self.frame_index,
                        pose_id,
                        landmark_id,
                        landmark.x,
                        landmark.y,
                        landmark.z,
                        landmark.visibility
                    ]
                )

def main():
    if len(sys.argv) < 2:
        print("Usage: python pose_landmark_extractor.py <video_path>")
        return

    video_path = sys.argv[1]
    output_csv_path = video_path.replace(".mp4", "_mediapipe.csv")
    
    model = "pose_landmarker.task"
    
    extractor = PoseLandmarkExtractor(video_path, model, output_csv_path)
    extractor.setup_csv()
    extractor.extract_landmarks()
    
if __name__ == "__main__":
    main()