import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


class PoseIDTracker:
    def __init__(self, key_landmarks=None, max_missing_frames=1000, max_reassignment_distance=0.5):
        self.KEY_LANDMARKS = key_landmarks if key_landmarks is not None else [0, 11, 12, 23, 24]
        self.MAX_MISSING_FRAMES = max_missing_frames
        self.MAX_REASSIGNMENT_DISTANCE = max_reassignment_distance

    def load_landmark_csv(self, path):
        return pd.read_csv(path)

    def group_landmarks_by_frame_and_pose(self, df):
        grouped = defaultdict(dict)
        for (frame, pose_id), group in df.groupby(['frame', 'pose_id']):
            landmarks = group.sort_values('landmark_id')[['x', 'y', 'z', 'confidence']].to_numpy()
            landmarks[:, 0] = 1.0 - landmarks[:, 0]
            grouped[frame][pose_id] = landmarks
        return grouped

    def average_distance(self, kp1, kp2):
        distances = []
        for i in self.KEY_LANDMARKS:
            try:
                if np.any(np.isnan(kp1[i])) or np.any(np.isnan(kp2[i])):
                    continue
                d = np.linalg.norm(kp1[i][:3] - kp2[i][:3])
                distances.append(d)
            except IndexError:
                continue
        return np.mean(distances) if distances else float('inf')

    def assign_consistent_ids_with_memory(self, grouped_landmarks):
        consistent_data = []
        # Fixed IDs: 0 = left, 1 = right
        tracked_people = {
            0: {'history': [], 'last_seen': -1, 'side': 'left'},
            1: {'history': [], 'last_seen': -1, 'side': 'right'}
        }

        for frame in tqdm(sorted(grouped_landmarks.keys()), desc="Tracking"):
            current_poses = grouped_landmarks[frame]
            assignments = {}
            used_ids = set()

            # Sort by x-position (landmark 0) to ensure left/right
            sorted_poses = sorted(
                current_poses.items(),
                key=lambda item: item[1][0][0] if not np.isnan(item[1][0][0]) else float('inf')
            )

            for curr_pose_id, landmarks in sorted_poses:
                x_mean = np.nanmean(landmarks[:, 0])
                side = 'left' if x_mean < 0.5 else 'right'
                person_id = 0 if side == 'left' else 1

                assignments[curr_pose_id] = person_id
                tracked_people[person_id]['history'].append(landmarks)
                tracked_people[person_id]['last_seen'] = frame
                used_ids.add(person_id)

            # Fill NaNs for missing people
            for person_id in [0, 1]:
                if person_id not in used_ids:
                    last_seen = tracked_people[person_id]['last_seen']
                    if frame - last_seen > self.MAX_MISSING_FRAMES:
                        tracked_people[person_id]['history'] = []
                        tracked_people[person_id]['last_seen'] = -1
                        continue
                    nan_landmarks = np.full((33, 4), np.nan)
                    consistent_data.append((frame, person_id, nan_landmarks))

            # Add data for this frame
            for curr_pose_id, person_id in assignments.items():
                landmarks = current_poses[curr_pose_id]
                consistent_data.append((frame, person_id, landmarks))

        return consistent_data

    def write_corrected_csv(self, consistent_data, output_path):
        with open(output_path, 'w') as f:
            f.write("frame,pose_id,landmark_id,x,y,z,confidence\n")
            for frame, pose_id, landmarks in consistent_data:
                for landmark_id, (x, y, z, confidence) in enumerate(landmarks):
                    f.write(f"{frame},{pose_id},{landmark_id},{x},{y},{z},{confidence}\n")

    def run(self, input_csv_path):
        output_csv_path = str(input_csv_path).replace(".csv", "_fixed.csv")
        df = self.load_landmark_csv(input_csv_path)
        df['x'] = 1.0 - df['x']  # flip x-coordinates
        grouped = self.group_landmarks_by_frame_and_pose(df)
        corrected = self.assign_consistent_ids_with_memory(grouped)
        self.write_corrected_csv(corrected, output_csv_path)
        print(f"✅ Done! Corrected CSV saved to {output_csv_path}")
        return output_csv_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pose_tracker.py <input_csv_path>")
        sys.exit(1)

    tracker = PoseIDTracker()
    tracker.run(sys.argv[1])