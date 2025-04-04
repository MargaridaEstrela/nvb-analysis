import sys
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# === CONFIGURATION ===
KEY_LANDMARKS = [0, 11, 12, 23, 24]  # Nose, shoulders, hips
MAX_MISSING_FRAMES = 1000  # How many frames to remember a person after they disappear


def load_landmark_csv(path):
    return pd.read_csv(path)


def group_landmarks_by_frame_and_pose(df):
    grouped = defaultdict(dict)
    for (frame, pose_id), group in df.groupby(['frame', 'pose_id']):
        landmarks = group.sort_values('landmark_id')[['x', 'y', 'z']].to_numpy()
        grouped[frame][pose_id] = landmarks
    return grouped


def average_distance(kp1, kp2, indices=KEY_LANDMARKS):
    distances = []
    for i in indices:
        try:
            d = np.linalg.norm(kp1[i] - kp2[i])
            distances.append(d)
        except IndexError:
            continue
    return np.mean(distances) if distances else float('inf')


def assign_consistent_ids_with_memory(grouped_landmarks):
    consistent_data = []
    tracked_people = {}  # person_id: {'landmarks': ..., 'last_seen': frame}
    next_person_id = 0

    for frame in tqdm(sorted(grouped_landmarks.keys()), desc="Tracking"):
        current_poses = grouped_landmarks[frame]
        assignments = {}
        used_ids = set()

        # Match current poses to previously tracked people
        for curr_pose_id, landmarks in current_poses.items():
            best_match = None
            best_dist = float('inf')
            for person_id, info in tracked_people.items():
                if frame - info['last_seen'] > MAX_MISSING_FRAMES or person_id in used_ids:
                    continue
                dist = average_distance(landmarks, info['landmarks'])
                if dist < best_dist:
                    best_dist = dist
                    best_match = person_id
            if best_match is not None:
                assignments[curr_pose_id] = best_match
                tracked_people[best_match] = {'landmarks': landmarks, 'last_seen': frame}
                used_ids.add(best_match)
            else:
                # New person
                assignments[curr_pose_id] = next_person_id
                tracked_people[next_person_id] = {'landmarks': landmarks, 'last_seen': frame}
                used_ids.add(next_person_id)
                next_person_id += 1

        # Add NaN placeholders for missing people
        for person_id in list(tracked_people.keys()):
            last_seen = tracked_people[person_id]['last_seen']
            if frame - last_seen > MAX_MISSING_FRAMES:
                del tracked_people[person_id]
                continue
            if person_id not in used_ids:
                nan_landmarks = np.full((33, 3), np.nan)
                consistent_data.append((frame, person_id, nan_landmarks))

        # Save matched people
        for curr_pose_id, person_id in assignments.items():
            landmarks = current_poses[curr_pose_id]
            consistent_data.append((frame, person_id, landmarks))

    return consistent_data

def write_corrected_csv(consistent_data, output_path):
    with open(output_path, 'w') as f:
        f.write("frame,pose_id,landmark_id,x,y,z\n")
        for frame, pose_id, landmarks in consistent_data:
            for landmark_id, (x, y, z) in enumerate(landmarks):
                f.write(f"{frame},{pose_id},{landmark_id},{x},{y},{z}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python pose_tracker.py <input_csv_path>")
        sys.exit(1)

    input_csv_path = sys.argv[1]
    output_csv_path = input_csv_path.replace(".csv", "_fixed.csv")

    df = load_landmark_csv(input_csv_path)
    grouped = group_landmarks_by_frame_and_pose(df)
    corrected = assign_consistent_ids_with_memory(grouped)
    write_corrected_csv(corrected, output_csv_path)

    print(f"✅ Done! Corrected CSV saved to {output_csv_path}")
    
if __name__ == "__main__":
    main()