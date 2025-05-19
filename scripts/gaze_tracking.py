import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm

class OpenFaceIDTracker:
    def __init__(self):
        self.KEY_LANDMARKS = [27, 30, 33, 36, 39, 42, 45] # Nose bridge, tip, and eyes
        self.MAX_MISSING_FRAMES = 50
        self.MAX_DISTANCE = 0.3
        self.REASSIGN_THRESHOLD = 0.3
        self.history = {}
        self.region_means = {0: [], 1: []}

    def group_landmarks(self, df):
        """
        Group all facial landmarks per frame and face_id.
        Returns a nested dictionary: { frame_number: { face_id: (landmarks, row_index) } }
        """
        grouped = defaultdict(dict)
        x_cols = [f'x_{i}' for i in range(68)]
        y_cols = [f'y_{i}' for i in range(68)]
        for (frame, face_id), grp in df.groupby(['frame', 'face_id']):
            row = grp.iloc[0]
            idx = row.name
            xs = row[x_cols].to_numpy()
            ys = row[y_cols].to_numpy()
            landmarks = np.stack([xs, ys], axis=1)  # shape (68, 2)
            grouped[int(frame)][int(face_id)] = (landmarks, idx)
        return grouped
    
    def average_distance(self, lm1, lm2):
        """
        Compute the mean distance between the same landmarks of two faces.
        Used to assess similarity between face detections.
        """
        dists = []
        
        for i in self.KEY_LANDMARKS:
            if np.isnan(lm1[i]).any() or np.isnan(lm2[i]).any():
                continue
            dists.append(np.linalg.norm(lm1[i] - lm2[i]))
            
        return np.mean(dists) if dists else float('inf')
    
    def get_mean_position(self, lm):
        valid = [lm[i] for i in self.KEY_LANDMARKS if not np.isnan(lm[i]).any()]
        return np.mean(valid, axis=0) if valid else np.array([np.nan, np.nan])

    def assign_consistent_ids(self, grouped):
        """
        Main function to assign consistent tracked IDs to detected face landmarks.
        Returns a dictionary mapping row_index → consistent tracked_id.
        """
        tracked = {}
        assignments = {}

        for frame in tqdm(sorted(grouped.keys()), desc='Tracking OpenFace IDs'):
            detections = grouped[frame]
            det_ids = list(detections.keys())

            if len(tracked) < 2 and len(det_ids) >= 2:
                x_means = {d: self.get_mean_position(detections[d][0])[0] for d in det_ids}
                left, right = sorted(x_means, key=x_means.get)[:2]
                
                for t_id, det in enumerate((left, right)):
                    lm, idx = detections[det]
                    tracked[t_id] = {'landmarks': lm, 'last_seen': frame, 'missing': 0}
                    assignments[idx] = t_id
                    
                    new_pos = self.get_mean_position(lm)
                    if len(self.region_means[t_id]) < 30:
                        self.region_means[t_id].append(new_pos)
                        
                continue

            pairs = []
            for t_id, info in tracked.items():
                for det_id, (lm, idx) in detections.items():
                    d = self.average_distance(info['landmarks'], lm)
                    pairs.append((d, t_id, det_id, idx))
                    
            pairs.sort(key=lambda x: x[0])

            used_tracks, used_dets = set(), set()

            for d, t_id, det_id, idx in pairs:
                if d > self.MAX_DISTANCE:
                    break
                if t_id in used_tracks or det_id in used_dets:
                    continue
                
                assignments[idx] = t_id
                tracked[t_id]['landmarks'] = detections[det_id][0]
                tracked[t_id]['last_seen'] = frame
                tracked[t_id]['missing'] = 0
                
                new_pos = self.get_mean_position(lm)
                if len(self.region_means[t_id]) < 30:
                    self.region_means[t_id].append(new_pos)
                    
                used_tracks.add(t_id)
                used_dets.add(det_id)

            for t_id in list(tracked):
                if tracked[t_id]['last_seen'] != frame:
                    tracked[t_id]['missing'] += 1
                    if tracked[t_id]['missing'] > self.MAX_MISSING_FRAMES:
                        del tracked[t_id]

            for det_id, (lm, idx) in detections.items():
                if det_id in used_dets:
                    continue

                found = False
                for past_idx, (past_lm, past_id) in self.history.items():
                    if self.average_distance(past_lm, lm) < self.REASSIGN_THRESHOLD:
                        if past_id not in tracked:
                            tracked[past_id] = {'landmarks': lm, 'last_seen': frame, 'missing': 0}
                            assignments[idx] = past_id
                            
                            new_pos = self.get_mean_position(lm)
                            if len(self.region_means[t_id]) < 30:
                                self.region_means[t_id].append(new_pos)
                            
                            found = True
                            break
                if found:
                    self.history[idx] = (lm, assignments[idx])
                    continue

                new_pos = self.get_mean_position(lm)
                mean_0 = np.mean(self.region_means[0], axis=0) if self.region_means[0] else None
                mean_1 = np.mean(self.region_means[1], axis=0) if self.region_means[1] else None
                dist_0 = np.linalg.norm(new_pos - mean_0) if mean_0 is not None else float('inf')
                dist_1 = np.linalg.norm(new_pos - mean_1) if mean_1 is not None else float('inf')

                t_id = 0 if dist_0 < dist_1 else 1
                tracked[t_id] = {'landmarks': lm, 'last_seen': frame, 'missing': 0}
                assignments[idx] = t_id
                self.region_means[t_id].append(new_pos)

                self.history[idx] = (lm, t_id)

        return assignments


    def run(self, openface_csv):
        """
        Load CSV, apply tracking, and write a new CSV with consistent tracked IDs.
        """
        df = pd.read_csv(openface_csv)
        grouped = self.group_landmarks(df)
        assignment_map = self.assign_consistent_ids(grouped)

        df['tracked_id'] = df.index.map(lambda i: assignment_map.get(i, -1))
        output_csv = openface_csv.replace('.csv', '_tracked.csv')
        df.to_csv(output_csv, index=False)
        print(f'✅ Done! Tracked CSV saved to {output_csv}')
        return output_csv

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python gaze_tracking.py <openface_csv>')
        sys.exit(1)
    tracker = OpenFaceIDTracker()
    tracker.run(sys.argv[1])
