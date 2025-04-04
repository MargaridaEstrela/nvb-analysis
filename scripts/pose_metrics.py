import pandas as pd

class PoseMetrics:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
    
    def frames_without_person(self, person_id):
        # Find frames where the specified person is missing
        frames_with_person = self.data[self.data['pose_id'] == person_id]['frame'].unique()
        total_frames = self.data['frame'].nunique()
        frames_without_person = total_frames - len(frames_with_person)
        
        return frames_without_person

    def get_metrics(self):
        # Calculate metrics for person 0 and person 1
        frames_without_person_0 = self.frames_without_person(0)
        frames_without_person_1 = self.frames_without_person(1)

        metrics = {
            "frames_without_person_0": frames_without_person_0,
            "frames_without_person_1": frames_without_person_1,
        }
        
        return metrics