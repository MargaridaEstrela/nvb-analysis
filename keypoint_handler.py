import pandas as pd
import numpy as np

class KeypointHandler:
    def __init__(self, data):
        # Separate data for person 0 and person 1
        self.data_person_0 = data[data['pose_id'] == 0].copy()
        self.data_person_1 = data[data['pose_id'] == 1].copy()
        self.epsilon = 1e-5      # Small non-zero value to prevent division by zero
        self.f_min = 0.7         # Minimum cutoff frequency
        self.beta = 0.0005        # Speed coefficient
        self.mean_alpha = 0.03   # Smoothing factor for EMA filter
        self.correction = 0.02   # Threshold value for correction
        self.speed_threshold = 0.1  # Speed threshold for correction

        # Baseline body proportions (from the anatomical image)
        self.base_upper_arm_length = 28.2
        self.base_forearm_length = 25.4
        self.base_arm_ratio = self.base_upper_arm_length / self.base_forearm_length
        self.base_head_to_body_ratio = 1 / 8

    def apply_speed_threshold_correction(self):
        # Correct joint positions if the speed exceeds a certain threshold
        for data in [self.data_person_0, self.data_person_1]:
            for landmark_id in data['landmark_id'].unique():
                landmark_data = data[data['landmark_id'] == landmark_id]
                
                prev_x, prev_y = landmark_data.iloc[0]['x'], landmark_data.iloc[0]['y']
                prev_time = 0
                
                corrected_x, corrected_y = [], []
                
                for i, row in landmark_data.iterrows():
                    current_time = row['frame']
                    dt = current_time - prev_time if prev_time != 0 else 1
                    if dt == 0:
                        dt = self.epsilon
                    
                    # Calculate velocity (speed)
                    vx = abs(row['x'] - prev_x) / dt
                    vy = abs(row['y'] - prev_y) / dt

                    # Correct if velocity exceeds speed threshold
                    if vx > self.speed_threshold or vy > self.speed_threshold:
                        new_x = prev_x  # Correct to previous value
                        new_y = prev_y
                    else:
                        new_x = row['x']
                        new_y = row['y']
                    
                    corrected_x.append(new_x)
                    corrected_y.append(new_y)

                    prev_x, prev_y = new_x, new_y
                    prev_time = current_time
                
                data.loc[data['landmark_id'] == landmark_id, 'x'] = corrected_x
                data.loc[data['landmark_id'] == landmark_id, 'y'] = corrected_y

    def apply_filters(self):
        # Apply One-Euro filter first, then Mean filter to smooth keypoints data for both persons
        self.data_person_0 = self._apply_filters_for_person(self.data_person_0)
        self.data_person_1 = self._apply_filters_for_person(self.data_person_1)

    def _apply_filters_for_person(self, data):
        data = self._apply_one_euro_filter(data)
        data = self._apply_mean_filter(data)
        return data

    def _apply_one_euro_filter(self, data):
        tau = 1 / (2 * np.pi * self.f_min)
        filtered_data = data.copy()

        for landmark_id in data['landmark_id'].unique():
            landmark_data = data[data['landmark_id'] == landmark_id]
            x_filtered, y_filtered, z_filtered = [], [], []

            prev_x, prev_y, prev_z = landmark_data.iloc[0]['x'], landmark_data.iloc[0]['y'], landmark_data.iloc[0]['z']
            prev_time = 0  

            for i, row in landmark_data.iterrows():
                current_time = row['frame']
                dt = current_time - prev_time if prev_time != 0 else 1
                if dt == 0:
                    dt = self.epsilon

                dx = abs(row['x'] - prev_x) / dt
                dy = abs(row['y'] - prev_y) / dt
                dz = abs(row['z'] - prev_z) / dt

                alpha_x = 1 / (1 + tau * dx)
                alpha_y = 1 / (1 + tau * dy)
                alpha_z = 1 / (1 + tau * dz)

                new_x = alpha_x * row['x'] + (1 - alpha_x) * prev_x
                new_y = alpha_y * row['y'] + (1 - alpha_y) * prev_y
                new_z = alpha_z * row['z'] + (1 - alpha_z) * prev_z

                x_filtered.append(new_x)
                y_filtered.append(new_y)
                z_filtered.append(new_z)

                prev_x, prev_y, prev_z = new_x, new_y, new_z
                prev_time = current_time

            filtered_data.loc[data['landmark_id'] == landmark_id, 'x'] = x_filtered
            filtered_data.loc[data['landmark_id'] == landmark_id, 'y'] = y_filtered
            filtered_data.loc[data['landmark_id'] == landmark_id, 'z'] = z_filtered

        return filtered_data

    def _apply_mean_filter(self, data):
        filtered_data = data.copy()

        for landmark_id in data['landmark_id'].unique():
            landmark_data = data[data['landmark_id'] == landmark_id]
            x_filtered, y_filtered, z_filtered = [], [], []

            prev_x, prev_y, prev_z = landmark_data.iloc[0]['x'], landmark_data.iloc[0]['y'], landmark_data.iloc[0]['z']
            for i, row in landmark_data.iterrows():
                new_x = self.mean_alpha * row['x'] + (1 - self.mean_alpha) * prev_x
                new_y = self.mean_alpha * row['y'] + (1 - self.mean_alpha) * prev_y
                new_z = self.mean_alpha * row['z'] + (1 - self.mean_alpha) * prev_z

                x_filtered.append(new_x)
                y_filtered.append(new_y)
                z_filtered.append(new_z)

                prev_x, prev_y, prev_z = new_x, new_y, new_z

            filtered_data.loc[data['landmark_id'] == landmark_id, 'x'] = x_filtered
            filtered_data.loc[data['landmark_id'] == landmark_id, 'y'] = y_filtered
            filtered_data.loc[data['landmark_id'] == landmark_id, 'z'] = z_filtered

        return filtered_data

    def calculate_head_length(self, data, nose_index=0, left_ear_index=7, right_ear_index=8):
        head_lengths = {}

        for pose_id in data["pose_id"].unique():
            person_data = data[data["pose_id"] == pose_id]
            
            nose_data = person_data[person_data["landmark_id"] == nose_index]
            if nose_data.empty:
                continue
            nose_y = nose_data.iloc[0]["y"]
            
            left_ear_data = person_data[person_data["landmark_id"] == left_ear_index]
            right_ear_data = person_data[person_data["landmark_id"] == right_ear_index]
            if left_ear_data.empty or right_ear_data.empty:
                continue
            
            left_ear_y = left_ear_data.iloc[0]["y"]
            right_ear_y = right_ear_data.iloc[0]["y"]
            avg_ear_y = (left_ear_y + right_ear_y) / 2
            
            head_length = abs(nose_y - avg_ear_y)
            head_lengths[pose_id] = head_length
    
        return head_lengths

    def calculate_arm_lengths(self, data, pose_id):
        limb_pairs = {
            "upper_arm_left": (11, 13),  # Left shoulder to left elbow
            "forearm_left": (13, 15),    # Left elbow to left wrist
            "upper_arm_right": (12, 14), # Right shoulder to right elbow
            "forearm_right": (14, 16),   # Right elbow to right wrist
        }
        
        limb_lengths = {}
        for limb_name, (start_landmark, end_landmark) in limb_pairs.items():
            start_data = data[(data["pose_id"] == pose_id) & (data["landmark_id"] == start_landmark)]
            end_data = data[(data["pose_id"] == pose_id) & (data["landmark_id"] == end_landmark)]
            
            if not start_data.empty and not end_data.empty:
                start_coords = start_data.iloc[0][["x", "y", "z"]].values
                end_coords = end_data.iloc[0][["x", "y", "z"]].values
                limb_length = np.linalg.norm(start_coords - end_coords)
                limb_lengths[limb_name] = limb_length

        return limb_lengths

    def normalize_z_values(self):
        # Combine data for both persons
        combined_data = pd.concat([self.data_person_0, self.data_person_1])
        head_lengths = self.calculate_head_length(combined_data)
        
        # Calculate arm lengths and inclination factors for each person
        arm_lengths_0 = self.calculate_arm_lengths(self.data_person_0, pose_id=0)
        arm_lengths_1 = self.calculate_arm_lengths(self.data_person_1, pose_id=1)

        inclination_0 = self._calculate_inclination_factor(arm_lengths_0)
        inclination_1 = self._calculate_inclination_factor(arm_lengths_1)

        # Normalize z-values for each person
        self.data_person_0 = self._normalize_z_for_person(self.data_person_0, head_lengths.get(0, 1), inclination_0)
        self.data_person_1 = self._normalize_z_for_person(self.data_person_1, head_lengths.get(1, 1), inclination_1)

    def _calculate_inclination_factor(self, arm_lengths):
        # Calculate inclination factor by comparing upper arm to forearm ratio with the baseline ratio
        upper_arm_length = (arm_lengths.get("upper_arm_left", 0) + arm_lengths.get("upper_arm_right", 0)) / 2
        forearm_length = (arm_lengths.get("forearm_left", 0) + arm_lengths.get("forearm_right", 0)) / 2
        
        if forearm_length == 0:
            return 1  # Avoid division by zero
        
        current_ratio = upper_arm_length / forearm_length
        inclination_factor = current_ratio / self.base_arm_ratio
        return inclination_factor

    def _normalize_z_for_person(self, data, head_length, inclination_factor):
        # Calculate body height with baseline head-to-body ratio and adjust by inclination factor
        body_height = 8 * head_length  # Body height assumed as 8 * head length based on baseline
        adjusted_body_height = body_height * inclination_factor  # Adjust based on inclination

        data['z'] = data['z'] / adjusted_body_height  # Scaling z values to be proportional to body height
        return data

    def process_data(self):
        # Wrapper function to call all data processing methods
        self.apply_speed_threshold_correction()  # Apply speed threshold correction first
        self.apply_filters()
        self.normalize_z_values()

    def get_processed_data(self):
        return pd.concat([self.data_person_0, self.data_person_1])
