import pandas as pd
import numpy as np

openpose_path = "csv/openpose.csv"
openpose_data = pd.read_csv(openpose_path)

# Add a 'frame' column assuming each row corresponds to a new frame
openpose_data["frame"] = openpose_data.index

# Prepare to reshape the data into long format
long_data = []

# Iterate over each person and each keypoint to reshape the data into the desired format
for person in ["P1", "P2"]:
    pose_id = 0 if person == "P1" else 1
    for k in range(1, 20):  # 19 keypoints per person
        # Check if x, y columns exist for this keypoint
        x_col = f"{person}_K{k}_x"
        y_col = f"{person}_K{k}_y"

        # Check if the required columns exist
        if x_col in openpose_data.columns and y_col in openpose_data.columns:
            # Extract x, y, (and z if available) columns along with frame and assign pose_id and landmark_id
            keypoint_data = openpose_data[["frame", x_col, y_col]].copy()
            keypoint_data = keypoint_data.rename(columns={x_col: "x", y_col: "y"})
            keypoint_data["pose_id"] = pose_id
            keypoint_data["landmark_id"] = k - 1  # Set landmark ID (0-based index)

            # Fill z column with NaN
            keypoint_data["z"] = pd.NA

            long_data.append(keypoint_data)

# Concatenate all data into a single long format DataFrame
openpose_long_formatted = pd.concat(long_data, ignore_index=True)
openpose_long_formatted = openpose_long_formatted[["frame", "pose_id", "landmark_id", "x", "y", "z"]]
openpose_long_formatted.to_csv("csv/openpose_formatted.csv", index=False)
