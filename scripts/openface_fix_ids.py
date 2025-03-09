import os
import pandas as pd

input_csv = "../csv/openface_0808.csv"
file_base, file_ext = os.path.splitext(input_csv)
output_csv = file_base + "_fixed" + file_ext

frame_width = 1920

df = pd.read_csv(input_csv)

# Ensure necessary columns exist
iris_landmarks = ['eye_lmk_x_27', 'eye_lmk_x_23', 'eye_lmk_x_51', 'eye_lmk_x_55']

for col in iris_landmarks:
    if col not in df.columns:
        raise ValueError(f"CSV must contain iris landmark column: {col}")

# Calculate the mean x-coordinate of the iris landmarks
df['iris_mean_x'] = df[iris_landmarks].mean(axis=1)

# Assign face_id based on the iris mean position
df.loc[df['iris_mean_x'] < frame_width / 2, 'face_id'] = 0  # Left side
df.loc[df['iris_mean_x'] > frame_width / 2, 'face_id'] = 1  # Right side

# Save formatted CSV
df.to_csv(output_csv, index=False)
print(f"Fixed CSV saved as {output_csv}")
