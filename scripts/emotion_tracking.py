import os
import sys
import cv2
import pandas as pd
import numpy as np
from rmn import RMN

if len(sys.argv) < 2:
    print("Usage: python emotion_tracking.py <video_path>")
    sys.exit()

# Initialize video and RMN model
video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)
model = RMN()

output_csv = "csv/emotions_tracking.csv"

data = []

# Load pre-trained DNN face detector from OpenCV
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt.txt",  # Path to Caffe 'deploy' prototxt file
    "res10_300x300_ssd_iter_140000.caffemodel"  # Path to pre-trained Caffe model
)

frame_number = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    height, width, _ = frame.shape

    # Prepare the frame for DNN face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    detected_faces = []

    # Iterate over all detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Filter out weak detections
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x_min, y_min, x_max, y_max) = box.astype("int")
            w = x_max - x_min
            h = y_max - y_min
            detected_faces.append((x_min, y_min, w, h))

    # Divide the frame in the middle and assign person IDs based on position
    for x, y, w, h in detected_faces:
        center_x = x + w // 2
        if center_x < width // 2:
            person_id = 0  # Left half of the frame
        else:
            person_id = 1  # Right half of the frame

        # Extract the face region and predict emotion
        face_region = frame[y:y+h, x:x+w]
        emotion_data = model.detect_emotion_for_single_frame(face_region)

        if emotion_data:
            emo_label = emotion_data[0]['emo_label']

            # Accumulate data instead of appending to a DataFrame
            data.append({
                "Person_ID": person_id,
                "Frame_Number": frame_number,
                "Emotion": emo_label
            })

            # Draw the rectangle and ID around the face
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # cv2.putText(frame, f"ID: {person_id}, {emo_label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    frame_number += 1

    # cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Create DataFrame from accumulated data and save to CSV
if data:
    df = pd.DataFrame(data, columns=["Person_ID", "Frame_Number", "Emotion"])
    df.to_csv(output_csv, index=False)
else:
    print("No data to save to CSV.")
