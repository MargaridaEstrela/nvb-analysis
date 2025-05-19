#!/bin/bash

# Base path to your experimental studies
BASE_PATH=~/Documents/projects/gaips/emoshow/experimental_studies/gaips

# Loop over sessions (adjust range as needed)
for i in {2..24}; do
  CSV_PATH="$BASE_PATH/$i/results/top.csv"

  if [ -f "$CSV_PATH" ]; then
    echo "Running gaze tracking on session $i..."
    python gaze_tracking.py "$CSV_PATH"
  else
    echo "CSV not found for session $i: $CSV_PATH"
  fi
done
