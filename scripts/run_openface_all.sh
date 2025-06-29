#!/bin/bash

# Change to OpenFace directory
cd ~/Documents/projects/repos/OpenFace || exit 1

# Base path to your experimental studies
BASE_PATH=~/Documents/projects/gaips/emoshow/experimental_studies/gaips

# Loop over session folders (2 to 10, change as needed)
for i in {1..31}; do
  VIDEO_PATH="$BASE_PATH/$i/videos/top.mp4"
  OUTPUT_DIR="$BASE_PATH/$i/results"

  if [ -f "$VIDEO_PATH" ]; then
    echo "Processing session $i..."
    ./build/bin/FaceLandmarkVidMulti \
      -f "$VIDEO_PATH" \
      -out_dir "$OUTPUT_DIR" \
      -gaze -pose -aus -landmarks -2Dfp -3Dfp -pdmparams -tracked
  else
    echo "Video not found for session $i: $VIDEO_PATH"
  fi
done
