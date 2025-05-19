#!/bin/bash

for i in {1..24}; do
    echo "Running with number: $i"
    python pose_estimation.py /Users/margaridaestrela/Documents/projects/gaips/emoshow/experimental_studies/gaips "$i"
done