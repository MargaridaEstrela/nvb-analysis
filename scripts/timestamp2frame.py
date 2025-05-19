import cv2
import csv

def parse_timestamp(ts_str):
    """Parses a timestamp in 'mm:ss' or 'ss' format into total seconds."""
    try:
        if ':' in ts_str:
            minutes, seconds = map(int, ts_str.split(':'))
            return minutes * 60 + seconds
        else:
            return float(ts_str)
    except:
        raise ValueError("Invalid time format. Use mm:ss or seconds (e.g., 2:05 or 125.0)")

def extract_frame_indices_interactive():
    video_path = input("Enter path to video: ").strip()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamps = []

    print("Enter timestamps for each round (format: mm:ss or ss). Press Enter to finish.")
    round_num = 1
    while True:
        user_input = input(f"Round {round_num}: ").strip()
        if user_input == "":
            break
        try:
            ts_seconds = parse_timestamp(user_input)
            frame_number = int(ts_seconds * fps)
            timestamps.append((f"Round {round_num}", user_input, frame_number))
            round_num += 1
        except ValueError as e:
            print(f"⚠️ {e}")

    cap.release()

    output_csv = "frames_by_round.csv"
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'time', 'frame'])
        writer.writerows(timestamps)

    print(f"\n✅ Saved {len(timestamps)} entries to {output_csv}")

if __name__ == "__main__":
    extract_frame_indices_interactive()