import cv2

def extract_frame(video_path, frame_number, output_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number >= total_frames:
        print(f"Error: Frame number exceeds total frames ({total_frames}).")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()

    if ret:
        cv2.imwrite(output_path, frame)
        print(f"Frame {frame_number} saved to {output_path}")
    else:
        print("Error: Cannot read frame.")

    cap.release()

# Example usage:
video_path = '/Users/margaridaestrela/Documents/projects/gaips/emoshow/experimental_studies/gaips/7/videos/top.mp4'
frame_number = 500  # change to the frame you want
output_path = 'frame_1000.jpg'
extract_frame(video_path, frame_number, output_path)