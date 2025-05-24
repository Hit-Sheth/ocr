import cv2
from config import SCOREBOARD_ROI
import os

# Path to your match video
VIDEO_PATH = "match_videos/sample_match2.mp4"

# Frame number to save (e.g., 300)
TARGET_FRAME = 30
OUTPUT_IMAGE = "outputs/sample_frame_with_roi1.jpg"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0
saved = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count == TARGET_FRAME:
        x, y, w, h = SCOREBOARD_ROI.values()
        # Draw rectangle on frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Save image
        os.makedirs("outputs", exist_ok=True)
        cv2.imwrite(OUTPUT_IMAGE, frame)
        print(f"Saved frame {TARGET_FRAME} with ROI to {OUTPUT_IMAGE}")
        saved = True
        break

    frame_count += 1

cap.release()

if not saved:
    print(f"Frame {TARGET_FRAME} not found.")
