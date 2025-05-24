# detect_overball.py

import cv2
import os
import json
from config import SCOREBOARD_ROI, FRAME_INTERVAL, OUTPUT_PATH
from utils.ocr_helpers import preprocess_and_ocr

def extract_scoreboard_from_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    ocr_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        print(f"Frame shape: {frame.shape}")
        if frame_count % FRAME_INTERVAL == 0:
            x, y, w, h = SCOREBOARD_ROI.values()
            scoreboard_region = frame[y:y+h, x:x+w]

            if scoreboard_region.size == 0:
                print(f"Empty frame at {frame_count}. Skipping.")
                continue

            text = preprocess_and_ocr(scoreboard_region)

            ocr_results.append({
                "frame": frame_count,
                "ocr_text": text
            })

        frame_count += 1

    cap.release()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(ocr_results, f, indent=2)

    print(f"OCR results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    extract_scoreboard_from_video("match_videos/sample_match2.mp4")
