import cv2
import os
import json
import re
from utils.ocr_helpers import preprocess_and_ocr

# Paths and constants
video_path = "match_videos/sample_match2.mp4"
OUTPUT_PATH = "outputs/overball_output.json"

# Define ROIs
SCOREBOARD_ROI = {"x": 500, "y": 622, "w": 300, "h": 34}
BATSMAN_ROI = {"x": 220, "y": 622, "w": 105, "h": 70}
BOWLER_ROI = {"x": 799, "y": 622, "w": 137, "h": 35}

# Get FPS
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
FPS = cap.get(cv2.CAP_PROP_FPS)
FRAME_INTERVAL = int(FPS)
cap.release()


def extract_scoreboard_from_video():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    ocr_results = []

    # State tracking
    previous_runs = 0
    previous_wickets = 0
    stricker_batsman = ""
    nonstricker_batsman = ""
    batting_team = ""
    bowling_team = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_INTERVAL == 0:
            # Extract regions
            sb_x, sb_y, sb_w, sb_h = SCOREBOARD_ROI.values()
            scoreboard_img = frame[sb_y:sb_y+sb_h, sb_x:sb_x+sb_w]

            bat_x, bat_y, bat_w, bat_h = BATSMAN_ROI.values()
            batsman_img = frame[bat_y:bat_y+bat_h, bat_x:bat_x+bat_w]

            bowl_x, bowl_y, bowl_w, bowl_h = BOWLER_ROI.values()
            bowler_img = frame[bowl_y:bowl_y+bowl_h, bowl_x:bowl_x+bowl_w]

            if scoreboard_img.size == 0 or batsman_img.size == 0 or bowler_img.size == 0:
                print(f"Empty ROI at frame {frame_count}. Skipping.")
                continue

            # OCR text
            scoreboard_text = preprocess_and_ocr(scoreboard_img)
            batsman_text = preprocess_and_ocr(batsman_img)
            bowler_text = preprocess_and_ocr(bowler_img)

            if not scoreboard_text.strip():
                print(f"No scoreboard text in frame {frame_count}. Skipping.")
                continue

            # Timestamp
            seconds = frame_count / FPS
            timestamp = f"{int(seconds // 3600):02d}:{int((seconds % 3600) // 60):02d}:{int(seconds % 60):02d}"

            # Extract runs and wickets
            score_match = re.search(r'(\d{1,3})-(\d)', scoreboard_text)
            if not score_match:
                print(f"Invalid score format in frame {frame_count}. Skipping.")
                continue

            runs = int(score_match.group(1))
            wickets = int(score_match.group(2))

            # Extract team names using 'V' as separator
            team_line = scoreboard_text.upper()
            if 'V' in team_line:
                team_parts = team_line.split('V', 1)
                bowling_team = re.sub(r'[^A-Z]', '', team_parts[0])
                # Extract only the first word (team name) from team_parts[1]
                batting_team = re.sub(r'[^A-Z]', '', team_parts[1].split()[0])

            # Extract batsman names
            batsman_line = batsman_text.upper()
            if '\n' in batsman_line and (wickets == previous_wickets + 1 or previous_wickets == 0):
                batsmen = batsman_line.split('\n', 1)
                stricker_batsman = re.sub(r'[^A-Z]', '', batsmen[0])
                nonstricker_batsman = re.sub(r'[^A-Z]', '', batsmen[1])

            # Swap batsmen if only 1 run difference (single)
            if runs == previous_runs + 1:
                stricker_batsman, nonstricker_batsman = nonstricker_batsman, stricker_batsman

            # Save current score for next iteration
            previous_runs = runs
            previous_wickets = wickets

            ocr_results.append({
                "frame": frame_count,
                "timestamp": timestamp,
                "stricker_batsman": stricker_batsman,
                "nonstricker_batsman": nonstricker_batsman,
                "bowler": bowler_text,
                "batting_team": batting_team,
                "bowling_team": bowling_team,
                "runs": runs,
                "wickets": scoreboard_text
            })

        frame_count += 1

    cap.release()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(ocr_results, f, indent=2)

    print(f"OCR results saved to {OUTPUT_PATH}")


extract_scoreboard_from_video()
