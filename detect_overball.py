import cv2
import os
import json
import re
from utils.ocr_helpers import preprocess_and_ocr

# Paths and constants
VIDEO_PATH = "match_videos/sample_match2.mp4"
OUTPUT_PATH = "outputs/overball_output.json"

# Compute sampling interval (one read per second)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
ret, frame = cap.read()
size = frame.shape
FPS = cap.get(cv2.CAP_PROP_FPS)
FRAME_INTERVAL = int(FPS)
cap.release()

# Define ROIs (all values converted to absolute pixel coordinates)
frame_width, frame_height = size[1], size[0]
SCOREBOARD_ROI = {
    "x": int(500 / 1280 * frame_width),
    "y": int(622 / 720 * frame_height),
    "w": int(350 / 1280 * frame_width),
    "h": int(70 / 720 * frame_height)
}
BATSMAN_ROI = {
    "x": int(220 / 1280 * frame_width),
    "y": int(622 / 720 * frame_height),
    "w": int(105 / 1280 * frame_width),
    "h": int(70 / 720 * frame_height)
}
BOWLER_ROI = {
    "x": int(799 / 1280 * frame_width),
    "y": int(622 / 720 * frame_height),
    "w": int(137 / 1280 * frame_width),
    "h": int(35 / 720 * frame_height)
}
OVER_ROI = {
    "x": int(715 / 1280 * frame_width),
    "y": int(620 / 720 * frame_height),
    "w": int(60 / 1280 * frame_width),
    "h": int(35 / 720 * frame_height)
}

def parse_over(text, prev_over):
    """Parse over as float, handling cricket over progression rules.
    Args:
        text: OCR text containing over info (e.g., "5.3", "6", "O4")
        prev_over: Previous valid over (float, e.g., 5.3)
    
    Returns:
        float: Parsed over value or None if invalid
    """
    if not text:
        return None

    # Normalize text (replace O with 0, remove spaces)
    normalized = text.upper().replace('O', '0').replace(' ', '.')
    
    # Try standard decimal formats first (5.3, 5,3)
    decimal_match = re.search(r'(\d+)[\.,](\d+)', normalized)
    if decimal_match:
        over_int = int(decimal_match.group(1))
        over_dec = int(decimal_match.group(2))
        
        # Validate decimal part (must be 0-6)
        if over_dec > 6:
            return None
            
        return float(f"{over_int}.{over_dec}")

    # Handle cases where only digit is visible (e.g., "4" meaning 5.4 after 5.3)
    if prev_over is not None and normalized and normalized[-1].isdigit():
        last_digit = int(normalized[-1])
        prev_int = int(prev_over)
        prev_dec = int(round((prev_over % 1) * 10))  # Get decimal part as integer
        
        # Case 1: Same decimal (e.g., seeing "3" after 5.3 → remains 5.3)
        if last_digit == prev_dec:
            return prev_over
            
        # Case 2: Next decimal (e.g., seeing "4" after 5.3 → 5.4)
        elif last_digit == prev_dec + 1:
            return float(f"{prev_int}.{last_digit}")
            
        # Case 3: Over completion (e.g., seeing "6" after 5.5 → 6.0)
        elif prev_dec == 5 and last_digit == 6:
            return float(prev_int + 1)
            
        # Case 4: New over starting (e.g., seeing "0" after 5.6 → 6.0)
        elif prev_dec == 6 and last_digit == 0:
            return float(prev_int + 1)

    return None


def format_time(frame_idx):
    """Convert frame index to H:MM:SS."""
    total_s = frame_idx / FPS
    h = int(total_s // 3600)
    m = int((total_s % 3600) // 60)
    s = int(total_s % 60)
    return f"{h}:{m:02d}:{s:02d}"


def create_ball_entry(ball_id, start, end, over, bowler,
                      striker, non_striker, bat_team,
                      bowl_team, runs, wkts, temp_runs, prev_wkts,
                      sc_text):
    """Build one ball's metadata dict, including extras."""
    delta_runs   = runs   - temp_runs
    delta_wkts   = wkts   - prev_wkts
    return {
        "ball_id": ball_id,
        "start": start,
        "end": end,
        "over": round(over,1),
        "bowler": bowler or None,
        "stricker_batsman": striker or None,
        "nonstricker_batsman": non_striker or None,
        "batting_team": bat_team or None,
        "bowling_team": bowl_team or None,
        "four":      (delta_runs == 4),
        "six":       (delta_runs == 6),
        "one":       (delta_runs == 1),
        "two":       (delta_runs == 2),
        "three":     (delta_runs == 3),  
        "wide_ball": ("WD" in sc_text.upper()),
        "no_ball":   ("NB" in sc_text.upper()),
        "wicket":    (delta_wkts >= 1),
        "runout":    (delta_wkts >= 1 and "RUNOUT" in sc_text.upper())
    }

def extract_ball_metadata():
    cap = cv2.VideoCapture(VIDEO_PATH)
    balls = []
    ball_id = 1

    # Track the “current” ball
    prev_over        = None
    ball_start_time  = None
    temp_runs        = None
    prev_wkts        = 0
    cur_striker      = ""
    cur_non_striker  = ""
    cur_bowler       = ""
    cur_bat_team     = ""
    cur_bowl_team    = ""
    prev_runs            = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            # if a ball was in progress, close it out
            if prev_over is not None:
                end_t = format_time(frame_idx)
                balls.append(create_ball_entry(
                    ball_id, ball_start_time, end_t, prev_over, cur_bowler,
                    cur_striker, cur_non_striker, cur_bat_team, cur_bowl_team,
                    temp_runs, curr_wkts, prev_runs, prev_wkts, sc_text
                ))
            break

        if frame_idx % FRAME_INTERVAL == 0:
            # Crop ROIs
            sbx,sby,sbw,sbh = SCOREBOARD_ROI.values()
            ovx,ovy,ovw,ovh = OVER_ROI.values()
            btx,bty,btw,bth = BATSMAN_ROI.values()
            blx,bly,blw,blh = BOWLER_ROI.values()

            sb_img = frame[sby:sby+sbh, sbx:sbx+sbw]
            ov_img = frame[ovy:ovy+ovh, ovx:ovx+ovw]
            bt_img = frame[bty:bty+bth, btx:btx+btw]
            bl_img = frame[bly:bly+blh, blx:blx+blw]

            # OCR
            sc_text   = preprocess_and_ocr(sb_img)
            over_text = preprocess_and_ocr(ov_img)
            bt_text   = preprocess_and_ocr(bt_img)
            bl_text   = preprocess_and_ocr(bl_img)

            # Parse over

            this_over = parse_over(over_text, prev_over)
            if this_over is None:
                frame_idx += 1
                continue
            print(f"Frame {frame_idx}: Over {this_over} | Score: {sc_text} ")

            # Parse score “runs-wkts”
            m = re.search(r'(\d+)-(\d+)', sc_text)
            if not m:
                frame_idx += 1
                continue
            curr_runs = int(m.group(1))
            curr_wkts = int(m.group(2))

            # Teams
            parts = re.split(' ', sc_text.upper(), 1)
            if len(parts)==2:
                cur_bowl_team = re.sub(r'[^A-Z]', '', parts[0])
                cur_bat_team  = re.sub(r'[^A-Z]', '', parts[1].split()[0])

            # Batsmen
            names = bt_text.upper()
            if ' ' in names and (prev_wkts == 0 or prev_wkts + 1 == curr_wkts):
                names = names.split(' ', 1)
                cur_striker, cur_non_striker = names[0], names[1]
            # if temp_runs != None:
            #     run_delta = curr_runs - temp_runs
            #     wicket_delta = curr_wkts - prev_wkts

            # Bowler
            cur_bowler = re.sub(r'[^A-Z ]','', bl_text.upper()).strip()

            tstemp = format_time(frame_idx)
            
            # New ball?
            if prev_over is None:
                # first ball
                ball_start_time = tstemp
                prev_over = this_over
                temp_runs = curr_runs
                prev_runs = curr_runs
                prev_wkts = curr_wkts
            elif this_over != prev_over:
                # close out previous ball
                balls.append(create_ball_entry(
                    ball_id, ball_start_time, tstemp,
                    prev_over, cur_bowler, cur_striker, cur_non_striker,
                    cur_bat_team, cur_bowl_team,
                    temp_runs, curr_wkts, prev_runs, prev_wkts, sc_text
                ))
                ball_id += 1

                # if odd-run, swap striker
                if ((temp_runs - prev_runs) % 2 == 1) ^ (str(this_over).endswith('.1')):
                    cur_striker, cur_non_striker = cur_non_striker, cur_striker
                # start next ball
                ball_start_time = tstemp
                prev_over  = this_over
                # temp_runs  = curr_runs
                prev_runs = temp_runs
                prev_wkts  = curr_wkts
            else:
                # same over, update current ball
                temp_runs = curr_runs
            print(f"  Ball {ball_id} | Runs: {curr_runs} (prev: {prev_runs})")
        frame_idx += 1

    cap.release()

    # save JSON
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(balls, f, indent=2)

    print(f"Generated {len(balls)} ball entries → {OUTPUT_PATH}")


if __name__ == "__main__":
    extract_ball_metadata()
