# Region of Interest for the scoreboard (x, y, width, height)
SCOREBOARD_ROI = {
    "x": 0,
    "y": 609,
    "w": 1280,   
    "h": 110
}

# Video settings
FPS = 30  # Frames per second of your input video
FRAME_INTERVAL = 30  # Process every 3 seconds if video is 30 FPS

# OCR settings (optional)
OCR_CONFIG = "--psm 6"  # Assume a single uniform block of text

# Output file path
OUTPUT_PATH = "outputs/overball_output.json"
