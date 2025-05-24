import cv2
import pytesseract
import numpy as np

def preprocess_and_ocr(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Optional: Apply thresholding to improve OCR
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Optional: Resize to improve accuracy
    resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # Run Tesseract OCR
    custom_config = r'--oem 3 --psm 6'  # OEM 3 = default, PSM 6 = assume a block of text
    text = pytesseract.image_to_string(resized, config=custom_config)

    return text.strip()
