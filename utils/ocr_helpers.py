import cv2
import numpy as np
import easyocr

# Initialize EasyOCR reader once
reader = easyocr.Reader(['en'])

def is_blurry(image, threshold=100):
    """Check if the image is blurry using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold

def preprocess_and_ocr(image):
    """Preprocess the image and extract text using EasyOCR."""
    if image is None or image.size == 0:
        return ""

    # Blur check
    if is_blurry(image):
        return ""

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Denoising
    denoised = cv2.fastNlMeansDenoising(enhanced, h=30)

    # Invert for better OCR (sometimes helps)
    inverted = cv2.bitwise_not(denoised)

    # Run EasyOCR
    try:
        results = reader.readtext(inverted, detail=0)
        return " ".join(results).strip()
    except Exception as e:
        return f"[ERROR] EasyOCR failed: {str(e)}"
