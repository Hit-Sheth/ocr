import cv2
import pytesseract
from config import OCR_CONFIG

def is_blurry(image, threshold=100):
    """Check if the image is blurry using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold

def preprocess_and_ocr(image):
    """Preprocess the image and extract text using OCR."""
    # Skip if image is empty
    if image is None or image.size == 0:
        return ""

    # Blur check
    if is_blurry(image):
        return ""

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, h=30)

    # Thresholding
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # OCR
    try:
        custom_config = OCR_CONFIG + " --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-:.()"
        text = pytesseract.image_to_string(thresh, config=custom_config)
        return text.strip()
    except pytesseract.TesseractNotFoundError:
        return "[ERROR] Tesseract not found."
