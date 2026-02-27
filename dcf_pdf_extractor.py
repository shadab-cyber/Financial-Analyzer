# pdf_text_extractor.py
import pdfplumber
import pytesseract
from PIL import Image
import pandas as pd
import json
import io

# cv2 is optional — only needed for scanned PDF preprocessing
try:
    import cv2
    import numpy as np
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

# ----------------------------------
# STEP 1: PAGE TYPE DETECTION
# ----------------------------------
def is_scanned_page(page, text_threshold=30):
    text = page.extract_text()
    return text is None or len(text.strip()) < text_threshold


# ----------------------------------
# STEP 2: OCR WITH PREPROCESSING
# ----------------------------------
def preprocess_image(img):
    if not _CV2_OK:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    gray = cv2.medianBlur(gray, 3)
    return gray


def ocr_page(page):
    pil_image = page.to_image(resolution=150).original
    if _CV2_OK:
        import numpy as np
        open_cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        processed = preprocess_image(open_cv_img)
        text = pytesseract.image_to_string(processed, config="--psm 6")
    else:
        # Fallback: OCR directly on PIL image without OpenCV preprocessing
        text = pytesseract.image_to_string(pil_image, config="--psm 6")
    return text


# ----------------------------------
# STEP 3: TABLE EXTRACTION (OPTIONAL)
# ----------------------------------
def extract_tables_text(page):
    tables = page.extract_tables()
    extracted = []
    for table in tables:
        df = pd.DataFrame(table)
        extracted.append(df)
    return extracted


# ----------------------------------
# STEP 4: MAIN EXTRACTOR
# ----------------------------------
def extract_financials_from_pdf(pdf_path):
    text_blocks = []
    tables_all = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            if is_scanned_page(page):
                text = ocr_page(page)
            else:
                text = page.extract_text()

            if text:
                text_blocks.append(text)

            tables_all.extend(extract_tables_text(page))

    return {
        "structured_financials": {
            "raw_text": text_blocks
        },
        "tables": [df.to_dict() for df in tables_all]
    }
