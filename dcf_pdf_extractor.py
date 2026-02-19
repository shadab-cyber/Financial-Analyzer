# dcf_pdf_extractor.py
import platform
import pdfplumber
import pytesseract
import cv2
import numpy as np
from PIL import Image
import pandas as pd

# ✅ FIX: Cross-platform Tesseract path
# Windows → use the known install location
# Linux/Mac (all cloud servers) → tesseract is on PATH after: apt-get install tesseract-ocr
if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# On Linux/Mac no path needed — tesseract is found via PATH automatically


# -----------------------------
# SCANNED PAGE CHECK
# -----------------------------
def is_scanned_page(page, threshold=30):
    text = page.extract_text()
    return text is None or len(text.strip()) < threshold


# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    gray = cv2.medianBlur(gray, 3)
    return gray


def ocr_page(page):
    pil_img = page.to_image(resolution=300).original
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    processed = preprocess_image(img)
    return pytesseract.image_to_string(processed, config="--psm 6")


# -----------------------------
# EXTRACT RAW TEXT
# -----------------------------
def extract_text_from_pdf(pdf_path):
    texts = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            if is_scanned_page(page):
                text = ocr_page(page)
            else:
                text = page.extract_text()

            if text:
                texts.append(text.lower())

    return texts


# -----------------------------
# EXTRACT FCF NUMBERS (BASIC)
# -----------------------------
def extract_fcf_from_text(text_blocks):
    fcf_values = []

    for text in text_blocks:
        for line in text.splitlines():
            if "free cash flow" in line or "net cash from operating" in line:
                numbers = [int(s.replace(",", "")) for s in line.split()
                           if s.replace(",", "").isdigit()]
                if numbers:
                    fcf_values.append(numbers[-1])

    return fcf_values
