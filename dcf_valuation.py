# dcf_valuation.py
import re
from pdf_text_extractor import extract_financials_from_pdf


# =========================
# SAFE NUMBER PARSER
# =========================
def clean_number(value):
    if value is None:
        return None

    value = str(value).strip()

    if value.startswith("(") and value.endswith(")"):
        value = "-" + value[1:-1]

    value = value.lower()
    value = value.replace(",", "")
    value = value.replace("₹", "").replace("rs", "").replace("cr", "").replace("crore", "")
    value = re.sub(r"[^0-9.\-]", "", value)

    if value in ("", "-", ".", None):
        return None

    try:
        return float(value)
    except ValueError:
        return None


def extract_numbers_from_line(line):
    tokens = re.findall(r"[-()]?[0-9,]+(?:\.[0-9]+)?", line)
    numbers = [clean_number(t) for t in tokens]
    return [n for n in numbers if n is not None]


# =========================
# KEYWORDS
# =========================
CFO_KEYWORDS = [
    "net cash from operating",
    "cash flow from operating",
    "cash generated from operations",
    "operating activities"
]

CAPEX_KEYWORDS = [
    "purchase of property",
    "purchase of plant",
    "capital expenditure",
    "capex",
    "acquisition of fixed assets"
]


# =========================
# STEP 1 — EXTRACT CFO
# =========================
def extract_cfo(text_blocks):
    extracted = []

    for text in text_blocks:
        lines = text.lower().splitlines()
        for i, line in enumerate(lines):
            if any(k in line for k in CFO_KEYWORDS):
                combined = line
                for j in range(1, 4):
                    if i + j < len(lines):
                        combined += " " + lines[i + j]

                nums = extract_numbers_from_line(combined)
                extracted.extend(nums)

    return extracted


# =========================
# STEP 2 — EXTRACT CAPEX
# =========================
def extract_capex(text_blocks):
    extracted = []

    for text in text_blocks:
        lines = text.lower().splitlines()
        for i, line in enumerate(lines):
            if any(k in line for k in CAPEX_KEYWORDS):
                combined = line
                for j in range(1, 4):
                    if i + j < len(lines):
                        combined += " " + lines[i + j]

                nums = extract_numbers_from_line(combined)
                extracted.extend(nums)

    return extracted


# =========================
# STEP 3 — COMPUTE FCF (UPDATED)
# =========================
def calculate_fcf_from_text(text_blocks):
    cfo = extract_cfo(text_blocks)
    capex = extract_capex(text_blocks)

    if not cfo or not capex:
        raise ValueError("Unable to extract CFO or CapEx from annual report")

    years = min(len(cfo), len(capex))
    fcf = []

    for i in range(min(years, 5)):  # enforce 5 years here
        fcf.append(round(cfo[i] - abs(capex[i]), 2))

    return fcf, cfo, capex


# =========================
# EXTRACTION CONFIDENCE
# =========================
def extraction_confidence(cfo, capex):
    score = 0

    if len(cfo) >= 5:
        score += 50
    if len(capex) >= 5:
        score += 30
    if all(isinstance(x, (int, float)) for x in cfo + capex):
        score += 20

    return score


# =========================
# DCF CONFIGURATION
# =========================
FORECAST_YEARS = 5
WACC = 0.11
TERMINAL_GROWTH = 0.04


# =========================
# AVG FCF GROWTH
# =========================
def calculate_avg_fcf_growth(fcf_list):
    growth_rates = []
    for i in range(1, len(fcf_list)):
        prev = fcf_list[i - 1]
        curr = fcf_list[i]
        if prev not in (None, 0):
            growth_rates.append((curr - prev) / abs(prev))
    return sum(growth_rates) / len(growth_rates) if growth_rates else 0.0


# =========================
# FORECAST FCF
# =========================
def forecast_fcf(last_fcf, growth_rate):
    forecast = []
    current = last_fcf
    for _ in range(FORECAST_YEARS):
        current *= (1 + growth_rate)
        forecast.append(round(current, 2))
    return forecast


# =========================
# DISCOUNT FCF
# =========================
def discount_cash_flows(cash_flows, rate):
    return [round(cf / ((1 + rate) ** i), 2) for i, cf in enumerate(cash_flows, 1)]


# =========================
# TERMINAL VALUE
# =========================
def calculate_terminal_value(last_forecast_fcf):
    return round(
        (last_forecast_fcf * (1 + TERMINAL_GROWTH)) / (WACC - TERMINAL_GROWTH),
        2
    )


def discount_terminal_value(tv, years, rate):
    return round(tv / ((1 + rate) ** years), 2)


def calculate_enterprise_value(pv_fcf, pv_terminal):
    return round(sum(pv_fcf) + pv_terminal, 2)


# =========================
# MASTER PIPELINE
# =========================
def run_dcf_from_pdf_text(text_blocks):
    historical_fcf, cfo, capex = calculate_fcf_from_text(text_blocks)

    avg_growth = calculate_avg_fcf_growth(historical_fcf)
    forecast = forecast_fcf(historical_fcf[-1], avg_growth)
    discounted_fcf = discount_cash_flows(forecast, WACC)

    terminal_value = calculate_terminal_value(forecast[-1])
    discounted_terminal = discount_terminal_value(
        terminal_value, FORECAST_YEARS, WACC
    )

    enterprise_value = calculate_enterprise_value(
        discounted_fcf, discounted_terminal
    )

    return {
        "Historical FCF (₹ Cr)": historical_fcf,
        "Average FCF Growth Rate (%)": round(avg_growth * 100, 2),
        "Forecast FCF (₹ Cr)": forecast,
        "Discounted FCF (₹ Cr)": discounted_fcf,
        "Terminal Value (₹ Cr)": terminal_value,
        "Discounted Terminal Value (₹ Cr)": discounted_terminal,
        "Enterprise Value (₹ Cr)": enterprise_value,
        "Confidence Score (%)": extraction_confidence(cfo, capex),
        "Assumptions": {
            "WACC (%)": int(WACC * 100),
            "Terminal Growth (%)": int(TERMINAL_GROWTH * 100),
            "Forecast Years": FORECAST_YEARS
        }
    }

# =========================
# PDF → TEXT → DCF WRAPPER
# =========================
from pdf_text_extractor import extract_financials_from_pdf

def run_dcf_from_pdfs(pdf_paths):
    """
    Accepts list of PDF file paths
    Returns DCF valuation result
    """
    all_text_blocks = []

    for pdf in pdf_paths:
        extracted = extract_financials_from_pdf(pdf)
        all_text_blocks.extend(extracted["structured_financials"]["raw_text"])

    return run_dcf_from_pdf_text(all_text_blocks)
