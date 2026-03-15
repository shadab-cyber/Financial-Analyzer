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
                # Take only the FIRST number on the matched line — the current
                # year figure. Using .extend grabbed every number (prior years,
                # note references, page numbers) causing wildly wrong FCF.
                if nums:
                    extracted.append(nums[0])

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
                # Same fix as extract_cfo — only take the first number.
                if nums:
                    extracted.append(nums[0])

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
    """
    Calculate average YoY FCF growth rate with safety caps.

    Problems with the naive average:
    · Sign changes (negative → positive or vice versa) produce growth rates of
      −200%, +500%, etc. that make the average meaningless.
    · A single outlier year dominates.

    Fix:
    · Cap each individual YoY rate at ±50% before averaging.
    · Detect sign changes and flag them.
    · If more than half the YoY rates hit the cap, mark the result as
      'unreliable' so the caller can warn the user.

    Returns:
        (avg_growth_rate, warnings_list)
    """
    growth_rates = []
    warnings = []
    sign_changes = 0

    for i in range(1, len(fcf_list)):
        prev = fcf_list[i - 1]
        curr = fcf_list[i]

        if prev is None or prev == 0:
            warnings.append(f"Year {i}: skipped (prior FCF = 0 or missing)")
            continue

        # Detect sign change
        if (prev < 0 < curr) or (curr < 0 < prev):
            sign_changes += 1
            warnings.append(
                f"Year {i}: FCF sign changed ({round(prev,1)} → {round(curr,1)}) "
                f"— growth rate may be misleading"
            )

        raw_rate = (curr - prev) / abs(prev)

        # Cap at ±50% to prevent one outlier year from dominating the average
        CAP = 0.50
        if abs(raw_rate) > CAP:
            warnings.append(
                f"Year {i}: growth rate {round(raw_rate*100,1)}% capped at "
                f"{'+'  if raw_rate > 0 else ''}{int(CAP*100)}%"
            )
            raw_rate = CAP if raw_rate > 0 else -CAP

        growth_rates.append(raw_rate)

    if not growth_rates:
        return 0.0, ["No valid growth rates — defaulting to 0%"]

    avg = sum(growth_rates) / len(growth_rates)

    # Flag if more than half the rates were capped or involved sign changes
    n_capped = sum(1 for w in warnings if "capped" in w)
    if n_capped > len(growth_rates) / 2 or sign_changes >= 2:
        warnings.append(
            "⚠️ Growth rate estimate is unreliable due to volatile FCF history. "
            "Consider overriding manually."
        )

    return round(avg, 4), warnings


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
def calculate_terminal_value(last_forecast_fcf, wacc=None, terminal_growth=None):
    """
    Gordon Growth Model terminal value with g < WACC guard.

    If terminal_growth >= wacc the denominator goes to zero or negative,
    producing infinity or a negative TV which is financially nonsensical.
    We clamp g to wacc - 0.01 (i.e. at least 1pp below WACC) and warn.
    """
    w = wacc if wacc is not None else WACC
    g = terminal_growth if terminal_growth is not None else TERMINAL_GROWTH

    tv_warning = None
    if g >= w:
        g = w - 0.01
        tv_warning = (
            f"⚠️ Terminal growth ({round(g*100+1,1)}%) was ≥ WACC ({round(w*100,1)}%). "
            f"Clamped to {round(g*100,1)}% to avoid infinite terminal value."
        )

    tv = round((last_forecast_fcf * (1 + g)) / (w - g), 2)
    return tv, tv_warning


def discount_terminal_value(tv, years, rate):
    return round(tv / ((1 + rate) ** years), 2)


def calculate_enterprise_value(pv_fcf, pv_terminal):
    return round(sum(pv_fcf) + pv_terminal, 2)


# =========================
# MASTER PIPELINE
# =========================
def run_dcf_from_pdf_text(text_blocks, net_debt=0, shares_outstanding=None,
                           current_price=None, wacc=None, terminal_growth=None):
    """
    Complete DCF pipeline.

    New parameters (all optional with sensible defaults):
        net_debt           – Net Debt = Total Debt − Cash (₹ Cr). Used to
                             convert Enterprise Value → Equity Value.
                             Pass a negative number if company has net cash.
        shares_outstanding – Shares in Crore. Used for per-share intrinsic value.
        current_price      – Current market price (₹). Used for margin of safety.
        wacc               – Override module-level WACC constant (decimal, e.g. 0.12)
        terminal_growth    – Override module-level TERMINAL_GROWTH (decimal, e.g. 0.05)
    """
    w  = wacc            if wacc            is not None else WACC
    g  = terminal_growth if terminal_growth is not None else TERMINAL_GROWTH

    historical_fcf, cfo, capex = calculate_fcf_from_text(text_blocks)

    avg_growth, growth_warnings = calculate_avg_fcf_growth(historical_fcf)
    forecast = forecast_fcf(historical_fcf[-1], avg_growth)

    # Guard: negative last FCF makes forecasting unreliable
    if historical_fcf[-1] < 0:
        growth_warnings.append(
            "⚠️ Most recent FCF is negative. Forecast may be unreliable — "
            "consider using a normalised FCF base."
        )

    discounted_fcf = discount_cash_flows(forecast, w)

    terminal_value, tv_warning = calculate_terminal_value(forecast[-1], wacc=w, terminal_growth=g)
    if tv_warning:
        growth_warnings.append(tv_warning)

    discounted_terminal = discount_terminal_value(terminal_value, FORECAST_YEARS, w)
    enterprise_value    = calculate_enterprise_value(discounted_fcf, discounted_terminal)

    # ── Equity Value & per-share intrinsic value ──────────────────────────────
    # Enterprise Value = Equity Value + Net Debt
    # ∴  Equity Value  = Enterprise Value − Net Debt
    net_debt_used    = float(net_debt) if net_debt is not None else 0.0
    equity_value     = round(enterprise_value - net_debt_used, 2)

    intrinsic_per_share  = None
    margin_of_safety_pct = None

    if shares_outstanding and float(shares_outstanding) > 0:
        shares = float(shares_outstanding)
        # Equity value is in ₹ Crore; shares in Crore → per-share in ₹
        intrinsic_per_share = round((equity_value * 1e7) / (shares * 1e7), 2)
        # = equity_value / shares  (both in Crore, so Cr/Cr = ₹ per share)

        if current_price and float(current_price) > 0 and intrinsic_per_share:
            cmp = float(current_price)
            margin_of_safety_pct = round(
                (intrinsic_per_share - cmp) / cmp * 100, 2
            )

    return {
        "Historical FCF (₹ Cr)":        historical_fcf,
        "Average FCF Growth Rate (%)":   round(avg_growth * 100, 2),
        "Forecast FCF (₹ Cr)":           forecast,
        "Discounted FCF (₹ Cr)":         discounted_fcf,
        "Terminal Value (₹ Cr)":         terminal_value,
        "Discounted Terminal Value (₹ Cr)": discounted_terminal,
        "Enterprise Value (₹ Cr)":       enterprise_value,
        "Net Debt (₹ Cr)":               net_debt_used,
        "Equity Value (₹ Cr)":           equity_value,
        "Intrinsic Value per Share (₹)": intrinsic_per_share,
        "Current Market Price (₹)":      float(current_price) if current_price else None,
        "Margin of Safety (%)":          margin_of_safety_pct,
        "Confidence Score (%)":          extraction_confidence(cfo, capex),
        "Growth Warnings":               growth_warnings,
        "Assumptions": {
            "WACC (%)":           round(w * 100, 2),
            "Terminal Growth (%)": round(g * 100, 2),
            "Forecast Years":     FORECAST_YEARS,
            "Net Debt (₹ Cr)":    net_debt_used,
            "Shares (Cr)":        float(shares_outstanding) if shares_outstanding else None,
        }
    }

# =========================
# PDF → TEXT → DCF WRAPPER
# =========================

def run_dcf_from_pdfs(pdf_paths, net_debt=0, shares_outstanding=None,
                       current_price=None, wacc=None, terminal_growth=None):
    """
    Accepts list of PDF file paths.
    Returns DCF valuation result.
    """
    all_text_blocks = []

    for pdf in pdf_paths:
        extracted = extract_financials_from_pdf(pdf)
        all_text_blocks.extend(extracted["structured_financials"]["raw_text"])

    return run_dcf_from_pdf_text(
        all_text_blocks,
        net_debt=net_debt,
        shares_outstanding=shares_outstanding,
        current_price=current_price,
        wacc=wacc,
        terminal_growth=terminal_growth,
    )
