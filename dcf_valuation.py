# dcf_valuation.py
import re
from pdf_text_extractor import extract_financials_from_pdf


# =============================================================================
# NUMBER PARSER
# =============================================================================

def clean_number(value):
    """
    Parse a string token into a float.
    Handles Indian accounting format: (1,234.56) → -1234.56
    """
    if value is None:
        return None
    value = str(value).strip()

    # Parentheses = negative in accounting notation
    if re.match(r'^\([\d,. ]+\)$', value):
        value = '-' + value[1:-1]

    value = re.sub(r'[₹रु]', '', value)
    value = re.sub(r'\b(rs\.?|inr|cr\.?|crore|lakh|lakhs)\b', '', value, flags=re.I)
    value = value.replace(',', '').strip()
    value = re.sub(r'[^0-9.\-]', '', value)

    if value in ('', '-', '.'):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def extract_first_number(line):
    """
    Extract the first valid accounting number from a line.
    Ignores: single digits, note refs (1-99), years (1900-2099).
    """
    tokens = re.findall(r'\([0-9,]+(?:\.[0-9]+)?\)|[-]?[0-9,]+(?:\.[0-9]+)?', line)
    for tok in tokens:
        n = clean_number(tok)
        if n is None:
            continue
        abs_n = abs(n)
        if abs_n < 100 or (1900 <= abs_n <= 2099):
            continue
        return n
    return None


def extract_numbers_context(lines, idx, window=3):
    """Search for a number on line[idx] and the next window lines."""
    for offset in range(window + 1):
        j = idx + offset
        if j >= len(lines):
            break
        n = extract_first_number(lines[j])
        if n is not None:
            return n, offset
    return None, None


# =============================================================================
# KEYWORD PATTERNS
# =============================================================================

CFO_TOTAL_PHRASES = [
    r"net cash (generated from|from|used in|inflow from|outflow from|flow from) operating",
    r"net cash (generated|generated from) operations",
    r"cash (generated|used|inflow|outflow) (from|in) operating activities",
    r"cash flow from operating activities",
    r"cash flows from operating activities",
    r"net cash flow from operating activities",
    r"net cash flows from operating activities",
    r"cash inflow from operations",
    r"cash generated from operations",
    r"net cash from operations",
    r"total.*operating activities",
    r"operating activities.*total",
    r"cash from operating activit",
    r"operating cash flow",
    r"\bcfo\b",
]

CAPEX_PHRASES = [
    r"purchase of (property|plant|ppe|fixed assets|tangible|intangible)",
    r"acquisition of (property|plant|ppe|fixed assets|tangible|intangible)",
    r"payment for (property|plant|ppe|fixed assets|tangible|intangible)",
    r"addition(s)? to (property|plant|ppe|fixed assets|tangible|intangible)",
    r"capital expenditure",
    r"\bcapex\b",
    r"purchase of capital assets",
    r"investment in (fixed assets|capital assets|ppe)",
    r"cash paid for (property|plant|assets)",
    r"net purchase of fixed assets",
    r"fixed assets purchased",
    r"purchase of investments.*fixed",
]

CFO_EXCLUSION_PHRASES = [
    r"depreciation", r"amortis", r"amortiz",
    r"working capital", r"trade receivable", r"trade payable",
    r"inventories", r"provisions", r"deferred tax",
    r"interest paid", r"interest received", r"dividend",
    r"income tax", r"tax paid", r"adjustment for",
    r"profit before", r"profit after", r"loss before",
    r"other income", r"finance cost",
]


def _compile(phrases):
    return [re.compile(p, re.IGNORECASE) for p in phrases]


_CFO_RE   = _compile(CFO_TOTAL_PHRASES)
_CAPEX_RE = _compile(CAPEX_PHRASES)
_EXCL_RE  = _compile(CFO_EXCLUSION_PHRASES)


def _matches_any(line, patterns):
    for pat in patterns:
        if pat.search(line):
            return True
    return False


# =============================================================================
# EXTRACT CFO
# =============================================================================

def extract_cfo(text_blocks):
    results = []

    for text in text_blocks:
        lines = text.splitlines()
        candidates = []

        for i, line in enumerate(lines):
            ll = line.lower().strip()
            if not ll:
                continue
            if not _matches_any(ll, _CFO_RE):
                continue
            if _matches_any(ll, _EXCL_RE):
                continue

            n, offset = extract_numbers_context(lines, i, window=3)
            if n is None:
                continue

            confidence = 3 - offset
            if re.search(r'\b(net|total)\b', ll):
                confidence += 2

            candidates.append((i, n, confidence))

        if not candidates:
            continue

        best = sorted(candidates, key=lambda x: (x[2], x[0]), reverse=True)[0]
        results.append(best[1])

    return results


# =============================================================================
# EXTRACT CAPEX
# =============================================================================

def extract_capex(text_blocks):
    results = []

    for text in text_blocks:
        lines = text.splitlines()
        candidates = []

        for i, line in enumerate(lines):
            ll = line.lower().strip()
            if not ll:
                continue
            if not _matches_any(ll, _CAPEX_RE):
                continue

            n, offset = extract_numbers_context(lines, i, window=3)
            if n is None:
                continue

            confidence = 3 - offset
            candidates.append((i, abs(n), confidence))

        if not candidates:
            continue

        best = sorted(candidates, key=lambda x: (x[2], x[1]), reverse=True)[0]
        results.append(best[1])

    return results


# =============================================================================
# MULTI-BLOCK STRATEGY
# =============================================================================

def _split_into_year_blocks(text_blocks):
    if len(text_blocks) >= 2:
        return text_blocks

    combined = '\n'.join(text_blocks)
    year_markers = list(re.finditer(
        r'\b(march\s+31[,\s]+20\d{2}|31(st)?\s+march[,\s]+20\d{2}|fy\s?20\d{2})\b',
        combined, re.IGNORECASE
    ))
    if len(year_markers) >= 2:
        blocks = []
        for idx, m in enumerate(year_markers):
            start = m.start()
            end   = year_markers[idx + 1].start() if idx + 1 < len(year_markers) else len(combined)
            blocks.append(combined[start:end])
        return blocks

    return text_blocks


def _extract_all_values(text_blocks, patterns, exclusions, positive=False):
    results = []
    for text in text_blocks:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            ll = line.lower().strip()
            if not ll:
                continue
            if not _matches_any(ll, patterns):
                continue
            if exclusions and _matches_any(ll, exclusions):
                continue
            n, _ = extract_numbers_context(lines, i, window=3)
            if n is not None:
                results.append(abs(n) if positive else n)
    return results


# =============================================================================
# COMPUTE FCF
# =============================================================================

def calculate_fcf_from_text(text_blocks):
    blocks = _split_into_year_blocks(text_blocks)

    cfo   = extract_cfo(blocks)
    capex = extract_capex(blocks)

    # Fallback: treat entire corpus as one block
    if not cfo or not capex:
        combined = ['\n'.join(text_blocks)]
        if not cfo:
            cfo   = _extract_all_values(combined, _CFO_RE,   _EXCL_RE, positive=False)
        if not capex:
            capex = _extract_all_values(combined, _CAPEX_RE, [],       positive=True)

    if not cfo:
        raise ValueError(
            "Could not find CFO (Cash from Operating Activities) in the uploaded PDFs.\n"
            "Common reasons:\n"
            "  • The PDF is a scanned image — use a text-based digital annual report.\n"
            "  • The cash flow statement uses an unusual label — try Manual Input mode.\n"
            "  • The PDF is in a language other than English."
        )
    if not capex:
        raise ValueError(
            "Could not find CAPEX (Purchase of Fixed Assets / Capital Expenditure) "
            "in the uploaded PDFs.\n"
            "Try Manual Input mode and enter the values directly from Screener.in."
        )

    years = min(len(cfo), len(capex))
    fcf   = [round(cfo[i] - abs(capex[i]), 2) for i in range(min(years, 5))]

    return fcf, cfo[:5], capex[:5]


# =============================================================================
# EXTRACTION CONFIDENCE
# =============================================================================

def extraction_confidence(cfo, capex):
    score = 0
    n = min(len(cfo), len(capex))

    if n >= 5:   score += 40
    elif n >= 3: score += 25
    elif n >= 2: score += 10

    positive_cfo = sum(1 for v in cfo if v > 0)
    if cfo and positive_cfo / len(cfo) >= 0.6:
        score += 20

    if cfo and capex:
        avg_cfo   = sum(cfo)   / len(cfo)
        avg_capex = sum(capex) / len(capex)
        if avg_capex < abs(avg_cfo) * 2:
            score += 20

    if all(isinstance(x, (int, float)) for x in cfo + capex):
        score += 20

    return min(score, 100)


# =============================================================================
# DCF CONFIGURATION
# =============================================================================

FORECAST_YEARS  = 5
WACC            = 0.11
TERMINAL_GROWTH = 0.04


# =============================================================================
# AVG FCF GROWTH
# =============================================================================

def calculate_avg_fcf_growth(fcf_list):
    growth_rates = []
    warnings     = []
    sign_changes = 0

    for i in range(1, len(fcf_list)):
        prev = fcf_list[i - 1]
        curr = fcf_list[i]

        if prev is None or prev == 0:
            warnings.append(f"Year {i}: skipped (prior FCF = 0 or missing)")
            continue

        if (prev < 0 < curr) or (curr < 0 < prev):
            sign_changes += 1
            warnings.append(
                f"Year {i}: FCF sign changed ({round(prev,1)} → {round(curr,1)}) "
                "— growth rate may be misleading"
            )

        raw_rate = (curr - prev) / abs(prev)
        CAP      = 0.50
        if abs(raw_rate) > CAP:
            warnings.append(
                f"Year {i}: growth rate {round(raw_rate*100,1)}% capped at "
                f"{'+'if raw_rate>0 else ''}{int(CAP*100)}%"
            )
            raw_rate = CAP if raw_rate > 0 else -CAP

        growth_rates.append(raw_rate)

    if not growth_rates:
        return 0.0, ["No valid growth rates — defaulting to 0%"]

    avg      = sum(growth_rates) / len(growth_rates)
    n_capped = sum(1 for w in warnings if "capped" in w)
    if n_capped > len(growth_rates) / 2 or sign_changes >= 2:
        warnings.append(
            "⚠️ Growth rate estimate is unreliable due to volatile FCF history. "
            "Consider overriding manually."
        )

    return round(avg, 4), warnings


# =============================================================================
# FORECAST / DISCOUNT / TERMINAL VALUE
# =============================================================================

def forecast_fcf(last_fcf, growth_rate, forecast_years=None):
    n = forecast_years if forecast_years and forecast_years > 0 else FORECAST_YEARS
    forecast = []
    current  = last_fcf
    for _ in range(n):
        current *= (1 + growth_rate)
        forecast.append(round(current, 2))
    return forecast


def discount_cash_flows(cash_flows, rate):
    return [round(cf / ((1 + rate) ** i), 2) for i, cf in enumerate(cash_flows, 1)]


def calculate_terminal_value(last_forecast_fcf, wacc=None, terminal_growth=None):
    w = wacc            if wacc            is not None else WACC
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


# =============================================================================
# MASTER PIPELINE
# =============================================================================

def run_dcf_from_pdf_text(text_blocks, net_debt=0, shares_outstanding=None,
                           current_price=None, wacc=None, terminal_growth=None,
                           forecast_years=None):
    w  = wacc            if wacc            is not None else WACC
    g  = terminal_growth if terminal_growth is not None else TERMINAL_GROWTH
    fy = int(forecast_years) if forecast_years and int(forecast_years) > 0 else FORECAST_YEARS

    historical_fcf, cfo, capex = calculate_fcf_from_text(text_blocks)

    avg_growth, growth_warnings = calculate_avg_fcf_growth(historical_fcf)
    forecast = forecast_fcf(historical_fcf[-1], avg_growth, forecast_years=fy)

    if historical_fcf[-1] < 0:
        growth_warnings.append(
            "⚠️ Most recent FCF is negative. Forecast may be unreliable — "
            "consider using a normalised FCF base."
        )

    discounted_fcf = discount_cash_flows(forecast, w)
    terminal_value, tv_warning = calculate_terminal_value(forecast[-1], wacc=w, terminal_growth=g)
    if tv_warning:
        growth_warnings.append(tv_warning)

    discounted_terminal = discount_terminal_value(terminal_value, fy, w)
    enterprise_value    = calculate_enterprise_value(discounted_fcf, discounted_terminal)

    net_debt_used = float(net_debt) if net_debt is not None else 0.0
    equity_value  = round(enterprise_value - net_debt_used, 2)

    intrinsic_per_share  = None
    margin_of_safety_pct = None

    if shares_outstanding and float(shares_outstanding) > 0:
        shares = float(shares_outstanding)
        intrinsic_per_share = round(equity_value / shares, 2)

        if current_price and float(current_price) > 0:
            cmp = float(current_price)
            margin_of_safety_pct = round(
                (intrinsic_per_share - cmp) / cmp * 100, 2
            )

    return {
        "Historical FCF (₹ Cr)":            historical_fcf,
        "Historical CFO (₹ Cr)":             cfo,
        "Historical CAPEX (₹ Cr)":           capex,
        "Average FCF Growth Rate (%)":       round(avg_growth * 100, 2),
        "Forecast FCF (₹ Cr)":              forecast,
        "Discounted FCF (₹ Cr)":            discounted_fcf,
        "Terminal Value (₹ Cr)":            terminal_value,
        "Discounted Terminal Value (₹ Cr)":  discounted_terminal,
        "Enterprise Value (₹ Cr)":          enterprise_value,
        "Net Debt (₹ Cr)":                  net_debt_used,
        "Equity Value (₹ Cr)":              equity_value,
        "Intrinsic Value per Share (₹)":    intrinsic_per_share,
        "Current Market Price (₹)":         float(current_price) if current_price else None,
        "Margin of Safety (%)":             margin_of_safety_pct,
        "Confidence Score (%)":             extraction_confidence(cfo, capex),
        "Growth Warnings":                  growth_warnings,
        "Assumptions": {
            "WACC (%)":             round(w * 100, 2),
            "Terminal Growth (%)":   round(g * 100, 2),
            "Forecast Years":        fy,
            "Net Debt (₹ Cr)":      net_debt_used,
            "Shares (Cr)":          float(shares_outstanding) if shares_outstanding else None,
        }
    }


# =============================================================================
# PDF → TEXT → DCF WRAPPER
# =============================================================================

def run_dcf_from_pdfs(pdf_paths, net_debt=0, shares_outstanding=None,
                       current_price=None, wacc=None, terminal_growth=None):
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
