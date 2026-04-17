# msme_analyzer.py
"""
MSME Analyzer — Core Business Logic
=====================================
All MSME analysis logic in one file.
Mirrors the pattern of Financial_Modelling.py, dcf_valuation.py, etc.

Public functions (called by msme_routes.py):
    run_gst_analysis(file_paths: list)          → dict
    run_bank_analysis(file_path: str)           → dict
    run_combined_analysis(gst: dict, bank: dict)→ dict
"""

import os
import re
import json
import base64
import logging
import io
from datetime import datetime

import pandas as pd
import requests

logger = logging.getLogger(__name__)


# =============================================================================
# SHARED HELPERS
# =============================================================================

def _fmt(val) -> float:
    """Safe float conversion — returns 0.0 on null/error."""
    if val is None:
        return 0.0
    if isinstance(val, float) and (val != val):   # NaN check
        return 0.0
    try:
        return float(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return 0.0


def _clean_amount(val) -> float:
    """
    Convert a raw cell value to a positive float.
    Handles: commas, Dr/Cr suffixes, parentheses, currency symbols, whitespace.
    """
    if val is None:
        return 0.0
    s = str(val).strip()
    if s.lower() in ("", "-", "nil", "n/a", "nan", "none"):
        return 0.0
    s = re.sub(r"[^\d.\-]", "", s.replace(",", ""))
    try:
        return abs(float(s))
    except ValueError:
        return 0.0


_DATE_FMTS = [
    "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y",
    "%d-%b-%Y", "%d-%b-%y", "%d %b %Y", "%d %b %y",
    "%Y-%m-%d", "%m/%d/%Y",
]


def _parse_date(s):
    """Try all common Indian bank date formats. Returns datetime or None."""
    s = str(s).strip()
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


# =============================================================================
# TRANSACTION CATEGORIZER
# =============================================================================

_CATEGORY_RULES = {
    "GST Payment":       ["gst", "igst", "cgst", "sgst", "gst payment"],
    "EMI / Loan":        ["emi", "loan repay", "equated", "loan ded", "hdfc loan", "sbi loan"],
    "Salary":            ["salary", "sal/", "payroll", "neft-sal", "sal credit", "wage"],
    "Rent":              ["rent", "rental", "lease"],
    "Vendor Payment":    ["neft", "rtgs", "imps", "vendor", "supplier"],
    "Customer Receipt":  ["received", "receipt", "recd", "sale proceeds"],
    "Bank Charges":      ["charges", "fee", "commission", "penalty", "sms chrg", "annual fee"],
    "Cash":              ["cash deposit", "cash withdrawal", "atm"],
    "UPI":               ["upi/", "upi-", "phonepe", "gpay", "paytm", "bhim"],
    "Cheque":            ["clg/", "clearing", "chq", "cheque"],
    "Interest Income":   ["interest cr", "int pd", "savings interest", "int cr"],
    "Insurance":         ["insurance", "lic", "premium"],
    "Tax":               ["tds", "advance tax", "income tax"],
}


def _categorize(description: str) -> str:
    d = description.lower()
    for cat, keywords in _CATEGORY_RULES.items():
        if any(k in d for k in keywords):
            return cat
    return "Other"


# =============================================================================
# SECTION 1 — GST ANALYSIS
# =============================================================================

def _parse_gstr3b_file(file_path: str) -> dict:
    """
    Parse a single GSTR-3B JSON file downloaded from gst.gov.in.
    Returns per-month financial figures dict.
    Raises on malformed input.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Period string "032024" → "Mar 2024"
    ret_period = data.get("ret_period", "012024")
    try:
        month  = int(ret_period[:2])
        year   = int(ret_period[2:])
        period = datetime(year, month, 1).strftime("%b %Y")
    except Exception:
        period = ret_period

    gstin = data.get("gstin", "N/A")

    # ── Outward supplies (your revenue) ──────────────────────────────────
    sup             = data.get("sup_details", {}).get("osup_det", {})
    taxable_revenue = _fmt(sup.get("txval", 0))
    cgst            = _fmt(sup.get("camt",  0))
    sgst            = _fmt(sup.get("samt",  0))
    igst            = _fmt(sup.get("iamt",  0))
    tax_collected   = cgst + sgst + igst

    # ── Input Tax Credit (tax paid on purchases) ──────────────────────────
    itc_list  = data.get("itc_elg", {}).get("itc_avl", [])
    total_itc = sum(
        _fmt(i.get("iamt", 0)) + _fmt(i.get("camt", 0)) + _fmt(i.get("samt", 0))
        for i in itc_list
    )

    net_tax  = max(tax_collected - total_itc, 0.0)

    # ── Late fee / interest (non-zero = filed late) ───────────────────────
    intr     = data.get("intr_ltfee", {}).get("intr_details", {})
    late_fee = _fmt(intr.get("iamt", 0)) + _fmt(intr.get("camt", 0)) + _fmt(intr.get("samt", 0))

    return {
        "period":             period,
        "gstin":              gstin,
        "taxable_revenue":    round(taxable_revenue, 2),
        "tax_collected":      round(tax_collected,   2),
        "itc_claimed":        round(total_itc,       2),
        "net_tax_paid":       round(net_tax,         2),
        "late_fee":           round(late_fee,        2),
        "filed_on_time":      late_fee == 0,
        "effective_gst_rate": round(
            tax_collected / taxable_revenue * 100 if taxable_revenue > 0 else 0, 2
        ),
    }


def run_gst_analysis(file_paths: list) -> dict:
    """
    PUBLIC — Parse multiple GSTR-3B JSON files and return aggregate summary.

    Parameters
    ----------
    file_paths : list[str]  paths to uploaded GSTR-3B JSON files

    Returns
    -------
    dict — full GST summary, JSON-serialisable
    """
    records = []
    errors  = []

    for path in file_paths:
        try:
            records.append(_parse_gstr3b_file(path))
        except Exception as e:
            errors.append(f"{os.path.basename(path)}: {e}")

    if not records:
        return {"error": "No valid GSTR-3B files could be parsed.", "parse_errors": errors}

    # Sort chronologically
    records.sort(key=lambda r: datetime.strptime(r["period"], "%b %Y"))

    n           = len(records)
    total_rev   = sum(r["taxable_revenue"] for r in records)
    total_tax   = sum(r["tax_collected"]   for r in records)
    total_itc   = sum(r["itc_claimed"]     for r in records)
    total_net   = sum(r["net_tax_paid"]    for r in records)
    months_late = sum(1 for r in records if not r["filed_on_time"])

    # Revenue growth: last month vs first month
    growth = 0.0
    if n >= 2 and records[0]["taxable_revenue"] > 0:
        growth = round(
            (records[-1]["taxable_revenue"] - records[0]["taxable_revenue"])
            / records[0]["taxable_revenue"] * 100, 2
        )

    peak       = max(records, key=lambda r: r["taxable_revenue"])
    low        = min(records, key=lambda r: r["taxable_revenue"])
    compliance = round((1 - months_late / n) * 100, 1)

    if compliance >= 90:   gst_rating = "Excellent"
    elif compliance >= 70: gst_rating = "Average"
    else:                  gst_rating = "Poor"

    return {
        "gstin":                records[0]["gstin"],
        "months_analyzed":      n,
        "total_revenue":        round(total_rev, 2),
        "avg_monthly_revenue":  round(total_rev / n, 2),
        "total_tax_collected":  round(total_tax, 2),
        "total_itc_claimed":    round(total_itc, 2),
        "total_net_tax_paid":   round(total_net, 2),
        "itc_utilization_pct":  round(total_itc / total_tax * 100 if total_tax > 0 else 0, 2),
        "revenue_growth_pct":   growth,
        "peak_month":           peak["period"],
        "peak_revenue":         peak["taxable_revenue"],
        "low_month":            low["period"],
        "low_revenue":          low["taxable_revenue"],
        "compliance_score":     compliance,
        "gst_rating":           gst_rating,
        "months_filed_late":    months_late,
        "monthly_records":      records,
        "parse_errors":         errors,
    }


# =============================================================================
# SECTION 2 — GOOGLE VISION PDF EXTRACTION
# =============================================================================

_VISION_URL = "https://vision.googleapis.com/v1/images:annotate?key={}"


def _pil_to_base64(pil_image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _call_vision_api(b64_image: str, api_key: str) -> str:
    """
    Send one base64 JPEG to Google Vision DOCUMENT_TEXT_DETECTION.
    Returns extracted text. Returns '' on any failure — never raises.
    """
    payload = {
        "requests": [{
            "image":    {"content": b64_image},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
        }]
    }
    try:
        resp = requests.post(_VISION_URL.format(api_key), json=payload, timeout=30)
        if resp.status_code != 200:
            logger.warning(f"Vision API {resp.status_code}: {resp.text[:200]}")
            return ""
        data = resp.json()
        return data["responses"][0]["fullTextAnnotation"]["text"]
    except (KeyError, IndexError):
        return ""   # blank page
    except Exception as e:
        logger.warning(f"Vision API error: {e}")
        return ""


def _is_image_pdf(pdf_path: str, sample_pages: int = 3) -> bool:
    """
    Return True if PDF is image-based (no selectable text).
    Checks first N pages — if total chars < 50 → scanned PDF.
    """
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            pages = min(sample_pages, len(pdf.pages))
            chars = sum(len((pdf.pages[i].extract_text() or "").strip()) for i in range(pages))
        return chars < 50
    except Exception as e:
        logger.warning(f"PDF type check failed: {e}")
        return False


def _extract_pdf_text(pdf_path: str, max_pages: int = 25):
    """
    Extract text from any bank statement PDF.

    Digital PDF  → pdfplumber  (fast, free)
    Scanned PDF  → Google Vision API

    Returns
    -------
    (lines: list[str], method: str)
    method = 'text' | 'vision' | 'error'
    """
    # ── Path 1: Digital PDF via pdfplumber ────────────────────────────────
    if not _is_image_pdf(pdf_path):
        try:
            import pdfplumber
            lines = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    for ln in text.splitlines():
                        ln = ln.strip()
                        if ln:
                            lines.append(ln)
            if len(lines) > 30:
                logger.info(f"Digital PDF: {len(lines)} lines")
                return lines, "text"
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}")

    # ── Path 2: Scanned PDF → Google Vision ──────────────────────────────
    api_key = os.environ.get("GOOGLE_VISION_API_KEY", "")
    if not api_key:
        logger.error("GOOGLE_VISION_API_KEY not set in environment")
        return [], "error"

    logger.info(f"Scanned PDF — using Google Vision: {pdf_path}")
    try:
        import pdf2image
        images = pdf2image.convert_from_path(
            pdf_path,
            dpi=200,
            first_page=1,
            last_page=max_pages,
            fmt="jpeg",
            thread_count=1,    # Render free tier: single thread
            grayscale=False,   # Vision is more accurate with colour
        )
    except Exception as e:
        logger.error(f"pdf2image failed: {e}")
        return [], "error"

    lines = []
    for i, img in enumerate(images):
        try:
            text = _call_vision_api(_pil_to_base64(img), api_key)
            for ln in text.splitlines():
                ln = ln.strip()
                if ln:
                    lines.append(ln)
            logger.info(f"Vision page {i+1}: {len(text)} chars")
        except Exception as e:
            logger.warning(f"Vision page {i+1} error: {e}")
        finally:
            del img   # free memory on Render 512 MB

    del images
    logger.info(f"Vision total: {len(lines)} lines")
    return (lines, "vision") if lines else ([], "error")


# =============================================================================
# SECTION 3 — BANK STATEMENT PARSING
# =============================================================================

def _find_col(df: pd.DataFrame, candidates: list):
    """Return first column that contains any candidate substring (case-insensitive)."""
    for c in candidates:
        for col in df.columns:
            if c.lower() in str(col).lower():
                return col
    return None


def _find_header_row(df_raw: pd.DataFrame) -> int:
    """Detect which row is the actual column header."""
    keywords = [
        "date", "narration", "debit", "credit", "balance",
        "withdrawal", "deposit", "particulars", "description",
        "amount", "tran", "value date",
    ]
    for i, row in df_raw.iterrows():
        text  = " ".join(str(v) for v in row.values).lower()
        score = sum(1 for k in keywords if k in text)
        if score >= 3:
            return int(i)
    return 0


def _rows_to_df(rows: list):
    """Convert a list of transaction dicts to a sorted DataFrame."""
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _parse_excel_statement(file_path: str):
    """
    Parse bank statement from Excel (.xlsx/.xls) or CSV.
    Returns (DataFrame | None, error_str | None)
    """
    try:
        is_csv = file_path.lower().endswith(".csv")

        if is_csv:
            df_raw = None
            for enc in ("utf-8", "latin-1", "cp1252"):
                try:
                    df_raw = pd.read_csv(file_path, encoding=enc, header=None, dtype=str)
                    break
                except Exception:
                    continue
            if df_raw is None:
                return None, "Could not read CSV — try saving as .xlsx instead."
        else:
            df_raw = pd.read_excel(file_path, header=None, dtype=str)

        if df_raw is None or df_raw.empty:
            return None, "File is empty."

        hdr = _find_header_row(df_raw)
        if is_csv:
            df = pd.read_csv(file_path, skiprows=hdr, encoding="latin-1", dtype=str)
        else:
            df = pd.read_excel(file_path, skiprows=hdr, dtype=str)

        df.columns = [str(c).strip() for c in df.columns]

        date_col   = _find_col(df, ["Date", "Txn Date", "Transaction Date",
                                     "Value Date", "Tran Date", "Posting Date"])
        desc_col   = _find_col(df, ["Narration", "Description", "Particulars",
                                     "Transaction Remarks", "PARTICULARS", "Remarks"])
        debit_col  = _find_col(df, ["Debit", "Withdrawal", "Withdrawal Amt",
                                     "DR", "Dr", "Debit Amount"])
        credit_col = _find_col(df, ["Credit", "Deposit", "Deposit Amt",
                                     "CR", "Cr", "Credit Amount"])
        bal_col    = _find_col(df, ["Balance", "Closing Balance", "BAL", "Running Balance"])

        if not date_col:
            return None, (
                f"Date column not found. Columns detected: {list(df.columns)}\n"
                "Download the statement as Excel directly from your bank's net banking portal."
            )
        if not desc_col:
            return None, f"Description column not found. Columns: {list(df.columns)}"
        if not debit_col and not credit_col:
            return None, "No Debit/Credit columns found."

        rows = []
        for _, row in df.iterrows():
            date = _parse_date(row.get(date_col, ""))
            if date is None:
                continue
            desc   = str(row.get(desc_col,   "")).strip()
            debit  = _clean_amount(row.get(debit_col))  if debit_col  else 0.0
            credit = _clean_amount(row.get(credit_col)) if credit_col else 0.0
            bal    = _clean_amount(row.get(bal_col))    if bal_col    else 0.0
            if debit == 0.0 and credit == 0.0:
                continue
            rows.append({
                "date":        date,
                "month":       date.strftime("%b %Y"),
                "description": desc or "Transaction",
                "debit":       debit,
                "credit":      credit,
                "balance":     bal,
                "category":    _categorize(desc),
            })

        if not rows:
            return None, "No transactions found. Check the file is the correct format."

        return _rows_to_df(rows), None

    except Exception as e:
        logger.exception("_parse_excel_statement error")
        return None, f"Parse error: {e}"


_AMOUNT_RE    = re.compile(r"[\d,]+\.\d{2}")
_LINE_DATE_RE = re.compile(
    r"^(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}"
    r"|\d{1,2}[\s\-][A-Za-z]{3}[\s\-]\d{2,4})"
)


def _parse_pdf_text_lines(lines: list):
    """
    Convert raw text lines (pdfplumber or OCR) into a normalised DataFrame.
    Any line starting with a valid date and containing ≥1 amount = transaction.
    Returns (DataFrame | None, error_str | None)
    """
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = _LINE_DATE_RE.match(line)
        if not m:
            continue
        date = _parse_date(m.group(1))
        if date is None:
            continue
        amounts = _AMOUNT_RE.findall(line)
        if not amounts:
            continue

        # Build description: remove date prefix and all amounts
        desc = _AMOUNT_RE.sub("", line[m.end():])
        desc = re.sub(r"\s+", " ", desc).strip()
        desc = re.sub(r"[^\w\s/\-@()]", "", desc).strip()

        ll   = line.lower()
        val1 = _clean_amount(amounts[0])
        val2 = _clean_amount(amounts[1]) if len(amounts) >= 2 else None

        is_cr = any(k in ll for k in ["cr", "credit", "deposit", "received",
                                        "salary", "receipt", "refund"])
        is_dr = any(k in ll for k in ["dr", "debit", "withdrawal", "paid",
                                        "payment", "purchase"])

        if is_cr and not is_dr:
            debit, credit = 0.0, val1
        elif is_dr and not is_cr:
            debit, credit = val1, 0.0
        elif val2 is not None:
            debit, credit = val1, val2
        else:
            debit, credit = val1, 0.0

        if debit == 0.0 and credit == 0.0:
            continue

        rows.append({
            "date":        date,
            "month":       date.strftime("%b %Y"),
            "description": desc or "Transaction",
            "debit":       debit,
            "credit":      credit,
            "balance":     0.0,
            "category":    _categorize(desc),
        })

    if not rows:
        return None, (
            "Text extracted but no transactions detected.\n"
            "For best results, download the statement as Excel from net banking."
        )

    return _rows_to_df(rows), None


# =============================================================================
# SECTION 4 — BANK ANALYTICS ENGINE
# =============================================================================

def _compute_bank_analytics(df: pd.DataFrame) -> dict:
    """
    Full analytics pass over a normalised bank statement DataFrame.
    Returns a complete JSON-serialisable dict.
    """
    monthly = (
        df.groupby("month")
        .agg(
            total_inflow  = ("credit",  "sum"),
            total_outflow = ("debit",   "sum"),
            txn_count     = ("date",    "count"),
            closing_bal   = ("balance", "last"),
        )
        .reset_index()
    )
    monthly["net_cash"] = monthly["total_inflow"] - monthly["total_outflow"]

    tot_months          = int(df["month"].nunique())
    total_inflow        = float(df["credit"].sum())
    total_outflow       = float(df["debit"].sum())
    avg_monthly_inflow  = float(monthly["total_inflow"].mean())  if tot_months else 0.0
    avg_monthly_outflow = float(monthly["total_outflow"].mean()) if tot_months else 0.0
    avg_balance         = float(df["balance"].mean()) if df["balance"].sum() > 0 else 0.0
    min_balance         = float(df["balance"].min())  if df["balance"].sum() > 0 else 0.0

    # ── Stress indicators ──────────────────────────────────────────────────
    bounce_kw  = ["bounce", "returned", "dishonour", "insufficient", "unpaid"]
    bounced    = df[df["description"].str.lower().str.contains("|".join(bounce_kw), na=False)]

    od_kw      = ["od interest", "overdraft", "od charges"]
    od_rows    = df[df["description"].str.lower().str.contains("|".join(od_kw), na=False)]

    # ── GST payment consistency ────────────────────────────────────────────
    gst_df     = df[df["category"] == "GST Payment"]
    gst_months = int(gst_df["month"].nunique())
    gst_cons   = round(gst_months / tot_months * 100, 1) if tot_months else 0.0

    # ── EMI burden ─────────────────────────────────────────────────────────
    emi_df     = df[df["category"] == "EMI / Loan"]
    raw_emi    = emi_df.groupby("month")["debit"].sum().mean()
    avg_emi    = float(raw_emi) if not pd.isna(raw_emi) else 0.0
    emi_ratio  = round(avg_emi / avg_monthly_inflow * 100, 1) if avg_monthly_inflow else 0.0

    # ── Cash flow stability ────────────────────────────────────────────────
    pos_months   = int((monthly["net_cash"] > 0).sum())
    cf_stability = round(pos_months / len(monthly) * 100, 1) if len(monthly) else 0.0

    # ── Category breakdown ─────────────────────────────────────────────────
    cat_debit  = {k: round(float(v), 2) for k, v in df.groupby("category")["debit"].sum().items()}
    cat_credit = {k: round(float(v), 2) for k, v in df.groupby("category")["credit"].sum().items()}

    # ── Business Health Score (0–100) ──────────────────────────────────────
    score = 100
    score -= min(len(bounced) * 10, 30)          # bounces         (max −30)
    score -= min(len(od_rows) * 5,  20)          # overdraft use   (max −20)
    score -= round((100 - gst_cons) * 0.2)       # GST irregularity(max −20)
    if emi_ratio > 40:   score -= 20             # heavy EMI
    elif emi_ratio > 25: score -= 10
    neg_months = len(monthly) - pos_months
    score -= min(neg_months * 5, 20)             # negative cash months (max −20)
    score  = max(score, 0)

    if score >= 75:   bank_rating = "Healthy"
    elif score >= 50: bank_rating = "Moderate"
    else:             bank_rating = "At Risk"

    monthly_list = [
        {
            "month":         row["month"],
            "total_inflow":  round(float(row["total_inflow"]),  2),
            "total_outflow": round(float(row["total_outflow"]), 2),
            "net_cash":      round(float(row["net_cash"]),      2),
            "txn_count":     int(row["txn_count"]),
        }
        for _, row in monthly.iterrows()
    ]

    return {
        "total_months":             tot_months,
        "total_inflow":             round(total_inflow,        2),
        "total_outflow":            round(total_outflow,       2),
        "avg_monthly_inflow":       round(avg_monthly_inflow,  2),
        "avg_monthly_outflow":      round(avg_monthly_outflow, 2),
        "avg_balance":              round(avg_balance,         2),
        "min_balance":              round(min_balance,         2),
        "bounce_count":             len(bounced),
        "od_usage_count":           len(od_rows),
        "avg_monthly_emi":          round(avg_emi,  2),
        "emi_to_inflow_ratio":      emi_ratio,
        "gst_payment_consistency":  gst_cons,
        "cash_flow_stability":      cf_stability,
        "positive_cash_months":     pos_months,
        "category_debits":          cat_debit,
        "category_credits":         cat_credit,
        "monthly_summary":          monthly_list,
        "business_health_score":    score,
        "bank_rating":              bank_rating,
        "loan_eligibility":         round(avg_monthly_inflow * 4),
    }


# =============================================================================
# SECTION 5 — PUBLIC BANK ANALYSIS ENTRY POINT
# =============================================================================

def run_bank_analysis(file_path: str) -> dict:
    """
    PUBLIC — Accept bank statement in any format and return full analytics.

    Formats supported
    -----------------
    .xlsx / .xls  → direct column parse via pandas
    .csv          → direct column parse via pandas
    .pdf digital  → pdfplumber text extraction
    .pdf scanned  → Google Vision API OCR

    Parameters
    ----------
    file_path : str  path to uploaded file (already saved to disk)

    Returns
    -------
    dict — full bank analytics, JSON-serialisable
           may include 'warning' key if OCR was used
           will include 'error' key on failure
    """
    ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""

    if ext in ("xlsx", "xls", "csv"):
        df, err = _parse_excel_statement(file_path)
        if df is None:
            return {"error": err}
        method  = "text"
        warning = None

    elif ext == "pdf":
        lines, method = _extract_pdf_text(file_path)

        if not lines:
            return {
                "error": (
                    "Could not extract text from this PDF.\n"
                    "• If scanned: ensure GOOGLE_VISION_API_KEY is set on Render.\n"
                    "• Best option: download statement as Excel from net banking."
                )
            }

        warning = (
            "Scanned PDF processed via Google Vision OCR. "
            "Please verify the transaction amounts look correct."
            if method == "vision" else None
        )

        df, err = _parse_pdf_text_lines(lines)
        if df is None:
            return {"error": err}

    else:
        return {"error": f"Unsupported file type: .{ext}. Upload Excel (.xlsx), CSV, or PDF."}

    result = _compute_bank_analytics(df)
    result["extraction_method"] = method

    if warning:
        result["warning"] = warning

    return result


# =============================================================================
# SECTION 6 — COMBINED ANALYSIS
# =============================================================================

def _build_insights(gst: dict, bank: dict, coll_eff: float, dscr) -> list:
    """
    Build a list of plain-English insight dicts from analysis results.
    Each item: { type: 'good'|'warn'|'bad', icon: str, text: str }
    """
    ins = []

    # Collection efficiency
    if coll_eff >= 90:
        ins.append({"type": "good", "icon": "✅",
                    "text": f"Collection efficiency is excellent at {coll_eff}% — nearly all GST-declared revenue is being received in the bank."})
    elif coll_eff >= 75:
        ins.append({"type": "warn", "icon": "⚠️",
                    "text": f"Collection efficiency is {coll_eff}% — some invoiced revenue has not been collected. Follow up on outstanding receivables."})
    else:
        ins.append({"type": "bad", "icon": "❌",
                    "text": f"Collection efficiency is only {coll_eff}% — large gap between declared revenue and bank receipts. High receivables risk."})

    # DSCR
    if dscr is None:
        ins.append({"type": "good", "icon": "✅",
                    "text": "No EMI or loan repayments detected — business appears debt-free or loans are serviced from a separate account."})
    elif dscr >= 1.5:
        ins.append({"type": "good", "icon": "✅",
                    "text": f"DSCR of {dscr}x — your business comfortably generates enough cash to service all debt obligations."})
    elif dscr >= 1.0:
        ins.append({"type": "warn", "icon": "⚠️",
                    "text": f"DSCR of {dscr}x is tight — debt repayments are consuming a large share of monthly income. Avoid taking additional loans."})
    else:
        ins.append({"type": "bad", "icon": "❌",
                    "text": f"DSCR of {dscr}x is below 1.0 — monthly inflows may not fully cover current loan repayments. This is a serious concern for lenders."})

    # GST compliance
    comp = gst.get("compliance_score", 0)
    if comp >= 90:
        ins.append({"type": "good", "icon": "✅",
                    "text": f"GST compliance score is {comp}% — consistent on-time filing is a strong positive signal for lenders."})
    elif comp >= 70:
        ins.append({"type": "warn", "icon": "⚠️",
                    "text": f"GST compliance is {comp}% — some late filings recorded. This can reduce loan approval chances."})
    else:
        ins.append({"type": "bad", "icon": "❌",
                    "text": f"GST compliance is poor at {comp}% — multiple late filings detected. Most lenders will flag this."})

    # Bounces
    bounces = bank.get("bounce_count", 0)
    if bounces == 0:
        ins.append({"type": "good", "icon": "✅",
                    "text": "No cheque bounces or payment returns detected — clean payment record."})
    elif bounces <= 2:
        ins.append({"type": "warn", "icon": "⚠️",
                    "text": f"{bounces} cheque bounce(s) detected. Ensure adequate balance before issuing cheques."})
    else:
        ins.append({"type": "bad", "icon": "❌",
                    "text": f"{bounces} cheque bounces detected — this is a significant red flag. Banks may reject loan applications with multiple bounces."})

    # Revenue growth
    growth = gst.get("revenue_growth_pct", 0)
    if growth >= 10:
        ins.append({"type": "good", "icon": "📈",
                    "text": f"Revenue grew {growth}% over the period — strong growth trajectory is a positive signal for lenders."})
    elif growth > 0:
        ins.append({"type": "good", "icon": "📈",
                    "text": f"Revenue showing positive growth of {growth}%."})
    elif growth < -10:
        ins.append({"type": "bad", "icon": "📉",
                    "text": f"Revenue declined {abs(growth)}% over the period — lenders will scrutinize this carefully."})

    # Cash flow stability
    cf         = bank.get("cash_flow_stability", 0)
    pos_months = bank.get("positive_cash_months", 0)
    tot_months = bank.get("total_months", 0)
    if cf < 60:
        ins.append({"type": "warn", "icon": "⚠️",
                    "text": f"Cash flow was positive in only {pos_months} of {tot_months} months — irregular cash flows may concern lenders."})

    return ins


def run_combined_analysis(gst: dict, bank: dict) -> dict:
    """
    PUBLIC — Cross-analyse GST and bank data to produce combined insights.

    Parameters
    ----------
    gst  : dict  output of run_gst_analysis()
    bank : dict  output of run_bank_analysis()

    Returns
    -------
    dict — combined metrics + insights list, JSON-serialisable
    """
    gst_rev  = float(gst.get("total_revenue",      0))
    gst_tax  = float(gst.get("total_tax_collected", 0))
    b_inflow = float(bank.get("total_inflow",       0))
    avg_in   = float(bank.get("avg_monthly_inflow", 0))
    avg_emi  = float(bank.get("avg_monthly_emi",    0))

    # Collection efficiency: actual bank receipts vs GST-declared (incl. tax)
    declared = gst_rev + gst_tax
    coll_eff = round(b_inflow / declared * 100, 1) if declared > 0 else 0.0

    # DSCR
    dscr = round(avg_in / avg_emi, 2) if avg_emi > 0 else None

    if dscr is None:           dscr_label = "No EMI detected"
    elif dscr >= 2.0:          dscr_label = "Very comfortable"
    elif dscr >= 1.5:          dscr_label = "Comfortable"
    elif dscr >= 1.0:          dscr_label = "Tight — monitor closely"
    else:                      dscr_label = "Stressed — debt may be unsustainable"

    # Weighted overall score: bank 60% + GST compliance 40%
    bank_score = float(bank.get("business_health_score", 50))
    gst_score  = float(gst.get("compliance_score",       50))
    overall    = round(bank_score * 0.6 + gst_score * 0.4)

    if overall >= 75:   overall_rating = "Healthy"
    elif overall >= 50: overall_rating = "Moderate"
    else:               overall_rating = "Needs Attention"

    insights = _build_insights(gst, bank, coll_eff, dscr)

    return {
        "collection_efficiency": coll_eff,
        "dscr":                  dscr,
        "dscr_label":            dscr_label,
        "loan_eligibility":      round(avg_in * 4),
        "overall_score":         overall,
        "overall_rating":        overall_rating,
        "bank_score":            bank_score,
        "gst_score":             gst_score,
        "insights":              insights,
    }
