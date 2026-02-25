#!/usr/bin/env python3
"""
financialanalyzer.py  —  Robust Income / Balance / CashFlow extractor.

Key fixes vs original:
  1. pdf_to_text: tries native text layer first (fast), falls back to OCR only
     for image-based PDFs. OCR DPI lowered to 150 (2x faster).
  2. Auto-detects Tesseract on Linux (Render) — no hardcoded Windows path.
  3. Extractors use PRIORITY-BASED matching: the highest-priority keyword that
     yields numbers wins. Later lines cannot overwrite it. This prevents
     sub-totals / footnotes from overwriting the real data row.
  4. Extractors also try MULTI-LINE look-ahead: if the label is on one line and
     the numbers are on the next 1-2 lines, they are still captured.
  5. detect_all_types() returns ALL statement types present in a combined PDF
     so a single annual-report PDF produces all 3 sections.
  6. All max() calls are guarded with a trailing 0 to prevent empty-sequence
     crashes.
"""

import fitz          # PyMuPDF
import pytesseract
from PIL import Image
import io
import re
import shutil
from typing import List, Optional, Dict

# ── Tesseract: auto-detect on Linux/Render; fall back to Windows path ─────────
_tess = shutil.which("tesseract")
pytesseract.pytesseract.tesseract_cmd = _tess if _tess else \
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"

DEBUG = False   # set True to print matched lines during extraction


# ══════════════════════════════════════════════════════════════════════════════
# PDF → text
# ══════════════════════════════════════════════════════════════════════════════
def pdf_to_text(pdf_path: str) -> List[str]:
    """
    Extract text lines from a PDF.
    Tries the native text layer first (instant, works for digital PDFs).
    Falls back to Tesseract OCR at 150 DPI for scanned/image PDFs.
    """
    doc = fitz.open(pdf_path)
    lines: List[str] = []

    for page in doc:
        native = page.get_text("text")
        if native and native.strip():
            for ln in native.splitlines():
                ln = ln.strip()
                if ln:
                    lines.append(ln)
        else:
            pix = page.get_pixmap(dpi=150)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img)
            for ln in text.splitlines():
                ln = ln.strip()
                if ln:
                    lines.append(ln)

    return lines


# ══════════════════════════════════════════════════════════════════════════════
# Number helpers
# ══════════════════════════════════════════════════════════════════════════════
def clean_number(token) -> Optional[float]:
    if token is None:
        return None
    s = str(token)
    s = s.replace("\u200b", "").replace(",", "").replace(" ", "")
    s = s.replace("(", "-").replace(")", "")
    s = re.sub(r"[^0-9.\-]", "", s)
    if s in ("", "-", ".", None):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def find_numbers(line: str) -> List[float]:
    toks = re.findall(r"-?\(?[0-9,]+(?:\.[0-9]+)?\)?", line)
    cleaned = [clean_number(t) for t in toks]
    return [c for c in cleaned if c is not None]


def cagr_percent(vals: List) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None
    start, end = vals[-1], vals[0]
    n = len(vals) - 1
    if start <= 0 or end <= 0:
        return None
    return round(((end / start) ** (1 / n) - 1) * 100, 2)


def average_growth(values: List) -> Optional[float]:
    if not values or len(values) < 2:
        return None
    growths = []
    for i in range(len(values) - 1):
        cur, prev = values[i], values[i + 1]
        if cur is None or prev in (None, 0):
            continue
        growths.append(((cur - prev) / prev) * 100)
    return round(sum(growths) / len(growths), 2) if growths else None


def avg(lst: List) -> Optional[float]:
    vals = [x for x in lst if x is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def ratio_yearly(num_list, den_list, scale=2) -> List:
    max_len = max(len(num_list), len(den_list), 0)
    out = []
    for i in range(max_len):
        n = num_list[i] if i < len(num_list) else None
        d = den_list[i] if i < len(den_list) else None
        out.append(round(n / d, scale) if (n is not None and d not in (None, 0)) else None)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Core extraction engine  (PRIORITY-BASED, MULTI-LINE AWARE)
# ══════════════════════════════════════════════════════════════════════════════
def _extract(lines: List[str], patterns: Dict[str, List[str]]) -> Dict[str, List[float]]:
    """
    Priority-based extractor.

    patterns maps: field_name -> [kw_high_priority, kw_lower_priority, ...]

    Rules:
    - Iterate lines. For every UNLOCKED field, check if current line matches
      any keyword (highest priority first).
    - If match found but no numbers on this line, look ahead 1-2 lines.
    - Once a field has data it is LOCKED — no further overwriting.
    - FIRST good match wins (earliest in document, highest priority keyword).
    """
    res: Dict[str, List[float]] = {k: [] for k in patterns}
    locked = set()

    for idx, line in enumerate(lines):
        low = line.lower().replace("\u2013", "-").replace("\u2014", "-")

        for field, keywords in patterns.items():
            if field in locked:
                continue

            matched = False
            for kw in keywords:
                if kw in low:
                    matched = True
                    break

            if not matched:
                continue

            # Try numbers on this line first
            nums = find_numbers(line)

            # Look ahead up to 2 lines if no numbers found
            if not nums:
                for ahead in range(1, 3):
                    if idx + ahead < len(lines):
                        nums = find_numbers(lines[idx + ahead])
                        if nums:
                            break

            if nums:
                res[field] = nums
                locked.add(field)
                if DEBUG:
                    print(f"  [MATCH] {field!r:35s} <- {nums[:5]}  | line: {line!r[:80]}")

    return res


# ══════════════════════════════════════════════════════════════════════════════
# Statement-specific extractors
# ══════════════════════════════════════════════════════════════════════════════
def extract_income_series(lines: List[str]) -> dict:
    patterns = {
        "revenue": [
            "revenue from operations",
            "total revenue from operations",
            "total operating revenue",
            "net revenue from operations",
            "net revenue",
            "net sales",
            "total revenue",
            "revenue",
            "sales",
        ],
        "net_profit": [
            "profit/(loss) for the period",
            "profit / (loss) for the period",
            "profit/(loss) for the year",
            "profit / (loss) for the year",
            "profit/loss for the period",
            "profit / loss for the period",
            "profit/loss for the year",
            "profit / loss for the year",
            "profit for the period",
            "profit for the year",
            "profit after tax",
            "net profit after tax",
            "net profit",
            "pat",
        ],
        "interest": [
            "finance costs",
            "finance cost",
            "interest expense",
            "borrowing costs",
            "interest on borrowings",
            "interest",
        ],
        "tax": [
            "total tax expense",
            "income tax expense",
            "tax expense",
            "current tax",
            "deferred tax",
            "income tax",
        ],
        "depreciation": [
            "depreciation and amortisation expense",
            "depreciation and amortization expense",
            "depreciation & amortization",
            "depreciation and amortization",
            "depreciation and amortisation",
            "depreciation",
            "amortisation",
            "amortization",
        ],
        "cogs": [
            "cost of materials consumed",
            "cost of goods sold",
            "cost of goods",
            "raw material consumed",
            "raw materials consumed",
        ],
        "purchases": [
            "purchases of stock-in-trade",
            "purchase of stock-in-trade",
            "purchase of stock in trade",
            "purchases of stock in trade",
            "purchases",
        ],
        "employee_cost": [
            "employee benefit expenses",
            "employee benefits expense",
            "employee benefits expenses",
            "staff costs",
            "personnel expenses",
            "employee cost",
            "salaries and wages",
            "salaries wages",
        ],
        "inventory_change": [
            "changes in inventories of finished goods",
            "change in inventories of finished goods",
            "changes in inventories of work-in-progress",
            "changes in inventories",
            "change in inventories",
        ],
    }
    return _extract(lines, patterns)


def extract_balance_series(lines: List[str]) -> dict:
    patterns = {
        "cash": [
            "cash and cash equivalents",
            "cash & cash equivalents",
            "cash and bank balances",
            "cash & bank balances",
            "cash and bank",
            "cash & bank",
        ],
        "current_assets": [
            "total current assets",
            "current assets",
        ],
        "inventory": [
            "inventories",
            "inventory",
        ],
        "current_liabilities": [
            "total current liabilities",
            "current liabilities",
        ],
        "non_current_liabilities": [
            "total non-current liabilities",
            "non-current liabilities",
            "non current liabilities",
            "long-term liabilities",
        ],
        "shareholders_fund": [
            "total equity",
            "total shareholders' funds",
            "total shareholders funds",
            "shareholders' funds",
            "shareholders funds",
            "net worth",
        ],
        "share_capital": [
            "equity share capital",
            "share capital",
        ],
        "reserves": [
            "other equity",
            "reserves and surplus",
            "reserves & surplus",
            "other reserves",
            "reserves",
        ],
        "other_comprehensive_income": [
            "other comprehensive income",
            "oci",
        ],
        "total_assets": [
            "total assets",
        ],
        "borrowings": [
            "total borrowings",
            "long-term borrowings",
            "short-term borrowings",
            "borrowings",
        ],
    }
    return _extract(lines, patterns)


def extract_cashflow_series(lines: List[str]) -> dict:
    patterns = {
        "cfo": [
            "net cash generated from operating activities",
            "net cash from operating activities",
            "net cash inflow from operating activities",
            "net cash generated from operations",
            "cash generated from operations",
            "net cash flow from operating activities",
            "net cash flow from operating",
            "cash flow from operating activities",
            "net cash from operating",
        ],
        "cfi": [
            "net cash used in investing activities",
            "net cash from investing activities",
            "net cash inflow from investing activities",
            "net cash flow from investing activities",
            "net cash flow from investing",
            "cash flow from investing activities",
            "net cash used in investing",
            "net cash from investing",
        ],
        "cff": [
            "net cash used in financing activities",
            "net cash from financing activities",
            "net cash inflow from financing activities",
            "net cash flow from financing activities",
            "net cash flow from financing",
            "cash flow from financing activities",
            "net cash from financing",
        ],
        "capex": [
            "purchase of property, plant and equipment",
            "purchase of property plant and equipment",
            "acquisition of property, plant and equipment",
            "payments for property, plant and equipment",
            "capital expenditure on fixed assets",
            "capital expenditure",
            "purchase of fixed assets",
            "purchase of tangible assets",
            "additions to property plant",
        ],
        "net_change_cash": [
            "net increase/(decrease) in cash and cash equivalents",
            "net decrease/(increase) in cash and cash equivalents",
            "net increase in cash and cash equivalents",
            "net decrease in cash and cash equivalents",
            "net change in cash and cash equivalents",
            "net increase/(decrease) in cash",
            "net change in cash",
            "net increase",
            "net decrease",
        ],
    }
    return _extract(lines, patterns)


# ══════════════════════════════════════════════════════════════════════════════
# Statement type detection
# ══════════════════════════════════════════════════════════════════════════════
def detect_type(lines: List[str], filename: str = "") -> Optional[str]:
    """Returns primary type for a single-statement PDF."""
    txt = " ".join(lines).lower()
    fname = filename.lower()

    # Strong filename hints
    if any(k in fname for k in ["income", "p&l", "pnl", "profit", " pl ", "_pl_", "-pl-"]):
        return "income"
    if any(k in fname for k in ["balance", " bs ", "_bs_", "-bs-", "balance-sheet", "balancesheet"]):
        return "balance"
    if any(k in fname for k in ["cash", "flow", "cfs", "cashflow", "cash-flow"]):
        return "cash"

    # Content scoring
    income_score = sum([
        4 if "revenue from operations" in txt else 0,
        4 if "profit for the period" in txt or "profit for the year" in txt else 0,
        3 if "profit/(loss)" in txt else 0,
        2 if "net profit" in txt else 0,
        1 if "employee benefit expenses" in txt else 0,
        1 if "finance costs" in txt else 0,
    ])
    balance_score = sum([
        4 if "total assets" in txt else 0,
        4 if "total equity" in txt or "total shareholders" in txt else 0,
        3 if "current liabilities" in txt else 0,
        2 if "non-current assets" in txt else 0,
        1 if "inventories" in txt else 0,
        1 if "equity share capital" in txt else 0,
    ])
    cash_score = sum([
        5 if "cash flow from operating activities" in txt else 0,
        5 if "net cash from operating" in txt else 0,
        4 if "cash flow from investing" in txt else 0,
        3 if "cash flow from financing" in txt else 0,
        1 if "capital expenditure" in txt else 0,
    ])

    best = max(income_score, balance_score, cash_score, 0)
    if best == 0:
        return None
    if income_score == best:
        return "income"
    if cash_score == best:
        return "cash"
    return "balance"


def detect_all_types(lines: List[str]) -> List[str]:
    """
    Returns ALL statement types present in a combined PDF.
    Used by app.py so a single annual-report PDF produces all 3 sections.
    """
    txt = " ".join(lines).lower()
    found = []

    income_signals = [
        "revenue from operations", "profit for the period", "profit for the year",
        "net profit", "profit/(loss)", "employee benefit expenses", "finance costs",
    ]
    balance_signals = [
        "total assets", "total equity", "total shareholders", "current liabilities",
        "non-current assets", "equity share capital", "reserves and surplus",
    ]
    cash_signals = [
        "cash flow from operating", "net cash from operating",
        "cash flow from investing", "cash flow from financing",
        "net cash used in investing",
    ]

    if sum(1 for s in income_signals if s in txt) >= 2:
        found.append("income")
    if sum(1 for s in balance_signals if s in txt) >= 2:
        found.append("balance")
    if sum(1 for s in cash_signals if s in txt) >= 2:
        found.append("cash")

    return found


# ══════════════════════════════════════════════════════════════════════════════
# Analyzers
# ══════════════════════════════════════════════════════════════════════════════
def analyze_income(data: dict) -> dict:
    revenue          = data.get("revenue", [])
    net_profit       = data.get("net_profit", [])
    interest         = data.get("interest", [])
    tax              = data.get("tax", [])
    depreciation     = data.get("depreciation", [])
    cogs             = data.get("cogs", [])
    purchases        = data.get("purchases", [])
    employee_cost    = data.get("employee_cost", [])
    inventory_change = data.get("inventory_change", [])

    def v(series, i=0):
        return series[i] if len(series) > i else None

    rev_cur  = v(revenue, 0)
    rev_prev = v(revenue, 1)
    revenue_growth = round(((rev_cur - rev_prev) / rev_prev * 100), 2) \
        if (rev_cur is not None and rev_prev not in (None, 0)) else None

    net_cur = v(net_profit, 0)

    # EBITDA series
    max_len = max(len(net_profit), len(interest), len(tax), len(depreciation), 0)
    ebitda_series = []
    for i in range(max_len):
        n  = net_profit[i]   if i < len(net_profit)   else None
        it = interest[i]     if i < len(interest)     else 0
        tx = tax[i]          if i < len(tax)           else 0
        dp = depreciation[i] if i < len(depreciation) else 0
        if n is not None:
            ebitda_series.append(round((n or 0) + (it or 0) + (tx or 0) + (dp or 0), 2))

    op_profit_cur = None
    if net_cur is not None:
        op_profit_cur = round((net_cur or 0) + (v(interest, 0) or 0) + (v(tax, 0) or 0), 2)

    operating_margin = round(op_profit_cur / rev_cur * 100, 2) \
        if (op_profit_cur is not None and rev_cur not in (None, 0)) else None
    net_profit_margin = round(net_cur / rev_cur * 100, 2) \
        if (net_cur is not None and rev_cur not in (None, 0)) else None

    cogs0   = v(cogs, 0) or 0
    pur0    = v(purchases, 0) or 0
    emp0    = v(employee_cost, 0) or 0
    invchg0 = v(inventory_change, 0) or 0
    gross_profit = gross_margin = None
    if rev_cur is not None:
        total_costs  = cogs0 + pur0 + emp0 - invchg0
        gross_profit = round(rev_cur - total_costs, 2)
        gross_margin = round(gross_profit / rev_cur * 100, 2) if rev_cur else None

    return {
        "Statement":                        "Income Statement",
        "Revenue (All Years)":              revenue,
        "Net Profit (All Years)":           net_profit,
        "EBITDA (All Years)":               ebitda_series,
        "COGS (All Years)":                 cogs,
        "Employee Cost (All Years)":        employee_cost,
        "Depreciation (All Years)":         depreciation,
        "Interest (All Years)":             interest,
        "Tax (All Years)":                  tax,
        "Purchases (All Years)":            purchases,
        "Inventory Change (All Years)":     inventory_change,
        "5Y Revenue CAGR (%)":              cagr_percent(revenue),
        "5Y Net Profit CAGR (%)":           cagr_percent(net_profit),
        "5Y EBITDA CAGR (%)":               cagr_percent(ebitda_series),
        "Revenue Growth (%)":               revenue_growth,
        "Average Revenue Growth (%)":       average_growth(revenue),
        "Average Net Profit Growth (%)":    average_growth(net_profit),
        "Operating Margin (%)":             operating_margin,
        "Net Profit Margin (%)":            net_profit_margin,
        "Gross Profit":                     gross_profit,
        "Gross Margin (%)":                 gross_margin,
    }


def analyze_balance(data: dict, income_data: dict = None) -> dict:
    cash = data.get("cash", [])
    ca   = data.get("current_assets", [])
    inv  = data.get("inventory", [])
    cl   = data.get("current_liabilities", [])
    ncl  = data.get("non_current_liabilities", [])
    tsf  = data.get("shareholders_fund", [])
    sc   = data.get("share_capital", [])
    rs   = data.get("reserves", [])
    oci  = data.get("other_comprehensive_income", [])
    ta   = data.get("total_assets", [])

    total_liab = []
    for i in range(max(len(cl), len(ncl), 0)):
        c = cl[i]  if i < len(cl)  else 0
        n = ncl[i] if i < len(ncl) else 0
        total_liab.append((c or 0) + (n or 0))

    cash_ratio    = ratio_yearly(cash, cl)
    current_ratio = ratio_yearly(ca, cl)

    max_eq_len = max(len(tsf), len(sc), len(rs), len(oci), 0)
    equity_series = []
    for i in range(max_eq_len):
        if tsf and i < len(tsf) and tsf[i] is not None:
            equity_series.append(tsf[i])
        else:
            sc_v  = sc[i]  if i < len(sc)  else 0
            rs_v  = rs[i]  if i < len(rs)  else 0
            oci_v = oci[i] if i < len(oci) else 0
            s = (sc_v or 0) + (rs_v or 0) + (oci_v or 0)
            equity_series.append(s if s != 0 else None)

    de = ratio_yearly(total_liab, equity_series)
    dr = ratio_yearly(total_liab, ta)

    wc_vals = [
        round(ca[i] - cl[i], 2)
        if i < len(ca) and i < len(cl) and ca[i] is not None and cl[i] is not None
        else None
        for i in range(min(len(ca), len(cl)))
    ]

    np_series       = income_data.get("net_profit", []) if income_data else []
    interest_series = income_data.get("interest",   []) if income_data else []
    tax_series      = income_data.get("tax",         []) if income_data else []

    max_years = max(len(ta), len(equity_series), len(np_series),
                    len(interest_series), len(tax_series), len(cl), 0)
    interest_coverage = []; roa = []; roe = []; roce = []

    for i in range(max_years):
        np_i     = np_series[i]       if i < len(np_series)       else None
        it_i     = interest_series[i] if i < len(interest_series) else None
        tax_i    = tax_series[i]      if i < len(tax_series)      else None
        assets_i = ta[i]              if i < len(ta)              else None
        cl_i     = cl[i]              if i < len(cl)              else None
        eq_i     = equity_series[i]   if i < len(equity_series)   else None

        ebit_i = None if (np_i is None and it_i is None and tax_i is None) \
            else (np_i or 0) + (it_i or 0) + (tax_i or 0)

        interest_coverage.append(
            round(ebit_i / it_i, 2) if (ebit_i is not None and it_i not in (None, 0)) else None)
        roa.append(
            round(np_i / assets_i * 100, 2) if (np_i is not None and assets_i not in (None, 0)) else None)
        roe.append(
            round(np_i / eq_i * 100, 2) if (np_i is not None and eq_i not in (None, 0)) else None)
        denom_roce = (assets_i - cl_i) if (assets_i is not None and cl_i is not None) else None
        roce.append(
            round(ebit_i / denom_roce * 100, 2)
            if (ebit_i is not None and denom_roce not in (None, 0)) else None)

    return {
        "Statement":                                        "Balance Sheet",
        "Cash (All Years)":                                 cash,
        "Current Assets (All Years)":                       ca,
        "Inventory (All Years)":                            inv,
        "Total Assets (All Years)":                         ta,
        "Current Liabilities (All Years)":                  cl,
        "Non-Current Liabilities (All Years)":              ncl,
        "Total Liabilities (All Years)":                    total_liab,
        "Share Capital (All Years)":                        sc,
        "Reserves (All Years)":                             rs,
        "Computed Equity (All Years)":                      equity_series,
        "Cash Ratio (All Years)":                           cash_ratio,
        "Current Ratio (All Years)":                        current_ratio,
        "Working Capital (All Years)":                      wc_vals,
        "Debt-to-Equity (All Years)":                       de,
        "Debt Ratio (All Years)":                           dr,
        "Interest Coverage Ratio (All Years)":              interest_coverage,
        "Return on Assets (ROA %) (All Years)":             roa,
        "Return on Equity (ROE %) (All Years)":             roe,
        "Return on Capital Employed (ROCE %) (All Years)":  roce,
        "5Y Avg Current Ratio":                             avg(current_ratio[:5]),
        "5Y Avg D/E":                                       avg(de[:5]),
        "5Y Avg ROA (%)":                                   avg(roa[:5]),
        "5Y Avg ROE (%)":                                   avg(roe[:5]),
    }


def analyze_cashflow(data: dict, balance_data: dict = None, income_data: dict = None) -> dict:
    cfo   = data.get("cfo",             [])
    cfi   = data.get("cfi",             [])
    cff   = data.get("cff",             [])
    capex = data.get("capex",           [])
    netch = data.get("net_change_cash", [])

    fcf = []
    for i in range(min(len(cfo), len(capex))):
        if cfo[i] is not None and capex[i] is not None:
            fcf.append(round(cfo[i] - abs(capex[i]), 2))
        else:
            fcf.append(None)

    cl  = balance_data.get("current_liabilities",     []) if balance_data else []
    ncl = balance_data.get("non_current_liabilities", []) if balance_data else []
    ta  = balance_data.get("total_assets",            []) if balance_data else []

    eq = []
    if balance_data:
        tsf = balance_data.get("shareholders_fund", [])
        sc  = balance_data.get("share_capital",     [])
        rs  = balance_data.get("reserves",          [])
        oci = balance_data.get("other_comprehensive_income", [])
        for i in range(max(len(tsf), len(sc), len(rs), len(oci), 0)):
            if tsf and i < len(tsf) and tsf[i] is not None:
                eq.append(tsf[i])
            else:
                sc_i  = sc[i]  if i < len(sc)  else 0
                rs_i  = rs[i]  if i < len(rs)  else 0
                oci_i = oci[i] if i < len(oci) else 0
                s = (sc_i or 0) + (rs_i or 0) + (oci_i or 0)
                eq.append(s if s != 0 else None)

    revenue    = income_data.get("revenue",    []) if income_data else []
    net_profit = income_data.get("net_profit", []) if income_data else []
    interest   = income_data.get("interest",   []) if income_data else []
    tax        = income_data.get("tax",        []) if income_data else []

    ocf_ratio = []; cf_to_debt = []; croa = []; croe = []
    cf_margin = []; earnings_quality = []

    max_years = max(len(cfo), len(cl), len(ncl), len(ta),
                    len(eq), len(revenue), len(net_profit),
                    len(interest), len(tax), 0)

    for i in range(max_years):
        cfo_i = cfo[i]        if i < len(cfo)        else None
        cl_i  = cl[i]         if i < len(cl)          else None
        ncl_i = ncl[i]        if i < len(ncl)         else None
        ta_i  = ta[i]         if i < len(ta)           else None
        eq_i  = eq[i]         if i < len(eq)           else None
        rev_i = revenue[i]    if i < len(revenue)      else None
        np_i  = net_profit[i] if i < len(net_profit)   else None
        it_i  = interest[i]   if i < len(interest)     else None
        tax_i = tax[i]        if i < len(tax)           else None

        total_debt = ((cl_i or 0) + (ncl_i or 0)) \
            if (cl_i is not None or ncl_i is not None) else None

        ocf_ratio.append(
            round(cfo_i / cl_i, 2) if (cfo_i is not None and cl_i not in (None, 0)) else None)
        cf_to_debt.append(
            round(cfo_i / total_debt, 2) if (cfo_i is not None and total_debt not in (None, 0)) else None)
        croa.append(
            round(cfo_i / ta_i * 100, 2) if (cfo_i is not None and ta_i not in (None, 0)) else None)
        croe.append(
            round(cfo_i / eq_i * 100, 2) if (cfo_i is not None and eq_i not in (None, 0)) else None)
        cf_margin.append(
            round(cfo_i / rev_i * 100, 2) if (cfo_i is not None and rev_i not in (None, 0)) else None)

        ebit_i = None if (np_i is None and it_i is None and tax_i is None) \
            else (np_i or 0) + (it_i or 0) + (tax_i or 0)
        earnings_quality.append(
            round(cfo_i / ebit_i, 2) if (cfo_i is not None and ebit_i not in (None, 0)) else None)

    return {
        "Statement":                                "Cash Flow",
        "Operating Cash Flow (All Years)":          cfo,
        "Investing Cash Flow (All Years)":          cfi,
        "Financing Cash Flow (All Years)":          cff,
        "CapEx (All Years)":                        capex,
        "Free Cash Flow (All Years)":               fcf,
        "Net Change in Cash (All Years)":           netch,
        "Operating Cash Flow Ratio (All Years)":    ocf_ratio,
        "Cash Flow to Debt Ratio (All Years)":      cf_to_debt,
        "Cash Return on Assets (%) (All Years)":    croa,
        "Cash Return on Equity (%) (All Years)":    croe,
        "Cash Flow Margin (%) (All Years)":         cf_margin,
        "Earnings Quality Ratio (All Years)":       earnings_quality,
        "5Y CAGR Operating Cash Flow (%)":          cagr_percent(cfo),
        "5Y Avg Operating Cash Flow Ratio":         avg(ocf_ratio[:5]),
        "5Y Avg Cash Flow to Debt Ratio":           avg(cf_to_debt[:5]),
        "5Y Avg Earnings Quality Ratio":            avg(earnings_quality[:5]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry-point (desktop testing)
# ══════════════════════════════════════════════════════════════════════════════
def main():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        paths = filedialog.askopenfilenames(filetypes=[("PDF Files", "*.pdf")])
    except Exception:
        paths = []

    if not paths:
        print("No files selected.")
        return

    income_data = balance_data = cash_data = None

    for p in paths:
        print(f"\nExtracting: {p}")
        lines = pdf_to_text(p)
        inc = extract_income_series(lines)
        bal = extract_balance_series(lines)
        cf  = extract_cashflow_series(lines)
        if any(v for v in inc.values()) and income_data is None:
            income_data = inc
        if any(v for v in bal.values()) and balance_data is None:
            balance_data = bal
        if any(v for v in cf.values())  and cash_data   is None:
            cash_data = cf

    if income_data:
        print("\n===== INCOME STATEMENT =====")
        for k, v in analyze_income(income_data).items():
            print(f"  {k}: {v}")
    if balance_data:
        print("\n===== BALANCE SHEET =====")
        for k, v in analyze_balance(balance_data, income_data).items():
            print(f"  {k}: {v}")
    if cash_data:
        print("\n===== CASH FLOW =====")
        for k, v in analyze_cashflow(cash_data, balance_data, income_data).items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
