#!/usr/bin/env python3
"""
financialanalyzer.py
Extracts financial data from Indian company PDFs (Screener, BSE, NSE, Annual Reports).

Key design:
- pdfplumber for text extraction (preserves table row structure far better than PyMuPDF)
- Per-row matching: label cell matched against keywords, number cells collected separately
- Priority-lock: first match wins — sub-totals / footnotes cannot overwrite the real row
- Look-ahead: if label row has no numbers, checks next 2 rows
- Covers Screener.in exports, BSE filings, NSE filings, standalone annual reports
"""

import re
import shutil
from typing import List, Optional, Dict

# ── Tesseract (only needed for scanned PDFs) ──────────────────────────────────
try:
    import pytesseract
    _tess = shutil.which("tesseract")
    pytesseract.pytesseract.tesseract_cmd = _tess if _tess else \
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    _TESS_OK = True
except ImportError:
    _TESS_OK = False

DEBUG = False   # True → print every matched row to stdout


# ══════════════════════════════════════════════════════════════════════════════
# PDF → lines  (pdfplumber primary, OCR fallback)
# ══════════════════════════════════════════════════════════════════════════════
def pdf_to_text(pdf_path: str) -> List[str]:
    """
    Extract ALL text from a PDF using multiple strategies simultaneously.
    Never skips a page. Always extracts both table rows and plain text.
    """
    import pdfplumber

    lines: List[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_lines = set()  # deduplicate within a page

            # Strategy 1: structured table rows (best for financial tables)
            # Each row → cells joined by " | " so label+numbers are on one line
            try:
                tables = page.extract_tables()
                for table in (tables or []):
                    for row in (table or []):
                        if not row:
                            continue
                        cells = [re.sub(r'\s+', ' ', str(c).strip()) if c else "" for c in row]
                        cells = [c for c in cells if c]
                        if cells:
                            page_lines.add(" | ".join(cells))
            except Exception:
                pass

            # Strategy 2: plain text extraction (catches text outside tables)
            try:
                text = page.extract_text()
                if text:
                    for ln in text.splitlines():
                        ln = ln.strip()
                        if ln:
                            page_lines.add(ln)
            except Exception:
                pass

            # Strategy 3: OCR fallback if nothing extracted so far
            if not page_lines and _TESS_OK:
                try:
                    img = page.to_image(resolution=150).original
                    ocr_text = pytesseract.image_to_string(img)
                    for ln in ocr_text.splitlines():
                        ln = ln.strip()
                        if ln:
                            page_lines.add(ln)
                except Exception:
                    pass

            lines.extend(page_lines)

    return lines


# ══════════════════════════════════════════════════════════════════════════════
# Number helpers
# ══════════════════════════════════════════════════════════════════════════════
def clean_number(token) -> Optional[float]:
    if token is None:
        return None
    s = str(token).strip()
    # Remove unicode spaces, currency symbols, unit suffixes
    s = re.sub(r'[\u200b\xa0\u202f\u2009]', '', s)
    s = re.sub(r'(?i)(cr\.?|crore|lakh|lakhs|mn|million|bn|billion|rs\.?|inr|\u20b9|%)', '', s)
    s = s.replace(',', '').replace(' ', '')
    s = s.replace('(', '-').replace(')', '')
    s = re.sub(r'[^0-9.\-]', '', s)
    if not s or s in ('-', '.', '--', '---'):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def find_numbers(line: str) -> List[float]:
    """Extract all numeric values from a line of text."""
    # Match: optional leading minus, optional opening paren, digits+commas, optional decimal
    toks = re.findall(r'-?\(?\d[\d,]*(?:\.\d+)?\)?', line)
    result = []
    for t in toks:
        n = clean_number(t)
        if n is not None:
            result.append(n)
    return result


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
# Core extraction engine
# ══════════════════════════════════════════════════════════════════════════════
def _normalize(text: str) -> str:
    """Normalize a line for keyword matching."""
    t = text.lower()
    # Unicode dashes → hyphen
    t = re.sub(r'[\u2013\u2014\u2015]', '-', t)
    # Unicode quotes
    t = re.sub(r'[\u2018\u2019\u201c\u201d]', "'", t)
    # Collapse whitespace
    t = re.sub(r'\s+', ' ', t)
    return t.strip()


def _extract(lines: List[str], patterns: Dict[str, List[str]]) -> Dict[str, List[float]]:
    """
    Priority-locked extractor.

    For each line:
      1. Normalize it.
      2. For each UNLOCKED field, check keywords in priority order.
      3. If match found, extract numbers from this line.
         If no numbers, look ahead up to 3 lines.
      4. Lock the field once it has data.
    """
    res: Dict[str, List[float]] = {k: [] for k in patterns}
    locked: set = set()

    for idx, line in enumerate(lines):
        norm = _normalize(line)

        for field, keywords in patterns.items():
            if field in locked:
                continue

            matched = any(kw in norm for kw in keywords)
            if not matched:
                continue

            # Collect numbers from this line
            nums = find_numbers(line)

            # Look ahead if no numbers found here
            if not nums:
                for ahead in range(1, 4):
                    if idx + ahead < len(lines):
                        nums = find_numbers(lines[idx + ahead])
                        if nums:
                            break

            if nums:
                res[field] = nums
                locked.add(field)
                if DEBUG:
                    print(f"  [MATCH] {field:<35s}  kw matched  nums={nums[:6]}")
                    print(f"          line: {line[:100]!r}")

    return res


# ══════════════════════════════════════════════════════════════════════════════
# Extractors  (keyword lists cover all common Indian financial statement formats)
# ══════════════════════════════════════════════════════════════════════════════
def extract_income_series(lines: List[str]) -> dict:
    patterns = {
        # ── Revenue ──────────────────────────────────────────────────────────
        "revenue": [
            "revenue from operations",
            "revenues from operations",
            "total revenue from operations",
            "net revenue from operations",
            "total operating revenue",
            "gross revenue from operations",
            "income from operations",
            "net sales and other operating revenue",
            "net sales",
            "sales and other income",
            "total income from operations",
            "net revenue",
            "total revenue",
            "turnover",
            "revenue",
            "sales",
        ],
        # ── Net Profit ────────────────────────────────────────────────────────
        "net_profit": [
            "profit/(loss) for the period",
            "profit / (loss) for the period",
            "profit/(loss) for the year",
            "profit / (loss) for the year",
            "loss/(profit) for the period",
            "profit/loss for the period",
            "profit / loss for the period",
            "profit/loss for the year",
            "profit for the period attributable",
            "profit for the year attributable",
            "profit for the period",
            "profit for the year",
            "profit after tax",
            "profit after taxation",
            "net profit after tax",
            "net profit for the year",
            "net profit for the period",
            "profit after exceptional",
            "net profit",
            "pat",
        ],
        # ── Interest / Finance Costs ──────────────────────────────────────────
        "interest": [
            "finance costs",
            "finance cost",
            "interest and finance charges",
            "interest expense",
            "interest on borrowings",
            "interest on loans",
            "borrowing costs",
            "financial charges",
            "interest",
        ],
        # ── Tax ───────────────────────────────────────────────────────────────
        "tax": [
            "total tax expense",
            "total income tax expense",
            "income tax expense",
            "tax expense",
            "provision for tax",
            "provision for income tax",
            "current tax expense",
            "current tax",
            "deferred tax charge",
            "deferred tax",
            "income tax",
            "tax on profit",
        ],
        # ── Depreciation ─────────────────────────────────────────────────────
        "depreciation": [
            "depreciation and amortisation expense",
            "depreciation and amortization expense",
            "depreciation, amortisation",
            "depreciation & amortization",
            "depreciation and amortization",
            "depreciation and amortisation",
            "depreciation of tangible assets",
            "amortisation of intangible assets",
            "depreciation on right",
            "depreciation",
            "amortisation",
            "amortization",
        ],
        # ── COGS ─────────────────────────────────────────────────────────────
        "cogs": [
            "cost of materials consumed",
            "consumption of raw material",
            "raw material consumed",
            "raw materials consumed",
            "material consumed",
            "cost of goods sold",
            "cost of goods",
            "cost of products sold",
            "cost of revenue",
        ],
        # ── Purchases ─────────────────────────────────────────────────────────
        "purchases": [
            "purchases of stock-in-trade",
            "purchase of stock-in-trade",
            "purchase of traded goods",
            "purchases of traded goods",
            "purchases of stock in trade",
            "purchase of stock in trade",
            "trading purchases",
            "purchases",
        ],
        # ── Employee Cost ─────────────────────────────────────────────────────
        "employee_cost": [
            "employee benefit expenses",
            "employee benefits expense",
            "employee benefits expenses",
            "employee remuneration and benefits",
            "personnel expenses",
            "staff expenses",
            "staff costs",
            "salaries wages and bonus",
            "salaries and wages",
            "salaries, wages",
            "employee cost",
            "manpower cost",
        ],
        # ── Inventory Change ──────────────────────────────────────────────────
        "inventory_change": [
            "changes in inventories of finished goods, work-in-progress",
            "changes in inventories of finished goods",
            "change in inventories of finished goods",
            "(increase)/decrease in inventories",
            "increase/(decrease) in inventories",
            "changes in inventories",
            "change in inventories",
        ],
    }
    return _extract(lines, patterns)


def extract_balance_series(lines: List[str]) -> dict:
    patterns = {
        # ── Cash ──────────────────────────────────────────────────────────────
        "cash": [
            "cash and cash equivalents",
            "cash & cash equivalents",
            "cash and bank balances",
            "cash & bank balances",
            "cash and bank",
            "cash & bank",
            "balances with banks",
            "cash in hand",
        ],
        # ── Current Assets ────────────────────────────────────────────────────
        "current_assets": [
            "total current assets",
            "current assets",
        ],
        # ── Inventory ─────────────────────────────────────────────────────────
        "inventory": [
            "inventories",
            "inventory",
            "stock-in-trade",
        ],
        # ── Current Liabilities ───────────────────────────────────────────────
        "current_liabilities": [
            "total current liabilities",
            "total current liabilities and provisions",
            "current liabilities and provisions",
            "current liabilities",
        ],
        # ── Non-Current Liabilities ───────────────────────────────────────────
        "non_current_liabilities": [
            "total non-current liabilities",
            "non-current liabilities",
            "non current liabilities",
            "long-term liabilities",
        ],
        # ── Shareholders' Funds / Equity ──────────────────────────────────────
        "shareholders_fund": [
            "total equity",
            "total equity attributable",
            "total shareholders' funds",
            "total shareholders funds",
            "shareholders' funds",
            "shareholders funds",
            "total net worth",
            "net worth",
        ],
        # ── Share Capital ─────────────────────────────────────────────────────
        "share_capital": [
            "equity share capital",
            "paid-up share capital",
            "paid up share capital",
            "share capital",
        ],
        # ── Reserves ─────────────────────────────────────────────────────────
        "reserves": [
            "other equity",
            "reserves and surplus",
            "reserves & surplus",
            "other reserves and surplus",
            "other reserves",
            "retained earnings",
            "reserves",
        ],
        # ── OCI ───────────────────────────────────────────────────────────────
        "other_comprehensive_income": [
            "other comprehensive income",
            "total other comprehensive income",
            "oci",
        ],
        # ── Total Assets ──────────────────────────────────────────────────────
        "total_assets": [
            "total assets",
        ],
        # ── Borrowings ────────────────────────────────────────────────────────
        "borrowings": [
            "total borrowings",
            "long term borrowings",
            "long-term borrowings",
            "short term borrowings",
            "short-term borrowings",
            "total debt",
            "borrowings",
        ],
    }
    return _extract(lines, patterns)


def extract_cashflow_series(lines: List[str]) -> dict:
    patterns = {
        # ── Operating ─────────────────────────────────────────────────────────
        "cfo": [
            "net cash generated from operating activities",
            "net cash from operating activities",
            "net cash inflow from operating activities",
            "net cash outflow from operating activities",
            "net cash used in operating activities",
            "net cash generated from operations",
            "cash generated from operations",
            "net cash flow from operating activities",
            "net cash flow from operations",
            "cash flow from operating activities",
            "net cash from operating",
            "cash from operations",
        ],
        # ── Investing ─────────────────────────────────────────────────────────
        "cfi": [
            "net cash used in investing activities",
            "net cash from investing activities",
            "net cash inflow from investing activities",
            "net cash outflow from investing activities",
            "net cash flow from investing activities",
            "net cash flow from investing",
            "cash flow from investing activities",
            "net cash used in investing",
            "net cash from investing",
        ],
        # ── Financing ─────────────────────────────────────────────────────────
        "cff": [
            "net cash used in financing activities",
            "net cash from financing activities",
            "net cash inflow from financing activities",
            "net cash outflow from financing activities",
            "net cash flow from financing activities",
            "net cash flow from financing",
            "cash flow from financing activities",
            "net cash from financing",
        ],
        # ── Capex ─────────────────────────────────────────────────────────────
        "capex": [
            "purchase of property, plant and equipment",
            "purchase of property plant and equipment",
            "acquisition of property, plant and equipment",
            "payments for property, plant and equipment",
            "purchase of fixed assets",
            "purchase of tangible assets",
            "additions to fixed assets",
            "capital expenditure on fixed assets",
            "capital expenditure",
            "additions to property",
        ],
        # ── Net Change in Cash ────────────────────────────────────────────────
        "net_change_cash": [
            "net increase/(decrease) in cash and cash equivalents",
            "net decrease/(increase) in cash and cash equivalents",
            "net increase in cash and cash equivalents",
            "net decrease in cash and cash equivalents",
            "net change in cash and cash equivalents",
            "net (decrease)/increase in cash",
            "net increase/(decrease) in cash",
            "net change in cash",
        ],
    }
    return _extract(lines, patterns)


# ══════════════════════════════════════════════════════════════════════════════
# Type detection
# ══════════════════════════════════════════════════════════════════════════════
def detect_type(lines: List[str], filename: str = "") -> Optional[str]:
    txt = _normalize(" ".join(lines))
    fname = filename.lower()

    if any(k in fname for k in ["income", "p&l", "pnl", "profit", "_pl", "-pl"]):
        return "income"
    if any(k in fname for k in ["balance", "_bs", "-bs", "balancesheet", "balance-sheet"]):
        return "balance"
    if any(k in fname for k in ["cash", "cashflow", "cash-flow", "cfs"]):
        return "cash"

    i = sum([4 if "revenue from operations" in txt else 0,
             4 if "profit for the period" in txt or "profit for the year" in txt else 0,
             2 if "net profit" in txt else 0, 1 if "finance costs" in txt else 0])
    b = sum([4 if "total assets" in txt else 0,
             4 if "total equity" in txt or "total shareholders" in txt else 0,
             3 if "current liabilities" in txt else 0, 1 if "equity share capital" in txt else 0])
    c = sum([5 if "cash flow from operating activities" in txt else 0,
             5 if "net cash from operating" in txt else 0,
             4 if "cash flow from investing" in txt else 0,
             3 if "cash flow from financing" in txt else 0])

    best = max(i, b, c, 0)
    if best == 0: return None
    if i == best: return "income"
    if c == best: return "cash"
    return "balance"


def detect_all_types(lines: List[str]) -> List[str]:
    txt = _normalize(" ".join(lines))
    found = []
    if sum(1 for s in ["revenue from operations", "profit for the period", "profit for the year",
                        "net profit", "finance costs", "employee benefit"] if s in txt) >= 2:
        found.append("income")
    if sum(1 for s in ["total assets", "total equity", "total shareholders", "current liabilities",
                        "equity share capital", "reserves and surplus"] if s in txt) >= 2:
        found.append("balance")
    if sum(1 for s in ["cash flow from operating", "net cash from operating",
                        "cash flow from investing", "cash flow from financing",
                        "net cash used in investing"] if s in txt) >= 2:
        found.append("cash")
    return found


# ══════════════════════════════════════════════════════════════════════════════
# Analyzers  (unchanged logic, only crash-guards added)
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

    def v(s, i=0): return s[i] if len(s) > i else None

    rev_cur, rev_prev = v(revenue, 0), v(revenue, 1)
    net_cur = v(net_profit, 0)

    revenue_growth = round(((rev_cur - rev_prev) / rev_prev * 100), 2) \
        if (rev_cur is not None and rev_prev not in (None, 0)) else None

    max_len = max(len(net_profit), len(interest), len(tax), len(depreciation), 0)
    ebitda_series = []
    for i in range(max_len):
        n  = net_profit[i]   if i < len(net_profit)   else None
        it = interest[i]     if i < len(interest)     else 0
        tx = tax[i]          if i < len(tax)           else 0
        dp = depreciation[i] if i < len(depreciation) else 0
        if n is not None:
            ebitda_series.append(round((n or 0)+(it or 0)+(tx or 0)+(dp or 0), 2))

    op = None
    if net_cur is not None:
        op = round((net_cur or 0)+(v(interest,0) or 0)+(v(tax,0) or 0), 2)

    opm = round(op/rev_cur*100, 2) if (op is not None and rev_cur not in (None,0)) else None
    npm = round(net_cur/rev_cur*100, 2) if (net_cur is not None and rev_cur not in (None,0)) else None

    cogs0, pur0, emp0, inv0 = (v(cogs,0) or 0),(v(purchases,0) or 0),(v(employee_cost,0) or 0),(v(inventory_change,0) or 0)
    gp = gm = None
    if rev_cur is not None:
        tc = cogs0+pur0+emp0-inv0
        gp = round(rev_cur-tc, 2)
        gm = round(gp/rev_cur*100, 2) if rev_cur else None

    return {
        "Statement":                     "Income Statement",
        "Revenue (All Years)":           revenue,
        "Net Profit (All Years)":        net_profit,
        "EBITDA (All Years)":            ebitda_series,
        "COGS (All Years)":              cogs,
        "Employee Cost (All Years)":     employee_cost,
        "Depreciation (All Years)":      depreciation,
        "Interest (All Years)":          interest,
        "Tax (All Years)":               tax,
        "Purchases (All Years)":         purchases,
        "Inventory Change (All Years)":  inventory_change,
        "5Y Revenue CAGR (%)":           cagr_percent(revenue),
        "5Y Net Profit CAGR (%)":        cagr_percent(net_profit),
        "5Y EBITDA CAGR (%)":            cagr_percent(ebitda_series),
        "Revenue Growth (%)":            revenue_growth,
        "Average Revenue Growth (%)":    average_growth(revenue),
        "Average Net Profit Growth (%)": average_growth(net_profit),
        "Operating Margin (%)":          opm,
        "Net Profit Margin (%)":         npm,
        "Gross Profit":                  gp,
        "Gross Margin (%)":              gm,
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
        total_liab.append((c or 0)+(n or 0))

    current_ratio = ratio_yearly(ca, cl)
    cash_ratio    = ratio_yearly(cash, cl)

    max_eq = max(len(tsf), len(sc), len(rs), len(oci), 0)
    equity = []
    for i in range(max_eq):
        if tsf and i < len(tsf) and tsf[i] is not None:
            equity.append(tsf[i])
        else:
            s = (sc[i] if i<len(sc) else 0 or 0) + \
                (rs[i] if i<len(rs) else 0 or 0) + \
                (oci[i] if i<len(oci) else 0 or 0)
            equity.append(s if s else None)

    de = ratio_yearly(total_liab, equity)
    dr = ratio_yearly(total_liab, ta)

    wc = [round(ca[i]-cl[i],2) if i<len(ca) and i<len(cl) and ca[i] is not None and cl[i] is not None else None
          for i in range(min(len(ca),len(cl)))]

    nps = income_data.get("net_profit",[]) if income_data else []
    its = income_data.get("interest",  []) if income_data else []
    txs = income_data.get("tax",       []) if income_data else []

    max_y = max(len(ta),len(equity),len(nps),len(its),len(txs),len(cl),0)
    ic=[]; roa=[]; roe=[]; roce=[]
    for i in range(max_y):
        np_i = nps[i] if i<len(nps) else None
        it_i = its[i] if i<len(its) else None
        tx_i = txs[i] if i<len(txs) else None
        ta_i = ta[i]  if i<len(ta)  else None
        cl_i = cl[i]  if i<len(cl)  else None
        eq_i = equity[i] if i<len(equity) else None
        ebit = None if (np_i is None and it_i is None and tx_i is None) \
            else (np_i or 0)+(it_i or 0)+(tx_i or 0)
        ic.append(round(ebit/it_i,2) if (ebit is not None and it_i not in (None,0)) else None)
        roa.append(round(np_i/ta_i*100,2) if (np_i is not None and ta_i not in (None,0)) else None)
        roe.append(round(np_i/eq_i*100,2) if (np_i is not None and eq_i not in (None,0)) else None)
        dr_ = (ta_i-cl_i) if (ta_i is not None and cl_i is not None) else None
        roce.append(round(ebit/dr_*100,2) if (ebit is not None and dr_ not in (None,0)) else None)

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
        "Computed Equity (All Years)":                      equity,
        "Cash Ratio (All Years)":                           cash_ratio,
        "Current Ratio (All Years)":                        current_ratio,
        "Working Capital (All Years)":                      wc,
        "Debt-to-Equity (All Years)":                       de,
        "Debt Ratio (All Years)":                           dr,
        "Interest Coverage Ratio (All Years)":              ic,
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
    for i in range(min(len(cfo),len(capex))):
        if cfo[i] is not None and capex[i] is not None:
            fcf.append(round(cfo[i]-abs(capex[i]),2))
        else:
            fcf.append(None)

    cl  = balance_data.get("current_liabilities",     []) if balance_data else []
    ncl = balance_data.get("non_current_liabilities", []) if balance_data else []
    ta  = balance_data.get("total_assets",            []) if balance_data else []
    eq  = []
    if balance_data:
        tsf = balance_data.get("shareholders_fund",[])
        sc  = balance_data.get("share_capital",    [])
        rs  = balance_data.get("reserves",         [])
        oci = balance_data.get("other_comprehensive_income",[])
        for i in range(max(len(tsf),len(sc),len(rs),len(oci),0)):
            if tsf and i<len(tsf) and tsf[i] is not None:
                eq.append(tsf[i])
            else:
                s = (sc[i] if i<len(sc) else 0 or 0)+(rs[i] if i<len(rs) else 0 or 0)+(oci[i] if i<len(oci) else 0 or 0)
                eq.append(s if s else None)

    rev = income_data.get("revenue",    []) if income_data else []
    np_ = income_data.get("net_profit", []) if income_data else []
    it_ = income_data.get("interest",   []) if income_data else []
    tx_ = income_data.get("tax",        []) if income_data else []

    ocfr=[]; cfd=[]; croa=[]; croe=[]; cfm=[]; eq_=[]; 
    max_y = max(len(cfo),len(cl),len(ncl),len(ta),len(eq),len(rev),len(np_),len(it_),len(tx_),0)

    for i in range(max_y):
        c_  = cfo[i] if i<len(cfo) else None
        cl_ = cl[i]  if i<len(cl)  else None
        nc_ = ncl[i] if i<len(ncl) else None
        ta_ = ta[i]  if i<len(ta)  else None
        eq_i= eq[i]  if i<len(eq)  else None
        rv_ = rev[i] if i<len(rev) else None
        np_i= np_[i] if i<len(np_) else None
        it_i= it_[i] if i<len(it_) else None
        tx_i= tx_[i] if i<len(tx_) else None
        td  = ((cl_ or 0)+(nc_ or 0)) if (cl_ is not None or nc_ is not None) else None
        ocfr.append(round(c_/cl_,2)  if (c_ is not None and cl_ not in (None,0)) else None)
        cfd.append(round(c_/td,2)    if (c_ is not None and td  not in (None,0)) else None)
        croa.append(round(c_/ta_*100,2) if (c_ is not None and ta_ not in (None,0)) else None)
        croe.append(round(c_/eq_i*100,2) if (c_ is not None and eq_i not in (None,0)) else None)
        cfm.append(round(c_/rv_*100,2) if (c_ is not None and rv_ not in (None,0)) else None)
        ebit= None if (np_i is None and it_i is None and tx_i is None) \
            else (np_i or 0)+(it_i or 0)+(tx_i or 0)
        eq_.append(round(c_/ebit,2) if (c_ is not None and ebit not in (None,0)) else None)

    return {
        "Statement":                             "Cash Flow",
        "Operating Cash Flow (All Years)":       cfo,
        "Investing Cash Flow (All Years)":       cfi,
        "Financing Cash Flow (All Years)":       cff,
        "CapEx (All Years)":                     capex,
        "Free Cash Flow (All Years)":            fcf,
        "Net Change in Cash (All Years)":        netch,
        "Operating Cash Flow Ratio (All Years)": ocfr,
        "Cash Flow to Debt Ratio (All Years)":   cfd,
        "Cash Return on Assets (%) (All Years)": croa,
        "Cash Return on Equity (%) (All Years)": croe,
        "Cash Flow Margin (%) (All Years)":      cfm,
        "Earnings Quality Ratio (All Years)":    eq_,
        "5Y CAGR Operating Cash Flow (%)":       cagr_percent(cfo),
        "5Y Avg Operating Cash Flow Ratio":      avg(ocfr[:5]),
        "5Y Avg Cash Flow to Debt Ratio":        avg(cfd[:5]),
        "5Y Avg Earnings Quality Ratio":         avg(eq_[:5]),
    }


def main():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        paths = filedialog.askopenfilenames(filetypes=[("PDF Files","*.pdf")])
    except Exception:
        paths = []
    if not paths:
        print("No files selected."); return
    income_data = balance_data = cash_data = None
    for p in paths:
        print(f"\nExtracting: {p}")
        lines = pdf_to_text(p)
        print(f"  {len(lines)} lines extracted")
        inc = extract_income_series(lines)
        bal = extract_balance_series(lines)
        cf  = extract_cashflow_series(lines)
        print(f"  Income  fields: {[k for k,v in inc.items() if v]}")
        print(f"  Balance fields: {[k for k,v in bal.items() if v]}")
        print(f"  Cash    fields: {[k for k,v in cf.items() if v]}")
        if any(v for v in inc.values()) and income_data is None: income_data = inc
        if any(v for v in bal.values()) and balance_data is None: balance_data = bal
        if any(v for v in cf.values())  and cash_data   is None: cash_data = cf
    if income_data:
        print("\n===== INCOME STATEMENT =====")
        for k,v in analyze_income(income_data).items(): print(f"  {k}: {v}")
    if balance_data:
        print("\n===== BALANCE SHEET =====")
        for k,v in analyze_balance(balance_data,income_data).items(): print(f"  {k}: {v}")
    if cash_data:
        print("\n===== CASH FLOW =====")
        for k,v in analyze_cashflow(cash_data,balance_data,income_data).items(): print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
