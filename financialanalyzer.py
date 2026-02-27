#!/usr/bin/env python3
"""
financialanalyzer.py
Handles Moneycontrol/Screener/BSE/NSE financial PDFs.
Moneycontrol PDFs are IMAGE-based screenshots — OCR is required.
"""

import re
import shutil
from typing import List, Optional, Dict

try:
    import pytesseract
    _tess = shutil.which("tesseract")
    pytesseract.pytesseract.tesseract_cmd = _tess if _tess else r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    _TESS_OK = True
except ImportError:
    _TESS_OK = False

DEBUG = False

def is_image_pdf(pdf_path: str) -> bool:
    """
    Quick check: returns True if the PDF has no text layer (image/scanned PDF).
    Checks only first 3 pages for speed.
    """
    try:
        import fitz
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            if i >= 3:
                break
            text = page.get_text("text").strip()
            if len(text) > 50:   # found real text
                doc.close()
                return False
        doc.close()
        return True
    except Exception:
        pass
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                if i >= 3:
                    break
                text = page.extract_text()
                if text and len(text.strip()) > 50:
                    return False
        return True
    except Exception:
        return False



def _find_financial_pages(doc) -> List[int]:
    """
    Scan page text quickly to find pages containing financial statements.
    Returns list of page indices. Falls back to last 60% of doc if nothing found.
    """
    TRIGGERS = [
        "statement of profit", "profit and loss", "profit & loss",
        "balance sheet", "statement of financial position",
        "cash flow", "income statement",
        "revenue from operations", "total assets", "total equity",
        "standalone financial", "consolidated financial",
    ]
    total = len(doc)
    hits = []
    for i, page in enumerate(doc):
        try:
            snippet = page.get_text("text")[:500].lower()
        except Exception:
            snippet = ""
        if any(t in snippet for t in TRIGGERS):
            # Include a window of ±5 pages around each hit
            start = max(0, i - 2)
            end   = min(total, i + 30)
            hits.extend(range(start, end))

    if hits:
        return sorted(set(hits))

    # Fallback: process last 60% of document (financials are always near the end)
    start = max(0, int(total * 0.40))
    return list(range(start, total))


def pdf_to_text(pdf_path: str) -> List[str]:
    """
    Extract text from financial statement pages only.
    For BSE annual reports (150-300 pages), scans for relevant pages first
    so we only process ~30-60 pages instead of the whole document.
    """
    lines: List[str] = []

    # Strategy 1: PyMuPDF — smart page selection
    try:
        import fitz
        doc = fitz.open(pdf_path)
        pages_to_read = _find_financial_pages(doc)

        for i in pages_to_read:
            page = doc[i]
            # Plain text
            try:
                text = page.get_text("text")
                if text and text.strip():
                    for ln in text.splitlines():
                        ln = ln.strip()
                        if ln:
                            lines.append(ln)
            except Exception:
                pass
            # Row reconstruction (groups spans by Y coordinate → label | num | num)
            try:
                raw  = page.get_text("rawdict")
                rows: Dict[int, list] = {}
                for block in raw.get("blocks", []):
                    for bline in block.get("lines", []):
                        y = int(bline["bbox"][1])
                        for span in bline.get("spans", []):
                            t = span["text"].strip()
                            if t:
                                rows.setdefault(y, []).append((span["bbox"][0], t))
                for y in sorted(rows):
                    cells = [t for _, t in sorted(rows[y])]
                    joined = " | ".join(cells)
                    if joined not in lines:
                        lines.append(joined)
            except Exception:
                pass

        doc.close()
    except ImportError:
        pass
    except Exception:
        pass

    # Strategy 2: pdfplumber (fallback for PDFs fitz can't read)
    if not lines:
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                total = len(pdf.pages)
                start = max(0, int(total * 0.40))
                for page in pdf.pages[start:]:
                    try:
                        for table in (page.extract_tables() or []):
                            for row in table:
                                cells = [re.sub(r'\s+', ' ', str(c).strip()) if c else "" for c in row]
                                cells = [c for c in cells if c]
                                if cells:
                                    lines.append(" | ".join(cells))
                    except Exception:
                        pass
                    try:
                        text = page.extract_text()
                        if text:
                            for ln in text.splitlines():
                                ln = ln.strip()
                                if ln:
                                    lines.append(ln)
                    except Exception:
                        pass
        except Exception:
            pass

    # Strategy 3: OCR (last resort — only if no text found at all)
    if not lines and _TESS_OK:
        try:
            import fitz
            from PIL import Image
            import io
            doc  = fitz.open(pdf_path)
            pages_to_read = _find_financial_pages(doc)
            for i in pages_to_read:
                page = doc[i]
                pix  = page.get_pixmap(dpi=200)
                img  = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img, config="--psm 6 --oem 3")
                for ln in text.splitlines():
                    ln = ln.strip()
                    if ln:
                        lines.append(ln)
            doc.close()
        except Exception:
            pass

    return lines


def clean_number(token) -> Optional[float]:
    if token is None:
        return None
    s = str(token).strip()
    s = re.sub(r'[\u200b\xa0\u202f]', '', s)
    s = re.sub(r'(?i)(cr\.?|crore|lakh|lakhs|mn|million|bn|billion|rs\.?|inr|\u20b9)', '', s)
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


def _normalize(text: str) -> str:
    t = text.lower()
    t = re.sub(r'[\u2013\u2014\u2015]', '-', t)
    t = re.sub(r'[\u2018\u2019\u201c\u201d]', "'", t)
    t = re.sub(r'\s+', ' ', t)
    return t.strip()


def _extract(lines: List[str], patterns: Dict[str, List[str]]) -> Dict[str, List[float]]:
    res: Dict[str, List[float]] = {k: [] for k in patterns}
    locked: set = set()
    for idx, line in enumerate(lines):
        norm = _normalize(line)
        for field, keywords in patterns.items():
            if field in locked:
                continue
            if not any(kw in norm for kw in keywords):
                continue
            nums = find_numbers(line)
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
                    print(f"  [MATCH] {field:<35s} nums={nums[:6]}  line={line[:80]!r}")
    return res


def extract_income_series(lines: List[str]) -> dict:
    patterns = {
        "revenue": [
            "revenue from operations [net]",
            "revenue from operations [gross]",
            "total operating revenues",
            "revenue from operations",
            "total revenue from operations",
            "net revenue from operations",
            "total operating revenue",
            "net sales",
            "net revenue",
            "total revenue",
            "turnover",
            "revenue",
            "sales",
        ],
        "net_profit": [
            "profit/loss for the period",
            "profit/loss from continuing operations",
            "profit/loss after tax and before extraordinary items",
            "profit/(loss) for the period",
            "profit/(loss) for the year",
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
            "interest and finance charges",
            "borrowing costs",
            "interest",
        ],
        "tax": [
            "total tax expenses",
            "tax expenses-continued operations",
            "total tax expense",
            "tax expense",
            "provision for tax",
            "current tax",
            "income tax expense",
            "income tax",
        ],
        "depreciation": [
            "depreciation and amortisation expenses",
            "depreciation and amortization expenses",
            "depreciation and amortisation expense",
            "depreciation and amortization expense",
            "depreciation & amortization",
            "depreciation and amortisation",
            "depreciation and amortization",
            "depreciation",
            "amortisation",
        ],
        "cogs": [
            "cost of materials consumed",
            "cost of goods sold",
            "raw material consumed",
            "raw materials consumed",
            "material consumed",
        ],
        "purchases": [
            "purchase of stock-in trade",
            "purchase of stock-in-trade",
            "purchases of stock-in-trade",
            "purchases of stock in trade",
            "purchase of traded goods",
            "purchases",
        ],
        "employee_cost": [
            "employee benefit expenses",
            "employee benefits expense",
            "employee benefits expenses",
            "personnel expenses",
            "staff costs",
            "employee cost",
            "salaries and wages",
        ],
        "inventory_change": [
            "changes in inventories of fg,wip and stock-in trade",
            "changes in inventories of fg, wip and stock-in trade",
            "changes in inventories of finished goods",
            "change in inventories of finished goods",
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
        ],
        "shareholders_fund": [
            "total equity",
            "total shareholders' funds",
            "total shareholders funds",
            "shareholders' funds",
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
            "reserves",
        ],
        "other_comprehensive_income": [
            "other comprehensive income",
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
            "net cash from operating activities",
            "net cash generated from operating activities",
            "net cash used in operating activities",
            "cash flow from operating activities",
            "net cash from operating",
        ],
        "cfi": [
            "net cash used in investing activities",
            "net cash from investing activities",
            "cash flow from investing activities",
            "net cash from investing",
        ],
        "cff": [
            "net cash used in financing activities",
            "net cash from financing activities",
            "cash flow from financing activities",
            "net cash from financing",
        ],
        "capex": [
            "purchase of property, plant and equipment",
            "capital expenditure",
            "purchase of fixed assets",
            "purchase of tangible assets",
        ],
        "net_change_cash": [
            "net increase/(decrease) in cash and cash equivalents",
            "net decrease/(increase) in cash and cash equivalents",
            "net increase in cash and cash equivalents",
            "net decrease in cash and cash equivalents",
            "net change in cash and cash equivalents",
            "net change in cash",
        ],
    }
    return _extract(lines, patterns)


def detect_type(lines: List[str], filename: str = "") -> Optional[str]:
    txt = _normalize(" ".join(lines))
    fname = filename.lower()
    if any(k in fname for k in ["income", "p&l", "pnl", "profit", "_pl", "-pl"]):
        return "income"
    if any(k in fname for k in ["balance", "_bs", "-bs", "balancesheet"]):
        return "balance"
    if any(k in fname for k in ["cash", "cashflow", "cash-flow", "cfs"]):
        return "cash"
    i = sum([4 if "revenue from operations" in txt else 0,
             4 if "profit/loss for the period" in txt or "profit for the year" in txt else 0,
             2 if "finance costs" in txt else 0])
    b = sum([4 if "total assets" in txt else 0,
             4 if "total equity" in txt or "total shareholders" in txt else 0,
             3 if "current liabilities" in txt else 0])
    c = sum([5 if "cash flow from operating" in txt else 0,
             4 if "cash flow from investing" in txt else 0])
    best = max(i, b, c, 0)
    if best == 0:
        return None
    if i == best:
        return "income"
    if c == best:
        return "cash"
    return "balance"


def detect_all_types(lines: List[str]) -> List[str]:
    txt = _normalize(" ".join(lines))
    found = []
    if sum(1 for s in ["revenue from operations", "profit/loss for the period",
                        "profit for the year", "finance costs", "employee benefit"] if s in txt) >= 2:
        found.append("income")
    if sum(1 for s in ["total assets", "total equity", "total shareholders",
                        "current liabilities", "equity share capital"] if s in txt) >= 2:
        found.append("balance")
    if sum(1 for s in ["cash flow from operating", "net cash from operating",
                        "cash flow from investing", "cash flow from financing"] if s in txt) >= 2:
        found.append("cash")
    return found


def analyze_income(data: dict) -> dict:
    revenue = data.get("revenue", [])
    net_profit = data.get("net_profit", [])
    interest = data.get("interest", [])
    tax = data.get("tax", [])
    depreciation = data.get("depreciation", [])
    cogs = data.get("cogs", [])
    purchases = data.get("purchases", [])
    employee_cost = data.get("employee_cost", [])
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
            ebitda_series.append(round((n or 0) + (it or 0) + (tx or 0) + (dp or 0), 2))

    op = round((net_cur or 0) + (v(interest, 0) or 0) + (v(tax, 0) or 0), 2) if net_cur is not None else None
    opm = round(op / rev_cur * 100, 2) if (op is not None and rev_cur not in (None, 0)) else None
    npm = round(net_cur / rev_cur * 100, 2) if (net_cur is not None and rev_cur not in (None, 0)) else None

    cogs0 = v(cogs, 0) or 0
    pur0  = v(purchases, 0) or 0
    emp0  = v(employee_cost, 0) or 0
    inv0  = v(inventory_change, 0) or 0
    gp = gm = None
    if rev_cur is not None:
        tc = cogs0 + pur0 + emp0 - inv0
        gp = round(rev_cur - tc, 2)
        gm = round(gp / rev_cur * 100, 2) if rev_cur else None

    return {
        "Statement": "Income Statement",
        "Revenue (All Years)": revenue,
        "Net Profit (All Years)": net_profit,
        "EBITDA (All Years)": ebitda_series,
        "COGS (All Years)": cogs,
        "Employee Cost (All Years)": employee_cost,
        "Depreciation (All Years)": depreciation,
        "Interest (All Years)": interest,
        "Tax (All Years)": tax,
        "Purchases (All Years)": purchases,
        "Inventory Change (All Years)": inventory_change,
        "5Y Revenue CAGR (%)": cagr_percent(revenue),
        "5Y Net Profit CAGR (%)": cagr_percent(net_profit),
        "5Y EBITDA CAGR (%)": cagr_percent(ebitda_series),
        "Revenue Growth (%)": revenue_growth,
        "Average Revenue Growth (%)": average_growth(revenue),
        "Average Net Profit Growth (%)": average_growth(net_profit),
        "Operating Margin (%)": opm,
        "Net Profit Margin (%)": npm,
        "Gross Profit": gp,
        "Gross Margin (%)": gm,
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

    current_ratio = ratio_yearly(ca, cl)
    cash_ratio    = ratio_yearly(cash, cl)

    max_eq = max(len(tsf), len(sc), len(rs), len(oci), 0)
    equity = []
    for i in range(max_eq):
        if tsf and i < len(tsf) and tsf[i] is not None:
            equity.append(tsf[i])
        else:
            s = (sc[i] if i < len(sc) else 0 or 0) + \
                (rs[i] if i < len(rs) else 0 or 0) + \
                (oci[i] if i < len(oci) else 0 or 0)
            equity.append(s if s else None)

    de = ratio_yearly(total_liab, equity)
    dr = ratio_yearly(total_liab, ta)
    wc = [round(ca[i] - cl[i], 2)
          if i < len(ca) and i < len(cl) and ca[i] is not None and cl[i] is not None
          else None for i in range(min(len(ca), len(cl)))]

    nps = income_data.get("net_profit", []) if income_data else []
    its = income_data.get("interest",   []) if income_data else []
    txs = income_data.get("tax",        []) if income_data else []

    max_y = max(len(ta), len(equity), len(nps), len(its), len(txs), len(cl), 0)
    ic = []; roa = []; roe = []; roce = []
    for i in range(max_y):
        np_i = nps[i] if i < len(nps) else None
        it_i = its[i] if i < len(its) else None
        tx_i = txs[i] if i < len(txs) else None
        ta_i = ta[i]  if i < len(ta)  else None
        cl_i = cl[i]  if i < len(cl)  else None
        eq_i = equity[i] if i < len(equity) else None
        ebit = None if (np_i is None and it_i is None and tx_i is None) \
            else (np_i or 0) + (it_i or 0) + (tx_i or 0)
        ic.append(round(ebit / it_i, 2) if (ebit is not None and it_i not in (None, 0)) else None)
        roa.append(round(np_i / ta_i * 100, 2) if (np_i is not None and ta_i not in (None, 0)) else None)
        roe.append(round(np_i / eq_i * 100, 2) if (np_i is not None and eq_i not in (None, 0)) else None)
        dr_ = (ta_i - cl_i) if (ta_i is not None and cl_i is not None) else None
        roce.append(round(ebit / dr_ * 100, 2) if (ebit is not None and dr_ not in (None, 0)) else None)

    return {
        "Statement": "Balance Sheet",
        "Cash (All Years)": cash, "Current Assets (All Years)": ca,
        "Inventory (All Years)": inv, "Total Assets (All Years)": ta,
        "Current Liabilities (All Years)": cl, "Non-Current Liabilities (All Years)": ncl,
        "Total Liabilities (All Years)": total_liab, "Share Capital (All Years)": sc,
        "Reserves (All Years)": rs, "Computed Equity (All Years)": equity,
        "Cash Ratio (All Years)": cash_ratio, "Current Ratio (All Years)": current_ratio,
        "Working Capital (All Years)": wc, "Debt-to-Equity (All Years)": de,
        "Debt Ratio (All Years)": dr, "Interest Coverage Ratio (All Years)": ic,
        "Return on Assets (ROA %) (All Years)": roa, "Return on Equity (ROE %) (All Years)": roe,
        "Return on Capital Employed (ROCE %) (All Years)": roce,
        "5Y Avg Current Ratio": avg(current_ratio[:5]), "5Y Avg D/E": avg(de[:5]),
        "5Y Avg ROA (%)": avg(roa[:5]), "5Y Avg ROE (%)": avg(roe[:5]),
    }


def analyze_cashflow(data: dict, balance_data: dict = None, income_data: dict = None) -> dict:
    cfo   = data.get("cfo", [])
    cfi   = data.get("cfi", [])
    cff   = data.get("cff", [])
    capex = data.get("capex", [])
    netch = data.get("net_change_cash", [])

    fcf = [round(cfo[i] - abs(capex[i]), 2)
           if cfo[i] is not None and capex[i] is not None else None
           for i in range(min(len(cfo), len(capex)))]

    cl  = balance_data.get("current_liabilities",     []) if balance_data else []
    ncl = balance_data.get("non_current_liabilities", []) if balance_data else []
    ta  = balance_data.get("total_assets",            []) if balance_data else []
    eq  = []
    if balance_data:
        tsf = balance_data.get("shareholders_fund", [])
        sc  = balance_data.get("share_capital",     [])
        rs  = balance_data.get("reserves",          [])
        oci = balance_data.get("other_comprehensive_income", [])
        for i in range(max(len(tsf), len(sc), len(rs), len(oci), 0)):
            if tsf and i < len(tsf) and tsf[i] is not None:
                eq.append(tsf[i])
            else:
                s = (sc[i] if i < len(sc) else 0 or 0) + \
                    (rs[i] if i < len(rs) else 0 or 0) + \
                    (oci[i] if i < len(oci) else 0 or 0)
                eq.append(s if s else None)

    rev = income_data.get("revenue",    []) if income_data else []
    np_ = income_data.get("net_profit", []) if income_data else []
    it_ = income_data.get("interest",   []) if income_data else []
    tx_ = income_data.get("tax",        []) if income_data else []

    ocfr=[]; cfd=[]; croa=[]; croe=[]; cfm=[]; eqr=[]
    max_y = max(len(cfo), len(cl), len(ncl), len(ta), len(eq),
                len(rev), len(np_), len(it_), len(tx_), 0)
    for i in range(max_y):
        c_   = cfo[i] if i < len(cfo) else None
        cl_  = cl[i]  if i < len(cl)  else None
        nc_  = ncl[i] if i < len(ncl) else None
        ta_  = ta[i]  if i < len(ta)  else None
        eq_i = eq[i]  if i < len(eq)  else None
        rv_  = rev[i] if i < len(rev) else None
        np_i = np_[i] if i < len(np_) else None
        it_i = it_[i] if i < len(it_) else None
        tx_i = tx_[i] if i < len(tx_) else None
        td = ((cl_ or 0) + (nc_ or 0)) if (cl_ is not None or nc_ is not None) else None
        ocfr.append(round(c_ / cl_, 2)      if (c_ is not None and cl_  not in (None, 0)) else None)
        cfd.append(round(c_ / td, 2)        if (c_ is not None and td   not in (None, 0)) else None)
        croa.append(round(c_ / ta_ * 100, 2)  if (c_ is not None and ta_  not in (None, 0)) else None)
        croe.append(round(c_ / eq_i * 100, 2) if (c_ is not None and eq_i not in (None, 0)) else None)
        cfm.append(round(c_ / rv_ * 100, 2)   if (c_ is not None and rv_  not in (None, 0)) else None)
        ebit = None if (np_i is None and it_i is None and tx_i is None) \
            else (np_i or 0) + (it_i or 0) + (tx_i or 0)
        eqr.append(round(c_ / ebit, 2) if (c_ is not None and ebit not in (None, 0)) else None)

    return {
        "Statement": "Cash Flow",
        "Operating Cash Flow (All Years)": cfo, "Investing Cash Flow (All Years)": cfi,
        "Financing Cash Flow (All Years)": cff, "CapEx (All Years)": capex,
        "Free Cash Flow (All Years)": fcf, "Net Change in Cash (All Years)": netch,
        "Operating Cash Flow Ratio (All Years)": ocfr, "Cash Flow to Debt Ratio (All Years)": cfd,
        "Cash Return on Assets (%) (All Years)": croa, "Cash Return on Equity (%) (All Years)": croe,
        "Cash Flow Margin (%) (All Years)": cfm, "Earnings Quality Ratio (All Years)": eqr,
        "5Y CAGR Operating Cash Flow (%)": cagr_percent(cfo),
        "5Y Avg Operating Cash Flow Ratio": avg(ocfr[:5]),
        "5Y Avg Cash Flow to Debt Ratio": avg(cfd[:5]),
        "5Y Avg Earnings Quality Ratio": avg(eqr[:5]),
    }


def main():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        paths = filedialog.askopenfilenames(filetypes=[("PDF Files", "*.pdf")])
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
        print(f"  Income : {[k for k,v in inc.items() if v]}")
        print(f"  Balance: {[k for k,v in bal.items() if v]}")
        print(f"  Cash   : {[k for k,v in cf.items() if v]}")
        if any(v for v in inc.values()) and income_data is None: income_data = inc
        if any(v for v in bal.values()) and balance_data is None: balance_data = bal
        if any(v for v in cf.values())  and cash_data   is None: cash_data = cf
    if income_data:
        print("\n===== INCOME STATEMENT =====")
        for k, v in analyze_income(income_data).items(): print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
