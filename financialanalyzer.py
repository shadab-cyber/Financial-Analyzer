#!/usr/bin/env python3
"""
financialanalyzer.py
Full upgraded merged Income / Balance / CashFlow analyzer (console output).
Year format assumed: Mar-YY, Mar-24, etc. (tolerant to similar headings).
"""

import tkinter as tk
from tkinter import filedialog
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re
import math
from typing import List

# ---------- CONFIG ----------
import shutil as _shutil
_tess = _shutil.which("tesseract")
if _tess:
    pytesseract.pytesseract.tesseract_cmd = _tess
else:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR	esseract.exe"

DEBUG = False  # Set True to print OCR previews and matched lines for debugging

# --------------------
# Utilities
# --------------------
def pdf_to_text(pdf_path: str) -> List[str]:
    """Extract text from PDF — uses native text layer first, OCR only for scanned pages."""
    doc = fitz.open(pdf_path)
    lines = []
    for page in doc:
        # Try native text first (instant for digital PDFs)
        native_text = page.get_text("text")
        if native_text and len(native_text.strip()) > 50:
            for ln in native_text.splitlines():
                ln = ln.strip()
                if ln:
                    lines.append(ln)
        else:
            # Scanned page — OCR at 150 DPI (sufficient for numbers, much faster)
            pix = page.get_pixmap(dpi=150)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img)
            for ln in text.splitlines():
                ln = ln.strip()
                if ln:
                    lines.append(ln)
    return lines


def clean_number(token):
    if token is None:
        return None
    s = str(token)
    s = s.replace("\u200b", "")
    s = s.replace(",", "").replace(" ", "")
    s = s.replace("(", "-").replace(")", "")
    s = re.sub(r"[^0-9.\-]", "", s)
    if s in ("", "-", ".", None):
        return None
    try:
        return float(s)
    except:
        return None


def find_numbers(line: str) -> List[float]:
    toks = re.findall(r"[-+]?[0-9,]+(?:\.[0-9]+)?", line)
    cleaned = [clean_number(t) for t in toks]
    return [c for c in cleaned if c is not None]


def cagr_percent(vals: List[float]):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None
    start, end = vals[-1], vals[0]
    n = len(vals) - 1
    if start <= 0 or end <= 0:
        return None
    return round(((end / start) ** (1 / n) - 1) * 100, 2)


def average_growth(values: List[float]):
    if not values or len(values) < 2:
        return None
    growths = []
    for i in range(len(values) - 1):
        cur = values[i]; prev = values[i+1]
        if cur is None or prev in (None, 0):
            continue
        growths.append(((cur - prev) / prev) * 100)
    return round(sum(growths)/len(growths), 2) if growths else None


def avg(lst: List[float]):
    vals = [x for x in lst if x is not None]
    return round(sum(vals)/len(vals), 2) if vals else None


def safe_div(n, d, scale=2):
    try:
        if n is None or d in (None, 0):
            return None
        return round(n / d, scale)
    except:
        return None


def ratio_yearly(num_list: List[float], den_list: List[float], scale=2) -> List[float]:
    """Compute year-wise ratio for lists (aligns by index; returns list length = max(len(num), len(den)))."""
    max_len = max(len(num_list), len(den_list))
    out = []
    for i in range(max_len):
        n = num_list[i] if i < len(num_list) else None
        d = den_list[i] if i < len(den_list) else None
        if n is None or d in (None, 0):
            out.append(None)
        else:
            out.append(round(n / d, scale))
    return out


# --------------------
# Extractors (robust keywords)
# --------------------
def extract_income_series(lines: List[str]) -> dict:
    patterns = {
        "revenue": [
            "revenue from operations", "total revenue from operations",
            "total operating revenue", "net revenue", "net sales",
            "revenue", "sales", "turnover", "total income from operations",
            "income from operations"
        ],
        "net_profit": [
            "profit/loss for the period", "profit / loss for the period",
            "profit /loss for the period", "profit/loss for the year",
            "profit / loss for the year", "profit for the period",
            "profit/(loss) for the period", "profit / (loss) for the period",
            "profit after tax", "profit after taxation",
            "net profit after tax", "net profit", "pat",
            "profit for the year", "profit for the period",
            "profit/(loss) after tax", "profit / (loss) after tax",
            "net income", "earnings after tax",
            "total comprehensive income for the period",
            "total comprehensive income for the year"
        ],
        "interest": [
            "finance costs", "finance cost", "interest expense",
            "interest and finance charges", "borrowing costs",
            "interest on borrowings", "interest"
        ],
        "tax": [
            "tax expense", "income tax expense", "income tax",
            "total tax expense", "provision for tax",
            "current tax", "deferred tax", "tax"
        ],
        "depreciation": [
            "depreciation and amortisation expense",
            "depreciation & amortization", "depreciation and amortisation",
            "depreciation and amortization", "depreciation", "amortisation",
            "amortization"
        ],
        "cogs": [
            "cost of materials consumed", "cost of goods sold",
            "cost of goods", "raw material consumed", "material cost",
            "cost of revenue"
        ],
        "purchases": [
            "purchase of stock in trade", "purchases of stock in trade",
            "purchase of stock-in-trade", "purchases of traded goods",
            "purchases"
        ],
        "employee_cost": [
            "employee benefit expenses", "employee benefits expense",
            "employee benefits expenses", "staff costs",
            "personnel expenses", "salaries and wages",
            "remuneration and benefits", "employee cost", "staff expenses"
        ],
        "inventory_change": [
            "change in inventories of finished goods",
            "changes in inventories of finished goods",
            "change in inventories", "changes in inventories",
            "(increase)/decrease in inventories"
        ]
    }
    res = {k: [] for k in patterns}
    matched_keys = set()
    for line in lines:
        low = line.lower().replace("\u2013", "-").replace("\u2014", "-")
        for k, kws in patterns.items():
            if k in matched_keys:
                continue  # first match wins
            if any(kw in low for kw in kws):
                nums = find_numbers(line)
                if nums:
                    res[k] = nums
                    matched_keys.add(k)
    return res

def extract_balance_series(lines: List[str]) -> dict:
    patterns = {
        "cash": [
            "cash and cash equivalents", "cash & cash equivalents",
            "cash & bank", "cash and bank balances",
            "cash and bank", "bank balances", "balances with banks"
        ],
        "current_assets": [
            "total current assets", "current assets"
        ],
        "inventory": ["inventories", "inventory", "stock in trade"],
        "current_liabilities": [
            "total current liabilities", "current liabilities"
        ],
        "non_current_liabilities": [
            "total non-current liabilities", "non-current liabilities",
            "non current liabilities", "long-term liabilities"
        ],
        "shareholders_fund": [
            "total shareholders funds", "total shareholders' funds",
            "total equity", "net worth",
            "shareholders funds", "shareholders' equity",
            "total stockholders equity", "equity"
        ],
        "share_capital": [
            "equity share capital", "share capital",
            "paid up share capital", "issued share capital"
        ],
        "reserves": [
            "reserves and surplus", "reserves & surplus",
            "other equity", "retained earnings",
            "other reserves", "free reserves", "reserves"
        ],
        "other_comprehensive_income": [
            "other comprehensive income", "other comprehensive loss",
            "items of other comprehensive income", "oci"
        ],
        "total_assets": [
            "total assets"
        ]
    }
    res = {k: [] for k in patterns}
    matched_keys = set()
    for line in lines:
        low = line.lower().replace("\u2013", "-").replace("\u2014", "-")
        for k, kws in patterns.items():
            if k in matched_keys:
                continue
            if any(kw in low for kw in kws):
                nums = find_numbers(line)
                if nums:
                    res[k] = nums
                    matched_keys.add(k)
    return res

def extract_cashflow_series(lines: List[str]) -> dict:
    patterns = {
        "cfo": [
            "net cash generated from operating activities",
            "net cash from operating activities",
            "net cash used in operating activities",
            "cash flow from operating activities",
            "net cash from operating",
            "cash generated from operations",
            "net cash generated from operations"
        ],
        "cfi": [
            "net cash used in investing activities",
            "net cash from investing activities",
            "cash flow from investing activities",
            "net cash used in investing",
            "cash flows from investing"
        ],
        "cff": [
            "net cash from financing activities",
            "net cash used in financing activities",
            "cash flow from financing activities",
            "net cash from financing",
            "cash flows from financing"
        ],
        "capex": [
            "purchase of property, plant and equipment",
            "purchase of property plant and equipment",
            "purchase of property",
            "acquisition of property",
            "capital expenditure",
            "purchase of fixed assets",
            "additions to fixed assets",
            "purchase of tangible assets"
        ],
        "net_change_cash": [
            "net increase/(decrease) in cash and cash equivalents",
            "net decrease/(increase) in cash and cash equivalents",
            "net increase in cash and cash equivalents",
            "net decrease in cash and cash equivalents",
            "net change in cash and cash equivalents",
            "net increase", "net decrease", "net change in cash"
        ]
    }
    res = {k: [] for k in patterns}
    matched_keys = set()
    for line in lines:
        low = line.lower().replace("–", "-").replace("—", "-")
        for k, kws in patterns.items():
            if k in matched_keys:
                continue
            if any(kw in low for kw in kws):
                nums = find_numbers(line)
                if nums:
                    res[k] = nums
                    matched_keys.add(k)
    return res


# --------------------
# Analyzers
# --------------------
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

    def v(series, i=0): return series[i] if len(series) > i else None

    rev_cur = v(revenue, 0)
    rev_prev = v(revenue, 1)
    revenue_growth = round(((rev_cur - rev_prev) / rev_prev * 100), 2) if (rev_cur not in (None,) and rev_prev not in (None, 0)) else None
    avg_rev_growth = average_growth(revenue)
    rev_cagr = cagr_percent(revenue)

    net_cur = v(net_profit, 0)
    avg_net_growth = average_growth(net_profit)

    # EBITDA
    max_len = max(len(net_profit), len(interest), len(tax), len(depreciation))
    ebitda_series = []
    for i in range(max_len):
        n = net_profit[i] if i < len(net_profit) else None
        it = interest[i] if i < len(interest) else 0
        tx = tax[i] if i < len(tax) else 0
        dp = depreciation[i] if i < len(depreciation) else 0
        if n is not None:
            ebitda_series.append(round((n or 0) + (it or 0) + (tx or 0) + (dp or 0), 2))
    ebitda_cur = ebitda_series[0] if ebitda_series else None
    avg_ebitda_growth = average_growth(ebitda_series)

    # EBIT estimate (Net + Interest + Tax)
    op_profit_cur = None
    if net_cur is not None:
        op_profit_cur = round((net_cur or 0) + (v(interest, 0) or 0) + (v(tax, 0) or 0), 2)
    operating_margin = round(op_profit_cur / rev_cur * 100, 2) if (op_profit_cur not in (None,) and rev_cur not in (None, 0)) else None

    # Gross profit approx
    cogs0 = v(cogs, 0) or 0
    purchases0 = v(purchases, 0) or 0
    emp0 = v(employee_cost, 0) or 0
    invchg0 = v(inventory_change, 0) or 0
    gross_profit = None
    gross_margin = None
    if rev_cur is not None:
        total_costs = (cogs0 + purchases0 + emp0 - invchg0)
        gross_profit = round(rev_cur - total_costs, 2)
        if rev_cur not in (None, 0):
            gross_margin = round(gross_profit / rev_cur * 100, 2)

    cost_to_sales = round(((cogs0 + purchases0 + emp0 - invchg0) / rev_cur) * 100, 2) if rev_cur not in (None, 0) else None
    employee_cost_ratio = round(emp0 / rev_cur * 100, 2) if rev_cur not in (None, 0) else None
    net_profit_margin = round(net_cur / rev_cur * 100, 2) if (net_cur not in (None,) and rev_cur not in (None, 0)) else None

    return {
        "Statement": "Income Statement",
        "Revenue (All Years)": revenue,
        "Revenue (Current Year)": rev_cur,
        "Revenue Growth (%)": revenue_growth,
        "Average Revenue Growth (%)": avg_rev_growth,
        "Revenue CAGR (%)": rev_cagr,

        "Net Profit (All Years)": net_profit,
        "Net Profit (Current Year)": net_cur,
        "Average Net Profit Growth (%)": avg_net_growth,

        "EBITDA (All Years)": ebitda_series,
        "EBITDA (Current Year)": ebitda_cur,
        "Average EBITDA Growth (%)": avg_ebitda_growth,

        "Operating Profit (EBIT, est) (Current Year)": op_profit_cur,
        "Operating Margin (%)": operating_margin,

        "Gross Profit": gross_profit,
        "Gross Margin (%)": gross_margin,
        "Cost to Sales (%)": cost_to_sales,
        "Employee Cost Ratio (%)": employee_cost_ratio,
        "Net Profit Margin (%)": net_profit_margin
    }


def analyze_balance(data: dict, income_data: dict = None) -> dict:
    cash = data.get("cash", [])
    ca = data.get("current_assets", [])
    inv = data.get("inventory", [])
    cl = data.get("current_liabilities", [])
    ncl = data.get("non_current_liabilities", [])
    tsf = data.get("shareholders_fund", [])
    sc = data.get("share_capital", [])
    rs = data.get("reserves", [])
    oci = data.get("other_comprehensive_income", [])
    ta = data.get("total_assets", [])

    # total liabilities
    total_liab = []
    for i in range(max(len(cl), len(ncl))):
        c = cl[i] if i < len(cl) else 0
        n = ncl[i] if i < len(ncl) else 0
        total_liab.append((c or 0) + (n or 0))

    cash_ratio = ratio_yearly(cash, cl)
    current_ratio = ratio_yearly(ca, cl)

    # compute equity_series: prefer tsf, else sc+rs+oci
    max_eq_len = max(len(tsf), len(sc), len(rs), len(oci))
    equity_series = []
    for i in range(max_eq_len):
        if tsf and i < len(tsf) and tsf[i] is not None:
            equity_series.append(tsf[i])
        else:
            sc_v = sc[i] if i < len(sc) else 0
            rs_v = rs[i] if i < len(rs) else 0
            oci_v = oci[i] if i < len(oci) else 0
            s = (sc_v or 0) + (rs_v or 0) + (oci_v or 0)
            equity_series.append(s if s != 0 else None)

    de = ratio_yearly(total_liab, equity_series)
    dr = ratio_yearly(total_liab, ta)
    fl = ratio_yearly(ta, equity_series)
    er = ratio_yearly(equity_series, ta)

    # Debt to Capital year-wise
    dc = []
    for i in range(max(len(total_liab), len(equity_series))):
        tl = total_liab[i] if i < len(total_liab) else 0
        eq_i = equity_series[i] if i < len(equity_series) else None
        denom = None
        if eq_i not in (None, 0) or tl not in (None, 0):
            denom = (tl or 0) + (eq_i or 0)
        dc.append(round(tl / denom, 2) if denom else None)

    # Working capital
    wc_vals = [round(ca[i] - cl[i], 2) if i < len(ca) and i < len(cl) and ca[i] is not None and cl[i] is not None else None for i in range(min(len(ca), len(cl)))]
    wcr = ratio_yearly(ca, cl)

    # Additional ratios needing income_data (Net Profit, Interest, Tax)
    np_series = (income_data.get("net_profit", []) if income_data else [])
    interest_series = (income_data.get("interest", []) if income_data else [])
    tax_series = (income_data.get("tax", []) if income_data else [])

    max_years = max(len(ta), len(equity_series), len(np_series), len(interest_series), len(tax_series), len(cl))
    interest_coverage = []
    roa = []
    roe = []
    roce = []
    equity_ratio = []

    for i in range(max_years):
        np_i = np_series[i] if i < len(np_series) else None
        it_i = interest_series[i] if i < len(interest_series) else None
        tax_i = tax_series[i] if i < len(tax_series) else None
        assets_i = ta[i] if i < len(ta) else None
        curr_liab_i = cl[i] if i < len(cl) else None
        eq_i = equity_series[i] if i < len(equity_series) else None

        # EBIT compute: Net + Interest + Tax
        if np_i is None and it_i is None and tax_i is None:
            ebit_i = None
        else:
            ebit_i = (np_i or 0) + (it_i or 0) + (tax_i or 0)

        # Interest coverage
        interest_coverage.append(round(ebit_i / it_i, 2) if (ebit_i is not None and it_i not in (None, 0)) else None)

        # ROA: NetProfit / TotalAssets *100
        roa.append(round((np_i / assets_i) * 100, 2) if (np_i is not None and assets_i not in (None, 0)) else None)

        # ROE: NetProfit / Equity *100
        roe.append(round((np_i / eq_i) * 100, 2) if (np_i is not None and eq_i not in (None, 0)) else None)

        # ROCE: EBIT / (TotalAssets - CurrentLiabilities) *100
        denom_roce = (assets_i - curr_liab_i) if (assets_i is not None and curr_liab_i is not None) else None
        roce.append(round((ebit_i / denom_roce) * 100, 2) if (ebit_i is not None and denom_roce not in (None, 0)) else None)

        # Equity ratio
        equity_ratio.append(round(eq_i / assets_i, 2) if (eq_i not in (None,) and assets_i not in (None, 0)) else None)

    return {
        "Statement": "Balance Sheet",
        "Cash (All Years)": cash,
        "Current Assets (All Years)": ca,
        "Inventory (All Years)": inv,
        "Total Assets (All Years)": ta,
        "Current Liabilities (All Years)": cl,
        "Non-Current Liabilities (All Years)": ncl,
        "Total Liabilities (All Years)": total_liab,
        "Share Capital (All Years)": sc,
        "Reserves (All Years)": rs,
        "Other Comprehensive Income (All Years)": oci,
        "Total Shareholders Fund (All Years if present)": tsf,
        "Computed Equity (All Years)": equity_series,

        "Cash Ratio (All Years)": cash_ratio,
        "Current Ratio (All Years)": current_ratio,
        "Quick Ratio (All Years)": None,  # needs inventory mapping for quick ratio
        "Working Capital (All Years)": wc_vals,
        "Working Capital Ratio (All Years)": wcr,

        "Debt-to-Equity (All Years)": de,
        "Debt Ratio (All Years)": dr,
        "Financial Leverage (All Years)": fl,
        "Equity Ratio (All Years)": equity_ratio,
        "Debt to Capital (All Years)": dc,

        "Interest Coverage Ratio (All Years)": interest_coverage,
        "Return on Assets (ROA %) (All Years)": roa,
        "Return on Equity (ROE %) (All Years)": roe,
        "Return on Capital Employed (ROCE %) (All Years)": roce,

        "5Y Avg Current Ratio": avg(current_ratio[:5]),
        "5Y Avg D/E": avg(de[:5]),
    }


def analyze_cashflow(data: dict, balance_data: dict = None, income_data: dict = None) -> dict:
    cfo = data.get("cfo", [])
    cfi = data.get("cfi", [])
    cff = data.get("cff", [])
    capex = data.get("capex", [])
    netch = data.get("net_change_cash", [])

    # Free cash flow
    fcf = []
    for i in range(min(len(cfo), len(capex))):
        if cfo[i] is not None and capex[i] is not None:
            fcf.append(round(cfo[i] - capex[i], 2))
        else:
            fcf.append(None)

    # prepare supporting series
    cl = (balance_data.get("current_liabilities", []) if balance_data else [])
    ncl = (balance_data.get("non_current_liabilities", []) if balance_data else [])
    ta = (balance_data.get("total_assets", []) if balance_data else [])
    eq = None
    if balance_data:
        tsf = balance_data.get("shareholders_fund", [])
        sc = balance_data.get("share_capital", [])
        rs = balance_data.get("reserves", [])
        oci = balance_data.get("other_comprehensive_income", [])
        max_eq_y = max(len(tsf), len(sc), len(rs), len(oci))
        eq_list = []
        for i in range(max_eq_y):
            if tsf and i < len(tsf) and tsf[i] is not None:
                eq_list.append(tsf[i])
            else:
                sc_i = sc[i] if i < len(sc) else 0
                rs_i = rs[i] if i < len(rs) else 0
                oci_i = oci[i] if i < len(oci) else 0
                s = (sc_i or 0) + (rs_i or 0) + (oci_i or 0)
                eq_list.append(s if s != 0 else None)
        eq = eq_list
    revenue = (income_data.get("revenue", []) if income_data else [])

    # EBIT from income_data if possible
    net_profit = (income_data.get("net_profit", []) if income_data else [])
    interest = (income_data.get("interest", []) if income_data else [])
    tax = (income_data.get("tax", []) if income_data else [])

    ocf_ratio = []; cf_to_debt = []; croa = []; croe = []; cf_margin = []; earnings_quality = []

    max_years = max(len(cfo), len(cl), len(ncl), len(ta), len(eq) if eq else 0, len(revenue), len(net_profit), len(interest), len(tax))

    for i in range(max_years):
        cfo_val = cfo[i] if i < len(cfo) else None
        curr_liab = cl[i] if i < len(cl) else None
        non_curr_liab = ncl[i] if i < len(ncl) else None
        total_debt = None
        if curr_liab is not None or non_curr_liab is not None:
            total_debt = (curr_liab or 0) + (non_curr_liab or 0)
        total_assets_i = ta[i] if i < len(ta) else None
        equity_i = eq[i] if eq and i < len(eq) else None
        revenue_i = revenue[i] if i < len(revenue) else None

        ocf_ratio.append(round(cfo_val / curr_liab, 2) if (cfo_val is not None and curr_liab not in (None, 0)) else None)
        cf_to_debt.append(round(cfo_val / total_debt, 2) if (cfo_val is not None and total_debt not in (None, 0)) else None)
        croa.append(round((cfo_val / total_assets_i) * 100, 2) if (cfo_val is not None and total_assets_i not in (None, 0)) else None)
        croe.append(round((cfo_val / equity_i) * 100, 2) if (cfo_val is not None and equity_i not in (None, 0)) else None)
        cf_margin.append(round((cfo_val / revenue_i) * 100, 2) if (cfo_val is not None and revenue_i not in (None, 0)) else None)

        # Earnings quality
        np_i = net_profit[i] if i < len(net_profit) else None
        it_i = interest[i] if i < len(interest) else None
        tax_i = tax[i] if i < len(tax) else None
        if np_i is None and it_i is None and tax_i is None:
            earnings_quality.append(None)
        else:
            ebit_i = (np_i or 0) + (it_i or 0) + (tax_i or 0)
            earnings_quality.append(round(cfo_val / ebit_i, 2) if (cfo_val is not None and ebit_i not in (None, 0)) else None)

    return {
        "Statement": "Cash Flow",
        "Operating Cash Flow (All Years)": cfo,
        "Investing Cash Flow (All Years)": cfi,
        "Financing Cash Flow (All Years)": cff,
        "CapEx (All Years)": capex,
        "Free Cash Flow (All Years)": fcf,
        "Net Change in Cash (All Years)": netch,
        "Operating Cash Flow Ratio (All Years)": ocf_ratio,
        "Cash Flow to Debt Ratio (All Years)": cf_to_debt,
        "Cash Return on Assets (CROA %) (All Years)": croa,
        "Cash Return on Equity (CROE %) (All Years)": croe,
        "Cash Flow Margin (%) (All Years)": cf_margin,
        "Earnings Quality Ratio (All Years)": earnings_quality,
        "5Y Avg Operating Cash Flow Ratio": avg(ocf_ratio[:5]),
        "5Y Avg Cash Flow to Debt Ratio": avg(cf_to_debt[:5]),
        "5Y Avg Earnings Quality Ratio": avg(earnings_quality[:5]),
        "5Y CAGR Operating Cash Flow (%)": cagr_percent(cfo),
    }


# --------------------
# Detect type
# --------------------
def detect_type(lines: List[str], filename: str = "") -> str:
    """
    Detect which financial statement type a PDF is.
    Returns 'income', 'balance', 'cash', or None (unknown / multi-statement).
    Priority: filename keywords > strong content signals > scoring.
    """
    txt = " ".join(lines).lower()
    fname = filename.lower().replace("_", " ").replace("-", " ")

    # --- Filename-based detection (strongest signal) ---
    if any(k in fname for k in ["income statement", "profit and loss", "p&l", "pnl",
                                  "profit loss", "statement of profit"]):
        return "income"
    if any(k in fname for k in ["balance sheet", "financial position", "bs_"]):
        return "balance"
    if any(k in fname for k in ["cash flow", "cashflow", "cfs", "statement of cash"]):
        return "cash"

    # Loose filename check (less specific)
    if any(k in fname for k in ["income", "profit", "pl"]):
        return "income"
    if any(k in fname for k in ["balance", "sheet"]):
        return "balance"
    if any(k in fname for k in ["cash", "flow"]):
        return "cash"

    # --- Content scoring (more reliable than single-keyword check) ---
    income_score = 0
    balance_score = 0
    cash_score = 0

    income_keywords = [
        "revenue from operations", "total income", "other income",
        "cost of materials consumed", "employee benefit expense",
        "finance costs", "depreciation and amortisation",
        "profit before tax", "profit after tax", "net profit",
        "earnings per share", "basic eps", "diluted eps"
    ]
    balance_keywords = [
        "total assets", "non-current assets", "current assets",
        "property plant and equipment", "trade receivables",
        "total equity and liabilities", "shareholders fund",
        "total current liabilities", "non-current liabilities",
        "total equity", "retained earnings"
    ]
    cash_keywords = [
        "cash flow from operating", "cash flow from investing",
        "cash flow from financing", "net cash generated",
        "net cash used in", "cash and cash equivalents at end",
        "opening cash balance", "closing cash balance"
    ]

    for kw in income_keywords:
        if kw in txt:
            income_score += 1
    for kw in balance_keywords:
        if kw in txt:
            balance_score += 1
    for kw in cash_keywords:
        if kw in txt:
            cash_score += 1

    # If scores are very close, it's likely a multi-statement PDF — return None
    # so app.py will run all 3 extractors
    scores = {"income": income_score, "balance": balance_score, "cash": cash_score}
    top = max(scores, key=scores.get)
    top_score = scores[top]

    if top_score == 0:
        return None  # no signal at all

    # Check if another type is close — if so treat as multi-statement
    other_scores = [v for k, v in scores.items() if k != top]
    if any(s >= top_score * 0.6 for s in other_scores):
        return None  # multi-statement PDF — let app.py extract all types

    return top

# --------------------
# Main program (reads all files first)
# --------------------
def main():
    print("📂 Select your financial statement PDFs (Income, Balance Sheet, Cash Flow). Year format assumed: Mar-YY etc.")
    root = tk.Tk(); root.withdraw()
    paths = filedialog.askopenfilenames(filetypes=[("PDF Files", "*.pdf")])
    if not paths:
        print("❌ No files selected. Exiting.")
        return

    extracted = []
    for p in paths:
        print(f"\n🔎 OCR extracting: {p}")
        try:
            lines = pdf_to_text(p)
        except Exception as e:
            print(f"⚠️ Failed to OCR {p}: {e}")
            lines = []
        stype = detect_type(lines, p)
        row = {"path": p, "type": stype, "lines": lines, "data": None}
        if stype == "income":
            row["data"] = extract_income_series(lines)
        elif stype == "balance":
            row["data"] = extract_balance_series(lines)
        elif stype == "cash":
            row["data"] = extract_cashflow_series(lines)
        else:
            # try all and pick best
            inc = extract_income_series(lines)
            bal = extract_balance_series(lines)
            cf = extract_cashflow_series(lines)
            counts = (sum(1 for v in inc.values() if v), sum(1 for v in bal.values() if v), sum(1 for v in cf.values() if v))
            best = max(range(3), key=lambda i: counts[i])
            if best == 0:
                row["type"] = "income"; row["data"] = inc
            elif best == 1:
                row["type"] = "balance"; row["data"] = bal
            else:
                row["type"] = "cash"; row["data"] = cf
        extracted.append(row)

    # canonical datasets
    income_data = None; balance_data = None; cash_data = None
    for row in extracted:
        if row["type"] == "income" and income_data is None:
            income_data = row["data"]
        if row["type"] == "balance" and balance_data is None:
            balance_data = row["data"]
        if row["type"] == "cash" and cash_data is None:
            cash_data = row["data"]

    # print outputs (in order of input)
    for row in extracted:
        p = row["path"]; stype = row["type"]
        print(f"\n📄 File: {p}")
        if DEBUG:
            print("--- OCR preview (first 40 lines) ---")
            print("\n".join(row["lines"][:40]))
        if stype == "income":
            res = analyze_income(row["data"])
            print("\n===== INCOME STATEMENT =====")
            for k, v in res.items():
                print(f"{k}: {v}")
        elif stype == "balance":
            res = analyze_balance(row["data"], income_data=income_data)
            print("\n===== BALANCE SHEET =====")
            for k, v in res.items():
                print(f"{k}: {v}")
        elif stype == "cash":
            res = analyze_cashflow(row["data"], balance_data=balance_data, income_data=income_data)
            print("\n===== CASH FLOW STATEMENT =====")
            for k, v in res.items():
                print(f"{k}: {v}")
        else:
            print("⚠️ Unknown / unprocessed file. Preview:")
            print("\n".join(row["lines"][:40]))

    print("\n✅ All files processed.")

if __name__ == "__main__":
    main()
