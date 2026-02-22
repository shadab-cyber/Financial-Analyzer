#!/usr/bin/env python3
"""
financialanalyzer.py
Full upgraded merged Income / Balance / CashFlow analyzer (console output).
Year format assumed: Mar-YY, Mar-24, etc. (tolerant to similar headings).
"""

# ❌ REMOVED: import tkinter as tk
# ❌ REMOVED: from tkinter import filedialog
# These don't work on Linux servers (Render, AWS, etc.)

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re
import math
import platform
from typing import List

# ---------- CONFIG ----------
# ✅ FIXED: Cross-platform Tesseract path
if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# On Linux (Render, AWS, etc.), tesseract is on PATH automatically

DEBUG = False  # Set True to print OCR previews and matched lines for debugging

# --------------------
# Utilities
# --------------------
def pdf_to_text(pdf_path: str) -> List[str]:
    """OCR each page and return cleaned non-empty lines."""
    doc = fitz.open(pdf_path)
    lines = []
    for page in doc:
        pix = page.get_pixmap(dpi=300)
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
    # NOTE: net_profit pattern includes the user-provided form "Profit/Loss for the Period"
    patterns = {
        "revenue": ["revenue from operations", "total operating revenue", "revenue", "net sales", "sales"],
        "net_profit": [
            "profit/loss for the period",
            "profit / loss for the period",
            "profit /loss for the period",
            "profit/loss for the year",
            "profit / loss for the year",
            "profit for the period",
            "profit/(loss) for the period",
            "profit / (loss) for the period",
            "profit after tax", "net profit", "pat", "profit for the year", "profit for the period"
        ],
        "interest": ["finance costs", "finance cost", "interest expense", "interest"],
        "tax": ["tax expense", "income tax", "current tax", "deferred tax", "tax"],
        "depreciation": ["depreciation & amortization", "depreciation and amortisation", "depreciation", "amortisation"],
        "cogs": ["cost of materials consumed", "cost of goods sold", "cost of goods"],
        "purchases": ["purchase of stock in trade", "purchases of stock in trade", "purchases"],
        "employee_cost": ["employee benefit expenses", "employee benefits expense", "personnel expenses", "employee cost"],
        "inventory_change": ["change in inventories", "changes in inventories"]
    }
    res = {k: [] for k in patterns}
    for line in lines:
        low = line.lower().replace("\u2013", "-")
        for k, kws in patterns.items():
            if any(kw in low for kw in kws):
                nums = find_numbers(line)
                if nums:
                    res[k] = nums
    return res


def extract_balance_series(lines: List[str]) -> dict:
    patterns = {
        "cash": ["cash and cash equivalents", "cash & bank", "cash and bank balances", "bank balances"],
        "current_assets": ["current assets"],
        "inventory": ["inventories", "inventory"],
        "current_liabilities": ["current liabilities", "total current liabilities"],
        "non_current_liabilities": ["non-current liabilities", "non current liabilities", "long-term liabilities"],
        "shareholders_fund": ["total shareholders funds", "total shareholders' funds", "total equity", "net worth", "shareholders funds"],
        "share_capital": ["share capital", "equity share capital"],
        "reserves": ["reserves & surplus", "reserves and surplus", "other equity", "other reserves", "reserves"],
        "other_comprehensive_income": ["other comprehensive income", "oci"],
        "total_assets": ["total assets"]
    }
    res = {k: [] for k in patterns}
    for line in lines:
        low = line.lower().replace("\u2013", "-")
        for k, kws in patterns.items():
            if any(kw in low for kw in kws):
                nums = find_numbers(line)
                if nums:
                    res[k] = nums
    return res


def extract_cashflow_series(lines: List[str]) -> dict:
    patterns = {
        "cfo": ["cash flow from operating", "net cash from operating", "net cash generated from operations", "cash generated from operations"],
        "cfi": ["cash flow from investing", "net cash used in investing"],
        "cff": ["cash flow from financing", "net cash from financing"],
        "capex": ["purchase of property", "capital expenditure", "purchase of fixed assets"],
        "net_change_cash": ["net increase", "net decrease", "net change in cash", "net increase/(decrease) in cash"]
    }
    res = {k: [] for k in patterns}
    for line in lines:
        low = line.lower()
        for k, kws in patterns.items():
            if any(kw in low for kw in kws):
                nums = find_numbers(line)
                if nums:
                    res[k] = nums
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

    ebitda = []
    npm = []
    opm = []
    max_y = max(len(revenue), len(net_profit), len(interest), len(tax), len(depreciation))

    for i in range(max_y):
        np = net_profit[i] if i < len(net_profit) else None
        it = interest[i] if i < len(interest) else None
        tx = tax[i] if i < len(tax) else None
        dep = depreciation[i] if i < len(depreciation) else None
        rev = revenue[i] if i < len(revenue) else None

        if np is None and it is None and tx is None and dep is None:
            ebitda.append(None)
        else:
            eb = (np or 0) + (it or 0) + (tx or 0) + (dep or 0)
            ebitda.append(eb if eb != 0 else None)

        if rev in (None, 0):
            npm.append(None)
            opm.append(None)
        else:
            npm.append(round((np / rev) * 100, 2) if np is not None else None)
            eb_val = ebitda[i]
            opm.append(round((eb_val / rev) * 100, 2) if eb_val is not None else None)

    return {
        "Statement": "Income Statement",
        "Revenue (All Years)": revenue,
        "Net Profit (All Years)": net_profit,
        "EBITDA (All Years)": ebitda,
        "Interest (All Years)": interest,
        "Tax (All Years)": tax,
        "Depreciation (All Years)": depreciation,
        "COGS (All Years)": cogs,
        "Purchases (All Years)": purchases,
        "Employee Cost (All Years)": employee_cost,
        "Inventory Change (All Years)": inventory_change,
        "Net Profit Margin (%) (All Years)": npm,
        "Operating Margin (%) (All Years)": opm,
        "5Y Revenue CAGR (%)": cagr_percent(revenue),
        "5Y Net Profit CAGR (%)": cagr_percent(net_profit),
        "5Y EBITDA CAGR (%)": cagr_percent(ebitda),
        "5Y Avg Net Profit Margin (%)": avg(npm[:5]),
        "5Y Avg Operating Margin (%)": avg(opm[:5])
    }


def analyze_balance(data: dict, income_data: dict = None) -> dict:
    current_assets = data.get("current_assets", [])
    current_liabilities = data.get("current_liabilities", [])
    non_current_liabilities = data.get("non_current_liabilities", [])
    cash = data.get("cash", [])
    inventory = data.get("inventory", [])
    shareholders_fund = data.get("shareholders_fund", [])
    share_capital = data.get("share_capital", [])
    reserves = data.get("reserves", [])
    oci = data.get("other_comprehensive_income", [])
    total_assets = data.get("total_assets", [])

    revenue = (income_data.get("revenue", []) if income_data else [])

    # construct computed equity if not directly available
    eq = []
    max_eq_y = max(len(shareholders_fund), len(share_capital), len(reserves), len(oci))
    for i in range(max_eq_y):
        if shareholders_fund and i < len(shareholders_fund) and shareholders_fund[i] is not None:
            eq.append(shareholders_fund[i])
        else:
            sc = share_capital[i] if i < len(share_capital) else 0
            rs = reserves[i] if i < len(reserves) else 0
            oc = oci[i] if i < len(oci) else 0
            s = (sc or 0) + (rs or 0) + (oc or 0)
            eq.append(s if s != 0 else None)

    # ratio calcs
    cr = []; de = []; dr = []; wc = []; qr = []; ato = []
    max_years = max(len(current_assets), len(current_liabilities), len(non_current_liabilities), len(total_assets), len(eq) if eq else 0, len(revenue), len(cash), len(inventory))

    for i in range(max_years):
        ca = current_assets[i] if i < len(current_assets) else None
        cl = current_liabilities[i] if i < len(current_liabilities) else None
        ncl = non_current_liabilities[i] if i < len(non_current_liabilities) else None
        tot_debt = None
        if cl is not None or ncl is not None:
            tot_debt = (cl or 0) + (ncl or 0)

        c = cash[i] if i < len(cash) else None
        inv = inventory[i] if i < len(inventory) else None
        ta = total_assets[i] if i < len(total_assets) else None
        e = eq[i] if eq and i < len(eq) else None
        rev = revenue[i] if i < len(revenue) else None

        # current ratio
        cr.append(round(ca / cl, 2) if (ca is not None and cl not in (None, 0)) else None)
        # debt to equity
        de.append(round(tot_debt / e, 2) if (tot_debt is not None and e not in (None, 0)) else None)
        # debt ratio
        dr.append(round(tot_debt / ta, 2) if (tot_debt is not None and ta not in (None, 0)) else None)
        # working capital
        wc.append(round(ca - cl, 2) if (ca is not None and cl is not None) else None)
        # quick ratio
        quick_assets = None
        if ca is not None:
            quick_assets = ca - (inv or 0)
        qr.append(round(quick_assets / cl, 2) if (quick_assets is not None and cl not in (None, 0)) else None)
        # asset turnover
        ato.append(round(rev / ta, 2) if (rev is not None and ta not in (None, 0)) else None)

    roe = []
    roa = []
    if income_data:
        net_profit = income_data.get("net_profit", [])
        for i in range(max(len(net_profit), len(eq) if eq else 0, len(total_assets))):
            np = net_profit[i] if i < len(net_profit) else None
            e = eq[i] if eq and i < len(eq) else None
            ta = total_assets[i] if i < len(total_assets) else None
            roe.append(round((np / e) * 100, 2) if (np is not None and e not in (None, 0)) else None)
            roa.append(round((np / ta) * 100, 2) if (np is not None and ta not in (None, 0)) else None)

    return {
        "Statement": "Balance Sheet",
        "5Y Avg Current Ratio": avg(cr[:5]),
        "5Y Avg D/E": avg(de[:5]),
        "Cash (All Years)": cash,
        "Cash Ratio (All Years)": ratio_yearly(cash, current_liabilities, 2),
        "Computed Equity (All Years)": eq if eq else None,
        "Current Assets (All Years)": current_assets,
        "Current Liabilities (All Years)": current_liabilities,
        "Current Ratio (All Years)": cr,
        "Debt Ratio (All Years)": dr,
        "Debt-to-Capital (All Years)": de,
        "Debt-to-Equity (All Years)": de,
        "Equity Ratio (All Years)": [round(e / ta, 2) if (e not in (None, 0) and ta not in (None, 0)) else None 
                                      for e, ta in zip(eq if eq else [], total_assets)],
        "Financial Leverage (All Years)": [round(ta / e, 2) if (e not in (None, 0) and ta not in (None, 0)) else None
                                            for e, ta in zip(eq if eq else [], total_assets)],
        "Interest Coverage Ratio (All Years)": (ratio_yearly((income_data.get("net_profit", []) if income_data else []), 
                                                              (income_data.get("interest", []) if income_data else []), 2)),
        "Inventory (All Years)": inventory,
        "Non-Current Liabilities (All Years)": non_current_liabilities,
        "Other Comprehensive Income (All Years)": oci,
        "Quick Ratio (All Years)": qr,
        "Reserves (All Years)": reserves,
        "Return on Assets (ROA %) (All Years)": roa if roa else None,
        "Return on Equity (ROE %) (All Years)": roe if roe else None,
        "Return on Capital Employed (ROCE %) (All Years)": [
            round(((np or 0) + (it or 0)) / ((ta or 0) - (cl or 0)) * 100, 2) 
            if (income_data and i < len(income_data.get("net_profit", [])) and i < len(income_data.get("interest", [])) 
                and i < len(total_assets) and i < len(current_liabilities) 
                and (((ta or 0) - (cl or 0)) != 0)) 
            else None
            for i, (np, it, ta, cl) in enumerate(zip(
                (income_data.get("net_profit", []) if income_data else []),
                (income_data.get("interest", []) if income_data else []),
                total_assets, current_liabilities
            ))
        ],
        "Share Capital (All Years)": share_capital,
        "Total Assets (All Years)": total_assets,
        "Total Shareholders Fund (All Years if present)": shareholders_fund,
        "Working Capital (All Years)": wc,
        "Working Capital Ratio (All Years)": [round(w / rev, 2) if (w is not None and rev not in (None, 0)) else None 
                                               for w, rev in zip(wc, revenue if revenue else [])],
        "5Y Avg ROE (%)": avg(roe[:5]) if roe else None,
        "5Y Avg ROA (%)": avg(roa[:5]) if roa else None,
    }


def analyze_cashflow(data: dict, balance_data: dict = None, income_data: dict = None) -> dict:
    cfo = data.get("cfo", [])
    cfi = data.get("cfi", [])
    cff = data.get("cff", [])
    capex = data.get("capex", [])
    netch = data.get("net_change_cash", [])

    # free cash flow
    fcf = []
    for i in range(max(len(cfo), len(capex))):
        o = cfo[i] if i < len(cfo) else None
        c = capex[i] if i < len(capex) else None
        if o is None and c is None:
            fcf.append(None)
        else:
            fcf.append((o or 0) - abs(c or 0))

    # from balance
    cl = (balance_data.get("current_liabilities", []) if balance_data else [])
    ncl = (balance_data.get("non_current_liabilities", []) if balance_data else [])
    ta = (balance_data.get("total_assets", []) if balance_data else [])

    # equity
    eq = []
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
    txt = " ".join(lines).lower()
    fname = filename.lower()
    if any(k in fname for k in ["income", "p&l", "pnl", "profit", "pl"]):
        return "income"
    if any(k in fname for k in ["balance", "sheet", "bs", "balance-sheet"]):
        return "balance"
    if any(k in fname for k in ["cash", "flow", "cfs", "cashflow"]):
        return "cash"
    if "cash flow from operating" in txt or "net cash from operating" in txt:
        return "cash"
    if "profit for the period" in txt or "profit/(loss) for the period" in txt or "profit/loss for the period" in txt or "revenue from operations" in txt:
        return "income"
    if "total assets" in txt or "total shareholders" in txt or "current liabilities" in txt:
        return "balance"
    # fallback scoring
    score = 0
    for k in ["revenue", "net profit", "ebitda"]:
        if k in txt:
            score += 1
    for k in ["total assets", "shareholders", "current liabilities"]:
        if k in txt:
            score -= 1
    for k in ["cash flow", "net cash"]:
        if k in txt:
            score += 0.5
    if score >= 1:
        return "income"
    if score <= -1:
        return "balance"
    return None


# ❌ REMOVED: main() function with tkinter GUI
# This file is now used ONLY as a library for the Flask web app
# The Flask routes in app.py handle file uploads via web interface

if __name__ == "__main__":
    print("⚠️  This file is designed to be imported by the Flask web app.")
    print("    Please run: python app.py")
    print("    Then visit: http://127.0.0.1:5000/")
