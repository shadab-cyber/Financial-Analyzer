#!/usr/bin/env python3
"""
financialanalyzer.py - NO TESSERACT VERSION
Uses pdfplumber for text extraction (works on all servers, no OCR needed)
This version works with digital PDFs (not scanned/image PDFs)
"""

import pdfplumber  # ✅ Already in requirements.txt, no system dependencies
import re
from typing import List

DEBUG = False

# --------------------
# Utilities
# --------------------
def pdf_to_text(pdf_path: str) -> List[str]:
    """Extract text from PDF using pdfplumber (no OCR needed)."""
    lines = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    for ln in text.splitlines():
                        ln = ln.strip()
                        if ln:
                            lines.append(ln)
    except Exception as e:
        print(f"Error extracting text: {e}")
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
# Extractors
# --------------------
def extract_income_series(lines: List[str]) -> dict:
    patterns = {
        "revenue": ["revenue from operations", "total operating revenue", "revenue", "net sales", "sales"],
        "net_profit": [
            "profit/loss for the period", "profit / loss for the period",
            "profit for the period", "profit after tax", "net profit", "pat"
        ],
        "interest": ["finance costs", "finance cost", "interest expense", "interest"],
        "tax": ["tax expense", "income tax", "current tax", "tax"],
        "depreciation": ["depreciation & amortization", "depreciation and amortisation", "depreciation"],
        "cogs": ["cost of materials consumed", "cost of goods sold"],
        "purchases": ["purchase of stock in trade", "purchases"],
        "employee_cost": ["employee benefit expenses", "personnel expenses"],
        "inventory_change": ["change in inventories"]
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
        "cash": ["cash and cash equivalents", "cash & bank", "bank balances"],
        "current_assets": ["current assets"],
        "inventory": ["inventories", "inventory"],
        "current_liabilities": ["current liabilities"],
        "non_current_liabilities": ["non-current liabilities", "long-term liabilities"],
        "shareholders_fund": ["total shareholders funds", "total equity", "net worth"],
        "share_capital": ["share capital"],
        "reserves": ["reserves & surplus", "reserves and surplus", "other equity"],
        "other_comprehensive_income": ["other comprehensive income"],
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
        "cfo": ["cash flow from operating", "net cash from operating"],
        "cfi": ["cash flow from investing", "net cash used in investing"],
        "cff": ["cash flow from financing"],
        "capex": ["purchase of property", "capital expenditure"],
        "net_change_cash": ["net increase", "net decrease", "net change in cash"]
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

    cr = []; de = []; dr = []; wc = []; qr = []; ato = []
    max_years = max(len(current_assets), len(current_liabilities), len(non_current_liabilities), 
                    len(total_assets), len(eq) if eq else 0, len(revenue), len(cash), len(inventory))

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

        cr.append(round(ca / cl, 2) if (ca is not None and cl not in (None, 0)) else None)
        de.append(round(tot_debt / e, 2) if (tot_debt is not None and e not in (None, 0)) else None)
        dr.append(round(tot_debt / ta, 2) if (tot_debt is not None and ta not in (None, 0)) else None)
        wc.append(round(ca - cl, 2) if (ca is not None and cl is not None) else None)
        quick_assets = None
        if ca is not None:
            quick_assets = ca - (inv or 0)
        qr.append(round(quick_assets / cl, 2) if (quick_assets is not None and cl not in (None, 0)) else None)
        ato.append(round(rev / ta, 2) if (rev is not None and ta not in (None, 0)) else None)

    roe = []; roa = []
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
        "Current Assets (All Years)": current_assets,
        "Current Liabilities (All Years)": current_liabilities,
        "Current Ratio (All Years)": cr,
        "Debt-to-Equity (All Years)": de,
        "Total Assets (All Years)": total_assets,
        "Working Capital (All Years)": wc,
        "Return on Equity (ROE %) (All Years)": roe if roe else None,
        "Return on Assets (ROA %) (All Years)": roa if roa else None,
        "5Y Avg ROE (%)": avg(roe[:5]) if roe else None,
        "5Y Avg ROA (%)": avg(roa[:5]) if roa else None,
    }


def analyze_cashflow(data: dict, balance_data: dict = None, income_data: dict = None) -> dict:
    cfo = data.get("cfo", [])
    cfi = data.get("cfi", [])
    cff = data.get("cff", [])
    capex = data.get("capex", [])
    netch = data.get("net_change_cash", [])

    fcf = []
    for i in range(max(len(cfo), len(capex))):
        o = cfo[i] if i < len(cfo) else None
        c = capex[i] if i < len(capex) else None
        if o is None and c is None:
            fcf.append(None)
        else:
            fcf.append((o or 0) - abs(c or 0))

    return {
        "Statement": "Cash Flow",
        "Operating Cash Flow (All Years)": cfo,
        "Investing Cash Flow (All Years)": cfi,
        "Financing Cash Flow (All Years)": cff,
        "CapEx (All Years)": capex,
        "Free Cash Flow (All Years)": fcf,
        "Net Change in Cash (All Years)": netch,
        "5Y CAGR Operating Cash Flow (%)": cagr_percent(cfo),
    }


# --------------------
# Detect type
# --------------------
def detect_type(lines: List[str], filename: str = "") -> str:
    txt = " ".join(lines).lower()
    fname = filename.lower()
    
    if any(k in fname for k in ["income", "p&l", "profit"]):
        return "income"
    if any(k in fname for k in ["balance", "sheet", "bs"]):
        return "balance"
    if any(k in fname for k in ["cash", "flow"]):
        return "cash"
        
    if "revenue from operations" in txt or "net profit" in txt:
        return "income"
    if "total assets" in txt or "shareholders" in txt:
        return "balance"
    if "cash flow from operating" in txt:
        return "cash"
        
    return None


if __name__ == "__main__":
    print("⚠️  This file is designed to be imported by the Flask web app.")
    print("    Please run: python app.py")
    print("    Then visit: http://127.0.0.1:5000/")
