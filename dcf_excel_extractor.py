# dcf_excel_extractor.py
"""
Extract CFO and CAPEX from a Screener.in Excel export for DCF valuation.
Uses the same sheet-reading logic as Financial_Modelling.py.
"""
import re as _re
import pandas as pd
import numpy as np


def fmt(v):
    if pd.isna(v):
        return 0.0
    return float(v)


def _read_cf_sheet(file):
    """Try all known Screener Cash Flow sheet names."""
    for name in ("Cash Flow Data", "Cash Flow Statement", "Cash Flows",
                 "Cashflow", "Cash flow", "Cash Flow"):
        try:
            return pd.read_excel(file, engine="openpyxl",
                                 sheet_name=name, header=None)
        except Exception:
            continue
    return pd.DataFrame()


def _is_year_value(val):
    """Return True if val looks like a fiscal year label from Screener.in."""
    if pd.isna(val):
        return False
    # datetime / Timestamp object
    if hasattr(val, 'year'):
        return True
    # numeric year e.g. 2023.0
    if isinstance(val, (int, float)) and 2000 <= float(val) <= 2100:
        return True
    if isinstance(val, str):
        s = val.strip()
        # "Mar 2024", "March 2024", "Mar-2024"
        if _re.match(r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-]20\d{2}$', s, _re.I):
            return True
        # "2024-03-31" ISO style
        if _re.match(r'^20\d{2}[\-\/]\d{2}[\-\/]\d{2}$', s):
            return True
        # bare year "2024"
        if _re.match(r'^20\d{2}$', s):
            return True
        # pandas parser as last resort
        try:
            parsed = pd.to_datetime(s, dayfirst=False)
            if 2000 <= parsed.year <= 2100:
                return True
        except Exception:
            pass
    return False


def _year_label(val):
    """Convert any year value to a readable 'Mar YYYY' string."""
    if pd.isna(val):
        return None
    if hasattr(val, 'year'):
        return val.strftime("Mar %Y")
    if isinstance(val, (int, float)) and 2000 <= float(val) <= 2100:
        return f"Mar {int(val)}"
    if isinstance(val, str):
        s = val.strip()
        # Already "Mar 2024" style
        if _re.match(r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-]20\d{2}$', s, _re.I):
            return s.replace('-', ' ')
        try:
            return pd.to_datetime(s).strftime("Mar %Y")
        except Exception:
            return s
    return str(val)


def _find_year_row(df, start_col=1, min_dates=3):
    for row_idx in range(min(df.shape[0], 40)):
        date_count = sum(
            1 for col in range(start_col, df.shape[1])
            if _is_year_value(df.iloc[row_idx, col])
        )
        if date_count >= min_dates:
            return row_idx
    return 0


def find_row(df, label):
    """Find row by exact then partial label match."""
    if df.empty:
        return None
    label_l = label.strip().lower()
    for i in range(df.shape[0]):
        val = df.iloc[i, 0]
        if isinstance(val, str) and val.strip().lower() == label_l:
            return i
    for i in range(df.shape[0]):
        val = df.iloc[i, 0]
        if isinstance(val, str) and label_l in val.strip().lower():
            return i
    return None


# Screener.in uses these exact row labels
CFO_LABELS = [
    "Cash from Operating Activity",
    "Cash from Operating Activities",
    "Net Cash from Operating Activities",
    "Cash Generated from Operations",
    "Net cash from operating",
    "Operating Cash Flow",
    "CFO",
]

CAPEX_LABELS = [
    "Fixed Assets Purchased",
    "Purchase of Fixed Assets",
    "Capital Expenditure",
    "Capex",
    "Purchase of Property, Plant",
    "Addition to Fixed Assets",
    "Net Fixed Assets Purchased",
]


def _find_any_row(df, labels):
    for label in labels:
        r = find_row(df, label)
        if r is not None:
            return r
    return None


def extract_dcf_from_excel(filepath):
    """
    Read a Screener.in Excel file and return:
    {
        'years':  ['Mar 2019', 'Mar 2020', ...],
        'cfo':    [1234.5, 1456.2, ...],
        'capex':  [300.1, 450.0, ...],   # always positive
        'fcf':    [934.4, 1006.2, ...],
        'source': 'excel'
    }
    Raises ValueError with a helpful message if data cannot be found.
    """
    cf_df = _read_cf_sheet(filepath)

    if cf_df.empty:
        try:
            cf_df = pd.read_excel(filepath, engine="openpyxl",
                                  sheet_name="Balance Sheet & P&L", header=None)
        except Exception:
            pass

    if cf_df.empty:
        raise ValueError(
            "No Cash Flow sheet found in this Excel file.\n"
            "Please upload a Screener.in Excel export — go to any company page on "
            "Screener.in → click 'Export to Excel' at the top."
        )

    YEAR_ROW  = _find_year_row(cf_df)
    START_COL = 1

    # Extract year labels
    years = []
    for col in range(START_COL, cf_df.shape[1]):
        val = cf_df.iloc[YEAR_ROW, col]
        if pd.isna(val):
            break
        label = _year_label(val)
        if label:
            years.append(label)
        else:
            break

    n = len(years)
    if n == 0:
        raise ValueError(
            "Could not detect year columns in the Excel file. "
            "Make sure this is a Screener.in export."
        )

    def row_values(row_idx):
        return [fmt(cf_df.iloc[row_idx, START_COL + i]) for i in range(n)]

    # ── Find CFO ──────────────────────────────────────────────────────────
    cfo_row = _find_any_row(cf_df, CFO_LABELS)
    if cfo_row is None:
        raise ValueError(
            "Could not find Cash from Operating Activities in the Excel.\n"
            "Expected one of: " + ", ".join(f'"{l}"' for l in CFO_LABELS[:4]) + "\n"
            "Make sure you are uploading a Screener.in Excel export."
        )
    cfo = row_values(cfo_row)

    # ── Find CAPEX ────────────────────────────────────────────────────────
    capex_row = _find_any_row(cf_df, CAPEX_LABELS)
    if capex_row is None:
        raise ValueError(
            "Could not find Capital Expenditure / Fixed Assets Purchased in the Excel.\n"
            "Expected one of: " + ", ".join(f'"{l}"' for l in CAPEX_LABELS[:4]) + "\n"
            "Make sure you are uploading a Screener.in Excel export."
        )
    capex_raw = row_values(capex_row)
    capex = [abs(v) for v in capex_raw]   # always positive

    # ── Compute FCF ───────────────────────────────────────────────────────
    fcf = [round(cfo[i] - capex[i], 2) for i in range(n)]

    # Strip trailing zeros (years with no data)
    while len(years) > 2 and cfo[-1] == 0 and capex[-1] == 0:
        years.pop(); cfo.pop(); capex.pop(); fcf.pop()

    return {
        "years":  years,
        "cfo":    [round(v, 2) for v in cfo],
        "capex":  [round(v, 2) for v in capex],
        "fcf":    fcf,
        "source": "excel",
    }


def extract_wacc_inputs(filepath):
    result = {
        'interest':  None,
        'debt':      None,
        'tax_rate':  None,
        'equity':    None,
    }
    try:
        bs_df = pd.read_excel(filepath, engine='openpyxl',
                              sheet_name='Balance Sheet & P&L', header=None)
        if bs_df.empty:
            return result

        YEAR_ROW  = _find_year_row(bs_df)
        START_COL = 1

        def latest(label):
            r = find_row(bs_df, label)
            if r is None:
                return None
            for col in range(bs_df.shape[1] - 1, START_COL - 1, -1):
                v = fmt(bs_df.iloc[r, col])
                if v != 0:
                    return v
            return None

        result['interest'] = latest('Interest')
        result['debt']     = latest('Borrowings')

        r_tax = find_row(bs_df, 'Tax')
        r_pbt = find_row(bs_df, 'Profit before tax')
        if r_tax is not None and r_pbt is not None:
            for col in range(bs_df.shape[1] - 1, START_COL - 1, -1):
                tax = fmt(bs_df.iloc[r_tax, col])
                pbt = fmt(bs_df.iloc[r_pbt, col])
                if pbt and tax and pbt > 0:
                    result['tax_rate'] = round(tax / pbt * 100, 1)
                    break

        eq  = latest('Equity Share Capital') or 0
        res = latest('Reserves')             or 0
        if eq + res > 0:
            result['equity'] = round(eq + res, 2)

    except Exception:
        pass

    return result


def extract_ebitda(filepath):
    try:
        bs_df = pd.read_excel(filepath, engine='openpyxl',
                              sheet_name='Balance Sheet & P&L', header=None)
        if bs_df.empty:
            return None, None

        YEAR_ROW  = _find_year_row(bs_df)
        START_COL = 1

        def latest_val(label):
            r = find_row(bs_df, label)
            if r is None:
                return None
            for col in range(bs_df.shape[1] - 1, START_COL - 1, -1):
                v = fmt(bs_df.iloc[r, col])
                if v != 0:
                    return v
            return None

        def latest_year():
            for col in range(bs_df.shape[1] - 1, START_COL - 1, -1):
                val = bs_df.iloc[YEAR_ROW, col]
                if pd.notna(val):
                    return _year_label(val) or str(val)
            return None

        op_profit = latest_val('Operating Profit') or latest_val('EBIT') or 0
        dep       = latest_val('Depreciation') or 0
        ebitda    = op_profit + dep

        return (round(ebitda, 2) if ebitda != 0 else None), latest_year()

    except Exception:
        return None, None


def extract_latest_fcf(filepath):
    try:
        data = extract_dcf_from_excel(filepath)
        if not data['fcf'] or not data['years']:
            return None, None
        return data['fcf'][-1], data['years'][-1]
    except Exception:
        return None, None
