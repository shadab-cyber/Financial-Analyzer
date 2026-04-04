# dcf_excel_extractor.py  VERSION=3
"""
Extract CFO and CAPEX from a Screener.in Excel export for DCF valuation.

Handles two Screener.in layouts:
  A) Named CF sheet with real values  (older exports)
  B) Named CF sheet with formula refs (newer exports) → reads 'Data Sheet' directly
"""
import re
import pandas as pd
import numpy as np

VERSION = "v3-datasheet-fallback"

# ─── Utilities ────────────────────────────────────────────────────────────────

def fmt(v):
    try:
        if v is None: return 0.0
        f = float(v)
        return 0.0 if pd.isna(f) else f
    except Exception:
        return 0.0


def _is_year(val):
    """True if val looks like a Screener.in fiscal year label."""
    if val is None: return False
    if hasattr(val, 'year'): return True                          # datetime
    if isinstance(val, (int, float)):
        try: return 2000 <= float(val) <= 2100
        except: return False
    if isinstance(val, str):
        s = val.strip()
        patterns = [
            r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-]20\d{2}$',
            r'^20\d{2}[\-\/]\d{2}[\-\/]\d{2}$',   # ISO date
            r'^20\d{2}$',                           # bare year
            r'^20\d{2}[\-\u2013]\d{2,4}$',         # Indian FY: 2023-24
            r'^FY\s*(\d{2}|\d{4})$',               # FY2024 / FY24
        ]
        for p in patterns:
            if re.match(p, s, re.IGNORECASE): return True
        try:
            parsed = pd.to_datetime(s, dayfirst=False)
            return 2000 <= parsed.year <= 2100
        except Exception:
            pass
    return False


def _to_label(val):
    """Convert a year value to 'Mar YYYY' display label."""
    if val is None: return None
    if hasattr(val, 'year'): return val.strftime('Mar %Y')
    if isinstance(val, (int, float)):
        try: return f'Mar {int(val)}'
        except: return str(val)
    if isinstance(val, str):
        s = val.strip()
        # Mar 2024 style
        if re.match(r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-]20\d{2}$', s, re.I):
            return s.replace('-', ' ')
        # Indian FY: 2023-24 → Mar 2024
        m = re.match(r'^(20\d{2})[\-\u2013](\d{2,4})$', s)
        if m: return f'Mar {int(m.group(1)) + 1}'
        # FY2024 → Mar 2024
        m = re.match(r'^FY\s*(20\d{2})$', s, re.I)
        if m: return f'Mar {m.group(1)}'
        # FY24 → Mar 2024
        m = re.match(r'^FY\s*(\d{2})$', s, re.I)
        if m: return f'Mar 20{m.group(1)}'
        try: return pd.to_datetime(s).strftime('Mar %Y')
        except: return s
    return str(val)


def _find_row_label(df, label):
    """Return row index where col-0 matches label (exact then partial)."""
    if df.empty: return None
    ll = label.strip().lower()
    for i in range(df.shape[0]):
        v = df.iloc[i, 0]
        if isinstance(v, str) and v.strip().lower() == ll: return i
    for i in range(df.shape[0]):
        v = df.iloc[i, 0]
        if isinstance(v, str) and ll in v.strip().lower(): return i
    return None


def _find_row_any(df, labels):
    """Return (row_idx, matched_label) for the first label found."""
    for lbl in labels:
        r = _find_row_label(df, lbl)
        if r is not None: return r, lbl
    return None, None


# ─── Sheet loading ────────────────────────────────────────────────────────────

def _load_cf_dataframe(filepath):
    """
    Return a DataFrame whose rows are:
      row N:  [label, yr1_val, yr2_val, ...]
    where row 0 has year headers in col 1+.

    Strategy:
      1. Try named CF sheets → use if they have real (non-formula) year values
      2. Fall back to the 'Data Sheet' hidden tab (newer Screener exports)
    """
    import openpyxl

    CF_SHEET_NAMES = [
        "Cash Flow", "Cash Flow Data", "Cash Flow Statement",
        "Cash Flows", "Cashflow", "Cash flow",
    ]

    # --- Strategy 1: named CF sheet with real values ---
    for name in CF_SHEET_NAMES:
        try:
            # data_only=True so we get cached values, not formula strings
            wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
            if name not in wb.sheetnames:
                continue
            ws = wb[name]
            rows = [list(r) for r in ws.iter_rows(values_only=True)]
            if not rows:
                continue
            df = pd.DataFrame(rows)
            # Check if any row has real year values
            for ri in range(min(len(rows), 50)):
                ycount = sum(1 for c in range(1, df.shape[1]) if _is_year(df.iloc[ri, c]))
                if ycount >= 2:
                    return df          # found a good sheet
        except Exception:
            continue

    # --- Strategy 2: 'Data Sheet' hidden tab (newer exports) ---
    try:
        wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
        if 'Data Sheet' not in wb.sheetnames:
            return pd.DataFrame(), []

        ws = wb['Data Sheet']
        all_rows = [list(r) for r in ws.iter_rows(values_only=True)]

        # Locate the CASH FLOW section
        cf_start = None
        for i, row in enumerate(all_rows):
            first = str(row[0] or '').strip().upper()
            if first.startswith('CASH FLOW') or first == 'CASH FLOW:':
                cf_start = i
                break

        if cf_start is None:
            return pd.DataFrame()

        # Collect rows until 4+ consecutive blank rows or a new major section
        section, blanks = [], 0
        for row in all_rows[cf_start:]:
            if all(v is None for v in row):
                blanks += 1
                if blanks >= 4: break
            else:
                blanks = 0
                section.append(row)

        if not section:
            return pd.DataFrame()

        df = pd.DataFrame(section)
        # The first data row is the section header (e.g. 'CASH FLOW:');
        # the SECOND row should be the year/date header row — verify
        return df

    except Exception:
        return pd.DataFrame()


# ─── Main public API ──────────────────────────────────────────────────────────

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
    # Investing CF as fallback proxy
    "Cash from Investing Activity",
    "Cash from Investing Activities",
    "Net Cash from Investing Activities",
]


def extract_dcf_from_excel(filepath):
    """
    Returns:
    {
        'years':          ['Mar 2019', ...],
        'cfo':            [1234.5, ...],
        'capex':          [300.1, ...],
        'fcf':            [934.4, ...],
        'capex_source':   str,
        'capex_is_proxy': bool,
        'source':         'excel',
        'version':        VERSION,
    }
    Raises ValueError with diagnostic detail on failure.
    """
    import openpyxl

    # --- Detect available sheet names for diagnostics ---
    try:
        wb_info = openpyxl.load_workbook(filepath, read_only=True)
        sheet_names = wb_info.sheetnames
    except Exception as e:
        raise ValueError(f"[{VERSION}] Cannot open Excel file: {e}")

    df = _load_cf_dataframe(filepath)

    if df is None or df.empty:
        raise ValueError(
            f"[{VERSION}] No Cash Flow data found. "
            f"Sheets in file: {sheet_names}. "
            f"Expected a Screener.in export with a 'Cash Flow' or 'Data Sheet' tab."
        )

    # --- Find year header row ---
    year_row_idx = None
    for ri in range(min(df.shape[0], 60)):
        ycount = sum(1 for c in range(1, df.shape[1]) if _is_year(df.iloc[ri, c]))
        if ycount >= 2:
            year_row_idx = ri
            break

    if year_row_idx is None:
        # Show first 5 rows for diagnosis
        sample = []
        for ri in range(min(5, df.shape[0])):
            sample.append(str([df.iloc[ri, c] for c in range(min(6, df.shape[1]))]))
        raise ValueError(
            f"[{VERSION}] Could not detect year columns. "
            f"Sheets: {sheet_names}. "
            f"First rows of CF data: {'; '.join(sample)}"
        )

    # --- Extract years and their column positions ---
    years, year_cols = [], []
    for c in range(1, df.shape[1]):
        val = df.iloc[year_row_idx, c]
        if _is_year(val):
            years.append(_to_label(val))
            year_cols.append(c)

    n = len(years)
    if n == 0:
        raise ValueError(f"[{VERSION}] Year row found at {year_row_idx} but no parseable year values. Sheets: {sheet_names}")

    def row_vals(row_idx):
        return [fmt(df.iloc[row_idx, c]) for c in year_cols]

    # --- CFO ---
    cfo_row, _ = _find_row_any(df, CFO_LABELS)
    if cfo_row is None:
        row0_labels = [str(df.iloc[i, 0]) for i in range(df.shape[0]) if df.iloc[i, 0] is not None]
        raise ValueError(
            f"[{VERSION}] Could not find Cash from Operating Activities. "
            f"Row labels found: {row0_labels}"
        )
    cfo = row_vals(cfo_row)

    # --- CAPEX ---
    capex_row, capex_label = _find_row_any(df, CAPEX_LABELS)
    if capex_row is None:
        row0_labels = [str(df.iloc[i, 0]) for i in range(df.shape[0]) if df.iloc[i, 0] is not None]
        raise ValueError(
            f"[{VERSION}] Could not find CAPEX / Fixed Assets. "
            f"Row labels found: {row0_labels}"
        )
    capex = [abs(fmt(df.iloc[capex_row, c])) for c in year_cols]
    is_proxy = 'Investing' in (capex_label or '')

    # --- FCF ---
    fcf = [round(cfo[i] - capex[i], 2) for i in range(n)]

    # Strip trailing all-zero years
    while len(years) > 2 and cfo[-1] == 0 and capex[-1] == 0:
        years.pop(); cfo.pop(); capex.pop(); fcf.pop()

    return {
        'years':          years,
        'cfo':            [round(v, 2) for v in cfo],
        'capex':          [round(v, 2) for v in capex],
        'fcf':            fcf,
        'capex_source':   capex_label or 'Unknown',
        'capex_is_proxy': is_proxy,
        'source':         'excel',
        'version':        VERSION,
    }


# ─── WACC / EBITDA helpers ────────────────────────────────────────────────────

def _load_bspl_dataframe(filepath):
    """Load Balance Sheet & P&L data, with Data Sheet fallback."""
    import openpyxl
    NAMES = ['Balance Sheet & P&L', 'Profit & Loss', 'Balance Sheet']
    for name in NAMES:
        try:
            wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
            if name not in wb.sheetnames: continue
            ws = wb[name]
            rows = [list(r) for r in ws.iter_rows(values_only=True)]
            df = pd.DataFrame(rows)
            for ri in range(min(len(rows), 50)):
                if sum(1 for c in range(1, df.shape[1]) if _is_year(df.iloc[ri, c])) >= 2:
                    return df
        except Exception:
            continue

    # Data Sheet fallback — find PROFIT & LOSS section
    try:
        wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
        if 'Data Sheet' not in wb.sheetnames: return pd.DataFrame()
        ws = wb['Data Sheet']
        all_rows = [list(r) for r in ws.iter_rows(values_only=True)]
        start = None
        for i, row in enumerate(all_rows):
            if row and row[0] and str(row[0]).strip().upper().startswith('PROFIT'):
                start = i; break
        if start is None: return pd.DataFrame()
        section, blanks = [], 0
        for row in all_rows[start:]:
            if all(v is None for v in row):
                blanks += 1
                if blanks >= 4: break
            else:
                blanks = 0
                section.append(row)
        return pd.DataFrame(section) if section else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def extract_wacc_inputs(filepath):
    result = {'interest': None, 'debt': None, 'tax_rate': None, 'equity': None}
    try:
        df = _load_bspl_dataframe(filepath)
        if df.empty: return result

        def latest(label):
            r = _find_row_label(df, label)
            if r is None: return None
            for c in range(df.shape[1]-1, 0, -1):
                v = fmt(df.iloc[r, c])
                if v != 0: return v
            return None

        result['interest'] = latest('Interest')
        result['debt']     = latest('Borrowings')

        r_tax = _find_row_label(df, 'Tax')
        r_pbt = _find_row_label(df, 'Profit before tax')
        if r_tax is not None and r_pbt is not None:
            for c in range(df.shape[1]-1, 0, -1):
                tax = fmt(df.iloc[r_tax, c])
                pbt = fmt(df.iloc[r_pbt, c])
                if pbt > 0 and tax:
                    result['tax_rate'] = round(tax/pbt*100, 1)
                    break

        eq  = latest('Equity Share Capital') or 0
        res = latest('Reserves') or 0
        if eq + res > 0: result['equity'] = round(eq + res, 2)
    except Exception:
        pass
    return result


def extract_ebitda(filepath):
    try:
        df = _load_bspl_dataframe(filepath)
        if df.empty: return None, None

        yr_row = None
        for ri in range(min(df.shape[0], 50)):
            if sum(1 for c in range(1, df.shape[1]) if _is_year(df.iloc[ri, c])) >= 2:
                yr_row = ri; break

        def latest_val(label):
            r = _find_row_label(df, label)
            if r is None: return None
            for c in range(df.shape[1]-1, 0, -1):
                v = fmt(df.iloc[r, c])
                if v != 0: return v
            return None

        def latest_year():
            if yr_row is None: return None
            for c in range(df.shape[1]-1, 0, -1):
                v = df.iloc[yr_row, c]
                if v is not None and _is_year(v): return _to_label(v)
            return None

        op  = latest_val('Operating Profit') or latest_val('EBIT') or 0
        dep = latest_val('Depreciation') or 0
        ebitda = op + dep
        return (round(ebitda, 2) if ebitda else None), latest_year()
    except Exception:
        return None, None


def extract_latest_fcf(filepath):
    try:
        data = extract_dcf_from_excel(filepath)
        if not data['fcf']: return None, None
        return data['fcf'][-1], data['years'][-1]
    except Exception:
        return None, None
