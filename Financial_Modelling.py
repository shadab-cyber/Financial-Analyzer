import pandas as pd
import numpy as np
try:
    from dateutil.relativedelta import relativedelta
    _HAS_RELATIVEDELTA = True
except ImportError:
    _HAS_RELATIVEDELTA = False

# ===============================
# HELPERS
# ===============================
def fmt(v):
    if pd.isna(v):
        return 0
    return float(v)

def find_row(df, label):
    """Find a row by label — exact match first, then partial (handles Screener variations)."""
    if df.empty:
        return None
    label_l = label.strip().lower()
    # Pass 1: exact match
    for i in range(df.shape[0]):
        val = df.iloc[i, 0]
        if isinstance(val, str) and val.strip().lower() == label_l:
            return i
    # Pass 2: substring match (minor label differences across Screener versions)
    for i in range(df.shape[0]):
        val = df.iloc[i, 0]
        if isinstance(val, str) and label_l in val.strip().lower():
            return i
    return None


def _read_cf_sheet(file):
    """Try multiple possible Cash Flow sheet names from Screener exports."""
    for name in ("Cash Flow Data", "Cash Flow Statement", "Cash Flows",
                 "Cashflow", "Cash flow", "Cash Flow"):
        try:
            return pd.read_excel(file, engine="openpyxl",
                                 sheet_name=name, header=None)
        except Exception:
            continue
    return pd.DataFrame()


# ===============================
# MAIN FUNCTION
# ===============================
def run_historical_fs(file):

    # ===============================
    # READ SHEETS
    # ===============================
    bs_df = pd.read_excel(file, engine="openpyxl", sheet_name="Balance Sheet & P&L", header=None)
    cf_df = _read_cf_sheet(file)

    # ===============================
    # YEARS (B16 onward)
    # ===============================
    YEAR_ROW = 15
    START_COL = 1

    years = []
    col = START_COL
    while col < bs_df.shape[1]:
        val = bs_df.iloc[YEAR_ROW, col]
        if pd.isna(val):
            break
        years.append(pd.to_datetime(val).strftime("%b-%y"))
        col += 1

    n = len(years)

    # ======================================================
    # 🔴 INCOME STATEMENT (UNCHANGED – FULL 26 ITEMS)
    # ======================================================
    def val(label):
        r = find_row(bs_df, label)
        return [fmt(bs_df.iloc[r, START_COL + i]) if r is not None else 0 for i in range(n)]

    sales = val("Sales")
    sales_growth = [""] + [round(((sales[i]/sales[i-1]) - 1)*100, 2) for i in range(1, n)]

    raw = val("Raw Material Cost")
    power = val("Power and Fuel")
    other_mfr = val("Other Mfr. Exp")
    emp = val("Employee Cost")
    change_inv = val("Change in Inventory")

    cogs = [raw[i] + power[i] + other_mfr[i] + emp[i] - change_inv[i] for i in range(n)]
    cogs_pct = [round((cogs[i]/sales[i])*100, 2) if sales[i] else 0 for i in range(n)]

    gross_profit = [sales[i] - cogs[i] for i in range(n)]
    gross_margin = [round((gross_profit[i]/sales[i])*100, 2) if sales[i] else 0 for i in range(n)]

    sga = [val("Selling and admin")[i] + val("Other Expenses")[i] for i in range(n)]
    sga_pct = [round((sga[i]/sales[i])*100, 2) if sales[i] else 0 for i in range(n)]

    ebitda = [gross_profit[i] - sga[i] for i in range(n)]
    ebitda_pct = [round((ebitda[i]/sales[i])*100, 2) if sales[i] else 0 for i in range(n)]

    interest = val("Interest")
    interest_pct = [round((interest[i]/sales[i])*100, 2) if sales[i] else 0 for i in range(n)]

    depreciation = val("Depreciation")
    depreciation_pct = [round((depreciation[i]/sales[i])*100, 2) if sales[i] else 0 for i in range(n)]

    ebt = [ebitda[i] - interest[i] - depreciation[i] for i in range(n)]
    ebt_pct = [round((ebt[i]/sales[i])*100, 2) if sales[i] else 0 for i in range(n)]

    tax = val("Tax")
    effective_tax = [round((tax[i]/ebt[i])*100, 2) if ebt[i] else 0 for i in range(n)]

    net_profit = [ebt[i] - tax[i] for i in range(n)]
    net_margin = [round((net_profit[i]/sales[i])*100, 2) if sales[i] else 0 for i in range(n)]

    equity_shares = val("Adjusted Equity Shares in Cr")
    eps = [round(net_profit[i]/equity_shares[i], 2) if equity_shares[i] else 0 for i in range(n)]
    eps_growth = [""] + [round(((eps[i]/eps[i-1]) - 1)*100, 2) if eps[i-1] else 0 for i in range(1, n)]

    dividend_amt = val("Dividend Amount")
    dividend_per_share = [round(dividend_amt[i]/equity_shares[i], 2) if equity_shares[i] else 0 for i in range(n)]
    payout_ratio = [round((dividend_per_share[i]/eps[i])*100, 2) if eps[i] else 0 for i in range(n)]
    # FIX 1: retained earnings % = (EPS - DPS) / EPS * 100
    retained_earnings_pct = [
        round(((eps[i] - dividend_per_share[i]) / eps[i]) * 100, 2)
        if eps[i] and eps[i] > dividend_per_share[i]
        else 0
        for i in range(n)
    ]
    retained_earnings = retained_earnings_pct  # alias for downstream compatibility

    income_statement = [
        ("Sales", sales), ("Sales Growth %", sales_growth),
        ("COGS", cogs), ("COGS % of Sales", cogs_pct),
        ("Gross Profit", gross_profit), ("Gross Margin %", gross_margin),
        ("Selling & General Expenses", sga), ("S&G % Sales", sga_pct),
        ("EBITDA", ebitda), ("EBITDA % Sales", ebitda_pct),
        ("Interest", interest), ("Interest % Sales", interest_pct),
        ("Depreciation", depreciation), ("Depreciation % Sales", depreciation_pct),
        ("EBT", ebt), ("EBT % Sales", ebt_pct),
        ("Tax", tax), ("Effective Tax Rate", effective_tax),
        ("Net Profit", net_profit), ("Net Margins", net_margin),
        ("No. of Equity Shares", equity_shares),
        ("EPS", eps), ("EPS Growth %", eps_growth),
        ("Dividend per Share", dividend_per_share),
        ("Dividend Payout %", payout_ratio),
        ("Retained Earnings %", retained_earnings_pct),
    ]

    # ======================================================
    # 🟢 BALANCE SHEET (UNCHANGED)
    # ======================================================
    def bs(label):
        r = find_row(bs_df, label)
        return [fmt(bs_df.iloc[r, START_COL + i]) if r is not None else 0 for i in range(n)]

    equity = bs("Equity Share Capital")
    reserves = bs("Reserves")
    borrowings = bs("Borrowings")
    other_liab = bs("Other Liabilities")
    total_liab = [equity[i] + reserves[i] + borrowings[i] + other_liab[i] for i in range(n)]

    net_block = bs("Net Block")
    cwip = bs("Capital Work in Progress")
    investments = bs("Investments")
    other_assets = bs("Other Assets")
    total_nca = [net_block[i] + cwip[i] + investments[i] + other_assets[i] for i in range(n)]

    receivables = bs("Receivables")
    inventory = bs("Inventory")
    cash_bank = bs("Cash & Bank")
    total_ca = [receivables[i] + inventory[i] + cash_bank[i] for i in range(n)]

    total_assets = [total_ca[i] + total_nca[i] for i in range(n)]

    balance_sheet = [
        ("Equity Share Capital", equity),
        ("Reserves", reserves),
        ("Borrowings", borrowings),
        ("Other Liabilities", other_liab),
        ("Total Liabilities", total_liab),
        ("", [""]*n),
        ("Fixed Asset Net Block", net_block),
        ("Capital Work in Progress", cwip),
        ("Investments", investments),
        ("Other Assets", other_assets),
        ("Total Non Current Assets", total_nca),
        ("", [""]*n),
        ("Receivables", receivables),
        ("Inventory", inventory),
        ("Cash & Bank", cash_bank),
        ("Total Current Assets", total_ca),
        ("", [""]*n),
        ("Total Assets", total_assets),
    ]

    # ======================================================
    # 🔵 CASH FLOW STATEMENT (NEW)
    # ======================================================
    def cf(label):
        r = find_row(cf_df, label)
        return [fmt(cf_df.iloc[r, START_COL + i]) if r is not None else 0 for i in range(n)]

    profit_ops = cf("Profit from operations")
    recv = cf("Receivables")
    inv = cf("Inventory")
    pay = cf("Payables")
    loans = cf("Loans Advances")
    other_wc = cf("Other WC items")
    wc_change = cf("Working capital changes")
    direct_tax = cf("Direct taxes")

    cfo = [profit_ops[i] + recv[i] + inv[i] + pay[i] + loans[i] + other_wc[i] + wc_change[i] + direct_tax[i] for i in range(n)]

    fa_p = cf("Fixed assets purchased")
    fa_s = cf("Fixed assets sold")
    inv_p = cf("Investments purchased")
    inv_s = cf("Investments sold")
    int_r = cf("Interest received")
    div_r = cf("Dividends received")
    grp = cf("Investment in group cos")
    red = cf("Redemp n Canc of Shares")
    acq = cf("Acquisition of companies")
    icd = cf("Inter corporate deposits")
    other_inv = cf("Other investing items")

    cfi = [fa_p[i] + fa_s[i] + inv_p[i] + inv_s[i] + int_r[i] + div_r[i] + grp[i] + red[i] + acq[i] + icd[i] + other_inv[i] for i in range(n)]

    proc_sh = cf("Proceeds from shares")
    red_deb = cf("Redemption of debentures")
    proc_borr = cf("Proceeds from borrowings")
    repay_borr = cf("Repayment of borrowings")
    int_paid = cf("Interest paid fin")
    div_paid = cf("Dividends paid")
    fin_liab = cf("Financial liabilities")
    other_fin = cf("Other financing items")

    cff = [proc_sh[i] + red_deb[i] + proc_borr[i] + repay_borr[i] + int_paid[i] + div_paid[i] + fin_liab[i] + other_fin[i] for i in range(n)]

    net_cf = [cfo[i] + cfi[i] + cff[i] for i in range(n)]

    cash_flow = [
        ("OPERATING ACTIVITIES", [""]*n),
        ("Profit from Operations", profit_ops),
        ("Receivables", recv),
        ("Inventory", inv),
        ("Payables", pay),
        ("Loans Advances", loans),
        ("Other WC items", other_wc),
        ("Working Capital Changes", wc_change),
        ("Direct Taxes", direct_tax),
        ("Cash from Operating Activities", cfo),
        ("", [""]*n),
        ("INVESTING ACTIVITIES", [""]*n),
        ("Fixed Assets Purchased", fa_p),
        ("Fixed Assets Sold", fa_s),
        ("Investments Purchased", inv_p),
        ("Investments Sold", inv_s),
        ("Interest Received", int_r),
        ("Dividends Received", div_r),
        ("Investment in Group Cos", grp),
        ("Redemp n Canc of Shares", red),
        ("Acquisition of Companies", acq),
        ("Inter Corporate Deposits", icd),
        ("Other Investing Items", other_inv),
        ("Cash from Investing Activities", cfi),
        ("", [""]*n),
        ("FINANCING ACTIVITIES", [""]*n),
        ("Proceeds from Shares", proc_sh),
        ("Redemption of Debentures", red_deb),
        ("Proceeds from Borrowings", proc_borr),
        ("Repayment of Borrowings", repay_borr),
        ("Interest Paid", int_paid),
        ("Dividends Paid", div_paid),
        ("Financial Liabilities", fin_liab),
        ("Other Financing Items", other_fin),
        ("Cash from Financing Activities", cff),
        ("", [""]*n),
        ("NET CASH FLOW", net_cf),
    ]

    return {
        "years": years,
        "income_statement": income_statement,
        "balance_sheet": balance_sheet,
        "cash_flow": cash_flow
    }


# ===============================
# RATIO ANALYSIS FUNCTION
# ===============================
def run_ratio_analysis(file):
    """
    Calculate all financial ratios from the Excel file
    Returns structured ratio analysis data
    """
    
    # First, get Historical FS data (we'll reuse many calculations)
    historical_data = run_historical_fs(file)
    years = historical_data["years"]
    n = len(years)
    
    # Read Balance Sheet for CFO data
    bs_df = pd.read_excel(file, engine="openpyxl", sheet_name="Balance Sheet & P&L", header=None)
    YEAR_ROW = 15
    START_COL = 1
    
    # Helper function to get values from historical data
    def get_hist_values(label):
        for item_label, values in historical_data["income_statement"]:
            if item_label == label:
                return values
        for item_label, values in historical_data["balance_sheet"]:
            if item_label == label:
                return values
        return [0] * n
    
    # Get all required values from Historical FS
    sales = get_hist_values("Sales")
    sales_growth = get_hist_values("Sales Growth %")
    ebitda = get_hist_values("EBITDA")
    ebt = get_hist_values("EBT")
    net_profit = get_hist_values("Net Profit")
    dividend_per_share = get_hist_values("Dividend per Share")
    gross_margin = get_hist_values("Gross Margin %")
    ebitda_margin = get_hist_values("EBITDA % Sales")
    ebt_margin = get_hist_values("EBT % Sales")
    net_margin = get_hist_values("Net Margins")
    sga_pct = get_hist_values("S&G % Sales")
    depreciation = get_hist_values("Depreciation")
    depreciation_pct = get_hist_values("Depreciation % Sales")
    interest = get_hist_values("Interest")
    retained_earnings = get_hist_values("Retained Earnings")
    
    # Balance Sheet items
    equity = get_hist_values("Equity Share Capital")
    reserves = get_hist_values("Reserves")
    borrowings = get_hist_values("Borrowings")
    other_liab = get_hist_values("Other Liabilities")
    receivables = get_hist_values("Receivables")
    inventory = get_hist_values("Inventory")
    net_block = get_hist_values("Fixed Asset Net Block")
    total_assets = get_hist_values("Total Assets")
    
    # Get CFO from Balance Sheet & P&L
    cfo_row = find_row(bs_df, "Cash from Operating Activity")
    cfo = [fmt(bs_df.iloc[cfo_row, START_COL + i]) if cfo_row is not None else 0 for i in range(n)]
    
    # ======================================================
    # CALCULATE ALL RATIOS
    # ======================================================
    
    # 1. GROWTH RATIOS
    # Sales Growth - already calculated in Historical FS
    
    # EBITDA Growth
    ebitda_growth = [""] + [
        round(((ebitda[i]/ebitda[i-1]) - 1)*100, 2) if ebitda[i-1] and ebitda[i-1] != 0 else 0 
        for i in range(1, n)
    ]
    
    # EBT Growth
    ebt_growth = [""] + [
        round(((ebt[i]/ebt[i-1]) - 1)*100, 2) if ebt[i-1] and ebt[i-1] != 0 else 0
        for i in range(1, n)
    ]
    
    # Net Profit Growth
    net_profit_growth = [""] + [
        round(((net_profit[i]/net_profit[i-1]) - 1)*100, 2) if net_profit[i-1] and net_profit[i-1] != 0 else 0
        for i in range(1, n)
    ]
    
    # Dividend Growth
    dividend_growth = [""] + [
        round(((dividend_per_share[i]/dividend_per_share[i-1]) - 1)*100, 2) if dividend_per_share[i-1] and dividend_per_share[i-1] != 0 else 0
        for i in range(1, n)
    ]
    
    # 2. PROFITABILITY MARGINS (already calculated in Historical FS)
    # Gross Margin, EBITDA Margin, EBT Margin, Net Margin
    
    # EBIT Margin = (EBITDA - Depreciation) / Sales
    ebit = [ebitda[i] - depreciation[i] for i in range(n)]
    ebit_margin = [round((ebit[i]/sales[i])*100, 2) if sales[i] else 0 for i in range(n)]
    
    # 3. EXPENSE RATIOS (already in Historical FS)
    # SalesExpenses%Sales, Depreciation%Sales
    
    # 4. RETURN & COVERAGE RATIOS
    # FIX 2: ROCE = EBIT / Capital Employed (CFA standard)
    # Capital Employed = Total Assets - Current Liabilities (Other Liabilities proxy)
    total_assets_ra = get_hist_values("Total Assets")
    other_liab_ra   = get_hist_values("Other Liabilities")
    capital_employed = [
        total_assets_ra[i] - other_liab_ra[i] for i in range(n)
    ]
    roce = [
        round((ebit[i] / capital_employed[i]) * 100, 2)
        if capital_employed[i] else 0
        for i in range(n)
    ]
    
    # Retained Earnings% - already in Historical FS
    
    # Return on Equity% = Net Profit / (Equity + Reserves)
    roe = [
        round((net_profit[i]/(equity[i] + reserves[i]))*100, 2)
        if (equity[i] + reserves[i]) else 0
        for i in range(n)
    ]
    
    # FIX 3: SSGR = (Retained Earnings% / 100) × ROE%
    # retained_earnings[i] is now e.g. 60.0 (meaning 60%)
    # ROE is also in % — result is in %
    retained_earnings_ra = get_hist_values("Retained Earnings %")
    self_sustained_growth = [
        round((retained_earnings_ra[i] / 100) * roe[i], 2)
        if isinstance(retained_earnings_ra[i], (int, float)) and isinstance(roe[i], (int, float))
        else 0
        for i in range(n)
    ]
    
    # Interest Coverage Ratio = (EBITDA - Depreciation) / Interest
    interest_coverage = [
        round(ebit[i]/interest[i], 1) if interest[i] else 0
        for i in range(n)
    ]
    
    # 5. TURNOVER RATIOS
    # Debt Turnover Ratio = Sales / Receivables
    debt_turnover = [
        round(sales[i]/receivables[i], 1) if receivables[i] else 0
        for i in range(n)
    ]
    
    # Creditor Turnover Ratio = Sales / Other Liabilities
    creditor_turnover = [
        round(sales[i]/other_liab[i], 1) if other_liab[i] else 0
        for i in range(n)
    ]
    
    # Inventory Turnover = Sales / Inventory
    inventory_turnover = [
        round(sales[i]/inventory[i], 1) if inventory[i] else 0
        for i in range(n)
    ]
    
    # Fixed Asset Turnover = Sales / Fixed Asset Net Block
    fixed_asset_turnover = [
        round(sales[i]/net_block[i], 1) if net_block[i] else 0
        for i in range(n)
    ]
    
    # Capital Turnover Ratio = Sales / (Equity + Reserves)
    capital_turnover = [
        round(sales[i]/(equity[i] + reserves[i]), 1) if (equity[i] + reserves[i]) else 0
        for i in range(n)
    ]
    
    # 6. DAYS RATIOS
    # Debtor Days = 365 / Debt Turnover Ratio
    debtor_days = [
        round(365/debt_turnover[i], 1) if debt_turnover[i] else 0
        for i in range(n)
    ]
    
    # Payable Days = 365 / Creditor Turnover Ratio
    payable_days = [
        round(365/creditor_turnover[i], 1) if creditor_turnover[i] else 0
        for i in range(n)
    ]
    
    # Inventory Days = 365 / Inventory Turnover
    inventory_days = [
        round(365/inventory_turnover[i], 1) if inventory_turnover[i] else 0
        for i in range(n)
    ]
    
    # Cash Conversion Cycle = (Debtor Days + Inventory Days) - Payable Days
    cash_conversion_cycle = [
        round((debtor_days[i] + inventory_days[i]) - payable_days[i], 1)
        for i in range(n)
    ]
    
    # 7. CASH FLOW RATIOS
    # CFO/Sales
    cfo_sales = [
        round((cfo[i]/sales[i])*100, 2) if sales[i] else 0
        for i in range(n)
    ]
    
    # CFO/Total Assets
    cfo_total_assets = [
        round((cfo[i]/total_assets[i])*100, 2) if total_assets[i] else 0
        for i in range(n)
    ]
    
    # CFO/Total Debt
    cfo_total_debt = [
        round((cfo[i]/borrowings[i])*100, 2) if borrowings[i] else 0
        for i in range(n)
    ]
    
    # ======================================================
    # BUILD RATIO ANALYSIS STRUCTURE (WITH GAPS)
    # ======================================================
    ratio_analysis = [
        # GROWTH RATIOS
        ("Sales Growth", sales_growth),
        ("EBITDA Growth", ebitda_growth),
        ("EBT Growth", ebt_growth),
        ("Net Profit Growth", net_profit_growth),
        ("Dividend Growth", dividend_growth),
        ("", [""]*n),  # GAP
        
        # PROFITABILITY MARGINS
        ("Gross Margin", gross_margin),
        ("EBITDA Margin", ebitda_margin),
        ("EBIT Margin", ebit_margin),
        ("EBT Margin", ebt_margin),
        ("Net Profit Margin", net_margin),
        ("", [""]*n),  # GAP
        
        # EXPENSE RATIOS
        ("SalesExpenses%Sales", sga_pct),
        ("Depreciation%Sales", depreciation_pct),
        ("", [""]*n),  # GAP
        
        # RETURN & COVERAGE RATIOS
        ("ROCE", roce),
        ("Retained Earnings %", retained_earnings_ra),
        ("Return on Equity%", roe),
        ("Self Sustained Growth Rate", self_sustained_growth),
        ("Interest Coverage Ratio", interest_coverage),
        ("", [""]*n),  # GAP
        
        # TURNOVER RATIOS
        ("Debt Turnover Ratio", debt_turnover),
        ("Creditor Turnover Ratio", creditor_turnover),
        ("Inventery Turnover", inventory_turnover),
        ("Fixed Asset Turnover", fixed_asset_turnover),
        ("Capital Turnover Ratio", capital_turnover),
        ("", [""]*n),  # GAP
        
        # DAYS RATIOS
        ("Debtor Days", debtor_days),
        ("Payable Days", payable_days),
        ("Inventory Days", inventory_days),
        ("Cash Conversion Cycle", cash_conversion_cycle),
        ("", [""]*n),  # GAP
        
        # CASH FLOW RATIOS
        ("CFO/Sales", cfo_sales),
        ("CFO/Total Assets", cfo_total_assets),
        ("CFO/Total Debt", cfo_total_debt),
    ]
    
    return {
        "years": years,
        "ratio_analysis": ratio_analysis
    }


# ===============================
# COMMON SIZE STATEMENT FUNCTION
# ===============================
def run_common_size_statement(file):
    """
    Calculate Common Size Income Statement and Balance Sheet
    Returns structured common size data
    """
    
    # Read Balance Sheet & P&L for data
    bs_df = pd.read_excel(file, engine="openpyxl", sheet_name="Balance Sheet & P&L", header=None)
    YEAR_ROW = 15
    START_COL = 1
    
    # Get years
    years = []
    col = START_COL
    while col < bs_df.shape[1]:
        val = bs_df.iloc[YEAR_ROW, col]
        if pd.isna(val):
            break
        years.append(pd.to_datetime(val).strftime("%b-%y"))
        col += 1
    
    n = len(years)
    
    # Helper function to get values from Balance Sheet & P&L
    def bs_val(label):
        r = find_row(bs_df, label)
        return [fmt(bs_df.iloc[r, START_COL + i]) if r is not None else 0 for i in range(n)]
    
    # ======================================================
    # COMMON SIZE INCOME STATEMENT
    # ======================================================
    
    # Get base values
    sales = bs_val("Sales")
    
    # Calculate common size percentages (all relative to Sales)
    raw_material = bs_val("Raw Material Cost")
    raw_material_pct = [round((raw_material[i]/sales[i])*100, 2) if sales[i] else 0 for i in range(n)]
    
    change_inv = bs_val("Change in Inventory")
    change_inv_pct = [round((change_inv[i]/sales[i])*100, 2) if sales[i] else 0 for i in range(n)]
    
    power_fuel = bs_val("Power and Fuel")
    power_fuel_pct = [round((power_fuel[i]/sales[i])*100, 2) if sales[i] else 0 for i in range(n)]
    
    other_mfr = bs_val("Other Mfr. Exp")
    other_mfr_pct = [round((other_mfr[i]/sales[i])*100, 2) if sales[i] else 0 for i in range(n)]
    
    employee = bs_val("Employee Cost")
    employee_pct = [round((employee[i]/sales[i])*100, 2) if sales[i] else 0 for i in range(n)]
    
    selling_admin = bs_val("Selling and admin")
    selling_admin_pct = [round((selling_admin[i]/sales[i])*100, 2) if sales[i] else 0 for i in range(n)]
    
    other_exp = bs_val("Other Expenses")
    other_exp_pct = [round((other_exp[i]/sales[i])*100, 2) if sales[i] else 0 for i in range(n)]
    
    other_income = bs_val("Other Income")
    other_income_pct = [round((other_income[i]/sales[i])*100, 2) if sales[i] else 0 for i in range(n)]
    
    depreciation = bs_val("Depreciation")
    depreciation_pct = [round((depreciation[i]/sales[i])*100, 2) if sales[i] else 0 for i in range(n)]
    
    interest = bs_val("Interest")
    interest_pct = [round((interest[i]/sales[i])*100, 2) if sales[i] else 0 for i in range(n)]
    
    pbt = bs_val("Profit before tax")
    pbt_pct = [round((pbt[i]/sales[i])*100, 2) if sales[i] else 0 for i in range(n)]
    
    tax = bs_val("Tax")
    tax_pct = [round((tax[i]/sales[i])*100, 2) if sales[i] else 0 for i in range(n)]
    
    net_profit = bs_val("Net profit")
    net_profit_pct = [round((net_profit[i]/sales[i])*100, 2) if sales[i] else 0 for i in range(n)]
    
    dividend = bs_val("Dividend Amount")
    dividend_pct = [round((dividend[i]/sales[i])*100, 2) if sales[i] else 0 for i in range(n)]
    
    ebitda = bs_val("EBITDA")
    ebitda_pct = [round((ebitda[i]/sales[i])*100, 2) if sales[i] else 0 for i in range(n)]
    
    # Sales is always 100%
    sales_pct = [100.0] * n
    
    common_size_income = [
        ("Sales", sales_pct),
        ("Raw Material Cost", raw_material_pct),
        ("Change in Inventory", change_inv_pct),
        ("Power and Fuel", power_fuel_pct),
        ("Other Mfr. Exp", other_mfr_pct),
        ("Employee Cost", employee_pct),
        ("Selling and admin", selling_admin_pct),
        ("Other Expenses", other_exp_pct),
        ("Other Income", other_income_pct),
        ("Depreciation", depreciation_pct),
        ("Interest", interest_pct),
        ("Profit before tax", pbt_pct),
        ("Tax", tax_pct),
        ("Net profit", net_profit_pct),
        ("Dividend Amount", dividend_pct),
        ("EBITDA", ebitda_pct),
    ]
    
    # ======================================================
    # COMMON SIZE BALANCE SHEET
    # ======================================================
    
    # Get base values
    total_liab = bs_val("Total")  # Row 60 - Total Liabilities
    total_assets = bs_val("Total Assets")  # Row 74
    
    # Liabilities (relative to Total Liabilities)
    total_liab_pct = [100.0] * n  # Total is always 100%
    
    equity = bs_val("Equity Share Capital")
    equity_pct = [round((equity[i]/total_liab[i])*100, 2) if total_liab[i] else 0 for i in range(n)]
    
    reserves = bs_val("Reserves")
    reserves_pct = [round((reserves[i]/total_liab[i])*100, 2) if total_liab[i] else 0 for i in range(n)]
    
    borrowings = bs_val("Borrowings")
    borrowings_pct = [round((borrowings[i]/total_liab[i])*100, 2) if total_liab[i] else 0 for i in range(n)]
    
    other_liab = bs_val("Other Liabilities")
    other_liab_pct = [round((other_liab[i]/total_liab[i])*100, 2) if total_liab[i] else 0 for i in range(n)]
    
    # Assets (relative to Total Assets)
    total_assets_pct = [100.0] * n  # Total Assets is always 100%
    
    net_block = bs_val("Net Block")
    net_block_pct = [round((net_block[i]/total_assets[i])*100, 2) if total_assets[i] else 0 for i in range(n)]
    
    cwip = bs_val("Capital Work in Progress")
    cwip_pct = [round((cwip[i]/total_assets[i])*100, 2) if total_assets[i] else 0 for i in range(n)]
    
    investments = bs_val("Investments")
    investments_pct = [round((investments[i]/total_assets[i])*100, 2) if total_assets[i] else 0 for i in range(n)]
    
    other_assets = bs_val("Other Assets")
    other_assets_pct = [round((other_assets[i]/total_assets[i])*100, 2) if total_assets[i] else 0 for i in range(n)]
    
    receivables = bs_val("Receivables")
    receivables_pct = [round((receivables[i]/total_assets[i])*100, 2) if total_assets[i] else 0 for i in range(n)]
    
    inventory = bs_val("Inventory")
    inventory_pct = [round((inventory[i]/total_assets[i])*100, 2) if total_assets[i] else 0 for i in range(n)]
    
    cash_bank = bs_val("Cash & Bank")
    cash_bank_pct = [round((cash_bank[i]/total_assets[i])*100, 2) if total_assets[i] else 0 for i in range(n)]
    
    common_size_balance = [
        ("Total Liabilities", total_liab_pct),
        ("Equity Share Capital", equity_pct),
        ("Reserves", reserves_pct),
        ("Borrowings", borrowings_pct),
        ("Other Liabilities", other_liab_pct),
        ("", [""]*n),  # GAP
        ("Total Assets", total_assets_pct),
        ("Net Block", net_block_pct),
        ("Capital Work in Progress", cwip_pct),
        ("Investments", investments_pct),
        ("Other Assets", other_assets_pct),
        ("Receivables", receivables_pct),
        ("Inventory", inventory_pct),
        ("Cash & Bank", cash_bank_pct),
    ]
    
    return {
        "years": years,
        "common_size_income": common_size_income,
        "common_size_balance": common_size_balance
    }


# ===============================
# FORECASTING FUNCTION
# ===============================
def run_forecasting(file, forecast_years=5):
    """
    Forecast Sales, EBITDA, Net Profit, and EPS using a mean-reversion model.

    Method:
    - Sales growth: weighted CAGR (recent 3Y weighted 2x vs older history)
      that mean-reverts toward long-run GDP (7% nominal) by year 5.
    - EBITDA margin: reverts from current margin toward its own 5-year average.
    - Net Profit and EPS derived from forecasted EBITDA × historical effective tax rate.
    - Caps: sales growth capped at 50% / -30%, margin capped at 60% / 0%.

    This avoids the straight-line extrapolation flaw of linear regression and
    reflects real analyst practice (mean-reversion DCF modelling).
    """
    import numpy as np

    # Get Historical FS data
    historical_data = run_historical_fs(file)
    years = historical_data["years"]
    n = len(years)

    def get_hist_values(label):
        for item_label, values in historical_data["income_statement"]:
            if item_label == label:
                return values
        for item_label, values in historical_data.get("balance_sheet", []):
            if item_label == label:
                return values
        return [0] * n

    sales           = get_hist_values("Sales")
    ebitda          = get_hist_values("EBITDA")
    net_profit      = get_hist_values("Net Profit")
    eps             = get_hist_values("EPS")
    depreciation    = get_hist_values("Depreciation")
    interest        = get_hist_values("Interest")
    tax             = get_hist_values("Tax")
    ebt             = get_hist_values("EBT")
    equity_shares   = get_hist_values("No. of Equity Shares")

    # ─── Sales Growth: Weighted CAGR ──────────────────────────────────────────
    # Use last 3Y CAGR (weight=2) blended with full-history CAGR (weight=1)
    def safe_cagr(start, end, periods):
        if start and end and start > 0 and periods > 0:
            return (end / start) ** (1 / periods) - 1
        return 0.0

    # Use only positive sales values
    valid_sales = [(i, s) for i, s in enumerate(sales) if s > 0]
    if len(valid_sales) >= 4:
        cagr_3y   = safe_cagr(valid_sales[-4][1], valid_sales[-1][1], 3)
        cagr_full = safe_cagr(valid_sales[0][1],  valid_sales[-1][1], len(valid_sales) - 1)
    elif len(valid_sales) >= 2:
        cagr_3y   = safe_cagr(valid_sales[-2][1], valid_sales[-1][1], 1)
        cagr_full = cagr_3y
    else:
        cagr_3y = cagr_full = 0.07

    blended_growth = (2 * cagr_3y + 1 * cagr_full) / 3
    blended_growth = max(-0.30, min(0.50, blended_growth))  # cap ±50%/-30%

    # Long-run mean reversion target: 7% nominal GDP
    LT_GROWTH = 0.07

    # ─── EBITDA margin: revert toward 5Y mean ────────────────────────────────
    valid_ebitda_margins = [
        ebitda[i] / sales[i] for i in range(n)
        if sales[i] > 0 and ebitda[i] > 0
    ]
    current_margin  = valid_ebitda_margins[-1] if valid_ebitda_margins else 0.15
    avg_margin_5y   = (sum(valid_ebitda_margins[-5:]) / len(valid_ebitda_margins[-5:])
                       if len(valid_ebitda_margins) >= 2 else current_margin)
    # Reversion speed: 30% per year toward 5Y average
    REVERSION_SPEED = 0.30

    # ─── Tax rate: 5Y average of profitable years ────────────────────────────
    tax_rates = []
    for i in range(n):
        if ebt[i] > 0 and tax[i] >= 0:
            r = min(max(tax[i] / ebt[i], 0), 0.50)
            tax_rates.append(r)
    avg_tax_rate = sum(tax_rates) / len(tax_rates) if tax_rates else 0.25

    # ─── Depreciation as % of last sales ─────────────────────────────────────
    dep_ratios = [depreciation[i] / sales[i] for i in range(n) if sales[i] > 0]
    avg_dep_ratio = sum(dep_ratios[-3:]) / len(dep_ratios[-3:]) if dep_ratios else 0.04

    # ─── Interest: keep flat at last year value ───────────────────────────────
    last_interest = interest[-1] if interest else 0

    # ─── Shares: use last known value ────────────────────────────────────────
    last_shares = equity_shares[-1] if equity_shares and equity_shares[-1] else 1

    # ─── Generate forecasts year by year ─────────────────────────────────────
    last_sales = valid_sales[-1][1] if valid_sales else 1

    sales_forecast, ebitda_forecast, np_forecast, eps_forecast = [], [], [], []
    margin_t = current_margin

    for yr in range(1, forecast_years + 1):
        # Sales growth: linearly mean-reverts from blended_growth → LT_GROWTH
        t_weight = (yr - 1) / max(forecast_years - 1, 1)  # 0 → 1 over forecast window
        growth_t = blended_growth * (1 - t_weight) + LT_GROWTH * t_weight
        growth_t = max(-0.30, min(0.50, growth_t))

        s_t = round(last_sales * (1 + growth_t) ** yr, 2)
        sales_forecast.append(s_t)

        # EBITDA margin: partial reversion toward avg_margin_5y
        margin_t = margin_t + REVERSION_SPEED * (avg_margin_5y - margin_t)
        margin_t = max(0.0, min(0.60, margin_t))
        eb_t = round(s_t * margin_t, 2)
        ebitda_forecast.append(eb_t)

        # Net Profit: EBITDA - Dep - Interest, taxed at avg_tax_rate
        dep_t    = round(s_t * avg_dep_ratio, 2)
        ebit_t   = eb_t - dep_t
        ebt_t    = ebit_t - last_interest
        tax_t    = round(max(ebt_t, 0) * avg_tax_rate, 2)
        np_t     = round(ebt_t - tax_t, 2)
        np_forecast.append(np_t)
        eps_t    = round(np_t / last_shares, 2) if last_shares else 0
        eps_forecast.append(eps_t)

    # ─── Growth rates for output ──────────────────────────────────────────────
    def growth_series(vals):
        out = [""]
        for i in range(1, len(vals)):
            if vals[i-1] and vals[i-1] != 0:
                out.append(round(((vals[i] / vals[i-1]) - 1) * 100, 2))
            else:
                out.append("")
        return out

    sales_growth_fc  = growth_series(sales_forecast)
    ebitda_growth_fc = growth_series(ebitda_forecast)
    eps_growth_fc    = growth_series(eps_forecast)

    # ─── EBITDA margins for output ────────────────────────────────────────────
    ebitda_margins_fc = [
        round((ebitda_forecast[i] / sales_forecast[i]) * 100, 2) if sales_forecast[i] else 0
        for i in range(forecast_years)
    ]

    # ─── Methodology note ────────────────────────────────────────────────────
    method_note = (
        f"Blended CAGR: {round(blended_growth*100,1)}% "
        f"(3Y CAGR {round(cagr_3y*100,1)}% × 2 + Full-history CAGR {round(cagr_full*100,1)}% × 1) / 3. "
        f"Mean-reverts to {round(LT_GROWTH*100,0):.0f}% by Year {forecast_years}. "
        f"EBITDA margin {round(current_margin*100,1)}% → reverts to 5Y avg {round(avg_margin_5y*100,1)}% "
        f"at {round(REVERSION_SPEED*100,0):.0f}% speed/yr. "
        f"Tax rate {round(avg_tax_rate*100,1)}% (5Y avg profitable years)."
    )

    # ─── Sales growth rates ───────────────────────────────────────────────────
    sales_growth = [""] + [
        round(((sales[i] / sales[i-1]) - 1) * 100, 2) if sales[i-1] else 0
        for i in range(1, n)
    ]
    ebitda_growth = [""] + [
        round(((ebitda[i] / ebitda[i-1]) - 1) * 100, 2) if ebitda[i-1] else 0
        for i in range(1, n)
    ]
    eps_growth = [""] + [
        round(((eps[i] / eps[i-1]) - 1) * 100, 2) if eps[i-1] else 0
        for i in range(1, n)
    ]
    
    # Generate future years (dates)
    # Extract the last year and add years sequentially
    from datetime import datetime
    try:
        from dateutil.relativedelta import relativedelta
        _use_relativedelta = True
    except ImportError:
        _use_relativedelta = False
    last_year_str = years[-1]  # e.g., "Mar-24"
    last_date = datetime.strptime(last_year_str + "-01", "%b-%y-%d")
    forecast_year_labels = []
    for i in range(1, forecast_years + 1):
        if _use_relativedelta:
            future_date = last_date + relativedelta(years=i)
        else:
            # Fallback: add 366 days per year to handle leap years
            from datetime import timedelta
            future_date = last_date + timedelta(days=int(365.25 * i))
        forecast_year_labels.append(future_date.strftime("%b-%y"))
    
    # ─── Build combined output ────────────────────────────────────────────────
    all_years = years + forecast_year_labels

    return {
        "years": all_years,
        "forecast_years_count": forecast_years,
        "historical_years_count": n,
        "methodology": method_note,

        # Sales
        "sales": {
            "historical":   sales,
            "forecasted":   sales_forecast,
            "growth":       sales_growth_fc,
            "hist_growth":  sales_growth,
        },

        # EBITDA
        "ebitda": {
            "historical":   ebitda,
            "forecasted":   ebitda_forecast,
            "growth":       ebitda_growth_fc,
            "hist_growth":  ebitda_growth,
            "margins_pct":  ebitda_margins_fc,
        },

        # Net Profit
        "net_profit": {
            "historical":   net_profit,
            "forecasted":   np_forecast,
        },

        # EPS
        "eps": {
            "historical":   eps,
            "forecasted":   eps_forecast,
            "growth":       eps_growth_fc,
            "hist_growth":  eps_growth,
        },

        # Assumptions used
        "assumptions": {
            "blended_growth_rate_pct": round(blended_growth * 100, 2),
            "cagr_3y_pct":             round(cagr_3y * 100, 2),
            "cagr_full_history_pct":   round(cagr_full * 100, 2),
            "long_run_growth_target_pct": round(LT_GROWTH * 100, 1),
            "current_ebitda_margin_pct":  round(current_margin * 100, 2),
            "target_ebitda_margin_pct":   round(avg_margin_5y * 100, 2),
            "avg_tax_rate_pct":           round(avg_tax_rate * 100, 2),
            "avg_dep_ratio_pct":          round(avg_dep_ratio * 100, 2),
        },
    }


# ===============================
# FCFF (FREE CASH FLOW TO FIRM) FUNCTION
# ===============================
def run_fcff(file):
    """
    Calculate Free Cash Flow to Firm (FCFF)
    Uses CFO-based method: FCFF = CFO + Interest × (1 - Tax Rate) - CAPEX
    """
    
    # Read Balance Sheet & P&L for Interest and Tax data
    bs_df = pd.read_excel(file, engine="openpyxl", sheet_name="Balance Sheet & P&L", header=None)

    # Read Cash Flow Data for CFO and CAPEX
    cf_df = _read_cf_sheet(file)
    
    YEAR_ROW_BS = 15
    START_COL = 1
    
    # Get years from Balance Sheet (these are the main years we'll use)
    years = []
    col = START_COL
    while col < bs_df.shape[1]:
        val = bs_df.iloc[YEAR_ROW_BS, col]
        if pd.isna(val):
            break
        years.append(pd.to_datetime(val).strftime("%b-%y"))
        col += 1
    
    n = len(years)
    
    # Helper function to get values from Balance Sheet & P&L
    def bs_val(label):
        r = find_row(bs_df, label)
        return [fmt(bs_df.iloc[r, START_COL + i]) if r is not None else 0 for i in range(n)]
    
    # Helper function to get values from Cash Flow Data
    # Cash Flow Data starts from 2015, we need to align with Balance Sheet years
    def cf_val(label):
        # Find the row
        cf_row = None
        for i in range(cf_df.shape[0]):
            val = cf_df.iloc[i, 0]
            if isinstance(val, str) and label.lower() in val.lower():
                cf_row = i
                break
        
        if cf_row is None:
            return [0] * n
        
        # Cash Flow years start from column 1
        # We need to match years from Balance Sheet
        cf_years = []
        for col in range(1, cf_df.shape[1]):
            year_val = cf_df.iloc[0, col]
            if pd.notna(year_val):
                cf_years.append(pd.to_datetime(year_val).strftime("%b-%y"))
        
        # Align Cash Flow data with Balance Sheet years
        result = []
        for bs_year in years:
            if bs_year in cf_years:
                idx = cf_years.index(bs_year)
                result.append(fmt(cf_df.iloc[cf_row, idx + 1]))
            else:
                result.append(0)
        
        return result
    
    # Get required components
    # From Balance Sheet & P&L
    interest = bs_val("Interest")
    tax = bs_val("Tax")
    pbt = bs_val("Profit before tax")
    sales = bs_val("Sales")
    
    # From Cash Flow Data
    cfo = cf_val("Cash from Operating Activity")
    capex_purchased = cf_val("Fixed assets purchased")
    capex_sold = cf_val("Fixed assets sold")
    
    # Calculate Tax Rate
    # Use effective tax rate, handling edge cases
    tax_rate = []
    for i in range(n):
        if pbt[i] != 0 and pbt[i] > 0:
            rate = abs(tax[i] / pbt[i])
            # Cap tax rate at reasonable limits (0% to 50%)
            rate = min(max(rate, 0), 0.5)
            tax_rate.append(round(rate, 4))
        else:
            # Use average tax rate from profitable years
            tax_rate.append(0.30)  # Default 30%
    
    # Calculate average tax rate for reference
    profitable_years_tax = [tax_rate[i] for i in range(n) if pbt[i] > 0 and tax_rate[i] > 0]
    avg_tax_rate = round(sum(profitable_years_tax) / len(profitable_years_tax), 4) if profitable_years_tax else 0.30
    
    # Use average tax rate for years with issues
    for i in range(n):
        if tax_rate[i] == 0.30 and pbt[i] <= 0:
            tax_rate[i] = avg_tax_rate
    
    # Calculate Net CAPEX (purchases are negative, sales are positive)
    net_capex = [capex_purchased[i] + capex_sold[i] for i in range(n)]
    
    # Calculate Interest After Tax
    interest_after_tax = [round(interest[i] * (1 - tax_rate[i]), 2) for i in range(n)]
    
    # Calculate FCFF
    # FCFF = CFO + Interest × (1 - Tax Rate) - CAPEX
    # Note: CAPEX is already negative in the data
    fcff = [round(cfo[i] + interest_after_tax[i] + net_capex[i], 2) for i in range(n)]
    
    # Calculate FCFF Growth Rates
    fcff_growth = [""] + [
        round(((fcff[i] / fcff[i-1]) - 1) * 100, 2) if fcff[i-1] and fcff[i-1] != 0 else 0
        for i in range(1, n)
    ]
    
    # Calculate FCFF Margin (FCFF / Sales)
    fcff_margin = [round((fcff[i] / sales[i]) * 100, 2) if sales[i] else 0 for i in range(n)]
    
    # Build the output structure
    fcff_data = [
        ("Cash from Operations (CFO)", cfo),
        ("Interest Expense", interest),
        ("Tax Rate", [round(tr * 100, 2) for tr in tax_rate]),
        ("Interest After Tax", interest_after_tax),
        ("CAPEX (Net)", net_capex),
        ("", [""]*n),  # GAP
        ("FCFF (Free Cash Flow to Firm)", fcff),
        ("FCFF Growth %", fcff_growth),
        ("FCFF Margin %", fcff_margin),
    ]
    
    return {
        "years": years,
        "fcff_data": fcff_data,
        "avg_tax_rate": round(avg_tax_rate * 100, 2),
        "method": "CFO-based: FCFF = CFO + Interest × (1 - Tax Rate) - CAPEX"
    }


# ===============================
# WACC (WEIGHTED AVERAGE COST OF CAPITAL) FUNCTION
# ===============================
def run_wacc(file, beta=None, risk_free_rate=7.1, equity_risk_premium=5.5, cost_of_equity_override=None):
    """
    Calculate Weighted Average Cost of Capital (WACC)
    Formula: WACC = (E/V × Re) + (D/V × Rd × (1 - Tax Rate))
    Cost of Equity via CAPM: Re = Rf + Beta * (Rm - Rf)

    Parameters:
    - file: Excel file path
    - beta: Stock beta (from yfinance or user input). Default 1.0 if not provided.
    - risk_free_rate: 10Y G-Sec yield in % (default 7.1%)
    - equity_risk_premium: India ERP in % (default 5.5%)
    - cost_of_equity_override: If set, skips CAPM and uses this value directly
    """
    # CAPM-based cost of equity — auto-fetches beta from yfinance if not provided
    beta_source = "user_provided"
    if cost_of_equity_override is not None:
        cost_of_equity = cost_of_equity_override
        beta_used = beta or 1.0
        beta_source = "overridden"
    else:
        # Try to resolve beta: user arg → yfinance → default 1.0
        beta_used = None
        if beta is not None:
            beta_used = beta
            beta_source = "user_provided"
        else:
            # Auto-fetch from yfinance using ticker from Excel meta row
            try:
                import yfinance as yf
            except ImportError:
                yf = None
            # Try to read ticker from meta rows of Excel (rows 0-15, col 0 label / col 1 value)
            _ticker_beta = None
            try:
                import pandas as _pd2
                _bs = _pd2.read_excel(file, engine="openpyxl",
                                      sheet_name="Balance Sheet & P&L", header=None)
                for _r in range(15):
                    _lbl = str(_bs.iloc[_r, 0]).lower() if not _pd2.isna(_bs.iloc[_r, 0]) else ""
                    if "ticker" in _lbl or "symbol" in _lbl or "bse" in _lbl or "nse" in _lbl:
                        _ticker_val = str(_bs.iloc[_r, 1]).strip()
                        if _ticker_val and _ticker_val.lower() not in ("nan", "none", ""):
                            # Normalise: add .NS for NSE if no exchange suffix
                            if "." not in _ticker_val:
                                _ticker_val = _ticker_val + ".NS"
                            _ticker_beta = _ticker_val
                            break
            except Exception:
                pass

            if yf is not None and _ticker_beta:
                try:
                    _info = yf.Ticker(_ticker_beta).fast_info
                    _beta_yf = getattr(_info, "beta", None)
                    if _beta_yf is None:
                        _info2 = yf.Ticker(_ticker_beta).info
                        _beta_yf = _info2.get("beta", None)
                    if _beta_yf and 0.1 <= float(_beta_yf) <= 5.0:
                        beta_used = round(float(_beta_yf), 3)
                        beta_source = f"yfinance ({_ticker_beta})"
                except Exception:
                    pass

            if beta_used is None:
                beta_used = 1.0
                beta_source = "default (beta=1.0 — add Ticker row to Excel for auto-fetch)"

        cost_of_equity = round(risk_free_rate + beta_used * equity_risk_premium, 2)
    
    # Read Balance Sheet & P&L
    bs_df = pd.read_excel(file, engine="openpyxl", sheet_name="Balance Sheet & P&L", header=None)
    
    YEAR_ROW = 15
    START_COL = 1
    
    # Get years
    years = []
    col = START_COL
    while col < bs_df.shape[1]:
        val = bs_df.iloc[YEAR_ROW, col]
        if pd.isna(val):
            break
        years.append(pd.to_datetime(val).strftime("%b-%y"))
        col += 1
    
    n = len(years)
    
    # Helper function to get values
    def bs_val(label):
        r = find_row(bs_df, label)
        return [fmt(bs_df.iloc[r, START_COL + i]) if r is not None else 0 for i in range(n)]
    
    # Get META data (rows 0-10)
    def get_meta_value(label):
        for i in range(15):
            val = bs_df.iloc[i, 0]
            if isinstance(val, str) and label.lower() in val.lower():
                return fmt(bs_df.iloc[i, 1])
        return None
    
    # Get current market cap and shares
    current_mcap = get_meta_value("Market Capitalization")
    num_shares = get_meta_value("Number of shares")
    current_price = get_meta_value("Current Price")
    
    # Get financial data
    borrowings = bs_val("Borrowings")
    interest = bs_val("Interest")
    tax = bs_val("Tax")
    pbt = bs_val("Profit before tax")
    equity_capital = bs_val("Equity Share Capital")
    reserves = bs_val("Reserves")
    
    # FIX 5: Use actual historical price × shares for market cap
    book_equity = [equity_capital[i] + reserves[i] for i in range(n)]

    # Dynamically find PRICE row (do not hardcode row 89 — it varies by Screener export)
    num_shares_row = find_row(bs_df, "Adjusted Equity Shares in Cr")
    PRICE_ROW = find_row(bs_df, "Price") or find_row(bs_df, "Market Price") or find_row(bs_df, "Share Price")
    market_equity = []

    for i in range(n):
        try:
            price  = fmt(bs_df.iloc[PRICE_ROW, START_COL + i]) if PRICE_ROW is not None else None
            shares = fmt(bs_df.iloc[num_shares_row, START_COL + i]) if num_shares_row is not None else None
            if price and shares and price > 0 and shares > 0:
                market_equity.append(round(price * shares, 2))
            elif i == n - 1 and current_mcap:
                market_equity.append(current_mcap)
            else:
                # Fallback: book value × 1.5 (conservative estimate)
                market_equity.append(round(book_equity[i] * 1.5, 2))
        except Exception:
            if i == n - 1 and current_mcap:
                market_equity.append(current_mcap)
            else:
                market_equity.append(round(book_equity[i] * 1.5, 2))
    
    # Market Value of Debt (D) - use book value as proxy
    market_debt = borrowings
    
    # Total Value (V = E + D)
    total_value = [market_equity[i] + market_debt[i] for i in range(n)]
    
    # Weights
    equity_weight = [round((market_equity[i] / total_value[i]) * 100, 2) if total_value[i] else 0 for i in range(n)]
    debt_weight = [round((market_debt[i] / total_value[i]) * 100, 2) if total_value[i] else 0 for i in range(n)]
    
    # Calculate Cost of Debt (Rd)
    cost_of_debt = [round((interest[i] / market_debt[i]) * 100, 2) if market_debt[i] else 0 for i in range(n)]
    
    # Calculate Tax Rate (smart handling)
    tax_rate = []
    for i in range(n):
        if pbt[i] != 0 and pbt[i] > 0:
            rate = abs(tax[i] / pbt[i])
            rate = min(max(rate, 0), 0.5)
            tax_rate.append(round(rate * 100, 2))
        else:
            tax_rate.append(30.0)  # Default
    
    # Calculate average tax rate from profitable years
    profitable_years_tax = [tax_rate[i] for i in range(n) if pbt[i] > 0 and tax_rate[i] > 0]
    avg_tax_rate = round(sum(profitable_years_tax) / len(profitable_years_tax), 2) if profitable_years_tax else 30.0
    
    # Use average tax rate for loss years
    for i in range(n):
        if pbt[i] <= 0:
            tax_rate[i] = avg_tax_rate
    
    # After-tax Cost of Debt
    after_tax_cod = [round(cost_of_debt[i] * (1 - tax_rate[i]/100), 2) for i in range(n)]
    
    # Cost of Equity (Re) - user provided, constant across years
    cost_of_equity_list = [cost_of_equity] * n
    
    # Calculate WACC
    # WACC = (E/V × Re) + (D/V × Rd × (1 - Tax Rate))
    wacc = []
    for i in range(n):
        e_component = (equity_weight[i] / 100) * cost_of_equity
        d_component = (debt_weight[i] / 100) * after_tax_cod[i]
        wacc_value = round(e_component + d_component, 2)
        wacc.append(wacc_value)
    
    # Build the output structure
    capm_beta  = beta if beta is not None else 1.0
    wacc_data = [
        ("Cost of Equity Method", [beta_source]*n),
        ("Risk-Free Rate (%)", [risk_free_rate]*n),
        ("Beta Used", [round(beta_used,3)]*n),
        ("Equity Risk Premium (%)", [equity_risk_premium]*n),
        ("Cost of Equity Re (%)", [cost_of_equity]*n),
        ("", [""]*n),
        ("Market Value of Equity (E)", market_equity),
        ("Market Value of Debt (D)", market_debt),
        ("Total Value (V = E + D)", total_value),
        ("", [""]*n),  # GAP
        ("Equity Weight (E/V)", equity_weight),
        ("Debt Weight (D/V)", debt_weight),
        ("", [""]*n),  # GAP
        ("Cost of Debt (Rd)", cost_of_debt),
        ("Tax Rate", tax_rate),
        ("After-Tax Cost of Debt", after_tax_cod),
        ("Cost of Equity (Re)", cost_of_equity_list),
        ("", [""]*n),  # GAP
        ("WACC", wacc),
    ]
    
    return {
        "years": years,
        "wacc_data": wacc_data,
        "current_wacc": wacc[-1],
        "avg_wacc": round(sum(wacc) / len(wacc), 2),
        "cost_of_equity_used": cost_of_equity,
        "capm_components": {"beta": beta_used, "beta_source": beta_source, "risk_free_rate": risk_free_rate, "equity_risk_premium": equity_risk_premium},
        "avg_tax_rate": avg_tax_rate,
        "current_market_cap": current_mcap,
        "formula": "WACC = (E/V × Re) + (D/V × Rd × (1 - Tax Rate))"
    }


# ===============================
# TERMINAL VALUE & DCF VALUATION FUNCTION
# ===============================
def run_terminal_value_dcf(file, cost_of_equity=13.0, growth_rate=4.0, forecast_years=5, cost_of_equity_override=None):
    """
    Calculate Terminal Value and complete DCF Valuation
    Uses Gordon Growth Model: TV = FCFF(n) × (1 + g) / (WACC - g)
    
    Parameters:
    - file: Excel file path
    - cost_of_equity: Cost of Equity (Re) in percentage
    - growth_rate: Perpetual growth rate (g) in percentage
    - forecast_years: Number of years to forecast (default 5)
    """
    
    # Import numpy for calculations
    import numpy as np
    
    # Get WACC first
    _coe = cost_of_equity_override if cost_of_equity_override is not None else cost_of_equity
    wacc_result = run_wacc(file, cost_of_equity_override=_coe)
    current_wacc = wacc_result["current_wacc"]
    wacc_decimal = current_wacc / 100
    
    # Clamp growth rate to be safely below WACC (never raise — just fix it)
    if current_wacc <= 0:
        current_wacc = 12.0   # sensible default if WACC calculation failed
        wacc_decimal = current_wacc / 100
    if growth_rate >= current_wacc:
        growth_rate = round(current_wacc * 0.4, 2)   # cap at 40% of WACC
    
    growth_decimal = growth_rate / 100
    
    # Get historical FCFF data
    fcff_result = run_fcff(file)
    historical_years = fcff_result["years"]
    
    # Extract historical FCFF values from fcff_data
    fcff_values = None
    for label, values in fcff_result["fcff_data"]:
        if label == "FCFF (Free Cash Flow to Firm)":
            fcff_values = values
            break
    
    if fcff_values is None or all(v == 0 for v in fcff_values):
        # FCFF could not be computed — return informative error dict
        return {
            "error": "Could not calculate FCFF from this Excel file. "
                     "Ensure your Screener Excel has a 'Cash Flow Data' sheet.",
            "years": historical_years,
            "fcff_data": [],
        }
    
    n_historical = len(fcff_values)
    
    # =====================================================
    # FORECAST FCFF using Linear Regression
    # =====================================================
    
    # Prepare data for linear regression
    year_weights = list(range(1, n_historical + 1))
    
    # Filter out any zero or missing values for regression
    valid_indices = [i for i in range(n_historical) if fcff_values[i] and fcff_values[i] != 0]
    
    if len(valid_indices) < 3:
        # Not enough data points, use simple average growth
        avg_fcff = sum([fcff_values[i] for i in valid_indices]) / len(valid_indices)
        forecasted_fcff = [avg_fcff * (1.05 ** i) for i in range(1, forecast_years + 1)]
    else:
        # Linear regression
        x = np.array([year_weights[i] for i in valid_indices])
        y = np.array([fcff_values[i] for i in valid_indices])
        
        # Calculate slope (m) and intercept (b)
        n_points = len(x)
        m = (n_points * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n_points * np.sum(x ** 2) - np.sum(x) ** 2)
        b = (np.sum(y) - m * np.sum(x)) / n_points
        
        # Forecast future FCFF
        future_years = list(range(n_historical + 1, n_historical + forecast_years + 1))
        forecasted_fcff = [round(m * yr + b, 2) for yr in future_years]
    
    # Generate forecast year labels (leap-year safe)
    from datetime import datetime
    try:
        from dateutil.relativedelta import relativedelta as _rd
        _rdok = True
    except ImportError:
        from datetime import timedelta as _td
        _rdok = False
    last_year_str = historical_years[-1]
    last_date = datetime.strptime(last_year_str + "-01", "%b-%y-%d")
    forecast_year_labels = []
    for i in range(1, forecast_years + 1):
        fd = last_date + (_rd(years=i) if _rdok else _td(days=int(365.25*i)))
        forecast_year_labels.append(fd.strftime("%b-%y"))
    
    # =====================================================
    # TERMINAL VALUE CALCULATION
    # =====================================================
    
    # Last forecast year FCFF
    fcff_terminal_year = forecasted_fcff[-1]
    
    # Terminal Value using Gordon Growth Model
    # TV = FCFF(n) × (1 + g) / (WACC - g)
    terminal_value = round((fcff_terminal_year * (1 + growth_decimal)) / (wacc_decimal - growth_decimal), 2)
    
    # =====================================================
    # PRESENT VALUE CALCULATIONS
    # =====================================================
    
    # Discount factors for each forecast year
    discount_factors = [round(1 / ((1 + wacc_decimal) ** i), 4) for i in range(1, forecast_years + 1)]
    
    # Present Value of each forecasted FCFF
    pv_fcffs = [round(forecasted_fcff[i] * discount_factors[i], 2) for i in range(forecast_years)]
    
    # Present Value of Terminal Value (discounted from last forecast year)
    pv_terminal_value = round(terminal_value * discount_factors[-1], 2)
    
    # Total Present Value of Forecasted FCFFs
    total_pv_fcffs = round(sum(pv_fcffs), 2)
    
    # =====================================================
    # ENTERPRISE VALUE & EQUITY VALUE
    # =====================================================
    
    # Enterprise Value = PV of Forecasted FCFFs + PV of Terminal Value
    enterprise_value = round(total_pv_fcffs + pv_terminal_value, 2)
    
    # Get latest debt and cash from Balance Sheet
    bs_df = pd.read_excel(file, engine="openpyxl", sheet_name="Balance Sheet & P&L", header=None)
    
    def get_latest_bs_value(label):
        r = find_row(bs_df, label)
        if r is not None:
            for col in range(bs_df.shape[1] - 1, 0, -1):
                val = bs_df.iloc[r, col]
                if pd.notna(val):
                    return fmt(val)
        return 0
    
    # Get current values
    borrowings = get_latest_bs_value("Borrowings")
    cash_bank = get_latest_bs_value("Cash & Bank")
    
    # Net Debt = Borrowings - Cash
    net_debt = round(borrowings - cash_bank, 2)
    
    # Equity Value = Enterprise Value - Net Debt
    equity_value = round(enterprise_value - net_debt, 2)
    
    # Get number of shares
    def get_meta_value(label):
        for i in range(15):
            val = bs_df.iloc[i, 0]
            if isinstance(val, str) and label.lower() in val.lower():
                return fmt(bs_df.iloc[i, 1])
        return None
    
    num_shares = get_meta_value("Number of shares")
    current_price = get_meta_value("Current Price")
    
    # Value per Share
    value_per_share = round(equity_value / num_shares, 2) if num_shares else 0
    
    # Upside/Downside
    upside_downside = 0
    if current_price and value_per_share:
        upside_downside = round(((value_per_share / current_price) - 1) * 100, 2)
    
    # =====================================================
    # SENSITIVITY ANALYSIS
    # =====================================================
    
    # Test different growth rates
    sensitivity_growth_rates = [2.0, 3.0, 4.0, 5.0, 6.0]
    sensitivity_results = []
    
    for g in sensitivity_growth_rates:
        if g < current_wacc:
            g_decimal = g / 100
            tv_sens = round((fcff_terminal_year * (1 + g_decimal)) / (wacc_decimal - g_decimal), 2)
            pv_tv_sens = round(tv_sens * discount_factors[-1], 2)
            ev_sens = round(total_pv_fcffs + pv_tv_sens, 2)
            eq_sens = round(ev_sens - net_debt, 2)
            val_per_share_sens = round(eq_sens / num_shares, 2) if num_shares else 0
            
            sensitivity_results.append({
                "growth_rate": g,
                "terminal_value": tv_sens,
                "enterprise_value": ev_sens,
                "equity_value": eq_sens,
                "value_per_share": val_per_share_sens
            })
    
    # =====================================================
    # BUILD OUTPUT STRUCTURE
    # =====================================================
    
    # FCFF Forecast Table
    fcff_forecast_table = []
    for i in range(forecast_years):
        fcff_forecast_table.append({
            "year": forecast_year_labels[i],
            "fcff": forecasted_fcff[i],
            "discount_factor": discount_factors[i],
            "present_value": pv_fcffs[i]
        })
    
    # Terminal Value Breakdown
    terminal_value_breakdown = {
        "fcff_terminal_year": fcff_terminal_year,
        "growth_rate": growth_rate,
        "wacc": current_wacc,
        "terminal_value": terminal_value,
        "discount_factor": discount_factors[-1],
        "pv_terminal_value": pv_terminal_value
    }
    
    # Valuation Summary
    valuation_summary = {
        "pv_forecasted_fcffs": total_pv_fcffs,
        "pv_terminal_value": pv_terminal_value,
        "enterprise_value": enterprise_value,
        "less_net_debt": net_debt,
        "equity_value": equity_value,
        "shares_outstanding": num_shares,
        "value_per_share": value_per_share,
        "current_market_price": current_price,
        "upside_downside": upside_downside
    }
    
    # ──────────────────────────────────────────────────────────────
    # MONTE CARLO SIMULATION (1000 runs)
    # Varies: WACC ±1.5%, Growth rate ±1%, FCFF terminal ±15%
    # Gives a realistic valuation range rather than a point estimate
    # ──────────────────────────────────────────────────────────────
    mc_vps_list = []
    try:
        _mc_runs = 1000
        _rng = np.random.default_rng(seed=42)
        _base_vps = valuation_summary.get("value_per_share", 0) or 0
        _shares_mc = valuation_summary.get("shares_outstanding") or 1
        _net_debt_mc = valuation_summary.get("less_net_debt", 0) or 0

        for _ in range(_mc_runs):
            # Sample from normal distributions around base assumptions
            _w  = max(wacc_decimal + _rng.normal(0, 0.015), 0.04)   # WACC ± 1.5%
            _g  = min(growth_decimal + _rng.normal(0, 0.01),         # growth ± 1%
                      _w - 0.005)                                     # must stay < WACC
            _g  = max(_g, 0.01)
            _tf = fcff_terminal_year * (1 + _rng.normal(0, 0.15))    # FCFF ± 15%

            _tv_mc = (_tf * (1 + _g)) / (_w - _g)
            _pv_tv_mc = _tv_mc / ((1 + _w) ** forecast_years)
            _pv_fcff_mc = sum(
                forecasted_fcff[i] / ((1 + _w) ** (i + 1))
                for i in range(forecast_years)
            )
            _ev_mc = _pv_fcff_mc + _pv_tv_mc
            _eq_mc = _ev_mc - _net_debt_mc
            _vps_mc = _eq_mc / _shares_mc if _shares_mc else 0
            mc_vps_list.append(_vps_mc)

        mc_arr = sorted(mc_vps_list)
        mc_results = {
            "simulations":       _mc_runs,
            "p10_value_per_share": round(mc_arr[int(_mc_runs * 0.10)], 2),
            "p25_value_per_share": round(mc_arr[int(_mc_runs * 0.25)], 2),
            "p50_value_per_share": round(mc_arr[int(_mc_runs * 0.50)], 2),
            "p75_value_per_share": round(mc_arr[int(_mc_runs * 0.75)], 2),
            "p90_value_per_share": round(mc_arr[int(_mc_runs * 0.90)], 2),
            "mean_value_per_share": round(sum(mc_arr) / _mc_runs, 2),
            "std_value_per_share":  round(float(np.std(mc_arr)), 2),
            "bear_case_p10":       round(mc_arr[int(_mc_runs * 0.10)], 2),
            "base_case_p50":       round(mc_arr[int(_mc_runs * 0.50)], 2),
            "bull_case_p90":       round(mc_arr[int(_mc_runs * 0.90)], 2),
            "interpretation": (
                f"Monte Carlo ({_mc_runs} runs): "
                f"10th–90th percentile value range = "
                f"₹{round(mc_arr[int(_mc_runs*0.10)],0):.0f} – "
                f"₹{round(mc_arr[int(_mc_runs*0.90)],0):.0f} per share."
            )
        }
    except Exception:
        mc_results = {"error": "Monte Carlo simulation failed", "simulations": 0}

    return {
        "historical_years": historical_years,
        "forecast_years": forecast_year_labels,
        "forecast_period": forecast_years,
        "fcff_forecast_table": fcff_forecast_table,
        "terminal_value_breakdown": terminal_value_breakdown,
        "valuation_summary": valuation_summary,
        "sensitivity_analysis": sensitivity_results,
        "monte_carlo": mc_results,
        "wacc_used": current_wacc,
        "growth_rate_used": growth_rate,
        "cost_of_equity_used": cost_of_equity
    }


# ===============================
# ALTMAN Z-SCORE (BANKRUPTCY RISK) FUNCTION
# ===============================
def run_altman_zscore(file):
    """
    Calculate Altman Z-Score for bankruptcy risk assessment
    
    Z-Score Formula (Manufacturing Companies):
    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    
    Where:
    X1 = Working Capital / Total Assets
    X2 = Retained Earnings / Total Assets
    X3 = EBIT / Total Assets
    X4 = Market Value of Equity / Total Liabilities
    X5 = Sales / Total Assets
    
    Interpretation:
    - Z > 2.99: Safe Zone (Low bankruptcy risk)
    - 1.81 < Z < 2.99: Grey Zone (Moderate risk)
    - Z < 1.81: Distress Zone (High bankruptcy risk)
    
    Returns:
    - Z-Score for each historical year
    - Risk classification
    - Component breakdown
    """
    
    # Read Excel file
    bs_df = pd.read_excel(file, engine="openpyxl", sheet_name="Balance Sheet & P&L", header=None)
    
    # Get years
    YEAR_ROW = 15
    START_COL = 1
    
    years = []
    col = START_COL
    while col < bs_df.shape[1]:
        val = bs_df.iloc[YEAR_ROW, col]
        if pd.isna(val):
            break
        years.append(pd.to_datetime(val).strftime("%b-%y"))
        col += 1
    
    n = len(years)
    
    # Helper function
    def val(label):
        r = find_row(bs_df, label)
        return [fmt(bs_df.iloc[r, START_COL + i]) if r is not None else 0 for i in range(n)]
    
    # ===============================
    # EXTRACT REQUIRED DATA
    # ===============================
    
    # P&L Data
    sales = val("Sales")
    ebitda = val("EBITDA")
    depreciation = val("Depreciation")
    
    # Balance Sheet Data
    equity = val("Equity Share Capital")
    reserves = val("Reserves")
    borrowings = val("Borrowings")
    other_liabilities = val("Other Liabilities")
    total_assets = val("Total Assets")
    
    receivables = val("Receivables")
    inventory = val("Inventory")
    cash_bank = val("Cash & Bank")
    
    # Metadata
    num_shares_in_cr = val("Adjusted Equity Shares in Cr")  # Adjusted shares per year
    
    # Dynamically find PRICE row instead of hardcoding row 89
    PRICE_ROW_ALT = find_row(bs_df, "Price") or find_row(bs_df, "Market Price") or find_row(bs_df, "Share Price")
    stock_prices = []
    for i in range(n):
        try:
            price = fmt(bs_df.iloc[PRICE_ROW_ALT, START_COL + i]) if PRICE_ROW_ALT is not None else 0
        except Exception:
            price = 0
        stock_prices.append(price)
    
    # ===============================
    # CALCULATE COMPONENTS
    # ===============================
    
    z_scores = []
    z_components = []
    risk_classifications = []
    
    for i in range(n):
        # Calculate EBIT
        ebit = ebitda[i] - depreciation[i]
        
        # Calculate Working Capital
        current_assets = receivables[i] + inventory[i] + cash_bank[i]
        current_liabilities = borrowings[i] + other_liabilities[i]
        working_capital = current_assets - current_liabilities
        
        # Calculate Total Liabilities
        total_liabilities = borrowings[i] + other_liabilities[i]
        
        # Calculate Market Value of Equity (use historical price for each year)
        market_value_equity = num_shares_in_cr[i] * stock_prices[i]
        
        # Retained Earnings (using Reserves as proxy)
        retained_earnings = reserves[i]
        
        # Calculate X components
        X1 = working_capital / total_assets[i] if total_assets[i] else 0
        X2 = retained_earnings / total_assets[i] if total_assets[i] else 0
        X3 = ebit / total_assets[i] if total_assets[i] else 0
        X4 = market_value_equity / total_liabilities if total_liabilities else 0
        X5 = sales[i] / total_assets[i] if total_assets[i] else 0
        
        # Calculate Z-Score
        z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
        
        # Risk Classification
        if z > 2.99:
            risk = "Safe Zone"
            status = "Low Risk"
        elif z > 1.81:
            risk = "Grey Zone"
            status = "Moderate Risk"
        else:
            risk = "Distress Zone"
            status = "High Risk"
        
        z_scores.append(round(z, 4))
        risk_classifications.append(f"{risk} - {status}")
        
        z_components.append({
            "year": years[i],
            "X1_working_capital_ratio": round(X1, 4),
            "X2_retained_earnings_ratio": round(X2, 4),
            "X3_ebit_ratio": round(X3, 4),
            "X4_market_equity_ratio": round(X4, 4),
            "X5_asset_turnover": round(X5, 4),
            "z_score": round(z, 4),
            "risk_classification": f"{risk} - {status}"
        })
    
    # ===============================
    # CALCULATE TRENDS
    # ===============================
    
    # Z-Score trend
    z_score_change = []
    for i in range(n):
        if i == 0:
            z_score_change.append("")
        else:
            change = z_scores[i] - z_scores[i-1]
            z_score_change.append(round(change, 4))
    
    # Latest year analysis
    latest_year = years[-1]
    latest_z_score = z_scores[-1]
    latest_risk = risk_classifications[-1]
    
    # Average Z-Score over all years
    avg_z_score = round(sum(z_scores) / len(z_scores), 4)
    
    # Determine trend direction
    if len(z_scores) >= 3:
        recent_trend = z_scores[-1] - z_scores[-3]
        if recent_trend > 0.5:
            trend_direction = "Improving"
        elif recent_trend < -0.5:
            trend_direction = "Deteriorating"
        else:
            trend_direction = "Stable"
    else:
        trend_direction = "Insufficient data"
    
    # ===============================
    # INTERPRETATION & RECOMMENDATIONS
    # ===============================
    
    interpretation = []
    recommendations = []
    
    if latest_z_score > 2.99:
        interpretation.append("The company is in the SAFE ZONE with strong financial health.")
        interpretation.append("Low probability of bankruptcy in the near term.")
        recommendations.append("Monitor ongoing performance and maintain financial discipline")
    elif latest_z_score > 1.81:
        interpretation.append("The company is in the GREY ZONE with moderate financial stress.")
        interpretation.append("Some warning signs present - requires careful monitoring.")
        recommendations.append("Improve working capital management")
        recommendations.append("Focus on profitability and debt reduction")
        recommendations.append("Monitor liquidity closely")
    else:
        interpretation.append("The company is in the DISTRESS ZONE with high bankruptcy risk.")
        interpretation.append("Significant financial distress indicators present.")
        recommendations.append("URGENT: Implement restructuring plan")
        recommendations.append("Improve cash flow from operations immediately")
        recommendations.append("Negotiate with creditors for debt relief")
        recommendations.append("Consider asset sales or capital infusion")
    
    # Add trend-based insights
    if trend_direction == "Improving":
        interpretation.append(f"Positive trend: Z-Score has improved over the past 3 years.")
    elif trend_direction == "Deteriorating":
        interpretation.append(f"Warning: Z-Score has deteriorated over the past 3 years.")
        recommendations.append("Address declining financial health immediately")
    
    # ===============================
    # RETURN COMPREHENSIVE RESULTS
    # ===============================
    
    return {
        "years": years,
        "z_scores": z_scores,
        "z_score_change": z_score_change,
        "risk_classifications": risk_classifications,
        "z_components_detail": z_components,
        "summary": {
            "latest_year": latest_year,
            "latest_z_score": latest_z_score,
            "latest_risk_classification": latest_risk,
            "average_z_score": avg_z_score,
            "trend_direction": trend_direction
        },
        "interpretation": interpretation,
        "recommendations": recommendations,
        "risk_zones": {
            "safe_zone": "> 2.99 (Low Risk)",
            "grey_zone": "1.81 - 2.99 (Moderate Risk)",
            "distress_zone": "< 1.81 (High Risk)"
        }
    }


# ===============================
# SCENARIO ANALYSIS (BULL/BASE/BEAR) FUNCTION
# ===============================
def run_scenario_analysis(file, forecast_years=5):
    """
    Run Bull/Base/Bear scenario analysis with different assumptions
    Returns 3 complete DCF valuations with varying parameters
    
    Parameters:
    - file: Excel file path
    - forecast_years: Number of years to forecast (default 5)
    """
    
    # Define scenario parameters
    scenarios = {
        "bull": {
            "name": "Bull Case (Optimistic)",
            "cost_of_equity": 11.0,      # Lower risk premium
            "terminal_growth": 6.0,       # High GDP-aligned growth
            "revenue_growth_adj": 1.50,   # 50% higher than base
            "ebitda_margin_adj": 1.50,    # Margin expansion (150 bps)
            "description": "Strong growth, margin expansion, lower risk"
        },
        "base": {
            "name": "Base Case (Most Likely)",
            "cost_of_equity": 13.0,      # Current assumption
            "terminal_growth": 4.0,       # Moderate growth
            "revenue_growth_adj": 1.00,   # Base growth
            "ebitda_margin_adj": 0.00,    # Current margins
            "description": "Moderate growth, stable margins, current risk"
        },
        "bear": {
            "name": "Bear Case (Pessimistic)",
            "cost_of_equity": 15.0,      # Higher risk premium
            "terminal_growth": 2.5,       # Conservative growth
            "revenue_growth_adj": 0.40,   # 60% lower than base (recession)
            "ebitda_margin_adj": -1.50,   # Margin compression (-150 bps)
            "description": "Weak growth, margin compression, higher risk"
        }
    }
    
    # Store results for each scenario
    scenario_results = {}

    # FIX 6: Get base FCFF first — then apply scenario multipliers
    try:
        base_fcff_result = run_fcff(file)
        base_fcff_values = None
        for label, vals in base_fcff_result["fcff_data"]:
            if label == "FCFF (Free Cash Flow to Firm)":
                base_fcff_values = vals
                break
        last_base_fcff = base_fcff_values[-1] if base_fcff_values else 0
    except Exception:
        last_base_fcff = 0

    # Scenario FCFF multipliers — derived from company's own growth volatility
    # Bull = base + 1 std dev of historical FCFF growth
    # Bear = base - 1.5 std dev (asymmetric — downturns are larger than upturns)
    try:
        import numpy as _np2
        _fcff_vals_clean = [v for v in (base_fcff_values or []) if v and v != 0]
        if len(_fcff_vals_clean) >= 3:
            _fcff_growths = [
                (_fcff_vals_clean[i] / _fcff_vals_clean[i-1]) - 1
                for i in range(1, len(_fcff_vals_clean))
                if _fcff_vals_clean[i-1] > 0
            ]
            if _fcff_growths:
                _std = float(_np2.std(_fcff_growths))
                _std = max(0.10, min(0.40, _std))   # floor 10%, cap 40%
                _bull_mult = round(1.0 + _std, 2)
                _bear_mult = round(1.0 - _std * 1.5, 2)
                _bear_mult = max(0.40, _bear_mult)  # bear floor: 40% of base
            else:
                _bull_mult, _bear_mult = 1.25, 0.65
        else:
            _bull_mult, _bear_mult = 1.25, 0.65
    except Exception:
        _bull_mult, _bear_mult = 1.30, 0.60

    scenario_fcff_multiplier = {
        "bull": _bull_mult,
        "base": 1.00,
        "bear": _bear_mult,
    }

    # Run DCF for each scenario
    for scenario_key, params in scenarios.items():
        try:
            # Run Terminal Value DCF with scenario parameters
            dcf_result = run_terminal_value_dcf(
                file,
                cost_of_equity_override=params["cost_of_equity"],
                growth_rate=params["terminal_growth"],
                forecast_years=forecast_years
            )

            # Apply FCFF multiplier to scenario valuation
            fcff_adj = scenario_fcff_multiplier.get(scenario_key, 1.0)
            if fcff_adj != 1.0 and last_base_fcff != 0:
                try:
                    wacc_d = dcf_result["wacc_used"] / 100
                    g_d    = params["terminal_growth"] / 100
                    adj_last_fcff = last_base_fcff * fcff_adj
                    forecasted = [adj_last_fcff * ((1 + g_d) ** i) for i in range(1, forecast_years + 1)]
                    discounted = [fcf / ((1 + wacc_d) ** i) for i, fcf in enumerate(forecasted, 1)]
                    tv    = (forecasted[-1] * (1 + g_d)) / (wacc_d - g_d)
                    pv_tv = tv / ((1 + wacc_d) ** forecast_years)
                    ev    = round(sum(discounted) + pv_tv, 2)
                    vs    = dcf_result["valuation_summary"]
                    net_debt = vs.get("less_net_debt", 0)
                    equity_val = round(ev - net_debt, 2)
                    shares = vs.get("shares_outstanding") or 1
                    vps = round(equity_val / shares, 2) if shares else 0
                    cmp = vs.get("current_market_price") or 0
                    updown = round(((vps / cmp) - 1) * 100, 2) if cmp else 0
                    dcf_result["valuation_summary"].update({
                        "enterprise_value": ev, "equity_value": equity_val,
                        "value_per_share": vps, "upside_downside": updown,
                        "fcff_scenario_multiplier": fcff_adj,
                    })
                    dcf_result["terminal_value_breakdown"]["terminal_value"] = round(tv, 2)
                except Exception:
                    pass  # keep original if adjustment fails

            # Get WACC for this scenario
            wacc_used = dcf_result["wacc_used"]
            
            # Store scenario result
            scenario_results[scenario_key] = {
                "name": params["name"],
                "description": params["description"],
                "assumptions": {
                    "cost_of_equity": params["cost_of_equity"],
                    "wacc": wacc_used,
                    "terminal_growth": params["terminal_growth"],
                    "revenue_growth_adj": params["revenue_growth_adj"],
                    "ebitda_margin_adj": params["ebitda_margin_adj"]
                },
                "valuation": dcf_result["valuation_summary"],
                "terminal_value": dcf_result["terminal_value_breakdown"]["terminal_value"],
                "enterprise_value": dcf_result["valuation_summary"]["enterprise_value"],
                "equity_value": dcf_result["valuation_summary"]["equity_value"],
                "value_per_share": dcf_result["valuation_summary"]["value_per_share"],
                "current_price": dcf_result["valuation_summary"]["current_market_price"],
                "upside_downside": dcf_result["valuation_summary"]["upside_downside"]
            }
            
        except Exception as e:
            print(f"Error in {scenario_key} scenario: {e}")
            scenario_results[scenario_key] = None
    
    # Calculate valuation range
    bull_value = scenario_results["bull"]["value_per_share"] if scenario_results.get("bull") else 0
    base_value = scenario_results["base"]["value_per_share"] if scenario_results.get("base") else 0
    bear_value = scenario_results["bear"]["value_per_share"] if scenario_results.get("bear") else 0
    
    valuation_range = {
        "min": bear_value,
        "max": bull_value,
        "range_percent": round(((bull_value - bear_value) / bear_value) * 100, 2) if bear_value else 0
    }
    
    # Calculate probability-weighted expected value
    # Default probabilities: Bull 15%, Base 60%, Bear 25%
    probabilities = {
        "bull": 0.15,
        "base": 0.60,
        "bear": 0.25
    }
    
    expected_value = round(
        bull_value * probabilities["bull"] +
        base_value * probabilities["base"] +
        bear_value * probabilities["bear"],
        2
    )
    
    # Get current market price
    current_price = scenario_results["base"]["current_price"] if scenario_results.get("base") else 0
    
    # Calculate expected value vs market
    expected_vs_market = round(((expected_value / current_price) - 1) * 100, 2) if current_price else 0
    
    # Create comparison summary
    comparison_summary = {
        "bull": {
            "value_per_share": bull_value,
            "upside_downside": scenario_results["bull"]["upside_downside"] if scenario_results.get("bull") else 0,
            "recommendation": "BUY" if scenario_results["bull"]["upside_downside"] > 20 else "HOLD"
        },
        "base": {
            "value_per_share": base_value,
            "upside_downside": scenario_results["base"]["upside_downside"] if scenario_results.get("base") else 0,
            "recommendation": "BUY" if scenario_results["base"]["upside_downside"] > 20 else ("HOLD" if scenario_results["base"]["upside_downside"] > -10 else "SELL")
        },
        "bear": {
            "value_per_share": bear_value,
            "upside_downside": scenario_results["bear"]["upside_downside"] if scenario_results.get("bear") else 0,
            "recommendation": "SELL" if scenario_results["bear"]["upside_downside"] < -20 else "HOLD"
        }
    }
    
    # Overall recommendation based on expected value
    if expected_vs_market > 20:
        overall_recommendation = "BUY"
        overall_rationale = "Expected value significantly above market price"
    elif expected_vs_market > -10:
        overall_recommendation = "HOLD"
        overall_rationale = "Expected value close to market price"
    else:
        overall_recommendation = "SELL"
        overall_rationale = "Expected value below market price"
    
    # Risk/Reward Analysis
    upside_potential = round(((bull_value / current_price) - 1) * 100, 2) if current_price else 0
    downside_risk = round(((bear_value / current_price) - 1) * 100, 2) if current_price else 0
    
    risk_reward = {
        "upside_potential": upside_potential,
        "downside_risk": abs(downside_risk),
        "risk_reward_ratio": round(abs(upside_potential / downside_risk), 2) if downside_risk else 0,
        "interpretation": "Favorable" if abs(upside_potential) > abs(downside_risk) else "Unfavorable"
    }
    
    return {
        "scenarios": scenario_results,
        "comparison_summary": comparison_summary,
        "valuation_range": valuation_range,
        "probabilities": probabilities,
        "expected_value": expected_value,
        "expected_vs_market": expected_vs_market,
        "current_market_price": current_price,
        "overall_recommendation": overall_recommendation,
        "overall_rationale": overall_rationale,
        "risk_reward": risk_reward
    }


# ===============================
# ROIC (RETURN ON INVESTED CAPITAL)
# ===============================
def run_roic(file):
    """
    ROIC = NOPAT / Invested Capital
    NOPAT = EBIT × (1 - effective tax rate)
    Invested Capital = Total Equity + Borrowings - Cash & Bank

    Interpretation:
    - ROIC > 25%: Exceptional — strong competitive moat
    - 15–25%: Good — above cost of capital
    - 10–15%: Fair — average business
    - < 10%: Poor — may be destroying value
    """
    historical = run_historical_fs(file)
    years = historical["years"]
    n = len(years)

    def gv(label):
        for lbl, vals in historical["income_statement"] + historical["balance_sheet"]:
            if lbl == label:
                return vals
        return [0] * n

    ebitda       = gv("EBITDA")
    depreciation = gv("Depreciation")
    tax          = gv("Tax")
    ebt          = gv("EBT")
    equity       = gv("Equity Share Capital")
    reserves     = gv("Reserves")
    borrowings   = gv("Borrowings")
    cash_bank    = gv("Cash & Bank")
    sales        = gv("Sales")

    nopat_list, invested_capital_list, roic_list = [], [], []

    for i in range(n):
        ebit = ebitda[i] - depreciation[i]

        # Effective tax rate (capped 0–50%)
        if ebt[i] and ebt[i] > 0:
            tax_rate = min(max(abs(tax[i] / ebt[i]), 0), 0.5)
        else:
            tax_rate = 0.30

        nopat = round(ebit * (1 - tax_rate), 2)

        invested_capital = round(
            equity[i] + reserves[i] + borrowings[i] - cash_bank[i], 2
        )

        roic = round((nopat / invested_capital) * 100, 2) if invested_capital else 0

        nopat_list.append(nopat)
        invested_capital_list.append(invested_capital)
        roic_list.append(roic)

    roic_change = [""] + [
        round(roic_list[i] - roic_list[i - 1], 2) for i in range(1, n)
    ]

    # ROIC vs implicit cost of capital (WACC ~11% default)
    wacc_benchmark = 11.0
    value_creation = [
        round(roic_list[i] - wacc_benchmark, 2) for i in range(n)
    ]

    def classify(r):
        if r > 25:   return "Exceptional ✅"
        if r > 15:   return "Good ✅"
        if r > 10:   return "Fair ⚠️"
        return "Poor ❌"

    latest_roic = roic_list[-1] if roic_list else 0
    avg_roic    = round(sum(roic_list) / n, 2) if n else 0

    return {
        "years": years,
        "roic_data": [
            ("NOPAT (₹ Cr)",          nopat_list),
            ("Invested Capital (₹ Cr)", invested_capital_list),
            ("",                        [""]*n),
            ("ROIC %",                  roic_list),
            ("ROIC Change (pp)",        roic_change),
            ("Value Creation vs WACC",  value_creation),
        ],
        "summary": {
            "latest_roic":       latest_roic,
            "avg_roic":          avg_roic,
            "classification":    classify(latest_roic),
            "wacc_benchmark":    wacc_benchmark,
        },
        "interpretation": {
            "exceptional": "> 25% — Competitive moat likely",
            "good":        "15–25% — Above cost of capital",
            "fair":        "10–15% — Average business",
            "poor":        "< 10%  — May be destroying value",
        }
    }


# ===============================
# PIOTROSKI F-SCORE
# ===============================
def run_piotroski(file):
    """
    9 binary signals summing to score 0–9.
    8–9: Strong buy signal
    5–7: Average / neutral
    0–4: Weak / avoid

    Signals:
    Profitability (4): ROA positive, CFO positive, ROA improving, earnings quality (CFO > NP)
    Leverage (3):      D/E falling, no equity dilution, current ratio improving
    Efficiency (2):    Gross margin improving, asset turnover improving
    """
    historical = run_historical_fs(file)
    years = historical["years"]
    n     = len(years)

    def gv(label):
        for lbl, vals in historical["income_statement"] + historical["balance_sheet"]:
            if lbl == label:
                return vals
        return [0] * n

    net_profit   = gv("Net Profit")
    gross_margin = gv("Gross Margin %")
    sales        = gv("Sales")
    equity       = gv("Equity Share Capital")
    reserves     = gv("Reserves")
    borrowings   = gv("Borrowings")
    receivables  = gv("Receivables")
    inventory    = gv("Inventory")
    cash_bank    = gv("Cash & Bank")

    # Get total assets from balance sheet
    total_assets = gv("Total Assets")

    # Try to get CFO from cash flow section
    cf_cfo = [0] * n
    for lbl, vals in historical.get("cash_flow", []):
        if lbl == "Cash from Operating Activities":
            cf_cfo = vals
            break

    score_rows = []

    for i in range(1, n):  # need prior year for deltas
        ta   = total_assets[i]   or 1
        ta_p = total_assets[i-1] or 1

        roa_cur  = net_profit[i]   / ta
        roa_prev = net_profit[i-1] / ta_p

        eq_cur  = equity[i]   + reserves[i]
        eq_prev = equity[i-1] + reserves[i-1]
        de_cur  = borrowings[i]   / eq_cur  if eq_cur  else 0
        de_prev = borrowings[i-1] / eq_prev if eq_prev else 0

        # Current ratio approximation
        cur_assets_cur  = receivables[i]   + inventory[i]   + cash_bank[i]
        cur_assets_prev = receivables[i-1] + inventory[i-1] + cash_bank[i-1]
        cr_cur  = cur_assets_cur  / (borrowings[i]   + 1)
        cr_prev = cur_assets_prev / (borrowings[i-1] + 1)

        at_cur  = sales[i]   / ta
        at_prev = sales[i-1] / ta_p

        # ── 9 Signals ──────────────────────────────────────────
        f1 = 1 if roa_cur > 0                        else 0  # ROA positive
        f2 = 1 if cf_cfo[i] > 0                      else 0  # CFO positive
        f3 = 1 if roa_cur > roa_prev                 else 0  # ROA improving
        f4 = 1 if cf_cfo[i] > net_profit[i]          else 0  # Earnings quality
        f5 = 1 if de_cur < de_prev                   else 0  # D/E falling
        f6 = 1 if eq_cur >= eq_prev                  else 0  # No dilution
        f7 = 1 if cr_cur > cr_prev                   else 0  # Liquidity improving
        f8 = 1 if gross_margin[i] > gross_margin[i-1] else 0 # Gross margin rising
        f9 = 1 if at_cur > at_prev                   else 0  # Asset turnover rising

        total = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9

        if total >= 8:    grade = "Strong ✅"
        elif total >= 5:  grade = "Average ⚠️"
        else:             grade = "Weak ❌"

        score_rows.append({
            "year":                   years[i],
            "f1_roa_positive":        f1,
            "f2_cfo_positive":        f2,
            "f3_roa_improving":       f3,
            "f4_earnings_quality":    f4,
            "f5_debt_falling":        f5,
            "f6_no_dilution":         f6,
            "f7_liquidity_improving": f7,
            "f8_gross_margin_rising": f8,
            "f9_asset_turnover":      f9,
            "total":                  total,
            "grade":                  grade,
        })

    latest_score = score_rows[-1]["total"] if score_rows else 0
    latest_grade = score_rows[-1]["grade"] if score_rows else "N/A"

    return {
        "years":        [r["year"] for r in score_rows],
        "scores":       score_rows,
        "total_scores": [r["total"] for r in score_rows],
        "grades":       [r["grade"] for r in score_rows],
        "summary": {
            "latest_score": latest_score,
            "latest_grade": latest_grade,
            "signal_definitions": {
                "F1": "ROA > 0 (profitable)",
                "F2": "CFO > 0 (cash generative)",
                "F3": "ROA improved YoY",
                "F4": "CFO > Net Profit (quality earnings)",
                "F5": "Debt/Equity falling",
                "F6": "No equity dilution",
                "F7": "Current ratio improving",
                "F8": "Gross margin improving",
                "F9": "Asset turnover improving",
            },
            "grading": {"8-9": "Strong — Institutional buy signal",
                        "5-7": "Average — Hold / neutral",
                        "0-4": "Weak — Avoid / short signal"},
        }
    }


# ===============================
# DUPONT ANALYSIS (3-FACTOR ROE)
# ===============================
def run_dupont(file):
    """
    DuPont decomposes ROE into 3 drivers:
    ROE = Net Profit Margin × Asset Turnover × Financial Leverage (Equity Multiplier)

    This tells you WHY ROE is high or low:
    - High margin  → quality/pricing power (e.g. luxury brands)
    - High turnover → efficiency (e.g. retail, FMCG)
    - High leverage → financial risk (e.g. banks, NBFCs)
    """
    historical = run_historical_fs(file)
    years = historical["years"]
    n     = len(years)

    def gv(label):
        for lbl, vals in historical["income_statement"] + historical["balance_sheet"]:
            if lbl == label:
                return vals
        return [0] * n

    net_profit   = gv("Net Profit")
    net_margin   = gv("Net Margins")        # already in %
    sales        = gv("Sales")
    total_assets = gv("Total Assets")
    equity       = gv("Equity Share Capital")
    reserves     = gv("Reserves")

    # Factor 1: Net Profit Margin (already computed)
    # Factor 2: Asset Turnover = Sales / Total Assets
    asset_turnover = [
        round(sales[i] / total_assets[i], 4) if total_assets[i] else 0
        for i in range(n)
    ]

    # Factor 3: Equity Multiplier = Total Assets / Total Equity
    total_equity = [equity[i] + reserves[i] for i in range(n)]
    equity_multiplier = [
        round(total_assets[i] / total_equity[i], 4) if total_equity[i] else 0
        for i in range(n)
    ]

    # DuPont ROE = (Net Margin / 100) × Asset Turnover × Equity Multiplier × 100
    roe_dupont = [
        round((net_margin[i] / 100) * asset_turnover[i] * equity_multiplier[i] * 100, 2)
        for i in range(n)
    ]

    # Actual ROE for cross-check
    roe_actual = [
        round((net_profit[i] / total_equity[i]) * 100, 2) if total_equity[i] else 0
        for i in range(n)
    ]

    # Identify dominant driver for latest year
    latest = n - 1
    nm = net_margin[latest]
    at = asset_turnover[latest]
    em = equity_multiplier[latest]

    # Normalise each factor relative to typical benchmark
    nm_score = nm / 10        # 10% net margin = benchmark
    at_score = at / 1.0       # 1.0 asset turnover = benchmark
    em_score = em / 2.0       # 2.0 equity multiplier = benchmark

    dominant = max(
        ("Profit Margins", nm_score),
        ("Asset Efficiency", at_score),
        ("Financial Leverage", em_score),
        key=lambda x: x[1]
    )[0]

    return {
        "years": years,
        "dupont_data": [
            ("Net Profit Margin %",     net_margin),
            ("Asset Turnover (x)",      [round(v, 2) for v in asset_turnover]),
            ("Equity Multiplier (x)",   [round(v, 2) for v in equity_multiplier]),
            ("",                        [""]*n),
            ("ROE via DuPont %",        roe_dupont),
            ("ROE Actual % (verify)",   roe_actual),
        ],
        "summary": {
            "latest_net_margin":      nm,
            "latest_asset_turnover":  round(at, 2),
            "latest_equity_mult":     round(em, 2),
            "latest_roe_dupont":      roe_dupont[latest],
            "latest_roe_actual":      roe_actual[latest],
            "dominant_roe_driver":    dominant,
            "interpretation": (
                "ROE is primarily driven by " + dominant + ". "
                + ("Strong pricing power — sustainable ROE." if dominant == "Profit Margins"
                   else "High efficiency — ROE resilient." if dominant == "Asset Efficiency"
                   else "Leverage-driven ROE — monitor debt risk.")
            )
        }
    }
