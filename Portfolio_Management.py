"""
Portfolio Management Module
Complete portfolio analysis, risk metrics, and optimization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    import yfinance as yf
    LIBRARIES_AVAILABLE = True
except ImportError:
    LIBRARIES_AVAILABLE = False
    print("Warning: yfinance not installed")




def convert_to_json_serializable(obj):
    """
    Recursively convert numpy/pandas types to Python native types for JSON serialization
    """
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj



# ── NSE symbol aliases ──────────────────────────────────────────────────────
SYMBOL_MAP = {
    'INFOSYS':'INFY','TATA MOTORS':'TATAMOTORS','TATA MOTOR':'TATAMOTORS',
    'HDFC BANK':'HDFCBANK','ICICI BANK':'ICICIBANK',
    'STATE BANK':'SBIN','STATE BANK OF INDIA':'SBIN',
    'RELIANCE INDUSTRIES':'RELIANCE',
    'TATA CONSULTANCY':'TCS','TATA CONSULTANCY SERVICES':'TCS',
    'BHARTI AIRTEL':'BHARTIARTL','AIRTEL':'BHARTIARTL',
    'AXIS BANK':'AXISBANK','KOTAK BANK':'KOTAKBANK',
    'KOTAK MAHINDRA':'KOTAKBANK','KOTAK MAHINDRA BANK':'KOTAKBANK',
    'HUL':'HINDUNILVR','HINDUSTAN UNILEVER':'HINDUNILVR',
    'MARUTI SUZUKI':'MARUTI','MAHINDRA':'M&M',
    "L&T":'LT','LARSEN':'LT','LARSEN AND TOUBRO':'LT','LARSEN & TOUBRO':'LT',
    'ASIAN PAINTS':'ASIANPAINT','BAJAJ FINANCE':'BAJFINANCE',
    'SUN PHARMA':'SUNPHARMA','ADANI ENTERPRISES':'ADANIENT','ADANI':'ADANIENT',
    'POWER GRID':'POWERGRID','COAL INDIA':'COALINDIA',
    'ULTRATECH':'ULTRACEMCO','ULTRATECH CEMENT':'ULTRACEMCO',
    'NESTLE':'NESTLEIND','HERO MOTOCORP':'HEROMOTOCO','HERO':'HEROMOTOCO',
    'TECH MAHINDRA':'TECHM','EICHER MOTORS':'EICHERMOT',
    'SHREE CEMENT':'SHREECEM','PIDILITE':'PIDILITIND',
    'BERGER PAINTS':'BERGEPAINT','DIVIS LAB':'DIVISLAB',
    'DR REDDY':'DRREDDY','APOLLO HOSPITAL':'APOLLOHOSP',
    'BAJAJ AUTO':'BAJAJ-AUTO','VEDANTA':'VEDL',
    'JSW STEEL':'JSWSTEEL','TATA STEEL':'TATASTEEL',
    'INDUSIND BANK':'INDUSINDBK','AVENUE SUPERMARTS':'DMART',
    'TATA MOTORS DVR':'TATAMOTORDVR',
}


def resolve_symbol(raw: str) -> str:
    """Map company name / alias → correct NSE ticker."""
    key = ' '.join(raw.strip().upper().split())   # normalise whitespace
    return SYMBOL_MAP.get(key, key.replace(' ', ''))


def fetch_price_for_symbol(raw_symbol: str):
    """
    Fetch the current market price for one NSE stock.
    Tries yfinance (info → history).  Returns (price, source) or (None, reason).
    Called by the /get-price Flask proxy route.
    """
    symbol = resolve_symbol(raw_symbol)

    if not LIBRARIES_AVAILABLE:
        return None, 'yfinance not installed'

    # ── Try yfinance .info ────────────────────────────────────────────────
    for suffix in ('.NS', '.BO'):
        try:
            t = yf.Ticker(symbol + suffix)
            info = t.info
            for field in ('currentPrice', 'regularMarketPrice'):
                val = info.get(field)
                if val and float(val) > 0:
                    print(f'✅ {symbol}{suffix} via info.{field}: ₹{val}')
                    return float(val), f'yfinance info ({suffix})'
        except Exception as e:
            print(f'   info failed {symbol}{suffix}: {e}')

    # ── Try yfinance .history ─────────────────────────────────────────────
    for suffix in ('.NS', '.BO'):
        for period in ('5d', '1mo'):
            try:
                t = yf.Ticker(symbol + suffix)
                h = t.history(period=period)
                if not h.empty:
                    price = float(h['Close'].iloc[-1])
                    if price > 0:
                        print(f'✅ {symbol}{suffix} via history({period}): ₹{price}')
                        return price, f'yfinance history ({period}{suffix})'
            except Exception as e:
                print(f'   history failed {symbol}{suffix} {period}: {e}')

    return None, 'all sources failed'



def _xirr_newton(cashflows_dates):
    """
    Internal XIRR implementation using Newton-Raphson.
    cashflows_dates: list of (datetime, amount) — negative = investment, positive = current value.
    Returns annualised rate (float) or None on failure.
    """
    if not cashflows_dates or len(cashflows_dates) < 2:
        return None
    dates   = [d for d, _ in cashflows_dates]
    amounts = [a for _, a in cashflows_dates]
    t0      = dates[0]
    years   = [(d - t0).days / 365.25 for d in dates]

    def npv(r):
        return sum(a / (1 + r) ** t for a, t in zip(amounts, years))

    def dnpv(r):
        return sum(-t * a / (1 + r) ** (t + 1) for a, t in zip(amounts, years))

    r = 0.1   # initial guess
    for _ in range(100):
        f  = npv(r)
        df = dnpv(r)
        if df == 0:
            break
        r_new = r - f / df
        if abs(r_new - r) < 1e-6:
            return round(r_new * 100, 2)   # return as %
        r = r_new
    return None


def calculate_xirr(portfolio_data):
    """
    Calculate XIRR (Extended IRR) for each holding and for the portfolio.

    XIRR properly accounts for:
    · When each lot was purchased (buy_date)
    · The size of each purchase
    · The current value today

    Returns a dict with per-holding XIRR% and portfolio-level XIRR%.
    Fallback: simple annualised return if buy_date is missing.
    """
    from datetime import date as _date
    today = datetime.now()
    holding_xirr = []

    for h in portfolio_data:
        buy_date_raw = h.get('buy_date', 'N/A')
        invested     = h['invested_value']
        current      = h['current_value']

        if buy_date_raw and buy_date_raw != 'N/A':
            try:
                buy_dt  = pd.to_datetime(buy_date_raw).to_pydatetime()
                cfs     = [(buy_dt, -invested), (today, current)]
                xi      = _xirr_newton(cfs)
            except Exception:
                xi = None
        else:
            xi = None

        if xi is None:
            # Simple annualised fallback
            years = max((today - pd.to_datetime(buy_date_raw).to_pydatetime()).days / 365.25, 1/365)                      if buy_date_raw and buy_date_raw != 'N/A' else 1
            raw   = (current / invested - 1) if invested else 0
            xi    = round(((1 + raw) ** (1 / years) - 1) * 100, 2)

        holding_xirr.append({'symbol': h['symbol'], 'xirr_pct': xi})

    # Portfolio-level XIRR: one cashflow per holding at its buy_date
    portfolio_cfs = []
    for h in portfolio_data:
        buy_date_raw = h.get('buy_date', 'N/A')
        invested     = h['invested_value']
        if buy_date_raw and buy_date_raw != 'N/A':
            try:
                buy_dt = pd.to_datetime(buy_date_raw).to_pydatetime()
                portfolio_cfs.append((buy_dt, -invested))
            except Exception:
                portfolio_cfs.append((today, -invested))
        else:
            portfolio_cfs.append((today, -invested))

    total_current = sum(h['current_value'] for h in portfolio_data)
    portfolio_cfs.append((today, total_current))
    portfolio_cfs.sort(key=lambda x: x[0])

    port_xirr = _xirr_newton(portfolio_cfs)

    return {
        'portfolio_xirr_pct': port_xirr,
        'holdings': holding_xirr,
    }

def fetch_portfolio_data(holdings):
    """
    Build portfolio metrics.
    Expects each holding to already have `current_price` injected by the
    /get-price proxy route called from the frontend.
    Falls back to yfinance if `current_price` is absent.
    """
    portfolio_data = []

    for holding in holdings:
        try:
            symbol    = holding['symbol']
            quantity  = float(holding['quantity'])
            buy_price = float(holding['buy_price'])

            # Price was fetched by the frontend via /get-price proxy
            if holding.get('current_price') and float(holding['current_price']) > 0:
                current_price = float(holding['current_price'])
                source = holding.get('price_source', 'proxy')
            else:
                # Direct server-side fallback
                current_price, source = fetch_price_for_symbol(symbol)
                if current_price is None:
                    current_price = buy_price
                    source = 'buy_price (all failed)'

            invested_value = quantity * buy_price
            current_value  = quantity * current_price
            gain_loss      = current_value - invested_value
            gain_loss_pct  = ((current_price / buy_price) - 1) * 100 if buy_price else 0

            print(f'💰 {symbol}: buy ₹{buy_price}  CMP ₹{current_price:.2f}  [{source}]')

            portfolio_data.append({
                'symbol':         symbol,
                'symbol_full':    symbol + '.NS',   # full yfinance ticker used by risk_metrics
                'quantity':       quantity,
                'buy_price':      buy_price,
                'buy_date':       holding.get('buy_date', 'N/A'),
                'current_price':  round(current_price, 2),
                'invested_value': round(invested_value, 2),
                'current_value':  round(current_value, 2),
                'gain_loss':      round(gain_loss, 2),
                'gain_loss_pct':  round(gain_loss_pct, 2),
                'price_source':   source,
            })

        except Exception as e:
            print(f'❌ {holding.get("symbol","?")}: {e}')

    return portfolio_data


def calculate_portfolio_summary(portfolio_data):
    """
    Calculate portfolio summary statistics
    """
    if not portfolio_data:
        return {
            'total_invested': 0,
            'total_current_value': 0,
            'total_gain_loss': 0,
            'total_gain_loss_pct': 0,
            'num_holdings': 0
        }
    
    total_invested = sum(h['invested_value'] for h in portfolio_data)
    total_current = sum(h['current_value'] for h in portfolio_data)
    total_gain_loss = total_current - total_invested
    total_gain_loss_pct = ((total_current / total_invested) - 1) * 100 if total_invested > 0 else 0
    
    # Best and worst performers
    sorted_by_performance = sorted(portfolio_data, key=lambda x: x['gain_loss_pct'], reverse=True)
    best = sorted_by_performance[0] if sorted_by_performance else None
    worst = sorted_by_performance[-1] if sorted_by_performance else None
    
    return {
        'total_invested': round(total_invested, 2),
        'total_current_value': round(total_current, 2),
        'total_gain_loss': round(total_gain_loss, 2),
        'total_gain_loss_pct': round(total_gain_loss_pct, 2),
        'num_holdings': len(portfolio_data),
        'best_performer': {
            'symbol': best['symbol'],
            'gain_loss_pct': round(best['gain_loss_pct'], 2)
        } if best else None,
        'worst_performer': {
            'symbol': worst['symbol'],
            'gain_loss_pct': round(worst['gain_loss_pct'], 2)
        } if worst else None
    }


def calculate_asset_allocation(portfolio_data):
    """
    Calculate asset allocation breakdown
    """
    total_value = sum(h['current_value'] for h in portfolio_data)
    
    if total_value == 0:
        return {
            'equity_pct': 0,
            'cash_pct': 100,
            'debt_pct': 0,
            'holdings': []
        }
    
    holdings_allocation = []
    for holding in portfolio_data:
        weight = (holding['current_value'] / total_value) * 100
        holdings_allocation.append({
            'symbol': holding['symbol'],
            'value': round(holding['current_value'], 2),
            'weight': round(weight, 2)
        })
    
    holdings_allocation.sort(key=lambda x: x['weight'], reverse=True)

    # ── Sector classification ─────────────────────────────────────────────────
    # Map NSE tickers to broad sectors.  Unlisted symbols fall into 'Other'.
    _SECTOR_MAP = {
        # Information Technology
        'TCS':'IT','INFY':'IT','WIPRO':'IT','HCLTECH':'IT','TECHM':'IT',
        'MPHASIS':'IT','LTIM':'IT','COFORGE':'IT','PERSISTENT':'IT',
        # Banking & Finance
        'HDFCBANK':'Banking','ICICIBANK':'Banking','SBIN':'Banking',
        'KOTAKBANK':'Banking','AXISBANK':'Banking','INDUSINDBK':'Banking',
        'BANDHANBNK':'Banking','FEDERALBNK':'Banking','IDFCFIRSTB':'Banking',
        'BAJFINANCE':'NBFC','BAJAJFINSV':'NBFC','CHOLAFIN':'NBFC',
        'MUTHOOTFIN':'NBFC','MANAPPURAM':'NBFC',
        # FMCG
        'HINDUNILVR':'FMCG','ITC':'FMCG','NESTLEIND':'FMCG','BRITANNIA':'FMCG',
        'DABUR':'FMCG','MARICO':'FMCG','GODREJCP':'FMCG','EMAMILTD':'FMCG',
        'TATACONSUM':'FMCG','COLPAL':'FMCG',
        # Auto
        'TATAMOTORS':'Auto','MARUTI':'Auto','M&M':'Auto','HEROMOTOCO':'Auto',
        'BAJAJ-AUTO':'Auto','EICHERMOT':'Auto','ASHOKLEY':'Auto',
        'BOSCHLTD':'Auto','MOTHERSON':'Auto',
        # Pharma & Healthcare
        'SUNPHARMA':'Pharma','DRREDDY':'Pharma','CIPLA':'Pharma',
        'DIVISLAB':'Pharma','BIOCON':'Pharma','LUPIN':'Pharma',
        'TORNTPHARM':'Pharma','AUROPHARMA':'Pharma',
        'APOLLOHOSP':'Healthcare','FORTIS':'Healthcare',
        # Energy & Oil
        'RELIANCE':'Energy','ONGC':'Energy','BPCL':'Energy','IOC':'Energy',
        'HINDPETRO':'Energy','GAIL':'Energy','PETRONET':'Energy',
        'NTPC':'Power','POWERGRID':'Power','TATAPOWER':'Power',
        'ADANIGREEN':'Power','TORNTPOWER':'Power',
        # Metals & Mining
        'TATASTEEL':'Metals','JSWSTEEL':'Metals','HINDALCO':'Metals',
        'VEDL':'Metals','NMDC':'Metals','COALINDIA':'Metals',
        'NATIONALUM':'Metals','SAIL':'Metals',
        # Cement & Construction
        'ULTRACEMCO':'Cement','SHREECEM':'Cement','AMBUJACEM':'Cement',
        'ACC':'Cement','RAMCOCEM':'Cement',
        'LT':'Construction','ADANIPORTS':'Infrastructure',
        # Telecom
        'BHARTIARTL':'Telecom',
        # Consumer Durables & Retail
        'TITAN':'ConsumerDurables','ASIANPAINT':'ConsumerDurables',
        'PIDILITIND':'ConsumerDurables','BERGEPAINT':'ConsumerDurables',
        'HAVELLS':'ConsumerDurables','VOLTAS':'ConsumerDurables',
        'DMART':'Retail','TRENT':'Retail',
        # Chemicals
        'SRF':'Chemicals','DEEPAKNTR':'Chemicals','AARTIIND':'Chemicals',
        'NAVINFLUOR':'Chemicals','PIIND':'Chemicals',
    }

    # yfinance sector cache (populated once per run for unmapped symbols)
    _yf_sector_cache = {}

    sector_totals = {}
    for h in holdings_allocation:
        sym = h['symbol']
        sector = _SECTOR_MAP.get(sym)
        if sector is None:
            # Fallback: query yfinance for sector field
            if sym not in _yf_sector_cache:
                try:
                    info = yf.Ticker(sym + '.NS').info
                    _yf_sector_cache[sym] = info.get('sector') or info.get('industry') or 'Other'
                except Exception:
                    _yf_sector_cache[sym] = 'Other'
            sector = _yf_sector_cache[sym]
        sector_totals[sector] = sector_totals.get(sector, 0) + h['weight']

    sector_breakdown = [
        {'sector': sec, 'weight': round(wt, 2)}
        for sec, wt in sorted(sector_totals.items(), key=lambda x: x[1], reverse=True)
    ]

    return {
        'holdings':          holdings_allocation,
        'sector_breakdown':  sector_breakdown,
        'total_value':       round(total_value, 2),
    }


def calculate_risk_metrics(portfolio_data, days=252, risk_free_rate=0.071):
    """
    Calculate portfolio risk metrics.

    Metrics returned:
      · Beta vs NIFTY 50
      · Annualised volatility
      · Sharpe ratio  (total return − Rf) / σ
      · Sortino ratio (total return − Rf) / downside_σ   ← new
      · Max drawdown
      · VaR 95% 1-day (parametric, normal distribution)  ← new
      · CVaR 95% 1-day (expected shortfall beyond VaR)   ← new
      · Correlation matrix between holdings              ← new
    """
    if not LIBRARIES_AVAILABLE or not portfolio_data:
        return {'portfolio_beta': None, 'portfolio_volatility': None,
                'sharpe_ratio': None, 'max_drawdown': None,
                'sortino_ratio': None, 'var_95': None, 'cvar_95': None,
                'correlation_matrix': None, 'note': 'Insufficient data'}

    try:
        symbols      = [h['symbol_full'] for h in portfolio_data]
        raw_weights  = [h['current_value'] for h in portfolio_data]
        total_value  = sum(raw_weights)
        weights      = [w / total_value for w in raw_weights]

        end_date   = datetime.now()
        start_date = end_date - timedelta(days=days)

        returns_data   = []
        valid_symbols  = []
        valid_weights  = []
        short_symbols  = []   # display names (strip .NS/.BO)

        for i, symbol in enumerate(symbols):
            try:
                hist = yf.Ticker(symbol).history(start=start_date, end=end_date)
                if len(hist) > 20:
                    returns_data.append(hist['Close'].pct_change().dropna())
                    valid_symbols.append(symbol)
                    short_symbols.append(symbol.replace('.NS', '').replace('.BO', ''))
                    valid_weights.append(weights[i])
            except Exception:
                continue

        if len(returns_data) < 2:
            return {'portfolio_beta': None, 'portfolio_volatility': None,
                    'sharpe_ratio': None, 'max_drawdown': None,
                    'sortino_ratio': None, 'var_95': None, 'cvar_95': None,
                    'correlation_matrix': None, 'note': 'Insufficient historical data'}

        w_sum = sum(valid_weights)
        valid_weights = [w / w_sum for w in valid_weights]

        returns_df = pd.concat(returns_data, axis=1)
        returns_df.columns = short_symbols
        returns_df = returns_df.dropna()

        if len(returns_df) < 20:
            return {'portfolio_beta': None, 'portfolio_volatility': None,
                    'sharpe_ratio': None, 'max_drawdown': None,
                    'sortino_ratio': None, 'var_95': None, 'cvar_95': None,
                    'correlation_matrix': None, 'note': 'Insufficient aligned data'}

        portfolio_returns = returns_df.dot(valid_weights)

        # ── Volatility ────────────────────────────────────────────────────────
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)

        # ── Beta vs NIFTY ─────────────────────────────────────────────────────
        portfolio_beta = 1.0
        try:
            nifty_hist    = yf.Ticker('^NSEI').history(start=start_date, end=end_date)
            nifty_returns = nifty_hist['Close'].pct_change().dropna()
            aligned       = pd.concat([portfolio_returns, nifty_returns], axis=1).dropna()
            if len(aligned) > 20:
                cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0][1]
                bv  = aligned.iloc[:, 1].var()
                portfolio_beta = cov / bv if bv != 0 else 1.0
        except Exception:
            pass

        # ── Return metrics ────────────────────────────────────────────────────
        avg_return = portfolio_returns.mean() * 252

        # ── Sharpe ────────────────────────────────────────────────────────────
        rf_daily   = risk_free_rate / 252
        excess     = portfolio_returns - rf_daily
        sharpe     = (excess.mean() / portfolio_returns.std() * np.sqrt(252)
                      if portfolio_returns.std() > 0 else 0)

        # ── Sortino ───────────────────────────────────────────────────────────
        downside   = excess[excess < 0]
        down_std   = downside.std() * np.sqrt(252) if len(downside) > 1 else portfolio_vol
        sortino    = ((avg_return - risk_free_rate) / down_std) if down_std > 0 else 0

        # ── Max Drawdown ──────────────────────────────────────────────────────
        cum        = (1 + portfolio_returns).cumprod()
        running_mx = cum.expanding().max()
        drawdown   = (cum - running_mx) / running_mx
        max_dd     = drawdown.min() * 100

        # ── VaR and CVaR (95%, 1-day, parametric) ────────────────────────────
        # VaR = μ - 1.645σ  (one-tailed 5% quantile under normality)
        mu_daily  = portfolio_returns.mean()
        sig_daily = portfolio_returns.std()
        var_95    = round(-(mu_daily - 1.645 * sig_daily) * total_value, 2)   # ₹ loss
        var_95_pct = round((mu_daily - 1.645 * sig_daily) * 100, 3)            # %

        # CVaR = E[loss | loss > VaR]  (parametric: μ - σ * φ(1.645)/0.05)
        from scipy.stats import norm as _norm
        cvar_factor = _norm.pdf(1.645) / 0.05
        cvar_95     = round(-(mu_daily - cvar_factor * sig_daily) * total_value, 2)
        cvar_95_pct = round((mu_daily - cvar_factor * sig_daily) * 100, 3)

        # ── Correlation matrix ────────────────────────────────────────────────
        corr_df     = returns_df.corr().round(3)
        corr_matrix = {
            'labels': short_symbols,
            'matrix': corr_df.values.tolist(),
        }

        return {
            'portfolio_beta':       round(portfolio_beta, 2),
            'portfolio_volatility': round(portfolio_vol * 100, 2),
            'sharpe_ratio':         round(sharpe, 2),
            'sortino_ratio':        round(sortino, 2),
            'max_drawdown':         round(max_dd, 2),
            'avg_annual_return':    round(avg_return * 100, 2),
            'var_95_pct':           var_95_pct,
            'var_95_inr':           var_95,
            'cvar_95_pct':          cvar_95_pct,
            'cvar_95_inr':          cvar_95,
            'correlation_matrix':   corr_matrix,
            'risk_free_rate_used':  risk_free_rate,
        }

    except Exception as e:
        return {
            'portfolio_beta': None, 'portfolio_volatility': None,
            'sharpe_ratio': None, 'max_drawdown': None,
            'sortino_ratio': None, 'var_95': None, 'cvar_95': None,
            'correlation_matrix': None, 'note': f'Error: {str(e)}'
        }


def calculate_diversification(portfolio_data):
    """
    Calculate diversification metrics
    """
    if not portfolio_data:
        return {
            'concentration_score': 0,
            'num_holdings': 0,
            'largest_position_pct': 0,
            'top_5_concentration': 0
        }
    
    total_value = sum(h['current_value'] for h in portfolio_data)
    weights = [h['current_value'] / total_value for h in portfolio_data]
    
    # Herfindahl Index
    herfindahl_index = sum(w**2 for w in weights) * 100
    
    # Largest position
    largest_position_pct = max(weights) * 100
    
    # Top 5 concentration
    sorted_weights = sorted(weights, reverse=True)
    top_5_concentration = sum(sorted_weights[:min(5, len(sorted_weights))]) * 100
    
    # Diversification score
    diversification_score = 100 - herfindahl_index
    
    return {
        'concentration_score': round(herfindahl_index, 2),
        'diversification_score': round(diversification_score, 2),
        'num_holdings': len(portfolio_data),
        'largest_position_pct': round(largest_position_pct, 2),
        'top_5_concentration': round(top_5_concentration, 2)
    }


def generate_rebalancing_signals(portfolio_data, allocation, target_weights=None):
    """
    Generate rebalancing recommendations.

    target_weights: optional dict {symbol: target_pct}, e.g. {'RELIANCE':20,'INFY':15}.
    If absent, equal-weight targets are used.
    """
    signals  = []
    holdings = allocation['holdings']
    total_val = allocation.get('total_value') or sum(h['current_value'] for h in portfolio_data)
    n = len(holdings)

    # Build target dict
    if target_weights and isinstance(target_weights, dict):
        targets   = {sym.upper(): float(pct) for sym, pct in target_weights.items()}
        assigned  = sum(targets.values())
        unassigned = [h['symbol'] for h in holdings if h['symbol'].upper() not in targets]
        remainder  = max(0, 100 - assigned)
        default_t  = round(remainder / len(unassigned), 2) if unassigned else 0
        for sym in unassigned:
            targets[sym.upper()] = default_t
    else:
        eq = round(100 / n, 2) if n else 0
        targets = {h['symbol'].upper(): eq for h in holdings}

    for holding in holdings:
        sym      = holding['symbol'].upper()
        cur_w    = holding['weight']
        tgt_w    = targets.get(sym, round(100 / n, 2))
        dev      = round(cur_w - tgt_w, 2)
        trade_inr = round((dev / 100) * total_val, 0)
        brokerage = round(abs(trade_inr) * 0.001, 0)
        severity  = 'HIGH' if abs(dev) > 10 else 'MEDIUM' if abs(dev) > 5 else 'LOW'
        action    = 'SELL' if dev > 2 else 'BUY' if dev < -2 else 'HOLD'

        signals.append({
            'symbol':         holding['symbol'],
            'current_weight': round(cur_w, 2),
            'target_weight':  round(tgt_w, 2),
            'deviation':      dev,
            'trade_inr':      int(-trade_inr),
            'brokerage_est':  int(brokerage),
            'action':         action,
            'severity':       severity,
            'type': 'OVERWEIGHT' if dev > 5 else 'UNDERWEIGHT' if dev < -5 else 'BALANCED',
            'recommendation': (
                f"{'Reduce' if dev > 2 else 'Increase' if dev < -2 else 'Hold'} "
                f"{holding['symbol']} — {cur_w:.1f}% vs {tgt_w:.1f}% target "
                f"({'overweight' if dev > 2 else 'underweight' if dev < -2 else 'on target'} "
                f"by {abs(dev):.1f}pp)"
            ),
        })

    signals.sort(key=lambda x: abs(x['deviation']), reverse=True)

    # Portfolio-level warnings
    top3 = sum(h['weight'] for h in sorted(holdings, key=lambda x: x['weight'], reverse=True)[:3])
    if top3 > 60:
        signals.append({
            'symbol': 'Portfolio', 'current_weight': round(top3, 2), 'target_weight': 60,
            'deviation': round(top3 - 60, 2), 'trade_inr': 0, 'brokerage_est': 0,
            'action': 'DIVERSIFY', 'severity': 'MEDIUM', 'type': 'TOP_CONCENTRATION',
            'recommendation': f'Top 3 stocks = {top3:.1f}% — consider adding more positions',
        })

    for h in portfolio_data:
        if h['gain_loss_pct'] > 30:
            signals.append({
                'symbol': h['symbol'], 'current_weight': round((h['current_value'] / total_val) * 100, 2),
                'target_weight': None, 'deviation': None, 'trade_inr': 0, 'brokerage_est': 0,
                'action': 'BOOK_PROFIT', 'severity': 'LOW', 'type': 'PROFIT_BOOKING',
                'recommendation': f"{h['symbol']} up {h['gain_loss_pct']:.1f}% — consider booking partial profits",
            })

    return signals


def compare_with_benchmark(portfolio_data, days=252):
    """
    Compare portfolio with NIFTY 50.

    Portfolio return is calculated as the weighted average of each holding's
    return over the SAME period used for the NIFTY fetch (last `days` calendar
    days).  For holdings bought more recently than `start_date`, the holding's
    own buy_date is used as the start so the comparison is apples-to-apples.

    Weighting: current_value weight (i.e. money-weighted, not equal-weighted).
    """
    if not LIBRARIES_AVAILABLE or not portfolio_data:
        return {
            'portfolio_return': None,
            'benchmark_return': None,
            'outperformance':   None,
            'note': 'Insufficient data'
        }

    try:
        end_date   = datetime.now()
        start_date = end_date - timedelta(days=days)

        # ── Fetch NIFTY return over the window ────────────────────────────────
        nifty      = yf.Ticker('^NSEI')
        nifty_hist = nifty.history(start=start_date, end=end_date)

        if len(nifty_hist) < 2:
            nifty_return = 0.0
        else:
            nifty_return = round(
                ((nifty_hist['Close'].iloc[-1] / nifty_hist['Close'].iloc[0]) - 1) * 100, 2
            )

        # ── Weighted portfolio return ─────────────────────────────────────────
        # For each holding we use its actual buy price and current price.
        # If the user supplied a buy_date we use that as the effective holding
        # start (capped to start_date if older) so the period is consistent.
        # The weight is current_value so larger positions have more impact.
        total_value = sum(h['current_value'] for h in portfolio_data)
        if total_value == 0:
            return {
                'portfolio_return': None,
                'benchmark_return': round(nifty_return, 2),
                'outperformance':   None,
                'note': 'Zero portfolio value'
            }

        weighted_return = 0.0
        for h in portfolio_data:
            weight  = h['current_value'] / total_value
            ret_pct = h['gain_loss_pct']   # (current_price / buy_price - 1) * 100

            # Scale to the benchmark window if we have a buy_date
            buy_date_raw = h.get('buy_date', 'N/A')
            if buy_date_raw and buy_date_raw != 'N/A':
                try:
                    buy_dt = pd.to_datetime(buy_date_raw)
                    holding_days = (end_date - buy_dt).days
                    if holding_days > 0 and holding_days < days:
                        # Annualise then scale to the benchmark window length
                        annual_ret  = (1 + ret_pct / 100) ** (365 / holding_days) - 1
                        ret_pct     = ((1 + annual_ret) ** (days / 365) - 1) * 100
                except Exception:
                    pass   # keep raw gain_loss_pct if date parsing fails

            weighted_return += weight * ret_pct

        portfolio_return = round(weighted_return, 2)
        outperformance   = round(portfolio_return - nifty_return, 2)

        return {
            'portfolio_return': portfolio_return,
            'benchmark_return': nifty_return,
            'outperformance':   outperformance,
            'benchmark':        'NIFTY 50',
            'period_days':      days,
        }

    except Exception as e:
        return {
            'portfolio_return': None,
            'benchmark_return': None,
            'outperformance':   None,
            'note': f'Error: {str(e)}'
        }


def run_portfolio_analysis(holdings):
    """
    Complete portfolio analysis
    
    Args:
        holdings: List of holdings dicts
    
    Returns:
        Complete analysis results
    """
    try:
        # Fetch current data
        portfolio_data = fetch_portfolio_data(holdings)
        
        if not portfolio_data:
            return {
                'success': False,
                'error': 'No valid holdings data found'
            }
        
        # Calculate all metrics
        summary        = calculate_portfolio_summary(portfolio_data)
        allocation     = calculate_asset_allocation(portfolio_data)
        risk_metrics   = calculate_risk_metrics(portfolio_data)
        diversification = calculate_diversification(portfolio_data)
        rebalancing    = generate_rebalancing_signals(
                            portfolio_data, allocation,
                            target_weights=holdings[0].get('_target_weights') if holdings else None
                         )
        benchmark      = compare_with_benchmark(portfolio_data)
        xirr_data      = calculate_xirr(portfolio_data)

        # Equity curve vs time (weekly points — 52 per year instead of 365)
        # Daily granularity adds ~15 KB to the response for no visual benefit
        # since the chart renders at ~800px width.
        try:
            from portfolio_transformer import generate_historical_data
            daily_curve  = generate_historical_data(
                current_value=summary['total_current_value'],
                initial_value=summary['total_invested'],
                days=365
            )
            # Downsample to weekly (every 7th point) + always include last point
            equity_curve = daily_curve[::7]
            if daily_curve and equity_curve[-1] != daily_curve[-1]:
                equity_curve.append(daily_curve[-1])
        except Exception:
            equity_curve = []

        result = {
            'success': True,
            'holdings': portfolio_data,
            'summary': {**summary,
                        'portfolio_xirr_pct': xirr_data.get('portfolio_xirr_pct')},
            'allocation':          allocation,
            'risk_metrics':        risk_metrics,
            'diversification':     diversification,
            'rebalancing_signals': rebalancing,
            'benchmark_comparison': benchmark,
            'xirr':                xirr_data,
            'equity_curve':        equity_curve,
        }
        
        # Convert all numpy/pandas types to JSON-serializable types
        return convert_to_json_serializable(result)
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
