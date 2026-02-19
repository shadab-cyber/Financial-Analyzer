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
    
    equity_pct = 100.0
    cash_pct = 0.0
    debt_pct = 0.0
    
    return {
        'equity_pct': round(equity_pct, 2),
        'cash_pct': round(cash_pct, 2),
        'debt_pct': round(debt_pct, 2),
        'holdings': holdings_allocation
    }


def calculate_risk_metrics(portfolio_data, days=252):
    """
    Calculate portfolio risk metrics
    """
    if not LIBRARIES_AVAILABLE or not portfolio_data:
        return {
            'portfolio_beta': None,
            'portfolio_volatility': None,
            'sharpe_ratio': None,
            'max_drawdown': None,
            'note': 'Insufficient data'
        }
    
    try:
        symbols = [h['symbol_full'] for h in portfolio_data]
        weights = [h['current_value'] for h in portfolio_data]
        total_value = sum(weights)
        weights = [w / total_value for w in weights]
        
        # Fetch historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        returns_data = []
        valid_symbols = []
        valid_weights = []
        
        for i, symbol in enumerate(symbols):
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                if len(hist) > 20:
                    returns = hist['Close'].pct_change().dropna()
                    returns_data.append(returns)
                    valid_symbols.append(symbol)
                    valid_weights.append(weights[i])
            except:
                continue
        
        if len(returns_data) < 2:
            return {
                'portfolio_beta': None,
                'portfolio_volatility': None,
                'sharpe_ratio': None,
                'max_drawdown': None,
                'note': 'Insufficient historical data'
            }
        
        # Normalize weights
        weight_sum = sum(valid_weights)
        valid_weights = [w / weight_sum for w in valid_weights]
        
        # Align returns
        returns_df = pd.concat(returns_data, axis=1)
        returns_df.columns = valid_symbols
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 20:
            return {
                'portfolio_beta': None,
                'portfolio_volatility': None,
                'sharpe_ratio': None,
                'max_drawdown': None,
                'note': 'Insufficient aligned data'
            }
        
        # Portfolio returns
        portfolio_returns = returns_df.dot(valid_weights)
        
        # Volatility
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)
        
        # Beta vs NIFTY
        try:
            nifty = yf.Ticker('^NSEI')
            nifty_hist = nifty.history(start=start_date, end=end_date)
            nifty_returns = nifty_hist['Close'].pct_change().dropna()
            
            aligned_data = pd.concat([portfolio_returns, nifty_returns], axis=1).dropna()
            if len(aligned_data) > 20:
                port_ret = aligned_data.iloc[:, 0]
                bench_ret = aligned_data.iloc[:, 1]
                
                covariance = np.cov(port_ret, bench_ret)[0][1]
                benchmark_variance = bench_ret.var()
                portfolio_beta = covariance / benchmark_variance if benchmark_variance != 0 else 1.0
            else:
                portfolio_beta = 1.0
        except:
            portfolio_beta = 1.0
        
        # Sharpe Ratio
        risk_free_rate = 0.07
        avg_return = portfolio_returns.mean() * 252
        sharpe_ratio = (avg_return - risk_free_rate) / portfolio_vol if portfolio_vol != 0 else 0
        
        # Max Drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        return {
            'portfolio_beta': round(portfolio_beta, 2),
            'portfolio_volatility': round(portfolio_vol * 100, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown': round(max_drawdown, 2),
            'avg_annual_return': round(avg_return * 100, 2)
        }
    
    except Exception as e:
        return {
            'portfolio_beta': None,
            'portfolio_volatility': None,
            'sharpe_ratio': None,
            'max_drawdown': None,
            'note': f'Error: {str(e)}'
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


def generate_rebalancing_signals(portfolio_data, allocation):
    """
    Generate rebalancing recommendations
    """
    signals = []
    
    for holding in allocation['holdings']:
        weight = holding['weight']
        
        if weight > 25:
            signals.append({
                'type': 'OVERWEIGHT',
                'symbol': holding['symbol'],
                'current_weight': round(weight, 2),
                'recommendation': f'Reduce position - currently {weight:.1f}% (max recommended: 25%)',
                'action': 'SELL',
                'severity': 'HIGH'
            })
        elif weight > 20:
            signals.append({
                'type': 'CONCENTRATION_RISK',
                'symbol': holding['symbol'],
                'current_weight': round(weight, 2),
                'recommendation': f'Consider reducing - currently {weight:.1f}% (recommended: <20%)',
                'action': 'REVIEW',
                'severity': 'MEDIUM'
            })
    
    # Top 3 concentration
    if len(allocation['holdings']) >= 3:
        top_3_weight = sum(h['weight'] for h in allocation['holdings'][:3])
        if top_3_weight > 60:
            signals.append({
                'type': 'TOP_CONCENTRATION',
                'symbol': 'Portfolio',
                'current_weight': round(top_3_weight, 2),
                'recommendation': f'Top 3 stocks represent {top_3_weight:.1f}% - consider diversifying',
                'action': 'DIVERSIFY',
                'severity': 'MEDIUM'
            })
    
    # Profit booking
    for holding in portfolio_data:
        if holding['gain_loss_pct'] > 30:
            signals.append({
                'type': 'PROFIT_BOOKING',
                'symbol': holding['symbol'],
                'current_weight': round((holding['current_value'] / sum(h['current_value'] for h in portfolio_data)) * 100, 2),
                'recommendation': f'Up {holding["gain_loss_pct"]:.1f}% - consider booking partial profits',
                'action': 'BOOK_PROFIT',
                'severity': 'LOW'
            })
    
    return signals


def compare_with_benchmark(portfolio_data, days=252):
    """
    Compare portfolio with NIFTY 50
    """
    if not LIBRARIES_AVAILABLE or not portfolio_data:
        return {
            'portfolio_return': None,
            'benchmark_return': None,
            'outperformance': None,
            'note': 'Insufficient data'
        }
    
    try:
        # Simplified - use average gain as proxy
        avg_portfolio_return = sum(h['gain_loss_pct'] for h in portfolio_data) / len(portfolio_data)
        
        # Get NIFTY performance
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        nifty = yf.Ticker('^NSEI')
        nifty_hist = nifty.history(start=start_date, end=end_date)
        
        if len(nifty_hist) > 0:
            nifty_return = ((nifty_hist['Close'].iloc[-1] / nifty_hist['Close'].iloc[0]) - 1) * 100
            outperformance = avg_portfolio_return - nifty_return
        else:
            nifty_return = 0
            outperformance = avg_portfolio_return
        
        return {
            'portfolio_return': round(avg_portfolio_return, 2),
            'benchmark_return': round(nifty_return, 2),
            'outperformance': round(outperformance, 2),
            'benchmark': 'NIFTY 50'
        }
    
    except Exception as e:
        return {
            'portfolio_return': None,
            'benchmark_return': None,
            'outperformance': None,
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
        summary = calculate_portfolio_summary(portfolio_data)
        allocation = calculate_asset_allocation(portfolio_data)
        risk_metrics = calculate_risk_metrics(portfolio_data)
        diversification = calculate_diversification(portfolio_data)
        rebalancing = generate_rebalancing_signals(portfolio_data, allocation)
        benchmark = compare_with_benchmark(portfolio_data)
        
        result = {
            'success': True,
            'holdings': portfolio_data,
            'summary': summary,
            'allocation': allocation,
            'risk_metrics': risk_metrics,
            'diversification': diversification,
            'rebalancing_signals': rebalancing,
            'benchmark_comparison': benchmark
        }
        
        # Convert all numpy/pandas types to JSON-serializable types
        return convert_to_json_serializable(result)
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
