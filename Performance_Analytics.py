"""
Performance Analytics Module
Comprehensive portfolio performance analysis with 15+ metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

try:
    import yfinance as yf
    LIBRARIES_AVAILABLE = True
except ImportError:
    LIBRARIES_AVAILABLE = False
    print("Warning: yfinance not installed")




def convert_to_json_serializable(obj):
    """
    Recursively convert numpy/pandas types to Python native types for JSON serialization
    
    Args:
        obj: Any object that might contain numpy/pandas types
    
    Returns:
        JSON-serializable version of the object
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


# ============================================
# 1. OVERALL PORTFOLIO PERFORMANCE
# ============================================

def calculate_portfolio_returns(holdings_history, cash_flows):
    """
    Calculate comprehensive return metrics
    
    Args:
        holdings_history: List of daily portfolio snapshots
        cash_flows: List of deposits/withdrawals with dates
    
    Returns:
        Dictionary with return metrics
    """
    if not holdings_history:
        return {'error': 'No portfolio history'}
    
    # Convert to DataFrame
    df = pd.DataFrame(holdings_history)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Calculate returns
    initial_value = df.iloc[0]['total_value']
    current_value = df.iloc[-1]['total_value']
    
    # Absolute return
    absolute_return = current_value - initial_value
    absolute_return_pct = (absolute_return / initial_value * 100) if initial_value > 0 else 0
    
    # CAGR (Compound Annual Growth Rate)
    days = (df.iloc[-1]['date'] - df.iloc[0]['date']).days
    years = days / 365.25
    cagr = ((current_value / initial_value) ** (1 / years) - 1) * 100 if years > 0 and initial_value > 0 else 0
    
    # XIRR (Internal Rate of Return with cash flows)
    xirr = calculate_xirr(cash_flows, current_value)
    
    # Period-wise returns
    returns = {
        '1D': calculate_period_return(df, days=1),
        '1W': calculate_period_return(df, days=7),
        '1M': calculate_period_return(df, days=30),
        '3M': calculate_period_return(df, days=90),
        '6M': calculate_period_return(df, days=180),
        '1Y': calculate_period_return(df, days=365),
        'YTD': calculate_ytd_return(df),
        'MTD': calculate_mtd_return(df),
        'Since_Inception': absolute_return_pct
    }
    
    # Best/Worst periods
    df['daily_return'] = df['total_value'].pct_change() * 100
    best_day = {
        'date': df.loc[df['daily_return'].idxmax(), 'date'].strftime('%Y-%m-%d'),
        'return': df['daily_return'].max()
    } if len(df) > 1 else None
    
    worst_day = {
        'date': df.loc[df['daily_return'].idxmin(), 'date'].strftime('%Y-%m-%d'),
        'return': df['daily_return'].min()
    } if len(df) > 1 else None
    
    return {
        'total_return_amount': round(absolute_return, 2),
        'total_return_pct': round(absolute_return_pct, 2),
        'cagr': round(cagr, 2),
        'xirr': round(xirr, 2),
        'period_returns': {k: round(v, 2) for k, v in returns.items()},
        'best_day': best_day,
        'worst_day': worst_day,
        'current_value': round(current_value, 2),
        'initial_value': round(initial_value, 2)
    }


def calculate_xirr(cash_flows, final_value):
    """
    Calculate XIRR (Internal Rate of Return)
    """
    if not cash_flows:
        return 0
    
    try:
        from scipy.optimize import newton
        
        dates = [cf['date'] for cf in cash_flows]
        amounts = [cf['amount'] for cf in cash_flows]
        
        # Add final value as last cash flow
        dates.append(datetime.now())
        amounts.append(-final_value)
        
        # Convert to days from start
        start_date = min(dates)
        days = [(d - start_date).days for d in dates]
        
        # XIRR function
        def xirr_func(rate):
            return sum([amt / ((1 + rate) ** (day / 365.25)) for amt, day in zip(amounts, days)])
        
        # Solve for rate
        rate = newton(xirr_func, 0.1)
        return rate * 100
    
    except:
        return 0


def calculate_period_return(df, days):
    """Calculate return for specific period"""
    if len(df) < 2:
        return 0
    
    cutoff_date = df.iloc[-1]['date'] - timedelta(days=days)
    period_df = df[df['date'] >= cutoff_date]
    
    if len(period_df) < 2:
        return 0
    
    start_value = period_df.iloc[0]['total_value']
    end_value = period_df.iloc[-1]['total_value']
    
    return ((end_value / start_value) - 1) * 100 if start_value > 0 else 0


def calculate_ytd_return(df):
    """Year-to-date return"""
    current_year = datetime.now().year
    ytd_df = df[df['date'].dt.year == current_year]
    
    if len(ytd_df) < 2:
        return 0
    
    start_value = ytd_df.iloc[0]['total_value']
    end_value = ytd_df.iloc[-1]['total_value']
    
    return ((end_value / start_value) - 1) * 100 if start_value > 0 else 0


def calculate_mtd_return(df):
    """Month-to-date return"""
    current_month = datetime.now().month
    current_year = datetime.now().year
    mtd_df = df[(df['date'].dt.month == current_month) & (df['date'].dt.year == current_year)]
    
    if len(mtd_df) < 2:
        return 0
    
    start_value = mtd_df.iloc[0]['total_value']
    end_value = mtd_df.iloc[-1]['total_value']
    
    return ((end_value / start_value) - 1) * 100 if start_value > 0 else 0


def generate_equity_curve(holdings_history):
    """Generate equity curve data for charting"""
    df = pd.DataFrame(holdings_history)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    return {
        'dates': df['date'].dt.strftime('%Y-%m-%d').tolist(),
        'values': df['total_value'].tolist()
    }


# ============================================
# 2. RISK-ADJUSTED PERFORMANCE
# ============================================

def calculate_risk_adjusted_metrics(holdings_history):
    """
    Calculate Sharpe, Sortino, Calmar, Volatility, Max Drawdown
    """
    df = pd.DataFrame(holdings_history)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    if len(df) < 2:
        return {'error': 'Insufficient data'}
    
    # Calculate daily returns
    df['daily_return'] = df['total_value'].pct_change()
    returns = df['daily_return'].dropna()
    
    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(252) * 100
    
    # Sharpe Ratio (risk-free rate = 7%)
    risk_free_rate = 0.07
    excess_returns = returns.mean() * 252 - risk_free_rate
    sharpe_ratio = excess_returns / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
    
    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else returns.std() * np.sqrt(252)
    sortino_ratio = excess_returns / downside_std if downside_std != 0 else 0
    
    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    # Calmar Ratio (return / max drawdown)
    annual_return = returns.mean() * 252 * 100
    calmar_ratio = abs(annual_return / max_drawdown) if max_drawdown != 0 else 0
    
    # Value at Risk (VaR 95%)
    var_95 = returns.quantile(0.05) * 100
    
    # Conditional VaR (CVaR - average of worst 5%)
    cvar_95 = returns[returns <= returns.quantile(0.05)].mean() * 100
    
    return {
        'volatility': round(volatility, 2),
        'sharpe_ratio': round(sharpe_ratio, 2),
        'sortino_ratio': round(sortino_ratio, 2),
        'calmar_ratio': round(calmar_ratio, 2),
        'max_drawdown': round(max_drawdown, 2),
        'var_95': round(var_95, 2),
        'cvar_95': round(cvar_95, 2),
        'annual_return': round(annual_return, 2)
    }


# ============================================
# 3. BENCHMARK COMPARISON
# ============================================

def compare_with_benchmark(holdings_history, benchmark='^NSEI'):
    """
    Compare portfolio with benchmark (NIFTY 50)
    """
    if not LIBRARIES_AVAILABLE:
        return {'error': 'Libraries not available'}
    
    df = pd.DataFrame(holdings_history)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    if len(df) < 2:
        return {'error': 'Insufficient data'}
    
    # Portfolio returns
    df['portfolio_return'] = df['total_value'].pct_change()
    
    # Fetch benchmark data
    try:
        start_date = df.iloc[0]['date']
        end_date = df.iloc[-1]['date']
        
        nifty = yf.Ticker(benchmark)
        nifty_hist = nifty.history(start=start_date, end=end_date)
        nifty_hist['bench_return'] = nifty_hist['Close'].pct_change()
        
        # Align dates
        merged = pd.merge(df[['date', 'portfolio_return']], 
                         nifty_hist[['bench_return']], 
                         left_on='date', right_index=True, how='inner')
        
        merged = merged.dropna()
        
        if len(merged) < 2:
            return {'error': 'Cannot align benchmark data'}
        
        # Calculate metrics
        portfolio_ret = merged['portfolio_return']
        bench_ret = merged['bench_return']
        
        # Beta
        covariance = np.cov(portfolio_ret, bench_ret)[0][1]
        bench_variance = bench_ret.var()
        beta = covariance / bench_variance if bench_variance != 0 else 1
        
        # Alpha
        portfolio_annual = portfolio_ret.mean() * 252
        bench_annual = bench_ret.mean() * 252
        risk_free = 0.07
        alpha = (portfolio_annual - risk_free) - beta * (bench_annual - risk_free)
        
        # Tracking Error
        tracking_diff = portfolio_ret - bench_ret
        tracking_error = tracking_diff.std() * np.sqrt(252) * 100
        
        # Information Ratio
        information_ratio = (tracking_diff.mean() / tracking_diff.std() * np.sqrt(252)) if tracking_diff.std() != 0 else 0
        
        # Correlation
        correlation = portfolio_ret.corr(bench_ret)
        
        # R-squared
        r_squared = correlation ** 2
        
        # Total returns
        portfolio_total = ((1 + portfolio_ret).prod() - 1) * 100
        bench_total = ((1 + bench_ret).prod() - 1) * 100
        
        return {
            'portfolio_return': round(portfolio_total, 2),
            'benchmark_return': round(bench_total, 2),
            'alpha': round(alpha * 100, 2),
            'beta': round(beta, 2),
            'tracking_error': round(tracking_error, 2),
            'information_ratio': round(information_ratio, 2),
            'correlation': round(correlation, 2),
            'r_squared': round(r_squared, 2),
            'outperformance': round(portfolio_total - bench_total, 2),
            'benchmark_name': 'NIFTY 50'
        }
    
    except Exception as e:
        return {'error': f'Benchmark comparison failed: {str(e)}'}


# ============================================
# 4. ATTRIBUTION ANALYSIS
# ============================================

def calculate_attribution(holdings_history, holdings_detail):
    """
    Performance attribution: Market, Sector, Stock Selection, Timing
    """
    # This is simplified - full Brinson-Fachler model requires sector benchmarks
    
    df = pd.DataFrame(holdings_history)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    if len(df) < 2:
        return {'error': 'Insufficient data'}
    
    # Portfolio return
    portfolio_return = ((df.iloc[-1]['total_value'] / df.iloc[0]['total_value']) - 1) * 100
    
    # Fetch NIFTY for market effect
    try:
        nifty = yf.Ticker('^NSEI')
        start_date = df.iloc[0]['date']
        end_date = df.iloc[-1]['date']
        nifty_hist = nifty.history(start=start_date, end=end_date)
        
        market_return = ((nifty_hist['Close'].iloc[-1] / nifty_hist['Close'].iloc[0]) - 1) * 100
    except:
        market_return = 0
    
    # Attribution components
    # Market effect: Portfolio Beta * Market Return
    market_effect = market_return  # Simplified
    
    # Stock selection effect: Excess return from picking stocks
    stock_selection = portfolio_return - market_return
    
    # Timing effect: Simplified as variance from buy/sell timing
    timing_effect = 0  # Placeholder - needs transaction data
    
    return {
        'total_return': round(portfolio_return, 2),
        'market_effect': round(market_effect, 2),
        'stock_selection_effect': round(stock_selection, 2),
        'timing_effect': round(timing_effect, 2),
        'sector_allocation_effect': 0,  # Placeholder
        'interaction_effect': round(portfolio_return - market_effect - stock_selection, 2)
    }


# ============================================
# 5. TRADE-LEVEL PERFORMANCE
# ============================================

def analyze_trades(trades):
    """
    Analyze trade-level performance
    
    Args:
        trades: List of completed trades with entry/exit
    """
    if not trades:
        return {'error': 'No trades'}
    
    winning_trades = [t for t in trades if t['profit'] > 0]
    losing_trades = [t for t in trades if t['profit'] < 0]
    
    total_trades = len(trades)
    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    
    # Win rate
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    
    # Average profit/loss
    avg_profit = sum([t['profit'] for t in winning_trades]) / win_count if win_count > 0 else 0
    avg_loss = sum([t['profit'] for t in losing_trades]) / loss_count if loss_count > 0 else 0
    
    # Profit factor
    gross_profit = sum([t['profit'] for t in winning_trades])
    gross_loss = abs(sum([t['profit'] for t in losing_trades]))
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0
    
    # Payoff ratio
    payoff_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
    
    # Expectancy
    expectancy = (win_rate / 100 * avg_profit) + ((100 - win_rate) / 100 * avg_loss)
    
    # Best/Worst trades
    best_trade = max(trades, key=lambda x: x['profit'])
    worst_trade = min(trades, key=lambda x: x['profit'])
    
    return {
        'total_trades': total_trades,
        'winning_trades': win_count,
        'losing_trades': loss_count,
        'win_rate': round(win_rate, 2),
        'avg_profit': round(avg_profit, 2),
        'avg_loss': round(avg_loss, 2),
        'profit_factor': round(profit_factor, 2),
        'payoff_ratio': round(payoff_ratio, 2),
        'expectancy': round(expectancy, 2),
        'best_trade': {
            'symbol': best_trade['symbol'],
            'profit': round(best_trade['profit'], 2)
        },
        'worst_trade': {
            'symbol': worst_trade['symbol'],
            'profit': round(worst_trade['profit'], 2)
        }
    }


# ============================================
# 6. DRAWDOWN & RECOVERY ANALYSIS
# ============================================

def analyze_drawdowns(holdings_history):
    """
    Detailed drawdown analysis
    """
    df = pd.DataFrame(holdings_history)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    if len(df) < 2:
        return {'error': 'Insufficient data'}
    
    # Calculate drawdown
    df['cumulative'] = df['total_value'] / df.iloc[0]['total_value']
    df['running_max'] = df['cumulative'].expanding().max()
    df['drawdown'] = (df['cumulative'] - df['running_max']) / df['running_max'] * 100
    
    # Max drawdown
    max_dd = df['drawdown'].min()
    max_dd_date = df.loc[df['drawdown'].idxmin(), 'date'].strftime('%Y-%m-%d')
    
    # Drawdown duration
    in_drawdown = False
    drawdown_start = None
    drawdowns = []
    
    for idx, row in df.iterrows():
        if row['drawdown'] < 0 and not in_drawdown:
            in_drawdown = True
            drawdown_start = row['date']
        elif row['drawdown'] == 0 and in_drawdown:
            in_drawdown = False
            drawdowns.append({
                'start': drawdown_start,
                'end': row['date'],
                'duration': (row['date'] - drawdown_start).days
            })
    
    # Average drawdown
    avg_dd = df['drawdown'][df['drawdown'] < 0].mean() if len(df['drawdown'][df['drawdown'] < 0]) > 0 else 0
    
    # Drawdown count
    dd_count = len([dd for dd in df['drawdown'] if dd < -10])
    
    return {
        'max_drawdown': round(max_dd, 2),
        'max_drawdown_date': max_dd_date,
        'avg_drawdown': round(avg_dd, 2),
        'drawdown_count_over_10': dd_count,
        'current_drawdown': round(df.iloc[-1]['drawdown'], 2),
        'drawdown_data': df[['date', 'drawdown']].to_dict('records')[-100:]  # Last 100 points
    }


# ============================================
# 7. CONSISTENCY METRICS
# ============================================

def calculate_consistency(holdings_history):
    """
    Calculate consistency and stability metrics
    """
    df = pd.DataFrame(holdings_history)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    if len(df) < 30:
        return {'error': 'Need at least 30 days of data'}
    
    # Monthly returns
    df['month'] = df['date'].dt.to_period('M')
    monthly_returns = df.groupby('month').apply(
        lambda x: ((x['total_value'].iloc[-1] / x['total_value'].iloc[0]) - 1) * 100 if len(x) > 1 else 0
    )
    
    # Profitable months
    profitable_months = (monthly_returns > 0).sum()
    total_months = len(monthly_returns)
    pct_profitable_months = (profitable_months / total_months * 100) if total_months > 0 else 0
    
    # Return stability (coefficient of variation)
    monthly_std = monthly_returns.std()
    monthly_mean = monthly_returns.mean()
    cv = abs(monthly_std / monthly_mean) if monthly_mean != 0 else 0
    
    # Consistency score (0-100)
    consistency_score = 100 - min(cv * 10, 100)
    
    return {
        'profitable_months': profitable_months,
        'total_months': total_months,
        'pct_profitable_months': round(pct_profitable_months, 2),
        'monthly_std': round(monthly_std, 2),
        'consistency_score': round(consistency_score, 2),
        'monthly_returns': monthly_returns.tolist()
    }


# ============================================
# 8. CONTRIBUTION ANALYSIS
# ============================================

def calculate_stock_contribution(holdings_detail, portfolio_return):
    """
    Calculate each stock's contribution to portfolio return and risk
    """
    total_value = sum([h['current_value'] for h in holdings_detail])
    
    contributions = []
    for holding in holdings_detail:
        weight = holding['current_value'] / total_value if total_value > 0 else 0
        stock_return = holding['gain_loss_pct']
        
        # Return contribution
        return_contribution = weight * stock_return
        
        contributions.append({
            'symbol': holding['symbol'],
            'weight': round(weight * 100, 2),
            'return': round(stock_return, 2),
            'return_contribution': round(return_contribution, 2),
            'risk_contribution': round(weight * 100, 2)  # Simplified
        })
    
    contributions.sort(key=lambda x: x['return_contribution'], reverse=True)
    
    return {
        'contributions': contributions,
        'top_contributor': contributions[0] if contributions else None,
        'worst_contributor': contributions[-1] if contributions else None
    }


# ============================================
# 9. BEHAVIOURAL ANALYTICS
# ============================================

def analyze_behavioral_patterns(trades, signals):
    """
    Analyze behavioral patterns and mistakes
    """
    if not trades:
        return {'error': 'No trade data'}
    
    # Overtrading penalty
    recommended_trades = len(signals) if signals else 0
    actual_trades = len(trades)
    overtrading_pct = ((actual_trades - recommended_trades) / recommended_trades * 100) if recommended_trades > 0 else 0
    
    # Early exits (sold before reaching target)
    early_exits = [t for t in trades if t.get('early_exit', False)]
    early_exit_cost = sum([t.get('missed_profit', 0) for t in early_exits])
    
    # Discipline score
    discipline_score = 100 - min(abs(overtrading_pct), 100)
    
    return {
        'overtrading_pct': round(overtrading_pct, 2),
        'early_exits': len(early_exits),
        'early_exit_cost': round(early_exit_cost, 2),
        'discipline_score': round(discipline_score, 2),
        'recommended_trades': recommended_trades,
        'actual_trades': actual_trades
    }


# ============================================
# 10. SCENARIO & STRESS TESTING
# ============================================

def run_stress_scenarios(holdings_detail, portfolio_value):
    """
    Run stress test scenarios
    """
    scenarios = [
        {'name': 'Market -10%', 'market_change': -0.10, 'beta': 1.0},
        {'name': 'Market -20% (Correction)', 'market_change': -0.20, 'beta': 1.0},
        {'name': 'Market -30% (Crash)', 'market_change': -0.30, 'beta': 1.0},
        {'name': '2008 Financial Crisis', 'market_change': -0.52, 'beta': 1.2},
        {'name': '2020 COVID Crash', 'market_change': -0.38, 'beta': 1.15},
        {'name': 'Volatility Spike (+50%)', 'market_change': 0, 'vol_increase': 1.5}
    ]
    
    results = []
    for scenario in scenarios:
        market_change = scenario.get('market_change', 0)
        beta = scenario.get('beta', 1.0)
        
        # Estimate portfolio impact
        portfolio_change = market_change * beta
        new_value = portfolio_value * (1 + portfolio_change)
        loss_amount = new_value - portfolio_value
        
        results.append({
            'scenario': scenario['name'],
            'current_value': round(portfolio_value, 2),
            'stressed_value': round(new_value, 2),
            'loss_amount': round(loss_amount, 2),
            'loss_pct': round(portfolio_change * 100, 2)
        })
    
    return {'scenarios': results}


# ============================================
# 11. TIME-WEIGHTED VS MONEY-WEIGHTED RETURNS
# ============================================

def calculate_twr_vs_mwr(holdings_history, cash_flows):
    """
    Calculate Time-Weighted Return vs Money-Weighted Return (XIRR)
    """
    df = pd.DataFrame(holdings_history)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    if len(df) < 2:
        return {'error': 'Insufficient data'}
    
    # Time-Weighted Return (removes cash flow impact)
    twr = ((df.iloc[-1]['total_value'] / df.iloc[0]['total_value']) - 1) * 100
    
    # Money-Weighted Return (XIRR - includes cash flow timing)
    mwr = calculate_xirr(cash_flows, df.iloc[-1]['total_value'])
    
    # Timing impact (difference)
    timing_impact = mwr - twr
    
    return {
        'time_weighted_return': round(twr, 2),
        'money_weighted_return': round(mwr, 2),
        'timing_impact': round(timing_impact, 2),
        'interpretation': 'Positive = good timing, Negative = bad timing'
    }


# ============================================
# 12. TAX-ADJUSTED PERFORMANCE
# ============================================

def calculate_tax_adjusted_returns(holdings_detail):
    """
    Calculate returns after LTCG/STCG tax
    """
    total_gains = 0
    ltcg_gains = 0
    stcg_gains = 0
    
    for holding in holdings_detail:
        gain = holding.get('gain_loss', 0)
        if gain > 0:
            holding_period_days = (datetime.now() - datetime.strptime(holding.get('buy_date', '2024-01-01'), '%Y-%m-%d')).days
            
            if holding_period_days > 365:
                ltcg_gains += gain
            else:
                stcg_gains += gain
            
            total_gains += gain
    
    # Tax calculation (India)
    ltcg_tax = max(0, (ltcg_gains - 100000) * 0.10) if ltcg_gains > 100000 else 0
    stcg_tax = stcg_gains * 0.15
    total_tax = ltcg_tax + stcg_tax
    
    # After-tax gains
    after_tax_gains = total_gains - total_tax
    tax_efficiency = (after_tax_gains / total_gains * 100) if total_gains > 0 else 0
    
    return {
        'pre_tax_gains': round(total_gains, 2),
        'ltcg_gains': round(ltcg_gains, 2),
        'stcg_gains': round(stcg_gains, 2),
        'ltcg_tax': round(ltcg_tax, 2),
        'stcg_tax': round(stcg_tax, 2),
        'total_tax': round(total_tax, 2),
        'after_tax_gains': round(after_tax_gains, 2),
        'tax_efficiency': round(tax_efficiency, 2)
    }


# ============================================
# COMPLETE PERFORMANCE ANALYSIS
# ============================================

def run_complete_performance_analysis(holdings_history, holdings_detail, cash_flows, trades=None, signals=None):
    """
    Run complete performance analytics
    
    Args:
        holdings_history: Daily portfolio snapshots
        holdings_detail: Current holdings with details
        cash_flows: List of deposits/withdrawals
        trades: Completed trades (optional)
        signals: Trading signals followed (optional)
    
    Returns:
        Complete performance analysis
    """
    try:
        # Calculate portfolio value
        current_value = holdings_history[-1]['total_value'] if holdings_history else 0
        
        # 1. Overall Performance
        overall = calculate_portfolio_returns(holdings_history, cash_flows)
        
        # 2. Risk-Adjusted
        risk_adjusted = calculate_risk_adjusted_metrics(holdings_history)
        
        # 3. Benchmark Comparison
        benchmark = compare_with_benchmark(holdings_history)
        
        # 4. Attribution
        attribution = calculate_attribution(holdings_history, holdings_detail)
        
        # 5. Trade-Level (if trades provided)
        trade_analysis = analyze_trades(trades) if trades else None
        
        # 6. Drawdown
        drawdown = analyze_drawdowns(holdings_history)
        
        # 7. Consistency
        consistency = calculate_consistency(holdings_history)
        
        # 8. Contribution
        contribution = calculate_stock_contribution(holdings_detail, overall.get('total_return_pct', 0))
        
        # 9. Behavioral (if trades and signals provided)
        behavioral = analyze_behavioral_patterns(trades, signals) if trades and signals else None
        
        # 10. Stress Testing
        stress = run_stress_scenarios(holdings_detail, current_value)
        
        # 11. TWR vs MWR
        twr_mwr = calculate_twr_vs_mwr(holdings_history, cash_flows)
        
        # 12. Tax-Adjusted
        tax_adjusted = calculate_tax_adjusted_returns(holdings_detail)
        
        # Equity curve
        equity_curve = generate_equity_curve(holdings_history)
        
        result = {
            'success': True,
            'overall_performance': overall,
            'risk_adjusted': risk_adjusted,
            'benchmark_comparison': benchmark,
            'attribution': attribution,
            'trade_analysis': trade_analysis,
            'drawdown': drawdown,
            'consistency': consistency,
            'contribution': contribution,
            'behavioral': behavioral,
            'stress_testing': stress,
            'twr_vs_mwr': twr_mwr,
            'tax_adjusted': tax_adjusted,
            'equity_curve': equity_curve
        }
        
        # Convert all numpy/pandas types to JSON-serializable types
        return convert_to_json_serializable(result)
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
