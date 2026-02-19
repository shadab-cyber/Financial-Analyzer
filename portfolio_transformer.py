"""
Portfolio to Performance Data Transformer
Converts Portfolio Management data to Performance Analytics format
"""

from datetime import datetime, timedelta
import random


def transform_portfolio_to_performance(portfolio_analysis_result):
    """
    Transform Portfolio Management result into Performance Analytics format
    
    Args:
        portfolio_analysis_result: Result from run_portfolio_analysis()
    
    Returns:
        Dict with holdings_history, holdings_detail, cash_flows
    """
    if not portfolio_analysis_result.get('success'):
        raise ValueError("Portfolio analysis failed")
    
    holdings = portfolio_analysis_result.get('holdings', [])
    summary = portfolio_analysis_result.get('summary', {})
    
    # Generate historical data (simulated)
    holdings_history = generate_historical_data(
        current_value=summary.get('total_current_value', 0),
        initial_value=summary.get('total_invested', 0),
        days=365  # 1 year of history
    )
    
    # Current holdings detail
    holdings_detail = []
    for holding in holdings:
        holdings_detail.append({
            'symbol': holding.get('symbol'),
            'quantity': holding.get('quantity'),
            'buy_price': holding.get('buy_price'),
            'current_price': holding.get('current_price'),
            'buy_date': holding.get('buy_date', '2024-01-01'),
            'gain_loss': holding.get('gain_loss', 0),
            'gain_loss_pct': holding.get('gain_loss_pct', 0),
            'current_value': holding.get('current_value', 0)
        })
    
    # Generate cash flows
    cash_flows = generate_cash_flows(
        initial_investment=summary.get('total_invested', 0),
        holdings=holdings
    )
    
    return {
        'holdings_history': holdings_history,
        'holdings_detail': holdings_detail,
        'cash_flows': cash_flows
    }


def generate_historical_data(current_value, initial_value, days=365):
    """
    Generate simulated daily portfolio history
    """
    if current_value == 0 or initial_value == 0:
        return []
    
    history = []
    
    # Total return
    total_return = (current_value - initial_value) / initial_value
    
    # Start date
    start_date = datetime.now() - timedelta(days=days)
    
    for i in range(days + 1):
        date = start_date + timedelta(days=i)
        
        # Progress ratio
        progress = i / days
        
        # Add some volatility (random walk)
        volatility = random.uniform(-0.02, 0.02) if i > 0 else 0
        
        # Calculate value with trend + volatility
        trend_return = total_return * progress
        value = initial_value * (1 + trend_return + volatility)
        
        # Ensure positive value
        value = max(value, initial_value * 0.5)
        
        history.append({
            'date': date.strftime('%Y-%m-%d'),
            'total_value': round(value, 2)
        })
    
    # Set last value to actual current value
    history[-1]['total_value'] = current_value
    
    return history


def generate_cash_flows(initial_investment, holdings):
    """
    Generate cash flows based on holdings
    """
    cash_flows = []
    
    # Estimate initial investment date (1 year ago)
    oldest_date = datetime.now() - timedelta(days=365)
    
    # Check if holdings have buy_date
    for holding in holdings:
        buy_date_str = holding.get('buy_date', 'N/A')
        if buy_date_str != 'N/A':
            try:
                buy_date = datetime.strptime(buy_date_str, '%Y-%m-%d')
                if buy_date < oldest_date:
                    oldest_date = buy_date
            except:
                pass
    
    # Initial investment
    cash_flows.append({
        'date': oldest_date,
        'amount': initial_investment
    })
    
    return cash_flows


def generate_demo_performance_data():
    """
    Generate demo data for testing Performance Analytics
    """
    # Demo portfolio history (1 year)
    start_date = datetime.now() - timedelta(days=365)
    initial_value = 500000
    current_value = 650000
    
    holdings_history = []
    for i in range(366):
        date = start_date + timedelta(days=i)
        progress = i / 365
        
        # Add realistic market fluctuations
        volatility = random.uniform(-0.015, 0.015)
        trend = (current_value - initial_value) / initial_value * progress
        
        value = initial_value * (1 + trend + volatility)
        
        holdings_history.append({
            'date': date.strftime('%Y-%m-%d'),
            'total_value': round(value, 2)
        })
    
    # Ensure last value is exactly current value
    holdings_history[-1]['total_value'] = current_value
    
    # Demo holdings
    holdings_detail = [
        {
            'symbol': 'RELIANCE',
            'quantity': 50,
            'buy_price': 2400,
            'current_price': 2800,
            'buy_date': '2023-06-15',
            'gain_loss': 20000,
            'gain_loss_pct': 16.67,
            'current_value': 140000
        },
        {
            'symbol': 'TCS',
            'quantity': 30,
            'buy_price': 3200,
            'current_price': 3900,
            'buy_date': '2023-08-20',
            'gain_loss': 21000,
            'gain_loss_pct': 21.88,
            'current_value': 117000
        },
        {
            'symbol': 'HDFC',
            'quantity': 100,
            'buy_price': 1500,
            'current_price': 1750,
            'buy_date': '2023-10-10',
            'gain_loss': 25000,
            'gain_loss_pct': 16.67,
            'current_value': 175000
        },
        {
            'symbol': 'INFY',
            'quantity': 80,
            'buy_price': 1400,
            'current_price': 1650,
            'buy_date': '2024-01-05',
            'gain_loss': 20000,
            'gain_loss_pct': 17.86,
            'current_value': 132000
        },
        {
            'symbol': 'ICICIBANK',
            'quantity': 60,
            'buy_price': 900,
            'current_price': 1100,
            'buy_date': '2024-02-01',
            'gain_loss': 12000,
            'gain_loss_pct': 22.22,
            'current_value': 66000
        }
    ]
    
    # Demo cash flows
    cash_flows = [
        {'date': datetime(2023, 6, 15), 'amount': 120000},
        {'date': datetime(2023, 8, 20), 'amount': 96000},
        {'date': datetime(2023, 10, 10), 'amount': 150000},
        {'date': datetime(2024, 1, 5), 'amount': 112000},
        {'date': datetime(2024, 2, 1), 'amount': 54000}
    ]
    
    return {
        'holdings_history': holdings_history,
        'holdings_detail': holdings_detail,
        'cash_flows': cash_flows,
        'trades': None,
        'signals': None
    }
