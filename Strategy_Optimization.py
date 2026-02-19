"""
Strategy Optimization Module
Advanced strategy backtesting, optimization, and analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
import json

try:
    import yfinance as yf
    from scipy.optimize import minimize
    from scipy.stats import norm
    LIBRARIES_AVAILABLE = True
except ImportError:
    LIBRARIES_AVAILABLE = False
    print("Warning: Required libraries not installed")


# ============================================
# 1. STRATEGY DEFINITION
# ============================================

class Strategy:
    """Base strategy class"""
    
    def __init__(self, name, params):
        self.name = name
        self.params = params
        self.signals = []
        self.trades = []
        
    def generate_signals(self, data):
        """Generate buy/sell signals - to be overridden"""
        raise NotImplementedError
    
    def backtest(self, data, initial_capital=100000):
        """Run backtest on historical data"""
        signals = self.generate_signals(data)
        trades = self.execute_trades(data, signals, initial_capital)
        return self.calculate_metrics(trades, data)
    
    def execute_trades(self, data, signals, initial_capital):
        """Execute trades based on signals"""
        trades = []
        position = None
        capital = initial_capital
        
        for i in range(len(data)):
            if signals[i] == 1 and position is None:  # Buy signal
                # Enter long position
                position = {
                    'entry_date': data.index[i],
                    'entry_price': data['Close'].iloc[i],
                    'shares': capital / data['Close'].iloc[i],
                    'type': 'LONG'
                }
                
            elif signals[i] == -1 and position is not None:  # Sell signal
                # Exit position
                exit_price = data['Close'].iloc[i]
                profit = (exit_price - position['entry_price']) * position['shares']
                profit_pct = ((exit_price / position['entry_price']) - 1) * 100
                
                capital += profit
                
                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': data.index[i],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'shares': position['shares'],
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'capital': capital
                })
                
                position = None
        
        return trades
    
    def calculate_metrics(self, trades, data):
        """Calculate performance metrics"""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'cagr': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'expectancy': 0,
                'avg_return': 0,
                'profit_factor': 0
            }
        
        winning_trades = [t for t in trades if t['profit'] > 0]
        losing_trades = [t for t in trades if t['profit'] <= 0]
        
        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        # Calculate returns
        returns = [t['profit_pct'] for t in trades]
        avg_return = np.mean(returns) if returns else 0
        std_return = np.std(returns) if len(returns) > 1 else 0
        
        # CAGR
        initial_capital = 100000
        final_capital = float(trades[-1]['capital']) if trades else initial_capital
        
        try:
            days = (trades[-1]['exit_date'] - trades[0]['entry_date']).days if len(trades) > 0 else 365
            years = max(days / 365.25, 0.1)  # Minimum 0.1 year to avoid division issues
            cagr = ((final_capital / initial_capital) ** (1 / years) - 1) * 100
        except:
            cagr = 0
        
        # Sharpe Ratio
        risk_free_rate = 7  # 7% for India
        sharpe_ratio = ((avg_return - risk_free_rate) / std_return) if std_return != 0 else 0
        
        # Max Drawdown - Fixed to ensure numeric array
        equity_curve = [float(initial_capital)]
        for trade in trades:
            equity_curve.append(float(trade['capital']))
        
        equity_array = np.array(equity_curve, dtype=float)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max * 100
        max_drawdown = float(np.min(drawdown))
        
        # Expectancy
        avg_win = np.mean([t['profit'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['profit'] for t in losing_trades]) if losing_trades else 0
        expectancy = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)
        
        # Profit factor
        total_profit = sum([t['profit'] for t in winning_trades]) if winning_trades else 0
        total_loss = abs(sum([t['profit'] for t in losing_trades])) if losing_trades else 1
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': round(float(win_rate), 2),
            'cagr': round(float(cagr), 2),
            'sharpe_ratio': round(float(sharpe_ratio), 2),
            'max_drawdown': round(float(max_drawdown), 2),
            'expectancy': round(float(expectancy), 2),
            'avg_return': round(float(avg_return), 2),
            'profit_factor': round(float(profit_factor), 2)
        }


class EMAStrategy(Strategy):
    """EMA Crossover Strategy"""
    
    def __init__(self, fast_period=9, slow_period=21):
        params = {'fast': fast_period, 'slow': slow_period}
        super().__init__('EMA Crossover', params)
        
    def generate_signals(self, data):
        """Generate EMA crossover signals"""
        ema_fast = data['Close'].ewm(span=self.params['fast']).mean()
        ema_slow = data['Close'].ewm(span=self.params['slow']).mean()
        
        signals = np.zeros(len(data))
        
        for i in range(1, len(data)):
            # Buy when fast crosses above slow
            if ema_fast.iloc[i] > ema_slow.iloc[i] and ema_fast.iloc[i-1] <= ema_slow.iloc[i-1]:
                signals[i] = 1
            # Sell when fast crosses below slow
            elif ema_fast.iloc[i] < ema_slow.iloc[i] and ema_fast.iloc[i-1] >= ema_slow.iloc[i-1]:
                signals[i] = -1
        
        return signals


class RSIStrategy(Strategy):
    """RSI Mean Reversion Strategy"""
    
    def __init__(self, period=14, oversold=30, overbought=70):
        params = {'period': period, 'oversold': oversold, 'overbought': overbought}
        super().__init__('RSI Mean Reversion', params)
        
    def generate_signals(self, data):
        """Generate RSI signals"""
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.params['period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.params['period']).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        signals = np.zeros(len(data))
        
        for i in range(1, len(data)):
            # Buy when RSI crosses above oversold
            if rsi.iloc[i] > self.params['oversold'] and rsi.iloc[i-1] <= self.params['oversold']:
                signals[i] = 1
            # Sell when RSI crosses below overbought
            elif rsi.iloc[i] < self.params['overbought'] and rsi.iloc[i-1] >= self.params['overbought']:
                signals[i] = -1
        
        return signals


# ============================================
# 2. PARAMETER OPTIMIZATION
# ============================================

def optimize_parameters(symbol, strategy_type, param_ranges, start_date, end_date):
    """
    Optimize strategy parameters using grid search
    
    Args:
        symbol: Stock symbol
        strategy_type: 'EMA' or 'RSI'
        param_ranges: Dict of parameter ranges (can be lists or ranges)
        start_date: Start date for backtest
        end_date: End date for backtest
    
    Returns:
        Optimization results with best parameters
    """
    if not LIBRARIES_AVAILABLE:
        return {'error': 'Libraries not available'}
    
    try:
        # Fetch data
        ticker = yf.Ticker(f"{symbol}.NS")
        data = ticker.history(start=start_date, end=end_date)
        
        if len(data) < 50:
            return {'error': 'Insufficient data'}
        
        results = []
        
        if strategy_type == 'EMA':
            # Grid search for EMA parameters - handle both lists and ranges
            fast_range = param_ranges.get('fast', range(5, 20, 2))
            slow_range = param_ranges.get('slow', range(20, 60, 5))
            
            # Convert to list if needed
            if not isinstance(fast_range, list):
                fast_range = list(fast_range)
            if not isinstance(slow_range, list):
                slow_range = list(slow_range)
            
            for fast, slow in product(fast_range, slow_range):
                if fast >= slow:
                    continue
                
                try:
                    strategy = EMAStrategy(fast, slow)
                    metrics = strategy.backtest(data)
                    
                    results.append({
                        'fast': int(fast),
                        'slow': int(slow),
                        **metrics
                    })
                except Exception as e:
                    print(f"Error testing EMA({fast}, {slow}): {e}")
                    continue
        
        elif strategy_type == 'RSI':
            # Grid search for RSI parameters - handle both lists and ranges
            period_range = param_ranges.get('period', range(10, 20, 2))
            oversold_range = param_ranges.get('oversold', range(20, 40, 5))
            overbought_range = param_ranges.get('overbought', range(60, 80, 5))
            
            # Convert to list if needed
            if not isinstance(period_range, list):
                period_range = list(period_range)
            if not isinstance(oversold_range, list):
                oversold_range = list(oversold_range)
            if not isinstance(overbought_range, list):
                overbought_range = list(overbought_range)
            
            for period, oversold, overbought in product(period_range, oversold_range, overbought_range):
                if oversold >= overbought:
                    continue
                
                try:
                    strategy = RSIStrategy(period, oversold, overbought)
                    metrics = strategy.backtest(data)
                    
                    results.append({
                        'period': int(period),
                        'oversold': int(oversold),
                        'overbought': int(overbought),
                        **metrics
                    })
                except Exception as e:
                    print(f"Error testing RSI({period}, {oversold}, {overbought}): {e}")
                    continue
                
                results.append({
                    'period': period,
                    'oversold': oversold,
                    'overbought': overbought,
                    **metrics
                })
        
        # Sort by Sharpe ratio (risk-adjusted returns)
        results.sort(key=lambda x: x.get('sharpe_ratio', 0), reverse=True)
        
        # Create heatmap data for visualization
        heatmap_data = create_heatmap_data(results, strategy_type)
        
        return {
            'success': True,
            'total_combinations': len(results),
            'best_params': results[0] if results else None,
            'top_10': results[:10],
            'all_results': results,
            'heatmap_data': heatmap_data
        }
    
    except Exception as e:
        return {'error': str(e)}


def create_heatmap_data(results, strategy_type):
    """Create data for heatmap visualization"""
    if strategy_type == 'EMA':
        # Create 2D heatmap: fast vs slow
        heatmap = {}
        for r in results:
            key = f"{r['fast']},{r['slow']}"
            heatmap[key] = {
                'sharpe': r['sharpe_ratio'],
                'cagr': r['cagr'],
                'win_rate': r['win_rate']
            }
        return heatmap
    
    elif strategy_type == 'RSI':
        # Create 2D heatmap: oversold vs overbought
        heatmap = {}
        for r in results:
            key = f"{r['oversold']},{r['overbought']}"
            heatmap[key] = {
                'sharpe': r['sharpe_ratio'],
                'cagr': r['cagr'],
                'win_rate': r['win_rate']
            }
        return heatmap
    
    return {}


# ============================================
# 3. BACKTESTING ENGINE
# ============================================

def run_backtest(symbol, strategy_config, start_date, end_date, initial_capital=100000):
    """
    Run comprehensive backtest
    
    Args:
        symbol: Stock symbol
        strategy_config: Strategy configuration dict
        start_date: Start date
        end_date: End date
        initial_capital: Starting capital
    
    Returns:
        Complete backtest results
    """
    if not LIBRARIES_AVAILABLE:
        return {'success': False, 'error': 'Required libraries not available. Please install yfinance, pandas, numpy, scipy'}
    
    try:
        print(f"Fetching data for {symbol}.NS from {start_date} to {end_date}")
        
        # Fetch data
        ticker = yf.Ticker(f"{symbol}.NS")
        data = ticker.history(start=start_date, end=end_date)
        
        print(f"Data fetched: {len(data)} rows")
        
        if len(data) < 50:
            return {'success': False, 'error': f'Insufficient data: only {len(data)} days available. Need at least 50 days.'}
        
        # Create strategy
        strategy_type = strategy_config.get('type', 'EMA')
        
        print(f"Creating {strategy_type} strategy with config: {strategy_config}")
        
        if strategy_type == 'EMA':
            fast = int(strategy_config.get('fast', 9))
            slow = int(strategy_config.get('slow', 21))
            strategy = EMAStrategy(fast, slow)
        elif strategy_type == 'RSI':
            period = int(strategy_config.get('period', 14))
            oversold = int(strategy_config.get('oversold', 30))
            overbought = int(strategy_config.get('overbought', 70))
            strategy = RSIStrategy(period, oversold, overbought)
        else:
            return {'success': False, 'error': f'Unknown strategy type: {strategy_type}'}
        
        print("Generating signals...")
        # Run backtest
        signals = strategy.generate_signals(data)
        
        print("Executing trades...")
        trades = strategy.execute_trades(data, signals, initial_capital)
        
        print(f"Trades executed: {len(trades)}")
        
        if len(trades) == 0:
            return {
                'success': False,
                'error': 'No trades generated. Try adjusting strategy parameters or using a different date range.'
            }
        
        print("Calculating metrics...")
        metrics = strategy.calculate_metrics(trades, data)
        
        print("Generating curves...")
        # Generate equity curve
        equity_curve = generate_equity_curve(trades, initial_capital)
        
        # Generate drawdown curve
        drawdown_curve = generate_drawdown_curve(equity_curve)
        
        print("Backtest complete!")
        
        return {
            'success': True,
            'symbol': symbol,
            'strategy': strategy.name,
            'params': strategy.params,
            'metrics': metrics,
            'trades': trades[:100],  # Limit to first 100 trades
            'total_trades_count': len(trades),
            'equity_curve': equity_curve,
            'drawdown_curve': drawdown_curve,
            'period': {
                'start': start_date,
                'end': end_date,
                'days': (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days
            }
        }
    
    except Exception as e:
        import traceback
        print(f"Error in run_backtest: {str(e)}")
        print(traceback.format_exc())
        return {'success': False, 'error': f'Backtest failed: {str(e)}'}


def generate_equity_curve(trades, initial_capital):
    """Generate equity curve data"""
    if not trades:
        return {'dates': [], 'values': []}
    
    # Fixed: dates should be dates, values should be values
    dates = [trades[0]['entry_date'].strftime('%Y-%m-%d')]
    values = [initial_capital]
    
    for trade in trades:
        dates.append(trade['exit_date'].strftime('%Y-%m-%d'))
        values.append(float(trade['capital']))  # Ensure numeric
    
    return {
        'dates': dates,
        'values': values
    }


def generate_drawdown_curve(equity_curve):
    """Generate drawdown curve from equity curve"""
    values = equity_curve['values']
    dates = equity_curve['dates']
    
    if not values or len(values) == 0:
        return {'dates': [], 'values': []}
    
    # Convert to numpy array with proper dtype
    values_array = np.array(values, dtype=float)
    running_max = np.maximum.accumulate(values_array)
    drawdown = ((values_array - running_max) / running_max * 100).tolist()
    
    return {
        'dates': dates,
        'values': drawdown
    }


# ============================================
# 4. WALK-FORWARD ANALYSIS
# ============================================

def walk_forward_analysis(symbol, strategy_config, start_date, end_date, train_period=252, test_period=63):
    """
    Perform walk-forward analysis
    
    Args:
        symbol: Stock symbol
        strategy_config: Strategy configuration
        start_date: Start date
        end_date: End date
        train_period: Training period in days (252 = 1 year)
        test_period: Test period in days (63 = 3 months)
    
    Returns:
        Walk-forward analysis results
    """
    if not LIBRARIES_AVAILABLE:
        return {'error': 'Libraries not available'}
    
    try:
        # Fetch full dataset
        ticker = yf.Ticker(f"{symbol}.NS")
        full_data = ticker.history(start=start_date, end=end_date)
        
        if len(full_data) < train_period + test_period:
            return {'error': 'Insufficient data for walk-forward'}
        
        windows = []
        current_start = 0
        
        while current_start + train_period + test_period <= len(full_data):
            # Training period
            train_data = full_data.iloc[current_start:current_start + train_period]
            
            # Test period
            test_start = current_start + train_period
            test_data = full_data.iloc[test_start:test_start + test_period]
            
            # Optimize on training data
            # (simplified - in production, run full optimization)
            strategy_type = strategy_config.get('type', 'EMA')
            
            if strategy_type == 'EMA':
                strategy = EMAStrategy(strategy_config.get('fast', 9), strategy_config.get('slow', 21))
            else:
                strategy = RSIStrategy(strategy_config.get('period', 14), 
                                      strategy_config.get('oversold', 30),
                                      strategy_config.get('overbought', 70))
            
            # Test on out-of-sample data
            test_metrics = strategy.backtest(test_data)
            
            windows.append({
                'window': len(windows) + 1,
                'train_start': train_data.index[0].strftime('%Y-%m-%d'),
                'train_end': train_data.index[-1].strftime('%Y-%m-%d'),
                'test_start': test_data.index[0].strftime('%Y-%m-%d'),
                'test_end': test_data.index[-1].strftime('%Y-%m-%d'),
                'test_cagr': test_metrics['cagr'],
                'test_sharpe': test_metrics['sharpe_ratio'],
                'test_win_rate': test_metrics['win_rate'],
                'test_max_dd': test_metrics['max_drawdown']
            })
            
            # Move window forward
            current_start += test_period
        
        # Calculate aggregate statistics
        avg_cagr = np.mean([w['test_cagr'] for w in windows])
        avg_sharpe = np.mean([w['test_sharpe'] for w in windows])
        consistency = np.std([w['test_cagr'] for w in windows])
        
        return {
            'success': True,
            'total_windows': len(windows),
            'windows': windows,
            'aggregate': {
                'avg_oos_cagr': round(avg_cagr, 2),
                'avg_oos_sharpe': round(avg_sharpe, 2),
                'consistency_score': round(100 - consistency, 2),
                'profitable_windows': len([w for w in windows if w['test_cagr'] > 0])
            }
        }
    
    except Exception as e:
        return {'error': str(e)}


# ============================================
# 5. MARKET REGIME ANALYSIS
# ============================================

def detect_market_regime(data):
    """
    Detect market regime: Trending/Sideways/High Vol/Low Vol
    
    Args:
        data: OHLCV data
    
    Returns:
        Regime classification
    """
    # Calculate ADX for trend strength
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    
    # Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(14).mean()
    
    # Volatility (using ATR)
    volatility_pct = (atr / close) * 100
    
    # Classify regime
    regimes = []
    
    for i in range(len(data)):
        if i < 28:  # Need 28 periods for indicators
            regimes.append('UNKNOWN')
            continue
        
        adx_val = adx.iloc[i]
        vol_val = volatility_pct.iloc[i]
        
        # Trending if ADX > 25
        if adx_val > 25:
            if vol_val > 3:
                regimes.append('TRENDING_HIGH_VOL')
            else:
                regimes.append('TRENDING_LOW_VOL')
        else:
            # Sideways
            if vol_val > 3:
                regimes.append('SIDEWAYS_HIGH_VOL')
            else:
                regimes.append('SIDEWAYS_LOW_VOL')
    
    return regimes


def analyze_regime_performance(symbol, strategy_config, start_date, end_date):
    """
    Analyze strategy performance across different market regimes
    
    Returns:
        Performance breakdown by regime
    """
    if not LIBRARIES_AVAILABLE:
        return {'error': 'Libraries not available'}
    
    try:
        # Fetch data
        ticker = yf.Ticker(f"{symbol}.NS")
        data = ticker.history(start=start_date, end=end_date)
        
        # Detect regimes
        regimes = detect_market_regime(data)
        data['regime'] = regimes
        
        # Run backtest
        backtest_result = run_backtest(symbol, strategy_config, start_date, end_date)
        
        if not backtest_result.get('success'):
            return backtest_result
        
        trades = backtest_result['trades']
        
        # Classify trades by regime
        regime_performance = {}
        
        for trade in trades:
            # Find regime at entry
            entry_idx = data.index.get_loc(trade['entry_date'], method='nearest')
            regime = data.iloc[entry_idx]['regime']
            
            if regime not in regime_performance:
                regime_performance[regime] = {
                    'trades': [],
                    'total_trades': 0,
                    'winning_trades': 0,
                    'total_return': 0
                }
            
            regime_performance[regime]['trades'].append(trade)
            regime_performance[regime]['total_trades'] += 1
            if trade['profit'] > 0:
                regime_performance[regime]['winning_trades'] += 1
            regime_performance[regime]['total_return'] += trade['profit_pct']
        
        # Calculate metrics per regime
        for regime, perf in regime_performance.items():
            trades_count = perf['total_trades']
            perf['win_rate'] = round(perf['winning_trades'] / trades_count * 100, 2) if trades_count > 0 else 0
            perf['avg_return'] = round(perf['total_return'] / trades_count, 2) if trades_count > 0 else 0
            del perf['trades']  # Remove trades to reduce response size
        
        # Calculate regime distribution
        regime_distribution = data['regime'].value_counts().to_dict()
        total_days = len(data)
        regime_pct = {k: round(v / total_days * 100, 2) for k, v in regime_distribution.items()}
        
        return {
            'success': True,
            'regime_performance': regime_performance,
            'regime_distribution': regime_pct,
            'recommendation': generate_regime_recommendation(regime_performance)
        }
    
    except Exception as e:
        return {'error': str(e)}


def generate_regime_recommendation(regime_performance):
    """Generate recommendation based on regime analysis"""
    best_regime = None
    best_sharpe = -999
    
    for regime, perf in regime_performance.items():
        if perf['win_rate'] > 60 and perf['avg_return'] > 0:
            if best_regime is None or perf['avg_return'] > best_sharpe:
                best_regime = regime
                best_sharpe = perf['avg_return']
    
    if best_regime:
        return f"Strategy works best in {best_regime.replace('_', ' ')} conditions. Consider using only in this regime."
    else:
        return "Strategy shows inconsistent performance across regimes. Consider optimization or different strategy."


# ============================================
# 6. ROBUSTNESS TESTING
# ============================================

def test_robustness(symbol, strategy_config, start_date, end_date):
    """
    Test strategy robustness through parameter variations and stress tests
    
    Returns:
        Robustness analysis results
    """
    if not LIBRARIES_AVAILABLE:
        return {'error': 'Libraries not available'}
    
    try:
        base_result = run_backtest(symbol, strategy_config, start_date, end_date)
        
        if not base_result.get('success'):
            return base_result
        
        base_cagr = base_result['metrics']['cagr']
        base_sharpe = base_result['metrics']['sharpe_ratio']
        
        # Test parameter variations
        variations = []
        strategy_type = strategy_config.get('type', 'EMA')
        
        if strategy_type == 'EMA':
            fast = strategy_config.get('fast', 12)
            slow = strategy_config.get('slow', 26)
            
            # Test nearby parameters (±20%)
            fast_range = range(max(5, int(fast * 0.8)), int(fast * 1.2) + 1, 2)
            slow_range = range(max(10, int(slow * 0.8)), int(slow * 1.2) + 1, 5)
            
            for f, s in product(fast_range, slow_range):
                if f >= s:
                    continue
                
                test_config = strategy_config.copy()
                test_config['fast'] = f
                test_config['slow'] = s
                
                result = run_backtest(symbol, test_config, start_date, end_date)
                
                if result.get('success'):
                    variations.append({
                        'params': f"{f}/{s}",
                        'cagr': result['metrics']['cagr'],
                        'sharpe': result['metrics']['sharpe_ratio'],
                        'deviation_pct': abs(result['metrics']['cagr'] - base_cagr) / base_cagr * 100 if base_cagr != 0 else 0
                    })
        
        # Calculate robustness score
        if variations:
            profitable_variations = len([v for v in variations if v['cagr'] > 0])
            avg_deviation = np.mean([v['deviation_pct'] for v in variations])
            
            robustness_score = (profitable_variations / len(variations)) * 100
            stability_score = max(0, 100 - avg_deviation)
            
            overall_score = (robustness_score + stability_score) / 2
        else:
            overall_score = 0
        
        # Test with slippage and commissions
        slippage_results = test_slippage_impact(symbol, strategy_config, start_date, end_date)
        
        # Monte Carlo simulation
        monte_carlo_results = run_monte_carlo(base_result['trades'], iterations=1000)
        
        return {
            'success': True,
            'base_performance': {
                'cagr': base_cagr,
                'sharpe': base_sharpe
            },
            'parameter_variations': {
                'total_tested': len(variations),
                'profitable': len([v for v in variations if v['cagr'] > 0]),
                'variations': variations[:10]  # Top 10
            },
            'robustness_score': round(overall_score, 2),
            'slippage_analysis': slippage_results,
            'monte_carlo': monte_carlo_results
        }
    
    except Exception as e:
        return {'error': str(e)}


def test_slippage_impact(symbol, strategy_config, start_date, end_date):
    """Test impact of slippage and commissions"""
    slippage_levels = [0, 0.05, 0.10, 0.20]  # 0%, 0.05%, 0.1%, 0.2%
    results = []
    
    base_result = run_backtest(symbol, strategy_config, start_date, end_date)
    
    if not base_result.get('success'):
        return []
    
    base_cagr = base_result['metrics']['cagr']
    
    for slippage in slippage_levels:
        # Adjust CAGR based on slippage (simplified)
        adjusted_cagr = base_cagr * (1 - slippage / 100)
        
        results.append({
            'slippage_pct': slippage,
            'cagr': round(adjusted_cagr, 2),
            'impact': round(base_cagr - adjusted_cagr, 2)
        })
    
    return results


def run_monte_carlo(trades, iterations=1000):
    """Run Monte Carlo simulation on trade sequence"""
    if not trades or len(trades) < 10:
        return {'error': 'Insufficient trades for Monte Carlo'}
    
    trade_returns = [t['profit_pct'] for t in trades]
    
    # Run random simulations
    final_returns = []
    
    for _ in range(iterations):
        # Randomly sample trades with replacement
        sampled_returns = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
        cumulative_return = np.prod([1 + r/100 for r in sampled_returns]) - 1
        final_returns.append(cumulative_return * 100)
    
    # Calculate statistics
    return {
        'mean_return': round(np.mean(final_returns), 2),
        'median_return': round(np.median(final_returns), 2),
        'std_dev': round(np.std(final_returns), 2),
        'percentile_5': round(np.percentile(final_returns, 5), 2),
        'percentile_95': round(np.percentile(final_returns, 95), 2),
        'probability_profit': round(len([r for r in final_returns if r > 0]) / len(final_returns) * 100, 2)
    }


# ============================================
# 7. RISK OPTIMIZATION
# ============================================

def optimize_for_risk(symbol, strategy_type, param_ranges, start_date, end_date, objective='sharpe'):
    """
    Optimize strategy for risk-adjusted returns
    
    Args:
        objective: 'sharpe', 'calmar', 'sortino', 'min_dd'
    
    Returns:
        Risk-optimized parameters
    """
    if not LIBRARIES_AVAILABLE:
        return {'error': 'Libraries not available'}
    
    try:
        # Fetch data
        ticker = yf.Ticker(f"{symbol}.NS")
        data = ticker.history(start=start_date, end=end_date)
        
        results = []
        
        if strategy_type == 'EMA':
            fast_range = param_ranges.get('fast', range(5, 20, 2))
            slow_range = param_ranges.get('slow', range(20, 60, 5))
            
            for fast, slow in product(fast_range, slow_range):
                if fast >= slow:
                    continue
                
                strategy = EMAStrategy(fast, slow)
                metrics = strategy.backtest(data)
                
                # Calculate risk scores
                risk_score = calculate_risk_score(metrics, objective)
                
                results.append({
                    'fast': fast,
                    'slow': slow,
                    'risk_score': risk_score,
                    **metrics
                })
        
        # Sort by risk score
        results.sort(key=lambda x: x.get('risk_score', 0), reverse=True)
        
        return {
            'success': True,
            'objective': objective,
            'best_params': results[0] if results else None,
            'top_10': results[:10],
            'optimization_summary': {
                'total_tested': len(results),
                'best_score': results[0]['risk_score'] if results else 0
            }
        }
    
    except Exception as e:
        return {'error': str(e)}


def calculate_risk_score(metrics, objective):
    """Calculate risk score based on objective"""
    if objective == 'sharpe':
        return metrics.get('sharpe_ratio', 0)
    
    elif objective == 'calmar':
        cagr = metrics.get('cagr', 0)
        max_dd = abs(metrics.get('max_drawdown', 1))
        return cagr / max_dd if max_dd != 0 else 0
    
    elif objective == 'min_dd':
        # Higher score for lower drawdown
        return 100 + metrics.get('max_drawdown', 0)  # max_dd is negative
    
    else:
        return metrics.get('sharpe_ratio', 0)


# ============================================
# 8. POSITION SIZING OPTIMIZATION
# ============================================

def optimize_position_sizing(symbol, strategy_config, start_date, end_date):
    """
    Optimize position sizing methods
    
    Methods:
    - Fixed fractional (10%, 25%, 50%, 100%)
    - Kelly Criterion
    - Volatility-based
    """
    if not LIBRARIES_AVAILABLE:
        return {'error': 'Libraries not available'}
    
    try:
        # Run base backtest
        base_result = run_backtest(symbol, strategy_config, start_date, end_date)
        
        if not base_result.get('success'):
            return base_result
        
        base_trades = base_result['trades']
        win_rate = base_result['metrics']['win_rate'] / 100
        
        # Calculate optimal Kelly
        if base_trades:
            winning_trades = [t for t in base_trades if t['profit'] > 0]
            losing_trades = [t for t in base_trades if t['profit'] < 0]
            
            avg_win = np.mean([t['profit_pct'] for t in winning_trades]) if winning_trades else 0
            avg_loss = abs(np.mean([t['profit_pct'] for t in losing_trades])) if losing_trades else 1
            
            # Kelly formula: (win_rate * avg_win - (1-win_rate) * avg_loss) / avg_win
            kelly_pct = ((win_rate * avg_win) - ((1 - win_rate) * avg_loss)) / avg_win if avg_win != 0 else 0
            kelly_pct = max(0, min(kelly_pct, 1))  # Clamp between 0 and 1
        else:
            kelly_pct = 0
        
        # Test different position sizes
        sizing_methods = {
            'Fixed 10%': simulate_position_size(base_trades, 0.10),
            'Fixed 25%': simulate_position_size(base_trades, 0.25),
            'Fixed 50%': simulate_position_size(base_trades, 0.50),
            'Full Kelly': simulate_position_size(base_trades, kelly_pct),
            'Half Kelly': simulate_position_size(base_trades, kelly_pct / 2),
            'Quarter Kelly': simulate_position_size(base_trades, kelly_pct / 4)
        }
        
        # Rank by Sharpe ratio
        ranked = sorted(sizing_methods.items(), key=lambda x: x[1]['sharpe'], reverse=True)
        
        return {
            'success': True,
            'kelly_percentage': round(kelly_pct * 100, 2),
            'sizing_methods': sizing_methods,
            'recommended': ranked[0][0],
            'comparison': {
                'best_return': max([v['cagr'] for v in sizing_methods.values()]),
                'best_sharpe': max([v['sharpe'] for v in sizing_methods.values()]),
                'lowest_dd': max([v['max_dd'] for v in sizing_methods.values()])  # max because they're negative
            }
        }
    
    except Exception as e:
        return {'error': str(e)}


def simulate_position_size(trades, size_fraction):
    """Simulate portfolio with different position sizing"""
    if not trades:
        return {'cagr': 0, 'sharpe': 0, 'max_dd': 0}
    
    capital = 100000
    equity_curve = [capital]
    
    for trade in trades:
        position_size = capital * size_fraction
        profit = trade['profit_pct'] / 100 * position_size
        capital += profit
        equity_curve.append(capital)
    
    # Calculate metrics
    total_return = (capital / 100000 - 1) * 100
    days = (trades[-1]['exit_date'] - trades[0]['entry_date']).days
    years = days / 365.25
    cagr = ((capital / 100000) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    # Max drawdown
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (np.array(equity_curve) - running_max) / running_max * 100
    max_dd = np.min(drawdown)
    
    # Sharpe (simplified)
    returns = [(equity_curve[i] / equity_curve[i-1] - 1) for i in range(1, len(equity_curve))]
    sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if len(returns) > 0 and np.std(returns) != 0 else 0
    
    return {
        'cagr': round(cagr, 2),
        'sharpe': round(sharpe, 2),
        'max_dd': round(max_dd, 2),
        'final_capital': round(capital, 2)
    }


# ============================================
# 9. STRATEGY COMPARISON & RANKING
# ============================================

def compare_strategies(symbol, strategies_list, start_date, end_date):
    """
    Compare multiple strategies side-by-side
    
    Args:
        strategies_list: List of strategy configs
        
    Returns:
        Comparison results
    """
    if not LIBRARIES_AVAILABLE:
        return {'error': 'Libraries not available'}
    
    try:
        results = []
        
        for strategy_config in strategies_list:
            result = run_backtest(symbol, strategy_config, start_date, end_date)
            
            if result.get('success'):
                results.append({
                    'name': strategy_config.get('name', strategy_config['type']),
                    'type': strategy_config['type'],
                    'params': strategy_config,
                    'metrics': result['metrics']
                })
        
        # Rank by different criteria
        rankings = {
            'by_cagr': sorted(results, key=lambda x: x['metrics']['cagr'], reverse=True),
            'by_sharpe': sorted(results, key=lambda x: x['metrics']['sharpe_ratio'], reverse=True),
            'by_win_rate': sorted(results, key=lambda x: x['metrics']['win_rate'], reverse=True),
            'by_min_dd': sorted(results, key=lambda x: x['metrics']['max_drawdown'], reverse=True)
        }
        
        # Calculate diversification benefit (correlation between strategies)
        # Simplified - would need actual trade-by-trade correlation
        
        return {
            'success': True,
            'total_strategies': len(results),
            'strategies': results,
            'rankings': rankings,
            'winner': {
                'best_return': rankings['by_cagr'][0]['name'] if results else None,
                'best_risk_adjusted': rankings['by_sharpe'][0]['name'] if results else None,
                'most_consistent': rankings['by_win_rate'][0]['name'] if results else None
            }
        }
    
    except Exception as e:
        return {'error': str(e)}


# ============================================
# 10. STRATEGY DEGRADATION MONITORING
# ============================================

def monitor_degradation(symbol, strategy_config, start_date, end_date, window_size=90):
    """
    Monitor strategy performance degradation over time
    
    Args:
        window_size: Rolling window size in days
    
    Returns:
        Degradation analysis
    """
    if not LIBRARIES_AVAILABLE:
        return {'error': 'Libraries not available'}
    
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        data = ticker.history(start=start_date, end=end_date)
        
        # Create strategy
        strategy_type = strategy_config.get('type', 'EMA')
        if strategy_type == 'EMA':
            strategy = EMAStrategy(strategy_config.get('fast', 12), strategy_config.get('slow', 26))
        else:
            strategy = RSIStrategy(strategy_config.get('period', 14), 
                                  strategy_config.get('oversold', 30),
                                  strategy_config.get('overbought', 70))
        
        # Calculate rolling performance
        rolling_metrics = []
        
        for i in range(window_size, len(data), 30):  # Every 30 days
            window_data = data.iloc[i-window_size:i]
            metrics = strategy.backtest(window_data)
            
            rolling_metrics.append({
                'date': window_data.index[-1].strftime('%Y-%m-%d'),
                'cagr': metrics['cagr'],
                'sharpe': metrics['sharpe_ratio'],
                'win_rate': metrics['win_rate']
            })
        
        if len(rolling_metrics) < 2:
            return {'error': 'Insufficient data for degradation analysis'}
        
        # Calculate degradation score
        recent_performance = np.mean([m['cagr'] for m in rolling_metrics[-3:]])  # Last 3 periods
        historical_performance = np.mean([m['cagr'] for m in rolling_metrics[:-3]])  # All except last 3
        
        degradation_pct = ((recent_performance - historical_performance) / abs(historical_performance) * 100) if historical_performance != 0 else 0
        
        # Score: 100 = no degradation, 0 = severe degradation
        degradation_score = max(0, min(100, 100 + degradation_pct))
        
        # Detect regime mismatch
        current_regime = detect_market_regime(data.tail(90))
        regime_stability = len(set(current_regime[-30:])) <= 2  # Stable if <= 2 regimes in last 30 days
        
        # Generate recommendation
        if degradation_score < 50:
            recommendation = "⚠️ PAUSE STRATEGY - Significant performance degradation detected"
            action = "PAUSE"
        elif degradation_score < 70:
            recommendation = "⚠️ REVIEW REQUIRED - Strategy showing signs of degradation"
            action = "REVIEW"
        else:
            recommendation = "✅ CONTINUE - Strategy performing normally"
            action = "CONTINUE"
        
        return {
            'success': True,
            'degradation_score': round(degradation_score, 2),
            'recent_performance': round(recent_performance, 2),
            'historical_performance': round(historical_performance, 2),
            'change_pct': round(degradation_pct, 2),
            'rolling_metrics': rolling_metrics,
            'regime_stable': regime_stability,
            'recommendation': recommendation,
            'action': action
        }
    
    except Exception as e:
        return {'error': str(e)}


# ============================================
# COMPLETE STRATEGY OPTIMIZATION
# ============================================

def run_complete_optimization(symbol, strategy_config, start_date, end_date):
    """
    Run complete strategy optimization pipeline
    
    Args:
        symbol: Stock symbol
        strategy_config: Strategy configuration
        start_date: Start date
        end_date: End date
    
    Returns:
        Complete optimization results
    """
    try:
        results = {}
        
        # 1. Basic backtest
        results['backtest'] = run_backtest(symbol, strategy_config, start_date, end_date)
        
        # 2. Parameter optimization (if requested)
        if strategy_config.get('optimize_params', False):
            param_ranges = strategy_config.get('param_ranges', {})
            results['optimization'] = optimize_parameters(
                symbol,
                strategy_config['type'],
                param_ranges,
                start_date,
                end_date
            )
        
        # 3. Walk-forward analysis
        results['walk_forward'] = walk_forward_analysis(symbol, strategy_config, start_date, end_date)
        
        # 4. Regime analysis
        results['regime_analysis'] = analyze_regime_performance(symbol, strategy_config, start_date, end_date)
        
        # 5. Robustness testing
        results['robustness'] = test_robustness(symbol, strategy_config, start_date, end_date)
        
        # 6. Risk optimization
        if strategy_config.get('optimize_risk', False):
            param_ranges = strategy_config.get('param_ranges', {})
            results['risk_optimization'] = optimize_for_risk(
                symbol,
                strategy_config['type'],
                param_ranges,
                start_date,
                end_date,
                objective=strategy_config.get('risk_objective', 'sharpe')
            )
        
        # 7. Position sizing
        results['position_sizing'] = optimize_position_sizing(symbol, strategy_config, start_date, end_date)
        
        # 8. Degradation monitoring
        results['degradation'] = monitor_degradation(symbol, strategy_config, start_date, end_date)
        
        return {
            'success': True,
            'symbol': symbol,
            'strategy': strategy_config,
            'results': results,
            'summary': generate_optimization_summary(results)
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def generate_optimization_summary(results):
    """Generate executive summary of optimization results"""
    summary = {
        'overall_score': 0,
        'key_findings': [],
        'recommendations': []
    }
    
    # Score components
    scores = []
    
    # Backtest performance
    if results.get('backtest', {}).get('success'):
        metrics = results['backtest']['metrics']
        if metrics['sharpe_ratio'] > 1.5:
            scores.append(90)
            summary['key_findings'].append(f"✅ Excellent Sharpe ratio: {metrics['sharpe_ratio']}")
        elif metrics['sharpe_ratio'] > 1.0:
            scores.append(70)
            summary['key_findings'].append(f"✅ Good Sharpe ratio: {metrics['sharpe_ratio']}")
        else:
            scores.append(50)
            summary['key_findings'].append(f"⚠️ Low Sharpe ratio: {metrics['sharpe_ratio']}")
    
    # Walk-forward consistency
    if results.get('walk_forward', {}).get('success'):
        consistency = results['walk_forward']['aggregate']['consistency_score']
        scores.append(consistency)
        summary['key_findings'].append(f"Walk-forward consistency: {consistency}/100")
    
    # Robustness
    if results.get('robustness', {}).get('success'):
        robustness = results['robustness']['robustness_score']
        scores.append(robustness)
        summary['key_findings'].append(f"Robustness score: {robustness}/100")
    
    # Degradation
    if results.get('degradation', {}).get('success'):
        degradation = results['degradation']['degradation_score']
        scores.append(degradation)
        summary['recommendations'].append(results['degradation']['recommendation'])
    
    # Overall score
    summary['overall_score'] = round(np.mean(scores), 2) if scores else 0
    
    # Final recommendation
    if summary['overall_score'] > 75:
        summary['final_verdict'] = "✅ STRATEGY APPROVED - Ready for live trading"
    elif summary['overall_score'] > 60:
        summary['final_verdict'] = "⚠️ NEEDS IMPROVEMENT - Consider optimization"
    else:
        summary['final_verdict'] = "❌ NOT RECOMMENDED - Significant issues detected"
    
    return summary
