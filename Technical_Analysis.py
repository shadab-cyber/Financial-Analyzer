"""
Technical Analysis Module
Fetches stock data, calculates indicators, and generates trading signals
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# NOTE: Install these libraries when deploying:
# pip install yfinance ta plotly scikit-learn --break-system-packages

try:
    import yfinance as yf
    import ta
    from sklearn.ensemble import RandomForestClassifier
    LIBRARIES_AVAILABLE = True
except ImportError:
    LIBRARIES_AVAILABLE = False
    print("Warning: yfinance, ta, or sklearn not installed. Install with:")
    print("pip install yfinance ta scikit-learn --break-system-packages")

# ── In-memory response cache ──────────────────────────────────────────────────
# Keys:   "{SYMBOL}|{timeframe}|{date_str}"   (date_str = today's date for daily,
#          or today+current_hour for intraday ≤1h, for a tighter TTL bucket)
# Values: (timestamp_of_storage, result_dict)
# TTL:    5 minutes — balances freshness vs Yahoo Finance rate limits.
# This is process-local (reset on Render dyno restart), which is fine;
# the benefit is avoiding redundant downloads within the same session.
_CACHE: dict = {}
_CACHE_TTL_SECONDS = 300   # 5 minutes


def fetch_stock_data(symbol, timeframe='1d', period='1y'):
    """
    Fetch stock data from Yahoo Finance
    
    Args:
        symbol: Stock ticker (e.g., 'TATAMOTORS', 'RELIANCE')
        timeframe: '1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo'
        period: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y'
    
    Returns:
        DataFrame with OHLCV data
    """
    if not LIBRARIES_AVAILABLE:
        raise ImportError("Required libraries not installed")
    
    try:
        from datetime import datetime, timedelta
        
        # Clean symbol
        symbol = symbol.strip().upper()
        
        # Remove any existing exchange suffix
        symbol = symbol.replace('.NS', '').replace('.BO', '').replace('.BSE', '').replace('.NSE', '')
        
        # CRITICAL: Validate timeframe/period combinations
        # Intraday data (1m, 5m, 15m, 30m, 1h) only available for last 60 days
        intraday_intervals = ['1m', '5m', '15m', '30m', '60m', '1h']
        
        if timeframe in intraday_intervals:
            # For intraday, restrict period to max 60 days
            valid_intraday_periods = ['1d', '5d', '1mo']
            if period not in valid_intraday_periods:
                # Auto-correct to 1 month for intraday
                print(f"Warning: {timeframe} timeframe only supports short periods. Changing period from {period} to 1mo")
                period = '1mo'
        
        # Convert period to actual date range for better reliability
        end_date = datetime.now()
        
        period_to_days = {
            '1d': 1,
            '5d': 7,  # Add extra days for weekends
            '1mo': 35,  # ~1 month + buffer
            '3mo': 100,  # ~3 months + buffer
            '6mo': 200,  # ~6 months + buffer
            '1y': 400,  # ~1 year + buffer
            '2y': 800,  # ~2 years + buffer
            '5y': 2000  # ~5 years + buffer
        }
        
        days_back = period_to_days.get(period, 400)
        start_date = end_date - timedelta(days=days_back)

        # ── Symbol normalisation ──────────────────────────────────────────────
        # A small alias table for the handful of cases where a common shorthand
        # differs from the actual NSE ticker.  yfinance handles the rest once
        # we append .NS — no need for a 90-entry exhaustive dict.
        _ALIASES = {
            'TATA MOTORS': 'TATAMOTORS',
            'INFOSYS':     'INFY',
            'HDFC':        'HDFCBANK',
            'ICICI':       'ICICIBANK',
            'SBI':         'SBIN',
            'AIRTEL':      'BHARTIARTL',
            'KOTAK':       'KOTAKBANK',
            'HCL':         'HCLTECH',
            'BAJAJ':       'BAJFINANCE',
            'NESTLE':      'NESTLEIND',
            'MAHINDRA':    'M&M',
            'HUL':         'HINDUNILVR',
            'AXIS':        'AXISBANK',
        }
        symbol = _ALIASES.get(symbol, symbol)   # apply alias if found

        # Try NSE first (most common for Indian stocks)
        symbols_to_try = [
            f"{symbol}.NS",   # NSE
            f"{symbol}.BO",   # BSE
            symbol            # US stocks / already-suffixed
        ]

        data = None
        last_error = None
        successful_symbol = None

        for test_symbol in symbols_to_try:
            try:
                print(f"Trying {test_symbol} with {timeframe} interval from {start_date.date()} to {end_date.date()}")
                
                ticker = yf.Ticker(test_symbol)
                
                # Use date range instead of period for better reliability
                data = ticker.history(start=start_date, end=end_date, interval=timeframe)
                
                if not data.empty and len(data) > 0:
                    successful_symbol = test_symbol
                    # Data found, clean and return
                    data = data.reset_index()
                    
                    # Handle different column names
                    column_renames = {}
                    for col in data.columns:
                        col_lower = col.lower()
                        if col_lower == 'datetime':
                            column_renames[col] = 'Date'
                        else:
                            column_renames[col] = col.capitalize()
                    
                    data = data.rename(columns=column_renames)
                    
                    print(f"✅ Successfully fetched {len(data)} candles for {successful_symbol}")
                    return data
                    
            except Exception as e:
                last_error = str(e)
                print(f"❌ Failed for {test_symbol}: {str(e)}")
                continue
        
        # If we get here, no data was found
        raise ValueError(
            f"No data found for {symbol} with timeframe {timeframe}. "
            f"Try: (1) Different timeframe (use 5m or 15m with 1mo), "
            f"(2) Different stock (RELIANCE, TCS, INFY work best), "
            f"(3) Check symbol spelling"
        )
    
    except Exception as e:
        error_msg = str(e)
        
        # Provide helpful error messages
        if "No data found" in error_msg or "delisted" in error_msg.lower():
            raise Exception(
                f"Unable to fetch data for '{symbol}' with timeframe '{timeframe}' and period '{period}'. "
                f"This combination may not have data available. "
                f"Try: 5m with 1mo, or 15m with 1mo (these work reliably for Indian stocks)"
            )
        else:
            raise Exception(f"Error fetching data: {error_msg}")


def calculate_technical_indicators(data):
    """
    Calculate technical indicators
    
    Args:
        data: DataFrame with OHLCV columns
    
    Returns:
        DataFrame with added indicator columns
    """
    if not LIBRARIES_AVAILABLE:
        raise ImportError("Required libraries not installed")
    
    df = data.copy()
    
    if len(df) < 20:
        raise ValueError(f"Insufficient data: only {len(df)} candles. Need at least 20 for basic analysis.")
    
    try:
        # MOMENTUM INDICATORS
        # RSI (Relative Strength Index)
        if len(df) >= 14:
            df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        else:
            df['RSI'] = None
        
        # Stochastic Oscillator
        if len(df) >= 14:
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()
        else:
            df['Stoch_K'] = None
            df['Stoch_D'] = None
        
        # TREND INDICATORS
        # MACD (Moving Average Convergence Divergence)
        if len(df) >= 26:
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Hist'] = macd.macd_diff()
        else:
            df['MACD'] = None
            df['MACD_Signal'] = None
            df['MACD_Hist'] = None
        
        # Moving Averages
        if len(df) >= 20:
            df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
        else:
            df['SMA_20'] = df['Close'].rolling(window=min(10, len(df))).mean()
            
        if len(df) >= 50:
            df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
        else:
            df['SMA_50'] = df['Close'].rolling(window=min(20, len(df))).mean()
            
        if len(df) >= 200:
            df['SMA_200'] = ta.trend.SMAIndicator(df['Close'], window=200).sma_indicator()
        else:
            df['SMA_200'] = None
            
        if len(df) >= 12:
            df['EMA_12'] = ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator()
        else:
            df['EMA_12'] = df['Close'].ewm(span=min(6, len(df))).mean()
            
        if len(df) >= 26:
            df['EMA_26'] = ta.trend.EMAIndicator(df['Close'], window=26).ema_indicator()
        else:
            df['EMA_26'] = df['Close'].ewm(span=min(12, len(df))).mean()
        
        # ADX (Average Directional Index)
        if len(df) >= 14:
            df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
        else:
            df['ADX'] = None
        
        # VOLATILITY INDICATORS
        # Bollinger Bands
        if len(df) >= 20:
            bollinger = ta.volatility.BollingerBands(df['Close'])
            df['BB_Upper'] = bollinger.bollinger_hband()
            df['BB_Middle'] = bollinger.bollinger_mavg()
            df['BB_Lower'] = bollinger.bollinger_lband()
            df['BB_Width'] = bollinger.bollinger_wband()
        else:
            df['BB_Upper'] = None
            df['BB_Middle'] = None
            df['BB_Lower'] = None
            df['BB_Width'] = None
        
        # ATR (Average True Range)
        if len(df) >= 14:
            df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        else:
            df['ATR'] = None
        
        # VOLUME INDICATORS
        # OBV (On-Balance Volume)
        if len(df) >= 10:
            df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        else:
            df['OBV'] = None
        
        # Volume Moving Average
        if len(df) >= 20:
            df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        else:
            df['Volume_SMA_20'] = df['Volume'].rolling(window=min(10, len(df))).mean()
        
        # Money Flow Index
        if len(df) >= 14:
            df['MFI'] = ta.volume.MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume']).money_flow_index()
        else:
            df['MFI'] = None

        # ── VWAP (Volume-Weighted Average Price) ─────────────────────────────
        # Always computed; run_technical_analysis decides whether to expose it
        # based on timeframe. On intraday data (1m–1h) it resets each session.
        # On daily data it becomes a cumulative average, less meaningful but
        # still valid as a trend filter.
        _tp        = (df['High'] + df['Low'] + df['Close']) / 3
        _cum_tpvol = (_tp * df['Volume']).cumsum()
        _cum_vol   = df['Volume'].cumsum()
        df['VWAP'] = (_cum_tpvol / _cum_vol.replace(0, np.nan)).round(2)

        return df
    
    except Exception as e:
        # Return data with at least basic indicators
        print(f"Warning: Some indicators failed: {str(e)}")
        # Add basic moving averages as fallback
        if 'SMA_20' not in df.columns:
            df['SMA_20'] = df['Close'].rolling(window=min(20, len(df))).mean()
        if 'SMA_50' not in df.columns:
            df['SMA_50'] = df['Close'].rolling(window=min(20, len(df))).mean()
        return df


def detect_candlestick_patterns(data):
    """
    Detect candlestick patterns
    
    Args:
        data: DataFrame with OHLCV data
    
    Returns:
        List of detected patterns with details
    """
    df = data.copy().reset_index(drop=True)
    patterns_detected = []

    if len(df) < 3:
        return patterns_detected

    # ── Helper: scan the last `lookback` bars for patterns ───────────────────
    # We check each valid window position, not just the final 3 candles.
    # The most recent detected occurrence of each pattern is reported.
    lookback = min(len(df), 30)   # scan last 30 candles
    start_i  = len(df) - lookback

    # Trend context: measured over last 6 closes ending at bar i
    def _downtrend(i):
        if i < 5: return False
        closes = df['Close'].iloc[i-5:i+1].values
        return closes[-1] < closes[0]

    def _uptrend(i):
        if i < 5: return False
        closes = df['Close'].iloc[i-5:i+1].values
        return closes[-1] > closes[0]

    # Track which patterns already found so we only report the most recent hit
    found = set()

    def _add(pat_dict):
        key = pat_dict['pattern']
        if key not in found:
            found.add(key)
            patterns_detected.append(pat_dict)

    # Scan from most-recent backwards so first match = most recent occurrence
    for i in range(len(df) - 1, max(start_i, 2) - 1, -1):
        c2 = df.iloc[i]       # current / signal bar
        c1 = df.iloc[i - 1]  # one bar ago
        c0 = df.iloc[i - 2]  # two bars ago

        body2 = abs(c2['Close'] - c2['Open'])
        body1 = abs(c1['Close'] - c1['Open'])
        body0 = abs(c0['Close'] - c0['Open'])
        hi2   = c2['High'];  lo2 = c2['Low']
        hi1   = c1['High'];  lo1 = c1['Low']
        upper2 = hi2 - max(c2['Open'], c2['Close'])
        lower2 = min(c2['Open'], c2['Close']) - lo2
        upper1 = hi1 - max(c1['Open'], c1['Close'])
        lower1 = min(c1['Open'], c1['Close']) - lo1
        range2 = hi2 - lo2
        avg_body = (body0 + body1 + body2) / 3 if (body0 + body1 + body2) > 0 else 1

        # ── SINGLE-BAR BULLISH ────────────────────────────────────────────────

        # Hammer — long lower shadow, small body, any colour, after downtrend
        if (body2 > 0 and lower2 >= 2 * body2 and
                upper2 <= body2 * 0.3 and _downtrend(i)):
            _add({'pattern': 'Hammer', 'type': 'Bullish Reversal', 'strength': 'Medium',
                  'description': 'Long lower shadow after downtrend — buyers rejecting lower prices'})

        # Inverted Hammer — long upper shadow, small body, after downtrend
        if (body2 > 0 and upper2 >= 2 * body2 and
                lower2 <= body2 * 0.3 and _downtrend(i)):
            _add({'pattern': 'Inverted Hammer', 'type': 'Bullish Reversal', 'strength': 'Weak',
                  'description': 'Long upper shadow after downtrend — potential buying attempt'})

        # ── SINGLE-BAR BEARISH ────────────────────────────────────────────────

        # Shooting Star — long upper shadow, small body, after uptrend
        if (body2 > 0 and upper2 >= 2 * body2 and
                lower2 <= body2 * 0.3 and _uptrend(i)):
            _add({'pattern': 'Shooting Star', 'type': 'Bearish Reversal', 'strength': 'Medium',
                  'description': 'Long upper shadow after uptrend — sellers rejecting higher prices'})

        # Hanging Man — looks like Hammer but appears after uptrend (bearish)
        if (body2 > 0 and lower2 >= 2 * body2 and
                upper2 <= body2 * 0.3 and _uptrend(i)):
            _add({'pattern': 'Hanging Man', 'type': 'Bearish Reversal', 'strength': 'Weak',
                  'description': 'Hammer shape after uptrend — potential distribution'})

        # Doji — body < 10% of range
        if range2 > 0 and body2 < range2 * 0.1:
            _add({'pattern': 'Doji', 'type': 'Indecision', 'strength': 'Neutral',
                  'description': 'Open ≈ Close — market indecision, watch for breakout direction'})

        # ── TWO-BAR BULLISH ───────────────────────────────────────────────────

        # Bullish Engulfing
        if (c1['Close'] < c1['Open'] and c2['Close'] > c2['Open'] and
                c2['Open'] <= c1['Close'] and c2['Close'] >= c1['Open']):
            _add({'pattern': 'Bullish Engulfing', 'type': 'Bullish Reversal', 'strength': 'Strong',
                  'description': 'Green candle fully engulfs prior red candle — strong buying'})

        # Bullish Harami — small green inside prior large red
        if (c1['Close'] < c1['Open'] and c2['Close'] > c2['Open'] and
                c2['Open'] > c1['Close'] and c2['Close'] < c1['Open'] and
                body2 < body1 * 0.5):
            _add({'pattern': 'Bullish Harami', 'type': 'Bullish Reversal', 'strength': 'Medium',
                  'description': 'Small green candle inside large red candle — momentum slowing'})

        # Piercing Line — red then green closing above midpoint of red
        if (c1['Close'] < c1['Open'] and c2['Close'] > c2['Open'] and
                c2['Open'] < c1['Low'] and
                c2['Close'] > (c1['Open'] + c1['Close']) / 2 and
                c2['Close'] < c1['Open']):
            _add({'pattern': 'Piercing Line', 'type': 'Bullish Reversal', 'strength': 'Medium',
                  'description': 'Green close pierces more than halfway into prior red candle'})

        # Tweezer Bottom — two candles with matching lows after downtrend
        if (_downtrend(i) and abs(lo2 - lo1) / max(lo1, 0.01) < 0.002 and
                c1['Close'] < c1['Open'] and c2['Close'] > c2['Open']):
            _add({'pattern': 'Tweezer Bottom', 'type': 'Bullish Reversal', 'strength': 'Medium',
                  'description': 'Matching lows after downtrend — double rejection of support level'})

        # ── TWO-BAR BEARISH ───────────────────────────────────────────────────

        # Bearish Engulfing
        if (c1['Close'] > c1['Open'] and c2['Close'] < c2['Open'] and
                c2['Open'] >= c1['Close'] and c2['Close'] <= c1['Open']):
            _add({'pattern': 'Bearish Engulfing', 'type': 'Bearish Reversal', 'strength': 'Strong',
                  'description': 'Red candle fully engulfs prior green candle — strong selling'})

        # Bearish Harami — small red inside prior large green
        if (c1['Close'] > c1['Open'] and c2['Close'] < c2['Open'] and
                c2['Open'] < c1['Close'] and c2['Close'] > c1['Open'] and
                body2 < body1 * 0.5):
            _add({'pattern': 'Bearish Harami', 'type': 'Bearish Reversal', 'strength': 'Medium',
                  'description': 'Small red candle inside large green candle — momentum slowing'})

        # Dark Cloud Cover — green then red closing below midpoint of green
        if (c1['Close'] > c1['Open'] and c2['Close'] < c2['Open'] and
                c2['Open'] > c1['High'] and
                c2['Close'] < (c1['Open'] + c1['Close']) / 2 and
                c2['Close'] > c1['Open']):
            _add({'pattern': 'Dark Cloud Cover', 'type': 'Bearish Reversal', 'strength': 'Medium',
                  'description': 'Red close pierces more than halfway into prior green candle'})

        # Tweezer Top — two candles with matching highs after uptrend
        if (_uptrend(i) and abs(hi2 - hi1) / max(hi1, 0.01) < 0.002 and
                c1['Close'] > c1['Open'] and c2['Close'] < c2['Open']):
            _add({'pattern': 'Tweezer Top', 'type': 'Bearish Reversal', 'strength': 'Medium',
                  'description': 'Matching highs after uptrend — double rejection of resistance level'})

        # ── THREE-BAR BULLISH ─────────────────────────────────────────────────

        # Morning Star
        if (i >= 2 and c0['Close'] < c0['Open'] and
                body1 < avg_body * 0.4 and
                c2['Close'] > c2['Open'] and
                c2['Close'] > (c0['Open'] + c0['Close']) / 2):
            _add({'pattern': 'Morning Star', 'type': 'Bullish Reversal', 'strength': 'Strong',
                  'description': 'Red → small indecision → green above midpoint — classic 3-bar reversal'})

        # Three White Soldiers — three consecutive bullish candles, each closing higher
        if (i >= 2 and
                c0['Close'] > c0['Open'] and c1['Close'] > c1['Open'] and c2['Close'] > c2['Open'] and
                c1['Close'] > c0['Close'] and c2['Close'] > c1['Close'] and
                body0 > avg_body * 0.6 and body1 > avg_body * 0.6 and body2 > avg_body * 0.6):
            _add({'pattern': 'Three White Soldiers', 'type': 'Bullish Continuation', 'strength': 'Strong',
                  'description': 'Three consecutive strong green candles — sustained buying pressure'})

        # ── THREE-BAR BEARISH ─────────────────────────────────────────────────

        # Evening Star
        if (i >= 2 and c0['Close'] > c0['Open'] and
                body1 < avg_body * 0.4 and
                c2['Close'] < c2['Open'] and
                c2['Close'] < (c0['Open'] + c0['Close']) / 2):
            _add({'pattern': 'Evening Star', 'type': 'Bearish Reversal', 'strength': 'Strong',
                  'description': 'Green → small indecision → red below midpoint — classic 3-bar reversal'})

        # Three Black Crows — three consecutive bearish candles, each closing lower
        if (i >= 2 and
                c0['Close'] < c0['Open'] and c1['Close'] < c1['Open'] and c2['Close'] < c2['Open'] and
                c1['Close'] < c0['Close'] and c2['Close'] < c1['Close'] and
                body0 > avg_body * 0.6 and body1 > avg_body * 0.6 and body2 > avg_body * 0.6):
            _add({'pattern': 'Three Black Crows', 'type': 'Bearish Continuation', 'strength': 'Strong',
                  'description': 'Three consecutive strong red candles — sustained selling pressure'})

        # Stop early if we already have 5+ patterns — avoid cluttering the UI
        if len(patterns_detected) >= 5:
            break

    return patterns_detected


def generate_trading_signals(data):
    """
    Generate trading signals based on indicators
    
    Args:
        data: DataFrame with indicators
    
    Returns:
        Dictionary with signals and analysis
    """
    df = data.copy()
    
    if len(df) < 20:
        return {
            'signal': 'INSUFFICIENT_DATA',
            'confidence': 0,
            'buy_probability': 50,
            'sell_probability': 50,
            'bullish_signals': 0,
            'bearish_signals': 0,
            'signals_list': [f'Need at least 20 candles for analysis (have {len(df)})'],
            'current_price': round(df.iloc[-1]['Close'], 2),
            'indicators': {}
        }
    
    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) >= 2 else latest   # needed for crossover checks
    signals = []
    bullish_count = 0
    bearish_count = 0

    # ── ADX trend-strength gate ───────────────────────────────────────────────
    # ADX < 20 → no meaningful trend; oscillator signals are unreliable in
    # choppy/sideways markets. We still report them but at half weight, and
    # we add an explanatory note so the user knows.
    adx_value = latest.get('ADX') if not pd.isna(latest.get('ADX')) else None
    trending  = adx_value is not None and adx_value >= 20
    osc_weight = 1 if trending else 0  # weight multiplier for oscillator signals
    if adx_value is not None:
        if trending:
            signals.append(f'ADX {round(adx_value, 1)} — trending market (signals reliable)')
        else:
            signals.append(f'ADX {round(adx_value, 1)} — choppy/sideways market (oscillator signals suppressed)')

    # RSI Analysis (oscillator — suppressed when not trending)
    if not pd.isna(latest.get('RSI')):
        if latest['RSI'] < 30:
            signals.append('RSI Oversold (Bullish)')
            bullish_count += 2 * osc_weight
        elif latest['RSI'] > 70:
            signals.append('RSI Overbought (Bearish)')
            bearish_count += 2 * osc_weight
        elif latest['RSI'] < 40:
            signals.append('RSI Below 40 (Slightly Bullish)')
            bullish_count += 1 * osc_weight
        elif latest['RSI'] > 60:
            signals.append('RSI Above 60 (Slightly Bearish)')
            bearish_count += 1 * osc_weight

    # MACD Analysis — detect the actual crossover event, not sustained position
    # Bullish crossover: MACD crosses ABOVE Signal on this candle
    # Bearish crossover: MACD crosses BELOW Signal on this candle
    if (not pd.isna(latest.get('MACD')) and not pd.isna(latest.get('MACD_Signal'))
            and not pd.isna(prev.get('MACD')) and not pd.isna(prev.get('MACD_Signal'))):
        bullish_cross = (prev['MACD'] <= prev['MACD_Signal'] and
                         latest['MACD'] > latest['MACD_Signal'])
        bearish_cross = (prev['MACD'] >= prev['MACD_Signal'] and
                         latest['MACD'] < latest['MACD_Signal'])
        if bullish_cross:
            signals.append('MACD Bullish Crossover')
            bullish_count += 2
        elif bearish_cross:
            signals.append('MACD Bearish Crossover')
            bearish_count += 2
        else:
            # No crossover this candle — report sustained position at lower weight
            if latest['MACD'] > latest['MACD_Signal']:
                signals.append('MACD Above Signal (Bullish momentum)')
                bullish_count += 1
            elif latest['MACD'] < latest['MACD_Signal']:
                signals.append('MACD Below Signal (Bearish momentum)')
                bearish_count += 1
    
    # Moving Average Analysis
    if not pd.isna(latest.get('SMA_20')) and not pd.isna(latest.get('SMA_50')):
        if latest['Close'] > latest['SMA_20'] > latest['SMA_50']:
            signals.append('Price Above MAs (Bullish Trend)')
            bullish_count += 2
        elif latest['Close'] < latest['SMA_20'] < latest['SMA_50']:
            signals.append('Price Below MAs (Bearish Trend)')
            bearish_count += 2
        elif latest['SMA_20'] > latest['SMA_50']:
            signals.append('Golden Cross (Bullish)')
            bullish_count += 1
        elif latest['SMA_20'] < latest['SMA_50']:
            signals.append('Death Cross (Bearish)')
            bearish_count += 1
    
    # Bollinger Bands Analysis
    if not pd.isna(latest.get('BB_Upper')) and not pd.isna(latest.get('BB_Lower')):
        if latest['Close'] <= latest['BB_Lower']:
            signals.append('Price at Lower BB (Oversold)')
            bullish_count += 1
        elif latest['Close'] >= latest['BB_Upper']:
            signals.append('Price at Upper BB (Overbought)')
            bearish_count += 1
    
    # Stochastic Analysis (oscillator — ADX-gated)
    if not pd.isna(latest.get('Stoch_K')):
        if latest['Stoch_K'] < 20:
            signals.append('Stochastic Oversold (Bullish)')
            bullish_count += 1 * osc_weight
        elif latest['Stoch_K'] > 80:
            signals.append('Stochastic Overbought (Bearish)')
            bearish_count += 1 * osc_weight

    # MFI Analysis (volume-weighted oscillator — ADX-gated)
    if not pd.isna(latest.get('MFI')):
        if latest['MFI'] < 20:
            signals.append('Money Flow Oversold (Bullish)')
            bullish_count += 1 * osc_weight
        elif latest['MFI'] > 80:
            signals.append('Money Flow Overbought (Bearish)')
            bearish_count += 1 * osc_weight
    
    # Volume Analysis — high volume confirms the move
    if not pd.isna(latest.get('Volume_SMA_20')) and not pd.isna(latest.get('Volume')):
        if latest['Volume'] > latest['Volume_SMA_20'] * 1.5:
            signals.append('High Volume (Strong Move)')

    # SMA-200 — long-term trend filter (not ADX-gated; it IS a trend indicator)
    sma200 = latest.get('SMA_200')
    if not pd.isna(sma200) and sma200 and sma200 > 0:
        if latest['Close'] > sma200:
            signals.append('Price Above SMA-200 (Long-term Uptrend)')
            bullish_count += 1
        else:
            signals.append('Price Below SMA-200 (Long-term Downtrend)')
            bearish_count += 1

    # EMA-12 / EMA-26 crossover — faster signal than SMA crossover
    ema12 = latest.get('EMA_12'); ema26 = latest.get('EMA_26')
    prev_ema12 = prev.get('EMA_12'); prev_ema26 = prev.get('EMA_26')
    if (not pd.isna(ema12) and not pd.isna(ema26) and
            not pd.isna(prev_ema12) and not pd.isna(prev_ema26)):
        ema_bull_cross = prev_ema12 <= prev_ema26 and ema12 > ema26
        ema_bear_cross = prev_ema12 >= prev_ema26 and ema12 < ema26
        if ema_bull_cross:
            signals.append('EMA-12 Crossed Above EMA-26 (Bullish)')
            bullish_count += 2
        elif ema_bear_cross:
            signals.append('EMA-12 Crossed Below EMA-26 (Bearish)')
            bearish_count += 2
        elif ema12 > ema26:
            signals.append('EMA-12 Above EMA-26 (Bullish momentum)')
            bullish_count += 1
        else:
            signals.append('EMA-12 Below EMA-26 (Bearish momentum)')
            bearish_count += 1

    # OBV trend — rising OBV = buyers accumulating; falling = distribution
    # Compare current OBV to its 10-period average as a simple trend check
    if 'OBV' in df.columns:
        obv_series = df['OBV'].dropna()
        if len(obv_series) >= 10:
            obv_now  = float(obv_series.iloc[-1])
            obv_avg  = float(obv_series.iloc[-10:].mean())
            obv_prev = float(obv_series.iloc[-2]) if len(obv_series) >= 2 else obv_now
            if obv_now > obv_avg and obv_now > obv_prev:
                signals.append('OBV Rising — buying accumulation (Bullish)')
                bullish_count += 1
            elif obv_now < obv_avg and obv_now < obv_prev:
                signals.append('OBV Falling — selling distribution (Bearish)')
                bearish_count += 1
    
    # ── Bollinger Band squeeze / expansion ───────────────────────────────────
    # BB_Width at a multi-period low = coiling (volatility compression) →
    # imminent breakout.  We don't know direction yet, so we flag it neutrally
    # and let price action confirm.  BB_Width at a 20-period high = expansion
    # → confirms the current trend direction.
    if 'BB_Width' in df.columns:
        bb_w = df['BB_Width'].dropna()
        if len(bb_w) >= 20:
            current_w   = float(bb_w.iloc[-1])
            min_20      = float(bb_w.iloc[-20:].min())
            max_20      = float(bb_w.iloc[-20:].max())
            width_range = max_20 - min_20
            if width_range > 0:
                if current_w <= min_20 * 1.05:   # within 5% of 20-period low
                    signals.append('BB Squeeze — volatility coiling, breakout imminent')
                    # Don't add to bull/bear — direction unknown until break
                elif current_w >= max_20 * 0.95:  # near 20-period high
                    # Expansion confirms whichever direction price is moving
                    if latest['Close'] > latest.get('SMA_20', latest['Close']):
                        signals.append('BB Expansion — trend acceleration (Bullish)')
                        bullish_count += 1
                    else:
                        signals.append('BB Expansion — trend acceleration (Bearish)')
                        bearish_count += 1

    # ── RSI divergence ────────────────────────────────────────────────────────
    # Bullish divergence: price makes a lower low, RSI makes a higher low
    #   → momentum not confirming the new low → likely reversal up
    # Bearish divergence: price makes a higher high, RSI makes a lower high
    #   → momentum not confirming the new high → likely reversal down
    # We scan the last 20 bars to find a recent swing low/high pair.
    if 'RSI' in df.columns:
        _rsi  = df['RSI'].dropna()
        _prc  = df['Close'].reindex(_rsi.index)
        if len(_rsi) >= 10:
            # Pivot detection: local min/max with window=3
            _window = 3
            _lows_i  = [i for i in range(_window, len(_rsi) - _window)
                        if _prc.iloc[i] == _prc.iloc[i-_window:i+_window+1].min()]
            _highs_i = [i for i in range(_window, len(_rsi) - _window)
                        if _prc.iloc[i] == _prc.iloc[i-_window:i+_window+1].max()]

            # Bullish: two recent lows where price lower but RSI higher
            if len(_lows_i) >= 2:
                i1, i2 = _lows_i[-2], _lows_i[-1]
                if (_prc.iloc[i2] < _prc.iloc[i1] and
                        _rsi.iloc[i2] > _rsi.iloc[i1] and
                        (len(_rsi) - 1 - i2) <= 10):   # within last 10 bars
                    signals.append('RSI Bullish Divergence — price lower low, RSI higher low')
                    bullish_count += 2

            # Bearish: two recent highs where price higher but RSI lower
            if len(_highs_i) >= 2:
                i1, i2 = _highs_i[-2], _highs_i[-1]
                if (_prc.iloc[i2] > _prc.iloc[i1] and
                        _rsi.iloc[i2] < _rsi.iloc[i1] and
                        (len(_rsi) - 1 - i2) <= 10):
                    signals.append('RSI Bearish Divergence — price higher high, RSI lower high')
                    bearish_count += 2

    # ── MACD divergence ───────────────────────────────────────────────────────
    # Same principle applied to MACD histogram:
    # Bullish: price lower low, MACD_Hist higher (less negative)
    # Bearish: price higher high, MACD_Hist lower (less positive)
    if 'MACD_Hist' in df.columns:
        _hist = df['MACD_Hist'].dropna()
        _prc2 = df['Close'].reindex(_hist.index)
        if len(_hist) >= 10:
            _lows2  = [i for i in range(3, len(_hist) - 3)
                       if _prc2.iloc[i] == _prc2.iloc[i-3:i+4].min()]
            _highs2 = [i for i in range(3, len(_hist) - 3)
                       if _prc2.iloc[i] == _prc2.iloc[i-3:i+4].max()]

            if len(_lows2) >= 2:
                i1, i2 = _lows2[-2], _lows2[-1]
                if (_prc2.iloc[i2] < _prc2.iloc[i1] and
                        _hist.iloc[i2] > _hist.iloc[i1] and
                        (len(_hist) - 1 - i2) <= 10):
                    signals.append('MACD Histogram Bullish Divergence')
                    bullish_count += 1

            if len(_highs2) >= 2:
                i1, i2 = _highs2[-2], _highs2[-1]
                if (_prc2.iloc[i2] > _prc2.iloc[i1] and
                        _hist.iloc[i2] < _hist.iloc[i1] and
                        (len(_hist) - 1 - i2) <= 10):
                    signals.append('MACD Histogram Bearish Divergence')
                    bearish_count += 1

    # Determine Overall Signal
    total_signals = bullish_count + bearish_count
    
    if total_signals == 0:
        signal = 'HOLD'
        confidence = 50
        buy_probability = 50
        sell_probability = 50
        if len(signals) == 0:
            signals.append('Neutral market conditions - no strong signals detected')
    else:
        buy_probability = round((bullish_count / total_signals) * 100, 2)
        sell_probability = round((bearish_count / total_signals) * 100, 2)
        
        if buy_probability >= 60:
            signal = 'BUY'
            confidence = buy_probability
        elif sell_probability >= 60:
            signal = 'SELL'
            confidence = sell_probability
        else:
            signal = 'HOLD'
            confidence = max(buy_probability, sell_probability)
    
    return {
        'signal': signal,
        'confidence': round(confidence, 2),
        'buy_probability': buy_probability,
        'sell_probability': sell_probability,
        'bullish_signals': bullish_count,
        'bearish_signals': bearish_count,
        'signals_list': signals,
        'current_price': round(latest['Close'], 2),
        'indicators': {
            'RSI': round(latest.get('RSI', 0), 2) if not pd.isna(latest.get('RSI')) else None,
            'MACD': round(latest.get('MACD', 0), 2) if not pd.isna(latest.get('MACD')) else None,
            'MACD_Signal': round(latest.get('MACD_Signal', 0), 2) if not pd.isna(latest.get('MACD_Signal')) else None,
            'SMA_20': round(latest.get('SMA_20', 0), 2) if not pd.isna(latest.get('SMA_20')) else None,
            'SMA_50': round(latest.get('SMA_50', 0), 2) if not pd.isna(latest.get('SMA_50')) else None,
            'BB_Upper': round(latest.get('BB_Upper', 0), 2) if not pd.isna(latest.get('BB_Upper')) else None,
            'BB_Lower': round(latest.get('BB_Lower', 0), 2) if not pd.isna(latest.get('BB_Lower')) else None,
            'Volume': int(latest.get('Volume', 0)) if not pd.isna(latest.get('Volume')) else None,
            'ATR': round(latest.get('ATR', 0), 2) if not pd.isna(latest.get('ATR')) else None,
            'ADX': round(latest.get('ADX', 0), 2) if not pd.isna(latest.get('ADX')) else None,
            'OBV': int(latest.get('OBV', 0)) if not pd.isna(latest.get('OBV')) else None,
            'SMA_200': round(latest.get('SMA_200', 0), 2) if not pd.isna(latest.get('SMA_200')) else None,
            'EMA_12': round(latest.get('EMA_12', 0), 2) if not pd.isna(latest.get('EMA_12')) else None,
            'EMA_26': round(latest.get('EMA_26', 0), 2) if not pd.isna(latest.get('EMA_26')) else None,
        }
    }


def predict_next_candle(data):
    """
    Predict next-candle direction using a transparent multi-factor momentum model
    with a rolling backtest to show historical hit-rate on the same dataset.

    Why not Random Forest?
    ──────────────────────
    A standard trading dataset contains 50-250 rows after indicator warmup.
    A RandomForestClassifier trained on such small samples has no meaningful
    generalisation ability. This replacement uses five rule-based momentum
    signals. The model is backtested on the same historical window to produce
    an honest hit-rate shown alongside the prediction.
    """
    df = data.copy().reset_index(drop=True)

    if len(df) < 14:
        return {
            'predicted_direction': 'INSUFFICIENT_DATA',
            'predicted_change': 0,
            'confidence': 0,
            'note': f'Need at least 14 candles (have {len(df)}).',
            'model': 'Multi-factor momentum',
        }

    def _score_bar(row, prev_row):
        """Apply the 5 momentum votes to a single bar. Returns (direction, n_votes)."""
        v = {}
        rsi = row.get('RSI')
        if not pd.isna(rsi):
            v['RSI'] = 1 if rsi > 50 else -1

        mhist = row.get('MACD_Hist')
        mhist_p = prev_row.get('MACD_Hist')
        if not pd.isna(mhist) and not pd.isna(mhist_p):
            v['MACD'] = 1 if mhist > 0 else -1

        sma20 = row.get('SMA_20')
        if not pd.isna(sma20) and sma20 > 0:
            v['SMA20'] = 1 if row['Close'] > sma20 else -1

        sma50 = row.get('SMA_50')
        if not pd.isna(sma50) and sma50 > 0:
            v['SMA50'] = 1 if row['Close'] > sma50 else -1

        stoch = row.get('Stoch_K')
        if not pd.isna(stoch):
            v['Stoch'] = 1 if stoch > 50 else -1

        if not v:
            return 'NEUTRAL', 0
        bull = sum(1 for x in v.values() if x > 0)
        bear = sum(1 for x in v.values() if x < 0)
        return ('UP' if bull > bear else 'DOWN' if bear > bull else 'NEUTRAL'), len(v)

    # ── Rolling backtest ──────────────────────────────────────────────────────
    # Walk through bars [1 .. n-2], predict next-bar direction, compare to actual.
    # Skip NEUTRAL predictions — they are abstentions, not errors.
    backtest_window = min(len(df) - 2, 100)
    start_bt = max(1, len(df) - 1 - backtest_window)

    total_calls = 0
    correct_all = 0
    bt_log = []   # (predicted, actual, hit)

    for i in range(start_bt, len(df) - 1):
        pred_dir, _ = _score_bar(df.iloc[i], df.iloc[i - 1])
        if pred_dir == 'NEUTRAL':
            continue
        actual_dir = 'UP' if df.iloc[i + 1]['Close'] > df.iloc[i]['Close'] else 'DOWN'
        hit = (pred_dir == actual_dir)
        bt_log.append((pred_dir, actual_dir, hit))
        total_calls += 1
        if hit:
            correct_all += 1

    # Last-20 hit-rate
    calls_20 = correct_20 = 0
    for _, _, hit in bt_log[-20:]:
        calls_20 += 1
        if hit:
            correct_20 += 1

    # Current streak
    streak = 0
    streak_sign = None
    if bt_log:
        last_hit = bt_log[-1][2]
        streak_sign = 'W' if last_hit else 'L'
        streak = 1
        for _, _, hit in reversed(bt_log[:-1]):
            if hit == last_hit:
                streak += 1
            else:
                break

    hit_rate_all = round(correct_all / total_calls * 100, 1) if total_calls else None
    hit_rate_20  = round(correct_20 / calls_20 * 100, 1)     if calls_20  else None
    streak_label = f"{streak} {'win' if streak_sign == 'W' else 'loss'} streak" if streak and streak_sign else None

    # ── Current prediction ────────────────────────────────────────────────────
    latest = df.iloc[-1]
    prev   = df.iloc[-2]
    current_price = float(latest['Close'])

    if not pd.isna(latest.get('ATR')) and latest['ATR'] > 0:
        atr = float(latest['ATR'])
    else:
        atr = float(df['Close'].pct_change().tail(10).std() * current_price)
    atr = max(atr, current_price * 0.001)

    votes = {}
    rsi = latest.get('RSI')
    if not pd.isna(rsi):
        votes['RSI midline'] = 1 if rsi > 50 else -1

    mhist = latest.get('MACD_Hist')
    mhist_p = prev.get('MACD_Hist')
    if not pd.isna(mhist) and not pd.isna(mhist_p):
        votes['MACD histogram'] = 1 if mhist > 0 else -1

    sma20 = latest.get('SMA_20')
    if not pd.isna(sma20) and sma20 > 0:
        votes['Price vs SMA-20'] = 1 if current_price > sma20 else -1

    sma50 = latest.get('SMA_50')
    if not pd.isna(sma50) and sma50 > 0:
        votes['Price vs SMA-50'] = 1 if current_price > sma50 else -1

    stoch = latest.get('Stoch_K')
    if not pd.isna(stoch):
        votes['Stochastic %K'] = 1 if stoch > 50 else -1

    if not votes:
        return {
            'predicted_direction': 'UNKNOWN',
            'predicted_change': 0,
            'confidence': 0,
            'note': 'Insufficient indicators to score.',
            'model': 'Multi-factor momentum',
        }

    total      = len(votes)
    bull_votes = sum(1 for v in votes.values() if v > 0)
    bear_votes = sum(1 for v in votes.values() if v < 0)
    net_score  = bull_votes - bear_votes

    if net_score > 0:
        direction       = 'UP'
        predicted_price = round(current_price + atr * 0.5, 2)
    elif net_score < 0:
        direction       = 'DOWN'
        predicted_price = round(current_price - atr * 0.5, 2)
    else:
        direction       = 'NEUTRAL'
        predicted_price = round(current_price, 2)

    predicted_change   = round(((predicted_price / current_price) - 1) * 100, 2)
    majority_votes     = max(bull_votes, bear_votes)
    confidence         = round(50 + (majority_votes / total) * 40, 1)
    factor_lines       = [f"{'UP' if v > 0 else 'DN'} {fname}" for fname, v in votes.items()]

    return {
        'predicted_direction': direction,
        'predicted_price':     predicted_price,
        'predicted_change':    predicted_change,
        'confidence':          confidence,
        'current_price':       round(current_price, 2),
        'model': f'Multi-factor momentum ({total} factors: {bull_votes} bullish, {bear_votes} bearish)',
        'model_note': (
            'Confidence = fraction of factors agreeing (rule-based, auditable). '
            'Hit-rate = how often this same model was correct on recent history.'
        ),
        'factors':        factor_lines,
        'features_used':  total,
        'hit_rate_all':   hit_rate_all,
        'hit_rate_20':    hit_rate_20,
        'backtest_calls': total_calls,
        'streak':         streak,
        'streak_type':    streak_sign,
        'streak_label':   streak_label,
    }



def calculate_support_resistance(data, n_levels=3, swing_window=5):
    """
    Identify support and resistance levels from swing highs/lows + daily pivot.

    Method
    ──────
    1. Classic daily pivot (most recent complete session):
         P  = (High + Low + Close) / 3
         R1 = 2P − Low    S1 = 2P − High
         R2 = P + (High − Low)   S2 = P − (High − Low)
         R3 = High + 2(P − Low)  S3 = Low  − 2(High − P)

    2. Swing high / low detection across the full dataset:
       A bar is a swing high if its High > all neighbours within ±swing_window.
       A bar is a swing low  if its Low  < all neighbours within ±swing_window.
       We cluster nearby levels (within 0.5% of each other) and rank clusters
       by touch-count to return the strongest n_levels.

    Args:
        data          : DataFrame with High, Low, Close columns.
        n_levels      : How many support / resistance lines to return each side.
        swing_window  : Half-width of the rolling window for swing detection.

    Returns:
        dict with keys:
            pivot_levels  – list of {label, price, type} for classic pivots
            support       – list of {price, touches, strength} sorted desc touches
            resistance    – list of {price, touches, strength} sorted desc touches
            current_price – latest Close
    """
    df = data.copy().reset_index(drop=True)
    if len(df) < swing_window * 2 + 1:
        return {'pivot_levels': [], 'support': [], 'resistance': [],
                'current_price': round(float(df.iloc[-1]['Close']), 2)}

    current_price = float(df.iloc[-1]['Close'])

    # ── 1. Classic pivot (last completed bar) ────────────────────────────────
    last = df.iloc[-1]
    P  = (last['High'] + last['Low'] + last['Close']) / 3
    R1 = round(2 * P - last['Low'],  2)
    R2 = round(P + (last['High'] - last['Low']), 2)
    R3 = round(last['High'] + 2 * (P - last['Low']), 2)
    S1 = round(2 * P - last['High'], 2)
    S2 = round(P - (last['High'] - last['Low']), 2)
    S3 = round(last['Low'] - 2 * (last['High'] - P), 2)
    P  = round(P, 2)

    pivot_levels = [
        {'label': 'R3', 'price': R3, 'type': 'resistance'},
        {'label': 'R2', 'price': R2, 'type': 'resistance'},
        {'label': 'R1', 'price': R1, 'type': 'resistance'},
        {'label': 'P',  'price': P,  'type': 'pivot'},
        {'label': 'S1', 'price': S1, 'type': 'support'},
        {'label': 'S2', 'price': S2, 'type': 'support'},
        {'label': 'S3', 'price': S3, 'type': 'support'},
    ]

    # ── 2. Swing high / low scan ─────────────────────────────────────────────
    w = swing_window
    swing_highs = []
    swing_lows  = []

    for i in range(w, len(df) - w):
        window_slice = df.iloc[i - w: i + w + 1]
        if df.iloc[i]['High'] == window_slice['High'].max():
            swing_highs.append(float(df.iloc[i]['High']))
        if df.iloc[i]['Low'] == window_slice['Low'].min():
            swing_lows.append(float(df.iloc[i]['Low']))

    def cluster_levels(raw_prices, cluster_pct=0.005):
        """Merge levels within cluster_pct of each other, count touches."""
        if not raw_prices:
            return []
        prices = sorted(raw_prices)
        clusters = []
        current = [prices[0]]
        for p in prices[1:]:
            if abs(p - current[0]) / current[0] <= cluster_pct:
                current.append(p)
            else:
                clusters.append(current)
                current = [p]
        clusters.append(current)
        result = []
        for c in clusters:
            avg   = round(sum(c) / len(c), 2)
            touches = len(c)
            strength = 'Strong' if touches >= 3 else ('Medium' if touches == 2 else 'Weak')
            result.append({'price': avg, 'touches': touches, 'strength': strength})
        return sorted(result, key=lambda x: x['touches'], reverse=True)

    all_resistance = cluster_levels([p for p in swing_highs if p > current_price])
    all_support    = cluster_levels([p for p in swing_lows  if p < current_price])

    return {
        'pivot_levels':   pivot_levels,
        'support':        all_support[:n_levels],
        'resistance':     all_resistance[:n_levels],
        'current_price':  round(current_price, 2),
    }


def run_technical_analysis(symbol, timeframe='1d', period='1y'):
    """
    Complete technical analysis pipeline

    Args:
        symbol: Stock ticker
        timeframe: Candle timeframe
        period: Historical period

    Returns:
        Dictionary with complete analysis
    """
    # ── Cache check ───────────────────────────────────────────────────────────
    _now = datetime.now()
    # For intraday (≤1h) use hour-bucket so cache refreshes every hour at most.
    # For daily+ data use date-bucket so cache lasts the trading day.
    _intraday_tf = timeframe in ('1m', '5m', '15m', '30m', '60m', '1h')
    _bucket = _now.strftime('%Y-%m-%d_%H') if _intraday_tf else _now.strftime('%Y-%m-%d')
    _cache_key = f"{symbol.upper()}|{timeframe}|{period}|{_bucket}"

    if _cache_key in _CACHE:
        _stored_at, _cached_result = _CACHE[_cache_key]
        if (_now - _stored_at).total_seconds() < _CACHE_TTL_SECONDS:
            _cached_result['from_cache'] = True
            return _cached_result

    try:
        # 1. Fetch data
        data = fetch_stock_data(symbol, timeframe, period)
        
        # 2. Calculate indicators
        data = calculate_technical_indicators(data)
        
        # 3. Detect patterns
        patterns = detect_candlestick_patterns(data)
        
        # 4. Generate signals
        signals = generate_trading_signals(data)
        
        # 5. Predict next candle
        prediction = predict_next_candle(data)

        # 6. Support / Resistance levels
        sr_levels = calculate_support_resistance(data)

        # 7. Prepare chart data (last 100 candles for display)
        chart_data = data.tail(100).to_dict('records')

        # Convert timestamps to strings for JSON serialization
        for row in chart_data:
            if 'Date' in row and isinstance(row['Date'], pd.Timestamp):
                row['Date'] = row['Date'].strftime('%Y-%m-%d %H:%M:%S')

        # 8. VWAP — only meaningful for intraday timeframes
        intraday_tf = timeframe in ('1m', '5m', '15m', '30m', '60m', '1h')
        vwap_value  = None
        if 'VWAP' in data.columns and not pd.isna(data.iloc[-1].get('VWAP')):
            vwap_value = round(float(data.iloc[-1]['VWAP']), 2)

        return {
            'success': True,
            'symbol': symbol,
            'timeframe': timeframe,
            'period': period,
            'is_intraday': intraday_tf,
            'total_candles': len(data),
            'chart_data': chart_data,
            'patterns': patterns,
            'signals': signals,
            'prediction': prediction,
            'support_resistance': sr_levels,
            'vwap': vwap_value,
            'summary': {
                'current_price': signals['current_price'],
                'recommendation': signals['signal'],
                'confidence': signals['confidence'],
                'buy_probability': signals['buy_probability'],
                'sell_probability': signals['sell_probability'],
                'predicted_direction': prediction.get('predicted_direction', 'UNKNOWN'),
                'predicted_change': prediction.get('predicted_change', 0),
                'vwap': vwap_value,
                'nearest_support':    sr_levels['support'][0]['price']    if sr_levels['support']    else None,
                'nearest_resistance': sr_levels['resistance'][0]['price'] if sr_levels['resistance'] else None,
            }
        }

        # ── Cache store ───────────────────────────────────────────────────────
        result['from_cache'] = False
        _CACHE[_cache_key] = (datetime.now(), result)
        # Evict stale entries (keep cache lean — max 50 entries)
        if len(_CACHE) > 50:
            oldest_key = min(_CACHE, key=lambda k: _CACHE[k][0])
            del _CACHE[oldest_key]

        return result
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'symbol': symbol,
            'timeframe': timeframe,
            'period': period
        }
