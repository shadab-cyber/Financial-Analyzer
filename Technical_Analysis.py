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
        
        # Common symbol mappings for Indian stocks
        symbol_mappings = {
            'TATAMOTORS': 'TATAMOTORS',
            'TATA MOTORS': 'TATAMOTORS',
            'RELIANCE': 'RELIANCE',
            'TCS': 'TCS',
            'INFY': 'INFY',
            'INFOSYS': 'INFY',
            'HDFCBANK': 'HDFCBANK',
            'HDFC': 'HDFCBANK',
            'ICICIBANK': 'ICICIBANK',
            'ICICI': 'ICICIBANK',
            'SBIN': 'SBIN',
            'SBI': 'SBIN',
            'BHARTIARTL': 'BHARTIARTL',
            'AIRTEL': 'BHARTIARTL',
            'ITC': 'ITC',
            'KOTAKBANK': 'KOTAKBANK',
            'KOTAK': 'KOTAKBANK',
            'LT': 'LT',
            'AXISBANK': 'AXISBANK',
            'AXIS': 'AXISBANK',
            'MARUTI': 'MARUTI',
            'WIPRO': 'WIPRO',
            'HCLTECH': 'HCLTECH',
            'HCL': 'HCLTECH',
            'ASIANPAINT': 'ASIANPAINT',
            'ULTRACEMCO': 'ULTRACEMCO',
            'BAJFINANCE': 'BAJFINANCE',
            'BAJAJ': 'BAJFINANCE',
            'TITAN': 'TITAN',
            'NESTLEIND': 'NESTLEIND',
            'NESTLE': 'NESTLEIND',
            'SUNPHARMA': 'SUNPHARMA',
            'TATASTEEL': 'TATASTEEL',
            'ONGC': 'ONGC',
            'NTPC': 'NTPC',
            'POWERGRID': 'POWERGRID',
            'M&M': 'M&M',
            'MAHINDRA': 'M&M',
            'HINDUNILVR': 'HINDUNILVR',
            'HUL': 'HINDUNILVR'
        }
        
        # Apply mapping if exists
        if symbol in symbol_mappings:
            symbol = symbol_mappings[symbol]
        
        # Try NSE first (most common for Indian stocks)
        symbols_to_try = [
            f"{symbol}.NS",      # NSE
            f"{symbol}.BO",      # BSE
            symbol               # US stocks (no suffix)
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
    df = data.copy()
    patterns_detected = []
    
    if len(df) < 3:
        return patterns_detected
    
    # Get last 3 candles
    c0 = df.iloc[-3]  # 2 candles ago
    c1 = df.iloc[-2]  # 1 candle ago
    c2 = df.iloc[-1]  # Current candle
    
    # Calculate body and shadow sizes
    body_2 = abs(c2['Close'] - c2['Open'])
    upper_shadow_2 = c2['High'] - max(c2['Open'], c2['Close'])
    lower_shadow_2 = min(c2['Open'], c2['Close']) - c2['Low']
    
    body_1 = abs(c1['Close'] - c1['Open'])
    
    # BULLISH PATTERNS

    # Downtrend context: last 5 closes trending down (simple heuristic)
    recent_closes = df['Close'].tail(6).values
    in_downtrend = len(recent_closes) >= 5 and recent_closes[-2] < recent_closes[0]
    in_uptrend   = len(recent_closes) >= 5 and recent_closes[-2] > recent_closes[0]

    # Hammer (Bullish Reversal) — long lower shadow, small body, any colour
    # Classic definition: lower shadow ≥ 2× body, upper shadow ≤ 0.3× body,
    # appears after a downtrend.  Colour doesn't matter.
    if (lower_shadow_2 >= 2 * body_2 and
            upper_shadow_2 <= body_2 * 0.3 and
            body_2 > 0 and in_downtrend):
        patterns_detected.append({
            'pattern': 'Hammer',
            'type': 'Bullish Reversal',
            'strength': 'Medium',
            'description': 'Long lower shadow after downtrend — buying pressure absorbing sellers'
        })
    
    # Bullish Engulfing
    if (c1['Close'] < c1['Open'] and  # Previous red candle
        c2['Close'] > c2['Open'] and  # Current green candle
        c2['Open'] < c1['Close'] and  # Opens below previous close
        c2['Close'] > c1['Open']):    # Closes above previous open
        patterns_detected.append({
            'pattern': 'Bullish Engulfing',
            'type': 'Bullish Reversal',
            'strength': 'Strong',
            'description': 'Strong buying pressure engulfs previous candle'
        })
    
    # Morning Star (3-candle pattern)
    if len(df) >= 3:
        if (c0['Close'] < c0['Open'] and  # First candle red
            abs(c1['Close'] - c1['Open']) < body_2 * 0.3 and  # Second small body
            c2['Close'] > c2['Open'] and  # Third candle green
            c2['Close'] > (c0['Open'] + c0['Close']) / 2):  # Closes above midpoint
            patterns_detected.append({
                'pattern': 'Morning Star',
                'type': 'Bullish Reversal',
                'strength': 'Strong',
                'description': 'Three-candle bullish reversal pattern'
            })
    
    # BEARISH PATTERNS
    
    # Shooting Star (Bearish Reversal) — long upper shadow, small body, any colour,
    # appears after an uptrend.
    if (upper_shadow_2 >= 2 * body_2 and
            lower_shadow_2 <= body_2 * 0.3 and
            body_2 > 0 and in_uptrend):
        patterns_detected.append({
            'pattern': 'Shooting Star',
            'type': 'Bearish Reversal',
            'strength': 'Medium',
            'description': 'Long upper shadow after uptrend — sellers rejecting higher prices'
        })
    
    # Bearish Engulfing
    if (c1['Close'] > c1['Open'] and  # Previous green candle
        c2['Close'] < c2['Open'] and  # Current red candle
        c2['Open'] > c1['Close'] and  # Opens above previous close
        c2['Close'] < c1['Open']):    # Closes below previous open
        patterns_detected.append({
            'pattern': 'Bearish Engulfing',
            'type': 'Bearish Reversal',
            'strength': 'Strong',
            'description': 'Strong selling pressure engulfs previous candle'
        })
    
    # Evening Star
    if len(df) >= 3:
        if (c0['Close'] > c0['Open'] and  # First candle green
            abs(c1['Close'] - c1['Open']) < body_2 * 0.3 and  # Second small body
            c2['Close'] < c2['Open'] and  # Third candle red
            c2['Close'] < (c0['Open'] + c0['Close']) / 2):  # Closes below midpoint
            patterns_detected.append({
                'pattern': 'Evening Star',
                'type': 'Bearish Reversal',
                'strength': 'Strong',
                'description': 'Three-candle bearish reversal pattern'
            })
    
    # Doji (Indecision)
    if body_2 < (c2['High'] - c2['Low']) * 0.1:
        patterns_detected.append({
            'pattern': 'Doji',
            'type': 'Indecision',
            'strength': 'Neutral',
            'description': 'Market indecision, potential reversal'
        })
    
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
    
    # Volume Analysis
    if not pd.isna(latest.get('Volume_SMA_20')) and not pd.isna(latest.get('Volume')):
        if latest['Volume'] > latest['Volume_SMA_20'] * 1.5:
            signals.append('High Volume (Strong Move)')
    
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
        }
    }


def predict_next_candle(data):
    """
    Predict next-candle direction using a transparent multi-factor momentum model.

    Why not Random Forest?
    ──────────────────────
    A standard trading dataset contains 50–250 rows after indicator warmup.
    A RandomForestClassifier trained on such small samples and immediately
    evaluated on the very next row has no meaningful generalisation ability —
    its reported "confidence" is essentially the class prior, not a
    statistically valid probability.  Displaying 65–72% RF confidence to a
    retail investor implies a level of certainty the model cannot deliver.

    This replacement uses five rule-based momentum signals drawn from the same
    indicators already calculated by calculate_technical_indicators().  Each
    signal casts a directional vote (+1 = bullish, −1 = bearish).  The
    aggregate score is mapped to a direction and a signal-count-based
    confidence band that is honest about what it represents: "X out of 5
    momentum factors agree."  The approach is fully auditable — users can
    see exactly why the model says UP or DOWN.
    """
    df = data.copy()

    if len(df) < 14:
        return {
            'predicted_direction': 'INSUFFICIENT_DATA',
            'predicted_change': 0,
            'confidence': 0,
            'note': f'Need at least 14 candles (have {len(df)}).',
            'model': 'Multi-factor momentum',
        }

    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) >= 2 else latest
    current_price = float(latest['Close'])

    # ── ATR-based price range estimate ───────────────────────────────────────
    if not pd.isna(latest.get('ATR')) and latest['ATR'] > 0:
        atr = float(latest['ATR'])
    else:
        atr = float(df['Close'].pct_change().tail(10).std() * current_price)
    atr = atr if atr > 0 else current_price * 0.01

    # ── Five momentum votes ───────────────────────────────────────────────────
    # Each vote:  +1 = bullish,  −1 = bearish,  0 = neutral/unavailable
    votes = {}

    # 1. RSI momentum: above/below 50 midline
    rsi = latest.get('RSI')
    if not pd.isna(rsi):
        votes['RSI midline'] = 1 if rsi > 50 else -1

    # 2. MACD histogram direction (positive = buying pressure building)
    mhist = latest.get('MACD_Hist')
    mhist_p = prev.get('MACD_Hist')
    if not pd.isna(mhist) and not pd.isna(mhist_p):
        if mhist > 0 and mhist > mhist_p:
            votes['MACD histogram'] = 1    # positive and rising
        elif mhist < 0 and mhist < mhist_p:
            votes['MACD histogram'] = -1   # negative and falling
        elif mhist > 0:
            votes['MACD histogram'] = 1
        elif mhist < 0:
            votes['MACD histogram'] = -1

    # 3. Price vs SMA-20 (short-term trend)
    sma20 = latest.get('SMA_20')
    if not pd.isna(sma20) and sma20 > 0:
        votes['Price vs SMA-20'] = 1 if current_price > sma20 else -1

    # 4. Price vs SMA-50 (medium-term trend)
    sma50 = latest.get('SMA_50')
    if not pd.isna(sma50) and sma50 > 0:
        votes['Price vs SMA-50'] = 1 if current_price > sma50 else -1

    # 5. Stochastic %K vs 50 midline (momentum)
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
    net_score  = bull_votes - bear_votes  # range: −total … +total

    # Direction: majority of votes
    if net_score > 0:
        direction = 'UP'
        predicted_price  = round(current_price + atr * 0.5, 2)
    elif net_score < 0:
        direction = 'DOWN'
        predicted_price  = round(current_price - atr * 0.5, 2)
    else:
        direction = 'NEUTRAL'
        predicted_price  = round(current_price, 2)

    predicted_change = round(((predicted_price / current_price) - 1) * 100, 2)

    # Confidence: fraction of agreeing votes, scaled 50–90%
    # We cap at 90% because no single-candle model can be more certain than that.
    majority_votes  = max(bull_votes, bear_votes)
    raw_frac        = majority_votes / total                    # 0.5 … 1.0
    confidence      = round(50 + raw_frac * 40, 1)             # 70 … 90

    # Build readable factor summary
    factor_lines = []
    for fname, vote in votes.items():
        arrow = '▲' if vote > 0 else '▼'
        factor_lines.append(f'{arrow} {fname}')

    return {
        'predicted_direction': direction,
        'predicted_price': predicted_price,
        'predicted_change': predicted_change,
        'confidence': confidence,
        'current_price': round(current_price, 2),
        'model': f'Multi-factor momentum ({total} factors: {bull_votes} bullish, {bear_votes} bearish)',
        'model_note': (
            'Confidence reflects how many of the 5 momentum factors agree — '
            'not a statistical probability. This model is rule-based and auditable, '
            'not a trained ML classifier.'
        ),
        'factors': factor_lines,
        'features_used': total,
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
        
        # 6. Prepare chart data (last 100 candles for display)
        chart_data = data.tail(100).to_dict('records')
        
        # Convert timestamps to strings for JSON serialization
        for row in chart_data:
            if 'Date' in row and isinstance(row['Date'], pd.Timestamp):
                row['Date'] = row['Date'].strftime('%Y-%m-%d %H:%M:%S')
        
        return {
            'success': True,
            'symbol': symbol,
            'timeframe': timeframe,
            'period': period,
            'total_candles': len(data),
            'chart_data': chart_data,
            'patterns': patterns,
            'signals': signals,
            'prediction': prediction,
            'summary': {
                'current_price': signals['current_price'],
                'recommendation': signals['signal'],
                'confidence': signals['confidence'],
                'buy_probability': signals['buy_probability'],
                'sell_probability': signals['sell_probability'],
                'predicted_direction': prediction.get('predicted_direction', 'UNKNOWN'),
                'predicted_change': prediction.get('predicted_change', 0)
            }
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'symbol': symbol,
            'timeframe': timeframe,
            'period': period
        }
