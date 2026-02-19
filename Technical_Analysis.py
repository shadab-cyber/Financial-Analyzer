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
    
    # Hammer (Bullish Reversal)
    if (lower_shadow_2 > 2 * body_2 and 
        upper_shadow_2 < body_2 * 0.3 and
        c2['Close'] > c2['Open']):
        patterns_detected.append({
            'pattern': 'Hammer',
            'type': 'Bullish Reversal',
            'strength': 'Medium',
            'description': 'Long lower shadow suggests buying pressure'
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
    
    # Shooting Star (Bearish Reversal)
    if (upper_shadow_2 > 2 * body_2 and 
        lower_shadow_2 < body_2 * 0.3 and
        c2['Close'] < c2['Open']):
        patterns_detected.append({
            'pattern': 'Shooting Star',
            'type': 'Bearish Reversal',
            'strength': 'Medium',
            'description': 'Long upper shadow suggests selling pressure'
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
    signals = []
    bullish_count = 0
    bearish_count = 0
    
    # RSI Analysis
    if not pd.isna(latest.get('RSI')):
        if latest['RSI'] < 30:
            signals.append('RSI Oversold (Bullish)')
            bullish_count += 2
        elif latest['RSI'] > 70:
            signals.append('RSI Overbought (Bearish)')
            bearish_count += 2
        elif latest['RSI'] < 40:
            signals.append('RSI Below 40 (Slightly Bullish)')
            bullish_count += 1
        elif latest['RSI'] > 60:
            signals.append('RSI Above 60 (Slightly Bearish)')
            bearish_count += 1
    
    # MACD Analysis
    if not pd.isna(latest.get('MACD')) and not pd.isna(latest.get('MACD_Signal')):
        if latest['MACD'] > latest['MACD_Signal'] and latest['MACD'] > 0:
            signals.append('MACD Bullish Crossover')
            bullish_count += 2
        elif latest['MACD'] < latest['MACD_Signal'] and latest['MACD'] < 0:
            signals.append('MACD Bearish Crossover')
            bearish_count += 2
    
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
    
    # Stochastic Analysis
    if not pd.isna(latest.get('Stoch_K')):
        if latest['Stoch_K'] < 20:
            signals.append('Stochastic Oversold (Bullish)')
            bullish_count += 1
        elif latest['Stoch_K'] > 80:
            signals.append('Stochastic Overbought (Bearish)')
            bearish_count += 1
    
    # MFI Analysis
    if not pd.isna(latest.get('MFI')):
        if latest['MFI'] < 20:
            signals.append('Money Flow Oversold (Bullish)')
            bullish_count += 1
        elif latest['MFI'] > 80:
            signals.append('Money Flow Overbought (Bearish)')
            bearish_count += 1
    
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
            'ATR': round(latest.get('ATR', 0), 2) if not pd.isna(latest.get('ATR')) else None
        }
    }


def predict_next_candle(data):
    """
    Predict next candle direction using simple ML
    
    Args:
        data: DataFrame with indicators
    
    Returns:
        Dictionary with prediction details
    """
    df = data.copy()
    
    if len(df) < 30:
        return {
            'predicted_direction': 'INSUFFICIENT_DATA',
            'predicted_change': 0,
            'confidence': 0,
            'note': f'Need at least 30 candles for prediction (have {len(df)}). Get more historical data.'
        }
    
    try:
        # Prepare features - use only available ones
        available_features = []
        feature_cols = ['RSI', 'MACD', 'SMA_20', 'SMA_50', 'ATR', 'Volume', 'MFI', 'Stoch_K']
        
        for col in feature_cols:
            if col in df.columns:
                available_features.append(col)
        
        if len(available_features) < 3:
            return {
                'predicted_direction': 'INSUFFICIENT_INDICATORS',
                'predicted_change': 0,
                'confidence': 0,
                'note': 'Not enough indicators calculated for ML prediction'
            }
        
        # Remove rows with NaN in available features
        df_clean = df.dropna(subset=available_features)
        
        if len(df_clean) < 20:
            return {
                'predicted_direction': 'INSUFFICIENT_CLEAN_DATA',
                'predicted_change': 0,
                'confidence': 0,
                'note': 'Too many missing values in indicator data'
            }
        
        # Create target: 1 if next close > current close, 0 otherwise
        df_clean['Target'] = (df_clean['Close'].shift(-1) > df_clean['Close']).astype(int)
        
        # Remove last row (no target)
        df_clean = df_clean[:-1]
        
        if len(df_clean) < 15:
            return {
                'predicted_direction': 'UNKNOWN',
                'predicted_change': 0,
                'confidence': 0,
                'note': 'Need more data points for reliable ML prediction'
            }
        
        X = df_clean[available_features]
        y = df_clean['Target']
        
        # Train simple model with reduced complexity for small datasets
        from sklearn.ensemble import RandomForestClassifier
        n_estimators = min(30, len(df_clean) // 2)  # Adapt to data size
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=3, random_state=42)
        model.fit(X, y)
        
        # Predict for latest candle
        latest_features = df[available_features].iloc[[-1]]
        
        if latest_features.isna().any().any():
            return {
                'predicted_direction': 'UNKNOWN',
                'predicted_change': 0,
                'confidence': 0,
                'note': 'Latest candle has missing indicator values'
            }
        
        prediction = model.predict(latest_features)[0]
        probability = model.predict_proba(latest_features)[0]
        
        predicted_direction = 'UP' if prediction == 1 else 'DOWN'
        confidence = round(max(probability) * 100, 2)
        
        # Estimate price change based on ATR or recent volatility
        current_price = df.iloc[-1]['Close']
        
        if not pd.isna(df.iloc[-1].get('ATR')) and df.iloc[-1]['ATR'] > 0:
            atr = df.iloc[-1]['ATR']
        else:
            # Use recent price volatility as fallback
            recent_returns = df['Close'].pct_change().tail(10)
            atr = recent_returns.std() * current_price
        
        if predicted_direction == 'UP':
            predicted_price = current_price + (atr * 0.5)
            predicted_change = round(((predicted_price / current_price) - 1) * 100, 2)
        else:
            predicted_price = current_price - (atr * 0.5)
            predicted_change = round(((predicted_price / current_price) - 1) * 100, 2)
        
        return {
            'predicted_direction': predicted_direction,
            'predicted_price': round(predicted_price, 2),
            'predicted_change': predicted_change,
            'confidence': confidence,
            'current_price': round(current_price, 2),
            'model': f'Random Forest ({n_estimators} trees)',
            'features_used': len(available_features)
        }
    
    except Exception as e:
        return {
            'predicted_direction': 'ERROR',
            'predicted_change': 0,
            'confidence': 0,
            'note': f'Prediction error: {str(e)}'
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
