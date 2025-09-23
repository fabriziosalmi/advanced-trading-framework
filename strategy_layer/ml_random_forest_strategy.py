"""
Advanced Trading Framework - ML Random Forest Strategy

This module implements a machine learning trading strategy using Random Forest
classifier, encapsulating all the logic from data handling, feature engineering,
and model training for signal generation.

Author: Senior Python Software Architect
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import os
import pickle
import json
import logging
import random
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from datetime import datetime, timedelta
import pytz
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import weakref

if TYPE_CHECKING:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    RandomForestClassifier = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False
    print("scikit-learn not installed. Install with: pip install scikit-learn")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except (ImportError, TypeError, Exception) as e:
    yf = None
    YFINANCE_AVAILABLE = False
    print(f"Warning: yfinance not available in ml_random_forest_strategy ({e}). Using simulated data.")

try:
    from . import advanced_features
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    advanced_features = None
    ADVANCED_FEATURES_AVAILABLE = False
    print("Warning: advanced_features module not available. Using basic features only.")

from .strategy_base import Strategy
from .signals import TradingSignal, create_buy_signal, create_sell_signal
from core.error_handler import retry_with_backoff, graceful_degradation, error_handler, ErrorCategory, ErrorSeverity


class MLRandomForestStrategy(Strategy):
    """
    Machine Learning Random Forest Trading Strategy.
    
    This strategy encapsulates all ML logic including:
    - Data fetching and preprocessing
    - Feature engineering
    - Model training and persistence
    - Signal generation with confidence scoring
    
    The strategy automatically trains models for each ticker and 
    generates buy/sell signals based on ML predictions.
    """
    
    def __init__(self, confidence_threshold: float = 0.6):
        """
        Initialize ML Random Forest Strategy.
        
        Args:
            confidence_threshold: Minimum confidence required to generate a signal
        """
        super().__init__("MLRandomForest", confidence_threshold)
        
        if RandomForestClassifier is None:
            raise ImportError("scikit-learn is required for MLRandomForestStrategy")
        
        # Model storage
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.model_dir = "models"
        
        # Data cache with thread-safe access (using regular dict instead of WeakValueDictionary for thread safety)
        self.data_cache: Dict[str, Any] = {}
        self.data_cache_lock = threading.RLock()  # Reentrant lock for thread safety
        self.model_lock = threading.RLock()  # Lock for model loading/saving operations
        self.last_data_update: Dict[str, datetime] = {}
        
        # Thread pool for CPU-intensive operations
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="MLStrategy")
        
        # LRU cache for expensive computations
        self._feature_cache: Dict[str, Any] = {}
        self._regime_cache: Dict[str, str] = {}
        
        # Performance monitoring
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Feature configuration - comprehensive feature set
        self.lookback_period = 120  # Increased for better training data
        self.feature_columns = [
            # Trend Indicators
            'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'ema_5', 'ema_10', 'ema_20', 'ema_50',
            'price_to_sma_20', 'price_to_sma_50', 'sma_20_to_sma_50',
            
            # Momentum Indicators
            'rsi', 'rsi_oversold', 'rsi_overbought',
            'stoch_k', 'stoch_d', 'williams_r',
            'roc_5', 'roc_10', 'roc_20',
            
            # Volatility Indicators
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
            'atr', 'atr_ratio',
            'volatility_10', 'volatility_20', 'volatility_50',
            
            # Volume Indicators
            'volume_sma_10', 'volume_sma_20', 'volume_ratio',
            'obv', 'vpt', 'cmf',
            
            # Oscillators
            'macd', 'macd_signal', 'macd_histogram', 'cci',
            
            # Price Action
            'price_change_1d', 'price_change_3d', 'price_change_5d', 'price_change_10d',
            'daily_range', 'body_size', 'upper_shadow', 'lower_shadow',
            'gap_up', 'gap_down',
            
            # Trend Strength
            'adx', 'trend_up', 'trend_strength',
            
            # Statistical Features
            'returns_skew_20', 'returns_kurtosis_20', 'close_zscore_20',
            'momentum_10', 'momentum_20', 'acceleration',
            
            # Advanced Volatility Features
            'atr_14', 'bollinger_bandwidth_20', 'volatility_ratio_5_20',
            
            # Advanced Momentum Features
            'momentum_decay_3', 'momentum_decay_6', 'momentum_decay_12',
            'momentum_acceleration_3_6', 'momentum_acceleration_6_12',
            'roc_acceleration_3_10', 'roc_acceleration_6_10',
            
            # Market Context Features
            'spy_close', 'spy_returns', 'spy_volatility',
            'relative_strength', 'relative_momentum_5', 'relative_momentum_20',
            'market_trend', 'market_volatility_ratio'
        ]
        
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for monitoring.
        
        Returns:
            Dictionary with performance metrics
        """
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        with self.data_cache_lock:
            data_cache_size = len(self.data_cache)
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': cache_hit_rate,
            'models_loaded': len(self.models),
            'data_cache_size': data_cache_size,
            'feature_cache_size': len(self._feature_cache),
            'regime_cache_size': len(self._regime_cache)
        }
    
    def cleanup(self):
        """Clean up resources and shutdown thread pool."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        
        # Clear caches
        with self.data_cache_lock:
            self.data_cache.clear()
        self._feature_cache.clear()
        self._regime_cache.clear()
        
        self.logger.info("ML Strategy resources cleaned up")

    def _cleanup_expired_cache_entries(self):
        """Clean up expired cache entries to prevent memory leaks."""
        now = datetime.now()
        expired_tickers = []
        
        with self.data_cache_lock:
            for ticker, last_update in self.last_data_update.items():
                if (now - last_update).total_seconds() > 86400:  # 24 hours
                    expired_tickers.append(ticker)
            
            for ticker in expired_tickers:
                if ticker in self.data_cache:
                    del self.data_cache[ticker]
                if ticker in self.last_data_update:
                    del self.last_data_update[ticker]
        
        if expired_tickers:
            self.logger.debug(f"Cleaned up {len(expired_tickers)} expired cache entries")
    
    def __del__(self):
        """Ensure cleanup is called when object is destroyed."""
        self.cleanup()
    
    async def initialize(self) -> bool:
        """Initialize ML strategy components."""
        try:
            self.logger.info("Initializing ML Random Forest Strategy")
            
            if not YFINANCE_AVAILABLE:
                self.logger.warning("yfinance not available, strategy will have limited functionality")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ML strategy: {str(e)}")
            self.cleanup()  # Ensure cleanup on initialization failure
            return False
    
    def _get_model_path(self, ticker: str) -> tuple[str, str]:
        """
        Get file paths for model and scaler.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Tuple of (model_path, scaler_path)
        """
        model_path = os.path.join(self.model_dir, f"{ticker}_rf_model.pkl")
        scaler_path = os.path.join(self.model_dir, f"{ticker}_scaler.pkl")
        return model_path, scaler_path
    
    def _model_exists(self, ticker: str) -> bool:
        """
        Check if a trained model exists for the given ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            True if both model and scaler files exist, False otherwise
        """
        model_path, scaler_path = self._get_model_path(ticker)
        return os.path.exists(model_path) and os.path.exists(scaler_path)
    
    @retry_with_backoff(max_retries=3, base_delay=2.0, exceptions=(Exception,))
    def _fetch_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a ticker with intelligent caching.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        # Periodic cleanup of expired cache entries
        if random.random() < 0.1:  # 10% chance to cleanup on each call
            self._cleanup_expired_cache_entries()
        
        # Check cache first
        cache_key = f"{ticker}_data"
        now = datetime.now()
        
        with self.data_cache_lock:
            if (ticker in self.data_cache and 
                ticker in self.last_data_update and
                (now - self.last_data_update[ticker]).seconds < 3600):  # Cache for 1 hour
                self.cache_hits += 1
                self.logger.debug(f"Using cached data for {ticker}")
                return self.data_cache[ticker]
        
        self.cache_misses += 1
        try:
            if not YFINANCE_AVAILABLE:
                error_handler.handle_error(
                    ImportError("yfinance not available"),
                    ErrorCategory.DATA,
                    ErrorSeverity.HIGH,
                    {"ticker": ticker}
                )
                return None
            
            # Fetch data with longer period for robust training
            # Note: Using hourly data for more granular analysis and increased training data
            try:
                ticker_obj = yf.Ticker(ticker)
                data = ticker_obj.history(
                    period="2y",  # Reduced to 2 years for hourly data (yfinance limits)
                    interval='1h',
                    timeout=30  # 30 second timeout
                )
            except Exception as e:
                error_handler.handle_error(
                    e,
                    ErrorCategory.DATA,
                    ErrorSeverity.HIGH,
                    {"ticker": ticker, "operation": "yfinance_data_fetch"}
                )
                return None
            
            if data.empty:
                error_handler.handle_error(
                    ValueError(f"No data found for {ticker}"),
                    ErrorCategory.DATA,
                    ErrorSeverity.MEDIUM,
                    {"ticker": ticker, "period": "5y"}
                )
                return None
            
            # yfinance returns columns like 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'
            # Keep only the columns we need and standardize names
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Check if all required columns exist
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                self.logger.error(f"Missing columns {missing_columns} for {ticker}")
                return None
            
            # Select and rename columns
            data = data[required_columns].copy()
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Fetch additional market data for enhanced features
            extended_data = self._fetch_extended_market_data(data.index[0], data.index[-1])
            if extended_data is not None:
                # Merge extended data with main ticker data
                data = data.join(extended_data, how='left')
                # Forward fill missing values to handle weekends/holidays
                data = data.fillna(method='ffill')
                self.logger.info(f"✅ Extended market data merged: {len(extended_data.columns)} additional columns")
            
            self.logger.info(f"Fetched {len(data)} data points for {ticker} with {len(data.columns)} columns")
            
            # Step 1.2: Empirically verify timestamps
            self.logger.info(f"📅 First 5 timestamps for {ticker}: {data.index[:5].tolist()}")
            
            # Step 1.3: Verify data time interval
            if len(data) >= 2:
                time_delta = data.index[1] - data.index[0]
                self.logger.info(f"🕒 Data time interval verified: {time_delta}")
            else:
                self.logger.warning(f"⚠️  Insufficient data points to verify time interval for {ticker}")
            
            # Step 1.4: Filter for market hours (9:30 AM - 4:00 PM ET) for hourly data
            if str(time_delta).startswith('0 days 01:00:00'):  # Check if hourly data
                # Convert to Eastern Time
                eastern = pytz.timezone('US/Eastern')
                data.index = data.index.tz_convert(eastern) if data.index.tz else data.index.tz_localize('UTC').tz_convert(eastern)
                
                # Filter for market hours: 9:30 AM to 4:00 PM ET, Monday-Friday
                market_hours = (data.index.hour >= 9) & (data.index.hour <= 15) & (data.index.weekday < 5)
                # More precise: 9:30-16:00 (4 PM)
                market_hours = market_hours & (
                    ((data.index.hour == 9) & (data.index.minute >= 30)) |
                    ((data.index.hour > 9) & (data.index.hour < 16)) |
                    ((data.index.hour == 16) & (data.index.minute == 0))
                )
                
                data = data[market_hours]
                self.logger.info(f"🏛️ Filtered to market hours: {len(data)} hourly bars remaining")
            
            # Cache the data
            with self.data_cache_lock:
                self.data_cache[ticker] = data
                self.last_data_update[ticker] = now
            
            return data
            
        except Exception as e:
            error_handler.handle_error(
                e,
                ErrorCategory.DATA,
                ErrorSeverity.MEDIUM,
                {"ticker": ticker, "operation": "data_fetch"}
            )
            return None

    def _fetch_extended_market_data(self, start_date, end_date):
        """
        Fetch extended market data including VIX, commodities, Treasury yields, and sector ETFs.
        This provides additional context for more sophisticated trading signals.
        
        Args:
            start_date: Start date for data fetching
            end_date: End date for data fetching
            
        Returns:
            DataFrame with extended market data or None if fetching fails
        """
        try:
            self.logger.info("📊 Fetching extended market data...")
            
            # Define additional data sources
            additional_tickers = {
                'VIX': '^VIX',  # CBOE Volatility Index
                'SPY': 'SPY',   # S&P 500 ETF for market breadth
                'QQQ': 'QQQ',   # Nasdaq 100 ETF
                'IWM': 'IWM',   # Russell 2000 ETF
                'GLD': 'GLD',   # Gold ETF
                'USO': 'USO',   # Oil ETF
                'TLT': 'TLT',   # 20+ Year Treasury Bond ETF
                'IEF': 'IEF',   # 7-10 Year Treasury Bond ETF
                'SHY': 'SHY',   # 1-3 Year Treasury Bond ETF
                'XLE': 'XLE',   # Energy sector ETF
                'XLF': 'XLF',   # Financial sector ETF
                'XLK': 'XLK',   # Technology sector ETF
                'XLV': 'XLV',   # Healthcare sector ETF
                'XLI': 'XLI',   # Industrial sector ETF
            }
            
            extended_data = pd.DataFrame()
            
            for name, ticker in additional_tickers.items():
                try:
                    ticker_obj = yf.Ticker(ticker)
                    data = ticker_obj.history(
                        start=start_date,
                        end=end_date,
                        interval='1h',
                        timeout=15
                    )
                    
                    if not data.empty and 'Close' in data.columns:
                        # Rename close column to avoid conflicts
                        extended_data[f'{name}_close'] = data['Close']
                        self.logger.debug(f"✅ Fetched {name} ({ticker}): {len(data)} days")
                    else:
                        self.logger.warning(f"⚠️  No data for {name} ({ticker})")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️  Failed to fetch {name} ({ticker}): {str(e)}")
                    continue
            
            if extended_data.empty:
                self.logger.warning("⚠️  No extended market data could be fetched")
                return None
                
            # Calculate market breadth indicators
            if 'SPY_close' in extended_data.columns:
                # Simple market breadth: SPY returns
                extended_data['SPY_returns'] = extended_data['SPY_close'].pct_change()
            
            # Calculate yield spreads if we have Treasury data
            if 'TLT_close' in extended_data.columns and 'SHY_close' in extended_data.columns:
                # Rough approximation of yield spread (inverse relationship with price)
                extended_data['yield_spread_20y_2y'] = extended_data['SHY_close'] / extended_data['TLT_close']
            
            # Calculate commodity ratios
            if 'GLD_close' in extended_data.columns and 'USO_close' in extended_data.columns:
                extended_data['gold_oil_ratio'] = extended_data['GLD_close'] / extended_data['USO_close']
            
            self.logger.info(f"✅ Extended market data ready: {len(extended_data.columns)} indicators")
            return extended_data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch extended market data: {str(e)}")
            return None
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators with STRICT CAUSALITY GUARANTEE.

        This method processes data sequentially, day by day, ensuring that each feature
        calculation for day t uses ONLY data available up to and including day t.
        This eliminates any possibility of lookahead bias.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with technical indicators
        """
        self.logger.info(f"Starting CAUSAL _calculate_technical_indicators with data shape: {data.shape if data is not None else 'None'}")
        self.logger.info(f"Input data columns: {list(data.columns) if data is not None else 'None'}")

        df = data.copy()

        # Handle MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Normalize column names to lowercase for consistency
        df.columns = df.columns.str.lower()

        # Initialize result DataFrame with same index and all required columns
        result_df = df.copy()
        
        # Initialize all feature columns with NaN
        feature_columns = [
            # Core features only - optimized for hourly timeframe
            'rsi_7',            # Momentum indicator (reduced from 14 to 7 for hourly)
            'atr_7',            # Volatility indicator (reduced from 14 to 7)
            'sma_20',           # Needed for slope and price_to_sma calculations (reduced from 50 to 20)
            'sma_20_slope',     # Trend indicator (reduced from 50 to 20)
            'macd_hist',        # Momentum/trend indicator (kept same, MACD works well on hourly)
            'volume_ratio_10',  # Volume indicator (reduced from 20 to 10)
            'spy_returns',      # Market context
            'price_to_sma_20',  # Value/mean reversion (updated from 50 to 20)
            # Regime detection features
            'regime_bull',      # Bull market indicator
            'regime_bear',      # Bear market indicator
            'regime_neutral',   # Neutral market indicator
            'volatility_regime', # High/low volatility regime
            # Relative strength features
            'relative_strength',    # Stock return vs SPY return
            'relative_momentum_3',  # 3-hour relative momentum (reduced from 5)
            'relative_momentum_12', # 12-hour relative momentum (reduced from 20)
            # Momentum decay/acceleration features
            'momentum_decay_3',         # 3-hour momentum decay (reduced from 5)
            'momentum_decay_6',         # 6-hour momentum decay (reduced from 10)
            'momentum_acceleration_3_6', # Acceleration from 3h to 6h momentum (reduced from 5_10)
            'momentum_acceleration_6_12', # Acceleration from 6h to 12h momentum (reduced from 10_20)
            # Additional useful features
            'ema_10',           # Exponential moving average 10 (reduced from 20)
            'rsi_4',            # RSI 4-hour (reduced from 7)
            'bb_position',      # Bollinger Band position (-1 to 1)
            'stoch_k',          # Stochastic %K
            'williams_r',       # Williams %R
            # Advanced volatility features
            'bollinger_bandwidth_10',  # Bollinger Band Width (reduced from 20 to 10)
            'volatility_ratio_3_10',   # Volatility ratio (3h vs 10h, reduced from 5_20)
            # Momentum acceleration features
            'roc_acceleration_3_10',   # ROC acceleration (3h vs 10h, reduced from 5_20)
            'roc_acceleration_6_10',  # ROC acceleration (6h vs 10h, reduced from 10_20)
            # Additional momentum decay features
            'momentum_decay_12',       # 12-hour momentum decay (reduced from 20)
            # Oscillators and momentum indicators
            'cci_7',                  # Commodity Channel Index (7-hour, reduced from 14)
            'obv',                     # On-Balance Volume
            # Extended market data features
            'vix_close',               # VIX volatility index
            'vix_ma_10',               # VIX 10-hour moving average (reduced from 20)
            'vix_regime',              # VIX-based market regime (fear/greed)
            'spy_close',               # S&P 500 ETF price
            'qqq_close',               # Nasdaq 100 ETF price
            'iwm_close',               # Russell 2000 ETF price
            'gld_close',               # Gold ETF price
            'uso_close',               # Oil ETF price
            'yield_spread_20y_2y',     # Treasury yield spread
            'gold_oil_ratio',          # Gold to oil ratio
            'sector_tech_vs_energy',   # Tech sector vs energy sector
            'market_breadth',          # Market breadth (SPY vs Russell 2000)
            # Advanced technical indicators
            'ichimoku_tenkan',         # Ichimoku Tenkan-sen (Conversion Line)
            'ichimoku_kijun',          # Ichimoku Kijun-sen (Base Line)
            'ichimoku_senkou_a',       # Ichimoku Senkou Span A (Leading Span A)
            'ichimoku_senkou_b',       # Ichimoku Senkou Span B (Leading Span B)
            'ichimoku_chikou',         # Ichimoku Chikou Span (Lagging Span)
            'keltner_upper',           # Keltner Channel Upper Band
            'keltner_lower',           # Keltner Channel Lower Band
            'keltner_position',        # Position within Keltner Channel (-1 to 1)
            'parabolic_sar',           # Parabolic SAR
            'chaikin_money_flow',      # Chaikin Money Flow (11-hour, reduced from 21)
            'vwap',                    # Volume Weighted Average Price
            'vwap_deviation',          # Price deviation from VWAP
        ]
        
        for col in feature_columns:
            if col not in result_df.columns:
                result_df[col] = np.nan

        # Process data sequentially to ensure strict causality
        self.logger.info("🔄 Processing data sequentially for causality...")

        for i in range(len(df)):
            current_data = df.iloc[:i+1]  # Data up to and including current day

            # === CORE FEATURES ONLY - MINIMAL SET ===

            # RSI (7-hour, optimized for hourly data)
            if i >= 7:
                delta = current_data['close'].diff().tail(7)
                gain = (delta.where(delta > 0, 0)).mean()
                loss = (-delta.where(delta < 0, 0)).mean()
                rs = gain / loss if loss != 0 else 100
                rsi = 100 - (100 / (1 + rs))
                result_df.iloc[i, result_df.columns.get_loc('rsi_7')] = rsi

            # ATR (7-hour, optimized for hourly data)
            if i >= 6:
                high_low = current_data['high'].tail(7) - current_data['low'].tail(7)
                high_close = (current_data['high'].tail(7) - current_data['close'].shift(1).tail(7)).abs()
                low_close = (current_data['low'].tail(7) - current_data['close'].shift(1).tail(7)).abs()
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.mean()
                result_df.iloc[i, result_df.columns.get_loc('atr_7')] = atr

            # SMA 20 and slope (optimized for hourly data)
            if i >= 19:
                result_df.iloc[i, result_df.columns.get_loc('sma_20')] = current_data['close'].tail(20).mean()
                # Calculate SMA slope (change over last 3 hours, optimized for hourly)
                if i >= 22:
                    sma_20_now = result_df['sma_20'].iloc[i]
                    sma_20_prev = result_df['sma_20'].iloc[i-3]
                    result_df.iloc[i, result_df.columns.get_loc('sma_20_slope')] = (sma_20_now - sma_20_prev) / sma_20_prev if sma_20_prev != 0 else 0

            # Price to SMA 20 (optimized for hourly data)
            if i >= 19 and not pd.isna(result_df.iloc[i]['sma_20']):
                result_df.iloc[i, result_df.columns.get_loc('price_to_sma_20')] = (current_data['close'].iloc[-1] / result_df.iloc[i]['sma_20']) - 1

            # MACD Histogram
            if i >= 25:  # Need enough data for all EMAs
                close_data = current_data['close']
                ema_12 = close_data.ewm(span=12).mean()
                ema_26 = close_data.ewm(span=26).mean()
                macd = ema_12.iloc[-1] - ema_26.iloc[-1]
                macd_signal = pd.Series([macd]).ewm(span=9).mean().iloc[-1] if hasattr(result_df, 'macd') and len(result_df['macd'].dropna()) > 0 else macd
                macd_histogram = macd - macd_signal
                result_df.iloc[i, result_df.columns.get_loc('macd_hist')] = macd_histogram

            # Volume Ratio (10-hour, optimized for hourly data)
            if i >= 9:
                result_df.iloc[i, result_df.columns.get_loc('volume_ratio_10')] = current_data['volume'].iloc[-1] / current_data['volume'].tail(10).mean()

            # === ADDITIONAL USEFUL FEATURES ===
            
            # EMA 10 (optimized for hourly data)
            if i >= 9:
                result_df.iloc[i, result_df.columns.get_loc('ema_10')] = current_data['close'].ewm(span=10).mean().iloc[-1]

            # RSI 7 (shorter timeframe)
            if i >= 7:
                delta_7 = current_data['close'].diff().tail(7)
                gain_7 = (delta_7.where(delta_7 > 0, 0)).mean()
                loss_7 = (-delta_7.where(delta_7 < 0, 0)).mean()
                rs_7 = gain_7 / loss_7 if loss_7 != 0 else 100
                rsi_7 = 100 - (100 / (1 + rs_7))
                result_df.iloc[i, result_df.columns.get_loc('rsi_7')] = rsi_7

            # Bollinger Bands Position (-1 to 1, where price is relative to bands)
            if i >= 19:
                sma_20 = current_data['close'].tail(20).mean()
                std_20 = current_data['close'].tail(20).std()
                upper_band = sma_20 + (std_20 * 2)
                lower_band = sma_20 - (std_20 * 2)
                current_price = current_data['close'].iloc[-1]
                
                # Position: -1 (at lower band) to 1 (at upper band)
                if upper_band != lower_band:
                    bb_position = (current_price - sma_20) / ((upper_band - lower_band) / 2)
                    bb_position = max(-1, min(1, bb_position))  # Clamp to [-1, 1]
                else:
                    bb_position = 0
                result_df.iloc[i, result_df.columns.get_loc('bb_position')] = bb_position

            # Stochastic %K (14-day)
            if i >= 13:
                high_14 = current_data['high'].tail(14).max()
                low_14 = current_data['low'].tail(14).min()
                current_close = current_data['close'].iloc[-1]
                
                if high_14 != low_14:
                    stoch_k = ((current_close - low_14) / (high_14 - low_14)) * 100
                else:
                    stoch_k = 50  # Neutral when no range
                result_df.iloc[i, result_df.columns.get_loc('stoch_k')] = stoch_k

            # Williams %R (14-day)
            if i >= 13:
                high_14 = current_data['high'].tail(14).max()
                low_14 = current_data['low'].tail(14).min()
                current_close = current_data['close'].iloc[-1]
                
                if high_14 != low_14:
                    williams_r = ((high_14 - current_close) / (high_14 - low_14)) * -100
                else:
                    williams_r = -50  # Neutral when no range
                result_df.iloc[i, result_df.columns.get_loc('williams_r')] = williams_r

            # Commodity Channel Index (7-hour, optimized for hourly data)
            if i >= 6:
                # Calculate Typical Price: (High + Low + Close) / 3
                typical_prices = (current_data['high'] + current_data['low'] + current_data['close']).tail(7) / 3
                
                # Calculate SMA of Typical Price
                sma_tp = typical_prices.mean()
                
                # Calculate Mean Deviation
                mean_deviation = (typical_prices - sma_tp).abs().mean()
                
                # Calculate CCI
                if mean_deviation != 0:
                    current_tp = (current_data['high'].iloc[-1] + current_data['low'].iloc[-1] + current_data['close'].iloc[-1]) / 3
                    cci = (current_tp - sma_tp) / (0.015 * mean_deviation)
                else:
                    cci = 0  # Neutral when no deviation
                result_df.iloc[i, result_df.columns.get_loc('cci_7')] = cci

            # On-Balance Volume (OBV)
            if i >= 1:  # Need at least 2 hours for OBV
                # Get current and previous close prices
                current_close = current_data['close'].iloc[-1]
                prev_close = current_data['close'].iloc[-2]
                current_volume = current_data['volume'].iloc[-1]
                
                # Get previous OBV value (if exists)
                prev_obv = result_df.iloc[i-1]['obv'] if i > 1 and not pd.isna(result_df.iloc[i-1]['obv']) else 0
                
                # Calculate OBV
                if current_close > prev_close:
                    obv = prev_obv + current_volume  # Price up, add volume
                elif current_close < prev_close:
                    obv = prev_obv - current_volume  # Price down, subtract volume
                else:
                    obv = prev_obv  # Price unchanged, OBV unchanged
                
                result_df.iloc[i, result_df.columns.get_loc('obv')] = obv

            # Bollinger Band Width (10-hour, optimized for hourly data) - volatility measure
            if i >= 9:
                sma_10 = current_data['close'].tail(10).mean()
                std_10 = current_data['close'].tail(10).std()
                upper_band = sma_10 + (std_10 * 2)
                lower_band = sma_10 - (std_10 * 2)
                
                # Bandwidth = (upper - lower) / middle
                if sma_10 != 0:
                    bb_width = (upper_band - lower_band) / sma_10
                else:
                    bb_width = 0
                result_df.iloc[i, result_df.columns.get_loc('bollinger_bandwidth_10')] = bb_width

            # Volatility Ratio (3-hour vs 10-hour, optimized for hourly data)
            if i >= 9:
                # Calculate returns
                returns = current_data['close'].pct_change()
                
                # Short-term volatility (3-hour)
                vol_3 = returns.tail(3).std()
                
                # Long-term volatility (10-hour)
                vol_10 = returns.tail(10).std()
                
                # Ratio: short-term / long-term volatility
                if vol_10 != 0:
                    vol_ratio = vol_3 / vol_10
                else:
                    vol_ratio = 1.0  # Neutral when no long-term volatility
                result_df.iloc[i, result_df.columns.get_loc('volatility_ratio_3_10')] = vol_ratio

            # ROC Acceleration (3-hour vs 10-hour, optimized for hourly data)
            if i >= 9:
                # Calculate ROC for different periods
                roc_3 = (current_data['close'].iloc[-1] - current_data['close'].iloc[-4]) / current_data['close'].iloc[-4] if i >= 3 else 0
                roc_10 = (current_data['close'].iloc[-1] - current_data['close'].iloc[-11]) / current_data['close'].iloc[-11] if i >= 10 else 0
                
                # Acceleration: difference between short and long ROC
                roc_accel_3_10 = roc_3 - roc_10
                result_df.iloc[i, result_df.columns.get_loc('roc_acceleration_3_10')] = roc_accel_3_10

            # ROC Acceleration (6-hour vs 10-hour, optimized for hourly data)
            if i >= 9:
                # Calculate ROC for 6-hour and 10-hour
                roc_6 = (current_data['close'].iloc[-1] - current_data['close'].iloc[-7]) / current_data['close'].iloc[-7] if i >= 6 else 0
                roc_10 = (current_data['close'].iloc[-1] - current_data['close'].iloc[-11]) / current_data['close'].iloc[-11] if i >= 10 else 0
                
                # Acceleration: difference between 6h and 10h ROC
                roc_accel_6_10 = roc_6 - roc_10
                result_df.iloc[i, result_df.columns.get_loc('roc_acceleration_6_10')] = roc_accel_6_10

            # === MOMENTUM DECAY AND ACCELERATION FEATURES ===
            # Calculate momentum decay (how much momentum has weakened over time)
            if i >= 12:  # Need 12 hours for decay calculations
                # 12-hour momentum decay: current 12-hour momentum vs 12-hour momentum 12 hours ago
                current_momentum_12h = current_data['close'].pct_change(12).iloc[-1]
                past_momentum_12h = current_data['close'].pct_change(12).iloc[-13] if i >= 24 else current_momentum_12h
                momentum_decay_12 = current_momentum_12h - past_momentum_12h
                result_df.iloc[i, result_df.columns.get_loc('momentum_decay_12')] = momentum_decay_12

            if i >= 6:  # Need 6 hours for 3-hour decay
                # 3-hour momentum decay: current 3-hour momentum vs 3-hour momentum 3 hours ago
                current_momentum_3h = current_data['close'].pct_change(3).iloc[-1]
                past_momentum_3h = current_data['close'].pct_change(3).iloc[-4] if i >= 6 else current_momentum_3h
                momentum_decay_3 = current_momentum_3h - past_momentum_3h
                result_df.iloc[i, result_df.columns.get_loc('momentum_decay_3')] = momentum_decay_3

                # 6-hour momentum decay: current 6-hour momentum vs 6-hour momentum 6 hours ago
                current_momentum_6h = current_data['close'].pct_change(6).iloc[-1]
                past_momentum_6h = current_data['close'].pct_change(6).iloc[-7] if i >= 12 else current_momentum_6h
                momentum_decay_6 = current_momentum_6h - past_momentum_6h
                result_df.iloc[i, result_df.columns.get_loc('momentum_decay_6')] = momentum_decay_6

                # Momentum acceleration: rate of change of momentum
                momentum_3h = current_data['close'].pct_change(3).iloc[-1]
                momentum_6h = current_data['close'].pct_change(6).iloc[-1]
                momentum_acceleration_3_6 = momentum_6h - momentum_3h  # Difference between longer and shorter momentum
                result_df.iloc[i, result_df.columns.get_loc('momentum_acceleration_3_6')] = momentum_acceleration_3_6

            if i >= 12:  # Need 12 hours for 12-hour decay
                # 12-hour momentum decay: current 12-hour momentum vs 12-hour momentum 12 hours ago
                current_momentum_12h = current_data['close'].pct_change(12).iloc[-1]
                past_momentum_12h = current_data['close'].pct_change(12).iloc[-13] if i >= 24 else current_momentum_12h
                momentum_decay_12 = current_momentum_12h - past_momentum_12h
                result_df.iloc[i, result_df.columns.get_loc('momentum_decay_12')] = momentum_decay_12

            if i >= 12:  # Need 12 hours for 12-hour momentum comparison
                momentum_6h = current_data['close'].pct_change(6).iloc[-1]
                momentum_12h = current_data['close'].pct_change(12).iloc[-1]
                momentum_acceleration_6_12 = momentum_12h - momentum_6h  # Difference between 12h and 6h momentum
                result_df.iloc[i, result_df.columns.get_loc('momentum_acceleration_6_12')] = momentum_acceleration_6_12

            # === REGIME DETECTION FEATURES ===
            # Bull/Bear/Neutral market classification based on trend and momentum
            if i >= 20 and not pd.isna(result_df.iloc[i]['sma_20_slope']):
                price_to_sma = result_df.iloc[i]['price_to_sma_20']
                slope = result_df.iloc[i]['sma_20_slope']
                
                # Bull regime: price above SMA and positive slope
                if price_to_sma > 0.02 and slope > 0.001:  # 2% above SMA, positive slope
                    result_df.iloc[i, result_df.columns.get_loc('regime_bull')] = 1
                    result_df.iloc[i, result_df.columns.get_loc('regime_bear')] = 0
                    result_df.iloc[i, result_df.columns.get_loc('regime_neutral')] = 0
                # Bear regime: price below SMA and negative slope
                elif price_to_sma < -0.02 and slope < -0.001:  # 2% below SMA, negative slope
                    result_df.iloc[i, result_df.columns.get_loc('regime_bull')] = 0
                    result_df.iloc[i, result_df.columns.get_loc('regime_bear')] = 1
                    result_df.iloc[i, result_df.columns.get_loc('regime_neutral')] = 0
                # Neutral regime: everything else
                else:
                    result_df.iloc[i, result_df.columns.get_loc('regime_bull')] = 0
                    result_df.iloc[i, result_df.columns.get_loc('regime_bear')] = 0
                    result_df.iloc[i, result_df.columns.get_loc('regime_neutral')] = 1

            # Volatility regime classification
            if i >= 7 and not pd.isna(result_df.iloc[i]['atr_7']):
                # Compare current ATR to its 20-hour moving average (optimized for hourly data)
                if i >= 27:  # Need enough data for ATR MA (7 + 20)
                    atr_ma = result_df['atr_7'].iloc[i-20:i].mean()
                    current_atr = result_df.iloc[i]['atr_7']
                    # High volatility if ATR > 1.2 * ATR_MA, Low if ATR < 0.8 * ATR_MA
                    if current_atr > atr_ma * 1.2:
                        result_df.iloc[i, result_df.columns.get_loc('volatility_regime')] = 1  # High volatility
                    elif current_atr < atr_ma * 0.8:
                        result_df.iloc[i, result_df.columns.get_loc('volatility_regime')] = -1  # Low volatility
                    else:
                        result_df.iloc[i, result_df.columns.get_loc('volatility_regime')] = 0  # Normal volatility

        # === MARKET DATA INTEGRATION ===
        # Fetch SPY data for relative strength calculations
        try:
            spy_data = self._fetch_market_index_data(df.index[0], df.index[-1])
            if spy_data is not None:
                # Normalize datetime indexes to prevent comparison warnings
                df.index = pd.to_datetime(df.index, utc=True)
                spy_data.index = pd.to_datetime(spy_data.index, utc=True)

                # Process market data sequentially for relative strength
                for i in range(len(result_df)):
                    if i >= 1:  # Need at least 1 day for returns
                        # Find corresponding SPY data point
                        current_date = df.index[i]
                        spy_mask = spy_data.index <= current_date
                        spy_row = spy_data[spy_mask]
                        if len(spy_row) > 0:
                            spy_close = spy_row['spy_close'].iloc[-1]
                            spy_prev_close = spy_row['spy_close'].iloc[-2] if len(spy_row) > 1 else spy_close
                            spy_return = (spy_close - spy_prev_close) / spy_prev_close if spy_prev_close != 0 else 0
                            
                            # Update spy_returns with actual SPY data
                            result_df.iloc[i, result_df.columns.get_loc('spy_returns')] = spy_return
                            
                            # Calculate relative strength: stock return - SPY return
                            stock_return = result_df['close'].pct_change(1).iloc[i] if i >= 1 else 0
                            relative_strength = stock_return - spy_return
                            result_df.iloc[i, result_df.columns.get_loc('relative_strength')] = relative_strength
                            
                            # Calculate relative momentum (3-hour and 12-hour, optimized for hourly data)
                            if i >= 3:
                                stock_returns_3h = result_df['close'].pct_change(3).iloc[i] if i >= 3 else 0
                                spy_returns_3h = (spy_close - spy_row['spy_close'].iloc[-4]) / spy_row['spy_close'].iloc[-4] if len(spy_row) > 3 else 0
                                result_df.iloc[i, result_df.columns.get_loc('relative_momentum_3')] = stock_returns_3h - spy_returns_3h
                            
                            if i >= 12:
                                stock_returns_12h = result_df['close'].pct_change(12).iloc[i] if i >= 12 else 0
                                spy_returns_12h = (spy_close - spy_row['spy_close'].iloc[-13]) / spy_row['spy_close'].iloc[-13] if len(spy_row) > 12 else 0
                                result_df.iloc[i, result_df.columns.get_loc('relative_momentum_12')] = stock_returns_12h - spy_returns_12h
        except Exception as e:
            self.logger.warning(f"Failed to add market data features: {e}")
            # Fallback: use simplified market context
            for i in range(len(result_df)):
                if i >= 1:
                    stock_return = result_df['close'].pct_change(1).iloc[i] if i >= 1 else 0
                    # Use stock return as rough market proxy (scaled down)
                    result_df.iloc[i, result_df.columns.get_loc('spy_returns')] = stock_return * 0.7
                    # Relative strength as difference from market proxy
                    result_df.iloc[i, result_df.columns.get_loc('relative_strength')] = stock_return * 0.3
                    # Simplified momentum calculations
                    if i >= 3:
                        result_df.iloc[i, result_df.columns.get_loc('relative_momentum_3')] = result_df['close'].pct_change(3).iloc[i] * 0.3
                    if i >= 12:
                        result_df.iloc[i, result_df.columns.get_loc('relative_momentum_12')] = result_df['close'].pct_change(12).iloc[i] * 0.3

        # === EXTENDED MARKET DATA FEATURES ===
        try:
            for i in range(len(result_df)):
                current_data = result_df.iloc[:i+1]
                
                # VIX features
                if 'vix_close' in current_data.columns and not pd.isna(current_data['vix_close'].iloc[-1]):
                    result_df.iloc[i, result_df.columns.get_loc('vix_close')] = current_data['vix_close'].iloc[-1]
                    
                    # VIX 10-hour moving average (optimized for hourly data)
                    if i >= 9:
                        vix_ma_10 = current_data['vix_close'].tail(10).mean()
                        result_df.iloc[i, result_df.columns.get_loc('vix_ma_10')] = vix_ma_10
                        
                        # VIX regime: 0 (low fear) to 1 (high fear)
                        # VIX < 15: low fear, VIX > 30: high fear
                        current_vix = current_data['vix_close'].iloc[-1]
                        vix_regime = min(1.0, max(0.0, (current_vix - 15) / 15))  # Scale 15-30 to 0-1
                        result_df.iloc[i, result_df.columns.get_loc('vix_regime')] = vix_regime
                
                # ETF prices
                for etf in ['spy_close', 'qqq_close', 'iwm_close', 'gld_close', 'uso_close']:
                    if etf in current_data.columns and not pd.isna(current_data[etf].iloc[-1]):
                        result_df.iloc[i, result_df.columns.get_loc(etf)] = current_data[etf].iloc[-1]
                
                # Treasury yield spread
                if 'yield_spread_20y_2y' in current_data.columns and not pd.isna(current_data['yield_spread_20y_2y'].iloc[-1]):
                    result_df.iloc[i, result_df.columns.get_loc('yield_spread_20y_2y')] = current_data['yield_spread_20y_2y'].iloc[-1]
                
                # Gold to oil ratio
                if 'gold_oil_ratio' in current_data.columns and not pd.isna(current_data['gold_oil_ratio'].iloc[-1]):
                    result_df.iloc[i, result_df.columns.get_loc('gold_oil_ratio')] = current_data['gold_oil_ratio'].iloc[-1]
                
                # Sector comparisons
                if i >= 20:  # Need some history for meaningful comparisons (optimized for hourly data)
                    if ('xlk_close' in current_data.columns and 'xle_close' in current_data.columns and
                        not pd.isna(current_data['xlk_close'].iloc[-1]) and not pd.isna(current_data['xle_close'].iloc[-1])):
                        # Tech sector vs Energy sector (momentum comparison)
                        tech_return = current_data['xlk_close'].pct_change(20).iloc[-1] if i >= 20 else 0
                        energy_return = current_data['xle_close'].pct_change(20).iloc[-1] if i >= 20 else 0
                        sector_tech_vs_energy = tech_return - energy_return
                        result_df.iloc[i, result_df.columns.get_loc('sector_tech_vs_energy')] = sector_tech_vs_energy
                    
                    # Market breadth: SPY vs Russell 2000 (large cap vs small cap)
                    if ('spy_close' in current_data.columns and 'iwm_close' in current_data.columns and
                        not pd.isna(current_data['spy_close'].iloc[-1]) and not pd.isna(current_data['iwm_close'].iloc[-1])):
                        spy_return = current_data['spy_close'].pct_change(20).iloc[-1] if i >= 19 else 0
                        iwm_return = current_data['iwm_close'].pct_change(20).iloc[-1] if i >= 19 else 0
                        market_breadth = spy_return - iwm_return  # Positive = large caps outperforming
                        result_df.iloc[i, result_df.columns.get_loc('market_breadth')] = market_breadth
                        
        except Exception as e:
            self.logger.warning(f"Failed to calculate extended market features: {e}")

        # === ADVANCED TECHNICAL INDICATORS ===
        try:
            for i in range(len(result_df)):
                current_data = result_df.iloc[:i+1]
                
                # === ICHIMOKU CLOUD ===
                if i >= 25:  # Need enough data for Ichimoku calculations
                    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
                    tenkan_high = current_data['high'].tail(9).max()
                    tenkan_low = current_data['low'].tail(9).min()
                    tenkan_sen = (tenkan_high + tenkan_low) / 2
                    result_df.iloc[i, result_df.columns.get_loc('ichimoku_tenkan')] = tenkan_sen
                    
                    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
                    kijun_high = current_data['high'].tail(26).max()
                    kijun_low = current_data['low'].tail(26).min()
                    kijun_sen = (kijun_high + kijun_low) / 2
                    result_df.iloc[i, result_df.columns.get_loc('ichimoku_kijun')] = kijun_sen
                    
                    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, plotted 26 periods ahead
                    senkou_a = (tenkan_sen + kijun_sen) / 2
                    result_df.iloc[i, result_df.columns.get_loc('ichimoku_senkou_a')] = senkou_a
                    
                    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead
                    if i >= 51:
                        senkou_high = current_data['high'].tail(52).max()
                        senkou_low = current_data['low'].tail(52).min()
                        senkou_b = (senkou_high + senkou_low) / 2
                        result_df.iloc[i, result_df.columns.get_loc('ichimoku_senkou_b')] = senkou_b
                    
                    # Chikou Span (Lagging Span): Current close plotted 26 periods back
                    result_df.iloc[i, result_df.columns.get_loc('ichimoku_chikou')] = current_data['close'].iloc[-1]
                
                # === KELTNER CHANNELS ===
                if i >= 19:  # Need 20 periods for ATR calculation
                    # Calculate ATR for Keltner Channels (using 14-period ATR)
                    if i >= 13:
                        atr_values = []
                        for j in range(max(0, i-13), i+1):
                            if j >= 1:
                                high_low = current_data['high'].iloc[j] - current_data['low'].iloc[j]
                                high_close = abs(current_data['high'].iloc[j] - current_data['close'].iloc[j-1])
                                low_close = abs(current_data['low'].iloc[j] - current_data['close'].iloc[j-1])
                                tr = max(high_low, high_close, low_close)
                                atr_values.append(tr)
                        
                        if atr_values:
                            atr = sum(atr_values) / len(atr_values)
                            
                            # Keltner Channels: EMA(20) +/- 2*ATR(14)
                            ema_20 = current_data['close'].ewm(span=20).mean().iloc[-1]
                            keltner_upper = ema_20 + (atr * 2)
                            keltner_lower = ema_20 - (atr * 2)
                            
                            result_df.iloc[i, result_df.columns.get_loc('keltner_upper')] = keltner_upper
                            result_df.iloc[i, result_df.columns.get_loc('keltner_lower')] = keltner_lower
                            
                            # Position within Keltner Channel (-1 to 1)
                            current_price = current_data['close'].iloc[-1]
                            if keltner_upper != keltner_lower:
                                keltner_position = (current_price - ema_20) / ((keltner_upper - keltner_lower) / 2)
                                keltner_position = max(-1, min(1, keltner_position))
                                result_df.iloc[i, result_df.columns.get_loc('keltner_position')] = keltner_position
                
                # === PARABOLIC SAR ===
                if i >= 2:  # Need at least 3 periods
                    # Simplified Parabolic SAR calculation
                    # Start with acceleration factor of 0.02, max 0.2
                    if i == 2:
                        # Initialize SAR
                        sar = current_data['low'].iloc[0]  # Start at first low
                        acceleration = 0.02
                        extreme_point = current_data['high'].iloc[0]
                        is_uptrend = True
                        prev_sar = sar  # Initialize prev_sar for first calculation
                    else:
                        # Get previous SAR value
                        prev_sar = result_df.iloc[i-1]['parabolic_sar'] if not pd.isna(result_df.iloc[i-1]['parabolic_sar']) else current_data['low'].iloc[0]
                        acceleration = result_df.iloc[i-1].get('parabolic_acceleration', 0.02)
                        extreme_point = result_df.iloc[i-1].get('parabolic_extreme', current_data['high'].iloc[0])
                        is_uptrend = result_df.iloc[i-1].get('parabolic_trend', True)
                    
                    current_high = current_data['high'].iloc[-1]
                    current_low = current_data['low'].iloc[-1]
                    
                    # Calculate new SAR
                    if is_uptrend:
                        new_sar = prev_sar + acceleration * (extreme_point - prev_sar)
                        new_sar = min(new_sar, current_low)  # Don't go above current low
                        
                        if current_high > extreme_point:
                            extreme_point = current_high
                            acceleration = min(acceleration + 0.02, 0.2)
                        
                        # Check for trend reversal
                        if new_sar > current_low:
                            is_uptrend = False
                            new_sar = extreme_point
                            extreme_point = current_low
                            acceleration = 0.02
                    else:
                        new_sar = prev_sar - acceleration * (prev_sar - extreme_point)
                        new_sar = max(new_sar, current_high)  # Don't go below current high
                        
                        if current_low < extreme_point:
                            extreme_point = current_low
                            acceleration = min(acceleration + 0.02, 0.2)
                        
                        # Check for trend reversal
                        if new_sar < current_high:
                            is_uptrend = True
                            new_sar = extreme_point
                            extreme_point = current_high
                            acceleration = 0.02
                    
                    result_df.iloc[i, result_df.columns.get_loc('parabolic_sar')] = new_sar
                    
                    # Store intermediate values for next iteration
                    result_df.loc[result_df.index[i], 'parabolic_acceleration'] = acceleration
                    result_df.loc[result_df.index[i], 'parabolic_extreme'] = extreme_point
                    result_df.loc[result_df.index[i], 'parabolic_trend'] = is_uptrend
                
                # === CHAIKIN MONEY FLOW ===
                if i >= 20:  # Need 21 periods
                    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
                    # Money Flow Volume = Money Flow Multiplier * Volume
                    # Chaikin Money Flow = 21-period EMA of Money Flow Volume / 21-period EMA of Volume
                    
                    cmf_values = []
                    volumes = []
                    
                    for j in range(max(0, i-20), i+1):
                        high = current_data['high'].iloc[j]
                        low = current_data['low'].iloc[j]
                        close = current_data['close'].iloc[j]
                        volume = current_data['volume'].iloc[j]
                        
                        if high != low:
                            mfm = ((close - low) - (high - close)) / (high - low)
                            mfv = mfm * volume
                            cmf_values.append(mfv)
                            volumes.append(volume)
                    
                    if cmf_values and volumes:
                        # 21-period EMA of MFV and Volume
                        cmf = sum(cmf_values) / sum(volumes) if sum(volumes) != 0 else 0
                        result_df.iloc[i, result_df.columns.get_loc('chaikin_money_flow')] = cmf
                
                # === VWAP (Volume Weighted Average Price) ===
                if i >= 0:  # Can calculate from day 1
                    # VWAP = sum(Price * Volume) / sum(Volume) for the period
                    price_volume_sum = 0
                    volume_sum = 0
                    
                    for j in range(i+1):
                        typical_price = (current_data['high'].iloc[j] + current_data['low'].iloc[j] + current_data['close'].iloc[j]) / 3
                        price_volume_sum += typical_price * current_data['volume'].iloc[j]
                        volume_sum += current_data['volume'].iloc[j]
                    
                    if volume_sum > 0:
                        vwap = price_volume_sum / volume_sum
                        result_df.iloc[i, result_df.columns.get_loc('vwap')] = vwap
                        
                        # VWAP Deviation: (Current Price - VWAP) / VWAP
                        current_price = current_data['close'].iloc[-1]
                        vwap_deviation = (current_price - vwap) / vwap if vwap != 0 else 0
                        result_df.iloc[i, result_df.columns.get_loc('vwap_deviation')] = vwap_deviation
                        
        except Exception as e:
            self.logger.warning(f"Failed to calculate advanced technical indicators: {e}")

        # === ADVANCED FEATURES ===
        # Simplified - using only core features for baseline
        pass

        self.logger.info(f"✅ Causal feature engineering completed. Shape: {result_df.shape}")
        return result_df
    
    def _fetch_market_index_data(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """
        Fetch S&P 500 (SPY) data for market context features.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with SPY data or None if failed
        """
        try:
            # Fetch SPY data
            spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False)
            
            if spy_data.empty:
                return None
            
            # Handle MultiIndex columns from yfinance
            if isinstance(spy_data.columns, pd.MultiIndex):
                # Flatten MultiIndex columns
                spy_data.columns = spy_data.columns.get_level_values(0)
            
            # Calculate market features
            spy_data['spy_close'] = spy_data['Close']
            spy_data['spy_returns'] = spy_data['spy_close'].pct_change()
            spy_data['spy_volatility'] = spy_data['spy_returns'].rolling(20).std() * np.sqrt(252)
            
            # Keep only needed columns
            result = spy_data[['spy_close', 'spy_returns', 'spy_volatility']].copy()
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Failed to fetch SPY market data: {e}")
            return None
    
    def _prepare_features_and_target(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target variable for ML training with simplified target.
        
        Args:
            data: DataFrame with technical indicators
            
        Returns:
            Tuple of (features_df, target_series)
        """
        # Step 2: Simplified target - 5-day forward price prediction
        data['simple_target'] = (data['close'].shift(-5) > data['close']).astype(int)
        
        # Select feature columns - using only core features optimized for hourly
        core_features = [
            'rsi_7',            # Momentum indicator (optimized for hourly)
            'atr_7',            # Volatility indicator (optimized for hourly)
            'sma_20_slope',     # Trend indicator (optimized for hourly)
            'macd_hist',        # Momentum/trend indicator
            'volume_ratio_10',  # Volume indicator (optimized for hourly)
            'spy_returns',      # Market context
            'price_to_sma_20',  # Value/mean reversion (optimized for hourly)
            'regime_bull',      # Bull market regime
            'regime_bear',      # Bear market regime
            'regime_neutral',   # Neutral market regime
            'volatility_regime', # Volatility regime
            'relative_strength',    # Stock vs market performance
            'relative_momentum_3',  # Short-term relative momentum (optimized for hourly)
            'relative_momentum_12', # Long-term relative momentum (optimized for hourly)
            'momentum_decay_3',     # Short-term momentum decay (optimized for hourly)
            'momentum_decay_6',     # Medium-term momentum decay (optimized for hourly)
            'momentum_acceleration_3_6', # Momentum acceleration 3h->6h (optimized for hourly)
            'momentum_acceleration_6_12', # Momentum acceleration 6h->12h (optimized for hourly)
            # Additional useful features
            'ema_10',           # Exponential moving average 10 (optimized for hourly)
            'rsi_4',            # RSI 4-hour (optimized for hourly)
            'bb_position',      # Bollinger Band position
            'stoch_k',          # Stochastic %K
            'williams_r',       # Williams %R
            # Advanced volatility features
            'bollinger_bandwidth_10',  # Bollinger Band Width (optimized for hourly)
            'volatility_ratio_3_10',   # Volatility ratio (3h vs 10h, optimized for hourly)
            # Momentum acceleration features
            'roc_acceleration_3_10',   # ROC acceleration (3h vs 10h, optimized for hourly)
            'roc_acceleration_6_10',  # ROC acceleration (6h vs 10h, optimized for hourly)
            # Additional momentum decay features
            'momentum_decay_12',       # 12-hour momentum decay (optimized for hourly)
            # Oscillators and momentum indicators
            'cci_7',                  # Commodity Channel Index (7-hour, optimized for hourly)
            'obv',                     # On-Balance Volume
            # Extended market data features
            'vix_close',               # VIX volatility index
            'vix_ma_10',               # VIX 10-hour moving average (optimized for hourly)
            'vix_regime',              # VIX-based market regime
            'spy_close',               # S&P 500 ETF price
            'qqq_close',               # Nasdaq 100 ETF price
            'iwm_close',               # Russell 2000 ETF price
            'gld_close',               # Gold ETF price
            'uso_close',               # Oil ETF price
            'yield_spread_20y_2y',     # Treasury yield spread
            'gold_oil_ratio',          # Gold to oil ratio
            'sector_tech_vs_energy',   # Tech sector vs energy sector
            'market_breadth',          # Market breadth (SPY vs Russell 2000)
            # Advanced technical indicators
            'ichimoku_tenkan',         # Ichimoku Tenkan-sen (Conversion Line)
            'ichimoku_kijun',          # Ichimoku Kijun-sen (Base Line)
            'ichimoku_senkou_a',       # Ichimoku Senkou Span A (Leading Span A)
            'ichimoku_senkou_b',       # Ichimoku Senkou Span B (Leading Span B)
            'ichimoku_chikou',         # Ichimoku Chikou Span (Lagging Span)
            'keltner_upper',           # Keltner Channel Upper Band
            'keltner_lower',           # Keltner Channel Lower Band
            'keltner_position',        # Position within Keltner Channel (-1 to 1)
            'parabolic_sar',           # Parabolic SAR
            'chaikin_money_flow',      # Chaikin Money Flow (11-hour, optimized for hourly)
            'vwap',                    # Volume Weighted Average Price
            'vwap_deviation',          # Price deviation from VWAP
        ]
        
        self.logger.info(f"📊 Selecting {len(core_features)} core features from dataframe")
        self.logger.info(f"🎯 Core features: {core_features}")
        
        missing_features = [col for col in core_features if col not in data.columns]
        if missing_features:
            self.logger.error(f"❌ Missing core features: {missing_features}")
            return None, None
            
        feature_data = data[core_features].copy()
        target_data = data['simple_target'].copy()
        
        # Fill NaN values in features with 0
        feature_data = feature_data.fillna(0)
        
        # Only remove rows where target is NaN
        valid_indices = target_data.dropna().index
        feature_data = feature_data.loc[valid_indices]
        target_data = target_data.loc[valid_indices]
        
        # Remove last 5 rows (no target available due to shift(-5))
        if len(feature_data) > 5:
            feature_data = feature_data.iloc[:-5]
            target_data = target_data.iloc[:-5]
        
        return feature_data, target_data
    
    async def _train_model_for_ticker(self, ticker: str) -> bool:
        """
        Train an optimized Random Forest model for a specific ticker with hyperparameter tuning.
        Uses thread pool for CPU-intensive operations.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            self.logger.info(f"Training optimized model for {ticker}")
            
            # Fetch data
            data = self._fetch_data(ticker)
            if data is None or len(data) < 50:  # Reduced requirement for better compatibility
                self.logger.warning(f"Insufficient data for {ticker}")
                return False
            
            # Run CPU-intensive operations in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._train_model_sync, ticker, data
            )
            return result
                
        except Exception as e:
            error_handler.handle_error(
                e,
                ErrorCategory.MODEL,
                ErrorSeverity.HIGH,
                {"ticker": ticker, "operation": "model_training"}
            )
            return False
    
    def _train_model_sync(self, ticker: str, data: pd.DataFrame, skip_performance_check: bool = False) -> bool:
        """Synchronous model training logic (runs in thread pool)."""
        try:
            self.logger.info(f"🔄 Starting sync training for {ticker} with {len(data)} rows")
            
            # Calculate technical indicators
            self.logger.info("📊 Calculating technical indicators...")
            data_with_indicators = self._calculate_technical_indicators(data)
            self.logger.info(f"✅ Indicators calculated, dataframe shape: {data_with_indicators.shape}")
            self.logger.info(f"📋 Columns: {sorted(data_with_indicators.columns.tolist())}")
            
            # Prepare features and target
            self.logger.info("🎯 Preparing features and target...")
            features, target = self._prepare_features_and_target(data_with_indicators)
            
            # Split data with time-series awareness: 4 years training, 1 year testing
            # Assuming ~250 trading days per year, use 1000 days for training, rest for testing
            min_test_samples = 50  # Minimum samples for reliable testing
            if len(features) < 200:  # Too few samples overall
                self.logger.warning(f"Insufficient total samples for {ticker}: {len(features)}")
                return False
            
            # Use 80% for training, 20% for testing, but ensure minimum test samples
            if len(features) * 0.2 < min_test_samples:
                # If 20% would give too few test samples, use fixed test size
                test_size = min_test_samples
                split_idx = len(features) - test_size
            else:
                split_idx = int(len(features) * 0.8)
            
            # Ensure we have enough training data too
            if split_idx < 100:
                self.logger.warning(f"Insufficient training samples for {ticker}: {split_idx}")
                return False
                
            X_train = features.iloc[:split_idx]
            X_test = features.iloc[split_idx:]
            y_train = target.iloc[:split_idx]
            y_test = target.iloc[split_idx:]
            
            # Log split information with dates
            train_dates = features.index[:split_idx]
            test_dates = features.index[split_idx:]
            self.logger.info(f"📊 Train/Test Split for {ticker}:")
            self.logger.info(f"  - Training samples: {len(X_train)} ({train_dates.min()} to {train_dates.max()})")
            self.logger.info(f"  - Test samples: {len(X_test)} ({test_dates.min()} to {test_dates.max()})")
            self.logger.info(f"  - No overlap: Training ends {train_dates.max()}, Test starts {test_dates.min()}")
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Hyperparameter tuning with cross-validation
            best_model, best_params, best_score = self._optimize_hyperparameters(X_train_scaled, y_train)
            
            # Train final model with best parameters
            final_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
            final_model.fit(X_train_scaled, y_train)
            
            # Comprehensive evaluation
            evaluation_metrics = self._evaluate_model(final_model, X_test_scaled, y_test)
            
            self.logger.info(f"Model trained for {ticker} with:")
            self.logger.info(f"  - Best params: {best_params}")
            self.logger.info(f"  - CV Score: {best_score:.3f}")
            self.logger.info(f"  - Test Accuracy: {evaluation_metrics['accuracy']:.3f}")
            self.logger.info(f"  - Test Precision: {evaluation_metrics['precision']:.3f}")
            self.logger.info(f"  - Test Recall: {evaluation_metrics['recall']:.3f}")
            self.logger.info(f"  - Test F1-Score: {evaluation_metrics['f1_score']:.3f}")
            
            # Analyze feature importance for debugging
            self._analyze_feature_importance(final_model, features.columns.tolist(), ticker)
            
            # Only save if model meets minimum performance criteria (unless skip check)
            min_f1_threshold = self.config.get('strategy', {}).get('ml_random_forest', {}).get('min_f1_threshold', 0.4)  # Default to 0.4 for realistic baseline
            if skip_performance_check or evaluation_metrics['f1_score'] > min_f1_threshold:
                # Store model and scaler with thread safety
                with self.model_lock:
                    self.models[ticker] = final_model
                    self.scalers[ticker] = scaler
                    
                    # Save to disk
                    model_path, scaler_path = self._get_model_path(ticker)
                    joblib.dump(final_model, model_path)
                    joblib.dump(scaler, scaler_path)
                
                # Cache the processed data
                with self.data_cache_lock:
                    self.data_cache[ticker] = data_with_indicators
                self.last_data_update[ticker] = datetime.now()
                
                # Store evaluation metrics for later analysis
                self._store_model_metrics(ticker, evaluation_metrics, best_params)
                
                return True
            else:
                self.logger.warning(f"Model for {ticker} did not meet performance threshold (F1: {evaluation_metrics['f1_score']:.3f})")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to train model for {ticker}: {str(e)}")
            return False
    
    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: pd.Series) -> tuple:
        """
        Perform hyperparameter optimization using grid search with cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Tuple of (best_model, best_params, best_score)
        """
        from sklearn.model_selection import GridSearchCV
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': [None, 'balanced']
        }
        
        # Use time-series split for cross-validation
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Grid search
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            param_grid,
            cv=tscv,
            scoring='f1_weighted',  # Use F1 score for imbalanced classes
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
    def _evaluate_model(self, model, X_test: np.ndarray, y_test: pd.Series) -> Dict[str, float]:
        """
        Comprehensive model evaluation with multiple metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, classification_report
        )
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
        
        if y_pred_proba is not None and len(np.unique(y_test)) == 2:
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            except:
                metrics['roc_auc'] = 0.5
        
        return metrics
    
    def _analyze_feature_importance(self, model, feature_names: list, ticker: str):
        """
        Analyze and log feature importance for debugging data leakage.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            ticker: Ticker symbol for logging
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance_pairs = list(zip(feature_names, importances))
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            self.logger.info(f"🔍 Top 10 Feature Importances for {ticker}:")
            for i, (feature, importance) in enumerate(feature_importance_pairs[:10]):
                self.logger.info(f"  {i+1:2d}. {feature:<25} {importance:.4f}")
            
            # Check for suspiciously high importance (potential data leakage)
            max_importance = max(importances)
            if max_importance > 0.5:
                dominant_feature = feature_names[importances.argmax()]
                self.logger.warning(f"⚠️  POTENTIAL DATA LEAKAGE: {dominant_feature} has {max_importance:.3f} importance (>50%)")
                self.logger.warning("   This feature may be directly correlated with the target variable")
        else:
            self.logger.warning(f"Model for {ticker} does not have feature_importances_ attribute")
    
    def _store_model_metrics(self, ticker: str, metrics: Dict[str, float], params: Dict[str, Any]):
        """
        Store model performance metrics for analysis.
        
        Args:
            ticker: Stock ticker
            metrics: Evaluation metrics
            params: Best hyperparameters
        """
        # Create metrics file
        metrics_file = os.path.join(self.model_dir, f"{ticker}_metrics.json")
        
        metrics_data = {
            'ticker': ticker,
            'training_date': datetime.now().isoformat(),
            'hyperparameters': params,
            'evaluation_metrics': metrics,
            'feature_count': len(self.feature_columns)
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
    
    def _load_model_for_ticker(self, ticker: str) -> bool:
        """
        Load existing model for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        with self.model_lock:
            try:
                model_path, scaler_path = self._get_model_path(ticker)
                
                if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                    return False
                
                self.models[ticker] = joblib.load(model_path)
                self.scalers[ticker] = joblib.load(scaler_path)
                
                self.logger.info(f"Model loaded for {ticker}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to load model for {ticker}: {str(e)}")
                return False
    
    def _get_latest_features(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Get latest features for prediction.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Feature DataFrame for prediction or None if failed
        """
        try:
            # Check if we need to update data
            with self.data_cache_lock:
                should_update = (
                    ticker not in self.data_cache or
                    ticker not in self.last_data_update or
                    (datetime.now() - self.last_data_update[ticker]).total_seconds() > 3600  # 1 hour in seconds
                )
            
            if should_update:
                data = self._fetch_data(ticker)
                if data is not None:
                    with self.data_cache_lock:
                        self.data_cache[ticker] = self._calculate_technical_indicators(data)
                        self.last_data_update[ticker] = datetime.now()
            
            with self.data_cache_lock:
                if ticker not in self.data_cache:
                    return None
                
                # Get latest features as DataFrame to preserve column names
                latest_data = self.data_cache[ticker][self.feature_columns].iloc[-1:].fillna(0)
                
                if len(latest_data) == 0:
                    return None
                
                # Sanity checks on market data
                if not self._validate_market_data_sanity(self.data_cache[ticker]):
                    self.logger.warning(f"⚠️ Market data sanity check failed for {ticker} - skipping signal generation")
                    return None
            
            return latest_data
            
        except Exception as e:
            self.logger.error(f"Failed to get latest features for {ticker}: {str(e)}")
            return None
    
    def _validate_market_data_sanity(self, data: pd.DataFrame) -> bool:
        """
        Validate market data sanity before using it for predictions.
        
        Args:
            data: DataFrame with market data and technical indicators
            
        Returns:
            True if data passes sanity checks, False otherwise
        """
        try:
            if data is None or data.empty:
                self.logger.warning("Market data is None or empty")
                return False
            
            # Get the latest row
            latest = data.iloc[-1]
            
            # Check 1: Data is recent (not older than 1 hour)
            if 'timestamp' in data.columns:
                latest_timestamp = pd.to_datetime(latest['timestamp'])
                time_diff = datetime.now() - latest_timestamp.to_pydatetime()
                if time_diff.total_seconds() > 3600:  # 1 hour
                    self.logger.warning(f"Market data is too old: {time_diff.total_seconds()} seconds")
                    return False
            
            # Check 2: Closing price is realistic (not zero, not negative, not extremely high)
            close_price = latest.get('close', 0)
            if close_price <= 0:
                self.logger.warning(f"Invalid closing price: {close_price}")
                return False
            if close_price > 100000:  # Unrealistically high price
                self.logger.warning(f"Closing price too high: {close_price}")
                return False
            
            # Check 3: Volume is positive
            volume = latest.get('volume', 0)
            if volume <= 0:
                self.logger.warning(f"Invalid volume: {volume}")
                return False
            
            # Check 4: Price is within reasonable range (optional, can be adjusted)
            # For example, if close price is less than $0.01 or more than $10000, it might be suspicious
            if close_price < 0.01 or close_price > 10000:
                self.logger.warning(f"Closing price outside reasonable range: {close_price}")
                return False
            
            # All checks passed
            return True
            
        except Exception as e:
            self.logger.error(f"Error during market data sanity validation: {str(e)}")
            return False
    
    def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """
        Detect current market regime based on technical indicators.
        
        Args:
            data: DataFrame with technical indicators
            
        Returns:
            Market regime: 'bull', 'bear', 'sideways', 'volatile', 'unknown'
        """
        try:
            # Get latest values
            latest = data.iloc[-1]
            
            # Trend indicators
            adx = latest.get('adx', 20)
            trend_strength = latest.get('trend_strength', 0)
            sma_20_to_50 = latest.get('sma_20_to_sma_50', 0)
            
            # Volatility indicators
            volatility = latest.get('volatility_20', 0.02)
            bb_width = latest.get('bb_width', 0.1)
            
            # Momentum indicators
            rsi = latest.get('rsi', 50)
            macd_hist = latest.get('macd_histogram', 0)
            
            # Determine regime
            if adx > 25:  # Strong trend
                if sma_20_to_50 > 0.05 and trend_strength > 0.03:  # Bullish trend
                    return 'bull'
                elif sma_20_to_50 < -0.05 and trend_strength > 0.03:  # Bearish trend
                    return 'bear'
            
            if bb_width > 0.15 or volatility > 0.03:  # High volatility
                return 'volatile'
            
            if abs(sma_20_to_50) < 0.02 and adx < 20:  # Weak/no trend
                return 'sideways'
            
            return 'neutral'
            
        except Exception as e:
            self.logger.warning(f"Failed to detect market regime: {str(e)}")
            return 'unknown'
    
    def _adjust_confidence_for_regime(self, base_confidence: float, regime: str) -> float:
        """
        Adjust confidence threshold based on market regime.
        
        Args:
            base_confidence: Base confidence threshold
            regime: Current market regime
            
        Returns:
            Adjusted confidence threshold
        """
        regime_adjustments = {
            'bull': 0.65,      # Lower threshold in bull markets (easier to predict)
            'bear': 0.70,      # Higher threshold in bear markets (harder to predict)
            'sideways': 0.75,  # Higher threshold in sideways markets (random)
            'volatile': 0.75,  # Higher threshold in volatile markets (unpredictable)
            'neutral': 0.70,   # Standard threshold
            'unknown': 0.70    # Standard threshold
        }
        
        return max(base_confidence, regime_adjustments.get(regime, 0.70))
    
    def _get_regime_signal_multiplier(self, regime: str) -> float:
        """
        Get signal strength multiplier based on market regime.
        
        Args:
            regime: Current market regime
            
        Returns:
            Signal strength multiplier
        """
        multipliers = {
            'bull': 1.2,       # Amplify signals in bull markets
            'bear': 1.1,       # Moderate amplification in bear markets
            'sideways': 0.8,   # Reduce signals in sideways markets
            'volatile': 0.7,   # Significantly reduce signals in volatile markets
            'neutral': 1.0,    # No adjustment
            'unknown': 1.0     # No adjustment
        }
        
        return multipliers.get(regime, 1.0)
    
    def _calculate_risk_metrics(self, data: pd.DataFrame, regime: str) -> Dict[str, float]:
        """
        Calculate risk metrics for position sizing and stop loss levels.
        
        Args:
            data: DataFrame with technical indicators
            regime: Current market regime
            
        Returns:
            Dictionary of risk metrics
        """
        try:
            latest = data.iloc[-1]
            
            # Base risk metrics
            volatility = latest.get('volatility_20', 0.02)
            atr = latest.get('atr', latest.get('close', 100) * 0.02)
            trend_strength = latest.get('trend_strength', 0)
            
            # Adjust for regime
            regime_multipliers = {
                'bull': {'volatility': 0.8, 'atr': 0.9},
                'bear': {'volatility': 1.2, 'atr': 1.1},
                'sideways': {'volatility': 1.0, 'atr': 1.0},
                'volatile': {'volatility': 1.5, 'atr': 1.3},
                'neutral': {'volatility': 1.0, 'atr': 1.0},
                'unknown': {'volatility': 1.0, 'atr': 1.0}
            }
            
            mult = regime_multipliers.get(regime, {'volatility': 1.0, 'atr': 1.0})
            
            return {
                'adjusted_volatility': volatility * mult['volatility'],
                'adjusted_atr': atr * mult['atr'],
                'trend_strength': trend_strength,
                'sharpe_ratio': latest.get('sharpe_1d', 0),
                'max_drawdown': latest.get('close_zscore_20', 0)  # Using z-score as proxy
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate risk metrics: {str(e)}")
            return {
                'adjusted_volatility': 0.02,
                'adjusted_atr': 2.0,
                'trend_strength': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
    
    def _calculate_stop_levels(self, data: pd.DataFrame, signal_type: str, regime: str, risk_metrics: Dict[str, float]) -> tuple[float, float]:
        """
        Calculate dynamic stop loss and take profit levels based on risk metrics.
        
        Args:
            data: DataFrame with technical indicators
            signal_type: 'buy' or 'sell'
            regime: Current market regime
            risk_metrics: Risk metrics dictionary
            
        Returns:
            Tuple of (stop_loss_pct, take_profit_pct)
        """
        try:
            latest = data.iloc[-1]
            volatility = risk_metrics['adjusted_volatility']
            atr = risk_metrics['adjusted_atr']
            trend_strength = risk_metrics['trend_strength']
            
            # Base stop loss levels (as percentage)
            base_stop_loss = {
                'bull': 0.03,      # 3% in bull markets
                'bear': 0.04,      # 4% in bear markets
                'sideways': 0.02,  # 2% in sideways markets
                'volatile': 0.05,  # 5% in volatile markets
                'neutral': 0.035,  # 3.5% default
                'unknown': 0.035   # 3.5% default
            }
            
            # ATR-based stop loss (more sophisticated)
            current_price = latest.get('close', 100)
            atr_stop = (atr / current_price) * 2  # 2 ATR stop
            
            # Volatility-based stop loss
            vol_stop = volatility * 2.5  # 2.5 standard deviations
            
            # Use the most conservative stop loss
            stop_loss_pct = min(
                base_stop_loss.get(regime, 0.035),
                atr_stop,
                vol_stop,
                0.08  # Maximum 8% stop loss
            )
            
            # Take profit based on risk-reward ratio and trend strength
            risk_reward_ratio = 2.0 if trend_strength > 0.02 else 1.5  # Higher ratio for strong trends
            take_profit_pct = stop_loss_pct * risk_reward_ratio
            
            # Cap take profit at reasonable levels
            take_profit_pct = min(take_profit_pct, 0.15)  # Maximum 15% take profit
            
            return stop_loss_pct, take_profit_pct
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate stop levels: {str(e)}")
            return 0.035, 0.07  # Default 3.5% stop, 7% target
    
    def _generate_signal_reasoning(self, ticker: str, signal_type: str, confidence: float, 
                                  regime: str, risk_metrics: Dict[str, float], probabilities: np.ndarray) -> str:
        """
        Generate detailed reasoning for trading signals.
        
        Args:
            ticker: Stock ticker
            signal_type: 'buy' or 'sell'
            confidence: Model confidence
            regime: Market regime
            risk_metrics: Risk metrics
            probabilities: Prediction probabilities
            
        Returns:
            Detailed reasoning string
        """
        try:
            direction = "upward" if signal_type == "buy" else "downward"
            prob_buy = probabilities[1] if len(probabilities) > 1 else probabilities[0]
            prob_sell = probabilities[0] if len(probabilities) > 1 else 1 - probabilities[0]
            
            reasoning = (
                f"ML model predicts {direction} movement for {ticker} "
                f"with {confidence:.1%} confidence in {regime} market regime. "
                f"Buy probability: {prob_buy:.1%}, Sell probability: {prob_sell:.1%}. "
                f"Risk metrics: Volatility {risk_metrics['adjusted_volatility']:.1%}, "
                f"Trend strength {risk_metrics['trend_strength']:.1%}."
            )
            
            return reasoning
            
        except Exception as e:
            return f"ML model predicts {direction} movement with {confidence:.1%} confidence."
    
    @graceful_degradation(fallback_value=[], log_warning=True)
    async def generate_signals(self, tickers: List[str]) -> List[TradingSignal]:
        """
        Generate trading signals for given tickers using ML model.

        Args:
            tickers: List of ticker symbols to analyze

        Returns:
            List of TradingSignal objects
        """
        signals = []

        for ticker in tickers:
            try:
                # Get market data
                data = await self._get_market_data(ticker)
                if data is None or data.empty:
                    continue

                # Calculate technical indicators
                data_with_indicators = self._calculate_technical_indicators(data)

                # Prepare features for ML model
                features, _ = self._prepare_features_and_target(data_with_indicators)
                if features is None or features.empty:
                    continue

                # Check if model exists for this ticker and make prediction with thread safety
                with self.model_lock:
                    if ticker not in self.models:
                        self.logger.warning(f"No trained model available for {ticker}")
                        continue

                    # Get latest feature values
                    latest_features = features.iloc[-1:].values

                    # Make prediction
                    model = self.models[ticker]
                    prediction = model.predict_proba(latest_features)[0]
                    confidence = prediction[1]  # Probability of positive class (BUY)

                # Get current price
                current_price = data.iloc[-1]['close']

                # Generate signal based on confidence threshold
                if confidence >= self.confidence_threshold:
                    # Calculate position size
                    position_size = self.calculate_position_size(
                        current_price, self.portfolio.cash, confidence
                    )

                    if position_size > 0:
                        # Calculate stop loss and take profit levels
                        regime = self._detect_market_regime(data)
                        risk_metrics = self._calculate_risk_metrics(data)
                        stop_loss_pct, take_profit_pct = self._calculate_stop_levels(
                            data, 'buy', regime, risk_metrics
                        )
                        
                        # Calculate actual price levels
                        stop_loss_price = round(current_price * (1 - stop_loss_pct), 2)
                        take_profit_price = round(current_price * (1 + take_profit_pct), 2)
                        
                        signal = TradingSignal(
                            symbol=ticker,
                            side='BUY',
                            quantity=position_size,
                            price=current_price,
                            confidence=confidence,
                            stop_loss=stop_loss_price,
                            take_profit=take_profit_price,
                            timestamp=datetime.now(),
                            strategy_name=self.name
                        )
                        signals.append(signal)

            except Exception as e:
                self.logger.error(f"Error generating signal for {ticker}: {str(e)}")
                continue

        return signals

    async def _get_market_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Get market data for a ticker asynchronously.

        Args:
            ticker: Ticker symbol

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            # Use the existing _fetch_data method but make it async
            import asyncio
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, self._fetch_data, ticker)
            return data
        except Exception as e:
            self.logger.error(f"Error fetching market data for {ticker}: {str(e)}")
            return None

    def get_model_info(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get information about the model for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with model information or None
        """
        with self.model_lock:
            if ticker not in self.models:
                return None
            
            model = self.models[ticker]
            model_path, _ = self._get_model_path(ticker)
        
        # Get model file stats
        model_stats = {}
        if os.path.exists(model_path):
            stat = os.stat(model_path)
            model_stats = {
                'file_size': stat.st_size,
                'last_modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
        
        return {
            'model': model,
            'scaler': self.scalers.get(ticker),
            'metrics': self.model_metrics.get(ticker, {}),
            'stats': model_stats
        }
    
    # ===== WALK-FORWARD VALIDATION METHODS =====
    
    async def _fetch_data_async(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Async version of _fetch_data for walk-forward validation.
        """
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, self._fetch_data, ticker
        )
    
    async def _train_fold(self, ticker: str, train_data: pd.DataFrame, test_data: pd.DataFrame, fold_num: int):
        """
        Train a model on a single fold for walk-forward validation.
        
        Args:
            ticker: Stock ticker symbol
            train_data: Training data slice
            test_data: Testing data slice  
            fold_num: Fold number for logging
            
        Returns:
            Tuple of (predictions, true_labels) if successful, None otherwise
        """
        try:
            self.logger.debug(f"Training fold {fold_num} for {ticker}")
            
            # Process training data
            train_with_indicators = self._calculate_technical_indicators(train_data)
            X_train, y_train = self._prepare_features_and_target(train_with_indicators)
            
            # Process test data
            test_with_indicators = self._calculate_technical_indicators(test_data)
            X_test, y_test = self._prepare_features_and_target(test_with_indicators)
            
            # Scale features using training data scaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Quick training without hyperparameter optimization for speed
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            predictions = model.predict(X_test_scaled)
            
            return predictions.tolist(), y_test.tolist()
            
        except Exception as e:
            self.logger.error(f"❌ Exception in fold {fold_num}: {e}")
            return None
    
    async def _train_final_model_for_deployment(self, ticker: str, final_train_data: pd.DataFrame) -> bool:
        """
        Train the final model for deployment using the most recent data.
        
        Args:
            ticker: Stock ticker symbol
            final_train_data: Most recent training data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Training final deployment model for {ticker}")
            
            # Use the existing training logic but with the provided data slice
            # Pass flag to skip performance threshold check since walk-forward already validated
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._train_model_sync, ticker, final_train_data, True
            )
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Exception training final model: {e}")
            return False
       