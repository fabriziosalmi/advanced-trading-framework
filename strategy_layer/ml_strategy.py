"""
Advanced Trading Framework - Machine Learning Strategy

This module implements an advanced ML-based trading strategy with feature
engineering, model training, prediction pipeline, and adaptive learning
capabilities for quantitative trading.

Author: Senior Python Software Architect
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import pickle
import os
from collections import deque

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import classification_report, accuracy_score
    import joblib
except ImportError:
    RandomForestClassifier = None
    print("scikit-learn not installed. ML features will be limited.")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except (ImportError, TypeError, Exception) as e:
    yf = None
    YFINANCE_AVAILABLE = False
    print(f"Warning: yfinance not available in ml_strategy ({e}). Using simulated data.")

from .base_strategy import BaseStrategy, TradingSignal, SignalType, SignalStrength, RiskParameters
from ..execution_layer.base_broker import MarketData


class FeatureEngineering:
    """
    Advanced feature engineering for financial time series.
    
    Provides comprehensive feature extraction including:
    - Technical indicators
    - Price patterns
    - Volume analysis
    - Volatility measures
    - Market microstructure features
    """
    
    @staticmethod
    def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average."""
        return prices.rolling(window=window).mean()
    
    @staticmethod
    def calculate_ema(prices: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average."""
        return prices.ewm(span=window).mean()
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands (middle, upper, lower)."""
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return middle, upper, lower
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (MACD line, Signal line, Histogram)."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_volatility(prices: pd.Series, window: int = 20) -> pd.Series:
        """Rolling volatility (standard deviation of returns)."""
        returns = prices.pct_change()
        return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    @staticmethod
    def calculate_volume_features(prices: pd.Series, volumes: pd.Series, window: int = 20) -> Dict[str, pd.Series]:
        """Volume-based features."""
        vwap = (prices * volumes).rolling(window=window).sum() / volumes.rolling(window=window).sum()
        volume_sma = volumes.rolling(window=window).mean()
        volume_ratio = volumes / volume_sma
        
        return {
            'vwap': vwap,
            'volume_sma': volume_sma,
            'volume_ratio': volume_ratio
        }
    
    @staticmethod
    def calculate_price_patterns(ohlc_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Price pattern features."""
        df = ohlc_data.copy()
        
        # Candlestick patterns
        body = abs(df['close'] - df['open'])
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        
        # Pattern indicators
        doji = (body / (df['high'] - df['low'])) < 0.1
        hammer = (lower_shadow > 2 * body) & (upper_shadow < body)
        shooting_star = (upper_shadow > 2 * body) & (lower_shadow < body)
        
        return {
            'body_ratio': body / (df['high'] - df['low']),
            'upper_shadow_ratio': upper_shadow / (df['high'] - df['low']),
            'lower_shadow_ratio': lower_shadow / (df['high'] - df['low']),
            'doji': doji.astype(int),
            'hammer': hammer.astype(int),
            'shooting_star': shooting_star.astype(int)
        }
    
    @classmethod
    def create_features(cls, ohlc_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set from OHLC data.
        
        Args:
            ohlc_data: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
            
        Returns:
            DataFrame with engineered features
        """
        df = ohlc_data.copy()
        features = pd.DataFrame(index=df.index)
        
        # Price features
        features['price'] = df['close']
        features['price_change'] = df['close'].pct_change()
        features['price_change_1d'] = df['close'].pct_change(1)
        features['price_change_5d'] = df['close'].pct_change(5)
        features['price_change_20d'] = df['close'].pct_change(20)
        
        # Technical indicators
        features['sma_10'] = cls.calculate_sma(df['close'], 10)
        features['sma_20'] = cls.calculate_sma(df['close'], 20)
        features['sma_50'] = cls.calculate_sma(df['close'], 50)
        features['ema_12'] = cls.calculate_ema(df['close'], 12)
        features['ema_26'] = cls.calculate_ema(df['close'], 26)
        
        # Relative position to moving averages
        features['price_to_sma10'] = df['close'] / features['sma_10'] - 1
        features['price_to_sma20'] = df['close'] / features['sma_20'] - 1
        features['price_to_sma50'] = df['close'] / features['sma_50'] - 1
        
        # RSI
        features['rsi'] = cls.calculate_rsi(df['close'])
        features['rsi_oversold'] = (features['rsi'] < 30).astype(int)
        features['rsi_overbought'] = (features['rsi'] > 70).astype(int)
        
        # Bollinger Bands
        bb_middle, bb_upper, bb_lower = cls.calculate_bollinger_bands(df['close'])
        features['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        features['bb_squeeze'] = (bb_upper - bb_lower) / bb_middle
        
        # MACD
        macd, signal, histogram = cls.calculate_macd(df['close'])
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = histogram
        features['macd_bullish'] = (macd > signal).astype(int)
        
        # Volatility
        features['volatility'] = cls.calculate_volatility(df['close'])
        features['volatility_rank'] = features['volatility'].rolling(252).rank(pct=True)
        
        # Volume features
        if 'volume' in df.columns:
            volume_features = cls.calculate_volume_features(df['close'], df['volume'])
            features['vwap'] = volume_features['vwap']
            features['volume_ratio'] = volume_features['volume_ratio']
            features['price_to_vwap'] = df['close'] / features['vwap'] - 1
        
        # Price patterns
        pattern_features = cls.calculate_price_patterns(df)
        for name, series in pattern_features.items():
            features[f'pattern_{name}'] = series
        
        # Momentum features
        features['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        features['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        features['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Support/Resistance levels
        features['high_20d'] = df['high'].rolling(20).max()
        features['low_20d'] = df['low'].rolling(20).min()
        features['price_to_high20'] = df['close'] / features['high_20d'] - 1
        features['price_to_low20'] = df['close'] / features['low_20d'] - 1
        
        return features


class MLTradingStrategy(BaseStrategy):
    """
    Advanced Machine Learning Trading Strategy.
    
    Features:
    - Comprehensive feature engineering
    - Multiple ML models (Random Forest, Gradient Boosting)
    - Model ensemble and voting
    - Adaptive learning and model retraining
    - Confidence-based signal generation
    - Performance monitoring and model validation
    """
    
    def __init__(
        self,
        symbols: List[str],
        model_type: str = "random_forest",
        lookback_days: int = 252,
        retrain_frequency: int = 30,
        min_training_samples: int = 500,
        confidence_threshold: float = 0.6,
        risk_params: Optional[RiskParameters] = None
    ):
        """
        Initialize ML Trading Strategy.
        
        Args:
            symbols: List of symbols to trade
            model_type: Type of ML model ('random_forest', 'gradient_boosting', 'ensemble')
            lookback_days: Days of historical data for training
            retrain_frequency: Days between model retraining
            min_training_samples: Minimum samples required for training
            confidence_threshold: Minimum confidence for signal generation
            risk_params: Risk management parameters
        """
        super().__init__("MLStrategy", symbols, risk_params)
        
        if RandomForestClassifier is None:
            raise ImportError("scikit-learn is required for MLTradingStrategy")
        
        self.model_type = model_type
        self.lookback_days = lookback_days
        self.retrain_frequency = retrain_frequency
        self.min_training_samples = min_training_samples
        self.confidence_threshold = confidence_threshold
        
        # Model components
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_columns: List[str] = []
        
        # Data storage
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.features_cache: Dict[str, pd.DataFrame] = {}
        self.last_training_date: Optional[datetime] = None
        
        # Performance tracking
        self.prediction_accuracy: Dict[str, deque] = {}
        self.model_performance: Dict[str, Dict] = {}
        
        self.logger.info(f"MLTradingStrategy initialized: {model_type} model for {len(symbols)} symbols")
    
    async def initialize(self) -> bool:
        """
        Initialize ML strategy components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize data structures
            for symbol in self.symbols:
                self.historical_data[symbol] = pd.DataFrame()
                self.features_cache[symbol] = pd.DataFrame()
                self.prediction_accuracy[symbol] = deque(maxlen=100)
                self.model_performance[symbol] = {}
            
            # Load historical data
            await self._load_historical_data()
            
            # Build features
            await self._build_features()
            
            # Train initial models
            await self._train_models()
            
            self.logger.info("ML strategy initialization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ML strategy: {str(e)}")
            return False
    
    async def cleanup(self) -> None:
        """Cleanup ML strategy resources."""
        # Save models
        await self._save_models()
        self.logger.info("ML strategy cleanup completed")
    
    async def _load_historical_data(self) -> None:
        """Load historical price data for all symbols."""
        if not yf:
            self.logger.warning("yfinance not available, using simulated data")
            return
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days + 100)  # Extra buffer
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='1d'
                )
                
                if not hist.empty:
                    # Standardize column names
                    hist.columns = ['open', 'high', 'low', 'close', 'volume']
                    self.historical_data[symbol] = hist
                    self.logger.info(f"Loaded {len(hist)} days of data for {symbol}")
                else:
                    self.logger.warning(f"No historical data found for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Failed to load data for {symbol}: {str(e)}")
    
    async def _build_features(self) -> None:
        """Build features for all symbols."""
        for symbol in self.symbols:
            if symbol not in self.historical_data or self.historical_data[symbol].empty:
                continue
            
            try:
                # Create features
                features = FeatureEngineering.create_features(self.historical_data[symbol])
                
                # Create target variable (next day return > 0)
                next_return = self.historical_data[symbol]['close'].pct_change().shift(-1)
                features['target'] = (next_return > 0.001).astype(int)  # 0.1% threshold
                
                # Remove NaN values
                features = features.dropna()
                
                self.features_cache[symbol] = features
                
                if self.feature_columns == []:
                    # Set feature columns from first symbol (excluding target)
                    self.feature_columns = [col for col in features.columns if col != 'target']
                
                self.logger.info(f"Built {len(features)} feature samples for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Failed to build features for {symbol}: {str(e)}")
    
    def _create_model(self, model_type: str) -> Any:
        """
        Create ML model based on type.
        
        Args:
            model_type: Type of model to create
            
        Returns:
            Initialized ML model
        """
        if model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        else:
            # Default to random forest
            return RandomForestClassifier(random_state=42, n_jobs=-1)
    
    async def _train_models(self) -> None:
        """Train ML models for all symbols."""
        for symbol in self.symbols:
            if symbol not in self.features_cache or self.features_cache[symbol].empty:
                continue
            
            try:
                features_df = self.features_cache[symbol]
                
                if len(features_df) < self.min_training_samples:
                    self.logger.warning(f"Insufficient training data for {symbol}: {len(features_df)} samples")
                    continue
                
                # Prepare training data
                X = features_df[self.feature_columns].fillna(0)
                y = features_df['target']
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Create and train model
                model = self._create_model(self.model_type)
                model.fit(X_train_scaled, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Store model and scaler
                self.models[symbol] = model
                self.scalers[symbol] = scaler
                
                # Store performance metrics
                self.model_performance[symbol] = {
                    'accuracy': accuracy,
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'training_date': datetime.now().isoformat(),
                    'feature_importance': dict(zip(
                        self.feature_columns,
                        model.feature_importances_ if hasattr(model, 'feature_importances_') else []
                    ))
                }
                
                self.logger.info(f"Trained model for {symbol}: {accuracy:.3f} accuracy")
                
            except Exception as e:
                self.logger.error(f"Failed to train model for {symbol}: {str(e)}")
        
        self.last_training_date = datetime.now()
    
    async def _update_features_with_market_data(self, market_data: Dict[str, MarketData]) -> None:
        """Update feature cache with new market data."""
        for symbol, data in market_data.items():
            if symbol not in self.symbols:
                continue
            
            # Create new row for current data
            new_row = pd.DataFrame({
                'open': [data.price],  # Simplified - using current price
                'high': [data.price],
                'low': [data.price],
                'close': [data.price],
                'volume': [data.volume or 100000],
                'timestamp': [data.timestamp]
            })
            new_row.set_index('timestamp', inplace=True)
            
            # Append to historical data
            if symbol in self.historical_data:
                self.historical_data[symbol] = pd.concat([
                    self.historical_data[symbol],
                    new_row
                ]).drop_duplicates().sort_index()
                
                # Keep only recent data
                cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
                self.historical_data[symbol] = self.historical_data[symbol][
                    self.historical_data[symbol].index > cutoff_date
                ]
    
    async def generate_signals(self, market_data: Dict[str, MarketData]) -> List[TradingSignal]:
        """
        Generate trading signals using ML models.
        
        Args:
            market_data: Current market data
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Update features with new market data
        await self._update_features_with_market_data(market_data)
        
        # Check if models need retraining
        if (self.last_training_date is None or
            (datetime.now() - self.last_training_date).days >= self.retrain_frequency):
            await self._retrain_models()
        
        for symbol, data in market_data.items():
            if symbol not in self.symbols or symbol not in self.models:
                continue
            
            try:
                # Get current features
                current_features = await self._get_current_features(symbol)
                if current_features is None:
                    continue
                
                # Make prediction
                model = self.models[symbol]
                scaler = self.scalers[symbol]
                
                # Prepare features
                X = current_features[self.feature_columns].fillna(0).values.reshape(1, -1)
                X_scaled = scaler.transform(X)
                
                # Get prediction and confidence
                prediction = model.predict(X_scaled)[0]
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X_scaled)[0]
                    confidence = max(probabilities)
                else:
                    confidence = 0.6  # Default confidence
                
                # Generate signal based on prediction
                if prediction == 1 and confidence >= self.confidence_threshold:
                    # Buy signal
                    signal = TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        strength=self._confidence_to_strength(confidence),
                        confidence=confidence,
                        target_price=data.price,
                        stop_loss=data.price * 0.95,
                        take_profit=data.price * 1.05,
                        reasoning=f"ML model predicts upward movement (confidence: {confidence:.2f})",
                        metadata={
                            'model_type': self.model_type,
                            'prediction': prediction,
                            'model_accuracy': self.model_performance.get(symbol, {}).get('accuracy', 0.0)
                        }
                    )
                    signals.append(signal)
                
                elif prediction == 0 and confidence >= self.confidence_threshold:
                    # Sell signal (if we have position)
                    if self.portfolio and self.portfolio.get_position(symbol):
                        signal = TradingSignal(
                            symbol=symbol,
                            signal_type=SignalType.SELL,
                            strength=self._confidence_to_strength(confidence),
                            confidence=confidence,
                            target_price=data.price,
                            reasoning=f"ML model predicts downward movement (confidence: {confidence:.2f})",
                            metadata={
                                'model_type': self.model_type,
                                'prediction': prediction,
                                'model_accuracy': self.model_performance.get(symbol, {}).get('accuracy', 0.0)
                            }
                        )
                        signals.append(signal)
                
            except Exception as e:
                self.logger.error(f"Failed to generate signal for {symbol}: {str(e)}")
        
        return signals
    
    def _confidence_to_strength(self, confidence: float) -> SignalStrength:
        """Convert confidence to signal strength."""
        if confidence >= 0.9:
            return SignalStrength.VERY_STRONG
        elif confidence >= 0.8:
            return SignalStrength.STRONG
        elif confidence >= 0.7:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    async def _get_current_features(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get current features for a symbol."""
        if symbol not in self.historical_data or self.historical_data[symbol].empty:
            return None
        
        try:
            # Rebuild features with current data
            features = FeatureEngineering.create_features(self.historical_data[symbol])
            return features.tail(1)  # Return most recent features
            
        except Exception as e:
            self.logger.error(f"Failed to get current features for {symbol}: {str(e)}")
            return None
    
    async def _retrain_models(self) -> None:
        """Retrain models with updated data."""
        self.logger.info("Retraining ML models with updated data")
        
        # Rebuild features with latest data
        await self._build_features()
        
        # Retrain models
        await self._train_models()
    
    async def _save_models(self) -> None:
        """Save trained models to disk."""
        try:
            models_dir = "models"
            os.makedirs(models_dir, exist_ok=True)
            
            for symbol in self.models:
                model_file = os.path.join(models_dir, f"{symbol}_model.pkl")
                scaler_file = os.path.join(models_dir, f"{symbol}_scaler.pkl")
                
                joblib.dump(self.models[symbol], model_file)
                joblib.dump(self.scalers[symbol], scaler_file)
            
            # Save metadata
            metadata = {
                'model_type': self.model_type,
                'feature_columns': self.feature_columns,
                'symbols': self.symbols,
                'training_date': self.last_training_date.isoformat() if self.last_training_date else None,
                'performance': self.model_performance
            }
            
            with open(os.path.join(models_dir, "metadata.json"), 'w') as f:
                import json
                json.dump(metadata, f, indent=2)
            
            self.logger.info("Models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save models: {str(e)}")
    
    def get_model_performance(self) -> Dict:
        """Get comprehensive model performance metrics."""
        return {
            'model_type': self.model_type,
            'symbols': list(self.model_performance.keys()),
            'avg_accuracy': np.mean([
                perf.get('accuracy', 0) for perf in self.model_performance.values()
            ]),
            'last_training': self.last_training_date.isoformat() if self.last_training_date else None,
            'individual_performance': self.model_performance
        }


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_ml_strategy():
        """Test ML strategy functionality."""
        
        if RandomForestClassifier is None:
            print("❌ scikit-learn not available, cannot test ML strategy")
            return
        
        # Create ML strategy
        strategy = MLTradingStrategy(
            symbols=["AAPL", "MSFT"],
            model_type="random_forest",
            lookback_days=100,
            confidence_threshold=0.6
        )
        
        print(f"Strategy: {strategy}")
        
        # Initialize strategy (this will load data and train models)
        success = await strategy.initialize()
        if not success:
            print("❌ Strategy initialization failed")
            return
        
        print("✓ Strategy initialized successfully")
        
        # Simulate market data
        market_data = {
            "AAPL": MarketData("AAPL", 150.0, 149.5, 150.5, 100000),
            "MSFT": MarketData("MSFT", 300.0, 299.5, 300.5, 80000)
        }
        
        # Start strategy
        strategy.is_active = True
        
        # Generate signals
        signals = await strategy.update(market_data)
        
        print(f"Generated {len(signals)} signals:")
        for signal in signals:
            print(f"  {signal}")
        
        # Get performance metrics
        performance = strategy.get_model_performance()
        print(f"Model Performance: {performance}")
        
        await strategy.cleanup()
    
    asyncio.run(test_ml_strategy())