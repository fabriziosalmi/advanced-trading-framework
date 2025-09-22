"""
Advanced Trading Framework - Example Strategies

Example trading strategies for backtesting and live trading.

Author: Strategy Specialist
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from core.backtesting import BacktestStrategy, Position
import logging
import os

logger = logging.getLogger(__name__)


class MovingAverageCrossoverStrategy(BacktestStrategy):
    """
    Simple moving average crossover strategy.

    Buys when fast MA crosses above slow MA, sells when it crosses below.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("MovingAverageCrossover", config)
        self.fast_period = config.get('fast_period', 20)
        self.slow_period = config.get('slow_period', 50)
        self.position_size = config.get('position_size', 0.1)

    def generate_signals(self, data: pd.DataFrame, portfolio: Dict[str, Any],
                        cash: float) -> List[Dict[str, Any]]:
        signals = []

        for symbol in data.keys():
            if symbol not in data or data[symbol].empty:
                continue

            df = data[symbol].copy()

            # Handle MultiIndex columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten MultiIndex columns for single symbol
                df.columns = df.columns.droplevel(1)

            # Calculate moving averages
            df['fast_ma'] = df['Close'].rolling(window=self.fast_period).mean()
            df['slow_ma'] = df['Close'].rolling(window=self.slow_period).mean()

            # Generate crossover signals
            df['signal'] = 0
            df.loc[df['fast_ma'] > df['slow_ma'], 'signal'] = 1  # Buy signal
            df.loc[df['fast_ma'] < df['slow_ma'], 'signal'] = -1  # Sell signal

            # Get latest signal
            if len(df) > 0:
                latest_signal = df['signal'].iloc[-1]
                current_price = df['Close'].iloc[-1]

                # Check current position
                current_position = portfolio.get(symbol, Position("", 0, 0)).quantity

                if latest_signal == 1 and current_position <= 0:
                    # Buy signal and no position
                    quantity = (cash * self.position_size) / current_price
                    if quantity > 0:
                        signals.append({
                            'symbol': symbol,
                            'side': 'BUY',
                            'quantity': quantity,
                            'type': 'MARKET'
                        })

                elif latest_signal == -1 and current_position > 0:
                    # Sell signal and have position
                    signals.append({
                        'symbol': symbol,
                        'side': 'SELL',
                        'quantity': current_position,
                        'type': 'MARKET'
                    })

        return signals


class RSIStrategy(BacktestStrategy):
    """
    RSI-based mean reversion strategy.

    Buys when RSI is oversold (< 30), sells when RSI is overbought (> 70).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("RSIStrategy", config)
        self.rsi_period = config.get('rsi_period', 14)
        self.oversold_level = config.get('oversold_level', 30)
        self.overbought_level = config.get('overbought_level', 70)
        self.position_size = config.get('position_size', 0.1)

    def generate_signals(self, data: pd.DataFrame, portfolio: Dict[str, Any],
                        cash: float) -> List[Dict[str, Any]]:
        signals = []

        for symbol in data.keys():
            if symbol not in data or data[symbol].empty:
                continue

            df = data[symbol].copy()

            # Handle MultiIndex columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten MultiIndex columns for single symbol
                df.columns = df.columns.droplevel(1)

            # Calculate RSI
            df['rsi'] = self._calculate_rsi(df['Close'], self.rsi_period)

            # Generate signals
            latest_rsi = df['rsi'].iloc[-1] if len(df) > 0 else 50
            current_price = df['Close'].iloc[-1] if len(df) > 0 else 0
            current_position = portfolio.get(symbol, Position("", 0, 0)).quantity

            if latest_rsi < self.oversold_level and current_position <= 0:
                # Oversold - Buy
                quantity = (cash * self.position_size) / current_price
                if quantity > 0:
                    signals.append({
                        'symbol': symbol,
                        'side': 'BUY',
                        'quantity': quantity,
                        'type': 'MARKET'
                    })

            elif latest_rsi > self.overbought_level and current_position > 0:
                # Overbought - Sell
                signals.append({
                    'symbol': symbol,
                    'side': 'SELL',
                    'quantity': current_position,
                    'type': 'MARKET'
                    })

        return signals

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class MLRandomForestBacktestStrategy(BacktestStrategy):
    """
    ML-based strategy using pre-trained Random Forest models.

    Uses the same ML models from the live strategy for backtesting.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("MLRandomForestBacktest", config)
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        self.position_size = config.get('position_size', 0.05)
        self.models = {}  # Cache for loaded models
        self.scalers = {}  # Cache for loaded scalers

    def generate_signals(self, data: pd.DataFrame, portfolio: Dict[str, Any],
                        cash: float) -> List[Dict[str, Any]]:
        signals = []

        for symbol in data.keys():
            if symbol not in data or data[symbol].empty:
                continue

            df = data[symbol].copy()

            # Handle MultiIndex columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten MultiIndex columns for single symbol
                df.columns = df.columns.droplevel(1)

            # Load model and scaler if not cached
            if symbol not in self.models:
                try:
                    from joblib import load
                    model_path = f"models/{symbol}_rf_model.pkl"
                    scaler_path = f"models/{symbol}_scaler.pkl"

                    if os.path.exists(model_path) and os.path.exists(scaler_path):
                        self.models[symbol] = load(model_path)
                        self.scalers[symbol] = load(scaler_path)
                    else:
                        continue  # No model available
                except Exception as e:
                    self.logger.warning(f"Failed to load model for {symbol}: {e}")
                    continue

            # Calculate technical indicators
            features = self._calculate_features(df)

            if features is None or len(features) == 0:
                continue

            # Get latest features
            latest_features = features.iloc[-1:].values

            # Scale features
            try:
                scaled_features = self.scalers[symbol].transform(latest_features)
            except:
                continue

            # Make prediction
            prediction = self.models[symbol].predict_proba(scaled_features)[0]

            # Get confidence scores
            confidence_buy = prediction[1]  # Probability of positive class
            confidence_sell = prediction[0]  # Probability of negative class

            current_position = portfolio.get(symbol, Position("", 0, 0)).quantity
            current_price = df['Close'].iloc[-1]

            # Generate signals based on confidence
            if confidence_buy > self.confidence_threshold and current_position <= 0:
                # Strong buy signal
                quantity = (cash * self.position_size) / current_price
                if quantity > 0:
                    signals.append({
                        'symbol': symbol,
                        'side': 'BUY',
                        'quantity': quantity,
                        'type': 'MARKET',
                        'confidence': confidence_buy
                    })

            elif confidence_sell > self.confidence_threshold and current_position > 0:
                # Strong sell signal
                signals.append({
                    'symbol': symbol,
                    'side': 'SELL',
                    'quantity': current_position,
                    'type': 'MARKET',
                    'confidence': confidence_sell
                })

        return signals

    def _calculate_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate technical features for ML model."""
        try:
            # Ensure we have enough data
            if len(df) < 50:
                return None

            features = pd.DataFrame(index=df.index)

            # Price-based features
            features['returns'] = df['Close'].pct_change()
            features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

            # Moving averages
            for period in [5, 10, 20, 50]:
                features[f'ma_{period}'] = df['Close'].rolling(period).mean()
                features[f'ma_{period}_slope'] = features[f'ma_{period}'].diff()

            # Volatility
            features['volatility_20'] = df['Close'].rolling(20).std()
            features['volatility_50'] = df['Close'].rolling(50).std()

            # RSI
            features['rsi_14'] = self._calculate_rsi(df['Close'], 14)

            # MACD
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            features['macd'] = exp1 - exp2
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_hist'] = features['macd'] - features['macd_signal']

            # Bollinger Bands
            sma_20 = df['Close'].rolling(20).mean()
            std_20 = df['Close'].rolling(20).std()
            features['bb_upper'] = sma_20 + (std_20 * 2)
            features['bb_lower'] = sma_20 - (std_20 * 2)
            features['bb_position'] = (df['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])

            # Volume features
            features['volume_ma_20'] = df['Volume'].rolling(20).mean()
            features['volume_ratio'] = df['Volume'] / features['volume_ma_20']

            # Momentum
            for period in [5, 10, 20]:
                features[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1

            # Drop NaN values
            features = features.dropna()

            return features

        except Exception as e:
            self.logger.error(f"Error calculating features: {e}")
            return None

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class MeanReversionStrategy(BacktestStrategy):
    """
    Mean reversion strategy using Bollinger Bands.

    Buys when price touches lower band, sells when price touches upper band.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("MeanReversion", config)
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2.0)
        self.position_size = config.get('position_size', 0.05)

    def generate_signals(self, data: pd.DataFrame, portfolio: Dict[str, Any],
                        cash: float) -> List[Dict[str, Any]]:
        signals = []

        for symbol in data.keys():
            if symbol not in data or data[symbol].empty:
                continue

            df = data[symbol].copy()

            # Handle MultiIndex columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten MultiIndex columns for single symbol
                df.columns = df.columns.droplevel(1)

            # Calculate Bollinger Bands
            sma = df['Close'].rolling(self.bb_period).mean()
            std = df['Close'].rolling(self.bb_period).std()
            upper_band = sma + (std * self.bb_std)
            lower_band = sma - (std * self.bb_std)

            # Get latest values
            current_price = df['Close'].iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]

            current_position = portfolio.get(symbol, Position("", 0, 0)).quantity

            # Generate signals
            if not (np.isnan(current_upper) or np.isnan(current_lower)):
                if current_price <= current_lower and current_position <= 0:
                    # Price at lower band - Buy
                    quantity = (cash * self.position_size) / current_price
                    if quantity > 0:
                        signals.append({
                            'symbol': symbol,
                            'side': 'BUY',
                            'quantity': quantity,
                            'type': 'MARKET'
                        })

                elif current_price >= current_upper and current_position > 0:
                    # Price at upper band - Sell
                    signals.append({
                        'symbol': symbol,
                        'side': 'SELL',
                        'quantity': current_position,
                        'type': 'MARKET'
                    })

        return signals


# Strategy factory function
def create_strategy(strategy_name: str, config: Dict[str, Any]) -> BacktestStrategy:
    """
    Factory function to create strategy instances.

    Args:
        strategy_name: Name of the strategy
        config: Strategy configuration

    Returns:
        Strategy instance
    """
    strategies = {
        'moving_average': MovingAverageCrossoverStrategy,
        'rsi': RSIStrategy,
        'ml_random_forest': MLRandomForestBacktestStrategy,
        'mean_reversion': MeanReversionStrategy,
    }

    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    return strategies[strategy_name](config)