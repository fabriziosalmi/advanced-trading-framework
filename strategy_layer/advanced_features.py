"""
Advanced Trading Framework - Advanced Features Module

This module provides enhanced technical indicators and features for improved
trading signal generation, focusing on volatility and momentum analysis.

Author: Quantitative Analyst
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR) - a volatility indicator.

    ATR measures the average range between high and low prices over a period,
    providing insight into price volatility.

    Args:
        data: DataFrame with 'high', 'low', 'close' columns
        period: Period for ATR calculation (default: 14)

    Returns:
        Series containing ATR values
    """
    if not all(col in data.columns for col in ['high', 'low', 'close']):
        raise ValueError("Data must contain 'high', 'low', 'close' columns")

    # Calculate True Range
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift(1))
    low_close = np.abs(data['low'] - data['close'].shift(1))

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Calculate ATR using exponential moving average
    atr = true_range.ewm(span=period, adjust=False).mean()

    return atr


def calculate_bollinger_bandwidth(data: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """
    Calculate Bollinger Band Width - measures volatility of price movements.

    The width between upper and lower Bollinger Bands indicates volatility:
    wider bands = higher volatility, narrower bands = lower volatility.

    Args:
        data: DataFrame with 'close' column
        period: Moving average period (default: 20)
        std_dev: Standard deviation multiplier (default: 2.0)

    Returns:
        Series containing Bollinger Band Width values
    """
    if 'close' not in data.columns:
        raise ValueError("Data must contain 'close' column")

    # Calculate moving average and standard deviation
    ma = data['close'].rolling(window=period).mean()
    std = data['close'].rolling(window=period).std()

    # Calculate Bollinger Bands
    upper_band = ma + (std * std_dev)
    lower_band = ma - (std * std_dev)

    # Calculate bandwidth
    bandwidth = (upper_band - lower_band) / ma

    return bandwidth


def calculate_momentum_decay(data: pd.DataFrame, period: int = 10, decay_factor: float = 0.9) -> pd.Series:
    """
    Calculate Momentum with Exponential Decay.

    This provides a momentum indicator that gives more weight to recent price
    movements while still considering longer-term trends.

    Args:
        data: DataFrame with 'close' column
        period: Lookback period (default: 10)
        decay_factor: Exponential decay factor (default: 0.9)

    Returns:
        Series containing momentum with decay values
    """
    if 'close' not in data.columns:
        raise ValueError("Data must contain 'close' column")

    # Calculate simple momentum (current - past)
    momentum = data['close'] - data['close'].shift(period)

    # Apply exponential decay to give more weight to recent values
    weights = np.array([decay_factor ** i for i in range(period)])
    weights = weights / weights.sum()  # Normalize

    # Calculate weighted momentum
    momentum_decay = momentum.rolling(window=period).apply(
        lambda x: np.dot(x.values, weights), raw=False
    )

    return momentum_decay


def calculate_roc_acceleration(data: pd.DataFrame, short_period: int = 5, long_period: int = 20) -> pd.Series:
    """
    Calculate Rate of Change (ROC) Acceleration.

    This measures the acceleration/deceleration of price momentum by comparing
    short-term and long-term ROC values.

    Args:
        data: DataFrame with 'close' column
        short_period: Short-term ROC period (default: 5)
        long_period: Long-term ROC period (default: 20)

    Returns:
        Series containing ROC acceleration values
    """
    if 'close' not in data.columns:
        raise ValueError("Data must contain 'close' column")

    # Calculate ROC for different periods
    roc_short = (data['close'] - data['close'].shift(short_period)) / data['close'].shift(short_period)
    roc_long = (data['close'] - data['close'].shift(long_period)) / data['close'].shift(long_period)

    # Calculate acceleration (rate of change of ROC)
    roc_acceleration = roc_short - roc_long

    return roc_acceleration


def calculate_volatility_ratio(data: pd.DataFrame, short_period: int = 5, long_period: int = 20) -> pd.Series:
    """
    Calculate Volatility Ratio.

    Compares short-term volatility to long-term volatility to identify
    periods of increasing or decreasing price variability.

    Args:
        data: DataFrame with 'close' column
        short_period: Short-term volatility period (default: 5)
        long_period: Long-term volatility period (default: 20)

    Returns:
        Series containing volatility ratio values
    """
    if 'close' not in data.columns:
        raise ValueError("Data must contain 'close' column")

    # Calculate short-term volatility (standard deviation of returns)
    returns = data['close'].pct_change()
    vol_short = returns.rolling(window=short_period).std()

    # Calculate long-term volatility
    vol_long = returns.rolling(window=long_period).std()

    # Calculate ratio
    volatility_ratio = vol_short / vol_long

    return volatility_ratio


def get_advanced_features(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate all advanced features for the given data.

    Args:
        data: DataFrame with OHLCV data

    Returns:
        Dictionary containing all advanced feature series
    """
    features = {}

    try:
        # Volatility features
        features['atr_14'] = calculate_atr(data, period=14).fillna(0)
        features['bollinger_bandwidth_20'] = calculate_bollinger_bandwidth(data, period=20).fillna(0)
        features['volatility_ratio_5_20'] = calculate_volatility_ratio(data, short_period=5, long_period=20).fillna(1)

        # Momentum features
        features['momentum_decay_10'] = calculate_momentum_decay(data, period=10).fillna(0)
        features['momentum_decay_20'] = calculate_momentum_decay(data, period=20).fillna(0)
        features['roc_acceleration_5_20'] = calculate_roc_acceleration(data, short_period=5, long_period=20).fillna(0)
        features['roc_acceleration_10_20'] = calculate_roc_acceleration(data, short_period=10, long_period=20).fillna(0)

    except Exception as e:
        # Log error but don't crash - return empty features
        print(f"Warning: Error calculating advanced features: {e}")
        # Return empty series with same index, filled with 0
        empty_series = pd.Series(index=data.index, dtype=float).fillna(0)
        features = {name: empty_series.copy() for name in [
            'atr_14', 'bollinger_bandwidth_20', 'volatility_ratio_5_20',
            'momentum_decay_10', 'momentum_decay_20', 'roc_acceleration_5_20', 'roc_acceleration_10_20'
        ]}

    return features