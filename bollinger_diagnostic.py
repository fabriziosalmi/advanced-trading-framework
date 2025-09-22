"""
Diagnostic Script for Bollinger Bandwidth Lookahead Bias Analysis

This script creates a minimal example to verify that bollinger_bandwidth_20
calculations are strictly backward-looking and don't leak future data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_data():
    """Create a sample DataFrame with price data."""
    # Create 30 days of sample data
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    np.random.seed(42)  # For reproducible results

    # Generate realistic price data with trend and noise
    base_price = 100
    prices = []
    for i in range(30):
        # Add some trend and volatility
        trend = i * 0.5
        noise = np.random.normal(0, 2)
        price = base_price + trend + noise
        prices.append(max(price, 1))  # Ensure positive prices

    df = pd.DataFrame({
        'date': dates,
        'close': prices
    })
    df.set_index('date', inplace=True)

    return df

def calculate_bollinger_bandwidth_manual(df, period=20, std_dev=2.0):
    """
    Manual calculation of Bollinger Bandwidth to verify causality.

    For each day t, this should only use data from t-(period-1) to t.
    """
    bandwidths = []

    for i in range(len(df)):
        if i < period - 1:
            # Not enough data for calculation
            bandwidths.append(np.nan)
            continue

        # Use only historical data up to current day (inclusive)
        window_data = df['close'].iloc[max(0, i - period + 1):i + 1]

        # Calculate mean and std of the window
        ma = window_data.mean()
        std = window_data.std()

        # Calculate Bollinger Bands
        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)

        # Calculate bandwidth
        bandwidth = (upper_band - lower_band) / ma
        bandwidths.append(bandwidth)

    return pd.Series(bandwidths, index=df.index)

def calculate_bollinger_bandwidth_pandas(df, period=20, std_dev=2.0):
    """
    Pandas-based calculation (current implementation).
    """
    ma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()

    upper_band = ma + (std * std_dev)
    lower_band = ma - (std * std_dev)

    bandwidth = (upper_band - lower_band) / ma

    return bandwidth

def verify_causality(df, manual_bandwidth, pandas_bandwidth):
    """
    Verify that both calculations produce identical results.
    """
    # Compare the two methods
    comparison = pd.DataFrame({
        'manual': manual_bandwidth,
        'pandas': pandas_bandwidth
    })

    # Check if they're equal (allowing for small floating point differences)
    are_equal = np.allclose(manual_bandwidth.dropna(), pandas_bandwidth.dropna(), rtol=1e-10)

    print("=== Causality Verification ===")
    print(f"Calculations match: {are_equal}")

    if not are_equal:
        print("WARNING: Calculations differ!")
        print(comparison.dropna().head())
    else:
        print("✅ Both methods produce identical results")

    return are_equal

def demonstrate_backward_looking(df, day_index=25):
    """
    Demonstrate that the calculation for a specific day only uses past data.
    """
    print(f"\n=== Backward-Looking Demonstration for Day {day_index} ===")

    target_date = df.index[day_index]
    print(f"Target date: {target_date}")

    # Show the data used for calculation (should be up to and including target_date)
    window_size = 20
    start_idx = max(0, day_index - window_size + 1)
    end_idx = day_index + 1  # Include current day

    window_data = df.iloc[start_idx:end_idx]
    print(f"Data window used: {len(window_data)} days")
    print(f"From: {window_data.index[0]} To: {window_data.index[-1]}")
    print(f"Prices: {window_data['close'].values}")

    # Manual calculation
    ma = window_data['close'].mean()
    std = window_data['close'].std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    bandwidth = (upper - lower) / ma

    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")
    print(".6f")

    # Verify no future data is used
    future_data = df.iloc[day_index + 1:]
    if len(future_data) > 0:
        print(f"Future data (NOT used): {len(future_data)} days from {future_data.index[0]}")
    else:
        print("No future data available (end of dataset)")

if __name__ == "__main__":
    print("🔍 Bollinger Bandwidth Causality Diagnostic")
    print("=" * 50)

    # Create sample data
    df = create_sample_data()
    print(f"Created sample dataset: {len(df)} days")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    # Calculate bandwidth using both methods
    manual_bw = calculate_bollinger_bandwidth_manual(df)
    pandas_bw = calculate_bollinger_bandwidth_pandas(df)

    # Verify they match
    verify_causality(df, manual_bw, pandas_bw)

    # Demonstrate backward-looking nature
    demonstrate_backward_looking(df, day_index=25)

    print("\n" + "=" * 50)
    print("🎯 CONCLUSION:")
    print("If calculations match and only use historical data, then")
    print("bollinger_bandwidth_20 is NOT the source of lookahead bias.")
    print("The issue must be elsewhere in the feature engineering pipeline.")