#!/usr/bin/env python3
"""
Quick test to verify signal generation workflow from UI perspective.
"""
import asyncio
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy_layer.ml_random_forest_strategy import MLRandomForestStrategy
from strategy_layer.signals import TradingSignal

async def test_signal_generation():
    """Test the signal generation workflow."""
    print("🧪 Testing signal generation workflow...")

    # Initialize strategy with 50% confidence threshold
    strategy = MLRandomForestStrategy(confidence_threshold=0.3)
    await strategy.initialize()

    # Test tickers
    test_tickers = ['MSFT']  # Just test one ticker to avoid timeout

    print(f"📊 Generating signals for: {test_tickers}")
    print(f"🎯 Confidence threshold: {strategy.confidence_threshold}")

    try:
        # Generate signals
        print("🔍 Calling generate_signals...")
        signals = await strategy.generate_signals(test_tickers)
        print(f"✅ Generated {len(signals)} signals")

        # Check signal structure
        if signals:
            signal = signals[0]
            print(f"📋 Sample signal: {signal.ticker} {signal.action} @ ${signal.price:.2f} (confidence: {signal.confidence:.2%})")

            # Check if signal has required attributes
            required_attrs = ['ticker', 'action', 'price', 'confidence', 'timestamp', 'reasoning']
            for attr in required_attrs:
                if hasattr(signal, attr):
                    print(f"✅ Signal has {attr}: {getattr(signal, attr)}")
                else:
                    print(f"❌ Signal missing {attr}")

            # Test to_dict method
            signal_dict = signal.to_dict()
            print(f"✅ Signal to_dict() works: {len(signal_dict)} fields")

        else:
            print("⚠️ No signals generated - this might be expected if models aren't trained")

    except Exception as e:
        print(f"❌ Error generating signals: {str(e)}")
        import traceback
        traceback.print_exc()

    print("🏁 Signal generation test completed")

if __name__ == "__main__":
    asyncio.run(test_signal_generation())