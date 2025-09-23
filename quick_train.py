#!/usr/bin/env python3
"""
Quick Training Script - Train ML models easily

Usage:
    python quick_train.py
"""

import os
import sys
import asyncio
import logging
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy_layer.ml_random_forest_strategy import MLRandomForestStrategy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def quick_train():
    """Quick training function for common tickers."""
    print("🚀 Quick ML Model Training")
    print("=" * 40)
    
    # Common tickers to train
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    
    try:
        # Initialize strategy
        print("📊 Initializing ML Strategy...")
        strategy = MLRandomForestStrategy(confidence_threshold=0.3)
        await strategy.initialize()
        print("✅ Strategy initialized")
        
        # Train models for each ticker
        success_count = 0
        for i, ticker in enumerate(tickers, 1):
            print(f"\n[{i}/{len(tickers)}] Training {ticker}...")
            
            try:
                # Generate a signal (this will train the model if needed)
                signals = await strategy.generate_signals([ticker])
                print(f"✅ {ticker} model training completed")
                success_count += 1
                
            except Exception as e:
                print(f"❌ Failed to train {ticker}: {e}")
                continue
        
        print(f"\n🎯 Training Summary:")
        print(f"   Successful: {success_count}/{len(tickers)} models")
        
        if success_count > 0:
            print(f"🎉 Training completed! Models saved in 'models/' directory")
        else:
            print(f"💥 No models were trained successfully")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False
    
    return success_count > 0

if __name__ == "__main__":
    # Ensure required directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Run training
    asyncio.run(quick_train())