#!/usr/bin/env python3
"""
Quick test script to validate app initialization without running the full Streamlit UI.
"""

import asyncio
import sys
import os
import logging

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app import TradingApp
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

async def test_app_initialization():
    """Test basic app initialization."""
    print("🧪 Testing TradingApp initialization...")
    
    try:
        # Create app instance
        app = TradingApp()
        print("✅ TradingApp instance created")
        
        # Test configuration loading
        if app.config:
            print("✅ Configuration loaded successfully")
            print(f"   - Broker type: {app.config.get('broker', {}).get('type', 'unknown')}")
            print(f"   - Strategy count: {len(app.config.get('strategy', {}).get('active_strategies', []))}")
        else:
            print("❌ Configuration not loaded")
            return False
        
        # Test initialization
        print("🔄 Testing component initialization...")
        success = await app.initialize()
        
        if success:
            print("✅ App initialization successful!")
            
            # Test portfolio
            if app.portfolio:
                print(f"   - Portfolio equity: ${app.portfolio.calculate_total_equity():,.2f}")
                print(f"   - Portfolio cash: ${app.portfolio.current_cash:,.2f}")
            
            # Test broker
            if app.broker:
                print(f"   - Broker type: {type(app.broker).__name__}")
                print(f"   - Broker connected: {await app.broker.is_connected()}")
            
            # Test strategies
            print(f"   - Active strategies: {len(app.strategies)}")
            for strategy in app.strategies:
                print(f"     - {strategy.name} (confidence: {strategy.confidence_threshold})")
            
            return True
        else:
            print("❌ App initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during initialization: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_signal_generation():
    """Test signal generation."""
    print("\n🧪 Testing signal generation...")
    
    try:
        app = TradingApp()
        
        if await app.initialize():
            # Test signal generation with a small set of tickers
            test_tickers = ['AAPL', 'MSFT']
            print(f"🔄 Generating signals for: {test_tickers}")
            
            signals = await app.generate_signals(test_tickers)
            
            print(f"✅ Generated {len(signals)} signals")
            for signal in signals:
                print(f"   - {signal.ticker}: {signal.action} (confidence: {signal.confidence:.2%})")
            
            return True
        else:
            print("❌ Could not initialize app for signal testing")
            return False
            
    except Exception as e:
        print(f"❌ Error during signal generation test: {str(e)}")
        return False

async def main():
    """Run all tests."""
    print("🚀 Advanced Trading Framework - Initialization Test")
    print("=" * 60)
    
    # Test 1: Basic initialization
    init_success = await test_app_initialization()
    
    if init_success:
        # Test 2: Signal generation (only if init worked)
        signal_success = await test_signal_generation()
    else:
        signal_success = False
    
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    print(f"   Initialization: {'✅ PASS' if init_success else '❌ FAIL'}")
    print(f"   Signal Generation: {'✅ PASS' if signal_success else '❌ FAIL'}")
    
    if init_success and signal_success:
        print("\n🎉 All tests passed! App is ready to run.")
        print("💡 To start the full app, run: python run_app.py")
        return True
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)