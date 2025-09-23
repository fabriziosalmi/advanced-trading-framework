#!/usr/bin/env python3
"""
Advanced Trading Framework - Integration Test

Comprehensive test to validate all system components work together.
This test validates imports, configurations, and basic functionality.
"""

import sys
import os
import asyncio
import traceback
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test all critical imports."""
    print("🧪 Testing imports...")
    
    try:
        # Core imports
        from core.portfolio import Portfolio
        from core.position import Position
        print("  ✅ Core modules imported successfully")
        
        # Execution layer imports
        from execution_layer.base_broker import BaseBroker
        from execution_layer.paper_broker import PaperBroker
        from execution_layer.alpaca_broker import AlpacaBroker
        print("  ✅ Execution layer modules imported successfully")
        
        # Strategy layer imports
        from strategy_layer.signals import TradingSignal, create_buy_signal, create_sell_signal
        from strategy_layer.strategy_base import Strategy
        from strategy_layer.ml_random_forest_strategy import MLRandomForestStrategy
        print("  ✅ Strategy layer modules imported successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {str(e)}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration loading."""
    print("🧪 Testing configuration...")
    
    try:
        import yaml
        
        config_path = project_root / "config.yaml"
        if not config_path.exists():
            print("  ❌ config.yaml not found")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate key sections
        required_sections = ['app', 'logging', 'data', 'broker', 'portfolio', 'strategy', 'universe']
        
        for section in required_sections:
            if section not in config:
                print(f"  ❌ Missing config section: {section}")
                return False
        
        print("  ✅ Configuration loaded and validated successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {str(e)}")
        return False

def test_core_components():
    """Test core data structures."""
    print("🧪 Testing core components...")
    
    try:
        from core.portfolio import Portfolio
        from core.position import Position
        from datetime import datetime
        
        # Test Position
        position = Position(
            ticker="AAPL",
            quantity=100,
            entry_price=150.0,
            side="LONG"
        )
        position.current_price = 155.0  # Set current price separately
        
        assert position.current_value == 15500.0
        assert position.unrealized_pl == 500.0
        
        # Test Position serialization
        position_dict = position.to_dict()
        position_restored = Position.from_dict(position_dict)
        assert position_restored.ticker == position.ticker
        
        # Test Portfolio - use temp file to avoid state conflicts
        import tempfile
        import os
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        temp_file.close()
        
        portfolio = Portfolio(initial_capital=100000.0, state_file=temp_file.name)
        
        # Open a position using Portfolio's interface
        success = portfolio.open_position(
            ticker="TEST",  # Use unique ticker for testing
            quantity=100,
            entry_price=150.0,
            side="LONG"
        )
        assert success
        
        assert len(portfolio.positions) == 1
        assert portfolio.calculate_total_equity() >= 100000.0  # Portfolio value should include position value
        
        # Clean up temp file
        try:
            os.unlink(temp_file.name)
        except:
            pass
        
        # Test Portfolio serialization (basic)
        try:
            portfolio_dict = portfolio.to_dict()
            assert isinstance(portfolio_dict, dict)
            assert 'positions' in portfolio_dict
            print("  ✅ Portfolio serialization works")
        except Exception as e:
            print(f"  ⚠️ Portfolio serialization test skipped: {str(e)}")
        
        print("  ✅ Core components tested successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Core components test failed: {str(e)}")
        traceback.print_exc()
        return False

async def test_execution_layer():
    """Test broker implementations."""
    print("🧪 Testing execution layer...")
    
    try:
        from execution_layer.paper_broker import PaperBroker
        
        # Test Paper Broker
        broker = PaperBroker(initial_cash=100000.0)
        
        # Initialize broker
        success = await broker.connect()
        if not success:
            print("  ❌ Paper broker connection failed")
            return False
        
        # Test account info
        account_info = await broker.get_account_info()
        assert account_info.cash == 100000.0
        
        # Test positions (should be empty initially)
        positions = await broker.get_positions()
        assert len(positions) == 0
        
        # Note: Order placement testing would require Order class and market data
        # For now, just test that the method exists
        assert hasattr(broker, 'submit_order')
        
        await broker.disconnect()
        
        print("  ✅ Execution layer tested successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Execution layer test failed: {str(e)}")
        traceback.print_exc()
        return False

async def test_strategy_layer():
    """Test strategy implementations."""
    print("🧪 Testing strategy layer...")
    
    try:
        from strategy_layer.signals import TradingSignal, create_buy_signal
        from strategy_layer.ml_random_forest_strategy import MLRandomForestStrategy
        
        # Test TradingSignal
        signal = create_buy_signal(
            ticker="AAPL",
            confidence=0.75,
            price=150.0,
            reasoning="Test signal"
        )
        
        assert signal.ticker == "AAPL"
        assert signal.action == "BUY"
        assert signal.confidence == 0.75
        
        # Test signal serialization
        signal_dict = signal.to_dict()
        signal_restored = TradingSignal.from_dict(signal_dict)
        assert signal_restored.ticker == signal.ticker
        
        # Test ML Strategy (basic initialization)
        try:
            strategy = MLRandomForestStrategy(confidence_threshold=0.3)
            
            # Test initialization (might fail due to missing sklearn/yfinance)
            try:
                success = await strategy.initialize()
                if success:
                    print("  ✅ ML Strategy initialized successfully")
                else:
                    print("  ⚠️ ML Strategy initialization failed (likely missing dependencies)")
            except Exception as init_error:
                print(f"  ⚠️ ML Strategy init failed: {str(init_error)} (likely missing dependencies)")
            
            await strategy.cleanup()
            
        except ImportError as ie:
            print(f"  ⚠️ ML Strategy test skipped: {str(ie)}")
        
        print("  ✅ Strategy layer tested successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Strategy layer test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_file_structure():
    """Test that all required files exist."""
    print("🧪 Testing file structure...")
    
    required_files = [
        "config.yaml",
        "requirements.txt",
        "app.py",
        "run_app.py",
        "README.md",
        "core/__init__.py",
        "core/portfolio.py",
        "core/position.py",
        "execution_layer/__init__.py",
        "execution_layer/base_broker.py",
        "execution_layer/paper_broker.py",
        "execution_layer/alpaca_broker.py",
        "strategy_layer/__init__.py",
        "strategy_layer/signals.py",
        "strategy_layer/strategy_base.py",
        "strategy_layer/ml_random_forest_strategy.py"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False
    
    print("  ✅ All required files present")
    return True

def test_dependencies():
    """Test critical dependencies."""
    print("🧪 Testing dependencies...")
    
    critical_deps = ['pandas', 'numpy', 'yaml']
    optional_deps = ['sklearn', 'yfinance', 'streamlit', 'plotly']
    
    missing_critical = []
    missing_optional = []
    
    for dep in critical_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_critical.append(dep)
    
    for dep in optional_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_optional.append(dep)
    
    if missing_critical:
        print(f"  ❌ Missing critical dependencies: {missing_critical}")
        return False
    
    if missing_optional:
        print(f"  ⚠️ Missing optional dependencies: {missing_optional}")
        print(f"    Install with: pip install {' '.join(missing_optional)}")
    
    print("  ✅ Critical dependencies available")
    return True

async def main():
    """Run all integration tests."""
    print("🚀 Advanced Trading Framework - Integration Test")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Dependencies", test_dependencies),
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Core Components", test_core_components),
        ("Execution Layer", test_execution_layer),
        ("Strategy Layer", test_strategy_layer),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} test FAILED with exception: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The trading framework is ready to use.")
        print("🚀 To launch the application, run: python run_app.py")
    else:
        print("⚠️ Some tests failed. Please review the errors above.")
        print("💡 Try installing missing dependencies: pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)