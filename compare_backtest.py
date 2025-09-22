"""
Comparative Backtest Script - Advanced Features Impact Analysis

This script compares the performance of ML strategies with and without
advanced volatility and momentum features to quantify their impact.

Author: Quantitative Analyst
Version: 1.0.0
"""

import sys
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import asyncio
import yaml

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from core.backtesting import compare_strategies, BacktestEngine
from strategy_layer.ml_random_forest_strategy import MLRandomForestStrategy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/comparative_backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_baseline_strategy(config: Dict[str, Any]) -> MLRandomForestStrategy:
    """
    Create baseline ML strategy without advanced features.

    Args:
        config: Strategy configuration

    Returns:
        MLRandomForestStrategy with advanced features disabled
    """
    # Create config with advanced features disabled and lower confidence threshold for testing
    baseline_config = config.copy()
    baseline_config['strategy']['ml_random_forest']['use_advanced_features'] = False

    return MLRandomForestStrategy(confidence_threshold=0.3)


def create_enhanced_strategy(config: Dict[str, Any]) -> MLRandomForestStrategy:
    """
    Create enhanced ML strategy with advanced features.

    Args:
        config: Strategy configuration

    Returns:
        MLRandomForestStrategy with advanced features enabled
    """
    # Create config with advanced features enabled and lower confidence threshold for testing
    enhanced_config = config.copy()
    enhanced_config['strategy']['ml_random_forest']['use_advanced_features'] = True

    return MLRandomForestStrategy(confidence_threshold=0.3)


async def run_comparative_backtest(symbols: List[str], start_date: str, end_date: str,
                           config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run comparative backtest between baseline and enhanced strategies.

    Args:
        symbols: List of symbols to backtest
        start_date: Start date for backtest
        end_date: End date for backtest
        config: Backtest configuration

    Returns:
        Comparison results
    """
    logger.info("🚀 Starting comparative backtest analysis")
    logger.info(f"📊 Symbols: {symbols}")
    logger.info(f"📅 Period: {start_date} to {end_date}")

    try:
        # Create strategies
        baseline_strategy = create_baseline_strategy(config)
        enhanced_strategy = create_enhanced_strategy(config)

        strategies = [
            ("Baseline ML (No Advanced Features)", baseline_strategy),
            ("Enhanced ML (With Advanced Features)", enhanced_strategy)
        ]

        logger.info("🎯 Running comparative backtest...")

        # Run comparison
        results = await compare_strategies(strategies, symbols, start_date, end_date, config)

        if results['status'] == 'success':
            logger.info("✅ Comparative backtest completed successfully")
            return results
        else:
            logger.error(f"❌ Comparative backtest failed: {results.get('error', 'Unknown error')}")
            return results

    except Exception as e:
        error_msg = f"Critical error during comparative backtest: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return {'status': 'failed', 'error': error_msg}


def analyze_results(results: Dict[str, Any]) -> None:
    """
    Analyze and display comparative backtest results.

    Args:
        results: Backtest comparison results
    """
    if results['status'] != 'success':
        logger.error("Cannot analyze results - backtest failed")
        return

    print("\n" + "="*80)
    print("📊 COMPARATIVE BACKTEST RESULTS - ADVANCED FEATURES IMPACT")
    print("="*80)

    # Overall winner
    best_strategy = results['best_strategy']
    print(f"🏆 Best Performing Strategy: {best_strategy}")

    # Detailed results
    print("\n📈 Strategy Performance Comparison:")
    print("-" * 60)

    for i, result in enumerate(results['results'], 1):
        strategy_name = result['strategy_name']
        metrics = result['metrics']
        total_return = result['total_return']

        print(f"\n{i}. {strategy_name}")
        print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
        print(f"   Max Drawdown: {metrics.max_drawdown:.2f}")
        win_rate = result.get('win_rate', 'N/A')
        if isinstance(win_rate, (int, float)):
            print(f"   Win Rate: {win_rate:.1%}")
        else:
            print(f"   Win Rate: {win_rate}")
        print(f"   Total Trades: {result.get('total_trades', 'N/A')}")
        
        win_rate = result.get('win_rate', 'N/A')
        if isinstance(win_rate, (int, float)):
            print(f"   Win Rate: {win_rate:.1%}")
        else:
            print(f"   Win Rate: {win_rate}")

    # Comparison metrics
    comparison = results['comparison']
    print("\n🔍 Key Metrics Comparison:")
    print("-" * 40)

    baseline_sharpe = comparison['sharpe_ratios'][0]
    enhanced_sharpe = comparison['sharpe_ratios'][1]
    if abs(baseline_sharpe) > 1e-6:
        sharpe_improvement = ((enhanced_sharpe - baseline_sharpe) / abs(baseline_sharpe)) * 100
    else:
        sharpe_improvement = enhanced_sharpe * 100  # If baseline is 0, just show the enhanced value as percentage

    baseline_return = comparison['total_returns'][0]
    enhanced_return = comparison['total_returns'][1]
    if abs(baseline_return) > 1e-6:
        return_improvement = ((enhanced_return - baseline_return) / abs(baseline_return)) * 100
    else:
        return_improvement = enhanced_return * 100

    baseline_drawdown = comparison['max_drawdowns'][0]
    enhanced_drawdown = comparison['max_drawdowns'][1]
    if abs(baseline_drawdown) > 1e-6:
        drawdown_improvement = ((baseline_drawdown - enhanced_drawdown) / abs(baseline_drawdown)) * 100
    else:
        drawdown_improvement = (baseline_drawdown - enhanced_drawdown) * 100

    print(f"Sharpe Ratio Improvement: {sharpe_improvement:+.1f}%")
    print(f"Total Return Improvement: {return_improvement:+.1f}%")
    print(f"Max Drawdown Improvement: {drawdown_improvement:+.1f}%")

    # Analysis
    print("\n💡 Analysis:")
    if sharpe_improvement > 5:
        print("✅ Advanced features significantly improve risk-adjusted returns!")
    elif sharpe_improvement > 0:
        print("👍 Advanced features provide modest improvement in risk-adjusted returns.")
    else:
        print("⚠️ Advanced features did not improve performance in this backtest.")

    print("\n📝 Recommendations:")
    if best_strategy == "Enhanced ML (With Advanced Features)":
        print("- Consider using advanced features in production")
        print("- Monitor feature importance to understand which features contribute most")
    else:
        print("- Advanced features may not be beneficial for current market conditions")
        print("- Consider alternative feature engineering approaches")


async def main():
    """Main execution function."""
    try:
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file {config_path} not found")
            sys.exit(1)

        # Set backtest parameters
        symbols = ['AAPL', 'MSFT', 'GOOGL']  # Test on multiple symbols
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # 1 year backtest

        # Run comparative backtest
        results = await run_comparative_backtest(symbols, start_date, end_date, config)

        # Analyze and display results
        analyze_results(results)

        logger.info("🎉 Comparative backtest analysis completed")

    except Exception as e:
        error_msg = f"Critical error in main execution: {str(e)}"
        logger.error(f"❌ {error_msg}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())