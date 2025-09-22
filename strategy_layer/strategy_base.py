"""
Advanced Trading Framework - Strategy Base Class

This module defines the abstract base class for all trading strategies,
following the specification for simple signal generation with ticker lists.

Author: Senior Python Software Architect
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import logging
import json
import os
from datetime import datetime

from .signals import TradingSignal


class Strategy(ABC):
    """
    Abstract Base Class for all trading strategies.
    
    This class defines the standard interface that all strategy implementations
    must follow, ensuring consistency and interchangeability between different
    trading approaches.
    
    Key Features:
    - Signal generation for specific tickers
    - Configuration management
    - Performance tracking
    - State persistence
    """
    
    def __init__(
        self,
        name: str,
        confidence_threshold: float = 0.6,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the base strategy.
        
        Args:
            name: Strategy name for identification
            confidence_threshold: Minimum confidence for actionable signals
            config: Strategy-specific configuration
        """
        self.name = name
        self.confidence_threshold = confidence_threshold
        self.config = config or {}
        
        # Strategy state
        self.is_active = False
        self.last_update = None
        self.signal_history: List[TradingSignal] = []
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        self.logger.info(f"Strategy '{name}' initialized with confidence threshold: {confidence_threshold}")
    
    @abstractmethod
    async def generate_signals(self, tickers: List[str]) -> List[TradingSignal]:
        """
        Generate trading signals for given tickers.
        
        This is the core method that each strategy must implement
        to provide trading decisions for specific tickers.
        
        Args:
            tickers: List of ticker symbols to analyze
            
        Returns:
            List of TradingSignal objects
        """
        pass
    
    async def initialize(self) -> bool:
        """
        Initialize strategy-specific components.
        
        Called before strategy starts trading. Override this to:
        - Load models or indicators
        - Validate configuration
        - Setup data connections
        - Initialize internal state
        
        Returns:
            True if initialization successful, False otherwise
        """
        self.logger.info(f"Initializing strategy '{self.name}'")
        return True
    
    async def cleanup(self) -> None:
        """
        Cleanup strategy resources.
        
        Called when strategy is stopped. Override this to:
        - Save state or models
        - Close data connections
        - Cleanup temporary files
        """
        self.logger.info(f"Cleaning up strategy '{self.name}'")
    
    def filter_signals_by_confidence(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """
        Filter signals by confidence threshold.
        
        Args:
            signals: List of trading signals
            
        Returns:
            List of signals meeting confidence threshold
        """
        return [
            signal for signal in signals
            if signal.confidence >= self.confidence_threshold
        ]
    
    def add_signal_to_history(self, signal: TradingSignal) -> None:
        """
        Add signal to history for tracking.
        
        Args:
            signal: Signal to add to history
        """
        self.signal_history.append(signal)
        
        # Keep only recent signals (last 1000)
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate strategy performance metrics.
        
        Returns:
            Dictionary with performance statistics
        """
        if not self.signal_history:
            return {
                'total_signals': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'avg_confidence': 0.0
            }
        
        # Signal statistics
        total_signals = len(self.signal_history)
        buy_signals = sum(1 for s in self.signal_history if s.action == 'BUY')
        sell_signals = sum(1 for s in self.signal_history if s.action == 'SELL')
        avg_confidence = sum(s.confidence for s in self.signal_history) / total_signals
        
        # Recent activity (last 24 hours)
        from datetime import timedelta
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_signals = [
            s for s in self.signal_history 
            if s.timestamp and datetime.fromtimestamp(s.timestamp) > recent_cutoff
        ]
        
        return {
            'total_signals': total_signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'avg_confidence': avg_confidence,
            'recent_signals_24h': len(recent_signals),
            'last_signal_time': self.signal_history[-1].timestamp if self.signal_history else None,
            'confidence_threshold': self.confidence_threshold
        }
    
    def save_state(self, file_path: str) -> None:
        """
        Save strategy state to file.
        
        Args:
            file_path: Path to save state file
        """
        try:
            state_data = {
                'name': self.name,
                'confidence_threshold': self.confidence_threshold,
                'is_active': self.is_active,
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'config': self.config,
                'performance_metrics': self.get_performance_metrics(),
                'recent_signals': [
                    signal.to_dict() for signal in self.signal_history[-10:]
                ],  # Save last 10 signals
                'saved_at': datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            self.logger.info(f"Strategy state saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save strategy state: {str(e)}")
    
    def load_state(self, file_path: str) -> bool:
        """
        Load strategy state from file.
        
        Args:
            file_path: Path to load state file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                self.logger.info("No state file found")
                return False
            
            with open(file_path, 'r') as f:
                state_data = json.load(f)
            
            # Restore basic attributes
            self.is_active = state_data.get('is_active', False)
            if state_data.get('last_update'):
                self.last_update = datetime.fromisoformat(state_data['last_update'])
            
            # Restore configuration
            if 'config' in state_data:
                self.config.update(state_data['config'])
            
            self.logger.info(f"Strategy state loaded from {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load strategy state: {str(e)}")
            return False
    
    def __str__(self) -> str:
        """String representation of strategy."""
        status = "Active" if self.is_active else "Inactive"
        return f"{self.name}({status}, confidence_threshold={self.confidence_threshold})"
    
    def __repr__(self) -> str:
        """Detailed representation of strategy."""
        return (
            f"Strategy(name='{self.name}', "
            f"confidence_threshold={self.confidence_threshold}, "
            f"active={self.is_active})"
        )


# Example concrete strategy implementation for testing
class SimpleMovingAverageStrategy(Strategy):
    """
    Example strategy implementation using simple moving averages.
    
    This is a basic example strategy that demonstrates the Strategy
    interface. It generates buy signals when price is above the moving
    average and sell signals when below.
    """
    
    def __init__(self, confidence_threshold: float = 0.6, ma_period: int = 20):
        """
        Initialize Simple Moving Average strategy.
        
        Args:
            confidence_threshold: Minimum confidence for signals
            ma_period: Moving average period
        """
        super().__init__("SimpleMA", confidence_threshold)
        self.ma_period = ma_period
        self.price_history: Dict[str, List[float]] = {}
    
    async def initialize(self) -> bool:
        """Initialize strategy components."""
        self.logger.info(f"Initializing SimpleMA strategy with {self.ma_period} period")
        return True
    
    async def cleanup(self) -> None:
        """Cleanup strategy resources."""
        self.logger.info("Cleaning up SimpleMA strategy")
    
    async def generate_signals(self, tickers: List[str]) -> List[TradingSignal]:
        """
        Generate signals based on moving average crossover.
        
        Args:
            tickers: List of ticker symbols to analyze
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # For this example, we'll simulate price data
        # In real implementation, you'd fetch actual market data
        import random
        
        for ticker in tickers:
            # Simulate getting current price
            current_price = 100.0 + random.uniform(-10, 10)
            
            # Initialize price history if not exists
            if ticker not in self.price_history:
                self.price_history[ticker] = []
            
            # Update price history
            self.price_history[ticker].append(current_price)
            
            # Keep only required history
            if len(self.price_history[ticker]) > self.ma_period:
                self.price_history[ticker] = self.price_history[ticker][-self.ma_period:]
            
            # Calculate moving average
            if len(self.price_history[ticker]) >= self.ma_period:
                ma = sum(self.price_history[ticker]) / len(self.price_history[ticker])
                
                # Generate signals
                if current_price > ma * 1.02:  # 2% above MA
                    signal = TradingSignal(
                        ticker=ticker,
                        action="BUY",
                        confidence=0.7,
                        price=current_price,
                        stop_loss=ma * 0.95,
                        take_profit=current_price * 1.05,
                        reasoning=f"Price ${current_price:.2f} above MA ${ma:.2f}"
                    )
                    signals.append(signal)
                    self.add_signal_to_history(signal)
                
                elif current_price < ma * 0.98:  # 2% below MA
                    signal = TradingSignal(
                        ticker=ticker,
                        action="SELL",
                        confidence=0.7,
                        price=current_price,
                        reasoning=f"Price ${current_price:.2f} below MA ${ma:.2f}"
                    )
                    signals.append(signal)
                    self.add_signal_to_history(signal)
        
        # Filter by confidence threshold
        return self.filter_signals_by_confidence(signals)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_simple_strategy():
        """Test the SimpleMA strategy."""
        
        # Create strategy
        strategy = SimpleMovingAverageStrategy(confidence_threshold=0.6, ma_period=5)
        
        print(f"Strategy: {strategy}")
        
        # Initialize strategy
        await strategy.initialize()
        strategy.is_active = True
        
        # Test tickers
        tickers = ["AAPL", "MSFT", "GOOGL"]
        
        # Generate signals multiple times to build history
        for i in range(10):
            signals = await strategy.generate_signals(tickers)
            
            if signals:
                print(f"Iteration {i+1}: Generated {len(signals)} signals")
                for signal in signals:
                    print(f"  {signal}")
        
        # Get performance metrics
        metrics = strategy.get_performance_metrics()
        print(f"Performance Metrics: {metrics}")
        
        await strategy.cleanup()
    
    asyncio.run(test_simple_strategy())