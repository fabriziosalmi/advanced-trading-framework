"""
Advanced Trading Framework - Base Strategy Interface

This module defines the abstract base class for all trading strategies,
providing a standardized interface for signal generation, risk management,
position sizing, and strategy lifecycle management.

Author: Senior Python Software Architect
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
import os

from .signals import TradingSignal
from ..core.position import Position
from ..core.portfolio import Portfolio
from ..execution_layer.base_broker import BaseBroker, MarketData


class SignalType(Enum):
    """Enumeration for signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalStrength(Enum):
    """Enumeration for signal strength levels."""
    VERY_WEAK = "VERY_WEAK"
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"
    VERY_STRONG = "VERY_STRONG"


class RiskParameters:
    """
    Risk management parameters for strategies.
    
    Defines risk limits and position sizing rules
    to ensure safe trading operations.
    """
    
    def __init__(
        self,
        max_position_size: float = 0.1,  # 10% of portfolio
        max_total_exposure: float = 0.8,  # 80% of portfolio
        max_daily_loss: float = 0.02,    # 2% daily loss limit
        max_drawdown: float = 0.10,      # 10% max drawdown
        stop_loss_pct: float = 0.05,     # 5% stop loss
        take_profit_pct: float = 0.10,   # 10% take profit
        risk_per_trade: float = 0.01,    # 1% risk per trade
        max_open_positions: int = 10,    # Max concurrent positions
        min_confidence: float = 0.6      # Min signal confidence
    ):
        """
        Initialize risk parameters.
        
        Args:
            max_position_size: Maximum position size as % of portfolio
            max_total_exposure: Maximum total exposure as % of portfolio
            max_daily_loss: Maximum daily loss as % of portfolio
            max_drawdown: Maximum drawdown tolerance
            stop_loss_pct: Default stop loss percentage
            take_profit_pct: Default take profit percentage
            risk_per_trade: Risk per trade as % of portfolio
            max_open_positions: Maximum number of open positions
            min_confidence: Minimum signal confidence to act
        """
        self.max_position_size = max_position_size
        self.max_total_exposure = max_total_exposure
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.risk_per_trade = risk_per_trade
        self.max_open_positions = max_open_positions
        self.min_confidence = min_confidence
    
    def to_dict(self) -> Dict:
        """Convert risk parameters to dictionary."""
        return {
            'max_position_size': self.max_position_size,
            'max_total_exposure': self.max_total_exposure,
            'max_daily_loss': self.max_daily_loss,
            'max_drawdown': self.max_drawdown,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'risk_per_trade': self.risk_per_trade,
            'max_open_positions': self.max_open_positions,
            'min_confidence': self.min_confidence
        }


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    Defines the standard interface that all strategy implementations
    must follow, ensuring consistency and interchangeability between
    different trading approaches.
    
    Key Features:
    - Signal generation with confidence levels
    - Risk management integration
    - Position sizing calculations
    - Performance tracking
    - State persistence
    - Modular design for easy testing
    """
    
    def __init__(
        self,
        name: str,
        symbols: List[str],
        risk_params: Optional[RiskParameters] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize the base strategy.
        
        Args:
            name: Strategy name for identification
            symbols: List of symbols to trade
            risk_params: Risk management parameters
            config: Strategy-specific configuration
        """
        self.name = name
        self.symbols = symbols
        self.risk_params = risk_params or RiskParameters()
        self.config = config or {}
        
        # Strategy state
        self.is_active = False
        self.last_update = None
        self.signal_history: List[TradingSignal] = []
        self.performance_metrics: Dict = {}
        
        # Dependencies (set by framework)
        self.portfolio: Optional[Portfolio] = None
        self.broker: Optional[BaseBroker] = None
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        self.logger.info(f"Strategy '{name}' initialized for symbols: {symbols}")
    
    def set_dependencies(self, portfolio: Portfolio, broker: BaseBroker) -> None:
        """
        Set strategy dependencies.
        
        Args:
            portfolio: Portfolio manager instance
            broker: Broker interface instance
        """
        self.portfolio = portfolio
        self.broker = broker
        self.logger.info("Strategy dependencies set")
    
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
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize strategy-specific components.
        
        Called before strategy starts trading. Use this to:
        - Load models or indicators
        - Validate configuration
        - Setup data connections
        - Initialize internal state
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """
        Cleanup strategy resources.
        
        Called when strategy is stopped. Use this to:
        - Save state or models
        - Close data connections
        - Cleanup temporary files
        """
        pass
    
    async def start(self) -> bool:
        """
        Start the strategy.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.is_active:
            self.logger.warning("Strategy is already active")
            return True
        
        if not self.portfolio or not self.broker:
            self.logger.error("Strategy dependencies not set")
            return False
        
        # Initialize strategy
        if not await self.initialize():
            self.logger.error("Strategy initialization failed")
            return False
        
        self.is_active = True
        self.logger.info(f"Strategy '{self.name}' started")
        return True
    
    async def stop(self) -> None:
        """Stop the strategy."""
        if not self.is_active:
            return
        
        self.is_active = False
        await self.cleanup()
        self.logger.info(f"Strategy '{self.name}' stopped")
    
    async def update(self, market_data: Dict[str, MarketData]) -> List[TradingSignal]:
        """
        Update strategy with new market data and generate signals.
        
        Args:
            market_data: Current market data for tracked symbols
            
        Returns:
            List of trading signals
        """
        if not self.is_active:
            return []
        
        try:
            # Generate signals
            signals = await self.generate_signals(market_data)
            
            # Filter signals by confidence
            filtered_signals = [
                signal for signal in signals
                if signal.confidence >= self.risk_params.min_confidence
            ]
            
            # Store signal history
            self.signal_history.extend(filtered_signals)
            
            # Keep only recent signals (last 1000)
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]
            
            self.last_update = datetime.now()
            
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"Error updating strategy: {str(e)}")
            return []
    
    def calculate_position_size(
        self,
        signal: TradingSignal,
        current_price: float,
        portfolio_value: float
    ) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            signal: Trading signal
            current_price: Current market price
            portfolio_value: Total portfolio value
            
        Returns:
            Suggested position size in shares
        """
        # Use signal's position size if provided
        if signal.position_size is not None:
            return signal.position_size
        
        # Calculate based on risk parameters
        risk_amount = portfolio_value * self.risk_params.risk_per_trade
        
        # Calculate stop loss distance
        if signal.stop_loss:
            stop_distance = abs(current_price - signal.stop_loss)
        else:
            stop_distance = current_price * self.risk_params.stop_loss_pct
        
        # Position size = Risk Amount / Stop Distance
        if stop_distance > 0 and current_price > 0:
            position_size = risk_amount / stop_distance
        else:
            # Fallback to fixed percentage of portfolio
            max_position_value = portfolio_value * self.risk_params.max_position_size
            if current_price > 0:
                position_size = max_position_value / current_price
            else:
                position_size = 0  # Cannot calculate position size without valid price
        
        return max(1, int(position_size))  # At least 1 share, integer shares
    
    def validate_signal(self, signal: TradingSignal) -> bool:
        """
        Validate signal against risk parameters.
        
        Args:
            signal: Signal to validate
            
        Returns:
            True if signal passes validation, False otherwise
        """
        # Check confidence threshold
        if signal.confidence < self.risk_params.min_confidence:
            self.logger.debug(f"Signal rejected: confidence {signal.confidence} below threshold")
            return False
        
        # Check if symbol is in our universe
        if signal.symbol not in self.symbols:
            self.logger.debug(f"Signal rejected: {signal.symbol} not in strategy symbols")
            return False
        
        # Check maximum open positions
        if self.portfolio:
            current_positions = len(self.portfolio.get_all_open_positions())
            if (signal.signal_type in [SignalType.BUY] and 
                current_positions >= self.risk_params.max_open_positions):
                self.logger.debug("Signal rejected: maximum open positions reached")
                return False
        
        return True
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate strategy performance metrics.
        
        Returns:
            Dictionary with performance statistics
        """
        if not self.signal_history:
            return {'total_signals': 0, 'avg_confidence': 0.0}
        
        # Signal statistics
        total_signals = len(self.signal_history)
        buy_signals = sum(1 for s in self.signal_history if s.signal_type == SignalType.BUY)
        sell_signals = sum(1 for s in self.signal_history if s.signal_type == SignalType.SELL)
        avg_confidence = sum(s.confidence for s in self.signal_history) / total_signals
        
        # Recent activity (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_signals = [s for s in self.signal_history if s.timestamp > recent_cutoff]
        
        return {
            'total_signals': total_signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'avg_confidence': avg_confidence,
            'recent_signals_24h': len(recent_signals),
            'last_signal_time': self.signal_history[-1].timestamp.isoformat() if self.signal_history else None,
            'strategy_uptime': (datetime.now() - self.last_update).total_seconds() if self.last_update else 0
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
                'symbols': self.symbols,
                'is_active': self.is_active,
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'risk_params': self.risk_params.to_dict(),
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
        return f"{self.name}({status}, {len(self.symbols)} symbols)"
    
    def __repr__(self) -> str:
        """Detailed representation of strategy."""
        return (
            f"BaseStrategy(name='{self.name}', symbols={self.symbols}, "
            f"active={self.is_active})"
        )


# Example concrete strategy implementation for testing
class SimpleMovingAverageStrategy(BaseStrategy):
    """
    Example strategy implementation using simple moving averages.
    
    This is a basic example strategy that demonstrates the BaseStrategy
    interface. It generates buy signals when price is above the moving
    average and sell signals when below.
    """
    
    def __init__(self, symbols: List[str], ma_period: int = 20):
        """
        Initialize Simple Moving Average strategy.
        
        Args:
            symbols: List of symbols to trade
            ma_period: Moving average period
        """
        super().__init__("SimpleMA", symbols)
        self.ma_period = ma_period
        self.price_history: Dict[str, List[float]] = {symbol: [] for symbol in symbols}
    
    async def initialize(self) -> bool:
        """Initialize strategy components."""
        self.logger.info(f"Initializing SimpleMA strategy with {self.ma_period} period")
        return True
    
    async def cleanup(self) -> None:
        """Cleanup strategy resources."""
        self.logger.info("Cleaning up SimpleMA strategy")
    
    async def generate_signals(self, market_data: Dict[str, MarketData]) -> List[TradingSignal]:
        """
        Generate signals based on moving average crossover.
        
        Args:
            market_data: Current market data
            
        Returns:
            List of trading signals
        """
        signals = []
        
        for symbol, data in market_data.items():
            if symbol not in self.symbols:
                continue
            
            # Update price history
            self.price_history[symbol].append(data.price)
            
            # Keep only required history
            if len(self.price_history[symbol]) > self.ma_period:
                self.price_history[symbol] = self.price_history[symbol][-self.ma_period:]
            
            # Calculate moving average
            if len(self.price_history[symbol]) >= self.ma_period:
                ma = sum(self.price_history[symbol]) / len(self.price_history[symbol])
                current_price = data.price
                
                # Generate signals
                if current_price > ma * 1.02:  # 2% above MA
                    signal = TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        strength=SignalStrength.MODERATE,
                        confidence=0.6,
                        target_price=current_price,
                        stop_loss=ma * 0.95,
                        take_profit=current_price * 1.05,
                        reasoning=f"Price ${current_price:.2f} above MA ${ma:.2f}"
                    )
                    signals.append(signal)
                
                elif current_price < ma * 0.98:  # 2% below MA
                    signal = TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        strength=SignalStrength.MODERATE,
                        confidence=0.6,
                        target_price=current_price,
                        reasoning=f"Price ${current_price:.2f} below MA ${ma:.2f}"
                    )
                    signals.append(signal)
        
        return signals


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_simple_strategy():
        """Test the SimpleMA strategy."""
        
        # Create strategy
        strategy = SimpleMovingAverageStrategy(["AAPL", "MSFT"], ma_period=5)
        
        print(f"Strategy: {strategy}")
        
        # Simulate market data
        market_data = {
            "AAPL": MarketData("AAPL", 150.0, 149.5, 150.5, 100000),
            "MSFT": MarketData("MSFT", 300.0, 299.5, 300.5, 80000)
        }
        
        # Initialize strategy
        await strategy.initialize()
        strategy.is_active = True
        
        # Generate some signals by updating with different prices
        for i in range(10):
            # Simulate price movement
            for symbol in market_data:
                price_change = (i - 5) * 2.0  # Price trend
                market_data[symbol].price += price_change
            
            signals = await strategy.update(market_data)
            
            if signals:
                for signal in signals:
                    print(f"Signal {i}: {signal}")
        
        # Get performance metrics
        metrics = strategy.get_performance_metrics()
        print(f"Performance: {metrics}")
        
        await strategy.cleanup()
    
    asyncio.run(test_simple_strategy())