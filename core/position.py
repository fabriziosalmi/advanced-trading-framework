"""
Advanced Trading Framework - Position Data Structure

This module defines the Position class, which represents a trading position 
(long or short) in a specific security with all associated metadata.

Author: Senior Python Software Architect
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Optional
import time
import json


@dataclass
class Position:
    """
    Represents a trading position with comprehensive state tracking.
    
    A Position encapsulates all the information needed to track a trade
    from entry to exit, including profit/loss calculations and risk management.
    
    Attributes:
        ticker: The stock symbol (e.g., 'AAPL', 'MSFT')
        quantity: Number of shares (positive for long, negative for short)
        entry_price: The average price at which shares were acquired
        side: Position direction - 'LONG' (bought) or 'SHORT' (sold to open)
        entry_timestamp: Unix timestamp when position was opened
        stop_loss_price: Price level to automatically close at a loss
        take_profit_price: Price level to automatically close at a profit
        status: Current position state - 'OPEN' or 'CLOSED'
        current_price: Latest market price (updated externally)
    """
    
    ticker: str
    quantity: float
    entry_price: float
    side: str
    entry_timestamp: float = field(default_factory=time.time)
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    status: str = 'OPEN'
    current_price: float = field(default=0.0, init=False)
    
    def __post_init__(self) -> None:
        """
        Validate position parameters after initialization.
        
        Raises:
            ValueError: If position parameters are invalid
        """
        if self.quantity == 0:
            raise ValueError("Position quantity cannot be zero")
        
        if self.entry_price <= 0:
            raise ValueError("Entry price must be positive")
        
        if self.side not in ['LONG', 'SHORT']:
            raise ValueError("Side must be 'LONG' or 'SHORT'")
        
        if self.status not in ['OPEN', 'CLOSED']:
            raise ValueError("Status must be 'OPEN' or 'CLOSED'")
        
        # Initialize current price to entry price if not set
        if self.current_price == 0.0:
            self.current_price = self.entry_price
    
    def update_market_price(self, current_price: float) -> None:
        """
        Update the current market price for this position.
        
        This method should be called regularly to maintain accurate
        profit/loss calculations and risk management monitoring.
        
        Args:
            current_price: The latest market price for this security
            
        Raises:
            ValueError: If current_price is not positive
        """
        if current_price <= 0:
            raise ValueError("Current price must be positive")
        
        self.current_price = current_price
    
    @property
    def current_value(self) -> float:
        """
        Calculate the current market value of this position.
        
        Returns:
            Current market value (quantity * current_price)
        """
        return round(abs(self.quantity) * self.current_price, 2)
    
    @property
    def entry_value(self) -> float:
        """
        Calculate the entry value of this position.
        
        Returns:
            Entry value (quantity * entry_price)
        """
        return round(abs(self.quantity) * self.entry_price, 2)
    
    @property
    def unrealized_pl(self) -> float:
        """
        Calculate the unrealized profit/loss for this position.
        
        For LONG positions: (current_price - entry_price) * quantity
        For SHORT positions: (entry_price - current_price) * quantity
        
        Returns:
            Unrealized P&L in dollars
        """
        price_diff = self.current_price - self.entry_price
        
        if self.side == 'LONG':
            return round(price_diff * self.quantity, 2)
        else:  # SHORT
            return round(-price_diff * abs(self.quantity), 2)
    
    @property
    def unrealized_pl_pct(self) -> float:
        """
        Calculate the unrealized profit/loss as a percentage.
        
        Returns:
            Unrealized P&L as a percentage of entry value
        """
        if self.entry_value == 0:
            return 0.0
        
        return round((self.unrealized_pl / self.entry_value) * 100.0, 4)
    
    @property
    def is_profitable(self) -> bool:
        """
        Check if the position is currently profitable.
        
        Returns:
            True if unrealized P&L is positive, False otherwise
        """
        return self.unrealized_pl > 0
    
    @property
    def days_held(self) -> float:
        """
        Calculate how many days this position has been held.
        
        Returns:
            Number of days since position was opened
        """
        return (time.time() - self.entry_timestamp) / 86400.0  # 86400 seconds in a day
    
    def should_stop_loss(self) -> bool:
        """
        Check if position should be closed due to stop loss.
        
        Returns:
            True if current price has breached stop loss level
        """
        if self.stop_loss_price <= 0:
            return False
        
        if self.side == 'LONG':
            return self.current_price <= self.stop_loss_price
        else:  # SHORT
            # For SHORT positions, stop loss triggers when price goes UP
            return self.current_price >= self.stop_loss_price
    
    def should_take_profit(self) -> bool:
        """
        Check if position should be closed due to take profit.
        
        Returns:
            True if current price has reached take profit level
        """
        if self.take_profit_price <= 0:
            return False
        
        if self.side == 'LONG':
            return self.current_price >= self.take_profit_price
        else:  # SHORT
            # For SHORT positions, take profit triggers when price goes DOWN
            return self.current_price <= self.take_profit_price
    
    def to_dict(self) -> dict:
        """
        Convert position to dictionary for serialization.
        
        Returns:
            Dictionary representation of the position
        """
        return {
            'ticker': self.ticker,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'side': self.side,
            'entry_timestamp': self.entry_timestamp,
            'stop_loss_price': self.stop_loss_price,
            'take_profit_price': self.take_profit_price,
            'status': self.status,
            'current_price': self.current_price
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Position':
        """
        Create a Position instance from a dictionary.
        
        Args:
            data: Dictionary containing position data
            
        Returns:
            Position instance
        """
        position = cls(
            ticker=data['ticker'],
            quantity=data['quantity'],
            entry_price=data['entry_price'],
            side=data['side'],
            entry_timestamp=data.get('entry_timestamp', time.time()),
            stop_loss_price=data.get('stop_loss_price', 0.0),
            take_profit_price=data.get('take_profit_price', 0.0),
            status=data.get('status', 'OPEN')
        )
        position.current_price = data.get('current_price', data['entry_price'])
        return position
    
    def __str__(self) -> str:
        """
        String representation of the position.
        
        Returns:
            Human-readable position summary
        """
        return (f"Position({self.ticker}, {self.side}, {self.quantity} shares, "
                f"Entry: ${self.entry_price:.2f}, Current: ${self.current_price:.2f}, "
                f"P&L: ${self.unrealized_pl:.2f} ({self.unrealized_pl_pct:.1f}%))")
    
    def __repr__(self) -> str:
        """
        Detailed representation of the position.
        
        Returns:
            Complete position data representation
        """
        return (f"Position(ticker='{self.ticker}', quantity={self.quantity}, "
                f"entry_price={self.entry_price}, side='{self.side}', "
                f"status='{self.status}', unrealized_pl={self.unrealized_pl:.2f})")


# Example usage and testing functions
if __name__ == "__main__":
    # Example: Create a long position in AAPL
    position = Position(
        ticker="AAPL",
        quantity=100,
        entry_price=150.00,
        side="LONG",
        stop_loss_price=140.00,
        take_profit_price=165.00
    )
    
    print(f"Initial position: {position}")
    
    # Update market price and check P&L
    position.update_market_price(155.00)
    print(f"After price update: {position}")
    print(f"Should stop loss: {position.should_stop_loss()}")
    print(f"Should take profit: {position.should_take_profit()}")
    
    # Test serialization
    position_dict = position.to_dict()
    print(f"Serialized: {position_dict}")
    
    # Test deserialization
    new_position = Position.from_dict(position_dict)
    print(f"Deserialized: {new_position}")