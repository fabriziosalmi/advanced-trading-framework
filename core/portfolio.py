"""
Advanced Trading Framework - Portfolio Management

This module defines the Portfolio class, which manages multiple positions,
cash, and provides comprehensive portfolio analytics and state persistence.

Author: Senior Python Software Architect
Version: 1.0.0
"""

import json
import os
import time
from typing import Dict, List, Optional
from datetime import datetime, timezone
import logging

from .position import Position
from .validation import validate_inputs, validate_ticker, validate_positive_number, validate_side


class Portfolio:
    """
    Comprehensive portfolio management system.
    
    The Portfolio class manages multiple trading positions, tracks cash balances,
    calculates portfolio-level metrics, and provides state persistence functionality.
    
    Key Features:
    - Multi-position management with single position per ticker constraint
    - Real-time portfolio valuation and risk metrics
    - Automatic state persistence and recovery
    - Comprehensive trade history and analytics
    - Risk management integration
    
    Attributes:
        initial_capital: Starting capital amount
        current_cash: Available cash for trading
        positions: Dictionary mapping tickers to Position objects
        state_file_path: Path to portfolio state persistence file
        trade_history: List of completed trades for analytics
    """
    
    def __init__(self, initial_capital: float, state_file: str = "data/portfolio_state.json"):
        """
        Initialize a new portfolio or load existing state.
        
        Args:
            initial_capital: Starting capital for the portfolio
            state_file: Path to JSON file for state persistence
            
        Raises:
            ValueError: If initial_capital is not positive
        """
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        
        self.initial_capital = initial_capital
        self.current_cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.state_file_path = state_file
        self.trade_history: List[Dict] = []
        self.portfolio_value_history: List[Dict] = []  # Track daily portfolio values
        self.created_timestamp = time.time()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        
        # Attempt to load existing state
        if os.path.exists(state_file):
            if self.load_state():
                self.logger.info(f"Portfolio state loaded from {state_file}")
            else:
                self.logger.warning(f"Failed to load portfolio state from {state_file}")
        
        # Record initial portfolio value
        self.record_portfolio_value()
    
    @validate_inputs(
        ticker=validate_ticker,
        quantity=validate_positive_number,
        entry_price=validate_positive_number,
        side=validate_side
    )
    def open_position(
        self,
        ticker: str,
        quantity: float,
        entry_price: float,
        side: str,
        sl_price: float = 0.0,
        tp_price: float = 0.0
    ) -> bool:
        """
        Open a new trading position.
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            quantity: Number of shares to trade
            entry_price: Price per share for the trade
            side: 'LONG' or 'SHORT'
            sl_price: Stop loss price (optional)
            tp_price: Take profit price (optional)
            
        Returns:
            True if position opened successfully, False otherwise
            
        Raises:
            ValueError: If position parameters are invalid
        """
        try:
            # Validate inputs
            if ticker in self.positions:
                self.logger.warning(f"Position for {ticker} already exists")
                return False
            
            if quantity <= 0:
                raise ValueError("Quantity must be positive")
            
            if entry_price <= 0:
                raise ValueError("Entry price must be positive")
            
            # Calculate required capital
            required_capital = abs(quantity) * entry_price
            
            # Check available cash
            if required_capital > self.current_cash:
                self.logger.warning(
                    f"Insufficient cash for {ticker} position. "
                    f"Required: ${required_capital:.2f}, Available: ${self.current_cash:.2f}"
                )
                return False
            
            # Create new position
            position = Position(
                ticker=ticker,
                quantity=quantity,
                entry_price=entry_price,
                side=side,
                stop_loss_price=sl_price,
                take_profit_price=tp_price
            )
            
            # Update portfolio state
            self.positions[ticker] = position
            self.current_cash -= required_capital
            
            # Log the trade
            trade_record = {
                'timestamp': time.time(),
                'action': 'OPEN',
                'ticker': ticker,
                'quantity': quantity,
                'price': entry_price,
                'side': side,
                'value': required_capital
            }
            self.trade_history.append(trade_record)
            
            self.logger.info(
                f"Opened {side} position: {ticker} {quantity} shares @ ${entry_price:.2f}"
            )
            
            # Auto-save state
            self.save_state()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to open position for {ticker}: {str(e)}")
            return False
    
    def close_position(self, ticker: str, exit_price: float) -> float:
        """
        Close an existing position and realize P&L.
        
        Args:
            ticker: Stock symbol of position to close
            exit_price: Price at which to close the position
            
        Returns:
            Realized profit/loss from the closed position
            
        Raises:
            ValueError: If position doesn't exist or exit_price is invalid
        """
        if ticker not in self.positions:
            raise ValueError(f"No position found for {ticker}")
        
        if exit_price <= 0:
            raise ValueError("Exit price must be positive")
        
        position = self.positions[ticker]
        
        # Calculate realized P&L
        if position.side == 'LONG':
            realized_pl = round((exit_price - position.entry_price) * position.quantity, 2)
        else:  # SHORT
            realized_pl = round((position.entry_price - exit_price) * abs(position.quantity), 2)
        
        # Calculate exit value
        exit_value = round(abs(position.quantity) * exit_price, 2)
        
        # Update cash balance
        if position.side == 'LONG':
            self.current_cash += exit_value
        else:  # SHORT
            # For short positions, we return the initial margin plus/minus P&L
            self.current_cash += position.entry_value + realized_pl
        
        # Log the trade
        trade_record = {
            'timestamp': time.time(),
            'action': 'CLOSE',
            'ticker': ticker,
            'quantity': position.quantity,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'side': position.side,
            'realized_pl': realized_pl,
            'days_held': position.days_held
        }
        self.trade_history.append(trade_record)
        
        # Mark position as closed and remove from active positions
        position.status = 'CLOSED'
        del self.positions[ticker]
        
        self.logger.info(
            f"Closed {position.side} position: {ticker} @ ${exit_price:.2f}, "
            f"P&L: ${realized_pl:.2f}"
        )
        
        # Auto-save state
        self.save_state()
        
        return realized_pl
    
    def get_position(self, ticker: str) -> Optional[Position]:
        """
        Retrieve a position by ticker symbol.
        
        Args:
            ticker: Stock symbol to look up
            
        Returns:
            Position object if found, None otherwise
        """
        return self.positions.get(ticker)
    
    def get_all_open_positions(self) -> List[Position]:
        """
        Get all currently open positions.
        
        Returns:
            List of all open Position objects
        """
        return list(self.positions.values())
    
    def update_all_positions(self, price_data: Dict[str, float]) -> None:
        """
        Update market prices for all positions.
        
        Args:
            price_data: Dictionary mapping tickers to current prices
        """
        for ticker, position in self.positions.items():
            if ticker in price_data:
                position.update_market_price(price_data[ticker])
    
    def calculate_total_equity(self) -> float:
        """
        Calculate total portfolio equity (cash + position values).
        
        Returns:
            Total portfolio value in dollars
        """
        positions_value = sum(pos.current_value for pos in self.positions.values())
        return round(self.current_cash + positions_value, 2)
    
    def calculate_total_pl(self) -> float:
        """
        Calculate total unrealized P&L across all positions.
        
        Returns:
            Total unrealized profit/loss in dollars
        """
        return round(sum(pos.unrealized_pl for pos in self.positions.values()), 2)
    
    def calculate_total_pl_pct(self) -> float:
        """
        Calculate total portfolio return percentage.
        
        Returns:
            Portfolio return as percentage of initial capital
        """
        current_equity = self.calculate_total_equity()
        if self.initial_capital == 0:
            return 0.0  # Avoid division by zero
        return round(((current_equity - self.initial_capital) / self.initial_capital) * 100.0, 4)
    
    def get_portfolio_summary(self) -> Dict:
        """
        Generate comprehensive portfolio summary.
        
        Returns:
            Dictionary containing portfolio metrics and statistics
        """
        total_equity = self.calculate_total_equity()
        total_pl = self.calculate_total_pl()
        
        # Calculate realized P&L from trade history
        realized_pl = sum(
            trade.get('realized_pl', 0) 
            for trade in self.trade_history 
            if trade['action'] == 'CLOSE'
        )
        
        # Position breakdown
        long_positions = [p for p in self.positions.values() if p.side == 'LONG']
        short_positions = [p for p in self.positions.values() if p.side == 'SHORT']
        
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'initial_capital': self.initial_capital,
            'current_cash': self.current_cash,
            'total_equity': total_equity,
            'total_return_pct': self.calculate_total_pl_pct(),
            'unrealized_pl': total_pl,
            'realized_pl': realized_pl,
            'total_pl': total_pl + realized_pl,
            'position_count': len(self.positions),
            'long_positions': len(long_positions),
            'short_positions': len(short_positions),
            'cash_percentage': (self.current_cash / total_equity) * 100.0 if total_equity > 0 else 100.0,
            'positions_value': total_equity - self.current_cash,
            'days_active': (time.time() - self.created_timestamp) / 86400.0
        }
    
    def get_risk_metrics(self) -> Dict:
        """
        Calculate portfolio risk metrics.
        
        Returns:
            Dictionary containing risk analysis metrics
        """
        if not self.positions:
            return {'portfolio_var': 0.0, 'max_position_risk': 0.0, 'concentration_risk': 0.0}
        
        total_equity = self.calculate_total_equity()
        position_values = [pos.current_value for pos in self.positions.values()]
        
        # Portfolio concentration risk (largest position as % of portfolio)
        max_position_value = max(position_values) if position_values else 0
        concentration_risk = (max_position_value / total_equity) * 100.0 if total_equity > 0 else 0
        
        # Value at Risk (simplified - based on current unrealized losses)
        unrealized_losses = [pos.unrealized_pl for pos in self.positions.values() if pos.unrealized_pl < 0]
        portfolio_var = sum(unrealized_losses) if unrealized_losses else 0.0
        
        return {
            'portfolio_var': portfolio_var,
            'max_position_risk': concentration_risk,
            'concentration_risk': concentration_risk,
            'positions_at_risk': len(unrealized_losses),
            'avg_position_size_pct': (100.0 / len(self.positions)) if self.positions else 0.0
        }
    
    def record_portfolio_value(self) -> None:
        """
        Record current portfolio value for performance tracking.
        
        Stores daily portfolio value snapshots for performance analysis.
        Only records one value per day to avoid excessive data.
        """
        current_time = datetime.now(timezone.utc)
        current_date = current_time.date()
        current_value = self.calculate_total_equity()
        
        # Check if we already have a record for today
        today_records = [record for record in self.portfolio_value_history 
                        if record['date'] == current_date.isoformat()]
        
        if not today_records:
            # Add new daily record
            record = {
                'date': current_date.isoformat(),
                'timestamp': current_time.isoformat(),
                'portfolio_value': current_value,
                'cash': self.current_cash,
                'positions_value': current_value - self.current_cash,
                'num_positions': len(self.positions)
            }
            self.portfolio_value_history.append(record)
            
            # Keep only last 365 days to prevent excessive memory usage
            if len(self.portfolio_value_history) > 365:
                self.portfolio_value_history = self.portfolio_value_history[-365:]
    
    def get_portfolio_value_history(self) -> List[Dict]:
        """
        Get portfolio value history for performance analysis.
        
        Returns:
            List of daily portfolio value records
        """
        return self.portfolio_value_history.copy()
    
    def save_state(self) -> None:
        """
        Persist portfolio state to JSON file.
        
        Saves current cash, positions, and trade history to enable
        recovery after application restart.
        """
        try:
            state_data = {
                'initial_capital': self.initial_capital,
                'current_cash': self.current_cash,
                'created_timestamp': self.created_timestamp,
                'last_saved': time.time(),
                'positions': {
                    ticker: position.to_dict() 
                    for ticker, position in self.positions.items()
                },
                'trade_history': self.trade_history,
                'portfolio_value_history': self.portfolio_value_history
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.state_file_path), exist_ok=True)
            
            # Write to temporary file first, then atomic rename
            temp_file = self.state_file_path + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            # Atomic rename
            os.replace(temp_file, self.state_file_path)
            
            self.logger.debug(f"Portfolio state saved to {self.state_file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save portfolio state: {str(e)}")
    
    def load_state(self) -> bool:
        """
        Load portfolio state from JSON file.
        
        Returns:
            True if state loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(self.state_file_path):
                self.logger.info("No existing state file found")
                return False
            
            with open(self.state_file_path, 'r') as f:
                state_data = json.load(f)
            
            # Restore portfolio attributes
            self.initial_capital = state_data.get('initial_capital', self.initial_capital)
            self.current_cash = state_data.get('current_cash', self.current_cash)
            self.created_timestamp = state_data.get('created_timestamp', self.created_timestamp)
            self.trade_history = state_data.get('trade_history', [])
            self.portfolio_value_history = state_data.get('portfolio_value_history', [])
            
            # Restore positions
            self.positions = {}
            positions_data = state_data.get('positions', {})
            for ticker, position_data in positions_data.items():
                try:
                    position = Position.from_dict(position_data)
                    self.positions[ticker] = position
                except Exception as e:
                    self.logger.warning(f"Failed to restore position {ticker}: {str(e)}")
            
            self.logger.info(
                f"Portfolio state loaded: {len(self.positions)} positions, "
                f"${self.current_cash:.2f} cash"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load portfolio state: {str(e)}")
            return False
    
    def __str__(self) -> str:
        """
        String representation of portfolio.
        
        Returns:
            Human-readable portfolio summary
        """
        summary = self.get_portfolio_summary()
        return (
            f"Portfolio(Equity: ${summary['total_equity']:.2f}, "
            f"Cash: ${summary['current_cash']:.2f}, "
            f"Positions: {summary['position_count']}, "
            f"Return: {summary['total_return_pct']:.1f}%)"
        )
    
    def to_dict(self) -> dict:
        """
        Convert portfolio to dictionary representation.
        
        This method provides a comprehensive dictionary representation of the
        portfolio state, suitable for serialization and API responses.
        
        Returns:
            Dictionary containing all portfolio data
        """
        return {
            'initial_capital': self.initial_capital,
            'current_cash': self.current_cash,
            'created_timestamp': self.created_timestamp,
            'positions': {
                ticker: position.to_dict() 
                for ticker, position in self.positions.items()
            },
            'trade_history': self.trade_history,
            'portfolio_value_history': self.portfolio_value_history,
            'summary': self.get_portfolio_summary(),
            'risk_metrics': self.get_risk_metrics()
        }
    
    def sync_with_broker(self, broker_positions: Dict[str, float]) -> None:
        """
        Synchronize portfolio positions with broker's actual positions.
        
        This method reconciles the internal portfolio state with the broker's
        reported positions, handling cases where positions were modified
        externally (e.g., partial fills, manual closures).
        
        Args:
            broker_positions: Dictionary mapping tickers to quantities from broker
        """
        self.logger.info("🔄 Synchronizing portfolio with broker positions")
        
        synced_tickers = set()
        
        # Update existing positions or add new ones from broker
        for ticker, broker_qty in broker_positions.items():
            synced_tickers.add(ticker)
            
            if ticker in self.positions:
                # Update existing position quantity if it differs
                current_qty = self.positions[ticker].quantity
                if abs(current_qty - broker_qty) > 0.01:  # Allow small floating point differences
                    self.logger.info(f"📊 Updating {ticker} position: {current_qty} → {broker_qty}")
                    self.positions[ticker].quantity = broker_qty
                    
                    # Recalculate entry price based on remaining quantity
                    # This is a simplification; in reality, we'd need trade history
                    if broker_qty == 0:
                        # Position was closed externally
                        self.logger.warning(f"⚠️ Position {ticker} was closed externally")
                        del self.positions[ticker]
                    else:
                        # Adjust cash if quantity changed
                        qty_diff = broker_qty - current_qty
                        if qty_diff > 0:
                            # Additional shares acquired (perhaps partial fill)
                            self.logger.info(f"📈 Additional {qty_diff} shares of {ticker} acquired")
                        elif qty_diff < 0:
                            # Shares sold (partial close)
                            self.logger.info(f"📉 {abs(qty_diff)} shares of {ticker} sold externally")
            else:
                # New position from broker (shouldn't happen in normal operation)
                self.logger.warning(f"⚠️ New position {ticker} found at broker (qty: {broker_qty})")
                # We can't create a position without entry price, so we'll skip for now
        
        # Check for positions that exist locally but not at broker
        for ticker in list(self.positions.keys()):
            if ticker not in synced_tickers:
                self.logger.warning(f"⚠️ Position {ticker} exists locally but not at broker - removing")
                del self.positions[ticker]
        
        self.logger.info(f"✅ Portfolio sync completed. {len(self.positions)} positions active.")
    
    def __repr__(self) -> str:
        """
        Detailed representation of portfolio.
        
        Returns:
            Complete portfolio data representation
        """
        return (
            f"Portfolio(initial_capital={self.initial_capital}, "
            f"current_cash={self.current_cash}, "
            f"positions={len(self.positions)}, "
            f"equity={self.calculate_total_equity():.2f})"
        )


# Example usage and testing
if __name__ == "__main__":
    # Create a new portfolio
    portfolio = Portfolio(initial_capital=100000.0, state_file="test_portfolio.json")
    
    print(f"Initial portfolio: {portfolio}")
    
    # Open some positions
    portfolio.open_position("AAPL", 100, 150.00, "LONG", sl_price=140.00, tp_price=165.00)
    portfolio.open_position("MSFT", 50, 300.00, "LONG", sl_price=280.00, tp_price=330.00)
    
    print(f"After opening positions: {portfolio}")
    
    # Update prices and check portfolio
    portfolio.update_all_positions({"AAPL": 155.00, "MSFT": 310.00})
    
    summary = portfolio.get_portfolio_summary()
    print(f"Portfolio Summary: {summary}")
    
    risk_metrics = portfolio.get_risk_metrics()
    print(f"Risk Metrics: {risk_metrics}")
    
    # Close a position
    realized_pl = portfolio.close_position("AAPL", 158.00)
    print(f"Realized P&L from AAPL: ${realized_pl:.2f}")
    
    print(f"Final portfolio: {portfolio}")
    
    # Clean up test file
    if os.path.exists("test_portfolio.json"):
        os.remove("test_portfolio.json")