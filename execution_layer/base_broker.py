"""
Advanced Trading Framework - Base Broker Interface

This module defines the abstract base class for all broker implementations,
providing a standardized interface for order execution, account management,
and market data retrieval.

Author: Senior Python Software Architect
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple
from enum import Enum
import logging
from datetime import datetime


class OrderType(Enum):
    """Order type enumeration for standardized order handling."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side enumeration for buy/sell operations."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status enumeration for tracking order lifecycle."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderTimeInForce(Enum):
    """Time in force enumeration for order duration control."""
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


class Order:
    """
    Standardized order representation for all broker implementations.
    
    This class provides a unified order structure that can be used
    across different broker APIs while maintaining consistency.
    """
    
    def __init__(
        self,
        symbol: str,
        quantity: float,
        order_type: OrderType,
        side: OrderSide,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: OrderTimeInForce = OrderTimeInForce.DAY,
        order_id: Optional[str] = None
    ):
        """
        Initialize a new order.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            quantity: Number of shares to trade
            order_type: Type of order (market, limit, etc.)
            side: Buy or sell side
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: Order duration specification
            order_id: Unique order identifier (set by broker)
        """
        self.symbol = symbol
        self.quantity = abs(quantity)  # Always positive
        self.order_type = order_type
        self.side = side
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.time_in_force = time_in_force
        self.order_id = order_id
        self.status = OrderStatus.PENDING
        self.filled_quantity = 0.0
        self.avg_fill_price = 0.0
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.broker_order_id = None  # Broker-specific ID
        
    def to_dict(self) -> Dict:
        """Convert order to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'side': self.side.value,
            'limit_price': self.limit_price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force.value,
            'order_id': self.order_id,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'avg_fill_price': self.avg_fill_price,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'broker_order_id': self.broker_order_id
        }
    
    def __str__(self) -> str:
        return (
            f"Order({self.side.value.upper()} {self.quantity} {self.symbol} "
            f"{self.order_type.value.upper()} @ {self.limit_price or 'MKT'}, "
            f"Status: {self.status.value.upper()})"
        )


class AccountInfo:
    """
    Standardized account information structure.
    
    Provides consistent account data representation across
    different broker implementations.
    """
    
    def __init__(
        self,
        account_id: str,
        buying_power: float,
        cash: float,
        portfolio_value: float,
        equity: float,
        day_trade_count: int = 0,
        pattern_day_trader: bool = False
    ):
        """
        Initialize account information.
        
        Args:
            account_id: Unique account identifier
            buying_power: Available buying power
            cash: Available cash balance
            portfolio_value: Total portfolio value
            equity: Account equity
            day_trade_count: Number of day trades
            pattern_day_trader: PDT status flag
        """
        self.account_id = account_id
        self.buying_power = buying_power
        self.cash = cash
        self.portfolio_value = portfolio_value
        self.equity = equity
        self.day_trade_count = day_trade_count
        self.pattern_day_trader = pattern_day_trader
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert account info to dictionary."""
        return {
            'account_id': self.account_id,
            'buying_power': self.buying_power,
            'cash': self.cash,
            'portfolio_value': self.portfolio_value,
            'equity': self.equity,
            'day_trade_count': self.day_trade_count,
            'pattern_day_trader': self.pattern_day_trader,
            'updated_at': self.updated_at.isoformat()
        }


class MarketData:
    """
    Standardized market data structure for price information.
    
    Provides consistent market data representation across
    different data providers and broker APIs.
    """
    
    def __init__(
        self,
        symbol: str,
        price: float,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        volume: Optional[int] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Initialize market data.
        
        Args:
            symbol: Stock symbol
            price: Current/last price
            bid: Best bid price
            ask: Best ask price
            volume: Trading volume
            timestamp: Data timestamp
        """
        self.symbol = symbol
        self.price = price
        self.bid = bid
        self.ask = ask
        self.volume = volume
        self.timestamp = timestamp or datetime.now()
    
    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None
    
    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid-market price."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2.0
        return self.price
    
    def to_dict(self) -> Dict:
        """Convert market data to dictionary."""
        return {
            'symbol': self.symbol,
            'price': self.price,
            'bid': self.bid,
            'ask': self.ask,
            'volume': self.volume,
            'spread': self.spread,
            'mid_price': self.mid_price,
            'timestamp': self.timestamp.isoformat()
        }


class BaseBroker(ABC):
    """
    Abstract base class for all broker implementations.
    
    This class defines the standard interface that all broker implementations
    must follow, ensuring consistency and interchangeability between different
    trading platforms (Alpaca, Interactive Brokers, paper trading, etc.).
    
    Key Features:
    - Standardized order management
    - Account information retrieval
    - Market data access
    - Position tracking
    - Connection management
    - Error handling and logging
    """
    
    def __init__(self, name: str):
        """
        Initialize the base broker.
        
        Args:
            name: Human-readable broker name
        """
        self.name = name
        self.connected = False
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self._orders: Dict[str, Order] = {}
        
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the broker.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the broker."""
        pass
    
    @abstractmethod
    async def is_connected(self) -> bool:
        """
        Check if broker connection is active.
        
        Returns:
            True if connected, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Optional[AccountInfo]:
        """
        Retrieve current account information.
        
        Returns:
            AccountInfo object if successful, None otherwise
        """
        pass
    
    @abstractmethod
    async def submit_order(self, order: Order) -> Optional[str]:
        """
        Submit an order to the broker.
        
        Args:
            order: Order object to submit
            
        Returns:
            Order ID if successful, None otherwise
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: Unique order identifier
            
        Returns:
            True if cancellation successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """
        Get current status of an order.
        
        Args:
            order_id: Unique order identifier
            
        Returns:
            Updated Order object if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_open_orders(self) -> List[Order]:
        """
        Get all open orders for the account.
        
        Returns:
            List of open Order objects
        """
        pass
    
    @abstractmethod
    async def get_positions(self) -> Dict[str, float]:
        """
        Get current positions from the broker.
        
        Returns:
            Dictionary mapping symbols to quantities
        """
        pass
    
    @abstractmethod
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """
        Get current market data for a symbol.
        
        Args:
            symbol: Stock symbol to query
            
        Returns:
            MarketData object if successful, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_market_data_batch(self, symbols: List[str]) -> Dict[str, MarketData]:
        """
        Get market data for multiple symbols.
        
        Args:
            symbols: List of stock symbols to query
            
        Returns:
            Dictionary mapping symbols to MarketData objects
        """
        pass
    
    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1D"
    ) -> Optional[List[Dict]]:
        """
        Get historical price data for a symbol.
        
        Args:
            symbol: Stock symbol to query
            start_date: Start date for historical data
            end_date: End date for historical data
            timeframe: Data timeframe (1D, 1H, 1M, etc.)
            
        Returns:
            List of OHLCV dictionaries if successful, None otherwise
        """
        pass
    
    # Convenience methods for common operations
    async def buy_market(self, symbol: str, quantity: float) -> Optional[str]:
        """
        Submit a market buy order.
        
        Args:
            symbol: Stock symbol to buy
            quantity: Number of shares to buy
            
        Returns:
            Order ID if successful, None otherwise
        """
        order = Order(
            symbol=symbol,
            quantity=quantity,
            order_type=OrderType.MARKET,
            side=OrderSide.BUY
        )
        return await self.submit_order(order)
    
    async def sell_market(self, symbol: str, quantity: float) -> Optional[str]:
        """
        Submit a market sell order.
        
        Args:
            symbol: Stock symbol to sell
            quantity: Number of shares to sell
            
        Returns:
            Order ID if successful, None otherwise
        """
        order = Order(
            symbol=symbol,
            quantity=quantity,
            order_type=OrderType.MARKET,
            side=OrderSide.SELL
        )
        return await self.submit_order(order)
    
    async def buy_limit(
        self,
        symbol: str,
        quantity: float,
        limit_price: float,
        time_in_force: OrderTimeInForce = OrderTimeInForce.DAY
    ) -> Optional[str]:
        """
        Submit a limit buy order.
        
        Args:
            symbol: Stock symbol to buy
            quantity: Number of shares to buy
            limit_price: Maximum price to pay
            time_in_force: Order duration
            
        Returns:
            Order ID if successful, None otherwise
        """
        order = Order(
            symbol=symbol,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            limit_price=limit_price,
            time_in_force=time_in_force
        )
        return await self.submit_order(order)
    
    async def sell_limit(
        self,
        symbol: str,
        quantity: float,
        limit_price: float,
        time_in_force: OrderTimeInForce = OrderTimeInForce.DAY
    ) -> Optional[str]:
        """
        Submit a limit sell order.
        
        Args:
            symbol: Stock symbol to sell
            quantity: Number of shares to sell
            limit_price: Minimum price to accept
            time_in_force: Order duration
            
        Returns:
            Order ID if successful, None otherwise
        """
        order = Order(
            symbol=symbol,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            side=OrderSide.SELL,
            limit_price=limit_price,
            time_in_force=time_in_force
        )
        return await self.submit_order(order)
    
    async def stop_loss(
        self,
        symbol: str,
        quantity: float,
        stop_price: float
    ) -> Optional[str]:
        """
        Submit a stop loss order.
        
        Args:
            symbol: Stock symbol to sell
            quantity: Number of shares to sell
            stop_price: Stop trigger price
            
        Returns:
            Order ID if successful, None otherwise
        """
        order = Order(
            symbol=symbol,
            quantity=quantity,
            order_type=OrderType.STOP,
            side=OrderSide.SELL,
            stop_price=stop_price
        )
        return await self.submit_order(order)
    
    def get_cached_order(self, order_id: str) -> Optional[Order]:
        """
        Get order from local cache.
        
        Args:
            order_id: Order identifier
            
        Returns:
            Cached Order object if found, None otherwise
        """
        return self._orders.get(order_id)
    
    def cache_order(self, order: Order) -> None:
        """
        Cache an order locally.
        
        Args:
            order: Order object to cache
        """
        if order.order_id:
            self._orders[order.order_id] = order
    
    def __str__(self) -> str:
        """String representation of broker."""
        status = "Connected" if self.connected else "Disconnected"
        return f"{self.name}Broker({status})"
    
    def __repr__(self) -> str:
        """Detailed representation of broker."""
        return f"BaseBroker(name='{self.name}', connected={self.connected})"