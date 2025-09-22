"""
Advanced Trading Framework - Alpaca Broker Implementation

This module implements the Alpaca API broker for real trading capabilities,
providing live market data, order execution, and account management through
the Alpaca Markets API.

Author: Senior Python Software Architect
Version: 1.0.0
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import TimeFrame
except ImportError:
    tradeapi = None
    TimeFrame = None

from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from .base_broker import (
    BaseBroker, Order, OrderType, OrderSide, OrderStatus, OrderTimeInForce,
    AccountInfo, MarketData
)


class AlpacaBroker(BaseBroker):
    """
    Alpaca Markets broker implementation.
    
    Provides real trading capabilities through the Alpaca API including:
    - Live order execution
    - Real-time market data
    - Account management
    - Position tracking
    - Historical data access
    
    Requires Alpaca API credentials and handles both paper and live trading.
    """
    
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str = "https://paper-api.alpaca.markets",
        data_url: str = "https://data.alpaca.markets"
    ):
        """
        Initialize Alpaca broker connection.
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            base_url: Alpaca API base URL (paper or live)
            data_url: Alpaca data API URL
            
        Raises:
            ImportError: If alpaca_trade_api package is not installed
        """
        super().__init__("Alpaca")
        
        if tradeapi is None:
            raise ImportError(
                "alpaca-trade-api package is required for AlpacaBroker. "
                "Install with: pip install alpaca-trade-api"
            )
        
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.data_url = data_url
        self.api: Optional[tradeapi.REST] = None
        
        # Alpaca-specific order ID mapping
        self._alpaca_order_map: Dict[str, str] = {}  # our_id -> alpaca_id
        self._reverse_order_map: Dict[str, str] = {}  # alpaca_id -> our_id
        self._order_timestamps: Dict[str, datetime] = {}  # our_id -> creation timestamp
        
        # Periodic cleanup
        self._last_cleanup = datetime.now()
        self._cleanup_interval = timedelta(hours=1)  # Clean up every hour
        self._max_order_age = timedelta(days=7)  # Remove mappings older than 7 days
        
        self.logger.info(f"AlpacaBroker initialized with base_url: {base_url}")
    
    def _cleanup_order_mappings(self, order_id: str):
        """
        Clean up order ID mappings for completed/canceled orders.
        
        Args:
            order_id: Our order identifier
        """
        alpaca_order_id = self._alpaca_order_map.get(order_id)
        if alpaca_order_id:
            # Remove both mappings
            self._alpaca_order_map.pop(order_id, None)
            self._reverse_order_map.pop(alpaca_order_id, None)
            self._order_timestamps.pop(order_id, None)
            self.logger.debug(f"Cleaned up order mappings for {order_id}")
    
    def _cleanup_stale_mappings(self):
        """
        Clean up order mappings that are older than the maximum age.
        This prevents memory leaks from orders that were never properly cleaned up.
        """
        if datetime.now() - self._last_cleanup < self._cleanup_interval:
            return  # Not time for cleanup yet
        
        current_time = datetime.now()
        stale_orders = []
        
        for order_id, timestamp in self._order_timestamps.items():
            if current_time - timestamp > self._max_order_age:
                stale_orders.append(order_id)
        
        for order_id in stale_orders:
            self.logger.info(f"Cleaning up stale order mapping: {order_id}")
            self._cleanup_order_mappings(order_id)
        
        self._last_cleanup = current_time
        
        if stale_orders:
            self.logger.info(f"Cleaned up {len(stale_orders)} stale order mappings")
    
    async def connect(self) -> bool:
        """
        Establish connection to Alpaca API.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.api = tradeapi.REST(
                key_id=self.api_key,
                secret_key=self.secret_key,
                base_url=self.base_url,
                api_version='v2'
            )
            
            # Test connection by getting account info
            account = self.api.get_account()
            if account:
                self.connected = True
                self.logger.info(f"Connected to Alpaca API. Account: {account.id}")
                return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Alpaca API: {str(e)}")
            self.connected = False
            
        return False
    
    async def disconnect(self) -> None:
        """Disconnect from Alpaca API."""
        self.connected = False
        self.api = None
        
        # Clean up all order mappings on disconnect
        order_ids = list(self._alpaca_order_map.keys())
        for order_id in order_ids:
            self._cleanup_order_mappings(order_id)
        
        self.logger.info("Disconnected from Alpaca API and cleaned up order mappings")
    
    async def is_connected(self) -> bool:
        """
        Check if Alpaca API connection is active.
        
        Returns:
            True if connected, False otherwise
        """
        if not self.connected or not self.api:
            return False
        
        try:
            # Test connection with a simple API call
            self.api.get_account()
            return True
        except Exception as e:
            self.logger.warning(f"Connection test failed: {str(e)}")
            self.connected = False
            return False
    
    async def get_account_info(self) -> Optional[AccountInfo]:
        """
        Retrieve current account information from Alpaca.
        
        Returns:
            AccountInfo object if successful, None otherwise
        """
        if not await self.is_connected():
            return None
        
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_fixed(5),
            retry=retry_if_exception_type(Exception),
            reraise=True
        )
        def _get_account():
            return self.api.get_account()
        
        try:
            account = _get_account()
            
            return AccountInfo(
                account_id=account.id,
                buying_power=float(account.buying_power),
                cash=float(account.cash),
                portfolio_value=float(account.portfolio_value),
                equity=float(account.equity),
                day_trade_count=int(account.daytrade_count),
                pattern_day_trader=account.pattern_day_trader
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get account info after retries: {str(e)}")
            return None
    
    def _convert_order_to_alpaca(self, order: Order) -> Dict[str, Any]:
        """
        Convert our Order object to Alpaca API format.
        
        Args:
            order: Our standardized Order object
            
        Returns:
            Dictionary with Alpaca API order parameters
        """
        # Map our order types to Alpaca format
        alpaca_order_type = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP: "stop",
            OrderType.STOP_LIMIT: "stop_limit"
        }.get(order.order_type, "market")
        
        # Map our sides to Alpaca format
        alpaca_side = {
            OrderSide.BUY: "buy",
            OrderSide.SELL: "sell"
        }.get(order.side, "buy")
        
        # Map our time in force to Alpaca format
        alpaca_tif = {
            OrderTimeInForce.DAY: "day",
            OrderTimeInForce.GTC: "gtc",
            OrderTimeInForce.IOC: "ioc",
            OrderTimeInForce.FOK: "fok"
        }.get(order.time_in_force, "day")
        
        alpaca_order = {
            "symbol": order.symbol,
            "qty": str(int(order.quantity)),
            "side": alpaca_side,
            "type": alpaca_order_type,
            "time_in_force": alpaca_tif
        }
        
        # Add price parameters based on order type
        if order.order_type == OrderType.LIMIT:
            alpaca_order["limit_price"] = str(order.limit_price)
        elif order.order_type == OrderType.STOP:
            alpaca_order["stop_price"] = str(order.stop_price)
        elif order.order_type == OrderType.STOP_LIMIT:
            alpaca_order["limit_price"] = str(order.limit_price)
            alpaca_order["stop_price"] = str(order.stop_price)
        
        return alpaca_order
    
    def _convert_alpaca_to_order(self, alpaca_order: Any) -> Order:
        """
        Convert Alpaca API order to our Order object.
        
        Args:
            alpaca_order: Alpaca API order object
            
        Returns:
            Our standardized Order object
        """
        # Map Alpaca order types to our enum
        order_type_map = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "stop": OrderType.STOP,
            "stop_limit": OrderType.STOP_LIMIT
        }
        
        # Map Alpaca sides to our enum
        side_map = {
            "buy": OrderSide.BUY,
            "sell": OrderSide.SELL
        }
        
        # Map Alpaca status to our enum
        status_map = {
            "new": OrderStatus.SUBMITTED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "done_for_day": OrderStatus.CANCELLED,
            "canceled": OrderStatus.CANCELLED,
            "expired": OrderStatus.EXPIRED,
            "replaced": OrderStatus.CANCELLED,
            "pending_cancel": OrderStatus.CANCELLED,
            "pending_replace": OrderStatus.SUBMITTED,
            "accepted": OrderStatus.SUBMITTED,
            "pending_new": OrderStatus.PENDING,
            "accepted_for_bidding": OrderStatus.SUBMITTED,
            "stopped": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
            "suspended": OrderStatus.CANCELLED,
            "calculated": OrderStatus.SUBMITTED
        }
        
        # Map time in force
        tif_map = {
            "day": OrderTimeInForce.DAY,
            "gtc": OrderTimeInForce.GTC,
            "ioc": OrderTimeInForce.IOC,
            "fok": OrderTimeInForce.FOK
        }
        
        # Get our order ID from the mapping
        our_order_id = self._reverse_order_map.get(alpaca_order.id)
        
        order = Order(
            symbol=alpaca_order.symbol,
            quantity=float(alpaca_order.qty),
            order_type=order_type_map.get(alpaca_order.order_type, OrderType.MARKET),
            side=side_map.get(alpaca_order.side, OrderSide.BUY),
            limit_price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
            stop_price=float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
            time_in_force=tif_map.get(alpaca_order.time_in_force, OrderTimeInForce.DAY),
            order_id=our_order_id
        )
        
        # Update order status and fill information
        order.status = status_map.get(alpaca_order.status, OrderStatus.PENDING)
        order.filled_quantity = float(alpaca_order.filled_qty or 0)
        order.avg_fill_price = float(alpaca_order.filled_avg_price or 0)
        order.broker_order_id = alpaca_order.id
        
        # Parse timestamps
        try:
            order.created_at = datetime.fromisoformat(alpaca_order.created_at.replace('Z', '+00:00'))
            order.updated_at = datetime.fromisoformat(alpaca_order.updated_at.replace('Z', '+00:00'))
        except:
            # Fallback to current time if parsing fails
            order.created_at = datetime.now()
            order.updated_at = datetime.now()
        
        # Clean up mappings for terminal order states
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED, OrderStatus.REJECTED]:
            if our_order_id:
                self._cleanup_order_mappings(our_order_id)
        
        return order
    
    async def submit_order(self, order: Order) -> Optional[str]:
        """
        Submit an order to Alpaca.
        
        Args:
            order: Order object to submit
            
        Returns:
            Order ID if successful, None otherwise
        """
        if not await self.is_connected():
            self.logger.error("Cannot submit order: not connected to Alpaca")
            return None
        
        try:
            # Generate unique order ID
            order_id = str(uuid.uuid4())
            order.order_id = order_id
            
            # Convert to Alpaca format
            alpaca_order_data = self._convert_order_to_alpaca(order)
            
            # Submit order to Alpaca with retry
            @retry(
                stop=stop_after_attempt(3),
                wait=wait_fixed(5),
                retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
                reraise=True
            )
            def _submit_order():
                return self.api.submit_order(**alpaca_order_data)
            
            alpaca_order = _submit_order()
            
            # Store order mappings
            self._alpaca_order_map[order_id] = alpaca_order.id
            self._reverse_order_map[alpaca_order.id] = order_id
            self._order_timestamps[order_id] = datetime.now()
            
            # Update order with Alpaca information
            order.broker_order_id = alpaca_order.id
            order.status = OrderStatus.SUBMITTED
            
            # Cache the order
            self.cache_order(order)
            
            self.logger.info(f"Order submitted: {order}")
            return order_id
            
        except Exception as e:
            self.logger.error(f"Failed to submit order after retries: {str(e)}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order in Alpaca.
        
        Args:
            order_id: Our order identifier
            
        Returns:
            True if cancellation successful, False otherwise
        """
        if not await self.is_connected():
            return False
        
        try:
            # Get Alpaca order ID
            alpaca_order_id = self._alpaca_order_map.get(order_id)
            if not alpaca_order_id:
                self.logger.error(f"Cannot find Alpaca order ID for {order_id}")
                return False
            
            # Cancel order in Alpaca
            self.api.cancel_order(alpaca_order_id)
            
            # Update cached order status
            cached_order = self.get_cached_order(order_id)
            if cached_order:
                cached_order.status = OrderStatus.CANCELLED
                cached_order.updated_at = datetime.now()
            
            # Clean up order mappings since order is cancelled
            self._cleanup_order_mappings(order_id)
            
            self.logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """
        Get current status of an order from Alpaca.
        
        Args:
            order_id: Our order identifier
            
        Returns:
            Updated Order object if found, None otherwise
        """
        # Periodic cleanup of stale mappings
        self._cleanup_stale_mappings()
        
        if not await self.is_connected():
            return None
        
        try:
            # Get Alpaca order ID
            alpaca_order_id = self._alpaca_order_map.get(order_id)
            if not alpaca_order_id:
                return None
            
            # Get order from Alpaca
            alpaca_order = self.api.get_order(alpaca_order_id)
            
            # Convert to our format
            order = self._convert_alpaca_to_order(alpaca_order)
            
            # Update cache
            self.cache_order(order)
            
            return order
            
        except Exception as e:
            self.logger.error(f"Failed to get order status for {order_id}: {str(e)}")
            return None
    
    async def get_open_orders(self) -> List[Order]:
        """
        Get all open orders from Alpaca.
        
        Returns:
            List of open Order objects
        """
        if not await self.is_connected():
            return []
        
        try:
            alpaca_orders = self.api.list_orders(status='open')
            orders = []
            
            for alpaca_order in alpaca_orders:
                order = self._convert_alpaca_to_order(alpaca_order)
                orders.append(order)
                self.cache_order(order)
            
            return orders
            
        except Exception as e:
            self.logger.error(f"Failed to get open orders: {str(e)}")
            return []
    
    async def get_positions(self) -> Dict[str, float]:
        """
        Get current positions from Alpaca.
        
        Returns:
            Dictionary mapping symbols to quantities
        """
        if not await self.is_connected():
            return {}
        
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_fixed(5),
            retry=retry_if_exception_type(Exception),
            reraise=True
        )
        def _get_positions():
            return self.api.list_positions()
        
        try:
            alpaca_positions = _get_positions()
            positions = {}
            
            for position in alpaca_positions:
                positions[position.symbol] = float(position.qty)
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Failed to get positions after retries: {str(e)}")
            return {}
    
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """
        Get current market data for a symbol from Alpaca.
        
        Args:
            symbol: Stock symbol to query
            
        Returns:
            MarketData object if successful, None otherwise
        """
        if not await self.is_connected():
            return None
        
        try:
            # Get latest quote with retry
            @retry(
                stop=stop_after_attempt(3),
                wait=wait_fixed(5),
                retry=retry_if_exception_type(Exception),
                reraise=True
            )
            def _get_quote():
                return self.api.get_latest_quote(symbol)
            
            # Get latest trade for price with retry
            @retry(
                stop=stop_after_attempt(3),
                wait=wait_fixed(5),
                retry=retry_if_exception_type(Exception),
                reraise=True
            )
            def _get_trade():
                return self.api.get_latest_trade(symbol)
            
            quote = _get_quote()
            trade = _get_trade()
            
            return MarketData(
                symbol=symbol,
                price=float(trade.price),
                bid=float(quote.bid_price),
                ask=float(quote.ask_price),
                volume=int(trade.size),
                timestamp=datetime.fromisoformat(trade.timestamp.replace('Z', '+00:00'))
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get market data for {symbol} after retries: {str(e)}")
            return None
    
    async def get_market_data_batch(self, symbols: List[str]) -> Dict[str, MarketData]:
        """
        Get market data for multiple symbols from Alpaca.
        
        Args:
            symbols: List of stock symbols to query
            
        Returns:
            Dictionary mapping symbols to MarketData objects
        """
        if not await self.is_connected():
            return {}
        
        market_data = {}
        
        # Alpaca API doesn't have efficient batch quotes, so we'll request individually
        # In production, this could be optimized with concurrent requests
        for symbol in symbols:
            data = await self.get_market_data(symbol)
            if data:
                market_data[symbol] = data
        
        return market_data
    
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1D"
    ) -> Optional[List[Dict]]:
        """
        Get historical price data from Alpaca.
        
        Args:
            symbol: Stock symbol to query
            start_date: Start date for historical data
            end_date: End date for historical data
            timeframe: Data timeframe (1D, 1H, 1M, etc.)
            
        Returns:
            List of OHLCV dictionaries if successful, None otherwise
        """
        if not await self.is_connected():
            return None
        
        try:
            # Map timeframe to Alpaca format
            timeframe_map = {
                "1M": TimeFrame.Minute,
                "5M": TimeFrame.Minute,
                "15M": TimeFrame.Minute,
                "1H": TimeFrame.Hour,
                "1D": TimeFrame.Day
            }
            
            alpaca_timeframe = timeframe_map.get(timeframe, TimeFrame.Day)
            
            # Get bars from Alpaca with retry
            @retry(
                stop=stop_after_attempt(3),
                wait=wait_fixed(5),
                retry=retry_if_exception_type(Exception),
                reraise=True
            )
            def _get_bars():
                return self.api.get_bars(
                    symbol,
                    alpaca_timeframe,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    adjustment='raw'
                ).df
            
            bars = _get_bars()
            
            # Convert to list of dictionaries
            historical_data = []
            for index, row in bars.iterrows():
                historical_data.append({
                    'timestamp': index.isoformat(),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': int(row['volume'])
                })
            
            return historical_data
            
        except Exception as e:
            self.logger.error(f"Failed to get historical data for {symbol} after retries: {str(e)}")
            return None


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    import os
    
    async def test_alpaca_broker():
        """Test Alpaca broker functionality."""
        
        # Initialize with paper trading credentials
        broker = AlpacaBroker(
            api_key=os.getenv("ALPACA_API_KEY", "test_key"),
            secret_key=os.getenv("ALPACA_SECRET_KEY", "test_secret"),
            base_url="https://paper-api.alpaca.markets"
        )
        
        print(f"Broker: {broker}")
        
        # Test connection
        if await broker.connect():
            print("✓ Connected to Alpaca")
            
            # Test account info
            account = await broker.get_account_info()
            if account:
                print(f"✓ Account info: {account.to_dict()}")
            
            # Test market data
            market_data = await broker.get_market_data("AAPL")
            if market_data:
                print(f"✓ Market data: {market_data.to_dict()}")
            
            # Test positions
            positions = await broker.get_positions()
            print(f"✓ Positions: {positions}")
            
            await broker.disconnect()
        else:
            print("✗ Failed to connect to Alpaca")
    
    # Only run test if credentials are available
    if os.getenv("ALPACA_API_KEY"):
        asyncio.run(test_alpaca_broker())
    else:
        print("Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables to test")