"""
Advanced Trading Framework - Paper Trading Broker

This module implements a paper trading broker for backtesting and simulation,
providing realistic order fills, slippage modeling, and comprehensive
trading simulation without risking real capital.

Author: Senior Python Software Architect
Version: 1.0.0
"""

import asyncio
import uuid
import random
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import json
import os
from collections import defaultdict

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except (ImportError, TypeError, Exception) as e:
    yf = None
    YFINANCE_AVAILABLE = False
    print(f"Warning: yfinance not available ({e}). Using simulated market data.")

from .base_broker import (
    BaseBroker, Order, OrderType, OrderSide, OrderStatus, OrderTimeInForce,
    AccountInfo, MarketData
)


class PaperBroker(BaseBroker):
    """
    Paper trading broker for simulation and backtesting.
    
    Provides realistic trading simulation with:
    - Realistic order fills with slippage
    - Market data from Yahoo Finance
    - Portfolio tracking and P&L calculation
    - Latency simulation
    - Commission and fee modeling
    - Market hours simulation
    
    Perfect for strategy testing and development without real money.
    """
    
    def __init__(
        self,
        initial_cash: float = 100000.0,
        commission_per_trade: float = 0.0,
        commission_per_share: float = 0.0,
        slippage_bps: float = 5.0,
        latency_ms: int = 100,
        state_file: str = "data/paper_broker_state.json"
    ):
        """
        Initialize paper trading broker.
        
        Args:
            initial_cash: Starting cash balance
            commission_per_trade: Fixed commission per trade
            commission_per_share: Commission per share traded
            slippage_bps: Slippage in basis points (1 bp = 0.01%)
            latency_ms: Simulated order latency in milliseconds
            state_file: Path to state persistence file
        """
        super().__init__("Paper")
        
        # Account settings
        self.initial_cash = initial_cash
        self.current_cash = initial_cash
        self.commission_per_trade = commission_per_trade
        self.commission_per_share = commission_per_share
        self.slippage_bps = slippage_bps
        self.latency_ms = latency_ms
        self.state_file = state_file
        
        # Trading state
        self.positions: Dict[str, float] = {}
        self.pending_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.trade_history: List[Dict] = []
        self.account_id = f"paper_{uuid.uuid4().hex[:8]}"
        
        # Market data cache
        self.market_data_cache: Dict[str, MarketData] = {}
        self.cache_expiry = 60  # seconds
        
        # Performance tracking
        self.total_trades = 0
        self.total_commission_paid = 0.0
        self.created_at = datetime.now()
        
        # Load existing state if available
        self._load_state()
        
        self.logger.info(f"PaperBroker initialized with ${initial_cash:,.2f}")
    
    async def connect(self) -> bool:
        """
        Establish connection (always successful for paper trading).
        
        Returns:
            True (always successful)
        """
        if yf is None:
            self.logger.warning(
                "yfinance package not found. Market data will be simulated. "
                "Install with: pip install yfinance"
            )
        
        self.connected = True
        self.logger.info("Paper broker connected")
        return True
    
    async def disconnect(self) -> None:
        """Disconnect and save state."""
        self._save_state()
        self.connected = False
        self.logger.info("Paper broker disconnected")
    
    async def is_connected(self) -> bool:
        """
        Check if broker is connected.
        
        Returns:
            True if connected, False otherwise
        """
        return self.connected
    
    async def get_account_info(self) -> Optional[AccountInfo]:
        """
        Get current account information.
        
        Returns:
            AccountInfo object with current portfolio state
        """
        if not self.connected:
            return None
        
        # Calculate portfolio value
        portfolio_value = self.current_cash
        
        # Add position values
        for symbol, quantity in self.positions.items():
            market_data = await self.get_market_data(symbol)
            if market_data:
                portfolio_value += abs(quantity) * market_data.price
        
        return AccountInfo(
            account_id=self.account_id,
            buying_power=self.current_cash,  # Simplified: cash = buying power
            cash=self.current_cash,
            portfolio_value=portfolio_value,
            equity=portfolio_value,
            day_trade_count=0,  # Not tracked in paper trading
            pattern_day_trader=False
        )
    
    def _calculate_commission(self, quantity: float) -> float:
        """
        Calculate commission for a trade.
        
        Args:
            quantity: Number of shares traded
            
        Returns:
            Total commission amount
        """
        return self.commission_per_trade + (abs(quantity) * self.commission_per_share)
    
    def _apply_slippage(self, price: float, side: OrderSide, order_type: OrderType) -> float:
        """
        Apply realistic slippage to order execution price.
        
        Args:
            price: Base execution price
            side: Order side (buy/sell)
            order_type: Type of order
            
        Returns:
            Price with slippage applied
        """
        if order_type == OrderType.MARKET:
            # Market orders have more slippage
            slippage_factor = self.slippage_bps / 10000.0  # Convert bps to decimal
            
            # Add random component to slippage
            random_factor = random.uniform(0.5, 1.5)
            slippage = price * slippage_factor * random_factor
            
            if side == OrderSide.BUY:
                return price + slippage
            else:
                return price - slippage
        
        # Limit orders have minimal slippage (only when crossing the spread)
        return price
    
    async def _simulate_latency(self) -> None:
        """Simulate order processing latency."""
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000.0)
    
    async def submit_order(self, order: Order) -> Optional[str]:
        """
        Submit an order for paper trading execution.
        
        Args:
            order: Order object to submit
            
        Returns:
            Order ID if successful, None otherwise
        """
        if not self.connected:
            return None
        
        try:
            # Generate unique order ID
            order_id = str(uuid.uuid4())
            order.order_id = order_id
            order.status = OrderStatus.PENDING
            
            # Simulate latency
            await self._simulate_latency()
            
            # Get current market data
            market_data = await self.get_market_data(order.symbol)
            if not market_data:
                self.logger.error(f"Cannot get market data for {order.symbol}")
                order.status = OrderStatus.REJECTED
                return None
            
            # Check if we can execute immediately
            execution_price = None
            
            if order.order_type == OrderType.MARKET:
                # Market orders execute immediately at current price with slippage
                execution_price = self._apply_slippage(
                    market_data.price, order.side, order.order_type
                )
                
            elif order.order_type == OrderType.LIMIT:
                # Check if limit order can be filled immediately
                if order.side == OrderSide.BUY and order.limit_price >= market_data.price:
                    execution_price = min(order.limit_price, market_data.price)
                elif order.side == OrderSide.SELL and order.limit_price <= market_data.price:
                    execution_price = max(order.limit_price, market_data.price)
            
            if execution_price:
                # Execute order immediately
                success = await self._execute_order(order, execution_price)
                if success:
                    self.logger.info(f"Order executed: {order}")
                    return order_id
                else:
                    return None
            else:
                # Add to pending orders
                order.status = OrderStatus.SUBMITTED
                self.pending_orders[order_id] = order
                self.cache_order(order)
                self.logger.info(f"Order submitted: {order}")
                return order_id
                
        except Exception as e:
            self.logger.error(f"Failed to submit order: {str(e)}")
            return None
    
    async def _execute_order(self, order: Order, execution_price: float) -> bool:
        """
        Execute an order and update portfolio state.
        
        Args:
            order: Order to execute
            execution_price: Price at which to execute
            
        Returns:
            True if execution successful, False otherwise
        """
        try:
            # Simulate partial fills for large orders (more realistic simulation)
            max_fill_percentage = 1.0  # Default to full fill
            
            # For large orders (>100 shares), simulate partial fills
            if order.quantity > 100:
                max_fill_percentage = random.uniform(0.3, 1.0)  # Fill 30-100% of order
            
            filled_quantity = int(order.quantity * max_fill_percentage)
            
            # Ensure at least 1 share is filled if order is large enough
            if filled_quantity == 0 and order.quantity > 0:
                filled_quantity = 1
            
            # Calculate trade value and commission for filled portion
            trade_value = filled_quantity * execution_price
            commission = self._calculate_commission(filled_quantity)
            
            # Check cash requirements for buy orders (for filled portion)
            if order.side == OrderSide.BUY:
                total_cost = trade_value + commission
                if total_cost > self.current_cash:
                    # Cannot even fill partially, reject order
                    self.logger.warning(
                        f"Insufficient cash for order. Required: ${total_cost:.2f}, "
                        f"Available: ${self.current_cash:.2f}"
                    )
                    order.status = OrderStatus.REJECTED
                    return False
                
                # Deduct cash and add position
                self.current_cash -= total_cost
                self.positions[order.symbol] = self.positions.get(order.symbol, 0) + filled_quantity
                
            else:  # SELL
                # Check if we have enough shares to sell (for filled portion)
                current_position = self.positions.get(order.symbol, 0)
                if current_position < filled_quantity:
                    # Cannot fill even partially, reject order
                    self.logger.warning(
                        f"Insufficient shares to sell. Required: {filled_quantity}, "
                        f"Available: {current_position}"
                    )
                    order.status = OrderStatus.REJECTED
                    return False
                
                # Add cash and reduce position
                self.current_cash += trade_value - commission
                self.positions[order.symbol] -= filled_quantity
                
                # Remove position if quantity is zero
                if self.positions[order.symbol] == 0:
                    del self.positions[order.symbol]
            
            # Update order status based on fill
            if filled_quantity == order.quantity:
                order.status = OrderStatus.FILLED
                order.filled_quantity = filled_quantity
                order.avg_fill_price = execution_price
                order.updated_at = datetime.now()
            else:
                # Partial fill - create new order for remaining quantity
                order.status = OrderStatus.PARTIALLY_FILLED
                order.filled_quantity = filled_quantity
                order.avg_fill_price = execution_price
                order.updated_at = datetime.now()

                # Create new order for remaining quantity
                remaining_quantity = order.quantity - filled_quantity
                remaining_order = Order(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=remaining_quantity,
                    order_type=order.order_type,
                    limit_price=order.limit_price,
                    stop_price=order.stop_price
                )
                remaining_order.status = OrderStatus.SUBMITTED
                remaining_order_id = f"{order.order_id}_remaining_{remaining_quantity}"

                # Add remaining order to pending orders
                self.pending_orders[remaining_order_id] = remaining_order
                self.cache_order(remaining_order)

                self.logger.info(f"Created remaining order {remaining_order_id} for {remaining_quantity} shares")
            
            # Track performance metrics
            self.total_trades += 1
            self.total_commission_paid += commission
            
            # Record trade history
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': order.symbol,
                'side': order.side.value,
                'quantity': order.quantity,
                'price': execution_price,
                'commission': commission,
                'trade_value': trade_value,
                'order_id': order.order_id
            }
            self.trade_history.append(trade_record)
            
            # Move from pending to history
            if order.order_id in self.pending_orders:
                del self.pending_orders[order.order_id]
            self.order_history.append(order)
            
            # Save state
            self._save_state()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to execute order: {str(e)}")
            order.status = OrderStatus.REJECTED
            return False
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation successful, False otherwise
        """
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()
            
            # Move to history
            del self.pending_orders[order_id]
            self.order_history.append(order)
            
            self.logger.info(f"Order cancelled: {order_id}")
            self._save_state()
            return True
        
        return False
    
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """
        Get current status of an order.
        
        Args:
            order_id: Order ID to query
            
        Returns:
            Order object if found, None otherwise
        """
        # Check pending orders
        if order_id in self.pending_orders:
            return self.pending_orders[order_id]
        
        # Check order history
        for order in self.order_history:
            if order.order_id == order_id:
                return order
        
        # Check cached orders
        return self.get_cached_order(order_id)
    
    async def get_open_orders(self) -> List[Order]:
        """
        Get all open (pending) orders.
        
        Returns:
            List of pending Order objects
        """
        return list(self.pending_orders.values())
    
    async def get_positions(self) -> Dict[str, float]:
        """
        Get current positions.
        
        Returns:
            Dictionary mapping symbols to quantities
        """
        return self.positions.copy()
    
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """
        Get current market data for a symbol.
        
        Args:
            symbol: Stock symbol to query
            
        Returns:
            MarketData object if successful, None otherwise
        """
        try:
            # Check cache first
            if symbol in self.market_data_cache:
                cached_data = self.market_data_cache[symbol]
                age = (datetime.now() - cached_data.timestamp).total_seconds()
                if age < self.cache_expiry:
                    return cached_data
            
            # Get fresh data
            if yf:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    hist = ticker.history(period="1d", interval="1m").tail(1)
                    
                    if not hist.empty:
                        price = float(hist['Close'].iloc[-1])
                        volume = int(hist['Volume'].iloc[-1])
                        
                        # Simulate bid/ask spread
                        spread_pct = 0.001  # 0.1% spread
                        spread = price * spread_pct
                        bid = price - spread / 2
                        ask = price + spread / 2
                        
                        market_data = MarketData(
                            symbol=symbol,
                            price=price,
                            bid=bid,
                            ask=ask,
                            volume=volume,
                            timestamp=datetime.now()
                        )
                        
                        # Cache the data
                        self.market_data_cache[symbol] = market_data
                        return market_data
                except Exception as e:
                    self.logger.warning(f"Failed to fetch market data for {symbol} from yfinance: {str(e)}")
                    # Fall through to fallback logic
                    return market_data
            
            # Fallback: simulate market data
            self.logger.warning(f"Simulating market data for {symbol}")
            simulated_price = 100.0 + random.uniform(-10, 10)
            spread = simulated_price * 0.001
            
            market_data = MarketData(
                symbol=symbol,
                price=simulated_price,
                bid=simulated_price - spread / 2,
                ask=simulated_price + spread / 2,
                volume=random.randint(10000, 100000),
                timestamp=datetime.now()
            )
            
            self.market_data_cache[symbol] = market_data
            return market_data
            
        except Exception as e:
            self.logger.error(f"Failed to get market data for {symbol}: {str(e)}")
            return None
    
    async def get_market_data_batch(self, symbols: List[str]) -> Dict[str, MarketData]:
        """
        Get market data for multiple symbols.
        
        Args:
            symbols: List of stock symbols to query
            
        Returns:
            Dictionary mapping symbols to MarketData objects
        """
        market_data = {}
        
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
        Get historical price data for a symbol.
        
        Args:
            symbol: Stock symbol to query
            start_date: Start date for historical data
            end_date: End date for historical data
            timeframe: Data timeframe (1D, 1H, etc.)
            
        Returns:
            List of OHLCV dictionaries if successful, None otherwise
        """
        try:
            if not yf:
                self.logger.warning("yfinance not available, cannot get historical data")
                return None
            
            # Map timeframe to yfinance interval
            interval_map = {
                "1M": "1m",
                "5M": "5m",
                "15M": "15m",
                "1H": "1h",
                "1D": "1d"
            }
            
            interval = interval_map.get(timeframe, "1d")
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval=interval
            )
            
            if hist.empty:
                return None
            
            historical_data = []
            for index, row in hist.iterrows():
                historical_data.append({
                    'timestamp': index.isoformat(),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                })
            
            return historical_data
            
        except Exception as e:
            self.logger.error(f"Failed to get historical data for {symbol}: {str(e)}")
            return None
    
    def get_performance_summary(self) -> Dict:
        """
        Get trading performance summary.
        
        Returns:
            Dictionary with performance metrics
        """
        account_info = asyncio.create_task(self.get_account_info())
        
        # Calculate returns
        current_value = account_info.result().portfolio_value if account_info.done() else self.current_cash
        total_return = current_value - self.initial_cash
        return_pct = (total_return / self.initial_cash) * 100.0
        
        return {
            'initial_cash': self.initial_cash,
            'current_value': current_value,
            'total_return': total_return,
            'return_pct': return_pct,
            'total_trades': self.total_trades,
            'total_commission': self.total_commission_paid,
            'active_positions': len(self.positions),
            'pending_orders': len(self.pending_orders),
            'days_active': (datetime.now() - self.created_at).days
        }
    
    def _save_state(self) -> None:
        """Save broker state to file."""
        try:
            state_data = {
                'account_id': self.account_id,
                'initial_cash': self.initial_cash,
                'current_cash': self.current_cash,
                'positions': self.positions,
                'pending_orders': {oid: order.to_dict() for oid, order in self.pending_orders.items()},
                'trade_history': self.trade_history,
                'total_trades': self.total_trades,
                'total_commission_paid': self.total_commission_paid,
                'created_at': self.created_at.isoformat(),
                'last_saved': datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save state: {str(e)}")
    
    def _load_state(self) -> None:
        """Load broker state from file."""
        try:
            if not os.path.exists(self.state_file):
                return
            
            with open(self.state_file, 'r') as f:
                state_data = json.load(f)
            
            self.account_id = state_data.get('account_id', self.account_id)
            self.current_cash = state_data.get('current_cash', self.initial_cash)
            self.positions = state_data.get('positions', {})
            self.trade_history = state_data.get('trade_history', [])
            self.total_trades = state_data.get('total_trades', 0)
            self.total_commission_paid = state_data.get('total_commission_paid', 0.0)
            
            # Restore pending orders
            pending_orders_data = state_data.get('pending_orders', {})
            for order_id, order_data in pending_orders_data.items():
                # Recreate Order objects from saved data
                # This is simplified - in production you'd want proper deserialization
                pass
            
            if 'created_at' in state_data:
                self.created_at = datetime.fromisoformat(state_data['created_at'])
            
            self.logger.info(f"State loaded: ${self.current_cash:.2f} cash, {len(self.positions)} positions")
            
        except Exception as e:
            self.logger.error(f"Failed to load state: {str(e)}")


# Example usage and testing
if __name__ == "__main__":
    async def test_paper_broker():
        """Test paper broker functionality."""
        
        broker = PaperBroker(initial_cash=100000.0, slippage_bps=5.0)
        
        print(f"Broker: {broker}")
        
        # Connect
        await broker.connect()
        print("✓ Connected to paper broker")
        
        # Get account info
        account = await broker.get_account_info()
        print(f"✓ Account: {account.to_dict()}")
        
        # Get market data
        market_data = await broker.get_market_data("AAPL")
        if market_data:
            print(f"✓ Market data: {market_data.to_dict()}")
        
        # Submit a market buy order
        order_id = await broker.buy_market("AAPL", 100)
        if order_id:
            print(f"✓ Market buy order submitted: {order_id}")
            
            # Check order status
            order = await broker.get_order_status(order_id)
            if order:
                print(f"✓ Order status: {order}")
        
        # Check positions
        positions = await broker.get_positions()
        print(f"✓ Positions: {positions}")
        
        # Performance summary
        performance = broker.get_performance_summary()
        print(f"✓ Performance: {performance}")
        
        await broker.disconnect()
    
    asyncio.run(test_paper_broker())