"""
Response Models for FastAPI
Pydantic models for API responses
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum

class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    version: str = Field(..., description="API version")

class PortfolioResponse(BaseModel):
    total_value: float = Field(..., description="Total portfolio value")
    cash: float = Field(..., description="Available cash")
    positions: List[Dict[str, Any]] = Field(..., description="Current positions")
    daily_pnl: float = Field(..., description="Daily P&L")
    total_pnl: float = Field(..., description="Total P&L")

class PositionResponse(BaseModel):
    symbol: str = Field(..., description="Asset symbol")
    quantity: float = Field(..., description="Position quantity")
    avg_cost: float = Field(..., description="Average cost basis")
    current_price: float = Field(..., description="Current market price")
    market_value: float = Field(..., description="Current market value")
    unrealized_pnl: float = Field(..., description="Unrealized P&L")

class OrderResponse(BaseModel):
    order_id: str = Field(..., description="Order ID")
    symbol: str = Field(..., description="Asset symbol")
    side: OrderSide = Field(..., description="Order side")
    order_type: OrderType = Field(..., description="Order type")
    quantity: float = Field(..., description="Order quantity")
    price: Optional[float] = Field(None, description="Order price")
    status: OrderStatus = Field(..., description="Order status")
    timestamp: datetime = Field(..., description="Order timestamp")

class StrategyResponse(BaseModel):
    name: str = Field(..., description="Strategy name")
    status: str = Field(..., description="Strategy status")
    parameters: Dict[str, Any] = Field(..., description="Strategy parameters")
    performance: Dict[str, float] = Field(..., description="Performance metrics")

class BacktestResponse(BaseModel):
    strategy_name: str = Field(..., description="Strategy name")
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    total_return: float = Field(..., description="Total return")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    trades: int = Field(..., description="Number of trades")

class MarketDataResponse(BaseModel):
    symbol: str = Field(..., description="Asset symbol")
    price: float = Field(..., description="Current price")
    change: float = Field(..., description="Price change")
    change_percent: float = Field(..., description="Price change percentage")
    volume: Optional[int] = Field(None, description="Volume")
    timestamp: datetime = Field(..., description="Data timestamp")

class SystemMetricsResponse(BaseModel):
    cpu_usage: float = Field(..., description="CPU usage percentage")
    memory_usage: float = Field(..., description="Memory usage percentage")
    active_strategies: int = Field(..., description="Number of active strategies")
    open_positions: int = Field(..., description="Number of open positions")
    pending_orders: int = Field(..., description="Number of pending orders")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.now)