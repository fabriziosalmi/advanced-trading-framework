"""
Request Models for FastAPI
Pydantic models for API requests
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from datetime import datetime
from .responses import OrderType, OrderSide

class OrderRequest(BaseModel):
    symbol: str = Field(..., description="Asset symbol", min_length=1)
    side: OrderSide = Field(..., description="Order side")
    order_type: OrderType = Field(..., description="Order type")
    quantity: float = Field(..., description="Order quantity", gt=0)
    price: Optional[float] = Field(None, description="Order price", gt=0)

    @validator('price')
    def validate_price(cls, v, values):
        if values.get('order_type') in ['limit', 'stop_limit'] and v is None:
            raise ValueError('Price is required for limit and stop_limit orders')
        return v

class StrategyConfigRequest(BaseModel):
    name: str = Field(..., description="Strategy name")
    parameters: Dict[str, Any] = Field(..., description="Strategy parameters")
    symbols: List[str] = Field(..., description="Symbols to trade")
    enabled: bool = Field(True, description="Enable strategy")

class BacktestRequest(BaseModel):
    strategy_name: str = Field(..., description="Strategy name")
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    initial_capital: float = Field(100000, description="Initial capital", gt=0)
    symbols: List[str] = Field(..., description="Symbols to test")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Strategy parameters")

class PortfolioAllocationRequest(BaseModel):
    allocations: Dict[str, float] = Field(..., description="Asset allocations")

    @validator('allocations')
    def validate_allocations(cls, v):
        total = sum(v.values())
        if not 0.99 <= total <= 1.01:  # Allow small rounding errors
            raise ValueError('Allocations must sum to 1.0')
        return v