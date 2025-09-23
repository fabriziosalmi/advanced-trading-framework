"""
Dashboard Router
Provides dashboard summary data and real-time updates
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import sys
from pathlib import Path
import json
from datetime import datetime, timedelta
import asyncio

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from fastapi_app.models.responses import ErrorResponse
from fastapi_app.routers.portfolio import get_portfolio, get_real_time_price
from fastapi_app.routers.trading import _trading_state
from core.portfolio import Portfolio
from strategy_layer.ml_random_forest_strategy import MLRandomForestStrategy
from pydantic import BaseModel

router = APIRouter()

class DashboardSummary(BaseModel):
    portfolio_value: float
    daily_pnl: float
    active_strategies: int
    open_positions: int

@router.get("/summary", response_model=DashboardSummary)
async def get_dashboard_summary():
    """Get dashboard summary with key metrics for the 4 main cards."""
    try:
        # Import here to avoid circular imports
        from fastapi_app.routers.portfolio import get_portfolio, get_real_time_price
        
        portfolio = get_portfolio()
        
        # Calculate portfolio value
        positions_value = 0
        for symbol, position in portfolio.positions.items():
            current_price = get_real_time_price(symbol)
            positions_value += position.quantity * current_price

        portfolio_value = portfolio.current_cash + positions_value

        # Calculate daily P&L (simplified - in production would track from previous day)
        daily_pnl = 0.0  # Placeholder - would need historical tracking

        # Count active strategies (simplified - check if strategies are loaded)
        active_strategies = 1 if hasattr(portfolio, 'strategies') and portfolio.strategies else 0

        # Count open positions
        open_positions = len(portfolio.positions)

        return DashboardSummary(
            portfolio_value=round(portfolio_value, 2),
            daily_pnl=round(daily_pnl, 2),
            active_strategies=active_strategies,
            open_positions=open_positions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance/history")
async def get_performance_history(days: int = 30):
    """Get portfolio performance history for charting."""
    try:
        # In a real implementation, this would fetch from database
        # For now, return mock data
        history = []
        base_value = 100000.0

        for i in range(days):
            date = datetime.now() - timedelta(days=days-i-1)
            # Simulate some growth with random variation
            import random
            change = random.uniform(-0.02, 0.03)  # -2% to +3% daily
            base_value *= (1 + change)

            history.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": round(base_value, 2),
                "change": round(change * 100, 2)
            })

        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/portfolio/allocation")
async def get_portfolio_allocation():
    """Get portfolio allocation data for pie chart."""
    try:
        # Import here to avoid circular imports
        from fastapi_app.routers.portfolio import get_portfolio, get_real_time_price
        
        portfolio = get_portfolio()
        
        allocations = []
        total_value = portfolio.current_cash

        # Add cash allocation
        if portfolio.current_cash > 0:
            allocations.append({
                "asset": "Cash",
                "value": round(portfolio.current_cash, 2),
                "percentage": 0.0  # Will be calculated after total
            })

        # Add position allocations
        for symbol, position in portfolio.positions.items():
            current_price = get_real_time_price(symbol)
            value = position.quantity * current_price
            total_value += value

            allocations.append({
                "asset": symbol,
                "value": round(value, 2),
                "percentage": 0.0  # Will be calculated after total
            })

        # Calculate percentages
        for allocation in allocations:
            if total_value > 0:
                allocation["percentage"] = round((allocation["value"] / total_value) * 100, 2)

        return {"allocations": allocations, "total_value": round(total_value, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))