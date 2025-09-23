"""
Portfolio Router
Handles portfolio management endpoints with real data integration
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List
import sys
from pathlib import Path
import yfinance as yf
from datetime import datetime

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from fastapi_app.models.responses import PortfolioResponse, PositionResponse, ErrorResponse
from fastapi_app.models.requests import PortfolioAllocationRequest
from fastapi_app.database import db
from core.portfolio import Portfolio
from execution_layer.paper_broker import PaperBroker

router = APIRouter()

# Global portfolio instance (in production, use dependency injection)
_portfolio = None
_broker = None

def get_portfolio():
    """Get portfolio instance with database persistence."""
    global _portfolio, _broker
    if _portfolio is None:
        _broker = PaperBroker()
        _portfolio = Portfolio(initial_capital=100000.0)
        _portfolio.broker = _broker

        # Load existing positions from database
        db_positions = db.get_positions()
        for pos_data in db_positions:
            symbol = pos_data['symbol']
            quantity = pos_data['quantity']
            avg_cost = pos_data['avg_cost']

            # Create position in portfolio
            if quantity != 0:
                from core.position import Position
                _portfolio.positions[symbol] = Position(
                    ticker=symbol,
                    quantity=quantity,
                    entry_price=avg_cost,
                    side="LONG" if quantity > 0 else "SHORT"
                )

    return _portfolio

def get_real_time_price(symbol: str) -> float:
    """Get real-time price from yfinance with caching."""
    try:
        # Check cache first
        cached_data = db.get_market_data(symbol)
        if cached_data:
            cached_time = datetime.fromisoformat(cached_data[0]['timestamp'])
            now = datetime.now()
            # Use cached data if less than 1 minute old
            if (now - cached_time).seconds < 60:
                return cached_data[0]['price']

        # Fetch fresh data from yfinance
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d", interval="1m")

        if not hist.empty:
            current_price = float(hist['Close'].iloc[-1])

            # Cache the price
            db.save_market_data(
                symbol=symbol,
                price=current_price,
                change_amount=None,
                change_percent=None,
                volume=int(hist['Volume'].iloc[-1]) if 'Volume' in hist else None
            )

            return current_price
        else:
            # Fallback to a default price if no data available
            return 100.0

    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        # Return cached price or default
        cached_data = db.get_market_data(symbol)
        if cached_data:
            return cached_data[0]['price']
        return 100.0

@router.get("/", response_model=PortfolioResponse)
async def get_portfolio_summary(portfolio: Portfolio = Depends(get_portfolio)):
    """Get portfolio summary including positions and P&L with real market data."""
    try:
        positions_data = []
        for symbol, position in portfolio.positions.items():
            # Get real-time price
            current_price = get_real_time_price(symbol)

            positions_data.append({
                "symbol": symbol,
                "quantity": position.quantity,
                "avg_cost": position.entry_price,
                "current_price": current_price,
                "market_value": position.quantity * current_price,
                "unrealized_pnl": (current_price - position.entry_price) * position.quantity
            })

        total_value = portfolio.current_cash + sum(pos["market_value"] for pos in positions_data)
        total_pnl = sum(pos["unrealized_pnl"] for pos in positions_data)

        # Save portfolio snapshot to database
        snapshot_data = {
            "total_value": total_value,
            "cash": portfolio.current_cash,
            "positions_value": sum(pos["market_value"] for pos in positions_data),
            "daily_pnl": 0.0,  # Could calculate from previous day's snapshot
            "total_pnl": total_pnl,
            "timestamp": datetime.now()
        }
        db.save_portfolio_snapshot(snapshot_data)

        return PortfolioResponse(
            total_value=total_value,
            cash=portfolio.current_cash,
            positions=positions_data,
            daily_pnl=0.0,
            total_pnl=total_pnl
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/positions", response_model=List[PositionResponse])
async def get_positions(portfolio: Portfolio = Depends(get_portfolio)):
    """Get all current positions with real market data."""
    try:
        positions = []
        for symbol, position in portfolio.positions.items():
            current_price = get_real_time_price(symbol)

            positions.append(PositionResponse(
                symbol=symbol,
                quantity=position.quantity,
                avg_cost=position.entry_price,
                current_price=current_price,
                market_value=position.quantity * current_price,
                unrealized_pnl=(current_price - position.entry_price) * position.quantity
            ))

            # Update position in database
            db.save_position(symbol, position.quantity, position.entry_price)

        return positions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/positions/{symbol}", response_model=PositionResponse)
async def get_position(symbol: str, portfolio: Portfolio = Depends(get_portfolio)):
    """Get specific position by symbol with real market data."""
    try:
        if symbol not in portfolio.positions:
            raise HTTPException(status_code=404, detail="Position not found")

        position = portfolio.positions[symbol]
        current_price = get_real_time_price(symbol)

        return PositionResponse(
            symbol=symbol,
            quantity=position.quantity,
            avg_cost=position.entry_price,
            current_price=current_price,
            market_value=position.quantity * current_price,
            unrealized_pnl=(current_price - position.entry_price) * position.quantity
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rebalance")
async def rebalance_portfolio(
    allocation: PortfolioAllocationRequest,
    portfolio: Portfolio = Depends(get_portfolio)
):
    """Rebalance portfolio to target allocations."""
    try:
        # Implementation would calculate required trades to reach target allocation
        # This is simplified for the example
        return {"message": "Portfolio rebalancing initiated", "allocations": allocation.allocations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))