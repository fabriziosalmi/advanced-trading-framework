"""
Trading Router
Handles trading operations and order management with real data integration
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List
import sys
from pathlib import Path
from datetime import datetime
import uuid

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from fastapi_app.models.responses import OrderResponse, ErrorResponse
from fastapi_app.models.requests import OrderRequest
from fastapi_app.database import db
from execution_layer.paper_broker import PaperBroker
from core.portfolio import Portfolio
from strategy_layer.signals import TradingSignal
from pydantic import BaseModel

# Import shared portfolio instance
from fastapi_app.routers.portfolio import get_portfolio, get_real_time_price

# Request models for trading controls
class StartTradingRequest(BaseModel):
    mode: str = "manual"
    risk_settings: dict = None

class RiskSettings(BaseModel):
    max_daily_loss: float = 1000
    position_size: float = 1000

router = APIRouter()

@router.post("/orders", response_model=OrderResponse)
async def place_order(
    order: OrderRequest,
    portfolio: Portfolio = Depends(get_portfolio)
):
    """Place a new trading order with real market data and database persistence."""
    try:
        # Generate unique order ID
        order_id = str(uuid.uuid4())

        # Get current market price if no price specified
        execution_price = order.price
        if execution_price is None:
            execution_price = get_real_time_price(order.symbol)

        # Create trading signal from order request
        signal = TradingSignal(
            ticker=order.symbol,
            action="BUY" if order.side.value == "buy" else "SELL",
            confidence=1.0,
            price=execution_price
        )

        # Save order to database first
        order_data = {
            "order_id": order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "order_type": order.order_type.value,
            "quantity": order.quantity,
            "price": execution_price,
            "status": "pending",
            "timestamp": datetime.now(),
            "filled_quantity": 0,
            "filled_price": None
        }
        db.save_order(order_data)

        # Execute order through portfolio
        try:
            if order.side.value == "buy":
                success = portfolio.open_position(
                    ticker=order.symbol,
                    quantity=order.quantity,
                    entry_price=execution_price,
                    side="LONG"
                )
            else:
                # For sell orders, we need to close existing position or short
                existing_position = portfolio.get_position(order.symbol)
                if existing_position:
                    # Close existing long position
                    success = portfolio.close_position(order.symbol, execution_price) > 0
                else:
                    # Open short position
                    success = portfolio.open_position(
                        ticker=order.symbol,
                        quantity=order.quantity,
                        entry_price=execution_price,
                        side="SHORT"
                    )

            if not success:
                raise Exception(f"Failed to execute {order.side.value} order for {order.symbol}")

            # Update order status to filled
            order_data.update({
                "status": "filled",
                "filled_quantity": order.quantity,
                "filled_price": execution_price
            })
            db.save_order(order_data)

            # Update position in database
            if order.symbol in portfolio.positions:
                position = portfolio.positions[order.symbol]
                db.save_position(order.symbol, position.quantity, position.entry_price)

        except Exception as e:
            # Update order status to failed
            order_data.update({"status": "failed"})
            db.save_order(order_data)
            raise e

        return OrderResponse(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side.value,
            order_type=order.order_type.value,
            quantity=order.quantity,
            price=execution_price,
            status="filled",
            timestamp=datetime.now()
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/orders", response_model=List[OrderResponse])
async def get_orders(limit: int = 100):
    """Get all orders (historical and active) from database."""
    try:
        orders = db.get_orders(limit)
        return [
            OrderResponse(
                order_id=order["order_id"],
                symbol=order["symbol"],
                side=order["side"],
                order_type=order["order_type"],
                quantity=order["quantity"],
                price=order["price"],
                status=order["status"],
                timestamp=datetime.fromisoformat(order["timestamp"]) if isinstance(order["timestamp"], str) else order["timestamp"]
            )
            for order in orders
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/orders/{order_id}", response_model=OrderResponse)
async def get_order(order_id: str):
    """Get specific order by ID from database."""
    try:
        orders = db.get_orders(limit=1000)  # Get more orders to search
        for order in orders:
            if order["order_id"] == order_id:
                return OrderResponse(
                    order_id=order["order_id"],
                    symbol=order["symbol"],
                    side=order["side"],
                    order_type=order["order_type"],
                    quantity=order["quantity"],
                    price=order["price"],
                    status=order["status"],
                    timestamp=datetime.fromisoformat(order["timestamp"]) if isinstance(order["timestamp"], str) else order["timestamp"]
                )
        raise HTTPException(status_code=404, detail="Order not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/orders/{order_id}")
async def cancel_order(order_id: str):
    """Cancel an active order."""
    try:
        # Find the order and update its status
        orders = db.get_orders(limit=1000)
        for order in orders:
            if order["order_id"] == order_id:
                if order["status"] in ["pending", "partial"]:
                    # Update order status to cancelled
                    order_data = dict(order)
                    order_data["status"] = "cancelled"
                    db.save_order(order_data)
                    return {"message": f"Order {order_id} cancelled"}
                else:
                    raise HTTPException(status_code=400, detail=f"Cannot cancel order with status: {order['status']}")
        raise HTTPException(status_code=404, detail="Order not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/positions", response_model=List[dict])
async def get_trading_positions(portfolio: Portfolio = Depends(get_portfolio)):
    """Get current trading positions with real market data."""
    try:
        positions = []
        for symbol, position in portfolio.positions.items():
            current_price = get_real_time_price(symbol)
            positions.append({
                "symbol": symbol,
                "quantity": position.quantity,
                "avg_cost": position.entry_price,
                "current_price": current_price,
                "current_value": position.quantity * current_price,
                "unrealized_pnl": (current_price - position.entry_price) * position.quantity
            })
        return positions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Global trading state (in production, this should be in a database or cache)
_trading_state = {
    "is_running": False,
    "mode": "manual",  # "manual" or "automatic"
    "risk_settings": {
        "max_daily_loss": 1000,
        "position_size": 1000
    }
}

@router.post("/start")
async def start_trading(request: StartTradingRequest):
    """Start trading with specified mode and risk settings."""
    try:
        global _trading_state

        _trading_state["is_running"] = True
        _trading_state["mode"] = request.mode

        if request.risk_settings:
            _trading_state["risk_settings"].update(request.risk_settings)

        return {
            "message": f"Trading started in {request.mode} mode",
            "state": _trading_state
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop")
async def stop_trading():
    """Stop trading operations."""
    try:
        global _trading_state

        _trading_state["is_running"] = False

        return {
            "message": "Trading stopped",
            "state": _trading_state
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/emergency-stop")
async def emergency_stop(portfolio: Portfolio = Depends(get_portfolio)):
    """Emergency stop - liquidate all positions immediately."""
    try:
        global _trading_state

        _trading_state["is_running"] = False

        # Emergency liquidation logic
        liquidated_positions = []
        total_loss = 0

        for symbol, position in list(portfolio.positions.items()):
            current_price = get_real_time_price(symbol)
            if position.quantity > 0:  # Long position
                loss = (position.entry_price - current_price) * position.quantity
            else:  # Short position
                loss = (current_price - position.entry_price) * abs(position.quantity)

            # Close position at market price
            portfolio.close_position(symbol, current_price)
            liquidated_positions.append({
                "symbol": symbol,
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "exit_price": current_price,
                "loss": loss
            })
            total_loss += loss

        return {
            "message": "Emergency stop executed",
            "liquidated_positions": liquidated_positions,
            "total_loss": total_loss,
            "state": _trading_state
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_trading_status():
    """Get current trading status and settings."""
    try:
        global _trading_state
        return _trading_state
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))