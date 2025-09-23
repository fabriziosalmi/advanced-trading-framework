"""
Watchlist Router
Handles watchlist management for tracking symbols
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from fastapi_app.models.responses import ErrorResponse

router = APIRouter()

# In-memory watchlist store (in production, use database)
_watchlist: List[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

@router.get("/")
async def get_watchlist():
    """Get current watchlist symbols."""
    try:
        # Get current prices for watchlist symbols
        watchlist_data = []
        for symbol in _watchlist:
            try:
                # Try to get real price (simplified - in production use market data service)
                price = 100.0  # Placeholder
                change = 0.0   # Placeholder
                watchlist_data.append({
                    "symbol": symbol,
                    "price": price,
                    "change": change,
                    "change_percent": 0.0
                })
            except Exception as e:
                # Fallback with basic data
                watchlist_data.append({
                    "symbol": symbol,
                    "price": 100.0,
                    "change": 0.0,
                    "change_percent": 0.0
                })

        return {"symbols": watchlist_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/add")
async def add_to_watchlist(symbol: str):
    """Add symbol to watchlist."""
    try:
        if symbol not in _watchlist:
            _watchlist.append(symbol)
        return {"message": f"Added {symbol} to watchlist"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{symbol}")
async def remove_from_watchlist(symbol: str):
    """Remove symbol from watchlist."""
    try:
        if symbol in _watchlist:
            _watchlist.remove(symbol)
        return {"message": f"Removed {symbol} from watchlist"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))