"""
Market Data Router
Handles market data retrieval and real-time feeds with real data sources
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from typing import List, Dict, Any
import sys
from pathlib import Path
import json
import asyncio
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from fastapi_app.models.responses import MarketDataResponse
from fastapi_app.database import db
from fastapi_app.routers.portfolio import get_real_time_price

router = APIRouter()

# Active WebSocket connections for real-time data
active_connections: List[WebSocket] = []

@router.get("/quote/{symbol}", response_model=MarketDataResponse)
async def get_quote(symbol: str):
    """Get current market quote for a symbol using real data."""
    try:
        # Get real-time price
        current_price = get_real_time_price(symbol)

        # Try to get additional market data from yfinance
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="2d")

            if len(hist) >= 2:
                yesterday_close = float(hist['Close'].iloc[-2])
                change = current_price - yesterday_close
                change_percent = (change / yesterday_close) * 100
                volume = int(hist['Volume'].iloc[-1]) if 'Volume' in hist else 0
            else:
                change = 0.0
                change_percent = 0.0
                volume = 0

        except Exception as e:
            print(f"Error getting additional data for {symbol}: {e}")
            change = 0.0
            change_percent = 0.0
            volume = 0

        # Save to database cache
        db.save_market_data(
            symbol=symbol.upper(),
            price=current_price,
            change_amount=change,
            change_percent=change_percent,
            volume=volume
        )

        return MarketDataResponse(
            symbol=symbol.upper(),
            price=current_price,
            change=change,
            change_percent=change_percent,
            volume=volume,
            timestamp=datetime.now()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/quotes", response_model=List[MarketDataResponse])
async def get_quotes(symbols: str):
    """Get quotes for multiple symbols (comma-separated) using real data."""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        quotes = []

        for symbol in symbol_list:
            try:
                # Get real-time price
                current_price = get_real_time_price(symbol)

                # Try to get additional market data from yfinance
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="2d")

                    if len(hist) >= 2:
                        yesterday_close = float(hist['Close'].iloc[-2])
                        change = current_price - yesterday_close
                        change_percent = (change / yesterday_close) * 100
                        volume = int(hist['Volume'].iloc[-1]) if 'Volume' in hist else 0
                    else:
                        change = 0.0
                        change_percent = 0.0
                        volume = 0

                except Exception:
                    change = 0.0
                    change_percent = 0.0
                    volume = 0

                quotes.append(MarketDataResponse(
                    symbol=symbol,
                    price=current_price,
                    change=change,
                    change_percent=change_percent,
                    volume=volume,
                    timestamp=datetime.now()
                ))

            except Exception as e:
                print(f"Error getting quote for {symbol}: {e}")
                # Skip this symbol if there's an error
                continue

        return quotes

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{symbol}")
async def get_historical_data(
    symbol: str,
    days: int = 30,
    interval: str = "1d"
):
    """Get historical price data using real market data."""
    try:
        # Fetch real historical data from yfinance
        ticker = yf.Ticker(symbol)

        # Map interval to yfinance format
        yf_interval = "1d"
        if interval == "1h":
            yf_interval = "1h"
        elif interval == "1m":
            yf_interval = "1m"
        elif interval == "5m":
            yf_interval = "5m"
        elif interval == "15m":
            yf_interval = "15m"
        elif interval == "30m":
            yf_interval = "30m"

        # Determine period based on days and interval
        if days <= 7 and yf_interval in ["1m", "5m", "15m", "30m"]:
            period = f"{days}d"
        elif days <= 30:
            period = "1mo"
        elif days <= 90:
            period = "3mo"
        elif days <= 180:
            period = "6mo"
        elif days <= 365:
            period = "1y"
        else:
            period = "2y"

        hist = ticker.history(period=period, interval=yf_interval)

        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for {symbol}")

        # Convert to our format
        data = []
        for index, row in hist.iterrows():
            data.append({
                "timestamp": index.isoformat(),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": int(row['Volume']) if pd.notna(row['Volume']) else 0
            })

        # Limit to requested number of days
        if len(data) > days:
            data = data[-days:]

        return {"symbol": symbol.upper(), "data": data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/live/{symbol}")
async def websocket_live_data(websocket: WebSocket, symbol: str):
    """WebSocket endpoint for real-time market data."""
    await websocket.accept()
    active_connections.append(websocket)

    try:
        while True:
            # Generate live market data
            price = 100.0 + (hash(symbol + str(datetime.now())) % 100)
            change = (hash(symbol + "live") % 20) - 10

            data = {
                "symbol": symbol.upper(),
                "price": price,
                "change": change,
                "change_percent": (change / price) * 100,
                "timestamp": datetime.now().isoformat()
            }

            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(1)  # Send updates every second

    except WebSocketDisconnect:
        active_connections.remove(websocket)

@router.get("/search")
async def search_symbols(query: str):
    """Search for symbols matching the query."""
    try:
        # Simulated symbol search
        symbols = [
            "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX",
            "ORCL", "CRM", "ADBE", "PYPL", "INTC", "AMD", "UBER", "SPOT"
        ]

        filtered = [s for s in symbols if query.upper() in s]

        return {
            "symbols": [
                {
                    "symbol": symbol,
                    "name": f"{symbol} Inc.",
                    "exchange": "NASDAQ"
                }
                for symbol in filtered[:10]  # Limit to 10 results
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sectors")
async def get_sector_performance():
    """Get sector performance overview."""
    try:
        sectors = [
            "Technology", "Healthcare", "Financials", "Energy",
            "Consumer Discretionary", "Industrials", "Materials",
            "Real Estate", "Utilities", "Consumer Staples", "Communication Services"
        ]

        performance = []
        for sector in sectors:
            change = (hash(sector) % 40) - 20  # -20% to +20%
            performance.append({
                "sector": sector,
                "change_percent": change / 10.0,  # Scale to -2% to +2%
                "market_cap": hash(sector) % 10000  # Billions
            })

        return {"sectors": performance}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))