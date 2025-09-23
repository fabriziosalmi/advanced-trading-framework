"""
FastAPI Main Application
Advanced Trading Framework - REST API Backend

Provides REST API endpoints for the trading framework with automatic OpenAPI documentation.
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi import WebSocket, WebSocketDisconnect
import uvicorn
import os
import sys
from pathlib import Path
from typing import List, Any
from datetime import datetime
import asyncio

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from fastapi_app.routers import (
    portfolio,
    trading,
    strategies,
    monitoring,
    backtesting,
    data,
    watchlist,
    dashboard
)
from fastapi_app.models.responses import HealthResponse
from core.config_validator import validate_configuration

# Global WebSocket connections for real-time updates
active_connections: List[WebSocket] = []

app = FastAPI(
    title="Advanced Trading Framework API",
    description="Professional trading framework with ML strategies, portfolio management, and real-time execution",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["Portfolio"])
app.include_router(trading.router, prefix="/api/trading", tags=["Trading"])
app.include_router(strategies.router, prefix="/api/strategies", tags=["Strategies"])
app.include_router(monitoring.router, prefix="/api/monitoring", tags=["Monitoring"])
app.include_router(backtesting.router, prefix="/api/backtesting", tags=["Backtesting"])
app.include_router(data.router, prefix="/api/data", tags=["Market Data"])
app.include_router(watchlist.router, prefix="/api/watchlist", tags=["Watchlist"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="fastapi_app/static"), name="static")
app.mount("/static/models", StaticFiles(directory="models"), name="models")

@app.websocket("/ws/updates")
async def websocket_updates(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates."""
    await websocket.accept()
    active_connections.append(websocket)

    try:
        # Send initial system status
        from fastapi_app.routers.trading import _trading_state
        await websocket.send_json({
            "event": "system_status",
            "data": "Running" if _trading_state.get("is_running", False) else "Stopped",
            "timestamp": datetime.now().isoformat()
        })

        # Keep connection alive and send periodic updates
        while True:
            await asyncio.sleep(5)  # Update every 5 seconds

            # Send system status updates
            await websocket.send_json({
                "event": "system_status",
                "data": "Running" if _trading_state.get("is_running", False) else "Stopped",
                "timestamp": datetime.now().isoformat()
            })

            # Send P&L updates (simplified - would track actual daily P&L)
            await websocket.send_json({
                "event": "pnl_update",
                "data": 0.0,  # Placeholder
                "timestamp": datetime.now().isoformat()
            })

    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        if websocket in active_connections:
            active_connections.remove(websocket)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main frontend application."""
    with open("fastapi_app/static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=None,
        version="2.0.0"
    )

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    print("🚀 Starting Advanced Trading Framework API...")

    # Validate configuration
    if not validate_configuration("config.yaml"):
        raise RuntimeError("Configuration validation failed")

    print("✅ Configuration validated")
    print("📡 API available at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/api/docs")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )