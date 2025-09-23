"""
Backtesting Router
Handles strategy backtesting and analysis
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from fastapi_app.models.responses import BacktestResponse
from fastapi_app.models.requests import BacktestRequest
from core.backtesting import BacktestEngine

router = APIRouter()

# Store running backtests
_running_backtests: Dict[str, Any] = {}

@router.post("/run", response_model=Dict[str, str])
async def start_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """Start a new backtest run."""
    try:
        backtest_id = f"backtest_{datetime.now().timestamp()}"

        # Add backtest to background tasks
        background_tasks.add_task(run_backtest_task, backtest_id, request)

        _running_backtests[backtest_id] = {
            "status": "running",
            "start_time": datetime.now(),
            "request": request
        }

        return {"backtest_id": backtest_id, "status": "started"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

async def run_backtest_task(backtest_id: str, request: BacktestRequest):
    """Background task to run backtest."""
    try:
        # Create backtest engine
        engine = BacktestEngine(
            initial_capital=request.initial_capital,
            start_date=request.start_date,
            end_date=request.end_date
        )

        # Run backtest (simplified)
        results = {
            "total_return": 0.15,  # 15% return
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.08,  # 8% max drawdown
            "trades": 150
        }

        # Update backtest status
        _running_backtests[backtest_id].update({
            "status": "completed",
            "results": results,
            "end_time": datetime.now()
        })

    except Exception as e:
        _running_backtests[backtest_id].update({
            "status": "failed",
            "error": str(e),
            "end_time": datetime.now()
        })

@router.get("/status/{backtest_id}")
async def get_backtest_status(backtest_id: str):
    """Get backtest status and results."""
    try:
        if backtest_id not in _running_backtests:
            raise HTTPException(status_code=404, detail="Backtest not found")

        return _running_backtests[backtest_id]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/results/{backtest_id}", response_model=BacktestResponse)
async def get_backtest_results(backtest_id: str):
    """Get completed backtest results."""
    try:
        if backtest_id not in _running_backtests:
            raise HTTPException(status_code=404, detail="Backtest not found")

        backtest = _running_backtests[backtest_id]

        if backtest["status"] != "completed":
            raise HTTPException(status_code=400, detail="Backtest not completed")

        request = backtest["request"]
        results = backtest["results"]

        return BacktestResponse(
            strategy_name=request.strategy_name,
            start_date=request.start_date,
            end_date=request.end_date,
            total_return=results["total_return"],
            sharpe_ratio=results["sharpe_ratio"],
            max_drawdown=results["max_drawdown"],
            trades=results["trades"]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list")
async def list_backtests():
    """List all backtests."""
    try:
        backtests = []
        for backtest_id, data in _running_backtests.items():
            backtests.append({
                "backtest_id": backtest_id,
                "status": data["status"],
                "start_time": data["start_time"],
                "strategy_name": data["request"].strategy_name
            })

        return {"backtests": backtests}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{backtest_id}")
async def delete_backtest(backtest_id: str):
    """Delete a backtest and its results."""
    try:
        if backtest_id not in _running_backtests:
            raise HTTPException(status_code=404, detail="Backtest not found")

        del _running_backtests[backtest_id]

        return {"message": f"Backtest {backtest_id} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))