"""
Strategies Router
Handles ML strategy management and configuration with database persistence
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from fastapi_app.models.responses import StrategyResponse, ErrorResponse
from fastapi_app.models.requests import StrategyConfigRequest
from fastapi_app.database import db
from strategy_layer.ml_random_forest_strategy import MLRandomForestStrategy
from strategy_layer.lgbm_strategy import MLLGBMStrategy

router = APIRouter()

# Global strategies store (in production, use proper state management)
_strategies: Dict[str, Any] = {}

def load_strategies_from_db():
    """Load strategies from database into memory."""
    global _strategies
    db_strategies = db.get_strategies()
    for strategy_data in db_strategies:
        name = strategy_data['name']
        strategy_type = strategy_data['strategy_type']
        parameters = strategy_data['parameters']

        try:
            # Recreate strategy instances
            if strategy_type == "ml_random_forest":
                strategy = MLRandomForestStrategy(parameters)
            elif strategy_type == "lgbm":
                strategy = MLLGBMStrategy(parameters)
            else:
                print(f"Unknown strategy type: {strategy_type}")
                continue

            _strategies[name] = strategy
        except Exception as e:
            print(f"Error loading strategy {name}: {e}")

@router.get("/", response_model=List[StrategyResponse])
async def get_strategies():
    """Get all configured strategies from database."""
    try:
        # Load strategies from database if not already loaded
        if not _strategies:
            load_strategies_from_db()

        # Get strategies from database for fresh status
        db_strategies = db.get_strategies()
        strategies = []

        for strategy_data in db_strategies:
            # Get strategy performance metrics (simplified for now)
            performance = {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0
            }

            strategies.append(StrategyResponse(
                name=strategy_data['name'],
                status=strategy_data['status'],
                parameters=strategy_data['parameters'],
                performance=performance
            ))

        return strategies
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/", response_model=StrategyResponse)
async def create_strategy(config: StrategyConfigRequest):
    """Create and configure a new trading strategy with database persistence."""
    try:
        # Check if strategy already exists in database
        existing_strategies = db.get_strategies()
        for existing in existing_strategies:
            if existing['name'] == config.name:
                raise HTTPException(status_code=400, detail="Strategy already exists")

        # Determine strategy type from name or parameters
        strategy_type = config.parameters.get('strategy_type', 'ml_random_forest')
        if config.name.startswith("ml_random_forest") or strategy_type == "ml_random_forest":
            strategy_type = "ml_random_forest"
            strategy = MLRandomForestStrategy(config.parameters)
        elif config.name.startswith("lgbm") or strategy_type == "lgbm":
            strategy_type = "lgbm"
            strategy = MLLGBMStrategy(config.parameters)
        else:
            raise HTTPException(status_code=400, detail="Unknown strategy type")

        # Save strategy to database
        strategy_data = {
            "name": config.name,
            "strategy_type": strategy_type,
            "parameters": config.parameters,
            "symbols": config.parameters.get('symbols', ['AAPL']),
            "status": "inactive",
            "created_at": datetime.now()
        }
        db.save_strategy(strategy_data)

        # Store in memory
        _strategies[config.name] = strategy

        return StrategyResponse(
            name=config.name,
            status="created",
            parameters=config.parameters,
            performance={}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{strategy_name}", response_model=StrategyResponse)
async def get_strategy(strategy_name: str):
    """Get specific strategy details."""
    try:
        if strategy_name not in _strategies:
            raise HTTPException(status_code=404, detail="Strategy not found")

        strategy = _strategies[strategy_name]

        return StrategyResponse(
            name=strategy_name,
            status="active",
            parameters=getattr(strategy, 'config', {}),
            performance={}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{strategy_name}", response_model=StrategyResponse)
async def update_strategy(strategy_name: str, config: StrategyConfigRequest):
    """Update strategy configuration."""
    try:
        if strategy_name not in _strategies:
            raise HTTPException(status_code=404, detail="Strategy not found")

        # Update strategy parameters
        strategy = _strategies[strategy_name]
        if hasattr(strategy, 'update_config'):
            strategy.update_config(config.parameters)

        return StrategyResponse(
            name=strategy_name,
            status="updated",
            parameters=config.parameters,
            performance={}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{strategy_name}")
async def delete_strategy(strategy_name: str):
    """Delete a trading strategy."""
    try:
        if strategy_name not in _strategies:
            raise HTTPException(status_code=404, detail="Strategy not found")

        # Stop strategy if running
        strategy = _strategies[strategy_name]
        if hasattr(strategy, 'stop'):
            strategy.stop()

        del _strategies[strategy_name]

        return {"message": f"Strategy {strategy_name} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{strategy_name}/start")
async def start_strategy(strategy_name: str):
    """Start a trading strategy and update database status."""
    try:
        # Load strategy if not in memory
        if strategy_name not in _strategies:
            load_strategies_from_db()

        if strategy_name not in _strategies:
            raise HTTPException(status_code=404, detail="Strategy not found")

        strategy = _strategies[strategy_name]
        if hasattr(strategy, 'start'):
            strategy.start()

        # Update status in database
        strategies = db.get_strategies()
        for strategy_data in strategies:
            if strategy_data['name'] == strategy_name:
                strategy_data['status'] = 'active'
                db.save_strategy(strategy_data)
                break

        return {"message": f"Strategy {strategy_name} started"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{strategy_name}/stop")
async def stop_strategy(strategy_name: str):
    """Stop a trading strategy and update database status."""
    try:
        # Load strategy if not in memory
        if strategy_name not in _strategies:
            load_strategies_from_db()

        if strategy_name not in _strategies:
            raise HTTPException(status_code=404, detail="Strategy not found")

        strategy = _strategies[strategy_name]
        if hasattr(strategy, 'stop'):
            strategy.stop()

        # Update status in database
        strategies = db.get_strategies()
        for strategy_data in strategies:
            if strategy_data['name'] == strategy_name:
                strategy_data['status'] = 'inactive'
                db.save_strategy(strategy_data)
                break

        return {"message": f"Strategy {strategy_name} stopped"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))