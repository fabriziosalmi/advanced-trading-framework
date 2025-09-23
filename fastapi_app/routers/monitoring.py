"""
Monitoring Router
Handles system health and performance monitoring
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import sys
from pathlib import Path
import psutil
from datetime import datetime

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from fastapi_app.models.responses import SystemMetricsResponse, HealthResponse
from core.monitoring import get_system_health, get_system_metrics

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def get_health():
    """Get system health status."""
    try:
        health = get_system_health()
        return HealthResponse(
            status="healthy" if health.get("status") == "ok" else "unhealthy",
            timestamp=datetime.now(),
            version="2.0.0"
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            version="2.0.0"
        )

@router.get("/metrics", response_model=SystemMetricsResponse)
async def get_metrics():
    """Get system performance metrics."""
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        # Get trading-specific metrics from monitoring module
        trading_metrics = get_system_metrics()

        return SystemMetricsResponse(
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            active_strategies=trading_metrics.get("active_strategies", 0),
            open_positions=trading_metrics.get("open_positions", 0),
            pending_orders=trading_metrics.get("pending_orders", 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/logs")
async def get_logs(limit: int = 100):
    """Get recent system logs."""
    try:
        # This would read from log files
        # Simplified implementation
        logs = [
            {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "message": "System running normally",
                "component": "monitoring"
            }
        ]
        return {"logs": logs[-limit:]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance")
async def get_performance_stats():
    """Get detailed performance statistics."""
    try:
        # CPU and memory details
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Network stats
        net_io = psutil.net_io_counters()

        return {
            "cpu": {
                "count": cpu_count,
                "frequency": cpu_freq.current if cpu_freq else None,
                "usage_percent": psutil.cpu_percent(interval=1)
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100
            },
            "network": {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts")
async def get_alerts():
    """Get system alerts and warnings."""
    try:
        alerts = []

        # Check system resources
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            alerts.append({
                "type": "warning",
                "message": f"High memory usage: {memory.percent:.1f}%",
                "timestamp": datetime.now().isoformat()
            })

        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 80:
            alerts.append({
                "type": "warning",
                "message": f"High CPU usage: {cpu_percent:.1f}%",
                "timestamp": datetime.now().isoformat()
            })

        return {"alerts": alerts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))