"""
Advanced Trading Framework - Monitoring & Observability

This module provides comprehensive monitoring, health checks, and metrics
collection for the trading framework.

Author: Monitoring Specialist
Version: 1.0.0
"""

import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    UNKNOWN = "UNKNOWN"


@dataclass
class HealthCheck:
    """Represents a health check result."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert health check to dictionary."""
        return {
            'name': self.name,
            'status': self.status.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'duration_ms': self.duration_ms,
            'details': self.details
        }


@dataclass
class Metric:
    """Represents a metric measurement."""
    name: str
    value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags,
            'unit': self.unit
        }


class HealthChecker:
    """
    Comprehensive health checker for the trading framework.

    This class performs various health checks on system components,
    data sources, and trading functionality.
    """

    def __init__(self):
        self.checks: Dict[str, Callable[[], HealthCheck]] = {}
        self.register_default_checks()

    def register_check(self, name: str, check_func: Callable[[], HealthCheck]):
        """
        Register a health check function.

        Args:
            name: Name of the health check
            check_func: Function that performs the check and returns HealthCheck
        """
        self.checks[name] = check_func

    def register_default_checks(self):
        """Register default health checks."""
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("memory_usage", self._check_memory_usage)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("network_connectivity", self._check_network_connectivity)

    def run_all_checks(self) -> List[HealthCheck]:
        """
        Run all registered health checks.

        Returns:
            List of health check results
        """
        results = []
        for name, check_func in self.checks.items():
            try:
                start_time = time.time()
                result = check_func()
                result.duration_ms = (time.time() - start_time) * 1000
                results.append(result)
            except Exception as e:
                logger.error(f"Health check '{name}' failed: {e}")
                results.append(HealthCheck(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Check failed: {str(e)}"
                ))

        return results

    def get_overall_status(self, checks: List[HealthCheck]) -> HealthStatus:
        """
        Determine overall health status from check results.

        Args:
            checks: List of health check results

        Returns:
            Overall health status
        """
        if not checks:
            return HealthStatus.UNKNOWN

        # If any check is unhealthy, overall status is unhealthy
        if any(check.status == HealthStatus.UNHEALTHY for check in checks):
            return HealthStatus.UNHEALTHY

        # If any check is degraded, overall status is degraded
        if any(check.status == HealthStatus.DEGRADED for check in checks):
            return HealthStatus.DEGRADED

        # If all checks are healthy, overall status is healthy
        if all(check.status == HealthStatus.HEALTHY for check in checks):
            return HealthStatus.HEALTHY

        # Default to unknown
        return HealthStatus.UNKNOWN

    def _check_system_resources(self) -> HealthCheck:
        """Check basic system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None

            status = HealthStatus.HEALTHY
            message = f"CPU usage: {cpu_percent:.1f}%"

            if cpu_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"High CPU usage: {cpu_percent:.1f}%"
            elif cpu_percent > 70:
                status = HealthStatus.DEGRADED
                message = f"Elevated CPU usage: {cpu_percent:.1f}%"

            details = {
                'cpu_percent': cpu_percent,
                'load_average': load_avg
            }

            return HealthCheck(
                name="system_resources",
                status=status,
                message=message,
                details=details
            )

        except Exception as e:
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check system resources: {str(e)}"
            )

    def _check_memory_usage(self) -> HealthCheck:
        """Check system memory usage."""
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            status = HealthStatus.HEALTHY
            message = f"Memory usage: {memory_percent:.1f}%"

            if memory_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"High memory usage: {memory_percent:.1f}%"
            elif memory_percent > 75:
                status = HealthStatus.DEGRADED
                message = f"Elevated memory usage: {memory_percent:.1f}%"

            details = {
                'memory_percent': memory_percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'memory_available_gb': memory.available / (1024**3)
            }

            return HealthCheck(
                name="memory_usage",
                status=status,
                message=message,
                details=details
            )

        except Exception as e:
            return HealthCheck(
                name="memory_usage",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check memory usage: {str(e)}"
            )

    def _check_disk_space(self) -> HealthCheck:
        """Check disk space availability."""
        try:
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent

            status = HealthStatus.HEALTHY
            message = f"Disk usage: {disk_percent:.1f}%"

            if disk_percent > 95:
                status = HealthStatus.UNHEALTHY
                message = f"Critical disk usage: {disk_percent:.1f}%"
            elif disk_percent > 85:
                status = HealthStatus.DEGRADED
                message = f"High disk usage: {disk_percent:.1f}%"

            details = {
                'disk_percent': disk_percent,
                'disk_used_gb': disk.used / (1024**3),
                'disk_total_gb': disk.total / (1024**3),
                'disk_free_gb': disk.free / (1024**3)
            }

            return HealthCheck(
                name="disk_space",
                status=status,
                message=message,
                details=details
            )

        except Exception as e:
            return HealthCheck(
                name="disk_space",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check disk space: {str(e)}"
            )

    def _check_network_connectivity(self) -> HealthCheck:
        """Check network connectivity."""
        try:
            import socket
            # Try to connect to a reliable host
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            return HealthCheck(
                name="network_connectivity",
                status=HealthStatus.HEALTHY,
                message="Network connectivity is available"
            )
        except Exception as e:
            return HealthCheck(
                name="network_connectivity",
                status=HealthStatus.UNHEALTHY,
                message=f"Network connectivity check failed: {str(e)}"
            )


class MetricsCollector:
    """
    Metrics collection and reporting system.

    This class collects various metrics from the trading system
    and provides methods for querying and exporting metrics.
    """

    def __init__(self):
        self.metrics: List[Metric] = []
        self.max_metrics = 10000  # Prevent unbounded growth
        self.lock = threading.Lock()

    def record_metric(self, name: str, value: Any, tags: Optional[Dict[str, str]] = None, unit: Optional[str] = None):
        """
        Record a metric measurement.

        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for categorization
            unit: Optional unit of measurement
        """
        metric = Metric(name=name, value=value, tags=tags or {}, unit=unit)

        with self.lock:
            self.metrics.append(metric)
            if len(self.metrics) > self.max_metrics:
                # Remove oldest metrics
                self.metrics = self.metrics[-self.max_metrics:]

    def get_metrics(self, name: Optional[str] = None, tags: Optional[Dict[str, str]] = None,
                   since: Optional[datetime] = None) -> List[Metric]:
        """
        Query metrics with optional filters.

        Args:
            name: Filter by metric name
            tags: Filter by tags (all tags must match)
            since: Filter by timestamp

        Returns:
            List of matching metrics
        """
        with self.lock:
            metrics = self.metrics.copy()

        # Apply filters
        if name:
            metrics = [m for m in metrics if m.name == name]

        if tags:
            metrics = [m for m in metrics if all(m.tags.get(k) == v for k, v in tags.items())]

        if since:
            metrics = [m for m in metrics if m.timestamp >= since]

        return metrics

    def get_latest_metric(self, name: str, tags: Optional[Dict[str, str]] = None) -> Optional[Metric]:
        """
        Get the most recent metric with the given name and tags.

        Args:
            name: Metric name
            tags: Optional tags to match

        Returns:
            Most recent matching metric or None
        """
        metrics = self.get_metrics(name=name, tags=tags)
        return max(metrics, key=lambda m: m.timestamp) if metrics else None

    def get_metric_summary(self, name: str, tags: Optional[Dict[str, str]] = None,
                          since: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get summary statistics for a metric.

        Args:
            name: Metric name
            tags: Optional tags to filter
            since: Optional timestamp filter

        Returns:
            Dictionary with metric summary statistics
        """
        metrics = self.get_metrics(name=name, tags=tags, since=since)

        if not metrics:
            return {"count": 0}

        values = [m.value for m in metrics if isinstance(m.value, (int, float))]

        if not values:
            return {"count": len(metrics)}

        return {
            "count": len(metrics),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1],
            "unit": metrics[0].unit
        }


# Global instances
health_checker = HealthChecker()
metrics_collector = MetricsCollector()


def record_trading_metric(name: str, value: Any, tags: Optional[Dict[str, str]] = None):
    """
    Convenience function to record trading-related metrics.

    Args:
        name: Metric name
        value: Metric value
        tags: Optional tags
    """
    metrics_collector.record_metric(name, value, tags)


def get_system_health() -> Dict[str, Any]:
    """
    Get comprehensive system health information.

    Returns:
        Dictionary with health status and details
    """
    checks = health_checker.run_all_checks()
    overall_status = health_checker.get_overall_status(checks)

    return {
        "overall_status": overall_status.value,
        "checks": [check.to_dict() for check in checks],
        "timestamp": datetime.now().isoformat()
    }


def get_system_metrics() -> Dict[str, Any]:
    """
    Get comprehensive system metrics.

    Returns:
        Dictionary with various system metrics
    """
    try:
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Memory metrics
        memory = psutil.virtual_memory()

        # Disk metrics
        disk = psutil.disk_usage('/')

        # Network metrics (basic)
        network = psutil.net_io_counters()

        return {
            "cpu": {
                "percent": cpu_percent,
                "count": psutil.cpu_count()
            },
            "memory": {
                "percent": memory.percent,
                "used_gb": memory.used / (1024**3),
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3)
            },
            "disk": {
                "percent": disk.percent,
                "used_gb": disk.used / (1024**3),
                "total_gb": disk.total / (1024**3),
                "free_gb": disk.free / (1024**3)
            },
            "network": {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to collect system metrics: {e}")
        return {"error": str(e)}