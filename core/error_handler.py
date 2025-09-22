"""
Advanced Trading Framework - Error Handling Utilities

This module provides comprehensive error handling, recovery mechanisms,
and user-friendly error reporting for the trading framework.

Author: Error Handling Specialist
Version: 1.0.0
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from functools import wraps
from dataclasses import dataclass
from enum import Enum
import traceback

T = TypeVar('T')


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ErrorCategory(Enum):
    """Error categories for better classification."""
    NETWORK = "NETWORK"
    DATA = "DATA"
    MODEL = "MODEL"
    CONFIGURATION = "CONFIGURATION"
    BROKER = "BROKER"
    SYSTEM = "SYSTEM"
    VALIDATION = "VALIDATION"


@dataclass
class FrameworkError:
    """Structured error information."""
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    original_exception: Optional[Exception] = None
    context: Optional[Dict[str, Any]] = None
    timestamp: float = None
    recoverable: bool = True

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            'message': self.message,
            'category': self.category.value,
            'severity': self.severity.value,
            'original_exception': str(self.original_exception) if self.original_exception else None,
            'context': self.context or {},
            'timestamp': self.timestamp,
            'recoverable': self.recoverable
        }


class ErrorHandler:
    """
    Centralized error handling and recovery system.

    This class provides:
    - Structured error logging and reporting
    - Automatic retry mechanisms with exponential backoff
    - Graceful degradation strategies
    - User-friendly error messages
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize error handler.

        Args:
            logger: Logger instance to use for error reporting
        """
        self.logger = logger or logging.getLogger(__name__)
        self.error_history: List[FrameworkError] = []
        self.max_history_size = 1000

    def handle_error(
        self,
        error: Exception,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True
    ) -> FrameworkError:
        """
        Handle and log an error.

        Args:
            error: The exception that occurred
            category: Category of the error
            severity: Severity level
            context: Additional context information
            recoverable: Whether this error is recoverable

        Returns:
            Structured FrameworkError object
        """
        framework_error = FrameworkError(
            message=str(error),
            category=category,
            severity=severity,
            original_exception=error,
            context=context,
            recoverable=recoverable
        )

        # Add to history
        self.error_history.append(framework_error)
        if len(self.error_history) > self.max_history_size:
            self.error_history.pop(0)

        # Log the error
        self._log_error(framework_error)

        return framework_error

    def _log_error(self, error: FrameworkError):
        """Log error with appropriate level and formatting."""
        log_message = f"[{error.category.value}] {error.message}"

        if error.context:
            context_str = ", ".join(f"{k}={v}" for k, v in error.context.items())
            log_message += f" (Context: {context_str})"

        if error.original_exception:
            log_message += f" | Exception: {type(error.original_exception).__name__}"

        # Choose log level based on severity
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)

        # Log stack trace for high severity errors
        if error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            if error.original_exception:
                self.logger.debug("Full traceback:", exc_info=error.original_exception)

    def get_recent_errors(
        self,
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None,
        limit: int = 10
    ) -> List[FrameworkError]:
        """
        Get recent errors with optional filtering.

        Args:
            category: Filter by error category
            severity: Filter by error severity
            limit: Maximum number of errors to return

        Returns:
            List of recent FrameworkError objects
        """
        errors = self.error_history

        if category:
            errors = [e for e in errors if e.category == category]
        if severity:
            errors = [e for e in errors if e.severity == severity]

        return errors[-limit:]

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary statistics of errors."""
        if not self.error_history:
            return {"total_errors": 0}

        categories = {}
        severities = {}

        for error in self.error_history:
            cat = error.category.value
            sev = error.severity.value

            categories[cat] = categories.get(cat, 0) + 1
            severities[sev] = severities.get(sev, 0) + 1

        return {
            "total_errors": len(self.error_history),
            "by_category": categories,
            "by_severity": severities,
            "most_recent": self.error_history[-1].to_dict() if self.error_history else None
        }


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        backoff_factor: Factor to multiply delay by after each retry
        exceptions: Tuple of exceptions to catch and retry on

    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            delay = base_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break

                    delay = min(delay * backoff_factor, max_delay)
                    await asyncio.sleep(delay)

            # If we get here, all retries failed
            raise last_exception

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            delay = base_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break

                    delay = min(delay * backoff_factor, max_delay)
                    time.sleep(delay)

            # If we get here, all retries failed
            raise last_exception

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def graceful_degradation(
    fallback_value: Any = None,
    log_warning: bool = True
):
    """
    Decorator for graceful degradation on function failures.

    Args:
        fallback_value: Value to return if function fails
        log_warning: Whether to log a warning on failure

    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Union[T, Any]:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if log_warning:
                    logging.warning(f"Function {func.__name__} failed, using fallback: {e}")
                return fallback_value

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Union[T, Any]:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_warning:
                    logging.warning(f"Function {func.__name__} failed, using fallback: {e}")
                return fallback_value

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class RecoveryManager:
    """
    Manages recovery strategies for different types of failures.

    This class provides strategies for recovering from various types of errors,
    such as network failures, data issues, and system problems.
    """

    def __init__(self, error_handler: ErrorHandler):
        """
        Initialize recovery manager.

        Args:
            error_handler: Error handler instance for logging
        """
        self.error_handler = error_handler
        self.recovery_strategies = {
            ErrorCategory.NETWORK: self._recover_network_error,
            ErrorCategory.DATA: self._recover_data_error,
            ErrorCategory.MODEL: self._recover_model_error,
            ErrorCategory.BROKER: self._recover_broker_error,
            ErrorCategory.CONFIGURATION: self._recover_config_error,
        }

    def attempt_recovery(self, error: FrameworkError) -> bool:
        """
        Attempt to recover from an error.

        Args:
            error: The error to attempt recovery for

        Returns:
            True if recovery was successful, False otherwise
        """
        if not error.recoverable:
            return False

        strategy = self.recovery_strategies.get(error.category)
        if strategy:
            try:
                return strategy(error)
            except Exception as e:
                self.error_handler.handle_error(
                    e,
                    ErrorCategory.SYSTEM,
                    ErrorSeverity.HIGH,
                    {"recovery_attempt": True, "original_error": error.to_dict()}
                )
                return False

        return False

    def _recover_network_error(self, error: FrameworkError) -> bool:
        """Attempt to recover from network errors."""
        # For network errors, we might wait and retry
        time.sleep(5)  # Simple wait strategy
        return True  # Assume recovery successful for now

    def _recover_data_error(self, error: FrameworkError) -> bool:
        """Attempt to recover from data errors."""
        # For data errors, we might clear cache or use fallback data
        return False  # Data errors usually require manual intervention

    def _recover_model_error(self, error: FrameworkError) -> bool:
        """Attempt to recover from model errors."""
        # For model errors, we might retrain or use a backup model
        return False  # Model errors usually require retraining

    def _recover_broker_error(self, error: FrameworkError) -> bool:
        """Attempt to recover from broker errors."""
        # For broker errors, we might switch to paper trading
        return False  # Broker errors usually require manual intervention

    def _recover_config_error(self, error: FrameworkError) -> bool:
        """Attempt to recover from configuration errors."""
        # For config errors, we might use defaults
        return False  # Config errors usually require manual fixes


# Global error handler instance
error_handler = ErrorHandler()
recovery_manager = RecoveryManager(error_handler)


def get_user_friendly_message(error: FrameworkError) -> str:
    """
    Convert technical error messages to user-friendly messages.

    Args:
        error: FrameworkError to convert

    Returns:
        User-friendly error message
    """
    messages = {
        ErrorCategory.NETWORK: "Network connection issue. Please check your internet connection.",
        ErrorCategory.DATA: "Data retrieval problem. Market data may be temporarily unavailable.",
        ErrorCategory.MODEL: "AI model issue. The trading strategy may need retraining.",
        ErrorCategory.CONFIGURATION: "Configuration problem. Please check your settings.",
        ErrorCategory.BROKER: "Trading platform connection issue. Please verify your broker credentials.",
        ErrorCategory.SYSTEM: "System error occurred. Please restart the application.",
        ErrorCategory.VALIDATION: "Input validation error. Please check your inputs.",
    }

    base_message = messages.get(error.category, "An unexpected error occurred.")

    if error.severity == ErrorSeverity.CRITICAL:
        base_message += " This is a critical issue that requires immediate attention."
    elif error.severity == ErrorSeverity.HIGH:
        base_message += " Please review the logs for more details."

    return base_message