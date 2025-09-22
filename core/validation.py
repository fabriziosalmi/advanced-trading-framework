"""
Advanced Trading Framework - Validation Utilities

This module provides input validation decorators and utilities
for ensuring data integrity throughout the application.

Author: Validation Specialist
Version: 1.0.0
"""

import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Union, get_type_hints
import logging


logger = logging.getLogger(__name__)


def validate_inputs(**validators):
    """
    Decorator to validate function inputs based on provided validators.

    Args:
        **validators: Dictionary mapping parameter names to validation functions

    Returns:
        Decorated function with input validation

    Example:
        @validate_inputs(
            ticker=lambda x: isinstance(x, str) and x,
            quantity=lambda x: isinstance(x, (int, float)) and x > 0
        )
        def trade(ticker: str, quantity: float):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate each parameter that has a validator
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    try:
                        if not validator(value):
                            raise ValueError(f"Validation failed for parameter '{param_name}' with value: {value}")
                    except Exception as e:
                        logger.error(f"Input validation error in {func.__name__} for {param_name}: {e}")
                        raise ValueError(f"Invalid input for {param_name}: {e}") from e

            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_types(func: Callable) -> Callable:
    """
    Decorator to validate function parameter types based on type hints.

    Args:
        func: Function to validate

    Returns:
        Decorated function with type validation
    """
    type_hints = get_type_hints(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        for param_name, value in bound_args.arguments.items():
            if param_name in type_hints:
                expected_type = type_hints[param_name]
                if not _check_type(value, expected_type):
                    raise TypeError(f"Parameter '{param_name}' must be of type {expected_type}, got {type(value)}")

        return func(*args, **kwargs)
    return wrapper


def _check_type(value: Any, expected_type: Any) -> bool:
    """Check if a value matches the expected type."""
    try:
        # Handle Union types (Python 3.10+)
        if hasattr(expected_type, '__origin__') and expected_type.__origin__ is Union:
            return any(_check_type(value, t) for t in expected_type.__args__)

        # Handle Optional types
        if hasattr(expected_type, '__origin__') and expected_type.__origin__ is Union:
            # Check if it's Optional[T] which is Union[T, None]
            args = expected_type.__args__
            if len(args) == 2 and type(None) in args:
                if value is None:
                    return True
                return _check_type(value, args[0] if args[1] is type(None) else args[1])

        # Handle List types
        if hasattr(expected_type, '__origin__') and expected_type.__origin__ is list:
            if not isinstance(value, list):
                return False
            if expected_type.__args__:
                return all(_check_type(item, expected_type.__args__[0]) for item in value)
            return True

        # Handle Dict types
        if hasattr(expected_type, '__origin__') and expected_type.__origin__ is dict:
            if not isinstance(value, dict):
                return False
            if len(expected_type.__args__) >= 2:
                key_type, value_type = expected_type.__args__[0], expected_type.__args__[1]
                return all(
                    _check_type(k, key_type) and _check_type(v, value_type)
                    for k, v in value.items()
                )
            return True

        # Basic type checking
        return isinstance(value, expected_type)

    except Exception:
        return False


# Common validators
def validate_ticker(ticker: str) -> bool:
    """Validate stock ticker symbol."""
    if not isinstance(ticker, str):
        return False
    # Basic validation: uppercase letters, numbers, dots, max 10 chars
    import re
    return bool(re.match(r'^[A-Z0-9.]{1,10}$', ticker.upper()))


def validate_positive_number(value: Union[int, float]) -> bool:
    """Validate that value is a positive number."""
    try:
        return isinstance(value, (int, float)) and value > 0
    except (TypeError, ValueError):
        return False


def validate_non_negative_number(value: Union[int, float]) -> bool:
    """Validate that value is a non-negative number."""
    try:
        return isinstance(value, (int, float)) and value >= 0
    except (TypeError, ValueError):
        return False


def validate_percentage(value: Union[int, float]) -> bool:
    """Validate that value is a percentage between 0 and 1."""
    try:
        return isinstance(value, (int, float)) and 0 <= value <= 1
    except (TypeError, ValueError):
        return False


def validate_side(side: str) -> bool:
    """Validate trading side."""
    return isinstance(side, str) and side.upper() in ['LONG', 'SHORT']


def validate_date_string(date_str: str) -> bool:
    """Validate date string format."""
    from datetime import datetime
    try:
        datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return True
    except (ValueError, AttributeError):
        return False


class ValidationResult:
    """Result of a validation operation."""

    def __init__(self, is_valid: bool, errors: Optional[List[str]] = None):
        """
        Initialize validation result.

        Args:
            is_valid: Whether validation passed
            errors: List of error messages if validation failed
        """
        self.is_valid = is_valid
        self.errors = errors or []

    def __bool__(self) -> bool:
        return self.is_valid

    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append(error)

    def get_error_message(self) -> str:
        """Get combined error message."""
        return "; ".join(self.errors)


def validate_portfolio_config(config: Dict[str, Any]) -> ValidationResult:
    """
    Validate portfolio configuration.

    Args:
        config: Portfolio configuration dictionary

    Returns:
        ValidationResult indicating success or failure with errors
    """
    result = ValidationResult(True)

    # Validate initial capital
    initial_cash = config.get('initial_cash', 100000)
    if not validate_positive_number(initial_cash):
        result.add_error("initial_cash must be a positive number")
        result.is_valid = False

    # Validate position limits
    max_positions = config.get('max_positions', 5)
    if not isinstance(max_positions, int) or max_positions < 1 or max_positions > 100:
        result.add_error("max_positions must be an integer between 1 and 100")
        result.is_valid = False

    # Validate risk limits
    max_daily_loss = config.get('max_daily_loss', 0.01)
    if not validate_percentage(max_daily_loss):
        result.add_error("max_daily_loss must be a percentage between 0 and 1")
        result.is_valid = False

    return result


def validate_strategy_config(config: Dict[str, Any]) -> ValidationResult:
    """
    Validate strategy configuration.

    Args:
        config: Strategy configuration dictionary

    Returns:
        ValidationResult indicating success or failure with errors
    """
    result = ValidationResult(True)

    # Validate confidence threshold
    confidence_threshold = config.get('confidence_threshold', 0.6)
    if not validate_percentage(confidence_threshold):
        result.add_error("confidence_threshold must be a percentage between 0 and 1")
        result.is_valid = False

    # Validate ML strategy settings
    ml_config = config.get('ml_random_forest', {})
    if ml_config:
        lookback_period = ml_config.get('lookback_period', 120)
        if not isinstance(lookback_period, int) or lookback_period < 30 or lookback_period > 1000:
            result.add_error("lookback_period must be an integer between 30 and 1000")
            result.is_valid = False

    return result