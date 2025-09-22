"""
Advanced Trading Framework - Configuration Validator

This module provides comprehensive validation for the trading framework configuration,
ensuring all settings are valid and reasonable before the application starts.

Author: Configuration Validator
Version: 1.0.0
"""

import os
import yaml
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ValidationSeverity(Enum):
    """Severity levels for configuration validation issues."""
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class ValidationIssue:
    """Represents a configuration validation issue."""
    severity: ValidationSeverity
    field_path: str
    message: str
    suggestion: Optional[str] = None


class ConfigValidator:
    """
    Comprehensive configuration validator for the trading framework.

    This class validates all aspects of the configuration file to ensure:
    - Required fields are present
    - Values are within reasonable ranges
    - Dependencies between settings are valid
    - API keys and credentials are properly configured
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the configuration validator.

        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        self.issues: List[ValidationIssue] = []

    def validate_config(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """
        Validate the entire configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            List of validation issues found
        """
        self.issues = []
        self.config = config  # Store config for use in validation methods

        # Validate top-level sections
        self._validate_app_section(config.get('app', {}))
        self._validate_logging_section(config.get('logging', {}))
        self._validate_data_section(config.get('data', {}))
        self._validate_automation_section(config.get('automation', {}))
        self._validate_broker_section(config.get('broker', {}))
        self._validate_portfolio_section(config.get('portfolio', {}))
        self._validate_strategy_section(config.get('strategy', {}))
        self._validate_risk_section(config.get('risk', {}))
        self._validate_ui_section(config.get('ui', {}))
        self._validate_performance_section(config.get('performance', {}))
        self._validate_storage_section(config.get('storage', {}))
        self._validate_environment_section(config.get('environment', {}))

        # Cross-section validations
        self._validate_cross_section_dependencies(config)

        return self.issues

    def _add_issue(self, severity: ValidationSeverity, field_path: str,
                   message: str, suggestion: Optional[str] = None):
        """Add a validation issue to the issues list."""
        self.issues.append(ValidationIssue(severity, field_path, message, suggestion))

    def _validate_app_section(self, app_config: Dict[str, Any]):
        """Validate application settings."""
        if not app_config:
            self._add_issue(ValidationSeverity.WARNING, 'app',
                          "App section is missing or empty",
                          "Add basic app configuration with name, version, and debug settings")
            return

        # Validate version format
        version = app_config.get('version', '')
        if version and not self._is_valid_version(version):
            self._add_issue(ValidationSeverity.WARNING, 'app.version',
                          f"Version '{version}' may not follow semantic versioning",
                          "Use semantic versioning format: MAJOR.MINOR.PATCH")

        # Validate log level
        log_level = app_config.get('log_level', 'INFO')
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if log_level.upper() not in valid_levels:
            self._add_issue(ValidationSeverity.ERROR, 'app.log_level',
                          f"Invalid log level '{log_level}'. Must be one of: {', '.join(valid_levels)}")

    def _validate_logging_section(self, logging_config: Dict[str, Any]):
        """Validate logging configuration."""
        if not logging_config:
            self._add_issue(ValidationSeverity.INFO, 'logging',
                          "Logging section not configured, using defaults")
            return

        # Validate file settings
        file_config = logging_config.get('file', {})
        if file_config.get('enabled', True):
            max_size = file_config.get('max_size_mb', 100)
            if max_size < 1 or max_size > 1000:
                self._add_issue(ValidationSeverity.WARNING, 'logging.file.max_size_mb',
                              f"Log file max size {max_size}MB is outside recommended range (1-1000MB)")

            backup_count = file_config.get('backup_count', 5)
            if backup_count < 1 or backup_count > 50:
                self._add_issue(ValidationSeverity.WARNING, 'logging.file.backup_count',
                              f"Log backup count {backup_count} is outside recommended range (1-50)")

    def _validate_data_section(self, data_config: Dict[str, Any]):
        """Validate data configuration."""
        if not data_config:
            self._add_issue(ValidationSeverity.WARNING, 'data',
                          "Data section not configured")
            return

        # Validate provider
        provider = data_config.get('provider', 'yfinance')
        valid_providers = ['yfinance', 'alpaca', 'manual']
        if provider not in valid_providers:
            self._add_issue(ValidationSeverity.ERROR, 'data.provider',
                          f"Invalid data provider '{provider}'. Must be one of: {', '.join(valid_providers)}")

        # Validate refresh intervals
        intervals = data_config.get('refresh_intervals', {})
        for interval_name, interval_value in intervals.items():
            if interval_value < 1:
                self._add_issue(ValidationSeverity.ERROR, f'data.refresh_intervals.{interval_name}',
                              f"Refresh interval {interval_name} must be at least 1 second")

    def _validate_automation_section(self, automation_config: Dict[str, Any]):
        """Validate automation settings."""
        if not automation_config:
            return  # Automation is optional

        # Validate trading limits
        max_trades_hour = automation_config.get('max_trades_per_hour', 10)
        max_trades_day = automation_config.get('max_daily_trades', 50)

        if max_trades_hour > max_trades_day:
            self._add_issue(ValidationSeverity.WARNING, 'automation.max_trades_per_hour',
                          f"Hourly trade limit ({max_trades_hour}) exceeds daily limit ({max_trades_day})")

        # Validate circuit breaker
        circuit_breaker = automation_config.get('circuit_breaker_loss', 0.1)
        if circuit_breaker > 0.5:
            self._add_issue(ValidationSeverity.WARNING, 'automation.circuit_breaker_loss',
                          f"Circuit breaker loss threshold {circuit_breaker:.1%} is very high")

    def _validate_broker_section(self, broker_config: Dict[str, Any]):
        """Validate broker configuration."""
        if not broker_config:
            self._add_issue(ValidationSeverity.ERROR, 'broker',
                          "Broker section is required")
            return

        broker_type = broker_config.get('type', 'alpaca')
        valid_types = ['alpaca', 'paper', 'manual']
        if broker_type not in valid_types:
            self._add_issue(ValidationSeverity.ERROR, 'broker.type',
                          f"Invalid broker type '{broker_type}'. Must be one of: {', '.join(valid_types)}")

        # Validate Alpaca settings if using Alpaca broker
        if broker_type == 'alpaca':
            alpaca_config = broker_config.get('alpaca', {})
            api_key = alpaca_config.get('api_key', '')
            secret_key = alpaca_config.get('secret_key', '')

            if not api_key or api_key.startswith('${'):
                self._add_issue(ValidationSeverity.ERROR, 'broker.alpaca.api_key',
                              "Alpaca API key not configured",
                              "Set ALPACA_API_KEY environment variable or configure in config")

            if not secret_key or secret_key.startswith('${'):
                self._add_issue(ValidationSeverity.ERROR, 'broker.alpaca.secret_key',
                              "Alpaca secret key not configured",
                              "Set ALPACA_SECRET_KEY environment variable or configure in config")
        elif broker_type == 'paper':
            # Paper trading doesn't require API keys
            pass

    def _validate_portfolio_section(self, portfolio_config: Dict[str, Any]):
        """Validate portfolio configuration."""
        if not portfolio_config:
            self._add_issue(ValidationSeverity.ERROR, 'portfolio',
                          "Portfolio section is required")
            return

        # Validate initial cash
        initial_cash = portfolio_config.get('initial_cash', 100000.0)
        if initial_cash <= 0:
            self._add_issue(ValidationSeverity.ERROR, 'portfolio.initial_cash',
                          "Initial cash must be positive")

        # Validate position limits
        max_positions = portfolio_config.get('max_positions', 5)
        if max_positions < 1 or max_positions > 50:
            self._add_issue(ValidationSeverity.WARNING, 'portfolio.max_positions',
                          f"Max positions {max_positions} is outside recommended range (1-50)")

        # Validate risk limits
        max_daily_loss = portfolio_config.get('max_daily_loss', 0.0001)
        if max_daily_loss > 0.1:  # 10%
            self._add_issue(ValidationSeverity.WARNING, 'portfolio.max_daily_loss',
                          f"Max daily loss {max_daily_loss:.1%} is very high")

    def _validate_strategy_section(self, strategy_config: Dict[str, Any]):
        """Validate strategy configuration."""
        if not strategy_config:
            self._add_issue(ValidationSeverity.WARNING, 'strategy',
                          "Strategy section not configured")
            return

        # Validate active strategies
        active_strategies = strategy_config.get('active_strategies', [])
        if not active_strategies:
            self._add_issue(ValidationSeverity.WARNING, 'strategy.active_strategies',
                          "No active strategies configured")

        # Validate confidence threshold
        confidence_threshold = strategy_config.get('confidence_threshold', 0.6)
        if confidence_threshold < 0.1 or confidence_threshold > 0.95:
            self._add_issue(ValidationSeverity.WARNING, 'strategy.confidence_threshold',
                          f"Confidence threshold {confidence_threshold} is outside recommended range (0.1-0.95)")

        # Validate ML strategy settings
        ml_config = strategy_config.get('ml_random_forest', {})
        if ml_config:
            lookback = ml_config.get('lookback_period', 120)
            if lookback < 30 or lookback > 500:
                self._add_issue(ValidationSeverity.WARNING, 'strategy.ml_random_forest.lookback_period',
                              f"Lookback period {lookback} days is outside recommended range (30-500)")

    def _validate_risk_section(self, risk_config: Dict[str, Any]):
        """Validate risk management configuration."""
        if not risk_config:
            return  # Risk section is optional

        # Validate portfolio risk
        max_portfolio_risk = risk_config.get('max_portfolio_risk', 0.15)
        if max_portfolio_risk > 0.5:
            self._add_issue(ValidationSeverity.WARNING, 'risk.max_portfolio_risk',
                          f"Max portfolio risk {max_portfolio_risk:.1%} is very high")

        # Validate position limits
        position_limits = risk_config.get('position_limits', {})
        max_weight = position_limits.get('max_weight', 0.1)
        if max_weight > 0.5:
            self._add_issue(ValidationSeverity.WARNING, 'risk.position_limits.max_weight',
                          f"Max position weight {max_weight:.1%} is very high")

    def _validate_ui_section(self, ui_config: Dict[str, Any]):
        """Validate UI configuration."""
        if not ui_config:
            return  # UI section is optional

        # Validate refresh rates
        refresh_rates = ui_config.get('refresh_rates', {})
        for rate_name, rate_value in refresh_rates.items():
            if rate_value < 5:
                self._add_issue(ValidationSeverity.WARNING, f'ui.refresh_rates.{rate_name}',
                              f"UI refresh rate {rate_name} ({rate_value}s) is very frequent")

    def _validate_performance_section(self, performance_config: Dict[str, Any]):
        """Validate performance configuration."""
        if not performance_config:
            return  # Performance section is optional

        # Validate benchmark ticker
        benchmark = performance_config.get('benchmark', 'SPY')
        if not benchmark:
            self._add_issue(ValidationSeverity.WARNING, 'performance.benchmark',
                          "No benchmark ticker specified")

    def _validate_storage_section(self, storage_config: Dict[str, Any]):
        """Validate storage configuration."""
        if not storage_config:
            return  # Storage section is optional

        # Validate directories exist or can be created
        directories = ['data_dir', 'models_dir', 'logs_dir', 'reports_dir']
        for dir_key in directories:
            dir_path = storage_config.get(dir_key, dir_key.replace('_dir', ''))
            if dir_path and not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                except Exception as e:
                    self._add_issue(ValidationSeverity.WARNING, f'storage.{dir_key}',
                                  f"Cannot create directory {dir_path}: {e}")

    def _validate_environment_section(self, environment_config: Dict[str, Any]):
        """Validate environment variables configuration."""
        if not environment_config:
            return  # Environment section is optional

        # Check required environment variables based on broker type
        broker_type = self.config.get('broker', {}).get('type', 'paper')

        # Alpaca API keys are only required if using Alpaca broker
        if broker_type == 'alpaca':
            alpaca_api_key = environment_config.get('alpaca_api_key', '')
            alpaca_secret_key = environment_config.get('alpaca_secret_key', '')

            if not alpaca_api_key or alpaca_api_key.startswith('${'):
                self._add_issue(ValidationSeverity.ERROR, 'environment.alpaca_api_key',
                              "Required environment variable ALPACA_API_KEY is not set",
                              "Set ALPACA_API_KEY environment variable")

            if not alpaca_secret_key or alpaca_secret_key.startswith('${'):
                self._add_issue(ValidationSeverity.ERROR, 'environment.alpaca_secret_key',
                              "Required environment variable ALPACA_SECRET_KEY is not set",
                              "Set ALPACA_SECRET_KEY environment variable")

        # Other environment variables are optional
        optional_vars = ['database_url', 'redis_url', 'email_smtp_server',
                        'email_username', 'email_password', 'webhook_url']

        for var_name in optional_vars:
            var_value = environment_config.get(var_name, '')
            if var_value and var_value.startswith('${'):
                var_env = var_value.strip('${}')
                if not os.getenv(var_env):
                    self._add_issue(ValidationSeverity.WARNING, f'environment.{var_name}',
                                  f"Environment variable {var_env} referenced but not set",
                                  f"Set {var_env} environment variable or remove from config")

    def _validate_cross_section_dependencies(self, config: Dict[str, Any]):
        """Validate dependencies between different configuration sections."""
        # Check if automation is enabled but no broker configured
        automation_enabled = config.get('automation', {}).get('enabled', False)
        broker_type = config.get('broker', {}).get('type', 'paper')

        if automation_enabled and broker_type == 'manual':
            self._add_issue(ValidationSeverity.WARNING, 'automation.enabled',
                          "Automation enabled with manual broker - trades will not execute",
                          "Consider using 'paper' or 'alpaca' broker for automated trading")

        # Check if live trading enabled but using paper broker
        alpaca_account_type = config.get('broker', {}).get('alpaca', {}).get('account_type', 'paper')
        if automation_enabled and alpaca_account_type == 'paper':
            self._add_issue(ValidationSeverity.INFO, 'broker.alpaca.account_type',
                          "Using paper trading account with automation enabled")

    def _is_valid_version(self, version: str) -> bool:
        """Check if version string follows semantic versioning."""
        import re
        return bool(re.match(r'^\d+\.\d+\.\d+', version))

    def validate_config_file(self) -> Tuple[bool, List[ValidationIssue]]:
        """
        Validate the configuration file.

        Returns:
            Tuple of (is_valid, issues_list)
        """
        try:
            if not os.path.exists(self.config_path):
                self._add_issue(ValidationSeverity.ERROR, 'config_file',
                              f"Configuration file not found: {self.config_path}")
                return False, self.issues

            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            if not config:
                self._add_issue(ValidationSeverity.ERROR, 'config_file',
                              "Configuration file is empty or invalid YAML")
                return False, self.issues

            issues = self.validate_config(config)

            # Check if there are any ERROR level issues
            has_errors = any(issue.severity == ValidationSeverity.ERROR for issue in issues)

            return not has_errors, issues

        except yaml.YAMLError as e:
            self._add_issue(ValidationSeverity.ERROR, 'config_file',
                          f"Invalid YAML syntax: {e}")
            return False, self.issues
        except Exception as e:
            self._add_issue(ValidationSeverity.ERROR, 'config_file',
                          f"Failed to read configuration file: {e}")
            return False, self.issues

    def print_validation_report(self, issues: List[ValidationIssue]):
        """Print a formatted validation report."""
        if not issues:
            print("✅ Configuration validation passed - no issues found")
            return

        print("📋 Configuration Validation Report")
        print("=" * 50)

        errors = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        warnings = [i for i in issues if i.severity == ValidationSeverity.WARNING]
        infos = [i for i in issues if i.severity == ValidationSeverity.INFO]

        if errors:
            print(f"❌ {len(errors)} Error(s):")
            for issue in errors:
                print(f"   • {issue.field_path}: {issue.message}")
                if issue.suggestion:
                    print(f"     💡 {issue.suggestion}")
            print()

        if warnings:
            print(f"⚠️  {len(warnings)} Warning(s):")
            for issue in warnings:
                print(f"   • {issue.field_path}: {issue.message}")
                if issue.suggestion:
                    print(f"     💡 {issue.suggestion}")
            print()

        if infos:
            print(f"ℹ️  {len(infos)} Info:")
            for issue in infos:
                print(f"   • {issue.field_path}: {issue.message}")
                if issue.suggestion:
                    print(f"     💡 {issue.suggestion}")
            print()

        if errors:
            print("❌ Configuration has errors that must be fixed")
        elif warnings:
            print("⚠️  Configuration has warnings that should be reviewed")
        else:
            print("✅ Configuration is valid (only informational messages)")


def validate_configuration(config_path: str = "config.yaml") -> bool:
    """
    Convenience function to validate configuration and print report.

    Args:
        config_path: Path to configuration file

    Returns:
        True if configuration is valid, False otherwise
    """
    validator = ConfigValidator(config_path)
    is_valid, issues = validator.validate_config_file()
    validator.print_validation_report(issues)
    return is_valid


if __name__ == "__main__":
    # Run validation when script is executed directly
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    success = validate_configuration(config_path)
    sys.exit(0 if success else 1)