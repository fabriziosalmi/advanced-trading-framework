"""
Advanced Trading Framework - Risk Management

Comprehensive risk management system with portfolio optimization, VaR calculations,
stress testing, and dynamic position sizing.

Author: Risk Management Specialist
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from scipy.optimize import minimize
from scipy.stats import norm
import logging
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for a portfolio."""
    # Value at Risk
    var_95: float = 0.0
    var_99: float = 0.0
    expected_shortfall_95: float = 0.0
    expected_shortfall_99: float = 0.0

    # Portfolio risk
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0

    # Diversification metrics
    diversification_ratio: float = 0.0
    concentration_ratio: float = 0.0

    # Stress test results
    stress_test_results: Dict[str, float] = field(default_factory=dict)


@dataclass
class PortfolioAllocation:
    """Portfolio allocation with weights and constraints."""
    weights: Dict[str, float]
    expected_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'weights': self.weights,
            'expected_return': self.expected_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio
        }


class PortfolioOptimizer:
    """
    Modern Portfolio Theory implementation with efficient frontier calculation.

    Features:
    - Mean-variance optimization
    - Efficient frontier computation
    - Risk parity allocation
    - Black-Litterman model support
    - Constraints handling (weights, sector, etc.)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Optimization parameters
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        self.max_weight = config.get('max_weight', 0.3)
        self.min_weight = config.get('min_weight', 0.0)
        self.rebalance_threshold = config.get('rebalance_threshold', 0.05)

    def optimize_portfolio(self, returns: pd.DataFrame, method: str = 'efficient_frontier',
                          target_return: Optional[float] = None) -> PortfolioAllocation:
        """
        Optimize portfolio using specified method.

        Args:
            returns: Historical returns DataFrame (assets as columns)
            method: Optimization method ('efficient_frontier', 'min_variance', 'risk_parity', 'equal_weight')
            target_return: Target return for efficient frontier optimization

        Returns:
            Optimal portfolio allocation
        """
        if method == 'equal_weight':
            return self._equal_weight_allocation(returns)
        elif method == 'min_variance':
            return self._min_variance_allocation(returns)
        elif method == 'risk_parity':
            return self._risk_parity_allocation(returns)
        elif method == 'efficient_frontier':
            return self._efficient_frontier_allocation(returns, target_return)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

    def _equal_weight_allocation(self, returns: pd.DataFrame) -> PortfolioAllocation:
        """Equal weight allocation across all assets."""
        n_assets = len(returns.columns)
        if n_assets == 0:
            raise ValueError("Cannot allocate portfolio with no assets")
        
        weights = {asset: 1.0 / n_assets for asset in returns.columns}

        expected_return, volatility = self._calculate_portfolio_stats(returns, weights)

        return PortfolioAllocation(
            weights=weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=(expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        )

    def _min_variance_allocation(self, returns: pd.DataFrame) -> PortfolioAllocation:
        """Minimum variance portfolio optimization."""
        n_assets = len(returns.columns)
        if n_assets == 0:
            raise ValueError("Cannot allocate portfolio with no assets")
        
        def objective(weights):
            return self._calculate_portfolio_stats(returns, dict(zip(returns.columns, weights)))[1] ** 2

        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        ]
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]

        # Initial guess
        x0 = np.array([1.0 / n_assets] * n_assets)

        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        if not result.success:
            self.logger.warning(f"Optimization failed: {result.message}")
            return self._equal_weight_allocation(returns)

        weights = dict(zip(returns.columns, result.x))
        expected_return, volatility = self._calculate_portfolio_stats(returns, weights)

        return PortfolioAllocation(
            weights=weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=(expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        )

    def _risk_parity_allocation(self, returns: pd.DataFrame) -> PortfolioAllocation:
        """Risk parity allocation - equal risk contribution from each asset."""
        cov_matrix = returns.cov().values
        n_assets = len(returns.columns)
        if n_assets == 0:
            raise ValueError("Cannot allocate portfolio with no assets")

        def objective(weights):
            # Risk contribution of each asset
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if portfolio_vol == 0:
                return 0  # No volatility, all weights are equal
            risk_contributions = weights * (np.dot(cov_matrix, weights)) / portfolio_vol
            # Target equal risk contribution
            target = portfolio_vol / n_assets
            return np.sum((risk_contributions - target) ** 2)

        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        ]
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        x0 = np.array([1.0 / n_assets] * n_assets)

        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        if not result.success:
            self.logger.warning(f"Risk parity optimization failed: {result.message}")
            return self._equal_weight_allocation(returns)

        weights = dict(zip(returns.columns, result.x))
        expected_return, volatility = self._calculate_portfolio_stats(returns, weights)

        return PortfolioAllocation(
            weights=weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=(expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        )

    def _efficient_frontier_allocation(self, returns: pd.DataFrame, target_return: Optional[float] = None) -> PortfolioAllocation:
        """Efficient frontier portfolio optimization."""
        n_assets = len(returns.columns)
        if n_assets == 0:
            raise ValueError("Cannot allocate portfolio with no assets")
        
        if target_return is None:
            # Maximize Sharpe ratio
            def objective(weights):
                exp_return, vol = self._calculate_portfolio_stats(returns, dict(zip(returns.columns, weights)))
                return -(exp_return - self.risk_free_rate) / vol if vol > 0 else 0

            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            ]
        else:
            # Minimize volatility for target return
            def objective(weights):
                return self._calculate_portfolio_stats(returns, dict(zip(returns.columns, weights)))[1] ** 2

            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: self._calculate_portfolio_stats(returns, dict(zip(returns.columns, weights)))[0] - target_return},
            ]

        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        x0 = np.array([1.0 / n_assets] * n_assets)

        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        if not result.success:
            self.logger.warning(f"Efficient frontier optimization failed: {result.message}")
            return self._equal_weight_allocation(returns)

        weights = dict(zip(returns.columns, result.x))
        expected_return, volatility = self._calculate_portfolio_stats(returns, weights)

        return PortfolioAllocation(
            weights=weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=(expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        )

    def _calculate_portfolio_stats(self, returns: pd.DataFrame, weights: Dict[str, float]) -> Tuple[float, float]:
        """Calculate expected return and volatility for a portfolio."""
        # Filter weights to only include assets in returns
        weights_filtered = {k: v for k, v in weights.items() if k in returns.columns}
        assets = list(weights_filtered.keys())

        if not assets:
            return 0.0, 0.0

        # Normalize weights
        total_weight = sum(weights_filtered.values())
        if total_weight == 0:
            return 0.0, 0.0

        weights_array = np.array([weights_filtered[asset] / total_weight for asset in assets])

        # Expected returns
        expected_returns = returns[assets].mean().values

        # Covariance matrix
        cov_matrix = returns[assets].cov().values

        # Portfolio expected return and volatility
        portfolio_return = np.dot(weights_array, expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array)))

        return portfolio_return, portfolio_volatility

    def calculate_efficient_frontier(self, returns: pd.DataFrame, n_points: int = 50) -> List[PortfolioAllocation]:
        """
        Calculate the efficient frontier.

        Args:
            returns: Historical returns DataFrame
            n_points: Number of points on the frontier

        Returns:
            List of portfolio allocations along the efficient frontier
        """
        min_return = returns.mean().min()
        max_return = returns.mean().max()

        target_returns = np.linspace(min_return, max_return, n_points)
        frontier = []

        for target_return in target_returns:
            try:
                allocation = self._efficient_frontier_allocation(returns, target_return)
                frontier.append(allocation)
            except:
                continue

        return frontier


class ValueAtRiskCalculator:
    """
    Value at Risk (VaR) calculations using multiple methods.

    Methods:
    - Historical simulation
    - Parametric (normal distribution)
    - Monte Carlo simulation
    - Cornish-Fisher expansion
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.confidence_levels = config.get('confidence_levels', [0.95, 0.99])
        self.horizon_days = config.get('horizon_days', 1)
        self.monte_carlo_simulations = config.get('monte_carlo_simulations', 10000)

    def calculate_var(self, returns: pd.Series, method: str = 'historical') -> Dict[str, float]:
        """
        Calculate Value at Risk using specified method.

        Args:
            returns: Historical returns series
            method: Calculation method ('historical', 'parametric', 'monte_carlo')

        Returns:
            Dictionary with VaR values for different confidence levels
        """
        if method == 'historical':
            return self._historical_var(returns)
        elif method == 'parametric':
            return self._parametric_var(returns)
        elif method == 'monte_carlo':
            return self._monte_carlo_var(returns)
        else:
            raise ValueError(f"Unknown VaR method: {method}")

    def _historical_var(self, returns: pd.Series) -> Dict[str, float]:
        """Historical simulation VaR."""
        var_results = {}

        for confidence in self.confidence_levels:
            # Scale returns for horizon
            scaled_returns = returns * np.sqrt(self.horizon_days)
            var = np.percentile(scaled_returns, (1 - confidence) * 100)
            var_results[f'var_{int(confidence * 100)}'] = var

            # Expected Shortfall (CVaR)
            tail_returns = scaled_returns[scaled_returns <= var]
            es = tail_returns.mean() if len(tail_returns) > 0 else var
            var_results[f'es_{int(confidence * 100)}'] = es

        return var_results

    def _parametric_var(self, returns: pd.Series) -> Dict[str, float]:
        """Parametric VaR assuming normal distribution."""
        var_results = {}

        mean_return = returns.mean()
        volatility = returns.std()

        for confidence in self.confidence_levels:
            # Z-score for confidence level
            z_score = norm.ppf(1 - confidence)

            # Scale for horizon
            scaled_vol = volatility * np.sqrt(self.horizon_days)

            var = mean_return * self.horizon_days + z_score * scaled_vol
            var_results[f'var_{int(confidence * 100)}'] = var

            # Expected Shortfall for normal distribution
            es = mean_return * self.horizon_days - scaled_vol * norm.pdf(z_score) / (1 - confidence)
            var_results[f'es_{int(confidence * 100)}'] = es

        return var_results

    def _monte_carlo_var(self, returns: pd.Series) -> Dict[str, float]:
        """Monte Carlo simulation VaR."""
        var_results = {}

        # Fit distribution parameters
        mean_return = returns.mean()
        volatility = returns.std()

        # Generate random returns
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(mean_return, volatility, self.monte_carlo_simulations)

        # Scale for horizon
        scaled_returns = simulated_returns * np.sqrt(self.horizon_days)

        for confidence in self.confidence_levels:
            var = np.percentile(scaled_returns, (1 - confidence) * 100)
            var_results[f'var_{int(confidence * 100)}'] = var

            # Expected Shortfall
            tail_returns = scaled_returns[scaled_returns <= var]
            es = tail_returns.mean() if len(tail_returns) > 0 else var
            var_results[f'es_{int(confidence * 100)}'] = es

        return var_results

    def calculate_portfolio_var(self, returns: pd.DataFrame, weights: Dict[str, float],
                               method: str = 'parametric') -> Dict[str, float]:
        """
        Calculate portfolio-level VaR.

        Args:
            returns: Asset returns DataFrame
            weights: Portfolio weights
            method: VaR calculation method

        Returns:
            Portfolio VaR metrics
        """
        # Calculate portfolio returns
        weights_array = np.array([weights.get(asset, 0) for asset in returns.columns])
        portfolio_returns = returns.dot(weights_array)

        return self.calculate_var(portfolio_returns, method)


class StressTester:
    """
    Stress testing framework for portfolios.

    Features:
    - Historical scenario analysis
    - Custom stress scenarios
    - Reverse stress testing
    - Sensitivity analysis
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def run_historical_stress_tests(self, returns: pd.DataFrame, weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Run historical stress tests using past market crises.

        Args:
            returns: Historical returns DataFrame
            weights: Current portfolio weights

        Returns:
            Stress test results
        """
        # Define historical crisis periods
        crisis_periods = {
            '2008_Financial_Crisis': ('2008-09-01', '2009-03-31'),
            '2020_COVID_Crisis': ('2020-02-01', '2020-04-30'),
            '2022_Rate_Hike': ('2022-01-01', '2022-06-30'),
            'Dot_Com_Bubble': ('2000-03-01', '2002-10-31'),
            '2011_European_Debt': ('2011-07-01', '2012-03-31')
        }

        results = {}

        for crisis_name, (start_date, end_date) in crisis_periods.items():
            try:
                # Extract crisis period returns
                mask = (returns.index >= start_date) & (returns.index <= end_date)
                crisis_returns = returns[mask]

                if len(crisis_returns) == 0:
                    continue

                # Calculate portfolio performance during crisis
                weights_array = np.array([weights.get(asset, 0) for asset in returns.columns])
                portfolio_returns = crisis_returns.dot(weights_array)

                results[crisis_name] = {
                    'start_date': start_date,
                    'end_date': end_date,
                    'total_return': (1 + portfolio_returns).prod() - 1,
                    'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
                    'volatility': portfolio_returns.std() * np.sqrt(252),
                    'worst_day': portfolio_returns.min(),
                    'peak_to_trough': self._calculate_peak_to_trough(portfolio_returns)
                }

            except Exception as e:
                self.logger.warning(f"Failed to analyze {crisis_name}: {e}")

        return results

    def run_custom_stress_scenarios(self, returns: pd.DataFrame, weights: Dict[str, float],
                                   scenarios: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Run custom stress scenarios.

        Args:
            returns: Historical returns DataFrame for volatility estimation
            weights: Current portfolio weights
            scenarios: Dictionary of stress scenarios (asset -> shock)

        Returns:
            Custom stress test results
        """
        results = {}

        for scenario_name, shocks in scenarios.items():
            try:
                # Calculate portfolio impact
                portfolio_impact = 0.0
                for asset, shock in shocks.items():
                    if asset in weights:
                        portfolio_impact += weights[asset] * shock

                # Estimate volatility impact
                asset_volatilities = returns.std()
                portfolio_vol = np.sqrt(sum(weights.get(asset, 0) ** 2 * asset_volatilities[asset] ** 2
                                           for asset in returns.columns))

                results[scenario_name] = {
                    'portfolio_impact': portfolio_impact,
                    'estimated_volatility': portfolio_vol,
                    'asset_shocks': shocks
                }

            except Exception as e:
                self.logger.warning(f"Failed to analyze scenario {scenario_name}: {e}")

        return results

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        if running_max.min() == 0:
            return 0.0  # Avoid division by zero
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _calculate_peak_to_trough(self, returns: pd.Series) -> float:
        """Calculate peak-to-trough decline."""
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        trough = cumulative.expanding().min()
        return (trough - peak).min()


class PositionSizer:
    """
    Dynamic position sizing strategies.

    Strategies:
    - Kelly Criterion
    - Fixed Fractional
    - Volatility-based
    - Risk Parity
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.max_position_size = config.get('max_position_size', 0.1)  # 10% max
        self.risk_per_trade = config.get('risk_per_trade', 0.02)  # 2% risk per trade

    def calculate_position_size(self, method: str, **kwargs) -> float:
        """
        Calculate position size using specified method.

        Args:
            method: Sizing method ('kelly', 'fixed_fractional', 'volatility_based')
            **kwargs: Method-specific parameters

        Returns:
            Position size as fraction of portfolio
        """
        if method == 'kelly':
            return self._kelly_criterion(**kwargs)
        elif method == 'fixed_fractional':
            return self._fixed_fractional(**kwargs)
        elif method == 'volatility_based':
            return self._volatility_based(**kwargs)
        else:
            raise ValueError(f"Unknown position sizing method: {method}")

    def _kelly_criterion(self, win_rate: float, win_loss_ratio: float, **kwargs) -> float:
        """
        Kelly Criterion position sizing.

        Args:
            win_rate: Probability of winning
            win_loss_ratio: Average win / average loss ratio

        Returns:
            Position size fraction
        """
        if win_rate <= 0 or win_rate >= 1 or win_loss_ratio <= 0:
            return 0.0

        kelly_fraction = win_rate - ((1 - win_rate) / win_loss_ratio)

        # Apply full Kelly or fractional Kelly
        kelly_multiplier = self.config.get('kelly_multiplier', 0.5)  # Half Kelly
        position_size = max(0, kelly_fraction * kelly_multiplier)

        return min(position_size, self.max_position_size)

    def _fixed_fractional(self, **kwargs) -> float:
        """Fixed fractional position sizing."""
        return self.risk_per_trade

    def _volatility_based(self, volatility: float, target_volatility: float = 0.15, **kwargs) -> float:
        """
        Volatility-based position sizing.

        Args:
            volatility: Asset volatility
            target_volatility: Target portfolio volatility

        Returns:
            Position size fraction
        """
        if volatility <= 0:
            return 0.0

        position_size = target_volatility / volatility
        return min(position_size, self.max_position_size)


class RiskManager:
    """
    Comprehensive risk management system.

    Integrates all risk management components:
    - Portfolio optimization
    - VaR calculations
    - Stress testing
    - Position sizing
    - Risk monitoring
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize components
        self.portfolio_optimizer = PortfolioOptimizer(config.get('optimization', {}))
        self.var_calculator = ValueAtRiskCalculator(config.get('var', {}))
        self.stress_tester = StressTester(config.get('stress_testing', {}))
        self.position_sizer = PositionSizer(config.get('position_sizing', {}))

    def calculate_portfolio_risk(self, returns: pd.DataFrame, weights: Dict[str, float]) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for a portfolio.

        Args:
            returns: Historical returns DataFrame
            weights: Portfolio weights

        Returns:
            Comprehensive risk metrics
        """
        # Calculate VaR
        portfolio_returns = returns.dot(np.array([weights.get(asset, 0) for asset in returns.columns]))
        var_metrics = self.var_calculator.calculate_var(portfolio_returns)

        # Basic portfolio statistics
        expected_return = portfolio_returns.mean()
        volatility = portfolio_returns.std()

        # Sharpe and Sortino ratios
        risk_free_rate = self.config.get('risk_free_rate', 0.02)
        sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility > 0 else 0

        downside_returns = portfolio_returns[portfolio_returns < 0]
        sortino_ratio = expected_return / downside_returns.std() if len(downside_returns) > 0 else 0

        # Maximum drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        if running_max.min() == 0:
            max_drawdown = 0.0
        else:
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

        # Calmar ratio
        calmar_ratio = expected_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Diversification metrics
        weights_array = np.array([weights.get(asset, 0) for asset in returns.columns])
        diversification_ratio = 1 / np.sum(weights_array ** 2) if np.sum(weights_array ** 2) > 0 else 0
        concentration_ratio = np.max(weights_array)

        # Stress testing
        stress_results = self.stress_tester.run_historical_stress_tests(returns, weights)

        return RiskMetrics(
            var_95=var_metrics.get('var_95', 0.0),
            var_99=var_metrics.get('var_99', 0.0),
            expected_shortfall_95=var_metrics.get('es_95', 0.0),
            expected_shortfall_99=var_metrics.get('es_99', 0.0),
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            diversification_ratio=diversification_ratio,
            concentration_ratio=concentration_ratio,
            stress_test_results=stress_results
        )

    def optimize_portfolio(self, returns: pd.DataFrame, method: str = 'efficient_frontier') -> PortfolioAllocation:
        """Optimize portfolio using specified method."""
        return self.portfolio_optimizer.optimize_portfolio(returns, method)

    def calculate_position_size(self, method: str, **kwargs) -> float:
        """Calculate position size using specified method."""
        return self.position_sizer.calculate_position_size(method, **kwargs)

    def run_stress_tests(self, returns: pd.DataFrame, weights: Dict[str, float]) -> Dict[str, Any]:
        """Run comprehensive stress tests."""
        historical = self.stress_tester.run_historical_stress_tests(returns, weights)

        # Custom scenarios
        custom_scenarios = {
            'Market_Crash': {asset: -0.3 for asset in returns.columns},  # 30% drop
            'Tech_Sector_Crash': {'AAPL': -0.5, 'MSFT': -0.5, 'GOOGL': -0.5},
            'Rate_Hike_Shock': {'BND': -0.1, 'SPY': -0.05},  # Bond selloff, equity dip
        }

        custom = self.stress_tester.run_custom_stress_scenarios(returns, weights, custom_scenarios)

        return {
            'historical_stress_tests': historical,
            'custom_stress_scenarios': custom
        }

    def check_risk_limits(self, current_metrics: RiskMetrics, limits: Dict[str, float]) -> Dict[str, bool]:
        """
        Check if current risk metrics violate defined limits.

        Args:
            current_metrics: Current risk metrics
            limits: Risk limits dictionary

        Returns:
            Dictionary indicating which limits are violated
        """
        violations = {}

        # VaR limits
        violations['var_95_limit'] = current_metrics.var_95 < limits.get('max_var_95', -0.05)
        violations['var_99_limit'] = current_metrics.var_99 < limits.get('max_var_99', -0.10)

        # Volatility limit
        violations['volatility_limit'] = current_metrics.volatility > limits.get('max_volatility', 0.25)

        # Drawdown limit
        violations['drawdown_limit'] = current_metrics.max_drawdown < limits.get('max_drawdown', -0.20)

        # Concentration limit
        violations['concentration_limit'] = current_metrics.concentration_ratio > limits.get('max_concentration', 0.25)

        return violations


# Convenience functions
def optimize_portfolio(returns: pd.DataFrame, method: str = 'efficient_frontier',
                      config: Optional[Dict[str, Any]] = None) -> PortfolioAllocation:
    """Convenience function for portfolio optimization."""
    if config is None:
        config = {'risk_free_rate': 0.02, 'max_weight': 0.3}

    optimizer = PortfolioOptimizer(config)
    return optimizer.optimize_portfolio(returns, method)


def calculate_var(returns: pd.Series, method: str = 'historical',
                 config: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Convenience function for VaR calculation."""
    if config is None:
        config = {'confidence_levels': [0.95, 0.99]}

    calculator = ValueAtRiskCalculator(config)
    return calculator.calculate_var(returns, method)