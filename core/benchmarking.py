"""
Advanced Trading Framework - Benchmarking System

Comprehensive benchmarking system for comparing strategy performance against
market indices, peer strategies, and custom benchmarks.

Author: Benchmarking Specialist
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from scipy import stats
import logging
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results of benchmarking analysis."""
    strategy_name: str
    benchmark_name: str
    period_start: datetime
    period_end: datetime

    # Returns
    strategy_return: float = 0.0
    benchmark_return: float = 0.0
    excess_return: float = 0.0  # Strategy - Benchmark

    # Risk metrics
    strategy_volatility: float = 0.0
    benchmark_volatility: float = 0.0
    tracking_error: float = 0.0

    # Risk-adjusted metrics
    alpha: float = 0.0  # Jensen's Alpha
    beta: float = 0.0
    sharpe_ratio: float = 0.0
    information_ratio: float = 0.0

    # Performance attribution
    market_timing: float = 0.0
    security_selection: float = 0.0

    # Statistical significance
    alpha_t_stat: float = 0.0
    alpha_p_value: float = 0.0

    # Rolling metrics
    rolling_alpha: List[float] = field(default_factory=list)
    rolling_beta: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy_name': self.strategy_name,
            'benchmark_name': self.benchmark_name,
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'strategy_return': self.strategy_return,
            'benchmark_return': self.benchmark_return,
            'excess_return': self.excess_return,
            'strategy_volatility': self.strategy_volatility,
            'benchmark_volatility': self.benchmark_volatility,
            'tracking_error': self.tracking_error,
            'alpha': self.alpha,
            'beta': self.beta,
            'sharpe_ratio': self.sharpe_ratio,
            'information_ratio': self.information_ratio,
            'market_timing': self.market_timing,
            'security_selection': self.security_selection,
            'alpha_t_stat': self.alpha_t_stat,
            'alpha_p_value': self.alpha_p_value
        }


class BenchmarkProvider:
    """
    Provider for market index and benchmark data.

    Supports major indices, custom benchmarks, and peer group comparisons.
    """

    # Major market indices
    MAJOR_INDICES = {
        'SPY': 'S&P 500 ETF',
        'QQQ': 'NASDAQ 100 ETF',
        'IWM': 'Russell 2000 ETF',
        'EFA': 'MSCI EAFE ETF',
        'VWO': 'Emerging Markets ETF',
        'BND': 'Bloomberg Barclays U.S. Aggregate Bond ETF',
        'VNQ': 'Real Estate ETF',
        'GLD': 'Gold ETF',
        'USO': 'WTI Crude Oil ETF'
    }

    # Sector ETFs
    SECTOR_ETF = {
        'XLK': 'Technology',
        'XLV': 'Healthcare',
        'XLF': 'Financials',
        'XLE': 'Energy',
        'XLI': 'Industrials',
        'XLC': 'Communication Services',
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLU': 'Utilities',
        'XLRE': 'Real Estate',
        'XLB': 'Materials'
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Data cache settings
        self.cache_dir = config.get('cache_dir', 'data/benchmark_cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        # Risk-free rate for calculations
        self.risk_free_rate = config.get('risk_free_rate', 0.02)

    def get_benchmark_data(self, benchmark_symbol: str, start_date: str,
                          end_date: str, interval: str = '1d') -> pd.Series:
        """
        Get benchmark data for the specified period.

        Args:
            benchmark_symbol: Benchmark symbol (e.g., 'SPY', 'QQQ')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval

        Returns:
            Benchmark price series
        """
        cache_file = os.path.join(self.cache_dir, f"{benchmark_symbol}_{start_date}_{end_date}_{interval}.pkl")

        # Try to load from cache first
        if os.path.exists(cache_file):
            try:
                data = pd.read_pickle(cache_file)
                self.logger.info(f"Loaded {benchmark_symbol} benchmark data from cache")
                return data
            except Exception as e:
                self.logger.warning(f"Failed to load cache for {benchmark_symbol}: {e}")

        # Download fresh data
        try:
            self.logger.info(f"Downloading {benchmark_symbol} benchmark data")
            data = yf.download(benchmark_symbol, start=start_date, end=end_date,
                             interval=interval, progress=False)

            if data.empty:
                raise ValueError(f"No data available for benchmark {benchmark_symbol}")

            # Handle MultiIndex columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                # Flatten MultiIndex columns for single symbol
                data.columns = data.columns.droplevel(1)

            # Use adjusted close for total returns
            if 'Adj Close' in data.columns:
                prices = data['Adj Close']
            else:
                prices = data['Close']

            # Ensure prices is a Series (extract from single-column DataFrame if needed)
            if isinstance(prices, pd.DataFrame):
                prices = prices.iloc[:, 0]  # Take first (and only) column

            # Cache the data
            prices.to_pickle(cache_file)

            self.logger.info(f"Cached benchmark data for {benchmark_symbol} ({len(prices)} periods)")
            return prices

        except Exception as e:
            self.logger.error(f"Failed to get benchmark data for {benchmark_symbol}: {e}")
            return pd.Series()

    def get_multiple_benchmarks(self, benchmark_symbols: List[str], start_date: str,
                               end_date: str, interval: str = '1d') -> pd.DataFrame:
        """
        Get data for multiple benchmarks.

        Args:
            benchmark_symbols: List of benchmark symbols
            start_date: Start date
            end_date: End date
            interval: Data interval

        Returns:
            DataFrame with benchmark prices
        """
        benchmark_data = {}

        def load_benchmark(symbol: str) -> Tuple[str, pd.Series]:
            return symbol, self.get_benchmark_data(symbol, start_date, end_date, interval)

        with ThreadPoolExecutor(max_workers=min(len(benchmark_symbols), 5)) as executor:
            futures = [executor.submit(load_benchmark, symbol) for symbol in benchmark_symbols]
            for future in futures:
                symbol, data = future.result()
                if not data.empty:
                    benchmark_data[symbol] = data

        return pd.DataFrame(benchmark_data)

    def create_custom_benchmark(self, holdings: Dict[str, float], start_date: str,
                               end_date: str, rebalance_freq: str = 'M') -> pd.Series:
        """
        Create a custom benchmark from a portfolio of holdings.

        Args:
            holdings: Dictionary of symbol -> weight
            start_date: Start date
            end_date: End date
            rebalance_freq: Rebalancing frequency ('M' for monthly, 'Q' for quarterly)

        Returns:
            Custom benchmark price series
        """
        # Get price data for all holdings
        symbols = list(holdings.keys())
        price_data = self.get_multiple_benchmarks(symbols, start_date, end_date)

        if price_data.empty:
            return pd.Series()

        # Calculate weighted returns
        weights = np.array([holdings[symbol] for symbol in symbols])

        # Normalize prices to start at 100
        normalized_prices = price_data.div(price_data.iloc[0]) * 100

        # Calculate portfolio value
        portfolio_values = (normalized_prices * weights).sum(axis=1)

        return portfolio_values

    def get_peer_group_benchmark(self, peer_symbols: List[str], start_date: str,
                                end_date: str, weighting: str = 'equal') -> pd.Series:
        """
        Create a peer group benchmark.

        Args:
            peer_symbols: List of peer company symbols
            start_date: Start date
            end_date: End date
            weighting: Weighting method ('equal', 'market_cap')

        Returns:
            Peer group benchmark series
        """
        if weighting == 'equal':
            weights = {symbol: 1.0 / len(peer_symbols) for symbol in peer_symbols}
        elif weighting == 'market_cap':
            # Simplified market cap weighting (would need actual market cap data)
            weights = {symbol: 1.0 / len(peer_symbols) for symbol in peer_symbols}
            self.logger.warning("Market cap weighting not fully implemented, using equal weights")
        else:
            raise ValueError(f"Unknown weighting method: {weighting}")

        return self.create_custom_benchmark(weights, start_date, end_date)


class BenchmarkAnalyzer:
    """
    Comprehensive benchmark analysis and comparison.

    Calculates alpha, beta, tracking error, and other performance attribution metrics.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        self.rolling_window = config.get('rolling_window', 252)  # Trading days in a year

        self.benchmark_provider = BenchmarkProvider(config)

    def benchmark_strategy(self, strategy_returns: pd.Series, benchmark_symbol: str,
                          start_date: str, end_date: str, strategy_name: str = "Strategy") -> BenchmarkResult:
        """
        Benchmark a strategy against a market index.

        Args:
            strategy_returns: Strategy returns series
            benchmark_symbol: Benchmark symbol
            start_date: Analysis start date
            end_date: Analysis end date
            strategy_name: Name of the strategy

        Returns:
            Comprehensive benchmark analysis
        """
        # Get benchmark data
        benchmark_prices = self.benchmark_provider.get_benchmark_data(
            benchmark_symbol, start_date, end_date
        )

        if benchmark_prices.empty:
            raise ValueError(f"No benchmark data available for {benchmark_symbol}")

        # Calculate benchmark returns
        benchmark_returns = benchmark_prices.pct_change().dropna()

        # Align the data
        common_index = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common_index) < 30:  # Minimum 30 observations
            raise ValueError("Insufficient overlapping data for analysis")

        strategy_returns = strategy_returns.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]

        # Calculate basic metrics
        strategy_total_return = (1 + strategy_returns).prod() - 1
        benchmark_total_return = (1 + benchmark_returns).prod() - 1
        excess_return = strategy_total_return - benchmark_total_return

        # Risk metrics
        strategy_volatility = strategy_returns.std() * np.sqrt(252)  # Annualized
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252)

        # Calculate alpha and beta using CAPM
        excess_strategy_returns = strategy_returns - self.risk_free_rate / 252
        excess_benchmark_returns = benchmark_returns - self.risk_free_rate / 252

        # Linear regression for beta and alpha
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            excess_benchmark_returns.values, excess_strategy_returns.values
        )

        beta = slope
        alpha = intercept * 252  # Annualized alpha

        # Tracking error
        tracking_error = (strategy_returns - benchmark_returns).std() * np.sqrt(252)

        # Information ratio
        information_ratio = excess_return / tracking_error if tracking_error > 0 else 0

        # Sharpe ratio
        sharpe_ratio = (strategy_total_return - self.risk_free_rate) / strategy_volatility if strategy_volatility > 0 else 0

        # Statistical significance of alpha
        n = len(strategy_returns)
        alpha_se = std_err * np.sqrt(252)  # Annualized standard error
        alpha_t_stat = alpha / alpha_se if alpha_se > 0 else 0
        alpha_p_value = 2 * (1 - stats.t.cdf(abs(alpha_t_stat), n - 2))

        # Performance attribution (simplified)
        market_timing = beta * benchmark_total_return
        security_selection = alpha

        # Rolling beta and alpha
        rolling_beta = []
        rolling_alpha = []

        for i in range(self.rolling_window, len(common_index), 30):  # Monthly rolling
            window_returns = strategy_returns.iloc[i-self.rolling_window:i]
            window_benchmark = benchmark_returns.iloc[i-self.rolling_window:i]

            if len(window_returns) >= 30:
                try:
                    slope, intercept, _, _, _ = stats.linregress(
                        window_benchmark.values, window_returns.values
                    )
                    rolling_beta.append(slope)
                    rolling_alpha.append(intercept * 252)
                except:
                    continue

        return BenchmarkResult(
            strategy_name=strategy_name,
            benchmark_name=benchmark_symbol,
            period_start=pd.to_datetime(start_date),
            period_end=pd.to_datetime(end_date),
            strategy_return=strategy_total_return,
            benchmark_return=benchmark_total_return,
            excess_return=excess_return,
            strategy_volatility=strategy_volatility,
            benchmark_volatility=benchmark_volatility,
            tracking_error=tracking_error,
            alpha=alpha,
            beta=beta,
            sharpe_ratio=sharpe_ratio,
            information_ratio=information_ratio,
            market_timing=market_timing,
            security_selection=security_selection,
            alpha_t_stat=alpha_t_stat,
            alpha_p_value=alpha_p_value,
            rolling_alpha=rolling_alpha,
            rolling_beta=rolling_beta
        )

    def multi_benchmark_analysis(self, strategy_returns: pd.Series,
                               benchmark_symbols: List[str], start_date: str,
                               end_date: str, strategy_name: str = "Strategy") -> List[BenchmarkResult]:
        """
        Analyze strategy against multiple benchmarks.

        Args:
            strategy_returns: Strategy returns series
            benchmark_symbols: List of benchmark symbols
            start_date: Analysis start date
            end_date: Analysis end date
            strategy_name: Strategy name

        Returns:
            List of benchmark results
        """
        results = []

        for benchmark_symbol in benchmark_symbols:
            try:
                result = self.benchmark_strategy(
                    strategy_returns, benchmark_symbol, start_date, end_date, strategy_name
                )
                results.append(result)
            except Exception as e:
                self.logger.warning(f"Failed to benchmark against {benchmark_symbol}: {e}")

        return results

    def peer_analysis(self, strategy_returns: pd.Series, peer_strategies: Dict[str, pd.Series],
                     start_date: str, end_date: str, strategy_name: str = "Strategy") -> List[BenchmarkResult]:
        """
        Compare strategy against peer strategies.

        Args:
            strategy_returns: Strategy returns series
            peer_strategies: Dictionary of peer strategy name -> returns series
            start_date: Analysis start date
            end_date: Analysis end date
            strategy_name: Strategy name

        Returns:
            List of peer comparison results
        """
        results = []

        for peer_name, peer_returns in peer_strategies.items():
            try:
                # Create a combined series for analysis
                combined_data = pd.DataFrame({
                    'strategy': strategy_returns,
                    'peer': peer_returns
                }).dropna()

                if len(combined_data) < 30:
                    continue

                # Calculate metrics
                strategy_total_return = (1 + combined_data['strategy']).prod() - 1
                peer_total_return = (1 + combined_data['peer']).prod() - 1
                excess_return = strategy_total_return - peer_total_return

                # Risk metrics
                strategy_vol = combined_data['strategy'].std() * np.sqrt(252)
                peer_vol = combined_data['peer'].std() * np.sqrt(252)
                tracking_error = (combined_data['strategy'] - combined_data['peer']).std() * np.sqrt(252)

                # Alpha/Beta calculation
                excess_strategy = combined_data['strategy'] - self.risk_free_rate / 252
                excess_peer = combined_data['peer'] - self.risk_free_rate / 252

                slope, intercept, _, _, _ = stats.linregress(
                    excess_peer.values, excess_strategy.values
                )

                beta = slope
                alpha = intercept * 252

                # Information ratio
                info_ratio = excess_return / tracking_error if tracking_error > 0 else 0

                result = BenchmarkResult(
                    strategy_name=strategy_name,
                    benchmark_name=f"Peer: {peer_name}",
                    period_start=pd.to_datetime(start_date),
                    period_end=pd.to_datetime(end_date),
                    strategy_return=strategy_total_return,
                    benchmark_return=peer_total_return,
                    excess_return=excess_return,
                    strategy_volatility=strategy_vol,
                    benchmark_volatility=peer_vol,
                    tracking_error=tracking_error,
                    alpha=alpha,
                    beta=beta,
                    information_ratio=info_ratio
                )

                results.append(result)

            except Exception as e:
                self.logger.warning(f"Failed peer analysis with {peer_name}: {e}")

        return results

    def attribution_analysis(self, strategy_returns: pd.Series, benchmark_returns: pd.Series,
                           factor_returns: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Multi-factor attribution analysis.

        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns
            factor_returns: Factor returns (optional)

        Returns:
            Attribution breakdown
        """
        # Align data
        common_index = strategy_returns.index.intersection(benchmark_returns.index)
        strategy_returns = strategy_returns.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]

        excess_returns = strategy_returns - benchmark_returns

        # Basic attribution
        total_attribution = excess_returns.sum()

        # Timing vs selection (simplified)
        # This would be more sophisticated with actual factor models
        market_timing = 0.0
        security_selection = total_attribution - market_timing

        # Factor attribution (if factors provided)
        factor_attribution = {}
        if factor_returns is not None:
            # Multi-factor regression
            try:
                from sklearn.linear_model import LinearRegression

                # Align factor data
                factor_data = factor_returns.loc[common_index].dropna()
                common_idx = factor_data.index.intersection(excess_returns.index)
                y = excess_returns.loc[common_idx].values
                X = factor_data.loc[common_idx].values

                if len(X) > 0 and len(y) > 0:
                    model = LinearRegression()
                    model.fit(X, y)

                    for i, factor_name in enumerate(factor_returns.columns):
                        factor_attribution[factor_name] = model.coef_[i] * 252  # Annualized

            except Exception as e:
                self.logger.warning(f"Factor attribution failed: {e}")

        return {
            'total_excess_return': total_attribution,
            'market_timing': market_timing,
            'security_selection': security_selection,
            'factor_attribution': factor_attribution
        }


class BenchmarkReport:
    """
    Generate comprehensive benchmark reports and visualizations.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def generate_report(self, benchmark_results: List[BenchmarkResult],
                       output_format: str = 'dict') -> Any:
        """
        Generate a comprehensive benchmark report.

        Args:
            benchmark_results: List of benchmark results
            output_format: Output format ('dict', 'json', 'html')

        Returns:
            Report in specified format
        """
        if not benchmark_results:
            return {}

        # Summary statistics
        summary = {
            'total_benchmarks': len(benchmark_results),
            'best_alpha': max(benchmark_results, key=lambda x: x.alpha),
            'best_sharpe': max(benchmark_results, key=lambda x: x.sharpe_ratio),
            'best_information_ratio': max(benchmark_results, key=lambda x: x.information_ratio),
            'average_beta': np.mean([r.beta for r in benchmark_results]),
            'average_alpha': np.mean([r.alpha for r in benchmark_results])
        }

        # Detailed results
        detailed_results = [result.to_dict() for result in benchmark_results]

        # Statistical summary
        alphas = [r.alpha for r in benchmark_results]
        betas = [r.beta for r in benchmark_results]
        excess_returns = [r.excess_return for r in benchmark_results]

        statistical_summary = {
            'alpha_stats': {
                'mean': np.mean(alphas),
                'std': np.std(alphas),
                'min': np.min(alphas),
                'max': np.max(alphas),
                'significant_count': sum(1 for r in benchmark_results if r.alpha_p_value < 0.05)
            },
            'beta_stats': {
                'mean': np.mean(betas),
                'std': np.std(betas),
                'min': np.min(betas),
                'max': np.max(betas)
            },
            'excess_return_stats': {
                'mean': np.mean(excess_returns),
                'std': np.std(excess_returns),
                'positive_count': sum(1 for r in excess_returns if r > 0)
            }
        }

        report = {
            'summary': summary,
            'detailed_results': detailed_results,
            'statistical_summary': statistical_summary,
            'generated_at': datetime.now().isoformat()
        }

        if output_format == 'json':
            return json.dumps(report, indent=2, default=str)
        elif output_format == 'html':
            return self._generate_html_report(report)
        else:
            return report

    def _generate_html_report(self, report_data: Dict) -> str:
        """Generate HTML report from report data."""
        html = f"""
        <html>
        <head>
            <title>Benchmark Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Benchmark Analysis Report</h1>
            <p>Generated on: {report_data['generated_at']}</p>

            <div class="summary">
                <h2>Summary</h2>
                <p>Total Benchmarks: {report_data['summary']['total_benchmarks']}</p>
                <p>Best Alpha: {report_data['summary']['best_alpha']['strategy_name']} vs {report_data['summary']['best_alpha']['benchmark_name']} ({report_data['summary']['best_alpha']['alpha']:.2%})</p>
                <p>Average Alpha: {report_data['statistical_summary']['alpha_stats']['mean']:.2%}</p>
                <p>Average Beta: {report_data['statistical_summary']['beta_stats']['mean']:.2f}</p>
            </div>

            <h2>Detailed Results</h2>
            <table>
                <tr>
                    <th>Strategy</th>
                    <th>Benchmark</th>
                    <th>Strategy Return</th>
                    <th>Benchmark Return</th>
                    <th>Excess Return</th>
                    <th>Alpha</th>
                    <th>Beta</th>
                    <th>Sharpe</th>
                    <th>Information Ratio</th>
                </tr>
        """

        for result in report_data['detailed_results']:
            html += f"""
                <tr>
                    <td>{result['strategy_name']}</td>
                    <td>{result['benchmark_name']}</td>
                    <td>{result['strategy_return']:.2%}</td>
                    <td>{result['benchmark_return']:.2%}</td>
                    <td class="{'positive' if result['excess_return'] > 0 else 'negative'}">{result['excess_return']:.2%}</td>
                    <td class="{'positive' if result['alpha'] > 0 else 'negative'}">{result['alpha']:.2%}</td>
                    <td>{result['beta']:.2f}</td>
                    <td>{result['sharpe_ratio']:.2f}</td>
                    <td>{result['information_ratio']:.2f}</td>
                </tr>
            """

        html += """
            </table>
        </body>
        </html>
        """

        return html


# Convenience functions
def benchmark_strategy(strategy_returns: pd.Series, benchmark_symbol: str,
                      start_date: str, end_date: str, config: Optional[Dict[str, Any]] = None) -> BenchmarkResult:
    """Convenience function for single benchmark analysis."""
    if config is None:
        config = {'risk_free_rate': 0.02}

    analyzer = BenchmarkAnalyzer(config)
    return analyzer.benchmark_strategy(strategy_returns, benchmark_symbol, start_date, end_date)


def compare_against_peers(strategy_returns: pd.Series, peer_strategies: Dict[str, pd.Series],
                         start_date: str, end_date: str, config: Optional[Dict[str, Any]] = None) -> List[BenchmarkResult]:
    """Convenience function for peer analysis."""
    if config is None:
        config = {'risk_free_rate': 0.02}

    analyzer = BenchmarkAnalyzer(config)
    return analyzer.peer_analysis(strategy_returns, peer_strategies, start_date, end_date)