"""
Advanced Trading Framework - Backtesting Engine

Comprehensive backtesting framework with historical testing, performance metrics,
and realistic trading simulation.

Author: Backtesting Specialist
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import asyncio
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os

# Visualization imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Import benchmarking module
from .benchmarking import BenchmarkAnalyzer, BenchmarkResult

logger = logging.getLogger(__name__)


class BacktestStatus(Enum):
    """Backtest execution status."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class OrderType(Enum):
    """Order types for backtesting."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(Enum):
    """Order side for backtesting."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Trade:
    """Represents a completed trade in backtesting."""
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    slippage: float = 0.0
    order_type: OrderType = OrderType.MARKET

    @property
    def total_cost(self) -> float:
        """Total cost including commission and slippage."""
        return (self.quantity * self.price) + self.commission + self.slippage


@dataclass
class Position:
    """Represents a position in the portfolio during backtesting."""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def market_value(self) -> float:
        """Current market value of the position."""
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        return self.market_value - (self.quantity * self.avg_price)

    @property
    def unrealized_pnl_percent(self) -> float:
        """Unrealized profit/loss percentage."""
        if self.avg_price is None or self.avg_price == 0 or self.current_price is None:
            return 0.0
        return (self.current_price - self.avg_price) / self.avg_price


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio state at a point in time."""
    timestamp: datetime
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    total_value: float = 0.0

    def calculate_total_value(self) -> float:
        """Calculate total portfolio value."""
        position_value = sum(pos.market_value for pos in self.positions.values())
        self.total_value = self.cash + position_value
        return self.total_value


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for backtesting results."""
    # Basic metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0

    # Risk metrics
    value_at_risk_95: float = 0.0
    expected_shortfall_95: float = 0.0
    calmar_ratio: float = 0.0

    # Trading metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration: timedelta = field(default_factory=lambda: timedelta(0))

    # Benchmark comparison
    benchmark_return: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0


class BacktestStrategy(ABC):
    """Abstract base class for backtesting strategies."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame], portfolio: Dict[str, Position],
                        cash: float) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on market data and current portfolio.

        Args:
            data: Dictionary of market data DataFrames with OHLCV columns, keyed by symbol
            portfolio: Current portfolio positions
            cash: Available cash

        Returns:
            List of signal dictionaries with keys: 'symbol', 'side', 'quantity', 'type'
        """
        pass

    def on_trade_executed(self, trade: Trade, portfolio: Dict[str, Position]):
        """Callback when a trade is executed."""
        pass

    def on_portfolio_update(self, portfolio_snapshot: PortfolioSnapshot):
        """Callback when portfolio is updated."""
        pass


class BacktestEngine:
    """
    Comprehensive backtesting engine with realistic trading simulation.

    Features:
    - Historical data loading and caching
    - Realistic order execution with slippage and commissions
    - Performance tracking and metrics calculation
    - Walk-forward analysis support
    - Multi-strategy comparison
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Backtest parameters
        self.initial_cash = config.get('initial_cash', 100000.0)
        self.commission_per_trade = config.get('commission_per_trade', 0.0)
        self.commission_per_share = config.get('commission_per_share', 0.0)
        self.slippage_bps = config.get('slippage_bps', 0.1)  # Basis points
        self.min_commission = config.get('min_commission', 1.0)

        # Data settings
        self.data_cache_dir = config.get('data_cache_dir', 'data/cache')
        os.makedirs(self.data_cache_dir, exist_ok=True)

        # Initialize state
        self.portfolio_history: List[PortfolioSnapshot] = []
        self.trades: List[Trade] = []
        self.current_portfolio: Dict[str, Position] = {}
        self.current_cash = self.initial_cash
        self.status = BacktestStatus.PENDING

    def load_historical_data(self, symbols: List[str], start_date: str, end_date: str,
                           interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """
        Load historical market data for backtesting.

        Args:
            symbols: List of ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval ('1d', '1h', '30m', etc.)

        Returns:
            Dictionary mapping symbols to DataFrames with OHLCV data
        """
        data = {}

        def load_symbol_data(symbol: str) -> Tuple[str, pd.DataFrame]:
            cache_file = os.path.join(self.data_cache_dir, f"{symbol}_{start_date}_{end_date}_{interval}.pkl")

            # Try to load from cache first
            if os.path.exists(cache_file):
                try:
                    df = pd.read_pickle(cache_file)
                    self.logger.info(f"Loaded {symbol} data from cache")
                    return symbol, df
                except Exception as e:
                    self.logger.warning(f"Failed to load cache for {symbol}: {e}")

            # Download fresh data
            try:
                self.logger.info(f"Downloading {symbol} data from {start_date} to {end_date}")
                df = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)

                if df.empty:
                    raise ValueError(f"No data available for {symbol}")

                # Clean and prepare data
                df = df.dropna()
                df.index = pd.to_datetime(df.index)

                # Ensure required columns exist
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in required_cols):
                    raise ValueError(f"Missing required columns for {symbol}")

                # Save to cache
                df.to_pickle(cache_file)
                self.logger.info(f"Cached {symbol} data ({len(df)} rows)")

                return symbol, df

            except Exception as e:
                self.logger.error(f"Failed to load data for {symbol}: {e}")
                return symbol, pd.DataFrame()

        # Load data in parallel
        with ThreadPoolExecutor(max_workers=min(len(symbols), 10)) as executor:
            futures = [executor.submit(load_symbol_data, symbol) for symbol in symbols]
            for future in as_completed(futures):
                symbol, df = future.result()
                if not df.empty:
                    data[symbol] = df

        self.logger.info(f"Loaded data for {len(data)}/{len(symbols)} symbols")
        return data

    async def run_backtest(self, strategy: BacktestStrategy, data: Dict[str, pd.DataFrame],
                    start_date: str, end_date: str, benchmark_symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run a backtest with the given strategy and data.

        Args:
            strategy: Trading strategy to test
            data: Historical market data
            start_date: Backtest start date
            end_date: Backtest end date
            benchmark_symbols: List of benchmark symbols to compare against

        Returns:
            Dictionary with backtest results and metrics
        """
        self.status = BacktestStatus.RUNNING
        self.logger.info(f"Starting backtest for strategy: {strategy.name}")

        try:
            # Initialize portfolio
            self._initialize_portfolio()

            # Get all unique timestamps across all symbols
            all_timestamps = set()
            for df in data.values():
                all_timestamps.update(df.index)
            timestamps = sorted(list(all_timestamps))

            # Filter timestamps within backtest period
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            timestamps = [t for t in timestamps if start_dt <= t <= end_dt]

            self.logger.info(f"Running backtest over {len(timestamps)} time periods")

            # Run simulation
            for i, timestamp in enumerate(timestamps):
                try:
                    # Update current prices
                    current_data = self._get_data_at_timestamp(data, timestamp)

                    # Generate signals (handle both sync and async strategies)
                    try:
                        # Try async first
                        if asyncio.iscoroutinefunction(strategy.generate_signals):
                            signals = await strategy.generate_signals(current_data, self.current_portfolio.copy(), self.current_cash)
                        else:
                            signals = strategy.generate_signals(current_data, self.current_portfolio.copy(), self.current_cash)
                    except TypeError:
                        # Fallback to sync call
                        signals = strategy.generate_signals(current_data, self.current_portfolio.copy(), self.current_cash)

                    # Execute signals
                    for signal in signals:
                        self._execute_signal(signal, timestamp, current_data)

                    # Update portfolio snapshot
                    self._update_portfolio_snapshot(timestamp)

                    # Progress logging
                    if (i + 1) % 100 == 0:
                        self.logger.info(f"Processed {i + 1}/{len(timestamps)} periods")

                except Exception as e:
                    self.logger.error(f"Error at timestamp {timestamp}: {e}")
                    continue

            # Calculate final metrics
            # Perform benchmarking analysis if benchmarks provided
            benchmark_results = []
            if benchmark_symbols:
                self.logger.info(f"Performing benchmarking analysis against {len(benchmark_symbols)} benchmarks")
                benchmark_results = self._perform_benchmarking_analysis(
                    start_date, end_date, benchmark_symbols
                )

            metrics = self._calculate_performance_metrics(benchmark_results)

            # Generate performance report
            self.logger.info("Generating performance report")
            report_result = self.generate_performance_report()

            self.status = BacktestStatus.COMPLETED
            self.logger.info(f"Backtest completed successfully")

            return {
                'status': 'success',
                'strategy_name': strategy.name,
                'initial_cash': self.initial_cash,
                'final_value': self.portfolio_history[-1].total_value if self.portfolio_history else self.initial_cash,
                'total_return': metrics.total_return,
                'metrics': metrics,
                'trades': [trade.__dict__ for trade in self.trades],
                'portfolio_history': [snapshot.__dict__ for snapshot in self.portfolio_history],
                'benchmarking': benchmark_results,
                'performance_report': report_result
            }

        except Exception as e:
            self.status = BacktestStatus.FAILED
            self.logger.error(f"Backtest failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'strategy_name': strategy.name
            }

    def _initialize_portfolio(self):
        """Initialize portfolio state."""
        self.current_portfolio = {}
        self.current_cash = self.initial_cash
        self.trades = []
        self.portfolio_history = []

        # Create initial snapshot
        initial_snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            cash=self.current_cash,
            positions={}
        )
        initial_snapshot.calculate_total_value()
        self.portfolio_history.append(initial_snapshot)

    def _get_data_at_timestamp(self, data: Dict[str, pd.DataFrame], timestamp: datetime) -> Dict[str, pd.DataFrame]:
        """Get market data up to a specific timestamp."""
        current_data = {}

        for symbol, df in data.items():
            # Get all data up to and including the timestamp
            mask = df.index <= timestamp
            if mask.any():
                historical_data = df[mask].copy()
                current_data[symbol] = historical_data
                # Update position prices with latest data
                if symbol in self.current_portfolio:
                    latest_price = historical_data['Close'].iloc[-1]
                    self.current_portfolio[symbol].current_price = latest_price

        return current_data

    def _execute_signal(self, signal: Dict[str, Any], timestamp: datetime, current_data: Dict[str, pd.DataFrame]):
        """Execute a trading signal with realistic commission and slippage models."""
        symbol = signal['symbol']
        side = signal['side']
        quantity = signal['quantity']
        order_type = signal.get('type', OrderType.MARKET)

        if symbol not in current_data:
            return  # No data available

        data = current_data[symbol]

        # Get market conditions for realistic execution
        latest_bar = data.iloc[-1] if not data.empty else None
        if latest_bar is None:
            return

        # Calculate realistic execution price with advanced slippage model
        base_price = latest_bar['Close']
        spread = (latest_bar['High'] - latest_bar['Low']) / latest_bar['Close']  # Intraday volatility
        volume_ratio = latest_bar['Volume'] / data['Volume'].rolling(20).mean().iloc[-1] if len(data) > 20 else 1.0

        # Advanced slippage model based on order size and market conditions
        if side == OrderSide.BUY:
            # Buy orders: price increases with slippage
            slippage_factor = (
                self.slippage_bps / 10000 +  # Base slippage
                spread * 0.5 +  # Spread contribution
                (quantity / 1000) * 0.001 +  # Size impact (rough estimate)
                (1 / volume_ratio) * 0.002  # Liquidity impact
            )
            execution_price = base_price * (1 + slippage_factor)
        else:
            # Sell orders: price decreases with slippage
            slippage_factor = (
                self.slippage_bps / 10000 +  # Base slippage
                spread * 0.5 +  # Spread contribution
                (quantity / 1000) * 0.001 +  # Size impact
                (1 / volume_ratio) * 0.002  # Liquidity impact
            )
            execution_price = base_price * (1 - slippage_factor)

        # Advanced commission model
        # Base commission + per-share + percentage-based
        commission = (
            self.commission_per_trade +  # Fixed per trade
            abs(quantity) * self.commission_per_share +  # Per share
            abs(quantity * execution_price) * 0.0005  # 0.05% of trade value
        )
        commission = max(commission, self.min_commission)  # Minimum commission

        # Calculate total cost including slippage impact
        trade_value = quantity * execution_price
        total_cost = trade_value + commission

        # Check if we can execute the trade
        if side == OrderSide.BUY:
            if total_cost > self.current_cash:
                return  # Insufficient funds
        else:
            if symbol not in self.current_portfolio or self.current_portfolio[symbol].quantity < quantity:
                return  # Insufficient position

        # Create trade record with detailed cost breakdown
        trade = Trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=execution_price,
            timestamp=timestamp,
            commission=commission,
            slippage=abs(execution_price - base_price) * quantity,  # Slippage cost
            order_type=order_type
        )

        self.trades.append(trade)

        # Update portfolio
        if side == OrderSide.BUY:
            self.current_cash -= total_cost
            if symbol in self.current_portfolio:
                # Average down the position
                existing_qty = self.current_portfolio[symbol].quantity
                existing_cost = existing_qty * self.current_portfolio[symbol].avg_price
                new_cost = quantity * execution_price
                total_qty = existing_qty + quantity
                avg_price = (existing_cost + new_cost) / total_qty

                self.current_portfolio[symbol].quantity = total_qty
                self.current_portfolio[symbol].avg_price = avg_price
            else:
                # New position
                self.current_portfolio[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_price=execution_price,
                    current_price=execution_price,
                    timestamp=timestamp
                )
        else:  # SELL
            proceeds = trade_value - commission
            self.current_cash += proceeds
            if symbol in self.current_portfolio:
                self.current_portfolio[symbol].quantity -= quantity
                if self.current_portfolio[symbol].quantity <= 0:
                    del self.current_portfolio[symbol]

    def _update_portfolio_snapshot(self, timestamp: datetime):
        """Update portfolio history with current snapshot."""
        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            cash=self.current_cash,
            positions=self.current_portfolio.copy()
        )
        snapshot.calculate_total_value()
        self.portfolio_history.append(snapshot)

    def _calculate_performance_metrics(self, benchmark_results: Optional[List[BenchmarkResult]] = None) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        if not self.portfolio_history:
            return PerformanceMetrics()

        # Get portfolio values over time
        values = [snapshot.total_value for snapshot in self.portfolio_history]
        timestamps = [snapshot.timestamp for snapshot in self.portfolio_history]

        if len(values) < 2:
            return PerformanceMetrics()

        # Calculate returns
        returns = pd.Series(values, index=timestamps).pct_change().dropna()

        # Basic metrics
        total_return = (values[-1] - values[0]) / values[0]
        days = (timestamps[-1] - timestamps[0]).days
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        volatility = returns.std() * np.sqrt(252)  # Annualized

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0

        # Trading metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.side == OrderSide.SELL])  # Simplified
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Calculate trade P&Ls (simplified)
        trade_pnls = []
        for trade in self.trades:
            if trade.side == OrderSide.SELL:
                # Calculate P&L for sell trades
                pnl = (trade.price - self._get_avg_buy_price(trade.symbol)) * trade.quantity - trade.commission
                trade_pnls.append(pnl)

        avg_win = np.mean([p for p in trade_pnls if p > 0]) if any(p > 0 for p in trade_pnls) else 0
        avg_loss = abs(np.mean([p for p in trade_pnls if p < 0])) if any(p < 0 for p in trade_pnls) else 0
        profit_factor = sum(p for p in trade_pnls if p > 0) / abs(sum(p for p in trade_pnls if p < 0)) if sum(p < 0 for p in trade_pnls) != 0 else float('inf')

        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Benchmark comparison (use first benchmark if available)
        benchmark_return = 0.0
        alpha = 0.0
        beta = 0.0
        tracking_error = 0.0
        information_ratio = 0.0

        if benchmark_results and len(benchmark_results) > 0:
            # Use the first benchmark (typically S&P 500)
            primary_benchmark = benchmark_results[0]
            benchmark_return = primary_benchmark.benchmark_return
            alpha = primary_benchmark.alpha
            beta = primary_benchmark.beta
            tracking_error = primary_benchmark.tracking_error
            information_ratio = primary_benchmark.information_ratio

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            benchmark_return=benchmark_return,
            alpha=alpha,
            beta=beta,
            tracking_error=tracking_error,
            information_ratio=information_ratio
        )

    def _get_avg_buy_price(self, symbol: str) -> float:
        """Get average buy price for a symbol (simplified)."""
        if symbol in self.current_portfolio:
            return self.current_portfolio[symbol].avg_price
        return 0.0

    def _perform_benchmarking_analysis(self, start_date: str, end_date: str,
                                     benchmark_symbols: List[str]) -> List[BenchmarkResult]:
        """
        Perform benchmarking analysis against market indices.

        Args:
            start_date: Analysis start date
            end_date: Analysis end date
            benchmark_symbols: List of benchmark symbols

        Returns:
            List of benchmark analysis results
        """
        if not self.portfolio_history:
            return []

        # Calculate strategy returns from portfolio history
        portfolio_values = pd.Series(
            [snapshot.total_value for snapshot in self.portfolio_history],
            index=[snapshot.timestamp for snapshot in self.portfolio_history]
        )

        # Calculate daily returns
        strategy_returns = portfolio_values.pct_change().dropna()

        # Initialize benchmark analyzer
        benchmark_config = {
            'risk_free_rate': 0.02,
            'cache_dir': self.config.get('benchmark_cache_dir', 'data/benchmark_cache')
        }
        analyzer = BenchmarkAnalyzer(benchmark_config)

        # Perform benchmarking against each symbol
        benchmark_results = []
        for benchmark_symbol in benchmark_symbols:
            try:
                result = analyzer.benchmark_strategy(
                    strategy_returns, benchmark_symbol, start_date, end_date,
                    strategy_name=self.current_strategy_name if hasattr(self, 'current_strategy_name') else "Strategy"
                )
                benchmark_results.append(result)
                self.logger.info(f"Benchmark analysis completed for {benchmark_symbol}")
            except Exception as e:
                self.logger.error(f"Failed to benchmark against {benchmark_symbol}: {e}")

        return benchmark_results

    def generate_performance_report(self, output_dir: str = "reports") -> Dict[str, Any]:
        """
        Generate comprehensive performance report with visualizations.
        
        Args:
            output_dir: Directory to save report files
            
        Returns:
            Dictionary with report paths and summary
        """
        os.makedirs(output_dir, exist_ok=True)
        report_files = {}
        
        if not self.portfolio_history:
            self.logger.warning("No portfolio history available for reporting")
            return {"error": "No portfolio history available"}
        
        # Generate equity curve chart
        equity_chart_path = self._generate_equity_curve_chart(output_dir)
        if equity_chart_path:
            report_files['equity_curve'] = equity_chart_path
        
        # Generate drawdown chart
        drawdown_chart_path = self._generate_drawdown_chart(output_dir)
        if drawdown_chart_path:
            report_files['drawdown_chart'] = drawdown_chart_path
        
        # Generate returns distribution
        returns_chart_path = self._generate_returns_distribution_chart(output_dir)
        if returns_chart_path:
            report_files['returns_distribution'] = returns_chart_path
        
        # Generate trade analysis
        trade_analysis_path = self._generate_trade_analysis_chart(output_dir)
        if trade_analysis_path:
            report_files['trade_analysis'] = trade_analysis_path
        
        # Generate summary report
        summary_path = self._generate_summary_report(output_dir)
        if summary_path:
            report_files['summary_report'] = summary_path
        
        return {
            'status': 'success',
            'files': report_files,
            'total_files': len(report_files)
        }
    
    def _generate_equity_curve_chart(self, output_dir: str) -> Optional[str]:
        """Generate equity curve visualization."""
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available for chart generation")
            return None
        
        try:
            # Extract portfolio values
            timestamps = [snapshot.timestamp for snapshot in self.portfolio_history]
            values = [snapshot.total_value for snapshot in self.portfolio_history]
            
            # Create equity curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timestamps, 
                y=values,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ))
            
            # Add initial capital line
            fig.add_hline(
                y=self.initial_cash,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Initial Capital: ${self.initial_cash:,.0f}"
            )
            
            fig.update_layout(
                title="Portfolio Equity Curve",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                template="plotly_white"
            )
            
            # Save chart
            chart_path = os.path.join(output_dir, "equity_curve.html")
            fig.write_html(chart_path)
            
            return chart_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate equity curve chart: {e}")
            return None
    
    def _generate_drawdown_chart(self, output_dir: str) -> Optional[str]:
        """Generate drawdown visualization."""
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available for chart generation")
            return None
        
        try:
            # Calculate drawdown
            values = pd.Series([snapshot.total_value for snapshot in self.portfolio_history])
            rolling_max = values.expanding().max()
            drawdown = (values - rolling_max) / rolling_max * 100
            
            timestamps = [snapshot.timestamp for snapshot in self.portfolio_history]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=drawdown,
                fill='tozeroy',
                mode='lines',
                name='Drawdown',
                line=dict(color='red', width=1)
            ))
            
            fig.update_layout(
                title="Portfolio Drawdown",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                template="plotly_white"
            )
            
            # Save chart
            chart_path = os.path.join(output_dir, "drawdown_chart.html")
            fig.write_html(chart_path)
            
            return chart_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate drawdown chart: {e}")
            return None
    
    def _generate_returns_distribution_chart(self, output_dir: str) -> Optional[str]:
        """Generate returns distribution histogram."""
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available for chart generation")
            return None
        
        try:
            # Calculate daily returns
            values = pd.Series([snapshot.total_value for snapshot in self.portfolio_history])
            daily_returns = values.pct_change().dropna() * 100
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=daily_returns,
                nbinsx=50,
                name='Daily Returns',
                marker_color='lightblue'
            ))
            
            # Add normal distribution overlay
            mean_return = daily_returns.mean()
            std_return = daily_returns.std()
            
            fig.add_vline(
                x=mean_return,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_return:.2f}%"
            )
            
            fig.update_layout(
                title="Daily Returns Distribution",
                xaxis_title="Daily Return (%)",
                yaxis_title="Frequency",
                template="plotly_white"
            )
            
            # Save chart
            chart_path = os.path.join(output_dir, "returns_distribution.html")
            fig.write_html(chart_path)
            
            return chart_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate returns distribution chart: {e}")
            return None
    
    def _generate_trade_analysis_chart(self, output_dir: str) -> Optional[str]:
        """Generate trade analysis visualization."""
        if not PLOTLY_AVAILABLE or not self.trades:
            self.logger.warning("Plotly not available or no trades for analysis")
            return None
        
        try:
            # Convert trades to DataFrame for analysis
            trades_df = pd.DataFrame([{
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'side': trade.side.value,
                'quantity': trade.quantity,
                'price': trade.price,
                'commission': trade.commission,
                'total_cost': trade.total_cost
            } for trade in self.trades])
            
            # Create subplot figure
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Trade Volume Over Time", "Trade P&L Distribution", 
                              "Trades by Symbol", "Commission Costs"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                      [{"type": "domain"}, {"secondary_y": False}]]
            )
            
            # Trade volume over time
            daily_volume = trades_df.groupby(trades_df['timestamp'].dt.date)['quantity'].sum()
            fig.add_trace(
                go.Bar(x=daily_volume.index, y=daily_volume.values, name="Daily Volume"),
                row=1, col=1
            )
            
            # Trade P&L distribution (simplified - would need position tracking)
            if len(self.trades) > 1:
                # Calculate approximate P&L per trade (simplified)
                pnl_values = []
                for trade in self.trades:
                    if trade.side == OrderSide.SELL:
                        pnl_values.append(0)  # Would need to track entry/exit pairs
                
                if pnl_values:
                    fig.add_trace(
                        go.Histogram(x=pnl_values, name="Trade P&L", marker_color='green'),
                        row=1, col=2
                    )
            
            # Trades by symbol
            symbol_counts = trades_df['symbol'].value_counts()
            fig.add_trace(
                go.Pie(labels=symbol_counts.index, values=symbol_counts.values, name="Trades by Symbol"),
                row=2, col=1
            )
            
            # Commission costs over time
            fig.add_trace(
                go.Scatter(x=trades_df['timestamp'], y=trades_df['commission'], 
                          mode='lines+markers', name="Commission per Trade"),
                row=2, col=2
            )
            
            fig.update_layout(title="Trade Analysis Dashboard", template="plotly_white")
            
            # Save chart
            chart_path = os.path.join(output_dir, "trade_analysis.html")
            fig.write_html(chart_path)
            
            return chart_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate trade analysis chart: {e}")
            return None
    
    def _generate_summary_report(self, output_dir: str) -> Optional[str]:
        """Generate text summary report."""
        try:
            if not self.portfolio_history:
                return None
            
            final_value = self.portfolio_history[-1].total_value
            total_return = (final_value - self.initial_cash) / self.initial_cash * 100
            
            # Calculate metrics
            values = pd.Series([snapshot.total_value for snapshot in self.portfolio_history])
            daily_returns = values.pct_change().dropna()
            
            volatility = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 0 else 0
            max_drawdown = ((values - values.expanding().max()) / values.expanding().max()).min() * 100
            
            # Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02
            excess_returns = daily_returns - risk_free_rate/252
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
            
            report_content = f"""
# Backtest Performance Summary

## Overview
- Initial Capital: ${self.initial_cash:,.2f}
- Final Value: ${final_value:,.2f}
- Total Return: {total_return:.2f}%
- Total Trades: {len(self.trades)}

## Risk Metrics
- Annualized Volatility: {volatility:.2f}%
- Maximum Drawdown: {max_drawdown:.2f}%
- Sharpe Ratio: {sharpe_ratio:.2f}

## Trading Costs
- Total Commissions: ${sum(trade.commission for trade in self.trades):,.2f}
- Total Slippage: ${sum(trade.slippage for trade in self.trades):,.2f}
- Total Trading Costs: ${sum(trade.commission + trade.slippage for trade in self.trades):,.2f}

## Trade Analysis
- Buy Trades: {len([t for t in self.trades if t.side == OrderSide.BUY])}
- Sell Trades: {len([t for t in self.trades if t.side == OrderSide.SELL])}
- Average Commission per Trade: ${np.mean([t.commission for t in self.trades]):.2f}
- Average Slippage per Trade: ${np.mean([t.slippage for t in self.trades]):.2f}

Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            report_path = os.path.join(output_dir, "backtest_summary.md")
            with open(report_path, 'w') as f:
                f.write(report_content)
            
            return report_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary report: {e}")
            return None


class WalkForwardAnalyzer:
    """
    Walk-forward analysis for out-of-sample testing.

    This class implements walk-forward optimization to prevent overfitting
    by testing strategies on out-of-sample data.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.train_window = config.get('train_window_months', 12)
        self.test_window = config.get('test_window_months', 3)
        self.step_size = config.get('step_size_months', 1)

    def run_walk_forward_analysis(self, strategy_factory: Callable, data: Dict[str, pd.DataFrame], start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Run walk-forward analysis.

        Args:
            strategy_factory: Function that creates strategy instances
            data: Historical market data
            start_date: Analysis start date
            end_date: Analysis end date

        Returns:
            Dictionary with walk-forward results
        """
        self.logger.info("Starting walk-forward analysis")

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        results = []
        current_train_start = start_dt

        while current_train_start + pd.DateOffset(months=self.train_window + self.test_window) <= end_dt:
            # Define training period
            train_end = current_train_start + pd.DateOffset(months=self.train_window)
            test_end = train_end + pd.DateOffset(months=self.test_window)

            self.logger.info(f"Walk-forward step: Train {current_train_start.date()} to {train_end.date()}, Test {train_end.date()} to {test_end.date()}")

            # Optimize strategy on training data
            optimized_strategy = self._optimize_strategy(strategy_factory, data, current_train_start, train_end)

            # Test on out-of-sample data
            backtest_config = self.config.copy()
            backtest_config['initial_cash'] = 100000.0

            engine = BacktestEngine(backtest_config)
            test_result = engine.run_backtest(optimized_strategy, data, train_end.strftime('%Y-%m-%d'), test_end.strftime('%Y-%m-%d'))

            if test_result['status'] == 'success':
                results.append({
                    'train_start': current_train_start,
                    'train_end': train_end,
                    'test_start': train_end,
                    'test_end': test_end,
                    'result': test_result
                })

            # Move to next step
            current_train_start += pd.DateOffset(months=self.step_size)

        # Aggregate results
        summary = self._summarize_walk_forward_results(results)

        return {
            'status': 'success',
            'total_steps': len(results),
            'results': results,
            'summary': summary
        }

    def _optimize_strategy(self, strategy_factory: Callable, data: Dict[str, pd.DataFrame],
                          start_date: pd.Timestamp, end_date: pd.Timestamp) -> BacktestStrategy:
        """Optimize strategy parameters on training data (simplified)."""
        # For now, just return a default strategy
        # In a real implementation, this would optimize parameters
        return strategy_factory({})

    def _summarize_walk_forward_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Summarize walk-forward analysis results."""
        if not results:
            return {}

        returns = [r['result']['total_return'] for r in results]
        sharpe_ratios = [r['result']['metrics'].sharpe_ratio for r in results]

        return {
            'avg_return': np.mean(returns),
            'std_return': np.std(returns),
            'avg_sharpe': np.mean(sharpe_ratios),
            'std_sharpe': np.std(sharpe_ratios),
            'positive_periods': sum(1 for r in returns if r > 0),
            'total_periods': len(returns),
            'win_rate': sum(1 for r in returns if r > 0) / len(returns)
        }


# Convenience functions
def run_backtest(strategy: BacktestStrategy, symbols: List[str], start_date: str, end_date: str,
                config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to run a single backtest.

    Args:
        strategy: Trading strategy to test
        symbols: List of symbols to trade
        start_date: Backtest start date
        end_date: Backtest end date
        config: Backtest configuration

    Returns:
        Backtest results
    """
    if config is None:
        config = {
            'initial_cash': 100000.0,
            'commission_per_trade': 5.0,
            'slippage_bps': 5.0
        }

    engine = BacktestEngine(config)
    data = engine.load_historical_data(symbols, start_date, end_date)

    return engine.run_backtest(strategy, data, start_date, end_date)


async def compare_strategies(strategies: List[BacktestStrategy], symbols: List[str], start_date: str, end_date: str,
                      config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Compare multiple strategies on the same data.

    Args:
        strategies: List of strategies to compare
        symbols: List of symbols to trade
        start_date: Backtest start date
        end_date: Backtest end date
        config: Backtest configuration

    Returns:
        Comparison results
    """
    results = []

    for strategy in strategies:
        result = await run_backtest(strategy, symbols, start_date, end_date, config)
        if result['status'] == 'success':
            results.append(result)

    if not results:
        return {'status': 'failed', 'error': 'No successful backtests'}

    # Sort by Sharpe ratio
    results.sort(key=lambda x: x['metrics'].sharpe_ratio, reverse=True)

    return {
        'status': 'success',
        'best_strategy': results[0]['strategy_name'],
        'results': results,
        'comparison': {
            'sharpe_ratios': [r['metrics'].sharpe_ratio for r in results],
            'total_returns': [r['total_return'] for r in results],
            'max_drawdowns': [r['metrics'].max_drawdown for r in results]
        }
    }


# Convenience functions for easy backtesting
async def run_backtest(strategy: BacktestStrategy, symbols: List[str], start_date: str,
                end_date: str, config: Dict[str, Any], benchmark_symbols: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Convenience function to run a single backtest.

    Args:
        strategy: Trading strategy to test
        symbols: List of symbols to trade
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        config: Backtest configuration
        benchmark_symbols: Optional list of benchmark symbols

    Returns:
        Backtest results dictionary
    """
    try:
        # Create backtest engine
        engine = BacktestEngine(config)

        # Load historical data
        data = engine.load_historical_data(symbols, start_date, end_date)

        if not data:
            return {'status': 'failed', 'error': 'No data available for backtesting'}

        # Run backtest
        return await engine.run_backtest(strategy, data, start_date, end_date, benchmark_symbols)

    except Exception as e:
        return {'status': 'failed', 'error': str(e)}


async def compare_strategies(strategies: List[Tuple[str, BacktestStrategy]], symbols: List[str],
                      start_date: str, end_date: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare multiple strategies in backtesting.

    Args:
        strategies: List of (name, strategy) tuples
        symbols: List of symbols to trade
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        config: Backtest configuration

    Returns:
        Comparison results
    """
    results = []

    for name, strategy in strategies:
        result = await run_backtest(strategy, symbols, start_date, end_date, config)
        if result['status'] == 'success':
            results.append(result)

    if not results:
        return {'status': 'failed', 'error': 'No successful backtests'}

    # Sort by Sharpe ratio
    results.sort(key=lambda x: x['metrics'].sharpe_ratio, reverse=True)

    return {
        'status': 'success',
        'best_strategy': results[0]['strategy_name'],
        'results': results,
        'comparison': {
            'sharpe_ratios': [r['metrics'].sharpe_ratio for r in results],
            'total_returns': [r['total_return'] for r in results],
            'max_drawdowns': [r['metrics'].max_drawdown for r in results]
        }
    }