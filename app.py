"""
Advanced Trading Framework - Main Application

This module implements the main Streamlit application that provides a comprehensive
UI for the trading framework, integrating all components: brokers, portfolios,
strategies, and real-time monitoring.

Author: Senior Python Software Architect
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import logging
import logging.handlers
import yaml
import os
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import time

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not available. Install with: pip install python-dotenv")

# Framework imports
from core.portfolio import Portfolio
from core.position import Position
from execution_layer.paper_broker import PaperBroker
from execution_layer.alpaca_broker import AlpacaBroker
from strategy_layer.ml_random_forest_strategy import MLRandomForestStrategy
from strategy_layer.signals import TradingSignal
from core.config_validator import ConfigValidator
from core.monitoring import health_checker, metrics_collector, record_trading_metric, get_system_health, get_system_metrics

# Plotting
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Install with: pip install plotly")

# Data handling
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except (ImportError, TypeError, Exception) as e:
    YFINANCE_AVAILABLE = False
    yf = None
    print(f"Warning: yfinance not available in app.py ({e}). Using simulated data.")

# Configure Streamlit page - must be first Streamlit command
st.set_page_config(
    page_title="Advanced Trading Framework",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_custom_css():
    """Apply custom CSS for dark mode and improved styling."""

    # Check if dark mode is enabled
    dark_mode = st.session_state.get('dark_mode', False)

    if dark_mode:
        # Dark mode CSS
        st.markdown("""
        <style>
        /* Dark mode variables */
        :root {
            --bg-primary: #0e1117;
            --bg-secondary: #262730;
            --bg-tertiary: #31333f;
            --text-primary: #fafafa;
            --text-secondary: #c4c4c4;
            --accent-green: #00ff88;
            --accent-red: #ff6b6b;
            --accent-blue: #4dabf7;
            --accent-yellow: #ffd43b;
            --border-color: #464853;
        }

        /* Main app background */
        .stApp {
            background-color: var(--bg-primary);
            color: var(--text-primary);
        }

        /* Sidebar styling */
        .css-1d391kg {
            background-color: var(--bg-secondary);
        }

        /* Metrics containers */
        div[data-testid="metric-container"] {
            background-color: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }

        /* Enhanced metric styling */
        div[data-testid="metric-container"] > div {
            color: var(--text-primary);
        }

        /* Positive metrics (green) */
        div[data-testid="metric-container"]:has([data-testid="metric-value"]:contains("+")):not(:has([data-testid="metric-value"]:contains("-"))) {
            border-left: 4px solid var(--accent-green);
        }

        /* Negative metrics (red) */
        div[data-testid="metric-container"]:has([data-testid="metric-value"]:contains("-")) {
            border-left: 4px solid var(--accent-red);
        }

        /* Containers and expanders */
        .element-container {
            background-color: var(--bg-secondary);
            border-radius: 8px;
            padding: 0.5rem;
            margin: 0.25rem 0;
        }

        /* Dataframes */
        div[data-testid="stDataFrame"] {
            background-color: var(--bg-tertiary);
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }

        /* Charts */
        div[data-testid="stPlotlyChart"] {
            background-color: var(--bg-tertiary);
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid var(--border-color);
        }

        /* Buttons */
        .stButton > button {
            background-color: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            background-color: var(--accent-blue);
            border-color: var(--accent-blue);
        }

        /* Primary buttons */
        .stButton > button[kind="primary"] {
            background-color: var(--accent-blue);
            border-color: var(--accent-blue);
        }

        /* Selectbox and inputs */
        .stSelectbox > div > div {
            background-color: var(--bg-tertiary);
            border-color: var(--border-color);
        }

        /* Headers with trend indicators */
        .metric-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-weight: 600;
        }

        .trend-up {
            color: var(--accent-green);
        }

        .trend-down {
            color: var(--accent-red);
        }

        .trend-neutral {
            color: var(--text-secondary);
        }

        /* Section containers */
        .section-container {
            background-color: var(--bg-secondary);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid var(--border-color);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        /* Success/Error toast styling */
        .stToast {
            background-color: var(--bg-tertiary);
            border-radius: 8px;
            border-left: 4px solid var(--accent-green);
        }

        /* Scrollbars */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: var(--bg-primary);
        }

        ::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-secondary);
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        # Light mode CSS with improved styling
        st.markdown("""
        <style>
        /* Light mode variables */
        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f8f9fa;
            --bg-tertiary: #e9ecef;
            --text-primary: #212529;
            --text-secondary: #6c757d;
            --accent-green: #28a745;
            --accent-red: #dc3545;
            --accent-blue: #007bff;
            --accent-yellow: #ffc107;
            --border-color: #dee2e6;
        }

        /* Metrics containers */
        div[data-testid="metric-container"] {
            background-color: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }

        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }

        /* Positive metrics (green) */
        div[data-testid="metric-container"]:has([data-testid="metric-value"]:contains("+")):not(:has([data-testid="metric-value"]:contains("-"))) {
            border-left: 4px solid var(--accent-green);
        }

        /* Negative metrics (red) */
        div[data-testid="metric-container"]:has([data-testid="metric-value"]:contains("-")) {
            border-left: 4px solid var(--accent-red);
        }

        /* Section containers */
        .section-container {
            background-color: var(--bg-secondary);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid var(--border-color);
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        /* Charts */
        div[data-testid="stPlotlyChart"] {
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid var(--border-color);
        }

        /* Dataframes */
        div[data-testid="stDataFrame"] {
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }

        /* Headers with trend indicators */
        .metric-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-weight: 600;
        }

        .trend-up {
            color: var(--accent-green);
        }

        .trend-down {
            color: var(--accent-red);
        }

        .trend-neutral {
            color: var(--text-secondary);
        }
        </style>
        """, unsafe_allow_html=True)

# Apply CSS on every run
apply_custom_css()

# Configure logging globally
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add file handler for persistent logging
log_file_path = "logs/trading.log"
os.makedirs('logs', exist_ok=True)
# Use TimedRotatingFileHandler for daily rotation, keeping 7 backup files
file_handler = logging.handlers.TimedRotatingFileHandler(
    log_file_path,
    when='midnight',  # Rotate at midnight
    backupCount=7     # Keep 7 backup files
)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)


@st.cache_resource
def initialize_trading_app():
    """
    Initialize the trading application once and cache it.
    This prevents re-initialization on every Streamlit rerun.
    """
    app = TradingApp()
    asyncio.run(app.initialize())
    return app


def initialize_session_state():
    """Initialize all session state variables atomically."""
    # Check if already initialized to avoid re-initialization on reruns
    if '_session_initialized' in st.session_state:
        return

    # Initialize all session state variables in a single batch operation
    # This ensures atomicity and prevents partial initialization
    initial_state = {
        '_session_initialized': True,
        'app_initialized': False,
        'portfolio_data': {},
        'signals_data': [],
        'performance_data': {},
        'auto_trading_enabled': False,
        'last_update': 0,
        'dark_mode': False,
        'selected_tickers': [],
        'trading_active': False,
        'log_messages': [],
        'error_messages': [],
        'config_validated': False,
        'broker_connected': False,
        'last_portfolio_update': 0,
        'last_signals_update': 0,
        'last_performance_update': 0
    }

    # Use st.session_state.update() for atomic batch update
    st.session_state.update(initial_state)


class TradingApp:
    """
    Main Trading Application class that orchestrates all components.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the trading application."""
        self.config_path = config_path
        self.logger = self._setup_logging()  # Initialize with defaults first
        self.config = self._load_config()
        
        # Update logging with actual config
        self.logger = self._setup_logging(self.config)
        
        # Log any config warnings that were stored during loading
        if hasattr(self, '_config_warnings') and self._config_warnings:
            for warning in self._config_warnings[:3]:  # Log first 3 warnings
                self.logger.warning(f"Config warning - {warning.field_path}: {warning.message}")
        
        # Initialize components
        self.portfolio: Optional[Portfolio] = None
        self.broker = None
        self.strategies: List = []
        self.signals_history: List[TradingSignal] = []
        
        # App state
        self.initialized = False
        self.running = False
        self.last_update = None
        
        # Automation control
        self.auto_trading_enabled = False
        self.automation_loop: Optional[asyncio.AbstractEventLoop] = None
        self.automation_thread: Optional[threading.Thread] = None
        self.trading_loop_task: Optional[asyncio.Task] = None
        self.risk_monitoring_task: Optional[asyncio.Task] = None
        
        # Monitoring
        self.monitoring_enabled = True
        self.health_check_interval = 60  # seconds
        self.metrics_collection_interval = 30  # seconds
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Streamlit session state management
        self._init_session_state()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.cleanup()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with validation."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Validate configuration
                validator = ConfigValidator(self.config_path)
                is_valid, issues = validator.validate_config_file()
                
                if not is_valid:
                    error_messages = []
                    for issue in issues:
                        if issue.severity.name == 'ERROR':
                            error_messages.append(f"{issue.field_path}: {issue.message}")
                    
                    if error_messages:
                        st.error("Configuration validation failed:")
                        for msg in error_messages[:5]:  # Show first 5 errors
                            st.error(f"• {msg}")
                        if len(error_messages) > 5:
                            st.error(f"... and {len(error_messages) - 5} more errors")
                        st.error("Please fix the configuration issues and restart the application.")
                        return {}
                
                # Log warnings if any
                warnings = [i for i in issues if i.severity.name == 'WARNING']
                if warnings:
                    st.warning(f"Configuration has {len(warnings)} warning(s). Check logs for details.")
                    # Store warnings to log later when logger is available
                    self._config_warnings = warnings
                
                return config if config else {}
            else:
                st.error(f"Configuration file not found: {self.config_path}")
                return {}
        except Exception as e:
            st.error(f"Failed to load configuration: {str(e)}")
            return {}
    
    def _setup_logging(self, config: Optional[Dict[str, Any]] = None) -> logging.Logger:
        """Setup logging based on configuration."""
        if config is None:
            config = {}
            
        logger = logging.getLogger("TradingApp")
        
        # Only setup logging if it hasn't been set up before at the module level
        if not hasattr(self.__class__, '_logging_initialized'):
            # Create logs directory
            log_config = config.get('logging', {})
            if log_config.get('file', {}).get('enabled', True):
                os.makedirs('logs', exist_ok=True)
            
            # Configure logging
            logging.basicConfig(
                level=getattr(logging, log_config.get('console', {}).get('level', 'INFO')),
                format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            
            # Add file handler if enabled and not already added
            if log_config.get('file', {}).get('enabled', True):
                file_path = log_config.get('file', {}).get('path', 'logs/trading.log')
                # Check if file handler already exists
                root_logger = logging.getLogger()
                file_handler_exists = any(isinstance(h, (logging.FileHandler, logging.handlers.TimedRotatingFileHandler)) and 
                                        hasattr(h, 'baseFilename') and h.baseFilename.endswith(file_path) 
                                        for h in root_logger.handlers)
                
                if not file_handler_exists:
                    # Use TimedRotatingFileHandler for daily log rotation
                    backup_count = log_config.get('file', {}).get('backup_count', 7)
                    
                    file_handler = logging.handlers.TimedRotatingFileHandler(
                        file_path,
                        when='midnight',  # Rotate at midnight
                        backupCount=backup_count  # Keep this many backup files
                    )
                    file_handler.setFormatter(logging.Formatter(log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')))
                    root_logger.addHandler(file_handler)
            
            # Mark as initialized
            self.__class__._logging_initialized = True
        
        return logger
    
    def _init_session_state(self):
        """Initialize Streamlit session state variables."""
        # This is now handled globally, but keeping for compatibility
        pass
    
    async def initialize(self) -> bool:
        """Initialize all trading components."""
        try:
            self.logger.info("Initializing Trading Application")
            
            # Initialize Portfolio
            portfolio_config = self.config.get('portfolio', {})
            self.portfolio = Portfolio(
                initial_capital=portfolio_config.get('initial_cash', 100000.0)
            )
            
            # Load existing portfolio state if available
            try:
                self.portfolio.load_state()
                self.logger.info("Portfolio state loaded successfully")
            except Exception as e:
                self.logger.warning(f"Could not load portfolio state (first run?): {str(e)}")
            
            # Initialize Broker
            broker_config = self.config.get('broker', {})
            trading_environment = self.config.get('trading_environment', 'paper')
            
            if trading_environment == 'paper':
                paper_config = broker_config.get('paper', {})
                self.broker = PaperBroker(
                    initial_cash=paper_config.get('initial_cash', 100000.0),
                    commission_per_trade=paper_config.get('commission', 0.0),
                    slippage_bps=paper_config.get('slippage', 0.001) * 10000  # Convert to basis points
                )
            elif trading_environment == 'live':
                alpaca_config = broker_config.get('alpaca', {})
                self.broker = AlpacaBroker(
                    api_key=os.getenv('ALPACA_API_KEY'),
                    secret_key=os.getenv('ALPACA_SECRET_KEY'),
                    base_url=alpaca_config.get('base_url', 'https://api.alpaca.markets'),  # Live trading URL
                    paper=False  # Explicitly set to live trading
                )
            else:
                st.error(f"Unknown trading environment: {trading_environment}. Must be 'paper' or 'live'")
                return False
            
            # Initialize broker
            if not await self.broker.connect():
                st.error("Failed to initialize broker")
                return False
            
            # Sync portfolio with broker if using Alpaca
            if trading_environment == 'live':
                await self._sync_portfolio_with_alpaca()
            
            # Initialize Strategies
            strategy_config = self.config.get('strategy', {})
            active_strategies = strategy_config.get('active_strategies', ['MLRandomForestStrategy'])
            
            for strategy_name in active_strategies:
                if strategy_name == 'MLRandomForestStrategy':
                    ml_config = strategy_config.get('ml_random_forest', {})
                    strategy = MLRandomForestStrategy(
                        confidence_threshold=ml_config.get('confidence_threshold', 0.6)
                    )
                    
                    if await strategy.initialize():
                        self.strategies.append(strategy)
                        self.logger.info(f"Initialized strategy: {strategy_name}")
                    else:
                        self.logger.warning(f"Failed to initialize strategy: {strategy_name}")
                        # Clean up resources if initialization failed
                        strategy.cleanup()
            
            if not self.strategies:
                st.warning("No strategies initialized")
            
            self.initialized = True
            st.session_state.app_initialized = True
            self.logger.info("Trading Application initialized successfully")
            
            # Start monitoring
            await self.start_monitoring()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize trading application: {str(e)}")
            st.error(f"Initialization failed: {str(e)}")
            return False
    
    async def _sync_portfolio_with_alpaca(self):
        """Sync portfolio with real Alpaca account data."""
        try:
            self.logger.info("Syncing portfolio with Alpaca account")
            
            # Get real account info
            account_info = await self.broker.get_account_info()
            if account_info:
                # Update portfolio cash with real account cash
                self.portfolio.cash = float(account_info.cash)
                self.logger.info(f"Updated cash from Alpaca: ${account_info.cash}")
            
            # Get real positions
            alpaca_positions = await self.broker.get_positions()
            if alpaca_positions:
                # Clear existing positions and add real ones
                self.portfolio.positions = {}
                for symbol, quantity in alpaca_positions.items():
                    if quantity != 0:  # Only add non-zero positions
                        # Get current market price for the position
                        market_data = await self.broker.get_market_data(symbol)
                        if market_data:
                            position = Position(
                                ticker=symbol,
                                quantity=quantity,
                                avg_cost=market_data.price,  # Use current price as approximation
                                timestamp=datetime.now()
                            )
                            self.portfolio.add_position(position)
                            self.logger.info(f"Added position from Alpaca: {symbol} x {quantity}")
            
            # Save updated portfolio state
            self.portfolio.save_state()
            self.logger.info("Portfolio synced with Alpaca account successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to sync portfolio with Alpaca: {str(e)}")
    
    async def generate_signals(self, tickers: List[str]) -> List[TradingSignal]:
        """Generate trading signals from all strategies."""
        all_signals = []
        
        for strategy in self.strategies:
            try:
                signals = await strategy.generate_signals(tickers)
                all_signals.extend(signals)
                self.logger.info(f"Generated {len(signals)} signals from {strategy.name}")
            except Exception as e:
                self.logger.error(f"Failed to generate signals from {strategy.name}: {str(e)}")
        
        # Update signals history in session state
        self.signals_history.extend(all_signals)
        if 'signals_data' not in st.session_state:
            st.session_state.signals_data = []
        st.session_state.signals_data.extend([signal.to_dict() for signal in all_signals])
        # Keep only last 100 signals
        st.session_state.signals_data = st.session_state.signals_data[-100:]
        
        return all_signals
    
    async def execute_signals(self, signals: List[TradingSignal]) -> int:
        """Execute trading signals through the broker."""
        executed_count = 0
        
        for signal in signals:
            try:
                if signal.action.upper() == 'BUY':
                    # Check if position already exists
                    if self.portfolio.get_position(signal.ticker) is not None:
                        self.logger.info(f"Position for {signal.ticker} already exists, skipping BUY signal")
                        continue
                    
                    # Calculate position size based on portfolio
                    portfolio_value = self.portfolio.calculate_total_equity()
                    max_position_value = portfolio_value * self.config.get('portfolio', {}).get('max_position_size', 0.1)
                    
                    if signal.price:
                        shares = int(max_position_value / signal.price)
                        if shares > 0:
                            # Create and submit order
                            from execution_layer.base_broker import Order
                            order = Order(
                                symbol=signal.ticker,
                                side='buy',
                                quantity=shares,
                                order_type='market'
                            )
                            
                            order_id = await self.broker.submit_order(order)
                            if order_id:
                                # Use stop loss and take profit from signal, or fall back to defaults
                                stop_loss_price = signal.stop_loss
                                take_profit_price = signal.take_profit
                                
                                if stop_loss_price is None or take_profit_price is None:
                                    # Fall back to config defaults if signal doesn't have stop levels
                                    risk_config = self.config.get('portfolio', {})
                                    stop_loss_pct = risk_config.get('stop_loss_default', 0.05)
                                    take_profit_pct = risk_config.get('take_profit_default', 0.1)
                                    
                                    stop_loss_price = signal.price * (1 - stop_loss_pct)
                                    take_profit_price = signal.price * (1 + take_profit_pct)
                                
                                # Update portfolio to reflect new position
                                self.portfolio.open_position(
                                    ticker=signal.ticker,
                                    quantity=shares,
                                    entry_price=signal.price,
                                    side='LONG',
                                    sl_price=stop_loss_price,
                                    tp_price=take_profit_price
                                )
                                executed_count += 1
                                self.logger.info(f"Executed BUY order for {shares} shares of {signal.ticker}")
                
                elif signal.action.upper() == 'SELL':
                    # Check if we have position to sell
                    position = self.portfolio.get_position(signal.ticker)
                    if position and position.quantity > 0:
                        # Create and submit order
                        from execution_layer.base_broker import Order
                        order = Order(
                            symbol=signal.ticker,
                            side='sell',
                            quantity=position.quantity,
                            order_type='market'
                        )
                        
                        order_id = await self.broker.submit_order(order)
                        if order_id:
                            # Update portfolio to reflect position closure
                            self.portfolio.close_position(signal.ticker, signal.price or position.current_price)
                            executed_count += 1
                            self.logger.info(f"Executed SELL order for {position.quantity} shares of {signal.ticker}")
            
            except Exception as e:
                self.logger.error(f"Failed to execute signal for {signal.ticker}: {str(e)}")
        
        return executed_count
    
    async def monitor_risk_management(self):
        """Monitor all open positions for stop-loss and take-profit triggers."""
        try:
            if not self.portfolio:
                return
            
            open_positions = self.portfolio.get_all_open_positions()
            
            for position in open_positions:
                # Get current market price
                market_data = await self.broker.get_market_data(position.ticker)
                if not market_data:
                    continue
                
                current_price = market_data.price
                position.update_market_price(current_price)
                
                # Check stop-loss trigger
                if position.should_stop_loss():
                    self.logger.warning(f"Stop-loss triggered for {position.ticker} at ${current_price}")
                    
                    # Create sell order
                    from execution_layer.base_broker import Order
                    order = Order(
                        symbol=position.ticker,
                        side='sell',
                        quantity=position.quantity,
                        order_type='market'
                    )
                    
                    order_id = await self.broker.submit_order(order)
                    if order_id:
                        self.portfolio.close_position(position.ticker, current_price)
                        self.logger.info(f"Stop-loss executed for {position.ticker}")
                
                # Check take-profit trigger
                elif position.should_take_profit():
                    self.logger.info(f"Take-profit triggered for {position.ticker} at ${current_price}")
                    
                    # Create sell order
                    from execution_layer.base_broker import Order
                    order = Order(
                        symbol=position.ticker,
                        side='sell',
                        quantity=position.quantity,
                        order_type='market'
                    )
                    
                    order_id = await self.broker.submit_order(order)
                    if order_id:
                        self.portfolio.close_position(position.ticker, current_price)
                        self.logger.info(f"Take-profit executed for {position.ticker}")
                        
        except Exception as e:
            self.logger.error(f"Error in risk management monitoring: {str(e)}")

    async def liquidate_all_positions(self) -> int:
        """Emergency liquidation of all open positions."""
        liquidated_count = 0
        
        try:
            if not self.portfolio or not self.broker:
                return 0
            
            open_positions = self.portfolio.get_all_open_positions()
            
            for position in open_positions:
                try:
                    # Get current market price
                    market_data = await self.broker.get_market_data(position.ticker)
                    current_price = market_data.price if market_data else position.current_price
                    
                    # Create sell order
                    from execution_layer.base_broker import Order
                    order = Order(
                        symbol=position.ticker,
                        side='sell',
                        quantity=position.quantity,
                        order_type='market'
                    )
                    
                    order_id = await self.broker.submit_order(order)
                    if order_id:
                        # Close position in portfolio
                        self.portfolio.close_position(position.ticker, current_price)
                        liquidated_count += 1
                        self.logger.warning(f"EMERGENCY LIQUIDATION: Closed position in {position.ticker}")
                
                except Exception as e:
                    self.logger.error(f"Failed to liquidate position {position.ticker}: {str(e)}")
            
            self.logger.warning(f"Emergency liquidation completed: {liquidated_count} positions closed")
            
        except Exception as e:
            self.logger.error(f"Error in emergency liquidation: {str(e)}")
        
        return liquidated_count

    async def automated_trading_loop(self):
        """
        Main automated trading loop that runs continuously.
        
        This loop:
        1. Generates signals from all active strategies
        2. Executes valid signals automatically
        3. Updates portfolio state
        4. Handles errors gracefully
        5. Respects configured intervals
        """
        self.logger.info("🤖 Starting automated trading loop")
        
        try:
            # Get configuration
            automation_config = self.config.get('automation', {})
            data_config = self.config.get('data', {})
            universe_config = self.config.get('universe', {})
            
            signal_interval = automation_config.get('trading_interval', 
                                                  data_config.get('refresh_intervals', {}).get('real_time', 60))
            tickers = universe_config.get('default_tickers', ['AAPL', 'MSFT', 'GOOGL'])
            max_trades_per_hour = automation_config.get('max_trades_per_hour', 10)
            
            # Track trades for rate limiting
            recent_trades = []
            sync_counter = 0
            sync_interval = 5  # Sync every 5 cycles
            
            while self.auto_trading_enabled and self.initialized:
                loop_start_time = asyncio.get_event_loop().time()
                
                try:
                    self.logger.info(f"🔄 Trading loop iteration - analyzing {len(tickers)} tickers")
                    
                    # Periodic portfolio sync with broker
                    sync_counter += 1
                    if sync_counter >= sync_interval:
                        try:
                            broker_positions = await self.broker.get_positions()
                            self.portfolio.sync_with_broker(broker_positions)
                            sync_counter = 0
                            self.logger.info("🔄 Portfolio synchronized with broker")
                        except Exception as sync_error:
                            self.logger.warning(f"Failed to sync with broker: {str(sync_error)}")
                    
                    # Step 1: Check market regime before generating signals
                    from strategy_layer.market_regime_filter import get_market_regime_filter
                    
                    regime_filter = get_market_regime_filter()
                    current_regime = regime_filter.get_current_regime()
                    regime_favorable = regime_filter.is_regime_favorable_for_trading(current_regime)
                    regime_confidence = regime_filter.get_regime_confidence(current_regime)
                    
                    self.logger.info(f"📊 Market regime: {current_regime.value} (favorable: {regime_favorable}, confidence: {regime_confidence:.2f})")
                    
                    if not regime_favorable:
                        self.logger.info("⚠️ Market regime not favorable for trend-following - skipping signal generation")
                        signals = []
                    else:
                        # Step 2: Generate signals from all strategies
                        signals = await self.generate_signals(tickers)
                    
                    if signals:
                        self.logger.info(f"📡 Generated {len(signals)} trading signals")
                        
                        # Check rate limiting
                        now = asyncio.get_event_loop().time()
                        recent_trades = [t for t in recent_trades if now - t < 3600]  # Last hour
                        
                        if len(recent_trades) >= max_trades_per_hour:
                            self.logger.warning(f"⚠️ Rate limit reached: {len(recent_trades)} trades in last hour")
                        else:
                            # Step 3: Execute signals automatically
                            executed_count = await self.execute_signals(signals)
                            
                            if executed_count > 0:
                                self.logger.info(f"✅ Executed {executed_count} trades automatically")
                                recent_trades.extend([now] * executed_count)
                                
                                # Step 4: Save portfolio state after trades
                                try:
                                    self.portfolio.save_state()
                                    self.logger.debug("💾 Portfolio state saved")
                                except Exception as e:
                                    self.logger.warning(f"Failed to save portfolio state: {str(e)}")
                    
                    # Step 5: Update portfolio with latest market data
                    await self.update_portfolio()
                    
                except Exception as loop_error:
                    self.logger.error(f"❌ Error in trading loop iteration: {str(loop_error)}")
                    # Continue running despite errors
                
                # Step 5: Wait for next iteration
                elapsed = asyncio.get_event_loop().time() - loop_start_time
                sleep_time = max(0, signal_interval - elapsed)
                
                if sleep_time > 0:
                    self.logger.debug(f"⏱️ Sleeping for {sleep_time:.1f} seconds until next iteration")
                    await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            self.logger.info("🛑 Automated trading loop cancelled")
            raise
        except Exception as e:
            self.logger.error(f"❌ Fatal error in automated trading loop: {str(e)}")
        finally:
            self.logger.info("🏁 Automated trading loop stopped")

    async def automated_risk_monitoring_loop(self):
        """
        Continuous risk monitoring loop that checks stop-loss and take-profit conditions.
        
        This loop:
        1. Monitors all open positions continuously
        2. Triggers stop-loss and take-profit orders
        3. Updates portfolio state after risk management actions
        4. Runs at high frequency for responsive risk management
        """
        self.logger.info("🛡️ Starting automated risk monitoring loop")
        
        try:
            # Get configuration
            automation_config = self.config.get('automation', {})
            risk_monitoring_interval = automation_config.get('risk_monitoring_interval', 30)
            
            while self.auto_trading_enabled and self.initialized:
                loop_start_time = asyncio.get_event_loop().time()
                
                try:
                    # Only monitor if we have open positions
                    if self.portfolio and self.portfolio.positions:
                        self.logger.debug(f"🔍 Risk monitoring - checking {len(self.portfolio.positions)} positions")
                        await self.monitor_risk_management()
                
                except Exception as risk_error:
                    self.logger.error(f"❌ Error in risk monitoring iteration: {str(risk_error)}")
                    # Continue running despite errors
                
                # Wait for next iteration
                elapsed = asyncio.get_event_loop().time() - loop_start_time
                sleep_time = max(0, risk_monitoring_interval - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            self.logger.info("🛑 Risk monitoring loop cancelled")
            raise
        except Exception as e:
            self.logger.error(f"❌ Fatal error in risk monitoring loop: {str(e)}")
        finally:
            self.logger.info("🏁 Risk monitoring loop stopped")

    def start_automation_sync(self):
        """Synchronous wrapper to start automation in a separate thread."""
        if self.automation_thread and self.automation_thread.is_alive():
            self.logger.warning("⚠️ Automation thread already running")
            return
        
        def run_automation():
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.automation_loop = loop
            
            try:
                # Run the async start method
                loop.run_until_complete(self.start_automation())
                # Keep running until stopped
                loop.run_forever()
            except Exception as e:
                self.logger.error(f"Automation thread error: {e}")
            finally:
                # Clean up tasks
                if self.trading_loop_task and not self.trading_loop_task.done():
                    self.trading_loop_task.cancel()
                if self.risk_monitoring_task and not self.risk_monitoring_task.done():
                    self.risk_monitoring_task.cancel()
                loop.close()
        
        self.automation_thread = threading.Thread(target=run_automation, daemon=True)
        self.automation_thread.start()
        self.logger.info("🚀 Automation thread started")

    def stop_automation_sync(self):
        """Synchronous wrapper to stop automation."""
        if not self.automation_loop or not self.automation_thread:
            self.logger.warning("⚠️ No automation running")
            return
        
        try:
            # Stop the event loop to terminate the automation thread
            self.automation_loop.stop()
            # Wait for thread to finish
            if self.automation_thread.is_alive():
                self.automation_thread.join(timeout=2.0)
        except Exception as e:
            self.logger.error(f"Error stopping automation: {e}")
        
        # Clean up
        self.auto_trading_enabled = False
        self.automation_loop = None
        self.automation_thread = None
        self.trading_loop_task = None
        self.risk_monitoring_task = None
        self.logger.info("🛑 Automation stopped")

    async def stop_automation(self):
        """Stop automated trading and risk monitoring loops."""
        self.auto_trading_enabled = False
        
        # Cancel running tasks
        if self.trading_loop_task:
            self.trading_loop_task.cancel()
            try:
                await self.trading_loop_task
            except asyncio.CancelledError:
                pass
            self.trading_loop_task = None
        
        if self.risk_monitoring_task:
            self.risk_monitoring_task.cancel()
            try:
                await self.risk_monitoring_task
            except asyncio.CancelledError:
                pass
            self.risk_monitoring_task = None
        
        self.logger.info("🛑 Automated trading and risk monitoring stopped")

    async def cleanup(self):
        """Cleanup all async tasks and resources."""
        self.logger.info("🧹 Starting async cleanup...")

        # Stop automation tasks
        await self.stop_automation()

        # Stop monitoring tasks
        await self.stop_monitoring()

        # Clean up strategies
        for strategy in self.strategies:
            try:
                strategy.cleanup()
                self.logger.info(f"Cleaned up strategy: {strategy.name}")
            except Exception as e:
                self.logger.error(f"Error cleaning up strategy {strategy.name}: {e}")

        # Cancel any remaining tasks
        tasks_to_cancel = []

        # Check for any other async tasks that might be running
        if hasattr(self, 'trading_loop_task') and self.trading_loop_task and not self.trading_loop_task.done():
            tasks_to_cancel.append(self.trading_loop_task)

        if hasattr(self, 'risk_monitoring_task') and self.risk_monitoring_task and not self.risk_monitoring_task.done():
            tasks_to_cancel.append(self.risk_monitoring_task)

        if hasattr(self, 'monitoring_task') and self.monitoring_task and not self.monitoring_task.done():
            tasks_to_cancel.append(self.monitoring_task)

        # Cancel all remaining tasks
        for task in tasks_to_cancel:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Reset state
        self.initialized = False
        self.running = False
        self.auto_trading_enabled = False
        self.monitoring_enabled = False

        self.logger.info("✅ Async cleanup completed")

    def sync_cleanup(self):
        """Synchronous cleanup method for use in non-async contexts."""
        try:
            # Create a new event loop if one doesn't exist
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, we need to handle this differently
                    # For Streamlit, we'll just log and mark for cleanup
                    self.logger.warning("Cannot perform async cleanup in running event loop")
                    self._mark_for_cleanup()
                    return
            except RuntimeError:
                # No event loop, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run cleanup in the event loop
            loop.run_until_complete(self.cleanup())
            
        except Exception as e:
            self.logger.error(f"Error during synchronous cleanup: {str(e)}")
    
    def _mark_for_cleanup(self):
        """Mark the app for cleanup on next initialization."""
        self.logger.info("App marked for cleanup on next restart")
        # Reset critical state to prevent operations
        self.initialized = False
        self.running = False
        self.auto_trading_enabled = False

    async def update_portfolio(self):
        """Update portfolio with latest positions and market data."""
        try:
            if not self.broker or not self.portfolio:
                return
            
            # Get latest positions from broker
            positions = await self.broker.get_positions()
            
            # Update portfolio positions
            for ticker, broker_position in positions.items():
                if broker_position['shares'] > 0:
                    position = Position(
                        ticker=ticker,
                        shares=broker_position['shares'],
                        avg_price=broker_position['avg_price'],
                        current_price=broker_position.get('current_price', broker_position['avg_price']),
                        timestamp=datetime.now()
                    )
                    self.portfolio.add_position(position)
                else:
                    # Remove position if shares = 0
                    self.portfolio.remove_position(ticker)
            
            # Update cash balance
            account_info = await self.broker.get_account_info()
            if account_info:
                self.portfolio.cash = account_info.cash
            
            # Update session state
            st.session_state.portfolio_data = {
                'total_value': self.portfolio.calculate_total_equity(),
                'cash': self.portfolio.cash,
                'positions': {ticker: pos.to_dict() for ticker, pos in self.portfolio.positions.items()},
                'daily_pnl': 0.0,  # Will be calculated separately 
                'total_pnl': self.portfolio.calculate_total_pl(),
                'last_update': datetime.now().isoformat()
            }
            
            self.last_update = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Failed to update portfolio: {str(e)}")
    
    def render_header(self):
        """Render application header."""
        st.title("🚀 Advanced Trading Framework")
        st.markdown("---")
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_color = "🟢" if self.initialized else "🔴"
            st.metric("System Status", f"{status_color} {'Online' if self.initialized else 'Offline'}")
        
        with col2:
            if self.portfolio:
                st.metric("Portfolio Value", f"${self.portfolio.calculate_total_equity():,.2f}")
            else:
                st.metric("Portfolio Value", "N/A")
        
        with col3:
            st.metric("Active Strategies", len(self.strategies))
        
        with col4:
            if self.last_update:
                time_diff = datetime.now() - self.last_update
                st.metric("Last Update", f"{time_diff.seconds}s ago")
            else:
                st.metric("Last Update", "Never")
    
    def render_sidebar(self):
        """Render sidebar controls."""
        with st.sidebar:
            st.header("🎛️ Controls")

            # Theme toggle
            col1, col2 = st.columns([1, 1])
            with col1:
                dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.get('dark_mode', False), key='dark_mode_toggle')
            with col2:
                if st.button("🔄 Refresh"):
                    st.rerun()

            # Update session state
            st.session_state['dark_mode'] = dark_mode

            # Configuration section
            st.subheader("Configuration")
            
            # Enhanced ticker selection
            universe_config = self.config.get('universe', {})
            default_tickers = universe_config.get('default_tickers', ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'])

            selected_tickers = self.render_enhanced_ticker_selector(
                key="sidebar",
                default_selected=default_tickers,
                allow_multiple=True,
                show_sectors=True
            )

            # Update session state for compatibility
            st.session_state.selected_tickers = selected_tickers
            
            # Strategy controls - persist in session state
            st.subheader("Strategy Settings")
            
            if 'confidence_threshold' not in st.session_state:
                st.session_state.confidence_threshold = self.config.get('strategy', {}).get('confidence_threshold', 0.6)
            
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.5,
                max_value=0.95,
                value=st.session_state.confidence_threshold,
                step=0.05
            )
            st.session_state.confidence_threshold = confidence_threshold
            
            # Trading controls - persist in session state
            st.subheader("Trading Controls")
            
            if 'auto_trading' not in st.session_state:
                st.session_state.auto_trading = self.auto_trading_enabled
            
            auto_trading = st.checkbox("Auto Trading", value=st.session_state.auto_trading)
            st.session_state.auto_trading = auto_trading
            
            # Handle automation state changes
            if auto_trading != self.auto_trading_enabled:
                if auto_trading:
                    # Start automation
                    self.start_automation_sync()
                    st.success("🤖 Automated trading started!")
                else:
                    # Stop automation
                    self.stop_automation_sync()
                    st.info("🛑 Automated trading stopped")
                st.rerun()
            
            # Display automation status
            if self.auto_trading_enabled:
                st.success("🟢 Automation Active")
                if st.button("🛑 Stop Automation"):
                    self.stop_automation_sync()
                    st.rerun()
            else:
                st.info("🔴 Manual Mode")
            
            if st.button("🔄 Refresh Data"):
                st.rerun()
            
            if st.button("📊 Generate Signals"):
                if self.initialized and selected_tickers:
                    with st.spinner("Generating signals..."):
                        try:
                            # Create a new event loop for this async call
                            import asyncio
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            signals = loop.run_until_complete(self.generate_signals(selected_tickers))
                            loop.close()
                            
                            self.show_signal_notification(len(signals))
                            self.show_system_notification("Signal generation completed", "success")
                            if not self.auto_trading_enabled:
                                # In manual mode, ask user if they want to execute
                                if signals:
                                    st.success(f"Generated {len(signals)} signals")
                                    if st.button("⚡ Execute Signals"):
                                        executed = asyncio.run(self.execute_signals(signals))
                                        st.success(f"Executed {executed} trades")
                                else:
                                    st.info("No signals generated")
                            else:
                                st.success(f"Generated {len(signals)} signals (auto-execution enabled)")
                        except Exception as e:
                            st.error(f"Error generating signals: {str(e)}")
                            self.logger.error(f"Signal generation failed: {str(e)}")
                        st.rerun()
                else:
                    st.warning("System not initialized or no tickers selected")
            
            # Emergency controls
            st.subheader("🚨 Emergency Controls")
            
            if st.button("🚨 LIQUIDATE ALL POSITIONS", type="primary"):
                if self.initialized and self.portfolio and self.portfolio.positions:
                    with st.spinner("Liquidating all positions..."):
                        liquidated_count = asyncio.run(self.liquidate_all_positions())
                        st.success(f"Liquidated {liquidated_count} positions")
                        st.rerun()
                else:
                    st.warning("No positions to liquidate")
            
            # Risk management - persist in session state
            st.subheader("Risk Management")
            
            if 'max_position_size' not in st.session_state:
                st.session_state.max_position_size = int(self.config.get('portfolio', {}).get('max_position_size', 0.1) * 100)
            
            max_position_size = st.slider(
                "Max Position Size (%)",
                min_value=1,
                max_value=20,
                value=st.session_state.max_position_size,
                step=1
            )
            st.session_state.max_position_size = max_position_size
            
            if 'stop_loss_pct' not in st.session_state:
                st.session_state.stop_loss_pct = int(self.config.get('portfolio', {}).get('stop_loss_default', 0.05) * 100)
            
            stop_loss_pct = st.slider(
                "Stop Loss (%)",
                min_value=1,
                max_value=10,
                value=st.session_state.stop_loss_pct,
                step=1
            )
            st.session_state.stop_loss_pct = stop_loss_pct
            
            return {
                'selected_tickers': selected_tickers,
                'confidence_threshold': confidence_threshold,
                'auto_trading': auto_trading,
                'max_position_size': max_position_size / 100,
                'stop_loss_pct': stop_loss_pct / 100
            }
    
    def render_portfolio_tab(self):
        """Render portfolio overview tab."""
        st.header("💼 Portfolio Overview")

        if not self.portfolio:
            st.warning("Portfolio not initialized")
            return

        # Portfolio metrics in an expandable container
        with st.container():
            st.subheader("📊 Key Metrics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_value = self.portfolio.calculate_total_equity()
                st.metric("Total Value", f"${total_value:,.2f}")

            with col2:
                cash = self.portfolio.current_cash
                st.metric("Cash", f"${cash:,.2f}")

            with col3:
                total_pl = self.portfolio.calculate_total_pl()
                delta_indicator = "📈" if total_pl >= 0 else "📉"
                st.metric("Total P&L", f"${total_pl:,.2f}", delta=f"{delta_indicator}")

            with col4:
                total_pl_pct = self.portfolio.calculate_total_pl_pct()
                delta_indicator = "↗️" if total_pl_pct >= 0 else "↘️"
                st.metric("Total P&L %", f"{total_pl_pct:.2%}", delta=f"{delta_indicator}")
        
        # Positions table in expander
        with st.expander("📋 Current Positions", expanded=True):
            if self.portfolio.positions:
                positions_data = []
                for ticker, position in self.portfolio.positions.items():
                    positions_data.append({
                        'Ticker': ticker,
                        'Quantity': position.quantity,
                        'Entry Price': f"${position.entry_price:.2f}",
                        'Current Price': f"${position.current_price:.2f}",
                        'Market Value': f"${position.current_value:.2f}",
                        'P&L': f"${position.unrealized_pl:.2f}",
                        'P&L %': f"{position.unrealized_pl_pct:.2%}"
                    })

                self.render_enhanced_positions_table(positions_data)
            else:
                st.info("No current positions")
        
        # Trade History in expander
        with st.expander("📈 Recent Trade History", expanded=False):
            if self.portfolio.trade_history:
                # Display recent trades (last 50 for better filtering experience)
                recent_trades = self.portfolio.trade_history[-50:][::-1]  # Most recent first

                trade_data = []
                for trade in recent_trades:
                    trade_data.append({
                        'Timestamp': trade.get('timestamp', 'N/A'),
                        'Ticker': trade.get('ticker', 'N/A'),
                        'Action': trade.get('action', 'N/A'),
                        'Quantity': trade.get('quantity', 0),
                        'Price': f"${trade.get('price', 0):.2f}",
                        'Value': f"${trade.get('value', 0):.2f}",
                        'P&L': f"${trade.get('realized_pl', 0):.2f}" if 'realized_pl' in trade else 'N/A'
                    })

                self.render_enhanced_trade_history_table(trade_data)
            else:
                st.info("No trade history available")
    
    def render_signals_tab(self):
        """Render signals tab."""
        st.header("📡 Trading Signals")
        
        # Load signals from session state
        signals_data = st.session_state.get('signals_data', [])
        
        if not signals_data:
            st.info("No signals generated yet")
            return
        
        # Convert back to signal objects for display
        signals_history = []
        for signal_dict in signals_data:
            # Create a simple signal-like object for display
            signal = type('Signal', (), signal_dict)()
            # Convert timestamp to datetime if it's a number
            if hasattr(signal, 'timestamp') and isinstance(signal.timestamp, (int, float)):
                signal.timestamp = datetime.fromtimestamp(signal.timestamp)
            signals_history.append(signal)
        
        # Recent signals - enhanced with filtering
        recent_signals = signals_history[-100:]  # Last 100 signals for better filtering
        display_data = []

        for signal in recent_signals:
            display_data.append({
                'Timestamp': signal.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(signal.timestamp, 'strftime') else str(signal.timestamp),
                'Ticker': signal.ticker,
                'Action': signal.action,
                'Confidence': f"{signal.confidence:.2%}",
                'Price': f"${signal.price:.2f}" if signal.price else "N/A",
                'Reasoning': signal.reasoning[:100] + "..." if signal.reasoning and len(signal.reasoning) > 100 else (signal.reasoning or "N/A")
            })

        if display_data:
            self.render_enhanced_signals_table(display_data)
        else:
            st.info("No signals available")
        
        # Signal statistics
        if len(signals_history) > 0:
            st.subheader("📊 Signal Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                buy_signals = len([s for s in signals_history if s.action.upper() == 'BUY'])
                st.metric("Buy Signals", buy_signals)
            
            with col2:
                sell_signals = len([s for s in signals_history if s.action.upper() == 'SELL'])
                st.metric("Sell Signals", sell_signals)
            
            with col3:
                avg_confidence = np.mean([s.confidence for s in signals_history])
                st.metric("Avg Confidence", f"{avg_confidence:.2%}")
    
    def render_performance_tab(self):
        """Render enhanced performance analysis dashboard."""
        st.header("📈 Performance Dashboard")

        if not self.portfolio:
            st.warning("Portfolio not initialized")
            return

        # Get comprehensive performance metrics
        trading_metrics = self._calculate_trading_performance()
        risk_metrics = self._calculate_risk_metrics()

        # Performance Overview Section
        st.subheader("🎯 Performance Overview")
        
        # Key Performance Indicators (Top Row)
        col1, col2, col3, col4 = st.columns(4)

        current_value = self.portfolio.calculate_total_equity()
        initial_capital = self.portfolio.initial_capital
        total_return = ((current_value - initial_capital) / initial_capital) * 100

        with col1:
            delta_color = "normal" if total_return >= 0 else "inverse"
            st.metric(
                "Total Return",
                f"{total_return:+.2f}%",
                delta=f"${current_value - initial_capital:+,.2f}",
                delta_color=delta_color
            )

        with col2:
            st.metric(
                "Current Value",
                f"${current_value:,.2f}",
                delta=f"vs ${initial_capital:,.2f} initial"
            )

        with col3:
            sharpe_ratio = risk_metrics.get('sharpe_ratio', 0)
            sharpe_color = "normal" if sharpe_ratio > 1 else "inverse" if sharpe_ratio < 0.5 else "off"
            st.metric(
                "Sharpe Ratio",
                f"{sharpe_ratio:.2f}",
                delta="Good" if sharpe_ratio > 1 else "Poor" if sharpe_ratio < 0.5 else "Fair",
                delta_color=sharpe_color
            )

        with col4:
            max_drawdown = risk_metrics.get('max_drawdown_pct', 0)
            dd_color = "normal" if max_drawdown > -10 else "inverse"
            st.metric(
                "Max Drawdown",
                f"{max_drawdown:.1f}%",
                delta="Low Risk" if max_drawdown > -10 else "High Risk",
                delta_color=dd_color
            )
        
        # Trading Statistics (Second Row)
        st.subheader("📊 Trading Statistics")
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            win_rate = trading_metrics['win_rate']
            wr_color = "normal" if win_rate >= 50 else "inverse"
            st.metric(
                "Win Rate",
                f"{win_rate:.1f}%",
                delta=f"{trading_metrics['winning_trades']}/{trading_metrics['total_trades']} trades",
                delta_color=wr_color
            )

        with col6:
            profit_factor = trading_metrics['profit_factor']
            pf_color = "normal" if profit_factor > 1.5 else "inverse" if profit_factor < 1 else "off"
            st.metric(
                "Profit Factor",
                f"{profit_factor:.2f}",
                delta="Excellent" if profit_factor > 2 else "Good" if profit_factor > 1.5 else "Poor",
                delta_color=pf_color
            )

        with col7:
            st.metric(
                "Avg P/L per Trade",
                f"${trading_metrics['avg_pl_per_trade']:,.2f}",
                delta=f"Total: ${trading_metrics['total_pl']:,.2f}"
            )

        with col8:
            avg_trade_duration = risk_metrics.get('avg_trade_duration_days', 0)
            st.metric(
                "Avg Hold Period",
                f"{avg_trade_duration:.1f} days",
                delta=f"Range: {risk_metrics.get('min_trade_duration', 0):.0f}-{risk_metrics.get('max_trade_duration', 0):.0f} days"
            )
        
        # Enhanced Equity Curve
        if PLOTLY_AVAILABLE:
            st.subheader("📈 Portfolio Equity Curve")

            value_history = self.portfolio.get_portfolio_value_history()

            if value_history:
                df = pd.DataFrame(value_history)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')

                if not df.empty:
                    # Calculate returns for the chart
                    df['returns'] = df['portfolio_value'].pct_change().fillna(0)

                    # Create enhanced equity curve
                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02,
                        row_heights=[0.7, 0.3],
                        subplot_titles=['Portfolio Value', 'Daily Returns %']
                    )

                    # Main equity curve
                    fig.add_trace(
                        go.Scatter(
                            x=df['date'],
                            y=df['portfolio_value'],
                            mode='lines',
                            name='Portfolio Value',
                            line=dict(color='#1f77b4', width=2),
                            hovertemplate='Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
                        ),
                        row=1, col=1
                    )

                    # Add initial capital reference line
                    fig.add_hline(
                        y=initial_capital,
                        line_dash="dash",
                        line_color="red",
                        opacity=0.7,
                        annotation_text=f"Initial: ${initial_capital:,.0f}",
                        row=1, col=1
                    )

                    # Add daily returns bar chart
                    colors = ['green' if ret >= 0 else 'red' for ret in df['returns']]
                    fig.add_trace(
                        go.Bar(
                            x=df['date'],
                            y=df['returns'] * 100,
                            name='Daily Returns %',
                            marker_color=colors,
                            opacity=0.7,
                            hovertemplate='Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
                        ),
                        row=2, col=1
                    )

                    # Update layout
                    fig.update_layout(
                        title="Portfolio Performance Analysis",
                        height=600,
                        showlegend=True,
                        template="plotly_white",
                        hovermode='x unified'
                    )

                    fig.update_xaxes(title_text="Date", row=2, col=1)
                    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
                    fig.update_yaxes(title_text="Daily Return (%)", row=2, col=1)

                    st.plotly_chart(fig, width='stretch')

            else:
                st.info("📊 Portfolio value history will appear after the first trading day")
        
        # Trade Analysis Section
        st.subheader("🔍 Trade Analysis")

        closed_trades = [trade for trade in self.portfolio.trade_history if trade.get('action') == 'CLOSE']

        if closed_trades:
            # Enhanced trade history table
            trade_data = []
            for trade in closed_trades[-50:]:  # Last 50 trades
                trade_data.append({
                    'Date': trade.get('timestamp', ''),
                    'Ticker': trade.get('ticker', 'N/A'),
                    'Quantity': trade.get('quantity', 0),
                    'Entry Price': trade.get('entry_price', 0),
                    'Exit Price': trade.get('price', 0),
                    'P&L': trade.get('realized_pl', 0),
                    'Return %': ((trade.get('price', 0) / trade.get('entry_price', 1) - 1) * 100) if trade.get('entry_price', 0) > 0 else 0,
                    'Duration': trade.get('duration_days', 0)
                })

            if trade_data:
                self.render_enhanced_trade_history_table(trade_data)
        else:
            st.info("📋 No closed trades yet. Start trading to see detailed analysis.")
    
    def _calculate_trading_performance(self) -> Dict[str, float]:
        """
        Calculate trading performance metrics from closed trades.
        
        Returns:
            Dictionary with trading performance metrics
        """
        closed_trades = [trade for trade in self.portfolio.trade_history if trade.get('action') == 'CLOSE']
        
        if not closed_trades:
            return {
                'total_pl': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_pl_per_trade': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'largest_win': 0.0
            }
        
        # Calculate basic metrics
        total_pl = sum(trade.get('realized_pl', 0) for trade in closed_trades)
        total_trades = len(closed_trades)
        
        # Separate winning and losing trades
        winning_trades = [trade for trade in closed_trades if trade.get('realized_pl', 0) > 0]
        losing_trades = [trade for trade in closed_trades if trade.get('realized_pl', 0) < 0]
        
        winning_trades_count = len(winning_trades)
        losing_trades_count = len(losing_trades)
        
        # Win rate
        win_rate = (winning_trades_count / total_trades) * 100 if total_trades > 0 else 0
        
        # Average P/L per trade
        avg_pl_per_trade = total_pl / total_trades if total_trades > 0 else 0
        
        # Profit factor
        total_wins = sum(trade.get('realized_pl', 0) for trade in winning_trades)
        total_losses = abs(sum(trade.get('realized_pl', 0) for trade in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0
        
        # Largest win
        largest_win = max((trade.get('realized_pl', 0) for trade in closed_trades), default=0)
        
        return {
            'total_pl': total_pl,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_pl_per_trade': avg_pl_per_trade,
            'total_trades': total_trades,
            'winning_trades': winning_trades_count,
            'losing_trades': losing_trades_count,
            'largest_win': largest_win
        }
    
    def render_logs_tab(self):
        """Render logs tab with logs and debug info in separate columns."""
        st.header("📝 System Logs & Debug Info")

        # Create two columns
        col1, col2 = st.columns(2)

        # Left column: Logs
        with col1:
            st.subheader("📄 System Logs")
            log_file_path = "logs/trading.log"

            if os.path.exists(log_file_path):
                try:
                    with open(log_file_path, 'r') as f:
                        log_lines = f.readlines()

                    # Show last 100 lines
                    recent_logs = log_lines[-100:]

                    st.text_area(
                        "Recent Logs",
                        value=''.join(recent_logs),
                        height=400,
                        help="Showing last 100 log entries"
                    )

                    if st.button("📥 Download Full Logs"):
                        with open(log_file_path, 'r') as f:
                            st.download_button(
                                label="Download",
                                data=f.read(),
                                file_name=f"trading_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                                mime="text/plain"
                            )

                except Exception as e:
                    st.error(f"Failed to read log file: {str(e)}")
            else:
                st.info("No log file found")

        # Right column: Debug Info
        with col2:
            st.subheader("🔍 Debug Info")

            # Get trading app instance for debug info
            trading_app = st.session_state.get('trading_app')

            if trading_app:
                st.write(f"**Session initialized:** {st.session_state.get('app_initialized', False)}")
                st.write(f"**App initialized:** {trading_app.initialized}")
                st.write(f"**Portfolio exists:** {trading_app.portfolio is not None}")
                st.write(f"**Broker exists:** {trading_app.broker is not None}")
                st.write(f"**Strategies count:** {len(trading_app.strategies) if hasattr(trading_app, 'strategies') else 0}")
                st.write(f"**Signals in session:** {len(st.session_state.get('signals_data', []))}")

                # Portfolio details if available
                if trading_app.portfolio:
                    st.markdown("**Portfolio Details:**")
                    st.write(f"- Cash: ${trading_app.portfolio.cash:.2f}")
                    st.write(f"- Total Value: ${trading_app.portfolio.get_total_value():.2f}")
                    st.write(f"- Positions: {len(trading_app.portfolio.positions)}")

                # Broker details if available
                if trading_app.broker:
                    st.markdown("**Broker Details:**")
                    st.write(f"- Type: {type(trading_app.broker).__name__}")
                    st.write(f"- Connected: {trading_app.broker.connected}")

                # Configuration summary
                st.markdown("**Configuration:**")
                trading_env = self.config.get('trading_environment', 'paper')
                st.write(f"- Environment: {trading_env}")
                st.write(f"- Risk management: {self.config.get('risk_management', {}).get('enabled', False)}")
            else:
                st.warning("Trading app not initialized")
    
    def render_main_app(self):
        """Render the main application interface."""
        
        # Header
        self.render_header()
        
        # Live Trading Warning
        trading_environment = self.config.get('trading_environment', 'paper')
        if trading_environment == 'live':
            st.error("🚨 **LIVE TRADING MODE ACTIVE** 🚨")
            st.error("**WARNING:** This application is currently configured for LIVE TRADING with real money!")
            st.error("All trades executed will use actual funds. Please ensure you understand the risks.")
            st.markdown("---")
        
        # Sidebar
        controls = self.render_sidebar()

        # Main content container
        with st.container():
            # Main content tabs
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["💼 Portfolio", "📡 Signals", "📈 Performance", "📝 Logs", "⚙️ Settings", "🔍 Monitoring", "🔬 Backtesting", "📊 Technical Analysis"])

            with tab1:
                self.render_portfolio_tab()

            with tab2:
                self.render_signals_tab()

            with tab3:
                self.render_performance_tab()

            with tab4:
                self.render_logs_tab()

            with tab5:
                self.render_settings_tab()

            with tab6:
                self.render_monitoring_tab()

            with tab7:
                self.render_backtesting_tab()

            with tab8:
                self.render_technical_analysis_tab()

    def render_settings_tab(self):
        """Render the settings and configuration tab."""
        st.header("⚙️ Settings & Configuration")

        # Application Control Section
        st.subheader("Application Control")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🧹 Cleanup Async Tasks", type="secondary", help="Cancel all running async tasks and clean up resources"):
                with st.spinner("Cleaning up async tasks..."):
                    asyncio.run(self.cleanup())
                    st.success("✅ Async cleanup completed")
                    st.rerun()

        with col2:
            if st.button("🔄 Reset Session State", type="secondary", help="Reset all session state variables"):
                # Clear session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("✅ Session state reset")
                st.rerun()

        with col3:
            if st.button("🔧 Reinitialize App", type="secondary", help="Reinitialize the trading application"):
                with st.spinner("Reinitializing application..."):
                    # Clear cached app
                    initialize_trading_app.clear()
                    st.success("✅ Application cache cleared - will reinitialize on next run")
                    st.rerun()

        # Configuration Display
        st.subheader("Current Configuration")
        st.json(self.config)

        # System Information
        st.subheader("System Information")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Application State:**")
            st.write(f"- Initialized: {self.initialized}")
            st.write(f"- Running: {self.running}")
            st.write(f"- Auto Trading: {self.auto_trading_enabled}")
            st.write(f"- Monitoring: {self.monitoring_enabled}")

        with col2:
            st.write("**Active Tasks:**")
            trading_task = self.trading_loop_task is not None and not self.trading_loop_task.done() if self.trading_loop_task else False
            risk_task = self.risk_monitoring_task is not None and not self.risk_monitoring_task.done() if self.risk_monitoring_task else False
            monitoring_task = self.monitoring_task is not None and not self.monitoring_task.done() if self.monitoring_task else False

            st.write(f"- Trading Loop: {'🟢' if trading_task else '🔴'}")
            st.write(f"- Risk Monitoring: {'🟢' if risk_task else '🔴'}")
            st.write(f"- Health Monitoring: {'🟢' if monitoring_task else '🔴'}")

    def render_monitoring_tab(self):
        """Render the monitoring and health dashboard."""
        st.header("🔍 System Monitoring & Health")

        # Health Status Overview
        st.subheader("Health Status")

        health_status = self.get_health_status()

        col1, col2, col3 = st.columns(3)

        with col1:
            system_status = health_status.get('system', {}).get('overall_status', 'UNKNOWN')
            status_color = {
                'HEALTHY': '🟢',
                'DEGRADED': '🟡',
                'UNHEALTHY': '🔴',
                'UNKNOWN': '⚪'
            }.get(system_status, '⚪')

            st.metric("System Health", f"{status_color} {system_status}")

        with col2:
            app_health = health_status.get('application', {})
            components_healthy = sum([
                app_health.get('portfolio_initialized', False),
                app_health.get('broker_connected', False),
                app_health.get('strategies_loaded', False)
            ])
            total_components = 3

            st.metric("App Components", f"{components_healthy}/{total_components} Healthy")

        with col3:
            monitoring_active = health_status.get('application', {}).get('monitoring_active', False)
            st.metric("Monitoring", "🟢 Active" if monitoring_active else "🔴 Inactive")

        # Detailed Health Checks
        st.subheader("Health Check Details")

        if 'system' in health_status and 'checks' in health_status['system']:
            checks_df = []
            for check in health_status['system']['checks']:
                checks_df.append({
                    'Component': check['name'].replace('_', ' ').title(),
                    'Status': check['status'],
                    'Message': check['message'],
                    'Duration (ms)': f"{check['duration_ms']:.1f}"
                })

            if checks_df:
                st.dataframe(pd.DataFrame(checks_df), width='stretch')

        # Application Health Details
        st.subheader("Application Health")

        app_health = health_status.get('application', {})

        health_items = [
            ("Portfolio Initialized", app_health.get('portfolio_initialized', False)),
            ("Broker Connected", app_health.get('broker_connected', False)),
            ("Strategies Loaded", app_health.get('strategies_loaded', False)),
            ("Auto Trading Active", app_health.get('auto_trading_active', False)),
            ("Monitoring Active", app_health.get('monitoring_active', False))
        ]

        cols = st.columns(2)
        for i, (label, status) in enumerate(health_items):
            with cols[i % 2]:
                icon = "✅" if status else "❌"
                st.write(f"{icon} {label}")

        # Metrics Dashboard
        st.subheader("Key Metrics")

        metrics_summary = self.get_metrics_summary()

        if metrics_summary and 'error' not in metrics_summary:
            # Create metrics display
            metric_cols = st.columns(3)

            metrics_to_show = [
                ('portfolio_value', 'Portfolio Value', '$'),
                ('cpu_usage_percent', 'CPU Usage', '%'),
                ('memory_usage_percent', 'Memory Usage', '%'),
                ('signals_last_hour', 'Signals (Last Hour)', ''),
                ('healthy_strategies', 'Healthy Strategies', ''),
                ('unhealthy_positions', 'Unhealthy Positions', '')
            ]

            for i, (metric_key, display_name, unit) in enumerate(metrics_to_show):
                with metric_cols[i % 3]:
                    if metric_key in metrics_summary:
                        summary = metrics_summary[metric_key]
                        latest = summary.get('latest', 0)
                        avg = summary.get('avg', 0)

                        if metric_key == 'portfolio_value':
                            st.metric(display_name, f"${latest:,.0f}", f"Avg: ${avg:,.0f}")
                        elif unit == '%':
                            st.metric(display_name, f"{latest:.1f}{unit}", f"Avg: {avg:.1f}{unit}")
                        else:
                            st.metric(display_name, f"{latest}{unit}", f"Avg: {avg:.1f}{unit}")
        else:
            st.info("No metrics available yet. Metrics will appear after the system runs for a few minutes.")

        # System Resources
        st.subheader("System Resources")

        try:
            system_metrics = get_system_metrics()

            if 'error' not in system_metrics:
                # CPU and Memory
                col1, col2 = st.columns(2)

                with col1:
                    cpu_percent = system_metrics.get('cpu', {}).get('percent', 0)
                    st.metric("CPU Usage", f"{cpu_percent:.1f}%")

                    memory_info = system_metrics.get('memory', {})
                    memory_percent = memory_info.get('percent', 0)
                    memory_used = memory_info.get('used_gb', 0)
                    memory_total = memory_info.get('total_gb', 0)

                    st.metric("Memory Usage", f"{memory_percent:.1f}%",
                            f"{memory_used:.1f}GB / {memory_total:.1f}GB")

                with col2:
                    disk_info = system_metrics.get('disk', {})
                    disk_percent = disk_info.get('percent', 0)
                    disk_used = disk_info.get('used_gb', 0)
                    disk_total = disk_info.get('total_gb', 0)

                    st.metric("Disk Usage", f"{disk_percent:.1f}%",
                            f"{disk_used:.1f}GB / {disk_total:.1f}GB")

                    # Network (basic)
                    network_info = system_metrics.get('network', {})
                    bytes_sent = network_info.get('bytes_sent', 0) / (1024**3)  # GB
                    bytes_recv = network_info.get('bytes_recv', 0) / (1024**3)  # GB

                    st.metric("Network I/O", f"↓{bytes_recv:.2f}GB ↑{bytes_sent:.2f}GB")
            else:
                st.error(f"Failed to get system metrics: {system_metrics['error']}")

        except Exception as e:
            st.error(f"Error displaying system resources: {str(e)}")

        # Monitoring Controls
        st.subheader("Monitoring Controls")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Run Health Check Now", width='stretch'):
                with st.spinner("Running health checks..."):
                    # Force immediate health check
                    self.show_system_notification("Health check started", "info")
                    asyncio.run(self._perform_health_checks())
                    self.show_system_notification("Health check completed successfully", "success")
                    st.rerun()

        with col2:
            monitoring_status = "Disable" if self.monitoring_enabled else "Enable"
            if st.button(f"{'🔴' if self.monitoring_enabled else '🟢'} {monitoring_status} Monitoring",
                        width='stretch'):
                if self.monitoring_enabled:
                    asyncio.run(self.stop_monitoring())
                    self.monitoring_enabled = False
                    st.success("Monitoring disabled")
                else:
                    self.monitoring_enabled = True
                    asyncio.run(self.start_monitoring())
                    st.success("Monitoring enabled")
                st.rerun()

        # Recent Metrics History
        st.subheader("Recent Metrics History")

        try:
            # Get recent metrics for visualization
            recent_metrics = {}
            since = datetime.now() - timedelta(minutes=30)  # Last 30 minutes

            for metric_name in ['portfolio_value', 'cpu_usage_percent', 'memory_usage_percent']:
                metrics = metrics_collector.get_metrics(metric_name, since=since)
                if metrics:
                    recent_metrics[metric_name] = sorted(metrics, key=lambda x: x.timestamp)

            if recent_metrics:
                # Simple line chart for portfolio value if available
                if 'portfolio_value' in recent_metrics:
                    portfolio_data = recent_metrics['portfolio_value']
                    if len(portfolio_data) > 1:
                        chart_data = pd.DataFrame({
                            'timestamp': [m.timestamp for m in portfolio_data],
                            'value': [m.value for m in portfolio_data]
                        })
                        chart_data = chart_data.set_index('timestamp')

                        st.line_chart(chart_data, width='stretch')
                    else:
                        st.info("Need more data points for portfolio value chart")
            else:
                st.info("No recent metrics data available for charting")

        except Exception as e:
            st.error(f"Error displaying metrics history: {str(e)}")

    def render_backtesting_tab(self):
        """Render the backtesting interface."""
        st.header("🔬 Strategy Backtesting")

        # Import backtesting components
        try:
            from core.backtesting import BacktestEngine, run_backtest, compare_strategies
            from strategy_layer.backtest_strategies import create_strategy
            BACKTESTING_AVAILABLE = True
        except ImportError as e:
            st.error(f"Backtesting module not available: {e}")
            st.info("Install required dependencies: pip install scipy")
            BACKTESTING_AVAILABLE = False
            return

        if not BACKTESTING_AVAILABLE:
            return

        # Backtesting configuration
        st.subheader("Backtest Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            symbols_input = st.text_input(
                "Symbols (comma-separated)",
                value="AAPL,MSFT,GOOGL",
                help="Stock symbols to include in backtest"
            )
            symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]

        with col2:
            # Enhanced date range picker for backtesting
            start_date, end_date = self.render_enhanced_date_range_picker(
                key="backtesting",
                default_start=datetime(2023, 1, 1),
                default_end=datetime(2024, 1, 1),
                show_presets=True
            )

            if start_date is None or end_date is None:
                st.stop()

        with col3:
            initial_cash = st.number_input(
                "Initial Cash ($)",
                value=100000.0,
                min_value=1000.0,
                step=1000.0,
                help="Starting portfolio value"
            )

        # Benchmarking configuration
        st.subheader("Benchmarking Configuration")

        enable_benchmarking = st.checkbox(
            "Enable Benchmarking",
            value=True,
            help="Compare strategy performance against market indices"
        )

        benchmark_symbols = []
        if enable_benchmarking:
            col1, col2 = st.columns(2)

            with col1:
                benchmark_input = st.text_input(
                    "Benchmark Symbols (comma-separated)",
                    value="SPY,QQQ,IWM",
                    help="Market indices to compare against (e.g., SPY for S&P 500, QQQ for NASDAQ)"
                )
                benchmark_symbols = [s.strip().upper() for s in benchmark_input.split(',') if s.strip()]

            with col2:
                risk_free_rate = st.slider(
                    "Risk-Free Rate (%)",
                    min_value=0.0,
                    max_value=5.0,
                    value=2.0,
                    step=0.1,
                    help="Risk-free rate for Sharpe ratio and alpha calculations"
                ) / 100.0

        # Strategy selection
        st.subheader("Strategy Selection")

        available_strategies = {
            'moving_average': 'Moving Average Crossover',
            'rsi': 'RSI Mean Reversion',
            'ml_random_forest': 'ML Random Forest',
            'mean_reversion': 'Bollinger Band Mean Reversion'
        }

        selected_strategies = st.multiselect(
            "Select Strategies to Test",
            options=list(available_strategies.keys()),
            default=['moving_average'],
            format_func=lambda x: available_strategies[x],
            help="Choose one or more strategies to backtest"
        )

        # Strategy parameters
        if selected_strategies:
            st.subheader("Strategy Parameters")

            strategy_configs = {}
            for strategy_name in selected_strategies:
                with st.expander(f"⚙️ {available_strategies[strategy_name]} Parameters"):
                    if strategy_name == 'moving_average':
                        fast_period = st.slider("Fast MA Period", 5, 50, 20, key=f"ma_fast_{strategy_name}")
                        slow_period = st.slider("Slow MA Period", 20, 200, 50, key=f"ma_slow_{strategy_name}")
                        position_size = st.slider("Position Size (%)", 1, 50, 10, key=f"ma_size_{strategy_name}")
                        strategy_configs[strategy_name] = {
                            'fast_period': fast_period,
                            'slow_period': slow_period,
                            'position_size': position_size / 100.0
                        }

                    elif strategy_name == 'rsi':
                        rsi_period = st.slider("RSI Period", 5, 30, 14, key=f"rsi_period_{strategy_name}")
                        oversold = st.slider("Oversold Level", 10, 40, 30, key=f"rsi_oversold_{strategy_name}")
                        overbought = st.slider("Overbought Level", 60, 90, 70, key=f"rsi_overbought_{strategy_name}")
                        position_size = st.slider("Position Size (%)", 1, 50, 10, key=f"rsi_size_{strategy_name}")
                        strategy_configs[strategy_name] = {
                            'rsi_period': rsi_period,
                            'oversold_level': oversold,
                            'overbought_level': overbought,
                            'position_size': position_size / 100.0
                        }

                    elif strategy_name == 'ml_random_forest':
                        confidence = st.slider("Confidence Threshold", 0.1, 0.9, 0.6, key=f"ml_conf_{strategy_name}")
                        position_size = st.slider("Position Size (%)", 1, 50, 5, key=f"ml_size_{strategy_name}")
                        strategy_configs[strategy_name] = {
                            'confidence_threshold': confidence,
                            'position_size': position_size / 100.0
                        }

                    elif strategy_name == 'mean_reversion':
                        bb_period = st.slider("BB Period", 10, 50, 20, key=f"bb_period_{strategy_name}")
                        bb_std = st.slider("BB Standard Deviations", 1.0, 3.0, 2.0, key=f"bb_std_{strategy_name}")
                        position_size = st.slider("Position Size (%)", 1, 50, 5, key=f"bb_size_{strategy_name}")
                        strategy_configs[strategy_name] = {
                            'bb_period': bb_period,
                            'bb_std': bb_std,
                            'position_size': position_size / 100.0
                        }

        # Run backtest button
        if st.button("🚀 Run Backtest", type="primary", width='stretch'):
            if not selected_strategies:
                st.error("Please select at least one strategy to test.")
                self.show_system_notification("Please select strategies for backtesting", "warning")
                return

            if start_date >= end_date:
                st.error("End date must be after start date.")
                return

            with st.spinner("Running backtest... This may take a few minutes."):
                try:
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Convert dates
                    start_str = start_date.strftime('%Y-%m-%d')
                    end_str = end_date.strftime('%Y-%m-%d')

                    status_text.text("Loading historical data...")
                    progress_bar.progress(0.1)

                    # Run backtests for each strategy
                    results = []
                    for i, strategy_name in enumerate(selected_strategies):
                        status_text.text(f"Testing {available_strategies[strategy_name]} strategy...")
                        progress = 0.1 + (0.8 * i / len(selected_strategies))

                        try:
                            config = {
                                'initial_cash': initial_cash,
                                'commission_per_trade': 5.0,
                                'slippage_bps': 5.0
                            }
                            config.update(strategy_configs.get(strategy_name, {}))

                            # Add benchmarking config
                            if enable_benchmarking:
                                config['benchmark_symbols'] = benchmark_symbols
                                config['risk_free_rate'] = risk_free_rate

                            strategy = create_strategy(strategy_name, config)

                            # Run backtest with benchmarking
                            benchmark_symbols_param = benchmark_symbols if enable_benchmarking else None
                            result = run_backtest(strategy, symbols, start_str, end_str, config, benchmark_symbols_param)

                            if result['status'] == 'success':
                                results.append(result)
                                st.success(f"✅ {available_strategies[strategy_name]} backtest completed")
                                self.show_system_notification(f"{available_strategies[strategy_name]} backtest completed", "success")
                            else:
                                st.error(f"❌ {available_strategies[strategy_name]} backtest failed: {result.get('error', 'Unknown error')}")

                        except Exception as e:
                            st.error(f"❌ Error testing {available_strategies[strategy_name]}: {str(e)}")

                        progress_bar.progress(min(0.9, progress + 0.8 / len(selected_strategies)))

                    progress_bar.progress(1.0)
                    status_text.text("Backtesting completed!")

                    # Display results
                    if results:
                        self._display_backtest_results(results, symbols, start_str, end_str)
                    else:
                        st.error("No successful backtests to display.")

                except Exception as e:
                    st.error(f"Backtest failed: {str(e)}")
                    st.exception(e)

                finally:
                    progress_bar.empty()
                    status_text.empty()

    def _display_backtest_results(self, results: List[Dict], symbols: List[str],
                                start_date: str, end_date: str):
        """Display backtest results in an organized manner."""
        st.success("🎉 Backtesting completed successfully!")
        self.show_system_notification("All backtests completed successfully", "success")

        # Summary metrics comparison
        st.subheader("📊 Performance Summary")

        summary_data = []
        for result in results:
            metrics = result['metrics']
            row_data = {
                'Strategy': result['strategy_name'],
                'Total Return': f"{metrics.total_return:.2%}",
                'Annual Return': f"{metrics.annualized_return:.2%}",
                'Volatility': f"{metrics.volatility:.2%}",
                'Sharpe Ratio': f"{metrics.sharpe_ratio:.2f}",
                'Max Drawdown': f"{metrics.max_drawdown:.2%}",
                'Win Rate': f"{metrics.win_rate:.2%}",
                'Total Trades': metrics.total_trades,
                'Final Value': f"${result['final_value']:,.0f}"
            }

            # Add benchmark comparison if available
            if 'benchmarking' in result and result['benchmarking']:
                primary_benchmark = result['benchmarking'][0]  # First benchmark
                row_data.update({
                    'Benchmark': primary_benchmark.benchmark_name,
                    'Benchmark Return': f"{primary_benchmark.benchmark_return:.2%}",
                    'Excess Return': f"{primary_benchmark.excess_return:.2%}",
                    'Alpha': f"{primary_benchmark.alpha:.2%}",
                    'Beta': f"{primary_benchmark.beta:.2f}",
                    'Information Ratio': f"{primary_benchmark.information_ratio:.2f}"
                })

            summary_data.append(row_data)

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, width='stretch')

        # Best strategy highlight
        if results:
            best_result = max(results, key=lambda x: x['metrics'].sharpe_ratio)
            st.info(f"🏆 Best performing strategy: **{best_result['strategy_name']}** "
                   f"(Sharpe: {best_result['metrics'].sharpe_ratio:.2f})")

        # Detailed results for each strategy
        for result in results:
            with st.expander(f"📈 Detailed Results: {result['strategy_name']}", expanded=False):
                self._display_detailed_result(result, symbols, start_date, end_date)

    def _display_detailed_result(self, result: Dict, symbols: List[str],
                               start_date: str, end_date: str):
        """Display detailed results for a single strategy."""
        metrics = result['metrics']

        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Return", f"{metrics.total_return:.2%}")
            st.metric("Annual Return", f"{metrics.annualized_return:.2%}")

        with col2:
            st.metric("Volatility", f"{metrics.volatility:.2%}")
            st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")

        with col3:
            st.metric("Max Drawdown", f"{metrics.max_drawdown:.2%}")
            st.metric("Calmar Ratio", f"{metrics.calmar_ratio:.2f}")

        with col4:
            st.metric("Win Rate", f"{metrics.win_rate:.2%}")
            st.metric("Total Trades", metrics.total_trades)

        # Benchmark comparison section
        if 'benchmarking' in result and result['benchmarking']:
            st.subheader("🏁 Benchmark Comparison")

            for benchmark_result in result['benchmarking']:
                with st.expander(f"📊 vs {benchmark_result.benchmark_name}", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Strategy Return", f"{benchmark_result.strategy_return:.2%}")
                        st.metric("Benchmark Return", f"{benchmark_result.benchmark_return:.2%}")

                    with col2:
                        st.metric("Excess Return", f"{benchmark_result.excess_return:.2%}",
                                delta=f"{benchmark_result.excess_return:.2%}")
                        st.metric("Alpha", f"{benchmark_result.alpha:.2%}",
                                delta=f"{benchmark_result.alpha:.2%}")

                    with col3:
                        st.metric("Beta", f"{benchmark_result.beta:.2f}")
                        st.metric("Tracking Error", f"{benchmark_result.tracking_error:.2%}")

                    with col4:
                        st.metric("Information Ratio", f"{benchmark_result.information_ratio:.2f}")
                        st.metric("Sharpe Ratio", f"{benchmark_result.sharpe_ratio:.2f}")

                    # Statistical significance
                    if benchmark_result.alpha_p_value < 0.05:
                        st.success(f"✅ Alpha is statistically significant (p-value: {benchmark_result.alpha_p_value:.3f})")
                    else:
                        st.info(f"ℹ️ Alpha is not statistically significant (p-value: {benchmark_result.alpha_p_value:.3f})")

                    # Performance attribution
                    st.subheader("Performance Attribution")
                    attr_col1, attr_col2 = st.columns(2)

                    with attr_col1:
                        st.metric("Market Timing", f"{benchmark_result.market_timing:.2%}")
                        st.metric("Security Selection", f"{benchmark_result.security_selection:.2%}")

                    with attr_col2:
                        st.metric("Alpha t-stat", f"{benchmark_result.alpha_t_stat:.2f}")
                        st.metric("Benchmark Volatility", f"{benchmark_result.benchmark_volatility:.2%}")

        # Portfolio value chart
        if 'portfolio_history' in result and result['portfolio_history']:
            st.subheader("Portfolio Value Over Time")

            history_data = []
            for snapshot in result['portfolio_history']:
                history_data.append({
                    'Date': pd.to_datetime(snapshot['timestamp']),
                    'Portfolio Value': snapshot['total_value'],
                    'Cash': snapshot['cash']
                })

            if history_data:
                chart_df = pd.DataFrame(history_data)
                chart_df = chart_df.set_index('Date')

                if PLOTLY_AVAILABLE:
                    fig = px.line(chart_df, y='Portfolio Value',
                                title=f"{result['strategy_name']} - Portfolio Value")
                    fig.add_hline(y=result['initial_cash'], line_dash="dash",
                                annotation_text="Initial Investment")
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.line_chart(chart_df['Portfolio Value'], width='stretch')

        # Trade history
        if 'trades' in result and result['trades']:
            st.subheader("Trade History")

            trades_df = pd.DataFrame(result['trades'])
            if not trades_df.empty:
                # Format for display
                display_df = trades_df.copy()
                display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
                display_df['price'] = display_df['price'].round(2)
                display_df['commission'] = display_df['commission'].round(2)
                display_df['total_cost'] = display_df['total_cost'].round(2)

                st.dataframe(display_df, width='stretch')

                # Trade statistics
                buy_trades = len(display_df[display_df['side'] == 'BUY'])
                sell_trades = len(display_df[display_df['side'] == 'SELL'])

                st.info(f"📊 Total Trades: {len(display_df)} | "
                       f"Buys: {buy_trades} | Sells: {sell_trades}")

        # Risk metrics
        st.subheader("Risk Analysis")

        risk_col1, risk_col2 = st.columns(2)

        with risk_col1:
            st.metric("VaR (95%)", f"{metrics.var_95:.2%}")
            st.metric("Expected Shortfall (95%)", f"{metrics.expected_shortfall_95:.2%}")

        with risk_col2:
            st.metric("Sortino Ratio", f"{metrics.sortino_ratio:.2f}")
            st.metric("Profit Factor", f"{metrics.profit_factor:.2f}")

        # Trading metrics
        if metrics.total_trades > 0:
            st.subheader("Trading Performance")

            trade_col1, trade_col2, trade_col3 = st.columns(3)

            with trade_col1:
                st.metric("Winning Trades", f"{metrics.winning_trades} ({metrics.win_rate:.1%})")
                st.metric("Average Win", f"${metrics.avg_win:.2f}")

            with trade_col2:
                st.metric("Losing Trades", f"{metrics.losing_trades}")
                st.metric("Average Loss", f"${metrics.avg_loss:.2f}")

            with trade_col3:
                st.metric("Profit Factor", f"{metrics.profit_factor:.2f}")
                st.metric("Total P&L", f"${(metrics.winning_trades * metrics.avg_win) - (metrics.losing_trades * metrics.avg_loss):.2f}")

    # Monitoring Methods
    async def start_monitoring(self):
        """Start background monitoring tasks."""
        if not self.monitoring_enabled:
            return

        try:
            self.logger.info("Starting monitoring tasks")

            # Start monitoring task
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info("Monitoring tasks started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {str(e)}")

    async def stop_monitoring(self):
        """Stop background monitoring tasks."""
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.logger.info("Monitoring tasks stopped")

    async def _monitoring_loop(self):
        """Background monitoring loop."""
        last_health_check = 0
        last_metrics_collection = 0

        while self.monitoring_enabled:
            try:
                current_time = time.time()

                # Health checks
                if current_time - last_health_check >= self.health_check_interval:
                    await self._perform_health_checks()
                    last_health_check = current_time

                # Metrics collection
                if current_time - last_metrics_collection >= self.metrics_collection_interval:
                    await self._collect_metrics()
                    last_metrics_collection = current_time

                # Sleep for a short interval
                await asyncio.sleep(5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {str(e)}")
                await asyncio.sleep(10)  # Back off on errors

    async def _perform_health_checks(self):
        """Perform comprehensive health checks."""
        try:
            # System health
            system_health = get_system_health()

            # Record health status
            overall_status = system_health.get('overall_status', 'UNKNOWN')
            record_trading_metric('system_health_status', overall_status, {'component': 'system'})

            # Check portfolio health
            if self.portfolio:
                portfolio_value = self.portfolio.calculate_total_equity()
                record_trading_metric('portfolio_value', portfolio_value, {'component': 'portfolio'})

                # Check for unhealthy positions (large losses)
                unhealthy_positions = 0
                for position in self.portfolio.positions.values():
                    if position.unrealized_pnl < -1000:  # More than $1000 loss
                        unhealthy_positions += 1

                record_trading_metric('unhealthy_positions', unhealthy_positions, {'component': 'portfolio'})

            # Check broker connectivity
            if self.broker:
                broker_healthy = await self._check_broker_health()
                record_trading_metric('broker_health', 1 if broker_healthy else 0, {'component': 'broker'})

            # Check strategy health
            strategy_health_count = 0
            for strategy in self.strategies:
                if hasattr(strategy, 'is_healthy') and strategy.is_healthy:
                    strategy_health_count += 1

            record_trading_metric('healthy_strategies', strategy_health_count, {'component': 'strategies'})
            record_trading_metric('total_strategies', len(self.strategies), {'component': 'strategies'})

            self.logger.debug(f"Health check completed: {overall_status}")

        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")

    async def _collect_metrics(self):
        """Collect system and application metrics."""
        try:
            # System metrics
            system_metrics = get_system_metrics()
            if 'cpu' in system_metrics:
                record_trading_metric('cpu_usage_percent', system_metrics['cpu']['percent'], {'component': 'system'})
            if 'memory' in system_metrics:
                record_trading_metric('memory_usage_percent', system_metrics['memory']['percent'], {'component': 'system'})
                record_trading_metric('memory_used_gb', system_metrics['memory']['used_gb'], {'component': 'system'})

            # Application metrics
            if self.portfolio:
                record_trading_metric('portfolio_positions_count', len(self.portfolio.positions), {'component': 'portfolio'})
                record_trading_metric('portfolio_cash', self.portfolio.current_cash, {'component': 'portfolio'})

            # Signal metrics
            if hasattr(self, 'signals_history'):
                recent_signals = [s for s in self.signals_history
                                if (datetime.now() - s.timestamp).total_seconds() < 3600]  # Last hour
                record_trading_metric('signals_last_hour', len(recent_signals), {'component': 'signals'})

            # Strategy metrics
            for i, strategy in enumerate(self.strategies):
                if hasattr(strategy, 'performance_stats'):
                    stats = strategy.performance_stats
                    for key, value in stats.items():
                        record_trading_metric(f'strategy_{key}', value,
                                            {'component': 'strategy', 'strategy_id': i, 'strategy_name': strategy.name})

        except Exception as e:
            self.logger.error(f"Metrics collection failed: {str(e)}")

    async def _check_broker_health(self) -> bool:
        """Check if broker is healthy and responsive."""
        try:
            # Simple connectivity check - try to get account info
            if hasattr(self.broker, 'get_account_info'):
                account_info = await self.broker.get_account_info()
                return account_info is not None
            elif hasattr(self.broker, 'connect'):
                # For brokers without account info, just check connection
                return await self.broker.connect()
            else:
                return True  # Assume healthy if no check method available
        except Exception as e:
            self.logger.warning(f"Broker health check failed: {str(e)}")
            return False

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status for display."""
        try:
            system_health = get_system_health()

            # Add application-specific health info
            app_health = {
                'portfolio_initialized': self.portfolio is not None,
                'broker_connected': self.broker is not None,
                'strategies_loaded': len(self.strategies) > 0,
                'auto_trading_active': self.auto_trading_enabled,
                'monitoring_active': self.monitoring_enabled
            }

            return {
                'system': system_health,
                'application': app_health,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary for display."""
        try:
            # Get recent metrics (last 5 minutes)
            since = datetime.now() - timedelta(minutes=5)

            summaries = {}
            metric_names = [
                'portfolio_value', 'cpu_usage_percent', 'memory_usage_percent',
                'signals_last_hour', 'healthy_strategies', 'unhealthy_positions'
            ]

            for name in metric_names:
                summary = metrics_collector.get_metric_summary(name, since=since)
                if summary['count'] > 0:
                    summaries[name] = summary

            return summaries

        except Exception as e:
            return {'error': str(e)}

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for a given price dataframe."""
        try:
            # Ensure we have the required columns
            if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                return df

            # Simple Moving Averages
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            df['SMA_50'] = df['close'].rolling(window=50).mean()
            df['SMA_200'] = df['close'].rolling(window=200).mean()

            # Exponential Moving Averages
            df['EMA_12'] = df['close'].ewm(span=12).mean()
            df['EMA_26'] = df['close'].ewm(span=26).mean()

            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_histogram'] = df['MACD'] - df['MACD_signal']

            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            df['BB_middle'] = df['close'].rolling(window=bb_period).mean()
            bb_std_dev = df['close'].rolling(window=bb_period).std()
            df['BB_upper'] = df['BB_middle'] + (bb_std_dev * bb_std)
            df['BB_lower'] = df['BB_middle'] - (bb_std_dev * bb_std)

            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # Volume Moving Average
            df['Volume_MA'] = df['volume'].rolling(window=20).mean()

            return df

        except Exception as e:
            return df

    def get_stock_data(self, symbol: str, period: str = "6mo") -> pd.DataFrame:
        """Get stock data for technical analysis."""
        try:
            if not YFINANCE_AVAILABLE:
                # Return simulated data for demo
                dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
                np.random.seed(42)  # For consistent demo data
                price = 100 + np.random.randn(len(dates)).cumsum() * 2
                volume = np.random.randint(1000000, 5000000, len(dates))

                df = pd.DataFrame({
                    'open': price * (1 + np.random.randn(len(dates)) * 0.01),
                    'high': price * (1 + abs(np.random.randn(len(dates))) * 0.02),
                    'low': price * (1 - abs(np.random.randn(len(dates))) * 0.02),
                    'close': price,
                    'volume': volume
                }, index=dates)

                return df

            # Get real data using yfinance
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period)
            except Exception as e:
                st.warning(f"Failed to fetch market data for {symbol} from yfinance: {str(e)}")
                return pd.DataFrame()

            if df.empty:
                return pd.DataFrame()

            # Normalize column names
            df.columns = [col.lower() for col in df.columns]

            return df

        except Exception as e:
            return pd.DataFrame()

    def render_technical_analysis_tab(self):
        """Render the technical analysis tab with interactive charts."""
        st.header("📊 Technical Analysis")

        if not PLOTLY_AVAILABLE:
            st.error("Plotly is required for technical analysis charts")
            return

        # Enhanced controls section
        col1, col2 = st.columns([2, 1])

        with col1:
            # Enhanced ticker selector (single selection for technical analysis)
            available_tickers = st.session_state.get('selected_tickers', ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'])
            if not available_tickers:
                available_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']

            selected_tickers = self.render_enhanced_ticker_selector(
                key="technical_analysis",
                default_selected=[available_tickers[0]] if available_tickers else ['AAPL'],
                allow_multiple=False,
                show_sectors=False
            )
            selected_ticker = selected_tickers[0] if selected_tickers else None

        with col2:
            # Enhanced time period selector
            time_period = self.render_enhanced_time_period_selector(
                key="technical_analysis",
                default_period="6mo"
            )

        # Chart type selector
        chart_type = st.selectbox(
            "📊 Chart Type",
            options=["Candlestick", "OHLC", "Line"],
            help="Select chart visualization type",
            key="chart_type_selector"
        )

        if not selected_ticker:
            st.warning("Please select a ticker to analyze")
            return

        # Get stock data
        with st.spinner(f"Loading data for {selected_ticker}..."):
            df = self.get_stock_data(selected_ticker, time_period)

        if df.empty:
            st.error(f"No data available for {selected_ticker}")
            return

        # Calculate technical indicators
        df = self.calculate_technical_indicators(df)

        # Technical Indicators Selection
        st.subheader("📈 Chart Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Moving Averages**")
            show_sma20 = st.checkbox("SMA 20", value=True)
            show_sma50 = st.checkbox("SMA 50", value=True)
            show_sma200 = st.checkbox("SMA 200", value=False)
            show_ema12 = st.checkbox("EMA 12", value=False)
            show_ema26 = st.checkbox("EMA 26", value=False)

        with col2:
            st.markdown("**Other Indicators**")
            show_bollinger = st.checkbox("Bollinger Bands", value=True)
            show_volume = st.checkbox("Volume", value=True)
            show_rsi = st.checkbox("RSI", value=False)
            show_macd = st.checkbox("MACD", value=False)

        # Create the main price chart
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_width=[0.7, 0.15, 0.15],
            subplot_titles=[f'{selected_ticker} Price Chart', 'Volume', 'RSI']
        )

        # Price chart (candlestick or line)
        if chart_type == "Candlestick":
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name=selected_ticker,
                    increasing_line_color='green',
                    decreasing_line_color='red'
                ),
                row=1, col=1
            )
        elif chart_type == "OHLC":
            fig.add_trace(
                go.Ohlc(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name=selected_ticker
                ),
                row=1, col=1
            )
        else:  # Line chart
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['close'],
                    mode='lines',
                    name=f'{selected_ticker} Close',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )

        # Add Moving Averages
        if show_sma20 and 'SMA_20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['SMA_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='orange', width=1),
                    opacity=0.8
                ),
                row=1, col=1
            )

        if show_sma50 and 'SMA_50' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['SMA_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='red', width=1),
                    opacity=0.8
                ),
                row=1, col=1
            )

        if show_sma200 and 'SMA_200' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['SMA_200'],
                    mode='lines',
                    name='SMA 200',
                    line=dict(color='purple', width=1),
                    opacity=0.8
                ),
                row=1, col=1
            )

        if show_ema12 and 'EMA_12' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['EMA_12'],
                    mode='lines',
                    name='EMA 12',
                    line=dict(color='cyan', width=1, dash='dash'),
                    opacity=0.8
                ),
                row=1, col=1
            )

        if show_ema26 and 'EMA_26' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['EMA_26'],
                    mode='lines',
                    name='EMA 26',
                    line=dict(color='magenta', width=1, dash='dash'),
                    opacity=0.8
                ),
                row=1, col=1
            )

        # Add Bollinger Bands
        if show_bollinger and all(col in df.columns for col in ['BB_upper', 'BB_lower', 'BB_middle']):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BB_upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='gray', width=1),
                    opacity=0.5
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BB_lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='gray', width=1),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)',
                    opacity=0.5
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BB_middle'],
                    mode='lines',
                    name='BB Middle',
                    line=dict(color='gray', width=1, dash='dot'),
                    opacity=0.5
                ),
                row=1, col=1
            )

        # Add Volume
        if show_volume and 'volume' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['volume'],
                    name='Volume',
                    marker_color='rgba(0,150,255,0.6)',
                    yaxis='y2'
                ),
                row=2, col=1
            )

            if 'Volume_MA' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['Volume_MA'],
                        mode='lines',
                        name='Volume MA',
                        line=dict(color='red', width=1),
                        yaxis='y2'
                    ),
                    row=2, col=1
                )

        # Add RSI
        if show_rsi and 'RSI' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=2)
                ),
                row=3, col=1
            )

            # Add RSI reference lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=3, col=1)

        # Update layout
        fig.update_layout(
            title=f"Technical Analysis - {selected_ticker}",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_white",
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01
            )
        )

        # Remove range slider for cleaner look
        fig.update_layout(xaxis_rangeslider_visible=False)

        # Display the chart
        st.plotly_chart(fig, width='stretch')

        # Display MACD if selected
        if show_macd and all(col in df.columns for col in ['MACD', 'MACD_signal', 'MACD_histogram']):
            st.subheader("📊 MACD Indicator")

            macd_fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                row_heights=[0.7, 0.3],
                subplot_titles=['MACD Line & Signal', 'MACD Histogram']
            )

            # MACD Line and Signal
            macd_fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MACD'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )

            macd_fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MACD_signal'],
                    mode='lines',
                    name='Signal',
                    line=dict(color='red', width=2)
                ),
                row=1, col=1
            )

            # MACD Histogram
            colors = ['green' if x >= 0 else 'red' for x in df['MACD_histogram']]
            macd_fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['MACD_histogram'],
                    name='Histogram',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )

            macd_fig.update_layout(
                title="MACD Analysis",
                height=400,
                template="plotly_white",
                showlegend=True
            )

            st.plotly_chart(macd_fig, width='stretch')

        # Current values summary
        st.subheader("📋 Current Technical Values")

        if not df.empty:
            latest = df.iloc[-1]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Current Price", f"${latest['close']:.2f}")
                if 'SMA_20' in df.columns and not pd.isna(latest['SMA_20']):
                    st.metric("SMA 20", f"${latest['SMA_20']:.2f}")

            with col2:
                if 'RSI' in df.columns and not pd.isna(latest['RSI']):
                    rsi_val = latest['RSI']
                    rsi_color = "🔴" if rsi_val > 70 else "🟢" if rsi_val < 30 else "🟡"
                    st.metric("RSI", f"{rsi_val:.1f} {rsi_color}")

                if 'SMA_50' in df.columns and not pd.isna(latest['SMA_50']):
                    st.metric("SMA 50", f"${latest['SMA_50']:.2f}")

            with col3:
                if 'MACD' in df.columns and not pd.isna(latest['MACD']):
                    st.metric("MACD", f"{latest['MACD']:.4f}")

                if 'BB_upper' in df.columns and not pd.isna(latest['BB_upper']):
                    st.metric("BB Upper", f"${latest['BB_upper']:.2f}")

            with col4:
                if 'volume' in df.columns:
                    st.metric("Volume", f"{latest['volume']:,.0f}")

                if 'BB_lower' in df.columns and not pd.isna(latest['BB_lower']):
                    st.metric("BB Lower", f"${latest['BB_lower']:.2f}")

    def render_interactive_dataframe(self, df: pd.DataFrame, title: str = "",
                                   show_filters: bool = True, show_export: bool = True,
                                   default_sort_column: str = None, ascending: bool = False) -> pd.DataFrame:
        """
        Render an interactive dataframe with filtering, sorting, and export capabilities.

        Args:
            df: DataFrame to display
            title: Optional title for the table
            show_filters: Whether to show column filters
            show_export: Whether to show export button
            default_sort_column: Column to sort by default
            ascending: Sort order for default column

        Returns:
            Filtered and sorted DataFrame
        """
        if df.empty:
            st.info("No data available")
            return df

        # Create a copy to avoid modifying original
        display_df = df.copy()

        # Add title if provided
        if title:
            st.subheader(title)

        # Create controls row
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

        with col1:
            # Search functionality
            search_term = st.text_input(
                f"🔍 Search in {title or 'table'}",
                key=f"search_{title.replace(' ', '_').lower()}_{id(df)}",
                help="Search across all text columns"
            )

        with col2:
            # Sort controls
            sort_column = st.selectbox(
                "📊 Sort by",
                options=["None"] + list(df.columns),
                index=0 if not default_sort_column else (list(df.columns).index(default_sort_column) + 1 if default_sort_column in df.columns else 0),
                key=f"sort_{title.replace(' ', '_').lower()}_{id(df)}"
            )

        with col3:
            # Sort order
            sort_ascending = st.selectbox(
                "📈 Order",
                options=["Descending", "Ascending"],
                index=1 if ascending else 0,
                key=f"order_{title.replace(' ', '_').lower()}_{id(df)}"
            )

        with col4:
            # Export button
            if show_export and not df.empty:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Export CSV",
                    data=csv,
                    file_name=f"{title.replace(' ', '_').lower()}_export.csv",
                    mime="text/csv",
                    key=f"export_{title.replace(' ', '_').lower()}_{id(df)}"
                )

        # Apply search filter
        if search_term:
            # Search across all string columns
            mask = pd.Series(False, index=display_df.index)
            for col in display_df.columns:
                if display_df[col].dtype == 'object':  # String columns
                    mask |= display_df[col].astype(str).str.contains(search_term, case=False, na=False)
            display_df = display_df[mask]

        # Apply sorting
        if sort_column != "None" and sort_column in display_df.columns:
            ascending_bool = sort_ascending == "Ascending"
            display_df = display_df.sort_values(by=sort_column, ascending=ascending_bool)

        # Column filters (optional)
        if show_filters and not display_df.empty:
            st.markdown("#### 🎯 Column Filters")
            filter_cols = st.columns(min(len(display_df.columns), 4))

            for i, col in enumerate(display_df.columns):
                with filter_cols[i % 4]:
                    col_type = display_df[col].dtype

                    # Numeric columns - range filter
                    if pd.api.types.is_numeric_dtype(col_type):
                        if not display_df[col].isna().all():
                            min_val = float(display_df[col].min())
                            max_val = float(display_df[col].max())

                            if min_val != max_val:  # Only show filter if there's a range
                                selected_range = st.slider(
                                    f"Filter {col}",
                                    min_value=min_val,
                                    max_value=max_val,
                                    value=(min_val, max_val),
                                    key=f"filter_{col}_{id(df)}"
                                )
                                display_df = display_df[
                                    (display_df[col] >= selected_range[0]) &
                                    (display_df[col] <= selected_range[1])
                                ]

                    # String columns - multiselect filter
                    elif col_type == 'object':
                        unique_values = display_df[col].dropna().unique()
                        if len(unique_values) > 1 and len(unique_values) <= 20:  # Only for reasonable number of options
                            selected_values = st.multiselect(
                                f"Filter {col}",
                                options=sorted(unique_values),
                                default=sorted(unique_values),
                                key=f"filter_{col}_{id(df)}"
                            )
                            if selected_values:
                                display_df = display_df[display_df[col].isin(selected_values)]

        # Display row count
        st.caption(f"Showing {len(display_df)} of {len(df)} rows")

        # Display the filtered dataframe
        if not display_df.empty:
            st.dataframe(
                display_df,
                width='stretch',
                height=min(400, len(display_df) * 35 + 50)  # Dynamic height with max
            )
        else:
            st.info("No data matches the current filters")

        return display_df

    def render_enhanced_positions_table(self, positions_data: List[Dict]) -> None:
        """Render enhanced positions table with interactivity."""
        if not positions_data:
            st.info("No current positions")
            return

        df = pd.DataFrame(positions_data)

        # Convert numeric columns for proper sorting/filtering
        numeric_cols = ['Quantity', 'P&L', 'P&L %']
        for col in numeric_cols:
            if col in df.columns:
                if col.endswith('%'):
                    # Remove % and convert to float
                    df[f'{col}_numeric'] = df[col].str.replace('%', '').astype(float)
                elif col in ['Entry Price', 'Current Price', 'Market Value', 'P&L']:
                    # Remove $ and convert to float
                    df[f'{col}_numeric'] = df[col].str.replace('$', '').str.replace(',', '').astype(float)

        self.render_interactive_dataframe(
            df,
            title="📋 Current Positions",
            default_sort_column="P&L_numeric" if "P&L_numeric" in df.columns else "Ticker",
            ascending=False
        )

    def render_enhanced_trade_history_table(self, trade_data: List[Dict]) -> None:
        """Render enhanced trade history table with interactivity."""
        if not trade_data:
            st.info("No trade history available")
            return

        df = pd.DataFrame(trade_data)

        # Convert timestamp for proper sorting
        if 'Timestamp' in df.columns:
            df['Timestamp_sort'] = pd.to_datetime(df['Timestamp'])

        # Convert numeric columns
        numeric_cols = ['Quantity', 'Price', 'Value', 'P&L']
        for col in numeric_cols:
            if col in df.columns and df[col].dtype == 'object':
                # Remove $ and N/A, convert to float
                df[f'{col}_numeric'] = df[col].str.replace('$', '').str.replace(',', '').str.replace('N/A', '0').astype(float)

        self.render_interactive_dataframe(
            df,
            title="📈 Trade History",
            default_sort_column="Timestamp_sort" if "Timestamp_sort" in df.columns else "Timestamp",
            ascending=False
        )

    def render_enhanced_signals_table(self, display_data: List[Dict]) -> None:
        """Render enhanced signals table with interactivity."""
        if not display_data:
            st.info("No signals available")
            return

        df = pd.DataFrame(display_data)

        # Convert timestamp for proper sorting
        if 'Timestamp' in df.columns:
            df['Timestamp_sort'] = pd.to_datetime(df['Timestamp'])

        # Convert confidence to numeric for sorting/filtering
        if 'Confidence' in df.columns:
            df['Confidence_numeric'] = df['Confidence'].str.replace('%', '').astype(float)

        # Convert price to numeric
        if 'Price' in df.columns:
            df['Price_numeric'] = df['Price'].str.replace('$', '').str.replace(',', '').str.replace('N/A', '0').astype(float)

        # Add action-based filtering
        st.markdown("#### 🎯 Quick Filters")
        col1, col2, col3 = st.columns(3)

        with col1:
            show_buy = st.checkbox("🟢 Buy Signals", value=True, key="filter_buy_signals")
        with col2:
            show_sell = st.checkbox("🔴 Sell Signals", value=True, key="filter_sell_signals")
        with col3:
            confidence_threshold = st.slider(
                "Min Confidence %",
                min_value=0,
                max_value=100,
                value=0,
                step=5,
                key="confidence_threshold_filter"
            )

        # Apply quick filters
        filtered_df = df.copy()

        # Action filter
        action_filter = []
        if show_buy:
            action_filter.append('BUY')
        if show_sell:
            action_filter.append('SELL')

        if action_filter:
            filtered_df = filtered_df[filtered_df['Action'].isin(action_filter)]

        # Confidence filter
        if 'Confidence_numeric' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Confidence_numeric'] >= confidence_threshold]

        self.render_interactive_dataframe(
            filtered_df,
            title="📡 Trading Signals",
            default_sort_column="Timestamp_sort" if "Timestamp_sort" in filtered_df.columns else "Timestamp",
            ascending=False
        )

    def get_comprehensive_ticker_list(self) -> Dict[str, List[str]]:
        """Get comprehensive list of popular tickers organized by sectors."""
        return {
            'Technology': [
                'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'ADBE',
                'CRM', 'INTC', 'AMD', 'ORCL', 'CSCO', 'AVGO', 'QCOM', 'TXN', 'IBM', 'UBER',
                'LYFT', 'SPOT', 'ZOOM', 'SHOP', 'SQ', 'PYPL', 'NOW', 'TEAM', 'DDOG', 'SNOW',
                'PLTR', 'RBLX', 'TWTR', 'SNAP', 'PINS', 'DOCU', 'ZM', 'OKTA', 'CRWD'
            ],
            'Finance': [
                'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'USB', 'PNC', 'TFC',
                'COF', 'BLK', 'SPGI', 'CME', 'ICE', 'V', 'MA', 'PYPL', 'SQ', 'AFRM'
            ],
            'Healthcare': [
                'JNJ', 'PFE', 'UNH', 'MRK', 'ABT', 'TMO', 'DHR', 'BMY', 'LLY', 'AMGN',
                'GILD', 'REGN', 'VRTX', 'BIIB', 'ISRG', 'MDT', 'SYK', 'BSX', 'EW', 'DXCM',
                'MRNA', 'BNTX', 'ZTS', 'CVS', 'WBA', 'MCK', 'COR', 'ABC'
            ],
            'Consumer': [
                'KO', 'PEP', 'WMT', 'HD', 'MCD', 'DIS', 'NKE', 'SBUX', 'COST', 'TGT',
                'LOW', 'TJX', 'ROST', 'LULU', 'ETSY', 'ULTA', 'BBY', 'AMZN', 'TSLA'
            ],
            'Energy': [
                'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'OXY', 'HES',
                'HAL', 'BKR', 'ENPH', 'SEDG', 'FSLR', 'NEE', 'DUK'
            ],
            'Industrials': [
                'BA', 'CAT', 'GE', 'HON', 'MMM', 'UNP', 'UPS', 'FDX', 'LMT', 'RTX',
                'GD', 'NOC', 'DE', 'EMR', 'ETN', 'ITW', 'ROK', 'PH', 'CARR'
            ],
            'Materials': [
                'LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'GOLD', 'SCCO', 'MOS', 'CF',
                'NUE', 'STLD', 'PKG', 'IFF', 'FMC', 'LYB'
            ],
            'Real Estate': [
                'AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'PSA', 'O', 'WELL', 'AVB', 'EQR',
                'VTR', 'BXP', 'REG', 'HST', 'PEAK'
            ],
            'Communication': [
                'VZ', 'T', 'TMUS', 'CMCSA', 'CHTR', 'DISH', 'EA', 'TTWO', 'ATVI',
                'ROKU', 'TWTR', 'SNAP', 'PINS', 'MTCH'
            ],
            'Utilities': [
                'NEE', 'DUK', 'SO', 'D', 'EXC', 'AEP', 'SRE', 'PEG', 'ED', 'XEL',
                'WEC', 'ES', 'AWK', 'PPL', 'AEE'
            ],
            'ETFs': [
                'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'BND', 'AGG', 'LQD',
                'HYG', 'GLD', 'SLV', 'USO', 'XLF', 'XLK', 'XLE', 'XLI', 'XLV', 'XLU'
            ]
        }

    def render_enhanced_ticker_selector(self, key: str, default_selected: List[str] = None,
                                      allow_multiple: bool = True, show_sectors: bool = True) -> List[str]:
        """
        Render an enhanced ticker selector with search, sectors, and presets.

        Args:
            key: Unique key for the widget
            default_selected: Default selected tickers
            allow_multiple: Whether to allow multiple selection
            show_sectors: Whether to show sector organization

        Returns:
            List of selected ticker symbols
        """
        tickers_by_sector = self.get_comprehensive_ticker_list()
        all_tickers = []
        for sector_tickers in tickers_by_sector.values():
            all_tickers.extend(sector_tickers)
        all_tickers = sorted(list(set(all_tickers)))  # Remove duplicates and sort

        if default_selected is None:
            default_selected = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']

        # Initialize session state
        session_key = f"selected_tickers_{key}"
        if session_key not in st.session_state:
            st.session_state[session_key] = default_selected

        # Search functionality
        col1, col2 = st.columns([3, 1])

        with col1:
            search_term = st.text_input(
                "🔍 Search Tickers",
                key=f"ticker_search_{key}",
                help="Type ticker symbol or company name",
                placeholder="e.g., AAPL, Apple, Microsoft..."
            )

        with col2:
            if st.button("🔄 Reset", key=f"reset_{key}"):
                st.session_state[session_key] = default_selected
                st.rerun()

        # Filter tickers based on search
        if search_term:
            filtered_tickers = [t for t in all_tickers if search_term.upper() in t.upper()]
            if len(filtered_tickers) > 0:
                st.caption(f"Found {len(filtered_tickers)} matches")
        else:
            filtered_tickers = all_tickers

        # Main selector
        if allow_multiple:
            # Quick action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("✅ Select All", key=f"select_all_{key}"):
                    st.session_state[session_key] = filtered_tickers
                    st.rerun()

            with col2:
                if st.button("❌ Clear All", key=f"clear_all_{key}"):
                    st.session_state[session_key] = []
                    st.rerun()

            with col3:
                if st.button("🎲 Random 10", key=f"random_{key}"):
                    import random
                    st.session_state[session_key] = random.sample(filtered_tickers, min(10, len(filtered_tickers)))
                    st.rerun()

            # Multi-select with current selection
            selected = st.multiselect(
                f"Select Tickers ({len(st.session_state[session_key])}/{len(filtered_tickers)})",
                options=filtered_tickers,
                default=[t for t in st.session_state[session_key] if t in filtered_tickers],
                key=f"multiselect_{key}",
                help="Use search above to filter options"
            )
            st.session_state[session_key] = selected

        else:
            # Single select
            current_selection = st.session_state[session_key][0] if st.session_state[session_key] else filtered_tickers[0]
            selected_single = st.selectbox(
                "Select Ticker",
                options=filtered_tickers,
                index=filtered_tickers.index(current_selection) if current_selection in filtered_tickers else 0,
                key=f"selectbox_{key}"
            )
            st.session_state[session_key] = [selected_single]
            selected = [selected_single]

        return selected

    def render_enhanced_date_range_picker(self, key: str, default_start: datetime = None,
                                        default_end: datetime = None, show_presets: bool = True) -> tuple:
        """
        Render an enhanced date range picker with presets and validation.

        Args:
            key: Unique key for the widget
            default_start: Default start date
            default_end: Default end date
            show_presets: Whether to show preset buttons

        Returns:
            Tuple of (start_date, end_date)
        """
        if default_start is None:
            default_start = datetime.now() - timedelta(days=180)  # 6 months ago
        if default_end is None:
            default_end = datetime.now()

        if show_presets:
            st.markdown("#### 📅 Quick Date Ranges")

            preset_cols = st.columns(5)
            presets = {
                "Last Week": (datetime.now() - timedelta(days=7), datetime.now()),
                "Last Month": (datetime.now() - timedelta(days=30), datetime.now()),
                "Last 3M": (datetime.now() - timedelta(days=90), datetime.now()),
                "Last 6M": (datetime.now() - timedelta(days=180), datetime.now()),
                "Last Year": (datetime.now() - timedelta(days=365), datetime.now())
            }

            for i, (label, (start, end)) in enumerate(presets.items()):
                with preset_cols[i]:
                    if st.button(label, key=f"preset_{label.replace(' ', '_')}_{key}"):
                        st.session_state[f"start_date_{key}"] = start.date()
                        st.session_state[f"end_date_{key}"] = end.date()
                        st.rerun()

        # Date inputs
        col1, col2 = st.columns(2)

        with col1:
            start_date = st.date_input(
                "📅 Start Date",
                value=st.session_state.get(f"start_date_{key}", default_start.date()),
                key=f"start_input_{key}",
                help="Select the start date for analysis"
            )
            st.session_state[f"start_date_{key}"] = start_date

        with col2:
            end_date = st.date_input(
                "📅 End Date",
                value=st.session_state.get(f"end_date_{key}", default_end.date()),
                key=f"end_input_{key}",
                help="Select the end date for analysis"
            )
            st.session_state[f"end_date_{key}"] = end_date

        # Validation
        if start_date >= end_date:
            st.error("❌ End date must be after start date")
            return None, None

        if (end_date - start_date).days > 1095:  # 3 years
            st.warning("⚠️ Date range is longer than 3 years. This may impact performance.")

        # Show selected range info
        days_diff = (end_date - start_date).days
        st.caption(f"📊 Selected range: {days_diff} days ({start_date} to {end_date})")

        return start_date, end_date

    def render_enhanced_time_period_selector(self, key: str, default_period: str = "6mo") -> str:
        """
        Render an enhanced time period selector with descriptions.

        Args:
            key: Unique key for the widget
            default_period: Default time period

        Returns:
            Selected time period string
        """
        periods = {
            "1d": "1 Day (Intraday)",
            "5d": "5 Days",
            "1mo": "1 Month",
            "3mo": "3 Months",
            "6mo": "6 Months (Default)",
            "1y": "1 Year",
            "2y": "2 Years",
            "5y": "5 Years",
            "10y": "10 Years",
            "ytd": "Year to Date",
            "max": "Maximum Available"
        }

        # Quick preset buttons
        st.markdown("#### ⏰ Quick Time Periods")
        preset_cols = st.columns(6)
        quick_periods = ["1mo", "3mo", "6mo", "1y", "2y", "ytd"]

        for i, period in enumerate(quick_periods):
            with preset_cols[i]:
                if st.button(periods[period].split('(')[0].strip(), key=f"period_{period}_{key}"):
                    st.session_state[f"time_period_{key}"] = period
                    st.rerun()

        # Main selector
        current_period = st.session_state.get(f"time_period_{key}", default_period)

        selected_period = st.selectbox(
            "📊 Time Period",
            options=list(periods.keys()),
            index=list(periods.keys()).index(current_period) if current_period in periods else 2,
            format_func=lambda x: periods[x],
            key=f"period_select_{key}",
            help="Select the time period for data analysis"
        )

        st.session_state[f"time_period_{key}"] = selected_period

        # Show period info
        if selected_period in ["1d", "5d"]:
            st.info("ℹ️ Short periods may have limited technical indicator data")
        elif selected_period in ["5y", "10y", "max"]:
            st.info("ℹ️ Long periods may take longer to load and process")

        return selected_period

    def show_notification(self, message: str, notification_type: str = "success", icon: str = None):
        """
        Show a toast notification to the user.

        Args:
            message: The message to display
            notification_type: success, error, warning, info
            icon: Optional emoji icon
        """
        try:
            if notification_type == "success":
                st.toast(f"✅ {message}", icon="✅" if not icon else icon)
            elif notification_type == "error":
                st.toast(f"❌ {message}", icon="❌" if not icon else icon)
            elif notification_type == "warning":
                st.toast(f"⚠️ {message}", icon="⚠️" if not icon else icon)
            elif notification_type == "info":
                st.toast(f"ℹ️ {message}", icon="ℹ️" if not icon else icon)
            else:
                st.toast(message, icon=icon or "📢")
        except Exception as e:
            # Fallback to regular streamlit notifications if toast fails
            if notification_type == "success":
                st.success(message)
            elif notification_type == "error":
                st.error(message)
            elif notification_type == "warning":
                st.warning(message)
            else:
                st.info(message)

    def show_trade_notification(self, action: str, ticker: str, quantity: int, price: float):
        """Show specialized trade notifications."""
        if action.upper() == "BUY":
            self.show_notification(
                f"Bought {quantity} shares of {ticker} at ${price:.2f}",
                "success",
                "📈"
            )
        elif action.upper() == "SELL":
            self.show_notification(
                f"Sold {quantity} shares of {ticker} at ${price:.2f}",
                "success",
                "📉"
            )
        elif action.upper() == "CLOSE":
            self.show_notification(
                f"Closed position in {ticker} - {quantity} shares at ${price:.2f}",
                "info",
                "🔒"
            )

    def show_signal_notification(self, signal_count: int, ticker: str = None):
        """Show signal generation notifications."""
        if ticker:
            self.show_notification(
                f"New signal generated for {ticker}",
                "info",
                "📡"
            )
        else:
            self.show_notification(
                f"{signal_count} new signals generated",
                "info",
                "📡"
            )

    def show_system_notification(self, message: str, notification_type: str = "info"):
        """Show system status notifications."""
        system_icons = {
            "success": "🟢",
            "error": "🔴",
            "warning": "🟡",
            "info": "🔵"
        }

        self.show_notification(
            message,
            notification_type,
            system_icons.get(notification_type, "⚙️")
        )

    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive risk metrics for the portfolio."""
        try:
            if not self.portfolio or not hasattr(self.portfolio, 'trade_history'):
                return {
                    'sharpe_ratio': 0.0,
                    'max_drawdown_pct': 0.0,
                    'volatility_annualized': 0.0,
                    'var_5_percent': 0.0,
                    'calmar_ratio': 0.0,
                    'sortino_ratio': 0.0,
                    'avg_trade_duration_days': 0.0,
                    'min_trade_duration': 0.0,
                    'max_trade_duration': 0.0
                }

            # Get portfolio value history and trade history
            value_history = self.portfolio.get_portfolio_value_history()
            closed_trades = [trade for trade in self.portfolio.trade_history if trade.get('action') == 'CLOSE']

            if not value_history or len(value_history) < 2:
                return {
                    'sharpe_ratio': 0.0,
                    'max_drawdown_pct': 0.0,
                    'volatility_annualized': 0.0,
                    'var_5_percent': 0.0,
                    'calmar_ratio': 0.0,
                    'sortino_ratio': 0.0,
                    'avg_trade_duration_days': 0.0,
                    'min_trade_duration': 0.0,
                    'max_trade_duration': 0.0
                }

            # Convert to DataFrame for analysis
            df = pd.DataFrame(value_history)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

            # Calculate daily returns
            df['returns'] = df['portfolio_value'].pct_change().fillna(0)

            # Risk-free rate (approximate 2% annually)
            risk_free_rate_daily = 0.02 / 252

            # Sharpe Ratio
            if df['returns'].std() > 0:
                sharpe_ratio = (df['returns'].mean() - risk_free_rate_daily) / df['returns'].std() * np.sqrt(252)
            else:
                sharpe_ratio = 0.0

            # Max Drawdown
            df['running_max'] = df['portfolio_value'].expanding().max()
            df['drawdown'] = (df['portfolio_value'] / df['running_max'] - 1) * 100
            max_drawdown_pct = df['drawdown'].min()

            # Volatility (annualized)
            volatility_annualized = df['returns'].std() * np.sqrt(252) * 100

            # Value at Risk (5%) - use last portfolio value as approximation
            last_portfolio_value = df['portfolio_value'].iloc[-1] if not df.empty else 100000
            var_5_percent = np.percentile(df['returns'], 5) * last_portfolio_value

            # Calmar Ratio (annual return / max drawdown)
            annual_return = ((df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[0]) ** (252 / len(df)) - 1) * 100
            calmar_ratio = annual_return / abs(max_drawdown_pct) if max_drawdown_pct != 0 else 0

            # Sortino Ratio (downside deviation)
            downside_returns = df['returns'][df['returns'] < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                sortino_ratio = (df['returns'].mean() - risk_free_rate_daily) / downside_returns.std() * np.sqrt(252)
            else:
                sortino_ratio = 0.0

            # Trade duration metrics
            if closed_trades:
                durations = [trade.get('duration_days', 1) for trade in closed_trades if trade.get('duration_days', 0) > 0]
                if durations:
                    avg_trade_duration_days = np.mean(durations)
                    min_trade_duration = np.min(durations)
                    max_trade_duration = np.max(durations)
                else:
                    avg_trade_duration_days = min_trade_duration = max_trade_duration = 0.0
            else:
                avg_trade_duration_days = min_trade_duration = max_trade_duration = 0.0

            return {
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown_pct': float(max_drawdown_pct),
                'volatility_annualized': float(volatility_annualized),
                'var_5_percent': float(var_5_percent),
                'calmar_ratio': float(calmar_ratio),
                'sortino_ratio': float(sortino_ratio),
                'avg_trade_duration_days': float(avg_trade_duration_days),
                'min_trade_duration': float(min_trade_duration),
                'max_trade_duration': float(max_trade_duration)
            }

        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {str(e)}")
            return {
                'sharpe_ratio': 0.0,
                'max_drawdown_pct': 0.0,
                'volatility_annualized': 0.0,
                'var_5_percent': 0.0,
                'calmar_ratio': 0.0,
                'sortino_ratio': 0.0,
                'avg_trade_duration_days': 0.0,
                'min_trade_duration': 0.0,
                'max_trade_duration': 0.0
            }

    def render_header(self):
        """Render the application header with key metrics."""
        st.title("📈 Advanced Trading Framework")

        # Key metrics in header
        if self.portfolio:
            col1, col2, col3, col4 = st.columns(4)

            current_value = self.portfolio.calculate_total_equity()
            initial_capital = self.portfolio.initial_capital
            total_return = ((current_value - initial_capital) / initial_capital) * 100

            with col1:
                st.metric("Portfolio Value", f"${current_value:,.2f}")

            with col2:
                delta_color = "normal" if total_return >= 0 else "inverse"
                st.metric("Total Return", f"{total_return:+.2f}%", delta_color=delta_color)

            with col3:
                st.metric("Cash Available", f"${self.portfolio.cash:,.2f}")

            with col4:
                open_positions = len(self.portfolio.get_all_open_positions())
                st.metric("Open Positions", open_positions)

        st.markdown("---")

    def render_sidebar(self) -> Dict[str, Any]:
        """Render the sidebar with trading controls and navigation."""
        with st.sidebar:
            st.header("🎛️ Trading Controls")

            # Trading Mode Toggle
            st.subheader("Trading Mode")
            trading_mode = st.radio(
                "Select Mode:",
                ["Manual", "Automatic"],
                index=1 if self.auto_trading_enabled else 0,
                help="Manual: You control all trades. Automatic: AI executes trades based on signals."
            )

            # Update auto trading state
            new_auto_trading = (trading_mode == "Automatic")
            if new_auto_trading != self.auto_trading_enabled:
                self.auto_trading_enabled = new_auto_trading
                st.session_state.auto_trading_enabled = new_auto_trading

            # Trading Controls
            st.subheader("Trading Actions")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("▶️ Start Trading", type="primary", use_container_width=True,
                           disabled=self.auto_trading_enabled and self.trading_loop_task and not self.trading_loop_task.done()):
                    if not self.initialized:
                        st.error("Application not initialized")
                    elif self.auto_trading_enabled:
                        # Start automated trading
                        asyncio.run(self.start_automated_trading())
                        st.success("🤖 Automated trading started!")
                        st.rerun()
                    else:
                        st.info("Manual trading mode - use signal tabs to execute trades")

            with col2:
                if st.button("⏹️ Stop Trading", type="secondary", use_container_width=True,
                           disabled=not self.auto_trading_enabled or not (self.trading_loop_task and not self.trading_loop_task.done())):
                    if self.auto_trading_enabled:
                        asyncio.run(self.stop_automated_trading())
                        st.success("🛑 Automated trading stopped!")
                        st.rerun()
                    else:
                        st.info("Manual trading mode - no automated trading to stop")

            # Emergency Controls
            st.subheader("🚨 Emergency Controls")

            if st.button("🚨 Liquidate All", type="secondary", use_container_width=True,
                        help="Emergency liquidation of all positions"):
                if st.session_state.get('confirm_liquidate', False):
                    with st.spinner("Liquidating all positions..."):
                        liquidated = asyncio.run(self.liquidate_all_positions())
                        if liquidated > 0:
                            st.success(f"✅ Liquidated {liquidated} positions")
                            st.session_state.confirm_liquidate = False
                        else:
                            st.info("No positions to liquidate")
                        st.rerun()
                else:
                    st.warning("⚠️ Click again to confirm liquidation of ALL positions")
                    st.session_state.confirm_liquidate = True

            # Risk Management
            st.subheader("🛡️ Risk Management")

            # Stop loss percentage
            stop_loss_pct = st.slider(
                "Stop Loss %",
                min_value=0.1,
                max_value=10.0,
                value=5.0,
                step=0.1,
                help="Maximum loss percentage per trade"
            )

            # Take profit percentage
            take_profit_pct = st.slider(
                "Take Profit %",
                min_value=0.1,
                max_value=20.0,
                value=10.0,
                step=0.1,
                help="Profit target percentage per trade"
            )

            # Max positions
            max_positions = st.slider(
                "Max Open Positions",
                min_value=1,
                max_value=20,
                value=5,
                help="Maximum number of concurrent positions"
            )

            # System Status
            st.subheader("📊 System Status")

            status_col1, status_col2 = st.columns(2)

            with status_col1:
                if self.initialized:
                    st.success("✅ Initialized")
                else:
                    st.error("❌ Not Initialized")

                if self.broker and self.broker.connected:
                    st.success("✅ Broker Connected")
                else:
                    st.error("❌ Broker Disconnected")

            with status_col2:
                if self.strategies:
                    st.success(f"✅ {len(self.strategies)} Strategies")
                else:
                    st.error("❌ No Strategies")

                trading_status = "🟢 Active" if (self.trading_loop_task and not self.trading_loop_task.done()) else "🔴 Inactive"
                st.write(f"Trading: {trading_status}")

            # Quick Actions
            st.subheader("⚡ Quick Actions")

            if st.button("🔄 Refresh Data", use_container_width=True):
                with st.spinner("Refreshing data..."):
                    asyncio.run(self.update_portfolio())
                    st.success("✅ Data refreshed")
                    st.rerun()

            if st.button("📊 Generate Signals", use_container_width=True):
                with st.spinner("Generating signals..."):
                    tickers = self.config.get('universe', {}).get('default_tickers', ['AAPL', 'MSFT'])
                    signals = asyncio.run(self.generate_signals(tickers))
                    st.success(f"✅ Generated {len(signals)} signals")
                    st.rerun()

            # Configuration
            with st.expander("⚙️ Quick Config"):
                dark_mode = st.checkbox("Dark Mode", value=st.session_state.get('dark_mode', False))
                if dark_mode != st.session_state.get('dark_mode', False):
                    st.session_state.dark_mode = dark_mode
                    st.rerun()

                # Trading environment indicator
                trading_env = self.config.get('trading_environment', 'paper')
                if trading_env == 'live':
                    st.error("🔴 LIVE TRADING ACTIVE")
                else:
                    st.info("🟢 PAPER TRADING MODE")

        return {
            'trading_mode': trading_mode,
            'auto_trading': self.auto_trading_enabled,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'max_positions': max_positions
        }

    def render_main_app(self):
        """Render the main application interface."""

        # Header
        self.render_header()

        # Live Trading Warning
        trading_environment = self.config.get('trading_environment', 'paper')
        if trading_environment == 'live':
            st.error("🚨 **LIVE TRADING MODE ACTIVE** 🚨")
            st.error("**WARNING:** This application is currently configured for LIVE TRADING with real money!")
            st.error("All trades executed will use actual funds. Please ensure you understand the risks.")
            st.markdown("---")

        # Sidebar
        controls = self.render_sidebar()

        # Main content area with proper layout
        main_col1, main_col2 = st.columns([1, 3])  # Sidebar space, main content

        with main_col1:
            # This column can be used for additional controls or summary
            st.empty()

        with main_col2:
            # Main content tabs
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
                "💼 Portfolio",
                "📡 Signals",
                "📈 Performance",
                "📝 Logs",
                "⚙️ Settings",
                "🔍 Monitoring",
                "🔬 Backtesting",
                "📊 Technical Analysis",
                "🤖 ML Modeling"
            ])

            with tab1:
                self.render_portfolio_tab()

            with tab2:
                self.render_signals_tab()

            with tab3:
                self.render_performance_tab()

            with tab4:
                self.render_logs_tab()

            with tab5:
                self.render_settings_tab()

            with tab6:
                self.render_monitoring_tab()

            with tab7:
                self.render_backtesting_tab()

            with tab8:
                self.render_technical_analysis_tab()

    async def start_automated_trading(self):
        """Start the automated trading loop."""
        if not self.initialized:
            raise ValueError("Application not initialized")

        if self.trading_loop_task and not self.trading_loop_task.done():
            self.logger.warning("Automated trading already running")
            return

        try:
            self.logger.info("Starting automated trading system")

            # Start trading loop
            self.trading_loop_task = asyncio.create_task(self.automated_trading_loop())

            # Start risk monitoring
            self.risk_monitoring_task = asyncio.create_task(self.automated_risk_monitoring_loop())

            self.logger.info("Automated trading system started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start automated trading: {str(e)}")
            raise

    async def stop_automated_trading(self):
        """Stop the automated trading loop."""
        try:
            self.logger.info("Stopping automated trading system")

            # Cancel trading tasks
            if self.trading_loop_task and not self.trading_loop_task.done():
                self.trading_loop_task.cancel()
                try:
                    await self.trading_loop_task
                except asyncio.CancelledError:
                    pass

            if self.risk_monitoring_task and not self.risk_monitoring_task.done():
                self.risk_monitoring_task.cancel()
                try:
                    await self.risk_monitoring_task
                except asyncio.CancelledError:
                    pass

            self.logger.info("Automated trading system stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping automated trading: {str(e)}")

    async def update_portfolio(self):
        """Update portfolio data from broker."""
        try:
            if not self.portfolio or not self.broker:
                return

            # Update market prices for open positions
            open_positions = self.portfolio.get_all_open_positions()
            for position in open_positions:
                market_data = await self.broker.get_market_data(position.ticker)
                if market_data:
                    self.portfolio.update_position_price(position.ticker, market_data.price)

            # Update portfolio value
            self.portfolio.record_portfolio_value()

        except Exception as e:
            self.logger.error(f"Error updating portfolio: {str(e)}")

    def render_portfolio_tab(self):
        """Render the portfolio overview tab."""
        st.header("💼 Portfolio Overview")

        if not self.portfolio:
            st.warning("Portfolio not initialized")
            return

        # Portfolio Summary
        st.subheader("Portfolio Summary")

        col1, col2, col3, col4 = st.columns(4)

        current_value = self.portfolio.calculate_total_equity()
        initial_capital = self.portfolio.initial_capital
        total_return = ((current_value - initial_capital) / initial_capital) * 100

        with col1:
            st.metric("Total Value", f"${current_value:,.2f}")

        with col2:
            delta_color = "normal" if total_return >= 0 else "inverse"
            st.metric("Total Return", f"{total_return:+.2f}%", delta_color=delta_color)

        with col3:
            st.metric("Cash", f"${self.portfolio.cash:,.2f}")

        with col4:
            open_positions = len(self.portfolio.get_all_open_positions())
            st.metric("Open Positions", open_positions)

        # Current Positions
        st.subheader("Current Positions")

        open_positions = self.portfolio.get_all_open_positions()
        if open_positions:
            positions_data = []
            for pos in open_positions:
                positions_data.append({
                    'Ticker': pos.ticker,
                    'Quantity': pos.quantity,
                    'Avg Cost': f"${pos.avg_cost:.2f}",
                    'Current Price': f"${pos.current_price:.2f}",
                    'Market Value': f"${pos.current_price * pos.quantity:,.2f}",
                    'Unrealized P&L': f"${pos.unrealized_pnl:,.2f}",
                    'Return %': f"{((pos.current_price - pos.avg_cost) / pos.avg_cost) * 100:.2f}%"
                })

            st.dataframe(pd.DataFrame(positions_data), use_container_width=True)
        else:
            st.info("No open positions")

        # Recent Trades
        st.subheader("Recent Trades")

        if self.portfolio.trade_history:
            recent_trades = self.portfolio.trade_history[-10:]  # Last 10 trades
            trades_data = []
            for trade in recent_trades:
                trades_data.append({
                    'Date': trade.get('timestamp', ''),
                    'Ticker': trade.get('ticker', ''),
                    'Action': trade.get('action', ''),
                    'Quantity': trade.get('quantity', 0),
                    'Price': f"${trade.get('price', 0):.2f}",
                    'Value': f"${trade.get('quantity', 0) * trade.get('price', 0):,.2f}"
                })

            st.dataframe(pd.DataFrame(trades_data), use_container_width=True)
        else:
            st.info("No recent trades")

    def render_signals_tab(self):
        """Render the signals tab."""
        st.header("📡 Trading Signals")

        # Generate Signals Button
        col1, col2 = st.columns([1, 4])

        with col1:
            if st.button("🔄 Generate Signals", type="primary"):
                with st.spinner("Generating trading signals..."):
                    tickers = self.config.get('universe', {}).get('default_tickers', ['AAPL', 'MSFT'])
                    signals = asyncio.run(self.generate_signals(tickers))
                    st.success(f"✅ Generated {len(signals)} signals")
                    st.rerun()

        # Signals Display
        signals_history = st.session_state.get('signals_data', [])

        if signals_history:
            # Convert to display format
            display_data = []
            for signal in signals_history[-20:]:  # Show last 20 signals
                display_data.append({
                    'Timestamp': signal.get('timestamp', ''),
                    'Ticker': signal.get('ticker', ''),
                    'Action': signal.get('action', ''),
                    'Confidence': f"{signal.get('confidence', 0):.2%}",
                    'Price': f"${signal.get('price', 0):.2f}" if signal.get('price', 0) else "N/A",
                    'Reasoning': signal.get('reasoning', '')[:50] + "..." if signal.get('reasoning', '') and len(signal.get('reasoning', '')) > 50 else signal.get('reasoning', '')
                })

            st.dataframe(pd.DataFrame(display_data), use_container_width=True)

            # Signal Actions (Manual Trading)
            if not self.auto_trading_enabled:
                st.subheader("Manual Trading Actions")

                # Get latest signals
                latest_signals = [s for s in signals_history if s.get('timestamp')]
                latest_signals.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

                if latest_signals:
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if st.button("✅ Execute BUY Signals", type="primary"):
                            buy_signals = [s for s in latest_signals if s.get('action') == 'BUY']
                            if buy_signals:
                                executed = asyncio.run(self.execute_signals([TradingSignal(**s) for s in buy_signals]))
                                st.success(f"✅ Executed {executed} BUY orders")
                                st.rerun()
                            else:
                                st.info("No BUY signals to execute")

                    with col2:
                        if st.button("❌ Execute SELL Signals", type="secondary"):
                            sell_signals = [s for s in latest_signals if s.get('action') == 'SELL']
                            if sell_signals:
                                executed = asyncio.run(self.execute_signals([TradingSignal(**s) for s in sell_signals]))
                                st.success(f"✅ Executed {executed} SELL orders")
                                st.rerun()
                            else:
                                st.info("No SELL signals to execute")

                    with col3:
                        if st.button("🔄 Execute All Signals"):
                            if latest_signals:
                                executed = asyncio.run(self.execute_signals([TradingSignal(**s) for s in latest_signals]))
                                st.success(f"✅ Executed {executed} orders")
                                st.rerun()
                            else:
                                st.info("No signals to execute")
        else:
            st.info("No signals available. Click 'Generate Signals' to create trading signals.")

    def render_performance_tab(self):
        """Render enhanced performance analysis dashboard."""
        st.header("📈 Performance Dashboard")

        if not self.portfolio:
            st.warning("Portfolio not initialized")
            return

        # Get comprehensive performance metrics
        trading_metrics = self._calculate_trading_performance()
        risk_metrics = self._calculate_risk_metrics()

        # Performance Overview Section
        st.subheader("🎯 Performance Overview")

        # Key Performance Indicators (Top Row)
        col1, col2, col3, col4 = st.columns(4)

        current_value = self.portfolio.calculate_total_equity()
        initial_capital = self.portfolio.initial_capital
        total_return = ((current_value - initial_capital) / initial_capital) * 100

        with col1:
            delta_color = "normal" if total_return >= 0 else "inverse"
            st.metric(
                "Total Return",
                f"{total_return:+.2f}%",
                delta=f"${current_value - initial_capital:+,.2f}",
                delta_color=delta_color
            )

        with col2:
            st.metric(
                "Current Value",
                f"${current_value:,.2f}",
                delta=f"vs ${initial_capital:,.2f} initial"
            )

        with col3:
            sharpe_ratio = risk_metrics.get('sharpe_ratio', 0)
            sharpe_color = "normal" if sharpe_ratio > 1 else "inverse" if sharpe_ratio < 0.5 else "off"
            st.metric(
                "Sharpe Ratio",
                f"{sharpe_ratio:.2f}",
                delta="Good" if sharpe_ratio > 1 else "Poor" if sharpe_ratio < 0.5 else "Fair",
                delta_color=sharpe_color
            )

        with col4:
            max_drawdown = risk_metrics.get('max_drawdown_pct', 0)
            dd_color = "normal" if max_drawdown > -10 else "inverse"
            st.metric(
                "Max Drawdown",
                f"{max_drawdown:.1f}%",
                delta="Low Risk" if max_drawdown > -10 else "High Risk",
                delta_color=dd_color
            )

        # Trading Statistics (Second Row)
        st.subheader("📊 Trading Statistics")
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            win_rate = trading_metrics['win_rate']
            wr_color = "normal" if win_rate >= 50 else "inverse"
            st.metric(
                "Win Rate",
                f"{win_rate:.1f}%",
                delta=f"{trading_metrics['winning_trades']}/{trading_metrics['total_trades']} trades",
                delta_color=wr_color
            )

        with col6:
            profit_factor = trading_metrics['profit_factor']
            pf_color = "normal" if profit_factor > 1.5 else "inverse" if profit_factor < 1 else "off"
            st.metric(
                "Profit Factor",
                f"{profit_factor:.2f}",
                delta="Excellent" if profit_factor > 2 else "Good" if profit_factor > 1.5 else "Poor",
                delta_color=pf_color
            )

        with col7:
            st.metric(
                "Avg P/L per Trade",
                f"${trading_metrics['avg_pl_per_trade']:,.2f}",
                delta=f"Total: ${trading_metrics['total_pl']:,.2f}"
            )

        with col8:
            avg_trade_duration = risk_metrics.get('avg_trade_duration_days', 0)
            st.metric(
                "Avg Hold Period",
                f"{avg_trade_duration:.1f} days",
                delta=f"Range: {risk_metrics.get('min_trade_duration', 0):.0f}-{risk_metrics.get('max_trade_duration', 0):.0f} days"
            )

        # Equity Curve
        st.subheader("📈 Portfolio Equity Curve")

        value_history = self.portfolio.get_portfolio_value_history()

        if value_history and len(value_history) > 1:
            df = pd.DataFrame(value_history)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['portfolio_value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#1f77b4', width=2)
                ))

                fig.add_hline(
                    y=initial_capital,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Initial: ${initial_capital:,.0f}"
                )

                fig.update_layout(
                    title="Portfolio Value Over Time",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(df.set_index('date')['portfolio_value'], use_container_width=True)
        else:
            st.info("Portfolio value history will appear after trading activity")

    def render_logs_tab(self):
        """Render logs tab with logs and debug info in separate columns."""
        st.header("📝 System Logs & Debug Info")

        # Create two columns
        col1, col2 = st.columns(2)

        # Left column: Logs
        with col1:
            st.subheader("📄 System Logs")
            log_file_path = "logs/trading.log"

            if os.path.exists(log_file_path):
                try:
                    with open(log_file_path, 'r') as f:
                        log_lines = f.readlines()

                    # Show last 100 lines
                    recent_logs = log_lines[-100:]

                    st.text_area(
                        "Recent Logs",
                        value=''.join(recent_logs),
                        height=400,
                        help="Showing last 100 log entries"
                    )

                    if st.button("📥 Download Full Logs"):
                        with open(log_file_path, 'r') as f:
                            st.download_button(
                                label="Download",
                                data=f.read(),
                                file_name=f"trading_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                                mime="text/plain"
                            )

                except Exception as e:
                    st.error(f"Failed to read log file: {str(e)}")
            else:
                st.info("No log file found")

        # Right column: Debug Info
        with col2:
            st.subheader("🔍 Debug Info")

            # Get trading app instance for debug info
            trading_app = st.session_state.get('trading_app')

            if trading_app:
                st.write(f"**Session initialized:** {st.session_state.get('app_initialized', False)}")
                st.write(f"**App initialized:** {trading_app.initialized}")
                st.write(f"**Portfolio exists:** {trading_app.portfolio is not None}")
                st.write(f"**Broker exists:** {trading_app.broker is not None}")
                st.write(f"**Strategies count:** {len(trading_app.strategies) if hasattr(trading_app, 'strategies') else 0}")
                st.write(f"**Signals in session:** {len(st.session_state.get('signals_data', []))}")

                # Portfolio details if available
                if trading_app.portfolio:
                    st.markdown("**Portfolio Details:**")
                    st.write(f"- Cash: ${trading_app.portfolio.cash:.2f}")
                    st.write(f"- Total Value: ${trading_app.portfolio.get_total_value():.2f}")
                    st.write(f"- Positions: {len(trading_app.portfolio.positions)}")

                # Broker details if available
                if trading_app.broker:
                    st.markdown("**Broker Details:**")
                    st.write(f"- Type: {type(trading_app.broker).__name__}")
                    st.write(f"- Connected: {trading_app.broker.connected}")

                # Configuration summary
                st.markdown("**Configuration:**")
                trading_env = self.config.get('trading_environment', 'paper')
                st.write(f"- Environment: {trading_env}")
                st.write(f"- Risk management: {self.config.get('risk_management', {}).get('enabled', False)}")
            else:
                st.warning("Trading app not initialized")

    def render_settings_tab(self):
        """Render the settings and configuration tab."""
        st.header("⚙️ Settings & Configuration")

        # Application Control Section
        st.subheader("Application Control")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🧹 Cleanup Async Tasks", type="secondary", help="Cancel all running async tasks and clean up resources"):
                with st.spinner("Cleaning up async tasks..."):
                    asyncio.run(self.cleanup())
                    st.success("✅ Async cleanup completed")
                    st.rerun()

        with col2:
            if st.button("🔄 Reset Session State", type="secondary", help="Reset all session state variables"):
                # Clear session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("✅ Session state reset")
                st.rerun()

        with col3:
            if st.button("🔧 Reinitialize App", type="secondary", help="Reinitialize the trading application"):
                with st.spinner("Reinitializing application..."):
                    # Clear cached app
                    initialize_trading_app.clear()
                    st.success("✅ Application cache cleared - will reinitialize on next run")
                    st.rerun()

        # Configuration Display
        st.subheader("Current Configuration")
        st.json(self.config)

        # System Information
        st.subheader("System Information")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Application State:**")
            st.write(f"- Initialized: {self.initialized}")
            st.write(f"- Running: {self.running}")
            st.write(f"- Auto Trading: {self.auto_trading_enabled}")
            st.write(f"- Monitoring: {self.monitoring_enabled}")

        with col2:
            st.write("**Active Tasks:**")
            trading_task = self.trading_loop_task is not None and not self.trading_loop_task.done() if self.trading_loop_task else False
            risk_task = self.risk_monitoring_task is not None and not self.risk_monitoring_task.done() if self.risk_monitoring_task else False
            monitoring_task = self.monitoring_task is not None and not self.monitoring_task.done() if self.monitoring_task else False

            st.write(f"- Trading Loop: {'🟢' if trading_task else '🔴'}")
            st.write(f"- Risk Monitoring: {'🟢' if risk_task else '🔴'}")
            st.write(f"- Health Monitoring: {'🟢' if monitoring_task else '🔴'}")

    def render_monitoring_tab(self):
        """Render the monitoring and health dashboard."""
        st.header("🔍 System Monitoring & Health")

        # Health Status Overview
        st.subheader("Health Status")

        health_status = self.get_health_status()

        col1, col2, col3 = st.columns(3)

        with col1:
            system_status = health_status.get('system', {}).get('overall_status', 'UNKNOWN')
            status_color = {
                'HEALTHY': '🟢',
                'DEGRADED': '🟡',
                'UNHEALTHY': '🔴',
                'UNKNOWN': '⚪'
            }.get(system_status, '⚪')

            st.metric("System Health", f"{status_color} {system_status}")

        with col2:
            app_health = health_status.get('application', {})
            components_healthy = sum([
                app_health.get('portfolio_initialized', False),
                app_health.get('broker_connected', False),
                app_health.get('strategies_loaded', False)
            ])
            total_components = 3

            st.metric("App Components", f"{components_healthy}/{total_components} Healthy")

        with col3:
            if self.portfolio:
                open_positions = len(self.portfolio.get_all_open_positions())
                st.metric("Open Positions", open_positions)
            else:
                st.metric("Open Positions", "N/A")

        # System Resources
        st.subheader("System Resources")

        try:
            system_metrics = get_system_metrics()

            if system_metrics:
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    cpu_percent = system_metrics.get('cpu', {}).get('percent', 0)
                    st.metric("CPU Usage", f"{cpu_percent:.1f}%")

                with col2:
                    memory_info = system_metrics.get('memory', {})
                    memory_percent = memory_info.get('percent', 0)
                    memory_used = memory_info.get('used_gb', 0)
                    memory_total = memory_info.get('total_gb', 0)

                    st.metric("Memory Usage", f"{memory_percent:.1f}%",
                            f"{memory_used:.1f}GB / {memory_total:.1f}GB")

                with col3:
                    disk_info = system_metrics.get('disk', {})
                    disk_percent = disk_info.get('percent', 0)
                    disk_used = disk_info.get('used_gb', 0)
                    disk_total = disk_info.get('total_gb', 0)

                    st.metric("Disk Usage", f"{disk_percent:.1f}%",
                            f"{disk_used:.1f}GB / {disk_total:.1f}GB")

                with col4:
                    # Network (basic)
                    network_info = system_metrics.get('network', {})
                    bytes_sent = network_info.get('bytes_sent', 0) / (1024**3)  # GB
                    bytes_recv = network_info.get('bytes_recv', 0) / (1024**3)  # GB

                    st.metric("Network I/O", f"↓{bytes_recv:.2f}GB ↑{bytes_sent:.2f}GB")
            else:
                st.error("Failed to get system metrics")
        except Exception as e:
            st.error(f"Error displaying system resources: {str(e)}")

        # Monitoring Controls
        st.subheader("Monitoring Controls")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Run Health Check Now", width='stretch'):
                with st.spinner("Running health checks..."):
                    # Force immediate health check
                    self.show_system_notification("Health check started", "info")
                    asyncio.run(self._perform_health_checks())
                    self.show_system_notification("Health check completed successfully", "success")
                    st.rerun()

        with col2:
            monitoring_status = "Disable" if self.monitoring_enabled else "Enable"
            if st.button(f"{'🔴' if self.monitoring_enabled else '🟢'} {monitoring_status} Monitoring",
                        width='stretch'):
                if self.monitoring_enabled:
                    asyncio.run(self.stop_monitoring())
                    self.monitoring_enabled = False
                    st.success("Monitoring disabled")
                else:
                    self.monitoring_enabled = True
                    asyncio.run(self.start_monitoring())
                    st.success("Monitoring enabled")
                st.rerun()

    def render_backtesting_tab(self):
        """Render the backtesting interface."""
        st.header("🔬 Strategy Backtesting")

        # Import backtesting components
        try:
            from core.backtesting import BacktestEngine, run_backtest, compare_strategies
            from strategy_layer.backtest_strategies import create_strategy
            BACKTESTING_AVAILABLE = True
        except ImportError as e:
            st.error(f"Backtesting module not available: {e}")
            st.info("Install required dependencies: pip install scipy")
            BACKTESTING_AVAILABLE = False
            return

        if not BACKTESTING_AVAILABLE:
            return

        st.info("Backtesting functionality is available but UI implementation is in progress.")

    def render_technical_analysis_tab(self):
        """Render the technical analysis tab with interactive charts."""
        st.header("📊 Technical Analysis")

        if not PLOTLY_AVAILABLE:
            st.error("Plotly is required for technical analysis charts")
            return

        st.info("Technical analysis functionality is available but UI implementation is in progress.")


# Main entry point
def main():
    """Main application entry point."""

    # Initialize session state
    initialize_session_state()

    # Get cached trading app instance
    trading_app = initialize_trading_app()

    # Update portfolio data periodically (not on every rerun)
    current_time = time.time()
    last_update = st.session_state.get('last_update', 0)

    if current_time - last_update > 60:  # Update every minute
        if trading_app.initialized:
            asyncio.run(trading_app.update_portfolio())
            # Record daily portfolio value for performance tracking
            trading_app.portfolio.record_portfolio_value()
            # Save the updated portfolio state with the new value history
            trading_app.portfolio.save_state()
            st.session_state.last_update = current_time

    # Render main interface
    trading_app.render_main_app()


if __name__ == "__main__":
    main()