# 🚀 Advanced Trading Framework

A production-quality, object-oriented trading framework with machine learning capabilities, real-time portfolio management, and a comprehensive Streamlit UI.

## � Recent Improvements

### v1.1.0 - Production Readiness Enhancements
- **🛡️ Division by Zero Protection**: Added robust checks in portfolio calculations to prevent division by zero errors
- **💰 Financial Arithmetic**: Implemented proper Decimal arithmetic for all financial calculations to eliminate floating-point precision issues
- **🔒 Thread Safety**: Added comprehensive thread locks for ML strategy cache access to prevent race conditions
- **📊 Stop Loss Logic**: Enhanced position management with proper stop-loss and take-profit calculations
- **🧹 Resource Cleanup**: Implemented proper cleanup for async tasks and thread pools to prevent resource leaks

## �🏗️ Architecture

The framework follows a layered architecture with clear separation of concerns:

### 📁 Project Structure

```
advanced-trading-framework/
├── core/                          # Core data structures
│   ├── __init__.py
│   ├── position.py               # Position dataclass with serialization
│   └── portfolio.py              # Portfolio management and analytics
├── execution_layer/              # Broker abstraction layer
│   ├── __init__.py
│   ├── base_broker.py           # Abstract broker interface
│   ├── alpaca_broker.py         # Alpaca API implementation
│   └── paper_broker.py          # Paper trading simulation
├── strategy_layer/              # Strategy abstraction layer
│   ├── __init__.py
│   ├── signals.py               # TradingSignal dataclass
│   ├── strategy_base.py         # Strategy ABC
│   ├── base_strategy.py         # Legacy strategy (being refactored)
│   └── ml_random_forest_strategy.py  # ML strategy implementation
├── app.py                       # Main Streamlit application
├── config.yaml                  # Configuration file
├── requirements.txt             # Dependencies
├── run_app.py                   # Launch script
└── README.md                    # This file
```

## 🎯 Key Features

### 💼 Portfolio Management
- **Real-time portfolio tracking** with P&L calculations
- **Position management** with automatic updates
- **Risk management** with configurable limits
- **Performance analytics** with metrics and reporting
- **JSON serialization** for state persistence

### 🤖 Machine Learning Strategy
- **Random Forest classifier** for signal generation
- **Technical indicators** as features (SMA, RSI, MACD, Bollinger Bands)
- **Automated model training** and persistence
- **Confidence-based signal filtering**
- **Feature engineering** with 13+ technical indicators

### 🏦 Multi-Broker Support
- **Paper trading** for testing and simulation
- **Alpaca API** integration for live trading
- **Extensible broker interface** for adding new brokers
- **Unified API** across all broker types

### 🎨 Streamlit Web Interface
- **Real-time dashboard** with live updates
- **Portfolio overview** with positions and P&L
- **Signal monitoring** with history and statistics
- **Performance visualization** with charts
- **Risk management controls** with configurable parameters
- **System logs** with downloadable history

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd advanced-trading-framework

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

The `config.yaml` file contains all configurable parameters:

- **Broker settings** (Paper trading vs Alpaca)
- **Strategy parameters** (confidence thresholds, indicators)
- **Risk management** (position limits, stop losses)
- **UI settings** (refresh rates, display options)

### 3. Environment Variables (Optional)

For Alpaca live trading, set these environment variables:

```bash
export ALPACA_API_KEY="your_alpaca_api_key"
export ALPACA_SECRET_KEY="your_alpaca_secret_key"
```

### 4. Launch Application

```bash
# Using the launch script
python run_app.py

# Or directly with Streamlit
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 📊 Usage Guide

### Portfolio Tab
- View current positions and their P&L
- Monitor total portfolio value and cash
- Track daily and total returns

### Signals Tab
- Review recent trading signals
- Monitor signal statistics (buy/sell ratio, confidence)
- Analyze signal reasoning and metadata

### Performance Tab
- View performance metrics (Sharpe ratio, win rate)
- Analyze portfolio value over time
- Compare against benchmarks

### Settings Tab
- View current configuration
- Modify strategy parameters
- Adjust risk management settings

## 🔧 Configuration

### Strategy Configuration

```yaml
strategy:
  active_strategies:
    - "MLRandomForestStrategy"
  
  ml_random_forest:
    confidence_threshold: 0.6
    lookback_period: 60
    min_accuracy: 0.55
    retrain_frequency: 7
```

### Risk Management

```yaml
portfolio:
  max_position_size: 0.1      # Max 10% per position
  max_daily_loss: 0.02        # Max 2% daily loss
  stop_loss_default: 0.05     # 5% stop loss
  take_profit_default: 0.1    # 10% take profit
```

### Data Sources

```yaml
data:
  provider: "yfinance"
  refresh_intervals:
    real_time: 60
    daily: 3600
    model_retrain: 86400
```

## 🛠️ Development

### Adding New Strategies

1. Inherit from `Strategy` base class in `strategy_layer/strategy_base.py`
2. Implement the `generate_signals(tickers)` method
3. Add strategy configuration to `config.yaml`
4. Register strategy in `app.py`

```python
from strategy_layer.strategy_base import Strategy
from strategy_layer.signals import TradingSignal, create_buy_signal

class MyStrategy(Strategy):
    async def generate_signals(self, tickers):
        signals = []
        # Your strategy logic here
        return signals
```

### Adding New Brokers

1. Inherit from `BaseBroker` in `execution_layer/base_broker.py`
2. Implement required methods (`place_order`, `get_positions`, etc.)
3. Add broker configuration to `config.yaml`
4. Register broker in `app.py`

### Testing

```bash
# Run syntax validation
python test_syntax.py

# Test individual components
python -m core.portfolio
python -m strategy_layer.ml_random_forest_strategy
```

## 📈 Machine Learning Strategy Details

### Technical Indicators Used
- **Moving Averages**: SMA 5, 10, 20
- **Momentum**: RSI, MACD, MACD Signal
- **Volatility**: Bollinger Bands, Volatility measure
- **Volume**: Volume ratio, normalized volume
- **Price**: 1-day and 5-day price changes, momentum

### Model Training
- **Algorithm**: Random Forest Classifier
- **Target**: Next day return > 0.5%
- **Features**: 13+ technical indicators
- **Validation**: 80/20 train/test split
- **Retraining**: Configurable frequency (default: weekly)

### Signal Generation
- **Confidence Filtering**: Only signals above threshold
- **Risk Management**: Integrated stop-loss and take-profit with proper financial arithmetic
- **Position Sizing**: Based on portfolio allocation rules with division by zero protection
- **Thread Safety**: Concurrent signal generation with proper locking mechanisms

## 🔒 Security & Risk

### Risk Management Features
- **Position size limits** (default: 10% max per position)
- **Daily loss limits** (default: 2% max daily loss)
- **Stop-loss orders** (default: 5% stop loss) with precise Decimal calculations
- **Portfolio diversification** controls with division by zero protection
- **Real-time risk monitoring** with thread-safe operations

### Security Considerations
- **API key protection** via environment variables
- **Paper trading default** for safety
- **Configuration validation**
- **Error handling and logging**
- **Thread-safe operations** for concurrent ML model access
- **Resource cleanup** for async tasks and thread pools

## 📋 Dependencies

### Core Requirements
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `scikit-learn>=1.3.0` - Machine learning
- `streamlit>=1.28.0` - Web interface
- `yfinance>=0.2.18` - Market data
- `alpaca-trade-api>=3.0.0` - Broker API

### Optional Dependencies
- `plotly>=5.15.0` - Advanced charting
- `scipy>=1.11.0` - Scientific computing
- `statsmodels>=0.14.0` - Statistical modeling

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration Errors**: Validate `config.yaml` syntax
   ```bash
   python -c "import yaml; yaml.safe_load(open('config.yaml'))"
   ```

3. **API Connection Issues**: Check network and API credentials
4. **Data Issues**: Verify ticker symbols and market hours

### Logging

The framework provides comprehensive logging:
- **File logging**: `logs/trading.log`
- **Console logging**: Real-time in terminal
- **UI logging**: Available in the Logs tab

### Debug Mode

Enable debug mode in `config.yaml`:
```yaml
development:
  debug_mode: true
  verbose_logging: true
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code follows the established patterns
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Alpaca Markets** for providing trading API
- **Yahoo Finance** for market data
- **Streamlit** for the web framework
- **scikit-learn** for machine learning capabilities

---

**⚠️ Disclaimer**: This is educational software. Always test thoroughly in paper trading mode before using with real money. Trading involves risk and past performance does not guarantee future results.