# Migration Guide: Streamlit to FastAPI

This document outlines the migration from Streamlit to FastAPI with modern web UI.

## Overview

The migration replaces:
- **Streamlit** → **FastAPI** with REST API and Swagger docs
- **Streamlit UI** → **HTML5 + Tailwind CSS + Bootstrap Icons**
- **Direct function calls** → **REST API endpoints**

## New Architecture

### Backend (FastAPI)
- **Main App**: `fastapi_app/main.py`
- **API Routes**: `fastapi_app/routers/`
- **Data Models**: `fastapi_app/models/`
- **Static Files**: `fastapi_app/static/`

### Frontend (HTML5/Tailwind/Bootstrap)
- **Main UI**: `fastapi_app/static/index.html`
- **JavaScript**: `fastapi_app/static/js/main.js`
- **Responsive Design**: Mobile-first with Tailwind CSS
- **Icons**: Bootstrap Icons

## API Endpoints

### Portfolio Management
- `GET /api/portfolio/` - Portfolio summary
- `GET /api/portfolio/positions` - Current positions
- `GET /api/portfolio/positions/{symbol}` - Specific position
- `POST /api/portfolio/rebalance` - Rebalance portfolio

### Trading Operations
- `POST /api/trading/orders` - Place order
- `GET /api/trading/orders` - List orders
- `GET /api/trading/orders/{order_id}` - Get order details
- `DELETE /api/trading/orders/{order_id}` - Cancel order

### Strategy Management
- `GET /api/strategies/` - List strategies
- `POST /api/strategies/` - Create strategy
- `GET /api/strategies/{name}` - Get strategy details
- `PUT /api/strategies/{name}` - Update strategy
- `DELETE /api/strategies/{name}` - Delete strategy
- `POST /api/strategies/{name}/start` - Start strategy
- `POST /api/strategies/{name}/stop` - Stop strategy

### Backtesting
- `POST /api/backtesting/run` - Start backtest
- `GET /api/backtesting/status/{id}` - Get backtest status
- `GET /api/backtesting/results/{id}` - Get results
- `GET /api/backtesting/list` - List backtests

### Monitoring
- `GET /api/monitoring/health` - System health
- `GET /api/monitoring/metrics` - Performance metrics
- `GET /api/monitoring/logs` - System logs
- `GET /api/monitoring/alerts` - System alerts

### Market Data
- `GET /api/data/quote/{symbol}` - Current quote
- `GET /api/data/quotes?symbols=AAPL,GOOGL` - Multiple quotes
- `GET /api/data/history/{symbol}` - Historical data
- `WebSocket /api/data/live/{symbol}` - Real-time data

## Installation

1. **Install FastAPI dependencies**:
```bash
pip install -r requirements_fastapi.txt
```

2. **Run the FastAPI application**:
```bash
python run_fastapi.py
```

3. **Access the application**:
- Main UI: http://localhost:8000
- API Documentation: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

## Key Features

### Modern UI
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Dark Mode**: Toggle between light and dark themes
- **Real-time Updates**: Live data updates without page refresh
- **Interactive Charts**: Chart.js for portfolio and performance visualization

### API Features
- **Automatic Documentation**: Swagger UI and ReDoc
- **Data Validation**: Pydantic models for request/response validation
- **WebSocket Support**: Real-time market data streaming
- **CORS Enabled**: Cross-origin requests for frontend integration

### Production Ready
- **Error Handling**: Comprehensive error responses
- **Background Tasks**: Async backtesting and long-running operations
- **Monitoring**: Health checks and system metrics
- **Logging**: Structured logging with log levels

## Streamlit vs FastAPI Comparison

| Feature | Streamlit | FastAPI |
|---------|-----------|---------|
| UI Framework | Python-based widgets | HTML5/CSS/JavaScript |
| API | Limited | Full REST API |
| Documentation | Manual | Automatic (Swagger) |
| Mobile Support | Limited | Responsive |
| Customization | Limited | Full control |
| Performance | Good | Excellent |
| Scalability | Limited | High |
| Real-time | Polling | WebSockets |

## Migration Benefits

1. **No Streamlit Limits**: Remove Streamlit's deployment and customization restrictions
2. **Modern UI**: Professional, responsive interface
3. **API First**: RESTful API for integration with other systems
4. **Better Performance**: Async operations and efficient data handling
5. **Mobile Support**: Responsive design works on all devices
6. **Professional Documentation**: Auto-generated API docs
7. **Real-time Features**: WebSocket support for live data
8. **Deployment Flexibility**: Deploy anywhere (Docker, cloud, etc.)

## Backward Compatibility

The core trading framework (`core/`, `strategy_layer/`, `execution_layer/`) remains unchanged. Only the presentation layer has been replaced.

## Testing

The FastAPI application includes:
- Health check endpoint
- API endpoint testing
- Error handling validation
- Performance monitoring

Run tests with:
```bash
pytest fastapi_app/tests/
```

## Deployment

The FastAPI application can be deployed using:
- **Docker**: Containerized deployment
- **Cloud Platforms**: Heroku, AWS, GCP, Azure
- **Traditional Servers**: Linux/Windows servers
- **Kubernetes**: Scalable container orchestration

## Support

For issues or questions about the migration:
1. Check the API documentation at `/api/docs`
2. Review the JavaScript console for frontend errors
3. Check server logs for backend issues
4. Refer to FastAPI documentation: https://fastapi.tiangolo.com/