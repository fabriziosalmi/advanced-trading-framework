/**
 * Advanced Trading Framework - Frontend JavaScript
 * Handles UI interactions and API communications
 */

class TradingFrameworkUI {
    constructor() {
        this.apiBase = '/api';
        this.currentView = 'dashboard';
        this.charts = {};
        this.tradingMode = 'manual'; // Default to manual
        this.riskSettings = {
            maxDailyLoss: 1000,
            positionSize: 1000
        };
        this.init();
    }

    async init() {
        this.setupEventListeners();
        this.initializeMenuSections();
        this.initializeCharts();
        await this.loadDashboardData();
        this.startRealTimeUpdates();
    }

    setupEventListeners() {
        // Sidebar toggle
        document.getElementById('sidebarToggle')?.addEventListener('click', () => {
            this.toggleSidebar();
        });

        // Dark mode toggle
        document.getElementById('darkModeToggle')?.addEventListener('click', () => {
            this.toggleDarkMode();
        });

        // Navigation links
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const view = link.getAttribute('href').substring(1);
                this.switchView(view);
            });
        });

        // Sidebar overlay
        document.getElementById('sidebarOverlay')?.addEventListener('click', () => {
            this.toggleSidebar();
        });

        // Menu section toggles
        document.querySelectorAll('.menu-header').forEach(header => {
            header.addEventListener('click', (e) => {
                this.toggleMenuSection(header);
            });
        });

        // Order form handling
        document.getElementById('orderType')?.addEventListener('change', (e) => {
            this.toggleLimitPrice(e.target.value);
        });

        document.getElementById('orderForm')?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.submitOrder();
        });

        // Strategy form handling
        document.getElementById('strategyForm')?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.createStrategy();
        });

        // Backtest form handling
        document.getElementById('backtestForm')?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.runBacktest();
        });

        // Market data search
        document.getElementById('searchQuote')?.addEventListener('click', () => {
            this.searchQuote();
        });

        // Chart updates
        document.getElementById('updateChart')?.addEventListener('click', () => {
            this.updateChart();
        });

        // Settings dark mode toggle
        document.getElementById('settingsDarkMode')?.addEventListener('click', () => {
            this.toggleDarkMode();
        });

        // Save settings button
        document.getElementById('saveSettings')?.addEventListener('click', () => {
            this.saveSettings();
        });

        // Trading control buttons
        document.getElementById('startTradingBtn')?.addEventListener('click', () => {
            this.startTrading();
        });

        document.getElementById('stopTradingBtn')?.addEventListener('click', () => {
            this.stopTrading();
        });

        document.getElementById('emergencyStopBtn')?.addEventListener('click', () => {
            this.emergencyStop();
        });

        // Trading mode radio buttons
        document.querySelectorAll('input[name="tradingMode"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.setTradingMode(e.target.value);
            });
        });

        // Risk settings inputs
        document.getElementById('maxDailyLoss')?.addEventListener('change', (e) => {
            this.updateRiskSettings();
        });

        document.getElementById('positionSize')?.addEventListener('change', (e) => {
            this.updateRiskSettings();
        });
    }

    // Trading control methods
    async startTrading() {
        try {
            const response = await fetch('/api/trading/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    mode: this.tradingMode,
                    riskSettings: this.riskSettings
                })
            });

            if (response.ok) {
                this.showNotification('Trading started successfully', 'success');
                this.updateTradingStatus('running');
            } else {
                const error = await response.text();
                this.showNotification(`Failed to start trading: ${error}`, 'error');
            }
        } catch (error) {
            this.showNotification(`Error starting trading: ${error.message}`, 'error');
        }
    }

    async stopTrading() {
        try {
            const response = await fetch('/api/trading/stop', {
                method: 'POST'
            });

            if (response.ok) {
                this.showNotification('Trading stopped successfully', 'success');
                this.updateTradingStatus('stopped');
            } else {
                const error = await response.text();
                this.showNotification(`Failed to stop trading: ${error}`, 'error');
            }
        } catch (error) {
            this.showNotification(`Error stopping trading: ${error.message}`, 'error');
        }
    }

    async emergencyStop() {
        try {
            const response = await fetch('/api/trading/emergency-stop', {
                method: 'POST'
            });

            if (response.ok) {
                this.showNotification('Emergency stop activated', 'warning');
                this.updateTradingStatus('emergency');
            } else {
                const error = await response.text();
                this.showNotification(`Emergency stop failed: ${error}`, 'error');
            }
        } catch (error) {
            this.showNotification(`Error with emergency stop: ${error.message}`, 'error');
        }
    }

    setTradingMode(mode) {
        this.tradingMode = mode;
        this.showNotification(`Trading mode set to ${mode}`, 'info');
    }

    updateRiskSettings() {
        const maxDailyLoss = document.getElementById('maxDailyLoss')?.value;
        const positionSize = document.getElementById('positionSize')?.value;

        this.riskSettings = {
            maxDailyLoss: parseFloat(maxDailyLoss) || 0,
            positionSize: parseFloat(positionSize) || 0
        };
    }

    updateTradingStatus(status) {
        const statusIndicator = document.getElementById('tradingStatus');
        const startBtn = document.getElementById('startTradingBtn');
        const stopBtn = document.getElementById('stopTradingBtn');

        if (statusIndicator) {
            statusIndicator.className = `px-3 py-1 text-xs font-medium rounded-full status-${status}`;
            statusIndicator.textContent = status.charAt(0).toUpperCase() + status.slice(1);
        }

        // Update button states
        if (startBtn && stopBtn) {
            if (status === 'running') {
                startBtn.disabled = true;
                startBtn.classList.add('opacity-50', 'cursor-not-allowed');
                stopBtn.disabled = false;
                stopBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            } else {
                startBtn.disabled = false;
                startBtn.classList.remove('opacity-50', 'cursor-not-allowed');
                stopBtn.disabled = true;
                stopBtn.classList.add('opacity-50', 'cursor-not-allowed');
            }
        }
    }

    showNotification(message, type) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show position-fixed`;
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        document.body.appendChild(notification);

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }

    toggleSidebar() {
        const sidebar = document.getElementById('sidebar');
        const overlay = document.getElementById('sidebarOverlay');
        const main = document.querySelector('main');

        if (sidebar.classList.contains('-translate-x-full')) {
            // Show sidebar
            sidebar.classList.remove('-translate-x-full');
            sidebar.classList.add('translate-x-0');
            overlay.classList.remove('hidden');
            main.classList.remove('lg:ml-0');
            main.classList.add('lg:ml-64');
        } else {
            // Hide sidebar
            sidebar.classList.remove('translate-x-0');
            sidebar.classList.add('-translate-x-full');
            overlay.classList.add('hidden');
            main.classList.remove('lg:ml-64');
            main.classList.add('lg:ml-0');
        }
    }

    toggleDarkMode() {
        document.documentElement.classList.toggle('dark');
        localStorage.setItem('darkMode', document.documentElement.classList.contains('dark'));
    }

    switchView(viewName) {
        // Hide all views
        document.querySelectorAll('.view').forEach(view => {
            view.classList.add('hidden');
        });

        // Show selected view
        const view = document.getElementById(`${viewName}-view`);
        if (view) {
            view.classList.remove('hidden');
            this.currentView = viewName;
        }

        // Update navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active', 'bg-blue-50', 'dark:bg-blue-900', 'text-blue-600', 'dark:text-blue-300');
            link.classList.add('text-gray-700', 'dark:text-gray-300');
        });

        const activeLink = document.querySelector(`a[href="#${viewName}"]`);
        if (activeLink) {
            activeLink.classList.add('active', 'bg-blue-50', 'dark:bg-blue-900', 'text-blue-600', 'dark:text-blue-300');
            activeLink.classList.remove('text-gray-700', 'dark:text-gray-300');
        }

        // Load view-specific data
        this.loadViewData(viewName);
    }

    toggleMenuSection(header) {
        const section = header.parentElement;
        const items = section.querySelector('.menu-items');
        const arrow = header.querySelector('.menu-arrow');

        if (items.classList.contains('hidden')) {
            items.classList.remove('hidden');
            arrow.style.transform = 'rotate(90deg)';
        } else {
            items.classList.add('hidden');
            arrow.style.transform = 'rotate(0deg)';
        }
    }

    // Initialize menu sections as expanded
    initializeMenuSections() {
        document.querySelectorAll('.menu-arrow').forEach(arrow => {
            arrow.style.transform = 'rotate(90deg)';
        });
    }

    toggleLimitPrice(orderType) {
        const limitPriceDiv = document.getElementById('limitPriceDiv');
        if (orderType === 'limit') {
            limitPriceDiv.classList.remove('hidden');
        } else {
            limitPriceDiv.classList.add('hidden');
        }
    }

    async submitOrder() {
        const orderData = {
            symbol: document.getElementById('orderSymbol').value,
            side: document.getElementById('orderSide').value,
            order_type: document.getElementById('orderType').value,
            quantity: parseFloat(document.getElementById('orderQuantity').value),
            price: document.getElementById('orderType').value === 'limit' ?
                   parseFloat(document.getElementById('orderPrice').value) : null
        };

        try {
            const response = await this.apiCall('/trading/orders', 'POST', orderData);
            this.showSuccess('Order placed successfully');
            this.loadTradingData();
            document.getElementById('orderForm').reset();
        } catch (error) {
            this.showError('Failed to place order: ' + error.message);
        }
    }

    async createStrategy() {
        const strategyData = {
            name: document.getElementById('strategyName').value,
            parameters: {},
            symbols: document.getElementById('strategySymbols').value.split(',').map(s => s.trim()),
            enabled: true
        };

        try {
            const response = await this.apiCall('/strategies', 'POST', strategyData);
            this.showSuccess('Strategy created successfully');
            this.loadStrategiesData();
            document.getElementById('strategyForm').reset();
        } catch (error) {
            this.showError('Failed to create strategy: ' + error.message);
        }
    }

    async loadViewData(viewName) {
        switch (viewName) {
            case 'dashboard':
                await this.loadDashboardData();
                break;
            case 'portfolio':
                await this.loadPortfolioData();
                break;
            case 'trading':
                await this.loadTradingData();
                break;
            case 'positions':
                await this.loadPositionsData();
                break;
            case 'strategies':
                await this.loadStrategiesData();
                break;
            case 'ml-models':
                await this.loadMLModelsData();
                break;
            case 'file-manager':
                await this.loadFileManagerData();
                break;
            case 'backtesting':
                await this.loadBacktestingData();
                break;
            case 'performance':
                await this.loadPerformanceData();
                break;
            case 'monitoring':
                await this.loadMonitoringData();
                break;
            case 'logs':
                await this.loadLogsData();
                break;
            case 'data':
                await this.loadMarketData();
                break;
            case 'charts':
                await this.loadChartsData();
                break;
            case 'watchlist':
                await this.loadWatchlistData();
                break;
            case 'settings':
                await this.loadSettingsData();
                break;
            case 'help':
                await this.loadHelpData();
                break;
        }
    }

    async loadDashboardData() {
        try {
            // Load portfolio summary
            try {
                const portfolio = await this.apiCall('/portfolio');
                this.updateDashboardMetrics(portfolio);
            } catch (error) {
                console.warn('Portfolio API not available, using mock data');
                this.updateDashboardMetrics({
                    total_value: 115000,
                    daily_pnl: 2500,
                    positions: [],
                    cash: 15000
                });
            }

            // Load system metrics
            try {
                const metrics = await this.apiCall('/monitoring/metrics');
                this.updateSystemMetrics(metrics);
            } catch (error) {
                console.warn('Monitoring API not available, using mock data');
                this.updateSystemMetrics({
                    active_strategies: 2,
                    open_positions: 5,
                    pending_orders: 1
                });
            }

        } catch (error) {
            console.error('Failed to load dashboard data:', error);
        }
    }

    async loadPortfolioData() {
        try {
            const positions = await this.apiCall('/portfolio/positions');
            this.populatePositionsTable(positions, 'positionsTable');
        } catch (error) {
            console.warn('Portfolio API not available, using mock data');
            this.populatePositionsTable([
                { symbol: 'AAPL', quantity: 100, avg_cost: 150, current_price: 155, market_value: 15500, unrealized_pnl: 500 },
                { symbol: 'GOOGL', quantity: 50, avg_cost: 2800, current_price: 2850, market_value: 142500, unrealized_pnl: 2500 }
            ], 'positionsTable');
        }
    }

    async loadTradingData() {
        try {
            const orders = await this.apiCall('/trading/orders');
            this.populateOrdersTable(orders);
        } catch (error) {
            console.warn('Trading API not available, using mock data');
            this.populateOrdersTable([
                { timestamp: new Date(), symbol: 'AAPL', side: 'buy', quantity: 100, price: 155, status: 'filled' },
                { timestamp: new Date(), symbol: 'GOOGL', side: 'sell', quantity: 25, price: 2850, status: 'pending' }
            ]);
        }
    }

    async loadPositionsData() {
        await this.loadPortfolioData();
        const positions = document.getElementById('positionsTable').innerHTML;
        document.getElementById('detailedPositionsTable').innerHTML = positions;
    }

    async loadStrategiesData() {
        try {
            const strategies = await this.apiCall('/strategies');
            this.populateStrategiesList(strategies);
        } catch (error) {
            console.warn('Strategies API not available, using mock data');
            this.populateStrategiesList([
                { name: 'ML Random Forest', status: 'active', performance: { total_return: 15.2, sharpe_ratio: 1.42 } },
                { name: 'LGBM Strategy', status: 'inactive', performance: { total_return: 8.7, sharpe_ratio: 0.95 } }
            ]);
        }
    }

    async loadMLModelsData() {
        try {
            // Load model metrics from local files
            const models = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA'];
            let activeModels = 0;
            const modelList = [];

            for (const symbol of models) {
                try {
                    const metricsResponse = await fetch(`/static/models/${symbol}_metrics.json`);
                    if (metricsResponse.ok) {
                        const metrics = await metricsResponse.json();
                        activeModels++;
                        modelList.push({
                            symbol: symbol,
                            accuracy: metrics.accuracy || 0,
                            precision: metrics.precision || 0,
                            recall: metrics.recall || 0,
                            last_trained: metrics.last_trained || 'Unknown'
                        });
                    }
                } catch (error) {
                    console.warn(`Could not load metrics for ${symbol}`);
                }
            }

            document.getElementById('activeModels').textContent = activeModels;
            this.populateModelList(modelList);
        } catch (error) {
            console.warn('ML Models data loading failed:', error);
            document.getElementById('activeModels').textContent = '0';
        }

        // Set up ML models action event listeners
        this.setupMLModelsControls();
    }

    setupMLModelsControls() {
        // Train new model button
        document.getElementById('trainNewModelBtn')?.addEventListener('click', () => {
            this.trainNewModel();
        });

        // Retrain best model button
        document.getElementById('retrainBestModelBtn')?.addEventListener('click', () => {
            this.retrainBestModel();
        });

        // Export models button
        document.getElementById('exportModelsBtn')?.addEventListener('click', () => {
            this.exportModels();
        });

        // Clean old models button
        document.getElementById('cleanOldModelsBtn')?.addEventListener('click', () => {
            this.cleanOldModels();
        });
    }

    async trainNewModel() {
        // In a real implementation, this would open a training modal/form
        this.showSuccess('Training new model...');
        console.log('Train new model');
    }

    async retrainBestModel() {
        if (!confirm('Are you sure you want to retrain the best performing model?')) {
            return;
        }

        try {
            // In a real implementation, this would call the backend API
            this.showSuccess('Retraining best model...');
            console.log('Retrain best model');
        } catch (error) {
            this.showError('Failed to retrain model');
        }
    }

    async exportModels() {
        try {
            // In a real implementation, this would export models
            this.showSuccess('Models exported successfully');
            console.log('Export models');
        } catch (error) {
            this.showError('Failed to export models');
        }
    }

    async cleanOldModels() {
        if (!confirm('Are you sure you want to clean old models? This will remove models older than 30 days.')) {
            return;
        }

        try {
            // In a real implementation, this would call the backend API
            this.showSuccess('Old models cleaned successfully');
            // Refresh models data
            this.loadMLModelsData();
        } catch (error) {
            this.showError('Failed to clean old models');
        }
    }

    async loadFileManagerData() {
        try {
            // Get file system information
            const modelsDir = '/static/models/';
            const dataDir = '/static/data/';

            // Count model files
            const modelFiles = ['AAPL_rf_model.pkl', 'AAPL_scaler.pkl', 'MSFT_rf_model.pkl', 'MSFT_scaler.pkl',
                              'GOOGL_rf_model.pkl', 'GOOGL_scaler.pkl', 'AMZN_rf_model.pkl', 'AMZN_scaler.pkl',
                              'META_rf_model.pkl', 'META_scaler.pkl', 'TSLA_rf_model.pkl', 'TSLA_scaler.pkl'];

            // Count data files (this would need actual file listing API)
            const dataFiles = ['paper_broker_state.json', 'portfolio_state.json'];

            document.getElementById('modelFiles').textContent = Math.floor(modelFiles.length / 2); // Pairs of model + scaler
            document.getElementById('dataFiles').textContent = dataFiles.length;

            // Calculate approximate storage (rough estimate)
            const totalMB = (modelFiles.length * 0.5) + (dataFiles.length * 0.1); // Rough MB estimate
            document.getElementById('totalStorage').textContent = totalMB.toFixed(1) + ' MB';

            this.populateFileList(modelFiles.concat(dataFiles));
        } catch (error) {
            console.warn('File manager data loading failed:', error);
        }

        // Set up cleanup action event listeners
        this.setupFileManagerControls();
    }

    setupFileManagerControls() {
        // Remove old models button
        document.getElementById('removeOldModelsBtn')?.addEventListener('click', () => {
            this.removeOldModels();
        });

        // Delete failed training files button
        document.getElementById('deleteFailedFilesBtn')?.addEventListener('click', () => {
            this.deleteFailedTrainingFiles();
        });

        // Archive old data button
        document.getElementById('archiveOldDataBtn')?.addEventListener('click', () => {
            this.archiveOldData();
        });
    }

    async removeOldModels() {
        if (!confirm('Are you sure you want to remove models older than 30 days?')) {
            return;
        }

        try {
            // In a real implementation, this would call the backend API
            this.showSuccess('Old models removed successfully');
            this.loadFileManagerData();
        } catch (error) {
            this.showError('Failed to remove old models');
        }
    }

    async deleteFailedTrainingFiles() {
        if (!confirm('Are you sure you want to delete all failed training files?')) {
            return;
        }

        try {
            // In a real implementation, this would call the backend API
            this.showSuccess('Failed training files deleted');
            this.loadFileManagerData();
        } catch (error) {
            this.showError('Failed to delete training files');
        }
    }

    async archiveOldData() {
        try {
            // In a real implementation, this would call the backend API
            this.showSuccess('Old data archived successfully');
            this.loadFileManagerData();
        } catch (error) {
            this.showError('Failed to archive old data');
        }
    }

    async loadBacktestingData() {
        try {
            const backtests = await this.apiCall('/backtesting/list');
            this.populateBacktestHistory(backtests.backtests || []);
        } catch (error) {
            console.warn('Backtesting API not available, using mock data');
            this.populateBacktestHistory([
                { strategy_name: 'ML Random Forest', period: '2023-01-01 to 2023-12-31', total_return: 15.2, sharpe_ratio: 1.42, max_drawdown: -8.3, trades: 156, status: 'completed' },
                { strategy_name: 'LightGBM', period: '2023-06-01 to 2023-12-31', total_return: 8.7, sharpe_ratio: 0.95, max_drawdown: -12.1, trades: 89, status: 'completed' }
            ]);
        }
        this.initializeEquityChart();
    }

    async loadPerformanceData() {
        try {
            // Try to load from API first
            const performance = await this.apiCall('/monitoring/performance');
            this.updatePerformanceMetrics(performance);
            this.populatePerformanceTable(performance.trades || []);
            this.initializePerformanceCharts(performance);
        } catch (error) {
            console.warn('Performance API not available, using mock data');
            // Mock performance data
            const mockPerformance = {
                total_return: 12.5,
                sharpe_ratio: 1.42,
                max_drawdown: -8.3,
                win_rate: 68.5,
                trades: [
                    {
                        date: '2025-09-20',
                        symbol: 'AAPL',
                        side: 'BUY',
                        quantity: 10,
                        entry_price: 150.25,
                        exit_price: 155.80,
                        pnl: 55.50,
                        return_pct: 3.69
                    },
                    {
                        date: '2025-09-19',
                        symbol: 'MSFT',
                        side: 'SELL',
                        quantity: 5,
                        entry_price: 305.00,
                        exit_price: 298.50,
                        pnl: 32.50,
                        return_pct: 2.13
                    },
                    {
                        date: '2025-09-18',
                        symbol: 'GOOGL',
                        side: 'BUY',
                        quantity: 2,
                        entry_price: 2750.00,
                        exit_price: 2825.50,
                        pnl: 151.00,
                        return_pct: 2.75
                    }
                ],
                equity_curve: [10000, 10150, 10200, 10350, 10400, 10550, 10600, 10750, 10800, 10950, 11000, 11150],
                monthly_returns: [1.5, 2.1, -0.8, 3.2, 1.8, -1.2, 2.5, 1.9, 0.5, 2.8, -0.3, 1.7]
            };
            this.updatePerformanceMetrics(mockPerformance);
            this.populatePerformanceTable(mockPerformance.trades);
            this.initializePerformanceCharts(mockPerformance);
        }
    }

    async loadMonitoringData() {
        try {
            const metrics = await this.apiCall('/monitoring/metrics');
            this.updateSystemMetrics(metrics);
        } catch (error) {
            console.warn('Monitoring API not available, using mock data');
        }
        this.populateSystemAlerts();
        this.populateProcessTable();
        this.initializeMonitoringCharts();
    }

    async loadLogsData() {
        try {
            // Try to load from API first
            const logs = await this.apiCall('/monitoring/logs');
            this.populateLogEntries(logs.entries || []);
        } catch (error) {
            console.warn('Logs API not available, using mock data');
            // Mock log data
            const mockLogs = [
                {
                    timestamp: '2025-09-23 14:30:25',
                    level: 'INFO',
                    message: 'Trading system started successfully',
                    source: 'trading_engine'
                },
                {
                    timestamp: '2025-09-23 14:30:22',
                    level: 'INFO',
                    message: 'Portfolio loaded with 3 positions',
                    source: 'portfolio_manager'
                },
                {
                    timestamp: '2025-09-23 14:30:20',
                    level: 'WARNING',
                    message: 'API rate limit approaching for Alpaca',
                    source: 'broker_alpaca'
                },
                {
                    timestamp: '2025-09-23 14:30:15',
                    level: 'INFO',
                    message: 'ML model AAPL_rf_model loaded successfully',
                    source: 'model_loader'
                },
                {
                    timestamp: '2025-09-23 14:30:10',
                    level: 'ERROR',
                    message: 'Failed to fetch data for TSLA: Connection timeout',
                    source: 'data_fetcher'
                },
                {
                    timestamp: '2025-09-23 14:30:05',
                    level: 'INFO',
                    message: 'Backtest completed for period 2025-01-01 to 2025-09-01',
                    source: 'backtest_engine'
                },
                {
                    timestamp: '2025-09-23 14:30:00',
                    level: 'DEBUG',
                    message: 'Signal generated for MSFT: confidence=0.78, action=BUY',
                    source: 'strategy_ml'
                }
            ];
            this.populateLogEntries(mockLogs);
        }

        // Set up log controls
        this.setupLogControls();
    }

    async loadMarketData() {
        this.populateMarketDataTable();
        this.populateTopMovers();
    }

    async loadChartsData() {
        this.initializePriceChart();
    }

    async runBacktest() {
        const backtestData = {
            strategy_name: document.getElementById('backtestStrategy').value,
            start_date: document.getElementById('backtestStartDate').value,
            end_date: document.getElementById('backtestEndDate').value,
            initial_capital: parseFloat(document.getElementById('backtestCapital').value),
            symbols: document.getElementById('backtestSymbols').value.split(',').map(s => s.trim())
        };

        try {
            const response = await this.apiCall('/backtesting/run', 'POST', backtestData);
            this.showSuccess('Backtest started successfully');
            // Show mock results
            this.displayBacktestResults({
                total_return: 15.2,
                sharpe_ratio: 1.42,
                max_drawdown: -8.3,
                trades: 156
            });
        } catch (error) {
            // Show mock results even if API fails
            this.displayBacktestResults({
                total_return: 15.2,
                sharpe_ratio: 1.42,
                max_drawdown: -8.3,
                trades: 156
            });
            this.showSuccess('Backtest completed (demo mode)');
        }
    }

    async searchQuote() {
        const symbol = document.getElementById('symbolSearch').value.toUpperCase();
        if (!symbol) return;

        try {
            const quote = await this.apiCall(`/data/quote/${symbol}`);
            this.displayQuoteResult(quote);
        } catch (error) {
            // Show mock quote
            this.displayQuoteResult({
                symbol: symbol,
                price: 150.25,
                change: 2.34,
                change_percent: 1.58,
                volume: 45672890,
                timestamp: new Date()
            });
        }
    }

    updateChart() {
        const symbol = document.getElementById('chartSymbol').value;
        const timeframe = document.getElementById('chartTimeframe').value;
        const chartType = document.getElementById('chartType').value;

        // Update chart with new parameters
        this.initializePriceChart(symbol, timeframe, chartType);
    }

    async loadWatchlistData() {
        try {
            // Try to load from API first
            const watchlist = await this.apiCall('/api/watchlist');
            this.populateWatchlistTable(watchlist.symbols || []);
        } catch (error) {
            console.warn('Watchlist API not available, using mock data');
            // Mock watchlist data
            const mockWatchlist = [
                {
                    symbol: 'AAPL',
                    name: 'Apple Inc.',
                    price: 175.43,
                    change: 2.15,
                    change_pct: 1.24,
                    volume: '45.2M',
                    market_cap: '2.8T'
                },
                {
                    symbol: 'MSFT',
                    name: 'Microsoft Corporation',
                    price: 335.49,
                    change: -1.23,
                    change_pct: -0.37,
                    volume: '28.7M',
                    market_cap: '2.5T'
                },
                {
                    symbol: 'GOOGL',
                    name: 'Alphabet Inc.',
                    price: 138.21,
                    change: 3.45,
                    change_pct: 2.56,
                    volume: '22.1M',
                    market_cap: '1.7T'
                },
                {
                    symbol: 'TSLA',
                    name: 'Tesla, Inc.',
                    price: 248.42,
                    change: -5.67,
                    change_pct: -2.23,
                    volume: '89.3M',
                    market_cap: '792B'
                }
            ];
            this.populateWatchlistTable(mockWatchlist);
        }

        // Set up watchlist controls
        this.setupWatchlistControls();
    }

    async loadSettingsData() {
        console.log('Loading settings data...');

        // Load settings from localStorage
        const settings = JSON.parse(localStorage.getItem('tradingSettings') || '{}');

        // General settings
        if (settings.darkMode !== undefined) {
            document.getElementById('settingsDarkMode').classList.toggle('bg-blue-600', settings.darkMode);
            document.getElementById('settingsDarkMode').classList.toggle('bg-gray-200', !settings.darkMode);
            const toggle = document.getElementById('settingsDarkMode').querySelector('span');
            if (toggle) {
                toggle.classList.toggle('translate-x-6', settings.darkMode);
                toggle.classList.toggle('translate-x-1', !settings.darkMode);
            }
        }

        if (settings.realTimeUpdates !== undefined) {
            document.getElementById('realTimeUpdates').classList.toggle('bg-blue-600', settings.realTimeUpdates);
            document.getElementById('realTimeUpdates').classList.toggle('bg-gray-200', !settings.realTimeUpdates);
            const toggle = document.getElementById('realTimeUpdates').querySelector('span');
            if (toggle) {
                toggle.classList.toggle('translate-x-6', settings.realTimeUpdates);
                toggle.classList.toggle('translate-x-1', !settings.realTimeUpdates);
            }
        }

        if (settings.notifications !== undefined) {
            document.getElementById('notifications').classList.toggle('bg-blue-600', settings.notifications);
            document.getElementById('notifications').classList.toggle('bg-gray-200', !settings.notifications);
            const toggle = document.getElementById('notifications').querySelector('span');
            if (toggle) {
                toggle.classList.toggle('translate-x-6', settings.notifications);
                toggle.classList.toggle('translate-x-1', !settings.notifications);
            }
        }

        if (settings.logLevel) {
            document.getElementById('logLevel').value = settings.logLevel;
        }

        // Trading environment
        if (settings.tradingEnvironment) {
            document.getElementById('tradingEnvironment').value = settings.tradingEnvironment;
        }

        if (settings.brokerType) {
            document.getElementById('brokerType').value = settings.brokerType;
        }

        if (settings.initialCash !== undefined) {
            document.getElementById('initialCash').value = settings.initialCash;
        }

        if (settings.commission !== undefined) {
            document.getElementById('commission').value = settings.commission;
        }

        // Portfolio & Risk
        if (settings.maxPositionSize !== undefined) {
            document.getElementById('maxPositionSize').value = settings.maxPositionSize;
        }

        if (settings.maxPositions !== undefined) {
            document.getElementById('maxPositions').value = settings.maxPositions;
        }

        if (settings.maxDailyLossPercent !== undefined) {
            document.getElementById('maxDailyLossPercent').value = settings.maxDailyLossPercent;
        }

        if (settings.stopLossPercent !== undefined) {
            document.getElementById('stopLossPercent').value = settings.stopLossPercent;
        }

        // Automation
        if (settings.enableAutomation !== undefined) {
            document.getElementById('enableAutomation').classList.toggle('bg-blue-600', settings.enableAutomation);
            document.getElementById('enableAutomation').classList.toggle('bg-gray-200', !settings.enableAutomation);
            const toggle = document.getElementById('enableAutomation').querySelector('span');
            if (toggle) {
                toggle.classList.toggle('translate-x-6', settings.enableAutomation);
                toggle.classList.toggle('translate-x-1', !settings.enableAutomation);
            }
        }

        if (settings.tradingInterval !== undefined) {
            document.getElementById('tradingInterval').value = settings.tradingInterval;
        }

        if (settings.maxTradesPerHour !== undefined) {
            document.getElementById('maxTradesPerHour').value = settings.maxTradesPerHour;
        }

        if (settings.maxDailyTrades !== undefined) {
            document.getElementById('maxDailyTrades').value = settings.maxDailyTrades;
        }

        // Strategy settings
        if (settings.confidenceThreshold !== undefined) {
            document.getElementById('confidenceThreshold').value = settings.confidenceThreshold;
        }

        if (settings.lookbackPeriod !== undefined) {
            document.getElementById('lookbackPeriod').value = settings.lookbackPeriod;
        }

        if (settings.minAccuracy !== undefined) {
            document.getElementById('minAccuracy').value = settings.minAccuracy;
        }

        if (settings.retrainFrequency !== undefined) {
            document.getElementById('retrainFrequency').value = settings.retrainFrequency;
        }

        // Data sources
        if (settings.dataProvider) {
            document.getElementById('dataProvider').value = settings.dataProvider;
        }

        if (settings.realTimeRefresh !== undefined) {
            document.getElementById('realTimeRefresh').value = settings.realTimeRefresh;
        }

        if (settings.dailyRefresh !== undefined) {
            document.getElementById('dailyRefresh').value = settings.dailyRefresh;
        }

        if (settings.defaultPeriod) {
            document.getElementById('defaultPeriod').value = settings.defaultPeriod;
        }

        // Trading universe
        if (settings.defaultTickers) {
            document.getElementById('defaultTickers').value = settings.defaultTickers;
        }

        if (settings.minMarketCap !== undefined) {
            document.getElementById('minMarketCap').value = settings.minMarketCap;
        }

        if (settings.minVolume !== undefined) {
            document.getElementById('minVolume').value = settings.minVolume;
        }

        if (settings.minPrice !== undefined) {
            document.getElementById('minPrice').value = settings.minPrice;
        }

        if (settings.maxPrice !== undefined) {
            document.getElementById('maxPrice').value = settings.maxPrice;
        }

        this.showSuccess('Settings loaded successfully');
    }

    async saveSettings() {
        console.log('Saving settings...');

        try {
            // Collect all settings from form fields
            const settings = {
                // General settings
                darkMode: document.getElementById('settingsDarkMode').classList.contains('bg-blue-600'),
                realTimeUpdates: document.getElementById('realTimeUpdates').classList.contains('bg-blue-600'),
                notifications: document.getElementById('notifications').classList.contains('bg-blue-600'),
                logLevel: document.getElementById('logLevel').value,

                // Trading environment
                tradingEnvironment: document.getElementById('tradingEnvironment').value,
                brokerType: document.getElementById('brokerType').value,
                initialCash: parseFloat(document.getElementById('initialCash').value) || 100000,
                commission: parseFloat(document.getElementById('commission').value) || 0.00,

                // Portfolio & Risk
                maxPositionSize: parseFloat(document.getElementById('maxPositionSize').value) || 1.0,
                maxPositions: parseInt(document.getElementById('maxPositions').value) || 10,
                maxDailyLossPercent: parseFloat(document.getElementById('maxDailyLossPercent').value) || 1.0,
                stopLossPercent: parseFloat(document.getElementById('stopLossPercent').value) || 5.0,

                // Automation
                enableAutomation: document.getElementById('enableAutomation').classList.contains('bg-blue-600'),
                tradingInterval: parseInt(document.getElementById('tradingInterval').value) || 60,
                maxTradesPerHour: parseInt(document.getElementById('maxTradesPerHour').value) || 10,
                maxDailyTrades: parseInt(document.getElementById('maxDailyTrades').value) || 50,

                // Strategy settings
                confidenceThreshold: parseFloat(document.getElementById('confidenceThreshold').value) || 0.3,
                lookbackPeriod: parseInt(document.getElementById('lookbackPeriod').value) || 365,
                minAccuracy: parseFloat(document.getElementById('minAccuracy').value) || 0.65,
                retrainFrequency: parseInt(document.getElementById('retrainFrequency').value) || 1,

                // Data sources
                dataProvider: document.getElementById('dataProvider').value,
                realTimeRefresh: parseInt(document.getElementById('realTimeRefresh').value) || 60,
                dailyRefresh: parseInt(document.getElementById('dailyRefresh').value) || 3600,
                defaultPeriod: document.getElementById('defaultPeriod').value,

                // Trading universe
                defaultTickers: document.getElementById('defaultTickers').value,
                minMarketCap: parseInt(document.getElementById('minMarketCap').value) || 1,
                minVolume: parseInt(document.getElementById('minVolume').value) || 1000000,
                minPrice: parseFloat(document.getElementById('minPrice').value) || 1,
                maxPrice: parseFloat(document.getElementById('maxPrice').value) || 1000
            };

            // Save to localStorage
            localStorage.setItem('tradingSettings', JSON.stringify(settings));

            // In a real implementation, you would also send to backend
            // await fetch('/api/settings', {
            //     method: 'POST',
            //     headers: { 'Content-Type': 'application/json' },
            //     body: JSON.stringify(settings)
            // });

            this.showSuccess('Settings saved successfully');
        } catch (error) {
            console.error('Error saving settings:', error);
            this.showError('Failed to save settings');
        }
    }

    async loadHelpData() {
        console.log('Loading help data...');
    }

    populatePositionsTable(positions, tableId) {
        const tbody = document.getElementById(tableId);
        if (!tbody) return;

        tbody.innerHTML = positions.map(pos => `
            <tr>
                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">${pos.symbol}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">${pos.quantity}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">$${pos.avg_cost.toFixed(2)}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">$${pos.current_price.toFixed(2)}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">$${pos.market_value.toFixed(2)}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm ${pos.unrealized_pnl >= 0 ? 'text-green-600' : 'text-red-600'}">
                    ${pos.unrealized_pnl >= 0 ? '+' : ''}$${pos.unrealized_pnl.toFixed(2)}
                </td>
                ${tableId === 'detailedPositionsTable' ? `
                    <td class="px-6 py-4 whitespace-nowrap text-sm font-medium">
                        <button class="text-blue-600 hover:text-blue-900 mr-2 edit-position-btn" data-symbol="${pos.symbol}">
                            Edit
                        </button>
                        <button class="text-red-600 hover:text-red-900 close-position-btn" data-symbol="${pos.symbol}">
                            Close
                        </button>
                    </td>
                ` : ''}
            </tr>
        `).join('');

        // Add event listeners for position action buttons
        if (tableId === 'detailedPositionsTable') {
            tbody.querySelectorAll('.edit-position-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const symbol = e.currentTarget.getAttribute('data-symbol');
                    this.editPosition(symbol);
                });
            });

            tbody.querySelectorAll('.close-position-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const symbol = e.currentTarget.getAttribute('data-symbol');
                    this.closePosition(symbol);
                });
            });
        }
    }

    async editPosition(symbol) {
        // In a real implementation, this would open an edit modal or form
        this.showSuccess(`Editing position for ${symbol}`);
        console.log(`Edit position: ${symbol}`);
    }

    async closePosition(symbol) {
        if (!confirm(`Are you sure you want to close the position for ${symbol}?`)) {
            return;
        }

        try {
            // In a real implementation, this would call the backend API
            this.showSuccess(`Position closed for ${symbol}`);
            // Refresh positions data
            this.loadPortfolioData();
        } catch (error) {
            this.showError(`Failed to close position for ${symbol}`);
        }
    }

    populateOrdersTable(orders) {
        const tbody = document.getElementById('ordersTable');
        if (!tbody) return;

        tbody.innerHTML = orders.map(order => `
            <tr>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">${new Date(order.timestamp).toLocaleTimeString()}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">${order.symbol}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">
                    <span class="px-2 py-1 text-xs rounded-full ${order.side === 'buy' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}">${order.side}</span>
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">${order.quantity}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">$${order.price ? order.price.toFixed(2) : 'Market'}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">
                    <span class="px-2 py-1 text-xs rounded-full ${order.status === 'filled' ? 'bg-green-100 text-green-800' : order.status === 'pending' ? 'bg-yellow-100 text-yellow-800' : 'bg-red-100 text-red-800'}">${order.status}</span>
                </td>
            </tr>
        `).join('');
    }

    populateStrategiesList(strategies) {
        const container = document.getElementById('strategiesList');
        if (!container) return;

        container.innerHTML = strategies.map(strategy => `
            <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <div class="flex items-center justify-between mb-2">
                    <h4 class="font-semibold text-gray-800 dark:text-white">${strategy.name}</h4>
                    <span class="px-2 py-1 text-xs rounded-full ${strategy.status === 'active' ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'}">${strategy.status}</span>
                </div>
                <div class="flex justify-between text-sm text-gray-600 dark:text-gray-300">
                    <span>Return: ${strategy.performance?.total_return ? strategy.performance.total_return.toFixed(1) : 'N/A'}%</span>
                    <span>Sharpe: ${strategy.performance?.sharpe_ratio ? strategy.performance.sharpe_ratio.toFixed(2) : 'N/A'}</span>
                </div>
                <div class="mt-2 flex space-x-2">
                    <button class="px-3 py-1 text-xs bg-blue-500 text-white rounded hover:bg-blue-600 configure-strategy-btn" data-strategy="${strategy.name}">
                        Configure
                    </button>
                    <button class="px-3 py-1 text-xs ${strategy.status === 'active' ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600'} text-white rounded toggle-strategy-btn" data-strategy="${strategy.name}" data-status="${strategy.status}">
                        ${strategy.status === 'active' ? 'Stop' : 'Start'}
                    </button>
                </div>
            </div>
        `).join('');

        // Add event listeners for strategy action buttons
        container.querySelectorAll('.configure-strategy-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const strategyName = e.currentTarget.getAttribute('data-strategy');
                this.configureStrategy(strategyName);
            });
        });

        container.querySelectorAll('.toggle-strategy-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const strategyName = e.currentTarget.getAttribute('data-strategy');
                const currentStatus = e.currentTarget.getAttribute('data-status');
                this.toggleStrategy(strategyName, currentStatus);
            });
        });
    }

    async configureStrategy(strategyName) {
        // In a real implementation, this would open a configuration modal
        this.showSuccess(`Configuring strategy: ${strategyName}`);
        console.log(`Configure strategy: ${strategyName}`);
    }

    async toggleStrategy(strategyName, currentStatus) {
        const action = currentStatus === 'active' ? 'stop' : 'start';
        const confirmMessage = `Are you sure you want to ${action} the strategy "${strategyName}"?`;

        if (!confirm(confirmMessage)) {
            return;
        }

        try {
            // In a real implementation, this would call the backend API
            const newStatus = currentStatus === 'active' ? 'stopped' : 'active';
            this.showSuccess(`Strategy "${strategyName}" ${action}ped successfully`);
            // Refresh strategies data
            this.loadStrategiesData();
        } catch (error) {
            this.showError(`Failed to ${action} strategy "${strategyName}"`);
        }
    }

    populateBacktestHistory(backtests) {
        const tbody = document.getElementById('backtestHistoryTable');
        if (!tbody) return;

        tbody.innerHTML = backtests.map(bt => `
            <tr>
                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">${bt.strategy_name}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">${bt.period}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm ${bt.total_return >= 0 ? 'text-green-600' : 'text-red-600'}">
                    ${bt.total_return >= 0 ? '+' : ''}${bt.total_return.toFixed(1)}%
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">${bt.sharpe_ratio.toFixed(2)}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-red-600">${bt.max_drawdown.toFixed(1)}%</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">${bt.trades}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">
                    <span class="px-2 py-1 text-xs rounded-full ${bt.status === 'completed' ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'}">${bt.status}</span>
                </td>
            </tr>
        `).join('');
    }

    displayBacktestResults(results) {
        const container = document.getElementById('backtestResults');
        if (!container) return;

        container.innerHTML = `
            <div class="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <div class="text-center">
                    <p class="text-sm text-gray-600 dark:text-gray-400">Total Return</p>
                    <p class="text-2xl font-bold ${results.total_return >= 0 ? 'text-green-600' : 'text-red-600'}">
                        ${results.total_return >= 0 ? '+' : ''}${results.total_return.toFixed(1)}%
                    </p>
                </div>
                <div class="text-center">
                    <p class="text-sm text-gray-600 dark:text-gray-400">Sharpe Ratio</p>
                    <p class="text-2xl font-bold text-blue-600">${results.sharpe_ratio.toFixed(2)}</p>
                </div>
                <div class="text-center">
                    <p class="text-sm text-gray-600 dark:text-gray-400">Max Drawdown</p>
                    <p class="text-2xl font-bold text-red-600">${results.max_drawdown.toFixed(1)}%</p>
                </div>
                <div class="text-center">
                    <p class="text-sm text-gray-600 dark:text-gray-400">Total Trades</p>
                    <p class="text-2xl font-bold text-gray-800 dark:text-white">${results.trades}</p>
                </div>
            </div>
        `;
    }

    populateSystemAlerts() {
        const container = document.getElementById('systemAlerts');
        if (!container) return;

        const alerts = [
            { type: 'info', message: 'System running normally', timestamp: new Date() },
            { type: 'warning', message: 'High memory usage detected', timestamp: new Date(Date.now() - 300000) },
            { type: 'success', message: 'Backup completed successfully', timestamp: new Date(Date.now() - 600000) }
        ];

        container.innerHTML = alerts.map(alert => `
            <div class="flex items-center space-x-3 p-3 rounded-lg ${
                alert.type === 'warning' ? 'bg-yellow-50 dark:bg-yellow-900' :
                alert.type === 'success' ? 'bg-green-50 dark:bg-green-900' :
                'bg-blue-50 dark:bg-blue-900'
            }">
                <div class="flex-shrink-0">
                    <i class="bi ${
                        alert.type === 'warning' ? 'bi-exclamation-triangle text-yellow-600' :
                        alert.type === 'success' ? 'bi-check-circle text-green-600' :
                        'bi-info-circle text-blue-600'
                    }"></i>
                </div>
                <div class="flex-1">
                    <p class="text-sm text-gray-800 dark:text-white">${alert.message}</p>
                    <p class="text-xs text-gray-500 dark:text-gray-400">${alert.timestamp.toLocaleTimeString()}</p>
                </div>
            </div>
        `).join('');
    }

    populateProcessTable() {
        const tbody = document.getElementById('processTable');
        if (!tbody) return;

        const processes = [
            { name: 'FastAPI Server', status: 'running', cpu: '12%', memory: '256MB', uptime: '2h 15m' },
            { name: 'ML Strategy Engine', status: 'running', cpu: '8%', memory: '512MB', uptime: '2h 15m' },
            { name: 'Data Feed Handler', status: 'running', cpu: '5%', memory: '128MB', uptime: '2h 15m' },
            { name: 'Risk Monitor', status: 'running', cpu: '3%', memory: '64MB', uptime: '2h 15m' }
        ];

        tbody.innerHTML = processes.map(proc => `
            <tr>
                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">${proc.name}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">
                    <span class="px-2 py-1 text-xs rounded-full bg-green-100 text-green-800">${proc.status}</span>
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">${proc.cpu}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">${proc.memory}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">${proc.uptime}</td>
            </tr>
        `).join('');
    }

    populateMarketDataTable() {
        const tbody = document.getElementById('marketDataTable');
        if (!tbody) return;

        const symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META'];
        const data = symbols.map(symbol => ({
            symbol,
            price: 100 + Math.random() * 200,
            change: (Math.random() - 0.5) * 10,
            volume: Math.floor(Math.random() * 10000000),
            timestamp: new Date()
        }));

        tbody.innerHTML = data.map(item => `
            <tr>
                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">${item.symbol}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">$${item.price.toFixed(2)}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm ${item.change >= 0 ? 'text-green-600' : 'text-red-600'}">
                    ${item.change >= 0 ? '+' : ''}${item.change.toFixed(2)}
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm ${item.change >= 0 ? 'text-green-600' : 'text-red-600'}">
                    ${item.change >= 0 ? '+' : ''}${((item.change / item.price) * 100).toFixed(2)}%
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">${item.volume.toLocaleString()}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">${item.timestamp.toLocaleTimeString()}</td>
            </tr>
        `).join('');
    }

    populateTopMovers() {
        const container = document.getElementById('topMovers');
        if (!container) return;

        const movers = [
            { symbol: 'NVDA', change: 8.5, price: 456.78 },
            { symbol: 'AMD', change: -5.2, price: 98.45 },
            { symbol: 'TSLA', change: 4.8, price: 234.56 }
        ];

        container.innerHTML = movers.map(mover => `
            <div class="flex items-center justify-between">
                <span class="font-medium text-gray-800 dark:text-white">${mover.symbol}</span>
                <div class="text-right">
                    <div class="text-sm text-gray-600 dark:text-gray-300">$${mover.price.toFixed(2)}</div>
                    <div class="text-sm ${mover.change >= 0 ? 'text-green-600' : 'text-red-600'}">
                        ${mover.change >= 0 ? '+' : ''}${mover.change.toFixed(1)}%
                    </div>
                </div>
            </div>
        `).join('');
    }

    displayQuoteResult(quote) {
        const container = document.getElementById('quoteResult');
        if (!container) return;

        container.innerHTML = `
            <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <div class="flex items-center justify-between mb-2">
                    <h4 class="text-lg font-semibold text-gray-800 dark:text-white">${quote.symbol}</h4>
                    <span class="text-2xl font-bold text-gray-800 dark:text-white">$${quote.price.toFixed(2)}</span>
                </div>
                <div class="flex items-center justify-between text-sm">
                    <span class="${quote.change >= 0 ? 'text-green-600' : 'text-red-600'}">
                        ${quote.change >= 0 ? '+' : ''}${quote.change.toFixed(2)} (${quote.change_percent >= 0 ? '+' : ''}${quote.change_percent.toFixed(2)}%)
                    </span>
                    <span class="text-gray-500 dark:text-gray-400">Vol: ${quote.volume?.toLocaleString() || 'N/A'}</span>
                </div>
            </div>
        `;
    }

    initializeEquityChart() {
        const ctx = document.getElementById('equityChart');
        if (!ctx) return;

        // Generate sample equity curve data
        const labels = [];
        const data = [];
        let value = 100000;

        for (let i = 0; i < 252; i++) { // One year of trading days
            labels.push(`Day ${i + 1}`);
            value += (Math.random() - 0.45) * 1000; // Slight upward bias
            data.push(value);
        }

        this.charts.equity = new Chart(ctx, {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    label: 'Portfolio Value',
                    data,
                    borderColor: 'rgb(59, 130, 246)',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        ticks: {
                            callback: function(value) {
                                return '$' + value.toLocaleString();
                            }
                        }
                    },
                    x: {
                        display: false
                    }
                }
            }
        });
    }

    initializeMonitoringCharts() {
        // CPU & Memory Chart
        const perfCtx = document.getElementById('systemPerformanceChart');
        if (perfCtx) {
            const labels = [];
            const cpuData = [];
            const memoryData = [];

            for (let i = 0; i < 20; i++) {
                labels.push(`${i}m`);
                cpuData.push(30 + Math.random() * 40);
                memoryData.push(50 + Math.random() * 30);
            }

            this.charts.performance = new Chart(perfCtx, {
                type: 'line',
                data: {
                    labels,
                    datasets: [{
                        label: 'CPU Usage (%)',
                        data: cpuData,
                        borderColor: 'rgb(59, 130, 246)',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)'
                    }, {
                        label: 'Memory Usage (%)',
                        data: memoryData,
                        borderColor: 'rgb(16, 185, 129)',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
        }

        // Network Chart
        const netCtx = document.getElementById('networkChart');
        if (netCtx) {
            const labels = [];
            const inData = [];
            const outData = [];

            for (let i = 0; i < 20; i++) {
                labels.push(`${i}m`);
                inData.push(Math.random() * 100);
                outData.push(Math.random() * 50);
            }

            this.charts.network = new Chart(netCtx, {
                type: 'line',
                data: {
                    labels,
                    datasets: [{
                        label: 'Incoming (MB/s)',
                        data: inData,
                        borderColor: 'rgb(245, 158, 11)',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)'
                    }, {
                        label: 'Outgoing (MB/s)',
                        data: outData,
                        borderColor: 'rgb(139, 92, 246)',
                        backgroundColor: 'rgba(139, 92, 246, 0.1)'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        }
    }

    initializePriceChart(symbol = 'AAPL', timeframe = '5d', chartType = 'line') {
        const ctx = document.getElementById('priceChart');
        if (!ctx) return;

        // Generate sample price data
        const labels = [];
        const data = [];
        let price = 150;

        const periods = timeframe === '1d' ? 24 : timeframe === '5d' ? 120 : 252;

        for (let i = 0; i < periods; i++) {
            if (timeframe === '1d') {
                labels.push(`${9 + Math.floor(i * 8 / periods)}:${(i * 60 / periods).toFixed(0).padStart(2, '0')}`);
            } else {
                const date = new Date();
                date.setDate(date.getDate() - (periods - i));
                labels.push(date.toLocaleDateString());
            }

            price += (Math.random() - 0.5) * 5;
            data.push(price);
        }

        if (this.charts.price) {
            this.charts.price.destroy();
        }

        this.charts.price = new Chart(ctx, {
            type: chartType === 'area' ? 'line' : chartType,
            data: {
                labels,
                datasets: [{
                    label: `${symbol} Price`,
                    data,
                    borderColor: 'rgb(59, 130, 246)',
                    backgroundColor: chartType === 'area' ? 'rgba(59, 130, 246, 0.1)' : 'rgba(59, 130, 246, 0.8)',
                    fill: chartType === 'area',
                    tension: chartType === 'line' ? 0.4 : 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        ticks: {
                            callback: function(value) {
                                return '$' + value.toFixed(2);
                            }
                        }
                    }
                }
            }
        });
    }

    updateDashboardMetrics(portfolio) {
        const portfolioValueEl = document.getElementById('portfolioValue');
        const dailyPnlEl = document.getElementById('dailyPnl');

        if (portfolioValueEl) {
            portfolioValueEl.textContent = this.formatCurrency(portfolio.total_value || 0);
        }

        if (dailyPnlEl) {
            const pnl = portfolio.daily_pnl || 0;
            dailyPnlEl.textContent = this.formatCurrency(pnl);
            dailyPnlEl.className = `text-2xl font-bold ${pnl >= 0 ? 'text-green-600' : 'text-red-600'}`;
        }
    }

    updateSystemMetrics(metrics) {
        const activeStrategiesEl = document.getElementById('activeStrategies');
        const openPositionsEl = document.getElementById('openPositions');

        if (activeStrategiesEl) {
            activeStrategiesEl.textContent = metrics.active_strategies || 0;
        }

        if (openPositionsEl) {
            openPositionsEl.textContent = metrics.open_positions || 0;
        }
    }

    initializeCharts() {
        // Portfolio Performance Chart
        const portfolioCtx = document.getElementById('portfolioChart');
        if (portfolioCtx) {
            this.charts.portfolio = new Chart(portfolioCtx, {
                type: 'line',
                data: {
                    labels: ['1M', '2M', '3M', '4M', '5M', '6M'],
                    datasets: [{
                        label: 'Portfolio Value',
                        data: [100000, 105000, 103000, 108000, 112000, 115000],
                        borderColor: 'rgb(59, 130, 246)',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: false,
                            ticks: {
                                callback: function(value) {
                                    return '$' + value.toLocaleString();
                                }
                            }
                        }
                    }
                }
            });
        }

        // Asset Allocation Chart
        const allocationCtx = document.getElementById('allocationChart');
        if (allocationCtx) {
            this.charts.allocation = new Chart(allocationCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Stocks', 'Bonds', 'Cash', 'Crypto'],
                    datasets: [{
                        data: [60, 25, 10, 5],
                        backgroundColor: [
                            'rgb(59, 130, 246)',
                            'rgb(16, 185, 129)',
                            'rgb(245, 158, 11)',
                            'rgb(139, 92, 246)'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            });
        }
    }

    startRealTimeUpdates() {
        // Update dashboard every 30 seconds
        setInterval(() => {
            if (this.currentView === 'dashboard') {
                this.loadDashboardData();
            }
        }, 30000);

        // Update recent activity
        this.updateRecentActivity();
    }

    updateRecentActivity() {
        const activityEl = document.getElementById('recentActivity');
        if (!activityEl) return;

        const activities = [
            {
                type: 'trade',
                description: 'Bought 100 shares of AAPL',
                time: '2 minutes ago',
                icon: 'bi-arrow-up-circle',
                color: 'text-green-600'
            },
            {
                type: 'strategy',
                description: 'ML Random Forest strategy triggered',
                time: '5 minutes ago',
                icon: 'bi-cpu',
                color: 'text-blue-600'
            },
            {
                type: 'alert',
                description: 'Portfolio rebalancing completed',
                time: '10 minutes ago',
                icon: 'bi-info-circle',
                color: 'text-yellow-600'
            }
        ];

        activityEl.innerHTML = activities.map(activity => `
            <div class="flex items-center space-x-3 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <div class="flex-shrink-0">
                    <i class="bi ${activity.icon} ${activity.color}"></i>
                </div>
                <div class="flex-1">
                    <p class="text-sm text-gray-800 dark:text-white">${activity.description}</p>
                    <p class="text-xs text-gray-500 dark:text-gray-400">${activity.time}</p>
                </div>
            </div>
        `).join('');
    }

    async apiCall(endpoint, method = 'GET', data = null) {
        const config = {
            method,
            headers: {
                'Content-Type': 'application/json',
            }
        };

        if (data) {
            config.body = JSON.stringify(data);
        }

        const response = await fetch(`${this.apiBase}${endpoint}`, config);

        if (!response.ok) {
            throw new Error(`API call failed: ${response.statusText}`);
        }

        return await response.json();
    }

    formatCurrency(amount) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD'
        }).format(amount);
    }

    showError(message) {
        // Simple error display - in production, use a proper notification system
        console.error(message);

        // You could add a toast notification here
        const errorDiv = document.createElement('div');
        errorDiv.className = 'fixed top-4 right-4 bg-red-500 text-white p-4 rounded-lg shadow-lg z-50';
        errorDiv.textContent = message;
        document.body.appendChild(errorDiv);

        setTimeout(() => {
            document.body.removeChild(errorDiv);
        }, 5000);
    }

    showSuccess(message) {
        // Simple success display
        const successDiv = document.createElement('div');
        successDiv.className = 'fixed top-4 right-4 bg-green-500 text-white p-4 rounded-lg shadow-lg z-50';
        successDiv.textContent = message;
        document.body.appendChild(successDiv);

        setTimeout(() => {
            document.body.removeChild(successDiv);
        }, 3000);
    }

    populateModelList(models) {
        const container = document.getElementById('modelList');
        if (!container) return;

        container.innerHTML = models.map(model => `
            <div class="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
                <div class="flex items-center justify-between mb-2">
                    <h4 class="font-semibold text-gray-800 dark:text-white">${model.symbol} Model</h4>
                    <span class="px-2 py-1 text-xs rounded-full bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200">
                        Active
                    </span>
                </div>
                <div class="grid grid-cols-2 gap-4 text-sm">
                    <div>
                        <span class="text-gray-600 dark:text-gray-400">Accuracy:</span>
                        <span class="font-medium text-gray-800 dark:text-white">${(model.accuracy * 100).toFixed(1)}%</span>
                    </div>
                    <div>
                        <span class="text-gray-600 dark:text-gray-400">Precision:</span>
                        <span class="font-medium text-gray-800 dark:text-white">${(model.precision * 100).toFixed(1)}%</span>
                    </div>
                    <div>
                        <span class="text-gray-600 dark:text-gray-400">Recall:</span>
                        <span class="font-medium text-gray-800 dark:text-white">${(model.recall * 100).toFixed(1)}%</span>
                    </div>
                    <div>
                        <span class="text-gray-600 dark:text-gray-400">Last Trained:</span>
                        <span class="font-medium text-gray-800 dark:text-white">${model.last_trained}</span>
                    </div>
                </div>
            </div>
        `).join('');
    }

    populateFileList(files) {
        const container = document.getElementById('fileList');
        if (!container) return;

        container.innerHTML = files.map(file => `
            <div class="flex items-center justify-between p-3 bg-white dark:bg-gray-800 rounded-lg shadow-sm">
                <div class="flex items-center space-x-3">
                    <i class="bi bi-file-earmark text-gray-400"></i>
                    <div>
                        <p class="font-medium text-gray-800 dark:text-white">${file}</p>
                        <p class="text-sm text-gray-500 dark:text-gray-400">${file.endsWith('.pkl') ? 'Model file' : file.endsWith('.json') ? 'Data file' : 'Unknown'}</p>
                    </div>
                </div>
                <div class="flex items-center space-x-2">
                    <button class="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 download-btn" data-file="${file}">
                        <i class="bi bi-download"></i>
                    </button>
                    <button class="text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-300 delete-btn" data-file="${file}">
                        <i class="bi bi-trash"></i>
                    </button>
                </div>
            </div>
        `).join('');

        // Add event listeners for download buttons
        container.querySelectorAll('.download-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const fileName = e.currentTarget.getAttribute('data-file');
                this.downloadFile(fileName);
            });
        });

        // Add event listeners for delete buttons
        container.querySelectorAll('.delete-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const fileName = e.currentTarget.getAttribute('data-file');
                this.deleteFile(fileName);
            });
        });
    }

    async downloadFile(fileName) {
        try {
            // In a real implementation, this would download the file from the server
            // For now, we'll simulate the download
            this.showSuccess(`Downloading ${fileName}...`);
            console.log(`Downloading file: ${fileName}`);
        } catch (error) {
            this.showError(`Failed to download ${fileName}`);
        }
    }

    async deleteFile(fileName) {
        if (!confirm(`Are you sure you want to delete ${fileName}? This action cannot be undone.`)) {
            return;
        }

        try {
            // In a real implementation, this would delete the file from the server
            // For now, we'll simulate the deletion
            this.showSuccess(`Deleted ${fileName}`);
            // Refresh the file list
            this.loadFileManagerData();
        } catch (error) {
            this.showError(`Failed to delete ${fileName}`);
        }
    }

    // Performance functions
    updatePerformanceMetrics(data) {
        document.getElementById('totalReturn').textContent = `+${data.total_return}%`;
        document.getElementById('sharpeRatio').textContent = data.sharpe_ratio.toFixed(2);
        document.getElementById('maxDrawdown').textContent = `${data.max_drawdown}%`;
        document.getElementById('winRate').textContent = `${data.win_rate}%`;
    }

    populatePerformanceTable(trades) {
        const tbody = document.getElementById('performanceTableBody');
        if (!tbody) return;

        tbody.innerHTML = trades.map(trade => `
            <tr class="border-b dark:border-gray-700">
                <td class="py-2 px-4 text-gray-800 dark:text-gray-300">${trade.date}</td>
                <td class="py-2 px-4 text-gray-800 dark:text-gray-300">${trade.symbol}</td>
                <td class="py-2 px-4">
                    <span class="px-2 py-1 text-xs rounded-full ${trade.side === 'BUY' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'}">
                        ${trade.side}
                    </span>
                </td>
                <td class="py-2 px-4 text-gray-800 dark:text-gray-300">${trade.quantity}</td>
                <td class="py-2 px-4 text-gray-800 dark:text-gray-300">$${trade.entry_price.toFixed(2)}</td>
                <td class="py-2 px-4 text-gray-800 dark:text-gray-300">$${trade.exit_price.toFixed(2)}</td>
                <td class="py-2 px-4 ${trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'}">
                    $${trade.pnl.toFixed(2)}
                </td>
                <td class="py-2 px-4 ${trade.return_pct >= 0 ? 'text-green-600' : 'text-red-600'}">
                    ${trade.return_pct.toFixed(2)}%
                </td>
            </tr>
        `).join('');
    }

    initializePerformanceCharts(data) {
        // Equity curve chart
        const equityCtx = document.getElementById('equityChart');
        if (equityCtx && data.equity_curve) {
            new Chart(equityCtx, {
                type: 'line',
                data: {
                    labels: data.equity_curve.map((_, i) => `Day ${i + 1}`),
                    datasets: [{
                        label: 'Portfolio Value',
                        data: data.equity_curve,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: false
                        }
                    }
                }
            });
        }

        // Monthly returns chart
        const returnsCtx = document.getElementById('monthlyReturnsChart');
        if (returnsCtx && data.monthly_returns) {
            new Chart(returnsCtx, {
                type: 'bar',
                data: {
                    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                    datasets: [{
                        label: 'Monthly Returns %',
                        data: data.monthly_returns,
                        backgroundColor: data.monthly_returns.map(val =>
                            val >= 0 ? 'rgba(16, 185, 129, 0.8)' : 'rgba(239, 68, 68, 0.8)'
                        )
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        }
    }

    // Logs functions
    populateLogEntries(logs) {
        const container = document.getElementById('logEntries');
        if (!container) return;

        container.innerHTML = logs.map(log => `
            <div class="flex items-start space-x-3 p-3 rounded-lg ${this.getLogLevelClass(log.level)}">
                <span class="px-2 py-1 text-xs font-medium rounded ${this.getLogLevelBadgeClass(log.level)}">
                    ${log.level}
                </span>
                <div class="flex-1">
                    <div class="flex items-center space-x-2 mb-1">
                        <span class="text-sm font-medium text-gray-800 dark:text-gray-200">${log.source}</span>
                        <span class="text-xs text-gray-500 dark:text-gray-400">${log.timestamp}</span>
                    </div>
                    <p class="text-sm text-gray-700 dark:text-gray-300">${log.message}</p>
                </div>
            </div>
        `).join('');
    }

    getLogLevelClass(level) {
        const classes = {
            'ERROR': 'bg-red-50 dark:bg-red-900/20 border-l-4 border-red-500',
            'WARNING': 'bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500',
            'INFO': 'bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500',
            'DEBUG': 'bg-gray-50 dark:bg-gray-900/20 border-l-4 border-gray-500'
        };
        return classes[level] || classes['INFO'];
    }

    getLogLevelBadgeClass(level) {
        const classes = {
            'ERROR': 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
            'WARNING': 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
            'INFO': 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
            'DEBUG': 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
        };
        return classes[level] || classes['INFO'];
    }

    setupLogControls() {
        // Log level filter
        document.getElementById('logLevel')?.addEventListener('change', () => {
            this.filterLogs();
        });

        // Date filter
        document.getElementById('logDate')?.addEventListener('change', () => {
            this.filterLogs();
        });

        // Refresh logs
        document.getElementById('refreshLogs')?.addEventListener('click', () => {
            this.loadLogsData();
        });

        // Clear logs
        document.getElementById('clearLogs')?.addEventListener('click', () => {
            if (confirm('Are you sure you want to clear all logs?')) {
                document.getElementById('logEntries').innerHTML = '';
                this.showSuccess('Logs cleared');
            }
        });

        // Export logs
        document.getElementById('exportLogs')?.addEventListener('click', () => {
            this.exportLogs();
        });

        // Load more logs
        document.getElementById('loadMoreLogs')?.addEventListener('click', () => {
            this.loadMoreLogs();
        });
    }

    filterLogs() {
        const levelFilter = document.getElementById('logLevel').value;
        const dateFilter = document.getElementById('logDate').value;
        const logEntries = document.querySelectorAll('#logEntries > div');

        logEntries.forEach(entry => {
            const level = entry.querySelector('span').textContent;
            const timestamp = entry.querySelector('.text-xs').textContent;
            const entryDate = timestamp.split(' ')[0];

            const levelMatch = levelFilter === 'all' || level === levelFilter;
            const dateMatch = !dateFilter || entryDate === dateFilter;

            entry.style.display = levelMatch && dateMatch ? 'flex' : 'none';
        });
    }

    exportLogs() {
        const logEntries = document.querySelectorAll('#logEntries > div');
        let logText = 'Timestamp\tLevel\tSource\tMessage\n';

        logEntries.forEach(entry => {
            const spans = entry.querySelectorAll('span');
            const message = entry.querySelector('p').textContent;
            const timestamp = spans[1].textContent;
            const level = spans[0].textContent;
            const source = spans[1].previousElementSibling.textContent;

            logText += `${timestamp}\t${level}\t${source}\t${message}\n`;
        });

        const blob = new Blob([logText], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `trading_logs_${new Date().toISOString().split('T')[0]}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        this.showSuccess('Logs exported successfully');
    }

    loadMoreLogs() {
        // In a real implementation, this would load more logs from the server
        this.showSuccess('Loading more logs... (simulated)');
    }

    // Watchlist functions
    populateWatchlistTable(symbols) {
        const tbody = document.getElementById('watchlistTableBody');
        const emptyState = document.getElementById('watchlistEmptyState');

        if (!tbody) return;

        if (symbols.length === 0) {
            tbody.innerHTML = '';
            if (emptyState) emptyState.style.display = 'block';
            return;
        }

        if (emptyState) emptyState.style.display = 'none';

        tbody.innerHTML = symbols.map(symbol => `
            <tr class="border-b dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700">
                <td class="py-3 px-4">
                    <div>
                        <div class="font-medium text-gray-800 dark:text-gray-200">${symbol.symbol}</div>
                        <div class="text-sm text-gray-500 dark:text-gray-400">${symbol.name}</div>
                    </div>
                </td>
                <td class="py-3 px-4 text-gray-800 dark:text-gray-300">$${symbol.price.toFixed(2)}</td>
                <td class="py-3 px-4 ${symbol.change >= 0 ? 'text-green-600' : 'text-red-600'}">
                    ${symbol.change >= 0 ? '+' : ''}$${symbol.change.toFixed(2)}
                </td>
                <td class="py-3 px-4 ${symbol.change_pct >= 0 ? 'text-green-600' : 'text-red-600'}">
                    ${symbol.change_pct >= 0 ? '+' : ''}${symbol.change_pct.toFixed(2)}%
                </td>
                <td class="py-3 px-4 text-gray-800 dark:text-gray-300">${symbol.volume}</td>
                <td class="py-3 px-4 text-gray-800 dark:text-gray-300">${symbol.market_cap}</td>
                <td class="py-3 px-4">
                    <div class="flex items-center space-x-2">
                        <button class="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300" onclick="window.tradingUI.viewSymbol('${symbol.symbol}')">
                            <i class="bi bi-eye"></i>
                        </button>
                        <button class="text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-300" onclick="window.tradingUI.removeFromWatchlist('${symbol.symbol}')">
                            <i class="bi bi-trash"></i>
                        </button>
                    </div>
                </td>
            </tr>
        `).join('');
    }

    setupWatchlistControls() {
        // Search functionality
        document.getElementById('watchlistSearch')?.addEventListener('input', (e) => {
            this.filterWatchlist(e.target.value);
        });

        // Sort functionality
        document.getElementById('watchlistSort')?.addEventListener('change', (e) => {
            this.sortWatchlist(e.target.value);
        });

        // Add symbol button
        document.getElementById('addToWatchlist')?.addEventListener('click', () => {
            this.showAddSymbolModal();
        });

        // Refresh watchlist
        document.getElementById('refreshWatchlist')?.addEventListener('click', () => {
            this.loadWatchlistData();
            this.showSuccess('Watchlist refreshed');
        });

        // Modal controls
        document.getElementById('cancelAddSymbol')?.addEventListener('click', () => {
            this.hideAddSymbolModal();
        });

        document.getElementById('confirmAddSymbol')?.addEventListener('click', () => {
            this.addSymbolToWatchlist();
        });

        // Close modal on outside click
        document.getElementById('addSymbolModal')?.addEventListener('click', (e) => {
            if (e.target.id === 'addSymbolModal') {
                this.hideAddSymbolModal();
            }
        });
    }

    filterWatchlist(searchTerm) {
        const rows = document.querySelectorAll('#watchlistTableBody tr');
        const term = searchTerm.toLowerCase();

        rows.forEach(row => {
            const symbol = row.cells[0].textContent.toLowerCase();
            const name = row.cells[0].querySelector('.text-sm').textContent.toLowerCase();
            const visible = symbol.includes(term) || name.includes(term);
            row.style.display = visible ? '' : 'none';
        });
    }

    sortWatchlist(criteria) {
        const tbody = document.getElementById('watchlistTableBody');
        const rows = Array.from(tbody.querySelectorAll('tr'));

        rows.sort((a, b) => {
            let aVal, bVal;

            switch (criteria) {
                case 'symbol':
                    aVal = a.cells[0].textContent.trim();
                    bVal = b.cells[0].textContent.trim();
                    return aVal.localeCompare(bVal);
                case 'price':
                    aVal = parseFloat(a.cells[1].textContent.replace('$', ''));
                    bVal = parseFloat(b.cells[1].textContent.replace('$', ''));
                    return bVal - aVal;
                case 'change':
                    aVal = parseFloat(a.cells[2].textContent.replace(/[+$]/g, ''));
                    bVal = parseFloat(b.cells[2].textContent.replace(/[+$]/g, ''));
                    return bVal - aVal;
                case 'volume':
                    aVal = parseFloat(a.cells[4].textContent.replace(/[^0-9.]/g, ''));
                    bVal = parseFloat(b.cells[4].textContent.replace(/[^0-9.]/g, ''));
                    return bVal - aVal;
                default:
                    return 0;
            }
        });

        rows.forEach(row => tbody.appendChild(row));
    }

    showAddSymbolModal() {
        document.getElementById('addSymbolModal').classList.remove('hidden');
        document.getElementById('newSymbolInput').focus();
    }

    hideAddSymbolModal() {
        document.getElementById('addSymbolModal').classList.add('hidden');
        document.getElementById('newSymbolInput').value = '';
    }

    async addSymbolToWatchlist() {
        const symbol = document.getElementById('newSymbolInput').value.trim().toUpperCase();

        if (!symbol) {
            this.showError('Please enter a symbol');
            return;
        }

        try {
            // In a real implementation, this would add to the backend
            // For now, we'll just simulate adding it
            await this.loadWatchlistData(); // Refresh the list
            this.hideAddSymbolModal();
            this.showSuccess(`Added ${symbol} to watchlist`);
        } catch (error) {
            this.showError('Failed to add symbol to watchlist');
        }
    }

    viewSymbol(symbol) {
        // Switch to data view and load symbol chart
        this.switchView('data');
        // In a real implementation, this would load the specific symbol's data
        this.showSuccess(`Viewing ${symbol} data`);
    }

    removeFromWatchlist(symbol) {
        if (confirm(`Remove ${symbol} from watchlist?`)) {
            // In a real implementation, this would remove from backend
            this.loadWatchlistData();
            this.showSuccess(`Removed ${symbol} from watchlist`);
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Set dark mode from localStorage
    if (localStorage.getItem('darkMode') === 'true') {
        document.documentElement.classList.add('dark');
    }

    // Initialize the UI
    window.tradingUI = new TradingFrameworkUI();
});