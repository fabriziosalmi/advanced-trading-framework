import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  Alert,
} from '@mui/material';

// Components
import MetricCard from './components/dashboard/MetricCard';
import TradingControl from './components/dashboard/TradingControl';
import PerformanceChart from './components/dashboard/PerformanceChart';
import AllocationPieChart from './components/dashboard/AllocationPieChart';

// Hooks
import useWebSocket from '../hooks/useWebSocket';

const Dashboard = () => {
  const [dashboardData, setDashboardData] = useState({
    portfolio_value: 0,
    daily_pnl: 0,
    active_strategies: 0,
    open_positions: 0,
  });
  const [systemStatus, setSystemStatus] = useState('Stopped');
  const [error, setError] = useState(null);

  // WebSocket for real-time updates
  const { isConnected } = useWebSocket((message) => {
    if (message.event === 'system_status') {
      setSystemStatus(message.data);
    } else if (message.event === 'pnl_update') {
      setDashboardData(prev => ({
        ...prev,
        daily_pnl: message.data
      }));
    } else if (message.event === 'new_trade') {
      // Refresh dashboard data when new trade occurs
      fetchDashboardData();
    }
  });

  // Fetch dashboard summary data
  const fetchDashboardData = async () => {
    try {
      const response = await fetch('/api/dashboard/summary');
      if (!response.ok) throw new Error('Failed to fetch dashboard data');
      const data = await response.json();
      setDashboardData(data);
      setError(null);
    } catch (err) {
      setError('Failed to load dashboard data');
      console.error('Dashboard data fetch error:', err);
    }
  };

  useEffect(() => {
    fetchDashboardData();
    // Refresh every 30 seconds
    const interval = setInterval(fetchDashboardData, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom sx={{ color: 'primary.main', fontWeight: 'bold' }}>
          📊 Trading Dashboard
        </Typography>
        <Typography variant="body1" sx={{ color: 'text.secondary', mb: 2 }}>
          Real-time portfolio monitoring and trading controls
        </Typography>

        {/* System Status Indicator */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
          <Typography variant="body2" sx={{ color: 'text.secondary' }}>
            System Status:
          </Typography>
          <Box
            sx={{
              px: 2,
              py: 0.5,
              borderRadius: 1,
              backgroundColor: systemStatus === 'Running' ? 'success.main' : 'warning.main',
              color: 'white',
              fontSize: '0.875rem',
              fontWeight: 'bold'
            }}
          >
            {systemStatus}
          </Box>
          {!isConnected && (
            <Typography variant="caption" sx={{ color: 'warning.main' }}>
              ⚠️ WebSocket disconnected
            </Typography>
          )}
        </Box>

        {/* Error Alert */}
        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}
      </Box>

      {/* Metric Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Portfolio Value"
            value={`$${dashboardData.portfolio_value.toLocaleString()}`}
            icon="💰"
            color="primary"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Daily P&L"
            value={`$${dashboardData.daily_pnl.toFixed(2)}`}
            icon="📈"
            color={dashboardData.daily_pnl >= 0 ? "success" : "error"}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Active Strategies"
            value={dashboardData.active_strategies}
            icon="🎯"
            color="secondary"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Open Positions"
            value={dashboardData.open_positions}
            icon="📊"
            color="info"
          />
        </Grid>
      </Grid>

      {/* Trading Controls */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12}>
          <Paper sx={{ p: 3, backgroundColor: 'background.paper' }}>
            <TradingControl onStatusChange={setSystemStatus} />
          </Paper>
        </Grid>
      </Grid>

      {/* Charts */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3, backgroundColor: 'background.paper' }}>
            <PerformanceChart />
          </Paper>
        </Grid>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, backgroundColor: 'background.paper' }}>
            <AllocationPieChart />
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Dashboard;