import React, { useState, useEffect } from 'react';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
  Typography,
  Box,
  Chip,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  ShowChart as SignalsIcon,
  Timeline as PerformanceIcon,
  Receipt as LogsIcon,
  Settings as SettingsIcon,
  Monitor as MonitoringIcon,
  Replay as BacktestingIcon,
  Analytics as TechnicalIcon,
  AccountBalance as PortfolioIcon,
  ShoppingCart as OrdersIcon,
  Psychology as StrategiesIcon,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import axios from 'axios';

const drawerWidth = 280;

const Sidebar = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [sidebarData, setSidebarData] = useState({
    positions: [],
    orders: [],
    strategies: [],
    logs: []
  });

  // Fetch sidebar data
  useEffect(() => {
    const fetchSidebarData = async () => {
      try {
        const [positionsRes, ordersRes, strategiesRes, logsRes] = await Promise.all([
          axios.get('/api/portfolio/positions'),
          axios.get('/api/trading/orders?limit=10'),
          axios.get('/api/strategies'),
          axios.get('/api/monitoring/logs?limit=5')
        ]);

        setSidebarData({
          positions: positionsRes.data,
          orders: ordersRes.data,
          strategies: strategiesRes.data,
          logs: logsRes.data.logs || []
        });
      } catch (error) {
        console.error('Error fetching sidebar data:', error);
      }
    };

    fetchSidebarData();
    // Refresh every 30 seconds
    const interval = setInterval(fetchSidebarData, 30000);
    return () => clearInterval(interval);
  }, []);

  const menuItems = [
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/dashboard' },
    { text: 'Portfolio', icon: <PortfolioIcon />, path: '/portfolio' },
    { text: 'Signals', icon: <SignalsIcon />, path: '/signals' },
    { text: 'Performance', icon: <PerformanceIcon />, path: '/performance' },
    { text: 'Orders', icon: <OrdersIcon />, path: '/orders' },
    { text: 'Strategies', icon: <StrategiesIcon />, path: '/strategies' },
    { text: 'Logs', icon: <LogsIcon />, path: '/logs' },
    { text: 'Settings', icon: <SettingsIcon />, path: '/settings' },
    { text: 'Monitoring', icon: <MonitoringIcon />, path: '/monitoring' },
    { text: 'Backtesting', icon: <BacktestingIcon />, path: '/backtesting' },
    { text: 'Technical Analysis', icon: <TechnicalIcon />, path: '/technical' },
  ];

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: drawerWidth,
          boxSizing: 'border-box',
          backgroundColor: 'background.paper',
          borderRight: '1px solid rgba(255, 255, 255, 0.12)',
        },
      }}
    >
      <Box sx={{ p: 2 }}>
        <Typography variant="h6" component="div" sx={{ color: 'primary.main', fontWeight: 'bold' }}>
          🚀 Trading Framework
        </Typography>
        <Typography variant="body2" sx={{ color: 'text.secondary', mt: 1 }}>
          Advanced Algorithmic Trading
        </Typography>
      </Box>

      <Divider />

      <List>
        {menuItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton
              selected={location.pathname === item.path}
              onClick={() => navigate(item.path)}
              sx={{
                '&.Mui-selected': {
                  backgroundColor: 'rgba(0, 255, 136, 0.1)',
                  '&:hover': {
                    backgroundColor: 'rgba(0, 255, 136, 0.2)',
                  },
                },
              }}
            >
              <ListItemIcon sx={{ color: location.pathname === item.path ? 'primary.main' : 'text.secondary' }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText
                primary={item.text}
                sx={{
                  '& .MuiListItemText-primary': {
                    color: location.pathname === item.path ? 'primary.main' : 'text.primary',
                  },
                }}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>

      <Divider />

      {/* Quick Stats */}
      <Box sx={{ p: 2 }}>
        <Typography variant="subtitle2" sx={{ mb: 1, color: 'text.secondary' }}>
          Quick Stats
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
          <Chip
            label={`${sidebarData.positions.length} Positions`}
            size="small"
            variant="outlined"
            sx={{ borderColor: 'primary.main', color: 'primary.main' }}
          />
        </Box>
        <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
          <Chip
            label={`${sidebarData.strategies.length} Strategies`}
            size="small"
            variant="outlined"
            sx={{ borderColor: 'secondary.main', color: 'secondary.main' }}
          />
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Chip
            label={`${sidebarData.orders.length} Recent Orders`}
            size="small"
            variant="outlined"
            sx={{ borderColor: 'text.secondary', color: 'text.secondary' }}
          />
        </Box>
      </Box>

      {/* Recent Logs */}
      <Box sx={{ p: 2, flexGrow: 1 }}>
        <Typography variant="subtitle2" sx={{ mb: 1, color: 'text.secondary' }}>
          Recent Activity
        </Typography>
        <Box sx={{ maxHeight: 200, overflowY: 'auto' }}>
          {sidebarData.logs.slice(0, 3).map((log, index) => (
            <Typography
              key={index}
              variant="caption"
              sx={{
                display: 'block',
                color: 'text.secondary',
                fontSize: '0.7rem',
                mb: 0.5,
                fontFamily: 'monospace'
              }}
            >
              {log.timestamp} {log.message}
            </Typography>
          ))}
        </Box>
      </Box>
    </Drawer>
  );
};

export default Sidebar;