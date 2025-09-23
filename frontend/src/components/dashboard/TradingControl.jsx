import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Typography,
  Grid,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  Chip,
} from '@mui/material';
import {
  PlayArrow as StartIcon,
  Stop as StopIcon,
  Warning as EmergencyIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';
import axios from 'axios';

const TradingControl = ({ onStatusChange }) => {
  const [tradingMode, setTradingMode] = useState('manual');
  const [riskSettings, setRiskSettings] = useState({
    max_daily_loss: 1000,
    position_size: 1000,
  });
  const [systemStatus, setSystemStatus] = useState('Stopped');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  // Fetch current trading status
  const fetchStatus = async () => {
    try {
      const response = await axios.get('/api/trading/status');
      const status = response.data;
      setSystemStatus(status.is_running ? 'Running' : 'Stopped');
      setTradingMode(status.mode);
      setRiskSettings(status.risk_settings);
      onStatusChange(status.is_running ? 'Running' : 'Stopped');
    } catch (err) {
      console.error('Error fetching trading status:', err);
    }
  };

  useEffect(() => {
    fetchStatus();
    // Refresh status every 10 seconds
    const interval = setInterval(fetchStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleStartTrading = async () => {
    setLoading(true);
    setError(null);
    try {
      await axios.post('/api/trading/start', {
        mode: tradingMode,
        risk_settings: riskSettings,
      });
      setSuccess('Trading started successfully');
      setSystemStatus('Running');
      onStatusChange('Running');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError('Failed to start trading');
      console.error('Start trading error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleStopTrading = async () => {
    setLoading(true);
    setError(null);
    try {
      await axios.post('/api/trading/stop');
      setSuccess('Trading stopped successfully');
      setSystemStatus('Stopped');
      onStatusChange('Stopped');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError('Failed to stop trading');
      console.error('Stop trading error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleEmergencyStop = async () => {
    if (!window.confirm('Are you sure you want to execute an emergency stop? This will liquidate all positions immediately.')) {
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await axios.post('/api/trading/emergency-stop');
      setSuccess('Emergency stop executed - all positions liquidated');
      setSystemStatus('Stopped');
      onStatusChange('Stopped');
      console.log('Emergency stop result:', response.data);
      setTimeout(() => setSuccess(null), 5000);
    } catch (err) {
      setError('Failed to execute emergency stop');
      console.error('Emergency stop error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleModeChange = async (newMode) => {
    try {
      await axios.post('/api/trading/mode', newMode);
      setTradingMode(newMode);
      setSuccess(`Trading mode changed to ${newMode}`);
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError('Failed to change trading mode');
      console.error('Mode change error:', err);
    }
  };

  const handleRiskSettingsUpdate = async () => {
    try {
      await axios.post('/api/settings/risk', riskSettings);
      setSuccess('Risk settings updated successfully');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError('Failed to update risk settings');
      console.error('Risk settings error:', err);
    }
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom sx={{ color: 'primary.main', fontWeight: 'bold' }}>
        🎮 Trading Control Panel
      </Typography>

      {/* Status Display */}
      <Box sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Typography variant="body1" sx={{ color: 'text.secondary' }}>
            Current Status:
          </Typography>
          <Chip
            label={systemStatus}
            color={systemStatus === 'Running' ? 'success' : 'default'}
            variant={systemStatus === 'Running' ? 'filled' : 'outlined'}
          />
        </Box>
      </Box>

      {/* Alerts */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      {success && (
        <Alert severity="success" sx={{ mb: 2 }}>
          {success}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Trading Mode */}
        <Grid item xs={12} md={6}>
          <FormControl fullWidth>
            <InputLabel>Trading Mode</InputLabel>
            <Select
              value={tradingMode}
              label="Trading Mode"
              onChange={(e) => handleModeChange(e.target.value)}
              disabled={systemStatus === 'Running'}
            >
              <MenuItem value="manual">Manual</MenuItem>
              <MenuItem value="automatic">Automatic</MenuItem>
            </Select>
          </FormControl>
        </Grid>

        {/* Risk Settings */}
        <Grid item xs={12} md={6}>
          <Box sx={{ display: 'flex', gap: 2 }}>
            <TextField
              label="Max Daily Loss ($)"
              type="number"
              value={riskSettings.max_daily_loss}
              onChange={(e) => setRiskSettings(prev => ({
                ...prev,
                max_daily_loss: parseFloat(e.target.value) || 0
              }))}
              size="small"
              disabled={systemStatus === 'Running'}
            />
            <TextField
              label="Position Size ($)"
              type="number"
              value={riskSettings.position_size}
              onChange={(e) => setRiskSettings(prev => ({
                ...prev,
                position_size: parseFloat(e.target.value) || 0
              }))}
              size="small"
              disabled={systemStatus === 'Running'}
            />
            <Button
              variant="outlined"
              startIcon={<SettingsIcon />}
              onClick={handleRiskSettingsUpdate}
              disabled={systemStatus === 'Running'}
              sx={{ minWidth: 'auto' }}
            >
              Update
            </Button>
          </Box>
        </Grid>

        {/* Control Buttons */}
        <Grid item xs={12}>
          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
            <Button
              variant="contained"
              color="success"
              startIcon={<StartIcon />}
              onClick={handleStartTrading}
              disabled={systemStatus === 'Running' || loading}
              size="large"
            >
              Start Trading
            </Button>

            <Button
              variant="contained"
              color="warning"
              startIcon={<StopIcon />}
              onClick={handleStopTrading}
              disabled={systemStatus === 'Stopped' || loading}
              size="large"
            >
              Stop Trading
            </Button>

            <Button
              variant="contained"
              color="error"
              startIcon={<EmergencyIcon />}
              onClick={handleEmergencyStop}
              disabled={loading}
              size="large"
            >
              Emergency Stop
            </Button>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
};

export default TradingControl;