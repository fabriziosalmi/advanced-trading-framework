import React, { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import {
  Box,
  Typography,
  CircularProgress,
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import axios from 'axios';

const PerformanceChart = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [timeRange, setTimeRange] = useState(30);

  const fetchPerformanceData = async (days = 30) => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(`/api/performance/history?days=${days}`);
      setData(response.data.history);
    } catch (err) {
      setError('Failed to load performance data');
      console.error('Performance data fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPerformanceData(timeRange);
  }, [timeRange]);

  const handleTimeRangeChange = (newRange) => {
    setTimeRange(newRange);
  };

  const formatTooltipValue = (value, name) => {
    if (name === 'value') {
      return [`$${value.toLocaleString()}`, 'Portfolio Value'];
    }
    if (name === 'change') {
      return [`${value >= 0 ? '+' : ''}${value.toFixed(2)}%`, 'Daily Change'];
    }
    return [value, name];
  };

  const formatYAxisValue = (value) => {
    return `$${(value / 1000).toFixed(0)}K`;
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error">
        {error}
      </Alert>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h6" sx={{ color: 'primary.main', fontWeight: 'bold' }}>
          📈 Portfolio Performance
        </Typography>

        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel>Time Range</InputLabel>
          <Select
            value={timeRange}
            label="Time Range"
            onChange={(e) => handleTimeRangeChange(e.target.value)}
          >
            <MenuItem value={7}>7 Days</MenuItem>
            <MenuItem value={30}>30 Days</MenuItem>
            <MenuItem value={90}>90 Days</MenuItem>
            <MenuItem value={180}>6 Months</MenuItem>
            <MenuItem value={365}>1 Year</MenuItem>
          </Select>
        </FormControl>
      </Box>

      <Box sx={{ width: '100%', height: 400 }}>
        <ResponsiveContainer>
          <LineChart
            data={data}
            margin={{
              top: 5,
              right: 30,
              left: 20,
              bottom: 5,
            }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis
              dataKey="date"
              stroke="#c4c4c4"
              fontSize={12}
              tick={{ fill: '#c4c4c4' }}
            />
            <YAxis
              stroke="#c4c4c4"
              fontSize={12}
              tick={{ fill: '#c4c4c4' }}
              tickFormatter={formatYAxisValue}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#262730',
                border: '1px solid rgba(255,255,255,0.12)',
                borderRadius: '8px',
                color: '#fafafa',
              }}
              formatter={formatTooltipValue}
              labelStyle={{ color: '#fafafa' }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="value"
              stroke="#00ff88"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, fill: '#00ff88' }}
              name="Portfolio Value"
            />
          </LineChart>
        </ResponsiveContainer>
      </Box>

      {/* Summary Stats */}
      {data.length > 0 && (
        <Box sx={{ mt: 2, display: 'flex', gap: 4, flexWrap: 'wrap' }}>
          <Typography variant="body2" sx={{ color: 'text.secondary' }}>
            Start Value: ${data[0]?.value.toLocaleString()}
          </Typography>
          <Typography variant="body2" sx={{ color: 'text.secondary' }}>
            End Value: ${data[data.length - 1]?.value.toLocaleString()}
          </Typography>
          <Typography
            variant="body2"
            sx={{
              color: data[data.length - 1]?.value >= data[0]?.value ? 'success.main' : 'error.main'
            }}
          >
            Change: {data.length > 1 ?
              `${((data[data.length - 1].value - data[0].value) / data[0].value * 100).toFixed(2)}%` :
              '0.00%'
            }
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default PerformanceChart;