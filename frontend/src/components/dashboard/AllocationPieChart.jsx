import React, { useState, useEffect } from 'react';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  Legend,
} from 'recharts';
import {
  Box,
  Typography,
  CircularProgress,
  Alert,
} from '@mui/material';
import axios from 'axios';

const COLORS = ['#00ff88', '#4dabf7', '#ffd43b', '#ff6b6b', '#9775fa', '#ffa94d'];

const AllocationPieChart = () => {
  const [data, setData] = useState([]);
  const [totalValue, setTotalValue] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchAllocationData = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get('/api/portfolio/allocation');
      setData(response.data.allocations);
      setTotalValue(response.data.total_value);
    } catch (err) {
      setError('Failed to load allocation data');
      console.error('Allocation data fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAllocationData();
    // Refresh every 60 seconds
    const interval = setInterval(fetchAllocationData, 60000);
    return () => clearInterval(interval);
  }, []);

  const formatTooltipValue = (value, name) => {
    return [`$${value.toLocaleString()}`, name];
  };

  const formatLegendValue = (value, entry) => {
    return `${entry.payload.asset}: ${entry.payload.percentage.toFixed(1)}%`;
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 300 }}>
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
      <Typography variant="h6" sx={{ color: 'primary.main', fontWeight: 'bold', mb: 3 }}>
        🥧 Asset Allocation
      </Typography>

      <Typography variant="body2" sx={{ color: 'text.secondary', mb: 2 }}>
        Total Portfolio Value: ${totalValue.toLocaleString()}
      </Typography>

      <Box sx={{ width: '100%', height: 300 }}>
        <ResponsiveContainer>
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ asset, percentage }) => `${asset}: ${percentage.toFixed(1)}%`}
              outerRadius={80}
              fill="#8884d8"
              dataKey="value"
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip
              formatter={formatTooltipValue}
              contentStyle={{
                backgroundColor: '#262730',
                border: '1px solid rgba(255,255,255,0.12)',
                borderRadius: '8px',
                color: '#fafafa',
              }}
            />
            <Legend
              formatter={formatLegendValue}
              wrapperStyle={{ color: '#c4c4c4' }}
            />
          </PieChart>
        </ResponsiveContainer>
      </Box>

      {/* Allocation Details */}
      <Box sx={{ mt: 2 }}>
        {data.map((item, index) => (
          <Box
            key={item.asset}
            sx={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              py: 0.5,
              borderBottom: index < data.length - 1 ? '1px solid rgba(255,255,255,0.1)' : 'none'
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Box
                sx={{
                  width: 12,
                  height: 12,
                  borderRadius: '50%',
                  backgroundColor: COLORS[index % COLORS.length],
                }}
              />
              <Typography variant="body2" sx={{ color: 'text.primary' }}>
                {item.asset}
              </Typography>
            </Box>
            <Box sx={{ textAlign: 'right' }}>
              <Typography variant="body2" sx={{ color: 'text.primary', fontWeight: 'bold' }}>
                ${item.value.toLocaleString()}
              </Typography>
              <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                {item.percentage.toFixed(1)}%
              </Typography>
            </Box>
          </Box>
        ))}
      </Box>
    </Box>
  );
};

export default AllocationPieChart;