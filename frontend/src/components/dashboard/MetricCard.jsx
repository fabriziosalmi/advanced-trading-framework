import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  useTheme,
} from '@mui/material';

const MetricCard = ({ title, value, icon, color = 'primary' }) => {
  const theme = useTheme();

  return (
    <Card
      sx={{
        backgroundColor: 'background.paper',
        border: `1px solid ${theme.palette.divider}`,
        borderRadius: 2,
        transition: 'all 0.3s ease',
        '&:hover': {
          boxShadow: theme.shadows[4],
          transform: 'translateY(-2px)',
        },
      }}
    >
      <CardContent sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Typography
            variant="h6"
            component="div"
            sx={{
              color: 'text.primary',
              fontWeight: 600,
              fontSize: '0.875rem',
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
            }}
          >
            {title}
          </Typography>
          <Box sx={{ fontSize: '1.5rem' }}>
            {icon}
          </Box>
        </Box>

        <Typography
          variant="h4"
          component="div"
          sx={{
            color: theme.palette[color]?.main || theme.palette.primary.main,
            fontWeight: 'bold',
            fontSize: '1.75rem',
          }}
        >
          {value}
        </Typography>
      </CardContent>
    </Card>
  );
};

export default MetricCard;