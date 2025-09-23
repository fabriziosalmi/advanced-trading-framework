# Trading Framework Frontend

Modern React-based frontend for the Advanced Trading Framework.

## Tech Stack

- **React 18** with Vite for fast development
- **Material-UI (MUI)** for components and theming
- **Recharts** for data visualization
- **Axios** for API communication
- **React Router** for navigation

## Features

### Dashboard
- Real-time portfolio metrics (4 main cards)
- Trading control panel with start/stop/emergency controls
- Portfolio performance chart with time range selection
- Asset allocation pie chart
- WebSocket integration for live updates

### Navigation
- Sidebar with all main sections
- Quick stats display
- Recent activity feed

### Real-time Updates
- WebSocket connection for live data
- Automatic reconnection on disconnect
- System status indicators

## Development Setup

1. **Install dependencies:**
   ```bash
   cd frontend
   npm install
   ```

2. **Start development server:**
   ```bash
   npm run dev
   ```

3. **Build for production:**
   ```bash
   npm run build
   ```

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── Sidebar.jsx              # Main navigation sidebar
│   │   ├── dashboard/
│   │   │   ├── MetricCard.jsx       # Portfolio metric cards
│   │   │   ├── TradingControl.jsx   # Trading controls panel
│   │   │   ├── PerformanceChart.jsx # Portfolio performance chart
│   │   │   └── AllocationPieChart.jsx # Asset allocation chart
│   ├── pages/
│   │   └── Dashboard.jsx            # Main dashboard page
│   ├── hooks/
│   │   └── useWebSocket.js          # WebSocket connection hook
│   ├── App.jsx                      # Main app component
│   └── main.jsx                     # App entry point
├── public/
├── package.json
├── vite.config.js
└── index.html
```

## API Integration

The frontend communicates with the FastAPI backend through:

- **REST endpoints** for data fetching and control actions
- **WebSocket** for real-time updates (`/ws/updates`)

### Key API Endpoints Used

- `GET /api/dashboard/summary` - Dashboard metrics
- `GET /api/performance/history` - Performance chart data
- `GET /api/portfolio/allocation` - Allocation chart data
- `POST /api/trading/start|stop|emergency-stop` - Trading controls
- `POST /api/trading/mode` - Trading mode settings
- `POST /api/settings/risk` - Risk management settings

## Styling

Uses Material-UI with a custom dark theme matching the original Streamlit design:

- Dark background (`#0e1117`)
- Green accent color (`#00ff88`) for positive metrics
- Blue secondary color (`#4dabf7`)
- Consistent spacing and typography

## WebSocket Events

The frontend listens for these WebSocket events:

- `system_status` - Updates system running/stopped status
- `pnl_update` - Real-time P&L updates
- `new_trade` - Notification of new trades executed

## Development Notes

- Proxy configuration in `vite.config.js` routes API calls to the backend
- Hot reload enabled for fast development
- ESLint configured for code quality
- Responsive design works on desktop and mobile

## Deployment

Build the frontend and serve the `dist` folder:

```bash
npm run build
```

The built files can be served by any static file server or integrated with the FastAPI backend.