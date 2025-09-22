"""
Advanced Trading Framework - Market Regime Filter

This module analyzes market conditions to determine the current regime
(Trending, Ranging, or Volatile) and provides filtering logic for trading decisions.

Author: Senior Python Software Architect
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    yf = None
    YFINANCE_AVAILABLE = False


class MarketRegime(Enum):
    """Market regime classifications."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class MarketRegimeFilter:
    """
    Market regime filter that analyzes broader market conditions
    to determine if current conditions are favorable for trend-following strategies.
    
    Uses S&P 500 (SPY) data to calculate:
    - ADX for trend strength
    - Volatility measures
    - Moving average relationships
    """
    
    def __init__(self, adx_threshold: float = 25.0, volatility_percentile: float = 75.0):
        """
        Initialize market regime filter.
        
        Args:
            adx_threshold: ADX value above which market is considered trending
            volatility_percentile: Percentile threshold for volatility classification
        """
        self.adx_threshold = adx_threshold
        self.volatility_percentile = volatility_percentile
        self.logger = logging.getLogger(__name__)
        
        # Cache for market data
        self.market_data: Optional[pd.DataFrame] = None
        self.last_update: Optional[datetime] = None
        self.regime_history: Dict[datetime, MarketRegime] = {}
        
        # Historical volatility baseline
        self.volatility_baseline: Optional[float] = None
        
    def update_market_data(self, force_refresh: bool = False) -> bool:
        """
        Update market data for regime analysis.
        
        Args:
            force_refresh: Force refresh of market data
            
        Returns:
            True if data updated successfully
        """
        if not YFINANCE_AVAILABLE:
            self.logger.warning("yfinance not available for market regime analysis")
            return False
            
        # Check if refresh needed
        now = datetime.now()
        if (not force_refresh and 
            self.last_update and 
            (now - self.last_update).seconds < 3600):  # Update every hour
            return True
            
        try:
            # Fetch 6 months of SPY data for analysis
            end_date = now
            start_date = end_date - timedelta(days=180)
            
            self.logger.info(f"Fetching SPY market data from {start_date.date()} to {end_date.date()}")
            spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False)
            
            if spy_data.empty:
                self.logger.error("Failed to fetch SPY data")
                return False
                
            # Calculate technical indicators
            self.market_data = self._calculate_market_indicators(spy_data)
            self.last_update = now
            
            # Update volatility baseline
            if self.volatility_baseline is None:
                self.volatility_baseline = self.market_data['volatility'].quantile(0.5)
            
            self.logger.info("Market data updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update market data: {e}")
            return False
    
    def _calculate_market_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market indicators for regime classification.
        
        Args:
            data: Raw SPY OHLCV data
            
        Returns:
            DataFrame with calculated indicators
        """
        df = data.copy()
        
        # Basic price data
        df['close'] = df['Close']
        df['high'] = df['High']
        df['low'] = df['Low']
        df['volume'] = df['Volume']
        
        # Returns
        df['returns'] = df['close'].pct_change()
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # Trend direction
        df['trend_direction'] = np.where(df['close'] > df['sma_20'], 1, -1)
        
        # ADX calculation
        df['adx'] = self._calculate_adx(df)
        
        # Volatility (20-day rolling std annualized)
        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)
        
        # Average True Range
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        df['atr_ratio'] = df['atr'] / df['close']
        
        # Bollinger Band width as volatility measure
        bb_period = 20
        bb_std = 2
        bb_middle = df['close'].rolling(bb_period).mean()
        bb_std_dev = df['close'].rolling(bb_period).std()
        df['bb_width'] = (bb_std_dev * bb_std * 2) / bb_middle
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).
        
        Args:
            df: DataFrame with high, low, close
            period: Period for ADX calculation
            
        Returns:
            ADX values series
        """
        # True Range
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Directional Movement
        plus_dm = (df['high'] - df['high'].shift(1)).clip(lower=0)
        minus_dm = (df['low'].shift(1) - df['low']).clip(lower=0)
        
        # Smooth TR and DM
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.ewm(span=period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period).mean() / atr)
        
        # Directional Index
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # ADX
        adx = dx.ewm(span=period).mean()
        
        return adx
    
    def get_current_regime(self) -> MarketRegime:
        """
        Determine the current market regime.
        
        Returns:
            Current market regime classification
        """
        if not self.update_market_data():
            return MarketRegime.UNKNOWN
            
        if self.market_data is None or self.market_data.empty:
            return MarketRegime.UNKNOWN
            
        # Get latest data point
        latest = self.market_data.iloc[-1]
        
        # Get recent volatility for comparison
        recent_volatility = latest['volatility']
        
        # Classify regime based on multiple factors
        adx = latest['adx']
        trend_direction = latest['trend_direction']
        bb_width = latest['bb_width']
        atr_ratio = latest['atr_ratio']
        
        # High ADX indicates trending market
        if adx > self.adx_threshold:
            if trend_direction > 0:
                regime = MarketRegime.TRENDING_UP
            else:
                regime = MarketRegime.TRENDING_DOWN
        else:
            # Low ADX - check for ranging vs volatile
            if (bb_width < 0.05 and atr_ratio < 0.02):  # Tight bands, low ATR
                regime = MarketRegime.RANGING
            else:
                regime = MarketRegime.VOLATILE
        
        # Store in history
        self.regime_history[datetime.now()] = regime
        
        return regime
    
    def is_regime_favorable_for_trading(self, regime: MarketRegime) -> bool:
        """
        Determine if the current regime is favorable for trend-following strategies.
        
        Args:
            regime: Current market regime
            
        Returns:
            True if favorable for trading, False otherwise
        """
        # Trend-following strategies work best in trending markets
        favorable_regimes = [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]
        
        return regime in favorable_regimes
    
    def get_regime_confidence(self, regime: MarketRegime) -> float:
        """
        Get confidence score for the regime classification.
        
        Args:
            regime: Market regime
            
        Returns:
            Confidence score between 0 and 1
        """
        if not self.market_data or self.market_data.empty:
            return 0.0
            
        latest = self.market_data.iloc[-1]
        adx = latest['adx']
        
        # Confidence based on ADX strength
        if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            confidence = min(adx / 50.0, 1.0)  # Scale ADX to 0-1
        elif regime == MarketRegime.RANGING:
            confidence = max(1.0 - (adx / self.adx_threshold), 0.0)
        else:  # VOLATILE
            confidence = 0.5  # Neutral confidence for volatile
            
        return confidence
    
    def get_regime_stats(self) -> Dict:
        """
        Get statistics about market regimes over time.
        
        Returns:
            Dictionary with regime statistics
        """
        if not self.regime_history:
            return {}
            
        regimes = list(self.regime_history.values())
        
        stats = {
            'total_observations': len(regimes),
            'regime_distribution': {},
            'most_common_regime': None,
            'favorable_percentage': 0.0
        }
        
        # Count regime occurrences
        for regime in MarketRegime:
            count = regimes.count(regime)
            stats['regime_distribution'][regime.value] = count
            
        # Find most common
        most_common = max(stats['regime_distribution'], key=stats['regime_distribution'].get)
        stats['most_common_regime'] = most_common
        
        # Calculate favorable percentage
        favorable_count = 0
        for regime in regimes:
            if self.is_regime_favorable_for_trading(regime):
                favorable_count += 1
                
        stats['favorable_percentage'] = favorable_count / len(regimes) if regimes else 0.0
        
        return stats


# Global instance for easy access
_market_regime_filter = None

def get_market_regime_filter() -> MarketRegimeFilter:
    """Get global market regime filter instance."""
    global _market_regime_filter
    if _market_regime_filter is None:
        _market_regime_filter = MarketRegimeFilter()
    return _market_regime_filter


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_regime_filter():
        """Test market regime filter functionality."""
        filter = MarketRegimeFilter()
        
        print("Testing Market Regime Filter...")
        
        # Update market data
        if filter.update_market_data(force_refresh=True):
            print("✓ Market data updated")
            
            # Get current regime
            regime = filter.get_current_regime()
            print(f"Current regime: {regime.value}")
            
            # Check if favorable
            favorable = filter.is_regime_favorable_for_trading(regime)
            print(f"Favorable for trading: {favorable}")
            
            # Get confidence
            confidence = filter.get_regime_confidence(regime)
            print(f"Regime confidence: {confidence:.2f}")
            
            # Get stats
            stats = filter.get_regime_stats()
            print(f"Regime stats: {stats}")
        else:
            print("✗ Failed to update market data")
    
    asyncio.run(test_regime_filter())