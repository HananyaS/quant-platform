"""
Technical indicators for quantitative analysis.

Provides common technical indicators used in trading strategies:
- Moving averages (SMA, EMA)
- RSI (Relative Strength Index)
- Bollinger Bands
- ATR (Average True Range)
- MACD
- Volatility measures
"""

import pandas as pd
import numpy as np
from typing import Tuple


class TechnicalIndicators:
    """
    Collection of technical indicators for financial time series.
    
    All methods are static and work on pandas Series or DataFrames.
    """
    
    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        """
        Simple Moving Average.
        
        Args:
            series: Price series
            window: Number of periods
            
        Returns:
            SMA series
        """
        return series.rolling(window=window).mean()
    
    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        """
        Exponential Moving Average.
        
        Args:
            series: Price series
            span: Number of periods for decay
            
        Returns:
            EMA series
        """
        return series.ewm(span=span, adjust=False).mean()
    
    @staticmethod
    def rsi(series: pd.Series, window: int = 14) -> pd.Series:
        """
        Relative Strength Index.
        
        Args:
            series: Price series
            window: Number of periods (default 14)
            
        Returns:
            RSI series (0-100)
        """
        delta = series.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def bollinger_bands(
        series: pd.Series,
        window: int = 20,
        num_std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands.
        
        Args:
            series: Price series
            window: Number of periods
            num_std: Number of standard deviations
            
        Returns:
            Tuple of (middle_band, upper_band, lower_band)
        """
        middle_band = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        
        return middle_band, upper_band, lower_band
    
    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14
    ) -> pd.Series:
        """
        Average True Range.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            window: Number of periods
            
        Returns:
            ATR series
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        return atr
    
    @staticmethod
    def macd(
        series: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence).
        
        Args:
            series: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        fast_ema = series.ewm(span=fast_period, adjust=False).mean()
        slow_ema = series.ewm(span=slow_period, adjust=False).mean()
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def volatility(
        series: pd.Series,
        window: int = 20,
        annualize: bool = True,
        trading_periods: int = 252
    ) -> pd.Series:
        """
        Rolling volatility (standard deviation of returns).
        
        Args:
            series: Price series
            window: Number of periods
            annualize: Whether to annualize the volatility
            trading_periods: Number of trading periods per year
            
        Returns:
            Volatility series
        """
        returns = series.pct_change()
        vol = returns.rolling(window=window).std()
        
        if annualize:
            vol = vol * np.sqrt(trading_periods)
        
        return vol
    
    @staticmethod
    def stochastic_oscillator(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            window: Lookback period
            smooth_k: %K smoothing period
            smooth_d: %D smoothing period
            
        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=window).min()
        highest_high = high.rolling(window=window).max()
        
        k_fast = 100 * (close - lowest_low) / (highest_high - lowest_low)
        k = k_fast.rolling(window=smooth_k).mean()
        d = k.rolling(window=smooth_d).mean()
        
        return k, d
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume.
        
        Args:
            close: Close price series
            volume: Volume series
            
        Returns:
            OBV series
        """
        direction = np.sign(close.diff())
        obv = (direction * volume).cumsum()
        
        return obv
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all common indicators to a DataFrame.
        
        Assumes DataFrame has columns: Open, High, Low, Close, Volume
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with added indicator columns
        """
        df = df.copy()
        
        # Ensure lowercase column names
        df.columns = df.columns.str.lower()
        
        # Price-based indicators
        if 'close' in df.columns:
            df['sma_20'] = TechnicalIndicators.sma(df['close'], 20)
            df['sma_50'] = TechnicalIndicators.sma(df['close'], 50)
            df['sma_200'] = TechnicalIndicators.sma(df['close'], 200)
            df['ema_12'] = TechnicalIndicators.ema(df['close'], 12)
            df['ema_26'] = TechnicalIndicators.ema(df['close'], 26)
            df['rsi_14'] = TechnicalIndicators.rsi(df['close'], 14)
            
            # Bollinger Bands
            bb_mid, bb_upper, bb_lower = TechnicalIndicators.bollinger_bands(
                df['close'], 20, 2.0
            )
            df['bb_middle'] = bb_mid
            df['bb_upper'] = bb_upper
            df['bb_lower'] = bb_lower
            df['bb_width'] = (bb_upper - bb_lower) / bb_mid
            
            # MACD
            macd, signal, hist = TechnicalIndicators.macd(df['close'])
            df['macd'] = macd
            df['macd_signal'] = signal
            df['macd_histogram'] = hist
            
            # Volatility
            df['volatility_20'] = TechnicalIndicators.volatility(df['close'], 20)
        
        # OHLC-based indicators
        if all(col in df.columns for col in ['high', 'low', 'close']):
            df['atr_14'] = TechnicalIndicators.atr(
                df['high'], df['low'], df['close'], 14
            )
            
            k, d = TechnicalIndicators.stochastic_oscillator(
                df['high'], df['low'], df['close'], 14
            )
            df['stoch_k'] = k
            df['stoch_d'] = d
        
        # Volume-based indicators
        if 'volume' in df.columns and 'close' in df.columns:
            df['obv'] = TechnicalIndicators.obv(df['close'], df['volume'])
        
        return df

