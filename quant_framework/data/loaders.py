"""
Data loaders for various sources (CSV, Yahoo Finance, APIs).

This module provides flexible data loading capabilities with standardized
preprocessing and cleaning methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
import pandas as pd
import numpy as np
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data loading."""
    symbol: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    columns: Optional[List[str]] = None
    dropna: bool = True
    remove_duplicates: bool = True


class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders.
    
    All data loaders should inherit from this class and implement
    the load() method.
    """
    
    def __init__(self, config: Optional[DataConfig] = None):
        """
        Initialize the data loader.
        
        Args:
            config: DataConfig object with loading parameters
        """
        self.config = config
        self.data: Optional[pd.DataFrame] = None
    
    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load data from the source.
        
        Returns:
            DataFrame with datetime index
        """
        pass
    
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the loaded data.
        
        Args:
            data: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df = data.copy()
        
        # Remove duplicates
        if self.config and self.config.remove_duplicates:
            df = df[~df.index.duplicated(keep='first')]
        
        # Handle missing values
        if self.config and self.config.dropna:
            df = df.dropna()
        
        # Sort by index
        df = df.sort_index()
        
        return df
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data (normalization, feature engineering, etc.).
        
        Args:
            data: Cleaned DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        df = data.copy()
        
        # Ensure proper column names
        df.columns = df.columns.str.lower()
        
        # Calculate basic returns if OHLC data is present
        if 'close' in df.columns:
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        return df
    
    def get_data(self) -> pd.DataFrame:
        """
        Complete pipeline: load -> clean -> preprocess.
        
        Returns:
            Processed DataFrame ready for analysis
        """
        if self.data is None:
            raw_data = self.load()
            cleaned_data = self.clean(raw_data)
            self.data = self.preprocess(cleaned_data)
        
        return self.data


class CSVDataLoader(BaseDataLoader):
    """
    Load data from CSV files.
    
    Example:
        loader = CSVDataLoader("data/SPY.csv", date_column='Date')
        data = loader.get_data()
    """
    
    def __init__(
        self,
        filepath: str,
        date_column: str = 'Date',
        config: Optional[DataConfig] = None
    ):
        """
        Initialize CSV data loader.
        
        Args:
            filepath: Path to CSV file
            date_column: Name of the date column
            config: Optional DataConfig object
        """
        super().__init__(config)
        self.filepath = Path(filepath)
        self.date_column = date_column
    
    def load(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Returns:
            DataFrame with datetime index
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {self.filepath}")
        
        df = pd.read_csv(
            self.filepath,
            parse_dates=[self.date_column],
            index_col=self.date_column
        )
        
        # Filter by date range if specified
        if self.config:
            if self.config.start_date:
                df = df[df.index >= self.config.start_date]
            if self.config.end_date:
                df = df[df.index <= self.config.end_date]
            
            # Select specific columns if specified
            if self.config.columns:
                df = df[self.config.columns]
        
        return df


class YahooDataLoader(BaseDataLoader):
    """
    Load data from Yahoo Finance with caching and retry logic.
    
    Features:
    - Automatic caching to avoid rate limits
    - Retry logic with exponential backoff
    - Fallback to cached data on error
    
    Example:
        loader = YahooDataLoader("AAPL", start="2020-01-01", end="2024-01-01")
        data = loader.get_data()
    """
    
    def __init__(
        self,
        symbol: str,
        start: str = "2020-01-01",
        end: Optional[str] = None,
        interval: str = "1d",
        use_cache: bool = True,
        max_retries: int = 3
    ):
        """
        Initialize Yahoo Finance data loader.
        
        Args:
            symbol: Stock ticker symbol
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD), defaults to today
            interval: Data interval (1d, 1h, etc.)
            use_cache: Whether to use cached data
            max_retries: Maximum number of retry attempts
        """
        config = DataConfig(symbol=symbol, start_date=start, end_date=end)
        super().__init__(config)
        self.symbol = symbol
        self.start = start
        self.end = end if end else pd.Timestamp.now().strftime('%Y-%m-%d')
        self.interval = interval
        self.use_cache = use_cache
        self.max_retries = max_retries
    
    def _get_cache_key(self) -> str:
        """Generate cache key for this data request."""
        return f"{self.symbol}_{self.start}_{self.end}_{self.interval}"
    
    def load(self) -> pd.DataFrame:
        """
        Load data from Yahoo Finance with caching and retry logic.
        
        Returns:
            DataFrame with datetime index and OHLCV data
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance is required for YahooDataLoader. "
                "Install it with: pip install yfinance"
            )
        
        # Try to load from cache first
        if self.use_cache:
            from quant_framework.utils.data_cache import get_cache
            cache = get_cache()
            cache_key = self._get_cache_key()
            
            cached_data = cache.get(cache_key, max_age_hours=24)
            if cached_data is not None:
                print(f"✓ Loading {self.symbol} from cache")
                return cached_data
        
        # Try to fetch from Yahoo Finance with retries
        import time
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                print(f"Fetching {self.symbol} from Yahoo Finance (attempt {attempt + 1}/{self.max_retries})...")
                
                ticker = yf.Ticker(self.symbol)
                df = ticker.history(
                    start=self.start,
                    end=self.end,
                    interval=self.interval
                )
                
                if df.empty:
                    raise ValueError(
                        f"No data retrieved for {self.symbol} "
                        f"from {self.start} to {self.end}"
                    )
                
                # Cache the data
                if self.use_cache:
                    cache.set(cache_key, df)
                    print(f"✓ Cached data for {self.symbol}")
                
                return df
                
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                
                # Check if it's a rate limit error
                if 'rate limit' in error_msg or 'too many requests' in error_msg or '429' in error_msg:
                    wait_time = (2 ** attempt) * 5  # Exponential backoff: 5, 10, 20 seconds
                    print(f"⚠ Rate limited. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    # For other errors, shorter wait
                    if attempt < self.max_retries - 1:
                        print(f"⚠ Error: {e}. Retrying in 2 seconds...")
                        time.sleep(2)
        
        # All retries failed - try to use stale cache as last resort
        if self.use_cache:
            print(f"⚠ All retries failed. Checking for stale cache...")
            cached_data = cache.get(cache_key, max_age_hours=None)  # Accept any age
            if cached_data is not None:
                print(f"✓ Using stale cached data for {self.symbol}")
                return cached_data
        
        # No cache available, raise the error
        raise RuntimeError(
            f"Failed to fetch data for {self.symbol} after {self.max_retries} attempts. "
            f"Last error: {last_error}\n\n"
            f"💡 Solutions:\n"
            f"1. Wait a few minutes and try again (rate limit usually clears)\n"
            f"2. Use CSV data instead: CSVDataLoader('data/{self.symbol}.csv')\n"
            f"3. Download data manually from Yahoo Finance\n"
            f"4. Use cached data if available (will be used automatically)"
        )


class APIDataLoader(BaseDataLoader):
    """
    Generic API data loader (placeholder for future implementations).
    
    This can be extended for specific APIs like:
    - Alpha Vantage
    - Quandl
    - Polygon.io
    - IEX Cloud
    """
    
    def __init__(
        self,
        api_key: str,
        endpoint: str,
        config: Optional[DataConfig] = None
    ):
        """
        Initialize API data loader.
        
        Args:
            api_key: API authentication key
            endpoint: API endpoint URL
            config: Optional DataConfig object
        """
        super().__init__(config)
        self.api_key = api_key
        self.endpoint = endpoint
    
    def load(self) -> pd.DataFrame:
        """
        Load data from API.
        
        Returns:
            DataFrame with datetime index
        """
        # Placeholder implementation
        raise NotImplementedError(
            "APIDataLoader is a placeholder. Implement specific API logic."
        )


class MultiTickerLoader:
    """
    Load data for multiple tickers simultaneously.
    
    Example:
        loader = MultiTickerLoader(
            symbols=['AAPL', 'GOOGL', 'MSFT'],
            loader_class=YahooDataLoader,
            start="2020-01-01"
        )
        data_dict = loader.load_all()
    """
    
    def __init__(
        self,
        symbols: List[str],
        loader_class: type,
        **loader_kwargs
    ):
        """
        Initialize multi-ticker loader.
        
        Args:
            symbols: List of ticker symbols
            loader_class: Data loader class to use
            **loader_kwargs: Arguments to pass to the loader
        """
        self.symbols = symbols
        self.loader_class = loader_class
        self.loader_kwargs = loader_kwargs
    
    def load_all(self) -> dict:
        """
        Load data for all symbols.
        
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        data_dict = {}
        
        for symbol in self.symbols:
            try:
                # Create loader instance for each symbol
                if self.loader_class == YahooDataLoader:
                    loader = self.loader_class(
                        symbol=symbol,
                        **self.loader_kwargs
                    )
                else:
                    loader = self.loader_class(**self.loader_kwargs)
                
                data_dict[symbol] = loader.get_data()
                
            except Exception as e:
                print(f"Error loading {symbol}: {e}")
                continue
        
        return data_dict

