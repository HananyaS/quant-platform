"""
Data caching utilities to avoid repeated API calls and rate limiting.
"""

import pandas as pd
from pathlib import Path
import pickle
from typing import Optional
from datetime import datetime, timedelta


class DataCache:
    """
    Simple file-based cache for market data.
    
    Avoids hitting API rate limits by caching downloaded data.
    """
    
    def __init__(self, cache_dir: str = ".cache"):
        """
        Initialize data cache.
        
        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        safe_key = key.replace('/', '_').replace(':', '_')
        return self.cache_dir / f"{safe_key}.pkl"
    
    def get(self, key: str, max_age_hours: Optional[int] = 24) -> Optional[pd.DataFrame]:
        """
        Get data from cache if it exists and is not too old.
        
        Args:
            key: Cache key (e.g., "AAPL_2020-01-01_2024-01-01")
            max_age_hours: Maximum age of cached data in hours (None = no limit)
            
        Returns:
            Cached DataFrame or None if not found/expired
        """
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        # Check age
        if max_age_hours is not None:
            file_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
            if datetime.now() - file_time > timedelta(hours=max_age_hours):
                return None
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")
            return None
    
    def set(self, key: str, data: pd.DataFrame) -> None:
        """
        Store data in cache.
        
        Args:
            key: Cache key
            data: DataFrame to cache
        """
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Warning: Could not save to cache: {e}")
    
    def clear(self, key: Optional[str] = None) -> None:
        """
        Clear cache.
        
        Args:
            key: Specific key to clear (None = clear all)
        """
        if key:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()
        else:
            # Clear all cache files
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()


# Global cache instance
_global_cache = DataCache()


def get_cache() -> DataCache:
    """Get the global cache instance."""
    return _global_cache

