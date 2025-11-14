"""
Logging utilities for the trading framework.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    verbose: bool = True
) -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Logger name
        log_file: Optional file path for logging
        level: Logging level
        verbose: Whether to log to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if verbose:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_default_log_file(name: str) -> str:
    """
    Get default log file path.
    
    Args:
        name: Logger name
        
    Returns:
        Log file path
    """
    timestamp = datetime.now().strftime("%Y%m%d")
    return f"logs/{name}_{timestamp}.log"

