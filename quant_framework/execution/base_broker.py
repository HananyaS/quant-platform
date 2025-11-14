"""
Base broker API for live trading execution.

Abstract interface for all broker implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum
import pandas as pd


class OrderSide(Enum):
    """Order side enum."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enum."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status enum."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Represents a trading order."""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    timestamp: Optional[pd.Timestamp] = None


@dataclass
class Position:
    """Represents a current position."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float = 0.0


@dataclass
class Account:
    """Represents account information."""
    balance: float
    equity: float
    buying_power: float
    cash: float
    positions_value: float


class BaseBrokerAPI(ABC):
    """
    Abstract base class for broker API implementations.
    
    All broker connectors should inherit from this class and implement
    the abstract methods.
    """
    
    def __init__(self, api_key: str, api_secret: str, paper_trading: bool = True):
        """
        Initialize broker API.
        
        Args:
            api_key: API key for authentication
            api_secret: API secret for authentication
            paper_trading: Whether to use paper trading mode
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper_trading = paper_trading
        self.connected = False
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the broker API.
        
        Returns:
            True if connection successful
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the broker API."""
        pass
    
    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Order:
        """
        Place a trading order.
        
        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            quantity: Number of shares/contracts
            order_type: Type of order
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            
        Returns:
            Order object with order details
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            True if cancellation successful
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Order:
        """
        Get the status of an order.
        
        Args:
            order_id: ID of the order
            
        Returns:
            Order object with current status
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """
        Get all current positions.
        
        Returns:
            List of Position objects
        """
        pass
    
    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position object or None if no position
        """
        pass
    
    @abstractmethod
    def get_balance(self) -> Account:
        """
        Get account balance and equity.
        
        Returns:
            Account object with balance information
        """
        pass
    
    @abstractmethod
    def get_market_price(self, symbol: str) -> float:
        """
        Get current market price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current market price
        """
        pass
    
    def close_position(self, symbol: str) -> Optional[Order]:
        """
        Close an existing position.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Order object for the closing trade
        """
        position = self.get_position(symbol)
        
        if position is None or position.quantity == 0:
            return None
        
        # Determine side (sell if long, buy if short)
        side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
        quantity = abs(position.quantity)
        
        return self.place_order(symbol, side, quantity, OrderType.MARKET)
    
    def close_all_positions(self) -> List[Order]:
        """
        Close all open positions.
        
        Returns:
            List of Order objects for closing trades
        """
        positions = self.get_positions()
        orders = []
        
        for position in positions:
            order = self.close_position(position.symbol)
            if order:
                orders.append(order)
        
        return orders

