"""
Paper trading implementation for simulated live trading.

Simulates a broker API without real money.
"""

from typing import Optional, List, Dict
import pandas as pd
import uuid
from datetime import datetime

from quant_framework.execution.base_broker import (
    BaseBrokerAPI,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    Position,
    Account
)


class PaperTrader(BaseBrokerAPI):
    """
    Paper trading implementation for testing strategies.
    
    Simulates order execution without real capital.
    
    Example:
        trader = PaperTrader(initial_capital=100000)
        trader.connect()
        order = trader.place_order("AAPL", OrderSide.BUY, 10, OrderType.MARKET)
    """
    
    def __init__(
        self,
        api_key: str = "paper",
        api_secret: str = "paper",
        initial_capital: float = 100000.0
    ):
        """
        Initialize paper trader.
        
        Args:
            api_key: Dummy API key
            api_secret: Dummy API secret
            initial_capital: Starting capital
        """
        super().__init__(api_key, api_secret, paper_trading=True)
        
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.price_cache: Dict[str, float] = {}
    
    def connect(self) -> bool:
        """Connect to paper trading."""
        self.connected = True
        print("Connected to Paper Trading")
        return True
    
    def disconnect(self) -> None:
        """Disconnect from paper trading."""
        self.connected = False
        print("Disconnected from Paper Trading")
    
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Order:
        """Place a paper trading order."""
        if not self.connected:
            raise RuntimeError("Not connected to paper trading")
        
        # Generate order ID
        order_id = str(uuid.uuid4())
        
        # Create order
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            stop_price=stop_price,
            order_id=order_id,
            status=OrderStatus.PENDING,
            timestamp=pd.Timestamp.now()
        )
        
        # For market orders, execute immediately
        if order_type == OrderType.MARKET:
            self._execute_order(order)
        
        self.orders[order_id] = order
        
        return order
    
    def _execute_order(self, order: Order) -> None:
        """Simulate order execution."""
        # Get current price (or use cached price)
        execution_price = self.get_market_price(order.symbol)
        
        # Calculate cost
        cost = execution_price * order.quantity
        
        # Check if we have enough cash
        if order.side == OrderSide.BUY and cost > self.cash:
            order.status = OrderStatus.REJECTED
            return
        
        # Update position
        if order.symbol not in self.positions:
            self.positions[order.symbol] = Position(
                symbol=order.symbol,
                quantity=0,
                entry_price=0,
                current_price=execution_price,
                market_value=0,
                unrealized_pnl=0
            )
        
        position = self.positions[order.symbol]
        
        # Update position based on order side
        if order.side == OrderSide.BUY:
            # Calculate new average entry price
            total_quantity = position.quantity + order.quantity
            total_cost = (position.quantity * position.entry_price) + cost
            position.entry_price = total_cost / total_quantity if total_quantity > 0 else 0
            position.quantity = total_quantity
            self.cash -= cost
        else:  # SELL
            position.quantity -= order.quantity
            self.cash += cost
            
            # Calculate realized P&L
            pnl = (execution_price - position.entry_price) * order.quantity
            position.realized_pnl += pnl
        
        # Update position market value
        position.current_price = execution_price
        position.market_value = position.quantity * execution_price
        position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_fill_price = execution_price
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status in [OrderStatus.PENDING, OrderStatus.OPEN]:
                order.status = OrderStatus.CANCELLED
                return True
        return False
    
    def get_order_status(self, order_id: str) -> Order:
        """Get order status."""
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")
        return self.orders[order_id]
    
    def get_positions(self) -> List[Position]:
        """Get all positions."""
        # Update current prices
        for symbol, position in self.positions.items():
            if position.quantity != 0:
                position.current_price = self.get_market_price(symbol)
                position.market_value = position.quantity * position.current_price
                position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
        
        # Return only non-zero positions
        return [p for p in self.positions.values() if p.quantity != 0]
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        if symbol in self.positions and self.positions[symbol].quantity != 0:
            position = self.positions[symbol]
            position.current_price = self.get_market_price(symbol)
            position.market_value = position.quantity * position.current_price
            position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
            return position
        return None
    
    def get_balance(self) -> Account:
        """Get account balance."""
        positions_value = sum(p.market_value for p in self.get_positions())
        equity = self.cash + positions_value
        
        return Account(
            balance=self.initial_capital,
            equity=equity,
            buying_power=self.cash,
            cash=self.cash,
            positions_value=positions_value
        )
    
    def get_market_price(self, symbol: str) -> float:
        """
        Get current market price.
        
        In paper trading, this uses cached prices or returns a default.
        In a real implementation, you would fetch from a data provider.
        """
        if symbol in self.price_cache:
            return self.price_cache[symbol]
        
        # Default price if not cached
        # In real use, you would fetch from Yahoo Finance or another source
        return 100.0
    
    def set_market_price(self, symbol: str, price: float) -> None:
        """
        Set market price for a symbol (for testing).
        
        Args:
            symbol: Trading symbol
            price: Current price
        """
        self.price_cache[symbol] = price

