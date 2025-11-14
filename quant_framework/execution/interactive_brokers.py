"""
Interactive Brokers API implementation.

Connector for Interactive Brokers trading platform.
"""

from typing import Optional, List
from quant_framework.execution.base_broker import (
    BaseBrokerAPI,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    Position,
    Account
)


class InteractiveBrokersAPI(BaseBrokerAPI):
    """
    Interactive Brokers API implementation.
    
    This is a stub/template. To use, you need to:
    1. Install ib_insync: pip install ib_insync
    2. Set up IB Gateway or TWS
    3. Implement the methods using IB API
    
    Example:
        broker = InteractiveBrokersAPI(
            host="127.0.0.1",
            port=7497,
            client_id=1
        )
        broker.connect()
    """
    
    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        paper_trading: bool = True
    ):
        """
        Initialize Interactive Brokers API.
        
        Args:
            api_key: Not used (IB uses different auth)
            api_secret: Not used (IB uses different auth)
            host: IB Gateway/TWS host
            port: IB Gateway/TWS port (7497 for paper, 7496 for live)
            client_id: Client ID for connection
            paper_trading: Use paper trading
        """
        super().__init__(api_key, api_secret, paper_trading)
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = None
    
    def connect(self) -> bool:
        """Connect to Interactive Brokers."""
        raise NotImplementedError(
            "InteractiveBrokersAPI.connect() needs to be implemented with ib_insync"
        )
    
    def disconnect(self) -> None:
        """Disconnect from Interactive Brokers."""
        raise NotImplementedError(
            "InteractiveBrokersAPI.disconnect() needs to be implemented"
        )
    
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Order:
        """Place an order through Interactive Brokers."""
        raise NotImplementedError(
            "InteractiveBrokersAPI.place_order() needs to be implemented"
        )
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        raise NotImplementedError(
            "InteractiveBrokersAPI.cancel_order() needs to be implemented"
        )
    
    def get_order_status(self, order_id: str) -> Order:
        """Get order status."""
        raise NotImplementedError(
            "InteractiveBrokersAPI.get_order_status() needs to be implemented"
        )
    
    def get_positions(self) -> List[Position]:
        """Get all positions."""
        raise NotImplementedError(
            "InteractiveBrokersAPI.get_positions() needs to be implemented"
        )
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        raise NotImplementedError(
            "InteractiveBrokersAPI.get_position() needs to be implemented"
        )
    
    def get_balance(self) -> Account:
        """Get account balance."""
        raise NotImplementedError(
            "InteractiveBrokersAPI.get_balance() needs to be implemented"
        )
    
    def get_market_price(self, symbol: str) -> float:
        """Get current market price."""
        raise NotImplementedError(
            "InteractiveBrokersAPI.get_market_price() needs to be implemented"
        )

