"""
Backtesting engine for trading strategies.

Simulates portfolio performance with realistic constraints:
- Transaction costs
- Slippage
- Position sizing
- Leverage limits
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from quant_framework.backtest.metrics import calc_all_metrics


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100000.0
    fee_perc: float = 0.001  # 0.1% per trade (if using percentage)
    fee_per_share: float = 0.0007  # $0.0007 per share (0.07 cents)
    fee_minimum: float = 2.5  # Minimum fee per trade
    use_per_share_fee: bool = True  # Use per-share fee instead of percentage
    slippage_perc: float = 0.0005  # 0.05% slippage
    leverage: float = 1.0  # No leverage by default
    position_size: float = 1.0  # Use full capital (if use_fixed_trade_value=False)
    use_fixed_trade_value: bool = False  # Use fixed dollar amount per trade
    fixed_trade_value: float = 10000.0  # Fixed dollar amount per trade
    max_position_pct: float = 0.95  # Maximum % of portfolio in single position
    risk_free_rate: float = 0.02  # 2% annual risk-free rate


class Backtester:
    """
    Backtesting engine for evaluating trading strategies.
    
    Simulates realistic trading with costs, slippage, and constraints.
    
    Example:
        bt = Backtester(initial_capital=100000, fee_perc=0.001)
        results = bt.run(data, signals)
        bt.print_summary()
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        fee_perc: float = 0.001,
        fee_per_share: float = 0.0007,
        fee_minimum: float = 2.5,
        use_per_share_fee: bool = True,
        slippage_perc: float = 0.0005,
        leverage: float = 1.0,
        position_size: float = 1.0,
        use_fixed_trade_value: bool = False,
        fixed_trade_value: float = 10000.0,
        max_position_pct: float = 0.95,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital
            fee_perc: Transaction fee as percentage (e.g., 0.001 = 0.1%)
            fee_per_share: Fee per share (e.g., 0.0007 = $0.0007 = 0.07 cents)
            fee_minimum: Minimum fee per trade (e.g., 2.5 = $2.50)
            use_per_share_fee: Use per-share fee structure (True) or percentage (False)
            slippage_perc: Slippage as percentage
            leverage: Maximum leverage allowed
            position_size: Fraction of capital to use per trade (0-1) if use_fixed_trade_value=False
            use_fixed_trade_value: Use fixed dollar amount per trade instead of percentage
            fixed_trade_value: Fixed dollar amount to use per trade
            max_position_pct: Maximum percentage of portfolio in single position (0-1)
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.config = BacktestConfig(
            initial_capital=initial_capital,
            fee_perc=fee_perc,
            fee_per_share=fee_per_share,
            fee_minimum=fee_minimum,
            use_per_share_fee=use_per_share_fee,
            slippage_perc=slippage_perc,
            leverage=leverage,
            position_size=position_size,
            use_fixed_trade_value=use_fixed_trade_value,
            fixed_trade_value=fixed_trade_value,
            max_position_pct=max_position_pct,
            risk_free_rate=risk_free_rate
        )
        
        self.results: Optional[Dict[str, Any]] = None
        self.equity_curve: Optional[pd.Series] = None
        self.trades: Optional[pd.DataFrame] = None
    
    def run(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        price_column: str = 'close'
    ) -> Dict[str, Any]:
        """
        Run backtest simulation.
        
        Args:
            data: DataFrame with price data
            signals: Series with position signals (+1, 0, -1)
            price_column: Name of price column to use
            
        Returns:
            Dictionary with backtest results and metrics
        """
        # Ensure data and signals are aligned
        df = data.copy()
        
        # Handle column name variations
        if price_column not in df.columns:
            if price_column.capitalize() in df.columns:
                price_column = price_column.capitalize()
            elif price_column.upper() in df.columns:
                price_column = price_column.upper()
            else:
                raise ValueError(
                    f"Price column '{price_column}' not found. "
                    f"Available columns: {df.columns.tolist()}"
                )
        
        prices = df[price_column]
        signals = signals.reindex(prices.index).fillna(0)
        
        # Initialize tracking variables
        capital = self.config.initial_capital
        position = 0  # Number of shares
        prev_signal = 0  # Track previous signal to detect changes
        equity = []
        cash = []
        position_values = []
        trade_log = []
        
        for i in range(len(prices)):
            timestamp = prices.index[i]
            price = prices.iloc[i]
            signal = signals.iloc[i]
            
            if pd.isna(price) or pd.isna(signal):
                equity.append(capital if i == 0 else equity[-1])
                cash.append(capital if i == 0 else cash[-1])
                position_values.append(0)
                continue
            
            # Calculate current position value
            position_value = position * price
            current_equity = capital + position_value
            
            # Determine if we should trade based on signal CHANGES
            # Rules:
            # 1. Only enter when signal changes (prevent multiple buys on same signal)
            # 2. Exit when signal goes to 0 (even in long-only mode)
            # 3. Use fixed trade value or percentage of portfolio
            
            should_trade = False
            target_shares = 0
            
            # Signal interpretation:
            # +1 = Want to be long
            # 0 = Want to be flat (exit all positions)
            # -1 = Want to be short
            
            if signal > 0:  # Long signal
                if position <= 0:  # Not currently long (flat or short)
                    should_trade = True
                    # Calculate trade size
                    if self.config.use_fixed_trade_value:
                        # Use fixed dollar amount, capped at max_position_pct of equity
                        trade_value = min(
                            self.config.fixed_trade_value,
                            current_equity * self.config.max_position_pct
                        )
                    else:
                        # Use percentage of equity
                        trade_value = current_equity * self.config.position_size * self.config.leverage
                        trade_value = min(trade_value, current_equity * self.config.max_position_pct)
                    
                    target_shares = trade_value / price if price > 0 else 0
                # else: already long, don't add to position
                else:
                    target_shares = position  # Keep current position
                    
            elif signal < 0:  # Short signal
                if position >= 0:  # Not currently short (flat or long)
                    should_trade = True
                    # Calculate trade size
                    if self.config.use_fixed_trade_value:
                        trade_value = min(
                            self.config.fixed_trade_value,
                            current_equity * self.config.max_position_pct
                        )
                    else:
                        trade_value = current_equity * self.config.position_size * self.config.leverage
                        trade_value = min(trade_value, current_equity * self.config.max_position_pct)
                    
                    target_shares = -trade_value / price if price > 0 else 0
                # else: already short, don't add to position
                else:
                    target_shares = position  # Keep current position
                    
            else:  # Signal == 0, exit all positions
                if abs(position) > 1e-6:  # Have a position to close
                    should_trade = True
                    target_shares = 0
            
            # Execute trade if needed
            shares_to_trade = target_shares - position
            
            if should_trade and abs(shares_to_trade) > 1e-6:  # Avoid tiny trades
                # Check if we have enough capital for the trade
                trade_cost = shares_to_trade * price
                
                # If buying, check we have enough cash
                if shares_to_trade > 0 and abs(trade_cost) > capital * 0.99:
                    # Not enough cash, reduce trade size
                    shares_to_trade = (capital * 0.98) / price
                    target_shares = position + shares_to_trade
                
                # Calculate trade value
                trade_value = abs(shares_to_trade * price)
                
                if trade_value > 0.01:  # Minimum $0.01 trade
                    # Apply slippage
                    slippage_cost = trade_value * self.config.slippage_perc
                    
                    # Apply transaction fees based on fee structure
                    if self.config.use_per_share_fee:
                        # Per-share fee with minimum
                        num_shares = abs(shares_to_trade)
                        per_share_cost = num_shares * self.config.fee_per_share
                        transaction_fee = max(per_share_cost, self.config.fee_minimum)
                    else:
                        # Percentage-based fee
                        transaction_fee = trade_value * self.config.fee_perc
                    
                    # Sanity check: fee shouldn't be more than trade value
                    if transaction_fee > trade_value * 0.5:
                        transaction_fee = trade_value * 0.01  # Cap at 1%
                    
                    # Update capital and position
                    capital -= shares_to_trade * price
                    capital -= (slippage_cost + transaction_fee)
                    position = target_shares
                    
                    # Check for negative capital (bankruptcy)
                    if capital < 0:
                        # Liquidate position and stop trading
                        capital += position * price  # Close position
                        position = 0
                        print(f"Warning: Bankruptcy at {timestamp}. Liquidating all positions.")
                    
                    # Log trade
                    trade_log.append({
                        'timestamp': timestamp,
                        'price': price,
                        'shares': shares_to_trade,
                        'position': position,
                        'value': trade_value,
                        'fees': transaction_fee,
                        'slippage': slippage_cost,
                        'signal': signal,
                        'capital': capital,
                        'pnl': 0  # Will calculate later
                    })
            
            # Update previous signal
            prev_signal = signal
            
            # Record equity
            position_value = position * price
            current_equity = capital + position_value
            
            equity.append(current_equity)
            cash.append(capital)
            position_values.append(position_value)
        
        # Create equity curve
        self.equity_curve = pd.Series(equity, index=prices.index)
        
        # Calculate returns
        returns = self.equity_curve.pct_change().fillna(0)
        
        # Calculate metrics
        metrics = calc_all_metrics(
            returns=returns,
            equity_curve=self.equity_curve,
            signals=signals,
            risk_free_rate=self.config.risk_free_rate,
            periods_per_year=252
        )
        
        # Store trades and calculate PnL
        if trade_log:
            self.trades = pd.DataFrame(trade_log)
            
            # Calculate PnL for each trade
            # PnL is calculated as the change in capital from previous trade
            if len(self.trades) > 1:
                prev_capital = self.config.initial_capital
                pnl_list = []
                for idx, row in self.trades.iterrows():
                    current_capital = row['capital'] + (row['position'] * row['price'])
                    pnl = current_capital - prev_capital
                    pnl_list.append(pnl)
                    prev_capital = current_capital
                self.trades['pnl'] = pnl_list
            elif len(self.trades) == 1:
                self.trades['pnl'] = self.trades['capital'] - self.config.initial_capital
        else:
            self.trades = pd.DataFrame()
        
        # Compile results
        self.results = {
            'equity_curve': self.equity_curve,
            'returns': returns,
            'metrics': metrics,
            'trades': self.trades,
            'final_equity': self.equity_curve.iloc[-1] if len(self.equity_curve) > 0 else 0,
            'total_return': metrics['total_return'],
            'config': self.config
        }
        
        return self.results
    
    def print_summary(self) -> None:
        """Print backtest summary statistics."""
        if self.results is None:
            print("No backtest results available. Run backtest first.")
            return
        
        metrics = self.results['metrics']
        
        print("=" * 60)
        print("BACKTEST SUMMARY")
        print("=" * 60)
        print(f"Initial Capital:    ${self.config.initial_capital:,.2f}")
        print(f"Final Equity:       ${self.results['final_equity']:,.2f}")
        print(f"Total Return:       {metrics['total_return']*100:.2f}%")
        print(f"Annual Return:      {metrics['annual_return']*100:.2f}%")
        print(f"Annual Volatility:  {metrics['annual_volatility']*100:.2f}%")
        print("-" * 60)
        print(f"Sharpe Ratio:       {metrics['sharpe_ratio']:.3f}")
        print(f"Sortino Ratio:      {metrics['sortino_ratio']:.3f}")
        print(f"Calmar Ratio:       {metrics['calmar_ratio']:.3f}")
        print(f"Max Drawdown:       {metrics['max_drawdown']*100:.2f}%")
        print("-" * 60)
        print(f"Win Rate:           {metrics['win_rate']*100:.2f}%")
        print(f"Profit Factor:      {metrics['profit_factor']:.3f}")
        
        if 'num_trades' in metrics:
            print(f"Number of Trades:   {metrics['num_trades']}")
            print(f"Turnover:           {metrics['turnover']:.3f}")
        
        print(f"VaR (95%):          {metrics['var_95']*100:.2f}%")
        print(f"CVaR (95%):         {metrics['cvar_95']*100:.2f}%")
        print("-" * 60)
        print(f"Fee Structure:      {'Per-Share' if self.config.use_per_share_fee else 'Percentage'}")
        if self.config.use_per_share_fee:
            print(f"  Per Share:        ${self.config.fee_per_share:.4f}")
            print(f"  Minimum:          ${self.config.fee_minimum:.2f}")
        else:
            print(f"  Fee Percent:      {self.config.fee_perc*100:.3f}%")
        print("=" * 60)
    
    def get_equity_curve(self) -> pd.Series:
        """
        Get the equity curve from the backtest.
        
        Returns:
            Series with portfolio values over time
        """
        if self.equity_curve is None:
            raise ValueError("No equity curve available. Run backtest first.")
        
        return self.equity_curve
    
    def get_trades(self) -> pd.DataFrame:
        """
        Get the trade log.
        
        Returns:
            DataFrame with all executed trades
        """
        if self.trades is None:
            raise ValueError("No trades available. Run backtest first.")
        
        return self.trades
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics.
        
        Returns:
            Dictionary with all calculated metrics
        """
        if self.results is None:
            raise ValueError("No results available. Run backtest first.")
        
        return self.results['metrics']

