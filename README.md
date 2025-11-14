# Quantitative Trading Framework

A comprehensive, modular Python framework for algorithmic trading that supports research, backtesting, and future live trading execution.

## 🚨 Critical Fixes - READ THIS FIRST!

### Three Major Bugs Fixed:

1. **✅ Position Sizing Fixed** - Was using 100% of portfolio on every trade!  
   Now uses fixed dollar amount (default: $10k per trade) with max % cap.

2. **✅ Exit Logic Fixed** - Long-only mode wasn't exiting positions!  
   Now properly exits when signal goes to 0 (exits to cash, doesn't go short).

3. **✅ Re-entry Prevention Fixed** - Multiple buy signals kept adding to position!  
   Now only enters once per signal cycle, prevents over-leveraging.

📚 **See:** `CRITICAL_BUGS_FIXED.md` for complete details

### Allow Short Feature (Already Implemented):

- ✅ **Default:** `allow_short=False` (long-only) - **Recommended for 95% of users**
- ⚠️ **Advanced:** `allow_short=True` (long-short) - For experienced traders only

**Why this matters:** Strategies go to CASH (not short) on bearish signals, dramatically improving performance!

📚 **See:** `QUICK_FIX_SUMMARY.md` and `ALLOW_SHORT_GUIDE.md` for details.

## 🎯 Features

- **Modular Architecture**: Clean separation of data, models, backtesting, and execution layers
- **Multiple Data Sources**: CSV, Yahoo Finance, and extensible API loaders
- **Rich Technical Indicators**: SMA, EMA, RSI, Bollinger Bands, MACD, ATR, and more
- **10+ Pre-built Strategies**: Momentum, Mean Reversion, RSI, MACD, Breakout, Turtle, and more
- **Comprehensive Backtesting**: Realistic simulation with transaction costs, slippage, and leverage
- **Performance Metrics**: Sharpe, Sortino, Calmar ratios, drawdowns, win rates, and more
- **Interactive Web UI**: Streamlit-based interface for easy research and backtesting
- **Strategy Comparison**: Side-by-side comparison with trade logs and metrics
- **Custom Strategy Builder**: Code editor for building and testing your own strategies
- **Visualization**: Equity curves, drawdowns, signal plots, and performance dashboards
- **Config-Driven**: YAML/JSON configuration files for reproducible experiments
- **Extensible**: Easy to add custom strategies and data sources
- **Live Trading Ready**: Broker API stubs for Alpaca, Interactive Brokers, and paper trading

## 📁 Project Structure

```
quant_framework/
├── data/                   # Data loading & preprocessing
│   ├── loaders.py          # CSV, Yahoo, API data loaders
│   └── indicators.py       # Technical indicators
├── models/                 # Strategy definitions
│   ├── base_strategy.py    # Abstract strategy interface
│   ├── momentum.py         # Moving average crossover
│   ├── mean_reversion.py   # Bollinger Bands strategy
│   ├── pairs_trading.py    # Statistical arbitrage
│   └── ml_volatility.py    # ML-based volatility model
├── backtest/               # Portfolio simulation
│   ├── backtester.py       # Backtesting engine
│   └── metrics.py          # Performance metrics
├── infra/                  # Orchestration
│   └── pipeline.py         # Trading pipeline
├── execution/              # Live trading connectors
│   ├── base_broker.py      # Broker API interface
│   ├── paper_trader.py     # Paper trading simulator
│   ├── alpaca_broker.py    # Alpaca integration
│   └── interactive_brokers.py  # IBKR integration
├── utils/                  # Utilities
│   ├── logger.py           # Logging setup
│   ├── plotting.py         # Visualization tools
│   ├── performance_report.py  # Report generation
│   └── config_loader.py    # Config management
├── configs/                # Configuration files
├── tests/                  # Unit tests
└── __init__.py

examples/                   # Example scripts
├── simple_backtest.py
├── multi_strategy_comparison.py
└── custom_strategy.py

main.py                     # Main entry point
requirements.txt            # Dependencies
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd quant

# Install dependencies
pip install -r requirements.txt
```

### 🌐 Web UI (Easiest Way!)

Launch the interactive web interface:

**Windows:**
```bash
launch_web_app.bat
```

**Mac/Linux:**
```bash
chmod +x launch_web_app.sh
./launch_web_app.sh
```

**Or manually:**
```bash
streamlit run web_app.py
```

The web app opens at `http://localhost:8501` with:
- 📊 10+ Built-in strategies
- 📅 Easy date/parameter configuration
- 📈 Interactive charts and metrics
- 🔄 **NEW:** Strategy comparison with allow_short control
- 💼 **NEW:** Detailed trade logs in comparisons
- 🔧 **NEW:** Custom strategy builder with code editor
- ⚖️ Strategy comparison tool
- 🎯 One-click backtesting

See [WEB_APP_GUIDE.md](WEB_APP_GUIDE.md) for detailed instructions.

### Basic Usage

```python
from quant_framework.data import YahooDataLoader
from quant_framework.models import MomentumStrategy
from quant_framework.backtest import Backtester
from quant_framework.infra import TradingPipeline

# Load data
data_loader = YahooDataLoader("AAPL", start="2020-01-01", end="2024-01-01")

# Create strategy
strategy = MomentumStrategy(short_window=20, long_window=50)

# Setup backtester
backtester = Backtester(initial_capital=100000, fee_perc=0.001)

# Run pipeline
pipeline = TradingPipeline(data_loader, strategy, backtester)
results = pipeline.run()
```

### Run Examples

```bash
# Run default momentum example
python main.py

# Run specific example
python main.py --example momentum
python main.py --example mean_reversion
python main.py --example comparison

# Run from config file
python main.py --config configs/example_momentum.yaml

# Run example scripts
python examples/simple_backtest.py
python examples/multi_strategy_comparison.py
python examples/custom_strategy.py
```

## 📊 Strategies

### Built-in Strategies

1. **Momentum Strategy** - Moving average crossover
   - Golden cross / Death cross signals
   - Configurable SMA or EMA periods

2. **Mean Reversion Strategy** - Bollinger Bands
   - Entry on band touches
   - Exit on mean reversion

3. **Pairs Trading Strategy** - Statistical arbitrage
   - Cointegration-based spread trading
   - Z-score entry/exit signals

4. **ML Volatility Model** - Machine learning template
   - Feature engineering pipeline
   - Volatility regime detection

### Creating Custom Strategies

```python
from quant_framework.models import BaseStrategy
import pandas as pd

class MyStrategy(BaseStrategy):
    def __init__(self, param1, param2):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Your strategy logic here
        signals = pd.Series(0, index=data.index)
        # +1 for long, -1 for short, 0 for neutral
        return signals
```

## 📈 Performance Metrics

The framework calculates comprehensive performance metrics:

- **Return Metrics**: Total return, annual return, volatility
- **Risk-Adjusted**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Drawdown**: Maximum drawdown, drawdown duration
- **Trading**: Win rate, profit factor, number of trades
- **Risk**: Value at Risk (VaR), Conditional VaR (CVaR)

## 🎨 Visualization

Built-in plotting functions:
- Equity curve with fill
- Drawdown underwater plot
- Trading signals overlaid on price
- Returns distribution
- Rolling Sharpe ratio
- Strategy comparison

## ⚙️ Configuration

Use YAML or JSON config files for reproducible experiments:

```yaml
strategy:
  name: "MomentumStrategy"
  type: "momentum"
  parameters:
    short_window: 20
    long_window: 50

data:
  source: "yahoo"
  symbol: "AAPL"
  start_date: "2020-01-01"
  end_date: "2024-01-01"

backtest:
  initial_capital: 100000
  fee_perc: 0.001
  slippage_perc: 0.0005
```

## 🧪 Testing

Run unit tests:

```bash
# Run all tests
pytest quant_framework/tests/

# Run specific test file
pytest quant_framework/tests/test_strategies.py

# Run with coverage
pytest --cov=quant_framework quant_framework/tests/
```

## 📡 Live Trading (Future)

The framework includes broker API stubs for future live trading:

```python
from quant_framework.execution import PaperTrader, AlpacaBroker

# Paper trading
trader = PaperTrader(initial_capital=100000)
trader.connect()
order = trader.place_order("AAPL", OrderSide.BUY, 10, OrderType.MARKET)

# Alpaca (requires API keys)
broker = AlpacaBroker(api_key="YOUR_KEY", api_secret="YOUR_SECRET")
broker.connect()
```

## 🔧 Technical Indicators

Available indicators via `TechnicalIndicators` class:

- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Relative Strength Index (RSI)
- Bollinger Bands
- Average True Range (ATR)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator
- On-Balance Volume (OBV)
- Volatility measures

## 📚 Documentation

Each module includes comprehensive docstrings with:
- Function/class descriptions
- Parameter specifications
- Return value descriptions
- Usage examples

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:

- Additional trading strategies
- More data sources (APIs)
- Advanced portfolio optimization
- Risk management modules
- Machine learning models
- Live trading broker integrations

## ⚠️ Disclaimer

This framework is for educational and research purposes only. Past performance does not guarantee future results. Always test strategies thoroughly before risking real capital. The authors are not responsible for any financial losses incurred using this software.

## 📄 License

MIT License - feel free to use this framework for your own projects.

## 🙏 Acknowledgments

Built with:
- pandas & numpy for data manipulation
- matplotlib for visualization
- yfinance for market data
- pytest for testing

## 📞 Contact

For questions, issues, or feature requests, please open an issue on GitHub.

---

**Happy Trading! 📈💰**

