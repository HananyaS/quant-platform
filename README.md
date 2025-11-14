# 📈 Quantitative Trading Platform

A comprehensive, professional-grade Python platform for algorithmic trading that supports strategy research, backtesting, machine learning, and deep learning.

## ⭐ Latest Updates (v2.0)

### 🆕 New Features
- **🌀 Fibonacci Retracement Strategy** - Trade bounces off key Fibonacci levels (23.6%, 38.2%, 50%, 61.8%, 78.6%)
- **🤖 Enhanced ML Models** - Full control over features, labels, train/test splits, and hyperparameters
- **🧠 PyTorch Deep Learning** - LSTM, GRU, Transformer, CNN, and MLP architectures with full configurability
- **📦 Modular Architecture** - Clean separation with `app_tabs/` module for better maintainability

### 🔧 Improvements
- **Default Parameters**: SPY ticker, $10,000 capital, 100% position sizing (retail-friendly defaults)
- **Enhanced Visualization**: Complete signal charts showing entries, exits, and position shading
- **Better Sample Retention**: Smart feature engineering preserves more data samples
- **Type Safety**: Fixed PyTorch dtype issues for reliable deep learning training

## 🎯 Features

### Core Platform
- **🏗️ Modular Architecture**: Clean separation with `app_tabs/` module system
- **📊 11 Built-in Strategies**: Momentum, Mean Reversion, RSI, MACD, Breakout, Turtle, Fibonacci, and more
- **💼 Realistic Backtesting**: Per-share fees, slippage, position sizing, and risk management
- **📈 Interactive Web UI**: Streamlit-based platform with 7 specialized tabs
- **🔄 Strategy Comparison**: Side-by-side analysis with normalized equity curves
- **📉 Complete Visualization**: Equity curves, drawdowns, trading signals with entries/exits

### Machine Learning
- **🤖 Classical ML**: Random Forest, XGBoost, LightGBM, SVM, Gradient Boosting
- **⚙️ Feature Engineering**: Technical, statistical, time-based, and lagged features
- **🎯 Label Definition**: Classification/regression with configurable thresholds
- **📏 Data Splits**: Holdout, time series CV, walk-forward validation
- **📊 Feature Importance**: Interactive visualizations of top features

### Deep Learning (PyTorch)
- **🧠 Architectures**: LSTM, GRU, Transformer, CNN, MLP
- **⚙️ Full Control**: Layers, hidden sizes, dropout, bidirectional, batch size
- **🎛️ Optimizers**: Adam, SGD, AdamW, RMSprop with configurable learning rates
- **📈 Real-time Training**: Live loss/accuracy plots with early stopping
- **🔧 Regularization**: Weight decay, dropout, batch normalization

### Data & Indicators
- **📡 Data Sources**: Yahoo Finance, CSV files, extensible API loaders
- **📊 Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic, and more
- **🎯 Performance Metrics**: Sharpe, Sortino, Calmar ratios, drawdowns, win rates, profit factor

## 📁 Project Structure

```
quant/
├── quant_framework/            # Core trading framework
│   ├── data/                   # Data loading & preprocessing
│   │   ├── loaders.py          # CSV, Yahoo, API data loaders
│   │   └── indicators.py       # 15+ technical indicators
│   ├── models/                 # 11 Built-in strategies
│   │   ├── base_strategy.py    # Abstract strategy interface
│   │   ├── momentum.py         # Moving average crossover
│   │   ├── mean_reversion.py   # Bollinger Bands strategy
│   │   ├── fibonacci_strategy.py  # 🆕 Fibonacci retracement
│   │   ├── rsi_strategy.py     # RSI oversold/overbought
│   │   ├── macd_strategy.py    # MACD crossover
│   │   ├── breakout_strategy.py # Donchian channel breakouts
│   │   ├── turtle_strategy.py  # Famous Turtle Trading
│   │   ├── stochastic_strategy.py  # Stochastic oscillator
│   │   ├── triple_ma_strategy.py   # Triple MA convergence
│   │   ├── pairs_trading.py    # Statistical arbitrage
│   │   └── buy_hold.py         # Baseline comparison
│   ├── ml/                     # 🆕 Machine Learning
│   │   ├── features.py         # Feature engineering
│   │   ├── classifiers.py      # RF, XGBoost, LightGBM, SVM, GB
│   │   ├── deep_models.py      # 🆕 PyTorch: LSTM, GRU, Transformer, CNN, MLP
│   │   ├── preprocessing.py    # Data scaling & transformation
│   │   └── trainer.py          # Model training & evaluation
│   ├── backtest/               # Portfolio simulation
│   │   ├── backtester.py       # Realistic backtesting engine
│   │   └── metrics.py          # 20+ performance metrics
│   ├── infra/                  # Orchestration
│   │   └── pipeline.py         # Trading pipeline
│   ├── research/               # Research tools
│   │   ├── optimizer.py        # Hyperparameter optimization
│   │   ├── portfolio.py        # Portfolio construction
│   │   └── feature_analysis.py # Feature importance
│   ├── execution/              # Live trading (future)
│   │   ├── base_broker.py      # Broker API interface
│   │   ├── paper_trader.py     # Paper trading simulator
│   │   ├── alpaca_broker.py    # Alpaca integration
│   │   └── interactive_brokers.py  # IBKR integration
│   ├── utils/                  # Utilities
│   │   ├── logger.py           # Logging setup
│   │   ├── plotting.py         # Matplotlib charts
│   │   ├── performance_report.py  # Report generation
│   │   └── config_loader.py    # Config management
│   ├── configs/                # Strategy YAML configs
│   └── tests/                  # Unit tests
│
├── app_tabs/                   # 🆕 Modular Streamlit UI
│   ├── __init__.py             # Tab exports
│   ├── helpers.py              # Shared utilities
│   ├── strategy_backtest.py    # Strategy backtesting tab
│   ├── compare_strategies.py   # Strategy comparison tab
│   ├── ml_models.py            # ML models tab
│   ├── deep_learning.py        # PyTorch deep learning tab
│   ├── optimization.py         # Hyperparameter optimization
│   ├── portfolio.py            # Portfolio construction
│   └── custom_strategy.py      # Custom strategy builder
│
├── examples/                   # Example scripts
│   ├── simple_backtest.py
│   ├── multi_strategy_comparison.py
│   ├── custom_strategy.py
│   └── demo_notebook.ipynb
│
├── app.py                      # 🚀 Main Streamlit app
├── START_HERE.sh               # Quick launcher (Mac/Linux)
├── START_HERE.bat              # Quick launcher (Windows)
├── START_HERE.md               # Comprehensive guide
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd quant

# Install dependencies (includes PyTorch, XGBoost, LightGBM)
pip install -r requirements.txt
```

### 🌐 Launch the Platform

The easiest way to start:

**Windows:**
```bash
START_HERE.bat
```

**Mac/Linux:**
```bash
chmod +x START_HERE.sh
./START_HERE.sh
```

**Or manually:**
```bash
streamlit run app.py
```

The platform opens at `http://localhost:8501` with **7 specialized tabs**:

#### 📊 Tab 1: Strategy Backtest
- 11 built-in strategies (including new Fibonacci!)
- Configurable parameters for each strategy
- Date range, initial capital, position sizing
- Interactive charts: Equity curve, drawdowns, trading signals
- Comprehensive metrics: Sharpe, Sortino, Calmar, win rate

#### 🔄 Tab 2: Compare Strategies
- Run multiple strategies side-by-side
- Normalized equity curves for fair comparison
- Detailed metrics comparison table
- Export results to CSV/JSON

#### 🤖 Tab 3: ML Models
- **Feature Engineering**: Technical, statistical, time-based, lagged features
- **5 Classifiers**: Random Forest, XGBoost, LightGBM, SVM, Gradient Boosting
- **Label Definition**: Classification/regression with custom thresholds
- **Data Splits**: Holdout, time series CV, walk-forward validation
- **Feature Importance**: Interactive visualizations

#### 🧠 Tab 4: Deep Learning (PyTorch)
- **5 Architectures**: LSTM, GRU, Transformer, CNN, MLP
- **Full Control**: Layers, hidden sizes, dropout, bidirectional
- **Optimizer Options**: Adam, SGD, AdamW, RMSprop
- **Real-time Training**: Live loss/accuracy plots
- **Backtesting**: Use trained model as trading strategy

#### 🎯 Tab 5: Optimization (Coming Soon)
- Hyperparameter grid/random search
- Bayesian optimization
- Walk-forward analysis

#### 💼 Tab 6: Portfolio (Coming Soon)
- Multi-asset portfolio construction
- Risk parity, mean-variance optimization
- Efficient frontier visualization

#### 🔧 Tab 7: Custom Strategy
- Build your own strategy with code editor
- Test immediately with full backtesting
- Save and reuse custom strategies

See [START_HERE.md](START_HERE.md) for comprehensive guide.

### 🎯 Default Settings

The platform comes with **retail-friendly defaults**:

- **Ticker**: SPY (S&P 500 ETF) - Liquid, low-spread, good for testing
- **Initial Capital**: $10,000 - Typical retail account size
- **Position Sizing**: 100% of portfolio - Maximum allocation per trade
- **Date Range**: Last 2 years - Recent market data
- **Fees**: $0.005/share (typical retail broker)
- **Slippage**: 0.05% - Conservative estimate
- **Allow Short**: False (Long-only) - Recommended for most users

**Why these defaults?**
- SPY has tight spreads and high liquidity
- $10K is achievable starting capital
- 100% position sizing shows max strategy potential
- Long-only avoids short selling complexity and costs

You can easily change any parameter in the UI!

### Basic Usage (Programmatic)

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

## 🤖 Machine Learning & Deep Learning

### ML Feature Engineering

The platform includes a comprehensive feature engineering pipeline:

```python
from quant_framework.ml import FeatureEngineering

fe = FeatureEngineering(
    lookback_periods=[5, 10, 20],
    include_technical=True,
    include_statistical=True,
    include_time_features=True,
    include_lagged_returns=True
)

X, y = fe.create_features(data, label_type='classification', threshold=0.0)
```

**Feature Categories:**
- **Technical**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic
- **Statistical**: Returns, volatility, skewness, kurtosis, correlation
- **Time-based**: Day of week, month, quarter, week of year, month start/end
- **Lagged**: Previous N-period returns and volume changes

**Label Options:**
- **Classification**: Direction prediction (1=up, -1=down, 0=neutral)
- **Regression**: Next-day return prediction

### Classical ML Models

Choose from 5 powerful classifiers:

1. **Random Forest** - Ensemble of decision trees, robust to overfitting
2. **XGBoost** - Gradient boosting, excellent for financial data
3. **LightGBM** - Fast gradient boosting, handles large datasets
4. **SVM** - Support Vector Machines with RBF kernel
5. **Gradient Boosting** - scikit-learn's gradient boosting

**Train/Test Strategies:**
- Holdout split (e.g., 80/20)
- Time series cross-validation (N splits)
- Walk-forward validation (rolling window)

### Deep Learning with PyTorch

Build sophisticated neural networks with full control:

#### Architectures

1. **LSTM** (Long Short-Term Memory)
   ```python
   # Captures long-term dependencies in price sequences
   - Bidirectional support
   - Multiple layers
   - Dropout regularization
   ```

2. **GRU** (Gated Recurrent Unit)
   ```python
   # Simpler than LSTM, often trains faster
   - Fewer parameters than LSTM
   - Good for shorter sequences
   ```

3. **Transformer** (Attention-based)
   ```python
   # State-of-the-art for sequence modeling
   - Multi-head attention
   - Positional encoding
   - Parallel processing
   ```

4. **CNN** (Convolutional Neural Network)
   ```python
   # Learns local patterns in price data
   - 1D convolutions for time series
   - Max pooling for feature reduction
   ```

5. **MLP** (Multi-Layer Perceptron)
   ```python
   # Simple feedforward network
   - Fast training
   - Good baseline model
   ```

**Training Features:**
- Real-time loss/accuracy visualization
- Early stopping to prevent overfitting
- Learning rate scheduling
- Batch normalization
- Weight decay (L2 regularization)
- Configurable optimizers (Adam, SGD, AdamW, RMSprop)

**Integration:**
- Trained models become trading strategies
- Full backtesting support
- Compare ML/DL strategies against traditional strategies

## 📊 Built-in Strategies

### All 11 Strategies

1. **📊 Buy & Hold** - Baseline benchmark
   - Buy at start, hold until end
   - Compare against to measure alpha

2. **📈 Momentum Strategy** - Moving average crossover
   - Golden cross / Death cross signals
   - Configurable SMA/EMA periods (default: 20/50)

3. **📉 Mean Reversion** - Bollinger Bands
   - Entry on band touches (2 std dev)
   - Exit on mean reversion
   - Works best in ranging markets

4. **🎯 RSI Strategy** - Oversold/overbought
   - Buy when RSI < 30 (oversold)
   - Sell when RSI > 70 (overbought)
   - Period: 14 days (configurable)

5. **⚡ MACD Strategy** - Trend following
   - MACD line crosses signal line
   - Histogram confirmation
   - Fast: 12, Slow: 26, Signal: 9

6. **💥 Breakout Strategy** - Donchian channels
   - Buy on N-day high breakout
   - Sell on N-day low breakout
   - Classic trend following (default: 20 days)

7. **🐢 Turtle Strategy** - Famous Turtle Trading
   - 20-day breakout entry
   - 10-day breakout exit
   - ATR-based position sizing
   - Includes stop-loss management

8. **📊 Stochastic Strategy** - Oscillator-based
   - %K and %D crossovers
   - Oversold (<20) and overbought (>80)
   - Fast stochastic: 14/3/3

9. **🔀 Triple MA Strategy** - Multiple timeframe convergence
   - Short (10), Medium (20), Long (50) MAs
   - All aligned = strong signal
   - Reduces false signals

10. **🌀 Fibonacci Strategy** - 🆕 Retracement levels
    - Identifies swings and key levels
    - Entry at 61.8% (golden ratio) retracement
    - Tolerance bands for flexibility
    - Long/short based on bounce direction

11. **⚖️ Pairs Trading** - Statistical arbitrage
    - Cointegration-based spread trading
    - Z-score entry/exit signals
    - Market-neutral strategy

### Creating Custom Strategies

All strategies inherit from `BaseStrategy`:

```python
from quant_framework.models import BaseStrategy
import pandas as pd
import numpy as np

class MyCustomStrategy(BaseStrategy):
    """
    Custom strategy template
    """
    def __init__(self, param1: int = 20, param2: float = 0.02, allow_short: bool = False):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        self.allow_short = allow_short
        self.position = 0  # Track current position
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with signals: +1 (long), -1 (short), 0 (neutral/exit)
        """
        signals = pd.Series(0, index=data.index)
        
        # Your strategy logic here
        # Example: Simple moving average crossover
        short_ma = data['close'].rolling(window=self.param1).mean()
        long_ma = data['close'].rolling(window=self.param1*2).mean()
        
        for i in range(1, len(data)):
            if self.position == 0:  # No position
                if short_ma.iloc[i] > long_ma.iloc[i]:
                    signals.iloc[i] = 1  # Buy signal
                    self.position = 1
                elif self.allow_short and short_ma.iloc[i] < long_ma.iloc[i]:
                    signals.iloc[i] = -1  # Short signal
                    self.position = -1
            
            elif self.position == 1:  # Long position
                if short_ma.iloc[i] < long_ma.iloc[i]:
                    signals.iloc[i] = 0  # Exit long
                    self.position = 0
            
            elif self.position == -1:  # Short position
                if short_ma.iloc[i] > long_ma.iloc[i]:
                    signals.iloc[i] = 0  # Exit short
                    self.position = 0
        
        return signals
```

**Using the Fibonacci Strategy:**

```python
from quant_framework.models import FibonacciStrategy

# Create Fibonacci retracement strategy
strategy = FibonacciStrategy(
    lookback_period=20,      # Period to identify swing high/low
    retracement_level=0.618, # Golden ratio (61.8%)
    tolerance=0.02,          # 2% tolerance band
    allow_short=False        # Long-only mode
)

# Use in backtesting
signals = strategy.generate_signals(data)
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

Available indicators via `TechnicalIndicators` class (`quant_framework/data/indicators.py`):

### Trend Indicators
- **SMA** - Simple Moving Average
- **EMA** - Exponential Moving Average
- **MACD** - Moving Average Convergence Divergence
- **ADX** - Average Directional Index

### Momentum Indicators
- **RSI** - Relative Strength Index
- **Stochastic** - Stochastic Oscillator (%K, %D)
- **ROC** - Rate of Change
- **Williams %R** - Williams Percent Range

### Volatility Indicators
- **Bollinger Bands** - Price envelopes based on standard deviation
- **ATR** - Average True Range
- **Keltner Channels** - EMA-based channels
- **Historical Volatility** - Rolling standard deviation

### Volume Indicators
- **OBV** - On-Balance Volume
- **Volume MA** - Volume moving average
- **VWAP** - Volume Weighted Average Price (intraday)

### Support/Resistance
- **Pivot Points** - Classic pivot levels
- **Donchian Channels** - N-period high/low channels
- **Fibonacci Retracements** - 🆕 Key retracement levels (23.6%, 38.2%, 50%, 61.8%, 78.6%)

## 📚 Documentation

Each module includes comprehensive docstrings with:
- Function/class descriptions
- Parameter specifications
- Return value descriptions
- Usage examples

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:

- Additional trading strategies (e.g., Ichimoku, Heikin-Ashi, Elliott Wave)
- More data sources (Polygon.io, Alpha Vantage, IEX Cloud)
- Advanced portfolio optimization (Black-Litterman, risk parity)
- Risk management modules (VaR, CVaR, tail risk)
- More ML/DL architectures (Attention, Transformers with custom heads)
- Live trading broker integrations (complete Alpaca/IBKR implementations)
- Options strategies and Greeks calculations

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## 📚 Documentation

- **[START_HERE.md](START_HERE.md)** - Comprehensive quick start guide
- **[CRITICAL_BUGS_FIXED.md](CRITICAL_BUGS_FIXED.md)** - Major bug fixes
- **[ALLOW_SHORT_GUIDE.md](ALLOW_SHORT_GUIDE.md)** - Long-only vs long-short
- **[FEE_STRUCTURE_GUIDE.md](FEE_STRUCTURE_GUIDE.md)** - Fee calculation details
- **[ML_SAMPLE_SIZE_FIX.md](ML_SAMPLE_SIZE_FIX.md)** - Feature engineering improvements
- **[PYTORCH_DTYPE_FIX.md](PYTORCH_DTYPE_FIX.md)** - Deep learning type safety

## ⚠️ Disclaimer

**IMPORTANT**: This platform is for **educational and research purposes only**. 

- Past performance does NOT guarantee future results
- Backtesting results may not reflect real-world trading
- Always paper trade strategies extensively before using real capital
- Consult a financial advisor before making investment decisions
- The authors are NOT responsible for any financial losses

**USE AT YOUR OWN RISK.**

## 📄 License

MIT License - Free to use for personal and commercial projects.

## 🙏 Built With

### Core Libraries
- **pandas** & **numpy** - Data manipulation and numerical computing
- **yfinance** - Market data from Yahoo Finance
- **matplotlib** & **plotly** - Visualization

### Web Interface
- **Streamlit** - Interactive web application framework

### Machine Learning
- **scikit-learn** - Classical ML algorithms and preprocessing
- **XGBoost** - Gradient boosting framework
- **LightGBM** - High-performance gradient boosting

### Deep Learning
- **PyTorch** - Neural network framework (LSTM, GRU, Transformer, CNN, MLP)
- **torchvision** - Computer vision utilities

### Testing & Quality
- **pytest** - Unit testing framework

## 📊 Performance Expectations

**Realistic expectations for algorithmic trading:**

- **Win Rate**: 40-60% is typical (not 80%+)
- **Sharpe Ratio**: >1.0 is good, >2.0 is excellent
- **Drawdown**: Expect 10-30% max drawdown even with good strategies
- **Market Regime**: Strategies perform differently in bull/bear/sideways markets
- **Overfitting**: High backtest returns often don't translate to live trading

**Tips for success:**
- Use walk-forward analysis, not just backtesting
- Test on multiple timeframes and symbols
- Include realistic transaction costs and slippage
- Consider market impact for larger trades
- Monitor correlation between strategies in portfolios

## 📞 Support

- **Issues**: Open an issue on GitHub for bugs
- **Features**: Request features via GitHub issues
- **Discussions**: Use GitHub Discussions for Q&A

---

**Happy Trading! 📈💰**

*Remember: The best strategy is proper risk management!*

