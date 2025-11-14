# 🚀 START HERE - Quick Launch Guide

Welcome to the Quantitative Trading Research Platform! This guide will get you up and running in minutes.

---

## 🎯 Fastest Start (One Command)

Simply run:

```bash
./START_HERE.sh
```

This script will:
1. Check/install dependencies
2. Launch the web app
3. Open your browser automatically

**First time?** You may need to make it executable:
```bash
chmod +x START_HERE.sh
./START_HERE.sh
```

---

## 📋 Manual Setup (If Needed)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages (takes 1-2 minutes).

### Step 2: Launch the Application

#### Unified Web Platform (Recommended)
```bash
streamlit run app.py
```
Opens at `http://localhost:8501` - Complete platform with:
- Strategy backtesting & comparison
- ML/DL model training
- Portfolio optimization
- Custom strategy builder

#### Command Line
```bash
python main.py --symbol AAPL --start 2020-01-01 --end 2024-01-01 --strategy momentum
```
For quick backtests without the GUI.

---

## Step 3: Use the Unified Platform

The app opens automatically at `http://localhost:8501`

### Platform Tabs:

1. **🎯 Strategy Backtest**
   - Select and configure strategies
   - Run single backtests
   - View equity curves, drawdowns, trading signals

2. **⚖️ Compare Strategies**
   - Select multiple strategies
   - Side-by-side performance comparison
   - Normalized equity curves

3. **🤖 ML Models** (Enhanced!)
   - **Feature Control**: Select technical, statistical, time, lagged features
   - **Lookback Periods**: Choose 5, 10, 14, 20, 50, 100, 200 days
   - **Label Definition**: Classification thresholds, volatility-adjusted, future high/low
   - **Data Splits**: Holdout, time series CV, walk-forward validation
   - **Models**: Random Forest, XGBoost, LightGBM, SVM, Gradient Boosting
   - **Hyperparameters**: Full control over model architecture
   - **Feature Importance**: Visualize top features

4. **🧠 Deep Learning** (PyTorch!)
   - **Architectures**: LSTM, GRU, Transformer, CNN, MLP
   - **Full Control**: Layers, hidden sizes, dropout, bidirectional
   - **Training**: Batch size, epochs, learning rate, optimizer
   - **Optimizers**: Adam, SGD, AdamW, RMSprop
   - **Regularization**: Weight decay, early stopping
   - **Real-time Visualization**: Training/validation loss and accuracy

5. **⚙️ Optimization**
   - Parameter optimization
   - Walk-forward analysis
   - Grid/random search

6. **💼 Portfolio**
   - Multi-asset portfolio optimization
   - Efficient frontier
   - Risk-return analysis

7. **🔧 Custom Strategy**
   - Build your own strategies
   - Code editor interface
   - Template builder

---

## 🎯 First Test Run

Try this to make sure everything works:

1. Launch the web app: `./START_HERE.sh`
2. Left sidebar: **New defaults** (SPY, $10,000, 100% position size)
3. Select "Momentum (MA Crossover)"
4. Click "🚀 Run Backtest"
5. Wait 10-30 seconds
6. See your results with equity curves, drawdowns, and trading signals!

---

## 💡 Command Line Usage

### Run backtests with custom parameters:
```bash
python main.py --symbol AAPL --start 2020-01-01 --end 2024-01-01 --strategy momentum
```

### Run example scripts:
```bash
# Simple backtest
python examples/simple_backtest.py

# Strategy comparison
python examples/multi_strategy_comparison.py

# Custom strategy
python examples/custom_strategy.py

# Use CSV data
python examples/use_csv_data.py
```

### Get help:
```bash
python main.py --help
```

---

## 📊 Available Strategies

### Built-in Strategies:
1. **Buy & Hold** - Baseline strategy (buy and hold entire period)
2. **Momentum** - Moving average crossover (trend following)
3. **Mean Reversion** - Bollinger Bands (buy low, sell high)
4. **RSI Strategy** - Overbought/oversold indicator
5. **MACD** - MACD line and signal line crossover
6. **Breakout** - Donchian channel breakouts
7. **Turtle Trading** - Famous turtle trading system
8. **Triple MA** - Three moving average alignment
9. **Stochastic** - Stochastic oscillator signals
10. **Fibonacci Retracement** - Trade bounces off Fibonacci levels (NEW!)

### Key Features:
- ✅ Long-only or long-short modes (`allow_short` parameter)
- ✅ Realistic fee structures (per-share or percentage)
- ✅ Position sizing options (fixed $ or % of portfolio)
- ✅ Complete signal visualization (entries, exits, holds)
- ✅ Comprehensive performance metrics

---

## ⚠️ Common Issues & Solutions

### "Rate Limit" Error (Yahoo Finance)
**Problem:** Too many data requests in short time  
**Solution:** 
- Wait 5 minutes between requests
- App automatically uses cached data when available
- Use CSV data: `python examples/use_csv_data.py`

### App Won't Start
**Problem:** Missing or outdated dependencies  
**Solution:**
```bash
pip install --upgrade streamlit plotly pandas numpy yfinance scikit-learn xgboost lightgbm torch torchvision
```

### No Data Loaded
**Problem:** Network issues or invalid symbol  
**Solution:**
- Check internet connection
- Try popular symbols: SPY, QQQ, AAPL, MSFT, GOOGL
- Verify date range (not too old, not future dates)

### Trading Signals Chart Empty
**Problem:** Chart not displaying  
**Solution:** Chart now automatically uses backtest data - should work after recent updates. Refresh page if needed.

### Import Errors
**Problem:** Module not found  
**Solution:**
```bash
# Install all required packages (including ML/DL)
pip install -r requirements.txt

# This now includes:
# - Core: streamlit, plotly, yfinance, pandas, numpy
# - ML: scikit-learn, xgboost, lightgbm
# - DL: torch, torchvision (PyTorch)
```

---

## 📚 Next Steps

1. ✅ **Run your first backtest** - Use the web app with default settings
2. 🔬 **Compare strategies** - Use "Compare Strategies" tab to see which works best
3. 📊 **Test different symbols** - Try various stocks (AAPL, SPY, TSLA, etc.)
4. ⚙️ **Experiment with parameters** - Adjust strategy settings to optimize
5. 🎓 **Build custom strategies** - See `examples/custom_strategy.py`
6. 📈 **Analyze results** - Study equity curves, drawdowns, and trading signals

---

## 📖 Additional Resources

### Documentation
- **README.md** - Complete documentation, features, installation
- **CONTRIBUTING.md** - Guidelines for contributing to the project

### Example Code
```
examples/
├── simple_backtest.py          # Basic backtest example
├── multi_strategy_comparison.py # Compare multiple strategies
├── custom_strategy.py           # Create your own strategy
├── use_csv_data.py             # Use your own data files
└── demo_notebook.ipynb         # Jupyter notebook demo
```

### Framework Structure
```
quant_framework/
├── data/          # Data loading and indicators
├── models/        # Trading strategies
├── backtest/      # Backtesting engine
├── ml/            # Machine learning models
├── research/      # Optimization and analysis
├── execution/     # Live trading (paper/real)
└── utils/         # Plotting and utilities
```

### Key Modules
- **Data Loaders**: `quant_framework.data.loaders`
- **Strategies**: `quant_framework.models`
- **Backtesting**: `quant_framework.backtest.Backtester`
- **Indicators**: `quant_framework.data.indicators`
- **Optimization**: `quant_framework.research.optimizer`

---

## 🎯 Quick Reference Commands

```bash
# Launch unified platform (easiest)
./START_HERE.sh

# Or manually
streamlit run app.py

# CLI backtest
python main.py --symbol AAPL --strategy momentum

# Run examples
python examples/simple_backtest.py
python examples/multi_strategy_comparison.py

# Install/update dependencies
pip install -r requirements.txt
```

---

## 🎉 You're Ready!

**Launch the app and start researching trading strategies!**

The unified platform provides everything in one place:
- 📊 Strategy backtesting with interactive charts
- 📈 Complete performance metrics & trading signals
- 🔄 Side-by-side strategy comparison
- 🤖 ML/DL model training & evaluation
- ⚙️ Strategy optimization tools
- 💼 Portfolio optimization
- 🎨 Customizable parameters & custom strategies

**No coding required for basic research - everything in one unified interface!**

---

## 🆘 Getting Help

- **Issues?** Check the "Common Issues" section above
- **Questions?** Read the full README.md
- **Examples?** Explore the `examples/` directory
- **Custom strategies?** See `examples/custom_strategy.py`

**Happy Trading! 📈🚀**

