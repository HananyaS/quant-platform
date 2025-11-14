# Contributing to Quantitative Trading Framework

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Project Structure](#project-structure)
5. [Coding Standards](#coding-standards)
6. [Testing](#testing)
7. [Pull Request Process](#pull-request-process)
8. [Areas for Contribution](#areas-for-contribution)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help maintain a welcoming environment
- No harassment or discrimination

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/quant_framework.git`
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes
6. Submit a pull request

## Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/quant_framework.git
cd quant_framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .  # Install in development mode

# Install development tools
pip install black flake8 mypy pytest pytest-cov
```

## Project Structure

```
quant_framework/
├── data/          # Data loading and preprocessing
├── models/        # Trading strategies
├── backtest/      # Backtesting engine
├── infra/         # Infrastructure and orchestration
├── execution/     # Live trading connectors
├── utils/         # Utilities
├── configs/       # Configuration files
└── tests/         # Unit tests
```

## Coding Standards

### Style Guide

- Follow PEP 8 conventions
- Use type hints for function signatures
- Use dataclasses for configuration objects
- Write docstrings for all public functions and classes

### Example

```python
from typing import Optional
import pandas as pd

def calculate_returns(
    prices: pd.Series,
    periods: int = 1,
    method: str = "simple"
) -> pd.Series:
    """
    Calculate returns from price series.
    
    Args:
        prices: Price series
        periods: Number of periods for calculation
        method: "simple" or "log" returns
        
    Returns:
        Series with calculated returns
        
    Raises:
        ValueError: If method is not supported
    """
    if method == "simple":
        return prices.pct_change(periods)
    elif method == "log":
        return np.log(prices / prices.shift(periods))
    else:
        raise ValueError(f"Unsupported method: {method}")
```

### Code Formatting

Use `black` for code formatting:

```bash
black quant_framework/
```

### Linting

Use `flake8` for linting:

```bash
flake8 quant_framework/ --max-line-length=100
```

### Type Checking

Use `mypy` for type checking:

```bash
mypy quant_framework/
```

## Testing

### Writing Tests

- Write tests for all new features
- Use pytest framework
- Aim for >80% code coverage
- Use fixtures for common test data

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest quant_framework/tests/test_strategies.py

# Run with coverage
pytest --cov=quant_framework --cov-report=html

# Run only unit tests (fast)
pytest -m "not integration"
```

### Test Example

```python
import pytest
from quant_framework.models import MomentumStrategy

def test_momentum_strategy_init():
    """Test momentum strategy initialization."""
    strategy = MomentumStrategy(short_window=10, long_window=20)
    assert strategy.short_window == 10
    assert strategy.long_window == 20

@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    # Create and return test data
    pass
```

## Pull Request Process

1. **Update Documentation**: Update README and docstrings
2. **Add Tests**: Include tests for new features
3. **Pass CI**: Ensure all tests pass
4. **Code Review**: Address review comments
5. **Squash Commits**: Clean up commit history if needed

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Added tests
- [ ] All tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## Areas for Contribution

### High Priority

1. **Additional Strategies**
   - More classical quant strategies
   - Machine learning models
   - Portfolio optimization

2. **Data Sources**
   - Additional API integrations
   - Alternative data sources
   - Real-time data feeds

3. **Risk Management**
   - Position sizing algorithms
   - Stop-loss implementation
   - Portfolio risk metrics

4. **Live Trading**
   - Complete broker integrations
   - Order management system
   - Real-time monitoring

### Medium Priority

5. **Performance**
   - Optimization of backtesting engine
   - Parallel processing
   - Caching mechanisms

6. **Visualization**
   - Interactive dashboards
   - Advanced charting
   - Portfolio analytics

7. **Documentation**
   - Tutorial notebooks
   - Strategy examples
   - Best practices guide

### Nice to Have

8. **Machine Learning**
   - Feature engineering utilities
   - Model training pipeline
   - Hyperparameter optimization

9. **Database Integration**
   - Data persistence
   - Historical data management
   - Results tracking

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Check existing issues before creating new ones

Thank you for contributing! 🎉

