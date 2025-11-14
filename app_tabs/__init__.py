"""
Streamlit app tabs - modular organization of UI tabs.

Each tab is in its own file for better maintainability.
"""

from .strategy_backtest import render_strategy_backtest_tab
from .compare_strategies import render_compare_strategies_tab
from .ml_models import render_ml_models_tab
from .deep_learning import render_deep_learning_tab
from .optimization import render_optimization_tab
from .portfolio import render_portfolio_tab
from .custom_strategy import render_custom_strategy_tab

__all__ = [
    'render_strategy_backtest_tab',
    'render_compare_strategies_tab',
    'render_ml_models_tab',
    'render_deep_learning_tab',
    'render_optimization_tab',
    'render_portfolio_tab',
    'render_custom_strategy_tab',
]

