"""
Performance reporting utilities.
"""

import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path


class PerformanceReport:
    """
    Generate comprehensive performance reports.
    
    Example:
        report = PerformanceReport(results)
        report.generate_html_report("report.html")
        report.print_summary()
    """
    
    def __init__(self, backtest_results: Dict[str, Any]):
        """
        Initialize performance report.
        
        Args:
            backtest_results: Dictionary with backtest results
        """
        self.results = backtest_results
        self.metrics = backtest_results.get('metrics', {})
        self.equity_curve = backtest_results.get('equity_curve')
        self.trades = backtest_results.get('trades')
    
    def print_summary(self) -> None:
        """Print summary report to console."""
        print("\n" + "=" * 70)
        print("PERFORMANCE REPORT")
        print("=" * 70)
        
        # Return metrics
        print("\n📊 RETURN METRICS")
        print("-" * 70)
        print(f"Total Return:           {self.metrics.get('total_return', 0)*100:>10.2f}%")
        print(f"Annual Return:          {self.metrics.get('annual_return', 0)*100:>10.2f}%")
        print(f"Annual Volatility:      {self.metrics.get('annual_volatility', 0)*100:>10.2f}%")
        
        # Risk metrics
        print("\n⚠️  RISK METRICS")
        print("-" * 70)
        print(f"Sharpe Ratio:           {self.metrics.get('sharpe_ratio', 0):>10.3f}")
        print(f"Sortino Ratio:          {self.metrics.get('sortino_ratio', 0):>10.3f}")
        print(f"Calmar Ratio:           {self.metrics.get('calmar_ratio', 0):>10.3f}")
        print(f"Max Drawdown:           {self.metrics.get('max_drawdown', 0)*100:>10.2f}%")
        print(f"VaR (95%):              {self.metrics.get('var_95', 0)*100:>10.2f}%")
        print(f"CVaR (95%):             {self.metrics.get('cvar_95', 0)*100:>10.2f}%")
        
        # Trading metrics
        print("\n📈 TRADING METRICS")
        print("-" * 70)
        print(f"Win Rate:               {self.metrics.get('win_rate', 0)*100:>10.2f}%")
        print(f"Profit Factor:          {self.metrics.get('profit_factor', 0):>10.3f}")
        
        if 'num_trades' in self.metrics:
            print(f"Number of Trades:       {self.metrics.get('num_trades', 0):>10.0f}")
            print(f"Turnover:               {self.metrics.get('turnover', 0):>10.3f}")
        
        print("\n" + "=" * 70)
    
    def generate_html_report(self, output_path: str) -> None:
        """
        Generate HTML report.
        
        Args:
            output_path: Path to save HTML report
        """
        html_content = self._create_html_content()
        
        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report saved to {output_path}")
    
    def _create_html_content(self) -> str:
        """Create HTML content for report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Trading Strategy Performance Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2E86AB;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .section {{
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }}
        .metric-label {{
            font-weight: bold;
        }}
        .metric-value {{
            color: #2E86AB;
        }}
        .positive {{
            color: #06A77D;
        }}
        .negative {{
            color: #D62828;
        }}
        h2 {{
            color: #333;
            border-bottom: 2px solid #2E86AB;
            padding-bottom: 10px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Trading Strategy Performance Report</h1>
        <p>Generated: {timestamp}</p>
    </div>
    
    <div class="section">
        <h2>Return Metrics</h2>
        <div class="metric">
            <span class="metric-label">Total Return:</span>
            <span class="metric-value {'positive' if self.metrics.get('total_return', 0) > 0 else 'negative'}">
                {self.metrics.get('total_return', 0)*100:.2f}%
            </span>
        </div>
        <div class="metric">
            <span class="metric-label">Annual Return:</span>
            <span class="metric-value {'positive' if self.metrics.get('annual_return', 0) > 0 else 'negative'}">
                {self.metrics.get('annual_return', 0)*100:.2f}%
            </span>
        </div>
        <div class="metric">
            <span class="metric-label">Annual Volatility:</span>
            <span class="metric-value">{self.metrics.get('annual_volatility', 0)*100:.2f}%</span>
        </div>
    </div>
    
    <div class="section">
        <h2>Risk Metrics</h2>
        <div class="metric">
            <span class="metric-label">Sharpe Ratio:</span>
            <span class="metric-value">{self.metrics.get('sharpe_ratio', 0):.3f}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Sortino Ratio:</span>
            <span class="metric-value">{self.metrics.get('sortino_ratio', 0):.3f}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Calmar Ratio:</span>
            <span class="metric-value">{self.metrics.get('calmar_ratio', 0):.3f}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Max Drawdown:</span>
            <span class="metric-value negative">{self.metrics.get('max_drawdown', 0)*100:.2f}%</span>
        </div>
        <div class="metric">
            <span class="metric-label">VaR (95%):</span>
            <span class="metric-value">{self.metrics.get('var_95', 0)*100:.2f}%</span>
        </div>
    </div>
    
    <div class="section">
        <h2>Trading Metrics</h2>
        <div class="metric">
            <span class="metric-label">Win Rate:</span>
            <span class="metric-value">{self.metrics.get('win_rate', 0)*100:.2f}%</span>
        </div>
        <div class="metric">
            <span class="metric-label">Profit Factor:</span>
            <span class="metric-value">{self.metrics.get('profit_factor', 0):.3f}</span>
        </div>
        {'<div class="metric"><span class="metric-label">Number of Trades:</span><span class="metric-value">' + 
         f"{self.metrics.get('num_trades', 0):.0f}</span></div>" if 'num_trades' in self.metrics else ''}
    </div>
</body>
</html>
"""
        return html
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export report as dictionary.
        
        Returns:
            Dictionary with all report data
        """
        return {
            'metrics': self.metrics,
            'timestamp': datetime.now().isoformat(),
            'equity_curve': self.equity_curve.to_dict() if self.equity_curve is not None else None,
            'trades': self.trades.to_dict() if self.trades is not None else None
        }
    
    def to_json(self, output_path: str) -> None:
        """
        Export report as JSON.
        
        Args:
            output_path: Path to save JSON file
        """
        import json
        
        data = self.to_dict()
        
        # Make numpy types JSON serializable
        def convert_types(obj):
            if isinstance(obj, (pd.Timestamp, datetime)):
                return obj.isoformat()
            elif isinstance(obj, (int, float)):
                return float(obj)
            return obj
        
        # Create simplified version for JSON
        json_data = {
            'metrics': {k: float(v) if isinstance(v, (int, float)) else str(v) 
                       for k, v in data['metrics'].items()},
            'timestamp': data['timestamp']
        }
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        
        print(f"JSON report saved to {output_path}")

