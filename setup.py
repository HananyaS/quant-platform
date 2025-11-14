"""
Setup script for the quantitative trading framework.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="quant_framework",
    version="1.0.0",
    author="Quantitative Trading Team",
    description="A modular framework for algorithmic trading",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quant_framework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "yfinance>=0.2.28",
        "matplotlib>=3.7.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "ml": [
            "scikit-learn>=1.3.0",
            "tensorflow>=2.13.0",
        ],
        "live": [
            "alpaca-trade-api>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "quant-backtest=main:main",
        ],
    },
)

