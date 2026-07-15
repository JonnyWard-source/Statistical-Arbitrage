# Statistical-Arbitrage
## Overview
This repository implements a research framework for statistical arbitrage pairs trading using:
- Cointegration-based pair selection
- Rolling z-score mean-reversion signals
- Walk-forward (out-of-sample) backtesting
- Transaction cost modelling
- Parameter sensitivity / optimisation analysis

This project is designed to replicate a simplified quantitative research pipeline used in systematic trading environments.

All detailed explanations, experimentation, and analysis are provided in the accompanying Jupyter notebooks.

## Key Features
### Pair Selection
- Linear regression-based hedge ratio estimation
- Cointegration testing via Engle–Granger methodology
- Half-life filtering of mean-reverting spreads
### Signal Generation
- Rolling z-score of spread
- Entry/exit thresholds for mean reversion trades
- Momentum-confirmed signal triggering
### Portfolio Construction
- Multi-pair aggregation
- Volatility targeting
- Simple risk-weighted allocation across pairs
### Backtesting Framework
- Walk-forward train/test splits
- Transaction cost simulation
- Position turnover penalties
- Performance tracking over multiple regimes
### Performance Metrics
- Annualised return
- Sharpe ratio
- Maximum drawdown
- Realised volatility
- Trade frequency

## Results (Summary)
Performance varies across market regimes, with typical results:
- Annualised return: ~10–15%
- Sharpe ratio: ~1.0–1.8
- Max drawdown: ~5–15%
- Volatility target: 10%
- Performance is not stable across all periods, reflecting realistic strategy decay and regime dependence.

## Important Notes
- This project is for research and educational purposes only
- It does not represent a production trading system
- Results are highly dependent on historical assumptions and parameter choices
- No guarantee of future performance or profitability

## Technologies Used
- Python 3.10+
- NumPy / Pandas
- Statsmodels
- Matplotlib
- yfinance
- Jupyter Notebooks

## Suggested Improvements (Future Work)
- Kalman filter-based dynamic hedge ratios
- Bayesian cointegration filtering
- Regime detection (volatility / correlation clustering)
- More realistic execution modelling (slippage, liquidity)
- Expansion to intraday data
