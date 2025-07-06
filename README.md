# Statistical-Arbitrage-Back-tester
Statistical Arbitrage pairs trading strategy bot, backtested on cointegrated S&P 500 pairs from 2018-2024

## Overview
This project implements a statistical arbitrage strategy*on S&P 500 equities using cointegration analysis** and **pairs trading techniques. It identifies cointegrated stock pairs, backtests the strategy historically (2018–2024), and builds a risk-parity portfolio with slippage-adjusted execution and leverage.

## Key Features
- **Cointegration Detection:** Automatically identifies statistically cointegrated pairs (p-value < 0.01).
- **Pairs Trading Backtest:** Multi-level z-score thresholding with stop-loss/take-profit rules.
- **Risk-Parity Portfolio Allocation:** Dynamically allocates capital based on realized volatility; supports customizable leverage.
- **Transaction Costs:** Models realistic slippage and liquidity costs in all trades.
- **Performance Comparison:** Benchmarks portfolio against the S&P 500 (SPY) total return.
- **Sensitivity Analysis:** Stress tests Sharpe ratio performance under varied slippage and market conditions.

## Results (2018–2024)
- **Cumulative Return:** +95% (with leverage)
- **Sharpe Ratio:** 1.56
- **Maximum Drawdown:** Refer to `top10_pairs_results.csv`
- Outperformance (with leverage) versus the S&P 500 over 6 years; ![Portfolio vs. S&P 500](images/performance_graph.png)

## Project Structure
data_loader.py # Data fetching and S&P 500 benchmark loader
├── main.py # Full strategy pipeline and plotting
├── data/ # Folder for price datasets
│ └── price_data.csv # Historical price data (user-provided)
├── top5_pairs_results.csv # Output: Top pairs backtest summary

## Requirements
- Python 3.8+
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `yfinance`, `statsmodels`

```bash
pip install pandas numpy matplotlib seaborn yfinance statsmodels
```

## Strategy Methodology

Pair Selection:
- Select pairs with >0.8 correlation and cointegration p-value < 0.01.
Trading Logic:
- Enter positions based on z-score thresholds (with scaling position size of .5 and 2).
- Exit positions on reversion or when stop-loss/take-profit triggers.
Portfolio Optimization:
- Risk-parity weighting based on inverse realized volatility.
- Leverage to scale returns.
Performance Reporting:
- Returns, Sharpe, volatility, drawdowns, slippage-adjusted metrics

#Notes
- For strategy and research purposes only
