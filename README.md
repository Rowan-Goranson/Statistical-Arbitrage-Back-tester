# Statistical-Arbitrage-Back-tester
Statistical Arbitrage pairs trading strategy bot, backtested on cointegrated S&P 500 pairs from 2018-2023

## Overview
This project implements a statistical arbitrage strategy*on S&P 500 equities using cointegration analysis** and **pairs trading techniques. It identifies cointegrated stock pairs, backtests the strategy historically (2018–2023), and builds a risk-parity portfolio with slippage-adjusted execution and leverage.

## Key Features
- **Cointegration Detection:** Automatically identifies statistically cointegrated pairs (p-value < 0.01).
- **Pairs Trading Backtest:** z-score entry/exit bands with per-trade stop-loss/take-profit rules.
- **Walk-Forward Validation:** Rolling 2-year formation / 6-month trading windows with frozen hedge ratios.
- **Portfolio & Leverage:** Equal-weight (walk-forward) or risk-parity (original) portfolios; leverage analysis with financing costs.
- **Transaction Costs:** Models realistic slippage and liquidity costs in all trades.
- **Performance Comparison:** Benchmarks portfolio against the S&P 500 (SPY) total return.
- **Sensitivity Analysis:** Stress tests Sharpe ratio performance under varied slippage and market conditions.

## Results

Out of sample (walk-forward, 2020–2023, 1x): **-19% cumulative, Sharpe -0.78**, vs. SPY +57%. An earlier in-sample version of this backtest showed +95–121% (Sharpe 0.7–1.6), but that came from selecting pairs and fitting hedge ratios on the same data it was scored on. See "Research process update" below for the side-by-side and what changed.

## Project Structure
src/
├── data_loader.py          # Data fetching and S&P 500 benchmark loader
├── backtest.py             # Pair backtest engine (spread, z-score, positions, costs)
├── main.py                 # Original in-sample pipeline (kept for comparison)
├── walk_forward.py         # Walk-forward out-of-sample backtest + leverage analysis
├── tune_walk_forward.py    # Nested parameter search (tuned on formation data only)
└── compare.py              # v1 / v2 / v3 side-by-side
data/price_data.csv         # Historical price data
results/                    # Walk-forward outputs, comparison table and charts
top10_pairs_results.csv     # In-sample top pairs summary

## Requirements
- Python 3.8+
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `yfinance`, `statsmodels`

```bash
pip install pandas numpy matplotlib seaborn yfinance statsmodels
```

## Strategy Methodology

Pair Selection:
- Select pairs with cointegration p-value < 0.01 (walk-forward: re-selected each fold on the formation window only).
Trading Logic:
- Enter at |z| > 2, exit inside |z| < 0.5; per-trade stop-loss/take-profit.
- Signals use the prior close (no same-bar look-ahead).
Portfolio Optimization:
- Risk-parity weighting based on inverse realized volatility.
- Leverage to scale returns.
Performance Reporting:
- Returns, Sharpe, volatility, drawdowns, slippage-adjusted metrics

#Notes
- For strategy and research purposes only

## Research process update

The original backtest (v1) selected pairs and fit hedge ratios on the full 2018-2023 sample and had a same-bar look-ahead and an argument-order bug. Fixing the bugs (v2) made results look *better* (Sharpe ~4), which exposed the in-sample fit as the dominant bias. v3 re-estimates everything walk-forward (2y formation / 6m trading, pairs picked by cointegration p-value only, frozen hedge ratio and spread stats while trading).

![comparison](results/research_process_comparison.png)

| Version (2020-2023, 5x unless noted) | Cum. return | Sharpe | Max DD |
|---|---|---|---|
| v1 Original (in-sample, look-ahead, arg bug) | +0.3% | 0.10 | -25% |
| v2 Bugs fixed, still in-sample | +1209% | 3.99 | -7% |
| v3 Walk-forward OOS, 5x | -87% | -1.38 | -88% |
| v3 Walk-forward OOS, 1x | -19% | -0.78 | -23% |

A nested parameter search (entry/exit z, p-cutoff, chosen on formation data only) did not help (1x Sharpe -1.04). Conclusion: pairs cointegrated in one 2-year window of mega-cap prices do not stay cointegrated out of sample; the earlier edge was in-sample fit. Next: sector-matched pairs, log prices, rolling hedge ratio.

Reproduce: `PYTHONPATH=src python src/walk_forward.py`, `src/tune_walk_forward.py`, then `src/compare.py`. v1/v2 return series in `results/` were generated from the original commit (`e481400`) and the fixed `main.py`.
