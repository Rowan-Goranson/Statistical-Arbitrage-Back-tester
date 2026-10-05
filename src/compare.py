"""Side-by-side: original in-sample backtest vs bug-fixed in-sample vs walk-forward OOS.

v1/v2 are the 5x risk-parity portfolios produced by the original / fixed main.py (saved to results/).
v3 is walk_forward.py (1x, rescaled to 5x with the same financing assumption).
All stats are on the common window so the comparison is apples-to-apples.
"""
import matplotlib.pyplot as plt
import pandas as pd

from walk_forward import stats, levered

R = 'results'
v1 = pd.read_csv(f'{R}/v1_original_5x_returns.csv', index_col=0, parse_dates=True).squeeze()
v2 = pd.read_csv(f'{R}/v2_fixed_insample_5x_returns.csv', index_col=0, parse_dates=True).squeeze()
v3_1x = pd.read_csv(f'{R}/v3_walkforward_1x_returns.csv', index_col=0, parse_dates=True).squeeze()

# Original code leaves NaN returns in the first days (5-day rolling vol warm-up), which turns every
# cumprod into NaN; treat them as 0 so the curve can be drawn at all.
v1 = v1.fillna(0)
common = v3_1x.index
series = {
    'v1  Original (in-sample, look-ahead, arg bug)': v1.reindex(common).fillna(0),
    'v2  Bugs fixed, still in-sample': v2.reindex(common).fillna(0),
    'v3  Walk-forward OOS, 5x': levered(v3_1x, 5),
    'v3  Walk-forward OOS, 1x': v3_1x,
}
colors = ['#c0392b', '#e67e22', '#1f4e79', '#7f8c8d']

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
rows = []
for (name, r), c in zip(series.items(), colors):
    eq = (1 + r).cumprod()
    ax.plot(eq - 1, label=name, color=c, lw=2)
    ax2.plot(eq / eq.cummax() - 1, color=c, lw=1.5)
    rows.append({'Version': name, **stats(r)})
ax.axhline(0, color='gray', ls='--', lw=1)
ax.set_title(f'Same strategy, corrected research process  ({common.min().date()} to {common.max().date()}; 5x unless noted)')
ax.set_ylabel('Cumulative return'); ax.legend(loc='upper left'); ax.grid(True, alpha=.4)
ax2.set_ylabel('Drawdown'); ax2.grid(True, alpha=.4)
plt.tight_layout(); plt.savefig(f'{R}/research_process_comparison.png', dpi=140)

table = pd.DataFrame(rows).set_index('Version')
table.to_csv(f'{R}/research_process_comparison.csv')
pd.set_option('display.float_format', lambda v: f'{v:,.2f}'); pd.set_option('display.width', 200)
print(table[['Cumulative Return', 'Annual Return', 'Annual Vol', 'Sharpe', 'Max Drawdown']])
