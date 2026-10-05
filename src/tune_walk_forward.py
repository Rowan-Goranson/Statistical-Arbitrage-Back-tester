"""Nested walk-forward tuning.

Per outer fold: choose (entry_z, exit_z, p_cutoff) using ONLY the formation window
(inner formation -> inner validation, best validation Sharpe), then refit on the full
formation window and trade the next unseen window. Also prints the static grid on the
OOS period for sensitivity only -- that table is data-snooped, not a result.
"""
import itertools
import os

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from backtest import backtest_pair, fit_spread_params
from walk_forward import FORMATION, TRADE, MAX_PAIRS, OUT, stats

ENTRY = [1.5, 2.0, 2.5]
EXIT = [0.0, 0.5]
PCUT = [0.01, 0.001]
GRID = list(itertools.product(ENTRY, EXIT, PCUT))
INNER_VAL = TRADE  # last 6m of formation used for validation


def sorted_pvalues(form):
    out = [(a, b, coint(form[a], form[b])[1]) for a, b in itertools.combinations(form.columns, 2)]
    return sorted(out, key=lambda x: x[2])


def run_window(form, trade, pvals, entry, exit_, pcut):
    chosen = [p for p in pvals if p[2] < pcut][:MAX_PAIRS]
    rets = {}
    for s1, s2, _ in chosen:
        res = backtest_pair(trade, s1, s2, entry_z=entry, exit_z=exit_, params=fit_spread_params(s1, s2, form))
        rets[f'{s1}-{s2}'] = res['daily_return']
    if not rets:
        return pd.Series(0.0, index=trade.index[1:]), 0
    return pd.DataFrame(rets).mean(axis=1).iloc[1:], len(chosen)


def sharpe(r):
    return r.mean() / r.std() * 252 ** 0.5 if r.std() > 0 else -np.inf


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv('data/price_data.csv', index_col=0, parse_dates=True)
    nested, static = [], {g: [] for g in GRID}
    log = []
    for start in range(0, len(df) - FORMATION - 1, TRADE):
        form = df.iloc[start:start + FORMATION]
        trade = df.iloc[start + FORMATION - 1:start + FORMATION + TRADE]
        if len(trade) < 20:
            break
        # inner split: tune on formation data only
        in_form, in_val = form.iloc[:-INNER_VAL], form.iloc[-INNER_VAL - 1:]
        inner_p = sorted_pvalues(in_form)
        scores = {g: sharpe(run_window(in_form, in_val, inner_p, *g)[0]) for g in GRID}
        best = max(scores, key=scores.get)
        # outer: refit on full formation, trade unseen window
        outer_p = sorted_pvalues(form)
        r, n = run_window(form, trade, outer_p, *best)
        nested.append(r)
        for g in GRID:
            static[g].append(run_window(form, trade, outer_p, *g)[0])
        log.append({'trade_start': r.index[0].date(), 'entry_z': best[0], 'exit_z': best[1],
                    'p_cutoff': best[2], 'inner_val_sharpe': round(scores[best], 2),
                    'pairs': n, 'fold_return': round((1 + r).prod() - 1, 4)})
        print(log[-1])

    pd.DataFrame(log).to_csv(f'{OUT}/tuned_folds.csv', index=False)
    oos = pd.concat(nested)
    print('\nNESTED-TUNED OOS (1x):', {k: round(v, 3) for k, v in stats(oos).items()})

    rows = []
    for g, parts in static.items():
        s = stats(pd.concat(parts))
        rows.append({'entry_z': g[0], 'exit_z': g[1], 'p_cutoff': g[2],
                     'Sharpe': s['Sharpe'], 'Cumulative': s['Cumulative Return'], 'MaxDD': s['Max Drawdown']})
    grid = pd.DataFrame(rows).sort_values('Sharpe', ascending=False)
    grid.to_csv(f'{OUT}/static_grid_snooped.csv', index=False)
    pd.set_option('display.float_format', lambda v: f'{v:,.3f}')
    print('\nSTATIC GRID on OOS (data-snooped, sensitivity only):')
    print(grid.to_string(index=False))
