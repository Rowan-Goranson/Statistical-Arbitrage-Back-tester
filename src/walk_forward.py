"""Walk-forward (out-of-sample) pairs backtest + leverage analysis.

Each fold: pick pairs and fit hedge ratio / spread mean / spread std on a formation window,
then trade the *next* window with those frozen estimates. Pair selection uses cointegration
p-value only (never backtest Sharpe). Positions start flat at each fold.
"""
import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

import data_loader
from backtest import backtest_pair, fit_spread_params

FORMATION = 504   # ~2y of trading days
TRADE = 126       # ~6m
P_CUTOFF = 0.01
MAX_PAIRS = 10
FINANCING = 0.05  # annual borrow rate paid on the levered portion (L-1)
LEVERAGE = [1, 2, 3, 5, 8, 10]
OUT = 'results'


def select_pairs(form_df):
    pairs = []
    for s1, s2 in combinations(form_df.columns, 2):
        p = coint(form_df[s1], form_df[s2])[1]
        if p < P_CUTOFF:
            pairs.append((s1, s2, p))
    pairs.sort(key=lambda x: x[2])
    return pairs, pairs[:MAX_PAIRS]


def walk_forward(df):
    fold_returns, log = [], []
    for start in range(0, len(df) - FORMATION - 1, TRADE):
        form = df.iloc[start:start + FORMATION]
        # include the last formation bar so the first trading day has a prior close
        trade = df.iloc[start + FORMATION - 1:start + FORMATION + TRADE]
        if len(trade) < 20:
            break
        passed, chosen = select_pairs(form)
        rets = {}
        for s1, s2, p in chosen:
            params = fit_spread_params(s1, s2, form)
            res = backtest_pair(trade, s1, s2, params=params)
            rets[f'{s1}-{s2}'] = res['daily_return']
        port = pd.DataFrame(rets).mean(axis=1) if rets else pd.Series(0.0, index=trade.index)
        port = port.iloc[1:]  # drop the seed bar
        fold_returns.append(port)
        log.append({'trade_start': port.index[0].date(), 'trade_end': port.index[-1].date(),
                    'pairs_passed': len(passed), 'pairs_traded': len(chosen),
                    'fold_return': (1 + port).prod() - 1})
        print(log[-1])
    return pd.concat(fold_returns), pd.DataFrame(log)


def stats(r):
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    ann_vol = r.std() * 252 ** 0.5
    return {
        'Cumulative Return': eq.iloc[-1] - 1,
        'Annual Return': r.mean() * 252,
        'Annual Vol': ann_vol,
        'Sharpe': r.mean() / r.std() * 252 ** 0.5 if r.std() > 0 else np.nan,
        'Max Drawdown': dd.min(),
        'Worst Day': r.min(),
    }


def levered(r, L, financing=FINANCING):
    lr = L * r - max(L - 1, 0) * financing / 252
    # account wiped out: once equity hits zero it stays there
    eq = (1 + lr).clip(lower=0).cumprod()
    return eq.pct_change().fillna(lr.iloc[0]).where(eq.shift(fill_value=1) > 0, 0)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv('data/price_data.csv', index_col=0, parse_dates=True)
    oos, log = walk_forward(df)
    log.to_csv(f'{OUT}/folds.csv', index=False)
    oos.to_csv(f'{OUT}/v3_walkforward_1x_returns.csv', header=['return'])

    spy = data_loader.load_spy_cumulative(oos.index.min().strftime('%Y-%m-%d'),
                                          (oos.index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
    spy_r = spy.squeeze().add(1).pct_change().reindex(oos.index).fillna(0)

    rows = []
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for L in LEVERAGE:
        lr = levered(oos, L)
        rows.append({'Leverage': L, **stats(lr)})
        ax1.plot((1 + lr).cumprod() - 1, label=f'{L}x')
        eq = (1 + lr).cumprod()
        ax2.plot(eq / eq.cummax() - 1, label=f'{L}x')
    ax1.plot((1 + spy_r).cumprod() - 1, color='k', ls='--', label='SPY')
    ax1.set_title(f'Walk-forward OOS cumulative return by leverage (financing {FINANCING:.0%} on borrowed portion)')
    ax1.legend(ncol=4); ax1.grid(True)
    ax2.set_title('Drawdown'); ax2.grid(True)
    plt.tight_layout(); plt.savefig(f'{OUT}/leverage_effect.png', dpi=130)

    table = pd.DataFrame(rows).set_index('Leverage')
    table.to_csv(f'{OUT}/leverage_table.csv')
    pd.set_option('display.float_format', lambda v: f'{v:,.3f}')
    print('\nOOS window:', oos.index.min().date(), '->', oos.index.max().date(), f'({len(oos)} days)')
    print('SPY buy&hold:', {k: round(v, 3) for k, v in stats(spy_r).items()})
    print(table)
