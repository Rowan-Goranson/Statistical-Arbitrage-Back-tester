import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_spread_params(s1: str, s2: str, df: pd.DataFrame):
    """OLS hedge ratio of s1 on s2, plus spread mean/std, all estimated on df."""
    x = sm.add_constant(df[s2])
    model = sm.OLS(df[s1], x).fit()
    hedge_ratio = model.params.iloc[1]
    spread = df[s1] - hedge_ratio * df[s2]
    return hedge_ratio, spread.mean(), spread.std()


#make sure s2 is the dominant stock when being called- can definitely add some type of check in the function later
def calculate_spread_and_zscore(s1: str, s2: str, df: pd.DataFrame, params=None):
    """params=(hedge_ratio, mu, sd) freezes the estimates (e.g. from a formation window);
    default fits them on df itself (in-sample)."""
    hedge_ratio, mu, sd = params if params is not None else fit_spread_params(s1, s2, df)
    spread = df[s1] - hedge_ratio * df[s2]
    zscore = (spread - mu) / sd
    return spread, zscore, hedge_ratio


#entry and exit are basic, can definitely be fine tuned
def backtest_pair(df, s1, s2, entry_z=2.0, exit_z=0.5, liquidity_factor=0.0005, volatility_factor=0.002, stop_loss=-15, take_profit=30, params=None):
    """Signal at close t-1 (z[t-1]) sets the position held over t-1 -> t, so no same-bar look-ahead.
    stop_loss / take_profit are per-trade PnL in % of capital.
    params=(hedge_ratio, mu, sd) trades df with frozen out-of-sample estimates."""
    spread, zscore, hedge_ratio = calculate_spread_and_zscore(s1, s2, df, params)

    position1 = [0]
    position2 = [0]
    daily_return = [0]

    rolling_volatility = df[s1].pct_change().rolling(window=5).std().fillna(0)

    trade_pnl = 0.0   # PnL of the current trade, fraction of capital
    stopped = False   # after a stop/take-profit, stay flat until z re-enters the exit band

    for i in range(1, len(zscore)):
        z = zscore.iloc[i-1]  # only information available at the prior close
        price1 = df[s1].iloc[i-1]
        price2 = df[s2].iloc[i-1]
        notional = price1 + abs(hedge_ratio) * price2  # dollar value of fully hedged position

        if abs(z) < exit_z:
            stopped = False

        prev1, prev2 = position1[-1], position2[-1]
        if stopped:
            pos1, pos2 = 0, 0
        elif z > entry_z:
            pos1, pos2 = -1 / notional, hedge_ratio / notional
        elif z < -entry_z:
            pos1, pos2 = 1 / notional, -hedge_ratio / notional
        elif abs(z) < exit_z:
            pos1, pos2 = 0, 0
        else:
            pos1, pos2 = prev1, prev2  # hold between exit and entry bands

        if (pos1 == 0 and prev1 != 0) or (pos1 != 0 and np.sign(pos1) != np.sign(prev1)):
            trade_pnl = 0.0  # new trade

        ret = pos1 * (df[s1].iloc[i] - price1) + pos2 * (df[s2].iloc[i] - price2)

        # Costs in % of capital: dollars traded on both legs
        traded_dollars = abs(pos1 - prev1) * price1 + abs(pos2 - prev2) * price2
        cost = (liquidity_factor + volatility_factor * rolling_volatility.iloc[i-1]) * traded_dollars
        pct_return = ret - cost

        trade_pnl += pct_return
        if pos1 != 0 and (
            (stop_loss is not None and trade_pnl <= stop_loss / 100)
            or (take_profit is not None and trade_pnl >= take_profit / 100)
        ):
            stopped = True  # flat from next bar (exit cost not modelled on this bar)

        position1.append(pos1)
        position2.append(pos2)
        daily_return.append(pct_return)

    results = pd.DataFrame({
        'spread': spread,
        'zscore': zscore,
        'position1': position1,
        'position2': position2,
        'daily_return': daily_return
    }, index=df.index)

    results['cumulative_returns'] = (1 + results['daily_return']).cumprod() - 1

    return results
