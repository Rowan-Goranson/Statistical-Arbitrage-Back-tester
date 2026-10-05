
import pandas as pd
import data_loader
from itertools import combinations
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#TODOS
'''
- Live?
- Rolling regression to update hedge ratio over time?
- set entry/exit zscores, liquitidy/voliatility pricing factor
'''

'''
- Notes: stop-losses near max historical drawdown (10-20% max drawdowns)
- Portfolio Optimization: risk partiy, and sharpe-ratio maximization
'''
df = pd.read_csv('data/price_data.csv', index_col=0, parse_dates = True)


correlation_matrix = df.corr()
strong_pairs = [
    (s1, s2)
    for s1, s2 in combinations(df.columns,2)
    if correlation_matrix.loc[s1, s2] > 0.8
]

cointegrated_pairs = []
for s1, s2 in strong_pairs:
    score, pvalue, _ = coint(df[s1], df[s2])
    if pvalue < 0.01:
        cointegrated_pairs.append((s1, s2, pvalue))

#sort for lowest p-values up front
cointegrated_pairs.sort(key=lambda x: x[2])

#for s1, s2, pvalue in cointegrated_pairs[:10]:
    #print(f'{s1} and {s2} -> p-value: {pvalue:.5f}')

from backtest import calculate_spread_and_zscore, backtest_pair

def plot_backtest_results(results, s1, s2):
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(results.index, results['spread'])
    axs[0].set_title(f'Spread between {s1} and {s2}')

    axs[1].plot(results.index, results['zscore'])
    axs[1].axhline(2.0, color='red', linestyle='--')
    axs[1].axhline(-2.0, color='green', linestyle='--')
    axs[1].set_title('Z-Score')

    axs[2].plot(results.index, results['cumulative_returns'])
    axs[2].set_title('Cumulative Returns')

    plt.tight_layout()
    plt.show()

'''
s1, s2 = 'MA', 'WMT'
results = backtest_pair(df, s1, s2)
plot_backtest_results(results, s1, s2)
'''


def batch_backtest(df, cointegrated_pairs, entry_z=2.0, exit_z=0.5, liquidity_factor=0.0005, volatility_factor=0.002):
    summary = []

    for s1, s2, pvalue in cointegrated_pairs:
        try:
            results = backtest_pair(df, s1, s2, entry_z=entry_z, exit_z=exit_z,
                                    liquidity_factor=liquidity_factor, volatility_factor=volatility_factor)
            daily_returns = results['daily_return']
            cum_return = results['cumulative_returns'].iloc[-1]
            sharpe = results['daily_return'].mean() / results['daily_return'].std() * (252**0.5)
            annual_return = daily_returns.mean() * 252
            annual_volatility = daily_returns.std() * (252**0.5)
            cumulative = results['cumulative_returns']
            peak = cumulative.cummax()
            drawdown = (cumulative - peak)
            max_drawdown = drawdown.min()
            
            summary.append((s1, s2, pvalue, cum_return, sharpe, annual_return, annual_volatility, max_drawdown))

        except Exception as e:
            print(f"Error backtesting {s1}-{s2}: {e}")

    summary_df = pd.DataFrame(summary, columns=['Stock1', 'Stock2', 'p-value', 'Cumulative Return', 'Sharpe Ratio', 'Annual Return', 'Annual Volatility', 'Max Drawdown'])
    
    return summary_df.sort_values(by='Sharpe Ratio', ascending=False)



#portfolio + graph



summary_df = batch_backtest(df, cointegrated_pairs)
top10 = summary_df.head(10)
top10.to_csv("top10_pairs_results.csv", index=False)

# Build portfolio based on top 10 pairs
portfolio_returns = pd.DataFrame(index=df.index)

# Backtest each pair and store their returns
for i, row in top10.iterrows():
    s1, s2 = row['Stock1'], row['Stock2']
    results = backtest_pair(df, s1, s2)
    pair_name = f"{s1}-{s2}"
    portfolio_returns[pair_name] = results['daily_return']

pair_volatility = portfolio_returns.std()
inv_vol = 1/ pair_volatility
risk_parity_weights = inv_vol / inv_vol.sum()
print("\nRisk Parity Weights:")
print(risk_parity_weights)

leverage_factor = 5
# Compute weighted portfolio returns and cumulative returns
portfolio_returns['Total'] = portfolio_returns.dot(risk_parity_weights)
portfolio_returns['Total'] *= leverage_factor
portfolio_returns['Cumulative'] = (1+portfolio_returns['Total']).cumprod()-1

# Plot
fig, ax1 = plt.subplots(figsize=(14, 7))
spy_cumulative = data_loader.load_spy_cumulative(df.index.min().strftime('%Y-%m-%d'), df.index.max().strftime('%Y-%m-%d'))
spy_cumulative = spy_cumulative.reindex(portfolio_returns.index, method='ffill')

# Plot Portfolio & SPY Returns
plt.plot(portfolio_returns.index, spy_cumulative, label='S&P 500 (SPY) Cumulative Return', color = 'green', linewidth=2, linestyle='-')
plt.plot(portfolio_returns.index, portfolio_returns['Cumulative'], color = 'blue', label='Portfolio Cumulative Return', linewidth=2)

# Breakeven line
plt.axhline(0, color='gray', linestyle='--', linewidth=1)

# Drawdown
portfolio_cum = portfolio_returns['Cumulative']
peak = portfolio_cum.cummax()
drawdown = (portfolio_cum - peak)
plt.fill_between(portfolio_returns.index, drawdown, 0, color='red', alpha=0.2, label='Drawdown')

# Rolling volatility on secondary y-axis
rolling_vol = portfolio_returns['Total'].rolling(window=30).std() * (252**0.5)
ax2 = plt.gca().twinx()
ax2.plot(portfolio_returns.index, rolling_vol, color='orange', linestyle='--', label='Rolling Volatility (annualized)')
ax2.set_ylabel('Annualized Volatility', color='orange')

# Titles and Labels
plt.title('Portfolio vs. S&P 500: Cumulative Returns, Drawdown & Volatility')
plt.xlabel('Date')
plt.ylabel('Cumulative Return', color='blue')
plt.grid(True)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

#plt.show()



#sensitivty test
liquidity_factors = [0.0001, 0.0005, 0.001, 0.002]
volatility_factors = [0.001, 0.002, 0.005, 0.01]

results = []

for lf in liquidity_factors:
    for vf in volatility_factors:
        print(f"\nTesting with liquidity_factor={lf}, volatility_factor={vf}")

        # Re-run portfolio backtest with these slippage factors
        portfolio_returns = pd.DataFrame(index=df.index)
        for _, row in top10.iterrows():
            s1, s2 = row['Stock1'], row['Stock2']
            res = backtest_pair(df, s1, s2,
                                liquidity_factor=lf,
                                volatility_factor=vf)
            pair_name = f"{s1}-{s2}"
            portfolio_returns[pair_name] = res['daily_return']

        portfolio_returns['Total'] = portfolio_returns.dot(risk_parity_weights)
        portfolio_returns['Total'] *= leverage_factor
        portfolio_returns['Cumulative'] = (1 + portfolio_returns['Total']).cumprod() - 1

        # Compute portfolio Sharpe & Return
        daily = portfolio_returns['Total'].dropna()
        sharpe = daily.mean() / daily.std() * (252**0.5)
        cum_return = portfolio_returns['Cumulative'].iloc[-1]

        results.append({
            'Liquidity Factor': lf,
            'Volatility Factor': vf,
            'Sharpe Ratio': sharpe,
            'Cumulative Return': cum_return
        })

# display

sensitivity_df = pd.DataFrame(results)
print(sensitivity_df.sort_values(by='Sharpe Ratio', ascending=False))

pivot_table = sensitivity_df.pivot(index='Liquidity Factor', columns='Volatility Factor',values='Sharpe Ratio')

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Portfolio Sharpe Ratio Sensitivity to Slippage")
plt.ylabel("Liquidity Factor")
plt.xlabel("Volatility Factor")
plt.show()
