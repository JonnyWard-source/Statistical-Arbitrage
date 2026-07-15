import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from tabulate import tabulate
from signals import (
    score_pair,
    compute_half_life,
    compute_zscore,
    compute_positions
)
from data import stocks

def compute_returns(rets, row, pos, beta, cost=0.001):
    """Calculate strategy returns after transaction costs."""
    a = row["Stock 1"]
    b = row["Stock 2"]

    weight_a = abs(beta) / (1 + abs(beta))
    weight_b = 1 / (1 + abs(beta))

    gross = pos.shift(1) * (weight_b * rets[b] - np.sign(beta) * weight_a * rets[a])

    turnover = pos.shift(1).diff().abs().fillna(0)
    costs = cost * turnover / 2
    net = gross - costs

    return net


# Rolling-window backtest parameters.
train_window = 2 * 252
test_window = 252
step = 252

dates = prices.index

# Run a sequence of rolling-window backtests and store the results.
start = 0
all_portfolios = []
results = []
while True:
    train_start = start
    train_end = start + train_window
    test_end = train_end + test_window

    if test_end >= len(dates):
        break

    start += step
    train_dates = dates[train_start:train_end]
    test_dates = dates[train_end:test_end]

    # Log-price training and test sets for stationarity and spread estimation.
    train = np.log(prices.loc[train_dates])
    test = np.log(prices.loc[test_dates])
    test_rets = prices.loc[test_dates].pct_change()

    # Screen all stock pairs for cointegration and a reasonable half-life.
    pairs = []
    for i in range(len(stocks)):
        X = sm.add_constant(train[stocks[i]])
        for j in range(i + 1, len(stocks)):
            model = sm.OLS(train[stocks[j]], X).fit()

            alpha = model.params["const"]
            beta = model.params[stocks[i]]

            spread = train[stocks[j]] - (beta * train[stocks[i]] + alpha)
            spread = spread.replace([np.inf, -np.inf], np.nan)
            spread = spread.dropna()

            spread_vol = spread.diff().std()
            _, pvalue, _ = coint(train[stocks[j]], train[stocks[i]])

            half_life = compute_half_life(spread)
            score = score_pair(pvalue, half_life, beta, spread)

            if pvalue < 0.01 and 2 < half_life < 60:
                pairs.append({
                    'Stock 1': stocks[i],
                    'Stock 2': stocks[j],
                    'alpha': alpha,
                    'beta': beta,
                    'adf_pvalue': pvalue,
                    'half_life': half_life,
                    'score': score,
                    'spread_vol': spread_vol
                })

    pairs = pd.DataFrame(pairs)
    pairs = pairs.sort_values('score', ascending=False)
    top_k = 7
    pairs = pairs.head(top_k)

    # Build a strategy for each selected pair and combine them into one portfolio.
    all_strats = []
    trade_counts = []
    for _, row in pairs.iterrows():
        z = compute_zscore(train, test, row)
        pos = compute_positions(z)

        trade_counts.append(pos.diff().abs().gt(0).sum())

        beta = pd.Series(row["beta"], index=test.index)
        strat = compute_returns(test_rets, row, pos, beta)
        all_strats.append(strat)

    strats = pd.concat(all_strats, axis=1)

    # Weight each strategy by its score adjusted for spread volatility.
    weights = pairs['score'] / pairs['spread_vol']
    weights /= weights.sum()
    weights = weights.values
    weights = weights[:strats.shape[1]]

    period_portfolio = strats.mul(weights, axis=1).sum(axis=1)

    # Scale the portfolio to a target volatility for comparability across windows.
    target_vol = 0.10
    realised_vol = period_portfolio.std() * np.sqrt(252)
    scale = target_vol / (realised_vol + 1e-8)
    period_portfolio *= scale

    all_portfolios.append(period_portfolio)
    portfolio = pd.concat(all_portfolios).sort_index()

    # Compute portfolio-level performance metrics.
    equity = (1 + portfolio.fillna(0)).cumprod()
    sharpe = portfolio.mean() / portfolio.std() * np.sqrt(252)

    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()

    years = (equity.index[-1] - equity.index[0]).days / 365
    ann_ret = (equity.iloc[-1] ** (1 / years)) - 1

    n_trades = np.sum(trade_counts)

    results.append({
        "Start": test_dates[0].date(),
        "End": test_dates[-1].date(),
        "Annual Return": ann_ret,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Annual Volatility": portfolio.std() * np.sqrt(252),
        "Trades": n_trades,
        "Pairs": len(pairs)
    })

# Summarise the backtest results across rolling windows.
results = pd.DataFrame(results)
summary = pd.DataFrame([{
    "Start": "Average",
    "End": "",
    "Annual Return": results["Annual Return"].mean(),
    "Sharpe": results["Sharpe"].mean(),
    "Max Drawdown": results["Max Drawdown"].mean(),
    "Annual Volatility": results["Annual Volatility"].mean(),
    "Trades": results["Trades"].mean(),
    "Pairs": results["Pairs"].mean()
}])

results = pd.concat([results, summary], ignore_index=True)

display = results.copy()
display["Annual Return"] = (display["Annual Return"] * 100).map("{:.2f}%".format)
display["Annual Volatility"] = (display["Annual Volatility"] * 100).map("{:.2f}%".format)
display["Sharpe"] = display["Sharpe"].map("{:.2f}".format)
display["Max Drawdown"] = (display["Max Drawdown"] * 100).map("{:.2f}%".format)
display["Trades"] = display["Trades"].map("{:.0f}".format)
display["Pairs"] = display["Pairs"].map("{:.0f}".format)

print(tabulate(display, headers='keys', tablefmt='github', showindex=False))
