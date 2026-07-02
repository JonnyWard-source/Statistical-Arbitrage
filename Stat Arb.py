"""Statistical arbitrage pairs-trading prototype.

This script downloads historical price data, screens for cointegrated stock pairs,
builds a mean-reversion signal using rolling z-scores, and evaluates a simple
backtest with transaction-cost adjustments. It is intended as a research project
for quantitative finance CV purposes rather than a production trading system.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import (adfuller, coint)
import statsmodels.api as sm
from tabulate import tabulate

# Universe of stocks used to search for tradable pairs.
stocks = ['KO', 'PEP', 'V', 'MA', 'XOM', 'CVX', 'MSFT', 'AAPL', 'LLOY.L', 'BARC.L', 'TSLA', 'RRL.XC',
          'TSCO.L', 'SHEL.L', 'QHE.L', 'MU', 'BIRG.L', 'MBG.DE', 'TMCO34.SA', 'BMW.DE','AXP', 'PYPL',
          'COF', 'BP', 'TTE', 'COP', 'EOG', 'OXY', 'GOOGL', 'META', 'AMZN', 'ORCL', 'CRM', 'NVDA', 'AMD',
          'QCOM', 'TXN', 'AVGO', 'INTC', 'ASML', 'HSBA.L', 'NWG.L', 'JPM',  'F', 'GM']

# Download daily adjusted close prices and forward-fill any missing values.
prices = yf.download(stocks, start='2020-01-01', end='2025-01-01')['Close']
prices = prices.ffill()


def score_pair(pvalue, half_life, beta, spread):
    """Rank a candidate pair using a simple composite score.

    Higher scores indicate stronger evidence of a mean-reverting relationship.
    """
    return (-2 * np.log(pvalue + 1e-8)
            - abs(half_life - 10) / 10
            - 0.05 * abs(beta))


def compute_half_life(spread):
    """Estimate the half-life of a spread using a simple autoregression."""
    lag = spread.shift(1)
    delta = spread - lag

    df = pd.concat([lag, delta], axis=1).dropna()

    X = sm.add_constant(df.iloc[:, 0])

    model_hl = sm.OLS(df.iloc[:, 1], X).fit()

    beta_hl = model_hl.params.iloc[1]

    if beta_hl >= 0:
        return np.inf

    return -np.log(2) / beta_hl


def compute_rolling_beta(train, test, a, b, window=60):
    """Estimate rolling betas for a pair over a rolling window."""
    data = pd.concat([train, test])

    beta = []
    idx = []

    for i in range(window, len(data)):
        y = data[b].iloc[i-window:i]
        x = sm.add_constant(data[a].iloc[i-window:i])

        model = sm.OLS(y, x).fit()

        beta.append(model.params.iloc[1])
        idx.append(data.index[i])

    beta = pd.Series(beta, index=idx)

    return beta.reindex(data.index).ffill()


def compute_zscore(train, test, row, window=80):
    """Construct a rolling z-score for a pair's spread over the test period."""
    a = row["Stock 1"]
    b = row["Stock 2"]

    beta = row["beta"]
    alpha = row["alpha"]

    train_spread = train[b] - (beta * train[a] + alpha)
    test_spread = test[b] - (beta * test[a] + alpha)

    spread = pd.concat([train_spread, test_spread])

    mean = spread.rolling(window).mean().shift(1)
    std = spread.rolling(window).std().shift(1)

    z = (spread - mean) / std

    return z.loc[test.index]


def compute_positions(z, entry=1.5, exit=0, max_days=30):
    """Generate simple long/short positions from a z-score signal."""
    pos = pd.Series(index=z.index, dtype=float)

    position = 0
    days_in_trade = 0

    for i in range(len(z)):
        if abs(z.iloc[i]) > 5:
            position = 0

        if pd.isna(z.iloc[i]):
            pos.iloc[i] = position
            continue

        if position == 0:
            days_in_trade = 0
            momentum = z.diff()

            if z.iloc[i] > entry and momentum.iloc[i] < 0:
                position = -1
            elif z.iloc[i] < -entry and momentum.iloc[i] > 0:
                position = 1
        else:
            days_in_trade += 1

            if abs(z.iloc[i]) < exit or days_in_trade >= max_days:
                position = 0
                days_in_trade = 0

        pos.iloc[i] = position

    return pos.fillna(0)


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