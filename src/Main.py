from data import download_prices
from backtest import run_backtest

prices = download_prices()

results = run_backtest(prices)

print(results)
