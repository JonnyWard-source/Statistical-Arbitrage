from data import download_prices
from backtest import run_backtest

def main():
    prices = download_prices()
    results = run_backtest(prices)
    print(results)

if __name__ == "__main__":
    main()
