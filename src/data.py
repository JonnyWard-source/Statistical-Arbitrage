import yfinance as yf

# Universe of stocks used to search for tradable pairs.
stocks = ['KO', 'PEP', 'V', 'MA', 'XOM', 'CVX', 'MSFT', 'AAPL', 'LLOY.L', 'BARC.L', 'TSLA', 'RRL.XC',
          'TSCO.L', 'SHEL.L', 'QHE.L', 'MU', 'BIRG.L', 'MBG.DE', 'TMCO34.SA', 'BMW.DE','AXP', 'PYPL',
          'COF', 'BP', 'TTE', 'COP', 'EOG', 'OXY', 'GOOGL', 'META', 'AMZN', 'ORCL', 'CRM', 'NVDA', 'AMD',
          'QCOM', 'TXN', 'AVGO', 'INTC', 'ASML', 'HSBA.L', 'NWG.L', 'JPM',  'F', 'GM']

def download_prices(start='2020-01-01', end='2025-01-01'):
  """Download daily close prices and forward-fill any missing values."""
  prices = yf.download(stocks, start=start, end=end)["Close"]
  return prices.ffill()
