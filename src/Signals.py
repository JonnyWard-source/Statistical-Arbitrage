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
