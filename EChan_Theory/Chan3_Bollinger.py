# %%
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import coint
# %%
# # # # # # # # 
# Load data
# # # # # # # #  
START = '2006-04-04'
END   = '2012-04-09' 

s1, s2 = 'EWA', 'EWC'
tickers = [s1, s2]

def load_data(ticker):
    df = yf.Ticker(ticker).history(start=START, end=END)['Close']
    return df

close = pd.DataFrame({ticker:load_data(ticker) for ticker in tickers})
# close[['log_GLD', 'log_USO']] = np.log(close)
# close.plot()


# %%
# # # # # # # # 
# Implement bollinger bands
# # # # # # # # 
def find_hedge(s1, s2):
    res_OLS = OLS(s1, add_constant(s2)).fit()
    _, slope = res_OLS.params

    return slope

lookback = 20
hedgeRatio = []

hedgeRatio = np.array([
    find_hedge(close[s2].iloc[t - lookback : t].dropna(), 
               close[s1].iloc[t - lookback : t].dropna()) 
               for t in range(lookback, len(close))
            ])

spread = close[s2].iloc[lookback : ].values - hedgeRatio * close[s1].iloc[lookback : ].values
spread = pd.Series(spread, index = close.index[lookback:])
spread.plot()



def find_z_score(spread, lookback):
    z = (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()
    return z  # just a Series, no DataFrame, no dropna


z_scores = find_z_score(spread, lookback)
z_scores

# %%
def get_pnl(spread, hedgeRatio, lookback, entryScore, exitScore):
    # align prices to whatever index spread has
    uso = close[s2].reindex(spread.index)
    gld = close[s1].reindex(spread.index)
    hr  = pd.Series(hedgeRatio, index=spread.index)

    # get z-scores
    z = find_z_score(spread, lookback)

    # Get numUnits
    # longs: enter when z < -entry, exit when z >= -exit
    numLong = pd.Series(np.nan, index=z.index)
    numLong.iloc[0] = 0
    numLong[z < -entryScore] = 1
    numLong[z >= -exitScore] = 0
    numLong = numLong.ffill()

    # shorts: enter when z > entry, exit when z <= exit
    numShort = pd.Series(np.nan, index=z.index)
    numShort.iloc[0] = 0
    numShort[z > entryScore] = -1
    numShort[z <= exitScore] = 0
    numShort = numShort.ffill()


    # units of each spread
    df = pd.DataFrame({'numUnits': numLong + numShort}, index=spread.index)

    # units of each item in spread
    df['pos_USO'] =  df['numUnits'] * uso
    df['pos_GLD'] = -df['numUnits'] * hr * gld

    # PnL = lagged dollar positions * daily returns, summed
    df['PnL'] = (df['pos_USO'].shift(1) * uso.pct_change() +
                 df['pos_GLD'].shift(1) * gld.pct_change())

    # Return = PnL / gross market val
    gross = df['pos_USO'].shift(1).abs() + df['pos_GLD'].shift(1).abs()
    df['Return']    = df['PnL'] / gross
    df['CumPnL']    = df['PnL'].cumsum()
    df['CumReturn'] = (1 + df['Return']).cumprod()

    return df
    


# %%
results = get_pnl(spread, hedgeRatio, lookback, 1, 0)
apr    = results['Return'].mean() * 252
sharpe = results['Return'].mean() / results['Return'].std() * np.sqrt(252)
print(f'APR:    {apr:.2%}')
print(f'Sharpe: {sharpe:.2f}')

plt.plot(results['CumReturn'], label='Cumulative Return')
plt.title(f'Cumulative Return — Ann. {apr:.2%} | Sharpe {sharpe:.2f}')
plt.legend()

plt.tight_layout()
plt.show()
# %%
