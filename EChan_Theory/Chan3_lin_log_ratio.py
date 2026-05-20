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
START = '2006-05-24'
END   = '2012-04-09' 

tickers = ['GLD', 'USO']

def load_data(ticker):
    df = yf.Ticker(ticker).history(start=START, end=END)['Close']
    return df

close = pd.DataFrame({ticker:load_data(ticker) for ticker in tickers})
close[['log_GLD', 'log_USO']] = np.log(close)
# close.plot()


# %%
# # # # # # # # 
# Rolling hedge ratios
# # # # # # # # 
def find_hedge(s1, s2):
    res_OLS = OLS(s1, add_constant(s2)).fit()
    _, slope = res_OLS.params

    return slope

lookback = 20
hedgeRatio = []

hedgeRatio = np.array([
    find_hedge(close['USO'].iloc[t - lookback : t].dropna(), 
               close['GLD'].iloc[t - lookback : t].dropna()) 
               for t in range(lookback, len(close))
            ])


spread = close['USO'].iloc[lookback : ].values - hedgeRatio * close['GLD'].iloc[lookback : ].values
spread = pd.Series(spread, index = close.index[lookback:])
spread.plot()

# %%

# # # # # # # # 
# Finding z scores
# # # # # # # # 


def get_PnL(spread, hedgeRatio, lookback):
    # align prices to whatever index spread has
    uso = close['USO'].reindex(spread.index)
    gld = close['GLD'].reindex(spread.index)
    hr  = pd.Series(hedgeRatio, index=spread.index)

    # units of each spread
    df = pd.DataFrame({
        'numUnits': -(spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()
    }, index=spread.index)

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

results = get_PnL(spread, hedgeRatio, lookback)
apr    = results['Return'].mean() * 252
sharpe = results['Return'].mean() / results['Return'].std() * np.sqrt(252)
print(f'APR:    {apr:.2%}')
print(f'Sharpe: {sharpe:.2f}')

# results['CumPnL'].plot()
results['CumReturn'].plot()
# %%
# # # # # # # # 
# Rolling log hedge ratios
# # # # # # # # 
lookback = 20
log_hedgeRatio = []

log_hedgeRatio = np.array([
    find_hedge(close['log_USO'].iloc[t - lookback : t].dropna(), 
               close['log_GLD'].iloc[t - lookback : t].dropna()) 
               for t in range(lookback, len(close))
            ])


log_spread = close['log_USO'].iloc[lookback : ].values - log_hedgeRatio * close['log_GLD'].iloc[lookback : ].values
log_spread = pd.Series(log_spread, index = close.index[lookback:])
# log_spread.plot()

log_results = get_PnL(log_spread, log_hedgeRatio, lookback)
apr    = log_results['Return'].mean() * 252
sharpe = log_results['Return'].mean() / log_results['Return'].std() * np.sqrt(252)
print(f'APR:    {apr:.2%}')
print(f'Sharpe: {sharpe:.2f}')

# log_results['CumPnL'].plot()
log_results['CumReturn'].plot()
# %%

# # # # # # # # 
# Using Ratio USO / GLD
# # # # # # # # 

ratio_spread = close['USO'] / close['GLD']
# close['Ratio'].plot()

ratio_results = get_PnL(ratio_spread, np.ones(len(ratio_spread)), lookback)
apr    = ratio_results['Return'].mean() * 252
sharpe = ratio_results['Return'].mean() / ratio_results['Return'].std() * np.sqrt(252)
print(f'APR:    {apr:.2%}')
print(f'Sharpe: {sharpe:.2f}')

# results['CumPnL'].plot()
ratio_results['CumReturn'].plot()