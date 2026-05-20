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
close.plot()

# %%
y = close[s2]
X = add_constant(close[s1])

yhat = pd.Series(np.nan, index = y.index) # meas prediction
eps = pd.Series(np.nan, index = y.index) # meas prediction error
Q = pd.Series(np.nan, index = y.index) # meas prediction error variance

P = np.zeros((2,2))
beta = pd.DataFrame(np.nan, index=X.index, columns=X.columns)
delta = 0.0001
Vw = delta / (1 - delta) * np.eye(2) # oovariance of eps in y(t) = x(t) beta(t) + eps
Ve = 0.001 # Cov of w in beta(t) = beta(t - 1) + w

beta.iloc[0] = [0,0] 
for t in range(len(y)):
    if t > 0:
        beta.iloc[t] = beta.iloc[t - 1]
        R = P + Vw # state covariance prediction of eps
    else:
        R = P.copy()
    
    x_t = X.iloc[t].values

    yhat.iloc[t] = x_t @ beta.iloc[t] # measurement prediction of new y
    Q.iloc[t] = x_t @ R @ x_t + Ve # measurement prediction variance

    # observe new y
    e = y.iloc[t] - yhat.iloc[t] # meas prediction error
    eps.iloc[t] = e
    K = R @ x_t / Q.iloc[t] # kalman gain
    beta.iloc[t] = beta.iloc[t] + K * e # state update
    P = R - np.outer(K,  x_t) @ R


spread = y - (X.values * beta.values).sum(axis = 1)
spread = pd.Series(spread, index=y.index)
spread.iloc[10:].plot()

# %%
fig, axes = plt.subplots(2, 1, figsize=(12, 6))
beta['EWA'].iloc[10:].plot(ax=axes[0], title='Slope (hedge ratio)')
beta['const'].iloc[10:].plot(ax=axes[1], title='Intercept')
plt.tight_layout()
plt.show()

# spread should be stationary
adf_stat, pvalue, *_ = adfuller(spread.dropna())
print(f'ADF: {adf_stat:.4f}, p={pvalue:.4f}')

# %%
def get_pnl(y, x, b, e, q):
    hr = b[x.name]

    numLong = pd.Series(np.nan, index=b.index)
    numLong.iloc[0] = 0
    numLong[e < -np.sqrt(q)] = 1
    numLong[e >= -np.sqrt(q)] = 0
    numLong = numLong.ffill()

    numShort = pd.Series(np.nan, index=b.index)
    numShort.iloc[0] = 0
    numShort[e > np.sqrt(q)] = -1
    numShort[e <= np.sqrt(q)] = 0
    numShort = numShort.ffill()

    df = pd.DataFrame({'numUnits': numLong + numShort}, index=b.index)

    df['pos_y'] = df['numUnits'] * y
    df['pos_x'] = -df['numUnits'] * x * hr

    df['PnL'] = (df['pos_y'].shift(1) * y.pct_change() +
                 df['pos_x'].shift(1) * x.pct_change())

    # Return = PnL / gross market val
    gross = df['pos_y'].shift(1).abs() + df['pos_x'].shift(1).abs()
    df['Return']    = df['PnL'] / gross
    df['CumPnL']    = df['PnL'].cumsum()
    df['CumReturn'] = (1 + df['Return']).cumprod()

    return df
# %%
results = get_pnl(close[s2].iloc[20:], close[s1].iloc[20:], beta.iloc[20:], eps.iloc[20:], Q.iloc[20:])
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
