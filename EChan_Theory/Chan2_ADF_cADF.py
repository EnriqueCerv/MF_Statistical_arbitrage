# %%
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
# %%
START = '2024-05-09'
END   = '2026-05-09' 

df = yf.Ticker('AAPL')
df = df.history(start=START, end=END)
df['Return'] = df['Close'].pct_change()
df = df.dropna()
ax = df.plot(y='Close', use_index=True, title='EWS', figsize=(10,5))
df.plot(y='Return', ax=ax, secondary_y=True, alpha=0.3, linewidth=1, style='--')
ax.set_ylabel('Price')
ax.right_ax.set_ylabel('Return')
# %%
from statsmodels.tsa.stattools import adfuller
result_price = adfuller(df['Close'])
result_ret = adfuller(df['Return'])

l = result_price[0]
halflife = -np.log(2) / l

print(f'ADF Statistic: {result_price[0]}')
print(f'p-value: {result_price[1]}')
print(f'halflife: {halflife}')

# Basically mean reverting with probability 1

# %%
# Strategy if not mean-reverting
halflife = round(halflife)
df['MktVal'] = -(df['Close'] - df['Close'].rolling(halflife).mean()) / df['Close'].rolling(halflife).std()
df['PnL'] = df['MktVal'].shift(1) * df['Return']
df['CumPnL'] = df['PnL'].cumsum()

# %%

## WITH COINTEGRATION
EWS = yf.Ticker('EWS').history(start=START, end=END)['Close']
EWU = yf.Ticker('EWU').history(start=START, end=END)['Close']
df = pd.DataFrame({'EWS':EWS, 'EWU':EWU}).dropna()

# 1. Run adf individually (should not be stationary)
from statsmodels.tsa.stattools import adfuller

res_EWS = adfuller(EWS)
print(f'EWS p-val: {res_EWS[1]}')
res_EWU = adfuller(EWU)
print(f'EWU p-val: {res_EWU[1]}')

# 2. Find optimal hedge ratio
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

X = add_constant(df['EWU'])
ols_res = OLS(df['EWS'], X).fit()
const, slope = ols_res.params
plt.scatter(const + slope * df['EWU'], df['EWS'])

# 3. Make portfolio
df['Spread'] = df['EWS'] - slope * df['EWU']

# 4. Test stationarity of new portfolio
res = adfuller(df['Spread'])
l = res[0]
halflife = -np.log(2) / l
print(f'ADF Statistic: {res[0]}')
print(f'p-value: {res[1]}')
print(f'halflife: {halflife}')

# 5. Make strategy if it is mean reverting
halflife = 2
df['MktVal'] = -(df['Spread'] - df['Spread'].rolling(halflife).mean()) / df['Spread'].rolling(halflife).std()
df['PnL'] = df['MktVal'].shift(1) * df['Spread'].pct_change()
df['CumPnL'] = df['PnL'].cumsum()

ax = df.plot(y='PnL', use_index=True, title='PnL', figsize=(10,5))
df.plot(y='CumPnL', ax=ax, secondary_y=True, alpha=0.3, linewidth=1, style='--')
ax.set_ylabel('PnL')
ax.right_ax.set_ylabel('CumPnL')
# %%


START = '2006-04-26'
END   = '2012-04-09' 


## WITH COINTEGRATION directly
stock1, stock2 = 'EWC', 'EWA'
df1 = yf.Ticker(stock1).history(start=START, end=END)['Close']
df2 = yf.Ticker(stock2).history(start=START, end=END)['Close']
plt.plot(df1)
plt.plot(df2)
plt.show()
df = pd.DataFrame({stock1:df1, stock2:df2}).dropna()

# 1. Run adf individually (should not be stationary)
from statsmodels.tsa.stattools import adfuller

res_stock1 = adfuller(df1)
print(f'stock1 p-val: {res_stock1[1]}')
res_stock2 = adfuller(df2)
print(f'stock2 p-val: {res_stock2[1]}')

# 2. Find optimal hedge ratio
from statsmodels.tsa.stattools import coint

t_stat, p_value, crit_values = coint(df[stock1], df[stock2])
# halflife = -np.log(2) / t_stat
print(f'CADF t-statistic: {t_stat:.4f}')
print(f'p-value: {p_value:.4f}')
print(f'Critical values (1%, 5%, 10%): {crit_values}')
# print(f'Halflife: {halflife}')

if p_value <= 0.10:
    res_OLS = OLS(df[stock1], add_constant(df[stock2])).fit()
    const, slope = res_OLS.params
    print(res_OLS.params)


    plt.scatter(const + slope * df[stock2], df[stock1])
    plt.show()

    # 3. Make portfolio
    df['Spread'] = df[stock1] - slope * df[stock2]
    df['Spread'].plot()
    adf_res = adfuller(df['Spread'])
    halflife = -np.log(2) / adf_res[0]
    print(f'Halflife: {halflife}')

    # 4. Make strategy if it is mean reverting
    halflife = 2
    df['MktVal'] = -(df['Spread'] - df['Spread'].rolling(halflife).mean()) / df['Spread'].rolling(halflife).std()
    df['PnL'] = df['MktVal'].shift(1) * df['Spread'].pct_change()
    df['CumPnL'] = df['PnL'].cumsum()

    ax = df.plot(y='CumPnL', use_index=True, title='CumPnL', figsize=(10,5))
    df.plot(y='PnL', ax=ax, secondary_y=True, alpha=0.3, linewidth=1, style='--')
    ax.set_ylabel('CumPnL')
    ax.right_ax.set_ylabel('PnL')

