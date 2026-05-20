# %%
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# %%
START = '2022-05-12'
END   = '2026-05-12' 

def load_data(ticker):
    df = yf.Ticker(ticker).history(start=START, end=END)['Close']
    return df

tickers = ['XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLU', 'XLB', 'XLP', 'XLY']
# tickers = ['EWH', 'EWS', 'EWY']
tickers = ['EWJ', 'EWH', 'EWS', 'EWY', 'EWA', 'EWC', 'EWG', 'EWQ', 'EWI', 'EWP', 'EWU', 'EWZ', 'EWW', 'EWT', 'EIDO', 'EWM']
tickers = ['EURUSD=X', 'GBPUSD=X', 'DKKUSD=X', 'SEKUSD=X', 'NOKUSD=X', 'ISKUSD=X']
# tickers = [
#     # 'JPYUSD=X',  # Japanese Yen
#     'AUDUSD=X',  # Australian Dollar
#     'NZDUSD=X',  # New Zealand Dollar
#     # 'HKDUSD=X',  # Hong Kong Dollar
#     'SGDUSD=X',  # Singapore Dollar
#     # 'CNYUSD=X',  # Chinese Yuan (onshore)
#     # 'CNHUSD=X',  # Chinese Yuan (offshore)
#     'KRWUSD=X',  # Korean Won
#     'TWDUSD=X',  # Taiwan Dollar
#     # 'INRUSD=X',  # Indian Rupee
#     'THBUSD=X',  # Thai Baht
#     # 'MYRUSD=X',  # Malaysian Ringgit
#     'IDRUSD=X',  # Indonesian Rupiah
#     # 'PHPUSD=X',  # Philippine Peso
# ]

df = pd.DataFrame({ticker:load_data(ticker) for ticker in tickers})

plt.figure(figsize=(10,6))
for ticker in tickers:
    plt.plot(df[ticker], label=ticker)
plt.legend()
plt.show()


# %%
result = coint_johansen(df, 0, 1)

print("Trace Statistic: \n", result.lr1)
print("Critical Values (90%, 95%, 99%): \n", result.cvt)
print("Eigen Statistic: \n", result.lr2)
print("Critical Values (90%, 95%, 99%): \n", result.cvm)
# %%
# Find eigenvectors:

evec = result.evec[:, 0]
evec
df['Spread'] = df[tickers] @ evec
plt.figure(figsize=(10,6))
plt.plot(df['Spread'], label = 'Spread')
plt.legend()
plt.show()
# %%
# Trade on the spread
# 1. Find halflife
def get_halflife(spread):
    X = add_constant(spread.iloc[:-1].values)
    y = spread.diff().dropna().values

    res = OLS(y, X).fit()
    constant_, lambda_ = res.params
    
    return -np.log(2)/lambda_ if lambda_ < 0 else np.inf

spread = df['Spread'].dropna()
halflife = get_halflife(spread)

# 2. Set trades:
lookback = round(halflife) if halflife < np.inf else int(2)
df['Trade'] = -(spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()
df['PnL'] = df['Trade'].shift(1) * df['Spread'].pct_change()
df['CumPnL'] = df['PnL'].cumsum()

ax = df.plot(y='CumPnL', use_index=True, title='CumPnL', figsize=(10,5))
df.plot(y='PnL', ax=ax, secondary_y=True, alpha=0.3, linewidth=1, style='--')
ax.set_ylabel('CumPnL')
ax.right_ax.set_ylabel('PnL')

# %%
df['Position'] = df['Trade'].clip(-1, 1)
df['Return']   = df['Position'].shift(1) * df['Spread'].pct_change()
df['CumReturn'] = (1 + df['Return']).cumprod()

n_days = len(df['Return'].dropna())
total_return = df['CumReturn'].iloc[-1] - 1

ann_return = (1 + total_return) ** (252 / n_days) - 1
ann_vol    = df['Return'].std() * np.sqrt(252)
sharpe     = ann_return / ann_vol

print(f'Total return:      {total_return:.2%}')
print(f'Annualised return: {ann_return:.2%}')
print(f'Annualised vol:    {ann_vol:.2%}')
print(f'Sharpe ratio:      {sharpe:.2f}')

plt.figure(figsize=(12, 6))

# 3. Cumulative return
plt.plot(df['CumReturn'], label='Cumulative Return')
plt.title(f'Cumulative Return — Ann. {ann_return:.2%} | Sharpe {sharpe:.2f}')
plt.legend()

plt.tight_layout()
plt.show()
# %%
