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

def sig_coint(close, s1, s2, p = 0.1):

    df = pd.DataFrame({s1:close[s1], s2:close[s2]}).dropna()

    # 1. Find optimal hedge ratio stat siginificance

    t_stat, p_value, crit_values = coint(df[s1], df[s2])
    # print(f'CADF t-statistic: {t_stat:.4f}')
    # print(f'p-value: {p_value:.4f}')
    # print(f'Critical values (1%, 5%, 10%): {crit_values}')

    if p_value <= p:
        return [(s1, s2)]
    else:
        return []
    
def get_halflife(spread):
    spread = spread.dropna().values
    delta = np.diff(spread)
    lagged = spread[:-1]
    
    # manually stack constant and lagged level
    X = np.column_stack([np.ones(len(lagged)), lagged])
    
    res = OLS(delta, X).fit()
    lambda_coef = res.params[1]  # params[0]=constant, params[1]=lambda
    
    if lambda_coef >= 0:
        return np.inf
    
    return -np.log(2) / lambda_coef

def mean_rev(close, s1, s2, plot=True):

    df = pd.DataFrame({s1:close[s1], s2:close[s2]}).dropna()

    res_OLS = OLS(df[s1], add_constant(df[s2])).fit()
    const, slope = res_OLS.params
    # print(res_OLS.params)

    # 2. Make portfolio
    df['Spread'] = df[s1] - slope * df[s2]
    halflife = round(get_halflife(df['Spread']))

    if not (2 <= halflife <= 252):
        return f'Halflife {halflife} not tradeable'

    df['MktVal'] = -(df['Spread'] - df['Spread'].rolling(halflife).mean()) / df['Spread'].rolling(halflife).std()
    df['PnL'] = df['MktVal'].shift(1) * df['Spread'].pct_change()
    df['CumPnL'] = df['PnL'].cumsum()

    # if plot and df['CumPnL'].iloc[-1] > 0:
    if plot:
        fig, ax = plt.subplots(figsize=(10,5))
        df['CumPnL'].plot(ax=ax, title=f'{s1}/{s2} PnL')
        ax2 = ax.twinx()
        df['PnL'].plot(ax=ax2, alpha=0.3, linewidth=1, style='--')
        ax.set_ylabel('CumPnL')
        ax2.set_ylabel('PnL')
        plt.show()

    return df

# %%
tickers = ['VOO', 'AAPL', 'SMH', 'TSM', 'AMD', 'BOTZ', 'NLR']
tickers += ['MSFT']
tickers += ['LMT', 'RTX', 'GD', 'NOC']
tickers += ['AMZN', 'GOOG', 'TSLA', 'JPM', 'META', 'ASML', 'V']
tickers += ['NVDA']
tickers += ['GLD', 'SLV', 'XLE']
tickers += ['EWJ', 'EWP', 'VPL'] 
tickers += ['VTI', 'VXUS', 'NEE', 'JNJ', 'UNH', 'PG', 'COST', 'WELL', 'SCCO', 'RIO']

START = '2020-05-12'
END   = '2026-05-12' 

def load_data(ticker):
    df = yf.Ticker(ticker).history(start=START, end=END)['Close']
    return df

close = pd.DataFrame({ticker:load_data(ticker) for ticker in tickers})

pairs = []
for i, t1 in enumerate(tickers):
    for t2 in tickers[i + 1:]:        
        pairs += sig_coint(close, t1, t2, p = 0.01)
# %%
for s1, s2 in pairs:
    mean_rev(close, s1, s2, True)