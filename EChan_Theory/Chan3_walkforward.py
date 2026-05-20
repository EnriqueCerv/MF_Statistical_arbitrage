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

def master(tickers, start, end, Kalman, entryScore, exitScore, lookback, cost_bps=20, plot=False):
    y = load_data(tickers[0], start, end, plot)
    x = load_data(tickers[1], start, end, plot)

    if Kalman:
        beta, eps, Q = fit_Kalman(y, x)
        hedge_ratios = beta[x.name]
        spread = find_spread_Kalman(y, x, beta, lookback, plot)
        positions = find_positions_Kalman(eps.iloc[lookback:], Q.iloc[lookback:])
        # positions = find_positions_Kalman(y.iloc[lookback:], x.iloc[lookback:], eps.iloc[lookback:], Q.iloc[lookback:])
    else:
        hedge_ratios = find_hedge_ratios(y, x, lookback)
        spread = find_spread(y, x, hedge_ratios, plot)
        z_scores = find_z_scores(spread, lookback)
        positions = find_positions(z_scores, entryScore, exitScore)
    
    results = find_pnl(y, x, hedge_ratios, positions, cost_bps)
    apr    = results['Return'].mean() * 252
    sharpe = results['Return'].mean() / results['Return'].std() * np.sqrt(252)

    if True:
        # Figure 1: Cumulative Return + daily Return
        fig1, ax1 = plt.subplots(figsize=(12, 5))

        color1 = 'tab:blue'
        ax1.plot(results['CumReturn'], color=color1, label='Cumulative Return')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Return', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)

        ax1b = ax1.twinx()
        color2 = 'tab:orange'
        ax1b.plot(results['Return'], color=color2, alpha=0.5, label='Daily Return')
        ax1b.set_ylabel('Daily Return', color=color2)
        ax1b.tick_params(axis='y', labelcolor=color2)

        plt.title(f'Returns — Ann. {apr:.2%} | Sharpe {sharpe:.2f}')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1b.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()
        plt.show()


        # Figure 2: Cumulative PnL + daily PnL
        fig2, ax2 = plt.subplots(figsize=(12, 5))

        ax2.plot(results['CumPnL'], color=color1, label='Cumulative PnL')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Cumulative PnL ($)', color=color1)
        ax2.tick_params(axis='y', labelcolor=color1)

        ax2b = ax2.twinx()
        ax2b.plot(results['PnL'], color=color2, alpha=0.5, label='Daily PnL')
        ax2b.set_ylabel('Daily PnL ($)', color=color2)
        ax2b.tick_params(axis='y', labelcolor=color2)

        plt.title('PnL')

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()
        plt.show()
    
    return results



# %%

# # # # # # # # # #
# Load Data
# # # # # # # # # #
def load_data(ticker, start, end, plot):
    df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)['Close']
    df.name = ticker
    
    if plot:
        df.plot()

    return df


# %%
# # # # # # # # # #
# For Kalman
# # # # # # # # # #

def fit_Kalman(y, x):
    X = add_constant(x)

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

    return beta, eps, Q


def find_spread_Kalman(y, x, beta, lookback, plot):
    spread = y - (add_constant(x).values * beta.values).sum(axis = 1)
    spread = pd.Series(spread, index=y.index)

    if plot:
        spread.iloc[lookback:].plot()
    
    return spread

def find_positions_Kalman(eps, Q):
    numLong = pd.Series(np.nan, index=eps.index)
    numLong.iloc[0] = 0
    numLong[eps < -np.sqrt(Q)] = 1
    numLong[eps >= -np.sqrt(Q)] = 0
    numLong = numLong.ffill()

    numShort = pd.Series(np.nan, index=eps.index)
    numShort.iloc[0] = 0
    numShort[eps > np.sqrt(Q)] = -1
    numShort[eps <= np.sqrt(Q)] = 0
    numShort = numShort.ffill()

    positions = pd.DataFrame({'numUnits' : numLong + numShort}, index=eps.index)
    return positions

# %%
# # # # # # # # # #
# For fixed bollinger bands
# # # # # # # # # #


def find_hedge_ratios(y, x, lookback):
    hedge_ratios = [find_hedge(
            y.iloc[t - lookback : t].dropna(), 
            x.iloc[t - lookback : t].dropna()) 
            for t in range(lookback, len(y))
            ]
    
    return pd.Series(hedge_ratios, index=y.index[lookback:], name='hedge_ratio')

def find_hedge(y, x):
    res_OLS = OLS(y, add_constant(x)).fit()
    _, slope = res_OLS.params

    return slope


def find_spread(y, x, hedge_ratios, plot=False):
    y_aligned = y.reindex(hedge_ratios.index)
    x_aligned = x.reindex(hedge_ratios.index)
    spread = y_aligned - x_aligned * hedge_ratios

    if plot:
        spread.plot()

    return spread

def find_z_scores(spread, lookback):
    z = (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()
    return z

def find_positions(z_scores, entryScore, exitScore):
    numLong = pd.Series(np.nan, index=z_scores.index)
    numLong.iloc[0] = 0
    numLong[z_scores < -entryScore] = 1
    numLong[z_scores >= -exitScore] = 0
    numLong = numLong.ffill()

    numShort = pd.Series(np.nan, index=z_scores.index)
    numShort.iloc[0] = 0
    numShort[z_scores > entryScore] = -1
    numShort[z_scores <= exitScore] = 0
    numShort = numShort.ffill()

    positions = pd.DataFrame({'numUnits' : numLong + numShort}, index=z_scores.index)
    return positions


# %%
# # # # # # # # # #
# For general pnl
# # # # # # # # # #
def find_pnl(y, x, hedge_ratios, positions, cost_bps):
    positions = positions.copy()
    positions['pos_y'] = positions['numUnits'] * y
    positions['pos_x'] = -positions['numUnits'] * x * hedge_ratios

    positions['PnL'] = (positions['pos_y'].shift(1) * y.pct_change() +
                 positions['pos_x'].shift(1) * x.pct_change())
    
    # turnover cost
    turnover = (positions['pos_y'].diff().abs().fillna(positions['pos_y'].abs()) +
                positions['pos_x'].diff().abs().fillna(positions['pos_x'].abs()))
    positions['Cost'] = turnover * cost_bps * 1e-4
    positions['PnL']  = positions['PnL'] - positions['Cost']

    # Return = PnL / gross market val
    gross = positions['pos_y'].shift(1).abs() + positions['pos_x'].shift(1).abs()
    positions['Return']    = positions['PnL'] / gross
    positions['CumPnL']    = positions['PnL'].cumsum()
    positions['CumReturn'] = (1 + positions['Return']).cumprod()

    return positions
# %%
START = '2006-04-04'
END   = '2012-04-09' 
s1, s2 = 'EWC', 'EWA'

# START = '2016-04-04'
# END   = '2026-04-09' 
# s1, s2 = 'GLD', 'RIO'

tickers = [s1, s2]

master(tickers, START, END, True, 1, 0.5, 20, cost_bps = 5, plot = False)
master(tickers[::-1], START, END, True, 1, 0.5, 20, cost_bps = 5, plot = False)
# %%
