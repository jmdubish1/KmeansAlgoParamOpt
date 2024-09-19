import numpy as np
import pandas as pd
from typing import Callable


def create_atr(data, n=12):
    high_low = data['High'] - data['Low']
    high_prev_close = np.abs(data['High'] - data['Close'].shift(1))
    low_prev_close = np.abs(data['Low'] - data['Close'].shift(1))
    true_range = np.maximum(high_low, high_prev_close)
    true_range = np.maximum(true_range, low_prev_close)

    # Calculate Average True Range (ATR)
    atr = np.zeros_like(data['Close'])
    atr[n - 1] = np.mean(true_range[:n])  # Initial ATR calculation

    for i in range(n, len(data['Close'])):
        atr[i] = ((atr[i - 1] * (n - 1)) + true_range[i]) / n

    return atr


def standardize_dailydata(daily_df, window_size=23):
    # Standardize by day over day % change
    for c in ['ATR_day']:
        daily_df[c] = daily_df[c]/daily_df['Close']*100

    # Standardize by difference from last window_size day average
    for c in ['Vol', 'OpenInt', 'VolAvg', 'OI']:
        c_mean = daily_df[c].rolling(window=window_size, min_periods=1).mean()
        daily_df[c] = daily_df[c]/c_mean

    return daily_df


def standardize_intradata(daily_df, sec_name, window_size=23):
    # Standardize by day over day % change
    for c in ['ATR_int']:
        daily_df[c] = daily_df[c]/daily_df[f'{sec_name}_Close']
        c_mean = daily_df[c].rolling(window=window_size, min_periods=1).mean()
        daily_df[c] = daily_df[c]/c_mean

    # Standardize by difference from last 23-day average
    for c in ['Vol_int']:
        c_mean = daily_df[c].rolling(window=window_size, min_periods=1).mean()
        daily_df[c] = daily_df[c]/c_mean

    return daily_df


def rsi(series, period):
    delta = series.pct_change()
    delta = delta.dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta <= 0]
    u[u.index[period-1]] = np.mean(u[:period]) #first value is sum of avg gains
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean(d[:period]) #first value is sum of avg losses
    d = d.drop(d.index[:(period-1)])
    rs = (pd.DataFrame.ewm(u, com=period-1, adjust=False).mean() /
          pd.DataFrame.ewm(d, com=period-1, adjust=False).mean())
    return 100 - 100 / (1 + rs)


def create_rsi(daily_df, other_sec, sec_name):
    securities = other_sec + [sec_name]
    rsi_cols = []
    for sec in securities:
        rsi_k_name = f'{sec}_RSI_k'
        rsi_d_name = f'{sec}_RSI_d'
        daily_df[rsi_k_name] = np.hstack([[0]*14, rsi(daily_df[f'{sec}_Close'], 14)])
        daily_df[rsi_d_name] = daily_df[rsi_k_name].rolling(window=9).mean()
        rsi_cols.append(rsi_k_name)
        rsi_cols.append(rsi_d_name)

    return daily_df, rsi_cols


def add_high_low_diff(daily_df, other_sec, sec_name, input_features):
    securities = other_sec + [sec_name]
    high_low_features = []
    for sec in securities:
        daily_df[f'{sec}_HL_diff'] = (daily_df[f'{sec}_High'] - daily_df[f'{sec}_Low'])/daily_df[f'{sec}_Close']
        high_low_features.append(f'{sec}_HL_diff')

    daily_set_cols2 = [col for col in input_features if not any(sub in col for sub in ['High', 'Low'])]
    drop_cols = [col for col in input_features if col not in daily_set_cols2]
    input_features = daily_set_cols2 + high_low_features
    daily_df.drop(columns=drop_cols, inplace=True)

    return daily_df, input_features


def scale_open_close(daily_df, sec_name, other_sec, input_features):
    securities = other_sec + [sec_name]
    scale_features = []
    for sec in securities:
        close_change = daily_df[f'{sec}_Close']/daily_df.loc[0, f'{sec}_Close']
        open_change = daily_df[f'{sec}_Open']/daily_df.loc[0, f'{sec}_Open']
        daily_df[f'{sec}_Close_Scale'] = close_change.fillna(0)
        daily_df[f'{sec}_Open_Scale'] = open_change.fillna(0)
        scale_features.append(f'{sec}_Open_Scale')
        scale_features.append(f'{sec}_Close_Scale')

    input_features = input_features + scale_features

    return daily_df, input_features


def get_open_close_diff(daily_df, other_sec, sec_name):
    securities = other_sec + [sec_name]
    for sec in securities:
        daily_df[f'{sec}_Close'] = daily_df[f'{sec}_Close'].pct_change()
        daily_df[f'{sec}_Open'] = daily_df[f'{sec}_Open'].pct_change()

    return daily_df



