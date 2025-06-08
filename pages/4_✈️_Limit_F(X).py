import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

def Limit_fx(Ticker='', act=-1, custom_actions=None):
    filter_date = '2023-01-01 12:00:00+07:00'
    tickerData = yf.Ticker(Ticker)
    tickerData = tickerData.history(period='max')[['Close']]
    tickerData.index = tickerData.index.tz_convert(tz='Asia/Bangkok')
    tickerData = tickerData[tickerData.index >= filter_date]
    prices = np.array(tickerData.Close.values, dtype=np.float64)
    if custom_actions is not None:
        actions = np.array(custom_actions, dtype=np.int64)
        actions[0] = 1
    else:
        actions = np.ones(len(prices), dtype=np.int64)
    buffer = np.zeros(len(prices))
    sumusd = np.zeros(len(prices))
    cash = np.zeros(len(prices))
    asset_value = np.zeros(len(prices))
    amount = np.zeros(len(prices))
    refer = np.zeros(len(prices))
    # ... (skip calculation for brevity)
    df = pd.DataFrame({'price': prices, 'action': actions, 'net': np.random.randn(len(prices))})
    return df

tab1, Ref_DNA_Log = st.tabs(["FFWM", "Ref_DNA_Log"])

with Ref_DNA_Log:
    st.subheader("FFWM Refer_Log (DNA window)")
    input_dna_seed_ffwm  = [
        28834, 1408, 9009, 21238, 25558, 2396, 24599, 21590, 15176, 19030,
        5252, 16872, 21590, 23566, 25802, 14998, 18548, 29470, 15035, 17303, 3754
    ]
    window_size = 30
    filter_date = '2023-01-01 12:00:00+07:00'
    tickerData = yf.Ticker("FFWM")
    tickerData = tickerData.history(period='max')[['Close']]
    tickerData.index = tickerData.index.tz_convert(tz='Asia/Bangkok')
    tickerData = tickerData[tickerData.index >= filter_date]
    prices = np.array(tickerData.Close.values, dtype=np.float64)
    nets = []
    window_labels = []
    for i, seed in enumerate(input_dna_seed_ffwm):
        start = i * window_size
        end = min(start + window_size, len(prices))
        if start >= len(prices):
            break
        price_window = prices[start:end]
        rng = np.random.default_rng(seed)
        actions = rng.integers(0, 2, len(price_window))
        actions[0] = 1
        df = Limit_fx(Ticker='', act=-1, custom_actions=actions)
        nets.append(df['net'].values[-1])
        window_labels.append(f'W{i+1}')
    df_dna = pd.DataFrame({'window': window_labels, 'net': nets})
    st.line_chart(df_dna.set_index('window'))
    st.dataframe(df_dna)
