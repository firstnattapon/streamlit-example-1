import pandas as pd
import numpy as np
from numba import njit
import yfinance as yf
import streamlit as st
import thingspeak
import json

st.set_page_config(page_title="Limit_F(X)", page_icon="✈️", layout="wide")

# โหลด config
with open('limit_fx_config.json', 'r') as f:
    config = json.load(f)
ASSETS = config['assets']
TICKERS = [a['symbol'] for a in ASSETS]

@njit(fastmath=True)
def calculate_optimized(action_list, price_list, fix=1500):
    action_array = np.asarray(action_list, dtype=np.int32)
    action_array[0] = 1
    price_array = np.asarray(price_list, dtype=np.float64)
    n = len(action_array)
    amount = np.empty(n, dtype=np.float64)
    buffer = np.zeros(n, dtype=np.float64)
    cash = np.empty(n, dtype=np.float64)
    asset_value = np.empty(n, dtype=np.float64)
    sumusd = np.empty(n, dtype=np.float64)
    initial_price = price_array[0]
    amount[0] = fix / initial_price
    cash[0] = fix
    asset_value[0] = amount[0] * initial_price
    sumusd[0] = cash[0] + asset_value[0]
    refer = -fix * np.log(initial_price / price_array)
    for i in range(1, n):
        curr_price = price_array[i]
        if action_array[i] == 0:
            amount[i] = amount[i-1]
            buffer[i] = 0
        else:
            amount[i] = fix / curr_price
            buffer[i] = amount[i-1] * curr_price - fix
        cash[i] = cash[i-1] + buffer[i]
        asset_value[i] = amount[i] * curr_price
        sumusd[i] = cash[i] + asset_value[i]
    return buffer, sumusd, cash, asset_value, amount, refer

def get_max_action(price_list, fix=1500):
    prices = np.asarray(price_list, dtype=np.float64)
    n = len(prices)
    if n < 2:
        return np.ones(n, dtype=int)
    dp = np.zeros(n, dtype=np.float64)
    path = np.zeros(n, dtype=int)
    initial_capital = float(fix * 2)
    dp[0] = initial_capital
    for i in range(1, n):
        max_prev_sumusd = 0
        best_j = 0
        for j in range(i):
            profit_from_j_to_i = fix * ((prices[i] / prices[j]) - 1)
            current_sumusd = dp[j] + profit_from_j_to_i
            if current_sumusd > max_prev_sumusd:
                max_prev_sumusd = current_sumusd
                best_j = j
        dp[i] = max_prev_sumusd
        path[i] = best_j
    actions = np.zeros(n, dtype=int)
    last_action_day = np.argmax(dp)
    current_day = last_action_day
    while current_day > 0:
        actions[current_day] = 1
        current_day = path[current_day]
    actions[0] = 1
    return actions

def Limit_fx(Ticker='', act=-1):
    filter_date = '2023-01-01 12:00:00+07:00'
    tickerData = yf.Ticker(Ticker)
    tickerData = tickerData.history(period='max')[['Close']]
    tickerData.index = tickerData.index.tz_convert(tz='Asia/Bangkok')
    tickerData = tickerData[tickerData.index >= filter_date]
    prices = np.array(tickerData.Close.values, dtype=np.float64)
    if act == -1:
        actions = np.array(np.ones(len(prices)), dtype=np.int64)
    elif act == -2:
        actions = get_max_action(prices)
    else:
        rng = np.random.default_rng(act)
        actions = rng.integers(0, 2, len(prices))
    buffer, sumusd, cash, asset_value, amount, refer = calculate_optimized(actions, prices)
    initial_capital = sumusd[0]
    df = pd.DataFrame({
        'price': prices,
        'action': actions,
        'buffer': np.round(buffer, 2),
        'sumusd': np.round(sumusd, 2),
        'cash': np.round(cash, 2),
        'asset_value': np.round(asset_value, 2),
        'amount': np.round(amount, 2),
        'refer': np.round(refer + initial_capital, 2),
        'net': np.round(sumusd - refer - initial_capital, 2)
    })
    df['sumusd'] = sumusd  # เพิ่มคอลัมน์นี้เพื่อใช้ใน Ref_index_Log
    df['buffer'] = buffer  # เพิ่มคอลัมน์นี้เพื่อใช้ใน Burn_Cash
    return df

def plot(Ticker='', act=-1):
    all = []
    all_id = []
    all.append(Limit_fx(Ticker, act=-1).net)
    all_id.append('min')
    all.append(Limit_fx(Ticker, act=act).net)
    all_id.append(f'fx_{act}')
    all.append(Limit_fx(Ticker, act=-2).net)
    all_id.append('max')
    chart_data = pd.DataFrame(np.array(all).T, columns=np.array(all_id))
    st.write('Refer_Log')
    st.line_chart(chart_data)
    df_plot = Limit_fx(Ticker, act=-1)[['buffer']].cumsum()
    st.write('Burn_Cash')
    st.line_chart(df_plot)
    st.write(Limit_fx(Ticker, act=-1))

channel_id = 2385118
write_api_key = 'IPSG3MMMBJEB9DY8'
client = thingspeak.Channel(channel_id, write_api_key, fmt='json')

# สร้าง tab อัตโนมัติ
tab_names = TICKERS + ['Burn_Cash', 'Ref_index_Log', 'cf_log']
tabs = st.tabs(tab_names)
tab_dict = dict(zip(tab_names, tabs))

# วน plot ในแต่ละ asset
for asset in ASSETS:
    symbol = asset['symbol']
    field = asset['field']
    with tab_dict[symbol]:
        act = client.get_field_last(field=str(field))
        act_js = int(json.loads(act)[f"field{field}"])
        plot(Ticker=symbol, act=act_js)

with tab_dict['Ref_index_Log']:
    tickers = TICKERS
    def get_prices(tickers, start_date):
        df_list = []
        for ticker in tickers:
            tickerData = yf.Ticker(ticker)
            tickerHist = tickerData.history(period='max')[['Close']]
            tickerHist.index = tickerHist.index.tz_convert(tz='Asia/Bangkok')
            tickerHist = tickerHist[tickerHist.index >= start_date]
            tickerHist = tickerHist.rename(columns={'Close': ticker})
            df_list.append(tickerHist[[ticker]])
        prices_df = pd.concat(df_list, axis=1)
        return prices_df
    filter_date = '2023-01-01 12:00:00+07:00'
    prices_df = get_prices(tickers, filter_date)
    prices_df = prices_df.dropna()
    int_st_list = prices_df.iloc[0][tickers]
    int_st = np.prod(int_st_list)
    initial_capital_per_stock = 3000
    initial_capital_Ref_index_Log = initial_capital_per_stock * len(tickers)
    def calculate_ref_log(row):
        int_end = np.prod(row[tickers])
        ref_log = initial_capital_Ref_index_Log + (-1500 * np.log(int_st / int_end))
        return ref_log
    prices_df['ref_log'] = prices_df.apply(calculate_ref_log, axis=1)
    prices_df = prices_df.reset_index()
    prices_df = prices_df.ref_log.values
    sumusd_ = {f'sumusd_{symbol}': Limit_fx(symbol, act=-1).sumusd for symbol in tickers}
    df_sumusd_ = pd.DataFrame(sumusd_)
    df_sumusd_['daily_sumusd'] = df_sumusd_.sum(axis=1)
    df_sumusd_['ref_log'] = prices_df
    total_initial_capital = sum([Limit_fx(symbol, act=-1).sumusd.iloc[0] for symbol in tickers])
    net_raw = df_sumusd_['daily_sumusd'] - df_sumusd_['ref_log'] - total_initial_capital
    net_at_index_0 = net_raw.iloc[0]
    df_sumusd_['net'] = net_raw - net_at_index_0
    df_sumusd_ = df_sumusd_.reset_index().set_index('index')
    st.line_chart(df_sumusd_.net)
    st.dataframe(df_sumusd_)

with tab_dict['Burn_Cash']:
    STOCK_SYMBOLS = TICKERS
    buffers = {f'buffer_{symbol}': Limit_fx(symbol, act=-1).buffer for symbol in STOCK_SYMBOLS}
    df_burn_cash = pd.DataFrame(buffers)
    df_burn_cash['daily_burn'] = df_burn_cash.sum(axis=1)
    df_burn_cash['cumulative_burn'] = df_burn_cash['daily_burn'].cumsum()
    st.line_chart(df_burn_cash['cumulative_burn'])
    df_burn_cash = df_burn_cash.reset_index(drop=True)
    st.dataframe(df_burn_cash)

import streamlit.components.v1 as components
def iframe(frame=''):
    src = frame
    st.components.v1.iframe(src, width=1500, height=800, scrolling=0)

with tab_dict['cf_log']:
    st.write('')
    st.write(' Rebalance   =  -fix * ln( t0 / tn )')
    st.write(' Net Profit  =  sumusd - refer - sumusd[0] (ต้นทุนเริ่มต้น)')
    st.write(' Ref_index_Log = initial_capital_Ref_index_Log + (-1500 * ln(int_st / int_end))')
    st.write(' Net in Ref_index_Log = (daily_sumusd - ref_log - total_initial_capital) - net_at_index_0')
    st.write('________')
    iframe(frame="https://monica.im/share/artifact?id=qpAkuKjBpuVz2cp9nNFRs3")
    st.write('________')
    iframe(frame="https://monica.im/share/artifact?id=wEjeaMxVW6MgDDm3xAZatX")
    st.write('________')
    iframe(frame="https://monica.im/share/artifact?id=ZfHT5iDP2Ypz82PCRw9nEK")
    st.write('________')
    iframe(frame="https://monica.im/share/chat?shareId=SUsEYhzSMwqIq3Cx")
