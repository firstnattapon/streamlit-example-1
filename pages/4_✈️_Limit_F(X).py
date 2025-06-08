import pandas as pd
import numpy as np
from numba import njit
import yfinance as yf
import streamlit as st
import thingspeak
import json

st.set_page_config(page_title="Limit_F(X)", page_icon="✈️" , layout = "wide" )
.
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

def Limit_fx(Ticker='', act=-1, custom_actions=None):
    filter_date = '2023-01-01 12:00:00+07:00'
    tickerData = yf.Ticker(Ticker)
    tickerData = tickerData.history(period='max')[['Close']]
    tickerData.index = tickerData.index.tz_convert(tz='Asia/Bangkok')
    tickerData = tickerData[tickerData.index >= filter_date]
    prices = np.array(tickerData.Close.values, dtype=np.float64)
    if custom_actions is not None:
        actions = np.array(custom_actions, dtype=np.int64)
        actions[0] = 1  # ensure first action is always 1
    elif act == -1:
        actions = np.ones(len(prices), dtype=np.int64)
    elif act == -2:
        actions = get_max_action(prices)
    else:
        rng = np.random.default_rng(act)
        actions = rng.integers(0, 2, len(prices))
        actions[0] = 1
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
    return df

def plot(Ticker='', act=-1):
    all = []
    all_id = []
    all.append(Limit_fx(Ticker, act=-1).net)
    all_id.append('min')
    all.append(Limit_fx(Ticker, act=act).net)
    all_id.append('fx_{}'.format(act))
    all.append(Limit_fx(Ticker, act=-2).net)
    all_id.append('max')
    chart_data = pd.DataFrame(np.array(all).T, columns=np.array(all_id))
    st.write('Refer_Log')
    st.line_chart(chart_data)
    df_plot = Limit_fx(Ticker, act=-1)
    df_plot = df_plot[['buffer']].cumsum()
    st.write('Burn_Cash')
    st.line_chart(df_plot)
    st.write(Limit_fx(Ticker, act=-1))

channel_id = 2385118
write_api_key = 'IPSG3MMMBJEB9DY8'
client = thingspeak.Channel(channel_id, write_api_key, fmt='json')

tab1, tab2, tab3, tab4, tab5, tab6, tab7, Burn_Cash, Ref_index_Log, cf_log, Ref_DNA_Log = st.tabs([
    "FFWM", "NEGG", "RIVN", 'APLS', 'NVTS', 'QXO(LV)', 'RXRX(LV)', 'Burn_Cash', 'Ref_index_Log', 'cf_log', 'Ref_DNA_Log'
])

with Ref_index_Log:
    tickers = ['FFWM', 'NEGG', 'RIVN', 'APLS', 'NVTS', 'QXO', 'RXRX']
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
    sumusd_ = {'sumusd_{}'.format(symbol): Limit_fx(symbol, act=-1).sumusd for symbol in tickers}
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

with Burn_Cash:
    STOCK_SYMBOLS = ['FFWM', 'NEGG', 'RIVN', 'APLS', 'NVTS', 'QXO', 'RXRX']
    buffers = {
        'buffer_{}'.format(symbol): Limit_fx(symbol, act=-1).buffer
        for symbol in STOCK_SYMBOLS}
    df_burn_cash = pd.DataFrame(buffers)
    df_burn_cash['daily_burn'] = df_burn_cash.sum(axis=1)
    df_burn_cash['cumulative_burn'] = df_burn_cash['daily_burn'].cumsum()
    st.line_chart(df_burn_cash['cumulative_burn'])
    df_burn_cash = df_burn_cash.reset_index(drop=True)
    st.dataframe(df_burn_cash)

with tab1:
    FFWM_act = client.get_field_last(field='{}'.format(2))
    FFWM_act_js = int(json.loads(FFWM_act)["field{}".format(2)])
    plot(Ticker='FFWM', act=FFWM_act_js)

with tab2:
    NEGG_act = client.get_field_last(field='{}'.format(3))
    NEGG_act_js = int(json.loads(NEGG_act)["field{}".format(3)])
    plot(Ticker='NEGG', act=NEGG_act_js)

with tab3:
    RIVN_act = client.get_field_last(field='{}'.format(4))
    RIVN_act_js = int(json.loads(RIVN_act)["field{}".format(4)])
    plot(Ticker='RIVN', act=RIVN_act_js)

with tab4:
    APLS_act = client.get_field_last(field='{}'.format(5))
    APLS_act_js = int(json.loads(APLS_act)["field{}".format(5)])
    plot(Ticker='APLS', act=APLS_act_js)

with tab5:
    NVTS_act = client.get_field_last(field='{}'.format(6))
    NVTS_act_js = int(json.loads(NVTS_act)["field{}".format(6)])
    plot(Ticker='NVTS', act=NVTS_act_js)

with tab6:
    QXO_act = client.get_field_last(field='{}'.format(7))
    QXO_act_js = int(json.loads(QXO_act)["field{}".format(7)])
    plot(Ticker='QXO', act=QXO_act_js)

with tab7:
    RXRX_act = client.get_field_last(field='{}'.format(8))
    RXRX_act_js = int(json.loads(RXRX_act)["field{}".format(8)])
    plot(Ticker='RXRX', act=RXRX_act_js)

import streamlit.components.v1 as components
def iframe(frame=''):
    src = frame
    st.components.v1.iframe(src, width=1500, height=800, scrolling=0)

with cf_log:
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

# -------------------------- NEW FEATURE: DNA WINDOW LOG --------------------------

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
    # Plot
    df_dna = pd.DataFrame({'window': window_labels, 'net': nets})
    st.line_chart(df_dna.set_index('window'))
    st.dataframe(df_dna)
