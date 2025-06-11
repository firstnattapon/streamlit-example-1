แน่นอนครับ!  
**Refactor** โค้ดให้สะอาดขึ้น, รองรับการเพิ่ม/ลด Asset จาก JSON, ลดการซ้ำซ้อน, อ่านง่าย, และแยกฟังก์ชันให้เหมาะสม  
(ผลลัพธ์ UI/โครงสร้างเหมือนเดิม 100%)

---

## ตัวอย่างโค้ด Refactor

````python
import pandas as pd
import numpy as np
from numba import njit
import yfinance as yf
import streamlit as st
import thingspeak
import json

st.set_page_config(page_title="Limit_F(X)", page_icon="✈️", layout="wide")

# === CONFIG LOADING ===
def load_config(path='limit_fx_config.json'):
    with open(path, 'r') as f:
        config = json.load(f)
    return config['assets']

ASSETS = load_config()
TICKERS = [a['symbol'] for a in ASSETS]

# === CORE CALCULATION FUNCTIONS ===
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

def Limit_fx(Ticker, act=-1):
    filter_date = '2023-01-01 12:00:00+07:00'
    tickerData = yf.Ticker(Ticker)
    tickerData = tickerData.history(period='max')[['Close']]
    tickerData.index = tickerData.index.tz_convert(tz='Asia/Bangkok')
    tickerData = tickerData[tickerData.index >= filter_date]
    prices = np.array(tickerData.Close.values, dtype=np.float64)
    if act == -1:
        actions = np.ones(len(prices), dtype=np.int64)
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
        'buffer': buffer,
        'sumusd': sumusd,
        'cash': cash,
        'asset_value': asset_value,
        'amount': amount,
        'refer': refer + initial_capital,
        'net': sumusd - refer - initial_capital
    })
    return df

def plot(Ticker, act):
    all_net = [
        Limit_fx(Ticker, act=-1).net,
        Limit_fx(Ticker, act=act).net,
        Limit_fx(Ticker, act=-2).net
    ]
    all_id = ['min', f'fx_{act}', 'max']
    chart_data = pd.DataFrame(np.array(all_net).T, columns=all_id)
    st.write('Refer_Log')
    st.line_chart(chart_data)
    df_plot = Limit_fx(Ticker, act=-1)[['buffer']].cumsum()
    st.write('Burn_Cash')
    st.line_chart(df_plot)
    st.write(Limit_fx(Ticker, act=-1))

# === THINGSPEAK ===
channel_id = 2385118
write_api_key = 'IPSG3MMMBJEB9DY8'
client = thingspeak.Channel(channel_id, write_api_key, fmt='json')

def get_act_from_thingspeak(field):
    act_json = client.get_field_last(field=str(field))
    return int(json.loads(act_json)[f"field{field}"])

# === TAB LAYOUT ===
tab_names = TICKERS + ['Burn_Cash', 'Ref_index_Log', 'cf_log']
tabs = st.tabs(tab_names)
tab_dict = dict(zip(tab_names, tabs))

# === MAIN ASSET TABS ===
for asset in ASSETS:
    symbol = asset['symbol']
    field = asset['field']
    with tab_dict[symbol]:
        act = get_act_from_thingspeak(field)
        plot(symbol, act)

# === REF_INDEX_LOG TAB ===
with tab_dict['Ref_index_Log']:
    def get_prices(tickers, start_date):
        df_list = []
        for ticker in tickers:
            tickerData = yf.Ticker(ticker)
            tickerHist = tickerData.history(period='max')[['Close']]
            tickerHist.index = tickerHist.index.tz_convert(tz='Asia/Bangkok')
            tickerHist = tickerHist[tickerHist.index >= start_date]
            tickerHist = tickerHist.rename(columns={'Close': ticker})
            df_list.append(tickerHist[[ticker]])
        return pd.concat(df_list, axis=1)
    filter_date = '2023-01-01 12:00:00+07:00'
    prices_df = get_prices(TICKERS, filter_date).dropna()
    int_st = np.prod(prices_df.iloc[0][TICKERS])
    initial_capital_per_stock = 3000
    initial_capital_Ref_index_Log = initial_capital_per_stock * len(TICKERS)
    def calculate_ref_log(row):
        int_end = np.prod(row[TICKERS])
        return initial_capital_Ref_index_Log + (-1500 * np.log(int_st / int_end))
    prices_df['ref_log'] = prices_df.apply(calculate_ref_log, axis=1)
    prices_df = prices_df.reset_index()
    ref_log_values = prices_df.ref_log.values
    sumusd_ = {f'sumusd_{symbol}': Limit_fx(symbol, act=-1).sumusd for symbol in TICKERS}
    df_sumusd_ = pd.DataFrame(sumusd_)
    df_sumusd_['daily_sumusd'] = df_sumusd_.sum(axis=1)
    df_sumusd_['ref_log'] = ref_log_values
    total_initial_capital = sum([Limit_fx(symbol, act=-1).sumusd.iloc[0] for symbol in TICKERS])
    net_raw = df_sumusd_['daily_sumusd'] - df_sumusd_['ref_log'] - total_initial_capital
    net_at_index_0 = net_raw.iloc[0]
    df_sumusd_['net'] = net_raw - net_at_index_0
    df_sumusd_ = df_sumusd_.reset_index().set_index('index')
    st.line_chart(df_sumusd_.net)
    st.dataframe(df_sumusd_)

# === BURN_CASH TAB ===
with tab_dict['Burn_Cash']:
    buffers = {f'buffer_{symbol}': Limit_fx(symbol, act=-1).buffer for symbol in TICKERS}
    df_burn_cash = pd.DataFrame(buffers)
    df_burn_cash['daily_burn'] = df_burn_cash.sum(axis=1)
    df_burn_cash['cumulative_burn'] = df_burn_cash['daily_burn'].cumsum()
    st.line_chart(df_burn_cash['cumulative_burn'])
    st.dataframe(df_burn_cash.reset_index(drop=True))

# === CF_LOG TAB ===
import streamlit.components.v1 as components
def iframe(frame=''):
    st.components.v1.iframe(frame, width=1500, height=800, scrolling=0)

with tab_dict['cf_log']:
    st.write('')
    st.write(' Rebalance   =  -fix * ln( t0 / tn )')
    st.write(' Net Profit  =  sumusd - refer - sumusd[0] (ต้นทุนเริ่มต้น)')
    st.write(' Ref_index_Log = initial_capital_Ref_index_Log + (-1500 * ln(int_st / int_end))')
    st.write(' Net in Ref_index_Log = (daily_sumusd - ref_log - total_initial_capital) - net_at_index_0')
    st.write('________')
    iframe("https://monica.im/share/artifact?id=qpAkuKjBpuVz2cp9nNFRs3")
    st.write('________')
    iframe("https://monica.im/share/artifact?id=wEjeaMxVW6MgDDm3xAZatX")
    st.write('________')
    iframe("https://monica.im/share/artifact?id=ZfHT5iDP2Ypz82PCRw9nEK")
    st.write('________')
    iframe("https://monica.im/share/chat?shareId=SUsEYhzSMwqIq3Cx")
````

---

### จุดเด่นของ Refactor นี้

- **Config-driven:** เพิ่ม/ลด asset ได้จาก JSON
- **ลดซ้ำซ้อน:** รวม logic ที่ซ้ำกัน, ฟังก์ชันอ่านง่าย, ใช้ชื่อสื่อความ
- **UI/ผลลัพธ์เหมือนเดิม:** ไม่กระทบต่อหน้าตาและการใช้งาน
- **ดูแลรักษาง่าย:** เพิ่ม asset ในไฟล์ JSON เดียว

---

**หากต้องการ refactor เพิ่มเติม หรือต้องการอธิบายแต่ละฟังก์ชัน แจ้งได้เลยครับ!**
