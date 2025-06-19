import pandas as pd
import numpy as np
from numba import njit
import yfinance as yf
import streamlit as st
import thingspeak
import json
import streamlit.components.v1 as components

st.set_page_config(page_title="Limit_F(X)", page_icon="✈️", layout="wide")

# === CONFIG LOADING ===
def load_config(path='limit_fx_config.json'):
    """Loads the asset configuration from a JSON file."""
    with open(path, 'r') as f:
        config = json.load(f)
    return config['assets']

ASSETS = load_config()
TICKERS = [a['symbol'] for a in ASSETS]

# === THINGSPEAK (REFACTORED FOR DYNAMIC CLIENTS) ===
@st.cache_data(ttl=300) # Cache ThingSpeak data for 5 minutes
def get_act_from_thingspeak(channel_id, api_key, field):
    """
    Fetches the last value from a specific field in a specific ThingSpeak channel.
    A new client is created for each call to support multiple channels.
    """
    try:
        # Create a client for this specific request
        client = thingspeak.Channel(channel_id, api_key, fmt='json')
        act_json = client.get_field_last(field=str(field))
        # Handle potential null values from ThingSpeak
        value = json.loads(act_json).get(f"field{field}")
        if value is None:
            st.warning(f"Field {field} on channel {channel_id} returned null. Using default value 0.")
            return 0
        return int(value)
    except Exception as e:
        st.error(f"Could not fetch data from ThingSpeak (Channel: {channel_id}, Field: {field}). Error: {e}")
        return 0 # Return a safe default value on error

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

@st.cache_data(ttl=600) # Cache data for 10 minutes
def Limit_fx(Ticker, act=-1):
    filter_date = '2023-01-01 12:00:00+07:00'
    tickerData = yf.Ticker(Ticker)
    tickerData = tickerData.history(period='max')[['Close']]
    tickerData.index = tickerData.index.tz_convert(tz='Asia/Bangkok')
    tickerData = tickerData[tickerData.index >= filter_date]
    prices = np.array(tickerData.Close.values, dtype=np.float64)
    if len(prices) == 0:
        return pd.DataFrame() # Return empty dataframe if no price data
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
    df_min = Limit_fx(Ticker, act=-1) 
    df_fx = Limit_fx(Ticker, act=act)
    df_max = Limit_fx(Ticker, act=-2)

    if df_min.empty or df_fx.empty or df_max.empty:
        st.error(f"Could not generate plot for {Ticker} due to missing data.")
        return

    chart_data = pd.DataFrame({
        'min': df_min.net,
        f'fx_{act}': df_fx.net,
        'max': df_max.net
    })
    st.write('Refer_Log')
    st.line_chart(chart_data)

    df_plot_burn = df_min[['buffer']].cumsum()
    st.write('Burn_Cash (Cumulative)')
    st.line_chart(df_plot_burn)

    st.write("Detailed Data (Min Action)")
    st.dataframe(df_min)

# === TAB LAYOUT ===
tab_names = TICKERS + ['Burn_Cash', 'Ref_index_Log', 'cf_log']
tabs = st.tabs(tab_names)
tab_dict = dict(zip(tab_names, tabs))

# === MAIN ASSET TABS (MODIFIED LOOP) ===
for asset in ASSETS:
    symbol = asset['symbol']
    with tab_dict[symbol]:
        # Get act by passing the specific credentials for this asset
        act = get_act_from_thingspeak(
            channel_id=asset['channel_id'],
            api_key=asset['write_api_key'],
            field=asset['field']
        )
        plot(symbol, act)

# === REF_INDEX_LOG TAB ===
with tab_dict['Ref_index_Log']:
    @st.cache_data(ttl=600)
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

    if not prices_df.empty:
        int_st = np.prod(prices_df.iloc[0][TICKERS])
        initial_capital_per_stock = 3000
        initial_capital_Ref_index_Log = initial_capital_per_stock * len(TICKERS)

        def calculate_ref_log(row):
            int_end = np.prod(row[TICKERS])
            return initial_capital_Ref_index_Log + (-1500 * np.log(int_st / int_end))

        prices_df['ref_log'] = prices_df.apply(calculate_ref_log, axis=1)
        ref_log_values = prices_df.ref_log.values
        
        sumusd_dfs = {f'sumusd_{symbol}': Limit_fx(symbol, act=-1).sumusd for symbol in TICKERS}
        df_sumusd_ = pd.DataFrame(sumusd_dfs)
        
        df_sumusd_['daily_sumusd'] = df_sumusd_.sum(axis=1)
        df_sumusd_['ref_log'] = ref_log_values
        
        total_initial_capital = sum([Limit_fx(symbol, act=-1).sumusd.iloc[0] for symbol in TICKERS])
        net_raw = df_sumusd_['daily_sumusd'] - df_sumusd_['ref_log'] - total_initial_capital
        net_at_index_0 = net_raw.iloc[0] if not net_raw.empty else 0
        df_sumusd_['net'] = net_raw - net_at_index_0
        
        st.line_chart(df_sumusd_['net'])
        st.dataframe(df_sumusd_)
    else:
        st.warning("Could not fetch price data for Ref_index_Log.")

# === BURN_CASH TAB (UPDATED) ===
with tab_dict['Burn_Cash']:
    # ดึงข้อมูล buffer จาก Limit_fx สำหรับทุก Tickers
    # ใช้ act=-1 (always buy) ตาม logic เดิมของแท็บนี้
    buffers = {f'buffer_{symbol}': Limit_fx(symbol, act=-1).buffer for symbol in TICKERS}
    df_burn_cash = pd.DataFrame(buffers)
    
    # คำนวณ daily และ cumulative burn
    df_burn_cash['daily_burn'] = df_burn_cash.sum(axis=1)
    df_burn_cash['cumulative_burn'] = df_burn_cash['daily_burn'].cumsum()

    # --- START: การคำนวณค่าความเสี่ยง ---
    
    # 1. 1-Day Burn (Max Daily Burn)
    max_daily_burn = df_burn_cash['daily_burn'].min()

    # 2. Peak-to-Trough (Overall Drawdown)
    cumulative_burn_series = df_burn_cash['cumulative_burn']
    if not cumulative_burn_series.empty:
        # คำนวณ drawdown จากจุดสูงสุด ไม่ใช่แค่ค่าต่ำสุด
        peak_index = cumulative_burn_series.idxmax()
        peak_to_trough_burn = cumulative_burn_series[peak_index] - cumulative_burn_series[peak_index:].min()
    else:
        peak_to_trough_burn = 0

    # 3. Rolling Window Burn (30-Day and 90-Day)
    if len(cumulative_burn_series) >= 30:
        rolling_30_day_change = cumulative_burn_series.rolling(window=30).apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)
        max_30_day_burn = rolling_30_day_change.min()
    else:
        max_30_day_burn = cumulative_burn_series.min() if not cumulative_burn_series.empty else 0

    if len(cumulative_burn_series) >= 90:
        rolling_90_day_change = cumulative_burn_series.rolling(window=90).apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)
        max_90_day_burn = rolling_90_day_change.min()
    else:
        max_90_day_burn = cumulative_burn_series.min() if not cumulative_burn_series.empty else 0


    # --- END: การคำนวณค่าความเสี่ยง ---
    
    st.header("Cash Burn Risk Analysis")
    st.info("การวิเคราะห์นี้อิงจากการ Backtest ด้วยกลยุทธ์ 'ซื้อทุกครั้ง' (act=-1) เพื่อประเมินความเสี่ยงสูงสุดที่เป็นไปได้")
    
    # แสดงผลในรูปแบบตารางที่สวยงาม
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Short-Term Risk")
        st.metric(label="🔥 1-Day Burn (Worst Day)", value=f"{max_daily_burn:,.2f} USD")
        st.metric(label="🔥 30-Day Burn (Worst Month)", value=f"{max_30_day_burn:,.2f} USD")
    
    with col2:
        st.subheader("Medium to Long-Term Risk")
        st.metric(label="🔥 90-Day Burn (Worst Quarter)", value=f"{max_90_day_burn:,.2f} USD")
        st.metric(label="🏔️ Peak-to-Trough Burn (Max Drawdown)", value=f"{peak_to_trough_burn:,.2f} USD")

    st.markdown("---")
    
    # พล็อตกราฟ
    st.subheader("Cumulative Cash Burn Over Time")
    st.line_chart(df_burn_cash['cumulative_burn'])
    
    with st.expander("View Detailed Burn Data"):
        st.dataframe(df_burn_cash)

# === CF_LOG TAB ===
def iframe(frame='', width=1500, height=800):
    components.iframe(frame, width=width, height=height, scrolling=True)

with tab_dict['cf_log']:
    st.markdown("""
    - **Rebalance**: `-fix * ln(t0 / tn)`
    - **Net Profit**: `sumusd - refer - sumusd[0]` (ต้นทุนเริ่มต้น)
    - **Ref_index_Log**: `initial_capital_Ref_index_Log + (-1500 * ln(int_st / int_end))`
    - **Net in Ref_index_Log**: `(daily_sumusd - ref_log - total_initial_capital) - net_at_index_0`
    - **Option P/L**: `(max(0, ราคาหุ้นปัจจุบัน - Strike) * contracts_or_shares) - (contracts_or_shares * premium_paid_per_share)`
    ---
    """)
    iframe("https://monica.im/share/artifact?id=qpAkuKjBpuVz2cp9nNFRs3")
    st.markdown("---")
    iframe("https://monica.im/share/artifact?id=wEjeaMxVW6MgDDm3xAZatX")
    st.markdown("---")
    iframe("https://monica.im/share/artifact?id=ZfHT5iDP2Ypz82PCRw9nEK")
    st.markdown("---")
    iframe("https://monica.im/share/chat?shareId=SUsEYhzSMwqIq3Cx")
