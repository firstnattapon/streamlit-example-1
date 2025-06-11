import pandas as pd
import numpy as np
from numba import njit
import yfinance as yf
import streamlit as st
import thingspeak
import json

# --- การตั้งค่าและโหลด Configuration ---
st.set_page_config(page_title="Limit_F(X)", page_icon="✈️" , layout = "wide" )

@st.cache_data
def load_config(filename="add_gen_config.json"):
    """
    โหลดการตั้งค่าจากไฟล์ JSON
    ใช้ cache เพื่อให้โหลดไฟล์แค่ครั้งเดียว
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"ไม่พบไฟล์ '{filename}'. กรุณาสร้างไฟล์นี้ตามรูปแบบที่กำหนด")
        return None # คืนค่า None เพื่อให้หยุดการทำงาน
    except json.JSONDecodeError:
        st.error(f"ไฟล์ '{filename}' มีรูปแบบ JSON ไม่ถูกต้อง")
        return None

# โหลด Config
config = load_config()

# ถ้าโหลด config ไม่สำเร็จ ให้หยุดการทำงานของแอป
if not config:
    st.stop()

# ดึงค่าตั้งค่าจาก config
SETTINGS = config.get("settings", {})
ASSETS = config.get("assets", [])
IFRAMES = config.get("iframes", [])

FILTER_DATE = SETTINGS.get("filter_date", "2023-01-01 12:00:00+07:00")
FIX_VALUE = SETTINGS.get("fix_value", 1500)
CHANNEL_ID = SETTINGS.get("thingspeak_channel_id")
WRITE_API_KEY = SETTINGS.get("thingspeak_write_api_key")

# สร้าง list ของ Ticker จาก config เพื่อใช้ในส่วนกลาง
ALL_TICKERS = [asset['ticker'] for asset in ASSETS]

# --- ส่วนของฟังก์ชันคำนวณ (เหมือนเดิม) ---
@njit(fastmath=True)
def calculate_optimized(action_list, price_list, fix=FIX_VALUE):
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

def get_max_action(price_list, fix=FIX_VALUE):
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

# --- ส่วนของฟังก์ชันหลักในการสร้างข้อมูลและกราฟ (Cache เพื่อประสิทธิภาพ) ---
@st.cache_data(ttl=3600) # Cache ข้อมูล 1 ชั่วโมง
def get_limit_fx_data(ticker, act=-1):
    tickerData = yf.Ticker(ticker)
    tickerData = tickerData.history(period='max')[['Close']]
    tickerData.index = tickerData.index.tz_convert(tz='Asia/Bangkok')
    tickerData = tickerData[tickerData.index >= FILTER_DATE]
    
    prices = np.array(tickerData.Close.values, dtype=np.float64)

    if act == -1: # min
        actions = np.ones(len(prices), dtype=np.int64)
    elif act == -2: # max
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
    return df

def plot_asset(ticker, act=-1):
    """
    ฟังก์ชันสำหรับพล็อตกราฟและแสดงข้อมูลของ Asset เดียว
    """
    # min
    df_min_net = get_limit_fx_data(ticker, act=-1).net
    
    # fx (ที่เลือก)
    df_fx_net = get_limit_fx_data(ticker, act=act).net
    
    # max
    df_max_net = get_limit_fx_data(ticker, act=-2).net

    chart_data = pd.DataFrame({
        'min': df_min_net,
        f'fx_{act}': df_fx_net,
        'max': df_max_net
    })
    
    st.write('Refer_Log')
    st.line_chart(chart_data)

    df_plot_burn = get_limit_fx_data(ticker, act=-1)
    df_burn_cumsum = df_plot_burn[['buffer']].cumsum()
    st.write('Burn_Cash')
    st.line_chart(df_burn_cumsum)
    st.write(df_plot_burn)

# --- ส่วนแสดงผลบน Streamlit ---

# เชื่อมต่อ Thingspeak
client = thingspeak.Channel(CHANNEL_ID, WRITE_API_KEY, fmt='json')

# สร้างชื่อ Tab ทั้งหมดแบบไดนามิก
tab_names = [asset['tab_name'] for asset in ASSETS]
static_tabs = ['Burn_Cash', 'Ref_index_Log', 'cf_log']
all_tabs = st.tabs(tab_names + static_tabs)

# --- สร้าง Tab สำหรับแต่ละ Asset โดยใช้ Loop ---
for i, asset in enumerate(ASSETS):
    with all_tabs[i]:
        try:
            field_num = asset['thingspeak_field']
            # ดึงค่า act จาก Thingspeak
            act_response = client.get_field_last(field=f'{field_num}')
            act_value = int(json.loads(act_response)[f"field{field_num}"])
            
            # เรียกใช้ฟังก์ชัน plot
            plot_asset(ticker=asset['ticker'], act=act_value)
        
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการดึงข้อมูลหรือแสดงผลสำหรับ {asset['ticker']}: {e}")
            st.warning("อาจเกิดจากไม่มีข้อมูลใน Thingspeak Field หรือการเชื่อมต่อมีปัญหา")

# --- Tab: Burn_Cash ---
with all_tabs[len(ASSETS)]: # Index ถัดจาก Asset tabs
    st.header("Cumulative Burn Cash (All Assets)")
    
    # ใช้ ALL_TICKERS ที่สร้างจาก config
    buffers = {
        f"buffer_{symbol}": get_limit_fx_data(symbol, act=-1).buffer
        for symbol in ALL_TICKERS
    }
    
    df_burn_cash = pd.DataFrame(buffers)
    df_burn_cash['daily_burn'] = df_burn_cash.sum(axis=1)
    df_burn_cash['cumulative_burn'] = df_burn_cash['daily_burn'].cumsum()
    
    st.line_chart(df_burn_cash['cumulative_burn'])
    with st.expander("ดูข้อมูลดิบ"):
        st.dataframe(df_burn_cash.reset_index(drop=True))

# --- Tab: Ref_index_Log ---
with all_tabs[len(ASSETS) + 1]:
    st.header("Reference Index Log (All Assets)")

    @st.cache_data(ttl=3600)
    def get_ref_index_data(tickers, start_date):
        df_list = []
        for ticker in tickers:
            try:
                tickerData = yf.Ticker(ticker)
                tickerHist = tickerData.history(period='max')[['Close']]
                if tickerHist.empty:
                    st.warning(f"ไม่พบข้อมูลสำหรับ Ticker: {ticker}")
                    continue
                tickerHist.index = tickerHist.index.tz_convert(tz='Asia/Bangkok')
                tickerHist = tickerHist[tickerHist.index >= start_date]
                tickerHist = tickerHist.rename(columns={'Close': ticker})
                df_list.append(tickerHist)
            except Exception as e:
                st.error(f"ไม่สามารถดาวน์โหลดข้อมูลสำหรับ {ticker}: {e}")
        
        if not df_list:
            return pd.Series(dtype=float)

        prices_df = pd.concat(df_list, axis=1).dropna()
        
        # Check if prices_df is empty after operations
        if prices_df.empty:
            st.warning("ไม่มีข้อมูลราคาร่วมกันในช่วงเวลาที่กำหนด")
            return pd.Series(dtype=float)

        valid_tickers = prices_df.columns.tolist() # Use only tickers that had data
        int_st_list = prices_df.iloc[0][valid_tickers]
        int_st = np.prod(int_st_list)
        
        initial_capital_per_stock = FIX_VALUE * 2
        initial_capital_ref_index_log = initial_capital_per_stock * len(valid_tickers)
        
        def calculate_ref_log(row):
            int_end = np.prod(row[valid_tickers])
            # Avoid division by zero or log of zero
            if int_st <= 0 or int_end <= 0:
                return np.nan
            return initial_capital_ref_index_log + (-FIX_VALUE * np.log(int_st / int_end))
        
        prices_df['ref_log'] = prices_df.apply(calculate_ref_log, axis=1)
        return prices_df['ref_log'].dropna()

    ref_log_series = get_ref_index_data(ALL_TICKERS, FILTER_DATE)
    
    if not ref_log_series.empty:
        sumusd_data = {
            f"sumusd_{symbol}": get_limit_fx_data(symbol, act=-1).sumusd 
            for symbol in ref_log_series.index.get_level_values(0).unique() if symbol in ALL_TICKERS
        }
        df_sumusd_ = pd.DataFrame(sumusd_data)
        df_sumusd_['daily_sumusd'] = df_sumusd_.sum(axis=1)
        
        # Align indices before calculation
        df_sumusd_, ref_log_series = df_sumusd_.align(ref_log_series, axis=0, join='inner')
        df_sumusd_['ref_log'] = ref_log_series
        
        total_initial_capital = sum(
            get_limit_fx_data(symbol, act=-1).sumusd.iloc[0] for symbol in df_sumusd_.columns if symbol.startswith('sumusd_')
        )
        
        net_raw = df_sumusd_['daily_sumusd'] - df_sumusd_['ref_log'] - total_initial_capital
        if not net_raw.empty:
            net_at_index_0 = net_raw.iloc[0]
            df_sumusd_['net'] = net_raw - net_at_index_0
            st.line_chart(df_sumusd_.net)
            with st.expander("ดูข้อมูลดิบ"):
                st.dataframe(df_sumusd_.reset_index())
    else:
        st.warning("ไม่สามารถสร้างกราฟ Ref_index_Log ได้ เนื่องจากไม่มีข้อมูล")


# --- Tab: cf_log ---
with all_tabs[len(ASSETS) + 2]:
    st.header("Calculation Formulas & References")

    st.write('`Rebalance = -fix * ln( t0 / tn )`')
    st.write('`Net Profit = sumusd - refer - sumusd[0] (ต้นทุนเริ่มต้น)`')
    st.write('`Ref_index_Log = initial_capital_Ref_index_Log + (-1500 * ln(int_st / int_end))`')
    st.write('`Net in Ref_index_Log = (daily_sumusd - ref_log - total_initial_capital) - net_at_index_0`')
    st.write('---')

    def iframe(src, width=1500, height=800, scrolling=False):
        st.components.v1.iframe(src, width=width, height=height, scrolling=scrolling)

    for frame in IFRAMES:
        # ----- จุดที่แก้ไข -----
        # เปลี่ยนจาก `frame=` เป็น `src=` เพื่อให้ตรงกับนิยามของฟังก์ชัน
        iframe(src=frame['src'])
        st.write('---')
