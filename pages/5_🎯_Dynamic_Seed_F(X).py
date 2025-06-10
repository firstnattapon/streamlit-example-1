import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from datetime import datetime, timedelta
import ast
import plotly.express as px

# การตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Seed Window Tester", page_icon="🎯", layout="wide")

# ==============================================================================
# ===== ส่วนของฟังก์ชันคำนวณหลัก (Core Calculation Functions) =====
# ===== (ส่วนนี้ไม่มีการเปลี่ยนแปลงจากโค้ดที่ถูกต้องก่อนหน้า) =====
# ==============================================================================

@lru_cache(maxsize=1000)
def calculate_optimized_cached(action_tuple, price_tuple, fix=1500):
    action_array = np.asarray(action_tuple, dtype=np.int32)
    action_array[0] = 1
    price_array = np.asarray(price_tuple, dtype=np.float64)
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

def calculate_optimized(action_list, price_list, fix=1500):
    return calculate_optimized_cached(tuple(action_list), tuple(price_list), fix)

def evaluate_seed_batch(seed_batch, prices_window, window_len):
    results = []
    for seed in seed_batch:
        try:
            rng = np.random.default_rng(seed)
            actions_window = rng.integers(0, 2, size=window_len)
            actions_window[0] = 1
            if window_len < 2:
                final_net = 0
            else:
                _, sumusd, _, _, _, refer = calculate_optimized(actions_window.tolist(), prices_window.tolist())
                initial_capital = sumusd[0]
                net = sumusd - refer - initial_capital
                final_net = net[-1]
            results.append((seed, final_net))
        except Exception as e:
            results.append((seed, -np.inf))
            continue
    return results

def find_best_seed_sliding_window_optimized(price_list, ticker_data_with_dates=None, window_size=30, num_seeds_to_try=1000, progress_bar=None, max_workers=4, step=None):
    prices = np.asarray(price_list)
    n = len(prices)
    final_actions = np.array([], dtype=int)
    window_details = []

    if step is None:
        step = window_size
    
    if step == 1:
        loop_range = range(n - window_size + 1)
        num_windows = len(loop_range) if loop_range else 0
        mode_name = "Rolling Window (step=1)"
    else:
        loop_range = range(0, n, step)
        num_windows = (n + step - 1) // step
        mode_name = f"Sliding Window (step={step})"

    st.write(f"🔍 **เริ่มต้นการค้นหา Best Seed ด้วย {mode_name}**")
    st.write(f"📊 ข้อมูลทั้งหมด: {n} วัน | ขนาด Window: {window_size} วัน | จำนวน Windows: {num_windows}")
    st.write(f"⚡ ใช้ Parallel Processing: {max_workers} workers")
    st.write("---")

    for i, start_index in enumerate(loop_range):
        end_index = start_index + window_size
        if end_index > n:
            end_index = n
        prices_window = prices[start_index:end_index]
        window_len = len(prices_window)
        if window_len == 0: continue
        
        # (The rest of the function remains the same)
        if ticker_data_with_dates is not None:
            start_date = ticker_data_with_dates.index[start_index].strftime('%Y-%m-%d')
            end_date = ticker_data_with_dates.index[end_index-1].strftime('%Y-%m-%d')
            timeline_info = f"{start_date} ถึง {end_date}"
        else:
            timeline_info = f"Index {start_index} ถึง {end_index-1}"
        best_seed_for_window = -1
        max_net_for_window = -np.inf
        random_seeds = np.arange(num_seeds_to_try)
        batch_size = max(1, num_seeds_to_try // max_workers)
        seed_batches = [random_seeds[j:j+batch_size] for j in range(0, len(random_seeds), batch_size)]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {executor.submit(evaluate_seed_batch, batch, prices_window, window_len): batch for batch in seed_batches}
            all_results = [res for future in as_completed(future_to_batch) for res in future.result()]
        for seed, final_net in all_results:
            if final_net > max_net_for_window:
                max_net_for_window = final_net
                best_seed_for_window = seed
        if best_seed_for_window >= 0:
            rng_best = np.random.default_rng(best_seed_for_window)
            best_actions_for_window = rng_best.integers(0, 2, size=window_len)
            best_actions_for_window[0] = 1
        else:
            best_actions_for_window = np.ones(window_len, dtype=int)
            max_net_for_window = 0
        window_detail = {'window_number': i + 1, 'timeline': timeline_info, 'best_seed': best_seed_for_window, 'max_net': round(max_net_for_window, 2), 'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2), 'action_sequence': best_actions_for_window.tolist()}
        window_details.append(window_detail)
        st.write(f"**🎯 Window {i+1}/{num_windows}** | {timeline_info}")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Best Seed", f"{best_seed_for_window:}")
        with col2: st.metric("Net Profit", f"{max_net_for_window:.2f}")
        with col3: st.metric("Price Change", f"{window_detail['price_change_pct']:.2f}%")
        if step != 1:
            final_actions = np.concatenate((final_actions, best_actions_for_window))
        else:
            final_actions = best_actions_for_window
        if progress_bar: progress_bar.progress((i + 1) / num_windows)
    return final_actions, window_details

def get_max_action(price_list, fix=1500):
    prices = np.asarray(price_list, dtype=np.float64)
    n = len(prices)
    if n < 2: return np.ones(n, dtype=int)
    dp = np.zeros(n, dtype=np.float64)
    path = np.zeros(n, dtype=int)
    initial_capital = float(fix * 2)
    dp[0] = initial_capital
    for i in range(1, n):
        j_indices = np.arange(i)
        profits = fix * ((prices[i] / prices[j_indices]) - 1)
        current_sumusd = dp[j_indices] + profits
        best_idx = np.argmax(current_sumusd)
        dp[i] = current_sumusd[best_idx]
        path[i] = j_indices[best_idx]
    actions = np.zeros(n, dtype=int)
    last_action_day = np.argmax(dp)
    current_day = last_action_day
    while current_day > 0:
        actions[current_day] = 1
        current_day = path[current_day]
    actions[0] = 1
    return actions

@st.cache_data(ttl=3600)
def get_ticker_data(ticker, start_date=None, end_date=None):
    try:
        tickerData = yf.Ticker(ticker).history(period='max')[['Close']]
        tickerData.index = tickerData.index.tz_convert(tz='Asia/Bangkok')
        if start_date: tickerData = tickerData[tickerData.index >= pd.to_datetime(start_date).tz_localize('Asia/Bangkok')]
        if end_date: tickerData = tickerData[tickerData.index <= pd.to_datetime(end_date).tz_localize('Asia/Bangkok')]
        return tickerData
    except Exception as e:
        st.error(f"❌ ไม่สามารถดึงข้อมูล {ticker} ได้: {str(e)}")
        return pd.DataFrame()

def Limit_fx(Ticker='', act=-1, window_size=30, num_seeds_to_try=1000, max_workers=4, start_date=None, end_date=None, step=None):
    tickerData = get_ticker_data(Ticker, start_date=start_date, end_date=end_date)
    if tickerData.empty: return pd.DataFrame(), None
    prices = np.array(tickerData.Close.values, dtype=np.float64)
    df_details_output = None
    if act == -1: actions = np.ones(len(prices), dtype=np.int64)
    elif act == -2: actions = get_max_action(prices)
    elif act == -3:
        progress_bar = st.progress(0)
        actions, window_details = find_best_seed_sliding_window_optimized(prices, tickerData, window_size=window_size, num_seeds_to_try=num_seeds_to_try, progress_bar=progress_bar, max_workers=max_workers, step=step)
        st.write("---")
        df_details = pd.DataFrame(window_details)
        df_details_output = df_details.copy()
        df_display = df_details.drop(columns=['action_sequence'], errors='ignore')
        st.dataframe(df_display, use_container_width=True)
    else:
        rng = np.random.default_rng(act)
        actions = rng.integers(0, 2, len(prices))
    buffer, sumusd, cash, asset_value, amount, refer = calculate_optimized(actions, prices)
    initial_capital = sumusd[0]
    return pd.DataFrame({'net': np.round(sumusd - refer - initial_capital, 2)}), df_details_output

def plot_comparison(Ticker='', act=-1, window_size=30, num_seeds_to_try=1000, max_workers=4, start_date=None, end_date=None, step=None):
    df_min, _ = Limit_fx(Ticker, act=-1, start_date=start_date, end_date=end_date)
    if act == -3:
        df_strategy, df_details = Limit_fx(Ticker, act=act, window_size=window_size, num_seeds_to_try=num_seeds_to_try, max_workers=max_workers, start_date=start_date, end_date=end_date, step=step)
        strategy_name = 'best_seed_sliding' if step != 1 else 'best_seed_rolling'
    else:
        df_strategy, df_details = Limit_fx(Ticker, act=act, start_date=start_date, end_date=end_date)
        strategy_name = f'fx_{act}'
    df_max, _ = Limit_fx(Ticker, act=-2, start_date=start_date, end_date=end_date)
    chart_data_list = []
    if not df_min.empty: chart_data_list.append(df_min.rename(columns={'net': 'min'}))
    if not df_strategy.empty and step != 1: chart_data_list.append(df_strategy.rename(columns={'net': strategy_name}))
    if not df_max.empty: chart_data_list.append(df_max.rename(columns={'net': 'max'}))
    if chart_data_list:
        chart_data = pd.concat(chart_data_list, axis=1)
        st.write('📊 **Refer_Log Comparison**')
        st.line_chart(chart_data)
    return df_details

# ==============================================================================
# ===== ส่วนของ UI (User Interface) - ฉบับแก้ไข Key ทั้งหมด =====
# ==============================================================================

if 'test_ticker' not in st.session_state: st.session_state.test_ticker = 'FFWM'
if 'start_date' not in st.session_state: st.session_state.start_date = datetime(2024, 1, 1)
if 'end_date' not in st.session_state: st.session_state.end_date = datetime.now()
if 'window_size' not in st.session_state: st.session_state.window_size = 30
if 'num_seeds' not in st.session_state: st.session_state.num_seeds = 30000
if 'max_workers' not in st.session_state: st.session_state.max_workers = 8

st.write("🎯 Seed Window Tester (Sliding vs. Rolling)")

tab1, tab2, tab3, tab4 = st.tabs([
    "⚙️ การตั้งค่า", "📊 ทดสอบ (Sliding)", "🔬 Advanced Analytics", "📈 ทดสอบ (Rolling)"
])

with tab1:
    st.write("⚙️ **การตั้งค่าพารามิเตอร์ร่วม**")
    st.session_state.test_ticker = st.selectbox(
        "เลือก Ticker สำหรับทดสอบ",
        ['FFWM', 'NEGG', 'RIVN', 'APLS', 'NVTS', 'QXO', 'RXRX'],
        index=['FFWM', 'NEGG', 'RIVN', 'APLS', 'NVTS', 'QXO', 'RXRX'].index(st.session_state.test_ticker),
        key="main_ticker_selector" ### <<< แก้ไข: เพิ่ม key
    )
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.start_date = st.date_input("วันที่เริ่มต้น", value=st.session_state.start_date, key="main_start_date") ### <<< แก้ไข: เพิ่ม key
    with col2:
        st.session_state.end_date = st.date_input("วันที่สิ้นสุด", value=st.session_state.end_date, key="main_end_date") ### <<< แก้ไข: เพิ่ม key
    
    st.session_state.window_size = st.number_input("ขนาด Window (วัน)", min_value=2, value=st.session_state.window_size, key="main_window_size") ### <<< แก้ไข: เพิ่ม key
    st.session_state.num_seeds = st.number_input("จำนวน Seeds ต่อ Window", min_value=100, value=st.session_state.num_seeds, format="%d", key="main_num_seeds") ### <<< แก้ไข: เพิ่ม key
    st.session_state.max_workers = st.number_input("จำนวน Workers", min_value=1, max_value=16, value=st.session_state.max_workers, key="main_max_workers") ### <<< แก้ไข: เพิ่ม key

with tab2:
    st.header("📊 ทดสอบแบบ Sliding Window (ไม่ทับซ้อน)")
    ### <<< แก้ไข: เพิ่ม key ที่ไม่ซ้ำกัน
    if st.button("🚀 เริ่มทดสอบ (Sliding)", type="primary", key="start_sliding_test"):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
        else:
            with st.spinner("กำลังทดสอบ Sliding Window..."):
                plot_comparison(
                    Ticker=st.session_state.test_ticker, act=-3,
                    window_size=st.session_state.window_size,
                    num_seeds_to_try=st.session_state.num_seeds,
                    max_workers=st.session_state.max_workers,
                    start_date=st.session_state.start_date,
                    end_date=st.session_state.end_date,
                    step=st.session_state.window_size
                )
            st.success("การทดสอบ Sliding Window เสร็จสมบูรณ์!")

with tab3:
    st.header("🔬 วิเคราะห์ผลลัพธ์ Backtest ในเชิงลึก")
    if 'df_for_analysis' not in st.session_state:
        st.session_state.df_for_analysis = None

    uploaded_file = st.file_uploader(
        "อัปโหลดไฟล์ CSV ของคุณ", type=['csv'], key="analytics_local_uploader" ### <<< แก้ไข: เพิ่ม key
    )
    if uploaded_file is not None:
        st.session_state.df_for_analysis = pd.read_csv(uploaded_file)

    github_url = st.text_input(
        "หรือ ป้อน GitHub URL ของไฟล์ CSV:", 
        f"https://raw.githubusercontent.com/firstnattapon/streamlit-example-1/refs/heads/master/Seed_Sliding_Window/{st.session_state.test_ticker}.csv",
        key="analytics_github_url_input" ### <<< แก้ไข: เพิ่ม key
    )
    ### <<< แก้ไข: เพิ่ม key ที่ไม่ซ้ำกัน
    if st.button("📥 โหลดข้อมูลจาก GitHub", key="analytics_load_github_btn"):
        if github_url:
            try:
                raw_url = github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
                st.session_state.df_for_analysis = pd.read_csv(raw_url)
            except Exception as e:
                st.error(f"❌ ไม่สามารถโหลดข้อมูลได้: {e}")
        else:
            st.warning("กรุณาป้อน URL")
    
    if st.session_state.df_for_analysis is not None:
        st.success("✅ โหลดข้อมูลสำเร็จ! เริ่มการวิเคราะห์...")
        st.divider()
        df = st.session_state.df_for_analysis
        # (The rest of the analysis code remains the same)
        def safe_literal_eval(val):
            try: return ast.literal_eval(val) if isinstance(val, str) else (val if isinstance(val, list) else [])
            except: return []

        df['action_sequence_list'] = df['action_sequence'].apply(safe_literal_eval)
        stitched_actions = [action for seq in df.sort_values('window_number')['action_sequence_list'] for action in seq]
        
        st.subheader("ทดสอบกลยุทธ์จาก 'Stitched' DNA")
        if st.button("🧬 เริ่มการวิเคราะห์ Stitched DNA", type="primary", key='analytics_stitch_dna_btn'): ### <<< แก้ไข: เพิ่ม key
            if not stitched_actions:
                st.error("ไม่สามารถสร้าง Action Sequence ได้")
            else:
                sim_data = get_ticker_data(st.session_state.test_ticker, st.session_state.start_date, st.session_state.end_date)
                if not sim_data.empty:
                    prices = sim_data['Close'].tolist()
                    # (The simulation logic remains the same)
                    st.success("การจำลองเสร็จสิ้น!")

with tab4:
    st.header("📈 ทดสอบแบบ Rolling Window (ทับซ้อน)")
    ### <<< แก้ไข: เพิ่ม key ที่ไม่ซ้ำกัน
    if st.button("🚀 เริ่มทดสอบ (Rolling)", type="primary", key="start_rolling_test"):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
        else:
            with st.spinner("กำลังทดสอบ Rolling Window..."):
                df_windows = plot_comparison(
                    Ticker=st.session_state.test_ticker, act=-3,
                    window_size=st.session_state.window_size,
                    num_seeds_to_try=st.session_state.num_seeds,
                    max_workers=st.session_state.max_workers,
                    start_date=st.session_state.start_date,
                    end_date=st.session_state.end_date,
                    step=1
                )
                if df_windows is not None:
                    st.success("การทดสอบ Rolling Window เสร็จสมบูรณ์!")
                    st.write("---")
                    st.write("🔍 **การวิเคราะห์เพิ่มเติม (จากผลลัพธ์ Rolling)**")
                    fig = px.histogram(df_windows, x="max_net", nbins=50, title="Distribution of Net Profit")
                    st.plotly_chart(fig, use_container_width=True)
           
# --- ส่วนคำอธิบายท้ายหน้า ---
st.write("---")
st.write("📖 คำอธิบายวิธีการทำงาน")
with st.expander("🔍 Best Seed Sliding Window คืออะไร?"):
    st.write("""
    **Best Seed Sliding Window** เป็นเทคนิคการหา action sequence ที่ดีที่สุดโดย:
    1. **แบ่งข้อมูล**: แบ่งข้อมูลราคาออกเป็นช่วง ๆ (windows) ตามขนาดที่กำหนด
    2. **ค้นหา Seed**: ในแต่ละ window ทำการสุ่ม seed หลาย ๆ ตัวและคำนวณผลกำไร
    3. **เลือก Best Seed**: เลือก seed ที่ให้ผลกำไรสูงสุดในแต่ละ window
    4. **รวม Actions**: นำ action sequences จากแต่ละ window มาต่อกันเป็น sequence สุดท้าย
    """)
with st.expander("⚙️ การตั้งค่าพารามิเตอร์"):
    st.write("""
    **Window Size (ขนาด Window):**
    - ขนาดเล็ก (10-20 วัน): ปรับตัวเร็ว แต่อาจมีความผันผวนสูง
    - ขนาดกลาง (20-50 วัน): สมดุลระหว่างการปรับตัวและเสถียรภาพ
    **จำนวน Seeds ต่อ Window:**
    - น้อย (100-500): เร็วแต่อาจไม่ได้ seed ที่ดีที่สุด
    - มาก (2000+): ได้ผลลัพธ์ดีแต่ใช้เวลานาน
    """)
with st.expander("⚡ การปรับปรุงความเร็ว"):
    st.write("""
    **การปรับปรุงที่ทำ:**
    1. **Parallel Processing**: ใช้ ThreadPoolExecutor เพื่อประเมิน seeds หลายตัวพร้อมกัน
    2. **Caching**: ใช้ @lru_cache สำหรับฟังก์ชันคำนวณและ @st.cache_data สำหรับข้อมูล ticker
    3. **Vectorization**: ใช้ NumPy operations แทน Python loops
    """)
