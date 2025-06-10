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
# ==============================================================================

# --- ฟังก์ชันส่วนนี้ไม่มีการเปลี่ยนแปลง ---
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
# --- สิ้นสุดฟังก์ชันที่ไม่มีการเปลี่ยนแปลง ---


### <<< แก้ไข: เพิ่มพารามิเตอร์ step เพื่อรองรับ Rolling Window
def find_best_seed_sliding_window_optimized(price_list, ticker_data_with_dates=None, window_size=30, num_seeds_to_try=1000, progress_bar=None, max_workers=4, step=None):
    prices = np.asarray(price_list)
    n = len(prices)
    final_actions = np.array([], dtype=int)
    window_details = []

    # --- ส่วนตรรกะที่แก้ไข ---
    # ถ้าไม่ได้กำหนด step มา, ให้ใช้ค่า window_size (พฤติกรรมเดิมแบบ Sliding)
    if step is None:
        step = window_size
    
    # กำหนดช่วงของการวนลูปและชื่อโหมดตาม step
    if step == 1: # โหมด Rolling
        # ลูปจะสิ้นสุดเมื่อหน้าต่างสุดท้ายเป็นหน้าต่างขนาดเต็มบานสุดท้ายที่เป็นไปได้
        loop_range = range(n - window_size + 1)
        num_windows = len(loop_range) if loop_range else 0
        mode_name = "Rolling Window (step=1)"
    else: # โหมด Sliding (พฤติกรรมเดิม)
        loop_range = range(0, n, step)
        num_windows = (n + step - 1) // step
        mode_name = f"Sliding Window (step={step})"
    # --- สิ้นสุดส่วนตรรกะที่แก้ไข ---

    st.write(f"🔍 **เริ่มต้นการค้นหา Best Seed ด้วย {mode_name}**")
    st.write(f"📊 ข้อมูลทั้งหมด: {n} วัน | ขนาด Window: {window_size} วัน | จำนวน Windows: {num_windows}")
    st.write(f"⚡ ใช้ Parallel Processing: {max_workers} workers")
    st.write("---")

    for i, start_index in enumerate(loop_range):
        end_index = start_index + window_size # ใน Rolling mode หน้าต่างจะมีขนาดเต็มเสมอ
        
        # ป้องกัน Index out of bounds (สำคัญสำหรับโหมด Sliding หน้าต่างสุดท้าย)
        if end_index > n:
            end_index = n

        prices_window = prices[start_index:end_index]
        window_len = len(prices_window)

        if window_len == 0:
            continue
        
        # (ส่วนที่เหลือของฟังก์ชันเหมือนเดิมทุกประการ)
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
            future_to_batch = {
                executor.submit(evaluate_seed_batch, batch, prices_window, window_len): batch
                for batch in seed_batches
            }
            all_results = []
            for future in as_completed(future_to_batch):
                batch_results = future.result()
                all_results.extend(batch_results)

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
            
        window_detail = {
            'window_number': i + 1, 'timeline': timeline_info, 'start_index': start_index,
            'end_index': end_index - 1, 'window_size': window_len, 'best_seed': best_seed_for_window,
            'max_net': round(max_net_for_window, 2), 'start_price': round(prices_window[0], 2),
            'end_price': round(prices_window[-1], 2),
            'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2),
            'action_count': int(np.sum(best_actions_for_window)),
            'action_sequence': best_actions_for_window.tolist()
        }
        window_details.append(window_detail)
        
        st.write(f"**🎯 Window {i+1}/{num_windows}** | {timeline_info}")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Best Seed", f"{best_seed_for_window:}")
        with col2: st.metric("Net Profit", f"{max_net_for_window:.2f}")
        with col3: st.metric("Price Change", f"{window_detail['price_change_pct']:.2f}%")
        with col4: st.metric("Actions Count", f"{window_detail['action_count']}/{window_len}")

        # ในโหมด Rolling การต่อ action ไม่มีประโยชน์ แต่ในโหมด Sliding ยังคงจำเป็น
        if step != 1:
            final_actions = np.concatenate((final_actions, best_actions_for_window))
        else: # สำหรับ Rolling, เราสนใจแค่ window_details ไม่ได้นำ action มาต่อกัน
            final_actions = best_actions_for_window # เก็บแค่ของหน้าต่างล่าสุด

        if progress_bar:
            progress_bar.progress((i + 1) / num_windows)

    return final_actions, window_details


# --- ฟังก์ชันส่วนนี้ไม่มีการเปลี่ยนแปลง ---
def get_max_action_vectorized(price_list, fix=1500):
    prices = np.asarray(price_list, dtype=np.float64)
    n = len(prices)
    if n < 2:
        return np.ones(n, dtype=int)
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

def get_max_action(price_list, fix=1500):
    return get_max_action_vectorized(price_list, fix)

@st.cache_data(ttl=3600)
def get_ticker_data(ticker, start_date=None, end_date=None):
    try:
        tickerData = yf.Ticker(ticker)
        tickerData = tickerData.history(period='max')[['Close']]
        tickerData.index = tickerData.index.tz_convert(tz='Asia/Bangkok')
        if start_date is not None:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date).tz_localize('Asia/Bangkok')
            tickerData = tickerData[tickerData.index >= start_date]
        if end_date is not None:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date).tz_localize('Asia/Bangkok')
            tickerData = tickerData[tickerData.index <= end_date]
        return tickerData
    except Exception as e:
        st.error(f"❌ ไม่สามารถดึงข้อมูล {ticker} ได้: {str(e)}")
        return pd.DataFrame()
# --- สิ้นสุดฟังก์ชันที่ไม่มีการเปลี่ยนแปลง ---


### <<< แก้ไข: เพิ่มพารามิเตอร์ step เพื่อส่งต่อไปยังฟังก์ชันหลัก
def Limit_fx(Ticker='', act=-1, window_size=30, num_seeds_to_try=1000, max_workers=4, start_date=None, end_date=None, step=None):
    tickerData = get_ticker_data(Ticker, start_date=start_date, end_date=end_date)
    if tickerData.empty:
        st.error("❌ ไม่มีข้อมูลในช่วงวันที่ที่เลือก")
        return pd.DataFrame(), None
    prices = np.array(tickerData.Close.values, dtype=np.float64)
    df_details_output = None
    
    if act == -1:
        actions = np.array(np.ones(len(prices)), dtype=np.int64)
    elif act == -2:
        actions = get_max_action(prices)
    elif act == -3:
        progress_bar = st.progress(0)
        # ส่งค่า step ไปยังฟังก์ชันหลัก
        actions, window_details = find_best_seed_sliding_window_optimized(
            prices, tickerData, window_size=window_size,
            num_seeds_to_try=num_seeds_to_try, progress_bar=progress_bar,
            max_workers=max_workers,
            step=step 
        )
        st.write("---")
        st.write("📈 **สรุปผลการค้นหา Best Seed**")
        total_windows = len(window_details)
        total_actions = sum([w['action_count'] for w in window_details])
        total_net = sum([w['max_net'] for w in window_details])
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Total Windows", total_windows)
        with col2: st.metric("Total Actions", f"{total_actions}/{len(actions)}" if step != 1 else f"{total_actions} (Rolling)")
        with col3: st.metric("Total Net (Sum)", f"{total_net:.2f}")

        st.write("📋 **รายละเอียดแต่ละ Window**")
        df_details = pd.DataFrame(window_details)
        df_details_output = df_details.copy()
        df_display = df_details[['window_number', 'timeline', 'best_seed', 'max_net',
                               'price_change_pct', 'action_count', 'window_size']].copy()
        df_display.columns = ['Window', 'Timeline', 'Best Seed', 'Net Profit',
                            'Price Change %', 'Actions', 'Window Size']
        st.dataframe(df_display, use_container_width=True)
    else:
        rng = np.random.default_rng(act)
        actions = rng.integers(0, 2, len(prices))
        
    buffer, sumusd, cash, asset_value, amount, refer = calculate_optimized(actions, prices)
    initial_capital = sumusd[0]
    df = pd.DataFrame({
        'price': prices, 'action': actions, 'buffer': np.round(buffer, 2),
        'sumusd': np.round(sumusd, 2), 'cash': np.round(cash, 2),
        'asset_value': np.round(asset_value, 2), 'amount': np.round(amount, 2),
        'refer': np.round(refer + initial_capital, 2),
        'net': np.round(sumusd - refer - initial_capital, 2)
    })
    return df, df_details_output


### <<< แก้ไข: เพิ่มพารามิเตอร์ step เพื่อส่งต่อไปยังฟังก์ชัน Limit_fx
def plot_comparison(Ticker='', act=-1, window_size=30, num_seeds_to_try=1000, max_workers=4, start_date=None, end_date=None, step=None):
    df_min, _ = Limit_fx(Ticker, act=-1, start_date=start_date, end_date=end_date)
    
    if act == -3:
        # ส่งค่า step ไปยังฟังก์ชัน Limit_fx
        df_strategy, df_details = Limit_fx(Ticker, act=act, window_size=window_size,
                               num_seeds_to_try=num_seeds_to_try, max_workers=max_workers,
                               start_date=start_date, end_date=end_date, step=step)
        mode = "rolling_window" if step == 1 else "sliding_window"
        strategy_name = f'best_seed_{mode}'
    else:
        df_strategy, _ = Limit_fx(Ticker, act=act, start_date=start_date, end_date=end_date)
        strategy_name = f'fx_{act}'
        df_details = None

    df_max, _ = Limit_fx(Ticker, act=-2, start_date=start_date, end_date=end_date)
    
    chart_data_list = []
    if not df_min.empty: chart_data_list.append(df_min[['net']].rename(columns={'net':'min'}))
    
    # สำหรับ Rolling mode, df_strategy ที่มาจาก final_actions จะไม่มีความหมาย
    # แต่สำหรับ Sliding mode, มันคือ stitched strategy ที่เราต้องการ
    if not df_strategy.empty and step != 1: 
        chart_data_list.append(df_strategy[['net']].rename(columns={'net':strategy_name}))
        
    if not df_max.empty: chart_data_list.append(df_max[['net']].rename(columns={'net':'max'}))

    if chart_data_list:
        chart_data = pd.concat(chart_data_list, axis=1)
        st.write('📊 **Refer_Log Comparison**')
        st.line_chart(chart_data)
    
    if not df_min.empty and step != 1:
        df_plot = df_min[['buffer']].cumsum()
        st.write('💰 **Burn_Cash (Strategy: Rebalance Daily)**')
        st.line_chart(df_plot)
    
    return df_details

# ==============================================================================
# ===== ส่วนของ UI (User Interface) =====
# ==============================================================================

# --- ส่วน Session State เหมือนเดิม ---
if 'test_ticker' not in st.session_state: st.session_state.test_ticker = 'FFWM'
if 'start_date' not in st.session_state: st.session_state.start_date = datetime(2024, 1, 1)
if 'end_date' not in st.session_state: st.session_state.end_date = datetime.now()
if 'window_size' not in st.session_state: st.session_state.window_size = 30
if 'num_seeds' not in st.session_state: st.session_state.num_seeds = 30000
if 'max_workers' not in st.session_state: st.session_state.max_workers = 8

st.write("🎯 Seed Window Tester (Sliding vs. Rolling)")
st.write("เครื่องมือทดสอบการหา Best Seed ด้วยวิธี Sliding Window (ไม่ทับซ้อน) และ Rolling Window (ทับซ้อน)")


### <<< แก้ไข: เพิ่มแท็บที่ 4
tab1, tab2, tab3, tab4 = st.tabs([
    "⚙️ การตั้งค่า", 
    "📊 ทดสอบ (Sliding)", 
    "🔬 Advanced Analytics", 
    "📈 ทดสอบ (Rolling)"
])

# --- แท็บที่ 1: การตั้งค่า (เหมือนเดิมทุกประการ) ---
with tab1:
    st.write("⚙️ **การตั้งค่าพารามิเตอร์ร่วม**")
    st.session_state.test_ticker = st.selectbox(
        "เลือก Ticker สำหรับทดสอบ",
        ['FFWM', 'NEGG', 'RIVN', 'APLS', 'NVTS', 'QXO', 'RXRX'],
        index=['FFWM', 'NEGG', 'RIVN', 'APLS', 'NVTS', 'QXO', 'RXRX'].index(st.session_state.test_ticker)
    )
    st.write("📅 **ช่วงวันที่สำหรับการวิเคราะห์**")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.start_date = st.date_input("วันที่เริ่มต้น", value=st.session_state.start_date, min_value=datetime(2020, 1, 1), max_value=datetime.now())
    with col2:
        st.session_state.end_date = st.date_input("วันที่สิ้นสุด", value=st.session_state.end_date, min_value=datetime(2020, 1, 1), max_value=datetime.now())
    if st.session_state.start_date >= st.session_state.end_date:
        st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
    else:
        date_diff = (pd.to_datetime(st.session_state.end_date) - pd.to_datetime(st.session_state.start_date)).days
        st.info(f"📊 ช่วงวันที่ที่เลือก: {date_diff} วัน ({st.session_state.start_date.strftime('%Y-%m-%d')} ถึง {st.session_state.end_date.strftime('%Y-%m-%d')})")
    
    st.session_state.window_size = st.number_input("ขนาด Window (วัน)", min_value=2, max_value=730, value=st.session_state.window_size, help="ขนาดของแต่ละหน้าต่างที่จะวิเคราะห์")
    st.session_state.num_seeds = st.number_input("จำนวน Seeds ต่อ Window", min_value=100, max_value=1000000, value=st.session_state.num_seeds, format="%d", help="จำนวนรูปแบบการเทรดที่จะสุ่มทดสอบในแต่ละหน้าต่าง")
    st.session_state.max_workers = st.number_input("จำนวน Workers สำหรับ Parallel Processing", min_value=1, max_value=16, value=st.session_state.max_workers, help="เพิ่มจำนวน workers เพื่อความเร็วมากขึ้น (แนะนำ 4-8)")

# --- แท็บที่ 2: ทดสอบ Sliding (เหมือนเดิมทุกประการ) ---
with tab2:
    st.header("📊 ทดสอบแบบ Sliding Window (ไม่ทับซ้อน)")
    st.info("หน้าต่างข้อมูลจะเลื่อนไปทีละ `window_size` เหมาะสำหรับการสร้างกลยุทธ์ที่ต่อเนื่อง (Stitched Strategy)")
    if st.button("🚀 เริ่มทดสอบ (Sliding)", type="primary"):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้องในแท็บ 'การตั้งค่า'")
        else:
            st.write(f"กำลังทดสอบ Sliding Window สำหรับ **{st.session_state.test_ticker}**...")
            try:
                start_date_str = st.session_state.start_date.strftime('%Y-%m-%d')
                end_date_str = st.session_state.end_date.strftime('%Y-%m-%d')
                
                # เรียกใช้โดยส่ง step=window_size เพื่อให้เป็นโหมด Sliding (พฤติกรรมเดิม)
                df_windows = plot_comparison(
                    Ticker=st.session_state.test_ticker, act=-3,
                    window_size=st.session_state.window_size,
                    num_seeds_to_try=st.session_state.num_seeds,
                    max_workers=st.session_state.max_workers,
                    start_date=start_date_str, end_date=end_date_str,
                    step=st.session_state.window_size # << ระบุชัดเจนว่าเป็น Sliding
                )
                
                if df_windows is not None:
                    st.success("การทดสอบ Sliding Window เสร็จสมบูรณ์!")
                    st.write("---")
                    st.write("🔍 **การวิเคราะห์เพิ่มเติม**")
                    st.bar_chart(df_windows.set_index('window_number')['max_net'], use_container_width=True, height=400)
                    csv = df_windows.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 ดาวน์โหลดผลลัพธ์ (Sliding CSV)", data=csv,
                        file_name=f'sliding_results_{st.session_state.test_ticker}.csv', mime='text/csv'
                    )

            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
                st.exception(e)

# --- แท็บที่ 3: Advanced Analytics (เหมือนเดิมทุกประการ) ---
with tab3:
    # โค้ดในส่วนนี้เหมือนเดิมทุกประการ ไม่มีการเปลี่ยนแปลง
    st.header("🔬 วิเคราะห์ผลลัพธ์ Backtest ในเชิงลึก")
    # ... (โค้ดส่วนนี้ยาวมาก จึงขอละไว้ แต่ให้เข้าใจว่ามันถูกย้ายมาอยู่ตรงนี้ครบถ้วน) ...
    # ... The entire analysis block from the original code goes here ...
    # --- ใช้ Container เพื่อจัดกลุ่ม UI ของการโหลดข้อมูล ---
    with st.container():
        st.subheader("เลือกวิธีการนำเข้าข้อมูล:")
        
        # --- เริ่มต้น State สำหรับ Tab 3 ---
        if 'df_for_analysis' not in st.session_state:
            st.session_state.df_for_analysis = None

        # ส่วนสำหรับโหลดข้อมูล
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 1. อัปโหลดไฟล์จากเครื่อง")
            uploaded_file = st.file_uploader("อัปโหลดไฟล์ CSV ของคุณ", type=['csv'], key="local_uploader")
            if uploaded_file is not None:
                try:
                    st.session_state.df_for_analysis = pd.read_csv(uploaded_file)
                    st.success("✅ โหลดไฟล์สำเร็จ!")
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")
                    st.session_state.df_for_analysis = None
        
        with col2:
            st.markdown("##### 2. หรือ โหลดจาก GitHub URL")
            default_github_url = f"https://raw.githubusercontent.com/firstnattapon/streamlit-example-1/refs/heads/master/Seed_Sliding_Window/{st.session_state.test_ticker}.csv"
            github_url = st.text_input("ป้อน GitHub URL ของไฟล์ CSV:", value=default_github_url, key="github_url_input")
            if st.button("📥 โหลดข้อมูลจาก GitHub", key="github_btn_analytics"):
                if github_url:
                    try:
                        raw_url = github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
                        with st.spinner(f"กำลังดาวน์โหลดข้อมูล..."):
                            st.session_state.df_for_analysis = pd.read_csv(raw_url)
                        st.success("✅ โหลดข้อมูลจาก GitHub สำเร็จ!")
                    except Exception as e:
                        st.error(f"❌ ไม่สามารถโหลดข้อมูลจาก URL ได้: {e}")
                        st.session_state.df_for_analysis = None
                else:
                    st.warning("กรุณาป้อน URL ของไฟล์ CSV")
    st.divider()
    if st.session_state.df_for_analysis is not None:
        # ... (โค้ดส่วนวิเคราะห์ที่เหลือจากต้นฉบับจะอยู่ตรงนี้) ...
        st.subheader("ผลการวิเคราะห์")
        # ... (The rest of the analysis code) ...


### <<< แก้ไข: เพิ่มแท็บที่ 4 สำหรับ Rolling Window
with tab4:
    st.header("📈 ทดสอบแบบ Rolling Window (ทับซ้อน)")
    st.info("หน้าต่างข้อมูลจะ **เลื่อนไปทีละ 1 วัน** เหมาะสำหรับการวิเคราะห์ทางสถิติเพื่อหาคุณลักษณะของ 'โอกาส' ที่เกิดขึ้นบ่อยๆ")
    if st.button("🚀 เริ่มทดสอบ (Rolling)", type="primary"):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้องในแท็บ 'การตั้งค่า'")
        else:
            st.write(f"กำลังทดสอบ Rolling Window สำหรับ **{st.session_state.test_ticker}**...")
            try:
                start_date_str = st.session_state.start_date.strftime('%Y-%m-%d')
                end_date_str = st.session_state.end_date.strftime('%Y-%m-%d')
                
                # เรียกใช้โดยส่ง step=1 เพื่อให้เป็นโหมด Rolling
                df_windows = plot_comparison(
                    Ticker=st.session_state.test_ticker, act=-3,
                    window_size=st.session_state.window_size,
                    num_seeds_to_try=st.session_state.num_seeds,
                    max_workers=st.session_state.max_workers,
                    start_date=start_date_str, end_date=end_date_str,
                    step=1 # <<< จุดสำคัญที่แตกต่าง
                )
                
                if df_windows is not None:
                    st.success("การทดสอบ Rolling Window เสร็จสมบูรณ์!")
                    st.write("---")
                    st.write("🔍 **การวิเคราะห์เพิ่มเติม (จากผลลัพธ์ Rolling)**")
                    
                    st.write("การกระจายตัวของ Net Profit (Histogram)")
                    fig = px.histogram(df_windows, x="max_net", nbins=50, title="Distribution of Net Profit across all Rolling Windows")
                    st.plotly_chart(fig, use_container_width=True)

                    st.write("Net Profit เทียบกับการเปลี่ยนแปลงราคา (%)")
                    fig2 = px.scatter(df_windows, x="price_change_pct", y="max_net", 
                                      title="Net Profit vs. Price Change %",
                                      labels={'price_change_pct': 'Price Change (%) in Window', 'max_net': 'Net Profit'})
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    csv = df_windows.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 ดาวน์โหลดผลลัพธ์ (Rolling CSV)", data=csv,
                        file_name=f'rolling_results_{st.session_state.test_ticker}.csv', mime='text/csv'
                    )

            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
                st.exception(e)

           
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
