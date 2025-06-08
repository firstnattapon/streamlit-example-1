import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from datetime import datetime, timedelta
 
st.set_page_config(page_title="Best Seed Sliding Window", page_icon="🎯", layout="wide")
 
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

def find_best_seed_sliding_window_optimized(price_list, ticker_data_with_dates=None, window_size=30, num_seeds_to_try=1000, progress_bar=None, max_workers=4):
    prices = np.asarray(price_list)
    n = len(prices)
    final_actions = np.array([], dtype=int)
    window_details = []
    num_windows = (n + window_size - 1) // window_size
    st.write("🔍 **เริ่มต้นการค้นหา Best Seed ด้วย Sliding Window (Optimized)**")
    st.write(f"📊 ข้อมูลทั้งหมด: {n} วัน | ขนาด Window: {window_size} วัน | จำนวน Windows: {num_windows}")
    st.write(f"⚡ ใช้ Parallel Processing: {max_workers} workers")
    st.write("---")
    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        window_len = len(prices_window)
        if window_len == 0:
            continue
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
        seed_batches = [random_seeds[i:i+batch_size] for i in range(0, len(random_seeds), batch_size)]
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
            'window_number': i + 1,
            'timeline': timeline_info,
            'start_index': start_index,
            'end_index': end_index - 1,
            'window_size': window_len,
            'best_seed': best_seed_for_window,
            'max_net': round(max_net_for_window, 2),
            'start_price': round(prices_window[0], 2),
            'end_price': round(prices_window[-1], 2),
            'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2),
            'action_count': int(np.sum(best_actions_for_window)),
            'action_sequence': best_actions_for_window.tolist()
        }
        window_details.append(window_detail)
        st.write(f"**🎯 Window {i+1}/{num_windows}** | {timeline_info}")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Best Seed", f"{best_seed_for_window:,}")
        with col2:
            st.metric("Net Profit", f"{max_net_for_window:.2f}")
        with col3:
            st.metric("Price Change", f"{window_detail['price_change_pct']:.2f}%")
        with col4:
            st.metric("Actions Count", f"{window_detail['action_count']}/{window_len}")
        final_actions = np.concatenate((final_actions, best_actions_for_window))
        if progress_bar:
            progress_bar.progress((i + 1) / num_windows)
    return final_actions, window_details

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
        if start_date is None and end_date is None:
            default_start = pd.to_datetime('2023-01-01 12:00:00').tz_localize('Asia/Bangkok')
            tickerData = tickerData[tickerData.index >= default_start]
        return tickerData
    except Exception as e:
        st.error(f"❌ ไม่สามารถดึงข้อมูล {ticker} ได้: {str(e)}")
        return pd.DataFrame()

def Limit_fx(Ticker='', act=-1, window_size=30, num_seeds_to_try=1000, max_workers=4, start_date=None, end_date=None):
    tickerData = get_ticker_data(Ticker, start_date=start_date, end_date=end_date)
    if tickerData.empty:
        st.error("❌ ไม่มีข้อมูลในช่วงวันที่ที่เลือก")
        return pd.DataFrame()
    prices = np.array(tickerData.Close.values, dtype=np.float64)
    if act == -1:
        actions = np.array(np.ones(len(prices)), dtype=np.int64)
    elif act == -2:
        actions = get_max_action(prices)
    elif act == -3:
        progress_bar = st.progress(0)
        actions, window_details = find_best_seed_sliding_window_optimized(
            prices, tickerData, window_size=window_size, 
            num_seeds_to_try=num_seeds_to_try, progress_bar=progress_bar,
            max_workers=max_workers
        )
        st.write("---")
        st.write("📈 **สรุปผลการค้นหา Best Seed**")
        total_windows = len(window_details)
        total_actions = sum([w['action_count'] for w in window_details])
        total_net = sum([w['max_net'] for w in window_details])
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Windows", total_windows)
        with col2:
            st.metric("Total Actions", f"{total_actions}/{len(actions)}")
        with col3:
            st.metric("Total Net (Sum)", f"{total_net:.2f}")
        st.write("📋 **รายละเอียดแต่ละ Window**")
        df_details = pd.DataFrame(window_details)
        df_display = df_details[['window_number', 'timeline', 'best_seed', 'max_net', 
                               'price_change_pct', 'action_count', 'window_size']].copy()
        df_display.columns = ['Window', 'Timeline', 'Best Seed', 'Net Profit', 
                            'Price Change %', 'Actions', 'Window Size']
        st.dataframe(df_display, use_container_width=True)
        st.session_state[f'window_details_{Ticker}'] = window_details
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

def plot_comparison(Ticker='', act=-1, window_size=30, num_seeds_to_try=1000, max_workers=4, start_date=None, end_date=None):
    all = []
    all_id = []
    all.append(Limit_fx(Ticker, act=-1, start_date=start_date, end_date=end_date).net)
    all_id.append('min')
    if act == -3:
        all.append(Limit_fx(Ticker, act=act, window_size=window_size, 
                           num_seeds_to_try=num_seeds_to_try, max_workers=max_workers,
                           start_date=start_date, end_date=end_date).net)
        all_id.append('best_seed')
    else:
        all.append(Limit_fx(Ticker, act=act, start_date=start_date, end_date=end_date).net)
        all_id.append('fx_{}'.format(act))
    all.append(Limit_fx(Ticker, act=-2, start_date=start_date, end_date=end_date).net)
    all_id.append('max')
    chart_data = pd.DataFrame(np.array(all).T, columns=np.array(all_id))
    st.write('📊 **Refer_Log Comparison**')
    st.line_chart(chart_data)
    df_plot = Limit_fx(Ticker, act=-1, start_date=start_date, end_date=end_date)
    if not df_plot.empty:
        df_plot = df_plot[['buffer']].cumsum()
        st.write('💰 **Burn_Cash**')
        st.line_chart(df_plot)
        st.write(Limit_fx(Ticker, act=-1, start_date=start_date, end_date=end_date))

if 'test_ticker' not in st.session_state:
    st.session_state.test_ticker = 'FFWM'
if 'start_date' not in st.session_state:
    st.session_state.start_date = datetime(2023, 1, 1)
if 'end_date' not in st.session_state:
    st.session_state.end_date = datetime.now()
if 'window_size' not in st.session_state:
    st.session_state.window_size = 30
if 'num_seeds' not in st.session_state:
    st.session_state.num_seeds = 1000
if 'max_workers' not in st.session_state:
    st.session_state.max_workers = 4

st.write("🎯 Best Seed Sliding Window Tester (Optimized)")
st.write("เครื่องมือทดสอบการหา Best Seed ด้วยวิธี Sliding Window สำหรับการเทรด (ปรับปรุงความเร็ว)")

tab1, tab2 = st.tabs(["การตั้งค่า", "ทดสอบ"])

with tab1:
    st.write("⚙️ การตั้งค่า")
    st.session_state.test_ticker = st.selectbox(
        "เลือก Ticker สำหรับทดสอบ", 
        ['FFWM', 'NEGG', 'RIVN', 'APLS', 'NVTS', 'QXO', 'RXRX'],
        index=['FFWM', 'NEGG', 'RIVN', 'APLS', 'NVTS', 'QXO', 'RXRX'].index(st.session_state.test_ticker)
    )
    st.write("📅 **ช่วงวันที่สำหรับการวิเคราะห์**")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.start_date = st.date_input(
            "วันที่เริ่มต้น",
            value=st.session_state.start_date,
            min_value=datetime(2024, 1, 1),
            max_value=datetime.now()
        )
    with col2:
        st.session_state.end_date = st.date_input(
            "วันที่สิ้นสุด",
            value=st.session_state.end_date,
            min_value=datetime(2024, 1, 1),
            max_value=datetime.now()
        )
    if st.session_state.start_date >= st.session_state.end_date:
        st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
    else:
        date_diff = (st.session_state.end_date - st.session_state.start_date).days
        st.info(f"📊 ช่วงวันที่ที่เลือก: {date_diff} วัน ({st.session_state.start_date.strftime('%Y-%m-%d')} ถึง {st.session_state.end_date.strftime('%Y-%m-%d')})")
    st.session_state.window_size = st.number_input(
        "ขนาด Window (วัน)", 
        min_value=2, max_value=730, value=st.session_state.window_size
    )
    st.session_state.num_seeds = st.number_input(
        "จำนวน Seeds ต่อ Window", 
        min_value=100, max_value=10000, value=st.session_state.num_seeds
    )
    st.session_state.max_workers = st.number_input(
        "จำนวน Workers สำหรับ Parallel Processing", 
        min_value=1, max_value=16, value=st.session_state.max_workers,
        help="เพิ่มจำนวน workers เพื่อความเร็วมากขึ้น (แนะนำ 4-8)"
    )

with tab2:
    st.write("---")
    if st.button("🚀 เริ่มทดสอบ Best Seed (Optimized)", type="primary"):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้องในแท็บ 'การตั้งค่า'")
        else:
            st.write(f"กำลังทดสอบ Best Seed สำหรับ **{st.session_state.test_ticker}** 📊")
            st.write(f"📅 ช่วงวันที่: {st.session_state.start_date.strftime('%Y-%m-%d')} ถึง {st.session_state.end_date.strftime('%Y-%m-%d')}")
            st.write(f"⚙️ พารามิเตอร์: Window Size = {st.session_state.window_size}, Seeds per Window = {st.session_state.num_seeds}, Workers = {st.session_state.max_workers}")
            st.write("---")
            try:
                start_date_str = st.session_state.start_date.strftime('%Y-%m-%d')
                end_date_str = st.session_state.end_date.strftime('%Y-%m-%d')
                plot_comparison(
                    Ticker=st.session_state.test_ticker, act=-3, 
                    window_size=st.session_state.window_size, 
                    num_seeds_to_try=st.session_state.num_seeds, 
                    max_workers=st.session_state.max_workers,
                    start_date=start_date_str, end_date=end_date_str
                )
                if f'window_details_{st.session_state.test_ticker}' in st.session_state:
                    st.write("---")
                    st.write("🔍 **การวิเคราะห์เพิ่มเติม**")
                    window_details = st.session_state[f'window_details_{st.session_state.test_ticker}']
                    df_windows = pd.DataFrame(window_details)
                    st.write("📊 **Net Profit แต่ละ Window**")
                    st.bar_chart(df_windows.set_index('window_number')['max_net'])
                    st.write("📈 **Price Change % แต่ละ Window**")
                    st.bar_chart(df_windows.set_index('window_number')['price_change_pct'])
                    st.write("🌱 **Seeds ที่เลือกใช้ในแต่ละ Window**")
                    seeds_df = df_windows[['window_number', 'timeline', 'best_seed', 'max_net']].copy()
                    seeds_df.columns = ['Window', 'Timeline', 'Selected Seed', 'Net Profit']
                    st.dataframe(seeds_df, use_container_width=True)
                    st.write("💾 **ดาวน์โหลดผลลัพธ์**")
                    csv = df_windows.to_csv(index=False)
                    st.download_button(
                        label="📥 ดาวน์โหลด Window Details (CSV)",
                        data=csv,
                        file_name=f'best_seed_results_{st.session_state.test_ticker}_{st.session_state.window_size}w_{st.session_state.num_seeds}s.csv',
                        mime='text/csv'
                    )
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
                st.write("กรุณาลองปรับพารามิเตอร์หรือเลือก ticker อื่น")
    st.write("---")
    st.write("📖 คำอธิบายวิธีการทำงาน")
    with st.expander("🔍 Best Seed Sliding Window คืออะไร?"):
        st.write("""
        **Best Seed Sliding Window** เป็นเทคนิคการหา action sequence ที่ดีที่สุดโดย:
        1. **แบ่งข้อมูล**: แบ่งข้อมูลราคาออกเป็นช่วง ๆ (windows) ตามขนาดที่กำหนด
        2. **ค้นหา Seed**: ในแต่ละ window ทำการสุ่ม seed หลาย ๆ ตัวและคำนวณผลกำไร
        3. **เลือก Best Seed**: เลือก seed ที่ให้ผลกำไรสูงสุดในแต่ละ window
        4. **รวม Actions**: นำ action sequences จากแต่ละ window มาต่อกันเป็น sequence สุดท้าย

        **ข้อดี:**
        - ปรับตัวได้ตามสภาวะตลาดในแต่ละช่วงเวลา
        - ลดความเสี่ยงจากการใช้ seed เดียวตลอดทั้งช่วง
        - สามารถควบคุมพารามิเตอร์ได้ (window size, จำนวน seeds)
        """)
    with st.expander("⚙️ การตั้งค่าพารามิเตอร์"):
        st.write("""
        **Window Size (ขนาด Window):**
        - ขนาดเล็ก (10-20 วัน): ปรับตัวเร็ว แต่อาจมีความผันผวนสูง
        - ขนาดกลาง (20-50 วัน): สมดุลระหว่างการปรับตัวและเสถียรภาพ
        - ขนาดใหญ่ (50+ วัน): เสถียรแต่ปรับตัวช้า

        **จำนวน Seeds ต่อ Window:**
        - น้อย (100-500): เร็วแต่อาจไม่ได้ seed ที่ดีที่สุด
        - กลาง (500-2000): สมดุลระหว่างเวลาและคุณภาพ
        - มาก (2000+): ได้ผลลัพธ์ดีแต่ใช้เวลานาน

        **จำนวน Workers:**
        - 1-2: สำหรับเครื่องที่มี CPU น้อย
        - 4: เหมาะสำหรับเครื่องทั่วไป
        - 6-8: สำหรับเครื่องที่มี CPU หลายคอร์
        """)
    with st.expander("⚡ การปรับปรุงความเร็ว"):
        st.write("""
        **การปรับปรุงที่ทำ:**
        1. **Parallel Processing**: ใช้ ThreadPoolExecutor เพื่อประเมิน seeds หลายตัวพร้อมกัน
        2. **Caching**: ใช้ @lru_cache สำหรับฟังก์ชันคำนวณและ @st.cache_data สำหรับข้อมูล ticker
        3. **Vectorization**: ใช้ NumPy operations แทน Python loops ในส่วนที่เป็นไปได้
        4. **Batch Processing**: แบ่ง seeds เป็น batches เพื่อลด overhead ของ threading
        5. **Memory Optimization**: ใช้ dtype ที่เหมาะสมและ pre-allocate arrays

        **ผลลัพธ์ที่คาดหวัง:**
        - เร็วขึ้น 2-4 เท่า (ขึ้นอยู่กับจำนวน CPU cores)
        - ใช้ memory มีประสิทธิภาพมากขึ้น
        - ผลลัพธ์เหมือนเดิมทุกประการ
        """)

 
