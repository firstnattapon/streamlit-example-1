# เริ่มต้นโค้ด
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from datetime import datetime, timedelta
import ast # <-- Import ที่จำเป็นสำหรับ Tab Analytics

# การตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Best Seed Sliding Window", page_icon="🎯", layout="wide")

# ==============================================================================
# ===== ส่วนของฟังก์ชันคำนวณหลัก (Core Calculation Functions) =====
# ==============================================================================

@lru_cache(maxsize=1000)
def calculate_optimized_cached(action_tuple, price_tuple, fix=1500):
    """ฟังก์ชันคำนวณหลักที่ถูกแคชเพื่อประสิทธิภาพสูงสุด"""
    action_array = np.asarray(action_tuple, dtype=np.int32)
    price_array = np.asarray(price_tuple, dtype=np.float64)
    n = len(price_array)
    if n == 0:
        return (np.array([]),) * 6 # คืนค่า tuple ของ array ว่างถ้าไม่มีข้อมูล

    # สร้าง copy เพื่อไม่ให้กระทบ action_tuple เดิม
    action_array_calc = action_array.copy()
    action_array_calc[0] = 1 # action แรกบังคับเป็น 1 เสมอ

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
        if action_array_calc[i] == 0:
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
    """Wrapper function เพื่อให้แน่ใจว่า input เป็น tuple ก่อนเรียก cached function"""
    return calculate_optimized_cached(tuple(action_list), tuple(price_list), fix)

def evaluate_seed_batch(seed_batch, prices_window, window_len):
    """ประเมินผลกำไรสำหรับกลุ่มของ seeds"""
    results = []
    for seed in seed_batch:
        try:
            rng = np.random.default_rng(seed)
            actions_window = rng.integers(0, 2, size=window_len)
            
            if window_len < 2:
                final_net = 0.0
            else:
                _, sumusd, _, _, _, refer = calculate_optimized(actions_window.tolist(), prices_window.tolist())
                initial_capital = sumusd[0]
                net = sumusd - refer - initial_capital
                final_net = net[-1]
            results.append((seed, final_net))
        except Exception:
            results.append((seed, -np.inf)) # กรณีเกิดข้อผิดพลาด ให้ค่าติดลบไปเลย
    return results

def find_best_seed_sliding_window_optimized(price_list, ticker_data_with_dates, window_size, num_seeds_to_try, progress_bar, max_workers):
    """ฟังก์ชันหลักสำหรับค้นหา Best Seed ด้วยวิธี Sliding Window (ฉบับปรับปรุง)"""
    prices = np.asarray(price_list)
    n = len(prices)
    final_actions = np.array([], dtype=int)
    window_details = []
    num_windows = (n + window_size - 1) // window_size

    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        window_len = len(prices_window)
        if window_len == 0: continue

        start_date = ticker_data_with_dates.index[start_index].strftime('%Y-%m-%d')
        end_date = ticker_data_with_dates.index[end_index-1].strftime('%Y-%m-%d')
        timeline_info = f"{start_date} ถึง {end_date}"

        best_seed_for_window = -1
        max_net_for_window = -np.inf
        
        # Parallel processing
        random_seeds = np.arange(num_seeds_to_try)
        batch_size = max(1, num_seeds_to_try // (max_workers * 4)) # แบ่งงานให้ละเอียดขึ้น
        seed_batches = [random_seeds[j:j+batch_size] for j in range(0, len(random_seeds), batch_size)]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {executor.submit(evaluate_seed_batch, batch, prices_window, window_len): batch for batch in seed_batches}
            all_results = []
            for future in as_completed(future_to_batch):
                all_results.extend(future.result())

        for seed, final_net in all_results:
            if final_net > max_net_for_window:
                max_net_for_window = final_net
                best_seed_for_window = seed

        # สร้าง action sequence ที่ดีที่สุดสำหรับ window นี้
        if best_seed_for_window >= 0:
            rng_best = np.random.default_rng(best_seed_for_window)
            best_actions_for_window = rng_best.integers(0, 2, size=window_len)
        else: # กรณีไม่เจอ seed ที่ดีกว่า 0 เลย (เป็นไปได้ยาก)
            best_actions_for_window = np.ones(window_len, dtype=int)
            max_net_for_window = 0.0 # กำหนดเป็น 0

        best_actions_for_window[0] = 1 # บังคับ action แรก

        window_detail = {
            'window_number': i + 1, 'timeline': timeline_info,
            'best_seed': best_seed_for_window, 'max_net': round(max_net_for_window, 2),
            'start_price': round(prices_window[0], 2), 'end_price': round(prices_window[-1], 2),
            'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2),
            'action_count': int(np.sum(best_actions_for_window)), 'window_size': window_len,
            'action_sequence': str(best_actions_for_window.tolist()) # <-- บันทึกเป็น string
        }
        window_details.append(window_detail)
        final_actions = np.concatenate((final_actions, best_actions_for_window))
        
        if progress_bar:
            progress_bar.progress((i + 1) / num_windows, text=f"Processing Window {i + 1}/{num_windows}")

    return final_actions, window_details

def get_max_action(price_list, fix=1500):
    """คำนวณ Action ที่ให้ผลตอบแทนสูงสุด (Perfect Foresight) ด้วย Dynamic Programming"""
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
    current_day = np.argmax(dp)
    while current_day > 0:
        actions[current_day] = 1
        current_day = path[current_day]
    actions[0] = 1
    return actions.tolist()

@st.cache_data(ttl=3600)
def get_ticker_data(ticker, start_date=None, end_date=None):
    """ดึงข้อมูลราคาหุ้นจาก yfinance พร้อมการจัดการ Timezone และ Caching"""
    try:
        data = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=True)[['Close']]
        if data.empty: return pd.DataFrame()
        
        # จัดการ Timezone ให้เป็นมาตรฐาน
        if data.index.tz is None:
            data = data.tz_localize('UTC').tz_convert('Asia/Bangkok')
        else:
            data = data.tz_convert('Asia/Bangkok')
        return data
    except Exception as e:
        st.error(f"❌ ไม่สามารถดึงข้อมูล {ticker} ได้: {str(e)}")
        return pd.DataFrame()

def run_strategy_and_get_results(ticker_data, strategy_type, window_size=30, num_seeds_to_try=1000, max_workers=4):
    """ฟังก์ชันกลางสำหรับรันกลยุทธ์และคืนผลลัพธ์ในรูปแบบ DataFrame"""
    if ticker_data.empty:
        return pd.DataFrame(), None
        
    prices = ticker_data['Close'].tolist()
    actions = []
    details = None
    
    if strategy_type == "REBALANCE_DAILY":
        actions = np.ones(len(prices), dtype=int).tolist()
    elif strategy_type == "PERFECT_FORESIGHT":
        actions = get_max_action(prices)
    elif strategy_type == "SLIDING_WINDOW":
        progress_bar = st.progress(0)
        st.info(f"🔍 เริ่มต้นการค้นหา Best Seed... | Ticker: {st.session_state.test_ticker} | Window: {window_size} วัน | Seeds: {num_seeds_to_try}")
        actions, details_list = find_best_seed_sliding_window_optimized(
            prices, ticker_data, window_size=window_size,
            num_seeds_to_try=num_seeds_to_try, progress_bar=progress_bar,
            max_workers=max_workers
        )
        progress_bar.empty()
        details = pd.DataFrame(details_list)
        
    buffer, sumusd, cash, asset_value, amount, refer = calculate_optimized(actions, prices)
    initial_capital = sumusd[0] if len(sumusd) > 0 else 0
    
    df = pd.DataFrame({
        'price': prices, 'action': actions, 'buffer': np.round(buffer, 2),
        'sumusd': np.round(sumusd, 2), 'cash': np.round(cash, 2),
        'asset_value': np.round(asset_value, 2), 'amount': np.round(amount, 2),
        'refer': np.round(refer + initial_capital, 2),
        'net': np.round(sumusd - refer - initial_capital, 2)
    }, index=ticker_data.index)
    
    return df, details

# ==============================================================================
# ===== ส่วนของ UI (User Interface) =====
# ==============================================================================

# --- กำหนดค่าเริ่มต้นใน Session State ---
if 'test_ticker' not in st.session_state: st.session_state.test_ticker = 'FFWM'
if 'start_date' not in st.session_state: st.session_state.start_date = datetime(2023, 1, 1)
if 'end_date' not in st.session_state: st.session_state.end_date = datetime.now()
if 'window_size' not in st.session_state: st.session_state.window_size = 30
if 'num_seeds' not in st.session_state: st.session_state.num_seeds = 10000
if 'max_workers' not in st.session_state: st.session_state.max_workers = 8
if 'df_for_analysis' not in st.session_state: st.session_state.df_for_analysis = None


st.title("🎯 Best Seed Sliding Window Tester (v2 Integrated)")
st.caption("เครื่องมือทดสอบและวิเคราะห์กลยุทธ์ Best Seed ด้วยวิธี Sliding Window")

tab1, tab2, tab3 = st.tabs(["⚙️ การตั้งค่า", "🚀 ทดสอบและเปรียบเทียบ", "📊 Advanced Analytics Dashboard"])

with tab1:
    st.header("การตั้งค่าพารามิเตอร์")
    st.session_state.test_ticker = st.selectbox(
        "เลือก Ticker สำหรับทดสอบ",
        ['FFWM', 'NEGG', 'RIVN', 'APLS', 'NVTS', 'QXO', 'RXRX', 'AGL'],
        index=['FFWM', 'NEGG', 'RIVN', 'APLS', 'NVTS', 'QXO', 'RXRX', 'AGL'].index(st.session_state.test_ticker)
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.start_date = st.date_input("วันที่เริ่มต้น", value=st.session_state.start_date)
    with col2:
        st.session_state.end_date = st.date_input("วันที่สิ้นสุด", value=st.session_state.end_date)
    
    if st.session_state.start_date >= st.session_state.end_date:
        st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
    
    st.session_state.window_size = st.number_input("ขนาด Window (วัน)", min_value=2, max_value=365, value=st.session_state.window_size)
    st.session_state.num_seeds = st.number_input("จำนวน Seeds ต่อ Window", min_value=100, max_value=100000, value=st.session_state.num_seeds, step=100)
    st.session_state.max_workers = st.slider("จำนวน Workers (สำหรับ Parallel Processing)", min_value=1, max_value=16, value=st.session_state.max_workers)

with tab2:
    st.header("ทดสอบและเปรียบเทียบผลลัพธ์")
    if st.button("🚀 เริ่มการทดสอบ", type="primary"):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้องในแท็บ 'การตั้งค่า'")
        else:
            with st.spinner("กำลังดึงข้อมูลและประมวลผล..."):
                start_date_str = st.session_state.start_date.strftime('%Y-%m-%d')
                end_date_str = st.session_state.end_date.strftime('%Y-%m-%d')
                
                ticker_data = get_ticker_data(st.session_state.test_ticker, start_date_str, end_date_str)

                if not ticker_data.empty:
                    df_strategy, df_details = run_strategy_and_get_results(
                        ticker_data, "SLIDING_WINDOW",
                        window_size=st.session_state.window_size,
                        num_seeds_to_try=st.session_state.num_seeds,
                        max_workers=st.session_state.max_workers
                    )
                    df_min, _ = run_strategy_and_get_results(ticker_data, "REBALANCE_DAILY")
                    df_max, _ = run_strategy_and_get_results(ticker_data, "PERFECT_FORESIGHT")
                    
                    st.success("การทดสอบเสร็จสมบูรณ์!")
                    
                    st.subheader("📈 เปรียบเทียบผลกำไรสุทธิ (Net Profit)")
                    chart_data = pd.DataFrame({
                        'Max Performance (Perfect)': df_max['net'],
                        'Best Seed Sliding Window': df_strategy['net'],
                        'Min Performance (Rebalance Daily)': df_min['net']
                    })
                    st.line_chart(chart_data)

                    if df_details is not None:
                        st.subheader("📋 สรุปผลลัพธ์ราย Window")
                        st.dataframe(df_details[['window_number', 'timeline', 'best_seed', 'max_net', 'price_change_pct', 'action_count', 'window_size']], use_container_width=True)
                        csv = df_details.to_csv(index=False)
                        st.download_button(
                            label="📥 ดาวน์โหลด Window Details (CSV)", data=csv,
                            file_name=f'best_seed_results_{st.session_state.test_ticker}.csv', mime='text/csv'
                        )
                else:
                    st.warning("ไม่พบข้อมูลในช่วงวันที่ที่เลือก")

with tab3:
    st.header("วิเคราะห์ผลลัพธ์ Backtest ในเชิงลึก")
    
    with st.container(border=True):
        st.subheader("1. นำเข้าข้อมูลผลลัพธ์ (ไฟล์ CSV จากการทดสอบ)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### อัปโหลดไฟล์จากเครื่อง")
            uploaded_file = st.file_uploader("เลือกไฟล์ CSV", type=['csv'], key="local_uploader")
            if uploaded_file is not None:
                try:
                    st.session_state.df_for_analysis = pd.read_csv(uploaded_file)
                    st.success("✅ โหลดไฟล์สำเร็จ!")
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")
                    st.session_state.df_for_analysis = None
        
        with col2:
            st.markdown("##### หรือ โหลดจาก GitHub URL")
            default_github_url = f"https://raw.githubusercontent.com/firstnattapon/streamlit-example-1/refs/heads/master/Seed_Sliding_Window/{st.session_state.test_ticker}.csv"
            github_url = st.text_input("ป้อน GitHub Raw URL:", value=default_github_url)
            if st.button("📥 โหลดจาก GitHub"):
                if github_url:
                    try:
                        with st.spinner("กำลังดาวน์โหลด..."):
                            st.session_state.df_for_analysis = pd.read_csv(github_url)
                        st.success("✅ โหลดข้อมูลจาก GitHub สำเร็จ!")
                    except Exception as e:
                        st.error(f"❌ ไม่สามารถโหลดข้อมูลจาก URL ได้: {e}")
                        st.session_state.df_for_analysis = None

    st.divider()

    if st.session_state.df_for_analysis is not None:
        st.header("2. ผลการวิเคราะห์")
        df_to_analyze = st.session_state.df_for_analysis.copy()

        try:
            # --- ตรวจสอบความสมบูรณ์ของไฟล์ ---
            required_cols = ['window_number', 'timeline', 'max_net', 'best_seed', 'price_change_pct', 'action_sequence', 'window_size']
            if not all(col in df_to_analyze.columns for col in required_cols):
                st.error(f"ไฟล์ CSV ไม่สมบูรณ์! กรุณาตรวจสอบว่ามีคอลัมน์เหล่านี้: {', '.join(required_cols)}")
            else:
                # --- เริ่มการวิเคราะห์ ---
                analysis_tabs = st.tabs(["🔬 ภาพรวมและสำรวจราย Window", "🧬 Stitched DNA Analysis"])

                with analysis_tabs[0]:
                    st.subheader("ภาพรวมประสิทธิภาพ (Overall Performance)")
                    df_to_analyze['result'] = np.where(df_to_analyze['max_net'] > 0, 'Win', 'Loss')
                    gross_profit = df_to_analyze[df_to_analyze['max_net'] > 0]['max_net'].sum()
                    gross_loss = abs(df_to_analyze[df_to_analyze['max_net'] < 0]['max_net'].sum())
                    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                    win_rate = (df_to_analyze['result'] == 'Win').mean() * 100
                    
                    kpi_cols = st.columns(4)
                    kpi_cols[0].metric("Total Net Profit", f"${df_to_analyze['max_net'].sum():,.2f}")
                    kpi_cols[1].metric("Win Rate", f"{win_rate:.2f}%")
                    kpi_cols[2].metric("Profit Factor", f"{profit_factor:.2f}")
                    kpi_cols[3].metric("Total Windows", f"{df_to_analyze.shape[0]}")
                    
                    st.subheader("สำรวจข้อมูลราย Window")
                    selected_window = st.selectbox(
                        'เลือก Window ที่ต้องการดูรายละเอียด:', 
                        options=df_to_analyze['window_number'], 
                        format_func=lambda x: f"Window #{x} | {df_to_analyze.loc[df_to_analyze['window_number'] == x, 'timeline'].iloc[0]}"
                    )
                    if selected_window:
                        window_data = df_to_analyze[df_to_analyze['window_number'] == selected_window].iloc[0]
                        st.markdown(f"**รายละเอียดของ Window #{selected_window}**")
                        w_cols = st.columns(3)
                        w_cols[0].metric("Net Profit", f"${window_data['max_net']:.2f}")
                        w_cols[1].metric("Best Seed", f"{window_data['best_seed']}")
                        w_cols[2].metric("Price Change", f"{window_data['price_change_pct']:.2f}%")
                        st.markdown(f"**Action Sequence:**")
                        st.code(window_data['action_sequence'], language='json')

                # ฟังก์ชัน Helper สำหรับแปลง string list เป็น list จริงๆ
                def safe_literal_eval(val):
                    if pd.isna(val): return []
                    try: return ast.literal_eval(val)
                    except (ValueError, SyntaxError): return []

                with analysis_tabs[1]:
                    st.subheader("ทดสอบกลยุทธ์จาก 'Stitched' DNA")
                    st.markdown("จำลองการเทรดโดยนำ **`action_sequence`** จากทุก Window ในไฟล์ CSV มา 'เย็บ' ต่อกันเป็นกลยุทธ์เดียว แล้วนำไปทดสอบกับข้อมูลจริงชุดใหม่เพื่อดูประสิทธิภาพ")

                    # --- สร้าง Stitched Actions ---
                    df_to_analyze['action_sequence_list'] = df_to_analyze['action_sequence'].apply(safe_literal_eval)
                    stitched_actions = [action for seq in df_to_analyze.sort_values('window_number')['action_sequence_list'] for action in seq]
                    
                    st.info(f"🧬 สร้าง 'Stitched DNA' สำเร็จ! มีทั้งหมด {len(stitched_actions)} actions.")

                    # --- ตั้งค่าการจำลอง ---
                    dna_cols = st.columns(2)
                    stitch_ticker = dna_cols[0].text_input("Ticker สำหรับจำลอง", value=st.session_state.test_ticker)
                    stitch_start_date = dna_cols[1].date_input("วันที่เริ่มต้นจำลอง", value=datetime(2024, 1, 1))

                    if st.button("🧬 เริ่มการวิเคราะห์ Stitched DNA", type="primary"):
                        if not stitched_actions:
                            st.error("ไม่สามารถสร้าง Action Sequence จากข้อมูลที่โหลดได้")
                        else:
                            with st.spinner(f"กำลังจำลองกลยุทธ์สำหรับ {stitch_ticker}..."):
                                sim_data = get_ticker_data(stitch_ticker, str(stitch_start_date), str(datetime.now()))
                                if sim_data.empty:
                                    st.error("ไม่สามารถดึงข้อมูลสำหรับจำลองได้")
                                else:
                                    prices = sim_data['Close'].tolist()
                                    n_total_prices = len(prices)
                                    
                                    # ตัด actions ให้มีความยาวไม่เกินจำนวนวันที่มีราคา
                                    final_actions_dna = stitched_actions[:n_total_prices]
                                    
                                    # จำลองกลยุทธ์ทั้ง 3 แบบบนข้อมูลชุดเดียวกัน
                                    df_dna, _ = run_strategy_and_get_results(sim_data.iloc[:len(final_actions_dna)], "CUSTOM", actions=final_actions_dna)
                                    df_max, _ = run_strategy_and_get_results(sim_data, "PERFECT_FORESIGHT")
                                    df_min, _ = run_strategy_and_get_results(sim_data, "REBALANCE_DAILY")
                                    
                                    # สร้าง DataFrame สำหรับพล็อตกราฟ
                                    plot_len = len(df_dna)
                                    plot_df = pd.DataFrame({
                                        'Max Performance (Perfect)': df_max['net'].iloc[:plot_len],
                                        'Stitched DNA Strategy': df_dna['net'],
                                        'Min Performance (Rebalance Daily)': df_min['net'].iloc[:plot_len]
                                    })
                                    
                                    st.subheader("Performance Comparison (Net Profit)")
                                    st.line_chart(plot_df)
                                    
                                    st.subheader("สรุปผลลัพธ์สุดท้าย (Final Net Profit)")
                                    final_net_dna = df_dna['net'].iloc[-1]
                                    final_net_min = df_min['net'].iloc[plot_len-1]
                                    
                                    metric_cols = st.columns(3)
                                    metric_cols[0].metric("Max Performance (at DNA End)", f"${df_max['net'].iloc[plot_len-1]:,.2f}")
                                    metric_cols[1].metric("Stitched DNA Strategy", f"${final_net_dna:,.2f}", 
                                                          delta=f"{final_net_dna - final_net_min:,.2f} vs Min", 
                                                          delta_color="normal")
                                    metric_cols[2].metric("Min Performance (at DNA End)", f"${final_net_min:,.2f}")

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการวิเคราะห์ข้อมูล: {e}")
            st.exception(e)
