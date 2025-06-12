# v2 (Corrected Full Code)
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import math
import json
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from datetime import datetime
from typing import List, Tuple, Dict, Any

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================

# --- การกำหนดประเภทของกลยุทธ์ให้ชัดเจนขึ้น แทนการใช้ Magic Number ---
class Strategy:
    REBALANCE_DAILY = "Rebalance Daily"
    PERFECT_FORESIGHT = "Perfect Foresight (Max)"
    SLIDING_WINDOW = "Best Seed Sliding Window"

def load_config(filepath: str = "dynamic_seed_config.json") -> Dict[str, Any]:
    """
    โหลดการตั้งค่าจากไฟล์ JSON พร้อม Fallback หากเกิดข้อผิดพลาด
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # หากไฟล์ไม่มีหรือผิดพลาด ให้ใช้ค่าเริ่มต้นสำรอง
        return {
            "assets": ["FFWM", "NEGG", "RIVN", "APLS", "NVTS", "QXO", "RXRX", "AGL"],
            "default_settings": {
                "selected_ticker": "FFWM", "start_date": "2025-04-28",
                "window_size": 30, "num_seeds": 30000, "max_workers": 8
            }
        }

def initialize_session_state(config: Dict[str, Any]):
    """
    ตั้งค่า Streamlit Session State จาก Configuration ที่โหลดมา
    """
    defaults = config.get('default_settings', {})

    if 'test_ticker' not in st.session_state:
        st.session_state.test_ticker = defaults.get('selected_ticker', 'FFWM')
    if 'start_date' not in st.session_state:
        try:
            st.session_state.start_date = datetime.strptime(defaults.get('start_date'), '%Y-%m-%d')
        except (ValueError, TypeError):
            st.session_state.start_date = datetime(2025, 4, 28)
    if 'end_date' not in st.session_state:
        st.session_state.end_date = datetime.now()
    if 'window_size' not in st.session_state:
        st.session_state.window_size = defaults.get('window_size', 30)
    if 'num_seeds' not in st.session_state:
        st.session_state.num_seeds = defaults.get('num_seeds', 30000)
    if 'max_workers' not in st.session_state:
        st.session_state.max_workers = defaults.get('max_workers', 8)
    if 'df_for_analysis' not in st.session_state:
        st.session_state.df_for_analysis = None

# ==============================================================================
# 2. Core Calculation & Data Functions
# ==============================================================================

@st.cache_data(ttl=3600)
def get_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    ดึงข้อมูลราคาปิดของ Ticker จาก yfinance และ cache ผลลัพธ์
    """
    try:
        data = yf.Ticker(ticker).history(start=start_date, end=end_date)[['Close']]
        if data.empty:
            return pd.DataFrame()
        if data.index.tz is None:
            data = data.tz_localize('UTC').tz_convert('Asia/Bangkok')
        else:
            data = data.tz_convert('Asia/Bangkok')
        return data
    except Exception as e:
        st.error(f"❌ ไม่สามารถดึงข้อมูล {ticker} ได้: {str(e)}")
        return pd.DataFrame()

@lru_cache(maxsize=2048)
def calculate_optimized_cached(action_tuple: Tuple[int, ...], price_tuple: Tuple[float, ...], fix: int = 1500) -> Tuple:
    """
    ฟังก์ชันคำนวณหลักที่ถูกแคชด้วย lru_cache
    """
    action_array = np.asarray(action_tuple, dtype=np.int32)
    price_array = np.asarray(price_tuple, dtype=np.float64)
    n = len(action_array)

    if n == 0:
        return (np.array([]),) * 6

    action_array_calc = action_array.copy()
    action_array_calc[0] = 1

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

def run_simulation(prices: List[float], actions: List[int], fix: int = 1500) -> pd.DataFrame:
    """
    จำลองการเทรดจากราคาและ Action ที่กำหนด และคืนค่าเป็น DataFrame
    """
    if not prices or not actions:
        return pd.DataFrame()

    buffer, sumusd, cash, asset_value, amount, refer = calculate_optimized_cached(tuple(actions), tuple(prices), fix)

    if len(sumusd) == 0:
        return pd.DataFrame()

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

# ==============================================================================
# 3. Strategy Action Generation
# ==============================================================================

def generate_actions_rebalance_daily(num_days: int) -> np.ndarray:
    """สร้าง Action สำหรับกลยุทธ์ Rebalance Daily (ซื้อทุกวัน)"""
    return np.ones(num_days, dtype=int)

def generate_actions_perfect_foresight(prices: List[float], fix: int = 1500) -> np.ndarray:
    """สร้าง Action สำหรับกลยุทธ์ Perfect Foresight (ผลตอบแทนสูงสุดที่เป็นไปได้)"""
    price_arr = np.asarray(prices, dtype=np.float64)
    n = len(price_arr)
    if n < 2:
        return np.ones(n, dtype=int)

    dp = np.zeros(n, dtype=np.float64)
    path = np.zeros(n, dtype=int)
    dp[0] = float(fix * 2)

    for i in range(1, n):
        j_indices = np.arange(i)
        profits = fix * ((price_arr[i] / price_arr[j_indices]) - 1)
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
    return actions

def find_best_seed_for_window(prices_window: np.ndarray, num_seeds_to_try: int, max_workers: int) -> Tuple[int, float, np.ndarray]:
    """ค้นหา Best Seed และ Actions สำหรับ Window ที่กำหนด"""
    window_len = len(prices_window)
    if window_len < 2:
        return 1, 0.0, np.ones(window_len, dtype=int)

    def evaluate_seed_batch(seed_batch: np.ndarray) -> List[Tuple[int, float]]:
        results = []
        for seed in seed_batch:
            rng = np.random.default_rng(seed)
            actions_window = rng.integers(0, 2, size=window_len)
            _, sumusd, _, _, _, refer = calculate_optimized_cached(tuple(actions_window), tuple(prices_window))
            net = sumusd - refer - sumusd[0]
            results.append((seed, net[-1]))
        return results

    best_seed_for_window = -1
    max_net_for_window = -np.inf

    random_seeds = np.arange(num_seeds_to_try)
    batch_size = max(1, num_seeds_to_try // (max_workers * 4))
    seed_batches = [random_seeds[j:j+batch_size] for j in range(0, len(random_seeds), batch_size)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_seed_batch, batch) for batch in seed_batches]
        for future in as_completed(futures):
            for seed, final_net in future.result():
                if final_net > max_net_for_window:
                    max_net_for_window = final_net
                    best_seed_for_window = seed

    if best_seed_for_window >= 0:
        rng_best = np.random.default_rng(best_seed_for_window)
        best_actions = rng_best.integers(0, 2, size=window_len)
    else:
        best_seed_for_window = 1
        best_actions = np.ones(window_len, dtype=int)
        max_net_for_window = 0.0

    best_actions[0] = 1
    return best_seed_for_window, max_net_for_window, best_actions

def generate_actions_sliding_window(
    ticker_data: pd.DataFrame, window_size: int, num_seeds_to_try: int, max_workers: int
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    สร้าง Action จากกลยุทธ์ Sliding Window และคืนค่า Actions พร้อมรายละเอียดของแต่ละ Window
    """
    prices = ticker_data['Close'].to_numpy()
    n = len(prices)
    final_actions = np.array([], dtype=int)
    window_details_list = []

    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="กำลังประมวลผล Sliding Windows...")

    st.write(f"📊 ข้อมูลทั้งหมด: {n} วัน | ขนาด Window: {window_size} วัน | จำนวน Windows: {num_windows}")
    st.write(f"⚡ ใช้ Parallel Processing: {max_workers} workers")
    st.write("---")

    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        window_len = len(prices_window)

        if window_len == 0: continue

        best_seed, max_net, best_actions = find_best_seed_for_window(
            prices_window, num_seeds_to_try, max_workers
        )
        final_actions = np.concatenate((final_actions, best_actions))

        start_date = ticker_data.index[start_index].strftime('%Y-%m-%d')
        end_date = ticker_data.index[end_index-1].strftime('%Y-%m-%d')
        detail = {
            'window_number': i + 1, 'timeline': f"{start_date} ถึง {end_date}",
            'best_seed': best_seed, 'max_net': round(max_net, 2),
            'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2),
            'action_count': int(np.sum(best_actions)), 'window_size': window_len,
            'action_sequence': best_actions.tolist()
        }
        window_details_list.append(detail)
        progress_bar.progress((i + 1) / num_windows, text=f"ประมวลผล Window {i+1}/{num_windows}")

    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)

# ==============================================================================
# 4. UI Rendering Functions
# ==============================================================================

def render_settings_tab(config: Dict[str, Any]):
    """แสดงผล UI สำหรับ Tab 'การตั้งค่า'"""
    st.write("⚙️ **การตั้งค่าพารามิเตอร์**")
    asset_list = config.get('assets', ['FFWM'])
    try:
        default_index = asset_list.index(st.session_state.test_ticker)
    except ValueError:
        default_index = 0

    st.session_state.test_ticker = st.selectbox(
        "เลือก Ticker สำหรับทดสอบ", options=asset_list, index=default_index
    )

    st.write("📅 **ช่วงวันที่สำหรับการวิเคราะห์**")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.start_date = st.date_input("วันที่เริ่มต้น", value=st.session_state.start_date)
    with col2:
        st.session_state.end_date = st.date_input("วันที่สิ้นสุด", value=st.session_state.end_date)

    if st.session_state.start_date >= st.session_state.end_date:
        st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
    else:
        st.info(f"ช่วงวันที่ที่เลือก: {st.session_state.start_date:%Y-%m-%d} ถึง {st.session_state.end_date:%Y-%m-%d}")

    st.session_state.window_size = st.number_input("ขนาด Window (วัน)", min_value=2, value=st.session_state.window_size)
    st.session_state.num_seeds = st.number_input("จำนวน Seeds ต่อ Window", min_value=100, value=st.session_state.num_seeds, format="%d")
    st.session_state.max_workers = st.number_input("จำนวน Workers", min_value=1, max_value=16, value=st.session_state.max_workers)


def display_comparison_charts(results: Dict[str, pd.DataFrame]):
    """แสดงกราฟเปรียบเทียบผลลัพธ์จากกลยุทธ์ต่างๆ"""
    chart_data_list = []
    # สร้าง DataFrame โดยใช้ index จากข้อมูลหลักเพื่อให้แน่ใจว่าทุก Series มี index เดียวกัน
    main_index = None
    for df in results.values():
        if not df.empty:
            main_index = df.index
            break
    
    if main_index is None:
        st.warning("ไม่มีข้อมูลสำหรับสร้างกราฟเปรียบเทียบ")
        return
        
    net_profits = {name: df['net'] for name, df in results.items() if not df.empty}
    chart_data = pd.DataFrame(net_profits)
    
    st.write('📊 **เปรียบเทียบกำไรสุทธิ (Net Profit)**')
    st.line_chart(chart_data)

    if Strategy.REBALANCE_DAILY in results and not results[Strategy.REBALANCE_DAILY].empty:
        df_min = results[Strategy.REBALANCE_DAILY]
        st.write(f'💰 **กระแสเงินสดสะสม (กลยุทธ์: {Strategy.REBALANCE_DAILY})**')
        st.line_chart(df_min['buffer'].cumsum())


def render_test_tab():
    """แสดงผล UI สำหรับ Tab 'ทดสอบ' และจัดการ Logic การรัน"""
    st.write("---")
    if st.button("🚀 เริ่มทดสอบ Best Seed (Optimized)", type="primary"):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้องในแท็บ 'การตั้งค่า'")
            return

        ticker = st.session_state.test_ticker
        start_date_str = st.session_state.start_date.strftime('%Y-%m-%d')
        end_date_str = st.session_state.end_date.strftime('%Y-%m-%d')

        st.info(f"กำลังดึงข้อมูลสำหรับ **{ticker}** | {start_date_str} ถึง {end_date_str}")
        ticker_data = get_ticker_data(ticker, start_date_str, end_date_str)

        if ticker_data.empty:
            st.error("ไม่พบข้อมูลสำหรับ Ticker และช่วงวันที่ที่เลือก")
            return

        prices = ticker_data['Close'].tolist()
        num_days = len(prices)

        with st.spinner("กำลังคำนวณกลยุทธ์ต่างๆ..."):
            st.write("🔍 **เริ่มต้นการค้นหา Best Seed ด้วย Sliding Window**")
            actions_sliding, df_windows = generate_actions_sliding_window(
                ticker_data, st.session_state.window_size,
                st.session_state.num_seeds, st.session_state.max_workers
            )

            actions_min = generate_actions_rebalance_daily(num_days)
            actions_max = generate_actions_perfect_foresight(prices)

            # run_simulation คืน DataFrame ที่ไม่มี index วันที่ เราต้องกำหนดให้มันทีหลัง
            results = {}
            for strategy_name, actions in {
                Strategy.SLIDING_WINDOW: actions_sliding.tolist(),
                Strategy.REBALANCE_DAILY: actions_min.tolist(),
                Strategy.PERFECT_FORESIGHT: actions_max.tolist()
            }.items():
                df = run_simulation(prices, actions)
                if not df.empty:
                    df.index = ticker_data.index
                results[strategy_name] = df

        st.success("การทดสอบเสร็จสมบูรณ์!")
        st.write("---")
        display_comparison_charts(results)

        st.write("📈 **สรุปผลการค้นหา Best Seed**")
        total_actions = df_windows['action_count'].sum()
        total_net = df_windows['max_net'].sum()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Windows", df_windows.shape[0])
        col2.metric("Total Actions", f"{total_actions}/{num_days}")
        col3.metric("Total Net (Sum)", f"${total_net:,.2f}")

        st.dataframe(df_windows[['window_number', 'timeline', 'best_seed', 'max_net', 'price_change_pct', 'action_count']], use_container_width=True)
        csv = df_windows.to_csv(index=False)
        st.download_button(
            label="📥 ดาวน์โหลด Window Details (CSV)", data=csv,
            file_name=f'best_seed_{ticker}_{st.session_state.window_size}w.csv', mime='text/csv'
        )

def render_analytics_tab():
    """แสดงผล UI สำหรับ Tab 'Advanced Analytics Dashboard'"""
    st.markdown("วิเคราะห์ผลลัพธ์ Backtest ในเชิงลึก")

    with st.container(border=True):
        st.markdown("เลือกวิธีการนำเข้าข้อมูล:")
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

            if st.button("📥 โหลดข้อมูลจาก GitHub"):
                if github_url:
                    try:
                        raw_url = github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
                        with st.spinner("กำลังดาวน์โหลดข้อมูล..."):
                            st.session_state.df_for_analysis = pd.read_csv(raw_url)
                        st.success("✅ โหลดข้อมูลจาก GitHub สำเร็จ!")
                    except Exception as e:
                        st.error(f"❌ ไม่สามารถโหลดข้อมูลจาก URL ได้: {e}")
                        st.session_state.df_for_analysis = None
                else:
                    st.warning("กรุณาป้อน URL ของไฟล์ CSV")

    st.divider()

    if st.session_state.df_for_analysis is not None:
        st.markdown("ผลการวิเคราะห์")
        df_to_analyze = st.session_state.df_for_analysis

        try:
            required_cols = ['window_number', 'timeline', 'max_net', 'best_seed', 'price_change_pct', 'action_sequence', 'window_size']
            if not all(col in df_to_analyze.columns for col in required_cols):
                st.error(f"ไฟล์ CSV ไม่สมบูรณ์! กรุณาตรวจสอบว่ามีคอลัมน์เหล่านี้ทั้งหมด: {', '.join(required_cols)}")
                return

            df = df_to_analyze.copy()
            df['result'] = np.where(df['max_net'] > 0, 'Win', 'Loss')

            overview_tab, stitched_dna_tab = st.tabs(["🔬 ภาพรวมและสำรวจราย Window", "🧬 Stitched DNA Analysis"])

            with overview_tab:
                # ... (ส่วนนี้ทำงานได้ดีอยู่แล้ว ไม่ต้องแก้ไข) ...
                st.markdown("ภาพรวมประสิทธิภาพ (Overall Performance)")
                gross_profit = df[df['max_net'] > 0]['max_net'].sum()
                gross_loss = abs(df[df['max_net'] <= 0]['max_net'].sum())
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                win_rate = (df['result'] == 'Win').mean() * 100
                
                kpi_cols = st.columns(4)
                kpi_cols[0].metric("Total Net Profit", f"${df['max_net'].sum():,.2f}")
                kpi_cols[1].metric("Win Rate", f"{win_rate:.2f}%")
                kpi_cols[2].metric("Profit Factor", f"{profit_factor:.2f}")
                kpi_cols[3].metric("Total Windows", f"{df.shape[0]}")

                st.subheader("สำรวจข้อมูลราย Window")
                format_func = lambda x: f"Window #{x} (Timeline: {df.loc[df['window_number'] == x, 'timeline'].iloc[0]})"
                selected_window = st.selectbox('เลือก Window:', options=df['window_number'], format_func=format_func)
                
                if selected_window:
                    window_data = df[df['window_number'] == selected_window].iloc[0]
                    st.markdown(f"**รายละเอียดของ Window #{selected_window}**")
                    w_cols = st.columns(3)
                    w_cols[0].metric("Net Profit", f"${window_data['max_net']:.2f}")
                    w_cols[1].metric("Best Seed", f"{window_data['best_seed']}")
                    w_cols[2].metric("Price Change", f"{window_data['price_change_pct']:.2f}%")
                    st.markdown("**Action Sequence:**")
                    st.code(window_data['action_sequence'], language='json')


            with stitched_dna_tab:
                st.subheader("ทดสอบกลยุทธ์จาก 'Stitched' DNA")
                st.markdown("จำลองการเทรดจริงโดยนำ `action_sequence` จากแต่ละ Window มา 'เย็บ' ต่อกัน และเปรียบเทียบกับ Benchmark")

                def safe_literal_eval(val):
                    if pd.isna(val) or val is None: return []
                    if isinstance(val, list): return val
                    if isinstance(val, str) and val.strip().startswith('[') and val.strip().endswith(']'):
                        try: return ast.literal_eval(val)
                        except (ValueError, SyntaxError): return []
                    return []

                df['action_sequence_list'] = df['action_sequence'].apply(safe_literal_eval)
                stitched_actions = [action for seq in df.sort_values('window_number')['action_sequence_list'] for action in seq]

                dna_cols = st.columns(2)
                stitch_ticker = dna_cols[0].text_input("Ticker สำหรับจำลอง", value=st.session_state.test_ticker, key='stitch_ticker_input')
                stitch_start_date = dna_cols[1].date_input("วันที่เริ่มต้นจำลอง", value=datetime(2024, 1, 1), key='stitch_date_input')

                if st.button("🧬 เริ่มการวิเคราะห์ Stitched DNA", type="primary"):
                    if not stitched_actions:
                        st.error("ไม่สามารถสร้าง Action Sequence จากข้อมูลที่โหลดได้")
                    else:
                        with st.spinner(f"กำลังจำลองกลยุทธ์สำหรับ {stitch_ticker}..."):
                            end_date = datetime.now()
                            sim_data = get_ticker_data(stitch_ticker, stitch_start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

                            if sim_data.empty:
                                st.error("ไม่สามารถดึงข้อมูลสำหรับจำลองได้")
                            else:
                                prices = sim_data['Close'].tolist()
                                n_total = len(prices)
                                final_actions_dna = stitched_actions[:n_total]

                                df_dna = run_simulation(prices[:len(final_actions_dna)], final_actions_dna)
                                df_max = run_simulation(prices, generate_actions_perfect_foresight(prices).tolist())
                                df_min = run_simulation(prices, generate_actions_rebalance_daily(n_total).tolist())

                                # !!!!!!!!!!!!!! THE FIX IS HERE !!!!!!!!!!!!!!
                                # We use .values to extract the raw NumPy array, ignoring the DataFrame's
                                # own RangeIndex. This prevents the index alignment problem when
                                # creating the new DataFrame with a DatetimeIndex.
                                plot_len = len(df_dna)
                                if plot_len > 0:
                                    plot_df = pd.DataFrame({
                                        'Max Performance (Perfect)': df_max['net'].values[:plot_len],
                                        'Stitched DNA Strategy': df_dna['net'].values,
                                        'Min Performance (Rebalance Daily)': df_min['net'].values[:plot_len]
                                    }, index=sim_data.index[:plot_len])

                                    st.subheader("Performance Comparison (Net Profit)")
                                    st.line_chart(plot_df)

                                    st.subheader("สรุปผลลัพธ์สุดท้าย (Final Net Profit)")
                                    final_net_dna = df_dna['net'].iloc[-1]
                                    final_net_min = df_min['net'].iloc[plot_len-1]
                                    metric_cols = st.columns(3)
                                    metric_cols[0].metric("Max Performance (at DNA End)", f"${df_max['net'].iloc[plot_len-1]:,.2f}")
                                    metric_cols[1].metric("Stitched DNA Strategy", f"${final_net_dna:,.2f}", delta=f"{final_net_dna - final_net_min:,.2f} vs Min", delta_color="normal")
                                    metric_cols[2].metric("Min Performance (at DNA End)", f"${final_net_min:,.2f}")
                                else:
                                    st.warning("ไม่มีข้อมูล DNA ที่จะแสดงผลได้")

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการวิเคราะห์ข้อมูล: {e}")
            st.exception(e)

# ==============================================================================
# 5. Main Application
# ==============================================================================
def main():
    st.set_page_config(page_title="Best Seed Sliding Window", page_icon="🎯", layout="wide")
    st.markdown("🎯 Best Seed Sliding Window Tester (Optimized)")
    st.caption("เครื่องมือทดสอบการหา Best Seed ด้วยวิธี Sliding Window (Refactored Version)")

    config = load_config()
    initialize_session_state(config)

    tab1, tab2, tab3 = st.tabs(["การตั้งค่า", "ทดสอบ", "📊 Advanced Analytics Dashboard"])

    with tab1:
        render_settings_tab(config)
    with tab2:
        render_test_tab()
    with tab3:
        render_analytics_tab()

    with st.expander("📖 คำอธิบายวิธีการทำงานและแนวคิดการ Refactor"):
        st.markdown("""
        **Best Seed Sliding Window** เป็นเทคนิคการหา action sequence ที่ดีที่สุดโดย:
        1. **แบ่งข้อมูล**: แบ่งข้อมูลราคาออกเป็นช่วง ๆ (windows) ตามขนาดที่กำหนด
        2. **ค้นหา Seed**: ในแต่ละ window ทำการสุ่ม seed หลาย ๆ ตัวและคำนวณผลกำไร
        3. **เลือก Best Seed**: เลือก seed ที่ให้ผลกำไรสูงสุดในแต่ละ window
        4. **รวม Actions**: นำ action sequences จากแต่ละ window มาต่อกันเป็น sequence สุดท้าย

        ---
        **หลักการ Refactoring ที่ใช้ในเวอร์ชันนี้:**
        - **Separation of Concerns**: แยกโค้ดส่วน UI, การคำนวณ, และการสร้างข้อมูลออกจากกันอย่างชัดเจน
        - **Single Responsibility**: แต่ละฟังก์ชันมีหน้าที่เดียว เช่น `get_ticker_data` ดึงข้อมูล, `run_simulation` จำลองการเทรด
        - **DRY (Don't Repeat Yourself)**: ลดการเขียนโค้ดซ้ำซ้อนโดยสร้างฟังก์ชันที่นำกลับมาใช้ใหม่ได้
        - **Readability**: ใช้ชื่อตัวแปรและฟังก์ชันที่สื่อความหมาย, เพิ่ม Type Hints และ Docstrings เพื่อให้โค้ดเข้าใจง่ายขึ้น
        """)

if __name__ == "__main__":
    main()
