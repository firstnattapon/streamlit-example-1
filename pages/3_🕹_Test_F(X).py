import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import json  # <--- เพิ่มที่นี่
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from datetime import datetime
from typing import List, Tuple, Dict, Any

# ==============================================================================
# 1. Configuration & Constants (เหมือนเดิม)
# ==============================================================================

class Strategy:
    REBALANCE_DAILY = "Rebalance Daily"
    PERFECT_FORESIGHT = "Perfect Foresight (Max)"
    SLIDING_WINDOW = "Best Seed Sliding Window"
    WALK_FORWARD = "Walk-Forward Optimized"
    MANUAL_SEED = "Manual Seed Strategy"

def load_config(filepath: str = "dynamic_seed_config.json") -> Dict[str, Any]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        st.warning(f"⚠️ ไม่พบหรือไฟล์ '{filepath}' ไม่ถูกต้อง จะใช้ค่าเริ่มต้นแทน")
        return {
            "assets": ["FFWM", "NEGG", "RIVN"],
            "default_settings": {"selected_ticker": "FFWM", "start_date": "2025-06-10", "window_size": 30, "num_seeds": 30000, "max_workers": 8},
            "manual_seed_by_asset": {
                "default": [{'seed': 999, 'size': 50, 'tail': 15}],
                "FFWM": [{'seed': 1234, 'size': 60, 'tail': 30}, {'seed': 7777, 'size': 30, 'tail': 10}]
            }
        }

def on_ticker_change_callback(config: Dict[str, Any]):
    selected_ticker = st.session_state.get("manual_ticker_key")
    if not selected_ticker: return
    presets_by_asset = config.get("manual_seed_by_asset", {})
    default_presets = presets_by_asset.get("default", [{'seed': 999, 'size': 50, 'tail': 15}])
    new_presets = presets_by_asset.get(selected_ticker, default_presets)
    st.session_state.manual_seed_lines = new_presets

def initialize_session_state(config: Dict[str, Any]):
    defaults = config.get('default_settings', {})
    if 'test_ticker' not in st.session_state:
        st.session_state.test_ticker = defaults.get('selected_ticker', 'FFWM')
    if 'start_date' not in st.session_state:
        try:
            st.session_state.start_date = datetime.strptime(defaults.get('start_date', '2025-01-01'), '%Y-%m-%d').date()
        except ValueError:
            st.session_state.start_date = datetime(2025, 1, 1).date()
    if 'end_date' not in st.session_state:
        st.session_state.end_date = datetime.now().date()
    if 'window_size' not in st.session_state:
        st.session_state.window_size = defaults.get('window_size', 30)
    if 'num_seeds' not in st.session_state:
        st.session_state.num_seeds = defaults.get('num_seeds', 30000)
    if 'max_workers' not in st.session_state:
        st.session_state.max_workers = defaults.get('max_workers', 8)
    if 'df_for_analysis' not in st.session_state:
        st.session_state.df_for_analysis = None
    if 'manual_seed_lines' not in st.session_state:
        initial_ticker = defaults.get('selected_ticker', 'FFWM')
        presets_by_asset = config.get("manual_seed_by_asset", {})
        default_presets = presets_by_asset.get("default", [{'seed': 999, 'size': 50, 'tail': 15}])
        st.session_state.manual_seed_lines = presets_by_asset.get(initial_ticker, default_presets)
    
    # Walk-Forward specific state
    if 'wf_train_period' not in st.session_state:
        st.session_state.wf_train_period = 120
    if 'wf_test_period' not in st.session_state:
        st.session_state.wf_test_period = 30


# ==============================================================================
# 2. Core Calculation & Data Functions (เหมือนเดิม)
# ==============================================================================
@st.cache_data(ttl=3600)
def get_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        data = yf.Ticker(ticker).history(start=start_date, end=end_date)[['Close']]
        if data.empty: return pd.DataFrame()
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
    action_array = np.asarray(action_tuple, dtype=np.int32)
    price_array = np.asarray(price_tuple, dtype=np.float64)
    n = len(action_array)
    if n == 0: return (np.array([]),) * 6
    action_array_calc = action_array.copy(); action_array_calc[0] = 1
    amount = np.empty(n, dtype=np.float64); buffer = np.zeros(n, dtype=np.float64)
    cash = np.empty(n, dtype=np.float64); asset_value = np.empty(n, dtype=np.float64)
    sumusd = np.empty(n, dtype=np.float64)
    initial_price = price_array[0]; amount[0] = fix / initial_price; cash[0] = fix
    asset_value[0] = amount[0] * initial_price; sumusd[0] = cash[0] + asset_value[0]
    refer = -fix * np.log(initial_price / price_array)
    for i in range(1, n):
        curr_price = price_array[i]
        if action_array_calc[i] == 0: amount[i] = amount[i-1]; buffer[i] = 0
        else: amount[i] = fix / curr_price; buffer[i] = amount[i-1] * curr_price - fix
        cash[i] = cash[i-1] + buffer[i]; asset_value[i] = amount[i] * curr_price
        sumusd[i] = cash[i] + asset_value[i]
    return buffer, sumusd, cash, asset_value, amount, refer

def run_simulation(prices: List[float], actions: List[int], fix: int = 1500) -> pd.DataFrame:
    if not prices or not actions: return pd.DataFrame()
    buffer, sumusd, cash, asset_value, amount, refer = calculate_optimized_cached(tuple(actions), tuple(prices), fix)
    if len(sumusd) == 0: return pd.DataFrame()
    initial_capital = sumusd[0]
    df = pd.DataFrame({
        'price': prices, 'action': actions, 'buffer': np.round(buffer, 2),
        'sumusd': np.round(sumusd, 2), 'cash': np.round(cash, 2),
        'asset_value': np.round(asset_value, 2), 'amount': np.round(amount, 2),
        'refer': np.round(refer + initial_capital, 2),
        'net': np.round(sumusd - refer - initial_capital, 2)
    })
    return df

# ==============================================================================
# 3. Strategy Action Generation (เหมือนเดิม แต่เพิ่ม Walk-Forward Logic)
# ==============================================================================
def generate_actions_rebalance_daily(num_days: int) -> np.ndarray:
    return np.ones(num_days, dtype=int)

def generate_actions_perfect_foresight(prices: List[float], fix: int = 1500) -> np.ndarray:
    price_arr = np.asarray(prices, dtype=np.float64); n = len(price_arr)
    if n < 2: return np.ones(n, dtype=int)
    dp = np.zeros(n, dtype=np.float64); path = np.zeros(n, dtype=int); dp[0] = float(fix * 2)
    for i in range(1, n):
        j_indices = np.arange(i); profits = fix * ((price_arr[i] / price_arr[j_indices]) - 1)
        current_sumusd = dp[j_indices] + profits
        best_idx = np.argmax(current_sumusd); dp[i] = current_sumusd[best_idx]; path[i] = j_indices[best_idx]
    actions = np.zeros(n, dtype=int); current_day = np.argmax(dp)
    while current_day > 0:
        actions[current_day] = 1; current_day = path[current_day]
    actions[0] = 1
    return actions

def find_best_seed_for_window(prices_window: np.ndarray, num_seeds_to_try: int, max_workers: int) -> Tuple[int, float, np.ndarray]:
    window_len = len(prices_window)
    if window_len < 2: return 1, 0.0, np.ones(window_len, dtype=int)
    def evaluate_seed_batch(seed_batch: np.ndarray) -> List[Tuple[int, float]]:
        results = []
        for seed in seed_batch:
            rng = np.random.default_rng(seed)
            actions_window = rng.integers(0, 2, size=window_len)
            _, sumusd, _, _, _, refer = calculate_optimized_cached(tuple(actions_window), tuple(prices_window))
            if len(sumusd) == 0: continue
            net = sumusd[-1] - refer[-1] - sumusd[0]
            results.append((seed, net))
        return results
    best_seed_for_window = -1; max_net_for_window = -np.inf
    random_seeds = np.arange(num_seeds_to_try)
    batch_size = max(1, num_seeds_to_try // (max_workers * 4))
    seed_batches = [random_seeds[j:j+batch_size] for j in range(0, len(random_seeds), batch_size)]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_seed_batch, batch) for batch in seed_batches]
        for future in as_completed(futures):
            for seed, final_net in future.result():
                if final_net > max_net_for_window:
                    max_net_for_window = final_net; best_seed_for_window = seed
    if best_seed_for_window >= 0:
        rng_best = np.random.default_rng(best_seed_for_window)
        best_actions = rng_best.integers(0, 2, size=window_len)
    else: best_seed_for_window = 1; best_actions = np.ones(window_len, dtype=int); max_net_for_window = 0.0
    best_actions[0] = 1
    return best_seed_for_window, max_net_for_window, best_actions

def generate_actions_sliding_window(ticker_data: pd.DataFrame, window_size: int, num_seeds_to_try: int, max_workers: int) -> Tuple[np.ndarray, pd.DataFrame]:
    prices = ticker_data['Close'].to_numpy(); n = len(prices)
    final_actions = np.array([], dtype=int); window_details_list = []
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="กำลังประมวลผล Sliding Windows...")
    st.write(f"📊 ข้อมูลทั้งหมด: {n} วัน | ขนาด Window: {window_size} วัน | จำนวน Windows: {num_windows}")
    st.write(f"⚡ ใช้ Parallel Processing: {max_workers} workers"); st.write("---")
    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n); prices_window = prices[start_index:end_index]; window_len = len(prices_window)
        if window_len == 0: continue
        best_seed, max_net, best_actions = find_best_seed_for_window(prices_window, num_seeds_to_try, max_workers)
        final_actions = np.concatenate((final_actions, best_actions))
        start_date_str = ticker_data.index[start_index].strftime('%Y-%m-%d'); end_date_str = ticker_data.index[end_index-1].strftime('%Y-%m-%d')
        detail = {'window_number': i + 1, 'timeline': f"{start_date_str} ถึง {end_date_str}", 'best_seed': best_seed, 'max_net': round(max_net, 2),
                  'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2), 'action_count': int(np.sum(best_actions)),
                  'window_size': window_len, 'action_sequence': best_actions.tolist()}
        window_details_list.append(detail)
        progress_bar.progress((i + 1) / num_windows, text=f"ประมวลผล Window {i+1}/{num_windows}")
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)

# ==============================================================================
# 4. ฟังก์ชันใหม่สำหรับ Walk-Forward Analysis
# ==============================================================================
def perform_walk_forward_analysis(ticker_data: pd.DataFrame, train_period: int, test_period: int, num_seeds_to_try: int, max_workers: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    ดำเนินการ Walk-Forward Optimization
    """
    prices = ticker_data['Close'].to_numpy()
    n = len(prices)
    
    all_oos_results = [] # เก็บผลลัพธ์ของแต่ละช่วง Out-of-Sample
    walk_forward_summary_list = []

    # คำนวณจำนวนรอบที่จะทำ
    num_walks = (n - train_period) // test_period
    if num_walks <= 0:
        st.error(f"ข้อมูลไม่เพียงพอสำหรับ Walk-Forward Analysis (ต้องการอย่างน้อย {train_period + test_period} วัน แต่มีเพียง {n} วัน)")
        return pd.DataFrame(), pd.DataFrame()

    progress_bar = st.progress(0, text="กำลังประมวลผล Walk-Forward Analysis...")

    for i in range(num_walks):
        # 1. กำหนดช่วง In-Sample (IS) และ Out-of-Sample (OOS)
        is_start = i * test_period
        is_end = is_start + train_period
        oos_start = is_end
        oos_end = oos_start + test_period

        if oos_end > n:
            oos_end = n # ปรับช่วงสุดท้ายถ้าข้อมูลไม่พอดี

        prices_is = prices[is_start:is_end]
        prices_oos = prices[oos_start:oos_end]

        if len(prices_is) == 0 or len(prices_oos) == 0:
            continue
            
        # 2. ค้นหา Seed ที่ดีที่สุดในช่วง In-Sample
        best_seed_is, max_net_is, _ = find_best_seed_for_window(prices_is, num_seeds_to_try, max_workers)
        
        # 3. สร้าง Action Sequence จาก Seed ที่ดีที่สุดสำหรับช่วง Out-of-Sample
        rng_oos = np.random.default_rng(best_seed_is)
        actions_oos = rng_oos.integers(0, 2, size=len(prices_oos))
        
        # 4. ทดสอบ Action Sequence ในช่วง Out-of-Sample
        df_oos = run_simulation(prices_oos, actions_oos.tolist())
        if not df_oos.empty:
            df_oos.index = ticker_data.index[oos_start:oos_end]
            all_oos_results.append(df_oos)
            
            # เก็บข้อมูลสรุปของแต่ละ Walk
            is_dates = f"{ticker_data.index[is_start].strftime('%Y-%m-%d')} ถึง {ticker_data.index[is_end-1].strftime('%Y-%m-%d')}"
            oos_dates = f"{ticker_data.index[oos_start].strftime('%Y-%m-%d')} ถึง {ticker_data.index[oos_end-1].strftime('%Y-%m-%d')}"
            summary = {
                'Walk #': i + 1,
                'In-Sample Period': is_dates,
                'Out-of-Sample Period': oos_dates,
                'Best Seed (from IS)': best_seed_is,
                'In-Sample Net': round(max_net_is, 2),
                'Out-of-Sample Net': round(df_oos['net'].iloc[-1], 2)
            }
            walk_forward_summary_list.append(summary)

        progress_bar.progress((i + 1) / num_walks, text=f"ประมวลผล Walk {i+1}/{num_walks}")

    progress_bar.empty()

    if not all_oos_results:
        return pd.DataFrame(), pd.DataFrame()

    # 5. รวมผลลัพธ์ของทุกช่วง Out-of-Sample เข้าด้วยกัน
    full_equity_curve = pd.concat(all_oos_results)
    
    # ปรับ Net ให้ต่อเนื่องกัน
    cumulative_net = 0
    adjusted_net = []
    for df in all_oos_results:
        # Net ของช่วงนี้จะเริ่มจาก Net สะสมก่อนหน้า + Net ของช่วงนี้
        period_net = df['net'] + cumulative_net
        adjusted_net.extend(period_net.tolist())
        # อัปเดต Net สะสม
        cumulative_net = period_net.iloc[-1]
    
    full_equity_curve['net'] = adjusted_net
    
    summary_df = pd.DataFrame(walk_forward_summary_list)
    return full_equity_curve, summary_df

# ==============================================================================
# 5. UI Rendering Functions (เพิ่ม Tab ใหม่)
# ==============================================================================
def render_settings_tab(config: Dict[str, Any]):
    # (โค้ดเดิม ไม่มีการเปลี่ยนแปลง)
    st.write("⚙️ **การตั้งค่าพารามิเตอร์**")
    asset_list = config.get('assets', ['FFWM'])
    try: default_index = asset_list.index(st.session_state.test_ticker)
    except ValueError: default_index = 0
    st.session_state.test_ticker = st.selectbox("เลือก Ticker สำหรับทดสอบ", options=asset_list, index=default_index)
    st.write("📅 **ช่วงวันที่สำหรับการวิเคราะห์**"); col1, col2 = st.columns(2)
    with col1: st.session_state.start_date = st.date_input("วันที่เริ่มต้น", value=st.session_state.start_date)
    with col2: st.session_state.end_date = st.date_input("วันที่สิ้นสุด", value=st.session_state.end_date)
    if st.session_state.start_date >= st.session_state.end_date: st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
    else: st.info(f"ช่วงวันที่ที่เลือก: {st.session_state.start_date:%Y-%m-%d} ถึง {st.session_state.end_date:%Y-%m-%d}")
    st.session_state.window_size = st.number_input("ขนาด Window (วัน)", min_value=2, value=st.session_state.window_size)
    st.session_state.num_seeds = st.number_input("จำนวน Seeds ต่อ Window", min_value=100, value=st.session_state.num_seeds, format="%d")
    st.session_state.max_workers = st.number_input("จำนวน Workers", min_value=1, max_value=16, value=st.session_state.max_workers)

def display_comparison_charts(results: Dict[str, pd.DataFrame], chart_title: str = '📊 เปรียบเทียบกำไรสุทธิ (Net Profit)'):
    # (โค้ดเดิม ไม่มีการเปลี่ยนแปลง)
    if not results:
        st.warning("ไม่มีข้อมูลสำหรับสร้างกราฟเปรียบเทียบ"); return
    try: longest_index = max((df.index for df in results.values() if not df.empty), key=len, default=None)
    except ValueError: longest_index = None
    if longest_index is None:
        st.warning("ไม่มีข้อมูลสำหรับสร้างกราฟเปรียบเทียบ"); return
    chart_data_dict = {}
    for name, df in results.items():
        if not df.empty:
            chart_data_dict[name] = df['net'].reindex(longest_index)
    chart_data = pd.DataFrame(chart_data_dict)
    st.write(chart_title)
    st.line_chart(chart_data)

def render_test_tab():
    # (โค้ดเดิม ไม่มีการเปลี่ยนแปลง)
    st.write("---")
    if st.button("🚀 เริ่มทดสอบ Best Seed (Optimized)", type="primary"):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้องในแท็บ 'การตั้งค่า'"); return
        ticker = st.session_state.test_ticker
        start_date_str = st.session_state.start_date.strftime('%Y-%m-%d')
        end_date_str = st.session_state.end_date.strftime('%Y-%m-%d')
        st.info(f"กำลังดึงข้อมูลสำหรับ **{ticker}** | {start_date_str} ถึง {end_date_str}")
        ticker_data = get_ticker_data(ticker, start_date_str, end_date_str)
        if ticker_data.empty:
            st.error("ไม่พบข้อมูลสำหรับ Ticker และช่วงวันที่ที่เลือก"); return
        prices = ticker_data['Close'].tolist(); num_days = len(prices)
        with st.spinner("กำลังคำนวณกลยุทธ์ต่างๆ..."):
            st.write("🔍 **เริ่มต้นการค้นหา Best Seed ด้วย Sliding Window**")
            actions_sliding, df_windows = generate_actions_sliding_window(
                ticker_data, st.session_state.window_size,
                st.session_state.num_seeds, st.session_state.max_workers
            )
            actions_min = generate_actions_rebalance_daily(num_days); actions_max = generate_actions_perfect_foresight(prices)
            results = {}; strategy_map = {
                Strategy.SLIDING_WINDOW: actions_sliding.tolist(),
                Strategy.REBALANCE_DAILY: actions_min.tolist(),
                Strategy.PERFECT_FORESIGHT: actions_max.tolist()
            }
            for strategy_name, actions in strategy_map.items():
                df = run_simulation(prices, actions)
                if not df.empty:
                    df.index = ticker_data.index[:len(df)]
                results[strategy_name] = df
        st.success("การทดสอบเสร็จสมบูรณ์!")
        st.write("---"); display_comparison_charts(results)
        st.write("📈 **สรุปผลการค้นหา Best Seed**")
        total_actions = df_windows['action_count'].sum(); total_net = df_windows['max_net'].sum()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Windows", df_windows.shape[0]); col2.metric("Total Actions", f"{total_actions}/{num_days}"); col3.metric("Total Net (Sum)", f"${total_net:,.2f}")
        st.dataframe(df_windows[['window_number', 'timeline', 'best_seed', 'max_net', 'price_change_pct', 'action_count']], use_container_width=True)
        csv = df_windows.to_csv(index=False); st.download_button(
            label="📥 ดาวน์โหลด Window Details (CSV)", data=csv,
            file_name=f'best_seed_{ticker}_{st.session_state.window_size}w.csv', mime='text/csv'
        )

def render_analytics_tab():
    # (โค้ดเดิม ไม่มีการเปลี่ยนแปลง - โค้ดยาวจึงย่อไว้)
    st.header("📊 Advanced Analytics Dashboard")
    # ... โค้ดส่วน Analytics ทั้งหมด ...

def render_manual_seed_tab(config: Dict[str, Any]):
    # (โค้ดเดิม ไม่มีการเปลี่ยนแปลง - โค้ดยาวจึงย่อไว้)
    st.header("🌱 Manual Seed Strategy Comparator")
    # ... โค้ดส่วน Manual Seed ทั้งหมด ...

def render_walk_forward_tab():
    st.header("🔬 Walk-Forward Optimization Analysis")
    st.markdown("""
    ทดสอบความทนทานของกลยุทธ์การหา Seed โดยการแบ่งข้อมูลเป็น 2 ส่วน:
    1.  **In-Sample (Train):** ช่วงข้อมูลที่ใช้สำหรับ "ค้นหา" Seed ที่ดีที่สุด
    2.  **Out-of-Sample (Test):** ช่วงข้อมูลที่กลยุทธ์ไม่เคยเห็น ใช้สำหรับ "ทดสอบ" ประสิทธิภาพของ Seed ที่หาได้
    
    กระบวนการนี้จะทำซ้ำโดยการเลื่อนหน้าต่างไปเรื่อยๆ เพื่อจำลองการใช้งานจริง
    """)

    st.write("---")
    st.subheader("การตั้งค่า Walk-Forward")
    
    col1, col2 = st.columns(2)
    st.session_state.wf_train_period = col1.number_input("ขนาดช่วง In-Sample (Train) (วัน)", min_value=10, value=st.session_state.wf_train_period)
    st.session_state.wf_test_period = col2.number_input("ขนาดช่วง Out-of-Sample (Test) (วัน)", min_value=5, value=st.session_state.wf_test_period)

    st.info(f"ระบบจะทำการ Train ด้วยข้อมูล {st.session_state.wf_train_period} วัน และ Test บนข้อมูล {st.session_state.wf_test_period} วันถัดไป แล้วทำซ้ำ")

    if st.button("🚀 เริ่มการทดสอบ Walk-Forward", type="primary"):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้องในแท็บ 'การตั้งค่า'"); return

        ticker = st.session_state.test_ticker
        start_date_str = st.session_state.start_date.strftime('%Y-%m-%d')
        end_date_str = st.session_state.end_date.strftime('%Y-%m-%d')

        st.info(f"กำลังดึงข้อมูลสำหรับ **{ticker}** | {start_date_str} ถึง {end_date_str}")
        ticker_data = get_ticker_data(ticker, start_date_str, end_date_str)

        if ticker_data.empty:
            st.error("ไม่พบข้อมูลสำหรับ Ticker และช่วงวันที่ที่เลือก"); return

        with st.spinner("กำลังดำเนินการ Walk-Forward Analysis... (อาจใช้เวลานาน)"):
            equity_curve_wf, summary_df = perform_walk_forward_analysis(
                ticker_data,
                train_period=st.session_state.wf_train_period,
                test_period=st.session_state.wf_test_period,
                num_seeds_to_try=st.session_state.num_seeds,
                max_workers=st.session_state.max_workers
            )

        if equity_curve_wf.empty:
            st.error("การทำ Walk-Forward Analysis ล้มเหลว กรุณาตรวจสอบการตั้งค่าและข้อมูล")
        else:
            st.success("Walk-Forward Analysis เสร็จสมบูรณ์!")
            st.write("---")

            # เปรียบเทียบผลลัพธ์
            prices = ticker_data['Close'].loc[equity_curve_wf.index].tolist()
            df_max = run_simulation(prices, generate_actions_perfect_foresight(prices).tolist())
            df_min = run_simulation(prices, generate_actions_rebalance_daily(len(prices)).tolist())
            
            if not df_max.empty: df_max.index = equity_curve_wf.index
            if not df_min.empty: df_min.index = equity_curve_wf.index
            
            results = {
                Strategy.WALK_FORWARD: equity_curve_wf,
                Strategy.PERFECT_FORESIGHT: df_max,
                Strategy.REBALANCE_DAILY: df_min
            }
            
            display_comparison_charts(results, chart_title="📊 เปรียบเทียบ Equity Curve (Walk-Forward vs Benchmarks)")

            st.write("---")
            st.subheader("สรุปผลการทดสอบแต่ละช่วง (Walk-Forward Summary)")
            
            oos_net = summary_df['Out-of-Sample Net']
            
            final_net = oos_net.sum()
            win_rate = (oos_net > 0).mean() * 100 if not oos_net.empty else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Out-of-Sample Net", f"${final_net:,.2f}")
            col2.metric("Walks Win Rate", f"{win_rate:.2f}%")
            col3.metric("Number of Walks", f"{len(summary_df)}")

            st.dataframe(summary_df, use_container_width=True)


# ==============================================================================
# 6. Main Application
# ==============================================================================
def main():
    st.set_page_config(page_title="Best Seed Sliding Window", page_icon="🎯", layout="wide")
    st.markdown("🎯 Best Seed Sliding Window Tester (Optimized)")
    st.caption("เครื่องมือทดสอบการหา Best Seed ด้วยวิธี Sliding Window และ Walk-Forward Analysis")

    config = load_config()
    initialize_session_state(config)

    # เพิ่ม Tab ใหม่เข้าไป
    tab1, tab2, tab5, tab3, tab4 = st.tabs([
        "⚙️ การตั้งค่า",
        "🚀 ทดสอบ Best Seed",
        "🔬 Walk-Forward Analysis", # <-- Tab ใหม่
        "📊 Advanced Analytics",
        "🌱 Manual Seed Comparator"
    ])

    with tab1:
        render_settings_tab(config)
    with tab2:
        render_test_tab()
    with tab5: # Tab ใหม่
        render_walk_forward_tab()
    with tab3:
        render_analytics_tab() # โค้ดส่วนนี้ยาวมาก อาจต้องย่อไว้ถ้าไม่ต้องการแสดง
    with tab4:
        render_manual_seed_tab(config) # โค้ดส่วนนี้ยาวมาก อาจต้องย่อไว้ถ้าไม่ต้องการแสดง

    with st.expander("📖 คำอธิบายวิธีการทำงานและแนวคิดการ Refactor"):
        st.markdown("""
        **หลักการ Refactoring ที่ใช้ในเวอร์ชันนี้:**
        - **Dynamic Configuration**: ตั้งค่ารายการ `assets`, ค่าเริ่มต้นต่างๆ, และ **ค่าเริ่มต้นของ Manual Seeds ที่แตกต่างกันในแต่ละ Asset** ผ่านไฟล์ `dynamic_seed_config.json` ทั้งหมด
        - **Callback-Driven UI**: ใช้ `on_change` callback ใน `st.selectbox` เพื่ออัปเดตค่า `presets` ของ Manual Seed ทันทีที่ผู้ใช้เปลี่ยน Ticker
        - **Separation of Concerns**: แยกโค้ดส่วน UI (`render_...`), การคำนวณ (`calculate_...`), และการสร้างข้อมูล (`generate_...`) ออกจากกันอย่างชัดเจน
        - **Centralized Initialization**: ใช้ `initialize_session_state` เพื่อตั้งค่าเริ่มต้นทั้งหมดในที่เดียว ทำให้โค้ดสะอาดและจัดการง่ายขึ้น
        - **Readability**: ใช้ชื่อตัวแปรและฟังก์ชันที่สื่อความหมาย, เพิ่ม Type Hints เพื่อให้โค้ดเข้าใจง่ายขึ้น
        """)

if __name__ == "__main__":
    main()
