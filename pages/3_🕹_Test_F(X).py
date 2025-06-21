import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from datetime import datetime
from typing import List, Tuple, Dict, Any

# ! NUMBA: Import Numba's Just-In-Time compiler for core acceleration
from numba import njit

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================

class Strategy:
    """คลาสสำหรับเก็บชื่อกลยุทธ์ต่างๆ เพื่อให้เรียกใช้ง่ายและลดข้อผิดพลาด"""
    REBALANCE_DAILY = "Rebalance Daily (Min)"
    PERFECT_FORESIGHT = "Perfect Foresight (Max)"
    SLIDING_WINDOW_RANDOM = "Best Seed (Random Search)"
    ARITHMETIC_SEQUENCE = "Arithmetic Sequence Optimizer"

def load_config(filepath: str = "dynamic_seed_config.json") -> Dict[str, Any]:
    """
    โหลดการตั้งค่าจากไฟล์ JSON
    หากไฟล์ไม่พบหรือมีข้อผิดพลาด จะคืนค่า default เพื่อให้โปรแกรมทำงานต่อได้
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        st.warning(f"⚠️ ไม่พบหรือไฟล์ '{filepath}' ไม่ถูกต้อง จะใช้ค่าเริ่มต้นแทน")
        return {
            "assets": ["FFWM", "NEGG", "RIVN", "BTC-USD", "NVDA", "TSLA", "META"],
            "default_settings": {
                "selected_ticker": "FFWM", "start_date": "2024-01-01", "window_size": 30,
                "num_seeds": 10000, "max_workers": 8,
                "as_num_trials": 5000, "as_master_seed": 42
            }
        }

def initialize_session_state(config: Dict[str, Any]):
    """
    ตั้งค่าเริ่มต้นสำหรับ Streamlit session state โดยใช้ค่าจาก config
    """
    defaults = config.get('default_settings', {})

    if 'test_ticker' not in st.session_state:
        st.session_state.test_ticker = defaults.get('selected_ticker', 'FFWM')
    if 'start_date' not in st.session_state:
        st.session_state.start_date = datetime.strptime(defaults.get('start_date', '2024-01-01'), '%Y-%m-%d').date()
    if 'end_date' not in st.session_state:
        st.session_state.end_date = datetime.now().date()
    if 'window_size' not in st.session_state:
        st.session_state.window_size = defaults.get('window_size', 30)
    
    # Random Search Parameters
    if 'num_seeds' not in st.session_state:
        st.session_state.num_seeds = defaults.get('num_seeds', 10000)
    if 'max_workers' not in st.session_state:
        st.session_state.max_workers = defaults.get('max_workers', 8)

    # Arithmetic Sequence Parameters
    if 'as_num_trials' not in st.session_state:
        st.session_state.as_num_trials = defaults.get('as_num_trials', 5000)
    if 'as_master_seed' not in st.session_state:
        st.session_state.as_master_seed = defaults.get('as_master_seed', 42)


# ==============================================================================
# 2. Core Calculation & Data Functions (With Numba Acceleration)
# ==============================================================================
@st.cache_data(ttl=3600)
def get_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """ ดึงข้อมูลราคาหุ้น/สินทรัพย์จาก Yahoo Finance และ Cache ผลลัพธ์ไว้ 1 ชั่วโมง """
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

@njit(cache=True)
def _calculate_simulation_numba(action_array: np.ndarray, price_array: np.ndarray, fix: int = 1500) -> Tuple:
    """ คำนวณผลลัพธ์การจำลองการเทรด (เร่งความเร็วด้วย Numba) """
    n = len(action_array)
    if n == 0 or len(price_array) == 0:
        empty_arr = np.empty(0, dtype=np.float64)
        return (empty_arr, empty_arr, empty_arr, empty_arr, empty_arr, empty_arr)

    action_array_calc = action_array.copy()
    if n > 0: action_array_calc[0] = 1 # บังคับซื้อวันแรก

    amount = np.empty(n, dtype=np.float64); buffer = np.zeros(n, dtype=np.float64)
    cash = np.empty(n, dtype=np.float64); asset_value = np.empty(n, dtype=np.float64)
    sumusd = np.empty(n, dtype=np.float64)

    initial_price = price_array[0]
    amount[0] = fix / initial_price; cash[0] = fix; asset_value[0] = amount[0] * initial_price
    sumusd[0] = cash[0] + asset_value[0]
    refer = -fix * np.log(initial_price / price_array)

    for i in range(1, n):
        curr_price = price_array[i]
        if action_array_calc[i] == 0:
            amount[i] = amount[i-1]; buffer[i] = 0.0
        else:
            amount[i] = fix / curr_price; buffer[i] = amount[i-1] * curr_price - fix
        cash[i] = cash[i-1] + buffer[i]; asset_value[i] = amount[i] * curr_price
        sumusd[i] = cash[i] + asset_value[i]
    return buffer, sumusd, cash, asset_value, amount, refer

@lru_cache(maxsize=8192)
def calculate_optimized_cached(action_tuple: Tuple[int, ...], price_tuple: Tuple[float, ...], fix: int = 1500) -> Tuple:
    """ Wrapper function สำหรับเรียกฟังก์ชัน Numba โดยใช้ Cache """
    action_array = np.asarray(action_tuple, dtype=np.int32)
    price_array = np.asarray(price_tuple, dtype=np.float64)
    return _calculate_simulation_numba(action_array, price_array, fix)

def run_simulation(prices: List[float], actions: List[int], fix: int = 1500) -> pd.DataFrame:
    """ สร้าง DataFrame ผลลัพธ์จากการจำลองการเทรด """
    min_len = min(len(prices), len(actions)); prices, actions = prices[:min_len], actions[:min_len]
    if not prices or not actions: return pd.DataFrame()
    buffer, sumusd, cash, asset_value, amount, refer = calculate_optimized_cached(tuple(actions), tuple(prices), fix)
    if len(sumusd) == 0: return pd.DataFrame()
    initial_capital = sumusd[0]
    return pd.DataFrame({
        'price': prices, 'action': actions, 'buffer': np.round(buffer, 2),
        'sumusd': np.round(sumusd, 2), 'cash': np.round(cash, 2),
        'asset_value': np.round(asset_value, 2), 'amount': np.round(amount, 2),
        'refer': np.round(refer + initial_capital, 2),
        'net': np.round(sumusd - refer - initial_capital, 2)
    })

# ==============================================================================
# 3. Strategy Action Generation
# ==============================================================================

# 3.1 Standard & Benchmark Strategies
def generate_actions_rebalance_daily(num_days: int) -> np.ndarray:
    return np.ones(num_days, dtype=np.int32)

def generate_actions_perfect_foresight(prices: List[float], fix: int = 1500) -> np.ndarray:
    price_arr = np.asarray(prices, dtype=np.float64); n = len(price_arr)
    if n < 2: return np.ones(n, dtype=int)
    dp = np.zeros(n, dtype=np.float64); path = np.zeros(n, dtype=int); dp[0] = float(fix * 2)
    for i in range(1, n):
        j_indices = np.arange(i); profits = fix * ((price_arr[i] / price_arr[j_indices]) - 1)
        current_sumusd = dp[j_indices] + profits
        best_idx = np.argmax(current_sumusd)
        dp[i] = current_sumusd[best_idx]; path[i] = j_indices[best_idx]
    actions = np.zeros(n, dtype=int); current_day = np.argmax(dp)
    while current_day > 0:
        actions[current_day] = 1; current_day = path[current_day]
    actions[0] = 1
    return actions

# 3.2 Standard Seed Generation (Random Search)
def find_best_seed_for_window(prices_window: np.ndarray, num_seeds_to_try: int, max_workers: int) -> Tuple[int, float, np.ndarray]:
    window_len = len(prices_window)
    if window_len < 2: return 1, 0.0, np.ones(window_len, dtype=int)
    def evaluate_seed_batch(seed_batch: np.ndarray) -> List[Tuple[int, float]]:
        results = []
        prices_tuple = tuple(prices_window)
        for seed in seed_batch:
            rng = np.random.default_rng(seed)
            actions_window = rng.integers(0, 2, size=window_len)
            _, sumusd, _, _, _, refer = calculate_optimized_cached(tuple(actions_window), prices_tuple)
            net = sumusd[-1] - refer[-1] - sumusd[0] if len(sumusd) > 0 else -np.inf
            results.append((seed, net))
        return results
    best_seed, max_net = -1, -np.inf
    random_seeds = np.arange(num_seeds_to_try)
    batch_size = max(1, num_seeds_to_try // (max_workers * 4 if max_workers > 0 else 1))
    seed_batches = [random_seeds[j:j+batch_size] for j in range(0, len(random_seeds), batch_size)]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_seed_batch, batch) for batch in seed_batches]
        for future in as_completed(futures):
            for seed, final_net in future.result():
                if final_net > max_net:
                    max_net = final_net; best_seed = seed
    rng_best = np.random.default_rng(best_seed if best_seed >= 0 else 1)
    best_actions = rng_best.integers(0, 2, size=window_len); best_actions[0] = 1
    return best_seed, max_net, best_actions

def generate_actions_sliding_window_random(ticker_data: pd.DataFrame, window_size: int, num_seeds_to_try: int, max_workers: int) -> Tuple[np.ndarray, pd.DataFrame]:
    prices = ticker_data['Close'].to_numpy(); n = len(prices)
    final_actions = np.array([], dtype=int); window_details_list = []
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="กำลังประมวลผล Sliding Windows (Random Search)...")
    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n); prices_window = prices[start_index:end_index]
        if len(prices_window) == 0: continue
        best_seed, max_net, best_actions = find_best_seed_for_window(prices_window, num_seeds_to_try, max_workers)
        final_actions = np.concatenate((final_actions, best_actions))
        start_date_str = ticker_data.index[start_index].strftime('%Y-%m-%d'); end_date_str = ticker_data.index[end_index-1].strftime('%Y-%m-%d')
        window_details_list.append({'window_number': i + 1, 'timeline': f"{start_date_str} ถึง {end_date_str}", 'best_seed': best_seed, 'max_net': round(max_net, 2), 'action_count': int(np.sum(best_actions)), 'window_size': len(best_actions)})
        progress_bar.progress((i + 1) / num_windows, text=f"ประมวลผล Window {i+1}/{num_windows}")
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)

# 3.3 NEW MODEL: Arithmetic Sequence Optimizer
@njit(cache=True)
def generate_actions_from_arithmetic_params(length: int, start_val: float, step: float) -> np.ndarray:
    """สร้าง Action จากพารามิเตอร์ของลำดับเลขคณิต (เร่งความเร็วด้วย Numba)"""
    if length == 0: return np.empty(0, dtype=np.int32)
    
    # 1. สร้างลำดับเลขคณิต
    n_indices = np.arange(length, dtype=np.float64)
    continuous_sequence = start_val + n_indices * step
    
    # 2. จำกัดค่าให้อยู่ในช่วง 0-1
    clipped_sequence = np.clip(continuous_sequence, 0.0, 1.0)
    
    # 3. แปลงเป็น Action 0 หรือ 1
    actions = (clipped_sequence > 0.5).astype(np.int32)
    actions[0] = 1 # บังคับซื้อวันแรก
    return actions

def find_best_arithmetic_params(prices_window: np.ndarray, num_trials: int, seed: int) -> Tuple[Dict, float, np.ndarray]:
    """ค้นหาพารามิเตอร์ (start_val, step) ที่ดีที่สุดสำหรับ Window ที่กำหนด"""
    window_len = len(prices_window)
    if window_len < 2:
        return {'start': 0.5, 'step': 0.0}, 0.0, np.ones(window_len, dtype=np.int32)

    rng = np.random.default_rng(seed)
    # สุ่มพารามิเตอร์ที่จะใช้ทดสอบ
    start_vals = rng.uniform(0.0, 1.0, num_trials)
    steps = rng.uniform(-0.1, 0.1, num_trials) # สุ่ม Step ในช่วงแคบๆ เพื่อไม่ให้ค่าแกว่งเกินไป
    
    best_params = {}; max_net = -np.inf; best_actions = np.array([])
    prices_tuple = tuple(prices_window)

    for i in range(num_trials):
        start = start_vals[i]
        step = steps[i]
        
        actions_window = generate_actions_from_arithmetic_params(window_len, start, step)
        
        # ! ACCELERATED CALL
        _, sumusd, _, _, _, refer = calculate_optimized_cached(tuple(actions_window), prices_tuple)
        
        net = sumusd[-1] - refer[-1] - sumusd[0] if len(sumusd) > 0 else -np.inf
        
        if net > max_net:
            max_net = net
            best_params = {'start': start, 'step': step}
            best_actions = actions_window
            
    if not best_params: # Fallback case
        best_params = {'start': 0.5, 'step': 0.0}
        best_actions = generate_actions_from_arithmetic_params(window_len, 0.5, 0.0)
        max_net = 0.0
        
    return best_params, max_net, best_actions

def generate_actions_sliding_window_arithmetic(ticker_data: pd.DataFrame, window_size: int, num_trials: int, master_seed: int) -> Tuple[np.ndarray, pd.DataFrame]:
    prices = ticker_data['Close'].to_numpy(); n = len(prices)
    final_actions = np.array([], dtype=np.int32); window_details_list = []
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="กำลังประมวลผล Arithmetic Sequence Optimizer...")

    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        if len(prices_window) < 2: continue
        
        window_seed = master_seed + i
        best_params, max_net, best_actions = find_best_arithmetic_params(prices_window, num_trials, seed=window_seed)
        
        final_actions = np.concatenate((final_actions, best_actions))
        
        start_date_str = ticker_data.index[start_index].strftime('%Y-%m-%d'); end_date_str = ticker_data.index[end_index-1].strftime('%Y-%m-%d')
        detail = {
            'window_number': i + 1, 'timeline': f"{start_date_str} ถึง {end_date_str}",
            'best_start_val': round(best_params['start'], 4), 'best_step': round(best_params['step'], 4),
            'max_net': round(max_net, 2), 'action_count': int(np.sum(best_actions)),
            'window_size': len(best_actions), 'window_seed': window_seed
        }
        window_details_list.append(detail)
        progress_bar.progress((i + 1) / num_windows, text=f"Optimizing Window {i+1}/{num_windows}")
        
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)

# ==============================================================================
# 4. UI Rendering Functions
# ==============================================================================
def render_settings_tab(config: Dict[str, Any]):
    st.write("⚙️ **การตั้งค่าพารามิเตอร์**")
    
    asset_list = config.get('assets', ['FFWM'])
    st.session_state.test_ticker = st.selectbox("เลือก Ticker สำหรับทดสอบ", options=asset_list, index=asset_list.index(st.session_state.test_ticker) if st.session_state.test_ticker in asset_list else 0)

    col1, col2 = st.columns(2)
    st.session_state.start_date = col1.date_input("วันที่เริ่มต้น", value=st.session_state.start_date)
    st.session_state.end_date = col2.date_input("วันที่สิ้นสุด", value=st.session_state.end_date)
    if st.session_state.start_date >= st.session_state.end_date:
        st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")

    st.divider()
    st.session_state.window_size = st.number_input("ขนาด Window (วัน)", min_value=2, value=st.session_state.window_size, help="ขนาดของแต่ละช่วงเวลาที่จะนำไปหา Action ที่ดีที่สุด")
    st.divider()
    
    col_rand, col_as = st.columns(2)
    with col_rand:
        st.subheader("พารามิเตอร์สำหรับ Random Search")
        st.session_state.num_seeds = st.number_input("จำนวน Seeds ต่อ Window", min_value=100, value=st.session_state.num_seeds, format="%d")
        st.session_state.max_workers = st.number_input("จำนวน Workers (CPU Cores)", min_value=1, max_value=16, value=st.session_state.max_workers)
    
    with col_as:
        st.subheader("พารามิเตอร์สำหรับ Arithmetic Sequence")
        st.session_state.as_num_trials = st.number_input("จำนวน Trials ต่อ Window", min_value=100, value=st.session_state.as_num_trials, format="%d", help="จำนวนคู่ (start, step) ที่จะสุ่มทดสอบ")
        st.session_state.as_master_seed = st.number_input("Master Seed", value=st.session_state.as_master_seed, format="%d", help="Seed หลักเพื่อผลลัพธ์ที่ทำซ้ำได้")

def display_comparison_charts(results: Dict[str, pd.DataFrame], chart_title: str = '📊 เปรียบเทียบกำไรสุทธิ (Net Profit)'):
    if not results: return
    valid_dfs = {name: df for name, df in results.items() if not df.empty and 'net' in df.columns}
    if not valid_dfs: return
    longest_index = max((df.index for df in valid_dfs.values()), key=len, default=None)
    if longest_index is None: return

    chart_data = pd.DataFrame(index=longest_index)
    for name, df in valid_dfs.items():
        chart_data[name] = df['net'].reindex(longest_index).ffill()
    
    st.line_chart(chart_data, use_container_width=True)
    
    st.subheader("สรุปผลลัพธ์สุดท้าย (Final Net Profit)")
    final_nets = {name: df['net'].iloc[-1] for name, df in valid_dfs.items()}
    sorted_strategies = sorted(final_nets, key=final_nets.get, reverse=True)
    
    metric_cols = st.columns(len(sorted_strategies))
    for i, name in enumerate(sorted_strategies):
        metric_cols[i].metric(name, f"${final_nets[name]:,.2f}")

def render_random_search_tab():
    st.info("กลยุทธ์นี้จะทำการ **'สุ่ม'** Action Sequence จำนวนมาก แล้วเลือกอันที่ให้ผลตอบแทนดีที่สุดในแต่ละ Window")
    if st.button("🚀 เริ่มทดสอบ Best Seed (Random Search)", type="primary"):
        run_full_test(Strategy.SLIDING_WINDOW_RANDOM)

def render_arithmetic_tab():
    st.info("กลยุทธ์นี้จะค้นหา **'ลำดับเลขคณิต'** ที่ดีที่สุด โดยการหาค่า `start` และ `step` ที่เหมาะสมเพื่อสร้าง Action Sequence ที่ให้ผลกำไรสูงสุด")
    if st.button("📈 เริ่มทดสอบ Arithmetic Sequence Optimizer", type="primary"):
        run_full_test(Strategy.ARITHMETIC_SEQUENCE)

def run_full_test(selected_strategy: str):
    """ฟังก์ชันกลางสำหรับรันการทดสอบและแสดงผล"""
    if st.session_state.start_date >= st.session_state.end_date:
        st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้องในแท็บ 'การตั้งค่า'"); return
        
    ticker = st.session_state.test_ticker
    start_date_str = st.session_state.start_date.strftime('%Y-%m-%d')
    end_date_str = st.session_state.end_date.strftime('%Y-%m-%d')
    
    st.info(f"กำลังดึงข้อมูลสำหรับ **{ticker}** | {start_date_str} ถึง {end_date_str}")
    ticker_data = get_ticker_data(ticker, start_date_str, end_date_str)
    if ticker_data.empty: st.error("ไม่พบข้อมูล"); return
    
    prices = ticker_data['Close'].to_numpy(); num_days = len(prices)
    
    with st.spinner(f"กำลังคำนวณกลยุทธ์ '{selected_strategy}'..."):
        # Generate actions for the selected strategy
        if selected_strategy == Strategy.SLIDING_WINDOW_RANDOM:
            actions_main, df_windows = generate_actions_sliding_window_random(
                ticker_data, st.session_state.window_size, st.session_state.num_seeds, st.session_state.max_workers
            )
        elif selected_strategy == Strategy.ARITHMETIC_SEQUENCE:
            actions_main, df_windows = generate_actions_sliding_window_arithmetic(
                ticker_data, st.session_state.window_size, st.session_state.as_num_trials, st.session_state.as_master_seed
            )
        else:
            st.error("กลยุทธ์ไม่ถูกต้อง"); return

        # Generate benchmark actions
        actions_min = generate_actions_rebalance_daily(num_days)
        actions_max = generate_actions_perfect_foresight(prices.tolist())
        
        # Run simulations
        results = {
            selected_strategy: run_simulation(prices.tolist(), actions_main.tolist()),
            Strategy.PERFECT_FORESIGHT: run_simulation(prices.tolist(), actions_max.tolist()),
            Strategy.REBALANCE_DAILY: run_simulation(prices.tolist(), actions_min.tolist())
        }
        for name, df in results.items():
            if not df.empty: df.index = ticker_data.index[:len(df)]
    
    st.success("การทดสอบเสร็จสมบูรณ์!")
    st.write("---")
    display_comparison_charts(results)
    
    st.write(f"📈 **สรุปผลการค้นหาด้วย {selected_strategy}**")
    st.dataframe(df_windows, use_container_width=True)

# ==============================================================================
# 5. Main Application
# ==============================================================================
def main():
    st.set_page_config(page_title="Dynamic Sequence Optimizer", page_icon="🎯", layout="wide")
    st.markdown("### 🎯 Dynamic Sequence Optimizer (Numba Accelerated)")
    st.caption("เครื่องมือทดสอบและเปรียบเทียบกลยุทธ์การหา Action Sequence ที่ดีที่สุด")

    config = load_config(); initialize_session_state(config)

    tab_list = ["⚙️ การตั้งค่า", "🚀 Best Seed (Random Search)", "📈 Arithmetic Sequence (ใหม่)"]
    tabs = st.tabs(tab_list)

    with tabs[0]: render_settings_tab(config)
    with tabs[1]: render_random_search_tab()
    with tabs[2]: render_arithmetic_tab()

    with st.expander("📖 คำอธิบายวิธีการทำงานและแนวคิด"):
        st.markdown("""
        ### หลักการทำงานของแต่ละโมเดล:

        1.  **🚀 Best Seed (Random Search)**:
            - **หลักการ**: Brute Force. ในแต่ละ `Window` จะทำการสุ่ม `Action Sequence` (ลำดับการซื้อ/ไม่ซื้อ) ขึ้นมาเป็นจำนวนมาก แล้วเลือกอันที่ให้กำไรสูงสุด
            - **ข้อดี**: ตรงไปตรงมา, เข้าใจง่าย
            - **ข้อเสีย**: เหมือนการงมเข็มในมหาสมุทร, ใช้พลังการประมวลผลสูง

        2.  **📈 Arithmetic Sequence Optimizer (ใหม่)**:
            - **หลักการ**: Parameter Search. แทนที่จะสุ่มทั้ง Sequence, โมเดลนี้จะค้นหา **พารามิเตอร์** ที่ดีที่สุดเพื่อ **สร้าง** Sequence ขึ้นมา
            - **วิธีการ**:
                - `Action Sequence` ถูกสร้างจาก **ลำดับเลขคณิต** ซึ่งกำหนดโดยค่า 2 ตัว: `start_val` (ค่าเริ่มต้น) และ `step` (ค่าที่เปลี่ยนแปลงในแต่ละวัน)
                - ลำดับที่ได้จะเป็นค่าต่อเนื่อง (เช่น 0.5, 0.52, 0.54, ...) ซึ่งจะถูกแปลงเป็น Action (ซื้อ/ไม่ซื้อ) โดยเทียบกับ 0.5
                - ระบบจะสุ่มทดสอบคู่ `(start_val, step)` เป็นจำนวนมาก (`Num Trials`) เพื่อหาคู่ที่ดีที่สุดสำหรับแต่ละ `Window`
            - **ข้อดี**: ลดความซับซ้อนของการค้นหาจากทั้ง Sequence มาเหลือแค่ 2 พารามิเตอร์ ทำให้มีโครงสร้างและทิศทางมากกว่าการสุ่มแบบสมบูรณ์
            - **ข้อเสีย**: ประสิทธิภาพขึ้นอยู่กับว่า "ลำดับเลขคณิต" สามารถอธิบายพฤติกรรมราคาที่เหมาะสมได้ดีเพียงใด

        ---
        **Core Technology**: การคำนวณผลการเทรด (Simulation) ซึ่งเป็นส่วนที่ทำงานช้าที่สุด ถูกเร่งความเร็วด้วย **Numba (`@njit`)** ทำให้การทดสอบทั้งหมดรวดเร็วยิ่งขึ้น
        """)

if __name__ == "__main__":
    main()
