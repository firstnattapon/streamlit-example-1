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
    SLIDING_WINDOW = "Best Seed Sliding Window"
    GRADIENT_DESCENT = "Gradient Descent Optimizer"

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
                "gd_iterations": 100, "gd_learning_rate": 0.1, "gd_master_seed": 42
            }
        }

def initialize_session_state(config: Dict[str, Any]):
    """
    ตั้งค่าเริ่มต้นสำหรับ Streamlit session state โดยใช้ค่าจาก config
    เพื่อให้ค่าต่างๆ ยังคงอยู่เมื่อผู้ใช้เปลี่ยนหน้าหรือทำ Action อื่นๆ
    """
    defaults = config.get('default_settings', {})

    if 'test_ticker' not in st.session_state:
        st.session_state.test_ticker = defaults.get('selected_ticker', 'FFWM')
    if 'start_date' not in st.session_state:
        try:
            st.session_state.start_date = datetime.strptime(defaults.get('start_date', '2024-01-01'), '%Y-%m-%d').date()
        except ValueError:
            st.session_state.start_date = datetime(2024, 1, 1).date()
    if 'end_date' not in st.session_state:
        st.session_state.end_date = datetime.now().date()
    if 'window_size' not in st.session_state:
        st.session_state.window_size = defaults.get('window_size', 30)
    
    # Random Seed Parameters
    if 'num_seeds' not in st.session_state:
        st.session_state.num_seeds = defaults.get('num_seeds', 10000)
    if 'max_workers' not in st.session_state:
        st.session_state.max_workers = defaults.get('max_workers', 8)

    # Gradient Descent Parameters
    if 'gd_iterations' not in st.session_state:
        st.session_state.gd_iterations = defaults.get('gd_iterations', 100)
    if 'gd_learning_rate' not in st.session_state:
        st.session_state.gd_learning_rate = defaults.get('gd_learning_rate', 0.1)
    if 'gd_master_seed' not in st.session_state:
        st.session_state.gd_master_seed = defaults.get('gd_master_seed', 42)


# ==============================================================================
# 2. Core Calculation & Data Functions (With Numba Acceleration)
# ==============================================================================
@st.cache_data(ttl=3600)
def get_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    ดึงข้อมูลราคาหุ้น/สินทรัพย์จาก Yahoo Finance และ Cache ผลลัพธ์ไว้ 1 ชั่วโมง
    """
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

# ! ACCELERATED: ฟังก์ชันแกนกลางที่ถูกเร่งความเร็วด้วย Numba
@njit(cache=True)
def _calculate_simulation_numba(action_array: np.ndarray, price_array: np.ndarray, fix: int = 1500) -> Tuple:
    """
    คำนวณผลลัพธ์การจำลองการเทรด (หัวใจของการคำนวณ)
    - ใช้ Numba @njit(cache=True) เพื่อคอมไพล์เป็น Machine Code ทำให้ทำงานเร็วมาก
    """
    n = len(action_array)
    if n == 0 or len(price_array) == 0:
        empty_arr = np.empty(0, dtype=np.float64)
        return (empty_arr, empty_arr, empty_arr, empty_arr, empty_arr, empty_arr)

    action_array_calc = action_array.copy()
    if n > 0: action_array_calc[0] = 1 # บังคับให้วันแรกต้อง Action=1 (ซื้อ) เสมอ

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
        if action_array_calc[i] == 0:  # Hold
            amount[i] = amount[i-1]
            buffer[i] = 0.0
        else:  # Rebalance
            amount[i] = fix / curr_price
            buffer[i] = amount[i-1] * curr_price - fix

        cash[i] = cash[i-1] + buffer[i]
        asset_value[i] = amount[i] * curr_price
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
    if not prices or not actions: return pd.DataFrame()
    min_len = min(len(prices), len(actions))
    prices, actions = prices[:min_len], actions[:min_len]
    
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
    """สร้าง Action สำหรับกลยุทธ์ Rebalance ทุกวัน (Min Performance)"""
    return np.ones(num_days, dtype=np.int32)

def generate_actions_perfect_foresight(prices: List[float], fix: int = 1500) -> np.ndarray:
    """สร้าง Action สำหรับกลยุทธ์ Perfect Foresight (Max Performance) โดยใช้ Dynamic Programming"""
    price_arr = np.asarray(prices, dtype=np.float64)
    n = len(price_arr)
    if n < 2: return np.ones(n, dtype=int)
    
    dp = np.zeros(n, dtype=np.float64); path = np.zeros(n, dtype=int)
    dp[0] = float(fix * 2)

    for i in range(1, n):
        j_indices = np.arange(i)
        profits = fix * ((price_arr[i] / price_arr[j_indices]) - 1)
        current_sumusd = dp[j_indices] + profits
        best_idx = np.argmax(current_sumusd)
        dp[i] = current_sumusd[best_idx]; path[i] = j_indices[best_idx]
        
    actions = np.zeros(n, dtype=int)
    current_day = np.argmax(dp)
    while current_day > 0:
        actions[current_day] = 1
        current_day = path[current_day]
    actions[0] = 1
    return actions

# 3.2 Standard Seed Generation (Original Logic)
def find_best_seed_for_window(prices_window: np.ndarray, num_seeds_to_try: int, max_workers: int) -> Tuple[int, float, np.ndarray]:
    """ ค้นหา Seed ที่ให้ค่า Net Profit สูงสุดสำหรับ Price Window ที่กำหนด (Brute Force) """
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

    best_seed_for_window, max_net_for_window = -1, -np.inf
    random_seeds = np.arange(num_seeds_to_try)
    batch_size = max(1, num_seeds_to_try // (max_workers * 4 if max_workers > 0 else 1))
    seed_batches = [random_seeds[j:j+batch_size] for j in range(0, len(random_seeds), batch_size)]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_seed_batch, batch) for batch in seed_batches]
        for future in as_completed(futures):
            for seed, final_net in future.result():
                if final_net > max_net_for_window:
                    max_net_for_window = final_net
                    best_seed_for_window = seed

    rng_best = np.random.default_rng(best_seed_for_window if best_seed_for_window >= 0 else 1)
    best_actions = rng_best.integers(0, 2, size=window_len)
    best_actions[0] = 1
    return best_seed_for_window, max_net_for_window, best_actions

def generate_actions_sliding_window(ticker_data: pd.DataFrame, window_size: int, num_seeds_to_try: int, max_workers: int) -> Tuple[np.ndarray, pd.DataFrame]:
    """ สร้าง Action Sequence ทั้งหมดโดยใช้กลยุทธ์ Sliding Window (Brute Force) """
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
        window_details_list.append({
            'window_number': i + 1, 'timeline': f"{start_date_str} ถึง {end_date_str}",
            'best_seed': best_seed, 'max_net': round(max_net, 2),
            'action_count': int(np.sum(best_actions)), 'window_size': len(best_actions)
        })
        progress_bar.progress((i + 1) / num_windows, text=f"ประมวลผล Window {i+1}/{num_windows}")
        
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)

# 3.3 NEW MODEL: Gradient Descent Optimizer
@njit(cache=True)
def _sigmoid(x: np.ndarray) -> np.ndarray:
    """ ฟังก์ชัน Sigmoid ที่เร่งความเร็วด้วย Numba """
    return 1.0 / (1.0 + np.exp(-x))

def find_best_solution_gradient_descent(prices_window: np.ndarray, iterations: int, learning_rate: float, seed: int) -> Tuple[float, np.ndarray]:
    """
    ค้นหา Action Sequence ที่ดีที่สุดโดยใช้ Gradient Descent
    - สร้าง Objective Function ที่ Differentiable ได้ เพื่อให้หา Gradient ได้
    - Objective: พยายามสร้าง Action ให้สอดคล้องกับการเปลี่ยนแปลงของราคาในวันถัดไป (ซื้อเมื่อคาดว่าราคาจะขึ้น)
    - ปรับค่าพารามิเตอร์ `theta` ด้วย Gradient Descent เพื่อลด Loss
    - แปลง `theta` ที่ได้ผลดีที่สุดเป็น Binary Actions (0 หรือ 1)
    """
    window_len = len(prices_window)
    if window_len < 2: return 0.0, np.ones(window_len, dtype=np.int32)
    
    # ใช้ PCG64 ซึ่งเป็น Generator ที่ทันสมัยและมีคุณสมบัติทางสถิติที่ดี
    rng = np.random.default_rng(np.random.PCG64(seed))
    # theta คือพารามิเตอร์ที่เราจะทำการ Optimize
    theta = rng.standard_normal(size=window_len, dtype=np.float64)

    # สัญญาณสำหรับการเรียนรู้: การเปลี่ยนแปลงของราคา (p_{i+1} - p_i)
    # เราใช้ np.diff และเติมค่าสุดท้ายเพื่อให้มีขนาดเท่ากับ window_len
    price_changes = np.diff(prices_window, append=prices_window[-1])
    
    # Optimization Loop
    for _ in range(iterations):
        soft_actions = _sigmoid(theta) # ค่า Action แบบต่อเนื่องระหว่าง 0-1
        
        # คำนวณ Gradient ของ Loss Function เทียบกับ theta
        # Loss = sum(-soft_action * price_change) -> พยายามทำให้ soft_action สูงเมื่อ price_change เป็นบวก
        # Gradient = -price_change * sigmoid_derivative
        gradient = -price_changes * soft_actions * (1.0 - soft_actions)
        
        # อัปเดตพารามิเตอร์ theta
        theta -= learning_rate * gradient

    # แปลง theta ที่ Optimize แล้วเป็น Binary Actions
    final_actions = (_sigmoid(theta) > 0.5).astype(np.int32)
    final_actions[0] = 1 # วันแรกต้องซื้อเสมอ

    # ประเมินผลลัพธ์สุดท้ายด้วย Simulation จริง
    _, sumusd, _, _, _, refer = calculate_optimized_cached(tuple(final_actions), tuple(prices_window))
    net = sumusd[-1] - refer[-1] - sumusd[0] if len(sumusd) > 0 else -np.inf
        
    return net, final_actions

def generate_actions_sliding_window_gd(ticker_data: pd.DataFrame, window_size: int, iterations: int, learning_rate: float, master_seed: int) -> Tuple[np.ndarray, pd.DataFrame]:
    """ สร้าง Action Sequence ทั้งหมดโดยใช้กลยุทธ์ Gradient Descent Optimizer """
    prices = ticker_data['Close'].to_numpy(); n = len(prices)
    final_actions = np.array([], dtype=np.int32); window_details_list = []
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="กำลังประมวลผล Gradient Descent Optimizer...")

    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        window_len = len(prices_window)
        if window_len < 2: continue
        
        window_seed = master_seed + i # ใช้ Seed ที่เปลี่ยนไปในแต่ละ Window แต่ยังคงเดิมทุกครั้งที่รัน
        max_net, best_actions = find_best_solution_gradient_descent(prices_window, iterations, learning_rate, seed=window_seed)
        
        final_actions = np.concatenate((final_actions, best_actions))
        
        start_date_str = ticker_data.index[start_index].strftime('%Y-%m-%d'); end_date_str = ticker_data.index[end_index-1].strftime('%Y-%m-%d')
        detail = {
            'window_number': i + 1, 'timeline': f"{start_date_str} ถึง {end_date_str}",
            'max_net': round(max_net, 2), 'action_count': int(np.sum(best_actions)),
            'window_size': window_len, 'window_seed': window_seed
        }
        window_details_list.append(detail)
        progress_bar.progress((i + 1) / num_windows, text=f"Optimizing Window {i+1}/{num_windows}")
        
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)

# ==============================================================================
# 4. UI Rendering Functions
# ==============================================================================
def render_settings_tab(config: Dict[str, Any]):
    """แสดงผล UI สำหรับ Tab การตั้งค่า"""
    st.write("⚙️ **การตั้งค่าพารามิเตอร์**")
    
    asset_list = config.get('assets', ['FFWM'])
    try:
        default_index = asset_list.index(st.session_state.test_ticker)
    except ValueError:
        default_index = 0
    st.session_state.test_ticker = st.selectbox("เลือก Ticker สำหรับทดสอบ", options=asset_list, index=default_index)

    st.write("📅 **ช่วงวันที่สำหรับการวิเคราะห์**")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.start_date = st.date_input("วันที่เริ่มต้น", value=st.session_state.start_date)
    with col2:
        st.session_state.end_date = st.date_input("วันที่สิ้นสุด", value=st.session_state.end_date)
    if st.session_state.start_date >= st.session_state.end_date:
        st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")

    st.divider()
    st.session_state.window_size = st.number_input("ขนาด Window (วัน)", min_value=2, value=st.session_state.window_size, help="ขนาดของแต่ละช่วงเวลาที่จะนำไปหา Action ที่ดีที่สุด")
    st.divider()
    
    col_rand, col_gd = st.columns(2)
    
    with col_rand:
        st.subheader("พารามิเตอร์สำหรับ Random Search")
        st.session_state.num_seeds = st.number_input("จำนวน Seeds ต่อ Window", min_value=100, value=st.session_state.num_seeds, format="%d", help="ยิ่งเยอะ ยิ่งมีโอกาสเจอ Seed ที่ดี แต่ใช้เวลามากขึ้น")
        st.session_state.max_workers = st.number_input("จำนวน Workers (CPU Cores)", min_value=1, max_value=16, value=st.session_state.max_workers, help="จำนวน Core ที่จะใช้ประมวลผลพร้อมกัน")
    
    with col_gd:
        st.subheader("พารามิเตอร์สำหรับ Gradient Descent")
        st.session_state.gd_iterations = st.number_input("จำนวน Iterations", min_value=10, value=st.session_state.gd_iterations, help="จำนวนรอบในการ Optimize")
        st.session_state.gd_learning_rate = st.number_input("Learning Rate", min_value=0.001, value=st.session_state.gd_learning_rate, format="%.3f", help="อัตราการเรียนรู้ของโมเดล")
        st.session_state.gd_master_seed = st.number_input("Master Seed", value=st.session_state.gd_master_seed, format="%d", help="Seed หลักเพื่อผลลัพธ์ที่ทำซ้ำได้")


def display_comparison_charts(results: Dict[str, pd.DataFrame], chart_title: str = '📊 เปรียบเทียบกำไรสุทธิ (Net Profit)'):
    """แสดงผลกราฟเปรียบเทียบผลลัพธ์จากหลายๆ กลยุทธ์"""
    if not results: return
    valid_dfs = {name: df for name, df in results.items() if not df.empty and 'net' in df.columns}
    if not valid_dfs: return
        
    try:
        longest_index = max((df.index for df in valid_dfs.values()), key=len)
    except ValueError:
        return

    chart_data = pd.DataFrame(index=longest_index)
    for name, df in valid_dfs.items():
        chart_data[name] = df['net'].reindex(longest_index).ffill()
    
    st.write(chart_title)
    st.line_chart(chart_data)
    
    st.subheader("สรุปผลลัพธ์สุดท้าย (Final Net Profit)")
    final_nets = {name: df['net'].iloc[-1] for name, df in valid_dfs.items()}
    sorted_strategies = sorted(final_nets, key=final_nets.get, reverse=True)
    
    metric_cols = st.columns(len(sorted_strategies))
    for i, name in enumerate(sorted_strategies):
        metric_cols[i].metric(name, f"${final_nets[name]:,.2f}")


def render_test_tab():
    """แสดงผล UI สำหรับ Tab Best Seed (Random)"""
    st.write("---")
    st.info("กลยุทธ์นี้จะทำการ **'สุ่ม'** Action Sequence จำนวนมาก (ตามที่ตั้งค่า `num_seeds`) แล้วเลือกอันที่ให้ผลตอบแทนดีที่สุดในแต่ละ Window")

    if st.button("🚀 เริ่มทดสอบ Best Seed (Random Search)", type="primary"):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้องในแท็บ 'การตั้งค่า'"); return
            
        ticker = st.session_state.test_ticker
        start_date_str = st.session_state.start_date.strftime('%Y-%m-%d')
        end_date_str = st.session_state.end_date.strftime('%Y-%m-%d')
        
        st.info(f"กำลังดึงข้อมูลสำหรับ **{ticker}** | {start_date_str} ถึง {end_date_str}")
        ticker_data = get_ticker_data(ticker, start_date_str, end_date_str)
        if ticker_data.empty: st.error("ไม่พบข้อมูล"); return
            
        prices = ticker_data['Close'].to_numpy(); num_days = len(prices)

        with st.spinner("กำลังคำนวณกลยุทธ์ต่างๆ (Core calculation is Numba-accelerated)..."):
            actions_sliding, df_windows = generate_actions_sliding_window(
                ticker_data, st.session_state.window_size, st.session_state.num_seeds, st.session_state.max_workers
            )
            actions_min = generate_actions_rebalance_daily(num_days)
            actions_max = generate_actions_perfect_foresight(prices.tolist())
            
            results = {
                Strategy.SLIDING_WINDOW: run_simulation(prices.tolist(), actions_sliding.tolist()),
                Strategy.PERFECT_FORESIGHT: run_simulation(prices.tolist(), actions_max.tolist()),
                Strategy.REBALANCE_DAILY: run_simulation(prices.tolist(), actions_min.tolist())
            }
            for name, df in results.items():
                if not df.empty: df.index = ticker_data.index[:len(df)]
        
        st.success("การทดสอบเสร็จสมบูรณ์!")
        st.write("---")
        display_comparison_charts(results)
        
        st.write("📈 **สรุปผลการค้นหา Best Seed (Random Search)**")
        st.dataframe(df_windows, use_container_width=True)


def render_gradient_descent_tab():
    """แสดงผล UI สำหรับ Tab Gradient Descent Optimizer"""
    st.write("---")
    st.info("กลยุทธ์ใหม่นี้ใช้ **Gradient Descent** เพื่อ 'เรียนรู้' และปรับปรุง Action Sequence ให้ดีที่สุดในแต่ละ Window โดยพยายามทำให้ใกล้เคียงกับ Perfect Foresight มากขึ้น แทนการสุ่มแบบ Brute-force")

    if st.button("📈 เริ่มทดสอบ Gradient Descent Optimizer", type="primary"):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้องในแท็บ 'การตั้งค่า'"); return
            
        ticker = st.session_state.test_ticker
        start_date_str = st.session_state.start_date.strftime('%Y-%m-%d')
        end_date_str = st.session_state.end_date.strftime('%Y-%m-%d')
        
        st.info(f"กำลังดึงข้อมูลสำหรับ **{ticker}** | {start_date_str} ถึง {end_date_str}")
        ticker_data = get_ticker_data(ticker, start_date_str, end_date_str)
        if ticker_data.empty: st.error("ไม่พบข้อมูล"); return
        
        prices = ticker_data['Close'].to_numpy(); num_days = len(prices)
        with st.spinner("กำลัง Optimize Action Sequences ด้วย Gradient Descent..."):
            actions_gd, df_windows = generate_actions_sliding_window_gd(
                ticker_data, st.session_state.window_size, 
                st.session_state.gd_iterations, st.session_state.gd_learning_rate,
                st.session_state.gd_master_seed
            )
            actions_min = generate_actions_rebalance_daily(num_days)
            actions_max = generate_actions_perfect_foresight(prices.tolist())
            
            results = {
                Strategy.GRADIENT_DESCENT: run_simulation(prices.tolist(), actions_gd.tolist()),
                Strategy.PERFECT_FORESIGHT: run_simulation(prices.tolist(), actions_max.tolist()),
                Strategy.REBALANCE_DAILY: run_simulation(prices.tolist(), actions_min.tolist())
            }
            for name, df in results.items():
                if not df.empty: df.index = ticker_data.index[:len(df)]

        st.success("การทดสอบเสร็จสมบูรณ์!")
        st.write("---")
        display_comparison_charts(results)
        
        st.write("📈 **สรุปผลการค้นหาด้วย Gradient Descent Optimizer**")
        st.dataframe(df_windows, use_container_width=True)

# ==============================================================================
# 5. Main Application
# ==============================================================================
def main():
    """ ฟังก์ชันหลักในการรัน Streamlit Application """
    st.set_page_config(page_title="Dynamic Seed Optimizer", page_icon="🎯", layout="wide")
    st.markdown("### 🎯 Dynamic Seed Optimizer (Numba Accelerated)")
    st.caption("เครื่องมือทดสอบและเปรียบเทียบกลยุทธ์การหา Action Sequence ที่ดีที่สุด")

    config = load_config()
    initialize_session_state(config)

    tab_list = [
        "⚙️ การตั้งค่า", 
        "🚀 Best Seed (Random Search)", 
        "📈 Gradient Descent Optimizer (ใหม่)"
    ]
    tabs = st.tabs(tab_list)

    with tabs[0]: render_settings_tab(config)
    with tabs[1]: render_test_tab()
    with tabs[2]: render_gradient_descent_tab()

    with st.expander("📖 คำอธิบายวิธีการทำงานและแนวคิด"):
        st.markdown("""
        ### หลักการทำงานของแต่ละโมเดล:

        1.  **Best Seed (Random Search)**:
            - **หลักการ**: Brute Force. ในแต่ละ `Window` จะทำการสุ่ม `Action Sequence` (ลำดับการซื้อ/ไม่ซื้อ) ขึ้นมาเป็นจำนวนมาก (ตามค่า `Num Seeds`).
            - **การเลือก**: ระบบจะเลือก `Action Sequence` ที่ให้กำไร (Net Profit) สูงที่สุดจากการสุ่มทั้งหมด มาใช้สำหรับ `Window` นั้นๆ
            - **ข้อดี**: เข้าใจง่าย, ตรงไปตรงมา
            - **ข้อเสีย**: เหมือนการงมเข็มในมหาสมุทร อาจไม่เจอคำตอบที่ดีที่สุดถ้าสุ่มไม่มากพอ และใช้พลังการประมวลผลสูง

        2.  **📈 Gradient Descent Optimizer (ใหม่)**:
            - **หลักการ**: Optimization. แทนที่จะสุ่มอย่างไร้ทิศทาง โมเดลนี้จะ "เรียนรู้" จากข้อมูลราคาเพื่อหา `Action Sequence` ที่ดีที่สุด
            - **วิธีการ**:
                - สร้าง `Action` ในรูปแบบต่อเนื่อง (ค่าระหว่าง 0-1) โดยใช้ฟังก์ชัน `Sigmoid`
                - กำหนดเป้าหมาย (Objective Function) คือ **"พยายามซื้อ (Action=1) ก่อนวันที่ราคาจะขึ้น และไม่ซื้อ (Action=0) ก่อนวันที่ราคาจะลง"**
                - ใช้เทคนิค **Gradient Descent** เพื่อคำนวณความชัน (Gradient) และค่อยๆ ปรับ `Action` ไปในทิศทางที่ทำให้ได้กำไรตามเป้าหมายมากขึ้นเรื่อยๆ จนครบจำนวน `Iterations`
                - แปลง `Action` ที่เรียนรู้เสร็จแล้วให้เป็น 0 หรือ 1 เพื่อนำไปใช้งานจริง
            - **ข้อดี**: มีหลักการและทิศทางในการค้นหา มีโอกาสเข้าใกล้คำตอบที่ดีที่สุด (Perfect Foresight) ได้มากกว่าการสุ่ม โดยใช้การคำนวณน้อยกว่า
            - **ข้อเสีย**: ผลลัพธ์ขึ้นอยู่กับการตั้งค่า Hyperparameters (เช่น Learning Rate, Iterations)

        3.  **Core Acceleration**:
            - การคำนวณผลการเทรด (Simulation) ซึ่งเป็นส่วนที่ทำงานช้าที่สุด ถูกเร่งความเร็วด้วย **Numba (`@njit`)** ซึ่งจะแปลงโค้ด Python เป็น Machine Code ทำให้การทดสอบทั้งหมดรวดเร็วยิ่งขึ้น
        """)

if __name__ == "__main__":
    main()
