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

# ! NUMBA: Import Numba's Just-In-Time compiler for core acceleration
from numba import njit

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================

class Strategy:
    """คลาสสำหรับเก็บชื่อกลยุทธ์ต่างๆ เพื่อให้เรียกใช้ง่ายและลดข้อผิดพลาด"""
    REBALANCE_DAILY = "Rebalance Daily"
    PERFECT_FORESIGHT = "Perfect Foresight (Max)"
    SLIDING_WINDOW_BRUTE_FORCE = "Best Seed (Brute-Force)"
    SEEDED_HEURISTIC = "Seeded Heuristic (Price-Triggered)" # ! NEW STRATEGY

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
            "assets": ["FFWM", "NEGG", "RIVN", "BTC-USD", "NVDA"],
            "default_settings": {
                "selected_ticker": "FFWM", "start_date": "2024-01-01", "window_size": 30,
                "num_seeds": 10000, "max_workers": 8
            },
            "manual_seed_by_asset": {
                "default": [{'seed': 999, 'size': 50, 'tail': 15}],
                "FFWM": [{'seed': 1234, 'size': 60, 'tail': 30}, {'seed': 7777, 'size': 30, 'tail': 10}]
            }
        }

def on_ticker_change_callback(config: Dict[str, Any]):
    """
    Callback ที่จะถูกเรียกเมื่อ Ticker ใน Tab Manual Seed เปลี่ยน
    เพื่อโหลดค่า Preset ของ Seed/Size/Tail ให้ตรงกับ Ticker ที่เลือก
    """
    selected_ticker = st.session_state.get("manual_ticker_key")
    if not selected_ticker:
        return
    presets_by_asset = config.get("manual_seed_by_asset", {})
    default_presets = presets_by_asset.get("default", [{'seed': 999, 'size': 50, 'tail': 15}])
    st.session_state.manual_seed_lines = presets_by_asset.get(selected_ticker, default_presets)

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
    if 'num_seeds' not in st.session_state:
        st.session_state.num_seeds = defaults.get('num_seeds', 10000)
    if 'max_workers' not in st.session_state:
        st.session_state.max_workers = defaults.get('max_workers', 8)

    if 'df_for_analysis' not in st.session_state:
        st.session_state.df_for_analysis = None

    if 'manual_seed_lines' not in st.session_state:
        initial_ticker = st.session_state.get('test_ticker', defaults.get('selected_ticker', 'FFWM'))
        presets_by_asset = config.get("manual_seed_by_asset", {})
        default_presets = presets_by_asset.get("default", [{'seed': 999, 'size': 50, 'tail': 15}])
        st.session_state.manual_seed_lines = presets_by_asset.get(initial_ticker, default_presets)


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
    if n > 0: action_array_calc[0] = 1

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
            buffer[i] = 0.0
        else:
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
    """สร้าง DataFrame ผลลัพธ์จากการจำลองการเทรด"""
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
    return np.ones(num_days, dtype=np.int32)

def generate_actions_perfect_foresight(prices: List[float], fix: int = 1500) -> np.ndarray:
    price_arr = np.asarray(prices, dtype=np.float64)
    n = len(price_arr)
    if n < 2: return np.ones(n, dtype=int)
    dp, path = np.zeros(n, dtype=np.float64), np.zeros(n, dtype=int)
    dp[0] = float(fix * 2)
    for i in range(1, n):
        j_indices = np.arange(i)
        profits = fix * ((price_arr[i] / price_arr[j_indices]) - 1)
        current_sumusd = dp[j_indices] + profits
        best_idx = np.argmax(current_sumusd)
        dp[i], path[i] = current_sumusd[best_idx], j_indices[best_idx]
    actions = np.zeros(n, dtype=int)
    current_day = np.argmax(dp)
    while current_day > 0:
        actions[current_day] = 1
        current_day = path[current_day]
    actions[0] = 1
    return actions

# 3.2 Brute-Force Seed Generation (Original Logic)
def find_best_seed_for_window_brute_force(prices_window: np.ndarray, num_seeds_to_try: int, max_workers: int) -> Tuple[int, float, np.ndarray]:
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
                    max_net, best_seed = final_net, seed

    if best_seed >= 0:
        best_actions = np.random.default_rng(best_seed).integers(0, 2, size=window_len)
    else:
        best_seed, best_actions, max_net = 1, np.ones(window_len, dtype=int), 0.0
    best_actions[0] = 1
    return best_seed, max_net, best_actions

def generate_actions_sliding_window_brute_force(ticker_data: pd.DataFrame, window_size: int, num_seeds_to_try: int, max_workers: int) -> Tuple[np.ndarray, pd.DataFrame]:
    prices, n = ticker_data['Close'].to_numpy(), len(ticker_data)
    final_actions, window_details_list = np.array([], dtype=int), []
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="กำลังประมวลผล Sliding Windows (Brute-Force)...")
    st.write(f"📊 ข้อมูล: {n} วัน | Window: {window_size} วัน | จำนวน Windows: {num_windows} | Seeds/Window: {num_seeds_to_try:,}")
    st.write(f"⚡ ใช้ Parallel Processing: {max_workers} workers")

    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        if len(prices_window) == 0: continue
        best_seed, max_net, best_actions = find_best_seed_for_window_brute_force(prices_window, num_seeds_to_try, max_workers)
        final_actions = np.concatenate((final_actions, best_actions))
        detail = {
            'window_number': i + 1,
            'timeline': f"{ticker_data.index[start_index].strftime('%Y-%m-%d')} ถึง {ticker_data.index[end_index-1].strftime('%Y-%m-%d')}",
            'best_seed': best_seed, 'max_net': round(max_net, 2),
            'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2),
            'action_count': int(np.sum(best_actions)), 'window_size': len(prices_window),
            'action_sequence': best_actions.tolist()
        }
        window_details_list.append(detail)
        progress_bar.progress((i + 1) / num_windows, text=f"ประมวลผล Window {i+1}/{num_windows}")
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)


# 3.3 ! NEW: Seeded Heuristic Strategy (Approximating Perfect Foresight)
@njit(cache=True)
def _generate_actions_seeded_heuristic_numba(prices: np.ndarray, threshold_up: float, threshold_down: float) -> np.ndarray:
    """
    สร้าง Action โดยใช้กฎที่ขึ้นกับราคา (Price-Triggered Rule)
    - Rebalance เมื่อราคาเปลี่ยนแปลงเกินเกณฑ์ที่กำหนด
    - ถูกเร่งความเร็วด้วย Numba เพื่อประสิทธิภาพสูงสุด
    """
    n = len(prices)
    if n == 0:
        return np.empty(0, dtype=np.int32)

    actions = np.zeros(n, dtype=np.int32)
    actions[0] = 1
    last_rebalance_price = prices[0]

    for i in range(1, n):
        change_pct = (prices[i] / last_rebalance_price) - 1.0
        # ตัดสินใจ Rebalance ถ้าเข้าเงื่อนไข
        if change_pct > threshold_up or change_pct < threshold_down:
            actions[i] = 1
            last_rebalance_price = prices[i]
        else:
            actions[i] = 0
    return actions

def find_best_heuristic_seed(prices_window: np.ndarray, num_seeds_to_try: int, max_workers: int) -> Tuple[int, float, np.ndarray, float, float]:
    """
    ค้นหา 'seed' ที่ให้ผลลัพธ์ดีที่สุด โดย seed จะใช้สร้างพารามิเตอร์ (thresholds)
    สำหรับ Heuristic Rule แทนการสร้าง action ตรงๆ
    """
    window_len = len(prices_window)
    if window_len < 2:
        return 1, 0.0, np.ones(window_len, dtype=int), 0.0, 0.0

    def evaluate_heuristic_seed_batch(seed_batch: np.ndarray) -> List[Tuple[int, float, float, float]]:
        results = []
        prices_tuple = tuple(prices_window)
        for seed in seed_batch:
            rng = np.random.default_rng(seed)
            # Seed ใช้สร้างพารามิเตอร์ ไม่ใช่ Action ตรงๆ
            # เกณฑ์ขาขึ้น: 1% - 50%, เกณฑ์ขาลง: -1% ถึง -30%
            threshold_up = rng.uniform(0.01, 0.50)
            threshold_down = rng.uniform(-0.30, -0.01)

            actions_window = _generate_actions_seeded_heuristic_numba(prices_window, threshold_up, threshold_down)
            _, sumusd, _, _, _, refer = calculate_optimized_cached(tuple(actions_window), prices_tuple)
            net = sumusd[-1] - refer[-1] - sumusd[0] if len(sumusd) > 0 else -np.inf
            results.append((seed, net, threshold_up, threshold_down))
        return results

    best_seed, max_net, best_thresh_up, best_thresh_down = -1, -np.inf, 0.0, 0.0
    random_seeds = np.arange(num_seeds_to_try)
    batch_size = max(1, num_seeds_to_try // (max_workers * 4 if max_workers > 0 else 1))
    seed_batches = [random_seeds[j:j + batch_size] for j in range(0, len(random_seeds), batch_size)]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_heuristic_seed_batch, batch) for batch in seed_batches]
        for future in as_completed(futures):
            for seed, net, t_up, t_down in future.result():
                if net > max_net:
                    max_net, best_seed, best_thresh_up, best_thresh_down = net, seed, t_up, t_down

    if best_seed >= 0:
        best_actions = _generate_actions_seeded_heuristic_numba(prices_window, best_thresh_up, best_thresh_down)
    else: # Fallback
        best_seed, max_net, best_thresh_up, best_thresh_down = 1, 0.0, 0.05, -0.05
        best_actions = _generate_actions_seeded_heuristic_numba(prices_window, best_thresh_up, best_thresh_down)

    return best_seed, max_net, best_actions, best_thresh_up, best_thresh_down

def generate_actions_sliding_window_heuristic(ticker_data: pd.DataFrame, window_size: int, num_seeds_to_try: int, max_workers: int) -> Tuple[np.ndarray, pd.DataFrame]:
    """สร้าง Action Sequence โดยใช้กลยุทธ์ Seeded Heuristic"""
    prices, n = ticker_data['Close'].to_numpy(), len(ticker_data)
    final_actions, window_details_list = np.array([], dtype=int), []
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="กำลังประมวลผล Seeded Heuristic Windows...")
    st.write(f"📊 ข้อมูล: {n} วัน | Window: {window_size} วัน | จำนวน Windows: {num_windows} | Seeds/Window: {num_seeds_to_try:,}")
    st.write(f"⚡ ใช้ Parallel Processing: {max_workers} workers")

    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        if len(prices_window) == 0: continue

        best_seed, max_net, best_actions, t_up, t_down = find_best_heuristic_seed(prices_window, num_seeds_to_try, max_workers)
        final_actions = np.concatenate((final_actions, best_actions))
        
        detail = {
            'window_number': i + 1,
            'timeline': f"{ticker_data.index[start_index].strftime('%Y-%m-%d')} ถึง {ticker_data.index[end_index-1].strftime('%Y-%m-%d')}",
            'best_seed': best_seed,
            'threshold_up': round(t_up, 4),
            'threshold_down': round(t_down, 4),
            'max_net': round(max_net, 2),
            'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2),
            'action_count': int(np.sum(best_actions)),
            'window_size': len(prices_window),
            'action_sequence': best_actions.tolist()
        }
        window_details_list.append(detail)
        progress_bar.progress((i + 1) / num_windows, text=f"ประมวลผล Heuristic Window {i+1}/{num_windows}")

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
    st.subheader("พารามิเตอร์สำหรับกลยุทธ์ Sliding Window")
    c1, c2, c3 = st.columns(3)
    st.session_state.window_size = c1.number_input("ขนาด Window (วัน)", min_value=2, value=st.session_state.window_size)
    st.session_state.num_seeds = c2.number_input("จำนวน Seeds ต่อ Window", min_value=100, value=st.session_state.num_seeds, format="%d", help="จำนวนการสุ่มเพื่อหาค่าที่ดีที่สุดในแต่ละ Window")
    st.session_state.max_workers = c3.number_input("จำนวน Workers (CPU Cores)", min_value=1, max_value=16, value=st.session_state.max_workers)
    
def display_comparison_charts(results: Dict[str, pd.DataFrame], chart_title: str = '📊 เปรียบเทียบกำไรสุทธิ (Net Profit)'):
    """แสดงผลกราฟเปรียบเทียบผลลัพธ์จากหลายๆ กลยุทธ์"""
    if not results: return
    valid_dfs = {name: df for name, df in results.items() if not df.empty and 'net' in df.columns}
    if not valid_dfs: return
    try: longest_index = max((df.index for df in valid_dfs.values()), key=len, default=None)
    except ValueError: longest_index = None
    if longest_index is None: return

    chart_data = pd.DataFrame(index=longest_index)
    for name, df in valid_dfs.items():
        chart_data[name] = df['net'].reindex(longest_index).ffill()
    st.write(chart_title)
    st.line_chart(chart_data)

def render_brute_force_test_tab():
    """แสดงผล UI สำหรับ Tab Best Seed (Brute-Force)"""
    st.write("---")
    st.markdown("### 🎲 ทดสอบ Best Seed ด้วย Brute-Force Search")
    st.info("กลยุทธ์นี้จะทำการสุ่ม Action (0 หรือ 1) สำหรับทุกวันใน Window เพื่อหาชุดที่ให้ผลตอบแทนดีที่สุด เป็นวิธีพื้นฐานที่สุด")
    if st.button("🚀 เริ่มทดสอบ Best Seed (Brute-Force)", type="primary"):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้องในแท็บ 'การตั้งค่า'"); return
            
        ticker = st.session_state.test_ticker; start_date_str, end_date_str = st.session_state.start_date.strftime('%Y-%m-%d'), st.session_state.end_date.strftime('%Y-%m-%d')
        st.info(f"กำลังดึงข้อมูลสำหรับ **{ticker}** | {start_date_str} ถึง {end_date_str}")
        ticker_data = get_ticker_data(ticker, start_date_str, end_date_str)
        if ticker_data.empty: st.error("ไม่พบข้อมูลสำหรับ Ticker และช่วงวันที่ที่เลือก"); return
            
        prices, num_days = ticker_data['Close'].to_numpy(), len(ticker_data)
        with st.spinner("กำลังคำนวณกลยุทธ์ต่างๆ..."):
            actions_sliding, df_windows = generate_actions_sliding_window_brute_force(
                ticker_data, st.session_state.window_size, st.session_state.num_seeds, st.session_state.max_workers
            )
            results = {
                Strategy.SLIDING_WINDOW_BRUTE_FORCE: run_simulation(prices.tolist(), actions_sliding.tolist()),
                Strategy.REBALANCE_DAILY: run_simulation(prices.tolist(), generate_actions_rebalance_daily(num_days).tolist()),
                Strategy.PERFECT_FORESIGHT: run_simulation(prices.tolist(), generate_actions_perfect_foresight(prices.tolist()).tolist())
            }
            for name, df in results.items():
                if not df.empty: df.index = ticker_data.index[:len(df)]
        
        st.success("การทดสอบเสร็จสมบูรณ์!")
        display_comparison_charts(results)
        
        st.write("📈 **สรุปผลการค้นหา Best Seed (Brute-Force)**")
        total_actions, total_net = df_windows['action_count'].sum(), df_windows['max_net'].sum()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Windows", df_windows.shape[0]); col2.metric("Total Actions", f"{total_actions}/{num_days}"); col3.metric("Total Net (Sum)", f"${total_net:,.2f}")
        st.dataframe(df_windows[['window_number', 'timeline', 'best_seed', 'max_net', 'price_change_pct', 'action_count']], use_container_width=True)
        csv = df_windows.to_csv(index=False)
        st.download_button(label="📥 ดาวน์โหลด Window Details (CSV)", data=csv, file_name=f'bruteforce_seed_{ticker}_{st.session_state.window_size}w.csv', mime='text/csv')

def render_heuristic_test_tab():
    """แสดงผล UI สำหรับ Tab ใหม่: Seeded Heuristic"""
    st.write("---")
    st.markdown("### 🎯 ทดสอบด้วย Seeded Heuristic (Price-Triggered)")
    st.info("""
    กลยุทธ์นี้เป็นแนวทางใหม่ที่ชาญฉลาดกว่า Brute-Force โดยได้รับแรงบันดาลใจจาก Perfect Foresight:
    - **หลักการ:** แทนที่จะสุ่ม Action โดยตรง, เราจะใช้ Seed เพื่อสุ่มหา **"กฎการตัดสินใจ"** ที่ดีที่สุด
    - **กฎ:** "Rebalance เมื่อราคาเปลี่ยนแปลง (ขึ้น/ลง) เกินเกณฑ์ (`threshold`) ที่กำหนด"
    - **ผลลัพธ์:** ได้ Action Sequence ที่ตอบสนองต่อพฤติกรรมราคาจริง และมีแนวโน้มใกล้เคียงกับ Perfect Foresight มากขึ้น
    """)
    if st.button("🚀 เริ่มทดสอบ Seeded Heuristic", type="primary", key="heuristic_test_button"):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้องในแท็บ 'การตั้งค่า'"); return
            
        ticker, start_date_str, end_date_str = st.session_state.test_ticker, st.session_state.start_date.strftime('%Y-%m-%d'), st.session_state.end_date.strftime('%Y-%m-%d')
        st.info(f"กำลังดึงข้อมูลสำหรับ **{ticker}** | {start_date_str} ถึง {end_date_str}")
        ticker_data = get_ticker_data(ticker, start_date_str, end_date_str)
        if ticker_data.empty: st.error("ไม่พบข้อมูลสำหรับ Ticker และช่วงวันที่ที่เลือก"); return
        
        prices, num_days = ticker_data['Close'].to_numpy(), len(ticker_data)
        with st.spinner("กำลังค้นหากฎที่ดีที่สุดในแต่ละ Window..."):
            actions_heuristic, df_windows = generate_actions_sliding_window_heuristic(
                ticker_data, st.session_state.window_size, st.session_state.num_seeds, st.session_state.max_workers
            )
            results = {
                Strategy.SEEDED_HEURISTIC: run_simulation(prices.tolist(), actions_heuristic.tolist()),
                Strategy.REBALANCE_DAILY: run_simulation(prices.tolist(), generate_actions_rebalance_daily(num_days).tolist()),
                Strategy.PERFECT_FORESIGHT: run_simulation(prices.tolist(), generate_actions_perfect_foresight(prices.tolist()).tolist())
            }
            for name, df in results.items():
                if not df.empty: df.index = ticker_data.index[:len(df)]
        
        st.success("การทดสอบเสร็จสมบูรณ์!")
        display_comparison_charts(results)
        
        st.write("📈 **สรุปผลการค้นหา Seeded Heuristic**")
        if not df_windows.empty:
            total_net, total_actions = df_windows['max_net'].sum(), df_windows['action_count'].sum()
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Net (Sum)", f"${total_net:,.2f}")
            col2.metric("Total Actions", f"{total_actions}/{num_days}")
            col3.metric("Total Windows", df_windows.shape[0])
            st.dataframe(df_windows[['window_number', 'timeline', 'best_seed', 'threshold_up', 'threshold_down', 'max_net', 'price_change_pct', 'action_count']], use_container_width=True)
            csv = df_windows.to_csv(index=False)
            st.download_button(label="📥 ดาวน์โหลด Heuristic Details (CSV)", data=csv, file_name=f'heuristic_seed_{ticker}_{st.session_state.window_size}w.csv', mime='text/csv')
        else:
            st.warning("ไม่พบข้อมูลผลลัพธ์")

def render_analytics_tab():
    # ... (ส่วนนี้เหมือนเดิม ไม่ต้องแก้ไข)
    st.header("📊 Advanced Analytics Dashboard")
    
    with st.container(border=True):
        st.subheader("เลือกวิธีการนำเข้าข้อมูล:")
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
        st.subheader("ผลการวิเคราะห์")
        df_to_analyze = st.session_state.df_for_analysis
        try:
            # Check for necessary columns
            required_cols = ['window_number', 'timeline', 'max_net', 'action_sequence']
            if not all(col in df_to_analyze.columns for col in required_cols):
                st.error(f"ไฟล์ CSV ไม่สมบูรณ์! ต้องมีคอลัมน์พื้นฐาน: {', '.join(required_cols)}")
                return
            
            df = df_to_analyze.copy()
            if 'result' not in df.columns:
                df['result'] = np.where(df['max_net'] > 0, 'Win', 'Loss')
            
            overview_tab, stitched_dna_tab = st.tabs(["🔬 ภาพรวมและสำรวจราย Window", "🧬 Stitched DNA Analysis"])

            with overview_tab:
                st.subheader("ภาพรวมประสิทธิภาพ (Overall Performance)")
                gross_profit = df[df['max_net'] > 0]['max_net'].sum()
                gross_loss = abs(df[df['max_net'] < 0]['max_net'].sum())
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                win_rate = (df['result'] == 'Win').mean() * 100
                
                kpi_cols = st.columns(4)
                kpi_cols[0].metric("Total Net Profit", f"${df['max_net'].sum():,.2f}")
                kpi_cols[1].metric("Win Rate", f"{win_rate:.2f}%")
                kpi_cols[2].metric("Profit Factor", f"{profit_factor:.2f}")
                kpi_cols[3].metric("Total Windows", f"{df.shape[0]}")
                
                st.subheader("สำรวจข้อมูลราย Window")
                selected_window = st.selectbox(
                    'เลือก Window ที่ต้องการดูรายละเอียด:',
                    options=df['window_number'],
                    format_func=lambda x: f"Window #{x} (Timeline: {df.loc[df['window_number'] == x, 'timeline'].iloc[0]})"
                )
                if selected_window:
                    window_data = df[df['window_number'] == selected_window].iloc[0]
                    st.markdown(f"**รายละเอียดของ Window #{selected_window}**")
                    st.dataframe(window_data)

            def safe_literal_eval(val):
                if pd.isna(val): return []
                if isinstance(val, list): return val
                if isinstance(val, str) and val.strip().startswith('[') and val.strip().endswith(']'):
                    try: return ast.literal_eval(val)
                    except: return []
                return []

            with stitched_dna_tab:
                st.subheader("ทดสอบกลยุทธ์จาก 'Stitched' DNA")
                st.markdown("จำลองการเทรดจริงโดยนำ **`action_sequence`** จากแต่ละ Window มา 'เย็บ' ต่อกัน และเปรียบเทียบกับ Benchmark")
                
                df['action_sequence_list'] = df['action_sequence'].apply(safe_literal_eval)
                df_sorted = df.sort_values('window_number')
                stitched_actions = [action for seq in df_sorted['action_sequence_list'] for action in seq]
                
                dna_cols = st.columns(2)
                stitch_ticker = dna_cols[0].text_input("Ticker สำหรับจำลอง", value=st.session_state.test_ticker, key='stitch_ticker_input')
                stitch_start_date = dna_cols[1].date_input("วันที่เริ่มต้นจำลอง", value=datetime.now().date() - pd.Timedelta(days=365), key='stitch_date_input')
                
                if st.button("🧬 เริ่มการวิเคราะห์ Stitched DNA แบบเปรียบเทียบ", type="primary", key='stitch_dna_btn'):
                    if not stitched_actions:
                        st.error("ไม่สามารถสร้าง Action Sequence จากข้อมูลที่โหลดได้")
                    else:
                        with st.spinner(f"กำลังจำลองกลยุทธ์สำหรับ {stitch_ticker}..."):
                            sim_data = get_ticker_data(stitch_ticker, str(stitch_start_date), str(datetime.now().date()))
                            if sim_data.empty:
                                st.error("ไม่สามารถดึงข้อมูลสำหรับจำลองได้")
                            else:
                                prices = sim_data['Close'].to_numpy()
                                n_total = len(prices)
                                
                                final_actions_dna = stitched_actions[:n_total]
                                df_dna = run_simulation(prices[:len(final_actions_dna)].tolist(), final_actions_dna)
                                df_max = run_simulation(prices.tolist(), generate_actions_perfect_foresight(prices.tolist()).tolist())
                                df_min = run_simulation(prices.tolist(), generate_actions_rebalance_daily(n_total).tolist())
                                
                                results_dna = {}
                                if not df_dna.empty:
                                    df_dna.index = sim_data.index[:len(df_dna)]
                                    results_dna['Stitched DNA'] = df_dna
                                if not df_max.empty:
                                    df_max.index = sim_data.index[:len(df_max)]
                                    results_dna[Strategy.PERFECT_FORESIGHT] = df_max
                                if not df_min.empty:
                                    df_min.index = sim_data.index[:len(df_min)]
                                    results_dna[Strategy.REBALANCE_DAILY] = df_min
                                    
                                st.subheader("Performance Comparison (Net Profit)")
                                display_comparison_charts(results_dna)
                                
                                st.subheader("สรุปผลลัพธ์สุดท้าย (Final Net Profit)")
                                metric_cols = st.columns(3)
                                final_net_max = results_dna.get(Strategy.PERFECT_FORESIGHT, pd.DataFrame({'net': [0]}))['net'].iloc[-1]
                                final_net_dna = results_dna.get('Stitched DNA', pd.DataFrame({'net': [0]}))['net'].iloc[-1]
                                final_net_min = results_dna.get(Strategy.REBALANCE_DAILY, pd.DataFrame({'net': [0]}))['net'].iloc[-1]
                                
                                metric_cols[0].metric("Max Performance", f"${final_net_max:,.2f}")
                                metric_cols[1].metric("Stitched DNA Strategy", f"${final_net_dna:,.2f}", delta=f"{final_net_dna - final_net_min:,.2f} vs Min", delta_color="normal")
                                metric_cols[2].metric("Min Performance", f"${final_net_min:,.2f}")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการวิเคราะห์ข้อมูล: {e}")
            st.exception(e)
            
def render_manual_seed_tab(config: Dict[str, Any]):
    # ... (ส่วนนี้เหมือนเดิม ไม่ต้องแก้ไข)
    st.header("🌱 Manual Seed Strategy Comparator")
    st.markdown("สร้างและเปรียบเทียบ Action Sequences โดยการตัดส่วนท้าย (`tail`) จาก Seed ที่กำหนด")
    
    with st.container(border=True):
        st.subheader("1. กำหนดค่า Input สำหรับการทดสอบ")
        col1, col2 = st.columns([1, 2])
        with col1:
            asset_list = config.get('assets', ['FFWM'])
            try:
                default_index = asset_list.index(st.session_state.get('manual_ticker_key', st.session_state.test_ticker))
            except (ValueError, KeyError):
                default_index = 0
            manual_ticker = st.selectbox("เลือก Ticker", options=asset_list, index=default_index, key="manual_ticker_key", on_change=on_ticker_change_callback, args=(config,))
        with col2:
            c1, c2 = st.columns(2)
            manual_start_date = c1.date_input("วันที่เริ่มต้น (Start Date)", value=st.session_state.start_date, key="manual_start_compare_tail")
            manual_end_date = c2.date_input("วันที่สิ้นสุด (End Date)", value=st.session_state.end_date, key="manual_end_compare_tail")
        
        if manual_start_date >= manual_end_date:
            st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
        
        st.divider()
        st.write("**กำหนดกลยุทธ์ (Seed/Size/Tail) ที่ต้องการเปรียบเทียบ:**")
        
        for i, line in enumerate(st.session_state.manual_seed_lines):
            cols = st.columns([1, 2, 2, 2])
            cols[0].write(f"**Line {i+1}**")
            line['seed'] = cols[1].number_input("Input Seed", value=line.get('seed', 1), min_value=0, key=f"seed_compare_tail_{i}")
            line['size'] = cols[2].number_input("Size (ขนาด Sequence เริ่มต้น)", value=line.get('size', 60), min_value=1, key=f"size_compare_tail_{i}")
            line['tail'] = cols[3].number_input("Tail (ส่วนท้ายที่จะใช้)", value=line.get('tail', 10), min_value=1, max_value=line.get('size', 60), key=f"tail_compare_tail_{i}")
            
        b_col1, b_col2, _ = st.columns([1,1,4])
        if b_col1.button("➕ เพิ่ม Line เปรียบเทียบ"):
            st.session_state.manual_seed_lines.append({'seed': np.random.randint(1, 10000), 'size': 50, 'tail': 20})
            st.rerun()
        if b_col2.button("➖ ลบ Line สุดท้าย"):
            if len(st.session_state.manual_seed_lines) > 1:
                st.session_state.manual_seed_lines.pop()
                st.rerun()
            else:
                st.warning("ต้องมีอย่างน้อย 1 line")
    
    st.write("---")
    if st.button("📈 เปรียบเทียบประสิทธิภาพ Seeds", type="primary", key="compare_manual_seeds_btn"):
        if manual_start_date >= manual_end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้อง")
            return
            
        with st.spinner("กำลังดึงข้อมูลและจำลองการเทรด..."):
            start_str, end_str = manual_start_date.strftime('%Y-%m-%d'), manual_end_date.strftime('%Y-%m-%d')
            ticker_data = get_ticker_data(manual_ticker, start_str, end_str)
            if ticker_data.empty:
                st.error(f"ไม่พบข้อมูลสำหรับ {manual_ticker} ในช่วงวันที่ที่เลือก")
                return

            prices = ticker_data['Close'].to_numpy()
            num_trading_days = len(prices)
            st.info(f"📊 พบข้อมูลราคา {num_trading_days} วันทำการในช่วงที่เลือก")
            
            results = {}
            max_sim_len = 0
            
            for i, line_info in enumerate(st.session_state.manual_seed_lines):
                input_seed, size_seed, tail_seed = line_info['seed'], line_info['size'], line_info['tail']
                if tail_seed > size_seed:
                    st.error(f"Line {i+1}: Tail ({tail_seed}) ต้องไม่มากกว่า Size ({size_seed})")
                    return
                
                rng_best = np.random.default_rng(input_seed)
                full_actions = rng_best.integers(0, 2, size=size_seed)
                actions_from_tail = full_actions[-tail_seed:].tolist()
                
                sim_len = min(num_trading_days, len(actions_from_tail))
                if sim_len == 0: continue
                
                df_line = run_simulation(prices[:sim_len].tolist(), actions_from_tail[:sim_len])
                if not df_line.empty:
                    df_line.index = ticker_data.index[:sim_len]
                    results[f"Seed {input_seed} (Tail {tail_seed})"] = df_line
                    max_sim_len = max(max_sim_len, sim_len)

            if not results:
                st.error("ไม่สามารถสร้างผลลัพธ์จาก Seed ที่กำหนดได้")
                return
            
            # Add benchmarks
            if max_sim_len > 0:
                prices_for_benchmark = prices[:max_sim_len].tolist()
                df_max = run_simulation(prices_for_benchmark, generate_actions_perfect_foresight(prices_for_benchmark).tolist())
                df_min = run_simulation(prices_for_benchmark, generate_actions_rebalance_daily(max_sim_len).tolist())
                if not df_max.empty:
                    df_max.index = ticker_data.index[:max_sim_len]
                    results[Strategy.PERFECT_FORESIGHT] = df_max
                if not df_min.empty:
                    df_min.index = ticker_data.index[:max_sim_len]
                    results[Strategy.REBALANCE_DAILY] = df_min
            
            st.success("การเปรียบเทียบเสร็จสมบูรณ์!")
            display_comparison_charts(results, chart_title="📊 Performance Comparison (Net Profit)")
            
            st.subheader("สรุปผลลัพธ์สุดท้าย (Final Net Profit)")
            
            sorted_names = [name for name in results.keys() if name not in [Strategy.PERFECT_FORESIGHT, Strategy.REBALANCE_DAILY]]
            display_order = [Strategy.PERFECT_FORESIGHT] + sorted(sorted_names) + [Strategy.REBALANCE_DAILY]
            
            final_results_list = [
                {'name': name, 'net': results[name]['net'].iloc[-1]}
                for name in display_order if name in results and not results[name].empty
            ]
            
            if final_results_list:
                final_metrics_cols = st.columns(len(final_results_list))
                for idx, item in enumerate(final_results_list):
                    final_metrics_cols[idx].metric(item['name'], f"${item['net']:,.2f}")

# ==============================================================================
# 5. Main Application
# ==============================================================================
def main():
    st.set_page_config(page_title="Intelligent Seed Finder", page_icon="🎯", layout="wide")
    st.markdown("### 🎯 Intelligent Seed Finder (Numba Accelerated)")
    st.caption("เครื่องมือทดสอบกลยุทธ์การหา Action Sequence ที่ดีที่สุด โดยเปรียบเทียบระหว่าง Brute-Force และ Heuristic")

    config = load_config()
    initialize_session_state(config)

    tab_list = [
        "⚙️ การตั้งค่า",
        "🚀 Seeded Heuristic", # ! NEW STRATEGY
        "🎲 Best Seed (Brute-Force)", # Renamed original strategy
        "📊 Advanced Analytics",
        "🌱 Forward Rolling Comparator"
    ]
    tabs = st.tabs(tab_list)

    with tabs[0]: render_settings_tab(config)
    with tabs[1]: render_heuristic_test_tab()
    with tabs[2]: render_brute_force_test_tab()
    with tabs[3]: render_analytics_tab()
    with tabs[4]: render_manual_seed_tab(config)

    with st.expander("📖 คำอธิบายกลยุทธ์"):
        st.markdown("""
        ### กลยุทธ์ที่ใช้ในเครื่องมือนี้

        1.  **🚀 Seeded Heuristic (Price-Triggered) - *Recommended***
            - **หลักการ:** เป็นกลยุทธ์ที่ชาญฉลาดซึ่งได้รับแรงบันดาลใจจาก `Perfect Foresight`
            - **วิธีการ:** แทนที่จะสุ่ม Action (0 หรือ 1) แบบมั่วๆ, ระบบจะใช้ `Seed` เพื่อสุ่มหา **"กฎการเทรด"** ที่ดีที่สุดสำหรับช่วงเวลานั้นๆ กฎที่ว่าคือ "จะ Rebalance (ซื้อ/ขาย) ก็ต่อเมื่อราคาเปลี่ยนแปลงขึ้นหรือลงเกินเกณฑ์ (`threshold`) ที่กำหนด"
            - **ข้อดี:** เป็นวิธีที่เข้าใกล้ผลลัพธ์ของ `Perfect Foresight` มากที่สุด เพราะการตัดสินใจอิงตามพฤติกรรมราคาจริง ทำให้มีประสิทธิภาพสูงแต่ยังคงความเร็วและควบคุมผลด้วย Seed ได้

        2.  **🎲 Best Seed (Brute-Force)**
            - **หลักการ:** เป็นวิธีพื้นฐานที่สุด คือการ "เดาสุ่ม"
            - **วิธีการ:** ระบบจะสร้าง `Action Sequence` (ชุดของ 0 และ 1) ขึ้นมาแบบสุ่มตาม `Seed` ที่กำหนด แล้วนำไปคำนวณผลตอบแทน ทำซ้ำหลายพันครั้งเพื่อหา `Seed` ที่ให้ผลลัพธ์ดีที่สุด
            - **ข้อเสีย:** ไม่ได้ใช้ข้อมูลราคาในการตัดสินใจ ทำให้ประสิทธิภาพด้อยกว่าวิธี Heuristic มาก และมีโอกาสน้อยที่จะเข้าใกล้ `Perfect Foresight`

        3.  **Perfect Foresight (Max) - *Benchmark***
            - **หลักการ:** กลยุทธ์อ้างอิงที่มีประสิทธิภาพสูงสุดที่เป็นไปได้ (เพดานบน)
            - **วิธีการ:** คำนวณโดยมองเห็นอนาคตทั้งหมด ทำให้รู้ว่าจะต้อง Rebalance วันไหนถึงจะได้กำไรสูงสุด ไม่สามารถใช้เทรดจริงได้ แต่ใช้เป็น Benchmark เพื่อวัดว่ากลยุทธ์อื่นดีแค่ไหน

        4.  **Rebalance Daily - *Benchmark***
            - **หลักการ:** กลยุทธ์อ้างอิงที่มีประสิทธิภาพต่ำสุด (พื้นล่าง)
            - **วิธีการ:** ทำการ Rebalance ทุกวันโดยไม่มีเงื่อนไข
        """)

if __name__ == "__main__":
    main()
