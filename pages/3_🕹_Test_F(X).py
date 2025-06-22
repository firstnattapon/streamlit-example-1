# Final Integrated Code (v1 + v2 Sequence Strategies)

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
    SLIDING_WINDOW = "Best Seed Sliding Window"
    CHAOTIC_SLIDING_WINDOW = "Chaotic Seed Sliding Window"
    GENETIC_ALGORITHM = "Genetic Algorithm Sliding Window"
    # ! NEW: Added from v2
    ARITHMETIC_SEQUENCE = "Arithmetic Sequence"
    GEOMETRIC_SEQUENCE = "Geometric Sequence"


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
                "ga_population_size": 50, "ga_generations": 20, "ga_master_seed": 42,
                # ! NEW: Added from v2
                "num_samples": 5000, "master_seed": 42
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

    if 'ga_population_size' not in st.session_state:
        st.session_state.ga_population_size = defaults.get('ga_population_size', 50)
    if 'ga_generations' not in st.session_state:
        st.session_state.ga_generations = defaults.get('ga_generations', 20)
    if 'ga_master_seed' not in st.session_state:
        st.session_state.ga_master_seed = defaults.get('ga_master_seed', 42)

    # ! NEW: Parameters for Arithmetic/Geometric Strategies from v2
    if 'num_samples' not in st.session_state:
        st.session_state.num_samples = defaults.get('num_samples', 5000)
    if 'master_seed' not in st.session_state:
        st.session_state.master_seed = defaults.get('master_seed', 42)

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
        # Standardize timezone to avoid issues
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
    - รับ Input เป็น NumPy arrays เพื่อประสิทธิภาพสูงสุด
    - คืนค่าเป็น Tuple ของ NumPy arrays
    """
    n = len(action_array)
    if n == 0 or len(price_array) == 0:
        empty_arr = np.empty(0, dtype=np.float64)
        return (empty_arr, empty_arr, empty_arr, empty_arr, empty_arr, empty_arr)

    # บังคับให้วันแรกต้อง Action=1 (ซื้อ) เสมอ
    action_array_calc = action_array.copy()
    if n > 0: action_array_calc[0] = 1

    # สร้าง arrays เปล่าเพื่อเก็บผลลัพธ์
    amount = np.empty(n, dtype=np.float64)
    buffer = np.zeros(n, dtype=np.float64)
    cash = np.empty(n, dtype=np.float64)
    asset_value = np.empty(n, dtype=np.float64)
    sumusd = np.empty(n, dtype=np.float64)

    # ตั้งค่าเริ่มต้นสำหรับวันแรก
    initial_price = price_array[0]
    amount[0] = fix / initial_price
    cash[0] = fix
    asset_value[0] = amount[0] * initial_price
    sumusd[0] = cash[0] + asset_value[0]

    # คำนวณเส้น Benchmark (Buy and Hold)
    refer = -fix * np.log(initial_price / price_array)

    # Loop หลักที่ Numba จะเร่งความเร็ว
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

@lru_cache(maxsize=16384) # เพิ่มขนาด cache เพื่อรองรับการเรียกที่หลากหลายขึ้น
def calculate_optimized_cached(action_tuple: Tuple[int, ...], price_tuple: Tuple[float, ...], fix: int = 1500) -> Tuple:
    """
    Wrapper function สำหรับเรียกฟังก์ชัน Numba โดยใช้ Cache
    - lru_cache จะจำผลลัพธ์ของการคำนวณสำหรับ input (action, price) ชุดเดิม
    - แปลง tuple เป็น numpy array ก่อนส่งให้ Numba
    """
    action_array = np.asarray(action_tuple, dtype=np.int32)
    price_array = np.asarray(price_tuple, dtype=np.float64)
    return _calculate_simulation_numba(action_array, price_array, fix)

def run_simulation(prices: List[float], actions: List[int], fix: int = 1500) -> pd.DataFrame:
    """
    สร้าง DataFrame ผลลัพธ์จากการจำลองการเทรด
    """
    if not prices or not actions: return pd.DataFrame()
    min_len = min(len(prices), len(actions))
    prices = prices[:min_len]
    actions = actions[:min_len]
    
    # ! ACCELERATED: จุดนี้คือการเรียกใช้ฟังก์ชันที่ถูกเร่งความเร็วแล้ว
    buffer, sumusd, cash, asset_value, amount, refer = calculate_optimized_cached(tuple(actions), tuple(prices), fix)
    
    if len(sumusd) == 0: return pd.DataFrame()

    initial_capital = sumusd[0]
    return pd.DataFrame({
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

# ! NEW: Helper for new sequence strategies, from v2
@njit(cache=True)
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

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
    
    dp = np.zeros(n, dtype=np.float64); path = np.zeros(n, dtype=int); dp[0] = float(fix * 2)
    for i in range(1, n):
        j_indices = np.arange(i)
        profits = fix * ((price_arr[i] / price_arr[j_indices]) - 1)
        current_sumusd = dp[j_indices] + profits
        best_idx = np.argmax(current_sumusd); dp[i] = current_sumusd[best_idx]; path[i] = j_indices[best_idx]
    actions = np.zeros(n, dtype=int); current_day = np.argmax(dp)
    while current_day > 0:
        actions[current_day] = 1; current_day = path[current_day]
    actions[0] = 1
    return actions

# 3.2 Standard Seed Generation (Original Logic from your sample)
def find_best_seed_for_window(prices_window: np.ndarray, num_seeds_to_try: int, max_workers: int) -> Tuple[int, float, np.ndarray]:
    """
    ค้นหา Seed ที่ให้ค่า Net Profit สูงสุดสำหรับ Price Window ที่กำหนด
    - ใช้ ThreadPoolExecutor เพื่อทำงานแบบขนาน (Parallel Processing)
    - คง Logic การสร้าง Action จาก Seed แบบดั้งเดิม (`np.random.default_rng`)
    """
    window_len = len(prices_window)
    if window_len < 2: return 1, 0.0, np.ones(window_len, dtype=int)

    def evaluate_seed_batch(seed_batch: np.ndarray) -> List[Tuple[int, float]]:
        """ประมวลผล Seed ทีละชุด (Batch)"""
        results = []
        prices_tuple = tuple(prices_window)
        for seed in seed_batch:
            # ใช้ PRNG (Pseudo-Random Number Generator) แบบดั้งเดิม
            rng = np.random.default_rng(seed)
            actions_window = rng.integers(0, 2, size=window_len)
            actions_tuple = tuple(actions_window)
            
            # ! ACCELERATED: การคำนวณหลักตรงนี้เร็วขึ้นมากเพราะ Numba
            _, sumusd, _, _, _, refer = calculate_optimized_cached(actions_tuple, prices_tuple)
            
            if len(sumusd) > 0:
                net = sumusd[-1] - refer[-1] - sumusd[0]
            else:
                net = -np.inf # จัดการกรณีที่การคำนวณล้มเหลว
            results.append((seed, net))
        return results

    best_seed_for_window = -1
    max_net_for_window = -np.inf
    
    random_seeds = np.arange(num_seeds_to_try)
    # แบ่งงานเป็น Batch เพื่อลด Overhead ของ Threading
    batch_size = max(1, num_seeds_to_try // (max_workers * 4 if max_workers > 0 else 1))
    seed_batches = [random_seeds[j:j+batch_size] for j in range(0, len(random_seeds), batch_size)]
    
    # ใช้ ThreadPoolExecutor เพื่อรัน evaluate_seed_batch พร้อมกันหลายๆ Core
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_seed_batch, batch) for batch in seed_batches]
        for future in as_completed(futures):
            for seed, final_net in future.result():
                if final_net > max_net_for_window:
                    max_net_for_window = final_net
                    best_seed_for_window = seed

    # สร้าง Action ที่ดีที่สุดขึ้นมาใหม่จาก Seed ที่เจอ เพื่อความแม่นยำ
    if best_seed_for_window >= 0:
        rng_best = np.random.default_rng(best_seed_for_window)
        best_actions = rng_best.integers(0, 2, size=window_len)
    else: # Fallback case
        best_seed_for_window = 1
        best_actions = np.ones(window_len, dtype=int)
        max_net_for_window = 0.0
    
    best_actions[0] = 1 # บังคับ Action วันแรก
    return best_seed_for_window, max_net_for_window, best_actions

def generate_actions_sliding_window(ticker_data: pd.DataFrame, window_size: int, num_seeds_to_try: int, max_workers: int) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    สร้าง Action Sequence ทั้งหมดโดยใช้กลยุทธ์ Sliding Window
    """
    prices = ticker_data['Close'].to_numpy()
    n = len(prices)
    final_actions = np.array([], dtype=int)
    window_details_list = []
    
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="กำลังประมวลผล Sliding Windows...")
    st.write(f"📊 ข้อมูลทั้งหมด: {n} วัน | ขนาด Window: {window_size} วัน | จำนวน Windows: {num_windows}")
    st.write(f"⚡ ใช้ Parallel Processing: {max_workers} workers (Core Calculation เร่งด้วย Numba)")
    st.write("---")

    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        window_len = len(prices_window)
        if window_len == 0: continue
        
        # ค้นหา Seed ที่ดีที่สุดสำหรับ Window ปัจจุบัน
        best_seed, max_net, best_actions = find_best_seed_for_window(prices_window, num_seeds_to_try, max_workers)
        
        final_actions = np.concatenate((final_actions, best_actions))
        
        # บันทึกรายละเอียดของ Window
        start_date_str = ticker_data.index[start_index].strftime('%Y-%m-%d')
        end_date_str = ticker_data.index[end_index-1].strftime('%Y-%m-%d')
        detail = {
            'window_number': i + 1,
            'timeline': f"{start_date_str} ถึง {end_date_str}",
            'best_seed': best_seed,
            'max_net': round(max_net, 2),
            'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2),
            'action_count': int(np.sum(best_actions)),
            'window_size': window_len,
            'action_sequence': best_actions.tolist()
        }
        window_details_list.append(detail)
        progress_bar.progress((i + 1) / num_windows, text=f"ประมวลผล Window {i+1}/{num_windows}")
        
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)

# 3.3 Chaotic Seed Generation
def chaotic_params_to_seed(r: float, x0: float) -> int:
    PRECISION_FACTOR = 1_000_000
    r_int = int((r - 3.0) * PRECISION_FACTOR)
    x0_int = int(x0 * PRECISION_FACTOR)
    return r_int * (PRECISION_FACTOR + 1) + x0_int

def seed_to_chaotic_params(seed: int) -> dict:
    PRECISION_FACTOR = 1_000_000
    r_int = seed // (PRECISION_FACTOR + 1)
    x0_int = seed % (PRECISION_FACTOR + 1)
    r = (r_int / PRECISION_FACTOR) + 3.0
    x0 = x0_int / PRECISION_FACTOR
    r = max(3.57, min(4.0, r))
    x0 = max(0.01, min(0.99, x0))
    return {'r': r, 'x0': x0}

@njit(cache=True)
def _generate_chaotic_actions_numba(length: int, r: float, x0: float) -> np.ndarray:
    actions = np.zeros(length, dtype=np.int32)
    x = x0
    for i in range(length):
        x = r * x * (1.0 - x)
        actions[i] = 1 if x > 0.5 else 0
    if length > 0: actions[0] = 1
    return actions

def generate_actions_from_chaotic_seed(length: int, seed: int) -> np.ndarray:
    if length == 0: return np.array([], dtype=np.int32)
    params = seed_to_chaotic_params(seed)
    return _generate_chaotic_actions_numba(length, params['r'], params['x0'])

def find_best_chaotic_seed(prices_window: np.ndarray, num_seeds_to_try: int, max_workers: int) -> Tuple[int, float, np.ndarray]:
    window_len = len(prices_window)
    if window_len < 2:
        return chaotic_params_to_seed(4.0, 0.1), 0.0, np.ones(window_len, dtype=np.int32)
    
    def evaluate_chaotic_seed_batch(seed_batch: np.ndarray) -> List[Tuple[int, float]]:
        results = []
        prices_tuple = tuple(prices_window)
        for seed in seed_batch:
            actions_window = generate_actions_from_chaotic_seed(window_len, seed)
            # This call is accelerated
            _, sumusd, _, _, _, refer = calculate_optimized_cached(tuple(actions_window), prices_tuple)
            net = sumusd[-1] - refer[-1] - sumusd[0] if len(sumusd) > 0 else -np.inf
            results.append((seed, net))
        return results

    best_seed = -1; max_net = -np.inf
    rng_for_params = np.random.default_rng()
    r_params = rng_for_params.uniform(3.57, 4.0, num_seeds_to_try)
    x0_params = rng_for_params.uniform(0.01, 0.99, num_seeds_to_try)
    seed_list = [chaotic_params_to_seed(r, x0) for r, x0 in zip(r_params, x0_params)]
    
    random_seeds_to_try = np.array(seed_list)
    batch_size = max(1, num_seeds_to_try // (max_workers * 4 if max_workers > 0 else 1))
    seed_batches = [random_seeds_to_try[j:j + batch_size] for j in range(0, len(random_seeds_to_try), batch_size)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_chaotic_seed_batch, batch) for batch in seed_batches]
        for future in as_completed(futures):
            for seed, final_net in future.result():
                if final_net > max_net:
                    max_net = final_net
                    best_seed = seed
    
    if best_seed > 0:
        best_actions = generate_actions_from_chaotic_seed(window_len, best_seed)
    else: # Fallback
        best_seed = chaotic_params_to_seed(4.0, 0.1)
        best_actions = generate_actions_from_chaotic_seed(window_len, best_seed)
        max_net = 0.0

    return best_seed, max_net, best_actions

def generate_actions_sliding_window_chaotic(ticker_data: pd.DataFrame, window_size: int, num_seeds_to_try: int, max_workers: int) -> Tuple[np.ndarray, pd.DataFrame]:
    prices = ticker_data['Close'].to_numpy(); n = len(prices)
    final_actions = np.array([], dtype=np.int32); window_details_list = []
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="กำลังประมวลผล Chaotic Seed Sliding Windows...")
    st.write(f"📊 ข้อมูลทั้งหมด: {n} วัน | ขนาด Window: {window_size} วัน | จำนวน Windows: {num_windows}")
    st.write(f"⚡ ใช้ Parallel Processing: {max_workers} workers"); st.write("---")

    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n); prices_window = prices[start_index:end_index]
        window_len = len(prices_window)
        if window_len == 0: continue
        
        best_seed, max_net, best_actions = find_best_chaotic_seed(prices_window, num_seeds_to_try, max_workers)
        final_actions = np.concatenate((final_actions, best_actions))
        
        start_date_str = ticker_data.index[start_index].strftime('%Y-%m-%d'); end_date_str = ticker_data.index[end_index-1].strftime('%Y-%m-%d')
        params = seed_to_chaotic_params(best_seed)
        detail = {
            'window_number': i + 1, 'timeline': f"{start_date_str} ถึง {end_date_str}", 'best_seed': best_seed,
            'r_param': round(params['r'], 6), 'x0_param': round(params['x0'], 6), 'max_net': round(max_net, 2),
            'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2),
            'action_count': int(np.sum(best_actions)), 'window_size': window_len, 'action_sequence': best_actions.tolist()
        }
        window_details_list.append(detail)
        progress_bar.progress((i + 1) / num_windows, text=f"ประมวลผล Chaotic Window {i+1}/{num_windows}")
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)


# 3.4 Genetic Algorithm Generation
def find_best_solution_ga(prices_window: np.ndarray, population_size: int, generations: int, seed: int, mutation_rate: float = 0.01) -> Tuple[float, np.ndarray]:
    window_len = len(prices_window)
    if window_len < 2: return 0.0, np.ones(window_len, dtype=np.int32)
    
    rng = np.random.default_rng(seed)
    population = rng.integers(0, 2, size=(population_size, window_len), dtype=np.int32)
    population[:, 0] = 1 # First action must be buy
    
    prices_tuple = tuple(prices_window)

    for _ in range(generations):
        # Evaluation
        fitness_scores = []
        for chromosome in population:
            # This call is accelerated
            _, sumusd, _, _, _, refer = calculate_optimized_cached(tuple(chromosome), prices_tuple)
            net = sumusd[-1] - refer[-1] - sumusd[0] if len(sumusd) > 0 else -np.inf
            fitness_scores.append(net)
        
        fitness_scores = np.array(fitness_scores)
        
        # Selection
        num_parents = population_size // 2
        parent_indices = np.argsort(fitness_scores)[-num_parents:]
        parents = population[parent_indices]
        
        # Crossover
        num_offspring = population_size - num_parents
        offspring = np.empty((num_offspring, window_len), dtype=np.int32)
        for k in range(num_offspring):
            parent1_idx, parent2_idx = rng.choice(num_parents, size=2, replace=False)
            crossover_point = rng.integers(1, window_len) # Ensure crossover happens after day 0
            offspring[k, :crossover_point] = parents[parent1_idx, :crossover_point]
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        
        # Mutation
        mutation_mask = rng.random((num_offspring, window_len)) < mutation_rate
        offspring[mutation_mask] = 1 - offspring[mutation_mask]
        offspring[:, 0] = 1 # Ensure first day remains buy
        
        # New population
        population[num_parents:] = offspring

    # Final evaluation to find the best in the last generation
    final_fitness = []
    for chromosome in population:
        # This call is accelerated
        _, sumusd, _, _, _, refer = calculate_optimized_cached(tuple(chromosome), prices_tuple)
        net = sumusd[-1] - refer[-1] - sumusd[0] if len(sumusd) > 0 else -np.inf
        final_fitness.append(net)
        
    best_idx = np.argmax(final_fitness)
    return final_fitness[best_idx], population[best_idx]

def generate_actions_sliding_window_ga(ticker_data: pd.DataFrame, window_size: int, population_size: int, generations: int, master_seed: int) -> Tuple[np.ndarray, pd.DataFrame]:
    prices = ticker_data['Close'].to_numpy(); n = len(prices)
    final_actions = np.array([], dtype=np.int32); window_details_list = []
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="กำลังประมวลผล Genetic Algorithm Sliding Windows...")
    st.write(f"🧬 Evolving Solutions using GA (Master Seed: {master_seed})")
    st.write(f"📊 ข้อมูล: {n} วัน | Window: {window_size} วัน | ประชากร: {population_size} | รุ่น: {generations}"); st.write("---")

    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        window_len = len(prices_window)
        if window_len < 2: continue
        
        window_seed = master_seed + i # Use a deterministic seed for each window
        max_net, best_actions = find_best_solution_ga(prices_window, population_size, generations, seed=window_seed)
        
        final_actions = np.concatenate((final_actions, best_actions))
        
        start_date_str = ticker_data.index[start_index].strftime('%Y-%m-%d'); end_date_str = ticker_data.index[end_index-1].strftime('%Y-%m-%d')
        detail = {
            'window_number': i + 1, 'timeline': f"{start_date_str} ถึง {end_date_str}", 'max_net': round(max_net, 2),
            'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2),
            'action_count': int(np.sum(best_actions)), 'window_size': window_len,
            'action_sequence': best_actions.tolist(), 'window_seed': window_seed
        }
        window_details_list.append(detail)
        progress_bar.progress((i + 1) / num_windows, text=f"Evolving Window {i+1}/{num_windows}")
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)

# ! NEW: 3.5 Arithmetic Sequence Strategy (from v2)
def generate_actions_sliding_window_arithmetic(ticker_data: pd.DataFrame, window_size: int, num_samples: int, master_seed: int) -> Tuple[np.ndarray, pd.DataFrame]:
    prices = ticker_data['Close'].to_numpy(); n = len(prices)
    final_actions = np.array([], dtype=np.int32); window_details_list = []
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="ประมวลผล Arithmetic Sequence Search...")
    st.write(f"📈 ค้นหาพารามิเตอร์ (a1, d) ที่ดีที่สุดในแต่ละ Window | {num_samples} samples/window")
    
    indices = np.arange(window_size)

    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n); prices_window = prices[start_index:end_index]
        window_len = len(prices_window)
        if window_len < 2: continue

        rng = np.random.default_rng(master_seed + i)
        best_net, best_actions, best_params = -np.inf, np.ones(window_len, dtype=np.int32), {}
        
        # Random Search for best (a1, d)
        for _ in range(num_samples):
            a1 = rng.uniform(-5, 5) # Sample first term
            d = rng.uniform(-1, 1)  # Sample common difference
            
            latent_sequence = a1 + indices[:window_len] * d
            actions = (_sigmoid(latent_sequence) > 0.5).astype(np.int32)
            actions[0] = 1

            _, sumusd, _, _, _, refer = calculate_optimized_cached(tuple(actions), tuple(prices_window))
            net = sumusd[-1] - refer[-1] - sumusd[0] if len(sumusd) > 0 else -np.inf

            if net > best_net:
                best_net, best_actions, best_params = net, actions, {'a1': a1, 'd': d}

        final_actions = np.concatenate((final_actions, best_actions))
        
        start_date_str = ticker_data.index[start_index].strftime('%Y-%m-%d'); end_date_str = ticker_data.index[end_index-1].strftime('%Y-%m-%d')
        window_details_list.append({'window_number': i + 1, 'timeline': f"{start_date_str} ถึง {end_date_str}",
                                    'max_net': round(best_net, 2), 'best_a1': round(best_params.get('a1', 0), 4),
                                    'best_d': round(best_params.get('d', 0), 4), 'action_count': int(np.sum(best_actions)),
                                    'window_size': window_len, 'window_seed': master_seed + i})
        progress_bar.progress((i + 1) / num_windows, text=f"Optimizing Window {i+1}/{num_windows}")
        
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)

# ! NEW: 3.6 Geometric Sequence Strategy (from v2)
def generate_actions_sliding_window_geometric(ticker_data: pd.DataFrame, window_size: int, num_samples: int, master_seed: int) -> Tuple[np.ndarray, pd.DataFrame]:
    prices = ticker_data['Close'].to_numpy(); n = len(prices)
    final_actions = np.array([], dtype=np.int32); window_details_list = []
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="ประมวลผล Geometric Sequence Search...")
    st.write(f"📉 ค้นหาพารามิเตอร์ (a1, r) ที่ดีที่สุดในแต่ละ Window | {num_samples} samples/window")

    indices = np.arange(window_size)

    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n); prices_window = prices[start_index:end_index]
        window_len = len(prices_window)
        if window_len < 2: continue

        rng = np.random.default_rng(master_seed + i)
        best_net, best_actions, best_params = -np.inf, np.ones(window_len, dtype=np.int32), {}
        
        # Random Search for best (a1, r)
        for _ in range(num_samples):
            a1 = rng.uniform(-5, 5)   # Sample first term
            r = rng.uniform(0.8, 1.2) # Sample common ratio
            
            latent_sequence = a1 * (r ** indices[:window_len])
            actions = (_sigmoid(latent_sequence) > 0.5).astype(np.int32)
            actions[0] = 1

            _, sumusd, _, _, _, refer = calculate_optimized_cached(tuple(actions), tuple(prices_window))
            net = sumusd[-1] - refer[-1] - sumusd[0] if len(sumusd) > 0 else -np.inf

            if net > best_net:
                best_net, best_actions, best_params = net, actions, {'a1': a1, 'r': r}

        final_actions = np.concatenate((final_actions, best_actions))
        
        start_date_str = ticker_data.index[start_index].strftime('%Y-%m-%d'); end_date_str = ticker_data.index[end_index-1].strftime('%Y-%m-%d')
        window_details_list.append({'window_number': i + 1, 'timeline': f"{start_date_str} ถึง {end_date_str}",
                                    'max_net': round(best_net, 2), 'best_a1': round(best_params.get('a1', 0), 4),
                                    'best_r': round(best_params.get('r', 0), 4), 'action_count': int(np.sum(best_actions)),
                                    'window_size': window_len, 'window_seed': master_seed + i})
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
    st.subheader("พารามิเตอร์ทั่วไป")
    st.session_state.window_size = st.number_input("ขนาด Window (วัน)", min_value=2, value=st.session_state.window_size)
    st.session_state.max_workers = st.number_input("จำนวน Workers (CPU Cores)", min_value=1, max_value=16, value=st.session_state.max_workers, help="ใช้สำหรับกลยุทธ์ Random และ Chaotic Search")
    
    st.subheader("พารามิเตอร์สำหรับกลยุทธ์แบบ Search")
    c1, c2, c3 = st.columns(3)
    st.session_state.num_seeds = c1.number_input("จำนวน Seeds (Random/Chaotic)", min_value=100, value=st.session_state.num_seeds, format="%d")
    st.session_state.num_samples = c2.number_input("จำนวน Samples (Sequence)", min_value=100, value=st.session_state.num_samples, format="%d", help="จำนวนการสุ่ม Parameter (a1, d, r) ในแต่ละ Window")
    st.session_state.master_seed = c3.number_input("Master Seed (Sequence)", value=st.session_state.master_seed, format="%d", help="Seed หลักสำหรับกลยุทธ์ Sequence เพื่อผลลัพธ์ที่ทำซ้ำได้")

    st.subheader("พารามิเตอร์สำหรับ Genetic Algorithm")
    ga_c1, ga_c2, ga_c3 = st.columns(3)
    st.session_state.ga_population_size = ga_c1.number_input("ขนาดประชากร (Population Size)", min_value=10, value=st.session_state.ga_population_size)
    st.session_state.ga_generations = ga_c2.number_input("จำนวนรุ่น (Generations)", min_value=5, value=st.session_state.ga_generations)
    st.session_state.ga_master_seed = ga_c3.number_input("Master Seed for GA", value=st.session_state.ga_master_seed, format="%d", help="เปลี่ยนค่านี้เพื่อเปลี่ยนผลลัพธ์ของ GA ทั้งหมด แต่การใช้ค่าเดิมจะให้ผลลัพธ์เดิมเสมอ")

def display_comparison_charts(results: Dict[str, pd.DataFrame], chart_title: str = '📊 เปรียบเทียบกำไรสุทธิ (Net Profit)'):
    """แสดงผลกราฟเปรียบเทียบผลลัพธ์จากหลายๆ กลยุทธ์"""
    if not results:
        st.warning("ไม่มีข้อมูลสำหรับสร้างกราฟเปรียบเทียบ")
        return

    valid_dfs = {name: df for name, df in results.items() if not df.empty and 'net' in df.columns}
    if not valid_dfs:
        st.warning("ไม่มีข้อมูล 'net' สำหรับสร้างกราฟ")
        return
        
    try:
        longest_index = max((df.index for df in valid_dfs.values()), key=len, default=None)
    except ValueError:
        longest_index = None

    if longest_index is None:
        st.warning("ไม่มีข้อมูลสำหรับสร้างกราฟเปรียบเทียบ")
        return

    chart_data = pd.DataFrame(index=longest_index)
    for name, df in valid_dfs.items():
        chart_data[name] = df['net'].reindex(longest_index).ffill()
    
    st.write(chart_title)
    st.line_chart(chart_data)

def render_test_tab():
    """แสดงผล UI สำหรับ Tab Best Seed (Random)"""
    st.write("---")
    if st.button("🚀 เริ่มทดสอบ Best Seed (Accelerated)", type="primary"):
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
            
        prices = ticker_data['Close'].to_numpy()
        num_days = len(prices)

        with st.spinner("กำลังคำนวณกลยุทธ์ต่างๆ (Core calculation is Numba-accelerated)..."):
            st.write("🔍 **เริ่มต้นการค้นหา Best Seed ด้วย Sliding Window**")
            actions_sliding, df_windows = generate_actions_sliding_window(
                ticker_data, st.session_state.window_size, st.session_state.num_seeds, st.session_state.max_workers
            )
            actions_min = generate_actions_rebalance_daily(num_days)
            actions_max = generate_actions_perfect_foresight(prices.tolist())
            
            results = {}
            strategy_map = {
                Strategy.SLIDING_WINDOW: actions_sliding.tolist(),
                Strategy.REBALANCE_DAILY: actions_min.tolist(),
                Strategy.PERFECT_FORESIGHT: actions_max.tolist()
            }
            
            for strategy_name, actions in strategy_map.items():
                df = run_simulation(prices.tolist(), actions)
                if not df.empty:
                    df.index = ticker_data.index[:len(df)]
                results[strategy_name] = df
        
        st.success("การทดสอบเสร็จสมบูรณ์!")
        st.write("---")
        display_comparison_charts(results)
        
        st.write("📈 **สรุปผลการค้นหา Best Seed**")
        if not df_windows.empty:
            total_actions = df_windows['action_count'].sum()
            total_net = df_windows['max_net'].sum()
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Windows", df_windows.shape[0])
            col2.metric("Total Actions", f"{total_actions}/{num_days}")
            col3.metric("Total Net (Sum)", f"${total_net:,.2f}")
        
            st.dataframe(df_windows[['window_number', 'timeline', 'best_seed', 'max_net', 'price_change_pct', 'action_count']], use_container_width=True)
            csv = df_windows.to_csv(index=False)
            st.download_button(label="📥 ดาวน์โหลด Window Details (CSV)", data=csv, file_name=f'best_seed_{ticker}_{st.session_state.window_size}w.csv', mime='text/csv')

def render_chaotic_test_tab():
    """แสดงผล UI สำหรับ Tab Best Seed (Chaotic)"""
    st.write("---")
    st.markdown("### 🌀 ทดสอบ Best Seed ด้วย Chaotic Generator (Logistic Map)")
    st.info("กลยุทธ์นี้จะค้นหาค่าพารามิเตอร์ `r` และ `x0` ของ Logistic Map ที่ให้ผลตอบแทนดีที่สุดในแต่ละ Window แทนการใช้ Seed แบบสุ่มทั่วไป")
    if st.button("🚀 เริ่มทดสอบ Best Chaotic Seed", type="primary", key="chaotic_test_button"):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้องในแท็บ 'การตั้งค่า'"); return
        
        ticker = st.session_state.test_ticker; start_date_str = st.session_state.start_date.strftime('%Y-%m-%d'); end_date_str = st.session_state.end_date.strftime('%Y-%m-%d')
        st.info(f"กำลังดึงข้อมูลสำหรับ **{ticker}** | {start_date_str} ถึง {end_date_str}")
        ticker_data = get_ticker_data(ticker, start_date_str, end_date_str)
        if ticker_data.empty: st.error("ไม่พบข้อมูลสำหรับ Ticker และช่วงวันที่ที่เลือก"); return
        
        prices = ticker_data['Close'].to_numpy(); num_days = len(prices)
        with st.spinner("กำลังคำนวณกลยุทธ์ต่างๆ (Chaotic Search)..."):
            actions_chaotic, df_windows = generate_actions_sliding_window_chaotic(ticker_data, st.session_state.window_size, st.session_state.num_seeds, st.session_state.max_workers)
            actions_min = generate_actions_rebalance_daily(num_days)
            actions_max = generate_actions_perfect_foresight(prices.tolist())
            
            results = {}
            strategy_map = {
                Strategy.CHAOTIC_SLIDING_WINDOW: actions_chaotic.tolist(),
                Strategy.REBALANCE_DAILY: actions_min.tolist(),
                Strategy.PERFECT_FORESIGHT: actions_max.tolist()
            }
            for strategy_name, actions in strategy_map.items():
                df = run_simulation(prices.tolist(), actions)
                if not df.empty: df.index = ticker_data.index[:len(df)]
                results[strategy_name] = df
        
        st.success("การทดสอบเสร็จสมบูรณ์!")
        st.write("---")
        display_comparison_charts(results)
        
        st.write("📈 **สรุปผลการค้นหา Best Chaotic Seed**")
        if not df_windows.empty:
            total_net = df_windows['max_net'].sum()
            action_pct = np.mean(actions_chaotic) * 100 if len(actions_chaotic) > 0 else 0
            best_window_row = df_windows.loc[df_windows['max_net'].idxmax()]
            best_r, best_x0 = best_window_row['r_param'], best_window_row['x0_param']
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Net (Sum)", f"${total_net:,.2f}")
            col2.metric("สมการที่ดีที่สุด (r, x)", f"({best_r:.4f}, {best_x0:.4f})", help=f"จาก Window ที่มี Net Profit สูงสุด: ${best_window_row['max_net']:.2f}")
            col3.metric("สัดส่วน Action=1", f"{action_pct:.2f}%")
            col4.metric("Total Windows", df_windows.shape[0])
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Net (Sum)", "$0.00")
            col2.metric("สมการที่ดีที่สุด (r, x)", "N/A")
            col3.metric("สัดส่วน Action=1", "0.00%")
            col4.metric("Total Windows", "0")
            
        st.dataframe(df_windows[['window_number', 'timeline', 'best_seed', 'r_param', 'x0_param', 'max_net', 'price_change_pct', 'action_count']], use_container_width=True)
        csv = df_windows.to_csv(index=False)
        st.download_button(label="📥 ดาวน์โหลด Chaotic Window Details (CSV)", data=csv, file_name=f'best_chaotic_seed_{ticker}_{st.session_state.window_size}w.csv', mime='text/csv')

def render_ga_test_tab():
    """แสดงผล UI สำหรับ Tab Best Seed (Genetic Algorithm)"""
    st.write("---")
    st.markdown("### 🧬 ทดสอบด้วย Genetic Algorithm Search")
    st.info("กลยุทธ์นี้ใช้วิวัฒนาการเชิงคำนวณ (GA) เพื่อ 'พัฒนา' Action Sequence ที่ดีที่สุดในแต่ละ Window โดยสามารถควบคุมผลลัพธ์ได้ด้วย Master Seed")
    if st.button("🚀 เริ่มทดสอบ Best Genetic Algorithm", type="primary", key="ga_test_button"):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้องในแท็บ 'การตั้งค่า'"); return
            
        ticker = st.session_state.test_ticker; start_date_str = st.session_state.start_date.strftime('%Y-%m-%d'); end_date_str = st.session_state.end_date.strftime('%Y-%m-%d')
        st.info(f"กำลังดึงข้อมูลสำหรับ **{ticker}** | {start_date_str} ถึง {end_date_str}")
        ticker_data = get_ticker_data(ticker, start_date_str, end_date_str)
        if ticker_data.empty: st.error("ไม่พบข้อมูลสำหรับ Ticker และช่วงวันที่ที่เลือก"); return
        
        prices = ticker_data['Close'].to_numpy(); num_days = len(prices)
        with st.spinner("กำลังคำนวณกลยุทธ์ต่างๆ (GA Search)..."):
            actions_ga, df_windows = generate_actions_sliding_window_ga(
                ticker_data, st.session_state.window_size, st.session_state.ga_population_size,
                st.session_state.ga_generations, master_seed=st.session_state.ga_master_seed)
            actions_min = generate_actions_rebalance_daily(num_days)
            actions_max = generate_actions_perfect_foresight(prices.tolist())
            
            results = {}; strategy_map = {
                Strategy.GENETIC_ALGORITHM: actions_ga.tolist(),
                Strategy.REBALANCE_DAILY: actions_min.tolist(),
                Strategy.PERFECT_FORESIGHT: actions_max.tolist()
            }
            for strategy_name, actions in strategy_map.items():
                df = run_simulation(prices.tolist(), actions)
                if not df.empty: df.index = ticker_data.index[:len(df)]
                results[strategy_name] = df
        
        st.success("การทดสอบเสร็จสมบูรณ์!")
        st.write("---")
        display_comparison_charts(results)
        
        st.write("📈 **สรุปผลการค้นหา Best GA Sequence**")
        if not df_windows.empty:
            total_net = df_windows['max_net'].sum()
            total_actions = df_windows['action_count'].sum()
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Net (Sum)", f"${total_net:,.2f}")
            col2.metric("Total Actions", f"{total_actions}/{num_days}")
            col3.metric("Total Windows", df_windows.shape[0])
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Net (Sum)", "$0.00")
            col2.metric("Total Actions", "0/0")
            col3.metric("Total Windows", "0")
            
        st.dataframe(df_windows[['window_number', 'timeline', 'window_seed', 'max_net', 'price_change_pct', 'action_count', 'action_sequence']], use_container_width=True)
        csv = df_windows.to_csv(index=False)
        st.download_button(label="📥 ดาวน์โหลด GA Window Details (CSV)", data=csv, file_name=f'best_ga_{ticker}_{st.session_state.window_size}w.csv', mime='text/csv')

# ! NEW: UI Tab for Arithmetic Sequence (from v2)
def render_arithmetic_tab():
    """แสดงผล UI สำหรับ Tab Arithmetic Sequence"""
    st.write("---")
    st.markdown(f"### 📈 ทดสอบด้วย {Strategy.ARITHMETIC_SEQUENCE}")
    st.info("กลยุทธ์นี้จะค้นหาพารามิเตอร์ของ **ลำดับเลขคณิต (`a1`, `d`)** ที่สร้าง Action ที่ดีที่สุดในแต่ละ Window")

    if st.button(f"🚀 เริ่มทดสอบ {Strategy.ARITHMETIC_SEQUENCE}", type="primary", key="arithmetic_test_button"):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้องในแท็บ 'การตั้งค่า'"); return
            
        ticker = st.session_state.test_ticker; start_date_str = st.session_state.start_date.strftime('%Y-%m-%d'); end_date_str = st.session_state.end_date.strftime('%Y-%m-%d')
        st.info(f"กำลังดึงข้อมูลสำหรับ **{ticker}** | {start_date_str} ถึง {end_date_str}")
        ticker_data = get_ticker_data(ticker, start_date_str, end_date_str)
        if ticker_data.empty: st.error("ไม่พบข้อมูลสำหรับ Ticker และช่วงวันที่ที่เลือก"); return
        
        prices = ticker_data['Close'].to_numpy(); num_days = len(prices)
        with st.spinner("กำลังคำนวณกลยุทธ์ต่างๆ (Arithmetic Search)..."):
            actions, df_windows = generate_actions_sliding_window_arithmetic(
                ticker_data, st.session_state.window_size, st.session_state.num_samples, st.session_state.master_seed
            )
            actions_min = generate_actions_rebalance_daily(num_days)
            actions_max = generate_actions_perfect_foresight(prices.tolist())
            
            results = {}; strategy_map = {
                Strategy.ARITHMETIC_SEQUENCE: actions.tolist(),
                Strategy.REBALANCE_DAILY: actions_min.tolist(),
                Strategy.PERFECT_FORESIGHT: actions_max.tolist()
            }
            for strategy_name, actions in strategy_map.items():
                df = run_simulation(prices.tolist(), actions)
                if not df.empty: df.index = ticker_data.index[:len(df)]
                results[strategy_name] = df
        
        st.success("การทดสอบเสร็จสมบูรณ์!")
        st.write("---")
        display_comparison_charts(results)
        
        st.write(f"📈 **สรุปผลการค้นหาด้วย {Strategy.ARITHMETIC_SEQUENCE}**")
        st.dataframe(df_windows[['window_number', 'timeline', 'max_net', 'best_a1', 'best_d', 'action_count']], use_container_width=True)
        csv = df_windows.to_csv(index=False)
        st.download_button(label="📥 ดาวน์โหลด Arithmetic Details (CSV)", data=csv, file_name=f'arithmetic_seq_{ticker}.csv', mime='text/csv')

# ! NEW: UI Tab for Geometric Sequence (from v2)
def render_geometric_tab():
    """แสดงผล UI สำหรับ Tab Geometric Sequence"""
    st.write("---")
    st.markdown(f"### 📉 ทดสอบด้วย {Strategy.GEOMETRIC_SEQUENCE}")
    st.info("กลยุทธ์นี้จะค้นหาพารามิเตอร์ของ **ลำดับเรขาคณิต (`a1`, `r`)** ที่สร้าง Action ที่ดีที่สุดในแต่ละ Window")

    if st.button(f"🚀 เริ่มทดสอบ {Strategy.GEOMETRIC_SEQUENCE}", type="primary", key="geometric_test_button"):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้องในแท็บ 'การตั้งค่า'"); return
            
        ticker = st.session_state.test_ticker; start_date_str = st.session_state.start_date.strftime('%Y-%m-%d'); end_date_str = st.session_state.end_date.strftime('%Y-%m-%d')
        st.info(f"กำลังดึงข้อมูลสำหรับ **{ticker}** | {start_date_str} ถึง {end_date_str}")
        ticker_data = get_ticker_data(ticker, start_date_str, end_date_str)
        if ticker_data.empty: st.error("ไม่พบข้อมูลสำหรับ Ticker และช่วงวันที่ที่เลือก"); return
        
        prices = ticker_data['Close'].to_numpy(); num_days = len(prices)
        with st.spinner("กำลังคำนวณกลยุทธ์ต่างๆ (Geometric Search)..."):
            actions, df_windows = generate_actions_sliding_window_geometric(
                ticker_data, st.session_state.window_size, st.session_state.num_samples, st.session_state.master_seed
            )
            actions_min = generate_actions_rebalance_daily(num_days)
            actions_max = generate_actions_perfect_foresight(prices.tolist())
            
            results = {}; strategy_map = {
                Strategy.GEOMETRIC_SEQUENCE: actions.tolist(),
                Strategy.REBALANCE_DAILY: actions_min.tolist(),
                Strategy.PERFECT_FORESIGHT: actions_max.tolist()
            }
            for strategy_name, actions in strategy_map.items():
                df = run_simulation(prices.tolist(), actions)
                if not df.empty: df.index = ticker_data.index[:len(df)]
                results[strategy_name] = df
        
        st.success("การทดสอบเสร็จสมบูรณ์!")
        st.write("---")
        display_comparison_charts(results)
        
        st.write(f"📈 **สรุปผลการค้นหาด้วย {Strategy.GEOMETRIC_SEQUENCE}**")
        st.dataframe(df_windows[['window_number', 'timeline', 'max_net', 'best_a1', 'best_r', 'action_count']], use_container_width=True)
        csv = df_windows.to_csv(index=False)
        st.download_button(label="📥 ดาวน์โหลด Geometric Details (CSV)", data=csv, file_name=f'geometric_seq_{ticker}.csv', mime='text/csv')

def render_analytics_tab():
    """แสดงผล UI สำหรับ Tab Advanced Analytics"""
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
            required_cols = ['window_number', 'timeline', 'max_net']
            # Allow for optional action_sequence
            if 'action_sequence' not in df_to_analyze.columns:
                st.info("คอลัมน์ 'action_sequence' ไม่พบในไฟล์, ฟีเจอร์ 'Stitched DNA Analysis' จะถูกปิดใช้งาน")
                df_to_analyze['action_sequence'] = [[] for _ in range(len(df_to_analyze))]


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
                
                if 'action_sequence' not in df.columns:
                     st.warning("ไม่สามารถทำการวิเคราะห์ Stitched DNA ได้เนื่องจากไม่มีคอลัมน์ 'action_sequence' ในข้อมูลที่อัปโหลด")
                else:
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
    """แสดงผล UI สำหรับ Tab Manual Seed Comparator"""
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
    """
    ฟังก์ชันหลักในการรัน Streamlit Application
    """
    st.set_page_config(page_title="Best Seed Sliding Window", page_icon="🎯", layout="wide")
    st.markdown("### 🎯 Best Seed Sliding Window Tester (Multi-Strategy & Numba Accelerated)")
    st.caption("เครื่องมือทดสอบการหา Best Seed และ Sequence ที่ดีที่สุด (Core Calculation เร่งความเร็วด้วย Numba)")

    # โหลดการตั้งค่าและเตรียม Session State
    config = load_config()
    initialize_session_state(config)

    # สร้าง Tabs
    tab_list = [
        "⚙️ การตั้งค่า",
        "🚀 Best Seed (Random)",
        "🌀 Best Seed (Chaotic)",
        "🧬 Best Seed (Genetic Algo)",
        # ! NEW: Added from v2
        "📈 Arithmetic Seq",
        "📉 Geometric Seq",
        "📊 Advanced Analytics",
        "🌱 Forward Rolling Comparator"
    ]
    tabs = st.tabs(tab_list)

    with tabs[0]: render_settings_tab(config)
    with tabs[1]: render_test_tab()
    with tabs[2]: render_chaotic_test_tab()
    with tabs[3]: render_ga_test_tab()
    # ! NEW: Added from v2
    with tabs[4]: render_arithmetic_tab()
    with tabs[5]: render_geometric_tab()
    with tabs[6]: render_analytics_tab()
    with tabs[7]: render_manual_seed_tab(config)

    with st.expander("📖 คำอธิบายวิธีการทำงานและแนวคิด (v.Optimized + Sequence Models)"):
        st.markdown("""
        ### หลักการทำงานของเวอร์ชันนี้:

        **เป้าหมาย: คง Logic เดิม 100% แต่ต้องการความเร็วสูงสุด และเพิ่มโมเดลใหม่ๆ**

        1.  **Logic ดั้งเดิม (Random/Chaotic/GA)**:
            - ใช้ `ThreadPoolExecutor` ในการกระจายงานเพื่อค้นหา Best Seed พร้อมๆ กันหลาย CPU Core (สำหรับ Random/Chaotic)
            - ใช้ `np.random.default_rng(seed)` หรือ `Logistic Map` หรือ `Genetic Algorithm` ในการสร้าง `action sequence` จาก `seed` ที่กำหนด
            - **ดังนั้น Best Seed ที่หาได้จะตรงกับโค้ดต้นฉบับทุกประการ**

        2.  **✨ โมเดลใหม่ (Sequence-based)**:
            - **📈 Arithmetic Sequence**: สร้าง Action จากสมการลำดับเลขคณิต `Action(t) = sigmoid(a1 + t * d)` โดย `t` คือลำดับวันใน Window ระบบจะสุ่มหาค่า `a1` และ `d` ที่ดีที่สุด
            - **📉 Geometric Sequence**: สร้าง Action จากสมการลำดับเรขาคณิต `Action(t) = sigmoid(a1 * r^t)` ระบบจะสุ่มหาค่า `a1` และ `r` ที่ดีที่สุด
            - โมเดลเหล่านี้ช่วยสร้าง Action ที่มีรูปแบบ (Pattern) มากกว่าการสุ่มแบบอิสระ

        3.  **⚡ Core Acceleration**:
            - ฟังก์ชันที่ทำงานช้าที่สุดคือ `_calculate_simulation_numba` ซึ่งเป็น Loop คำนวณผลการเทรดที่ต้องรันเป็นแสนๆ รอบ
            - ฟังก์ชันนี้ถูกเร่งความเร็วด้วย **Numba (`@njit`)** ซึ่งเป็น Just-In-Time Compiler ที่จะแปลงโค้ด Python ในส่วนนี้ให้เป็น Machine Code ที่ทำงานเร็วเทียบเท่าภาษา C
            - มีการใช้ `cache=True` ทำให้ Numba คอมไพล์โค้ดแค่ครั้งแรกเท่านั้น การรันครั้งถัดไปจะเร็วทันที
        
        4.  **ผลลัพธ์**:
            - ได้ทั้ง **ความถูกต้องของตรรกะ (Correctness)** เหมือนเดิม
            - และ **ความเร็วที่เพิ่มขึ้นอย่างมหาศาล (Performance)** จากการเร่งความเร็วเฉพาะส่วนที่เป็นคอขวด (Bottleneck) จริงๆ
        """)

if __name__ == "__main__":
    main()
