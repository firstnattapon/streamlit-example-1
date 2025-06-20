# v2 (แก้ไขแล้ว) Perfect Foresight (Max) ที่ทำงานถูกต้อง

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
from numba import njit #! NUMBA: Import Numba's Just-In-Time compiler

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================

class Strategy:
    REBALANCE_DAILY = "Rebalance Daily"
    PERFECT_FORESIGHT = "Perfect Foresight (Max)"
    SLIDING_WINDOW = "Best Seed Sliding Window"
    CHAOTIC_SLIDING_WINDOW = "Chaotic Seed Sliding Window"
    GENETIC_ALGORITHM = "Genetic Algorithm Sliding Window"

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
            "assets": ["FFWM", "NEGG", "RIVN", "BTC-USD"],
            "default_settings": {
                "selected_ticker": "FFWM", "start_date": "2024-01-01", "window_size": 30,
                "num_seeds": 10000, "max_workers": 8,
                "ga_population_size": 50, "ga_generations": 20,
                "ga_master_seed": 42
            },
            "manual_seed_by_asset": {
                "default": [{'seed': 999, 'size': 50, 'tail': 15}],
                "FFWM": [{'seed': 1234, 'size': 60, 'tail': 30}, {'seed': 7777, 'size': 30, 'tail': 10}]
            }
        }

def on_ticker_change_callback(config: Dict[str, Any]):
    """
    Callback ที่จะถูกเรียกเมื่อ Ticker ใน Tab Manual Seed เปลี่ยน
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

    if 'df_for_analysis' not in st.session_state:
        st.session_state.df_for_analysis = None

    if 'manual_seed_lines' not in st.session_state:
        initial_ticker = st.session_state.get('test_ticker', defaults.get('selected_ticker', 'FFWM'))
        presets_by_asset = config.get("manual_seed_by_asset", {})
        default_presets = presets_by_asset.get("default", [{'seed': 999, 'size': 50, 'tail': 15}])
        st.session_state.manual_seed_lines = presets_by_asset.get(initial_ticker, default_presets)


# ==============================================================================
# 2. Core Calculation & Data Functions
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

@njit(cache=True)
def _calculate_simulation_numba(action_array: np.ndarray, price_array: np.ndarray, fix: int = 1500) -> Tuple:
    n = len(action_array)
    if n == 0 or len(price_array) == 0:
        empty_arr = np.empty(0, dtype=np.float64)
        return (empty_arr, empty_arr, empty_arr, empty_arr, empty_arr, empty_arr)
    action_array_calc = action_array.copy()
    if n > 0: action_array_calc[0] = 1
    amount = np.empty(n, dtype=np.float64); buffer = np.zeros(n, dtype=np.float64)
    cash = np.empty(n, dtype=np.float64); asset_value = np.empty(n, dtype=np.float64)
    sumusd = np.empty(n, dtype=np.float64); initial_price = price_array[0]
    amount[0] = fix / initial_price; cash[0] = fix
    asset_value[0] = amount[0] * initial_price; sumusd[0] = cash[0] + asset_value[0]
    refer = -fix * np.log(initial_price / price_array)
    for i in range(1, n):
        curr_price = price_array[i]
        if action_array_calc[i] == 0:
            amount[i] = amount[i-1]; buffer[i] = 0.0
        else:
            amount[i] = fix / curr_price; buffer[i] = amount[i-1] * curr_price - fix
        cash[i] = cash[i-1] + buffer[i]
        asset_value[i] = amount[i] * curr_price; sumusd[i] = cash[i] + asset_value[i]
    return buffer, sumusd, cash, asset_value, amount, refer

@lru_cache(maxsize=4096)
def calculate_optimized_cached(action_tuple: Tuple[int, ...], price_tuple: Tuple[float, ...], fix: int = 1500) -> Tuple:
    action_array = np.asarray(action_tuple, dtype=np.int32)
    price_array = np.asarray(price_tuple, dtype=np.float64)
    return _calculate_simulation_numba(action_array, price_array, fix)

def run_simulation(prices: List[float], actions: List[int], fix: int = 1500) -> pd.DataFrame:
    if not prices or not actions: return pd.DataFrame()
    min_len = min(len(prices), len(actions)); prices = prices[:min_len]; actions = actions[:min_len]
    action_array = np.asarray(actions, dtype=np.int32)
    price_array = np.asarray(prices, dtype=np.float64)
    buffer, sumusd, cash, asset_value, amount, refer = _calculate_simulation_numba(action_array, price_array, fix)
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
def generate_actions_rebalance_daily(num_days: int) -> np.ndarray:
    return np.ones(num_days, dtype=np.int32)

# MODIFIED: Replaced the Numba version with the original v1 version for correctness
def generate_actions_perfect_foresight(prices: List[float], fix: int = 1500) -> np.ndarray:
    """
    This is the function from v1, which is confirmed to work correctly.
    It takes a list of prices as input.
    """
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


# 3.1 Standard Seed Generation
def find_best_seed_for_window(prices_window: np.ndarray, num_seeds_to_try: int, max_workers: int) -> Tuple[int, float, np.ndarray]:
    window_len = len(prices_window)
    if window_len < 2: return 1, 0.0, np.ones(window_len, dtype=np.int32)
    def evaluate_seed_batch(seed_batch: np.ndarray) -> List[Tuple[int, float]]:
        results = []
        for seed in seed_batch:
            rng = np.random.default_rng(seed)
            actions_window = rng.integers(0, 2, size=window_len, dtype=np.int32)
            _, sumusd, _, _, _, refer = calculate_optimized_cached(tuple(actions_window), tuple(prices_window))
            net = sumusd[-1] - refer[-1] - sumusd[0] if len(sumusd) > 0 else -np.inf
            results.append((seed, net))
        return results
    best_seed_for_window = -1; max_net_for_window = -np.inf
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
    if best_seed_for_window >= 0:
        rng_best = np.random.default_rng(best_seed_for_window)
        best_actions = rng_best.integers(0, 2, size=window_len, dtype=np.int32)
    else:
        best_seed_for_window = 1
        best_actions = np.ones(window_len, dtype=np.int32)
        max_net_for_window = 0.0
    best_actions[0] = 1
    return best_seed_for_window, max_net_for_window, best_actions

def generate_actions_sliding_window(ticker_data: pd.DataFrame, window_size: int, num_seeds_to_try: int, max_workers: int) -> Tuple[np.ndarray, pd.DataFrame]:
    prices = ticker_data['Close'].to_numpy()
    n = len(prices)
    final_actions = np.array([], dtype=np.int32)
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
        best_seed, max_net, best_actions = find_best_seed_for_window(prices_window, num_seeds_to_try, max_workers)
        final_actions = np.concatenate((final_actions, best_actions))
        start_date_str = ticker_data.index[start_index].strftime('%Y-%m-%d')
        end_date_str = ticker_data.index[end_index-1].strftime('%Y-%m-%d')
        detail = {
            'window_number': i + 1, 'timeline': f"{start_date_str} ถึง {end_date_str}", 'best_seed': best_seed, 'max_net': round(max_net, 2),
            'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2), 'action_count': int(np.sum(best_actions)),
            'window_size': window_len, 'action_sequence': best_actions.tolist()
        }
        window_details_list.append(detail)
        progress_bar.progress((i + 1) / num_windows, text=f"ประมวลผล Window {i+1}/{num_windows}")
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)

# 3.2 Chaotic Seed Generation
def chaotic_params_to_seed(r: float, x0: float) -> int:
    PRECISION_FACTOR = 1_000_000
    r_int = int((r - 3.0) * PRECISION_FACTOR)
    x0_int = int(x0 * PRECISION_FACTOR)
    seed = r_int * (PRECISION_FACTOR + 1) + x0_int
    return seed

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
    if length > 0:
        actions[0] = 1
    return actions

def generate_actions_from_chaotic_seed(length: int, seed: int) -> np.ndarray:
    if length == 0:
        return np.array([], dtype=np.int32)
    params = seed_to_chaotic_params(seed)
    return _generate_chaotic_actions_numba(length, params['r'], params['x0'])

def find_best_chaotic_seed(prices_window: np.ndarray, num_seeds_to_try: int, max_workers: int) -> Tuple[int, float, np.ndarray]:
    window_len = len(prices_window)
    if window_len < 2:
        default_seed = chaotic_params_to_seed(4.0, 0.1)
        return default_seed, 0.0, np.ones(window_len, dtype=np.int32)
    def evaluate_chaotic_seed_batch(seed_batch: np.ndarray) -> List[Tuple[int, float]]:
        results = []
        for seed in seed_batch:
            actions_window = generate_actions_from_chaotic_seed(window_len, seed)
            _, sumusd, _, _, _, refer = calculate_optimized_cached(tuple(actions_window), tuple(prices_window))
            net = sumusd[-1] - refer[-1] - sumusd[0] if len(sumusd) > 0 else -np.inf
            results.append((seed, net))
        return results
    best_seed = -1
    max_net = -np.inf
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
    else:
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
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        window_len = len(prices_window)
        if window_len == 0: continue
        best_seed, max_net, best_actions = find_best_chaotic_seed(prices_window, num_seeds_to_try, max_workers)
        final_actions = np.concatenate((final_actions, best_actions))
        start_date_str = ticker_data.index[start_index].strftime('%Y-%m-%d')
        end_date_str = ticker_data.index[end_index-1].strftime('%Y-%m-%d')
        params = seed_to_chaotic_params(best_seed)
        detail = {
            'window_number': i + 1, 'timeline': f"{start_date_str} ถึง {end_date_str}",
            'best_seed': best_seed, 'r_param': round(params['r'], 6), 'x0_param': round(params['x0'], 6),
            'max_net': round(max_net, 2),
            'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2),
            'action_count': int(np.sum(best_actions)), 'window_size': window_len,
            'action_sequence': best_actions.tolist()
        }
        window_details_list.append(detail)
        progress_bar.progress((i + 1) / num_windows, text=f"ประมวลผล Chaotic Window {i+1}/{num_windows}")
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)

# 3.3 Genetic Algorithm Generation with Seed
def find_best_solution_ga(prices_window: np.ndarray, population_size: int, generations: int, seed: int, mutation_rate: float = 0.01) -> Tuple[float, np.ndarray]:
    window_len = len(prices_window)
    if window_len < 2: return 0.0, np.ones(window_len, dtype=np.int32)

    rng = np.random.default_rng(seed)

    population = rng.integers(0, 2, size=(population_size, window_len), dtype=np.int32)
    population[:, 0] = 1
    for _ in range(generations):
        fitness_scores = []
        for chromosome in population:
            _, sumusd, _, _, _, refer = calculate_optimized_cached(tuple(chromosome), tuple(prices_window))
            net = sumusd[-1] - refer[-1] - sumusd[0] if len(sumusd) > 0 else -np.inf
            fitness_scores.append(net)
        fitness_scores = np.array(fitness_scores)
        num_parents = population_size // 2
        parent_indices = np.argsort(fitness_scores)[-num_parents:]
        parents = population[parent_indices]
        num_offspring = population_size - num_parents
        offspring = np.empty((num_offspring, window_len), dtype=np.int32)
        for k in range(num_offspring):
            parent1_idx, parent2_idx = rng.choice(num_parents, size=2, replace=False)
            crossover_point = rng.integers(1, window_len)
            offspring[k, :crossover_point] = parents[parent1_idx, :crossover_point]
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        mutation_mask = rng.random((num_offspring, window_len)) < mutation_rate
        offspring[mutation_mask] = 1 - offspring[mutation_mask]
        offspring[:, 0] = 1
        population[num_parents:] = offspring
    final_fitness = []
    for chromosome in population:
        _, sumusd, _, _, _, refer = calculate_optimized_cached(tuple(chromosome), tuple(prices_window))
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
    st.write(f"📊 ข้อมูล: {n} วัน | Window: {window_size} วัน | ประชากร: {population_size} | รุ่น: {generations}")
    st.write("---")
    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n); prices_window = prices[start_index:end_index]
        window_len = len(prices_window)
        if window_len < 2: continue

        window_seed = master_seed + i

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

# ==============================================================================
# 4. UI Rendering Functions
# ==============================================================================
def render_settings_tab(config: Dict[str, Any]):
    st.write("⚙️ **การตั้งค่าพารามิเตอร์**")
    asset_list = config.get('assets', ['FFWM'])
    try: default_index = asset_list.index(st.session_state.test_ticker)
    except ValueError: default_index = 0
    st.session_state.test_ticker = st.selectbox("เลือก Ticker สำหรับทดสอบ", options=asset_list, index=default_index)
    st.write("📅 **ช่วงวันที่สำหรับการวิเคราะห์**"); col1, col2 = st.columns(2)
    with col1: st.session_state.start_date = st.date_input("วันที่เริ่มต้น", value=st.session_state.start_date)
    with col2: st.session_state.end_date = st.date_input("วันที่สิ้นสุด", value=st.session_state.end_date)
    if st.session_state.start_date >= st.session_state.end_date: st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
    st.divider()
    st.subheader("พารามิเตอร์ทั่วไป")
    st.session_state.window_size = st.number_input("ขนาด Window (วัน)", min_value=2, value=st.session_state.window_size)
    st.subheader("พารามิเตอร์สำหรับ Random/Chaotic Seed")
    c1, c2 = st.columns(2)
    st.session_state.num_seeds = c1.number_input("จำนวน Seeds/Params ต่อ Window", min_value=100, value=st.session_state.num_seeds, format="%d")
    st.session_state.max_workers = c2.number_input("จำนวน Workers", min_value=1, max_value=16, value=st.session_state.max_workers)
    st.subheader("พารามิเตอร์สำหรับ Genetic Algorithm")
    ga_c1, ga_c2, ga_c3 = st.columns(3)
    st.session_state.ga_population_size = ga_c1.number_input("ขนาดประชากร (Population Size)", min_value=10, value=st.session_state.ga_population_size)
    st.session_state.ga_generations = ga_c2.number_input("จำนวนรุ่น (Generations)", min_value=5, value=st.session_state.ga_generations)
    st.session_state.ga_master_seed = ga_c3.number_input("Master Seed for GA", value=st.session_state.ga_master_seed, format="%d", help="เปลี่ยนค่านี้เพื่อเปลี่ยนผลลัพธ์ของ GA ทั้งหมด แต่การใช้ค่าเดิมจะให้ผลลัพธ์เดิมเสมอ")

def display_comparison_charts(results: Dict[str, pd.DataFrame], chart_title: str = '📊 เปรียบเทียบกำไรสุทธิ (Net Profit)'):
    if not results: st.warning("ไม่มีข้อมูลสำหรับสร้างกราฟเปรียบเทียบ"); return
    valid_dfs = {name: df for name, df in results.items() if not df.empty and 'net' in df.columns}
    if not valid_dfs: st.warning("ไม่มีข้อมูล 'net' สำหรับสร้างกราฟ"); return
    try: longest_index = max((df.index for df in valid_dfs.values()), key=len, default=None)
    except ValueError: longest_index = None
    if longest_index is None: st.warning("ไม่มีข้อมูลสำหรับสร้างกราฟเปรียบเทียบ"); return
    chart_data = pd.DataFrame(index=longest_index)
    for name, df in valid_dfs.items():
        chart_data[name] = df['net'].reindex(longest_index).ffill()
    st.write(chart_title); st.line_chart(chart_data)

def render_test_tab():
    st.write("---")
    if st.button("🚀 เริ่มทดสอบ Best Seed (Optimized)", type="primary"):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้องในแท็บ 'การตั้งค่า'"); return
        ticker = st.session_state.test_ticker; start_date_str = st.session_state.start_date.strftime('%Y-%m-%d'); end_date_str = st.session_state.end_date.strftime('%Y-%m-%d')
        st.info(f"กำลังดึงข้อมูลสำหรับ **{ticker}** | {start_date_str} ถึง {end_date_str}")
        ticker_data = get_ticker_data(ticker, start_date_str, end_date_str)
        if ticker_data.empty: st.error("ไม่พบข้อมูลสำหรับ Ticker และช่วงวันที่ที่เลือก"); return
        prices = ticker_data['Close'].to_numpy(); num_days = len(prices)
        with st.spinner("กำลังคำนวณกลยุทธ์ต่างๆ..."):
            st.write("🔍 **เริ่มต้นการค้นหา Best Seed ด้วย Sliding Window**")
            actions_sliding, df_windows = generate_actions_sliding_window(ticker_data, st.session_state.window_size, st.session_state.num_seeds, st.session_state.max_workers)
            actions_min = generate_actions_rebalance_daily(num_days)
            # MODIFIED: Call with .tolist()
            actions_max = generate_actions_perfect_foresight(prices.tolist())
            results = {}; strategy_map = {Strategy.SLIDING_WINDOW: actions_sliding.tolist(), Strategy.REBALANCE_DAILY: actions_min.tolist(), Strategy.PERFECT_FORESIGHT: actions_max.tolist()}
            for strategy_name, actions in strategy_map.items():
                df = run_simulation(prices.tolist(), actions)
                if not df.empty: df.index = ticker_data.index[:len(df)]
                results[strategy_name] = df
        st.success("การทดสอบเสร็จสมบูรณ์!")
        st.write("---"); display_comparison_charts(results)
        st.write("📈 **สรุปผลการค้นหา Best Seed**")
        total_actions = df_windows['action_count'].sum(); total_net = df_windows['max_net'].sum()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Windows", df_windows.shape[0]); col2.metric("Total Actions", f"{total_actions}/{num_days}"); col3.metric("Total Net (Sum)", f"${total_net:,.2f}")
        st.dataframe(df_windows[['window_number', 'timeline', 'best_seed', 'max_net', 'price_change_pct', 'action_count']], use_container_width=True)
        csv = df_windows.to_csv(index=False); st.download_button(label="📥 ดาวน์โหลด Window Details (CSV)", data=csv, file_name=f'best_seed_{ticker}_{st.session_state.window_size}w.csv', mime='text/csv')

def render_chaotic_test_tab():
    st.write("---")
    st.markdown("### 🌀 ทดสอบ Best Seed ด้วย Chaotic Generator (Logistic Map)")
    st.info("กลยุทธ์นี้จะค้นหาค่าพารามิเตอร์ `r` และ `x0` ของ Logistic Map ที่ให้
