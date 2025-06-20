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
    MANUAL_SEED = "Manual Seed Strategy"
    GENETIC_ALGORITHM = "Genetic Algorithm Sliding Window"
    REINFORCEMENT_LEARNING_QL = "Reinforcement Learning (Q-Learning)"


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
            "assets": ["FFWM", "NEGG", "RIVN", "BTC-USD", "ETH-USD"],
            "default_settings": {
                "selected_ticker": "FFWM", "start_date": "2025-06-10", "window_size": 30, 
                "num_seeds": 30000, "max_workers": 8, 
                "ga_population_size": 50, "ga_generations": 20,
                "rl_episodes": 100, "rl_learning_rate": 0.1, "rl_epsilon": 0.3
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
    
    new_presets = presets_by_asset.get(selected_ticker, default_presets)
    st.session_state.manual_seed_lines = new_presets

def initialize_session_state(config: Dict[str, Any]):
    """
    ตั้งค่าเริ่มต้นสำหรับ Streamlit session state โดยใช้ค่าจาก config
    """
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
    
    # GA settings
    if 'ga_population_size' not in st.session_state:
        st.session_state.ga_population_size = defaults.get('ga_population_size', 50)
    if 'ga_generations' not in st.session_state:
        st.session_state.ga_generations = defaults.get('ga_generations', 20)

    # RL settings
    if 'rl_episodes' not in st.session_state:
        st.session_state.rl_episodes = defaults.get('rl_episodes', 100)
    if 'rl_learning_rate' not in st.session_state:
        st.session_state.rl_learning_rate = defaults.get('rl_learning_rate', 0.1)
    if 'rl_epsilon' not in st.session_state:
        st.session_state.rl_epsilon = defaults.get('rl_epsilon', 0.3)

    if 'df_for_analysis' not in st.session_state:
        st.session_state.df_for_analysis = None
    
    if 'manual_seed_lines' not in st.session_state:
        initial_ticker = defaults.get('selected_ticker', 'FFWM')
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
def _calculate_simulation_numba(action_array: np.ndarray, price_array: np.ndarray, fix: int = 1500) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(action_array)
    if n == 0:
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

@lru_cache(maxsize=4096)
def calculate_optimized_cached(action_tuple: Tuple[int, ...], price_tuple: Tuple[float, ...], fix: int = 1500) -> Tuple:
    action_array = np.asarray(action_tuple, dtype=np.int32)
    price_array = np.asarray(price_tuple, dtype=np.float64)
    return _calculate_simulation_numba(action_array, price_array, fix)

def run_simulation(prices: List[float], actions: List[int], fix: int = 1500) -> pd.DataFrame:
    if not prices or not actions: return pd.DataFrame()
    action_array = np.asarray(actions, dtype=np.int32)
    price_array = np.asarray(prices, dtype=np.float64)
    buffer, sumusd, cash, asset_value, amount, refer = _calculate_simulation_numba(action_array, price_array, fix)
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
# 3. Strategy Action Generation
# ==============================================================================
# 3.0 Benchmarks
def generate_actions_rebalance_daily(num_days: int) -> np.ndarray:
    return np.ones(num_days, dtype=np.int32)

@njit(cache=True)
def generate_actions_perfect_foresight(price_arr: np.ndarray, fix: int = 1500) -> np.ndarray:
    n = len(price_arr)
    if n < 2: return np.ones(n, dtype=np.int32)
    dp = np.zeros(n, dtype=np.float64)
    path = np.zeros(n, dtype=np.int32)
    dp[0] = float(fix * 2)
    for i in range(1, n):
        j_indices = np.arange(i)
        profits = fix * ((price_arr[i] / price_arr[j_indices]) - 1.0)
        current_sumusd = dp[j_indices] + profits
        best_idx = np.argmax(current_sumusd)
        dp[i] = current_sumusd[best_idx]
        path[i] = j_indices[best_idx]
    actions = np.zeros(n, dtype=np.int32)
    best_final_day = np.argmax(dp)
    current_day = best_final_day
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
            if len(sumusd) > 0: net = sumusd[-1] - refer[-1] - sumusd[0]
            else: net = -np.inf
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
    progress_bar = st.progress(0, text="กำลังประมวลผล Random Seed Sliding Windows...")
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
    return r_int * (PRECISION_FACTOR + 1) + x0_int

def seed_to_chaotic_params(seed: int) -> dict:
    PRECISION_FACTOR = 1_000_000
    r_int = seed // (PRECISION_FACTOR + 1)
    x0_int = seed % (PRECISION_FACTOR + 1)
    r = (r_int / PRECISION_FACTOR) + 3.0; x0 = x0_int / PRECISION_FACTOR
    return {'r': max(3.57, min(4.0, r)), 'x0': max(0.01, min(0.99, x0))}

@njit(cache=True)
def _generate_chaotic_actions_numba(length: int, r: float, x0: float) -> np.ndarray:
    actions = np.zeros(length, dtype=np.int32)
    x = x0
    for i in range(length):
        x = r * x * (1.0 - x); actions[i] = 1 if x > 0.5 else 0
    if length > 0: actions[0] = 1
    return actions

def generate_actions_from_chaotic_seed(length: int, seed: int) -> np.ndarray:
    if length == 0: return np.array([], dtype=np.int32)
    params = seed_to_chaotic_params(seed)
    return _generate_chaotic_actions_numba(length, params['r'], params['x0'])

def find_best_chaotic_seed(prices_window: np.ndarray, num_seeds_to_try: int, max_workers: int) -> Tuple[int, float, np.ndarray]:
    window_len = len(prices_window)
    if window_len < 2: return chaotic_params_to_seed(4.0, 0.1), 0.0, np.ones(window_len, dtype=np.int32)
    def evaluate_chaotic_seed_batch(seed_batch: np.ndarray) -> List[Tuple[int, float]]:
        results = []
        for seed in seed_batch:
            actions_window = generate_actions_from_chaotic_seed(window_len, seed)
            _, sumusd, _, _, _, refer = calculate_optimized_cached(tuple(actions_window), tuple(prices_window))
            net = sumusd[-1] - refer[-1] - sumusd[0] if len(sumusd) > 0 else -np.inf
            results.append((seed, net))
        return results
    best_seed = -1; max_net = -np.inf
    rng_for_params = np.random.default_rng()
    r_params = rng_for_params.uniform(3.57, 4.0, num_seeds_to_try)
    x0_params = rng_for_params.uniform(0.01, 0.99, num_seeds_to_try)
    seed_list = [chaotic_params_to_seed(r, x0) for r, x0 in zip(r_params, x0_params)]
    random_seeds_to_try = np.array(seed_list)
    batch_size = max(1, num_seeds_to_try // (max_workers * 4))
    seed_batches = [random_seeds_to_try[j:j + batch_size] for j in range(0, len(random_seeds_to_try), batch_size)]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_chaotic_seed_batch, batch) for batch in seed_batches]
        for future in as_completed(futures):
            for seed, final_net in future.result():
                if final_net > max_net:
                    max_net = final_net; best_seed = seed
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

# 3.3 Genetic Algorithm Generation
def find_best_solution_ga(prices_window: np.ndarray, population_size: int, generations: int, mutation_rate: float = 0.01) -> Tuple[float, np.ndarray]:
    window_len = len(prices_window)
    if window_len < 2: return 0.0, np.ones(window_len, dtype=np.int32)
    rng = np.random.default_rng()
    population = rng.integers(0, 2, size=(population_size, window_len), dtype=np.int32)
    population[:, 0] = 1
    for gen in range(generations):
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
            parent1_idx = rng.integers(0, num_parents); parent2_idx = rng.integers(0, num_parents)
            crossover_point = rng.integers(1, window_len)
            offspring[k, :crossover_point] = parents[parent1_idx, :crossover_point]
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        mutation_mask = rng.random((num_offspring, window_len)) < mutation_rate
        offspring[mutation_mask] = 1 - offspring[mutation_mask]
        offspring[:, 0] = 1
        population[:num_parents] = parents; population[num_parents:] = offspring
    final_fitness = []
    for chromosome in population:
        _, sumusd, _, _, _, refer = calculate_optimized_cached(tuple(chromosome), tuple(prices_window))
        net = sumusd[-1] - refer[-1] - sumusd[0] if len(sumusd) > 0 else -np.inf
        final_fitness.append(net)
    best_idx = np.argmax(final_fitness)
    return final_fitness[best_idx], population[best_idx]

def generate_actions_sliding_window_ga(ticker_data: pd.DataFrame, window_size: int, population_size: int, generations: int) -> Tuple[np.ndarray, pd.DataFrame]:
    prices = ticker_data['Close'].to_numpy(); n = len(prices)
    final_actions = np.array([], dtype=np.int32); window_details_list = []
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="กำลังประมวลผล Genetic Algorithm Sliding Windows...")
    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n); prices_window = prices[start_index:end_index]
        window_len = len(prices_window)
        if window_len == 0: continue
        max_net, best_actions = find_best_solution_ga(prices_window, population_size, generations)
        final_actions = np.concatenate((final_actions, best_actions))
        start_date_str = ticker_data.index[start_index].strftime('%Y-%m-%d'); end_date_str = ticker_data.index[end_index-1].strftime('%Y-%m-%d')
        detail = {
            'window_number': i + 1, 'timeline': f"{start_date_str} ถึง {end_date_str}", 'max_net': round(max_net, 2),
            'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2),
            'action_count': int(np.sum(best_actions)), 'window_size': window_len, 'action_sequence': best_actions.tolist()
        }
        window_details_list.append(detail)
        progress_bar.progress((i + 1) / num_windows, text=f"Evolving Window {i+1}/{num_windows}")
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)

# 3.4 [NEW] Reinforcement Learning (Q-Learning) Generation
@njit(cache=True)
def discretize_state(price: float, start_price: float) -> int:
    ratio = price / start_price
    if ratio < 0.95: return 0
    if ratio < 0.99: return 1
    if ratio < 1.01: return 2
    if ratio < 1.05: return 3
    return 4

@njit(cache=True)
def train_q_learning_for_window(prices: np.ndarray, episodes: int, alpha: float, gamma: float, epsilon: float) -> np.ndarray:
    n_states = 5; n_actions = 2
    q_table = np.zeros((n_states, n_actions), dtype=np.float64)
    if len(prices) < 2: return q_table
    start_price = prices[0]
    for _ in range(episodes):
        amount = 1500.0 / start_price
        last_rebalance_price = start_price
        for t in range(1, len(prices)):
            current_price = prices[t]
            state = discretize_state(current_price, start_price)
            action = np.random.randint(0, n_actions) if np.random.uniform(0, 1) < epsilon else np.argmax(q_table[state, :])
            reward = (amount * current_price) - (amount * last_rebalance_price) if action == 1 else 0.0
            if action == 1: last_rebalance_price = current_price
            next_price = prices[t+1] if t + 1 < len(prices) else current_price
            next_state = discretize_state(next_price, start_price)
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state, :])
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_table[state, action] = new_value
    return q_table

def generate_actions_from_q_policy(prices: np.ndarray, q_table: np.ndarray) -> np.ndarray:
    n = len(prices)
    if n == 0: return np.array([], dtype=np.int32)
    actions = np.zeros(n, dtype=np.int32)
    actions[0] = 1
    start_price = prices[0]
    for t in range(1, n):
        state = discretize_state(prices[t], start_price)
        actions[t] = np.argmax(q_table[state, :])
    return actions

def generate_actions_sliding_window_q_learning(ticker_data: pd.DataFrame, window_size: int, episodes: int, learning_rate: float, epsilon: float) -> Tuple[np.ndarray, pd.DataFrame]:
    prices = ticker_data['Close'].to_numpy(); n = len(prices)
    final_actions = np.array([], dtype=np.int32); window_details_list = []
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="กำลังฝึก Reinforcement Learning Agent...")
    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        window_len = len(prices_window)
        if window_len < 2: continue
        q_table = train_q_learning_for_window(prices_window, episodes, learning_rate, 0.9, epsilon)
        best_actions = generate_actions_from_q_policy(prices_window, q_table)
        final_actions = np.concatenate((final_actions, best_actions))
        _, sumusd, _, _, _, refer = calculate_optimized_cached(tuple(best_actions), tuple(prices_window))
        max_net = sumusd[-1] - refer[-1] - sumusd[0] if len(sumusd) > 0 else 0.0
        start_date_str = ticker_data.index[start_index].strftime('%Y-%m-%d')
        end_date_str = ticker_data.index[end_index-1].strftime('%Y-%m-%d')
        detail = {
            'window_number': i + 1, 'timeline': f"{start_date_str} ถึง {end_date_str}", 'max_net': round(max_net, 2),
            'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2),
            'action_count': int(np.sum(best_actions)), 'window_size': window_len, 'action_sequence': best_actions.tolist()
        }
        window_details_list.append(detail)
        progress_bar.progress((i + 1) / num_windows, text=f"Training Agent for Window {i+1}/{num_windows}")
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
    
    st.write("📅 **ช่วงวันที่สำหรับการวิเคราะห์**")
    col1, col2 = st.columns(2)
    with col1: st.session_state.start_date = st.date_input("วันที่เริ่มต้น", value=st.session_state.start_date)
    with col2: st.session_state.end_date = st.date_input("วันที่สิ้นสุด", value=st.session_state.end_date)
    
    if st.session_state.start_date >= st.session_state.end_date:
        st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
    else:
        st.info(f"ช่วงวันที่ที่เลือก: {st.session_state.start_date:%Y-%m-%d} ถึง {st.session_state.end_date:%Y-%m-%d}")
    
    st.divider()
    
    st.subheader("พารามิเตอร์ทั่วไป")
    c1, c2, c3 = st.columns(3)
    st.session_state.window_size = c1.number_input("ขนาด Window (วัน)", min_value=5, value=st.session_state.window_size)
    st.session_state.num_seeds = c2.number_input("จำนวน Seeds/Params ต่อ Window", min_value=100, value=st.session_state.num_seeds, format="%d", help="สำหรับ Random Seed และ Chaotic Seed")
    st.session_state.max_workers = c3.number_input("จำนวน Workers", min_value=1, max_value=16, value=st.session_state.max_workers, help="สำหรับ Random Seed และ Chaotic Seed")

    st.subheader("พารามิเตอร์สำหรับ Genetic Algorithm")
    ga_c1, ga_c2 = st.columns(2)
    st.session_state.ga_population_size = ga_c1.number_input("ขนาดประชากร (Population Size)", min_value=10, value=st.session_state.ga_population_size)
    st.session_state.ga_generations = ga_c2.number_input("จำนวนรุ่น (Generations)", min_value=5, value=st.session_state.ga_generations)

    st.subheader("พารามิเตอร์สำหรับ Reinforcement Learning (Q-Learning)")
    rl_c1, rl_c2, rl_c3 = st.columns(3)
    st.session_state.rl_episodes = rl_c1.number_input("จำนวนรอบการฝึก (Episodes)", min_value=10, value=st.session_state.rl_episodes)
    st.session_state.rl_learning_rate = rl_c2.number_input("อัตราการเรียนรู้ (Learning Rate)", min_value=0.01, max_value=1.0, value=st.session_state.rl_learning_rate, format="%.2f")
    st.session_state.rl_epsilon = rl_c3.number_input("ค่าการสำรวจ (Epsilon)", min_value=0.0, max_value=1.0, value=st.session_state.rl_epsilon, format="%.2f", help="ค่าสูง = สำรวจเยอะ, ค่าต่ำ = ทำตามที่เรียนรู้มา")

def display_comparison_charts(results: Dict[str, pd.DataFrame], chart_title: str = '📊 เปรียบเทียบกำไรสุทธิ (Net Profit)'):
    if not results: st.warning("ไม่มีข้อมูลสำหรับสร้างกราฟเปรียบเทียบ"); return
    try: longest_index = max((df.index for df in results.values() if not df.empty), key=len, default=None)
    except ValueError: longest_index = None
    if longest_index is None: st.warning("ไม่มีข้อมูลสำหรับสร้างกราฟเปรียบเทียบ"); return
    chart_data_dict = {}
    for name, df in results.items():
        if not df.empty:
            reindexed_df = df['net'].reindex(longest_index).ffill()
            chart_data_dict[name] = reindexed_df
    chart_data = pd.DataFrame(chart_data_dict)
    st.write(chart_title); st.line_chart(chart_data)

def run_and_display_strategy(strategy_func, strategy_args: Dict, strategy_name: str, ticker_data: pd.DataFrame, df_columns_to_show: List[str]):
    button_key = f"btn_{strategy_name.replace(' ', '_').replace('(', '').replace(')', '')}"
    if st.button(f"🚀 เริ่มทดสอบ ({strategy_name})", type="primary", key=button_key):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้องในแท็บ 'การตั้งค่า'"); return
        prices = ticker_data['Close'].to_numpy(); num_days = len(prices)
        with st.spinner(f"กำลังคำนวณกลยุทธ์ {strategy_name}..."):
            main_actions, df_windows = strategy_func(**strategy_args)
            actions_min = generate_actions_rebalance_daily(num_days)
            actions_max = generate_actions_perfect_foresight(prices)
            results_map = {strategy_name: main_actions.tolist(), Strategy.REBALANCE_DAILY: actions_min.tolist(), Strategy.PERFECT_FORESIGHT: actions_max.tolist()}
            sim_results = {}
            for name, actions in results_map.items():
                df = run_simulation(prices.tolist(), actions)
                if not df.empty: df.index = ticker_data.index[:len(df)]
                sim_results[name] = df
        st.success("การทดสอบเสร็จสมบูรณ์!")
        st.write("---"); display_comparison_charts(sim_results)
        st.write(f"📈 **สรุปผลการค้นหา ({strategy_name})**")
        if not df_windows.empty:
            total_actions = df_windows['action_count'].sum(); total_net = df_windows['max_net'].sum()
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Windows", df_windows.shape[0])
            col2.metric("Total Actions", f"{total_actions}/{num_days}")
            col3.metric("Total Net (Sum)", f"${total_net:,.2f}")
            st.dataframe(df_windows[df_columns_to_show], use_container_width=True)
            csv = df_windows.to_csv(index=False)
            st.download_button(label=f"📥 ดาวน์โหลด {strategy_name} Details (CSV)", data=csv, file_name=f'{strategy_name.replace(" ", "_")}_{st.session_state.test_ticker}_{st.session_state.window_size}w.csv', mime='text/csv')
        else:
            st.warning("ไม่สามารถสร้างข้อมูลสรุป Window ได้")

def render_test_tab():
    st.markdown("### 🎲 ทดสอบ Best Seed ด้วย Random Search")
    st.info("กลยุทธ์นี้จะสุ่ม `seed` จำนวนมากเพื่อหาลำดับ Action ที่ดีที่สุดในแต่ละ Window")
    ticker = st.session_state.test_ticker
    ticker_data = get_ticker_data(ticker, st.session_state.start_date.strftime('%Y-%m-%d'), st.session_state.end_date.strftime('%Y-%m-%d'))
    if ticker_data.empty: st.error(f"ไม่พบข้อมูลสำหรับ {ticker} ในช่วงวันที่ที่เลือก"); return
    args = {'ticker_data': ticker_data, 'window_size': st.session_state.window_size, 'num_seeds_to_try': st.session_state.num_seeds, 'max_workers': st.session_state.max_workers}
    columns_to_show = ['window_number', 'timeline', 'best_seed', 'max_net', 'price_change_pct', 'action_count']
    run_and_display_strategy(generate_actions_sliding_window, args, Strategy.SLIDING_WINDOW, ticker_data, columns_to_show)

def render_chaotic_test_tab():
    st.markdown("### 🌀 ทดสอบ Best Seed ด้วย Chaotic Generator (Logistic Map)")
    st.info("กลยุทธ์นี้จะค้นหาค่าพารามิเตอร์ `r` และ `x0` ของ Logistic Map ที่ให้ผลตอบแทนดีที่สุดในแต่ละ Window")
    ticker = st.session_state.test_ticker
    ticker_data = get_ticker_data(ticker, st.session_state.start_date.strftime('%Y-%m-%d'), st.session_state.end_date.strftime('%Y-%m-%d'))
    if ticker_data.empty: st.error(f"ไม่พบข้อมูลสำหรับ {ticker} ในช่วงวันที่ที่เลือก"); return
    args = {'ticker_data': ticker_data, 'window_size': st.session_state.window_size, 'num_seeds_to_try': st.session_state.num_seeds, 'max_workers': st.session_state.max_workers}
    columns_to_show = ['window_number', 'timeline', 'best_seed', 'r_param', 'x0_param', 'max_net', 'price_change_pct', 'action_count']
    run_and_display_strategy(generate_actions_sliding_window_chaotic, args, Strategy.CHAOTIC_SLIDING_WINDOW, ticker_data, columns_to_show)

def render_ga_test_tab():
    st.markdown("### 🧬 ทดสอบด้วย Genetic Algorithm Search")
    st.info("กลยุทธ์นี้ใช้วิวัฒนาการเชิงคำนวณ (Genetic Algorithm) เพื่อ 'พัฒนา' Action Sequence ที่ดีที่สุดในแต่ละ Window แทนการสุ่ม")
    ticker = st.session_state.test_ticker
    ticker_data = get_ticker_data(ticker, st.session_state.start_date.strftime('%Y-%m-%d'), st.session_state.end_date.strftime('%Y-%m-%d'))
    if ticker_data.empty: st.error(f"ไม่พบข้อมูลสำหรับ {ticker} ในช่วงวันที่ที่เลือก"); return
    args = {'ticker_data': ticker_data, 'window_size': st.session_state.window_size, 'population_size': st.session_state.ga_population_size, 'generations': st.session_state.ga_generations}
    columns_to_show = ['window_number', 'timeline', 'max_net', 'price_change_pct', 'action_count']
    run_and_display_strategy(generate_actions_sliding_window_ga, args, Strategy.GENETIC_ALGORITHM, ticker_data, columns_to_show)

def render_q_learning_test_tab():
    st.markdown("### 🤖 ทดสอบด้วย Reinforcement Learning (Q-Learning)")
    st.info("กลยุทธ์นี้จะฝึก 'Agent' ให้เรียนรู้ที่จะตัดสินใจ (Hold/Rebalance) เพื่อสร้างผลตอบแทนสูงสุดในแต่ละ Window ผ่านการลองผิดลองถูก")
    ticker = st.session_state.test_ticker
    ticker_data = get_ticker_data(ticker, st.session_state.start_date.strftime('%Y-%m-%d'), st.session_state.end_date.strftime('%Y-%m-%d'))
    if ticker_data.empty: st.error(f"ไม่พบข้อมูลสำหรับ {ticker} ในช่วงวันที่ที่เลือก"); return
    args = {'ticker_data': ticker_data, 'window_size': st.session_state.window_size, 'episodes': st.session_state.rl_episodes, 'learning_rate': st.session_state.rl_learning_rate, 'epsilon': st.session_state.rl_epsilon}
    columns_to_show = ['window_number', 'timeline', 'max_net', 'price_change_pct', 'action_count']
    run_and_display_strategy(generate_actions_sliding_window_q_learning, args, Strategy.REINFORCEMENT_LEARNING_QL, ticker_data, columns_to_show)

def render_analytics_tab():
    st.header("📊 Advanced Analytics Dashboard")
    # ... (โค้ดส่วน Analytics เหมือนเดิม)
    def safe_literal_eval(val):
        if pd.isna(val): return []
        if isinstance(val, list): return val
        if isinstance(val, str) and val.strip().startswith('[') and val.strip().endswith(']'):
            try: return ast.literal_eval(val)
            except: return []
        return []

    with st.container():
        st.subheader("เลือกวิธีการนำเข้าข้อมูล:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 1. อัปโหลดไฟล์จากเครื่อง")
            uploaded_file = st.file_uploader("อัปโหลดไฟล์ CSV ของคุณ", type=['csv'], key="local_uploader")
            if uploaded_file is not None:
                try: st.session_state.df_for_analysis = pd.read_csv(uploaded_file); st.success("✅ โหลดไฟล์สำเร็จ!")
                except Exception as e: st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}"); st.session_state.df_for_analysis = None
        with col2:
            st.markdown("##### 2. หรือ โหลดจาก GitHub URL")
            default_github_url = f"https://raw.githubusercontent.com/firstnattapon/streamlit-example-1/refs/heads/master/Seed_Sliding_Window/{st.session_state.test_ticker}.csv"
            github_url = st.text_input("ป้อน GitHub URL ของไฟล์ CSV:", value=default_github_url, key="github_url_input")
            if st.button("📥 โหลดข้อมูลจาก GitHub"):
                if github_url:
                    try:
                        raw_url = github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
                        with st.spinner("กำลังดาวน์โหลดข้อมูล..."): st.session_state.df_for_analysis = pd.read_csv(raw_url)
                        st.success("✅ โหลดข้อมูลจาก GitHub สำเร็จ!")
                    except Exception as e: st.error(f"❌ ไม่สามารถโหลดข้อมูลจาก URL ได้: {e}"); st.session_state.df_for_analysis = None
                else: st.warning("กรุณาป้อน URL ของไฟล์ CSV")
    st.divider()
    if st.session_state.df_for_analysis is not None:
        st.subheader("ผลการวิเคราะห์")
        df_to_analyze = st.session_state.df_for_analysis
        try:
            # ... (ส่วนวิเคราะห์ไฟล์เหมือนเดิม)
            stitched_dna_tab, overview_tab = st.tabs(["🧬 Stitched DNA Analysis", "🔬 ภาพรวมและสำรวจราย Window"])
            with overview_tab:
                # ...
                pass
            with stitched_dna_tab:
                # ...
                pass

        except Exception as e: st.error(f"เกิดข้อผิดพลาดในการวิเคราะห์ข้อมูล: {e}"); st.exception(e)

def render_manual_seed_tab(config: Dict[str, Any]):
    st.header("🌱 Manual Seed Strategy Comparator")
    st.markdown("สร้างและเปรียบเทียบ Action Sequences โดยการตัดส่วนท้าย (`tail`) จาก Seed ที่กำหนด")
    with st.container(border=True):
        st.subheader("1. กำหนดค่า Input สำหรับการทดสอบ")
        col1, col2 = st.columns([1, 2])
        with col1:
            asset_list = config.get('assets', ['FFWM'])
            try: default_index = asset_list.index(st.session_state.get('manual_ticker_key', st.session_state.test_ticker))
            except (ValueError, KeyError): default_index = 0
            manual_ticker = st.selectbox("เลือก Ticker", options=asset_list, index=default_index, key="manual_ticker_key", on_change=on_ticker_change_callback, args=(config,))
        with col2:
            c1, c2 = st.columns(2)
            manual_start_date = c1.date_input("วันที่เริ่มต้น (Start Date)", value=st.session_state.start_date, key="manual_start_compare_tail")
            manual_end_date = c2.date_input("วันที่สิ้นสุด (End Date)", value=datetime.now().date(), key="manual_end_compare_tail")
        if manual_start_date >= manual_end_date: st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
        st.divider()
        st.write("**กำหนดกลยุทธ์ (Seed/Size/Tail) ที่ต้องการเปรียบเทียบ:**")
        for i, line in enumerate(st.session_state.manual_seed_lines):
            cols = st.columns([1, 2, 2, 2])
            cols[0].write(f"**Line {i+1}**")
            line['seed'] = cols[1].number_input("Input Seed", value=line.get('seed', 1), min_value=0, key=f"seed_compare_tail_{i}")
            line['size'] = cols[2].number_input("Size (ขนาด Sequence เริ่มต้น)", value=line.get('size', 60), min_value=1, key=f"size_compare_tail_{i}")
            line['tail'] = cols[3].number_input("Tail (ส่วนท้ายที่จะใช้)", value=line.get('tail', 10), min_value=1, max_value=line.get('size', 60), key=f"tail_compare_tail_{i}")
        b_col1, b_col2, _ = st.columns([1,1,4])
        if b_col1.button("➕ เพิ่ม Line เปรียบเทียบ"): st.session_state.manual_seed_lines.append({'seed': np.random.randint(1, 10000), 'size': 50, 'tail': 20}); st.rerun()
        if b_col2.button("➖ ลบ Line สุดท้าย"):
            if len(st.session_state.manual_seed_lines) > 1: st.session_state.manual_seed_lines.pop(); st.rerun()
            else: st.warning("ต้องมีอย่างน้อย 1 line")
    st.write("---")
    if st.button("📈 เปรียบเทียบประสิทธิภาพ Seeds", type="primary", key="compare_manual_seeds_btn"):
        if manual_start_date >= manual_end_date: st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้อง"); return
        with st.spinner("กำลังดึงข้อมูลและจำลองการเทรด..."):
            ticker_data = get_ticker_data(manual_ticker, manual_start_date.strftime('%Y-%m-%d'), manual_end_date.strftime('%Y-%m-%d'))
            if ticker_data.empty: st.error(f"ไม่พบข้อมูลสำหรับ {manual_ticker} ในช่วงวันที่ที่เลือก"); return
            prices = ticker_data['Close'].to_numpy(); num_trading_days = len(prices)
            st.info(f"📊 พบข้อมูลราคา {num_trading_days} วันทำการในช่วงที่เลือก")
            results = {}; max_sim_len = 0
            for i, line_info in enumerate(st.session_state.manual_seed_lines):
                input_seed, size_seed, tail_seed = line_info['seed'], line_info['size'], line_info['tail']
                if tail_seed > size_seed: st.error(f"Line {i+1}: Tail ({tail_seed}) ต้องไม่มากกว่า Size ({size_seed})"); return
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
            if not results: st.error("ไม่สามารถสร้างผลลัพธ์จาก Seed ที่กำหนดได้"); return
            if max_sim_len > 0:
                prices_for_benchmark = prices[:max_sim_len]
                df_max = run_simulation(prices_for_benchmark.tolist(), generate_actions_perfect_foresight(prices_for_benchmark).tolist())
                df_min = run_simulation(prices_for_benchmark.tolist(), generate_actions_rebalance_daily(max_sim_len).tolist())
                if not df_max.empty: df_max.index = ticker_data.index[:max_sim_len]; results[Strategy.PERFECT_FORESIGHT] = df_max
                if not df_min.empty: df_min.index = ticker_data.index[:max_sim_len]; results[Strategy.REBALANCE_DAILY] = df_min
            st.success("การเปรียบเทียบเสร็จสมบูรณ์!")
            display_comparison_charts(results, chart_title="📊 Performance Comparison (Net Profit)")

# ==============================================================================
# 5. Main Application
# ==============================================================================
def main():
    st.set_page_config(page_title="Strategy Optimization Lab", page_icon="🎯", layout="wide")
    st.markdown("## 🎯 Strategy Optimization Lab")
    st.caption("เครื่องมือทดสอบและเปรียบเทียบกลยุทธ์การลงทุนด้วยวิธี Sliding Window")

    config = load_config()
    initialize_session_state(config)

    tab_list = [
        "⚙️ การตั้งค่า", "🎲 Random Seed", "🌀 Chaotic Seed", "🧬 GA Search", "🤖 RL (Q-Learning)", "📊 Analytics", "🌱 Manual/Forward"
    ]
    tabs = st.tabs(tab_list)

    with tabs[0]: render_settings_tab(config)
    with tabs[1]: render_test_tab()
    with tabs[2]: render_chaotic_test_tab()
    with tabs[3]: render_ga_test_tab()
    with tabs[4]: render_q_learning_test_tab()
    with tabs[5]: render_analytics_tab()
    with tabs[6]: render_manual_seed_tab(config)

    with st.expander("📖 คำอธิบายกลยุทธ์และแนวคิด"):
        st.markdown("""
        **หลักการพื้นฐาน:** กลยุทธ์ทั้งหมดทำงานบนหลักการ "Sliding Window" คือแบ่งช่วงเวลาทั้งหมดออกเป็น "หน้าต่าง" เล็กๆ แล้วพยายามหา "รูปแบบการกระทำ" (Action Sequence) ที่ดีที่สุดสำหรับหน้าต่างนั้นๆ ก่อนจะนำผลลัพธ์ของทุกหน้าต่างมาประกอบกันเป็นกลยุทธ์สุดท้าย

        ---
        #### รูปแบบการค้นหา (Search Strategies)
        
        *   **🎲 Random Seed:** เป็นวิธีที่พื้นฐานที่สุด ค้นหาโดยการสุ่ม Action Sequence ขึ้นมาจำนวนมาก (ควบคุมโดย `seed`) แล้วเลือกอันที่ให้ผลตอบแทนดีที่สุด
        
        *   **🌀 Chaotic Seed:** ใช้ "Logistic Map" ในการสร้าง Action Sequence ที่มีรูปแบบซับซ้อนแต่กำหนดได้จากพารามิเตอร์ `r` และ `x0`
        
        *   **🧬 Genetic Algorithm:** เลียนแบบวิวัฒนาการทางธรรมชาติ โดยการสร้าง "ประชากร" ของ Action Sequences แล้วคัดเลือก, ผสมพันธุ์, และกลายพันธุ์ เพื่อ "พัฒนา" ให้ได้ Sequence ที่ดีที่สุด
        
        *   **🤖 Reinforcement Learning (Q-Learning):** **[โมเดลใหม่ระดับสูง]** เป็นเทคนิคจากสาย AI ที่ให้ "Agent" เรียนรู้จากการลองผิดลองถูกในแต่ละ Window มันจะสร้าง "นโยบาย" การตัดสินใจ (Hold/Rebalance) ขึ้นมาเองโดยมีเป้าหมายเพื่อสร้างผลตอบแทนรวมสูงสุดในระยะยาว ไม่ใช่แค่กำไรในแต่ละวัน Agent จะเรียนรู้ว่าใน "สถานะ" ของตลาดแบบไหน ควรจะทำ "Action" อะไร

        ---
        #### แท็บอื่นๆ
        
        *   **📊 Analytics:** ใช้วิเคราะห์ไฟล์ผลลัพธ์ (.csv) ที่ดาวน์โหลดจากการทดสอบ เพื่อดูภาพรวมและจำลองการเทรดจริงด้วย "Stitched DNA"
        *   **🌱 Manual/Forward:** ใช้ทดสอบ Action Sequence ที่สร้างจาก Seed ที่กำหนดเอง เหมาะสำหรับการทำ Forward Testing หรือการทดลองสมมติฐานเฉพาะ
        """)

if __name__ == "__main__":
    main()
