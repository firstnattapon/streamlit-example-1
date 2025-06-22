# main

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
    HYBRID_GA = "Hybrid (Random + GA)"
    ORIGINAL_DNA = "Original DNA (from Seed)"
    EVOLVED_DNA = "Evolved DNA (by GA)"


def load_config(filepath: str = "dynamic_seed_config.json") -> Dict[str, Any]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        st.warning(f"⚠️ ไม่พบหรือไฟล์ '{filepath}' ไม่ถูกต้อง จะใช้ค่าเริ่มต้นแทน")
        return {
            "assets": ["FFWM", "NEGG", "RIVN", "BTC-USD", "NVDA", "TSLA", "META"],
            "default_settings": { "selected_ticker": "FFWM", "start_date": "2024-01-01", "window_size": 30, "num_seeds": 10000, "max_workers": 8, "ga_population_size": 50, "ga_generations": 20, "ga_master_seed": 42 },
            "manual_seed_by_asset": { "default": [{'seed': 999, 'size': 50, 'tail': 15}], "FFWM": [{'seed': 1234, 'size': 60, 'tail': 30}, {'seed': 7777, 'size': 30, 'tail': 10}]}
        }

def initialize_session_state(config: Dict[str, Any]):
    defaults = config.get('default_settings', {})
    if 'test_ticker' not in st.session_state: st.session_state.test_ticker = defaults.get('selected_ticker', 'FFWM')
    if 'start_date' not in st.session_state:
        try: st.session_state.start_date = datetime.strptime(defaults.get('start_date', '2024-01-01'), '%Y-%m-%d').date()
        except ValueError: st.session_state.start_date = datetime(2024, 1, 1).date()
    if 'end_date' not in st.session_state: st.session_state.end_date = datetime.now().date()
    if 'window_size' not in st.session_state: st.session_state.window_size = defaults.get('window_size', 30)
    if 'num_seeds' not in st.session_state: st.session_state.num_seeds = defaults.get('num_seeds', 10000)
    if 'max_workers' not in st.session_state: st.session_state.max_workers = defaults.get('max_workers', 8)
    if 'ga_population_size' not in st.session_state: st.session_state.ga_population_size = defaults.get('ga_population_size', 50)
    if 'ga_generations' not in st.session_state: st.session_state.ga_generations = defaults.get('ga_generations', 20)
    if 'ga_master_seed' not in st.session_state: st.session_state.ga_master_seed = defaults.get('ga_master_seed', 42)
    if 'df_for_analysis' not in st.session_state: st.session_state.df_for_analysis = None
    if 'manual_seed_lines' not in st.session_state:
        initial_ticker = st.session_state.get('test_ticker', defaults.get('selected_ticker', 'FFWM'))
        presets_by_asset = config.get("manual_seed_by_asset", {})
        default_presets = presets_by_asset.get("default", [{'seed': 999, 'size': 50, 'tail': 15}])
        st.session_state.manual_seed_lines = presets_by_asset.get(initial_ticker, default_presets)

def on_ticker_change_callback(config: Dict[str, Any]):
    selected_ticker = st.session_state.get("manual_ticker_key")
    if not selected_ticker: return
    presets_by_asset = config.get("manual_seed_by_asset", {})
    default_presets = presets_by_asset.get("default", [{'seed': 999, 'size': 50, 'tail': 15}])
    st.session_state.manual_seed_lines = presets_by_asset.get(selected_ticker, default_presets)


# ==============================================================================
# 2. Core Calculation & Data Functions
# ==============================================================================
@st.cache_data(ttl=3600)
def get_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        data = yf.Ticker(ticker).history(start=start_date, end=end_date)[['Close']]
        if data.empty: return pd.DataFrame()
        if data.index.tz is None: data = data.tz_localize('UTC').tz_convert('Asia/Bangkok')
        else: data = data.tz_convert('Asia/Bangkok')
        return data
    except Exception as e:
        st.error(f"❌ ไม่สามารถดึงข้อมูล {ticker} ได้: {str(e)}"); return pd.DataFrame()

@njit(cache=True)
def _calculate_net_profit_numba(action_array: np.ndarray, price_array: np.ndarray, fix: int = 1500) -> float:
    n = len(action_array)
    if n == 0 or len(price_array) == 0 or n > len(price_array): return -np.inf

    action_array_calc = action_array.copy()
    action_array_calc[0] = 1
    initial_price = price_array[0]
    initial_capital = fix * 2.0
    refer_net = -fix * np.log(initial_price / price_array[n-1])
    cash = float(fix)
    amount = float(fix) / initial_price
    
    for i in range(1, n):
        curr_price = price_array[i]
        if action_array_calc[i] != 0:
            cash += amount * curr_price - fix
            amount = fix / curr_price
            
    final_sumusd = cash + (amount * price_array[n-1])
    net = final_sumusd - refer_net - initial_capital
    return net

def run_simulation(prices: List[float], actions: List[int], fix: int = 1500) -> pd.DataFrame:
    @njit
    def _full_sim_numba(action_arr, price_arr, fix_val):
        n = len(action_arr)
        empty = np.empty(0, dtype=np.float64)
        if n == 0 or len(price_arr) == 0: return empty, empty, empty, empty, empty, empty
        
        action_calc = action_arr.copy(); action_calc[0] = 1
        amount = np.empty(n, dtype=np.float64); buffer = np.zeros(n, dtype=np.float64)
        cash = np.empty(n, dtype=np.float64); asset_val = np.empty(n, dtype=np.float64)
        sumusd_val = np.empty(n, dtype=np.float64)
        
        init_price = price_arr[0]; amount[0] = fix_val / init_price; cash[0] = fix_val
        asset_val[0] = amount[0] * init_price; sumusd_val[0] = cash[0] + asset_val[0]
        refer = -fix_val * np.log(init_price / price_arr[:n])
        
        for i in range(1, n):
            curr_price = price_arr[i]
            if action_calc[i] == 0: amount[i] = amount[i-1]; buffer[i] = 0.0
            else: amount[i] = fix_val / curr_price; buffer[i] = amount[i-1] * curr_price - fix_val
            cash[i] = cash[i-1] + buffer[i]; asset_val[i] = amount[i] * curr_price
            sumusd_val[i] = cash[i] + asset_val[i]
        return buffer, sumusd_val, cash, asset_val, amount, refer

    if not prices or not actions: return pd.DataFrame()
    min_len = min(len(prices), len(actions))
    prices_arr = np.array(prices[:min_len], dtype=np.float64)
    actions_arr = np.array(actions[:min_len], dtype=np.int32)
    buffer, sumusd, cash, asset_value, amount, refer = _full_sim_numba(actions_arr, prices_arr, fix)
    if len(sumusd) == 0: return pd.DataFrame()
    initial_capital = sumusd[0]
    return pd.DataFrame({
        'price': prices_arr, 'action': actions_arr, 'buffer': np.round(buffer, 2),
        'sumusd': np.round(sumusd, 2), 'cash': np.round(cash, 2),
        'asset_value': np.round(asset_value, 2), 'amount': np.round(amount, 2),
        'refer': np.round(refer + initial_capital, 2),
        'net': np.round(sumusd - refer - initial_capital, 2)
    })

# ==============================================================================
# 3. Strategy Action Generation
# ==============================================================================
def generate_actions_rebalance_daily(num_days: int) -> np.ndarray: return np.ones(num_days, dtype=np.int32)

def generate_actions_perfect_foresight(prices: List[float], fix: int = 1500) -> np.ndarray:
    price_arr = np.asarray(prices, dtype=np.float64); n = len(price_arr)
    if n < 2: return np.ones(n, dtype=int)
    dp = np.zeros(n, dtype=np.float64); path = np.zeros(n, dtype=int); dp[0] = float(fix * 2)
    for i in range(1, n):
        j_indices = np.arange(i)
        profits = fix * ((price_arr[i] / price_arr[j_indices]) - 1)
        current_sumusd = dp[j_indices] + profits
        best_idx = np.argmax(current_sumusd); dp[i] = current_sumusd[best_idx]; path[i] = j_indices[best_idx]
    actions = np.zeros(n, dtype=int); current_day = np.argmax(dp)
    while current_day > 0: actions[current_day] = 1; current_day = path[current_day]
    actions[0] = 1
    return actions

def find_best_seed_for_window(prices_window: np.ndarray, num_seeds_to_try: int, max_workers: int) -> Tuple[int, float, np.ndarray]:
    window_len = len(prices_window)
    if window_len < 2: return 1, 0.0, np.ones(window_len, dtype=int)
    def evaluate_seed_batch(seed_batch: np.ndarray) -> List[Tuple[int, float]]:
        results = []
        for seed in seed_batch:
            rng = np.random.default_rng(seed)
            actions = rng.integers(0, 2, size=window_len)
            net = _calculate_net_profit_numba(actions, prices_window)
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
                if final_net > max_net: max_net, best_seed = final_net, seed
    if best_seed >= 0:
        rng_best = np.random.default_rng(best_seed)
        best_actions = rng_best.integers(0, 2, size=window_len)
    else: best_seed, best_actions, max_net = 1, np.ones(window_len, dtype=int), 0.0
    best_actions[0] = 1
    return best_seed, max_net, best_actions

def generate_actions_sliding_window(ticker_data: pd.DataFrame, window_size: int, num_seeds_to_try: int, max_workers: int) -> Tuple[np.ndarray, pd.DataFrame]:
    prices = ticker_data['Close'].to_numpy(); n = len(prices)
    final_actions = np.array([], dtype=int); window_details_list = []
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="กำลังประมวลผล Sliding Windows...")
    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n); prices_window = prices[start_index:end_index]
        if len(prices_window) == 0: continue
        best_seed, max_net, best_actions = find_best_seed_for_window(prices_window, num_seeds_to_try, max_workers)
        final_actions = np.concatenate((final_actions, best_actions))
        start_date = ticker_data.index[start_index]; end_date = ticker_data.index[end_index-1]
        detail = {
            'window_number': i + 1, 'timeline': f"{start_date.strftime('%Y-%m-%d')} ถึง {end_date.strftime('%Y-%m-%d')}",
            'best_seed': best_seed, 'max_net': round(max_net, 2),
            'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2),
            'action_count': int(np.sum(best_actions)), 'window_size': len(prices_window),
            'action_sequence': best_actions.tolist()
        }
        window_details_list.append(detail)
        progress_bar.progress((i + 1) / num_windows, text=f"ประมวลผล Window {i+1}/{num_windows}")
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)

def refine_solution_with_ga(prices_window: np.ndarray, initial_chromosome: np.ndarray, population_size: int, generations: int, seed: int, mutation_rate: float = 0.01, initial_population_mutation_rate: float = 0.1) -> Tuple[float, np.ndarray]:
    window_len = len(prices_window)
    if window_len < 2: return 0.0, np.ones(window_len, dtype=np.int32)
    rng = np.random.default_rng(seed)
    population = np.tile(initial_chromosome, (population_size, 1))
    mutation_mask = rng.random(population.shape) < initial_population_mutation_rate
    population[mutation_mask] = 1 - population[mutation_mask]
    population[0] = initial_chromosome; population[:, 0] = 1
    for _ in range(generations):
        fitness_scores = np.array([_calculate_net_profit_numba(chromo, prices_window) for chromo in population])
        num_parents = population_size // 2
        parent_indices = np.argsort(fitness_scores)[-num_parents:]
        parents = population[parent_indices]
        offspring = np.empty((population_size - num_parents, window_len), dtype=np.int32)
        parent_choices = rng.choice(num_parents, size=(len(offspring), 2), replace=True)
        for k, (p1_idx, p2_idx) in enumerate(parent_choices):
            crossover_point = rng.integers(1, window_len)
            offspring[k, :crossover_point] = parents[p1_idx, :crossover_point]
            offspring[k, crossover_point:] = parents[p2_idx, crossover_point:]
        mutation_mask = rng.random(offspring.shape) < mutation_rate
        offspring[mutation_mask] = 1 - offspring[mutation_mask]; offspring[:, 0] = 1
        population[0:num_parents] = parents; population[num_parents:] = offspring
    final_fitness = np.array([_calculate_net_profit_numba(chromo, prices_window) for chromo in population])
    best_idx = np.argmax(final_fitness)
    return final_fitness[best_idx], population[best_idx]

def generate_actions_sliding_window_hybrid_ga(ticker_data: pd.DataFrame, window_size: int, num_seeds_to_try: int, max_workers: int, population_size: int, generations: int, master_seed: int) -> Tuple[np.ndarray, pd.DataFrame]:
    prices = ticker_data['Close'].to_numpy(); n = len(prices)
    final_actions = np.array([], dtype=np.int32); window_details_list = []
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="กำลังประมวลผล Hybrid (Random + GA) Strategy...")
    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n); prices_window = prices[start_index:end_index]
        if len(prices_window) < 2: continue
        progress_bar.progress((i + 0.0) / num_windows, text=f"Window {i+1}/{num_windows} - Phase 1: Random Search...")
        initial_seed, initial_net, initial_actions = find_best_seed_for_window(prices_window, num_seeds_to_try, max_workers)
        progress_bar.progress((i + 0.5) / num_windows, text=f"Window {i+1}/{num_windows} - Phase 2: GA Refinement...")
        refined_net, refined_actions = refine_solution_with_ga(prices_window, initial_actions, population_size, generations, seed=master_seed + i)
        final_actions = np.concatenate((final_actions, refined_actions))
        start_date = ticker_data.index[start_index]; end_date = ticker_data.index[end_index-1]
        detail = {
            'window_number': i + 1, 'timeline': f"{start_date.strftime('%Y-%m-%d')} ถึง {end_date.strftime('%Y-%m-%d')}",
            'initial_best_seed': initial_seed, 'initial_max_net': round(initial_net, 2),
            'refined_max_net': round(refined_net, 2), 'net_improvement': round(refined_net - initial_net, 2),
            'final_action_count': int(np.sum(refined_actions)), 'window_size': len(prices_window)
        }
        window_details_list.append(detail)
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)

def evolve_sequence_with_ga(initial_sequence: np.ndarray, prices: np.ndarray, population_size: int, generations: int, ga_seed: int, mutation_rate: float = 0.01, initial_mutation_rate: float = 0.10, progress_bar=None) -> Tuple[np.ndarray, float]:
    sequence_length = len(initial_sequence)
    rng = np.random.default_rng(ga_seed)
    population = np.tile(initial_sequence, (population_size, 1))
    mutation_mask = rng.random(population.shape) < initial_mutation_rate
    population[mutation_mask] = 1 - population[mutation_mask]
    population[0] = initial_sequence; population[:, 0] = 1
    for gen in range(generations):
        fitness_scores = np.array([_calculate_net_profit_numba(chromo, prices) for chromo in population])
        num_parents = population_size // 2
        parent_indices = np.argsort(fitness_scores)[-num_parents:]
        parents = population[parent_indices]
        offspring = np.empty((population_size - num_parents, sequence_length), dtype=np.int32)
        parent_choices = rng.choice(num_parents, size=(len(offspring), 2), replace=True)
        for k, (p1_idx, p2_idx) in enumerate(parent_choices):
            crossover_point = rng.integers(1, sequence_length)
            offspring[k, :crossover_point] = parents[p1_idx, :crossover_point]
            offspring[k, crossover_point:] = parents[p2_idx, crossover_point:]
        mutation_mask = rng.random(offspring.shape) < mutation_rate
        offspring[mutation_mask] = 1 - offspring[mutation_mask]; offspring[:, 0] = 1
        population[0:num_parents] = parents; population[num_parents:] = offspring
        if progress_bar: progress_bar.progress((gen + 1) / generations, text=f"Evolving Generation {gen+1}/{generations}")
    final_fitness_scores = np.array([_calculate_net_profit_numba(chromo, prices) for chromo in population])
    best_idx = np.argmax(final_fitness_scores)
    return population[best_idx], final_fitness_scores[best_idx]

# ==============================================================================
# 4. UI Rendering Functions
# ==============================================================================
def display_comparison_charts(results: Dict[str, pd.DataFrame], chart_title: str = '📊 เปรียบเทียบกำไรสุทธิ (Net Profit)'):
    if not results: st.warning("ไม่มีข้อมูลสำหรับสร้างกราฟเปรียบเทียบ"); return
    valid_dfs = {name: df for name, df in results.items() if not df.empty and 'net' in df.columns}
    if not valid_dfs: st.warning("ไม่มีข้อมูล 'net' สำหรับสร้างกราฟ"); return
    try: longest_index = max((df.index for df in valid_dfs.values()), key=len, default=None)
    except ValueError: longest_index = None
    if longest_index is None: st.warning("ไม่มีข้อมูลสำหรับสร้างกราฟเปรียบเทียบ"); return
    chart_data = pd.DataFrame(index=longest_index)
    for name, df in valid_dfs.items(): chart_data[name] = df['net'].reindex(longest_index).ffill()
    st.write(chart_title); st.line_chart(chart_data)

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
    if st.session_state.start_date >= st.session_state.end_date: st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
    st.divider()
    st.subheader("พารามิเตอร์สำหรับกลยุทธ์ Sliding Window")
    c1, c2, c3 = st.columns(3)
    st.session_state.window_size = c1.number_input("ขนาด Window (วัน)", min_value=2, value=st.session_state.window_size)
    st.session_state.num_seeds = c2.number_input("จำนวน Seeds (Random Search)", min_value=100, value=st.session_state.num_seeds, format="%d")
    st.session_state.max_workers = c3.number_input("จำนวน Workers (CPU Cores)", min_value=1, max_value=16, value=st.session_state.max_workers)
    st.subheader("พารามิเตอร์สำหรับ Genetic Algorithm")
    ga_c1, ga_c2, ga_c3 = st.columns(3)
    st.session_state.ga_population_size = ga_c1.number_input("ขนาดประชากร (Population Size)", min_value=10, value=st.session_state.ga_population_size)
    st.session_state.ga_generations = ga_c2.number_input("จำนวนรุ่น (Generations)", min_value=5, value=st.session_state.ga_generations)
    st.session_state.ga_master_seed = ga_c3.number_input("Master Seed for GA", value=st.session_state.ga_master_seed, format="%d")

def render_random_seed_tab():
    st.write("---")
    st.markdown(f"### 🚀 {Strategy.SLIDING_WINDOW}")
    st.info("ค้นหา Seed ที่ให้ผลตอบแทนดีที่สุดในแต่ละ Window โดยใช้การสุ่ม (Random Search)")
    if st.button("🚀 เริ่มทดสอบ Best Seed", type="primary"):
        if st.session_state.start_date >= st.session_state.end_date: st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้อง"); return
        ticker = st.session_state.test_ticker
        with st.spinner(f"กำลังดึงข้อมูลและประมวลผลสำหรับ {ticker}..."):
            ticker_data = get_ticker_data(ticker, str(st.session_state.start_date), str(st.session_state.end_date))
            if ticker_data.empty: st.error("ไม่พบข้อมูลสำหรับ Ticker และช่วงวันที่ที่เลือก"); return
            actions, df_windows = generate_actions_sliding_window(ticker_data, st.session_state.window_size, st.session_state.num_seeds, st.session_state.max_workers)
            prices = ticker_data['Close'].to_numpy()
            results = {
                Strategy.SLIDING_WINDOW: run_simulation(prices.tolist(), actions.tolist()),
                Strategy.REBALANCE_DAILY: run_simulation(prices.tolist(), generate_actions_rebalance_daily(len(prices)).tolist()),
                Strategy.PERFECT_FORESIGHT: run_simulation(prices.tolist(), generate_actions_perfect_foresight(prices.tolist()).tolist())
            }
            for name, df in results.items():
                if not df.empty: df.index = ticker_data.index[:len(df)]
        st.success("การทดสอบเสร็จสมบูรณ์!"); display_comparison_charts(results)
        st.write("📈 **สรุปผลการค้นหา Best Seed**")
        if not df_windows.empty:
            total_net = df_windows['max_net'].sum(); total_actions = df_windows['action_count'].sum()
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Windows", df_windows.shape[0])
            col2.metric("Total Actions", f"{total_actions}/{len(prices)}")
            col3.metric("Total Net (Sum)", f"${total_net:,.2f}")
            st.dataframe(df_windows[['window_number', 'timeline', 'best_seed', 'max_net', 'action_count']], use_container_width=True)
            st.download_button("📥 Download Details (CSV)", df_windows.to_csv(index=False), f'best_seed_{ticker}.csv', 'text/csv')

def render_hybrid_ga_tab():
    st.write("---")
    st.markdown(f"### 🚀+🧬 {Strategy.HYBRID_GA}")
    st.info("ทำงาน 2 ขั้นตอน: 1. ใช้ Random Search หา `action_sequence` ที่ดีที่สุด 2. นำ Sequence นั้นมาเป็น 'ต้นแบบ' ให้ GA พัฒนาต่อยอด")
    if st.button(f"🚀 เริ่มทดสอบ {Strategy.HYBRID_GA}", type="primary"):
        if st.session_state.start_date >= st.session_state.end_date: st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้อง"); return
        ticker = st.session_state.test_ticker
        with st.spinner(f"กำลังดึงข้อมูลและประมวลผลสำหรับ {ticker}..."):
            ticker_data = get_ticker_data(ticker, str(st.session_state.start_date), str(st.session_state.end_date))
            if ticker_data.empty: st.error("ไม่พบข้อมูลสำหรับ Ticker และช่วงวันที่ที่เลือก"); return
            actions, df_windows = generate_actions_sliding_window_hybrid_ga(ticker_data, st.session_state.window_size, st.session_state.num_seeds, st.session_state.max_workers, st.session_state.ga_population_size, st.session_state.ga_generations, st.session_state.ga_master_seed)
            prices = ticker_data['Close'].to_numpy()
            results = {
                Strategy.HYBRID_GA: run_simulation(prices.tolist(), actions.tolist()),
                Strategy.REBALANCE_DAILY: run_simulation(prices.tolist(), generate_actions_rebalance_daily(len(prices)).tolist()),
                Strategy.PERFECT_FORESIGHT: run_simulation(prices.tolist(), generate_actions_perfect_foresight(prices.tolist()).tolist())
            }
            for name, df in results.items():
                if not df.empty: df.index = ticker_data.index[:len(df)]
        st.success("การทดสอบเสร็จสมบูรณ์!"); display_comparison_charts(results)
        st.write("📈 **สรุปผลการค้นหาด้วย Hybrid GA Strategy**")
        if not df_windows.empty:
            total_initial_net = df_windows['initial_max_net'].sum()
            total_refined_net = df_windows['refined_max_net'].sum()
            improvement_pct = ((total_refined_net / total_initial_net) - 1) * 100 if total_initial_net != 0 else float('inf') if total_refined_net > 0 else 0
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Net (Before GA)", f"${total_initial_net:,.2f}")
            col2.metric("Total Net (After GA)", f"${total_refined_net:,.2f}")
            col3.metric("Improvement", f"{improvement_pct:.2f}%", delta=f"${total_refined_net - total_initial_net:,.2f}")
            st.dataframe(df_windows[['window_number', 'timeline', 'initial_best_seed', 'initial_max_net', 'refined_max_net', 'net_improvement']], use_container_width=True)
            st.download_button("📥 Download Details (CSV)", df_windows.to_csv(index=False), f'hybrid_ga_{ticker}.csv', 'text/csv')

def render_evolve_dna_tab():
    st.write("---")
    st.markdown("### 🧬 Evolve DNA (GA)")
    st.markdown("นำ `action_sequence` ที่สร้างจาก **Seed ต้นแบบ** มาเป็นจุดเริ่มต้นให้กับ **Genetic Algorithm (GA)** เพื่อค้นหา Sequence ที่ถูกปรับปรุงให้ดีที่สุดภายใต้สภาวะตลาดที่กำหนด")
    with st.container(border=True):
        st.subheader("1. กำหนด DNA ต้นแบบ (Blueprint)")
        c1, c2 = st.columns(2)
        blueprint_seed = c1.number_input("Input Seed for Blueprint", value=16942, min_value=1, format="%d", key="blueprint_seed")
        blueprint_size = c2.number_input("Sequence Size (length of DNA)", value=60, min_value=10, key="blueprint_size")
    with st.container(border=True):
        st.subheader("2. กำหนดพารามิเตอร์ Genetic Algorithm")
        c1, c2, c3 = st.columns(3)
        ga_pop_size = c1.number_input("Population Size", value=2000, min_value=100, step=100, key="evolve_pop_size")
        ga_generations = c2.number_input("Generations", value=200, min_value=10, step=10, key="evolve_generations")
        ga_seed = c3.number_input("Master Seed for GA", value=42, format="%d", key="evolve_ga_seed")
    with st.container(border=True):
        st.subheader("3. กำหนดสนามทดสอบ (Simulation Environment)")
        c1, c2 = st.columns(2)
        sim_ticker = c1.selectbox("Ticker", options=load_config().get('assets', ['FFWM']), index=0, key="sim_ticker_evolve")
        sim_days_ago = c2.number_input("ใช้ข้อมูลย้อนหลัง (วันทำการ)", value=blueprint_size, min_value=blueprint_size, key="sim_days_evolve", help=f"ต้องมีค่าอย่างน้อยเท่ากับ Sequence Size ({blueprint_size})")
    if st.button("🧬 Start Evolution", type="primary", key="start_evolve_btn"):
        with st.spinner("กำลังเตรียมข้อมูลและวิวัฒนาการ DNA..."):
            rng_base = np.random.default_rng(blueprint_seed)
            initial_actions = rng_base.integers(0, 2, size=blueprint_size); initial_actions[0] = 1
            st.info(f"กำลังดึงข้อมูลราคา {sim_ticker} ย้อนหลัง {sim_days_ago} วันทำการ...")
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
            start_date = (pd.Timestamp.now() - pd.Timedelta(days=sim_days_ago * 2)).strftime('%Y-%m-%d')
            ticker_data = get_ticker_data(sim_ticker, start_date, end_date)
            if ticker_data.empty or len(ticker_data) < sim_days_ago: st.error(f"ไม่สามารถดึงข้อมูลราคาได้ครบ {sim_days_ago} วัน"); return
            prices_full = ticker_data['Close'].to_numpy()[-sim_days_ago:]
            prices_for_sim = prices_full[:blueprint_size]
            initial_net = _calculate_net_profit_numba(initial_actions, prices_for_sim)
            evo_progress_bar = st.progress(0, text="Starting Evolution...")
            evolved_actions, final_net = evolve_sequence_with_ga(initial_actions, prices_for_sim, ga_pop_size, ga_generations, ga_seed, progress_bar=evo_progress_bar)
            evo_progress_bar.empty()
            st.info("กำลังจำลองผลลัพธ์เพื่อสร้างกราฟ...")
            results_evo = {}; sim_dates = ticker_data.index[-sim_days_ago:][:blueprint_size]
            df_initial = run_simulation(prices_for_sim.tolist(), initial_actions.tolist())
            df_evolved = run_simulation(prices_for_sim.tolist(), evolved_actions.tolist())
            df_max = run_simulation(prices_for_sim.tolist(), generate_actions_perfect_foresight(prices_for_sim.tolist()).tolist())
            df_min = run_simulation(prices_for_sim.tolist(), generate_actions_rebalance_daily(len(prices_for_sim)).tolist())
            if not df_initial.empty: df_initial.index = sim_dates; results_evo[Strategy.ORIGINAL_DNA] = df_initial
            if not df_evolved.empty: df_evolved.index = sim_dates; results_evo[Strategy.EVOLVED_DNA] = df_evolved
            if not df_max.empty: df_max.index = sim_dates; results_evo[Strategy.PERFECT_FORESIGHT] = df_max
            if not df_min.empty: df_min.index = sim_dates; results_evo[Strategy.REBALANCE_DAILY] = df_min
        st.success("กระบวนการวิวัฒนาการเสร็จสมบูรณ์!"); st.write("---")
        st.subheader("📊 เปรียบเทียบประสิทธิภาพ (Net Profit)"); display_comparison_charts(results_evo)
        st.subheader("📈 สรุปผลลัพธ์สุดท้าย"); metric_cols = st.columns(2)
        metric_cols[0].metric(f"{Strategy.ORIGINAL_DNA} (Seed: {blueprint_seed})", f"${initial_net:,.2f}")
        metric_cols[1].metric(f"{Strategy.EVOLVED_DNA} (GA)", f"${final_net:,.2f}", delta=f"{final_net - initial_net:,.2f}")
        st.write("---"); st.subheader("🔬 รายละเอียดการเปลี่ยนแปลง DNA")
        dna_detail_df = pd.DataFrame({'Position': range(blueprint_size), 'Original': initial_actions, 'Evolved': evolved_actions, 'Changed': initial_actions != evolved_actions})
        st.dataframe(dna_detail_df, use_container_width=True, height=300)
        changes = dna_detail_df['Changed'].sum()
        st.markdown(f"**จำนวนยีนที่เปลี่ยนแปลง:** `{changes}` จาก `{blueprint_size}` ตำแหน่ง ({changes/blueprint_size:.2%})")

def render_analytics_tab():
    st.header("📊 Advanced Analytics Dashboard")
    # ... (Analytics code can be added here later if needed)
    st.info("ฟีเจอร์นี้ถูกปิดใช้งานชั่วคราวเพื่อลดความซับซ้อนของแอป")

def render_manual_seed_tab(config: Dict[str, Any]):
    st.header("🌱 Forward Rolling Comparator")
    st.markdown("สร้างและเปรียบเทียบ Action Sequences โดยการตัดส่วนท้าย (`tail`) จาก Seed ที่กำหนด")
    with st.container(border=True):
        st.subheader("1. กำหนดค่า Input")
        col1, col2 = st.columns([1, 2])
        asset_list = config.get('assets', ['FFWM'])
        try: default_index = asset_list.index(st.session_state.get('manual_ticker_key', st.session_state.test_ticker))
        except (ValueError, KeyError): default_index = 0
        manual_ticker = col1.selectbox("เลือก Ticker", options=asset_list, index=default_index, key="manual_ticker_key", on_change=on_ticker_change_callback, args=(config,))
        c1, c2 = col2.columns(2)
        manual_start_date = c1.date_input("วันที่เริ่มต้น", value=st.session_state.start_date, key="manual_start_compare_tail")
        manual_end_date = c2.date_input("วันที่สิ้นสุด", value=st.session_state.end_date, key="manual_end_compare_tail")
        if manual_start_date >= manual_end_date: st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
        st.divider(); st.write("**กำหนดกลยุทธ์ (Seed/Size/Tail) ที่ต้องการเปรียบเทียบ:**")
        for i, line in enumerate(st.session_state.manual_seed_lines):
            cols = st.columns([1, 2, 2, 2])
            cols[0].write(f"**Line {i+1}**")
            line['seed'] = cols[1].number_input("Input Seed", value=line.get('seed', 1), min_value=0, key=f"seed_compare_tail_{i}")
            line['size'] = cols[2].number_input("Size", value=line.get('size', 60), min_value=1, key=f"size_compare_tail_{i}")
            line['tail'] = cols[3].number_input("Tail", value=line.get('tail', 10), min_value=1, max_value=line.get('size', 60), key=f"tail_compare_tail_{i}")
        b_col1, b_col2, _ = st.columns([1,1,4])
        if b_col1.button("➕ Add Line"): st.session_state.manual_seed_lines.append({'seed': np.random.randint(1, 10000), 'size': 50, 'tail': 20}); st.rerun()
        if b_col2.button("➖ Remove Line"):
            if len(st.session_state.manual_seed_lines) > 1: st.session_state.manual_seed_lines.pop(); st.rerun()
            else: st.warning("ต้องมีอย่างน้อย 1 line")
    st.write("---")
    if st.button("📈 เปรียบเทียบ Seeds", type="primary", key="compare_manual_seeds_btn"):
        if manual_start_date >= manual_end_date: st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้อง"); return
        with st.spinner("กำลังดึงข้อมูลและจำลองการเทรด..."):
            ticker_data = get_ticker_data(manual_ticker, str(manual_start_date), str(manual_end_date))
            if ticker_data.empty: st.error(f"ไม่พบข้อมูลสำหรับ {manual_ticker}"); return
            prices = ticker_data['Close'].to_numpy()
            results = {}; max_sim_len = 0
            for i, line_info in enumerate(st.session_state.manual_seed_lines):
                input_seed, size_seed, tail_seed = line_info['seed'], line_info['size'], line_info['tail']
                if tail_seed > size_seed: st.error(f"Line {i+1}: Tail ({tail_seed}) ต้องไม่มากกว่า Size ({size_seed})"); return
                rng = np.random.default_rng(input_seed)
                actions = rng.integers(0, 2, size=size_seed)[-tail_seed:].tolist()
                sim_len = min(len(prices), len(actions))
                if sim_len == 0: continue
                df_line = run_simulation(prices[:sim_len].tolist(), actions[:sim_len])
                if not df_line.empty:
                    df_line.index = ticker_data.index[:sim_len]
                    results[f"Seed {input_seed} (Tail {tail_seed})"] = df_line
                    max_sim_len = max(max_sim_len, sim_len)
            if not results: st.error("ไม่สามารถสร้างผลลัพธ์ได้"); return
            if max_sim_len > 0:
                prices_bench = prices[:max_sim_len].tolist()
                df_max = run_simulation(prices_bench, generate_actions_perfect_foresight(prices_bench).tolist())
                df_min = run_simulation(prices_bench, generate_actions_rebalance_daily(max_sim_len).tolist())
                if not df_max.empty: df_max.index = ticker_data.index[:max_sim_len]; results[Strategy.PERFECT_FORESIGHT] = df_max
                if not df_min.empty: df_min.index = ticker_data.index[:max_sim_len]; results[Strategy.REBALANCE_DAILY] = df_min
            st.success("การเปรียบเทียบเสร็จสมบูรณ์!"); display_comparison_charts(results, chart_title="📊 Performance Comparison")
            final_results = [{'name': name, 'net': results[name]['net'].iloc[-1]} for name in results if not results[name].empty]
            if final_results: st.subheader("สรุปผลลัพธ์สุดท้าย"); st.dataframe(final_results)


# ==============================================================================
# 5. Main Application
# ==============================================================================
def main():
    st.set_page_config(page_title="DNA Strategy Lab", page_icon="🧬", layout="wide")
    st.markdown("### 🧬 DNA Strategy Lab (Best Seed & GA Evolution)")
    st.caption("เครื่องมือทดลองและพัฒนา Action Sequence ด้วย Numba-Accelerated GA")

    config = load_config()
    initialize_session_state(config)

    tab_list = [
        "⚙️ การตั้งค่า",
        "🚀 Best Seed (Random)",
        "🚀+🧬 Hybrid (Random + GA)",
        "🧬 Evolve DNA (GA)",
        "📊 Advanced Analytics",
        "🌱 Forward Rolling Comparator"
    ]
    
    render_functions = {
        "⚙️ การตั้งค่า": render_settings_tab,
        "🚀 Best Seed (Random)": render_random_seed_tab,
        "🚀+🧬 Hybrid (Random + GA)": render_hybrid_ga_tab,
        "🧬 Evolve DNA (GA)": render_evolve_dna_tab,
        "📊 Advanced Analytics": render_analytics_tab,
        "🌱 Forward Rolling Comparator": render_manual_seed_tab
    }
    
    tabs = st.tabs(tab_list)

    for i, tab_name in enumerate(tab_list):
        with tabs[i]:
            # FIX: Check if the function needs 'config' argument
            if tab_name in ["⚙️ การตั้งค่า", "🌱 Forward Rolling Comparator"]:
                render_functions[tab_name](config)
            else:
                render_functions[tab_name]()

if __name__ == "__main__":
    main()
