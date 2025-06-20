# ก่อนรัน ต้องติดตั้ง: pip install dtw-python
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
from numba import njit
from dtw import dtw #! DTW: Import Dynamic Time Warping

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
    PATTERN_MATCHING_DTW = "Pattern Matching (DTW) Sliding Window"


def load_config(filepath: str = "dynamic_seed_config.json") -> Dict[str, Any]:
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
                "pm_lookback_period": 750, "pm_num_analogs": 5
            },
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
    st.session_state.manual_seed_lines = presets_by_asset.get(selected_ticker, default_presets)

def initialize_session_state(config: Dict[str, Any]):
    defaults = config.get('default_settings', {})
    if 'test_ticker' not in st.session_state: st.session_state.test_ticker = defaults.get('selected_ticker', 'FFWM')
    if 'start_date' not in st.session_state:
        try: st.session_state.start_date = datetime.strptime(defaults.get('start_date', '2025-01-01'), '%Y-%m-%d').date()
        except ValueError: st.session_state.start_date = datetime(2025, 1, 1).date()
    if 'end_date' not in st.session_state: st.session_state.end_date = datetime.now().date()
    if 'window_size' not in st.session_state: st.session_state.window_size = defaults.get('window_size', 30)
    if 'num_seeds' not in st.session_state: st.session_state.num_seeds = defaults.get('num_seeds', 30000)
    if 'max_workers' not in st.session_state: st.session_state.max_workers = defaults.get('max_workers', 8)
    if 'ga_population_size' not in st.session_state: st.session_state.ga_population_size = defaults.get('ga_population_size', 50)
    if 'ga_generations' not in st.session_state: st.session_state.ga_generations = defaults.get('ga_generations', 20)
    
    # Pattern Matching settings
    if 'pm_lookback_period' not in st.session_state: st.session_state.pm_lookback_period = defaults.get('pm_lookback_period', 750)
    if 'pm_num_analogs' not in st.session_state: st.session_state.pm_num_analogs = defaults.get('pm_num_analogs', 5)

    if 'df_for_analysis' not in st.session_state: st.session_state.df_for_analysis = None
    if 'manual_seed_lines' not in st.session_state:
        initial_ticker = defaults.get('selected_ticker', 'FFWM')
        presets_by_asset = config.get("manual_seed_by_asset", {})
        default_presets = presets_by_asset.get("default", [{'seed': 999, 'size': 50, 'tail': 15}])
        st.session_state.manual_seed_lines = presets_by_asset.get(initial_ticker, default_presets)

# ==============================================================================
# 2. Core Calculation & Data Functions (No Changes)
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
    prices, actions = prices[:len(actions)], actions[:len(prices)]
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
# --- OMITTED FOR BREVITY: All functions for Benchmarks, Random Seed, Chaotic, GA are here ---
# 3.0 Benchmarks
def generate_actions_rebalance_daily(num_days: int) -> np.ndarray: return np.ones(num_days, dtype=np.int32)
@njit(cache=True)
def generate_actions_perfect_foresight(price_arr: np.ndarray, fix: int = 1500) -> np.ndarray:
    n = len(price_arr)
    if n < 2: return np.ones(n, dtype=np.int32)
    dp = np.zeros(n, dtype=np.float64); path = np.zeros(n, dtype=np.int32)
    dp[0] = float(fix * 2)
    for i in range(1, n):
        j_indices = np.arange(i); profits = fix * ((price_arr[i] / price_arr[j_indices]) - 1.0)
        current_sumusd = dp[j_indices] + profits; best_idx = np.argmax(current_sumusd)
        dp[i] = current_sumusd[best_idx]; path[i] = j_indices[best_idx]
    actions = np.zeros(n, dtype=np.int32); best_final_day = np.argmax(dp); current_day = best_final_day
    while current_day > 0: actions[current_day] = 1; current_day = path[current_day]
    actions[0] = 1
    return actions

# 3.1 & 3.2 & 3.3 ... All functions from previous steps are assumed to be here
# To save space, I will skip pasting them again. The new model is self-contained.

# 3.4 [NEW] Pattern Matching (DTW) Generation
@lru_cache(maxsize=1024)
def find_best_historical_analogs_dtw(target_pattern: tuple, historical_data: tuple, window_size: int, num_analogs: int) -> list:
    """Finds the best historical analogs using Dynamic Time Warping."""
    target_pattern = np.array(target_pattern)
    historical_data = np.array(historical_data)
    
    distances = []
    # Loop through historical data to find similar patterns
    for i in range(len(historical_data) - window_size):
        historical_pattern = historical_data[i : i + window_size]
        # Normalize patterns to compare shapes, not absolute values
        normalized_target = target_pattern / target_pattern[0]
        normalized_historical = historical_pattern / historical_pattern[0]
        
        # Calculate DTW distance
        distance = dtw(normalized_target, normalized_historical, keep_internals=True).distance
        
        # We need to know what happened *after* this historical pattern
        future_start = i + window_size
        future_end = future_start + window_size
        if future_end <= len(historical_data):
            distances.append((distance, future_start, future_end))
    
    # Sort by distance (smaller is better) and take the top N
    distances.sort(key=lambda x: x[0])
    return distances[:num_analogs]

def generate_actions_sliding_window_pattern_matching(
    ticker_data: pd.DataFrame, 
    window_size: int, 
    lookback_period: int, 
    num_analogs: int
) -> Tuple[np.ndarray, pd.DataFrame]:
    
    all_prices = ticker_data['Close'].to_numpy()
    n = len(all_prices)
    final_actions = np.array([], dtype=np.int32)
    window_details_list = []
    
    num_windows = (n - lookback_period) // window_size
    if num_windows <= 0:
        st.error(f"ข้อมูลไม่เพียงพอสำหรับ Lookback Period ({lookback_period}) และ Window Size ({window_size}) กรุณาลดค่า Lookback Period หรือเพิ่มช่วงวันที่ของข้อมูล")
        return np.array([]), pd.DataFrame()

    progress_bar = st.progress(0, text="กำลังค้นหารูปแบบราคาในอดีต (Pattern Matching)...")
    
    # We can only start predicting after the lookback_period
    for i, start_index in enumerate(range(lookback_period, n, window_size)):
        end_index = min(start_index + window_size, n)
        if end_index - start_index < window_size: continue # Skip partial windows at the end

        # 1. Define target pattern and historical search space
        target_pattern = all_prices[start_index:end_index]
        historical_data = all_prices[:start_index]

        # 2. Find best analogs from history
        analogs = find_best_historical_analogs_dtw(
            tuple(target_pattern), 
            tuple(historical_data), 
            window_size, 
            num_analogs
        )
        
        # 3. For each analog, find its "perfect future" action sequence
        future_actions_candidates = []
        if analogs:
            for _, future_start, future_end in analogs:
                future_prices = historical_data[future_start:future_end]
                if len(future_prices) == window_size:
                    # Find the best possible actions for that future
                    perfect_actions = generate_actions_perfect_foresight(future_prices)
                    future_actions_candidates.append(perfect_actions)

        # 4. Combine the candidates into a single action sequence (Voting)
        if future_actions_candidates:
            # Stack candidates and take the mode (most frequent value) for each day
            stacked_actions = np.vstack(future_actions_candidates)
            # scipy.stats.mode is better, but np.mean avoids a new dependency
            # A mean > 0.5 means '1' was more common than '0'
            voted_actions = (np.mean(stacked_actions, axis=0) > 0.5).astype(int)
            voted_actions[0] = 1 # Ensure first action is always 1
        else:
            # If no analogs found, default to rebalancing daily
            voted_actions = np.ones(window_size, dtype=np.int32)

        final_actions = np.concatenate((final_actions, voted_actions))
        
        # Calculate Net for reporting
        _, sumusd, _, _, _, refer = calculate_optimized_cached(tuple(voted_actions), tuple(target_pattern))
        max_net = sumusd[-1] - refer[-1] - sumusd[0] if len(sumusd) > 0 else 0.0
        
        start_date_str = ticker_data.index[start_index].strftime('%Y-%m-%d')
        end_date_str = ticker_data.index[end_index-1].strftime('%Y-%m-%d')
        detail = {
            'window_number': i + 1, 'timeline': f"{start_date_str} ถึง {end_date_str}", 'max_net': round(max_net, 2),
            'price_change_pct': round(((target_pattern[-1] / target_pattern[0]) - 1) * 100, 2),
            'action_count': int(np.sum(voted_actions)), 'analogs_found': len(future_actions_candidates),
            'window_size': window_len, 'action_sequence': voted_actions.tolist()
        }
        window_details_list.append(detail)
        progress_bar.progress((i + 1) / num_windows, text=f"Analyzing Window {i+1}/{num_windows}")
        
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)


# ==============================================================================
# 4. UI Rendering Functions
# ==============================================================================
def render_settings_tab(config: Dict[str, Any]):
    st.write("⚙️ **การตั้งค่าพารามิเตอร์**")
    asset_list = config.get('assets', ['FFWM']); default_index = asset_list.index(st.session_state.test_ticker) if st.session_state.test_ticker in asset_list else 0
    st.session_state.test_ticker = st.selectbox("เลือก Ticker สำหรับทดสอบ", options=asset_list, index=default_index)
    
    st.write("📅 **ช่วงวันที่สำหรับการวิเคราะห์**")
    col1, col2 = st.columns(2)
    with col1: st.session_state.start_date = st.date_input("วันที่เริ่มต้น", value=st.session_state.start_date)
    with col2: st.session_state.end_date = st.date_input("วันที่สิ้นสุด", value=st.session_state.end_date)
    
    if st.session_state.start_date >= st.session_state.end_date: st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
    else: st.info(f"ช่วงวันที่ที่เลือก: {st.session_state.start_date:%Y-%m-%d} ถึง {st.session_state.end_date:%Y-%m-%d}")
    
    st.divider()
    
    st.subheader("พารามิเตอร์ทั่วไป")
    c1, c2 = st.columns(2)
    st.session_state.window_size = c1.number_input("ขนาด Window (วัน)", min_value=10, value=st.session_state.window_size)
    st.session_state.max_workers = c2.number_input("จำนวน Workers (สำหรับ Random/Chaotic)", min_value=1, max_value=16, value=st.session_state.max_workers)

    st.subheader("พารามิเตอร์สำหรับ Random/Chaotic Seed")
    st.session_state.num_seeds = st.number_input("จำนวน Seeds/Params ต่อ Window", min_value=100, value=st.session_state.num_seeds, format="%d")
    
    st.subheader("พารามิเตอร์สำหรับ Genetic Algorithm")
    ga_c1, ga_c2 = st.columns(2)
    st.session_state.ga_population_size = ga_c1.number_input("ขนาดประชากร (Population Size)", min_value=10, value=st.session_state.ga_population_size)
    st.session_state.ga_generations = ga_c2.number_input("จำนวนรุ่น (Generations)", min_value=5, value=st.session_state.ga_generations)
    
    st.subheader("พารามิเตอร์สำหรับ Pattern Matching (DTW)")
    pm_c1, pm_c2 = st.columns(2)
    st.session_state.pm_lookback_period = pm_c1.number_input("ช่วงข้อมูลในอดีตสำหรับค้นหา (วัน)", min_value=100, value=st.session_state.pm_lookback_period, help="จำนวนวันที่ใช้เป็นฐานข้อมูลในการค้นหารูปแบบที่คล้ายกัน ยิ่งมากยิ่งดีแต่จะช้าลง")
    st.session_state.pm_num_analogs = pm_c2.number_input("จำนวนรูปแบบที่คล้ายที่สุดที่จะใช้ (Analogs)", min_value=1, max_value=20, value=st.session_state.pm_num_analogs, help="จำนวนรูปแบบที่คล้ายกันที่สุดในอดีตที่จะนำมาลงคะแนนเพื่อตัดสินใจ")


def display_comparison_charts(results: Dict[str, pd.DataFrame], chart_title: str = '📊 เปรียบเทียบกำไรสุทธิ (Net Profit)'):
    if not results: st.warning("ไม่มีข้อมูลสำหรับสร้างกราฟเปรียบเทียบ"); return
    try: longest_index = max((df.index for name, df in results.items() if not df.empty and 'net' in df.columns), key=len, default=None)
    except ValueError: longest_index = None
    if longest_index is None: st.warning("ไม่มีข้อมูลสำหรับสร้างกราฟเปรียบเทียบ"); return
    chart_data = pd.DataFrame(index=longest_index)
    for name, df in results.items():
        if not df.empty and 'net' in df.columns:
            chart_data[name] = df['net'].reindex(longest_index).ffill()
    st.write(chart_title); st.line_chart(chart_data)


def run_and_display_strategy(strategy_func, strategy_args: Dict, strategy_name: str, full_ticker_data: pd.DataFrame, df_columns_to_show: List[str]):
    button_key = f"btn_{strategy_name.replace(' ', '_').replace('(', '').replace(')', '')}"
    if st.button(f"🚀 เริ่มทดสอบ ({strategy_name})", type="primary", key=button_key):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้องในแท็บ 'การตั้งค่า'"); return
        
        with st.spinner(f"กำลังคำนวณกลยุทธ์ {strategy_name}..."):
            # For Pattern Matching, we need a larger dataset including the lookback period
            if strategy_name == Strategy.PATTERN_MATCHING_DTW:
                test_data = full_ticker_data
                prices_for_sim = test_data['Close'].to_numpy()[st.session_state.pm_lookback_period:]
                sim_dates = test_data.index[st.session_state.pm_lookback_period:]
            else:
                test_data = full_ticker_data
                prices_for_sim = test_data['Close'].to_numpy()
                sim_dates = test_data.index

            main_actions, df_windows = strategy_func(ticker_data=test_data, **strategy_args)
            
            num_days = len(prices_for_sim)
            if num_days == 0 or len(main_actions) == 0:
                st.error("ไม่สามารถสร้างผลลัพธ์ได้ ข้อมูลอาจไม่เพียงพอสำหรับการตั้งค่าปัจจุบัน"); return

            actions_min = generate_actions_rebalance_daily(num_days)
            actions_max = generate_actions_perfect_foresight(prices_for_sim)

            results_map = {
                strategy_name: main_actions.tolist(),
                Strategy.REBALANCE_DAILY: actions_min.tolist(),
                Strategy.PERFECT_FORESIGHT: actions_max.tolist()
            }
            sim_results = {}
            for name, actions in results_map.items():
                df = run_simulation(prices_for_sim.tolist(), actions)
                if not df.empty: df.index = sim_dates[:len(df)]
                sim_results[name] = df
        
        st.success("การทดสอบเสร็จสมบูรณ์!")
        st.write("---"); display_comparison_charts(sim_results, f"ผลการทดสอบ: {strategy_name}")
        st.write(f"📈 **สรุปผลการค้นหา ({strategy_name})**")
        
        if df_windows is not None and not df_windows.empty:
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

# --- UI Rendering for each Tab ---
def render_test_tab(ticker_data): # Random Seed
    st.markdown("### 🎲 ทดสอบ Best Seed ด้วย Random Search")
    st.info("กลยุทธ์นี้จะสุ่ม `seed` จำนวนมากเพื่อหาลำดับ Action ที่ดีที่สุดในแต่ละ Window")
    if ticker_data.empty: st.error("ไม่พบข้อมูลสำหรับ Ticker และช่วงวันที่ที่เลือก"); return
    args = {'window_size': st.session_state.window_size, 'num_seeds_to_try': st.session_state.num_seeds, 'max_workers': st.session_state.max_workers}
    columns_to_show = ['window_number', 'timeline', 'best_seed', 'max_net', 'price_change_pct', 'action_count']
    run_and_display_strategy(generate_actions_sliding_window, args, Strategy.SLIDING_WINDOW, ticker_data, columns_to_show)

def render_chaotic_test_tab(ticker_data):
    st.markdown("### 🌀 ทดสอบ Best Seed ด้วย Chaotic Generator")
    st.info("กลยุทธ์นี้จะค้นหาพารามิเตอร์ `r` และ `x0` ของ Logistic Map ที่ให้ผลตอบแทนดีที่สุดในแต่ละ Window")
    if ticker_data.empty: st.error("ไม่พบข้อมูลสำหรับ Ticker และช่วงวันที่ที่เลือก"); return
    args = {'window_size': st.session_state.window_size, 'num_seeds_to_try': st.session_state.num_seeds, 'max_workers': st.session_state.max_workers}
    columns_to_show = ['window_number', 'timeline', 'best_seed', 'r_param', 'x0_param', 'max_net', 'price_change_pct', 'action_count']
    run_and_display_strategy(generate_actions_sliding_window_chaotic, args, Strategy.CHAOTIC_SLIDING_WINDOW, ticker_data, columns_to_show)

def render_ga_test_tab(ticker_data):
    st.markdown("### 🧬 ทดสอบด้วย Genetic Algorithm Search")
    st.info("กลยุทธ์นี้ใช้วิวัฒนาการเชิงคำนวณ (GA) เพื่อ 'พัฒนา' Action Sequence ที่ดีที่สุดในแต่ละ Window")
    if ticker_data.empty: st.error("ไม่พบข้อมูลสำหรับ Ticker และช่วงวันที่ที่เลือก"); return
    args = {'window_size': st.session_state.window_size, 'population_size': st.session_state.ga_population_size, 'generations': st.session_state.ga_generations}
    columns_to_show = ['window_number', 'timeline', 'max_net', 'price_change_pct', 'action_count']
    run_and_display_strategy(generate_actions_sliding_window_ga, args, Strategy.GENETIC_ALGORITHM, ticker_data, columns_to_show)

def render_pattern_matching_tab(ticker_data):
    st.markdown("### 🔍 ทดสอบด้วย Pattern Matching (DTW)")
    st.info("กลยุทธ์นี้จะค้นหารูปแบบราคาในอดีตที่คล้ายกับปัจจุบันที่สุด แล้วนำ Action ที่เคยให้ผลลัพธ์ดีที่สุดในอนาคตของรูปแบบนั้นมาใช้")
    if ticker_data.empty: st.error("ไม่พบข้อมูลสำหรับ Ticker และช่วงวันที่ที่เลือก"); return
    args = {'window_size': st.session_state.window_size, 'lookback_period': st.session_state.pm_lookback_period, 'num_analogs': st.session_state.pm_num_analogs}
    columns_to_show = ['window_number', 'timeline', 'max_net', 'price_change_pct', 'action_count', 'analogs_found']
    run_and_display_strategy(generate_actions_sliding_window_pattern_matching, args, Strategy.PATTERN_MATCHING_DTW, ticker_data, columns_to_show)

# ... The render_analytics_tab and render_manual_seed_tab are omitted for brevity ...
def render_analytics_tab(): st.header("📊 Advanced Analytics Dashboard (Omitted)"); st.info("This tab is for analyzing saved CSV files.")
def render_manual_seed_tab(config): st.header("🌱 Manual/Forward Testing (Omitted)"); st.info("This tab is for testing custom seeds.")

# ==============================================================================
# 5. Main Application
# ==============================================================================
def main():
    st.set_page_config(page_title="Strategy Optimization Lab", page_icon="🎯", layout="wide")
    st.markdown("## 🎯 Strategy Optimization Lab")
    st.caption("เครื่องมือทดสอบและเปรียบเทียบกลยุทธ์การลงทุนด้วยวิธี Sliding Window")

    config = load_config()
    initialize_session_state(config)

    tab_list = ["⚙️ การตั้งค่า", "🎲 Random Seed", "🌀 Chaotic Seed", "🧬 GA Search", "🔍 Pattern Matching", "📊 Analytics", "🌱 Manual/Forward"]
    tabs = st.tabs(tab_list)

    # Fetch data once for all tabs to use
    # For Pattern Matching, we need to fetch more data to serve as the historical lookback
    start_date_obj = st.session_state.start_date
    pm_lookback_days = st.session_state.pm_lookback_period
    data_fetch_start_date = (start_date_obj - pd.Timedelta(days=pm_lookback_days * 1.5)).strftime('%Y-%m-%d') # Fetch 1.5x to account for non-trading days
    full_ticker_data = get_ticker_data(st.session_state.test_ticker, data_fetch_start_date, st.session_state.end_date.strftime('%Y-%m-%d'))
    
    # Filter data for non-pattern matching tabs
    standard_ticker_data = full_ticker_data.loc[full_ticker_data.index.date >= start_date_obj]

    with tabs[0]: render_settings_tab(config)
    with tabs[1]: render_test_tab(standard_ticker_data)
    with tabs[2]: render_chaotic_test_tab(standard_ticker_data)
    with tabs[3]: render_ga_test_tab(standard_ticker_data)
    with tabs[4]: render_pattern_matching_tab(full_ticker_data) # This tab needs the full dataset
    with tabs[5]: render_analytics_tab()
    with tabs[6]: render_manual_seed_tab(config)

    with st.expander("📖 คำอธิบายกลยุทธ์และแนวคิด"):
        st.markdown("""
        **หลักการพื้นฐาน:** กลยุทธ์ทั้งหมดทำงานบนหลักการ "Sliding Window" คือแบ่งช่วงเวลาทั้งหมดออกเป็น "หน้าต่าง" เล็กๆ แล้วพยายามหา "รูปแบบการกระทำ" (Action Sequence) ที่ดีที่สุดสำหรับหน้าต่างนั้นๆ ก่อนจะนำผลลัพธ์ของทุกหน้าต่างมาประกอบกันเป็นกลยุทธ์สุดท้าย

        ---
        #### รูปแบบการค้นหา (Search Strategies)
        *   **🎲 Random Seed:** ค้นหาโดยการสุ่ม Action Sequence จำนวนมาก
        *   **🌀 Chaotic Seed:** ใช้ "Logistic Map" ในการสร้าง Action Sequence ที่มีรูปแบบซับซ้อน
        *   **🧬 Genetic Algorithm:** เลียนแบบวิวัฒนาการทางธรรมชาติ เพื่อ "พัฒนา" Sequence ที่ดีที่สุด
        *   **🔍 Pattern Matching (DTW):** **[โมเดลใหม่ระดับสูง]** เป็นเทคนิคที่ค้นหารูปแบบราคาในอดีต (Historical Analogs) ที่มี "รูปทรง" คล้ายกับรูปแบบราคาในปัจจุบันมากที่สุดโดยใช้ Dynamic Time Warping (DTW) จากนั้นจะนำ Action Sequence ที่เคยให้ผลตอบแทนดีที่สุดในอนาคตของรูปแบบเหล่านั้นมา "ลงคะแนน" (Vote) เพื่อสร้างเป็น Action สำหรับปัจจุบัน เป็นกลยุทธ์ที่ขับเคลื่อนด้วยข้อมูลอย่างแท้จริง

        ---
        #### แท็บอื่นๆ
        *   **📊 Analytics:** ใช้วิเคราะห์ไฟล์ผลลัพธ์ (.csv) ที่ดาวน์โหลดจากการทดสอบ
        *   **🌱 Manual/Forward:** ใช้ทดสอบ Action Sequence ที่สร้างจาก Seed ที่กำหนดเอง
        """)

if __name__ == "__main__":
    main()
