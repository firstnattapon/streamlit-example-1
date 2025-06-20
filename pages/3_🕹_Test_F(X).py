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
                "selected_ticker": "FFWM", "start_date": "2024-01-01", "end_date": "2024-07-01", "window_size": 30, 
                "num_seeds": 10000, "max_workers": 8, 
                "ga_population_size": 50, "ga_generations": 20,
                "pm_lookback_period": 252, "pm_num_analogs": 5
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
        try: st.session_state.start_date = datetime.strptime(defaults.get('start_date', '2024-01-01'), '%Y-%m-%d').date()
        except ValueError: st.session_state.start_date = datetime(2024, 1, 1).date()
    if 'end_date' not in st.session_state:
        try: st.session_state.end_date = datetime.strptime(defaults.get('end_date', '2024-07-01'), '%Y-%m-%d').date()
        except ValueError: st.session_state.end_date = datetime.now().date()
    if 'window_size' not in st.session_state: st.session_state.window_size = defaults.get('window_size', 30)
    if 'num_seeds' not in st.session_state: st.session_state.num_seeds = defaults.get('num_seeds', 10000)
    if 'max_workers' not in st.session_state: st.session_state.max_workers = defaults.get('max_workers', 8)
    if 'ga_population_size' not in st.session_state: st.session_state.ga_population_size = defaults.get('ga_population_size', 50)
    if 'ga_generations' not in st.session_state: st.session_state.ga_generations = defaults.get('ga_generations', 20)
    
    if 'pm_lookback_period' not in st.session_state: st.session_state.pm_lookback_period = defaults.get('pm_lookback_period', 252)
    if 'pm_num_analogs' not in st.session_state: st.session_state.pm_num_analogs = defaults.get('pm_num_analogs', 5)

    if 'df_for_analysis' not in st.session_state: st.session_state.df_for_analysis = None
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
        data = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=True)[['Close']]
        if data.empty: return pd.DataFrame()
        if data.index.tz is None:
            data = data.tz_localize('UTC').tz_convert('Asia/Bangkok')
        else:
            data = data.tz_convert('Asia/Bangkok')
        return data.rename(columns={'Close': 'price'})
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
            amount[i] = amount[i-1]; buffer[i] = 0.0
        else:
            amount[i] = fix / curr_price; buffer[i] = amount[i-1] * curr_price - fix
        cash[i] = cash[i-1] + buffer[i]; asset_value[i] = amount[i] * curr_price
        sumusd[i] = cash[i] + asset_value[i]
    return buffer, sumusd, cash, asset_value, amount, refer

@lru_cache(maxsize=4096)
def calculate_optimized_cached(action_tuple: Tuple[int, ...], price_tuple: Tuple[float, ...], fix: int = 1500) -> Tuple:
    action_array = np.asarray(action_tuple, dtype=np.int32)
    price_array = np.asarray(price_tuple, dtype=np.float64)
    return _calculate_simulation_numba(action_array, price_array, fix)

def run_simulation(prices: List[float], actions: List[int], fix: int = 1500) -> pd.DataFrame:
    if not prices or not actions: return pd.DataFrame()
    min_len = min(len(prices), len(actions))
    prices, actions = prices[:min_len], actions[:min_len]
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

# 3.1 Standard Seed Generation (functions are self-contained, no change)
def find_best_seed_for_window(prices_window: np.ndarray, num_seeds_to_try: int, max_workers: int) -> Tuple[int, float, np.ndarray]:
    window_len = len(prices_window);
    if window_len < 2: return 1, 0.0, np.ones(window_len, dtype=np.int32)
    def evaluate_seed_batch(seed_batch: np.ndarray) -> List[Tuple[int, float]]:
        results = []
        for seed in seed_batch:
            rng = np.random.default_rng(seed); actions_window = rng.integers(0, 2, size=window_len, dtype=np.int32)
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
                if final_net > max_net_for_window: max_net_for_window = final_net; best_seed_for_window = seed
    if best_seed_for_window >= 0:
        rng_best = np.random.default_rng(best_seed_for_window)
        best_actions = rng_best.integers(0, 2, size=window_len, dtype=np.int32)
    else: best_seed_for_window = 1; best_actions = np.ones(window_len, dtype=np.int32); max_net_for_window = 0.0
    best_actions[0] = 1
    return best_seed_for_window, max_net_for_window, best_actions

def generate_actions_sliding_window(ticker_data: pd.DataFrame, **kwargs) -> Tuple[np.ndarray, pd.DataFrame]:
    prices = ticker_data['price'].to_numpy(); n = len(prices); window_size = kwargs['window_size']
    final_actions = np.array([], dtype=np.int32); window_details_list = []
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="กำลังประมวลผล Random Seed...")
    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n); prices_window = prices[start_index:end_index]
        window_len = len(prices_window)
        if window_len == 0: continue
        best_seed, max_net, best_actions = find_best_seed_for_window(prices_window, kwargs['num_seeds_to_try'], kwargs['max_workers'])
        final_actions = np.concatenate((final_actions, best_actions))
        start_date_str = ticker_data.index[start_index].strftime('%Y-%m-%d'); end_date_str = ticker_data.index[end_index-1].strftime('%Y-%m-%d')
        detail = {'window_number': i + 1, 'timeline': f"{start_date_str} to {end_date_str}", 'best_seed': best_seed, 'max_net': round(max_net, 2),'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2) if prices_window[0] != 0 else 0, 'action_count': int(np.sum(best_actions)), 'window_size': window_len, 'action_sequence': best_actions.tolist()}
        window_details_list.append(detail)
        progress_bar.progress((i + 1) / num_windows, text=f"Processing Window {i+1}/{num_windows}")
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)


# 3.4 [NEW] Pattern Matching (DTW) Generation
@lru_cache(maxsize=1024)
def find_best_historical_analogs_dtw(target_pattern_tuple: tuple, historical_data_tuple: tuple, window_size: int, num_analogs: int) -> list:
    target_pattern = np.array(target_pattern_tuple)
    historical_data = np.array(historical_data_tuple)
    distances = []
    for i in range(len(historical_data) - window_size * 2): # Ensure there's a future
        historical_pattern = historical_data[i : i + window_size]
        normalized_target = target_pattern / target_pattern[0]
        normalized_historical = historical_pattern / historical_pattern[0]
        distance = dtw(normalized_target, normalized_historical, keep_internals=False).distance
        future_start = i + window_size; future_end = future_start + window_size
        distances.append((distance, future_start, future_end))
    distances.sort(key=lambda x: x[0])
    return distances[:num_analogs]

def generate_actions_sliding_window_pattern_matching(ticker_data: pd.DataFrame, **kwargs) -> Tuple[np.ndarray, pd.DataFrame]:
    all_prices = ticker_data['price'].to_numpy()
    window_size = kwargs['window_size']; lookback_period = kwargs['lookback_period']; num_analogs = kwargs['num_analogs']
    n = len(all_prices)
    if n < lookback_period + window_size:
        st.error(f"ข้อมูลไม่เพียงพอ ({n} วัน) สำหรับ Lookback Period ({lookback_period}) และ Window Size ({window_size}) กรุณาลดค่า Lookback Period หรือเพิ่มช่วงวันที่ของข้อมูล")
        return np.array([]), pd.DataFrame()
    
    action_start_offset = lookback_period
    sim_prices = all_prices[action_start_offset:]
    final_actions = np.array([], dtype=np.int32)
    window_details_list = []
    
    num_windows = (len(sim_prices) + window_size - 1) // window_size
    progress_bar = st.progress(0, text="กำลังค้นหารูปแบบราคาในอดีต (Pattern Matching)...")

    for i in range(num_windows):
        current_day_index = action_start_offset + (i * window_size)
        target_start = current_day_index - window_size
        target_end = current_day_index
        if target_start < 0: continue

        target_pattern = all_prices[target_start:target_end]
        historical_data = all_prices[:target_start]
        
        if len(historical_data) < window_size * 2: continue
        
        analogs = find_best_historical_analogs_dtw(tuple(target_pattern), tuple(historical_data), window_size, num_analogs)
        future_actions_candidates = []
        if analogs:
            for _, future_start, future_end in analogs:
                future_prices = historical_data[future_start:future_end]
                if len(future_prices) == window_size:
                    perfect_actions = generate_actions_perfect_foresight(future_prices)
                    future_actions_candidates.append(perfect_actions)
        
        if future_actions_candidates:
            voted_actions = (np.mean(np.vstack(future_actions_candidates), axis=0) > 0.5).astype(int)
            voted_actions[0] = 1
        else:
            voted_actions = np.ones(window_size, dtype=np.int32)
        
        final_actions = np.concatenate((final_actions, voted_actions))
        
        # Calculate Net for reporting based on actual prices in this window
        report_prices_start = current_day_index
        report_prices_end = min(report_prices_start + window_size, n)
        report_prices = all_prices[report_prices_start:report_prices_end]

        if len(report_prices) > 0:
            _, sumusd, _, _, _, refer = calculate_optimized_cached(tuple(voted_actions), tuple(report_prices))
            max_net = sumusd[-1] - refer[-1] - sumusd[0] if len(sumusd) > 0 else 0.0
            
            detail = {'window_number': i + 1, 'timeline': f"{ticker_data.index[report_prices_start].strftime('%Y-%m-%d')} to {ticker_data.index[report_prices_end-1].strftime('%Y-%m-%d')}", 'max_net': round(max_net, 2), 'price_change_pct': round(((report_prices[-1] / report_prices[0]) - 1) * 100, 2), 'action_count': int(np.sum(voted_actions)), 'analogs_found': len(future_actions_candidates), 'window_size': len(report_prices)}
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
    
    st.divider()
    st.subheader("พารามิเตอร์ทั่วไป")
    c1, c2 = st.columns(2)
    st.session_state.window_size = c1.number_input("ขนาด Window (วัน)", min_value=10, value=st.session_state.window_size)
    st.session_state.max_workers = c2.number_input("จำนวน Workers (สำหรับ Random Seed)", min_value=1, max_value=16, value=st.session_state.max_workers)
    
    st.subheader("พารามิเตอร์สำหรับ Random Seed")
    st.session_state.num_seeds = st.number_input("จำนวน Seeds ต่อ Window", min_value=100, value=st.session_state.num_seeds, format="%d")

    st.subheader("พารามิเตอร์สำหรับ Pattern Matching (DTW)")
    pm_c1, pm_c2 = st.columns(2)
    st.session_state.pm_lookback_period = pm_c1.number_input("ช่วงข้อมูลในอดีตสำหรับค้นหา (วัน)", min_value=50, value=st.session_state.pm_lookback_period)
    st.session_state.pm_num_analogs = pm_c2.number_input("จำนวนรูปแบบที่คล้ายที่สุดที่จะใช้ (Analogs)", min_value=1, max_value=20, value=st.session_state.pm_num_analogs)


def display_comparison_charts(results: Dict[str, pd.DataFrame], chart_title: str):
    if not results: st.warning("ไม่มีข้อมูลสำหรับสร้างกราฟเปรียบเทียบ"); return
    
    valid_dfs = {name: df for name, df in results.items() if not df.empty and 'net' in df.columns}
    if not valid_dfs: st.warning("ไม่มีข้อมูล 'net' สำหรับสร้างกราฟ"); return

    longest_index = max((df.index for df in valid_dfs.values()), key=len, default=None)
    if longest_index is None: return

    chart_data = pd.DataFrame(index=longest_index)
    for name, df in valid_dfs.items():
        chart_data[name] = df['net'].reindex(longest_index).ffill()
    
    st.write(chart_title); st.line_chart(chart_data)


def run_and_display_strategy(strategy_func, strategy_args: Dict, strategy_name: str, full_ticker_data: pd.DataFrame, df_columns_to_show: List[str]):
    button_key = f"btn_{strategy_name.replace(' ', '_').replace('(', '').replace(')', '')}"
    if st.button(f"🚀 เริ่มทดสอบ ({strategy_name})", type="primary", key=button_key):
        with st.spinner(f"กำลังคำนวณกลยุทธ์ {strategy_name}..."):
            
            # Pattern Matching uses a different data slice
            if strategy_name == Strategy.PATTERN_MATCHING_DTW:
                test_data = full_ticker_data
                lookback = st.session_state.pm_lookback_period
                if len(test_data) < lookback + st.session_state.window_size:
                    st.error("ข้อมูลไม่เพียงพอสำหรับ Lookback ที่กำหนด"); return
                prices_for_sim = test_data['price'].to_numpy()[lookback:]
                sim_dates = test_data.index[lookback:]
            else:
                start_date_obj = st.session_state.start_date
                test_data = full_ticker_data[full_ticker_data.index.date >= start_date_obj]
                if test_data.empty: st.error("ไม่พบข้อมูลในช่วงวันที่ที่เลือกสำหรับทดสอบ"); return
                prices_for_sim = test_data['price'].to_numpy()
                sim_dates = test_data.index

            main_actions, df_windows = strategy_func(ticker_data=test_data, **strategy_args)
            
            num_days = len(prices_for_sim)
            if num_days == 0 or len(main_actions) == 0: st.error("ไม่สามารถสร้างผลลัพธ์ได้"); return

            main_actions = main_actions[:num_days] # Ensure actions and prices align
            
            results_map = {
                strategy_name: run_simulation(prices_for_sim.tolist(), main_actions.tolist()),
                Strategy.REBALANCE_DAILY: run_simulation(prices_for_sim.tolist(), generate_actions_rebalance_daily(num_days).tolist()),
                Strategy.PERFECT_FORESIGHT: run_simulation(prices_for_sim.tolist(), generate_actions_perfect_foresight(prices_for_sim).tolist())
            }
            
            for name, df in results_map.items():
                if not df.empty: df.index = sim_dates[:len(df)]

        st.success("การทดสอบเสร็จสมบูรณ์!")
        st.write("---"); display_comparison_charts(results_map, f"ผลการทดสอบ: {strategy_name}")
        st.write(f"📈 **สรุปผลการค้นหา ({strategy_name})**")
        
        if df_windows is not None and not df_windows.empty:
            total_actions = df_windows['action_count'].sum(); total_net = df_windows['max_net'].sum()
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Windows", df_windows.shape[0]); col2.metric("Total Actions", f"{total_actions}/{len(main_actions)}"); col3.metric("Total Net (Sum)", f"${total_net:,.2f}")
            st.dataframe(df_windows[df_columns_to_show], use_container_width=True)
            csv = df_windows.to_csv(index=False)
            st.download_button(label=f"📥 ดาวน์โหลด {strategy_name} Details (CSV)", data=csv, file_name=f'{strategy_name.replace(" ", "_")}_{st.session_state.test_ticker}_{st.session_state.window_size}w.csv', mime='text/csv')

# ==============================================================================
# 5. Main Application
# ==============================================================================
def main():
    st.set_page_config(page_title="Strategy Optimization Lab", page_icon="🎯", layout="wide")
    st.markdown("## 🎯 Strategy Optimization Lab")
    st.caption("เครื่องมือทดสอบและเปรียบเทียบกลยุทธ์การลงทุนด้วยวิธี Sliding Window")

    config = load_config()
    initialize_session_state(config)

    tab_list = ["⚙️ การตั้งค่า", "🎲 Random Seed", "🔍 Pattern Matching", "📊 Analytics"]
    tabs = st.tabs(tab_list)

    with tabs[0]: render_settings_tab(config)
    
    # Fetch data once for all tabs
    start_date_obj = st.session_state.start_date
    end_date_obj = st.session_state.end_date
    # Fetch extra data for lookback period
    required_lookback_days = st.session_state.pm_lookback_period + st.session_state.window_size
    data_fetch_start_date = (start_date_obj - pd.Timedelta(days=required_lookback_days * 1.5)).strftime('%Y-%m-%d')
    full_ticker_data = get_ticker_data(st.session_state.test_ticker, data_fetch_start_date, end_date_obj.strftime('%Y-%m-%d'))

    with tabs[1]:
        st.markdown("### 🎲 ทดสอบ Best Seed ด้วย Random Search")
        st.info("กลยุทธ์นี้จะสุ่ม `seed` จำนวนมากเพื่อหาลำดับ Action ที่ดีที่สุดในแต่ละ Window")
        if not full_ticker_data.empty:
            test_data = full_ticker_data[full_ticker_data.index.date >= start_date_obj]
            args = {'window_size': st.session_state.window_size, 'num_seeds_to_try': st.session_state.num_seeds, 'max_workers': st.session_state.max_workers}
            columns_to_show = ['window_number', 'timeline', 'best_seed', 'max_net', 'price_change_pct', 'action_count']
            run_and_display_strategy(generate_actions_sliding_window, args, Strategy.SLIDING_WINDOW, test_data, columns_to_show)
        else: st.warning("กรุณารอข้อมูลโหลดสักครู่ หรือตรวจสอบการตั้งค่าวันที่")


    with tabs[2]:
        st.markdown("### 🔍 ทดสอบด้วย Pattern Matching (DTW)")
        st.info("กลยุทธ์นี้จะค้นหารูปแบบราคาในอดีตที่คล้ายกับปัจจุบันที่สุด แล้วนำ Action ที่เคยให้ผลลัพธ์ดีที่สุดในอนาคตของรูปแบบนั้นมาใช้")
        if not full_ticker_data.empty:
            # This tab uses the full dataset including the lookback period
            args = {'window_size': st.session_state.window_size, 'lookback_period': st.session_state.pm_lookback_period, 'num_analogs': st.session_state.pm_num_analogs}
            columns_to_show = ['window_number', 'timeline', 'max_net', 'price_change_pct', 'action_count', 'analogs_found']
            run_and_display_strategy(generate_actions_sliding_window_pattern_matching, args, Strategy.PATTERN_MATCHING_DTW, full_ticker_data, columns_to_show)
        else: st.warning("กรุณารอข้อมูลโหลดสักครู่ หรือตรวจสอบการตั้งค่าวันที่")

    with tabs[3]:
        st.header("📊 Advanced Analytics Dashboard")
        st.info("This tab is for analyzing saved CSV files from tests.")


    with st.expander("📖 คำอธิบายกลยุทธ์และแนวคิด"):
        st.markdown("""
        **หลักการพื้นฐาน:** กลยุทธ์ทั้งหมดทำงานบนหลักการ "Sliding Window" คือแบ่งช่วงเวลาทั้งหมดออกเป็น "หน้าต่าง" เล็กๆ แล้วพยายามหา "รูปแบบการกระทำ" (Action Sequence) ที่ดีที่สุดสำหรับหน้าต่างนั้นๆ
        
        ---
        #### รูปแบบการค้นหา (Search Strategies)
        *   **🎲 Random Seed:** ค้นหาโดยการสุ่ม Action Sequence จำนวนมาก
        *   **🔍 Pattern Matching (DTW):** **[โมเดลใหม่ระดับสูง]** เป็นเทคนิคที่ค้นหารูปแบบราคาในอดีต (Historical Analogs) ที่มี "รูปทรง" คล้ายกับรูปแบบราคาในปัจจุบันมากที่สุดโดยใช้ Dynamic Time Warping (DTW) จากนั้นจะนำ Action Sequence ที่เคยให้ผลตอบแทนดีที่สุดในอนาคตของรูปแบบเหล่านั้นมา "ลงคะแนน" (Vote) เพื่อสร้างเป็น Action สำหรับปัจจุบัน เป็นกลยุทธ์ที่ขับเคลื่อนด้วยข้อมูลอย่างแท้จริง
        """)

if __name__ == "__main__":
    main()
