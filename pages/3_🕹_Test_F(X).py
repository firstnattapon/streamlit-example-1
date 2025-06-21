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
    ARITHMETIC_SEQUENCE = "Arithmetic Sequence"
    GEOMETRIC_SEQUENCE = "Geometric Sequence"

def load_config(filepath: str = "dynamic_seed_config.json") -> Dict[str, Any]:
    """ โหลดการตั้งค่าจากไฟล์ JSON """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        st.warning(f"⚠️ ไม่พบหรือไฟล์ '{filepath}' ไม่ถูกต้อง จะใช้ค่าเริ่มต้นแทน")
        return {
            "assets": ["FFWM", "NEGG", "RIVN", "BTC-USD", "NVDA", "TSLA", "META"],
            "default_settings": {
                "selected_ticker": "FFWM", "start_date": "2024-01-01", "window_size": 30,
                "num_samples": 5000, "max_workers": 8, "master_seed": 42
            }
        }

def initialize_session_state(config: Dict[str, Any]):
    """ ตั้งค่าเริ่มต้นสำหรับ Streamlit session state """
    defaults = config.get('default_settings', {})

    if 'test_ticker' not in st.session_state:
        st.session_state.test_ticker = defaults.get('selected_ticker', 'FFWM')
    if 'start_date' not in st.session_state:
        st.session_state.start_date = datetime.strptime(defaults.get('start_date', '2024-01-01'), '%Y-%m-%d').date()
    if 'end_date' not in st.session_state:
        st.session_state.end_date = datetime.now().date()
    if 'window_size' not in st.session_state:
        st.session_state.window_size = defaults.get('window_size', 30)
    
    # Parameters for Random, Arithmetic, and Geometric search
    if 'num_samples' not in st.session_state:
        st.session_state.num_samples = defaults.get('num_samples', 5000)
    if 'max_workers' not in st.session_state:
        st.session_state.max_workers = defaults.get('max_workers', 8)
    if 'master_seed' not in st.session_state:
        st.session_state.master_seed = defaults.get('master_seed', 42)


# ==============================================================================
# 2. Core Calculation & Data Functions (With Numba Acceleration)
# ==============================================================================
@st.cache_data(ttl=3600)
def get_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """ ดึงข้อมูลราคาหุ้น/สินทรัพย์จาก Yahoo Finance """
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
    """ คำนวณผลลัพธ์การจำลองการเทรด (หัวใจของการคำนวณ) """
    n = len(action_array)
    if n == 0 or len(price_array) == 0:
        empty_arr = np.empty(0, dtype=np.float64)
        return (empty_arr, empty_arr, empty_arr, empty_arr, empty_arr, empty_arr)

    action_array_calc = action_array.copy()
    if n > 0: action_array_calc[0] = 1

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

@lru_cache(maxsize=16384)
def calculate_optimized_cached(action_tuple: Tuple[int, ...], price_tuple: Tuple[float, ...], fix: int = 1500) -> Tuple:
    action_array = np.asarray(action_tuple, dtype=np.int32)
    price_array = np.asarray(price_tuple, dtype=np.float64)
    return _calculate_simulation_numba(action_array, price_array, fix)

def run_simulation(prices: List[float], actions: List[int], fix: int = 1500) -> pd.DataFrame:
    if not prices or not actions: return pd.DataFrame()
    min_len = min(len(prices), len(actions)); prices, actions = prices[:min_len], actions[:min_len]
    buffer, sumusd, cash, asset_value, amount, refer = calculate_optimized_cached(tuple(actions), tuple(prices), fix)
    if len(sumusd) == 0: return pd.DataFrame()
    initial_capital = sumusd[0]
    return pd.DataFrame({'price': prices, 'action': actions, 'buffer': np.round(buffer, 2),
                         'sumusd': np.round(sumusd, 2), 'cash': np.round(cash, 2),
                         'asset_value': np.round(asset_value, 2), 'amount': np.round(amount, 2),
                         'refer': np.round(refer + initial_capital, 2),
                         'net': np.round(sumusd - refer - initial_capital, 2)})

@njit(cache=True)
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

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
        j_indices = np.arange(i)
        profits = fix * ((price_arr[i] / price_arr[j_indices]) - 1)
        current_sumusd = dp[j_indices] + profits
        best_idx = np.argmax(current_sumusd); dp[i] = current_sumusd[best_idx]; path[i] = j_indices[best_idx]
    actions = np.zeros(n, dtype=int); current_day = np.argmax(dp)
    while current_day > 0:
        actions[current_day] = 1; current_day = path[current_day]
    actions[0] = 1
    return actions

# 3.2 Standard Random Seed Search
def generate_actions_sliding_window_random(ticker_data: pd.DataFrame, window_size: int, num_samples: int, max_workers: int) -> Tuple[np.ndarray, pd.DataFrame]:
    prices = ticker_data['Close'].to_numpy(); n = len(prices)
    final_actions = np.array([], dtype=int); window_details_list = []
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="ประมวลผล Random Search...")

    # Shared function for parallel execution
    def evaluate_seed_batch(seed_batch: np.ndarray, prices_tuple: Tuple[float,...]) -> List[Tuple[int, float]]:
        results = []
        window_len = len(prices_tuple)
        for seed in seed_batch:
            rng = np.random.default_rng(seed)
            actions_window = rng.integers(0, 2, size=window_len)
            _, sumusd, _, _, _, refer = calculate_optimized_cached(tuple(actions_window), prices_tuple)
            net = sumusd[-1] - refer[-1] - sumusd[0] if len(sumusd) > 0 else -np.inf
            results.append((seed, net))
        return results
    
    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n); prices_window = prices[start_index:end_index]
        window_len = len(prices_window)
        if window_len < 2: continue
        
        best_seed_for_window, max_net_for_window = -1, -np.inf
        random_seeds = np.arange(num_samples)
        batch_size = max(1, num_samples // (max_workers * 4 if max_workers > 0 else 1))
        seed_batches = [random_seeds[j:j+batch_size] for j in range(0, len(random_seeds), batch_size)]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(evaluate_seed_batch, batch, tuple(prices_window)) for batch in seed_batches]
            for future in as_completed(futures):
                for seed, final_net in future.result():
                    if final_net > max_net_for_window:
                        max_net_for_window = final_net; best_seed_for_window = seed

        rng_best = np.random.default_rng(best_seed_for_window if best_seed_for_window >= 0 else 1)
        best_actions = rng_best.integers(0, 2, size=window_len); best_actions[0] = 1
        final_actions = np.concatenate((final_actions, best_actions))
        
        start_date_str = ticker_data.index[start_index].strftime('%Y-%m-%d'); end_date_str = ticker_data.index[end_index-1].strftime('%Y-%m-%d')
        window_details_list.append({'window_number': i + 1, 'timeline': f"{start_date_str} ถึง {end_date_str}",
                                    'best_seed': best_seed_for_window, 'max_net': round(max_net_for_window, 2),
                                    'action_count': int(np.sum(best_actions)), 'window_size': window_len})
        progress_bar.progress((i + 1) / num_windows, text=f"ประมวลผล Window {i+1}/{num_windows}")
        
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)

# 3.3 NEW MODEL: Arithmetic Sequence
def generate_actions_sliding_window_arithmetic(ticker_data: pd.DataFrame, window_size: int, num_samples: int, master_seed: int) -> Tuple[np.ndarray, pd.DataFrame]:
    prices = ticker_data['Close'].to_numpy(); n = len(prices)
    final_actions = np.array([], dtype=np.int32); window_details_list = []
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="ประมวลผล Arithmetic Sequence Search...")
    
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

# 3.4 NEW MODEL: Geometric Sequence
def generate_actions_sliding_window_geometric(ticker_data: pd.DataFrame, window_size: int, num_samples: int, master_seed: int) -> Tuple[np.ndarray, pd.DataFrame]:
    prices = ticker_data['Close'].to_numpy(); n = len(prices)
    final_actions = np.array([], dtype=np.int32); window_details_list = []
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="ประมวลผล Geometric Sequence Search...")

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
    st.write("⚙️ **การตั้งค่าพารามิเตอร์**")
    asset_list = config.get('assets', ['FFWM'])
    st.session_state.test_ticker = st.selectbox("เลือก Ticker สำหรับทดสอบ", options=asset_list, index=asset_list.index(st.session_state.test_ticker) if st.session_state.test_ticker in asset_list else 0)

    col1, col2 = st.columns(2)
    st.session_state.start_date = col1.date_input("วันที่เริ่มต้น", value=st.session_state.start_date)
    st.session_state.end_date = col2.date_input("วันที่สิ้นสุด", value=st.session_state.end_date)
    if st.session_state.start_date >= st.session_state.end_date:
        st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")

    st.divider()
    st.subheader("พารามิเตอร์สำหรับการทดสอบ (ใช้ร่วมกัน)")
    c1, c2, c3 = st.columns(3)
    st.session_state.window_size = c1.number_input("ขนาด Window (วัน)", min_value=2, value=st.session_state.window_size)
    st.session_state.num_samples = c2.number_input("จำนวน Samples ต่อ Window", min_value=100, value=st.session_state.num_samples, format="%d", help="จำนวนการสุ่ม Seed หรือ Parameters ที่จะใช้ในแต่ละ Window")
    st.session_state.master_seed = c3.number_input("Master Seed", value=st.session_state.master_seed, format="%d", help="Seed หลักเพื่อผลลัพธ์ที่ทำซ้ำได้สำหรับโมเดล Arithmetic/Geometric")
    # st.session_state.max_workers = c3.number_input("จำนวน Workers (CPU Cores)", min_value=1, max_value=16, value=st.session_state.max_workers, help="สำหรับ Random Search เท่านั้น")


def display_comparison_charts(results: Dict[str, pd.DataFrame]):
    if not results: return
    valid_dfs = {name: df for name, df in results.items() if not df.empty and 'net' in df.columns}
    if not valid_dfs: return
    
    try: longest_index = max((df.index for df in valid_dfs.values()), key=len)
    except ValueError: return

    chart_data = pd.DataFrame(index=longest_index)
    for name, df in valid_dfs.items():
        chart_data[name] = df['net'].reindex(longest_index).ffill()
    
    st.line_chart(chart_data)
    
    st.subheader("สรุปผลลัพธ์สุดท้าย (Final Net Profit)")
    final_nets = {name: df['net'].iloc[-1] for name, df in valid_dfs.items()}
    sorted_strategies = sorted(final_nets, key=final_nets.get, reverse=True)
    
    metric_cols = st.columns(len(sorted_strategies))
    for i, name in enumerate(sorted_strategies):
        metric_cols[i].metric(name, f"${final_nets[name]:,.2f}")


def run_and_display_results(strategy_func, strategy_name, ticker_data, df_cols):
    """Helper function to run a strategy and display results."""
    prices = ticker_data['Close'].to_numpy(); num_days = len(prices)

    with st.spinner(f"กำลังคำนวณกลยุทธ์ {strategy_name}..."):
        # Call the appropriate strategy function with its required arguments
        if strategy_name == Strategy.SLIDING_WINDOW_RANDOM:
            actions, df_windows = strategy_func(ticker_data, st.session_state.window_size, st.session_state.num_samples, st.session_state.max_workers)
        else: # Arithmetic or Geometric
            actions, df_windows = strategy_func(ticker_data, st.session_state.window_size, st.session_state.num_samples, st.session_state.master_seed)
        
        actions_min = generate_actions_rebalance_daily(num_days)
        actions_max = generate_actions_perfect_foresight(prices.tolist())
        
        results = {
            strategy_name: run_simulation(prices.tolist(), actions.tolist()),
            Strategy.PERFECT_FORESIGHT: run_simulation(prices.tolist(), actions_max.tolist()),
            Strategy.REBALANCE_DAILY: run_simulation(prices.tolist(), actions_min.tolist())
        }
        for name, df in results.items():
            if not df.empty: df.index = ticker_data.index[:len(df)]
    
    st.success("การทดสอบเสร็จสมบูรณ์!")
    st.write("---")
    display_comparison_charts(results)
    st.write(f"📈 **สรุปผลการค้นหาด้วย {strategy_name}**")
    st.dataframe(df_windows[df_cols], use_container_width=True)

# Main UI Tabs
def render_random_search_tab(ticker_data):
    st.info("กลยุทธ์นี้จะ **'สุ่ม'** Action Sequence จำนวนมาก แล้วเลือกอันที่ให้ผลตอบแทนดีที่สุดในแต่ละ Window")
    if st.button("🚀 เริ่มทดสอบ Best Seed (Random Search)", type="primary"):
        run_and_display_results(generate_actions_sliding_window_random, Strategy.SLIDING_WINDOW_RANDOM, ticker_data,
                                ['window_number', 'timeline', 'best_seed', 'max_net', 'action_count'])

def render_arithmetic_tab(ticker_data):
    st.info("กลยุทธ์นี้จะค้นหาพารามิเตอร์ของ **ลำดับเลขคณิต (`a1`, `d`)** ที่สร้าง Action ที่ดีที่สุดในแต่ละ Window")
    if st.button("📈 เริ่มทดสอบ Arithmetic Sequence", type="primary"):
        run_and_display_results(generate_actions_sliding_window_arithmetic, Strategy.ARITHMETIC_SEQUENCE, ticker_data,
                                ['window_number', 'timeline', 'max_net', 'best_a1', 'best_d', 'action_count'])

def render_geometric_tab(ticker_data):
    st.info("กลยุทธ์นี้จะค้นหาพารามิเตอร์ของ **ลำดับเรขาคณิต (`a1`, `r`)** ที่สร้าง Action ที่ดีที่สุดในแต่ละ Window")
    if st.button("📉 เริ่มทดสอบ Geometric Sequence", type="primary"):
        run_and_display_results(generate_actions_sliding_window_geometric, Strategy.GEOMETRIC_SEQUENCE, ticker_data,
                                ['window_number', 'timeline', 'max_net', 'best_a1', 'best_r', 'action_count'])

# ==============================================================================
# 5. Main Application
# ==============================================================================
def main():
    st.set_page_config(page_title="Sequence Optimizer", page_icon="🎯", layout="wide")
    st.markdown("### 🎯 Sequence Based Strategy Optimizer")
    st.caption("เครื่องมือทดสอบและเปรียบเทียบกลยุทธ์การหา Action Sequence ที่ดีที่สุด")

    config = load_config()
    initialize_session_state(config)

    tab_list = ["⚙️ การตั้งค่า", "🚀 Random Search", "📈 Arithmetic Seq", "📉 Geometric Seq"]
    tab_settings, tab_random, tab_arithmetic, tab_geometric = st.tabs(tab_list)

    with tab_settings:
        render_settings_tab(config)

    # Pre-fetch data to be used by all test tabs
    ticker = st.session_state.test_ticker
    start_date_str = st.session_state.start_date.strftime('%Y-%m-%d')
    end_date_str = st.session_state.end_date.strftime('%Y-%m-%d')
    ticker_data = get_ticker_data(ticker, start_date_str, end_date_str)

    if ticker_data.empty:
        st.error(f"ไม่พบข้อมูลสำหรับ **{ticker}** ในช่วงวันที่ที่เลือก กรุณาปรับการตั้งค่าในแท็บ 'การตั้งค่า'")
        return

    with tab_random:
        render_random_search_tab(ticker_data)
    with tab_arithmetic:
        render_arithmetic_tab(ticker_data)
    with tab_geometric:
        render_geometric_tab(ticker_data)

    with st.expander("📖 คำอธิบายวิธีการทำงานของโมเดลใหม่"):
        st.markdown("""
        ### หลักการทำงานของแต่ละโมเดล:

        1.  **🚀 Random Search**:
            - **หลักการ**: Brute Force. สุ่ม Action (0 หรือ 1) ตรงๆ ในแต่ละวัน แล้วเลือกชุดที่ดีที่สุด
            - **พารามิเตอร์**: `Seed` (ตัวเลขสุ่ม)

        2.  **📈 Arithmetic Sequence**:
            - **หลักการ**: สร้างลำดับ Action จากสมการลำดับเลขคณิต `Action(t) = sigmoid(a1 + t * d)` โดย `t` คือลำดับวันใน Window
            - **การค้นหา**: สุ่มหาค่า `a1` (พจน์เริ่มต้น) และ `d` (ผลต่างร่วม) ที่ดีที่สุด
            - **ลักษณะ**: สร้าง Action ที่มีแนวโน้มเปลี่ยนแปลงอย่าง **คงที่** (ค่อยๆ เพิ่มขึ้น/ลดลง) ตลอดช่วง Window

        3.  **📉 Geometric Sequence**:
            - **หลักการ**: สร้างลำดับ Action จากสมการลำดับเรขาคณิต `Action(t) = sigmoid(a1 * r^t)`
            - **การค้นหา**: สุ่มหาค่า `a1` (พจน์เริ่มต้น) และ `r` (อัตราส่วนร่วม) ที่ดีที่สุด
            - **ลักษณะ**: สร้าง Action ที่มีแนวโน้มเปลี่ยนแปลงแบบ **ทวีคูณ** (เปลี่ยนแปลงช้าในช่วงแรกและเร็วขึ้นในช่วงหลัง หรือกลับกัน)

        **共通ขั้นตอน**:
        - ในแต่ละ Window ระบบจะสุ่มหาพารามิเตอร์ (Seed, a1, d, a1, r) ตามจำนวน `Num Samples`
        - นำพารามิเตอร์ชุดที่ให้กำไร (Net Profit) สูงสุดมาใช้สำหรับ Window นั้น
        - การคำนวณกำไรหลักถูกเร่งความเร็วด้วย **Numba** เพื่อให้การทดสอบรวดเร็ว
        """)

if __name__ == "__main__":
    main()
