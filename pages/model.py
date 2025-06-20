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

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================

class Strategy:
    REBALANCE_DAILY = "Rebalance Daily"
    PERFECT_FORESIGHT = "Perfect Foresight (Max)"
    SLIDING_WINDOW = "Best Seed Sliding Window"
    # เพิ่มกลยุทธ์ใหม่
    GREEDY_LOOKAHEAD = "Greedy Lookahead Optimizer"
    MANUAL_SEED = "Manual Seed Strategy"

def load_config(filepath: str = "dynamic_seed_config.json") -> Dict[str, Any]:
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
                # เพิ่มค่าเริ่มต้นสำหรับ Greedy Lookahead
                "greedy_lookahead_days": 5
            },
            "manual_seed_by_asset": {"default": [{'seed': 999, 'size': 50, 'tail': 15}]}
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
    # เพิ่ม state สำหรับ Greedy Lookahead
    if 'greedy_lookahead_days' not in st.session_state: st.session_state.greedy_lookahead_days = defaults.get('greedy_lookahead_days', 5)
    if 'df_for_analysis' not in st.session_state: st.session_state.df_for_analysis = None
    if 'manual_action_sequence' not in st.session_state: st.session_state.manual_action_sequence = "[1, 0, 1]"

# ==============================================================================
# 2. Core Calculation & Data Functions (No Changes)
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
def _calculate_simulation_numba(action_array: np.ndarray, price_array: np.ndarray, fix: int = 1500) -> Tuple:
    n = len(action_array)
    if n == 0 or len(price_array) == 0:
        empty_arr = np.empty(0, dtype=np.float64); return (empty_arr, empty_arr, empty_arr, empty_arr, empty_arr, empty_arr)
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
        if action_array_calc[i] == 0: amount[i] = amount[i-1]; buffer[i] = 0.0
        else: amount[i] = fix / curr_price; buffer[i] = amount[i-1] * curr_price - fix
        cash[i] = cash[i-1] + buffer[i]
        asset_value[i] = amount[i] * curr_price; sumusd[i] = cash[i] + asset_value[i]
    return buffer, sumusd, cash, asset_value, amount, refer

@lru_cache(maxsize=4096)
def calculate_optimized_cached(action_tuple: Tuple[int, ...], price_tuple: Tuple[float, ...], fix: int = 1500) -> Tuple:
    action_array = np.asarray(action_tuple, dtype=np.int32); price_array = np.asarray(price_tuple, dtype=np.float64)
    return _calculate_simulation_numba(action_array, price_array, fix)

def run_simulation(prices: List[float], actions: List[int], fix: int = 1500) -> pd.DataFrame:
    if not prices or not actions: return pd.DataFrame()
    min_len = min(len(prices), len(actions)); prices = prices[:min_len]; actions = actions[:min_len]
    action_array = np.asarray(actions, dtype=np.int32); price_array = np.asarray(prices, dtype=np.float64)
    buffer, sumusd, cash, asset_value, amount, refer = _calculate_simulation_numba(action_array, price_array, fix)
    if len(sumusd) == 0: return pd.DataFrame()
    initial_capital = sumusd[0]
    return pd.DataFrame({'price': prices, 'action': actions, 'buffer': np.round(buffer, 2),'sumusd': np.round(sumusd, 2), 'cash': np.round(cash, 2), 'asset_value': np.round(asset_value, 2), 'amount': np.round(amount, 2), 'refer': np.round(refer + initial_capital, 2), 'net': np.round(sumusd - refer - initial_capital, 2)})

# ==============================================================================
# 3. Strategy Action Generation
# ==============================================================================
def generate_actions_rebalance_daily(num_days: int) -> np.ndarray: return np.ones(num_days, dtype=np.int32)
@njit(cache=True)
def generate_actions_perfect_foresight(price_arr: np.ndarray, fix: int = 1500) -> np.ndarray:
    n = len(price_arr)
    if n < 2: return np.ones(n, dtype=np.int32)
    dp = np.zeros(n, dtype=np.float64); path = np.zeros(n, dtype=np.int32); dp[0] = float(fix * 2)
    for i in range(1, n):
        j_indices = np.arange(i); profits = fix * ((price_arr[i] / price_arr[j_indices]) - 1.0)
        current_sumusd = dp[j_indices] + profits; best_idx = np.argmax(current_sumusd)
        dp[i] = current_sumusd[best_idx]; path[i] = j_indices[best_idx]
    actions = np.zeros(n, dtype=np.int32); best_final_day = np.argmax(dp); current_day = best_final_day
    while current_day > 0: actions[current_day] = 1; current_day = path[current_day]
    actions[0] = 1
    return actions

# --- Random Seed functions (omitted for brevity, they are correct in previous versions) ---
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
    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        window_len = len(prices_window)
        if window_len == 0: continue
        best_seed, max_net, best_actions = find_best_seed_for_window(prices_window, num_seeds_to_try, max_workers)
        final_actions = np.concatenate((final_actions, best_actions))
        start_date_str = ticker_data.index[start_index].strftime('%Y-%m-%d')
        end_date_str = ticker_data.index[end_index-1].strftime('%Y-%m-%d')
        detail = {'window_number': i + 1, 'timeline': f"{start_date_str} ถึง {end_date_str}", 'best_seed': best_seed, 'max_net': round(max_net, 2), 'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2), 'action_count': int(np.sum(best_actions)), 'window_size': window_len, 'action_sequence': best_actions.tolist()}
        window_details_list.append(detail)
        progress_bar.progress((i + 1) / num_windows, text=f"ประมวลผล Window {i+1}/{num_windows}")
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)

# 3.4 [NEW] Greedy Lookahead Optimizer
@njit(cache=True)
def generate_greedy_lookahead_actions_for_window(prices: np.ndarray, lookahead_days: int) -> np.ndarray:
    n = len(prices)
    if n == 0: return np.array([], dtype=np.int32)
    
    actions = np.zeros(n, dtype=np.int32)
    actions[0] = 1 # บังคับ Rebalance วันแรกเสมอ
    
    for t in range(1, n):
        # ที่จุด t, เราต้องตัดสินใจว่าจะ Hold หรือ Rebalance
        current_price = prices[t]
        
        # 1. ถ้า Hold, ผลตอบแทนเฉพาะหน้าคือ 0
        
        # 2. ถ้า Rebalance, หาโอกาสทำกำไรที่ดีที่สุดใน lookahead_days ข้างหน้า
        lookahead_end = min(t + 1 + lookahead_days, n)
        future_prices = prices[t + 1 : lookahead_end]
        
        if len(future_prices) == 0:
            # ไม่มีอนาคตให้ดู, เลือก Hold
            actions[t] = 0
            continue
            
        # หาจุดสูงสุดในอนาคตอันใกล้
        max_future_price = np.max(future_prices)
        
        # "กำไรที่คาดว่าจะได้" ถ้า Rebalance วันนี้
        potential_profit = max_future_price - current_price
        
        # 3. ตัดสินใจ
        if potential_profit > 0:
            actions[t] = 1 # มีโอกาสทำกำไร -> Rebalance
        else:
            actions[t] = 0 # ไม่มีโอกาสทำกำไรในระยะสั้น -> Hold
            
    return actions

def generate_actions_sliding_window_greedy(ticker_data: pd.DataFrame, window_size: int, lookahead_days: int) -> Tuple[np.ndarray, pd.DataFrame]:
    prices = ticker_data['Close'].to_numpy()
    n = len(prices)
    final_actions = np.array([], dtype=np.int32)
    window_details_list = []
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="กำลังประมวลผล Greedy Lookahead Optimizer...")
    
    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        window_len = len(prices_window)
        if window_len == 0: continue
        
        # สร้าง Actions สำหรับ Window นี้โดยเฉพาะ
        actions_for_window = generate_greedy_lookahead_actions_for_window(prices_window, lookahead_days)
        final_actions = np.concatenate((final_actions, actions_for_window))
        
        # คำนวณ Net เพื่อแสดงผล
        _, sumusd, _, _, _, refer = calculate_optimized_cached(tuple(actions_for_window), tuple(prices_window))
        max_net = sumusd[-1] - refer[-1] - sumusd[0] if len(sumusd) > 0 else 0.0
        
        start_date_str = ticker_data.index[start_index].strftime('%Y-%m-%d')
        end_date_str = ticker_data.index[end_index-1].strftime('%Y-%m-%d')
        detail = {'window_number': i + 1, 'timeline': f"{start_date_str} ถึง {end_date_str}", 'max_net': round(max_net, 2), 'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2), 'action_count': int(np.sum(actions_for_window)), 'window_size': window_len, 'action_sequence': actions_for_window.tolist()}
        window_details_list.append(detail)
        progress_bar.progress((i + 1) / num_windows, text=f"Optimizing Window {i+1}/{num_windows}")
        
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)


# ==============================================================================
# 4. UI Rendering Functions
# ==============================================================================
def render_settings_tab(config: Dict[str, Any]):
    st.write("⚙️ **การตั้งค่าพารามิเตอร์**")
    asset_list = config.get('assets', ['FFWM']); default_index = asset_list.index(st.session_state.test_ticker) if st.session_state.test_ticker in asset_list else 0
    st.session_state.test_ticker = st.selectbox("เลือก Ticker สำหรับทดสอบ", options=asset_list, index=default_index)
    st.write("📅 **ช่วงวันที่สำหรับการวิเคราะห์**"); col1, col2 = st.columns(2)
    with col1: st.session_state.start_date = st.date_input("วันที่เริ่มต้น", value=st.session_state.start_date)
    with col2: st.session_state.end_date = st.date_input("วันที่สิ้นสุด", value=st.session_state.end_date)
    if st.session_state.start_date >= st.session_state.end_date: st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
    st.divider()
    st.subheader("พารามิเตอร์ทั่วไป")
    st.session_state.window_size = st.number_input("ขนาด Window (วัน)", min_value=10, value=st.session_state.window_size)
    st.subheader("พารามิเตอร์สำหรับ Random Seed")
    c1, c2 = st.columns(2)
    st.session_state.num_seeds = c1.number_input("จำนวน Seeds ต่อ Window", min_value=100, value=st.session_state.num_seeds, format="%d")
    st.session_state.max_workers = c2.number_input("จำนวน Workers", min_value=1, max_value=16, value=st.session_state.max_workers)
    st.subheader("พารามิเตอร์สำหรับ Greedy Lookahead Optimizer")
    st.session_state.greedy_lookahead_days = st.number_input("จำนวนวันที่มองไปข้างหน้า (Lookahead Days)", min_value=1, max_value=st.session_state.window_size-1, value=st.session_state.greedy_lookahead_days)

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
    if st.button("🚀 เริ่มทดสอบ Best Seed (Random)", type="primary"):
        if st.session_state.start_date >= st.session_state.end_date: st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้อง"); return
        ticker = st.session_state.test_ticker; start_date_str = st.session_state.start_date.strftime('%Y-%m-%d'); end_date_str = st.session_state.end_date.strftime('%Y-%m-%d')
        st.info(f"กำลังดึงข้อมูลสำหรับ **{ticker}** | {start_date_str} ถึง {end_date_str}")
        ticker_data = get_ticker_data(ticker, start_date_str, end_date_str)
        if ticker_data.empty: st.error("ไม่พบข้อมูลสำหรับ Ticker และช่วงวันที่ที่เลือก"); return
        prices = ticker_data['Close'].to_numpy(); num_days = len(prices)
        with st.spinner("กำลังคำนวณกลยุทธ์ต่างๆ..."):
            actions_sliding, df_windows = generate_actions_sliding_window(ticker_data, st.session_state.window_size, st.session_state.num_seeds, st.session_state.max_workers)
            results = {
                Strategy.SLIDING_WINDOW: run_simulation(prices.tolist(), actions_sliding.tolist()),
                Strategy.REBALANCE_DAILY: run_simulation(prices.tolist(), generate_actions_rebalance_daily(num_days).tolist()),
                Strategy.PERFECT_FORESIGHT: run_simulation(prices.tolist(), generate_actions_perfect_foresight(prices).tolist())
            }
            for name, df in results.items():
                if not df.empty: df.index = ticker_data.index[:len(df)]
        st.success("การทดสอบเสร็จสมบูรณ์!"); st.write("---"); display_comparison_charts(results)
        st.write("📈 **สรุปผลการค้นหา Best Seed**")
        total_actions = df_windows['action_count'].sum(); total_net = df_windows['max_net'].sum()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Windows", df_windows.shape[0]); col2.metric("Total Actions", f"{total_actions}/{num_days}"); col3.metric("Total Net (Sum)", f"${total_net:,.2f}")
        st.dataframe(df_windows[['window_number', 'timeline', 'best_seed', 'max_net', 'price_change_pct', 'action_count']], use_container_width=True)
        csv = df_windows.to_csv(index=False); st.download_button(label="📥 ดาวน์โหลด Window Details (CSV)", data=csv, file_name=f'best_seed_{ticker}_{st.session_state.window_size}w.csv', mime='text/csv')

def render_greedy_lookahead_tab():
    st.write("---")
    st.markdown("### 🧠 ทดสอบด้วย Greedy Lookahead Optimizer")
    st.info("กลยุทธ์นี้จะตัดสินใจในแต่ละวันโดย 'แอบดู' อนาคตในระยะสั้น (ตามที่กำหนด) เพื่อเลือก Action ที่ให้กำไรเฉพาะหน้าดีที่สุด เป็นการจำลองกลยุทธ์ที่เกือบจะสมบูรณ์แบบ")
    if st.button("🚀 เริ่มทดสอบ Greedy Lookahead", type="primary", key="greedy_test_button"):
        if st.session_state.start_date >= st.session_state.end_date: st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้อง"); return
        ticker = st.session_state.test_ticker; start_date_str = st.session_state.start_date.strftime('%Y-%m-%d'); end_date_str = st.session_state.end_date.strftime('%Y-%m-%d')
        st.info(f"กำลังดึงข้อมูลสำหรับ **{ticker}** | {start_date_str} ถึง {end_date_str}")
        ticker_data = get_ticker_data(ticker, start_date_str, end_date_str)
        if ticker_data.empty: st.error("ไม่พบข้อมูลสำหรับ Ticker และช่วงวันที่ที่เลือก"); return
        prices = ticker_data['Close'].to_numpy(); num_days = len(prices)
        with st.spinner("กำลังคำนวณกลยุทธ์ต่างๆ..."):
            actions_greedy, df_windows = generate_actions_sliding_window_greedy(ticker_data, st.session_state.window_size, st.session_state.greedy_lookahead_days)
            results = {
                Strategy.GREEDY_LOOKAHEAD: run_simulation(prices.tolist(), actions_greedy.tolist()),
                Strategy.REBALANCE_DAILY: run_simulation(prices.tolist(), generate_actions_rebalance_daily(num_days).tolist()),
                Strategy.PERFECT_FORESIGHT: run_simulation(prices.tolist(), generate_actions_perfect_foresight(prices).tolist())
            }
            for name, df in results.items():
                if not df.empty: df.index = ticker_data.index[:len(df)]
        st.success("การทดสอบเสร็จสมบูรณ์!"); st.write("---"); display_comparison_charts(results)
        st.write("📈 **สรุปผลการทำงานของ Greedy Lookahead**")
        if not df_windows.empty:
            total_actions = df_windows['action_count'].sum(); total_net = df_windows['max_net'].sum()
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Windows", df_windows.shape[0]); col2.metric("Total Actions", f"{total_actions}/{num_days}"); col3.metric("Total Net (Sum)", f"${total_net:,.2f}")
            st.dataframe(df_windows[['window_number', 'timeline', 'max_net', 'price_change_pct', 'action_count']], use_container_width=True)

# --- Other render functions (analytics, manual) are omitted for brevity. ---
def render_analytics_tab(): st.info("This is the Analytics Tab.")
def render_manual_seed_tab(config): st.info("This is the Manual/Forward Test Tab.")

# ==============================================================================
# 5. Main Application
# ==============================================================================
def main():
    st.set_page_config(page_title="Strategy Optimization Lab", page_icon="🎯", layout="wide")
    st.markdown("🎯 Strategy Optimization Lab")
    st.caption("เครื่องมือทดสอบและเปรียบเทียบกลยุทธ์การลงทุน")

    config = load_config()
    initialize_session_state(config)

    tab_list = ["⚙️ การตั้งค่า", "🎲 Best Seed (Random)", "🧠 Greedy Lookahead", "📊 Analytics", "🌱 Manual / Forward Test"]
    tabs = st.tabs(tab_list)

    with tabs[0]: render_settings_tab(config)
    with tabs[1]: render_test_tab()
    with tabs[2]: render_greedy_lookahead_tab()
    with tabs[3]: render_analytics_tab()
    with tabs[4]: render_manual_seed_tab(config)

    with st.expander("📖 คำอธิบายกลยุทธ์"):
        st.markdown("""
        **หลักการพื้นฐาน:** กลยุทธ์ทั้งหมดทำงานบนหลักการ "Sliding Window" คือแบ่งช่วงเวลาทั้งหมดออกเป็น "หน้าต่าง" เล็กๆ เพื่อวิเคราะห์และสร้าง Action Sequence
        
        **กลยุทธ์ที่ใช้ในการค้นหา:**
        - **🎲 Best Seed (Random):** เป็นกลยุทธ์พื้นฐานที่ค้นหาโดยการสุ่ม Action Sequence จำนวนมากเพื่อหาชุดที่ดีที่สุดในแต่ละ Window
        - **🧠 Greedy Lookahead Optimizer:** เป็นกลยุทธ์ขั้นสูงที่พยายามเข้าใกล้ **Perfect Foresight** โดยในแต่ละวันจะ 'แอบดู' ราคาในอนาคตเป็นระยะสั้นๆ (Lookahead) เพื่อตัดสินใจเลือก Action (Hold/Rebalance) ที่ให้กำไรเฉพาะหน้าดีที่สุด
        
        **Benchmark Strategies:**
        - **Rebalance Daily:** เส้นฐาน (Floor) ของประสิทธิภาพ คือการ Rebalance ทุกวันโดยไม่มีเงื่อนไข
        - **Perfect Foresight:** เพดานสูงสุด (Ceiling) ของประสิทธิภาพ คือกลยุทธ์ที่รู้ราคาในอนาคตทั้งหมดและเลือกวันที่ Rebalance ได้สมบูรณ์แบบ
        """)

if __name__ == "__main__":
    main()
