import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import math
import json
import ast
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
    BRUTE_FORCE_OPTIMIZER = "Brute-Force Optimizer"
    CHAOTIC_SLIDING_WINDOW = "Chaotic Seed Sliding Window"
    MANUAL_ACTION = "Manual Action Sequence"

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
                "num_seeds": 1000000
            }
        }

def initialize_session_state(config: Dict[str, Any]):
    defaults = config.get('default_settings', {})
    if 'test_ticker' not in st.session_state: st.session_state.test_ticker = defaults.get('selected_ticker', 'FFWM')
    if 'start_date' not in st.session_state:
        try: st.session_state.start_date = datetime.strptime(defaults.get('start_date', '2024-01-01'), '%Y-%m-%d').date()
        except ValueError: st.session_state.start_date = datetime(2024, 1, 1).date()
    if 'end_date' not in st.session_state: st.session_state.end_date = datetime.now().date()
    if 'window_size' not in st.session_state: st.session_state.window_size = defaults.get('window_size', 30)
    if 'num_seeds' not in st.session_state: st.session_state.num_seeds = defaults.get('num_seeds', 1000000)
    if 'df_for_analysis' not in st.session_state: st.session_state.df_for_analysis = None
    if 'manual_action_sequence' not in st.session_state: st.session_state.manual_action_sequence = "[1, 0, 1, 0, 1, 1, 0, 0, 1, 0]"

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

# [UPDATED] Fully JIT-compiled function for finding the best seed
@njit(cache=True)
def _find_best_seed_numba(prices_window: np.ndarray, num_seeds_to_try: int) -> Tuple[int, float]:
    window_len = prices_window.shape[0]
    max_net = -np.inf
    best_seed = -1
    for seed in range(num_seeds_to_try):
        # *** ใช้ np.random.seed ที่ Numba รู้จัก ***
        np.random.seed(seed)
        # *** ใช้ np.random.randint ที่ Numba รู้จัก ***
        actions_window = np.random.randint(0, 2, size=window_len).astype(np.int32)
        _, sumusd, _, _, _, refer = _calculate_simulation_numba(actions_window, prices_window)
        if len(sumusd) > 0:
            net = sumusd[-1] - refer[-1] - sumusd[0]
            if net > max_net:
                max_net = net
                best_seed = seed
    return best_seed, max_net

def find_best_seed_for_window(prices_window: np.ndarray, num_seeds_to_try: int) -> Tuple[int, float, np.ndarray]:
    window_len = len(prices_window)
    if window_len < 2: return 1, 0.0, np.ones(window_len, dtype=np.int32)
    best_seed, max_net = _find_best_seed_numba(prices_window, num_seeds_to_try)
    if best_seed >= 0:
        np.random.seed(best_seed)
        best_actions = np.random.randint(0, 2, size=window_len).astype(np.int32)
        best_actions[0] = 1
    else: 
        best_seed = 1; max_net = 0.0; best_actions = np.ones(window_len, dtype=np.int32)
    return best_seed, max_net, best_actions

def generate_actions_sliding_window_brute_force(ticker_data: pd.DataFrame, window_size: int, num_seeds: int) -> Tuple[np.ndarray, pd.DataFrame]:
    prices = ticker_data['Close'].to_numpy()
    n = len(prices)
    final_actions = np.array([], dtype=np.int32); window_details_list = []
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="กำลังประมวลผล Brute-Force Optimizer...")
    st.write(f"🚀 **Brute-Force Optimizer**")
    st.write(f"📊 ข้อมูลทั้งหมด: {n} วัน | ขนาด Window: {window_size} วัน | Seeds ต่อ Window: **{num_seeds:,}**")
    st.write("---")
    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        window_len = len(prices_window)
        if window_len == 0: continue
        best_seed, max_net, best_actions = find_best_seed_for_window(prices_window, num_seeds)
        final_actions = np.concatenate((final_actions, best_actions))
        start_date_str = ticker_data.index[start_index].strftime('%Y-%m-%d')
        end_date_str = ticker_data.index[end_index-1].strftime('%Y-%m-%d')
        detail = {'window_number': i + 1, 'timeline': f"{start_date_str} ถึง {end_date_str}", 'best_seed': best_seed, 'max_net': round(max_net, 2), 'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2), 'action_count': int(np.sum(best_actions)), 'window_size': window_len, 'action_sequence': best_actions.tolist()}
        window_details_list.append(detail)
        progress_bar.progress((i + 1) / num_windows, text=f"Brute-forcing Window {i+1}/{num_windows}")
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
    st.subheader("พารามิเตอร์สำหรับ Brute-Force Optimizer")
    st.session_state.window_size = st.number_input("ขนาด Window (วัน)", min_value=10, value=st.session_state.window_size)
    st.session_state.num_seeds = st.number_input("จำนวน Seeds ที่จะทดสอบต่อ Window", min_value=1000, max_value=10000000, value=st.session_state.num_seeds, format="%d", help="สามารถใส่ค่าเป็นล้านได้ การคำนวณถูก Optimize ด้วย Numba แล้ว")

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

def render_brute_force_tab():
    st.markdown("### 🚀 Brute-Force Optimizer")
    st.info("โมเดล 'เครื่องปั่นตัวเลข' ที่ถูกอัปเกรดให้มีความเร็วสูงสุดโดยใช้ Numba สามารถทดสอบ Seeds เป็นล้านค่าได้ในเวลาอันสั้นเพื่อค้นหาผลลัพธ์ที่ดีที่สุด")
    if st.button("🚀 เริ่มทดสอบ Brute-Force Optimizer", type="primary"):
        if st.session_state.start_date >= st.session_state.end_date: st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้อง"); return
        ticker = st.session_state.test_ticker; start_date_str = st.session_state.start_date.strftime('%Y-%m-%d'); end_date_str = st.session_state.end_date.strftime('%Y-%m-%d')
        st.info(f"กำลังดึงข้อมูลสำหรับ **{ticker}** | {start_date_str} ถึง {end_date_str}")
        ticker_data = get_ticker_data(ticker, start_date_str, end_date_str)
        if ticker_data.empty: st.error("ไม่พบข้อมูลสำหรับ Ticker และช่วงวันที่ที่เลือก"); return
        prices = ticker_data['Close'].to_numpy(); num_days = len(prices)
        
        with st.spinner("กำลัง Brute-Force... (อาจใช้เวลาสักครู่แต่เร็วกว่าเดิมมาก)"):
            import time; start_time = time.time()
            actions_brute, df_windows = generate_actions_sliding_window_brute_force(ticker_data, st.session_state.window_size, st.session_state.num_seeds)
            end_time = time.time(); st.success(f"Brute-Force เสร็จสิ้นใน {end_time - start_time:.2f} วินาที!")
            
            actions_min = generate_actions_rebalance_daily(num_days)
            actions_max = generate_actions_perfect_foresight(prices)
            results = {
                Strategy.BRUTE_FORCE_OPTIMIZER: run_simulation(prices.tolist(), actions_brute.tolist()),
                Strategy.REBALANCE_DAILY: run_simulation(prices.tolist(), actions_min.tolist()),
                Strategy.PERFECT_FORESIGHT: run_simulation(prices.tolist(), actions_max.tolist())
            }
            for name, df in results.items():
                if not df.empty: df.index = ticker_data.index[:len(df)]
        
        display_comparison_charts(results)
        st.write("📈 **สรุปผลการค้นหา Best Seed**")
        total_actions = df_windows['action_count'].sum(); total_net = df_windows['max_net'].sum()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Windows", df_windows.shape[0]); col2.metric("Total Actions", f"{total_actions}/{num_days}"); col3.metric("Total Net (Sum)", f"${total_net:,.2f}")
        st.dataframe(df_windows[['window_number', 'timeline', 'best_seed', 'max_net', 'price_change_pct', 'action_count']], use_container_width=True)
        csv = df_windows.to_csv(index=False); st.download_button(label="📥 ดาวน์โหลด Window Details (CSV)", data=csv, file_name=f'bruteforce_{ticker}_{st.session_state.window_size}w.csv', mime='text/csv')

def render_analytics_tab(): st.info("This is the Analytics Tab.")
def render_manual_seed_tab(config): st.info("This is the Manual/Forward Test Tab.")
    
# ==============================================================================
# 5. Main Application
# ==============================================================================
def main():
    st.set_page_config(page_title="Brute-Force Optimizer", page_icon="🚀", layout="wide")
    st.markdown("## 🚀 Brute-Force Optimizer Lab")
    st.caption("เครื่องมือทดสอบกลยุทธ์ด้วยการค้นหาแบบ Exhaustive Search ที่ความเร็วสูง")
    config = load_config()
    initialize_session_state(config)
    tab_list = ["⚙️ การตั้งค่า", "🚀 Brute-Force Optimizer", "📊 Analytics", "🌱 Manual / Forward Test"]
    tabs = st.tabs(tab_list)
    with tabs[0]: render_settings_tab(config)
    with tabs[1]: render_brute_force_tab()
    with tabs[2]: render_analytics_tab()
    with tabs[3]: render_manual_seed_tab(config)
    with st.expander("📖 คำอธิบายกลยุทธ์"):
        st.markdown("""
        **หลักการพื้นฐาน:** กลยุทธ์นี้ยังคงทำงานบนหลักการ "Sliding Window"
        **🚀 Brute-Force Optimizer:** นี่คือโมเดล "เครื่องปั่นตัวเลข" ที่ถูกยกระดับไปอีกขั้นโดยมีเป้าหมายคือ **ความเร็วสูงสุด**
        - **Numba-Optimized Loop:** ย้ายลูปการทดสอบ Seed ทั้งหมดเข้าไปทำงานในโค้ดที่ถูก JIT-compile ด้วย Numba ทำให้สามารถทดสอบ Seed เป็นล้านๆ ค่าได้ในเวลาเพียงไม่กี่วินาที
        - **ผลลัพธ์ที่ดีกว่าผ่านปริมาณ:** ด้วยความสามารถในการทดสอบในสเกลที่ใหญ่ขึ้นมหาศาล โอกาสในการค้นพบ Action Sequence ที่ให้ผลตอบแทนเข้าใกล้ **Perfect Foresight** จึงมีมากขึ้นอย่างมีนัยสำคัญ
        """)

if __name__ == "__main__":
    main()
