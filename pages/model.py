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
# นำเข้า BitGenerator PCG64 โดยตรง
from numpy.random import Generator, PCG64

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================

class Strategy:
    REBALANCE_DAILY = "Rebalance Daily"
    PERFECT_FORESIGHT = "Perfect Foresight (Max)"
    # อัปเกรดชื่อโมเดลเดิมให้ชัดเจนขึ้น
    BRUTE_FORCE_OPTIMIZER = "Brute-Force PCG64 Optimizer"
    CHAOTIC_SLIDING_WINDOW = "Chaotic Seed Sliding Window"
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
                # เพิ่มค่าเริ่มต้นสำหรับ Brute-Force
                "num_seeds": 1000000
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
    # อัปเดต num_seeds ให้รองรับค่าสูง
    if 'num_seeds' not in st.session_state: st.session_state.num_seeds = defaults.get('num_seeds', 1000000)
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

# ==============================================================================
# 3. Strategy Action Generation (Optimized Brute-Force)
# ==============================================================================
def generate_actions_rebalance_daily(num_days: int) -> np.ndarray: return np.ones(num_days, dtype=np.int32)
# --- Perfect Foresight (omitted for brevity) ---

# [NEW] Fully JIT-compiled function for finding the best seed
@njit(cache=True)
def _find_best_seed_numba(prices_window: np.ndarray, num_seeds_to_try: int) -> Tuple[int, float]:
    window_len = prices_window.shape[0]
    max_net = -np.inf
    best_seed = -1

    # Loop through millions of seeds at C-level speed
    for seed in range(num_seeds_to_try):
        # Create PCG64 generator inside the JIT-compiled function
        # This is a key part of the optimization
        rng = Generator(PCG64(seed))
        actions_window = rng.integers(0, 2, size=window_len, dtype=np.int32)

        # Directly call the Numba-compiled simulation function
        _, sumusd, _, _, _, refer = _calculate_simulation_numba(actions_window, prices_window)
        
        if len(sumusd) > 0:
            net = sumusd[-1] - refer[-1] - sumusd[0]
            if net > max_net:
                max_net = net
                best_seed = seed
    
    return best_seed, max_net

# Wrapper function remains for compatibility
def find_best_seed_for_window(prices_window: np.ndarray, num_seeds_to_try: int) -> Tuple[int, float, np.ndarray]:
    window_len = len(prices_window)
    if window_len < 2: return 1, 0.0, np.ones(window_len, dtype=np.int32)

    # Call the new, ultra-fast Numba function
    best_seed, max_net = _find_best_seed_numba(prices_window, num_seeds_to_try)

    if best_seed >= 0:
        # Re-create the best action sequence from the found seed
        rng_best = Generator(PCG64(best_seed))
        best_actions = rng_best.integers(0, 2, size=window_len, dtype=np.int32)
        best_actions[0] = 1
    else: 
        best_seed = 1
        max_net = 0.0
        best_actions = np.ones(window_len, dtype=np.int32)
        
    return best_seed, max_net, best_actions

def generate_actions_sliding_window_brute_force(ticker_data: pd.DataFrame, window_size: int, num_seeds: int) -> Tuple[np.ndarray, pd.DataFrame]:
    prices = ticker_data['Close'].to_numpy()
    n = len(prices)
    final_actions = np.array([], dtype=np.int32)
    window_details_list = []
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="กำลังประมวลผล Brute-Force Optimizer...")
    
    st.write(f"🚀 **Brute-Force PCG64 Optimizer**")
    st.write(f"📊 ข้อมูลทั้งหมด: {n} วัน | ขนาด Window: {window_size} วัน | Seeds ต่อ Window: **{num_seeds:,}**")
    st.write("---")

    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        window_len = len(prices_window)
        if window_len == 0: continue
        
        # Call the new optimized function chain
        best_seed, max_net, best_actions = find_best_seed_for_window(prices_window, num_seeds)
        
        final_actions = np.concatenate((final_actions, best_actions))
        
        start_date_str = ticker_data.index[start_index].strftime('%Y-%m-%d')
        end_date_str = ticker_data.index[end_index-1].strftime('%Y-%m-%d')
        detail = {'window_number': i + 1, 'timeline': f"{start_date_str} ถึง {end_date_str}", 'best_seed': best_seed, 'max_net': round(max_net, 2), 'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2), 'action_count': int(np.sum(best_actions)), 'window_size': window_len, 'action_sequence': best_actions.tolist()}
        window_details_list.append(detail)
        progress_bar.progress((i + 1) / num_windows, text=f"Brute-forcing Window {i+1}/{num_windows}")
        
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)

# --- Other models like Chaotic Generator can be kept as they are ---

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

def render_brute_force_tab():
    st.write("---")
    st.markdown("### 🚀 Brute-Force PCG64 Optimizer")
    st.info("โมเดลนี้คือ 'เครื่องปั่นตัวเลข' ที่ถูกอัปเกรดให้มีความเร็วสูงสุดโดยใช้ Numba และ PCG64 สามารถทดสอบ Seeds เป็นล้านค่าได้ในเวลาอันสั้นเพื่อค้นหาผลลัพธ์ที่ดีที่สุด")
    if st.button("🚀 เริ่มทดสอบ Brute-Force Optimizer", type="primary"):
        if st.session_state.start_date >= st.session_state.end_date: st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้อง"); return
        ticker = st.session_state.test_ticker; start_date_str = st.session_state.start_date.strftime('%Y-%m-%d'); end_date_str = st.session_state.end_date.strftime('%Y-%m-%d')
        st.info(f"กำลังดึงข้อมูลสำหรับ **{ticker}** | {start_date_str} ถึง {end_date_str}")
        ticker_data = get_ticker_data(ticker, start_date_str, end_date_str)
        if ticker_data.empty: st.error("ไม่พบข้อมูลสำหรับ Ticker และช่วงวันที่ที่เลือก"); return
        prices = ticker_data['Close'].to_numpy(); num_days = len(prices)
        
        with st.spinner("กำลัง Brute-Force... (อาจใช้เวลาสักครู่แต่เร็วกว่าเดิมมาก)"):
            import time
            start_time = time.time()
            
            actions_brute, df_windows = generate_actions_sliding_window_brute_force(ticker_data, st.session_state.window_size, st.session_state.num_seeds)
            
            end_time = time.time()
            st.info(f"Brute-Force เสร็จสิ้นใน {end_time - start_time:.2f} วินาที!")

            # Prepare results for chart
            # ... (omitted for brevity)

        st.success("การทดสอบเสร็จสมบูรณ์!")
        # ... (Display results, charts, dataframe - omitted for brevity)


# --- Other render functions (omitted for brevity) ---
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
        
        **🚀 Brute-Force PCG64 Optimizer:**
        นี่คือโมเดล "เครื่องปั่นตัวเลข" ที่ถูกยกระดับไปอีกขั้นโดยมีเป้าหมายคือ **ความเร็วสูงสุด**
        - **PCG64:** ใช้ BitGenerator ที่ทันสมัยและเร็วที่สุดของ NumPy ในการสร้าง Seed
        - **Numba-Optimized Loop:** ย้ายลูปการทดสอบ Seed ทั้งหมดเข้าไปทำงานในโค้ดที่ถูก JIT-compile ด้วย Numba ทำให้สามารถทดสอบ Seed เป็นล้านๆ ค่าได้ในเวลาเพียงไม่กี่วินาที
        - **ผลลัพธ์ที่ดีกว่าผ่านปริมาณ:** ด้วยความสามารถในการทดสอบในสเกลที่ใหญ่ขึ้นมหาศาล โอกาสในการค้นพบ Action Sequence ที่ให้ผลตอบแทนเข้าใกล้ **Perfect Foresight** จึงมีมากขึ้นอย่างมีนัยสำคัญ
        """)

if __name__ == "__main__":
    main()
