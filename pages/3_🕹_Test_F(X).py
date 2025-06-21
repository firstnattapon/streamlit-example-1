import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from datetime import datetime
from typing import List, Tuple, Dict, Any

# ! NUMBA: Import Numba's Just-In-Time compiler for core acceleration
from numba import njit, float64, int32
import math

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================

class Strategy:
    REBALANCE_DAILY = "Rebalance Daily (Min)"
    PERFECT_FORESIGHT = "Perfect Foresight (Max)"
    CHAOS_WALK_FORWARD = "Chaotic System (Walk-Forward)"

class ChaosEquation:
    LOGISTIC_MAP = "Logistic Map"
    SINE_MAP = "Sine Map"
    TENT_MAP = "Tent Map"

def initialize_session_state():
    """ตั้งค่าเริ่มต้นสำหรับ Streamlit session state"""
    if 'test_ticker' not in st.session_state: st.session_state.test_ticker = 'MARA'
    if 'start_date' not in st.session_state: st.session_state.start_date = datetime(2023, 1, 1).date()
    if 'end_date' not in st.session_state: st.session_state.end_date = datetime.now().date()
    if 'fix_capital' not in st.session_state: st.session_state.fix_capital = 1500
    if 'window_size' not in st.session_state: st.session_state.window_size = 60
    if 'num_params_to_try' not in st.session_state: st.session_state.num_params_to_try = 5000
    if 'selected_chaos_eq' not in st.session_state: st.session_state.selected_chaos_eq = ChaosEquation.LOGISTIC_MAP

# ==============================================================================
# 2. Core Calculation & Data Functions
# ==============================================================================
@st.cache_data(ttl=3600)
def get_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        data = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=True)[['Close']]
        if data.empty: return pd.DataFrame()
        return data
    except Exception as e:
        st.error(f"❌ ไม่สามารถดึงข้อมูล {ticker} ได้: {str(e)}"); return pd.DataFrame()

@njit(cache=True)
def _calculate_net_profit_numba(action_array: np.ndarray, price_array: np.ndarray, fix: int) -> float:
    n = len(action_array)
    if n < 2: return 0.0
    action_array_calc = action_array.copy(); action_array_calc[0] = 1
    cash, amount = fix, fix / price_array[0]
    
    for i in range(1, n):
        if action_array_calc[i] == 1:
            buffer = amount * price_array[i] - fix
            cash += buffer
            amount = fix / price_array[i]
    
    final_sumusd = cash + amount * price_array[-1]
    initial_sumusd = 2 * fix
    refer_profit = -fix * math.log(price_array[0] / price_array[-1])
    net = final_sumusd - (initial_sumusd + refer_profit)
    return net

# ==============================================================================
# 3. Chaotic Action Generation
# ==============================================================================

# --- Numba-accelerated Chaos Generators ---
@njit(float64[:](int32, float64, float64), cache=True)
def _generate_logistic_map(length, r, x0):
    x_series = np.empty(length, dtype=np.float64)
    x = x0
    for i in range(length):
        x = r * x * (1.0 - x)
        x_series[i] = x
    return x_series

@njit(float64[:](int32, float64, float64), cache=True)
def _generate_sine_map(length, r, x0):
    x_series = np.empty(length, dtype=np.float64)
    x = x0
    for i in range(length):
        x = r * math.sin(math.pi * x)
        x_series[i] = x
    return x_series

@njit(float64[:](int32, float64, float64), cache=True)
def _generate_tent_map(length, mu, x0):
    x_series = np.empty(length, dtype=np.float64)
    x = x0
    for i in range(length):
        x = mu * min(x, 1.0 - x)
        x_series[i] = x
    return x_series

def generate_actions_from_chaos(equation: str, length: int, param: float, x0: float) -> np.ndarray:
    """สร้าง Action Sequence จากสมการ Chaos ที่เลือก"""
    if equation == ChaosEquation.LOGISTIC_MAP:
        x_series = _generate_logistic_map(length, param, x0)
    elif equation == ChaosEquation.SINE_MAP:
        x_series = _generate_sine_map(length, param, x0)
    elif equation == ChaosEquation.TENT_MAP:
        x_series = _generate_tent_map(length, param, x0)
    else:
        raise ValueError("Unknown chaos equation")
    
    actions = (x_series > 0.5).astype(np.int32)
    if length > 0: actions[0] = 1
    return actions

# --- Optimizer ---
def find_best_chaos_params(prices_window: np.ndarray, equation: str, num_params_to_try: int, fix: int) -> Dict:
    """ค้นหาพารามิเตอร์ที่ดีที่สุดสำหรับสมการ Chaos ที่กำหนด"""
    window_len = len(prices_window)
    if window_len < 2: return {'best_param': 0, 'best_x0': 0, 'best_net': 0}
    
    # Define parameter ranges for each equation
    if equation == ChaosEquation.LOGISTIC_MAP: param_range = (3.57, 4.0)
    elif equation == ChaosEquation.SINE_MAP: param_range = (0.7, 1.0)
    elif equation == ChaosEquation.TENT_MAP: param_range = (1.0, 2.0)
    else: param_range = (0, 1)

    # Generate random parameters to test
    rng = np.random.default_rng()
    params_to_test = rng.uniform(param_range[0], param_range[1], num_params_to_try)
    x0s_to_test = rng.uniform(0.01, 0.99, num_params_to_try)
    
    best_net = -np.inf
    best_param, best_x0 = 0.0, 0.0

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(generate_actions_from_chaos, equation, window_len, p, x0): (p, x0) for p, x0 in zip(params_to_test, x0s_to_test)}
        
        for future in as_completed(futures):
            params_tuple = futures[future]
            try:
                actions = future.result()
                current_net = _calculate_net_profit_numba(actions, prices_window, fix)
                if current_net > best_net:
                    best_net = current_net
                    best_param, best_x0 = params_tuple
            except Exception:
                pass # Ignore errors from bad parameters

    return {'best_param': best_param, 'best_x0': best_x0, 'best_net': best_net}

# --- Walk-Forward Strategy ---
def generate_chaos_walk_forward(ticker_data: pd.DataFrame, equation: str, window_size: int, num_params: int, fix: int) -> Tuple[np.ndarray, pd.DataFrame]:
    prices = ticker_data['Close'].to_numpy()
    n = len(prices)
    final_actions, window_details = np.array([], dtype=int), []
    num_windows = n // window_size
    progress_bar = st.progress(0)
    
    # Initial actions for the first test window
    best_actions_for_next_window = np.ones(window_size, dtype=np.int32)

    for i in range(num_windows - 1):
        learn_start, learn_end = i * window_size, (i + 1) * window_size
        test_start, test_end = learn_end, learn_end + window_size
        
        learn_prices, learn_dates = prices[learn_start:learn_end], ticker_data.index[learn_start:learn_end]
        test_prices, test_dates = prices[test_start:test_end], ticker_data.index[test_start:test_end]
        
        # Learn from the past
        search_result = find_best_chaos_params(learn_prices, equation, num_params, fix)
        
        # Test on the future (using results from previous loop)
        final_actions = np.concatenate((final_actions, best_actions_for_next_window))
        walk_forward_net = _calculate_net_profit_numba(best_actions_for_next_window, test_prices, fix)
        
        window_details.append({
            'window_num': i + 1,
            'learn_period': f"{learn_dates[0]:%Y-%m-%d} to {learn_dates[-1]:%Y-%m-%d}",
            'best_param': round(search_result['best_param'], 4),
            'best_x0': round(search_result['best_x0'], 4),
            'test_period': f"{test_dates[0]:%Y-%m-%d} to {test_dates[-1]:%Y-%m-%d}",
            'walk_forward_net': round(walk_forward_net, 2)
        })
        
        # Prepare actions for the *next* loop
        best_actions_for_next_window = generate_actions_from_chaos(
            equation, window_size, search_result['best_param'], search_result['best_x0']
        )
        progress_bar.progress((i + 1) / (num_windows - 1))

    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details)

# --- Benchmarks ---
@njit(cache=True)
def _generate_perfect_foresight_numba(price_arr: np.ndarray, fix: int) -> np.ndarray:
    n=len(price_arr);actions=np.zeros(n,np.int32)
    if n<2:return np.ones(n,np.int32)
    dp,path=np.zeros(n),np.zeros(n,np.int32)
    dp[0]=float(fix*2)
    for i in range(1,n):
        profits=fix*((price_arr[i]/price_arr[:i])-1)
        current_sumusd=dp[:i]+profits
        best_j_idx=np.argmax(current_sumusd)
        dp[i],path[i]=current_sumusd[best_j_idx],best_j_idx
    current_day=np.argmax(dp)
    while current_day>0:actions[current_day],current_day=1,path[current_day]
    actions[0]=1
    return actions

# ==============================================================================
# 4. UI Rendering Functions
# ==============================================================================
def render_settings_tab():
    st.write("⚙️ **พารามิเตอร์พื้นฐาน**")
    c1, c2, c3 = st.columns(3)
    c1.text_input("Ticker", key="test_ticker")
    c2.date_input("วันที่เริ่มต้น", key="start_date")
    c3.date_input("วันที่สิ้นสุด", key="end_date")
    
    st.divider()
    st.write("🧠 **พารามิเตอร์สำหรับโมเดล Chaotic System**")
    s_c1, s_c2, s_c3 = st.columns(3)
    s_c1.selectbox("เลือกสมการ Chaos", [ChaosEquation.LOGISTIC_MAP, ChaosEquation.SINE_MAP, ChaosEquation.TENT_MAP], key="selected_chaos_eq")
    s_c2.number_input("ขนาด Window (วัน)", min_value=10, key="window_size")
    s_c3.number_input("จำนวนพารามิเตอร์ที่จะทดสอบ", min_value=1000, step=1000, key="num_params_to_try")

def render_model_tab():
    st.markdown(f"### 🧠 Chaotic System Optimizer: *{st.session_state.selected_chaos_eq}*")
    
    if st.button("🚀 เริ่มการค้นหาและวิเคราะห์", type="primary"):
        with st.spinner(f"ดึงข้อมูล **{st.session_state.test_ticker}**..."):
            ticker_data = get_ticker_data(st.session_state.test_ticker, str(st.session_state.start_date), str(st.session_state.end_date))
        if ticker_data.empty: return
        
        prices_np, prices_list = ticker_data['Close'].to_numpy(), ticker_data['Close'].tolist()
        
        with st.spinner(f"กำลังค้นหาพารามิเตอร์ที่ดีที่สุดสำหรับ '{st.session_state.selected_chaos_eq}' แบบ Walk-Forward..."):
            actions_chaos, df_windows = generate_chaos_walk_forward(
                ticker_data, st.session_state.selected_chaos_eq, st.session_state.window_size,
                st.session_state.num_params_to_try, st.session_state.fix_capital
            )
        st.success("วิเคราะห์เสร็จสมบูรณ์!")

        # Run simulations for comparison
        results = {}
        with st.spinner("กำลังจำลองกลยุทธ์เพื่อเปรียบเทียบ..."):
            sim_len = len(actions_chaos)
            strategy_map = {
                Strategy.CHAOS_WALK_FORWARD: actions_chaos.tolist(),
                Strategy.PERFECT_FORESIGHT: _generate_perfect_foresight_numba(prices_np[:sim_len], st.session_state.fix_capital).tolist(),
                Strategy.REBALANCE_DAILY: np.ones(sim_len, dtype=np.int32).tolist(),
            }
            for name, actions in strategy_map.items():
                net = _calculate_net_profit_numba(np.array(actions), prices_np[:sim_len], st.session_state.fix_capital)
                # Create a simple dataframe for charting
                results[name] = pd.DataFrame({'net': [net]}) # Storing final net for metric display
                # For charting, we need a series. Let's run the full simulation to get the cumulative net.
                # This is a bit inefficient but necessary for the line chart.
                full_sim_net = _calculate_simulation_numba(np.array(actions, dtype=np.int32), prices_np[:sim_len], st.session_state.fix_capital)
                results[f"{name}_chart"] = pd.DataFrame({'net': full_sim_net}, index=ticker_data.index[:sim_len])


        # Display charts and metrics
        chart_data = pd.DataFrame({name: df['net'] for name, df in results.items() if name.endswith('_chart')})
        st.line_chart(chart_data)

        if not df_windows.empty:
            final_net_chaos = df_windows['walk_forward_net'].sum()
            final_net_max = results[f"{Strategy.PERFECT_FORESIGHT}_chart"]['net'].iloc[-1]
            final_net_min = results[f"{Strategy.REBALANCE_DAILY}_chart"]['net'].iloc[-1]
            
            col1, col2, col3 = st.columns(3)
            col1.metric(f"🥇 {Strategy.PERFECT_FORESIGHT}", f"${final_net_max:,.2f}")
            col2.metric(f"🧠 {Strategy.CHAOS_WALK_FORWARD}", f"${final_net_chaos:,.2f}", delta=f"{final_net_chaos - final_net_min:,.2f} vs Min")
            col3.metric(f"🥉 {Strategy.REBALANCE_DAILY}", f"${final_net_min:,.2f}")
        
        st.dataframe(df_windows)

# ==============================================================================
# 5. Main Application
# ==============================================================================
def main():
    st.set_page_config(page_title="Chaotic System Optimizer", page_icon="🌀", layout="wide")
    st.markdown("### 🌀 Chaotic System Optimizer")
    st.caption("โมเดลค้นหาพารามิเตอร์ที่ดีที่สุดสำหรับสมการ Chaos ต่างๆ และตรวจสอบด้วย Walk-Forward Validation")

    initialize_session_state()

    tab_settings, tab_model = st.tabs(["⚙️ การตั้งค่า", "🚀 วิเคราะห์และแสดงผล"])
    with tab_settings: render_settings_tab()
    with tab_model: render_model_tab()
    
    with st.expander("📖 คำอธิบายสมการ Chaos ต่างๆ"):
        st.markdown("""
        ### 1. Logistic Map (Default)
        - **สมการ:** `x = r * x * (1 - x)`
        - **พารามิเตอร์ `r`:** ช่วง `[3.57, 4.0]`
        - **ลักษณะ:** เป็นสมการ Chaos ที่คลาสสิกที่สุด ให้พฤติกรรมที่ซับซ้อนและมีการแตกแขนง (Bifurcation) ที่สวยงาม เป็นมาตรฐานที่ดีในการเปรียบเทียบ

        ### 2. Sine Map
        - **สมการ:** `x = r * sin(π * x)`
        - **พารามิเตอร์ `r`:** ช่วง `[0.7, 1.0]`
        - **ลักษณะ:** ให้รูปแบบที่ "นุ่มนวล" และต่อเนื่องกว่า Logistic Map เนื่องจากใช้ฟังก์ชัน `sin` การกระจายตัวของค่า `x` จะแตกต่างออกไปอย่างชัดเจน

        ### 3. Tent Map
        - **สมการ:** `x = μ * min(x, 1 - x)`
        - **พารามิเตอร์ `μ`:** ช่วง `[1.0, 2.0]`
        - **ลักษณะ:** เรียบง่ายและคำนวณเร็วมาก มีจุดเด่นคือการให้ค่า `x` ที่มีการกระจายตัวสม่ำเสมอ (Uniform Distribution) ทั่วทั้งช่วง ทำให้ไม่เกิดการกระจุกตัวของ Action
        """)

if __name__ == "__main__":
    main()
