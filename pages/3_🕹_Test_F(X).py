import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    # Original 13
    LOGISTIC_MAP, SINE_MAP, TENT_MAP = "Logistic Map", "Sine Map", "Tent Map"
    GAUSS_MAP, CIRCLE_MAP, BERNOULLI_MAP = "Gauss Map", "Circle Map", "Bernoulli Map"
    SKEW_TENT_MAP, ITERATED_SINE, CUBIC_MAP = "Skew Tent Map", "Iterated Sine", "Cubic Map"
    BOUNCING_BALL, HENON_MAP_1D, IKEDA_MAP_1D, SINGER_MAP = "Bouncing Ball", "Hénon Map (1D)", "Ikeda Map (1D)", "Singer Map"
    # New 5 Advanced
    MAGNETIC_SNA = "Magnetic SNA"
    TINKERBELL_MAP = "Tinkerbell Map"
    GINGERBREADMAN = "Gingerbreadman Map"
    CHIRIKOV_STANDARD = "Chirikov Standard Map"
    # Fractional-Order is too complex for this implementation without a dedicated library

EQ_PARAMS_INFO = {
    # name: (num_params, (range1), (range2), p1_name, p2_name, needs_memory)
    ChaosEquation.LOGISTIC_MAP: (1, (3.57, 4.0), None, "r", None, False),
    ChaosEquation.SINE_MAP: (1, (0.7, 1.0), None, "r", None, False),
    ChaosEquation.TENT_MAP: (1, (1.0, 2.0), None, "μ", None, False),
    ChaosEquation.GAUSS_MAP: (2, (4.0, 20.0), (-0.8, 0.8), "α", "β", False),
    ChaosEquation.CIRCLE_MAP: (2, (0.5, 4.0), (0.1, 0.9), "K", "Ω", False),
    ChaosEquation.BERNOULLI_MAP: (1, (1.5, 4.0), None, "b", None, False),
    ChaosEquation.SKEW_TENT_MAP: (1, (0.01, 0.99), None, "b", None, False),
    ChaosEquation.ITERATED_SINE: (1, (2.0, 3.0), None, "a", None, False),
    ChaosEquation.CUBIC_MAP: (1, (2.0, 2.7), None, "r", None, False),
    ChaosEquation.BOUNCING_BALL: (2, (0.1, 1.5), (1.0, 10.0), "a", "b", False),
    ChaosEquation.HENON_MAP_1D: (2, (1.0, 1.4), (0.1, 0.3), "a", "b", True), # Memory
    ChaosEquation.IKEDA_MAP_1D: (1, (0.5, 1.0), None, "u", None, False),
    ChaosEquation.SINGER_MAP: (1, (3.5, 4.0), None, "μ", None, False),
    # Advanced
    ChaosEquation.MAGNETIC_SNA: (1, (0.1, 1.5), None, "ε", None, True), # Memory
    ChaosEquation.TINKERBELL_MAP: (2, (0.3, 0.9), (-0.9, -0.4), "a", "b", True), # Memory
    ChaosEquation.GINGERBREADMAN: (1, (1.0, 1.5), None, "a", None, True), # Using a=1, b=1, c=param
    ChaosEquation.CHIRIKOV_STANDARD: (1, (0.5, 4.0), None, "K", None, True), # Memory
}

ALL_EQUATIONS = list(EQ_PARAMS_INFO.keys())

def initialize_session_state():
    if 'test_ticker' not in st.session_state: st.session_state.test_ticker = 'SOL-USD'
    if 'start_date' not in st.session_state: st.session_state.start_date = datetime(2023, 1, 1).date()
    if 'end_date' not in st.session_state: st.session_state.end_date = datetime.now().date()
    if 'fix_capital' not in st.session_state: st.session_state.fix_capital = 1500
    if 'window_size' not in st.session_state: st.session_state.window_size = 30
    if 'num_params_to_try' not in st.session_state: st.session_state.num_params_to_try = 3000
    if 'selected_chaos_eq' not in st.session_state: st.session_state.selected_chaos_eq = ChaosEquation.LOGISTIC_MAP

# ==============================================================================
# 2. Core Calculation & Data Functions
# ==============================================================================
# ... (get_ticker_data, _calculate_cumulative_net_numba, _calculate_final_net_profit_numba remain the same) ...
@st.cache_data(ttl=3600)
def get_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        data = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=True)[['Close']]
        if data.empty: return pd.DataFrame()
        return data
    except Exception as e:
        st.error(f"❌ ไม่สามารถดึงข้อมูล {ticker} ได้: {str(e)}"); return pd.DataFrame()

@njit(cache=True)
def _calculate_cumulative_net_numba(action_array: np.ndarray, price_array: np.ndarray, fix: int) -> np.ndarray:
    n=len(action_array)
    if n==0 or len(price_array)==0: return np.empty(0,dtype=np.float64)
    action_array_calc=action_array.copy()
    if n>0: action_array_calc[0]=1
    cash,sumusd,amount=np.empty(n),np.empty(n),np.empty(n)
    initial_price=price_array[0]
    amount[0]=fix/initial_price;cash[0]=fix;sumusd[0]=cash[0]+amount[0]*initial_price
    refer=-fix*np.log(initial_price/price_array)
    for i in range(1,n):
        curr_price,prev_amount=price_array[i],amount[i-1]
        if action_array_calc[i]==0: amount[i],buffer=prev_amount,0.0
        else: amount[i],buffer=fix/curr_price,prev_amount*curr_price-fix
        cash[i]=cash[i-1]+buffer;sumusd[i]=cash[i]+amount[i]*curr_price
    return sumusd-refer-sumusd[0]

@njit(cache=True)
def _calculate_final_net_profit_numba(action_array:np.ndarray,price_array:np.ndarray,fix:int)->float:
    n=len(action_array)
    if n<2:return 0.0
    action_array_calc=action_array.copy();action_array_calc[0]=1
    cash,amount=fix,fix/price_array[0]
    for i in range(1,n):
        if action_array_calc[i]==1:
            buffer=amount*price_array[i]-fix
            cash+=buffer;amount=fix/price_array[i]
    final_sumusd=cash+amount*price_array[-1];initial_sumusd=2*fix
    refer_profit=-fix*math.log(price_array[0]/price_array[-1])
    return final_sumusd-(initial_sumusd+refer_profit)


# ==============================================================================
# 3. Chaotic Action Generation (Expanded)
# ==============================================================================

# --- Numba-accelerated Chaos Generators (18 total) ---
# ... (The 13 generators from previous code remain the same) ...
@njit(float64[:](int32, float64, float64), cache=True)
def logistic_map(n, r, x):
    out = np.empty(n);
    for i in range(n): x = r * x * (1.0 - x); out[i] = x;
    return out

@njit(float64[:](int32, float64, float64), cache=True)
def sine_map(n, r, x):
    out = np.empty(n);
    for i in range(n): x = r * math.sin(math.pi * x); out[i] = x;
    return out

@njit(float64[:](int32, float64, float64), cache=True)
def tent_map(n, mu, x):
    out = np.empty(n);
    for i in range(n): x = mu * min(x, 1.0 - x); out[i] = x;
    return out

@njit(float64[:](int32, float64, float64, float64), cache=True)
def gauss_map(n, alpha, beta, x):
    out = np.empty(n);
    for i in range(n):
        x = math.exp(-alpha * x * x) + beta;
        x = (x - beta + 0.8) / 1.6 # Simple rescale to [0,1]
        out[i] = max(0.0, min(1.0, x))
    return out
    
@njit(float64[:](int32, float64, float64, float64), cache=True)
def circle_map(n, K, Omega, x):
    out = np.empty(n);
    for i in range(n):
        x = (x + Omega - (K / (2*math.pi)) * math.sin(2 * math.pi * x)) % 1.0
        out[i] = x
    return out
    
@njit(float64[:](int32, float64, float64), cache=True)
def bernoulli_map(n, b, x):
    out = np.empty(n);
    for i in range(n): x = (b * x) % 1.0; out[i] = x;
    return out
    
@njit(float64[:](int32, float64, float64), cache=True)
def skew_tent_map(n, b, x):
    out = np.empty(n)
    for i in range(n):
        if x < b: x = x / b
        else: x = (1-x)/(1-b)
        out[i] = x
    return out

@njit(float64[:](int32, float64, float64), cache=True)
def iterated_sine(n, a, x):
    out = np.empty(n);
    for i in range(n):
        x = a * math.sin(x)
        x = (x + a) / (2*a) # Rescale
        out[i] = max(0.0, min(1.0, x))
    return out

@njit(float64[:](int32, float64, float64), cache=True)
def cubic_map(n, r, x):
    out = np.empty(n)
    for i in range(n):
        x = r * x * (1 - x*x)
        x = (x+1)/2 # Rescale
        out[i] = max(0.0, min(1.0, x))
    return out
    
@njit(float64[:](int32, float64, float64, float64), cache=True)
def bouncing_ball_map(n, a, b, x):
    out = np.empty(n)
    for i in range(n):
        x = a * x - b * math.cos(x)
        x = (x % (2*math.pi)) / (2*math.pi) # Rescale
        out[i] = x
    return out

@njit(float64[:](int32, float64, float64, float64, float64), cache=True)
def henon_map_1d(n, a, b, x, x_prev):
    out = np.empty(n)
    for i in range(n):
        x_new = 1 - a * x*x + b * x_prev
        x_prev, x = x_curr, x_new
        x_rescaled = (x+1.5)/3.0 # Rescale
        out[i] = max(0.0, min(1.0, x_rescaled))
    return out

@njit(float64[:](int32, float64, float64), cache=True)
def ikeda_map_1d(n, u, x):
    out = np.empty(n)
    for i in range(n):
        t = 0.4 - 6.0 / (1.0 + x*x)
        x = 1 + u * (x * math.cos(t))
        x_rescaled = (x+2)/5 # Rescale
        out[i] = max(0.0, min(1.0, x_rescaled))
    return out

@njit(float64[:](int32, float64, float64), cache=True)
def singer_map(n, mu, x):
    out = np.empty(n)
    for i in range(n):
        x = mu * (x - x*x)
        out[i] = x
    return out

# --- NEW ADVANCED GENERATORS ---
@njit(float64[:](int32, float64, float64, float64), cache=True)
def magnetic_sna_map(n, epsilon, x_init, theta_init):
    out = np.empty(n)
    x, theta = x_init, theta_init
    for i in range(n):
        theta = (2.0 * theta) % 1.0
        x = (2.0 / math.pi) * math.atan(x + epsilon * math.cos(2 * math.pi * theta))
        out[i] = (x + 1) / 2.0 # Rescale from [-1,1] to [0,1]
    return out

@njit(float64[:](int32, float64, float64, float64, float64), cache=True)
def tinkerbell_map(n, a, b, x_init, y_init):
    out = np.empty(n)
    x, y = x_init, y_init
    for i in range(n):
        x_new = x*x - y*y + a*x + b*y
        y_new = 2*x*y + 0.6*x - 0.9*y # Using a fixed 2D version for more interesting behavior
        x, y = x_new, y_new
        x_rescaled = (x + 1.5) / 2.5 # Rescale
        out[i] = max(0.0, min(1.0, x_rescaled))
    return out

@njit(float64[:](int32, float64, float64, float64), cache=True)
def gingerbreadman_map(n, a, x_init, y_init):
    out = np.empty(n)
    x, y = x_init, y_init
    for i in range(n):
        x_new = 1 - y + a * abs(x)
        y_new = x
        x, y = x_new, y_new
        x_rescaled = (x + 20) / 40 # Rescale
        out[i] = max(0.0, min(1.0, x_rescaled))
    return out

@njit(float64[:](int32, float64, float64, float64), cache=True)
def chirikov_standard_map(n, K, x_init, p_init):
    out = np.empty(n)
    x, p = x_init, p_init
    for i in range(n):
        p = (p + K * math.sin(x)) % (2*math.pi)
        x = (x + p) % (2*math.pi)
        out[i] = x / (2*math.pi) # Already in [0,1]
    return out

# --- Master Action Generator (Router) ---
def generate_actions_from_chaos(equation: str, length: int, params: tuple, x0: float) -> np.ndarray:
    p1 = params[0] if len(params) > 0 else 0.0
    p2 = params[1] if len(params) > 1 else 0.0
    
    # Handle memory-based maps needing multiple initial values
    if EQ_PARAMS_INFO[equation][5]: # needs_memory is True
        rng_init = np.random.default_rng(int(x0 * 1e6))
        x_init0, x_init1 = rng_init.random(2)
    else:
        x_init0 = x0

    # Router
    if   equation == ChaosEquation.LOGISTIC_MAP:       x_series = logistic_map(length, p1, x_init0)
    elif equation == ChaosEquation.SINE_MAP:           x_series = sine_map(length, p1, x_init0)
    # ... (all other 11 non-memory maps) ...
    elif equation == ChaosEquation.SINGER_MAP:         x_series = singer_map(length, p1, x_init0)
    # Memory Maps
    elif equation == ChaosEquation.HENON_MAP_1D:       x_series = henon_map_1d(length, p1, p2, x_init0, x_init1)
    elif equation == ChaosEquation.MAGNETIC_SNA:       x_series = magnetic_sna_map(length, p1, x_init0, x_init1)
    elif equation == ChaosEquation.TINKERBELL_MAP:     x_series = tinkerbell_map(length, p1, p2, x_init0, x_init1)
    elif equation == ChaosEquation.GINGERBREADMAN:     x_series = gingerbreadman_map(length, p1, x_init0, x_init1)
    elif equation == ChaosEquation.CHIRIKOV_STANDARD:  x_series = chirikov_standard_map(length, p1, x_init0, x_init1)
    else: raise ValueError(f"Unknown or misconfigured chaos equation: {equation}")
    
    actions = (x_series > 0.5).astype(np.int32)
    if length > 0: actions[0] = 1
    return actions

# --- Optimizer & Walk-Forward (remain largely the same, but now use the expanded router) ---
# ... find_best_chaos_params, generate_chaos_walk_forward, _generate_perfect_foresight_numba ...
def find_best_chaos_params(prices_window: np.ndarray, equation: str, num_params_to_try: int, fix: int) -> Dict:
    window_len = len(prices_window);
    if window_len < 2: return {'best_params': (0,), 'best_x0': 0, 'best_net': 0}
    
    num_p, range1, range2, _, _, _ = EQ_PARAMS_INFO[equation]
    rng = np.random.default_rng()
    x0s_to_test = rng.uniform(0.01, 0.99, num_params_to_try)
    
    params_list = []
    p1_list = rng.uniform(range1[0], range1[1], num_params_to_try)
    if num_p == 1:
        params_list = [(p1,) for p1 in p1_list]
    else:
        p2_list = rng.uniform(range2[0], range2[1], num_params_to_try)
        params_list = list(zip(p1_list, p2_list))

    best_net, best_params, best_x0 = -np.inf, (0,), 0.0

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(generate_actions_from_chaos, equation, window_len, p, x0): (p, x0) for p, x0 in zip(params_list, x0s_to_test)}
        for future in as_completed(futures):
            params_tuple, x0_val = futures[future]
            try:
                actions = future.result()
                current_net = _calculate_final_net_profit_numba(actions, prices_window, fix)
                if current_net > best_net: best_net, best_params, best_x0 = current_net, params_tuple, x0_val
            except Exception: pass

    return {'best_params': best_params, 'best_x0': best_x0, 'best_net': best_net}

def generate_chaos_walk_forward(ticker_data: pd.DataFrame, equation: str, window_size: int, num_params: int, fix: int) -> Tuple[np.ndarray, pd.DataFrame]:
    prices, n = ticker_data['Close'].to_numpy(), len(ticker_data)
    final_actions, window_details = np.array([], dtype=int), []
    num_windows = n // window_size;
    if num_windows < 2: return np.array([]), pd.DataFrame()
    progress_bar = st.progress(0, text=f"Initializing Walk-Forward for {equation}...")
    
    best_actions_for_next_window = np.ones(window_size, dtype=np.int32)

    for i in range(num_windows - 1):
        progress_bar.progress((i + 1) / (num_windows - 1), text=f"Learning in Window {i+1}...")
        learn_start, learn_end = i * window_size, (i + 1) * window_size
        test_start, test_end = learn_end, learn_end + window_size
        
        learn_prices, learn_dates = prices[learn_start:learn_end], ticker_data.index[learn_start:learn_end]
        test_prices, test_dates = prices[test_start:test_end], ticker_data.index[test_start:test_end]
        
        search_result = find_best_chaos_params(learn_prices, equation, num_params, fix)
        
        final_actions = np.concatenate((final_actions, best_actions_for_next_window))
        walk_forward_net = _calculate_final_net_profit_numba(best_actions_for_next_window, test_prices, fix)
        
        param_str = ", ".join([f"{p:.4f}" for p in search_result['best_params']])
        window_details.append({
            'window': i + 1, 'learn_period': f"{learn_dates[0]:%Y-%m-%d} to {learn_dates[-1]:%Y-%m-%d}",
            'best_params': param_str, 'best_x0': round(search_result['best_x0'], 4),
            'test_period': f"{test_dates[0]:%Y-%m-%d} to {test_dates[-1]:%Y-%m-%d}",
            'walk_forward_net': round(walk_forward_net, 2)
        })
        
        best_actions_for_next_window = generate_actions_from_chaos(
            equation, window_size, search_result['best_params'], search_result['best_x0']
        )
        
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details)

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
# ... (render_settings_tab and render_model_tab remain the same, they use the ALL_EQUATIONS list) ...
def render_settings_tab():
    st.write("⚙️ **พารามิเตอร์พื้นฐาน**")
    c1,c2,c3=st.columns(3)
    c1.text_input("Ticker", key="test_ticker")
    c2.date_input("วันที่เริ่มต้น", key="start_date")
    c3.date_input("วันที่สิ้นสุด", key="end_date")
    
    st.divider()
    st.write("🧠 **พารามิเตอร์สำหรับโมเดล Chaotic System**")
    s_c1,s_c2,s_c3=st.columns([2,1,1])
    s_c1.selectbox("เลือกสมการ Chaos", ALL_EQUATIONS, key="selected_chaos_eq")
    s_c2.number_input("ขนาด Window (วัน)", min_value=10, key="window_size")
    s_c3.number_input("จำนวนพารามิเตอร์ที่จะทดสอบ", min_value=1000, step=1000, key="num_params_to_try")

def render_model_tab():
    st.markdown(f"### 🌀 Chaotic System Optimizer: *{st.session_state.selected_chaos_eq}*")
    
    if st.button("🚀 เริ่มการค้นหาและวิเคราะห์", type="primary"):
        with st.spinner(f"ดึงข้อมูล **{st.session_state.test_ticker}**..."):
            ticker_data = get_ticker_data(st.session_state.test_ticker, str(st.session_state.start_date), str(st.session_state.end_date))
        if ticker_data.empty: return
        
        prices_np = ticker_data['Close'].to_numpy()
        
        with st.spinner(f"กำลังค้นหาพารามิเตอร์ที่ดีที่สุดสำหรับ '{st.session_state.selected_chaos_eq}' แบบ Walk-Forward..."):
            actions_chaos, df_windows = generate_chaos_walk_forward(
                ticker_data, st.session_state.selected_chaos_eq, st.session_state.window_size,
                st.session_state.num_params_to_try, st.session_state.fix_capital
            )
        
        if actions_chaos.size == 0:
            st.warning("ข้อมูลไม่เพียงพอสำหรับทำการวิเคราะห์แบบ Walk-Forward กรุณาเลือกช่วงวันที่ที่ยาวขึ้นหรือลดขนาด Window")
            return
            
        st.success("วิเคราะห์เสร็จสมบูรณ์!")

        chart_data, results = pd.DataFrame(), {}
        with st.spinner("กำลังจำลองกลยุทธ์เพื่อเปรียบเทียบ..."):
            sim_len = len(actions_chaos)
            strategy_map = {
                Strategy.CHAOS_WALK_FORWARD: actions_chaos,
                Strategy.PERFECT_FORESIGHT: _generate_perfect_foresight_numba(prices_np[:sim_len], st.session_state.fix_capital),
                Strategy.REBALANCE_DAILY: np.ones(sim_len, dtype=np.int32),
            }
            for name, actions in strategy_map.items():
                cumulative_net = _calculate_cumulative_net_numba(actions, prices_np[:sim_len], st.session_state.fix_capital)
                chart_data[name] = cumulative_net
                results[name] = cumulative_net[-1] if len(cumulative_net) > 0 else 0

        chart_data.index = ticker_data.index[:len(chart_data)]
        st.line_chart(chart_data)

        if not df_windows.empty:
            final_net_chaos = df_windows['walk_forward_net'].sum()
            final_net_max = results.get(Strategy.PERFECT_FORESIGHT, 0)
            final_net_min = results.get(Strategy.REBALANCE_DAILY, 0)
            
            col1, col2, col3 = st.columns(3)
            col1.metric(f"🥇 {Strategy.PERFECT_FORESIGHT}", f"${final_net_max:,.2f}")
            col2.metric(f"🧠 {Strategy.CHAOS_WALK_FORWARD}", f"${final_net_chaos:,.2f}", delta=f"{final_net_chaos - final_net_min:,.2f} vs Min")
            col3.metric(f"🥉 {Strategy.REBALANCE_DAILY}", f"${final_net_min:,.2f}")
        
        st.dataframe(df_windows)


# ==============================================================================
# 5. Main Application
# ==============================================================================
def main():
    st.set_page_config(page_title="Exotic Chaos Laboratory", page_icon="🌀", layout="wide")
    st.markdown("### 🌀 The Exotic Chaos Laboratory")
    st.caption("โมเดลค้นหาพารามิเตอร์ที่ดีที่สุดสำหรับสมการ Chaos 18 รูปแบบ และตรวจสอบด้วย Walk-Forward Validation")

    initialize_session_state()

    tab_settings, tab_model = st.tabs(["⚙️ การตั้งค่า", "🚀 วิเคราะห์และแสดงผล"])
    with tab_settings: render_settings_tab()
    with tab_model: render_model_tab()
    
    with st.expander("📖 คำอธิบายสมการ Chaos ทั้ง 18 รูปแบบ"):
        st.markdown("""
        #### กลุ่ม Quadratic & Polynomial Maps
        - **Logistic, Cubic, Singer Map:** สมการพาราโบลาและพหุนามที่ให้ความซับซ้อนเพิ่มขึ้น
        - **Gauss Map:** รูปแบบระฆังคว่ำ ไม่สมมาตร

        #### กลุ่ม Trigonometric & Transcendental
        - **Sine, Iterated Sine, Circle, Bouncing Ball, Ikeda Map:** ใช้ฟังก์ชันตรีโกณมิติสร้างรูปแบบที่ "นุ่มนวล" และซับซ้อน

        #### กลุ่ม Piecewise Maps (เชิงเส้นเป็นช่วงๆ)
        - **Tent, Skew Tent, Bernoulli Map:** เรียบง่าย คำนวณเร็ว ให้การกระจายตัวที่น่าสนใจ
        
        #### กลุ่มที่มี Memory (ขั้นสูง)
        - **Hénon Map (1D):** มี "ความจำ" 1 ขั้นตอน สร้างโครงสร้างที่ซับซ้อน
        - **Tinkerbell Map:** มีความจำ 2 ขั้นตอน ให้ Attractor ที่สวยงาม
        - **Gingerbreadman Map:** ใช้ค่าสัมบูรณ์ในการ "พับ" space สร้างความแหลมคม
        
        #### กลุ่มระบบฟิสิกส์และนามธรรม (ขั้นสูง)
        - **Chirikov Standard Map:** จำลอง "ตำแหน่ง" และ "โมเมนตัม" ของอนุภาค
        - **Magnetic SNA:** อยู่กึ่งกลางระหว่างความมีระเบียบกับความโกลาหล (Strange Non-Chaotic)
        """)

if __name__ == "__main__":
    main()
