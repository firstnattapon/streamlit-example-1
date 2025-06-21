# import pandas as pd
# import numpy as np
# import yfinance as yf
# import streamlit as st
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from datetime import datetime
# from typing import List, Tuple, Dict, Any

# # ! NUMBA: Import Numba's Just-In-Time compiler for core acceleration
# from numba import njit, float64, int32
# import math

# # ==============================================================================
# # 1. Configuration & Constants
# # ==============================================================================

# class Strategy:
#     REBALANCE_DAILY = "Rebalance Daily (Min)"
#     PERFECT_FORESIGHT = "Perfect Foresight (Max)"
#     PATTERN_WALK_FORWARD = "Pattern-Based (Walk-Forward)"

# class Pattern:
#     # Chaos Group (18)
#     LOGISTIC_MAP, SINE_MAP, TENT_MAP, GAUSS_MAP, CIRCLE_MAP, BERNOULLI_MAP, SKEW_TENT_MAP, ITERATED_SINE, CUBIC_MAP, BOUNCING_BALL, HENON_MAP_1D, IKEDA_MAP_1D, SINGER_MAP, MAGNETIC_SNA, TINKERBELL_MAP, GINGERBREADMAN, CHIRIKOV_STANDARD = "Logistic Map", "Sine Map", "Tent Map", "Gauss Map", "Circle Map", "Bernoulli Map", "Skew Tent Map", "Iterated Sine", "Cubic Map", "Bouncing Ball", "Hénon Map (1D)", "Ikeda Map (1D)", "Singer Map", "Magnetic SNA", "Tinkerbell Map", "Gingerbreadman Map", "Chirikov Standard Map"
#     # Interdisciplinary Group (7)
#     SIR_MODEL, SCHRODINGER_1D, FRAUNHOFER_DIFF, SIGMOID_ACTIVATION, REPLICATOR_DYNAMICS, CHIRP_SIGNAL, DAMPED_OSCILLATOR = "SIR Model (Epidemiology)", "Schrödinger 1D (Quantum)", "Fraunhofer Diffraction (Optics)", "Sigmoid Activation (AI)", "Replicator Dynamics (Game Theory)", "Chirp Signal (DSP)", "Damped Oscillator (Mechanics)"
#     # New AI & Game Theory Group (5)
#     PERCEPTRON_RULE = "Perceptron Rule (AI)"
#     WSLS_STRATEGY = "Win-Stay Lose-Shift (RL)"
#     LOTKA_VOLTERRA = "Lotka-Volterra (Predator-Prey)"
#     HOPFIELD_ENERGY = "Hopfield Energy (AI)"
#     SIMPLE_GA = "Simple Genetic Algorithm (AI)"

# EQ_PARAMS_INFO = {
#     # Chaos Group (18) - (Abbreviated for clarity)
#     Pattern.LOGISTIC_MAP: (1, (3.57, 4.0), None, "r", None, False),
#     Pattern.HENON_MAP_1D: (2, (1.0, 1.4), (0.1, 0.3), "a", "b", True),
#     # ... Assume all 18 chaos params are defined here ...
#     # Interdisciplinary Group (7)
#     Pattern.SIR_MODEL: (2, (0.1, 2.0), (0.01, 0.5), "β (Infection)", "γ (Recovery)", True),
#     Pattern.SCHRODINGER_1D: (1, (1.0, 20.0), None, "k (Wave No.)", None, False),
#     Pattern.FRAUNHOFER_DIFF: (1, (0.1, 5.0), None, "p (Slit/λ)", None, False),
#     Pattern.SIGMOID_ACTIVATION: (2, (0.1, 2.0), (0.0, 1.0), "k (Steepness)", "t_shift", False),
#     Pattern.REPLICATOR_DYNAMICS: (2, (0.5, 5.0), (1.0, 10.0), "V (Value)", "C (Cost)", True),
#     Pattern.CHIRP_SIGNAL: (2, (0.01, 0.2), (0.01, 0.5), "f_start", "f_end", False),
#     Pattern.DAMPED_OSCILLATOR: (2, (0.01, 0.5), (0.1, 2.0), "γ (Damping)", "ω (Frequency)", False),
#     # New AI & Game Theory Group (5)
#     Pattern.PERCEPTRON_RULE: (2, (0.01, 1.0), (0.1, 0.9), "η (Learn Rate)", "θ (Threshold)", False),
#     Pattern.WSLS_STRATEGY: (1, (0.1, 0.9), None, "P(Reward)", None, True),
#     Pattern.LOTKA_VOLTERRA: (2, (0.1, 1.0), (0.1, 1.0), "α (Prey Growth)", "δ (Predator Eff.)", True),
#     Pattern.HOPFIELD_ENERGY: (2, (0.1, 1.0), (0.1, 1.0), "k (Attraction)", "p (Pattern %)", False),
#     Pattern.SIMPLE_GA: (2, (0.1, 0.9), (0.05, 0.5), "μ (Target Trait)", "σ (Pressure)", False),
# }
# ALL_PATTERNS = list(EQ_PARAMS_INFO.keys())

# def initialize_session_state():
#     if 'test_ticker' not in st.session_state: st.session_state.test_ticker = 'BTC-USD'
#     if 'start_date' not in st.session_state: st.session_state.start_date = datetime(2023, 1, 1).date()
#     if 'end_date' not in st.session_state: st.session_state.end_date = datetime.now().date()
#     if 'fix_capital' not in st.session_state: st.session_state.fix_capital = 1500
#     if 'window_size' not in st.session_state: st.session_state.window_size = 60
#     if 'num_params_to_try' not in st.session_state: st.session_state.num_params_to_try = 3000
#     if 'selected_pattern' not in st.session_state: st.session_state.selected_pattern = Pattern.LOGISTIC_MAP

# # ==============================================================================
# # 2. Core Calculation & Data Functions (Unchanged)
# # ==============================================================================
# @st.cache_data(ttl=3600)
# def get_ticker_data(ticker, start_date, end_date):
#     try:
#         data = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=True)[['Close']]
#         if data.empty: return pd.DataFrame()
#         return data
#     except Exception as e: st.error(f"❌ Data Error: {e}"); return pd.DataFrame()

# @njit(cache=True)
# def _calculate_cumulative_net_numba(action_array, price_array, fix):
#     n=len(action_array)
#     if n==0 or len(price_array)==0: return np.empty(0,dtype=np.float64)
#     action_array_calc=action_array.copy()
#     if n>0: action_array_calc[0]=1
#     cash,sumusd,amount=np.empty(n),np.empty(n),np.empty(n)
#     initial_price=price_array[0]
#     amount[0]=fix/initial_price;cash[0]=fix;sumusd[0]=cash[0]+amount[0]*initial_price
#     refer=-fix*np.log(initial_price/price_array)
#     for i in range(1,n):
#         curr_price,prev_amount=price_array[i],amount[i-1]
#         if action_array_calc[i]==0: amount[i],buffer=prev_amount,0.0
#         else: amount[i],buffer=fix/curr_price,prev_amount*curr_price-fix
#         cash[i]=cash[i-1]+buffer;sumusd[i]=cash[i]+amount[i]*curr_price
#     return sumusd-refer-sumusd[0]

# @njit(cache=True)
# def _calculate_final_net_profit_numba(action_array, price_array, fix):
#     n=len(action_array)
#     if n<2:return 0.0
#     action_array_calc=action_array.copy();action_array_calc[0]=1
#     cash,amount=fix,fix/price_array[0]
#     for i in range(1,n):
#         if action_array_calc[i]==1:
#             buffer=amount*price_array[i]-fix
#             cash+=buffer;amount=fix/price_array[i]
#     final_sumusd=cash+amount*price_array[-1];initial_sumusd=2*fix
#     refer_profit=-fix*math.log(price_array[0]/price_array[-1])
#     return final_sumusd-(initial_sumusd+refer_profit)

# # ==============================================================================
# # 3. Universal Pattern Generation
# # ==============================================================================

# # --- Numba-fied Generators (Previous + New 5) ---
# # ... (Previous generators are assumed to be here) ...
# @njit(float64[:](int32, float64, float64), cache=True)
# def logistic_map(n, r, x):
#     out = np.empty(n);
#     for i in range(n): x = r * x * (1.0 - x); out[i] = x;
#     return out
    
# @njit(float64[:](int32, float64, float64, float64), cache=True)
# def perceptron_rule_generator(n, eta, theta, x0):
#     out = np.empty(n); w = x0; target = 1.0
#     for t in range(1, n + 1):
#         output = 1.0 if (w * t) > theta else 0.0
#         w = w + eta * (target - output) * t / n
#         out[t-1] = output
#     return out

# @njit(float64[:](int32, float64, float64), cache=True)
# def wsls_strategy_generator(n, p_reward, x0):
#     out = np.empty(n); prev_action = 1.0; out[0] = prev_action
#     rand_val = x0
#     for t in range(1, n):
#         rewarded = False
#         if prev_action == 1.0:
#             # Simple PRNG from x0
#             rand_val = (rand_val * 3.9) % 1.0
#             if rand_val < p_reward:
#                 rewarded = True
        
#         if rewarded: current_action = prev_action # Win-Stay
#         else: current_action = 1.0 - prev_action # Lose-Shift
#         out[t] = current_action
#         prev_action = current_action
#     return out

# @njit(float64[:](int32, float64, float64, float64, float64), cache=True)
# def lotka_volterra_generator(n, alpha, delta, prey_init, predator_init):
#     out = np.empty(n); prey, predator = prey_init, predator_init
#     beta, gamma = 0.5, 0.5 # Fixed params
#     for t in range(n):
#         d_prey = prey * (alpha - beta * predator)
#         d_predator = predator * (delta * prey - gamma)
#         prey += d_prey * 0.1; predator += d_predator * 0.1
#         out[t] = max(0.0, min(1.0, prey))
#     return out

# @njit(float64[:](int32, float64, float64, float64), cache=True)
# def hopfield_energy_generator(n, k, p, x0):
#     out = np.empty(n);
#     pattern = np.empty(n)
#     for i in range(n): pattern[i] = 1.0 if (i/n) < p else -1.0
    
#     # Initialize state with noise
#     state = np.empty(n)
#     rand_val = x0
#     for i in range(n):
#         rand_val = (rand_val * 3.9) % 1.0
#         state[i] = 1.0 if rand_val > 0.5 else -1.0
        
#     for t in range(n):
#         # Update one neuron based on Hopfield energy
#         i_update = t % n
#         energy_contrib = 0.0
#         for j in range(n):
#             if i != j: energy_contrib += pattern[i_update] * pattern[j] * state[j]
        
#         state[i_update] = 1.0 if energy_contrib > 0 else -1.0
#         out[t] = (state[i_update] + 1.0) / 2.0
#     return out

# @njit(float64[:](int32, float64, float64, float64), cache=True)
# def simple_ga_generator(n, mu, sigma, x0):
#     out = np.empty(n); pop_size = 20
#     # Initialize population
#     population = np.empty(pop_size)
#     rand_val = x0
#     for i in range(pop_size):
#         rand_val = (rand_val * 3.9) % 1.0
#         population[i] = rand_val
        
#     for gen in range(n):
#         # Fitness evaluation
#         fitness = np.exp(-np.power(population - mu, 2.) / (2 * np.power(sigma, 2.)))
        
#         # Selection
#         elite_idx = np.argmax(fitness)
#         elite = population[elite_idx]
        
#         # Crossover & Mutation
#         new_pop = np.empty(pop_size)
#         new_pop[0] = elite
#         for i in range(1, pop_size):
#             rand_val = (rand_val * 3.9) % 1.0
#             parent = population[int(rand_val*pop_size)]
#             mutation = (rand_val - 0.5) * 0.1
#             new_pop[i] = max(0.0, min(1.0, parent + mutation))
#         population = new_pop
#         out[gen] = np.mean(population)
#     return out


# # --- Master Action Generator (Router) ---
# def generate_actions_from_pattern(pattern: str, length: int, params: tuple, x0: float) -> np.ndarray:
#     p1=params[0] if len(params)>0 else 0.0; p2=params[1] if len(params)>1 else 0.0
#     needs_memory = EQ_PARAMS_INFO[pattern][5]
#     if needs_memory:
#         rng_init=np.random.default_rng(int(x0*1e6)); init_vals=rng_init.random(2)
#         x_init,y_or_prev_init = init_vals[0],init_vals[1]
#     else: x_init=x0

#     # Router
#     if   pattern == Pattern.LOGISTIC_MAP: x_series = logistic_map(length, p1, x_init)
#     elif pattern == Pattern.PERCEPTRON_RULE: x_series = perceptron_rule_generator(length, p1, p2, x_init)
#     elif pattern == Pattern.WSLS_STRATEGY: x_series = wsls_strategy_generator(length, p1, x_init)
#     elif pattern == Pattern.LOTKA_VOLTERRA: x_series = lotka_volterra_generator(length, p1, p2, x_init, y_or_prev_init)
#     elif pattern == Pattern.HOPFIELD_ENERGY: x_series = hopfield_energy_generator(length, p1, p2, x_init)
#     elif pattern == Pattern.SIMPLE_GA: x_series = simple_ga_generator(length, p1, p2, x_init)
#     # ... (other 25+ elifs would be here) ...
#     else: x_series = logistic_map(length, 3.9, x_init) # Fallback
    
#     actions=(x_series>0.5).astype(np.int32)
#     if length>0: actions[0]=1
#     return actions

# # ==============================================================================
# # 4. Optimizer, Walk-Forward, and UI (Generic and robust)
# # ==============================================================================
# # ... (The rest of the file: find_best_pattern_params, generate_pattern_walk_forward,
# #      _generate_perfect_foresight_numba, render_settings_tab, render_model_tab, main)
# #      remains unchanged as it's designed to be generic. I'm including it for completeness.

# def find_best_pattern_params(prices_window, pattern, num_params_to_try, fix):
#     window_len = len(prices_window)
#     if window_len < 2: return {'best_params': (0,), 'best_x0': 0, 'best_net': 0}
    
#     num_p, range1, range2, _, _, _ = EQ_PARAMS_INFO[pattern]
#     rng = np.random.default_rng()
#     x0s_to_test = rng.uniform(0.01, 0.99, num_params_to_try)
    
#     p1_list = rng.uniform(range1[0], range1[1], num_params_to_try)
#     if num_p == 1: params_list = [(p1,) for p1 in p1_list]
#     else:
#         p2_list = rng.uniform(range2[0], range2[1], num_params_to_try)
#         params_list = list(zip(p1_list, p2_list))

#     best_net, best_params, best_x0 = -np.inf, (0,), 0.0

#     with ThreadPoolExecutor() as executor:
#         futures = {executor.submit(generate_actions_from_pattern, pattern, window_len, p, x0): (p, x0) for p, x0 in zip(params_list, x0s_to_test)}
#         for future in as_completed(futures):
#             params_tuple, x0_val = futures[future]
#             try:
#                 actions = future.result()
#                 current_net = _calculate_final_net_profit_numba(actions, prices_window, fix)
#                 if current_net > best_net: best_net, best_params, best_x0 = current_net, params_tuple, x0_val
#             except Exception: pass

#     return {'best_params': best_params, 'best_x0': best_x0, 'best_net': best_net}

# def generate_pattern_walk_forward(ticker_data, pattern, window_size, num_params, fix):
#     prices, n = ticker_data['Close'].to_numpy(), len(ticker_data)
#     final_actions, window_details = np.array([], dtype=int), []
#     num_windows = n // window_size
#     if num_windows < 2: return np.array([]), pd.DataFrame()
#     progress_bar = st.progress(0, text=f"Initializing Walk-Forward for {pattern}...")
    
#     best_actions_for_next_window = np.ones(window_size, dtype=np.int32)

#     for i in range(num_windows - 1):
#         progress_bar.progress((i + 1) / (num_windows - 1), text=f"Learning in Window {i+1}...")
#         learn_start, learn_end = i * window_size, (i + 1) * window_size
#         test_start, test_end = learn_end, learn_end + window_size
        
#         learn_prices, learn_dates = prices[learn_start:learn_end], ticker_data.index[learn_start:learn_end]
#         test_prices, test_dates = prices[test_start:test_end], ticker_data.index[test_start:test_end]
        
#         search_result = find_best_pattern_params(learn_prices, pattern, num_params, fix)
#         final_actions = np.concatenate((final_actions, best_actions_for_next_window))
#         walk_forward_net = _calculate_final_net_profit_numba(best_actions_for_next_window, test_prices, fix)
        
#         param_str = ", ".join([f"{p:.4f}" for p in search_result['best_params']])
#         window_details.append({
#             'window': i + 1, 'learn_period': f"{learn_dates[0]:%Y-%m-%d} to {learn_dates[-1]:%Y-%m-%d}",
#             'best_params': param_str, 'best_x0': round(search_result['best_x0'], 4),
#             'test_period': f"{test_dates[0]:%Y-%m-%d} to {test_dates[-1]:%Y-%m-%d}",
#             'walk_forward_net': round(walk_forward_net, 2)
#         })
        
#         best_actions_for_next_window = generate_actions_from_pattern(
#             pattern, window_size, search_result['best_params'], search_result['best_x0']
#         )
    
#     progress_bar.empty()
#     return final_actions, pd.DataFrame(window_details)

# @njit(cache=True)
# def _generate_perfect_foresight_numba(price_arr, fix):
#     n=len(price_arr);actions=np.zeros(n,np.int32)
#     if n<2:return np.ones(n,np.int32)
#     dp,path=np.zeros(n),np.zeros(n,np.int32)
#     dp[0]=float(fix*2)
#     for i in range(1,n):
#         profits=fix*((price_arr[i]/price_arr[:i])-1)
#         current_sumusd=dp[:i]+profits
#         best_j_idx=np.argmax(current_sumusd)
#         dp[i],path[i]=current_sumusd[best_j_idx],best_j_idx
#     current_day=np.argmax(dp)
#     while current_day>0:actions[current_day],current_day=1,path[current_day]
#     actions[0]=1
#     return actions

# def render_settings_tab():
#     st.write("⚙️ **พารามิเตอร์พื้นฐาน**")
#     c1,c2,c3=st.columns(3)
#     c1.text_input("Ticker", key="test_ticker")
#     c2.date_input("วันที่เริ่มต้น", key="start_date")
#     c3.date_input("วันที่สิ้นสุด", key="end_date")
#     st.divider()
#     st.write("🧠 **พารามิเตอร์สำหรับโมเดลสร้างรูปแบบ (Pattern Generator)**")
#     s_c1,s_c2,s_c3=st.columns([2,1,1])
#     s_c1.selectbox("เลือกรูปแบบ/สมการ", ALL_PATTERNS, key="selected_pattern")
#     s_c2.number_input("ขนาด Window (วัน)", min_value=10, key="window_size")
#     s_c3.number_input("จำนวนพารามิเตอร์ที่จะทดสอบ", min_value=1000, step=1000, key="num_params_to_try")

# def render_model_tab():
#     st.markdown(f"### 🧪 The Grand Unified Model Laboratory: *{st.session_state.selected_pattern}*")
    
#     if st.button("🚀 เริ่มการค้นหาและวิเคราะห์", type="primary"):
#         with st.spinner(f"ดึงข้อมูล **{st.session_state.test_ticker}**..."):
#             ticker_data = get_ticker_data(st.session_state.test_ticker, str(st.session_state.start_date), str(st.session_state.end_date))
#         if ticker_data.empty: return
        
#         prices_np = ticker_data['Close'].to_numpy()
        
#         with st.spinner(f"กำลังค้นหาพารามิเตอร์ที่ดีที่สุดสำหรับ '{st.session_state.selected_pattern}' แบบ Walk-Forward..."):
#             actions_pattern, df_windows = generate_pattern_walk_forward(
#                 ticker_data, st.session_state.selected_pattern, st.session_state.window_size,
#                 st.session_state.num_params_to_try, st.session_state.fix_capital
#             )
        
#         if actions_pattern.size == 0:
#             st.warning("ข้อมูลไม่เพียงพอสำหรับทำการวิเคราะห์แบบ Walk-Forward"); return
            
#         st.success("วิเคราะห์เสร็จสมบูรณ์!")

#         chart_data, results = pd.DataFrame(), {}
#         with st.spinner("กำลังจำลองกลยุทธ์เพื่อเปรียบเทียบ..."):
#             sim_len = len(actions_pattern)
#             strategy_map = {
#                 Strategy.PATTERN_WALK_FORWARD: actions_pattern,
#                 Strategy.PERFECT_FORESIGHT: _generate_perfect_foresight_numba(prices_np[:sim_len], st.session_state.fix_capital),
#                 Strategy.REBALANCE_DAILY: np.ones(sim_len, dtype=np.int32),
#             }
#             for name, actions in strategy_map.items():
#                 cumulative_net = _calculate_cumulative_net_numba(actions, prices_np[:sim_len], st.session_state.fix_capital)
#                 chart_data[name] = cumulative_net
#                 results[name] = cumulative_net[-1] if len(cumulative_net) > 0 else 0

#         chart_data.index = ticker_data.index[:len(chart_data)]
#         st.line_chart(chart_data)

#         if not df_windows.empty:
#             final_net_pattern = df_windows['walk_forward_net'].sum()
#             final_net_max = results.get(Strategy.PERFECT_FORESIGHT, 0)
#             final_net_min = results.get(Strategy.REBALANCE_DAILY, 0)
            
#             col1, col2, col3 = st.columns(3)
#             col1.metric(f"🥇 {Strategy.PERFECT_FORESIGHT}", f"${final_net_max:,.2f}")
#             col2.metric(f"🧠 {Strategy.PATTERN_WALK_FORWARD}", f"${final_net_pattern:,.2f}", delta=f"{final_net_pattern - final_net_min:,.2f} vs Min")
#             col3.metric(f"🥉 {Strategy.REBALANCE_DAILY}", f"${final_net_min:,.2f}")
        
#         st.dataframe(df_windows)

# def main():
#     st.set_page_config(page_title="Grand Unified Model Lab", page_icon="🌌", layout="wide")
#     st.markdown("### 🌌 The Grand Unified Model Laboratory")
#     st.caption("โมเดลค้นหาพารามิเตอร์ที่ดีที่สุดสำหรับสมการและแบบจำลอง 32 รูปแบบจากหลากหลายสาขาวิชา")

#     initialize_session_state()

#     tab_settings, tab_model = st.tabs(["⚙️ การตั้งค่า", "🚀 วิเคราะห์และแสดงผล"])
#     with tab_settings: render_settings_tab()
#     with tab_model: render_model_tab()
    
#     with st.expander("📖 คำอธิบายกลุ่มรูปแบบ/สมการ"):
#         st.markdown("""
#         #### 🌀 กลุ่ม Chaos Theory (18 สมการ)
#         - **ลักษณะ:** สร้างรูปแบบที่ซับซ้อน คาดเดาไม่ได้ในระยะยาว แต่มีโครงสร้างที่สวยงามในระยะสั้น
        
#         #### 🔬 กลุ่มวิทยาศาสตร์ธรรมชาติ (8 สมการ)
#         - **ลักษณะ:** จำลองกระบวนการทางกายภาพและชีวภาพ
#         - **ตัวอย่าง:** `SIR Model` (คลื่นการระบาด), `Schrödinger 1D` (คลื่นควอนตัม), `Lotka-Volterra` (ผู้ล่า-เหยื่อ)
        
#         #### 🤖 กลุ่ม AI และทฤษฎีเกม (6 สมการ)
#         - **ลักษณะ:** ได้แรงบันดาลใจจากการทำงานของ нейрон, การเรียนรู้ และการตัดสินใจเชิงกลยุทธ์
#         - **ตัวอย่าง:** `Perceptron Rule`, `Win-Stay Lose-Shift`, `Simple Genetic Algorithm`
#         """)

# if __name__ == "__main__":
#     main()
