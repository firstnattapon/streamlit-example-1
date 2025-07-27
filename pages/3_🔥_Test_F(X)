# # main
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import streamlit as st
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from datetime import datetime
# from typing import List, Tuple, Dict, Any
# import json
 
# # ! NUMBA: Import Numba's Just-In-Time compiler for core acceleration
# from numba import njit
 
# # ==============================================================================
# # 1. Configuration & Constants
# # ==============================================================================
# st.set_page_config(page_title="Hybrid_Multi_Mutation", page_icon="🧬", layout="wide")

# # JSON Config data provided by the user
# CONFIG_JSON_STRING = """
# {
#   "assets": ["FFWM", "NEGG", "RIVN", "APLS", "NVTS", "QXO", "RXRX", "AGL" ,"FLNC" , "GERN" , "DYN"],
#   "default_settings": {
#     "selected_ticker": "FFWM", "start_date": "2024-01-01",
#     "window_size": 30, "num_seeds": 1000, "max_workers": 8,
#     "mutation_rate": 10.0, "num_mutations": 5
#   },
#   "manual_seed_by_asset": {}
# }
# """

# class Strategy:
#     """คลาสสำหรับเก็บชื่อกลยุทธ์ต่างๆ เพื่อให้เรียกใช้ง่ายและลดข้อผิดพลาด"""
#     REBALANCE_DAILY = "Rebalance Daily"
#     PERFECT_FORESIGHT = "Perfect Foresight (Max)"
#     HYBRID_MULTI_MUTATION = "Hybrid (Multi-Mutation)"
#     ORIGINAL_DNA = "Original DNA (Pre-Mutation)"

# @st.cache_data
# def load_config() -> Dict[str, Any]:
#     """Loads configuration from the JSON string."""
#     return json.loads(CONFIG_JSON_STRING)

# def initialize_session_state(config: Dict[str, Any]):
#     """Initializes session state variables from the config."""
#     defaults = config.get('default_settings', {})
#     if 'test_ticker' not in st.session_state: st.session_state.test_ticker = defaults.get('selected_ticker', 'FFWM')
#     if 'start_date' not in st.session_state:
#         try: st.session_state.start_date = datetime.strptime(defaults.get('start_date', '2024-01-01'), '%Y-%m-%d').date()
#         except ValueError: st.session_state.start_date = datetime(2024, 1, 1).date()
#     if 'end_date' not in st.session_state: st.session_state.end_date = datetime.now().date()
#     if 'window_size' not in st.session_state: st.session_state.window_size = defaults.get('window_size', 30)
#     if 'num_seeds' not in st.session_state: st.session_state.num_seeds = defaults.get('num_seeds', 1000)
#     if 'max_workers' not in st.session_state: st.session_state.max_workers = defaults.get('max_workers', 8)
#     if 'mutation_rate' not in st.session_state: st.session_state.mutation_rate = defaults.get('mutation_rate', 10.0)
#     if 'num_mutations' not in st.session_state: st.session_state.num_mutations = defaults.get('num_mutations', 5)

# # ==============================================================================
# # 2. Core Calculation & Data Functions (Unchanged)
# # ==============================================================================
# @st.cache_data(ttl=3600)
# def get_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
#     try:
#         data = yf.Ticker(ticker).history(start=start_date, end=end_date)[['Close']]
#         if data.empty: return pd.DataFrame()
#         if data.index.tz is None: data = data.tz_localize('UTC').tz_convert('Asia/Bangkok')
#         else: data = data.tz_convert('Asia/Bangkok')
#         return data
#     except Exception as e:
#         st.error(f"❌ ไม่สามารถดึงข้อมูล {ticker} ได้: {str(e)}"); return pd.DataFrame()

# @njit(cache=True)
# def _calculate_net_profit_numba(action_array: np.ndarray, price_array: np.ndarray, fix: int = 1500) -> float:
#     n = len(action_array)
#     if n == 0 or len(price_array) == 0 or n > len(price_array): return -np.inf
#     action_array_calc = action_array.copy(); action_array_calc[0] = 1
#     initial_price = price_array[0]; initial_capital = fix * 2.0
#     refer_net = -fix * np.log(initial_price / price_array[n-1])
#     cash = float(fix); amount = float(fix) / initial_price
#     for i in range(1, n):
#         curr_price = price_array[i]
#         if action_array_calc[i] != 0: cash += amount * curr_price - fix; amount = fix / curr_price
#     final_sumusd = cash + (amount * price_array[n-1])
#     net = final_sumusd - refer_net - initial_capital
#     return net

# def run_simulation(prices: List[float], actions: List[int], fix: int = 1500) -> pd.DataFrame:
#     @njit
#     def _full_sim_numba(action_arr, price_arr, fix_val):
#         n = len(action_arr); empty = np.empty(0, dtype=np.float64)
#         if n == 0 or len(price_arr) == 0: return empty, empty, empty, empty, empty, empty
#         action_calc = action_arr.copy(); action_calc[0] = 1
#         amount = np.empty(n, dtype=np.float64); buffer = np.zeros(n, dtype=np.float64)
#         cash = np.empty(n, dtype=np.float64); asset_val = np.empty(n, dtype=np.float64)
#         sumusd_val = np.empty(n, dtype=np.float64)
#         init_price = price_arr[0]; amount[0] = fix_val / init_price; cash[0] = fix_val
#         asset_val[0] = amount[0] * init_price; sumusd_val[0] = cash[0] + asset_val[0]
#         refer = -fix_val * np.log(init_price / price_arr[:n])
#         for i in range(1, n):
#             curr_price = price_arr[i]
#             if action_calc[i] == 0: amount[i] = amount[i-1]; buffer[i] = 0.0
#             else: amount[i] = fix_val / curr_price; buffer[i] = amount[i-1] * curr_price - fix_val
#             cash[i] = cash[i-1] + buffer[i]; asset_val[i] = amount[i] * curr_price; sumusd_val[i] = cash[i] + asset_val[i]
#         return buffer, sumusd_val, cash, asset_val, amount, refer
#     if not prices or not actions: return pd.DataFrame()
#     min_len = min(len(prices), len(actions))
#     prices_arr = np.array(prices[:min_len], dtype=np.float64); actions_arr = np.array(actions[:min_len], dtype=np.int32)
#     buffer, sumusd, cash, asset_value, amount, refer = _full_sim_numba(actions_arr, prices_arr, fix)
#     if len(sumusd) == 0: return pd.DataFrame()
#     initial_capital = sumusd[0]
#     return pd.DataFrame({
#         'price': prices_arr, 'action': actions_arr, 'buffer': np.round(buffer, 2),
#         'sumusd': np.round(sumusd, 2), 'cash': np.round(cash, 2), 'asset_value': np.round(asset_value, 2),
#         'amount': np.round(amount, 2), 'refer': np.round(refer + initial_capital, 2),
#         'net': np.round(sumusd - refer - initial_capital, 2)
#     })

# # ==============================================================================
# # 3. Strategy Action Generation (Mostly Unchanged)
# # ==============================================================================
# def generate_actions_rebalance_daily(num_days: int) -> np.ndarray: return np.ones(num_days, dtype=np.int32)
# def generate_actions_perfect_foresight(prices: List[float], fix: int = 1500) -> np.ndarray:
#     price_arr = np.asarray(prices, dtype=np.float64); n = len(price_arr)
#     if n < 2: return np.ones(n, dtype=int)
#     dp = np.zeros(n, dtype=np.float64); path = np.zeros(n, dtype=int); dp[0] = float(fix * 2)
#     for i in range(1, n):
#         j_indices = np.arange(i); profits = fix * ((price_arr[i] / price_arr[j_indices]) - 1)
#         current_sumusd = dp[j_indices] + profits
#         best_idx = np.argmax(current_sumusd); dp[i] = current_sumusd[best_idx]; path[i] = j_indices[best_idx]
#     actions = np.zeros(n, dtype=int); current_day = np.argmax(dp)
#     while current_day > 0: actions[current_day] = 1; current_day = path[current_day]
#     actions[0] = 1
#     return actions

# def find_best_seed_for_window(prices_window: np.ndarray, num_seeds_to_try: int, max_workers: int) -> Tuple[int, float, np.ndarray]:
#     window_len = len(prices_window)
#     if window_len < 2: return 1, 0.0, np.ones(window_len, dtype=int)
#     def evaluate_seed_batch(seed_batch: np.ndarray) -> List[Tuple[int, float]]:
#         results = []
#         for seed in seed_batch:
#             rng = np.random.default_rng(seed)
#             actions = rng.integers(0, 2, size=window_len)
#             net = _calculate_net_profit_numba(actions, prices_window)
#             results.append((seed, net))
#         return results
#     best_seed, max_net = -1, -np.inf
#     random_seeds = np.arange(num_seeds_to_try)
#     batch_size = max(1, num_seeds_to_try // (max_workers * 4 if max_workers > 0 else 1))
#     seed_batches = [random_seeds[j:j+batch_size] for j in range(0, len(random_seeds), batch_size)]
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = [executor.submit(evaluate_seed_batch, batch) for batch in seed_batches]
#         for future in as_completed(futures):
#             for seed, final_net in future.result():
#                 if final_net > max_net: max_net, best_seed = final_net, seed
#     if best_seed >= 0:
#         rng_best = np.random.default_rng(best_seed)
#         best_actions = rng_best.integers(0, 2, size=window_len)
#     else: best_seed, best_actions, max_net = 1, np.ones(window_len, dtype=int), 0.0
#     best_actions[0] = 1
#     return best_seed, max_net, best_actions

# def find_best_mutation_for_sequence(original_actions: np.ndarray, prices_window: np.ndarray, num_mutation_seeds: int, mutation_rate: float, max_workers: int) -> Tuple[int, float, np.ndarray]:
#     window_len = len(original_actions)
#     if window_len < 2: return 1, -np.inf, original_actions
#     def evaluate_mutation_seed_batch(seed_batch: np.ndarray) -> List[Tuple[int, float]]:
#         results = []
#         for seed in seed_batch:
#             mutation_rng = np.random.default_rng(seed)
#             mutated_actions = original_actions.copy()
#             mutation_mask = mutation_rng.random(window_len) < mutation_rate
#             mutated_actions[mutation_mask] = 1 - mutated_actions[mutation_mask]
#             mutated_actions[0] = 1
#             net = _calculate_net_profit_numba(mutated_actions, prices_window)
#             results.append((seed, net))
#         return results
#     best_mutation_seed, max_mutated_net = -1, -np.inf
#     mutation_seeds_to_try = np.arange(num_mutation_seeds)
#     batch_size = max(1, num_mutation_seeds // (max_workers * 4 if max_workers > 0 else 1))
#     seed_batches = [mutation_seeds_to_try[j:j+batch_size] for j in range(0, len(mutation_seeds_to_try), batch_size)]
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = [executor.submit(evaluate_mutation_seed_batch, batch) for batch in seed_batches]
#         for future in as_completed(futures):
#             for seed, net in future.result():
#                 if net > max_mutated_net: max_mutated_net, best_mutation_seed = net, seed
#     if best_mutation_seed >= 0:
#         mutation_rng = np.random.default_rng(best_mutation_seed)
#         final_mutated_actions = original_actions.copy()
#         mutation_mask = mutation_rng.random(window_len) < mutation_rate
#         final_mutated_actions[mutation_mask] = 1 - final_mutated_actions[mutation_mask]
#         final_mutated_actions[0] = 1
#     else:
#         best_mutation_seed, max_mutated_net, final_mutated_actions = -1, -np.inf, original_actions.copy()
#     return best_mutation_seed, max_mutated_net, final_mutated_actions

# def generate_actions_hybrid_multi_mutation(ticker_data: pd.DataFrame, window_size: int, num_seeds: int, max_workers: int, mutation_rate_pct: float, num_mutations: int, progress_placeholder) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
#     prices = ticker_data['Close'].to_numpy()
#     n = len(prices)
#     final_actions = np.array([], dtype=int)
#     original_actions_full = np.array([], dtype=int)
#     window_details_list = []
#     num_windows = (n + window_size - 1) // window_size
#     mutation_rate = mutation_rate_pct / 100.0
#     for i, start_index in enumerate(range(0, n, window_size)):
#         end_index = min(start_index + window_size, n)
#         prices_window = prices[start_index:end_index]
#         if len(prices_window) < 2: continue
        
#         progress_placeholder.text(f"Window {i+1}/{num_windows}: ค้นหา DNA ที่ดีที่สุด...")
#         dna_seed, current_best_net, current_best_actions = find_best_seed_for_window(prices_window, num_seeds, max_workers)
        
#         original_actions_window = current_best_actions.copy()
#         original_net_for_display = current_best_net
#         successful_mutation_seeds = []
#         for mutation_round in range(num_mutations):
#             progress_placeholder.text(f"Window {i+1}/{num_windows}: รอบการกลายพันธุ์ {mutation_round+1}/{num_mutations}...")
#             mutation_seed, mutated_net, mutated_actions = find_best_mutation_for_sequence(current_best_actions, prices_window, num_seeds, mutation_rate, max_workers)
#             if mutated_net > current_best_net:
#                 current_best_net, current_best_actions = mutated_net, mutated_actions
#                 successful_mutation_seeds.append(int(mutation_seed))
        
#         final_actions = np.concatenate((final_actions, current_best_actions))
#         original_actions_full = np.concatenate((original_actions_full, original_actions_window))
#         start_date = ticker_data.index[start_index]; end_date = ticker_data.index[end_index-1]
#         detail = {
#             'window': i + 1, 'timeline': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
#             'dna_seed': dna_seed, 'mutation_seeds': str(successful_mutation_seeds) if successful_mutation_seeds else "None",
#             'improvements': len(successful_mutation_seeds),
#             'original_net': round(original_net_for_display, 2), 'final_net': round(current_best_net, 2)
#         }
#         window_details_list.append(detail)
#     return original_actions_full, final_actions, pd.DataFrame(window_details_list)


# # ==============================================================================
# # 4. Simulation Tracer Class (Unchanged)
# # ==============================================================================
# class SimulationTracer:
#     def __init__(self, encoded_string: str):
#         self.encoded_string: str = encoded_string
#         self._decode_and_set_attributes()
#     def _decode_and_set_attributes(self):
#         encoded_string = self.encoded_string
#         if not isinstance(encoded_string, str) or not encoded_string.isdigit():
#             raise ValueError("Input ต้องเป็นสตริงที่ประกอบด้วยตัวเลขเท่านั้น")
#         decoded_numbers, idx = [], 0
#         while idx < len(encoded_string):
#             try:
#                 length_of_number = int(encoded_string[idx]); idx += 1
#                 number_str = encoded_string[idx : idx + length_of_number]; idx += length_of_number
#                 decoded_numbers.append(int(number_str))
#             except (IndexError, ValueError): raise ValueError(f"รูปแบบของสตริงไม่ถูกต้องที่ตำแหน่ง {idx}")
#         if len(decoded_numbers) < 3: raise ValueError("ข้อมูลในสตริงไม่ครบถ้วน (ต้องการอย่างน้อย 3 ค่า)")
#         self.action_length, self.mutation_rate, self.dna_seed = decoded_numbers[0], decoded_numbers[1], decoded_numbers[2]
#         self.mutation_seeds: List[int] = decoded_numbers[3:]
#         self.mutation_rate_float: float = self.mutation_rate / 100.0
#     def run(self) -> np.ndarray:
#         dna_rng = np.random.default_rng(seed=self.dna_seed)
#         current_actions = dna_rng.integers(0, 2, size=self.action_length)
#         current_actions[0] = 1
#         for m_seed in self.mutation_seeds:
#             mutation_rng = np.random.default_rng(seed=m_seed)
#             mutation_mask = mutation_rng.random(self.action_length) < self.mutation_rate_float
#             current_actions[mutation_mask] = 1 - current_actions[mutation_mask]
#             current_actions[0] = 1
#         return current_actions
#     def __str__(self) -> str:
#         return (f"✅ พารามิเตอร์ที่ถอดรหัสสำเร็จ:\n"
#                 f"- action_length: {self.action_length}\n"
#                 f"- mutation_rate: {self.mutation_rate} ({self.mutation_rate_float:.2f})\n"
#                 f"- dna_seed: {self.dna_seed}\n"
#                 f"- mutation_seeds: {self.mutation_seeds}")
#     @staticmethod
#     def encode(action_length: int, mutation_rate: int, dna_seed: int, mutation_seeds: List[int]) -> str:
#         all_numbers = [action_length, mutation_rate, dna_seed] + mutation_seeds
#         encoded_parts = [f"{len(str(num))}{num}" for num in all_numbers]
#         return "".join(encoded_parts)

# # ==============================================================================
# # 5. NEW & REFACTORED: Batch Processing and UI Rendering
# # ==============================================================================
# def run_single_asset_simulation(ticker: str, params: Dict[str, Any]) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
#     """Runs the full simulation pipeline for a single asset."""
#     ticker_data = get_ticker_data(ticker, str(params['start_date']), str(params['end_date']))
#     if ticker_data.empty:
#         st.warning(f"ไม่พบข้อมูลสำหรับ Ticker {ticker} ในช่วงวันที่ที่เลือก")
#         return None, None, None

#     # A placeholder for progress text updates within the simulation function
#     progress_placeholder = st.empty()
    
#     original_actions, final_actions, df_windows = generate_actions_hybrid_multi_mutation(
#         ticker_data, params['window_size'], params['num_seeds'], params['max_workers'],
#         params['mutation_rate'], params['num_mutations'], progress_placeholder
#     )
#     progress_placeholder.empty() # Clear the progress text after completion

#     prices = ticker_data['Close'].to_numpy()
#     results = {
#         Strategy.HYBRID_MULTI_MUTATION: run_simulation(prices.tolist(), final_actions.tolist()),
#         Strategy.ORIGINAL_DNA: run_simulation(prices.tolist(), original_actions.tolist()),
#         Strategy.REBALANCE_DAILY: run_simulation(prices.tolist(), generate_actions_rebalance_daily(len(prices)).tolist()),
#         Strategy.PERFECT_FORESIGHT: run_simulation(prices.tolist(), generate_actions_perfect_foresight(prices.tolist()).tolist())
#     }
#     for name, df in results.items():
#         if not df.empty:
#             df.index = ticker_data.index[:len(df)]
    
#     return results, df_windows, ticker_data

# def display_single_asset_results(ticker: str, results: Dict, df_windows: pd.DataFrame, ticker_data: pd.DataFrame):
#     """Renders the UI components for a single asset's results."""
#     st.markdown(f"### 📊 ผลลัพธ์สำหรับ {ticker}")

#     # Chart
#     chart_results = {k: v for k, v in results.items() if k != Strategy.ORIGINAL_DNA}
#     display_comparison_charts(chart_results)

#     # Metrics
#     st.write("#### สรุปผลการดำเนินงาน (Compounded Final Profit)")
#     metrics = {}
#     for name, df in results.items():
#         metrics[name] = df['net'].iloc[-1] if df is not None and not df.empty else 0.0

#     col1, col2, col3, col4 = st.columns(4)
#     col1.metric("Perfect Foresight", f"${metrics.get(Strategy.PERFECT_FORESIGHT, 0):,.2f}")
#     col2.metric("Hybrid Strategy", f"${metrics.get(Strategy.HYBRID_MULTI_MUTATION, 0):,.2f}")
#     col3.metric("Original DNA", f"${metrics.get(Strategy.ORIGINAL_DNA, 0):,.2f}")
#     col4.metric("Rebalance Daily", f"${metrics.get(Strategy.REBALANCE_DAILY, 0):,.2f}")

#     # Details Table
#     st.write("#### 📝 รายละเอียดผลลัพธ์ราย Window")
#     st.dataframe(df_windows, use_container_width=True)
#     st.download_button("📥 Download Details (CSV)", df_windows.to_csv(index=False), f'hybrid_details_{ticker}.csv', 'text/csv')

# def display_comparison_charts(results: Dict[str, pd.DataFrame], chart_title: str = 'เปรียบเทียบกำไรสุทธิ (Net Profit)'):
#     if not results: return
#     valid_dfs = {name: df for name, df in results.items() if not df.empty and 'net' in df.columns}
#     if not valid_dfs: return
#     try: longest_index = max((df.index for df in valid_dfs.values()), key=len, default=None)
#     except ValueError: longest_index = None
#     if longest_index is None: return
#     chart_data = pd.DataFrame(index=longest_index)
#     for name, df in valid_dfs.items(): chart_data[name] = df['net'].reindex(longest_index).ffill()
#     st.line_chart(chart_data)

# def render_settings_tab():
#     st.write("⚙️ **การตั้งค่าพารามิเตอร์**")
#     config = load_config()
#     asset_list = config.get('assets', ['FFWM'])

#     c1, c2 = st.columns(2)
#     st.session_state.test_ticker = c1.selectbox("เลือก Ticker สำหรับทดสอบ", options=asset_list, index=asset_list.index(st.session_state.test_ticker) if st.session_state.test_ticker in asset_list else 0)
#     st.session_state.window_size = c2.number_input("ขนาด Window (วัน)", min_value=2, value=st.session_state.window_size)

#     c1, c2 = st.columns(2)
#     st.session_state.start_date = c1.date_input("วันที่เริ่มต้น", value=st.session_state.start_date)
#     st.session_state.end_date = c2.date_input("วันที่สิ้นสุด", value=st.session_state.end_date)
#     if st.session_state.start_date >= st.session_state.end_date: st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")

#     st.divider()
#     st.subheader("พารามิเตอร์สำหรับกลยุทธ์")
#     c1, c2 = st.columns(2)
#     st.session_state.num_seeds = c1.number_input("จำนวน Seeds (สำหรับค้นหา DNA และ Mutation)", min_value=100, value=st.session_state.num_seeds, format="%d")
#     st.session_state.max_workers = c2.number_input("จำนวน Workers (CPU Cores)", min_value=1, max_value=16, value=st.session_state.max_workers)

#     c1, c2 = st.columns(2)
#     st.session_state.mutation_rate = c1.slider("อัตราการกลายพันธุ์ (Mutation Rate) %", min_value=0.0, max_value=50.0, value=st.session_state.mutation_rate, step=0.5)
#     st.session_state.num_mutations = c2.number_input("จำนวนรอบการกลายพันธุ์ (Multi-Mutation)", min_value=0, max_value=10, value=st.session_state.num_mutations)

# def render_hybrid_multi_mutation_tab():
#     st.markdown(f"### 🧬 {Strategy.HYBRID_MULTI_MUTATION}")

#     # New: Add a checkbox to run for all assets
#     run_all = st.checkbox("Run for All Assets in Config", value=False)
    
#     if st.button(f"🚀 Start Hybrid Multi-Mutation", type="primary"):
#         if st.session_state.start_date >= st.session_state.end_date:
#             st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้อง")
#             return

#         params = {
#             'start_date': st.session_state.start_date, 'end_date': st.session_state.end_date,
#             'window_size': st.session_state.window_size, 'num_seeds': st.session_state.num_seeds,
#             'max_workers': st.session_state.max_workers, 'mutation_rate': st.session_state.mutation_rate,
#             'num_mutations': st.session_state.num_mutations
#         }
        
#         if run_all:
#             # --- BATCH PROCESSING LOGIC ---
#             config = load_config()
#             all_assets = config.get('assets', [])
#             batch_results = {}
            
#             progress_bar = st.progress(0, "Starting batch processing...")
            
#             for i, ticker in enumerate(all_assets):
#                 progress_text = f"Processing asset {i+1}/{len(all_assets)}: {ticker}"
#                 progress_bar.progress((i) / len(all_assets), text=progress_text)
                
#                 try:
#                     with st.spinner(f"Running simulation for {ticker}..."):
#                         results, df_windows, ticker_data = run_single_asset_simulation(ticker, params)
#                     if results:
#                         batch_results[ticker] = {
#                             "results": results,
#                             "details_df": df_windows,
#                             "ticker_data": ticker_data
#                         }
#                 except Exception as e:
#                     st.error(f"An error occurred while processing {ticker}: {e}")

#             st.session_state.batch_results = batch_results
#             if 'single_asset_results' in st.session_state: # Clear old single results
#                 del st.session_state.single_asset_results
#             progress_bar.progress(1.0, "Batch processing complete!")

#         else:
#             # --- SINGLE ASSET LOGIC (Original behavior) ---
#             ticker = st.session_state.test_ticker
#             with st.spinner(f"กำลังดึงข้อมูลและประมวลผลสำหรับ {ticker}..."):
#                 results, df_windows, ticker_data = run_single_asset_simulation(ticker, params)
#                 if results:
#                     st.session_state.single_asset_results = {
#                         "ticker": ticker,
#                         "results": results,
#                         "details_df": df_windows,
#                         "ticker_data": ticker_data
#                     }
#                     if 'batch_results' in st.session_state: # Clear old batch results
#                         del st.session_state.batch_results

#     # --- DISPLAY RESULTS SECTION (Unified for both single and batch) ---
#     st.divider()

#     if 'batch_results' in st.session_state and st.session_state.batch_results:
#         st.success("การทดสอบแบบ Batch เสร็จสมบูรณ์!")
#         batch_data = st.session_state.batch_results
        
#         # Display Summary Table
#         summary_data = []
#         for ticker, data in batch_data.items():
#             final_nets = {name: df['net'].iloc[-1] for name, df in data['results'].items() if not df.empty}
#             summary_data.append({
#                 "Ticker": ticker,
#                 "Hybrid Net": final_nets.get(Strategy.HYBRID_MULTI_MUTATION, 0),
#                 "Perfect Net": final_nets.get(Strategy.PERFECT_FORESIGHT, 0),
#                 "Original DNA Net": final_nets.get(Strategy.ORIGINAL_DNA, 0),
#                 "Rebalance Net": final_nets.get(Strategy.REBALANCE_DAILY, 0),
#             })
#         summary_df = pd.DataFrame(summary_data)
#         st.write("### 📈 สรุปผลลัพธ์รวมทุก Asset")
#         st.dataframe(summary_df.style.format({col: "${:,.2f}" for col in summary_df.columns if col != 'Ticker'}), use_container_width=True)

#         # Display Detailed Results in Expanders
#         for ticker, data in batch_data.items():
#             with st.expander(f"ดูรายละเอียดสำหรับ {ticker}"):
#                 display_single_asset_results(ticker, data['results'], data['details_df'], data['ticker_data'])
    
#     elif 'single_asset_results' in st.session_state and st.session_state.single_asset_results:
#         st.success("การทดสอบเสร็จสมบูรณ์!")
#         data = st.session_state.single_asset_results
#         display_single_asset_results(data['ticker'], data['results'], data['details_df'], data['ticker_data'])

# def render_tracer_tab():
#     st.markdown("### 🔍 Action Sequence Tracer & Encoder")
#     st.info("เครื่องมือนี้ใช้สำหรับ 1. **ถอดรหัส (Decode)** String เพื่อจำลองผลลัพธ์ และ 2. **เข้ารหัส (Encode)** พารามิเตอร์เพื่อสร้าง String")

#     st.markdown("---")
#     st.markdown("#### 1. ถอดรหัส (Decode) String")
#     encoded_string = st.text_input("ป้อน Encoded String ที่นี่:", "26021034252903219354832053493", key="decoder_input")
#     if st.button("Trace & Simulate", type="primary", key="tracer_button"):
#         if not encoded_string: st.warning("กรุณาป้อน Encoded String")
#         else:
#             with st.spinner("..."):
#                 try:
#                     tracer = SimulationTracer(encoded_string=encoded_string)
#                     st.success("ถอดรหัสสำเร็จ!"); st.code(str(tracer), language='bash')
#                     final_actions = tracer.run()
#                     st.markdown("#### 🎉 ผลลัพธ์ Action Sequence สุดท้าย:")
#                     st.dataframe(pd.DataFrame(final_actions, columns=['Action']), use_container_width=True)
#                     st.code(str(final_actions))
#                 except ValueError as e: st.error(f"❌ เกิดข้อผิดพลาดในการประมวลผล: {e}")

#     st.divider()
#     st.markdown("#### 2. เข้ารหัส (Encode) พารามิเตอร์")
#     col1, col2 = st.columns(2)
#     with col1:
#         action_length_input = st.number_input("Action Length", min_value=1, value=60, key="enc_len")
#         dna_seed_input = st.number_input("DNA Seed", min_value=0, value=900, format="%d", key="enc_dna")
#     with col2:
#         mutation_rate_input = st.number_input("Mutation Rate (%)", min_value=0, value=10, key="enc_rate")
#         mutation_seeds_str = st.text_input("Mutation Seeds (คั่นด้วยจุลภาค ,)", "899, 530, 35, 814, 646", key="enc_seeds_str")
#     if st.button("Encode Parameters", key="encoder_button"):
#         try:
#             mutation_seeds_list = [int(s.strip()) for s in mutation_seeds_str.split(',')] if mutation_seeds_str.strip() else []
#             generated_string = SimulationTracer.encode(int(action_length_input), int(mutation_rate_input), int(dna_seed_input), mutation_seeds_list)
#             st.success("เข้ารหัสสำเร็จ!"); st.code(generated_string, language='text')
#         except (ValueError, TypeError) as e: st.error(f"❌ เกิดข้อผิดพลาด: {e}")

# # ==============================================================================
# # 6. Main Application
# # ==============================================================================
# def main():
#     st.markdown("### 🧬 Hybrid Strategy Lab (Multi-Mutation)")
#     st.caption("เครื่องมือทดลองและพัฒนากลยุทธ์ด้วย Numba-Accelerated Parallel Random Search")

#     config = load_config()
#     initialize_session_state(config)

#     tab_list = ["⚙️ การตั้งค่า", f"🧬 {Strategy.HYBRID_MULTI_MUTATION}", "🔍 Tracer"]
#     tabs = st.tabs(tab_list)

#     with tabs[0]:
#         render_settings_tab()
#     with tabs[1]:
#         render_hybrid_multi_mutation_tab()
#     with tabs[2]:
#         render_tracer_tab()

# if __name__ == "__main__":
#     main()
