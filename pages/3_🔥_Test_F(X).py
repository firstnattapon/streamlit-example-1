# main.py
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Tuple, Dict, Any

# ! NUMBA: Import Numba's Just-In-Time compiler for core acceleration
from numba import njit

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================
st.set_page_config(page_title="Multi-Ticker Hybrid Lab", page_icon="🧪", layout="wide")

class Strategy:
    """คลาสสำหรับเก็บชื่อกลยุทธ์ต่างๆ เพื่อให้เรียกใช้ง่ายและลดข้อผิดพลาด"""
    REBALANCE_DAILY = "Rebalance Daily"
    PERFECT_FORESIGHT = "Perfect Foresight (Max)"
    HYBRID_MULTI_MUTATION = "Hybrid (Multi-Mutation)"
    ORIGINAL_DNA = "Original DNA (Pre-Mutation)"

def load_tickers_config(filepath: str = "tickers_to_test.json") -> List[str]:
    """Loads a list of tickers to test from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config.get("assets", [])
    except FileNotFoundError:
        st.error(f"❌ ไม่พบไฟล์ตั้งค่า '{filepath}' กรุณาสร้างไฟล์นี้")
        return []
    except json.JSONDecodeError:
        st.error(f"❌ รูปแบบ JSON ในไฟล์ '{filepath}' ไม่ถูกต้อง")
        return []

def initialize_session_state():
    """Initializes session state with default parameters."""
    defaults = {
        "start_date": "2024-01-01", "window_size": 30, "num_seeds": 1000,
        "max_workers": 8, "mutation_rate": 10.0, "num_mutations": 5
    }
    if 'start_date' not in st.session_state:
        try:
            st.session_state.start_date = datetime.strptime(defaults['start_date'], '%Y-%m-%d').date()
        except ValueError:
            st.session_state.start_date = datetime(2024, 1, 1).date()
    if 'end_date' not in st.session_state: st.session_state.end_date = datetime.now().date()
    if 'window_size' not in st.session_state: st.session_state.window_size = defaults['window_size']
    if 'num_seeds' not in st.session_state: st.session_state.num_seeds = defaults['num_seeds']
    if 'max_workers' not in st.session_state: st.session_state.max_workers = defaults['max_workers']
    if 'mutation_rate' not in st.session_state: st.session_state.mutation_rate = defaults['mutation_rate']
    if 'num_mutations' not in st.session_state: st.session_state.num_mutations = defaults['num_mutations']
    if 'all_ticker_results' not in st.session_state: st.session_state.all_ticker_results = {}

# ==============================================================================
# 2. Core Calculation & Data Functions (Unchanged from App 1)
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
def _calculate_net_profit_numba(action_array: np.ndarray, price_array: np.ndarray, fix: int = 1500) -> float:
    n = len(action_array)
    if n == 0 or len(price_array) == 0 or n > len(price_array): return -np.inf
    action_array_calc = action_array.copy(); action_array_calc[0] = 1
    initial_price = price_array[0]; initial_capital = fix * 2.0
    refer_net = -fix * np.log(initial_price / price_array[n-1])
    cash = float(fix); amount = float(fix) / initial_price
    for i in range(1, n):
        curr_price = price_array[i]
        if action_array_calc[i] != 0: cash += amount * curr_price - fix; amount = fix / curr_price
    final_sumusd = cash + (amount * price_array[n-1])
    net = final_sumusd - refer_net - initial_capital
    return net

def run_simulation(prices: List[float], actions: List[int], fix: int = 1500) -> pd.DataFrame:
    @njit
    def _full_sim_numba(action_arr, price_arr, fix_val):
        n = len(action_arr); empty = np.empty(0, dtype=np.float64)
        if n == 0 or len(price_arr) == 0: return empty, empty, empty, empty, empty, empty
        action_calc = action_arr.copy(); action_calc[0] = 1
        amount = np.empty(n, dtype=np.float64); buffer = np.zeros(n, dtype=np.float64)
        cash = np.empty(n, dtype=np.float64); asset_val = np.empty(n, dtype=np.float64)
        sumusd_val = np.empty(n, dtype=np.float64)
        init_price = price_arr[0]; amount[0] = fix_val / init_price; cash[0] = fix_val
        asset_val[0] = amount[0] * init_price; sumusd_val[0] = cash[0] + asset_val[0]
        refer = -fix_val * np.log(init_price / price_arr[:n])
        for i in range(1, n):
            curr_price = price_arr[i]
            if action_calc[i] == 0: amount[i] = amount[i-1]; buffer[i] = 0.0
            else: amount[i] = fix_val / curr_price; buffer[i] = amount[i-1] * curr_price - fix_val
            cash[i] = cash[i-1] + buffer[i]; asset_val[i] = amount[i] * curr_price; sumusd_val[i] = cash[i] + asset_val[i]
        return buffer, sumusd_val, cash, asset_val, amount, refer

    if not prices or not actions: return pd.DataFrame()
    min_len = min(len(prices), len(actions))
    prices_arr = np.array(prices[:min_len], dtype=np.float64); actions_arr = np.array(actions[:min_len], dtype=np.int32)
    buffer, sumusd, cash, asset_value, amount, refer = _full_sim_numba(actions_arr, prices_arr, fix)
    if len(sumusd) == 0: return pd.DataFrame()
    initial_capital = sumusd[0]
    return pd.DataFrame({
        'price': prices_arr, 'action': actions_arr, 'buffer': np.round(buffer, 2),
        'sumusd': np.round(sumusd, 2), 'cash': np.round(cash, 2), 'asset_value': np.round(asset_value, 2),
        'amount': np.round(amount, 2), 'refer': np.round(refer + initial_capital, 2),
        'net': np.round(sumusd - refer - initial_capital, 2)
    })

# ==============================================================================
# 3. Strategy Action Generation (Unchanged from App 1)
# ==============================================================================
def generate_actions_rebalance_daily(num_days: int) -> np.ndarray: return np.ones(num_days, dtype=np.int32)
def generate_actions_perfect_foresight(prices: List[float], fix: int = 1500) -> np.ndarray:
    price_arr = np.asarray(prices, dtype=np.float64); n = len(price_arr)
    if n < 2: return np.ones(n, dtype=int)
    dp = np.zeros(n, dtype=np.float64); path = np.zeros(n, dtype=int); dp[0] = float(fix * 2)
    for i in range(1, n):
        j_indices = np.arange(i); profits = fix * ((price_arr[i] / price_arr[j_indices]) - 1)
        current_sumusd = dp[j_indices] + profits
        best_idx = np.argmax(current_sumusd); dp[i] = current_sumusd[best_idx]; path[i] = j_indices[best_idx]
    actions = np.zeros(n, dtype=int); current_day = np.argmax(dp)
    while current_day > 0: actions[current_day] = 1; current_day = path[current_day]
    actions[0] = 1
    return actions

def find_best_seed_for_window(prices_window: np.ndarray, num_seeds_to_try: int, max_workers: int) -> Tuple[int, float, np.ndarray]:
    window_len = len(prices_window)
    if window_len < 2: return 1, 0.0, np.ones(window_len, dtype=int)
    def evaluate_seed_batch(seed_batch: np.ndarray) -> List[Tuple[int, float]]:
        results = []
        for seed in seed_batch:
            rng = np.random.default_rng(seed)
            actions = rng.integers(0, 2, size=window_len)
            net = _calculate_net_profit_numba(actions, prices_window)
            results.append((seed, net))
        return results
    best_seed, max_net = -1, -np.inf
    random_seeds = np.arange(num_seeds_to_try)
    batch_size = max(1, num_seeds_to_try // (max_workers * 4 if max_workers > 0 else 1))
    seed_batches = [random_seeds[j:j+batch_size] for j in range(0, len(random_seeds), batch_size)]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_seed_batch, batch) for batch in seed_batches]
        for future in as_completed(futures):
            for seed, final_net in future.result():
                if final_net > max_net: max_net, best_seed = final_net, seed
    if best_seed >= 0:
        rng_best = np.random.default_rng(best_seed)
        best_actions = rng_best.integers(0, 2, size=window_len)
    else: best_seed, best_actions, max_net = 1, np.ones(window_len, dtype=int), 0.0
    best_actions[0] = 1
    return best_seed, max_net, best_actions

def find_best_mutation_for_sequence(
    original_actions: np.ndarray, prices_window: np.ndarray,
    num_mutation_seeds: int, mutation_rate: float, max_workers: int
) -> Tuple[int, float, np.ndarray]:
    window_len = len(original_actions)
    if window_len < 2: return 1, -np.inf, original_actions
    def evaluate_mutation_seed_batch(seed_batch: np.ndarray) -> List[Tuple[int, float]]:
        results = []
        for seed in seed_batch:
            mutation_rng = np.random.default_rng(seed)
            mutated_actions = original_actions.copy()
            mutation_mask = mutation_rng.random(window_len) < mutation_rate
            mutated_actions[mutation_mask] = 1 - mutated_actions[mutation_mask]
            mutated_actions[0] = 1
            net = _calculate_net_profit_numba(mutated_actions, prices_window)
            results.append((seed, net))
        return results
    best_mutation_seed, max_mutated_net = -1, -np.inf
    mutation_seeds_to_try = np.arange(num_mutation_seeds)
    batch_size = max(1, num_mutation_seeds // (max_workers * 4 if max_workers > 0 else 1))
    seed_batches = [mutation_seeds_to_try[j:j+batch_size] for j in range(0, len(mutation_seeds_to_try), batch_size)]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_mutation_seed_batch, batch) for batch in seed_batches]
        for future in as_completed(futures):
            for seed, net in future.result():
                if net > max_mutated_net:
                    max_mutated_net, best_mutation_seed = net, seed
    if best_mutation_seed >= 0:
        mutation_rng = np.random.default_rng(best_mutation_seed)
        final_mutated_actions = original_actions.copy()
        mutation_mask = mutation_rng.random(window_len) < mutation_rate
        final_mutated_actions[mutation_mask] = 1 - final_mutated_actions[mutation_mask]
        final_mutated_actions[0] = 1
    else:
        best_mutation_seed, max_mutated_net, final_mutated_actions = -1, -np.inf, original_actions.copy()
    return best_mutation_seed, max_mutated_net, final_mutated_actions

def generate_actions_hybrid_multi_mutation(
    ticker_data: pd.DataFrame, window_size: int, num_seeds: int, max_workers: int,
    mutation_rate_pct: float, num_mutations: int, status_placeholder: st.container
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    prices, n = ticker_data['Close'].to_numpy(), len(ticker_data)
    final_actions, original_actions_full = np.array([], dtype=int), np.array([], dtype=int)
    window_details_list = []
    mutation_rate = mutation_rate_pct / 100.0

    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        if len(prices_window) < 2: continue
        
        status_placeholder.text(f"Window {i+1}: Searching for Best DNA...")
        dna_seed, current_best_net, current_best_actions = find_best_seed_for_window(prices_window, num_seeds, max_workers)
        
        original_actions_window, original_net_for_display = current_best_actions.copy(), current_best_net
        successful_mutation_seeds = []
        for mutation_round in range(num_mutations):
            status_placeholder.text(f"Window {i+1}: Mutation Round {mutation_round+1}/{num_mutations}...")
            mutation_seed, mutated_net, mutated_actions = find_best_mutation_for_sequence(
                current_best_actions, prices_window, num_seeds, mutation_rate, max_workers
            )
            if mutated_net > current_best_net:
                current_best_net, current_best_actions = mutated_net, mutated_actions
                successful_mutation_seeds.append(int(mutation_seed))
        
        final_actions = np.concatenate((final_actions, current_best_actions))
        original_actions_full = np.concatenate((original_actions_full, original_actions_window))
        start_date, end_date = ticker_data.index[start_index], ticker_data.index[end_index-1]
        window_details_list.append({
            'window': i + 1, 'timeline': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            'dna_seed': dna_seed, 'mutation_seeds': str(successful_mutation_seeds) if successful_mutation_seeds else "None",
            'improvements': len(successful_mutation_seeds), 'original_net': round(original_net_for_display, 2),
            'final_net': round(current_best_net, 2)
        })
    return original_actions_full, final_actions, pd.DataFrame(window_details_list)

# ==============================================================================
# 4. Simulation Tracer Class (Unchanged from App 1)
# ==============================================================================
class SimulationTracer:
    def __init__(self, encoded_string: str):
        self.encoded_string: str = encoded_string
        self._decode_and_set_attributes()
    def _decode_and_set_attributes(self):
        encoded_string = self.encoded_string
        if not isinstance(encoded_string, str) or not encoded_string.isdigit():
            raise ValueError("Input ต้องเป็นสตริงที่ประกอบด้วยตัวเลขเท่านั้น")
        decoded_numbers, idx = [], 0
        while idx < len(encoded_string):
            try:
                length_of_number = int(encoded_string[idx]); idx += 1
                number_str = encoded_string[idx : idx + length_of_number]; idx += length_of_number
                decoded_numbers.append(int(number_str))
            except (IndexError, ValueError): raise ValueError(f"รูปแบบของสตริงไม่ถูกต้องที่ตำแหน่ง {idx}")
        if len(decoded_numbers) < 3: raise ValueError("ข้อมูลในสตริงไม่ครบถ้วน (ต้องการอย่างน้อย 3 ค่า)")
        self.action_length: int = decoded_numbers[0]
        self.mutation_rate: int = decoded_numbers[1]
        self.dna_seed: int = decoded_numbers[2]
        self.mutation_seeds: List[int] = decoded_numbers[3:]
        self.mutation_rate_float: float = self.mutation_rate / 100.0
    def run(self) -> np.ndarray:
        dna_rng = np.random.default_rng(seed=self.dna_seed)
        current_actions = dna_rng.integers(0, 2, size=self.action_length)
        current_actions[0] = 1
        for m_seed in self.mutation_seeds:
            mutation_rng = np.random.default_rng(seed=m_seed)
            mutation_mask = mutation_rng.random(self.action_length) < self.mutation_rate_float
            current_actions[mutation_mask] = 1 - current_actions[mutation_mask]
            current_actions[0] = 1
        return current_actions
    def __str__(self) -> str:
        return (f"✅ พารามิเตอร์ที่ถอดรหัสสำเร็จ:\n- action_length: {self.action_length}\n"
                f"- mutation_rate: {self.mutation_rate} ({self.mutation_rate_float:.2f})\n"
                f"- dna_seed: {self.dna_seed}\n- mutation_seeds: {self.mutation_seeds}")
    @staticmethod
    def encode(action_length: int, mutation_rate: int, dna_seed: int, mutation_seeds: List[int]) -> str:
        all_numbers = [action_length, mutation_rate, dna_seed] + mutation_seeds
        return "".join([f"{len(str(num))}{num}" for num in all_numbers])

# ==============================================================================
# 5. UI Rendering Functions (Adapted for Multi-Ticker)
# ==============================================================================
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

def render_settings_and_control_tab():
    st.write("⚙️ **การตั้งค่าพารามิเตอร์ (Global)**")
    st.info("การตั้งค่าในหน้านี้จะถูกนำไปใช้กับการทดสอบทุก Ticker")
    
    c1, c2 = st.columns(2)
    st.session_state.start_date = c1.date_input("วันที่เริ่มต้น", value=st.session_state.start_date)
    st.session_state.end_date = c2.date_input("วันที่สิ้นสุด", value=st.session_state.end_date)
    if st.session_state.start_date >= st.session_state.end_date: st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
    
    st.session_state.window_size = st.number_input("ขนาด Window (วัน)", min_value=2, value=st.session_state.window_size)
    st.divider()
    st.subheader("พารามิเตอร์สำหรับกลยุทธ์")
    c1, c2 = st.columns(2)
    st.session_state.num_seeds = c1.number_input("จำนวน Seeds (สำหรับค้นหา DNA และ Mutation)", min_value=100, value=st.session_state.num_seeds, format="%d")
    st.session_state.max_workers = c2.number_input("จำนวน Workers (CPU Cores)", min_value=1, max_value=16, value=st.session_state.max_workers)

    c1, c2 = st.columns(2)
    st.session_state.mutation_rate = c1.slider("อัตราการกลายพันธุ์ (Mutation Rate) %", min_value=0.0, max_value=50.0, value=st.session_state.mutation_rate, step=0.5)
    st.session_state.num_mutations = c2.number_input("จำนวนรอบการกลายพันธุ์ (Multi-Mutation)", min_value=0, max_value=10, value=st.session_state.num_mutations)
    
    st.divider()
    if st.button("🚀 เริ่มการทดสอบทุก Tickers", type="primary", use_container_width=True):
        st.session_state.run_triggered = True
        st.session_state.all_ticker_results = {} # Clear previous results

def render_ticker_results_tab(ticker: str, results_data: Dict[str, Any]):
    st.markdown(f"### 📈 สรุปผลลัพธ์สำหรับ **{ticker}**")
    
    sim_results = results_data['sim_results']
    chart_results = {k: v for k, v in sim_results.items() if k != Strategy.ORIGINAL_DNA}
    display_comparison_charts(chart_results)
    
    st.divider()
    
    df_windows = results_data.get('df_windows_details', pd.DataFrame())
    
    total_perfect_net = sim_results.get(Strategy.PERFECT_FORESIGHT, pd.DataFrame({'net': [0]}))['net'].iloc[-1]
    total_hybrid_net = sim_results.get(Strategy.HYBRID_MULTI_MUTATION, pd.DataFrame({'net': [0]}))['net'].iloc[-1]
    total_original_net = sim_results.get(Strategy.ORIGINAL_DNA, pd.DataFrame({'net': [0]}))['net'].iloc[-1]
    total_rebalance_net = sim_results.get(Strategy.REBALANCE_DAILY, pd.DataFrame({'net': [0]}))['net'].iloc[-1]

    st.write("#### สรุปผลการดำเนินงานโดยรวม (Compounded Final Profit)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Perfect Foresight", f"${total_perfect_net:,.2f}")
    col2.metric("Hybrid Strategy", f"${total_hybrid_net:,.2f}")
    col3.metric("Original DNA", f"${total_original_net:,.2f}")
    col4.metric("Rebalance Daily", f"${total_rebalance_net:,.2f}")

    st.write("---")
    st.write("#### 📝 รายละเอียดผลลัพธ์ราย Window")
    st.dataframe(df_windows, use_container_width=True)
    st.download_button("📥 Download Details (CSV)", df_windows.to_csv(index=False), f'hybrid_results_{ticker}.csv', 'text/csv', key=f'download_{ticker}')
    
    # --- Encoder Section ---
    st.divider()
    st.markdown("#### 🎁 Generate Encoded String from Window Result")
    
    if not df_windows.empty:
        window_to_encode = st.number_input( "Select Window #", min_value=1, max_value=len(df_windows), value=1, key=f"encoder_win_{ticker}")
        
        if st.button("Encode Selected Window", key=f"encoder_btn_{ticker}"):
            try:
                window_data = df_windows.iloc[window_to_encode - 1]
                ticker_data = results_data['ticker_data']
                start_idx = (window_to_encode - 1) * st.session_state.window_size
                action_len = min(st.session_state.window_size, len(ticker_data) - start_idx)
                
                mutation_seeds_str = window_data['mutation_seeds']
                mutation_seeds = []
                if mutation_seeds_str not in ["None", "[]"]:
                    mutation_seeds = [int(s.strip()) for s in mutation_seeds_str.strip('[]').split(',') if s.strip()]
                
                encoded_string = SimulationTracer.encode(
                    action_length=action_len, mutation_rate=int(st.session_state.mutation_rate),
                    dna_seed=int(window_data['dna_seed']), mutation_seeds=mutation_seeds
                )
                st.success(f"**Encoded String for {ticker} - Window #{window_to_encode}:**")
                st.code(encoded_string, language='text')
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")

def render_tracer_tab():
    st.markdown("### 🔍 Action Sequence Tracer & Encoder")
    st.info("เครื่องมือนี้ใช้สำหรับ 1. **ถอดรหัส (Decode)** String เพื่อจำลองผลลัพธ์ และ 2. **เข้ารหัส (Encode)** พารามิเตอร์เพื่อสร้าง String")
    
    # Decoder
    st.markdown("---")
    st.markdown("#### 1. ถอดรหัส (Decode) String")
    encoded_string = st.text_input("ป้อน Encoded String ที่นี่:", "23021034252903219354832053493", key="decoder_input")
    if st.button("Trace & Simulate", type="primary", key="tracer_button"):
        if encoded_string:
            with st.spinner("กำลังถอดรหัสและจำลอง..."):
                try:
                    tracer = SimulationTracer(encoded_string=encoded_string)
                    st.success("ถอดรหัสสำเร็จ!"); st.code(str(tracer), language='bash')
                    final_actions = tracer.run()
                    st.markdown("#### 🎉 ผลลัพธ์ Action Sequence สุดท้าย:")
                    st.dataframe(pd.DataFrame(final_actions, columns=['Action']), use_container_width=True)
                except ValueError as e: st.error(f"❌ เกิดข้อผิดพลาด: {e}")
        else: st.warning("กรุณาป้อน Encoded String")
    
    # Encoder
    st.divider()
    st.markdown("#### 2. เข้ารหัส (Encode) พารามิเตอร์")
    col1, col2 = st.columns(2)
    with col1:
        action_length = st.number_input("Action Length", 1, value=30, key="enc_len")
        dna_seed = st.number_input("DNA Seed", 0, value=1234, format="%d", key="enc_dna")
    with col2:
        mutation_rate = st.number_input("Mutation Rate (%)", 0, value=10, key="enc_rate")
        seeds_str = st.text_input("Mutation Seeds (คั่นด้วย ,)", "10,20,30", key="enc_seeds_str")
    if st.button("Encode Parameters", key="encoder_button"):
        try:
            seeds_list = [int(s.strip()) for s in seeds_str.split(',') if s.strip()] if seeds_str else []
            generated_string = SimulationTracer.encode(int(action_length), int(mutation_rate), int(dna_seed), seeds_list)
            st.success("เข้ารหัสสำเร็จ!"); st.code(generated_string, language='text')
        except (ValueError, TypeError) as e: st.error(f"❌ ข้อมูลไม่ถูกต้อง: {e}")

# ==============================================================================
# 6. Main Application
# ==============================================================================
def main():
    st.markdown("### 🧪 Multi-Ticker Hybrid Strategy Lab")
    st.caption("เครื่องมือทดสอบกลยุทธ์ Hybrid Multi-Mutation กับหลายสินทรัพย์พร้อมกัน")

    initialize_session_state()
    tickers_to_test = load_tickers_config()
    
    if not tickers_to_test:
        st.warning("ไม่พบรายชื่อ Tickers ในไฟล์ `tickers_to_test.json` หรือไฟล์มีปัญหา")
        return

    # --- Setup Tabs ---
    tab_names = ["⚙️ การตั้งค่า & ควบคุม"]
    # If results exist, add ticker tabs and the tracer tab
    if st.session_state.all_ticker_results:
        tab_names.extend(list(st.session_state.all_ticker_results.keys()))
        tab_names.append("🔍 Tracer")

    tabs = st.tabs(tab_names)

    # --- Tab 1: Settings and Control ---
    with tabs[0]:
        render_settings_and_control_tab()

    # --- Main Processing Block ---
    if st.session_state.get('run_triggered', False):
        st.session_state.run_triggered = False  # Reset trigger
        
        main_progress_bar = st.progress(0, text="Starting backtest for all tickers...")
        status_area = st.empty()

        for i, ticker in enumerate(tickers_to_test):
            main_progress_text = f"Processing Ticker {i+1}/{len(tickers_to_test)}: **{ticker}**"
            main_progress_bar.progress((i) / len(tickers_to_test), text=main_progress_text)
            status_area.info(f"⬇️ Downloading data for {ticker}...")
            
            ticker_data = get_ticker_data(ticker, str(st.session_state.start_date), str(st.session_state.end_date))
            if ticker_data.empty:
                status_area.error(f"⚠️ ไม่พบข้อมูลสำหรับ {ticker}, ข้ามไปยังตัวถัดไป")
                st.session_state.all_ticker_results[f"{ticker} (No Data)"] = {}
                continue
            
            status_area.info(f"⚙️ Running Hybrid Multi-Mutation for {ticker}...")
            
            # This placeholder will be used by the generation function to show window-level progress
            window_status_placeholder = status_area.empty()
            original_actions, final_actions, df_windows = generate_actions_hybrid_multi_mutation(
                ticker_data, st.session_state.window_size, st.session_state.num_seeds,
                st.session_state.max_workers, st.session_state.mutation_rate,
                st.session_state.num_mutations, window_status_placeholder
            )
            
            status_area.info(f"📊 Simulating results for {ticker}...")
            prices = ticker_data['Close'].to_numpy()
            
            sim_results = {
                Strategy.HYBRID_MULTI_MUTATION: run_simulation(prices.tolist(), final_actions.tolist()),
                Strategy.ORIGINAL_DNA: run_simulation(prices.tolist(), original_actions.tolist()),
                Strategy.REBALANCE_DAILY: run_simulation(prices.tolist(), generate_actions_rebalance_daily(len(prices)).tolist()),
                Strategy.PERFECT_FORESIGHT: run_simulation(prices.tolist(), generate_actions_perfect_foresight(prices.tolist()).tolist())
            }
            for name, df in sim_results.items():
                if not df.empty: df.index = ticker_data.index[:len(df)]
            
            # Store all results for this ticker in session state
            st.session_state.all_ticker_results[ticker] = {
                "sim_results": sim_results,
                "df_windows_details": df_windows,
                "ticker_data": ticker_data
            }

        main_progress_bar.progress(1.0, text="✅ Backtest complete for all tickers!")
        status_area.success("โหลดผลลัพธ์ในแท็บด้านบน")
        st.rerun() # Rerun to draw the new tabs

    # --- Render Result Tabs (if they exist) ---
    if len(tabs) > 1:
        # Loop through the ticker tabs (all except the first and last)
        for i, tab in enumerate(tabs[1:-1]):
            with tab:
                ticker_name = tab_names[i+1]
                render_ticker_results_tab(ticker_name, st.session_state.all_ticker_results[ticker_name])

        # Render the Tracer tab
        with tabs[-1]:
            render_tracer_tab()


if __name__ == "__main__":
    main()
