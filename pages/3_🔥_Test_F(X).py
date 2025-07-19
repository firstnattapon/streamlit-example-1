#pages/4_🧬_Hybrid_Multi_Mutation.py
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Tuple, Dict, Any
 
# ! NUMBA: Import Numba's Just-In-Time compiler for core acceleration
from numba import njit
 
# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================
st.set_page_config(page_title="Hybrid_Multi_Mutation", page_icon="🧬", layout="wide")

class Strategy:
    """คลาสสำหรับเก็บชื่อกลยุทธ์ต่างๆ เพื่อให้เรียกใช้ง่ายและลดข้อผิดพลาด"""
    REBALANCE_DAILY = "Rebalance Daily"
    PERFECT_FORESIGHT = "Perfect Foresight (Max)"
    HYBRID_MULTI_MUTATION = "Hybrid (Multi-Mutation)"
    ORIGINAL_DNA = "Original DNA (Pre-Mutation)"

def load_config(filepath: str = "hybrid_seed_config.json") -> Dict[str, Any]:
    # In a real app, this might load from a JSON file. For simplicity, it's a dict.
    return {
        "assets": ["FFWM", "NEGG", "RIVN", "APLS", "NVTS", "QXO", "RXRX", "AGL" ,"FLNC" , "GERN" , "DYN" ],
        "default_settings": {
            "selected_ticker": "FFWM", "start_date": "2024-01-01",
            "window_size": 30 , "num_seeds": 1000, "max_workers": 1, 
            "mutation_rate": 10.0, "num_mutations": 5
        }
    }

def initialize_session_state(config: Dict[str, Any]):
    defaults = config.get('default_settings', {})
    if 'test_ticker' not in st.session_state: st.session_state.test_ticker = defaults.get('selected_ticker', 'FFWM')
    if 'start_date' not in st.session_state:
        try: st.session_state.start_date = datetime.strptime(defaults.get('start_date', '2024-01-01'), '%Y-%m-%d').date()
        except ValueError: st.session_state.start_date = datetime(2024, 1, 1).date()
    if 'end_date' not in st.session_state: st.session_state.end_date = datetime.now().date()
    if 'window_size' not in st.session_state: st.session_state.window_size = defaults.get('window_size', 30 )  
    if 'num_seeds' not in st.session_state: st.session_state.num_seeds = defaults.get('num_seeds', 1000)
    if 'max_workers' not in st.session_state: st.session_state.max_workers = defaults.get('max_workers', 8)
    if 'mutation_rate' not in st.session_state: st.session_state.mutation_rate = defaults.get('mutation_rate', 10.0)
    if 'num_mutations' not in st.session_state: st.session_state.num_mutations = defaults.get('num_mutations', 5)

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
# 3. Strategy Action Generation (No Changes)
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
    original_actions: np.ndarray,
    prices_window: np.ndarray,
    num_mutation_seeds: int,
    mutation_rate: float,
    max_workers: int
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
                    max_mutated_net = net
                    best_mutation_seed = seed

    if best_mutation_seed >= 0:
        mutation_rng = np.random.default_rng(best_mutation_seed)
        final_mutated_actions = original_actions.copy()
        mutation_mask = mutation_rng.random(window_len) < mutation_rate
        final_mutated_actions[mutation_mask] = 1 - final_mutated_actions[mutation_mask]
        final_mutated_actions[0] = 1
    else:
        best_mutation_seed = -1
        max_mutated_net = -np.inf
        final_mutated_actions = original_actions.copy()

    return best_mutation_seed, max_mutated_net, final_mutated_actions

def generate_actions_hybrid_multi_mutation(
    ticker_data: pd.DataFrame,
    window_size: int,
    num_seeds: int,
    max_workers: int,
    mutation_rate_pct: float,
    num_mutations: int,
    progress_bar: st.progress # Pass progress bar to update it
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:

    prices = ticker_data['Close'].to_numpy()
    n = len(prices)
    final_actions = np.array([], dtype=int)
    original_actions_full = np.array([], dtype=int)
    window_details_list = []

    num_windows = (n + window_size - 1) // window_size
    mutation_rate = mutation_rate_pct / 100.0

    for i, start_index in enumerate(range(0, n, window_size)):
        progress_total_steps = num_mutations + 1
        current_step_in_window = 0

        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        if len(prices_window) < 2: continue

        progress_text = f"Window {i+1}/{num_windows} - Phase 1: Searching for Best DNA..."
        # This function updates its own progress bar now
        progress_bar.progress(current_step_in_window / progress_total_steps, text=progress_text)
        current_step_in_window += 1

        dna_seed, current_best_net, current_best_actions = find_best_seed_for_window(prices_window, num_seeds, max_workers)
        
        original_actions_window = current_best_actions.copy()
        original_net_for_display = current_best_net
        successful_mutation_seeds = []

        for mutation_round in range(num_mutations):
            progress_text = f"Window {i+1}/{num_windows} - Mutation Round {mutation_round+1}/{num_mutations}..."
            progress_bar.progress(current_step_in_window / progress_total_steps, text=progress_text)
            current_step_in_window += 1
            
            mutation_seed, mutated_net, mutated_actions = find_best_mutation_for_sequence(
                current_best_actions, prices_window, num_seeds, mutation_rate, max_workers
            )

            if mutated_net > current_best_net:
                current_best_net = mutated_net
                current_best_actions = mutated_actions
                successful_mutation_seeds.append(int(mutation_seed))

        final_actions = np.concatenate((final_actions, current_best_actions))
        original_actions_full = np.concatenate((original_actions_full, original_actions_window))

        start_date = ticker_data.index[start_index]; end_date = ticker_data.index[end_index-1]
        detail = {
            'window': i + 1, 'timeline': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            'dna_seed': dna_seed,
            'mutation_seeds': str(successful_mutation_seeds) if successful_mutation_seeds else "None",
            'improvements': len(successful_mutation_seeds),
            'original_net': round(original_net_for_display, 2),
            'final_net': round(current_best_net, 2)
        }
        window_details_list.append(detail)

    progress_bar.progress(1.0, text="Completed.")
    return original_actions_full, final_actions, pd.DataFrame(window_details_list)

# ==============================================================================
# 4. Simulation Tracer Class (No Changes)
# ==============================================================================
class SimulationTracer:
    """
    คลาสสำหรับห่อหุ้มกระบวนการทั้งหมด ตั้งแต่การถอดรหัสพารามิเตอร์
    ไปจนถึงการจำลองกระบวนการกลายพันธุ์ของ action sequence
    """
    
    def __init__(self, encoded_string: str):
        self.encoded_string: str = encoded_string
        self._decode_and_set_attributes()

    def _decode_and_set_attributes(self):
        encoded_string = self.encoded_string
        if not isinstance(encoded_string, str) or not encoded_string.isdigit():
            raise ValueError("Input ต้องเป็นสตริงที่ประกอบด้วยตัวเลขเท่านั้น")

        decoded_numbers = []
        idx = 0
        while idx < len(encoded_string):
            try:
                length_of_number = int(encoded_string[idx])
                idx += 1
                number_str = encoded_string[idx : idx + length_of_number]
                idx += length_of_number
                decoded_numbers.append(int(number_str))
            except (IndexError, ValueError):
                raise ValueError(f"รูปแบบของสตริงไม่ถูกต้องที่ตำแหน่ง {idx}")

        if len(decoded_numbers) < 3:
            raise ValueError("ข้อมูลในสตริงไม่ครบถ้วน (ต้องการอย่างน้อย 3 ค่า)")

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
        return (
            "✅ พารามิเตอร์ที่ถอดรหัสสำเร็จ:\n"
            f"- action_length: {self.action_length}\n"
            f"- mutation_rate: {self.mutation_rate} ({self.mutation_rate_float:.2f})\n"
            f"- dna_seed: {self.dna_seed}\n"
            f"- mutation_seeds: {self.mutation_seeds}"
        )

    @staticmethod
    def encode(
        action_length: int,
        mutation_rate: int,
        dna_seed: int,
        mutation_seeds: List[int]
    ) -> str:
        all_numbers = [action_length, mutation_rate, dna_seed] + mutation_seeds
        encoded_parts = [f"{len(str(num))}{num}" for num in all_numbers]
        return "".join(encoded_parts)

# ==============================================================================
# 5. UI Rendering Functions (RESTRUCTURED)
# ==============================================================================
def render_settings_tab():
    st.write("⚙️ **การตั้งค่าพารามิเตอร์**")
    config = load_config()
    asset_list = config.get('assets', ['FFWM'])

    c1, c2 = st.columns(2)
    # This ticker selection is now a general default, not the one controlling the run
    st.session_state.test_ticker = c1.selectbox("เลือก Ticker เริ่มต้น (Default)", options=asset_list, index=asset_list.index(st.session_state.get('test_ticker', 'FFWM')) if st.session_state.get('test_ticker', 'FFWM') in asset_list else 0)
    st.session_state.window_size = c2.number_input("ขนาด Window (วัน)", min_value=2, value=st.session_state.window_size)

    c1, c2 = st.columns(2)
    st.session_state.start_date = c1.date_input("วันที่เริ่มต้น", value=st.session_state.start_date)
    st.session_state.end_date = c2.date_input("วันที่สิ้นสุด", value=st.session_state.end_date)
    if st.session_state.start_date >= st.session_state.end_date: st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")

    st.divider()
    st.subheader("พารามิเตอร์สำหรับกลยุทธ์")
    c1, c2 = st.columns(2)
    st.session_state.num_seeds = c1.number_input("จำนวน Seeds (สำหรับค้นหา DNA และ Mutation)", min_value=100, value=st.session_state.num_seeds, format="%d")
    st.session_state.max_workers = c2.number_input("จำนวน Workers (CPU Cores)", min_value=1, max_value=16, value=st.session_state.max_workers)

    c1, c2 = st.columns(2)
    st.session_state.mutation_rate = c1.slider("อัตราการกลายพันธุ์ (Mutation Rate) %", min_value=0.0, max_value=50.0, value=st.session_state.mutation_rate, step=0.5)
    st.session_state.num_mutations = c2.number_input("จำนวนรอบการกลายพันธุ์ (Multi-Mutation)", min_value=0, max_value=10, value=st.session_state.num_mutations, help="จำนวนครั้งที่จะพยายามพัฒนายีนส์ต่อจากตัวที่ดีที่สุดในแต่ละ Window")
    
    # --- MOVED: Encoder Section ---
    st.divider()
    st.markdown("#### 🎁 Generate Encoded String from Window Result")
    
    if 'all_window_details' in st.session_state and st.session_state.all_window_details:
        st.info("เลือก Ticker และหมายเลข Window จากผลการทดสอบล่าสุดเพื่อสร้าง Encoded String")
        
        # Select Ticker from completed results
        available_tickers = list(st.session_state.all_window_details.keys())
        selected_ticker_for_encode = st.selectbox("Select Ticker to Encode", options=available_tickers)

        if selected_ticker_for_encode:
            df_windows = st.session_state.all_window_details[selected_ticker_for_encode]
            ticker_data_cache = st.session_state.all_ticker_data[selected_ticker_for_encode]

            c1, c2 = st.columns([1, 3])
            with c1:
                max_window = len(df_windows)
                window_to_encode = st.number_input(
                    "Select Window #", min_value=1, max_value=max_window, value=1, key="window_encoder_input"
                )
                
                # Calculate the default action length for the selected window
                try:
                    total_days = len(ticker_data_cache)
                    window_size = st.session_state.window_size
                    start_index = (window_to_encode - 1) * window_size
                    default_action_length = min(window_size, total_days - start_index)
                except (KeyError, TypeError):
                    default_action_length = st.session_state.get('window_size', 30)

                action_length_for_encoder = st.number_input(
                    "Action Length", min_value=1, value=default_action_length, key="action_length_for_encoder",
                    help="ความยาวของ action sequence สำหรับ window ที่เลือก (คำนวณอัตโนมัติ)"
                )

            with c2:
                st.write("") # for vertical alignment
                if st.button("Encode Selected Window", key="window_encoder_button", use_container_width=True):
                    try:
                        window_data = df_windows.iloc[window_to_encode - 1]
                        dna_seed = int(window_data['dna_seed'])
                        mutation_rate = int(st.session_state.mutation_rate)
                        
                        mutation_seeds_str = window_data['mutation_seeds']
                        mutation_seeds = []
                        if mutation_seeds_str not in ["None", "[]"]:
                            cleaned_str = mutation_seeds_str.strip('[]')
                            if cleaned_str:
                                mutation_seeds = [int(s.strip()) for s in cleaned_str.split(',')]

                        action_length_to_use = int(action_length_for_encoder)
                        
                        encoded_string = SimulationTracer.encode(
                            action_length=action_length_to_use,
                            mutation_rate=mutation_rate,
                            dna_seed=dna_seed,
                            mutation_seeds=mutation_seeds
                        )
                        
                        st.success(f"**Encoded String for {selected_ticker_for_encode} - Window #{window_to_encode}:**")
                        st.code(encoded_string, language='text')

                    except (IndexError, KeyError) as e:
                        st.error(f"ไม่สามารถหาข้อมูลได้: {e}")
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดระหว่างการเข้ารหัส: {e}")
    else:
        st.warning("กรุณาทำการทดสอบในแท็บ 'Hybrid (Multi-Mutation)' ก่อนเพื่อสร้างผลลัพธ์")

def render_hybrid_multi_mutation_tab():
    st.write("---")
    st.markdown(f"### 🧬 {Strategy.HYBRID_MULTI_MUTATION}")
    st.info("กลยุทธ์นี้ทำงานโดย: 1. ค้นหา 'DNA' ที่ดีที่สุดในแต่ละ Window 2. นำ DNA นั้นมาพยายาม 'กลายพันธุ์' (Mutate) ซ้ำๆ เพื่อหาผลลัพธ์ที่ดีกว่าเดิม")

    with st.expander("📖 คำอธิบายวิธีการทำงานและแนวคิด (Multi-Mutation)", expanded=False):
        # (Explanation content remains the same, so it's omitted for brevity)
        st.markdown("...")
        
    if st.button(f"🚀 Start Hybrid Multi-Mutation for ALL Tickers", type="primary", use_container_width=True):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้อง")
            return

        config = load_config()
        tickers_to_run = config.get('assets', [])
        if not tickers_to_run:
            st.error("ไม่พบรายชื่อ Ticker ในไฟล์ config")
            return

        # Initialize containers for results
        st.session_state.all_ticker_data = {}
        st.session_state.all_window_details = {}
        all_results_summary = []
        
        # Master progress bar for all tickers
        master_progress = st.progress(0, text="Initializing full test run...")
        
        for i, ticker in enumerate(tickers_to_run):
            master_progress.progress((i) / len(tickers_to_run), text=f"Processing Ticker {i+1}/{len(tickers_to_run)}: **{ticker}**")
            
            with st.status(f"Running simulation for {ticker}...", expanded=False) as status:
                ticker_data = get_ticker_data(ticker, str(st.session_state.start_date), str(st.session_state.end_date))
                if ticker_data.empty:
                    st.warning(f"ไม่พบข้อมูลสำหรับ {ticker}, ข้ามไป...")
                    continue
                
                st.session_state.all_ticker_data[ticker] = ticker_data
                
                # Create a dedicated progress bar for this ticker's windows
                ticker_progress_bar = st.progress(0, text=f"Initializing {ticker}...")

                original_actions, final_actions, df_windows = generate_actions_hybrid_multi_mutation(
                    ticker_data, st.session_state.window_size, st.session_state.num_seeds,
                    st.session_state.max_workers, st.session_state.mutation_rate,
                    st.session_state.num_mutations, ticker_progress_bar
                )
                
                st.session_state.all_window_details[ticker] = df_windows
                
                prices = ticker_data['Close'].to_numpy()
                
                results = {
                    Strategy.HYBRID_MULTI_MUTATION: run_simulation(prices.tolist(), final_actions.tolist()),
                    Strategy.ORIGINAL_DNA: run_simulation(prices.tolist(), original_actions.tolist()),
                    Strategy.REBALANCE_DAILY: run_simulation(prices.tolist(), generate_actions_rebalance_daily(len(prices)).tolist()),
                    Strategy.PERFECT_FORESIGHT: run_simulation(prices.tolist(), generate_actions_perfect_foresight(prices.tolist()).tolist())
                }

                # Extract final net profit for summary
                ticker_summary = {'Ticker': ticker}
                for name, df in results.items():
                    final_net = df['net'].iloc[-1] if not df.empty else 0.0
                    ticker_summary[name] = final_net
                all_results_summary.append(ticker_summary)

                status.update(label=f"✅ {ticker} simulation complete!", state="complete", expanded=False)
        
        master_progress.progress(1.0, text="All Tickers Processed!")

        # Store final summary DataFrame in session state
        st.session_state.summary_df = pd.DataFrame(all_results_summary).set_index('Ticker')

    # --- Display results section ---
    if 'summary_df' in st.session_state:
        st.success("การทดสอบสำหรับ Ticker ทั้งหมดเสร็จสมบูรณ์!")
        st.divider()
        st.write("### 📈 สรุปผลลัพธ์ (All Tickers)")
        st.write("ตารางสรุปกำไรสุทธิสุดท้าย (Compounded Final Net Profit) ของแต่ละ Ticker และกลยุทธ์")
        
        summary_df = st.session_state.summary_df
        # Display styled summary table
        st.dataframe(
            summary_df.style.format("{:,.2f}").background_gradient(cmap='Greens', subset=[Strategy.HYBRID_MULTI_MUTATION])
                                  .background_gradient(cmap='RdYlGn', subset=[Strategy.ORIGINAL_DNA])
                                  .highlight_max(axis=0, color='lightgreen'),
            use_container_width=True
        )

        # Optional: Chart of overall results
        st.write("#### เปรียบเทียบกำไรสุทธิของกลยุทธ์ Hybrid")
        st.bar_chart(summary_df[[Strategy.HYBRID_MULTI_MUTATION, Strategy.ORIGINAL_DNA]])
        
        st.divider()
        st.write("#### 📝 รายละเอียดผลลัพธ์ราย Window (เลือก Ticker เพื่อดูรายละเอียด)")
        
        if 'all_window_details' in st.session_state and st.session_state.all_window_details:
            selected_ticker_details = st.selectbox(
                "เลือก Ticker เพื่อดูรายละเอียดราย Window:",
                options=list(st.session_state.all_window_details.keys())
            )
            if selected_ticker_details:
                detail_df = st.session_state.all_window_details[selected_ticker_details]
                st.dataframe(detail_df, use_container_width=True)
                st.download_button(
                    f"📥 Download {selected_ticker_details} Details (CSV)",
                    detail_df.to_csv(index=False),
                    f'hybrid_multi_mutation_{selected_ticker_details}.csv',
                    'text/csv'
                )

def render_tracer_tab():
    # This function remains unchanged, so it is omitted for brevity
    st.markdown("### 🔍 Action Sequence Tracer & Encoder")
    # ... (same code as before) ...
    st.info("เครื่องมือนี้ใช้สำหรับ 1. **ถอดรหัส (Decode)** String เพื่อจำลองผลลัพธ์ และ 2. **เข้ารหัส (Encode)** พารามิเตอร์เพื่อสร้าง String")

    st.markdown("---")
    st.markdown("#### 1. ถอดรหัส (Decode) String")
    
    encoded_string = st.text_input(
        "ป้อน Encoded String ที่นี่:",
        "260210295355822131233176", # Example string for Window 13
        help="สตริงที่เข้ารหัสพารามิเตอร์ต่างๆ เช่น action_length, mutation_rate, dna_seed, และ mutation_seeds",
        key="decoder_input"
    )

    if st.button("Trace & Simulate", type="primary", key="tracer_button"):
        if not encoded_string:
            st.warning("กรุณาป้อน Encoded String")
        else:
            with st.spinner(f"กำลังถอดรหัสและจำลองสำหรับ: {encoded_string[:20]}..."):
                try:
                    tracer = SimulationTracer(encoded_string=encoded_string)
                    st.success("ถอดรหัสสำเร็จ!")
                    st.code(str(tracer), language='bash')
                    final_actions = tracer.run()
                    st.write("---")
                    st.markdown("#### 🎉 ผลลัพธ์ Action Sequence สุดท้าย:")
                    st.dataframe(pd.DataFrame(final_actions, columns=['Action']), use_container_width=True)
                    st.write("Raw Array:")
                    st.code(str(final_actions))
                except ValueError as e:
                    st.error(f"❌ เกิดข้อผิดพลาดในการประมวลผล: {e}")

    st.divider()
    
    st.markdown("#### 2. เข้ารหัส (Encode) พารามิเตอร์")
    st.write("ป้อนพารามิเตอร์เพื่อสร้าง Encoded String สำหรับการทดลองซ้ำ")

    col1, col2 = st.columns(2)
    with col1:
        action_length_input = st.number_input("Action Length", min_value=1, value=60, key="enc_len", help="ความยาวของ action sequence")
        dna_seed_input = st.number_input("DNA Seed", min_value=0, value=900, format="%d", key="enc_dna", help="Seed สำหรับสร้าง DNA ดั้งเดิม")
    with col2:
        mutation_rate_input = st.number_input("Mutation Rate (%)", min_value=0, value=10, key="enc_rate", help="อัตราการกลายพันธุ์เป็นเปอร์เซ็นต์ (เช่น 5 สำหรับ 5%)")
        mutation_seeds_str = st.text_input(
            "Mutation Seeds (คั่นด้วยจุลภาค ,)", 
            "899, 530, 35, 814, 646", 
            key="enc_seeds_str",
            help="ชุดของ Seed สำหรับการกลายพันธุ์แต่ละรอบ คั่นด้วยเครื่องหมายจุลภาค"
        )
        
    if st.button("Encode Parameters", key="encoder_button"):
        try:
            if mutation_seeds_str.strip():
                mutation_seeds_list = [int(s.strip()) for s in mutation_seeds_str.split(',')]
            else:
                mutation_seeds_list = []

            generated_string = SimulationTracer.encode(
                action_length=int(action_length_input),
                mutation_rate=int(mutation_rate_input),
                dna_seed=int(dna_seed_input),
                mutation_seeds=mutation_seeds_list
            )
            
            st.success("เข้ารหัสสำเร็จ! สามารถคัดลอก String ด้านล่างไปใช้ได้")
            st.code(generated_string, language='text')

        except (ValueError, TypeError) as e:
            st.error(f"❌ เกิดข้อผิดพลาด: กรุณาตรวจสอบว่า Mutation Seeds เป็นตัวเลขที่คั่นด้วยจุลภาคเท่านั้น ({e})")


# ==============================================================================
# 6. Main Application
# ==============================================================================
def main():
    st.markdown("### 🧬 Hybrid Strategy Lab (Multi-Mutation)")
    st.caption("เครื่องมือทดลองและพัฒนากลยุทธ์ด้วย Numba-Accelerated Parallel Random Search")

    config = load_config()
    initialize_session_state(config)

    tab_list = ["⚙️ การตั้งค่า", f"🧬 {Strategy.HYBRID_MULTI_MUTATION}", "🔍 Tracer"]
    tabs = st.tabs(tab_list)

    with tabs[0]:
        render_settings_tab()
    with tabs[1]:
        render_hybrid_multi_mutation_tab()
    with tabs[2]:
        render_tracer_tab()

if __name__ == "__main__":
    main()
