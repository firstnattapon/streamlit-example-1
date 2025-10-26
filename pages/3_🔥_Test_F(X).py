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
        "assets": ["FFWM", "NEGG", "RIVN", "APLS", "NVTS", "QXO", "RXRX", "AGL" ,"FLNC" , "GERN" , "DYN" , "DJT", "IBRX" , "SG" , "CLSK" , "LUNR" ],
        "default_settings": {
            "selected_ticker": "FFWM", "start_date": "2024-01-01",
            "window_size": 30, "num_seeds": 1000, "max_workers": 8, # 8 cores
            "mutation_rate": 10.0, "num_mutations": 5,
            "num_restarts": 5 # ! NEW: Goal 1
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
    if 'num_seeds' not in st.session_state: st.session_state.num_seeds = defaults.get('num_seeds', 1000)
    if 'max_workers' not in st.session_state: st.session_state.max_workers = defaults.get('max_workers', 8)
    if 'mutation_rate' not in st.session_state: st.session_state.mutation_rate = defaults.get('mutation_rate', 10.0)
    if 'num_mutations' not in st.session_state: st.session_state.num_mutations = defaults.get('num_mutations', 5)
    # ! NEW (Goal 1): Add num_restarts to session state
    if 'num_restarts' not in st.session_state: st.session_state.num_restarts = defaults.get('num_restarts', 5)


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
# 3. Strategy Action Generation
# ==============================================================================
def generate_actions_rebalance_daily(num_days: int) -> np.ndarray:
    return np.ones(num_days, dtype=np.int32)

def generate_actions_perfect_foresight(prices: List[float], fix: int = 1500) -> np.ndarray:
    price_arr = np.asarray(prices, dtype=np.float64)
    n = len(price_arr)
    if n < 2:
        a = np.ones(n, dtype=np.int32)
        if n > 0: a[0] = 1
        return a

    dp = np.full(n, -np.inf, dtype=np.float64)
    path = np.zeros(n, dtype=np.int32)
    dp[0] = float(fix * 2.0) 

    for i in range(1, n):
        j_indices = np.arange(i)
        profits = fix * ((price_arr[i] / price_arr[j_indices]) - 1.0) 
        cand = dp[j_indices] + profits                                
        best_idx = int(np.argmax(cand))
        dp[i] = cand[best_idx]
        path[i] = j_indices[best_idx]

    final_scores = dp + fix * ((price_arr[-1] / price_arr) - 1.0)
    end_idx = int(np.argmax(final_scores))

    actions = np.zeros(n, dtype=np.int32)
    while end_idx > 0:
        actions[end_idx] = 1
        end_idx = path[end_idx]
    actions[0] = 1
    return actions

def find_best_seed_for_window(
    prices_window: np.ndarray,
    num_seeds_to_try: int,
    max_workers: int,
    master_rng: np.random.Generator # ! NEW (Goal 1): Receive master RNG
) -> Tuple[int, float, np.ndarray]:
    
    window_len = len(prices_window)
    if window_len < 2: return 1, 0.0, np.ones(window_len, dtype=np.int32)

    def evaluate_seed_batch(seed_batch: np.ndarray) -> List[Tuple[int, float]]:
        results = []
        for seed in seed_batch:
            # ใช้ seed (ที่เป็น int ขนาดใหญ่) เพื่อสร้าง RNG เฉพาะสำหรับตัวมันเอง
            rng = np.random.default_rng(seed=int(seed)) 
            actions = rng.integers(0, 2, size=window_len).astype(np.int32)
            actions[0] = 1
            net = _calculate_net_profit_numba(actions, prices_window)
            results.append((int(seed), net))
        return results

    best_seed, max_net = -1, -np.inf
    
    # ! MODIFIED (Goal 1): Generate RANDOM seeds to test, not just 0..N
    random_seeds = master_rng.integers(0, 2**31 - 1, size=num_seeds_to_try, dtype=np.int64)
    
    batch_size = max(1, num_seeds_to_try // (max_workers * 4 if max_workers > 0 else 1))
    seed_batches = [random_seeds[j:j+batch_size] for j in range(0, len(random_seeds), batch_size)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_seed_batch, batch) for batch in seed_batches]
        for future in as_completed(futures):
            for seed, final_net in future.result():
                if final_net > max_net: max_net, best_seed = final_net, seed

    if best_seed >= 0:
        rng_best = np.random.default_rng(best_seed)
        best_actions = rng_best.integers(0, 2, size=window_len).astype(np.int32)
    else:
        best_seed, best_actions, max_net = 1, np.ones(window_len, dtype=np.int32), 0.0

    best_actions[0] = 1
    return best_seed, max_net, best_actions

def find_best_mutation_for_sequence(
    original_actions: np.ndarray,
    prices_window: np.ndarray,
    num_mutation_seeds: int,
    mutation_rate: float,
    max_workers: int,
    master_rng: np.random.Generator # ! NEW (Goal 1): Receive master RNG
) -> Tuple[int, float, np.ndarray]:

    window_len = len(original_actions)
    if window_len < 2: return 1, -np.inf, original_actions

    def evaluate_mutation_seed_batch(seed_batch: np.ndarray) -> List[Tuple[int, float]]:
        results = []
        for seed in seed_batch:
            mutation_rng = np.random.default_rng(seed=int(seed))
            mutated_actions = original_actions.copy()
            mutation_mask = mutation_rng.random(window_len) < mutation_rate
            mutated_actions[mutation_mask] = 1 - mutated_actions[mutation_mask]
            mutated_actions[0] = 1
            net = _calculate_net_profit_numba(mutated_actions.astype(np.int32), prices_window)
            results.append((int(seed), net))
        return results

    best_mutation_seed, max_mutated_net = -1, -np.inf
    
    # ! MODIFIED (Goal 1): Generate RANDOM mutation seeds, not just 0..N
    mutation_seeds_to_try = master_rng.integers(0, 2**31 - 1, size=num_mutation_seeds, dtype=np.int64)

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
        max_mutated_net = -np.inf # No profitable mutation found
        final_mutated_actions = original_actions.copy()

    return best_mutation_seed, max_mutated_net, final_mutated_actions

def generate_actions_hybrid_multi_mutation(
    ticker_data: pd.DataFrame,
    window_size: int,
    num_seeds: int,
    max_workers: int,
    mutation_rate_pct: float,
    num_mutations: int,
    num_restarts: int # ! NEW (Goal 1): Number of restarts
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:

    prices = ticker_data['Close'].to_numpy()
    n = len(prices)
    final_actions = np.array([], dtype=np.int32)
    original_actions_full = np.array([], dtype=np.int32)
    window_details_list = []

    num_windows = (n + window_size - 1) // window_size
    
    # ! NEW (Goal 1): New Progress Bar logic
    total_main_steps = num_windows * num_restarts
    current_main_step = 0
    progress_bar = st.progress(0, text="Initializing Hybrid Multi-Mutation Search...")
    
    mutation_rate = mutation_rate_pct / 100.0
    
    # ! NEW (Goal 1): Master RNG for reproducibility of the whole run
    window_master_rng = np.random.default_rng(seed=42)

    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        if len(prices_window) < 2: continue

        # ! NEW (Goal 1): Variables to store the "Champion of Champions" for this window
        best_champion_of_champions_net = -np.inf
        best_champion_of_champions_actions = np.ones(len(prices_window), dtype=np.int32)
        best_champion_original_actions = np.ones(len(prices_window), dtype=np.int32)
        best_champion_details = {}
        
        start_date = ticker_data.index[start_index]
        end_date = ticker_data.index[end_index-1]

        # ! NEW (Goal 1): This is the "Random-Restart Hill Climbing" loop
        for restart_num in range(num_restarts):
            
            # ! NEW (Goal 1): Update progress bar
            current_main_step += 1
            progress_text = f"Window {i+1}/{num_windows} - Restart {restart_num+1}/{num_restarts} - Phase 1: DNA Search..."
            progress_bar.progress(current_main_step / total_main_steps, text=progress_text)

            # ! NEW (Goal 1): Create unique, reproducible RNGs for this restart
            restart_seed = window_master_rng.integers(0, 2**31 - 1)
            dna_search_rng = np.random.default_rng(restart_seed)
            mutation_search_rng = np.random.default_rng(restart_seed + 1) # Use a different seed for mutation search

            # --- Start of Original Logic (now inside restart loop) ---

            # --- Phase 1: Find Initial Champion for this restart ---
            dna_seed, current_best_net, current_best_actions = find_best_seed_for_window(
                prices_window, num_seeds, max_workers, master_rng=dna_search_rng
            )

            original_actions_window_for_this_restart = current_best_actions.copy()
            original_net_for_this_restart = current_best_net
            successful_mutation_seeds = []
            
            progress_text = f"Window {i+1}/{num_windows} - Restart {restart_num+1}/{num_restarts} - Phase 2: Mutating..."
            progress_bar.progress(current_main_step / total_main_steps, text=progress_text)


            # --- Phase 2: Iterative Mutation for this restart ---
            for mutation_round in range(num_mutations):
                mutation_seed, mutated_net, mutated_actions = find_best_mutation_for_sequence(
                    current_best_actions, prices_window, num_seeds, mutation_rate, max_workers,
                    master_rng=mutation_search_rng # Use mutation-specific RNG
                )

                # Survival of the Fittest for this restart's champion
                if mutated_net > current_best_net:
                    current_best_net = mutated_net
                    current_best_actions = mutated_actions
                    successful_mutation_seeds.append(int(mutation_seed))

            # --- End of Original Logic ---

            # ! NEW (Goal 1): "Survival of the Fittest" among Restarts
            # Compare the final champion of this *restart* with the best champion found *so far* in this window
            if current_best_net > best_champion_of_champions_net:
                best_champion_of_champions_net = current_best_net
                best_champion_of_champions_actions = current_best_actions.copy()
                best_champion_original_actions = original_actions_window_for_this_restart.copy()
                
                # Store the details of this *winning* restart
                best_champion_details = {
                    'window': i + 1, 'timeline': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                    'dna_seed': dna_seed, # The DNA seed of the winner
                    'mutation_seeds': str(successful_mutation_seeds) if successful_mutation_seeds else "None",
                    'improvements': len(successful_mutation_seeds),
                    'original_net': round(original_net_for_this_restart, 2), # Original net of the winner
                    'final_net': round(current_best_net, 2) # Final net of the winner
                }

        # --- End of Restart Loop ---
        
        # Now, append the true champion's data
        if best_champion_details: # Ensure we found at least one valid result
            final_actions = np.concatenate((final_actions, best_champion_of_champions_actions.astype(np.int32)))
            original_actions_full = np.concatenate((original_actions_full, best_champion_original_actions.astype(np.int32)))
            window_details_list.append(best_champion_details)
        else:
            # Fallback in case something went wrong (e.g., window too short)
            fallback_actions = np.ones(len(prices_window), dtype=np.int32)
            final_actions = np.concatenate((final_actions, fallback_actions))
            original_actions_full = np.concatenate((original_actions_full, fallback_actions))


    progress_bar.empty()
    return original_actions_full, final_actions, pd.DataFrame(window_details_list)

# ==============================================================================
# 4. Simulation Tracer Class (for the new tab)
# ==============================================================================
# ! NO CHANGES NEEDED for Goal 1 & 2. This class works with the *output*
# ! of the simulation (the seeds), and the output format is preserved.
class SimulationTracer:
    """
    คลาสสำหรับห่อหุ้มกระบวนการทั้งหมด ตั้งแต่การถอดรหัสพารามเตอร์
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
                length_of_number = int(encoded_string[idx]); idx += 1
                number_str = encoded_string[idx : idx + length_of_number]; idx += length_of_number
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
        current_actions = dna_rng.integers(0, 2, size=self.action_length).astype(np.int32)
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
# 5. UI Rendering Functions
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

def render_settings_tab():
    st.write("⚙️ **การตั้งค่าพารามิเตอร์**")
    config = load_config()
    asset_list = config.get('assets', ['FFWM'])

    c1, c2 = st.columns(2)
    st.session_state.test_ticker = c1.selectbox("เลือก Ticker สำหรับทดสอบ", options=asset_list, index=asset_list.index(st.session_state.test_ticker) if st.session_state.test_ticker in asset_list else 0)
    st.session_state.window_size = c2.number_input("ขนาด Window (วัน)", min_value=2, value=st.session_state.window_size)

    c1, c2 = st.columns(2)
    st.session_state.start_date = c1.date_input("วันที่เริ่มต้น", value=st.session_state.start_date)
    st.session_state.end_date = c2.date_input("วันที่สิ้นสุด", value=st.session_state.end_date)
    if st.session_state.start_date >= st.session_state.end_date: st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")

    st.divider()
    st.subheader("พารามิเตอร์สำหรับกลยุทธ์")
    c1, c2 = st.columns(2)
    st.session_state.num_seeds = c1.number_input("จำนวน Seeds (สำหรับค้นหา DNA และ Mutation)", min_value=100, value=st.session_state.num_seeds, format="%d", help="จำนวน 'ผู้ท้าชิง' ที่จะสุ่มสร้างในแต่ละรอบ (ต่อ 1 Restart)")
    st.session_state.max_workers = c2.number_input("จำนวน Workers (CPU Cores)", min_value=1, max_value=16, value=st.session_state.max_workers)
    
    c1, c2, c3 = st.columns(3)
    st.session_state.mutation_rate = c1.slider("อัตราการกลายพันธุ์ (Mutation Rate) %", min_value=0.0, max_value=50.0, value=st.session_state.mutation_rate, step=0.5)
    st.session_state.num_mutations = c2.number_input("จำนวนรอบการกลายพันธุ์ (Multi-Mutation)", min_value=0, max_value=10, value=st.session_state.num_mutations, help="จำนวนครั้งที่จะพยายามพัฒนายีนส์ต่อจากตัวที่ดีที่สุดในแต่ละ Window (ต่อ 1 Restart)")
    
    # ! NEW (Goal 1): Add UI for num_restarts
    st.session_state.num_restarts = c3.number_input(
        "จำนวนรอบการค้นหา (Random-Restarts)", 
        min_value=1, 
        max_value=50, 
        value=st.session_state.num_restarts, 
        help="จำนวนครั้งที่จะรันกระบวนการค้นหา (เฟส 1+2) ใหม่ทั้งหมดในแต่ละ Window เพื่อแก้ปัญหา Local Optimum"
    )

def render_hybrid_multi_mutation_tab():
    st.write("---")
    st.markdown(f"### 🧬 {Strategy.HYBRID_MULTI_MUTATION}")
    st.info("กลยุทธ์นี้ทำงานโดย: 1. ค้นหา 'DNA' ที่ดีที่สุดในแต่ละ Window 2. นำ DNA นั้นมาพยายาม 'กลายพันธุ์' (Mutate) ซ้ำๆ เพื่อหาผลลัพธ์ที่ดีกว่าเดิม")

    with st.expander("📖 คำอธิบายวิธีการทำงานและแนวคิด (Multi-Mutation)"):
  

    if st.button(f"🚀 Start Hybrid Multi-Mutation", type="primary"):
        if st.session_state.start_date >= st.session_state.end_date: st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้อง"); return
        ticker = st.session_state.test_ticker
        with st.spinner(f"กำลังดึงข้อมูลและประมวลผลสำหรับ {ticker}..."):
            ticker_data = get_ticker_data(ticker, str(st.session_state.start_date), str(st.session_state.end_date))
            if ticker_data.empty: st.error("ไม่พบข้อมูลสำหรับ Ticker และช่วงวันที่ที่เลือก"); return

            original_actions, final_actions, df_windows = generate_actions_hybrid_multi_mutation(
                ticker_data, st.session_state.window_size, st.session_state.num_seeds,
                st.session_state.max_workers, st.session_state.mutation_rate,
                st.session_state.num_mutations,
                st.session_state.num_restarts # ! NEW (Goal 1): Pass new param
            )

            prices = ticker_data['Close'].to_numpy()

            results = {
                Strategy.HYBRID_MULTI_MUTATION: run_simulation(prices.tolist(), final_actions.tolist()),
                Strategy.ORIGINAL_DNA: run_simulation(prices.tolist(), original_actions.tolist()),
                Strategy.REBALANCE_DAILY: run_simulation(prices.tolist(), generate_actions_rebalance_daily(len(prices)).tolist()),
                Strategy.PERFECT_FORESIGHT: run_simulation(prices.tolist(), generate_actions_perfect_foresight(prices.tolist()).tolist())
            }
            for name, df in results.items():
                if not df.empty: df.index = ticker_data.index[:len(df)]

            st.session_state.simulation_results = results
            st.session_state.df_windows_details = df_windows
            st.session_state.ticker_data_cache = ticker_data

    if 'simulation_results' in st.session_state:
        st.success("การทดสอบเสร็จสมบูรณ์!")
        results = st.session_state.simulation_results
        chart_results = {k: v for k, v in results.items() if k != Strategy.ORIGINAL_DNA}
        display_comparison_charts(chart_results)

        st.divider()
        st.write("### 📈 สรุปผลลัพธ์")

        df_windows = st.session_state.get('df_windows_details', pd.DataFrame())

        if not df_windows.empty:
            perfect_df = results.get(Strategy.PERFECT_FORESIGHT)
            total_perfect_net = perfect_df['net'].iloc[-1] if perfect_df is not None and not perfect_df.empty else 0.0

            hybrid_df = results.get(Strategy.HYBRID_MULTI_MUTATION)
            total_hybrid_net = hybrid_df['net'].iloc[-1] if hybrid_df is not None and not hybrid_df.empty else 0.0

            original_df = results.get(Strategy.ORIGINAL_DNA)
            total_original_net = original_df['net'].iloc[-1] if original_df is not None and not original_df.empty else 0.0

            rebalance_df = results.get(Strategy.REBALANCE_DAILY)
            total_rebalance_net = rebalance_df['net'].iloc[-1] if rebalance_df is not None and not rebalance_df.empty else 0.0

            st.write("#### สรุปผลการดำเนินงานโดยรวม (Compounded Final Profit)")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Perfect Foresight", f"${total_perfect_net:,.2f}", help="กำไรสุทธิสุดท้ายจากการจำลองต่อเนื่องแบบ Perfect Foresight (ทบต้น)")
            col2.metric("Hybrid Strategy", f"${total_hybrid_net:,.2f}", help="กำไรสุทธิสุดท้ายจากการจำลองต่อเนื่องของกลยุทธ์ Hybrid ที่ผ่านการกลายพันธุ์แล้ว (ทบต้น)")
            col3.metric("Original Profits", f"${total_original_net:,.2f}", help="กำไรสุทธิสุดท้ายจากการจำลองต่อเนื่องของ 'DNA ดั้งเดิม' (ของแชมป์เปี้ยนตัวจริง) ก่อนการกลายพันธุ์ (ทบต้น)")
            col4.metric("Rebalance Daily", f"${total_rebalance_net:,.2f}", help="กำไรสุทธิสุดท้ายจากการจำลองต่อเนื่องของกลยุทธ์ Rebalance Daily (ทบต้น)")

            st.write("---")
            st.write("#### 📝 รายละเอียดผลลัพธ์ราย Window (ของ 'แชมป์เปี้ยน' ที่ชนะในแต่ละ Window)")
            st.dataframe(df_windows, use_container_width=True)
            ticker = st.session_state.get('test_ticker', 'TICKER')
            st.download_button("📥 Download Details (CSV)", df_windows.to_csv(index=False), f'hybrid_multi_mutation_{ticker}.csv', 'text/csv')

            st.divider()
            st.markdown("#### 🎁 Generate Encoded String from Window Result")
            st.info("เลือกหมายเลข Window จากตารางด้านบนเพื่อสร้าง Encoded String สำหรับนำไปใช้ในแท็บ 'Tracer'")

            c1, c2 = st.columns([1, 3])
            with c1:
                max_window = len(df_windows)
                if max_window == 0:
                    st.warning("ไม่มีข้อมูล Window")
                    # Set default action_length to avoid errors later
                    action_length_for_encoder = st.number_input(
                        "Action Length", min_value=1, value=st.session_state.window_size, key="action_length_for_encoder_disabled", disabled=True
                    )
                else:
                    window_to_encode = st.number_input(
                        "Select Window #", min_value=1, max_value=max_window, value=1, key="window_encoder_input"
                    )
                    try:
                        total_days = len(st.session_state.ticker_data_cache)
                        window_size = st.session_state.window_size
                        start_index = (window_to_encode - 1) * window_size
                        # คำนวณความยาว action ที่แท้จริงสำหรับ window นั้นๆ
                        actual_action_length = min(window_size, total_days - start_index)
                    except (KeyError, TypeError):
                        actual_action_length = st.session_state.get('window_size', 30)

                    action_length_for_encoder = st.number_input(
                        "Action Length",
                        min_value=1,
                        value=actual_action_length,
                        key="action_length_for_encoder",
                        help="ความยาวของ action sequence สำหรับ window ที่เลือก (คำนวณอัตโนมัติ)"
                    )

            with c2:
                st.write("")
                st.write("")
                if max_window > 0 and st.button("Encode Selected Window", key="window_encoder_button"):
                    try:
                        window_data = df_windows.iloc[window_to_encode - 1]
                        dna_seed = int(window_data['dna_seed'])
                        
                        # ! MODIFIED: ใช้ mutation_rate จาก session_state (เนื่องจากมันไม่ได้ถูกเก็บในตาราง)
                        # แปลงเป็น int หากเป็น float (เช่น 10.0 -> 10)
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
                        st.success(f"**Encoded String for Window #{window_to_encode}:**")
                        st.code(encoded_string, language='text')

                    except (IndexError, KeyError):
                        st.error(f"ไม่สามารถหาข้อมูลสำหรับ Window #{window_to_encode} ได้")
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดระหว่างการเข้ารหัส: {e}")

def render_tracer_tab():
    st.markdown("### 🔍 Action Sequence Tracer & Encoder")
    st.info("เครื่องมือนี้ใช้สำหรับ 1. **ถอดรหัส (Decode)** String เพื่อจำลองผลลัพธ์ และ 2. **เข้ารหัส (Encode)** พารามิเตอร์เพื่อสร้าง String")

    st.markdown("---")
    st.markdown("#### 1. ถอดรหัส (Decode) String")

    encoded_string = st.text_input(
        "ป้อน Encoded String ที่นี่:",
        "2302103860538865333056135842", # ! NEW: Example updated based on new logic
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
        action_length_input = st.number_input("Action Length", min_value=1, value=30, key="enc_len", help="ความยาวของ action sequence")
        dna_seed_input = st.number_input("DNA Seed", min_value=0, value=860, format="%d", key="enc_dna", help="Seed สำหรับสร้าง DNA ดั้งเดิม")
    with col2:
        mutation_rate_input = st.number_input("Mutation Rate (%)", min_value=0, value=10, key="enc_rate", help="อัตราการกลายพันธุ์เป็นเปอร์เซ็นต์ (เช่น 10 สำหรับ 10%)")
        mutation_seeds_str = st.text_input(
            "Mutation Seeds (คั่นด้วยจุลภาค ,)",
            "506, 164, 842, 308",
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
    st.markdown("### 🧬 Hybrid Strategy Lab (Random-Restart Hill Climbing)")
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
