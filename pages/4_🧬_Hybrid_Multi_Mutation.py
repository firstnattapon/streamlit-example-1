import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Tuple, Dict, Any
import json  # ! NEW: For Export Functionality

# ! NUMBA: Import Numba's Just-In-Time compiler for core acceleration
from numba import njit

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================
st.set_page_config(page_title="Hybrid_Multi_Mutation_Tabs", page_icon="🧬", layout="wide")

class Strategy:
    """คลาสสำหรับเก็บชื่อกลยุทธ์ต่างๆ"""
    REBALANCE_DAILY = "Rebalance Daily"
    PERFECT_FORESIGHT = "Perfect Foresight (Max)"
    HYBRID_MULTI_MUTATION = "Hybrid (Multi-Mutation)"
    ORIGINAL_DNA = "Original DNA (Pre-Mutation)"

def load_config() -> Dict[str, Any]:
    return {
        "assets": ["FSUN", "NEGG", "RIVN", "APLS", "NVTS", "QXO", "RXRX", "AGL" ,"FLNC" , "GERN" , "DYN" , "DJT", "IBRX" , "SG" , "CLSK" , "LUNR" ],
        "default_settings": {
            "start_date": "2024-01-01",
            "window_size": 30, "num_seeds": 1000, "max_workers": 1,
            "mutation_rate": 10.0, "num_mutations": 5
        }
    }

def initialize_session_state(config: Dict[str, Any]):
    defaults = config.get('default_settings', {})
    asset_list = config.get('assets', [])
    
    # ! GOAL: Default Select ALL Tickers
    if 'selected_tickers' not in st.session_state: 
        st.session_state.selected_tickers = asset_list
    
    if 'start_date' not in st.session_state:
        try: st.session_state.start_date = datetime.strptime(defaults.get('start_date', '2024-01-01'), '%Y-%m-%d').date()
        except ValueError: st.session_state.start_date = datetime(2024, 1, 1).date()
    if 'end_date' not in st.session_state: st.session_state.end_date = datetime.now().date()
    
    # Simulation Params
    if 'window_size' not in st.session_state: st.session_state.window_size = defaults.get('window_size', 30)
    if 'num_seeds' not in st.session_state: st.session_state.num_seeds = defaults.get('num_seeds', 1000)
    if 'max_workers' not in st.session_state: st.session_state.max_workers = defaults.get('max_workers', 8)
    if 'mutation_rate' not in st.session_state: st.session_state.mutation_rate = defaults.get('mutation_rate', 10.0)
    if 'num_mutations' not in st.session_state: st.session_state.num_mutations = defaults.get('num_mutations', 5)
    
    # ! GOAL Step 4: Global Encoding Settings
    if 'trace_target_window' not in st.session_state: st.session_state.trace_target_window = 1
    if 'trace_action_length' not in st.session_state: st.session_state.trace_action_length = 0 # 0 means Auto (Window Size)
    
    if 'batch_results' not in st.session_state: st.session_state.batch_results = {}

# ==============================================================================
# 2. Core Calculation & Data Functions
# ==============================================================================
@st.cache_data(ttl=3600)
def get_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        data = yf.Ticker(ticker).history(start=start_date, end=end_date)[['Close']]
        if data.empty: return pd.DataFrame()
        if data.index.tz is None: 
            data = data.tz_localize('UTC').tz_convert('Asia/Bangkok')
        else: 
            data = data.tz_convert('Asia/Bangkok')
        return data
    except Exception as e:
        return pd.DataFrame()

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

def find_best_seed_for_window(prices_window: np.ndarray, num_seeds_to_try: int, max_workers: int) -> Tuple[int, float, np.ndarray]:
    window_len = len(prices_window)
    if window_len < 2: return 1, 0.0, np.ones(window_len, dtype=np.int32)

    def evaluate_seed_batch(seed_batch: np.ndarray) -> List[Tuple[int, float]]:
        results = []
        for seed in seed_batch:
            rng = np.random.default_rng(seed)
            actions = rng.integers(0, 2, size=window_len).astype(np.int32)
            actions[0] = 1
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
        best_actions = rng_best.integers(0, 2, size=window_len).astype(np.int32)
    else:
        best_seed, best_actions, max_net = 1, np.ones(window_len, dtype=np.int32), 0.0
    best_actions[0] = 1
    return best_seed, max_net, best_actions

def find_best_mutation_for_sequence(original_actions: np.ndarray, prices_window: np.ndarray, num_mutation_seeds: int, mutation_rate: float, max_workers: int) -> Tuple[int, float, np.ndarray]:
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
            net = _calculate_net_profit_numba(mutated_actions.astype(np.int32), prices_window)
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
                if net > max_mutated_net: max_mutated_net = net; best_mutation_seed = seed
    if best_mutation_seed >= 0:
        mutation_rng = np.random.default_rng(best_mutation_seed)
        final_mutated_actions = original_actions.copy()
        mutation_mask = mutation_rng.random(window_len) < mutation_rate
        final_mutated_actions[mutation_mask] = 1 - final_mutated_actions[mutation_mask]
        final_mutated_actions[0] = 1
    else:
        best_mutation_seed = -1; max_mutated_net = -np.inf; final_mutated_actions = original_actions.copy()
    return best_mutation_seed, max_mutated_net, final_mutated_actions

def generate_actions_hybrid_multi_mutation(
    ticker_data: pd.DataFrame, window_size: int, num_seeds: int, max_workers: int, 
    mutation_rate_pct: float, num_mutations: int, progress_bar=None, ticker_name:str=""
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:

    prices = ticker_data['Close'].to_numpy()
    n = len(prices)
    final_actions = np.array([], dtype=np.int32)
    original_actions_full = np.array([], dtype=np.int32)
    window_details_list = []

    num_windows = (n + window_size - 1) // window_size
    mutation_rate = mutation_rate_pct / 100.0

    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        if len(prices_window) < 2: continue

        if progress_bar:
            progress_text = f"{ticker_name}: Window {i+1}/{num_windows}..."
            progress_bar.progress((i + 1) / num_windows, text=progress_text)

        dna_seed, current_best_net, current_best_actions = find_best_seed_for_window(prices_window, num_seeds, max_workers)
        original_actions_window = current_best_actions.copy()
        original_net_for_display = current_best_net
        successful_mutation_seeds = []

        for mutation_round in range(num_mutations):
            mutation_seed, mutated_net, mutated_actions = find_best_mutation_for_sequence(
                current_best_actions, prices_window, num_seeds, mutation_rate, max_workers
            )
            if mutated_net > current_best_net:
                current_best_net = mutated_net
                current_best_actions = mutated_actions
                successful_mutation_seeds.append(int(mutation_seed))

        final_actions = np.concatenate((final_actions, current_best_actions.astype(np.int32)))
        original_actions_full = np.concatenate((original_actions_full, original_actions_window.astype(np.int32)))

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

    return original_actions_full, final_actions, pd.DataFrame(window_details_list)

# ==============================================================================
# 4. Simulation Tracer Class
# ==============================================================================
class SimulationTracer:
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
        if len(decoded_numbers) < 3: raise ValueError("ข้อมูลในสตริงไม่ครบถ้วน")
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
        return (f"✅ Decoded: Len={self.action_length}, Rate={self.mutation_rate}%, DNA={self.dna_seed}, Mut={self.mutation_seeds}")

    @staticmethod
    def encode(action_length: int, mutation_rate: int, dna_seed: int, mutation_seeds: List[int]) -> str:
        all_numbers = [action_length, mutation_rate, dna_seed] + mutation_seeds
        encoded_parts = [f"{len(str(num))}{num}" for num in all_numbers]
        return "".join(encoded_parts)

# ==============================================================================
# 5. UI Rendering & Logic Functions
# ==============================================================================

# ! GOAL Step 3: Optimization & Refactoring
# Helper Function เพื่อใช้ซ้ำทั้งใน UI รายตัว และใน Batch Export
def generate_encoded_dna_for_ticker(
    df_windows: pd.DataFrame, 
    data_len: int, 
    target_win_num: int, 
    global_act_len: int, 
    mutation_rate: float, 
    window_size: int
) -> Tuple[str, int, int]:
    """
    Generate Encoded String logic extracted for reuse.
    Returns: (Encoded String, Actual Window Used, Final Action Length)
    """
    max_win = len(df_windows)
    use_win_num = target_win_num if target_win_num <= max_win else max_win
    
    start_idx = (use_win_num - 1) * window_size
    remaining = data_len - start_idx
    real_len = min(window_size, remaining)
    
    final_act_len = global_act_len if global_act_len > 0 else real_len
    
    row = df_windows.iloc[use_win_num - 1]
    dna_seed = int(row['dna_seed'])
    mut_seeds_str = row['mutation_seeds']
    
    mut_seeds = []
    if mut_seeds_str not in ["None", "[]"]:
        clean = mut_seeds_str.strip('[]')
        if clean: mut_seeds = [int(s.strip()) for s in clean.split(',')]
            
    encoded_string = SimulationTracer.encode(
        action_length=int(final_act_len),
        mutation_rate=int(mutation_rate),
        dna_seed=dna_seed,
        mutation_seeds=mut_seeds
    )
    
    return encoded_string, use_win_num, int(final_act_len)


def display_comparison_charts(results: Dict[str, pd.DataFrame], chart_title: str = '📊 เปรียบเทียบกำไรสุทธิ (Net Profit)'):
    valid_dfs = {name: df for name, df in results.items() if not df.empty and 'net' in df.columns}
    if not valid_dfs: st.warning("ไม่มีข้อมูลสำหรับสร้างกราฟ"); return
    try: longest_index = max((df.index for df in valid_dfs.values()), key=len, default=None)
    except ValueError: longest_index = None
    if longest_index is None: return
    chart_data = pd.DataFrame(index=longest_index)
    for name, df in valid_dfs.items(): chart_data[name] = df['net'].reindex(longest_index).ffill()
    st.write(chart_title); st.line_chart(chart_data)

def render_settings_tab():
    st.write("⚙️ **การตั้งค่าพารามิเตอร์**")
    config = load_config()
    asset_list = config.get('assets', [])

    # Default Select ALL Tickers
    st.session_state.selected_tickers = st.multiselect(
        "เลือก Tickers ที่ต้องการทดสอบ (เลือกได้หลายตัว)", 
        options=asset_list, 
        default=st.session_state.selected_tickers 
    )

    c1, c2 = st.columns(2)
    st.session_state.window_size = c1.number_input("ขนาด Window (วัน)", min_value=2, value=st.session_state.window_size)
    st.session_state.num_seeds = c2.number_input("จำนวน Seeds (DNA)", min_value=100, value=st.session_state.num_seeds, format="%d")

    c1, c2 = st.columns(2)
    st.session_state.start_date = c1.date_input("วันที่เริ่มต้น", value=st.session_state.start_date)
    st.session_state.end_date = c2.date_input("วันที่สิ้นสุด", value=st.session_state.end_date)

    st.divider()
    st.subheader("Parameters for Hybrid Strategy")
    c1, c2, c3 = st.columns(3)
    st.session_state.max_workers = c1.number_input("Max Workers", min_value=1, max_value=16, value=st.session_state.max_workers)
    st.session_state.mutation_rate = c2.slider("Mutation Rate (%)", 0.0, 50.0, st.session_state.mutation_rate, 0.5)
    st.session_state.num_mutations = c3.number_input("Mutation Rounds", min_value=0, max_value=10, value=st.session_state.num_mutations)

    # ! GOAL: Moved Inputs Here
    st.divider()
    with st.expander("🔌 **Global Encoding & Tracing Settings** (Advanced)", expanded=True):
        st.info("ตั้งค่า Default สำหรับการ Generate Encoded String ในหน้าผลลัพธ์")
        gc1, gc2 = st.columns(2)
        st.session_state.trace_target_window = gc1.number_input(
            "Target Window # (สำหรับ Encode)", 
            min_value=1, 
            value=st.session_state.trace_target_window,
            help="เลือก Window ลำดับที่ต้องการจะนำมาสร้าง Encoded String โดยอัตโนมัติในหน้าผลลัพธ์"
        )
        st.session_state.trace_action_length = gc2.number_input(
            "Action Length (0 = Auto/Window Size)", 
            min_value=0, 
            value=st.session_state.trace_action_length,
            help="กำหนดความยาว Action Sequence สำหรับ Encode (ใส่ 0 เพื่อใช้ความยาวจริงของ Window นั้นๆ)"
        )

def execute_batch_processing():
    tickers = st.session_state.selected_tickers
    if not tickers:
        st.error("กรุณาเลือก Ticker อย่างน้อย 1 ตัว"); return

    start_str = str(st.session_state.start_date)
    end_str = str(st.session_state.end_date)
    
    st.session_state.batch_results = {}
    
    overall_progress = st.progress(0, text="Starting Batch Process...")
    total_tickers = len(tickers)
    
    for idx, ticker in enumerate(tickers):
        ticker_data = get_ticker_data(ticker, start_str, end_str)
        if ticker_data.empty:
            st.warning(f"Skipping {ticker}: No Data Found.")
            continue
            
        original_actions, final_actions, df_windows = generate_actions_hybrid_multi_mutation(
            ticker_data, st.session_state.window_size, st.session_state.num_seeds,
            st.session_state.max_workers, st.session_state.mutation_rate,
            st.session_state.num_mutations, 
            progress_bar=overall_progress, 
            ticker_name=ticker
        )
        
        prices = ticker_data['Close'].to_numpy()
        sim_results = {
            Strategy.HYBRID_MULTI_MUTATION: run_simulation(prices.tolist(), final_actions.tolist()),
            Strategy.ORIGINAL_DNA: run_simulation(prices.tolist(), original_actions.tolist()),
            Strategy.REBALANCE_DAILY: run_simulation(prices.tolist(), generate_actions_rebalance_daily(len(prices)).tolist()),
            Strategy.PERFECT_FORESIGHT: run_simulation(prices.tolist(), generate_actions_perfect_foresight(prices.tolist()).tolist())
        }
        for name, df in sim_results.items():
            if not df.empty: df.index = ticker_data.index[:len(df)]
            
        st.session_state.batch_results[ticker] = {
            "sim_results": sim_results,
            "df_windows": df_windows,
            "data_len": len(ticker_data)
        }
        overall_progress.progress((idx + 1) / total_tickers, text=f"Completed {ticker} ({idx+1}/{total_tickers})")
        
    overall_progress.empty()
    st.success(f"✅ Processed {len(st.session_state.batch_results)} tickers successfully!")

def render_single_ticker_result(ticker: str, result_data: Dict[str, Any]):
    sim_results = result_data["sim_results"]
    df_windows = result_data["df_windows"]
    data_len = result_data["data_len"]
    
    chart_results = {k: v for k, v in sim_results.items() if k != Strategy.ORIGINAL_DNA}
    display_comparison_charts(chart_results, chart_title=f'📊 {ticker} - Net Profit Comparison')
    
    c1, c2, c3, c4 = st.columns(4)
    def get_final_net(strategy_name):
        df = sim_results.get(strategy_name)
        return df['net'].iloc[-1] if df is not None and not df.empty else 0.0

    c1.metric("Perfect Foresight", f"${get_final_net(Strategy.PERFECT_FORESIGHT):,.0f}")
    c2.metric("Hybrid Strategy", f"${get_final_net(Strategy.HYBRID_MULTI_MUTATION):,.0f}", delta_color="normal")
    c3.metric("Original DNA", f"${get_final_net(Strategy.ORIGINAL_DNA):,.0f}")
    c4.metric("Rebalance Daily", f"${get_final_net(Strategy.REBALANCE_DAILY):,.0f}")
    
    st.divider()
    st.write(f"📝 **Detailed Window Results ({ticker})**")
    with st.expander("Dataframe_Results"):
        st.dataframe(df_windows, use_container_width=True)
    
    # ! GOAL: Use Global Settings for Encoding (Refactored to use Helper)
    st.markdown(f"#### 🎁 Generate Encoded String for **{ticker}**")
    
    target_win_num = st.session_state.trace_target_window
    global_act_len = st.session_state.trace_action_length
    window_size = st.session_state.window_size
    
    # Calculate display parameters first for UI (logic duplication minimalized for display only)
    max_win = len(df_windows)
    use_win_num_disp = target_win_num if target_win_num <= max_win else max_win
    
    c_enc_1, c_enc_2 = st.columns([3, 1])
    with c_enc_1:
        st.info(f"Using Global Settings: **Window {use_win_num_disp}**, **Len {global_act_len if global_act_len>0 else 'Auto'}** (Rate: {st.session_state.mutation_rate}%)")
        if target_win_num > max_win:
             st.caption(f"⚠️ Window {target_win_num} เกินจำนวนที่มี (Max {max_win}). ใช้ Window {max_win} แทน")

    with c_enc_2:
        if st.button(f"Encode ({ticker})", key=f"btn_enc_{ticker}", use_container_width=True):
            try:
                # ! REFACTORED: Use helper function
                encoded, final_win, final_len = generate_encoded_dna_for_ticker(
                    df_windows, data_len, target_win_num, global_act_len, 
                    st.session_state.mutation_rate, window_size
                )
                
                st.success(f"Encoded String ({ticker} Win {final_win}):")
                st.code(encoded, language='text')
            except Exception as e:
                st.error(f"Encoding Error: {e}")

def render_methodology_expander():
    st.markdown(f"### 🧬 {Strategy.HYBRID_MULTI_MUTATION}")
    st.info("กลยุทธ์นี้ทำงานโดย: 1. ค้นหา 'DNA' ที่ดีที่สุดในแต่ละ Window 2. นำ DNA นั้นมาพยายาม 'กลายพันธุ์' (Mutate) ซ้ำๆ เพื่อหาผลลัพธ์ที่ดีกว่าเดิม")

    with st.expander("📖 คำอธิบายวิธีการทำงานและแนวคิด (Multi-Mutation)"):
        st.markdown(
            """
            แนวคิด **Hybrid (Multi-Mutation)** ได้รับแรงบันดาลใจจากกระบวนการ **วิวัฒนาการและการคัดเลือกสายพันธุ์ (Evolution & Selective Breeding)** โดยมีเป้าหมายเพื่อ "พัฒนา" รูปแบบการซื้อขาย (Actions) ที่ดีที่สุดให้ดียิ่งขึ้นไปอีกแบบซ้ำๆ ภายในแต่ละ Window

            แทนที่จะเปรียบเทียบระหว่าง "DNA ดั้งเดิม" กับ "การกลายพันธุ์แค่ครั้งเดียว" กลยุทธ์นี้จะนำผู้ชนะ (Champion) มาผ่านกระบวนการกลายพันธุ์ซ้ำๆ หลายรอบ เพื่อค้นหาการปรับปรุงที่ดีขึ้นเรื่อยๆ

            ---

            #### 🧬 กระบวนการทำงานในแต่ละ Window:

            1.  **เฟส 1: ค้นหา "แชมป์เปี้ยนตั้งต้น" (Initial Champion)**
                * โปรแกรมจะทำการสุ่ม Actions หรือ "DNA" ขึ้นมาตามจำนวน `num_seeds` ที่กำหนด
                * DNA ที่สร้างกำไร (Net Profit) ได้สูงสุด จะถูกคัดเลือกให้เป็น **"แชมป์เปี้ยนตัวแรก"**
                * `DNA_Original = argmax_{s in S_dna} [ Profit(Generate_DNA(s)) ]`

            2.  **เฟส 2: กระบวนการ "กลายพันธุ์ต่อเนื่อง" (Iterative Mutation)**
                * โปรแกรมจะเริ่มลูปการกลายพันธุ์ตามจำนวนรอบ (`num_mutations`) ที่กำหนด
                * **ในแต่ละรอบ:**
                    * **สร้างผู้ท้าชิง:** นำ Actions ของ **"แชมป์เปี้ยนปัจจุบัน"** มาเป็นต้นแบบ แล้วค้นหารูปแบบการกลายพันธุ์ (Mutation Pattern) ที่ดีที่สุดเพื่อสร้าง "ผู้ท้าชิง" (Challenger)
                    * `Challenger = argmax_{s_m in S_mutation} [ Profit(Mutate(Current_Champion, s_m)) ]`
                    * **คัดเลือกผู้ที่แข็งแกร่งที่สุด (Survival of the Fittest):** เปรียบเทียบกำไรระหว่าง "ผู้ท้าชิง" กับ "แชมป์เปี้ยนปัจจุบัน"
                        * **ถ้าผู้ท้าชิงชนะ:** ผู้ท้าชิงจะกลายเป็น **"แชมป์เปี้ยนคนใหม่"** และจะถูกนำไปใช้เป็นต้นแบบในรอบการกลายพันธุ์ถัดไป
                        * **ถ้าแชมป์เปี้ยนปัจจุบันชนะ:** แชมป์เปี้ยนจะยังคงตำแหน่งเดิม และถูกนำไปใช้เป็นต้นแบบในรอบถัดไป
            
            3.  **เฟส 3: ผลลัพธ์สุดท้าย**
                * หลังจากผ่านกระบวนการกลายพันธุ์ครบทุกรอบแล้ว **"แชมป์เปี้ยนตัวสุดท้าย"** ที่รอดมาได้ คือ Actions ที่จะถูกนำไปใช้สำหรับ Window นั้นจริงๆ

            ---
            
            #### ตัวอย่าง: (สมมติ `num_mutations = 2`)

            1.  **ค้นหา DNA ดั้งเดิม:** พบว่า Seed `5784` ให้กำไรดีที่สุด `Net Profit = $1,200`
                * **แชมป์เปี้ยนปัจจุบัน:** Actions จาก Seed `5784` (Profit: $1,200)

            2.  **Mutation รอบที่ 1:**
                * นำ Actions ของแชมป์เปี้ยน (Seed `5784`) ไปค้นหารูปแบบกลายพันธุ์ที่ดีที่สุด
                * พบว่า Mutation Seed `8871` สามารถพัฒนากำไรเป็น `$1,550` ได้
                * เนื่องจาก `$1,550 > $1,200` → ผู้ท้าชิงชนะ!
                * **แชมป์เปี้ยนคนใหม่:** Actions ที่กลายพันธุ์จาก Seed `8871` (Profit: $1,550)

            3.  **Mutation รอบที่ 2:**
                * นำ Actions ของแชมป์เปี้ยนคนใหม่ (ที่มาจาก Mutation Seed `8871`) ไปค้นหารูปแบบกลายพันธุ์ที่ดีที่สุดอีกครั้ง
                * พบว่า Mutation Seed `10524` สามารถพัฒนากำไรต่อได้เป็น `$1,620`
                * เนื่องจาก `$1,620 > $1,550` → ผู้ท้าชิงชนะอีกครั้ง!
                * **แชมป์เปี้ยนคนใหม่:** Actions ที่กลายพันธุ์จาก Seed `10524` (Profit: $1,620)

            4.  **จบกระบวนการ:** Actions สุดท้ายสำหรับ Window นี้คือ Actions ที่ให้กำไร `$1,620` ซึ่งเป็นผลลัพธ์จากการพัฒนาต่อยอดมา 2 รอบ

            แนวคิด **Hybrid (Multi-Mutation)** ได้รับแรงบันดาลใจจากกระบวนการ **วิวัฒนาการและการคัดเลือกสายพันธุ์ (Evolution & Selective Breeding)** โดยมีเป้าหมายเพื่อ "พัฒนา" รูปแบบการซื้อขาย (Actions) ที่ดีที่สุดให้ดียิ่งขึ้นไปอีกแบบซ้ำๆ ภายในแต่ละ Window

            ---

            ### 🔬 เจาะลึก Logic: หัวใจของกระบวนการกลายพันธุ์ (Mutation)
    
            กระบวนการกลายพันธุ์คือการนำรูปแบบการซื้อขาย (Actions) ของ **"แชมป์เปี้ยนปัจจุบัน"** มาทำการ **"ปรับปรุงเล็กน้อยอย่างสุ่ม"** เพื่อมองหาโอกาสที่จะพัฒนามันให้ดียิ่งขึ้นไปอีก เปรียบเสมือนการคัดเลือกสายพันธุ์เพื่อหาลักษณะเด่นที่ดีกว่าเดิม
    
            หัวใจสำคัญของกระบวนการนี้เกิดขึ้นภายในฟังก์ชัน `find_best_mutation_for_sequence` ซึ่งจะสร้าง "ผู้ท้าชิง" (Challenger) ขึ้นมาหลายพันราย โดยแต่ละรายจะถูกสร้างผ่าน 3 ขั้นตอนหลักดังนี้:
    
            ---
    
            #### ขั้นตอนที่ 1: 📜 สร้าง "แผนผังการกลายพันธุ์" (Mutation Blueprint)
    
            ในขั้นตอนนี้ โปรแกรมจะใช้ `mutation_seed` ที่ไม่ซ้ำกันเพื่อสร้าง "พิมพ์เขียว" ที่กำหนดว่ายีน (Action) ในวันไหนควรจะเปลี่ยนแปลง
    
            ```python
            # สร้างอาเรย์ของเลขสุ่ม (0.0 - 1.0) ตาม seed ที่กำหนด
            # แล้วเปรียบเทียบกับ mutation_rate เพื่อสร้าง "แผนผัง"
            mutation_mask = mutation_rng.random(window_len) < mutation_rate
            ```
    
            * **`mutation_rng.random(window_len)`**: สร้างชุดตัวเลขสุ่มขึ้นมา 1 ตัวต่อ 1 วันใน Window การใช้ `seed` ที่ต่างกันจะให้ชุดตัวเลขสุ่มที่ต่างกัน
            * **`< mutation_rate`**: นำตัวเลขสุ่มแต่ละตัวมาเทียบกับอัตราการกลายพันธุ์ (เช่น 5% หรือ 0.05)
                * ถ้าน้อยกว่า ➡️ `True` (ตำแหน่งนี้จะเกิดการเปลี่ยนแปลง)
                * ถ้ามากกว่า ➡️ `False` (ตำแหน่งนี้จะคงเดิม)
            * **ผลลัพธ์**: คือ "แผนผัง" ที่เป็น `True` / `False` ซึ่งเป็นเหมือนพิมพ์เขียวสำหรับการเปลี่ยนแปลงในขั้นตอนต่อไป
    
            **ตัวอย่าง:**
            * `original_actions`: `[1, 0, 1, 1]`
            * `mutation_rate`: 50% (0.5)
            * `เลขสุ่มที่สร้างได้`: `[0.23, 0.81, 0.99, 0.45]`
            * **แผนผัง (`mutation_mask`)**: `[True, False, False, True]`
    
            ---
    
            #### ขั้นตอนที่ 2: 🧬 ดำเนินการเปลี่ยนแปลงตามแผนผัง (Flipping the Genes)
    
            โปรแกรมจะนำ Actions ของแชมป์เปี้ยนมาคัดลอก แล้ว "พลิกค่า" เฉพาะในตำแหน่งที่แผนผังเป็น `True`
    
            ```python
            # คัดลอก Actions เดิมมา
            mutated_actions = original_actions.copy()
            
            # ใช้แผนผัง (mask) เพื่อเลือกตำแหน่งที่จะ "พลิกค่า"
            mutated_actions[mutation_mask] = 1 - mutated_actions[mutation_mask]
            ```
    
            * `1 - action` เป็นเทคนิคที่รวดเร็วในการพลิกค่า:
                * ถ้า Action เดิมเป็น `1` (ซื้อ) ➡️ `1 - 1` จะได้ `0` (ถือ)
                * ถ้า Action เดิมเป็น `0` (ถือ) ➡️ `1 - 0` จะได้ `1` (ซื้อ)
    
            **ตัวอย่าง (ต่อ):**
            * **ต้นฉบับ**: `[1, 0, 1, 1]`
            * **แผนผัง**: `[T, F, F, T]` (เปลี่ยนแปลงตำแหน่งที่ 0 และ 3)
            * **ตำแหน่ง 0**: `1` พลิกเป็น `0`
            * **ตำแหน่ง 3**: `1` พลิกเป็น `0`
            * **ผลลัพธ์หลังการพลิกยีน**: `[0, 0, 1, 0]`
    
            ---
    
            #### ขั้นตอนที่ 3: 🛡️ บังคับใช้กฎเหล็ก (The First-Day Rule)
    
            เพื่อรับประกันว่าทุกการจำลองจะเริ่มต้นด้วยการซื้อเสมอ โปรแกรมจะบังคับให้ Action ของวันแรกสุดเป็น `1` เสมอ ไม่ว่าการกลายพันธุ์จะให้ผลเป็นอย่างไรก็ตาม
    
            ```python
            # ไม่ว่าผลจะเป็นอย่างไร วันแรกต้องเป็น 1 เสมอ
            mutated_actions[0] = 1
            ```
    
            **ตัวอย่าง (สุดท้าย):**
            * **ผลลัพธ์จากการพลิกยีน**: `[0, 0, 1, 0]`
            * **บังคับกฎข้อแรก**: `[1, 0, 1, 0]`
    
            > ✨ **ผลลัพธ์สุดท้าย** คือ Actions ของ "ผู้ท้าชิง" หนึ่งราย ที่พร้อมจะถูกนำไปประเมินผลกำไรเพื่อท้าชิงตำแหน่งแชมป์เปี้ยนต่อไป กระบวนการทั้งหมดนี้จะเกิดขึ้นซ้ำๆ หลายพันครั้งเพื่อค้นหารูปแบบการกลายพันธุ์ที่ดีที่สุดเพียงหนึ่งเดียวในแต่ละรอบ
            """
        )
        
        code = """ ตัวอย่าง code
        import numpy as np
        dna_rng = np.random.default_rng(seed=239)
        current_actions = dna_rng.integers(0, 2, size=30)
        default_actions = current_actions.copy() 
        
        mutation_seeds = [30]
        #รอบที่ for loop
        m_seed = 30
        mutation_rng = np.random.default_rng(seed=30)
        mutation_mask = mutation_rng.random(30) < 0.10 # Mutation Rate 10(%)
        [0.72..., 0.39..., 0.03..., 0.58..., 0.41..., ...]
        [False False  True False False False False False False False False False
        False False False False False False False False False False  True False
        False False False False False False]
        
        current_actions[mutation_mask] = 1 - current_actions[mutation_mask] # Flipping the Genes
        current_actions[0] = 1
        default_actions[0] = 1
        
        print( "mutation_mask" , mutation_mask)
        print( "default_actions" , default_actions)
        print( "current_actions" , current_actions)
        """
        st.code(code, language="python")

def render_simulation_tabs():
    st.write("---")
    render_methodology_expander()
    
    st.divider()
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f"### 🚀 Batch Runner")
        st.caption("เลือก Tickers ในหน้า Settings แล้วกดปุ่มด้านขวาเพื่อเริ่มคำนวณ")
    with c2:
        if st.button("🚀 Start One-Click Loop All", type="primary", use_container_width=True):
            execute_batch_processing()
            
    # ! GOAL Step 1: Export JSON Feature
    if st.session_state.batch_results:
        st.write("---")
        exp_c1, exp_c2 = st.columns([3, 1])
        with exp_c1:
            st.markdown("#### 💾 Export Encoded Strings")
            st.caption(f"Settings: Window {st.session_state.trace_target_window}, Len {st.session_state.trace_action_length}, Rate {st.session_state.mutation_rate}%")
        
        with exp_c2:
            # Prepare data for JSON export
            export_payload = {
                "metadata": {
                    "trace_target_window": st.session_state.trace_target_window,
                    "trace_action_length": st.session_state.trace_action_length,
                    "mutation_rate_percent": st.session_state.mutation_rate,
                    "exported_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                },
                "tickers": {}
            }
            
            # Loop through all results and generate strings
            for ticker, data in st.session_state.batch_results.items():
                try:
                    encoded, final_win, final_len = generate_encoded_dna_for_ticker(
                        data["df_windows"], 
                        data["data_len"], 
                        st.session_state.trace_target_window, 
                        st.session_state.trace_action_length, 
                        st.session_state.mutation_rate, 
                        st.session_state.window_size
                    )
                    export_payload["tickers"][ticker] = encoded
                except Exception as e:
                    export_payload["tickers"][ticker] = f"Error: {str(e)}"

            json_str = json.dumps(export_payload, indent=4)
            st.download_button(
                label="💾 Download JSON (All Tickers)",
                data=json_str,
                file_name="encoded_strings.json",
                mime="application/json",
                use_container_width=True
            )

        tickers = list(st.session_state.batch_results.keys())
        st.write(f"✅ Results available for: {', '.join(tickers)}")
        tabs = st.tabs([f"📈 {t}" for t in tickers])
        for tab, ticker in zip(tabs, tickers):
            with tab:
                render_single_ticker_result(ticker, st.session_state.batch_results[ticker])
    else:
        st.info("ยังไม่มีผลลัพธ์ กรุณากดปุ่ม Start")

# ! GOAL: Restore Full Tracer (Decode + Encode)
def render_tracer_tab():
    st.markdown("### 🔍 Action Sequence Tracer & Encoder")
    st.info("เครื่องมือนี้ใช้สำหรับ 1. **ถอดรหัส (Decode)** String เพื่อจำลองผลลัพธ์ และ 2. **เข้ารหัส (Encode)** พารามิเตอร์เพื่อสร้าง String")

    st.markdown("---")
    st.markdown("#### 1. ถอดรหัส (Decode) String")

    encoded_string = st.text_input(
        "ป้อน Encoded String ที่นี่:",
        "26021034252903219354832053493",
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
    config = load_config()
    initialize_session_state(config)

    tab_list = ["⚙️ Settings", "🧬 Simulation Results", "🔍 Tracer"]
    tabs = st.tabs(tab_list)

    with tabs[0]:
        render_settings_tab()
    with tabs[1]:
        render_simulation_tabs()
    with tabs[2]:
        render_tracer_tab()

if __name__ == "__main__":
    main()
