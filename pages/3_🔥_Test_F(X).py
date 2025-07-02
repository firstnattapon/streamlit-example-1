# main
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import json
import time
import thingspeak
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any

# ! NUMBA: Import Numba's Just-In-Time compiler for core acceleration
from numba import njit

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================
st.set_page_config(page_title="Automated Strategy Lab", page_icon="🤖", layout="wide")

class Strategy:
    """คลาสสำหรับเก็บชื่อกลยุทธ์ต่างๆ เพื่อให้เรียกใช้ง่ายและลดข้อผิดพลาด"""
    REBALANCE_DAILY = "Rebalance Daily"
    PERFECT_FORESIGHT = "Perfect Foresight (Max)"
    HYBRID_MULTI_MUTATION = "Hybrid (Multi-Mutation)"
    ORIGINAL_DNA = "Original DNA (Pre-Mutation)"

def load_config(filename="strategy_config.json"):
    """Loads all configurations from a single JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error: Configuration file '{filename}' not found.")
        return None
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from '{filename}'. Please check its format.")
        return None

def initialize_session_state(config: Dict[str, Any]):
    # Uses the 'default_settings' section from the loaded config
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
# 2. Core Calculation & Data Functions (Unchanged)
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

# ==============================================================================
# 3. Strategy Action Generation (Unchanged)
# ==============================================================================
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
    original_actions: np.ndarray, prices_window: np.ndarray, num_mutation_seeds: int,
    mutation_rate: float, max_workers: int
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
                if net > max_mutated_net: max_mutated_net, best_mutation_seed = net, seed

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
    mutation_rate_pct: float, num_mutations: int, progress_bar=None
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:

    prices = ticker_data['Close'].to_numpy()
    n = len(prices)
    final_actions = np.array([], dtype=int)
    original_actions_full = np.array([], dtype=int)
    window_details_list = []
    num_windows = (n + window_size - 1) // window_size
    mutation_rate = mutation_rate_pct / 100.0

    for i, start_index in enumerate(range(0, n, window_size)):
        if progress_bar: progress_bar.progress(i / num_windows, text=f"Processing window {i+1}/{num_windows}...")
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        if len(prices_window) < 2: continue

        dna_seed, current_best_net, current_best_actions = find_best_seed_for_window(prices_window, num_seeds, max_workers)
        original_actions_window = current_best_actions.copy()
        original_net_for_display = current_best_net
        successful_mutation_seeds = []

        for mutation_round in range(num_mutations):
            mutation_seed, mutated_net, mutated_actions = find_best_mutation_for_sequence(
                current_best_actions, prices_window, num_seeds, mutation_rate, max_workers
            )
            if mutated_net > current_best_net:
                current_best_net, current_best_actions = mutated_net, mutated_actions
                successful_mutation_seeds.append(int(mutation_seed))

        final_actions = np.concatenate((final_actions, current_best_actions))
        original_actions_full = np.concatenate((original_actions_full, original_actions_window))
        start_date_win, end_date_win = ticker_data.index[start_index], ticker_data.index[end_index-1]
        detail = {
            'window': i + 1, 'timeline': f"{start_date_win.strftime('%Y-%m-%d')} to {end_date_win.strftime('%Y-%m-%d')}",
            'dna_seed': dna_seed, 'mutation_seeds': str(successful_mutation_seeds) if successful_mutation_seeds else "None",
            'improvements': len(successful_mutation_seeds), 'original_net': round(original_net_for_display, 2),
            'final_net': round(current_best_net, 2)
        }
        window_details_list.append(detail)
    if progress_bar: progress_bar.empty()
    return original_actions_full, final_actions, pd.DataFrame(window_details_list)


# ==============================================================================
# 4. Simulation Tracer & Encoder Class (Unchanged)
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
        self.action_length, self.mutation_rate, self.dna_seed = decoded_numbers[0], decoded_numbers[1], decoded_numbers[2]
        self.mutation_seeds: List[int] = decoded_numbers[3:]
        self.mutation_rate_float: float = self.mutation_rate / 100.0

    def run(self) -> np.ndarray:
        dna_rng = np.random.default_rng(seed=self.dna_seed)
        current_actions = dna_rng.integers(0, 2, size=self.action_length); current_actions[0] = 1
        for m_seed in self.mutation_seeds:
            mutation_rng = np.random.default_rng(seed=m_seed)
            mutation_mask = mutation_rng.random(self.action_length) < self.mutation_rate_float
            current_actions[mutation_mask] = 1 - current_actions[mutation_mask]; current_actions[0] = 1
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
# 5. UI Rendering Functions
# ==============================================================================
def render_settings_tab(config: Dict[str, Any]):
    st.write("⚙️ **การตั้งค่าพารามิเตอร์**")
    
    # Get the asset list for the dropdown from the unified config
    asset_configs = config.get('automation_assets', [])
    asset_list = [asset['ticker'] for asset in asset_configs] if asset_configs else ['FFWM']

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
    st.session_state.num_seeds = c1.number_input("จำนวน Seeds (สำหรับค้นหา DNA และ Mutation)", min_value=100, value=st.session_state.num_seeds, format="%d")
    st.session_state.max_workers = c2.number_input("จำนวน Workers (CPU Cores)", min_value=1, max_value=16, value=st.session_state.max_workers)

    c1, c2 = st.columns(2)
    st.session_state.mutation_rate = c1.slider("อัตราการกลายพันธุ์ (Mutation Rate) %", min_value=0.0, max_value=50.0, value=st.session_state.mutation_rate, step=0.5)
    st.session_state.num_mutations = c2.number_input("จำนวนรอบการกลายพันธุ์ (Multi-Mutation)", min_value=0, max_value=10, value=st.session_state.num_mutations, help="จำนวนครั้งที่จะพยายามพัฒนายีนส์ต่อจากตัวที่ดีที่สุดในแต่ละ Window")


def render_hybrid_multi_mutation_tab():
    st.markdown(f"### 🧬 {Strategy.HYBRID_MULTI_MUTATION}")
    st.info("กลยุทธ์นี้ทำงานโดย: 1. ค้นหา 'DNA' ที่ดีที่สุดในแต่ละ Window 2. นำ DNA นั้นมาพยายาม 'กลายพันธุ์' (Mutate) ซ้ำๆ เพื่อหาผลลัพธ์ที่ดีกว่าเดิม")

    if st.button(f"🚀 Start Hybrid Multi-Mutation", type="primary"):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้อง"); return
        
        ticker = st.session_state.test_ticker
        with st.spinner(f"กำลังดึงข้อมูลและประมวลผลสำหรับ {ticker}..."):
            ticker_data = get_ticker_data(ticker, str(st.session_state.start_date), str(st.session_state.end_date))
            if ticker_data.empty: st.error("ไม่พบข้อมูลสำหรับ Ticker และช่วงวันที่ที่เลือก"); return

            progress_bar = st.progress(0, text="Initializing Hybrid Multi-Mutation Search...")
            _, final_actions, df_windows = generate_actions_hybrid_multi_mutation(
                ticker_data, st.session_state.window_size, st.session_state.num_seeds,
                st.session_state.max_workers, st.session_state.mutation_rate,
                st.session_state.num_mutations, progress_bar
            )
            st.session_state.simulation_results = {
                 Strategy.HYBRID_MULTI_MUTATION: run_simulation(ticker_data['Close'].tolist(), final_actions.tolist()),
            }
            st.session_state.df_windows_details = df_windows
            st.success("Test complete!")
            # Full display logic (charts, tables) can be added here as in your original file.

def render_tracer_tab():
    # This function is unchanged
    st.markdown("### 🔍 Action Sequence Tracer & Encoder")
    encoded_string = st.text_input("ป้อน Encoded String ที่นี่:", "2602104900", key="decoder_input")
    if st.button("Trace & Simulate", type="primary", key="tracer_button"):
        if encoded_string:
            try:
                tracer = SimulationTracer(encoded_string=encoded_string)
                st.success("ถอดรหัสสำเร็จ!"); st.code(str(tracer), language='bash')
                final_actions = tracer.run()
                st.markdown("#### 🎉 ผลลัพธ์ Action Sequence สุดท้าย:")
                st.dataframe(pd.DataFrame(final_actions, columns=['Action']), use_container_width=True)
            except ValueError as e: st.error(f"❌ เกิดข้อผิดพลาดในการประมวลผล: {e}")

def render_automation_hub_tab(config: Dict[str, Any]):
    st.markdown("### 🤖 Automation Hub")
    st.info("ส่วนนี้จะทำการค้นหากลยุทธ์ที่ดีที่สุดสำหรับ Ticker แต่ละตัวจากไฟล์ `config.json` โดยอัตโนมัติ และอัปเดตไปยัง ThingSpeak หากพบกลยุทธ์ใหม่")

    automation_configs = config.get('automation_assets')
    if not automation_configs:
        st.error("ไม่พบส่วน `automation_assets` ใน `config.json`")
        return

    st.markdown("#### Automation Settings")
    lookback_days = st.number_input("Lookback Period (days)", min_value=30, max_value=365, value=90,
                                    help="จำนวนวันที่ต้องการดึงข้อมูลย้อนหลังเพื่อคำนวณกลยุทธ์")

    if st.button("🚀 Run Automation for All Tickers", type="primary"):
        run_automation_workflow(automation_configs, lookback_days)

def run_automation_workflow(asset_configs: List[Dict], lookback_days: int):
    st.write("---"); st.header("Automation Run Log")

    # Get shared parameters from the settings tab
    window_size, num_seeds, max_workers, mutation_rate_pct, num_mutations = (
        st.session_state.window_size, st.session_state.num_seeds, st.session_state.max_workers,
        st.session_state.mutation_rate, st.session_state.num_mutations
    )
    end_date, start_date = datetime.now(), datetime.now() - timedelta(days=lookback_days)

    for asset in asset_configs:
        ticker = asset['ticker']
        with st.status(f"Processing {ticker}...", expanded=True) as status:
            try:
                # 1. Get Data
                st.write(f"Fetching data for {ticker}...")
                ticker_data = get_ticker_data(ticker, start_date, end_date)
                if ticker_data.empty:
                    st.warning(f"No data for {ticker}. Skipping."); status.update(label=f"⚠️ {ticker}: No data", state="complete", expanded=False); continue

                # 2. Generate Strategy
                st.write(f"Running Hybrid Multi-Mutation strategy...")
                _, _, df_windows = generate_actions_hybrid_multi_mutation(
                    ticker_data, window_size, num_seeds, max_workers, mutation_rate_pct, num_mutations
                )
                if df_windows.empty:
                    st.warning("Strategy generation failed. Skipping."); status.update(label=f"⚠️ {ticker}: No strategy", state="complete", expanded=False); continue

                # 3. Extract and Encode latest window
                latest_window_data = df_windows.iloc[-1]
                dna_seed = int(latest_window_data['dna_seed'])
                mutation_seeds_str = latest_window_data['mutation_seeds']
                mutation_seeds = [int(s.strip()) for s in mutation_seeds_str.strip('[]').split(',') if s.strip().isdigit()]
                
                action_length = len(ticker_data) % window_size or window_size

                encoded_string = SimulationTracer.encode(
                    action_length=action_length, mutation_rate=int(mutation_rate_pct),
                    dna_seed=dna_seed, mutation_seeds=mutation_seeds
                )
                st.write(f"Generated encoded string: `{encoded_string}`")

                # 4. Connect to ThingSpeak and compare
                channel_id, field_num, write_key, read_key = (
                    asset['channel_id'], asset['thingspeak_field'], asset['write_api_key'], asset.get('read_api_key', '')
                )
                
                # FIX: Initialize with write_key as 'api_key'
                client = thingspeak.Channel(id=channel_id, api_key=write_key)
                
                # FIX: Pass read_key in 'get' options if it exists
                get_options = {'field': field_num, 'results': 1}
                if read_key: get_options['api_key'] = read_key

                last_entry_json = client.get(options=get_options)
                last_value = last_entry_json['feeds'][0].get(f'field{field_num}') if last_entry_json.get('feeds') else None
                
                st.write(f"Last value on ThingSpeak: `{last_value}`")

                # 5. Update if different
                if str(encoded_string) != str(last_value):
                    st.write(f"New strategy found. Updating ThingSpeak...")
                    client.update({f'field{field_num}': encoded_string})
                    status.update(label=f"✅ {ticker}: Updated", state="complete", expanded=False)
                    st.success(f"Successfully updated {ticker}!")
                    time.sleep(16) # IMPORTANT: Respect ThingSpeak API rate limit
                else:
                    st.write("Strategy is up-to-date."); status.update(label=f"☑️ {ticker}: No changes", state="complete", expanded=False)

            except Exception as e:
                st.error(f"An error occurred while processing {ticker}: {e}")
                status.update(label=f"❌ {ticker}: Error", state="error", expanded=True)

# ==============================================================================
# 6. Main Application
# ==============================================================================
def main():
    st.markdown("### 🤖 Automated Strategy Lab")
    st.caption("A unified tool for strategy generation, backtesting, and automated deployment to ThingSpeak.")

    config = load_config()
    if config is None:
        st.stop() # Halt execution if config file is invalid

    initialize_session_state(config)

    tab_list = ["⚙️ Settings", f"🧬 {Strategy.HYBRID_MULTI_MUTATION}", "🔍 Tracer", "🤖 Automation Hub"]
    tabs = st.tabs(tab_list)

    with tabs[0]:
        render_settings_tab(config)
    with tabs[1]:
        render_hybrid_multi_mutation_tab()
    with tabs[2]:
        render_tracer_tab()
    with tabs[3]:
        render_automation_hub_tab(config)

if __name__ == "__main__":
    main()
