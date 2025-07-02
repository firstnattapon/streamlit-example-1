import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any
import json
import time
import thingspeak
from numba import njit

# ==============================================================================
# 0. Configuration and Setup
# ==============================================================================
st.set_page_config(page_title="Strategy Automation", page_icon="🤖", layout="wide")

def load_config(filepath: str) -> Dict[str, Any]:
    """Loads configuration from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.error(f"Error loading config file '{filepath}': {e}")
        return {}

# ==============================================================================
# 1. Core Logic (Copied from main.py, mostly unchanged)
# These functions are the engine of the strategy simulation.
# ==============================================================================

@st.cache_data(ttl=3600)
def get_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        data = yf.Ticker(ticker).history(start=start_date, end=end_date)[['Close']]
        if data.empty: return pd.DataFrame()
        return data
    except Exception as e:
        st.error(f"❌ Could not fetch data for {ticker}: {str(e)}"); return pd.DataFrame()

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
    else: best_mutation_seed, max_mutated_net, final_mutated_actions = -1, -np.inf, original_actions.copy()
    return best_mutation_seed, max_mutated_net, final_mutated_actions

def generate_actions_hybrid_multi_mutation(ticker_data: pd.DataFrame, window_size: int, num_seeds: int, max_workers: int, mutation_rate_pct: float, num_mutations: int, progress_bar) -> pd.DataFrame:
    prices = ticker_data['Close'].to_numpy()
    n = len(prices)
    window_details_list = []
    num_windows = (n + window_size - 1) // window_size
    mutation_rate = mutation_rate_pct / 100.0

    for i, start_index in enumerate(range(0, n, window_size)):
        progress_total_steps = num_mutations + 1
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        if len(prices_window) < 2: continue
        
        progress_bar.progress((i * progress_total_steps) / (num_windows * progress_total_steps), text=f"Window {i+1}/{num_windows} - Phase 1: Finding Best DNA...")
        dna_seed, current_best_net, current_best_actions = find_best_seed_for_window(prices_window, num_seeds, max_workers)
        
        original_net_for_display = current_best_net
        successful_mutation_seeds = []
        for mutation_round in range(num_mutations):
            progress_bar.progress((i * progress_total_steps + 1 + mutation_round) / (num_windows * progress_total_steps), text=f"Window {i+1}/{num_windows} - Mutation {mutation_round+1}/{num_mutations}...")
            mutation_seed, mutated_net, mutated_actions = find_best_mutation_for_sequence(current_best_actions, prices_window, num_seeds, mutation_rate, max_workers)
            if mutated_net > current_best_net:
                current_best_net, current_best_actions = mutated_net, mutated_actions
                successful_mutation_seeds.append(int(mutation_seed))

        detail = {
            'window': i + 1, 'action_length': len(prices_window),
            'dna_seed': dna_seed, 'mutation_seeds': str(successful_mutation_seeds) if successful_mutation_seeds else "None",
            'original_net': round(original_net_for_display, 2), 'final_net': round(current_best_net, 2)
        }
        window_details_list.append(detail)
    return pd.DataFrame(window_details_list)

class SimulationTracer:
    @staticmethod
    def encode(action_length: int, mutation_rate: int, dna_seed: int, mutation_seeds: List[int]) -> str:
        all_numbers = [action_length, mutation_rate, dna_seed] + mutation_seeds
        encoded_parts = [f"{len(str(num))}{num}" for num in all_numbers]
        return "".join(encoded_parts)

# ==============================================================================
# 2. ThingSpeak Handlers
# ==============================================================================

def read_from_thingspeak(channel_id: int, read_api_key: str, field_num: int) -> str | None:
    """Reads the latest value from a specific ThingSpeak field."""
    try:
        channel = thingspeak.Channel(id=channel_id, api_key=read_api_key)
        latest_feed = channel.get_latest_feed()
        field_key = f'field{field_num}'
        if latest_feed and field_key in latest_feed:
            return latest_feed[field_key]
        return None
    except Exception as e:
        st.warning(f"Could not read from ThingSpeak Channel {channel_id}, Field {field_num}: {e}")
        return None

def write_to_thingspeak(channel_id: int, write_api_key: str, field_num: int, value: str) -> bool:
    """Writes a value to a specific ThingSpeak field."""
    try:
        channel = thingspeak.Channel(id=channel_id, write_key=write_api_key)
        response = channel.update({f'field{field_num}': value})
        if response and int(response) > 0:
            return True
        return False
    except Exception as e:
        st.error(f"Could not write to ThingSpeak Channel {channel_id}, Field {field_num}: {e}")
        return False

# ==============================================================================
# 3. Main Automation Process
# ==============================================================================

def run_automation_process(config: Dict[str, Any]):
    """The main function to orchestrate the automation loop."""
    settings = config.get('settings', {})
    assets = config.get('assets', [])

    if not settings or not assets:
        st.error("Configuration is missing 'settings' or 'assets'.")
        return

    results_container = st.container(border=True)
    results_container.header("Automation Log")

    for asset in assets:
        ticker = asset.get('ticker')
        ts_field = asset.get('thingspeak_field')
        ts_channel_id = asset.get('channel_id')
        ts_read_key = asset.get('read_api_key')
        ts_write_key = asset.get('write_api_key')

        if not all([ticker, ts_field, ts_channel_id, ts_read_key, ts_write_key]):
            results_container.error(f"Skipping asset due to incomplete configuration: {asset}")
            continue

        with results_container.status(f"Processing {ticker}...", expanded=True) as status:
            try:
                # 1. Get Data
                st.write(f"Fetching data for {ticker}...")
                end_date = datetime.now()
                start_date = end_date - timedelta(days=settings.get('start_date_period_days', 180))
                ticker_data = get_ticker_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                if ticker_data.empty:
                    status.update(label=f"⚠️ No data for {ticker}", state="error")
                    continue

                # 2. Run Hybrid Simulation
                st.write(f"Running Hybrid Multi-Mutation simulation for {ticker}...")
                progress_bar = st.progress(0)
                df_windows = generate_actions_hybrid_multi_mutation(
                    ticker_data,
                    settings['window_size'], settings['num_seeds'],
                    settings['max_workers'], settings['mutation_rate'],
                    settings['num_mutations'], progress_bar
                )
                progress_bar.empty()
                
                if df_windows.empty:
                    status.update(label=f"⚠️ Simulation failed for {ticker}", state="error")
                    continue

                # 3. Get Result for the LATEST window
                latest_window_data = df_windows.iloc[-1]
                st.write("Latest window result:")
                st.dataframe(latest_window_data.to_frame().T)

                # 4. Encode the result
                action_length = int(latest_window_data['action_length'])
                mutation_rate = int(settings['mutation_rate']) # Rate as int for encoding
                dna_seed = int(latest_window_data['dna_seed'])
                
                mutation_seeds_str = latest_window_data['mutation_seeds']
                mutation_seeds = []
                if mutation_seeds_str not in ["None", "[]"]:
                    mutation_seeds = [int(s.strip()) for s in mutation_seeds_str.strip('[]').split(',')]

                new_encoded_string = SimulationTracer.encode(
                    action_length=action_length,
                    mutation_rate=mutation_rate,
                    dna_seed=dna_seed,
                    mutation_seeds=mutation_seeds
                )
                st.write(f"**New Encoded String:** `{new_encoded_string}`")

                # 5. Compare with ThingSpeak
                st.write("Comparing with ThingSpeak...")
                last_thingspeak_value = read_from_thingspeak(ts_channel_id, ts_read_key, ts_field)
                st.write(f"**Last ThingSpeak Value:** `{last_thingspeak_value}`")

                # 6. Update if necessary
                if str(new_encoded_string) != str(last_thingspeak_value):
                    st.write("❗️ Values are different. Updating ThingSpeak...")
                    success = write_to_thingspeak(ts_channel_id, ts_write_key, ts_field, new_encoded_string)
                    if success:
                        status.update(label=f"✅ Successfully updated {ticker}!", state="complete", expanded=False)
                    else:
                        status.update(label=f"❌ Failed to update {ticker}", state="error")
                else:
                    status.update(label=f"👍 {ticker} is already up-to-date.", state="complete", expanded=False)

                # ThingSpeak has a rate limit of ~15 seconds for updates on the free plan
                st.write("Waiting 16 seconds to respect ThingSpeak API rate limit...")
                time.sleep(16)

            except Exception as e:
                status.update(label=f"❌ An error occurred processing {ticker}: {e}", state="error")


# ==============================================================================
# 4. Streamlit User Interface
# ==============================================================================

def main():
    st.title("🤖 Hybrid Strategy Automation Runner")
    st.caption("This tool automates the process of finding and updating the best strategy on ThingSpeak.")

    config_filepath = "automation_config.json"
    config = load_config(config_filepath)

    if not config:
        st.stop()

    st.subheader("Loaded Configuration")
    with st.expander("Show Configuration Details"):
        st.json(config)
        
    if st.button("🚀 Run Automation for All Assets", type="primary"):
        run_automation_process(config)

if __name__ == "__main__":
    main()
