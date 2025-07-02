# main_workflow.py
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import json
import thingspeak
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

# ! NUMBA: Import Numba's Just-In-Time compiler for core acceleration
from numba import njit

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================
st.set_page_config(page_title="Closed-Loop Hybrid Backtester", page_icon="🔄", layout="wide")
CONFIG_FILE_PATH = "add_gen_config.json" # <--- ตั้งชื่อไฟล์ที่นี่

def load_config(filepath: str = CONFIG_FILE_PATH) -> List[Dict[str, Any]]:
    """Loads asset configurations from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"❌ ไม่พบไฟล์ตั้งค่า '{filepath}'")
        return []
    except json.JSONDecodeError:
        st.error(f"❌ รูปแบบ JSON ในไฟล์ '{filepath}' ไม่ถูกต้อง")
        return []

def initialize_session_state():
    """Initializes session state with default parameters."""
    if 'results' not in st.session_state:
        st.session_state.results = {}

# ==============================================================================
# 2. Core Calculation & Data Functions (No changes needed)
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
        st.error(f"❌ [{ticker}] ไม่สามารถดึงข้อมูลได้: {e}")
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
        n = len(action_arr)
        if n == 0: return (np.empty(0, dtype=np.float64),) * 6
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

    min_len = min(len(prices), len(actions))
    prices_arr = np.array(prices[:min_len], dtype=np.float64)
    actions_arr = np.array(actions[:min_len], dtype=np.int32)
    buffer, sumusd, cash, asset_value, amount, refer = _full_sim_numba(actions_arr, prices_arr, fix)
    if len(sumusd) == 0: return pd.DataFrame()
    initial_capital = sumusd[0]
    return pd.DataFrame({
        'price': prices_arr, 'action': actions_arr, 'net': np.round(sumusd - refer - initial_capital, 2)
    })

def find_best_seed_for_window(prices_window: np.ndarray, num_seeds_to_try: int, max_workers: int) -> np.ndarray:
    window_len = len(prices_window)
    if window_len < 2: return np.ones(window_len, dtype=int)
    
    def evaluate_seed(seed):
        rng = np.random.default_rng(seed)
        actions = rng.integers(0, 2, size=window_len)
        net = _calculate_net_profit_numba(actions, prices_window)
        return seed, net

    best_seed, max_net = -1, -np.inf
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_seed, s) for s in range(num_seeds_to_try)]
        for future in as_completed(futures):
            seed, final_net = future.result()
            if final_net > max_net:
                max_net, best_seed = final_net, seed

    if best_seed != -1:
        rng_best = np.random.default_rng(best_seed)
        best_actions = rng_best.integers(0, 2, size=window_len)
        best_actions[0] = 1
        return best_actions
    return np.ones(window_len, dtype=int)

def find_best_mutation_for_sequence(original_actions: np.ndarray, prices_window: np.ndarray, num_mutation_seeds: int, mutation_rate: float, max_workers: int) -> np.ndarray:
    original_net = _calculate_net_profit_numba(original_actions, prices_window)
    max_mutated_net = original_net
    best_mutated_actions = original_actions.copy()

    def evaluate_mutation(seed):
        mutation_rng = np.random.default_rng(seed)
        mutated_actions = original_actions.copy()
        mutation_mask = mutation_rng.random(len(mutated_actions)) < mutation_rate
        mutated_actions[mutation_mask] = 1 - mutated_actions[mutation_mask]
        mutated_actions[0] = 1
        net = _calculate_net_profit_numba(mutated_actions, prices_window)
        return net, mutated_actions

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_mutation, s) for s in range(num_mutation_seeds)]
        for future in as_completed(futures):
            net, actions = future.result()
            if net > max_mutated_net:
                max_mutated_net = net
                best_mutated_actions = actions
    
    return best_mutated_actions

def generate_actions_hybrid_multi_mutation(ticker_data: pd.DataFrame, params: Dict[str, Any], status_placeholder: st.container) -> np.ndarray:
    prices = ticker_data['Close'].to_numpy()
    n = len(prices)
    final_actions = np.array([], dtype=int)
    mutation_rate = params['mutation_rate'] / 100.0

    num_windows = (n + params['window_size'] - 1) // params['window_size']
    for i, start_index in enumerate(range(0, n, params['window_size'])):
        end_index = min(start_index + params['window_size'], n)
        prices_window = prices[start_index:end_index]
        if len(prices_window) < 2: continue

        status_placeholder.text(f"Window {i+1}/{num_windows}: Searching for Best DNA...")
        current_best_actions = find_best_seed_for_window(prices_window, params['num_seeds'], params['max_workers'])

        for mutation_round in range(params['num_mutations']):
            status_placeholder.text(f"Window {i+1}/{num_windows}: Mutation Round {mutation_round+1}/{params['num_mutations']}...")
            mutated_actions = find_best_mutation_for_sequence(
                current_best_actions, prices_window, params['num_seeds'], mutation_rate, params['max_workers']
            )
            current_best_actions = mutated_actions

        final_actions = np.concatenate((final_actions, current_best_actions))
    
    return final_actions

# ==============================================================================
# 3. ThingSpeak & Workflow Logic
# ==============================================================================
def read_from_thingspeak(channel: thingspeak.Channel, field_id: int) -> Optional[int]:
    """Reads the last value from a specific field in a ThingSpeak channel."""
    try:
        last_entry = channel.get({'results': 1})
        if last_entry and 'feeds' in last_entry and last_entry['feeds']:
            field_value_str = last_entry['feeds'][0].get(f'field{field_id}')
            if field_value_str is not None:
                return int(float(field_value_str))
        return None
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการอ่านข้อมูลจาก ThingSpeak: {e}")
        return None

def run_backtest_and_update_workflow(asset_config: Dict[str, Any], params: Dict[str, Any]):
    """The main logic loop for a single asset."""
    ticker = asset_config['ticker']
    # <--- ปรับให้ตรงกับ key ใน JSON
    field_id = asset_config['thingspeak_field'] 
    
    st.info(f"🚀 เริ่มกระบวนการสำหรับ **{ticker}**...")
    status_container = st.container(border=True)
    
    # --- Connect to ThingSpeak ---
    with st.spinner(f"[{ticker}] กำลังเชื่อมต่อกับ ThingSpeak..."):
        try:
            # ใช้ `read_api_key` ถ้ามี, ถ้าไม่มีจะใช้ `write_api_key` สำหรับการอ่าน (สำหรับ Public Channel)
            read_key = asset_config.get('read_api_key') 
            channel = thingspeak.Channel(id=asset_config['channel_id'], write_key=asset_config['write_api_key'], read_key=read_key)
        except Exception as e:
            st.error(f"[{ticker}] ไม่สามารถสร้างการเชื่อมต่อ ThingSpeak ได้: {e}")
            return

    # --- 1. Read from API ---
    with st.spinner(f"[{ticker}] กำลังอ่านค่าล่าสุดจาก Field {field_id}..."):
        current_api_value = read_from_thingspeak(channel, field_id)
        if current_api_value is not None:
            status_container.metric("📊 ค่าปัจจุบันบน ThingSpeak", f"{current_api_value:,}")
        else:
            status_container.warning(f"ไม่พบค่าบน ThingSpeak Field {field_id} หรืออ่านไม่ได้")

    # --- 2. Calculate new value ---
    with st.spinner(f"[{ticker}] กำลังดึงข้อมูลราคา..."):
        ticker_data = get_ticker_data(ticker, str(params['start_date']), str(params['end_date']))
    
    if ticker_data.empty:
        st.error(f"[{ticker}] ไม่สามารถดึงข้อมูลราคาได้, หยุดกระบวนการ")
        return

    window_status_placeholder = st.empty()
    with st.spinner(f"[{ticker}] กำลังรัน Backtest... (อาจใช้เวลานาน)"):
        final_actions = generate_actions_hybrid_multi_mutation(ticker_data, params, window_status_placeholder)
        sim_df = run_simulation(ticker_data['Close'].tolist(), final_actions.tolist())
    
    window_status_placeholder.empty()

    if sim_df.empty or 'net' not in sim_df or sim_df['net'].empty:
        st.error(f"[{ticker}] การคำนวณล้มเหลว, ไม่ได้ผลลัพธ์ Net Profit")
        return

    newly_calculated_value = int(sim_df['net'].iloc[-1])
    status_container.metric("💻 ค่าที่คำนวณใหม่", f"{newly_calculated_value:,}")

    # Store results for display
    st.session_state.results[ticker] = {'sim_df': sim_df, 'ticker_data': ticker_data}
    
    # --- 3. Compare and Update ---
    if newly_calculated_value == current_api_value:
        st.success(f"✅ [{ticker}] ค่าที่คำนวณใหม่ ({newly_calculated_value:,}) ตรงกับค่าบน ThingSpeak ไม่จำเป็นต้องอัปเดต")
    else:
        st.info(f"[{ticker}] พบค่าใหม่! ({newly_calculated_value:,}) กำลังอัปเดต ThingSpeak...")
        with st.spinner(f"[{ticker}] Sending update to Field {field_id}..."):
            try:
                channel.update({f'field{field_id}': newly_calculated_value})
                st.success(f"🎉 [{ticker}] อัปเดต ThingSpeak สำเร็จ!")
            except Exception as e:
                st.error(f"[{ticker}] การอัปเดต ThingSpeak ล้มเหลว: {e}")

# ==============================================================================
# 4. Streamlit UI
# ==============================================================================
def render_results_display(ticker: str):
    """Displays the backtest results for a given ticker."""
    results = st.session_state.results.get(ticker)
    if not results:
        return
        
    st.write("---")
    st.subheader(f"ผลการ Backtest สำหรับ {ticker}")
    
    sim_df = results['sim_df']
    ticker_data = results['ticker_data']
    
    if len(sim_df) <= len(ticker_data):
        sim_df.index = ticker_data.index[:len(sim_df)]
    
    st.write("📈 กราฟกำไรสุทธิ (Net Profit)")
    st.line_chart(sim_df[['net']])

    with st.expander("ดูข้อมูลการจำลองโดยละเอียด"):
        st.dataframe(sim_df, use_container_width=True)

def main():
    st.title("🔄 Closed-Loop Hybrid Backtester & Updater")
    st.caption("รัน Backtest และอัปเดต ThingSpeak อัตโนมัติเมื่อพบผลลัพธ์ที่ดีกว่าเดิม")

    initialize_session_state()
    asset_configs = load_config()

    if not asset_configs:
        st.warning(f"กรุณาสร้างและตั้งค่าไฟล์ `{CONFIG_FILE_PATH}` ให้ถูกต้อง")
        return

    # Create Tabs from config
    tab_names = [config.get('tab_name', config['ticker']) for config in asset_configs]
    tabs = st.tabs(tab_names)

    for i, tab in enumerate(tabs):
        with tab:
            asset_config = asset_configs[i]
            ticker = asset_config['ticker']

            st.header(f"Asset: {ticker}")
            st.json(asset_config, expanded=False)

            with st.expander("⚙️ ปรับพารามิเตอร์การทดสอบ", expanded=True):
                params = {}
                c1, c2 = st.columns(2)
                params['start_date'] = c1.date_input("วันที่เริ่มต้น", datetime(2024, 1, 1).date(), key=f"start_{ticker}")
                params['end_date'] = c2.date_input("วันที่สิ้นสุด", datetime.now().date(), key=f"end_{ticker}")
                
                params['window_size'] = st.number_input("ขนาด Window (วัน)", min_value=2, value=30, key=f"win_{ticker}")
                
                c1, c2 = st.columns(2)
                params['num_seeds'] = c1.number_input("จำนวน Seeds", min_value=100, value=1000, format="%d", key=f"seeds_{ticker}")
                params['max_workers'] = c2.number_input("จำนวน Workers (CPU Cores)", min_value=1, max_value=16, value=8, key=f"work_{ticker}")

                c1, c2 = st.columns(2)
                params['mutation_rate'] = c1.slider("อัตราการกลายพันธุ์ (%)", 0.0, 50.0, 10.0, 0.5, key=f"rate_{ticker}")
                params['num_mutations'] = c2.number_input("จำนวนรอบการกลายพันธุ์", 0, 10, 5, key=f"mut_{ticker}")
            
            if st.button(f"รัน Backtest และ Update สำหรับ {ticker}", type="primary", use_container_width=True):
                 if params['start_date'] >= params['end_date']:
                     st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
                 else:
                     run_backtest_and_update_workflow(asset_config, params)

            # Display results if they exist in session state
            render_results_display(ticker)

if __name__ == "__main__":
    main()
