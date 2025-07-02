# final_workflow.py
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import json
import thingspeak
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Any, Optional

# ! NUMBA: Import Numba's Just-In-Time compiler for core acceleration
from numba import njit, prange

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================
st.set_page_config(page_title="Closed-Loop Hybrid Backtester", page_icon="🔄", layout="wide")
CONFIG_FILE_PATH = "add_gen_config.json"

def load_config(filepath: str = CONFIG_FILE_PATH) -> List[Dict[str, Any]]:
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
    if 'results' not in st.session_state:
        st.session_state.results = {}

# ==============================================================================
# 2. Core Calculation & Data Functions (MODIFIED)
# ==============================================================================
@st.cache_data(ttl=3600)
def get_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        data = yf.Ticker(ticker).history(start=start_date, end=end_date)[['Close']]
        if data.empty or data['Close'].isnull().any() or (data['Close'] <= 0).any():
            st.warning(f"[{ticker}] ข้อมูลราคาไม่สมบูรณ์ มีค่า null หรือ 0")
            return pd.DataFrame()
        if data.index.tz is None:
            data = data.tz_localize('UTC').tz_convert('Asia/Bangkok')
        else:
            data = data.tz_convert('Asia/Bangkok')
        return data
    except Exception as e:
        st.error(f"❌ [{ticker}] ไม่สามารถดึงข้อมูลได้: {e}")
        return pd.DataFrame()

@njit(cache=True, fastmath=True)
def _calculate_net_profit_numba(action_array: np.ndarray, price_array: np.ndarray, fix: int = 1500) -> float:
    n = len(action_array)
    if n == 0 or len(price_array) == 0 or n > len(price_array):
        return -np.inf
    # --- ป้องกัน Numba Error ---
    if np.any(price_array <= 0):
        return -np.inf

    action_array_calc = action_array.copy()
    action_array_calc[0] = 1
    initial_price = price_array[0]
    initial_capital = float(fix * 2.0)
    refer_net = -float(fix) * np.log(initial_price / price_array[n-1])
    cash = float(fix)
    amount = float(fix) / initial_price
    for i in prange(1, n):
        curr_price = price_array[i]
        if action_array_calc[i] != 0:
            cash += amount * curr_price - fix
            amount = fix / curr_price
    final_sumusd = cash + (amount * price_array[n-1])
    net = final_sumusd - refer_net - initial_capital
    return net

@njit(fastmath=True)
def _full_sim_numba(action_arr: np.ndarray, price_arr: np.ndarray, fix_val: int):
    n = len(action_arr)
    # --- ป้องกัน Numba Error ---
    if n == 0 or np.any(price_arr <= 0):
        empty_arr = np.empty(0, dtype=np.float64)
        return empty_arr, empty_arr, empty_arr, empty_arr, empty_arr, empty_arr

    action_calc = action_arr.copy()
    action_calc[0] = 1
    amount = np.empty(n, dtype=np.float64)
    buffer = np.zeros(n, dtype=np.float64)
    cash = np.empty(n, dtype=np.float64)
    asset_val = np.empty(n, dtype=np.float64)
    sumusd_val = np.empty(n, dtype=np.float64)
    init_price = price_arr[0]
    amount[0] = float(fix_val) / init_price
    cash[0] = float(fix_val)
    asset_val[0] = amount[0] * init_price
    sumusd_val[0] = cash[0] + asset_val[0]
    refer = -float(fix_val) * np.log(init_price / price_arr[:n])
    for i in prange(1, n):
        curr_price = price_arr[i]
        if action_calc[i] == 0:
            amount[i] = amount[i-1]
            buffer[i] = 0.0
        else:
            amount[i] = fix_val / curr_price
            buffer[i] = amount[i-1] * curr_price - fix_val
        cash[i] = cash[i-1] + buffer[i]
        asset_val[i] = amount[i] * curr_price
        sumusd_val[i] = cash[i] + asset_val[i]
    return buffer, sumusd_val, cash, asset_val, amount, refer

def run_simulation(prices: List[float], actions: List[int], fix: int = 1500) -> pd.DataFrame:
    min_len = min(len(prices), len(actions))
    prices_arr = np.array(prices[:min_len], dtype=np.float64)
    actions_arr = np.array(actions[:min_len], dtype=np.int32)
    
    buffer, sumusd, cash, asset_value, amount, refer = _full_sim_numba(actions_arr, prices_arr, fix)
    
    if len(sumusd) == 0:
        return pd.DataFrame()
        
    initial_capital = sumusd[0]
    net_profit = sumusd - refer - initial_capital
    return pd.DataFrame({'price': prices_arr, 'action': actions_arr, 'net': np.round(net_profit, 2)})

def find_best_seed_for_window(prices_window: np.ndarray, num_seeds: int, max_workers: int) -> np.ndarray:
    if len(prices_window) < 2: return np.ones(len(prices_window), dtype=int)
    best_actions = np.ones(len(prices_window), dtype=int)
    max_net = -np.inf
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(
            _calculate_net_profit_numba, 
            np.random.default_rng(s).integers(0, 2, size=len(prices_window)), 
            prices_window
        ): s for s in range(num_seeds)}
        
        for future in as_completed(futures):
            net = future.result()
            if net > max_net:
                max_net = net
                seed = futures[future]
                best_actions = np.random.default_rng(seed).integers(0, 2, size=len(prices_window))
                best_actions[0] = 1
    return best_actions
    
def find_best_mutation_for_sequence(original_actions, prices_window, num_seeds, mutation_rate, max_workers):
    current_best_actions = original_actions.copy()
    max_net = _calculate_net_profit_numba(original_actions, prices_window)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for s in range(num_seeds):
            rng = np.random.default_rng(s)
            mutated_actions = original_actions.copy()
            mutation_mask = rng.random(len(mutated_actions)) < mutation_rate
            mutated_actions[mutation_mask] = 1 - mutated_actions[mutation_mask]
            mutated_actions[0] = 1
            futures[executor.submit(_calculate_net_profit_numba, mutated_actions, prices_window)] = mutated_actions
        
        for future in as_completed(futures):
            net = future.result()
            if net > max_net:
                max_net = net
                current_best_actions = futures[future]
    return current_best_actions

def generate_hybrid_actions(ticker_data, params, status_placeholder):
    prices = ticker_data['Close'].to_numpy()
    n = len(prices)
    final_actions = np.array([], dtype=int)
    mutation_rate = params['mutation_rate'] / 100.0
    num_windows = (n + params['window_size'] - 1) // params['window_size']
    
    for i in range(0, n, params['window_size']):
        window_num = (i // params['window_size']) + 1
        end_index = min(i + params['window_size'], n)
        prices_window = prices[i:end_index]
        if len(prices_window) < 2: continue
        
        status_placeholder.text(f"Window {window_num}/{num_windows}: Finding best DNA...")
        current_actions = find_best_seed_for_window(prices_window, params['num_seeds'], params['max_workers'])
        
        for mut_round in range(params['num_mutations']):
            status_placeholder.text(f"Window {window_num}/{num_windows}: Mutation round {mut_round + 1}/{params['num_mutations']}...")
            current_actions = find_best_mutation_for_sequence(current_actions, prices_window, params['num_seeds'], mutation_rate, params['max_workers'])
        
        final_actions = np.concatenate((final_actions, current_actions))
    return final_actions

# ==============================================================================
# 3. ThingSpeak & Workflow Logic (MODIFIED)
# ==============================================================================
def read_from_thingspeak(channel: thingspeak.Channel, field_id: int) -> Optional[int]:
    try:
        # --- แก้ไขปัญหา ThingSpeak TypeError ---
        response_str = channel.get({'results': 1})
        
        # แปลง string ที่ได้เป็น dictionary
        response_data = json.loads(response_str)

        if response_data and 'feeds' in response_data and response_data['feeds']:
            field_value_str = response_data['feeds'][0].get(f'field{field_id}')
            if field_value_str is not None:
                return int(float(field_value_str))
        return None
    except json.JSONDecodeError:
        st.error("เกิดข้อผิดพลาดในการแปลงข้อมูล JSON จาก ThingSpeak")
        return None
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการอ่านข้อมูลจาก ThingSpeak: {e}")
        return None

def run_workflow(asset_config: Dict[str, Any], params: Dict[str, Any]):
    ticker = asset_config['ticker']
    field_id = asset_config['thingspeak_field']
    
    st.info(f"🚀 เริ่มกระบวนการสำหรับ **{ticker}**...")
    status_container = st.container(border=True)
    
    with st.spinner(f"[{ticker}] กำลังเชื่อมต่อกับ ThingSpeak..."):
        try:
            write_channel = thingspeak.Channel(id=asset_config['channel_id'], api_key=asset_config['write_api_key'])
            read_key = asset_config.get('read_api_key')
            read_channel = thingspeak.Channel(id=asset_config['channel_id'], api_key=read_key)
        except Exception as e:
            st.error(f"[{ticker}] ไม่สามารถสร้างการเชื่อมต่อ ThingSpeak ได้: {e}")
            return

    with st.spinner(f"[{ticker}] กำลังอ่านค่าล่าสุดจาก Field {field_id}..."):
        current_api_value = read_from_thingspeak(read_channel, field_id)
        if current_api_value is not None:
            status_container.metric("📊 ค่าปัจจุบันบน ThingSpeak", f"{current_api_value:,}")
        else:
            status_container.warning(f"ไม่พบค่าบน ThingSpeak Field {field_id} หรืออ่านไม่ได้")

    with st.spinner(f"[{ticker}] กำลังดึงข้อมูลราคา..."):
        ticker_data = get_ticker_data(ticker, str(params['start_date']), str(params['end_date']))
    
    if ticker_data.empty:
        st.error(f"[{ticker}] ไม่สามารถดึงข้อมูลราคาได้, หยุดกระบวนการ")
        return

    window_status_placeholder = st.empty()
    with st.spinner(f"[{ticker}] กำลังรัน Backtest... (อาจใช้เวลานาน)"):
        final_actions = generate_hybrid_actions(ticker_data, params, window_status_placeholder)
        sim_df = run_simulation(ticker_data['Close'].tolist(), final_actions.tolist())
    
    window_status_placeholder.empty()

    if sim_df.empty or 'net' not in sim_df or sim_df['net'].empty:
        st.error(f"[{ticker}] การคำนวณล้มเหลว, ไม่ได้ผลลัพธ์ Net Profit")
        return

    newly_calculated_value = int(sim_df['net'].iloc[-1])
    status_container.metric("💻 ค่าที่คำนวณใหม่", f"{newly_calculated_value:,}")

    st.session_state.results[ticker] = {'sim_df': sim_df, 'ticker_data': ticker_data}
    
    if newly_calculated_value == current_api_value:
        st.success(f"✅ [{ticker}] ค่าที่คำนวณใหม่ ({newly_calculated_value:,}) ตรงกับค่าบน ThingSpeak ไม่จำเป็นต้องอัปเดต")
    else:
        st.info(f"[{ticker}] พบค่าใหม่! ({newly_calculated_value:,}) กำลังอัปเดต ThingSpeak...")
        with st.spinner(f"[{ticker}] Sending update to Field {field_id}..."):
            try:
                write_channel.update({f'field{field_id}': newly_calculated_value})
                st.success(f"🎉 [{ticker}] อัปเดต ThingSpeak สำเร็จ!")
            except Exception as e:
                st.error(f"[{ticker}] การอัปเดต ThingSpeak ล้มเหลว: {e}")

# ==============================================================================
# 4. Streamlit UI
# ==============================================================================
def render_results_display(ticker: str):
    results = st.session_state.results.get(ticker)
    if not results: return
    
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
    initialize_session_state()
    asset_configs = load_config()

    if not asset_configs:
        st.warning(f"กรุณาสร้างและตั้งค่าไฟล์ `{CONFIG_FILE_PATH}` ให้ถูกต้อง")
        return

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
                params['window_size'] = st.number_input("ขนาด Window (วัน)", 2, value=30, key=f"win_{ticker}")
                c1, c2 = st.columns(2)
                params['num_seeds'] = c1.number_input("จำนวน Seeds", 100, value=1000, format="%d", key=f"seeds_{ticker}")
                params['max_workers'] = c2.number_input("Workers", 1, 16, value=8, key=f"work_{ticker}")
                c1, c2 = st.columns(2)
                params['mutation_rate'] = c1.slider("อัตรากลายพันธุ์ (%)", 0.0, 50.0, 10.0, 0.5, key=f"rate_{ticker}")
                params['num_mutations'] = c2.number_input("รอบการกลายพันธุ์", 0, 10, 5, key=f"mut_{ticker}")
            
            if st.button(f"รัน Workflow สำหรับ {ticker}", type="primary", use_container_width=True):
                 if params['start_date'] >= params['end_date']:
                     st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
                 else:
                     run_workflow(asset_config, params)
            render_results_display(ticker)

if __name__ == "__main__":
    main()
