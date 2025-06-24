# Final 
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import math
import json
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from datetime import datetime
from typing import List, Tuple, Dict, Any

# ! NUMBA: Import Numba's Just-In-Time compiler for core acceleration
from numba import njit

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================

class Strategy:
    """คลาสสำหรับเก็บชื่อกลยุทธ์ต่างๆ เพื่อให้เรียกใช้ง่ายและลดข้อผิดพลาด"""
    REBALANCE_DAILY = "Rebalance Daily"
    PERFECT_FORESIGHT = "Perfect Foresight (Max)"
    # ! NEW: กลยุทธ์ใหม่ตามที่ร้องขอ
    LFSR_SEARCH = "LFSR (Characteristic Poly)"
    
    # --- กลยุทธ์เก่า (เก็บไว้ในโค้ดแต่ไม่แสดงใน UI) ---
    SLIDING_WINDOW = "Best Seed Sliding Window"
    CHAOTIC_SLIDING_WINDOW = "Chaotic Seed Sliding Window"
    GENETIC_ALGORITHM = "Genetic Algorithm Sliding Window"
    ARITHMETIC_SEQUENCE = "Arithmetic Sequence"
    GEOMETRIC_SEQUENCE = "Geometric Sequence"


def load_config(filepath: str = "dynamic_seed_config.json") -> Dict[str, Any]:
    """
    โหลดการตั้งค่าจากไฟล์ JSON
    หากไฟล์ไม่พบหรือมีข้อผิดพลาด จะคืนค่า default เพื่อให้โปรแกรมทำงานต่อได้
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        st.warning(f"⚠️ ไม่พบหรือไฟล์ '{filepath}' ไม่ถูกต้อง จะใช้ค่าเริ่มต้นแทน")
        return {
            "assets": ["FFWM", "NEGG", "RIVN", "BTC-USD", "NVDA", "TSLA", "META"],
            "default_settings": {
                "selected_ticker": "FFWM", "start_date": "2024-01-01", "window_size": 30,
                "max_workers": 8, "num_samples": 5000, "master_seed": 42
            },
            "manual_seed_by_asset": {
                "default": [{'seed': 999, 'size': 50, 'tail': 15}],
                "FFWM": [{'seed': 1234, 'size': 60, 'tail': 30}, {'seed': 7777, 'size': 30, 'tail': 10}]
            }
        }

def on_ticker_change_callback(config: Dict[str, Any]):
    """
    Callback ที่จะถูกเรียกเมื่อ Ticker ใน Tab Manual Seed เปลี่ยน
    เพื่อโหลดค่า Preset ของ Seed/Size/Tail ให้ตรงกับ Ticker ที่เลือก
    """
    selected_ticker = st.session_state.get("manual_ticker_key")
    if not selected_ticker:
        return
    presets_by_asset = config.get("manual_seed_by_asset", {})
    default_presets = presets_by_asset.get("default", [{'seed': 999, 'size': 50, 'tail': 15}])
    st.session_state.manual_seed_lines = presets_by_asset.get(selected_ticker, default_presets)

def initialize_session_state(config: Dict[str, Any]):
    """
    ตั้งค่าเริ่มต้นสำหรับ Streamlit session state โดยใช้ค่าจาก config
    เพื่อให้ค่าต่างๆ ยังคงอยู่เมื่อผู้ใช้เปลี่ยนหน้าหรือทำ Action อื่นๆ
    """
    defaults = config.get('default_settings', {})

    if 'test_ticker' not in st.session_state:
        st.session_state.test_ticker = defaults.get('selected_ticker', 'FFWM')
    if 'start_date' not in st.session_state:
        try:
            st.session_state.start_date = datetime.strptime(defaults.get('start_date', '2024-01-01'), '%Y-%m-%d').date()
        except ValueError:
            st.session_state.start_date = datetime(2024, 1, 1).date()
    if 'end_date' not in st.session_state:
        st.session_state.end_date = datetime.now().date()
    if 'window_size' not in st.session_state:
        st.session_state.window_size = defaults.get('window_size', 30)
    if 'max_workers' not in st.session_state:
        st.session_state.max_workers = defaults.get('max_workers', 8)
    if 'num_samples' not in st.session_state:
        st.session_state.num_samples = defaults.get('num_samples', 5000)
    if 'master_seed' not in st.session_state:
        st.session_state.master_seed = defaults.get('master_seed', 42)

    if 'df_for_analysis' not in st.session_state:
        st.session_state.df_for_analysis = None

    if 'manual_seed_lines' not in st.session_state:
        initial_ticker = st.session_state.get('test_ticker', defaults.get('selected_ticker', 'FFWM'))
        presets_by_asset = config.get("manual_seed_by_asset", {})
        default_presets = presets_by_asset.get("default", [{'seed': 999, 'size': 50, 'tail': 15}])
        st.session_state.manual_seed_lines = presets_by_asset.get(initial_ticker, default_presets)


# ==============================================================================
# 2. Core Calculation & Data Functions (With Numba Acceleration)
# ==============================================================================
@st.cache_data(ttl=3600)
def get_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    ดึงข้อมูลราคาหุ้น/สินทรัพย์จาก Yahoo Finance และ Cache ผลลัพธ์ไว้ 1 ชั่วโมง
    """
    try:
        data = yf.Ticker(ticker).history(start=start_date, end=end_date)[['Close']]
        if data.empty: return pd.DataFrame()
        # Standardize timezone to avoid issues
        if data.index.tz is None:
            data = data.tz_localize('UTC').tz_convert('Asia/Bangkok')
        else:
            data = data.tz_convert('Asia/Bangkok')
        return data
    except Exception as e:
        st.error(f"❌ ไม่สามารถดึงข้อมูล {ticker} ได้: {str(e)}")
        return pd.DataFrame()

@njit(cache=True)
def _calculate_simulation_numba(action_array: np.ndarray, price_array: np.ndarray, fix: int = 1500) -> Tuple:
    """
    คำนวณผลลัพธ์การจำลองการเทรด (หัวใจของการคำนวณ)
    - ใช้ Numba @njit(cache=True) เพื่อคอมไพล์เป็น Machine Code ทำให้ทำงานเร็วมาก
    - รับ Input เป็น NumPy arrays เพื่อประสิทธิภาพสูงสุด
    - คืนค่าเป็น Tuple ของ NumPy arrays
    """
    n = len(action_array)
    if n == 0 or len(price_array) == 0:
        empty_arr = np.empty(0, dtype=np.float64)
        return (empty_arr, empty_arr, empty_arr, empty_arr, empty_arr, empty_arr)

    action_array_calc = action_array.copy()
    if n > 0: action_array_calc[0] = 1

    amount = np.empty(n, dtype=np.float64)
    buffer = np.zeros(n, dtype=np.float64)
    cash = np.empty(n, dtype=np.float64)
    asset_value = np.empty(n, dtype=np.float64)
    sumusd = np.empty(n, dtype=np.float64)

    initial_price = price_array[0]
    amount[0] = fix / initial_price
    cash[0] = fix
    asset_value[0] = amount[0] * initial_price
    sumusd[0] = cash[0] + asset_value[0]

    refer = -fix * np.log(initial_price / price_array)

    for i in range(1, n):
        curr_price = price_array[i]
        if action_array_calc[i] == 0:
            amount[i] = amount[i-1]
            buffer[i] = 0.0
        else:
            amount[i] = fix / curr_price
            buffer[i] = amount[i-1] * curr_price - fix

        cash[i] = cash[i-1] + buffer[i]
        asset_value[i] = amount[i] * curr_price
        sumusd[i] = cash[i] + asset_value[i]

    return buffer, sumusd, cash, asset_value, amount, refer

@lru_cache(maxsize=16384)
def calculate_optimized_cached(action_tuple: Tuple[int, ...], price_tuple: Tuple[float, ...], fix: int = 1500) -> Tuple:
    """
    Wrapper function สำหรับเรียกฟังก์ชัน Numba โดยใช้ Cache
    """
    action_array = np.asarray(action_tuple, dtype=np.int32)
    price_array = np.asarray(price_tuple, dtype=np.float64)
    return _calculate_simulation_numba(action_array, price_array, fix)

def run_simulation(prices: List[float], actions: List[int], fix: int = 1500) -> pd.DataFrame:
    """
    สร้าง DataFrame ผลลัพธ์จากการจำลองการเทรด
    """
    if not prices or not actions: return pd.DataFrame()
    min_len = min(len(prices), len(actions))
    prices, actions = prices[:min_len], actions[:min_len]
    
    buffer, sumusd, cash, asset_value, amount, refer = calculate_optimized_cached(tuple(actions), tuple(prices), fix)
    
    if len(sumusd) == 0: return pd.DataFrame()

    initial_capital = sumusd[0]
    return pd.DataFrame({
        'price': prices, 'action': actions, 'buffer': np.round(buffer, 2),
        'sumusd': np.round(sumusd, 2), 'cash': np.round(cash, 2),
        'asset_value': np.round(asset_value, 2), 'amount': np.round(amount, 2),
        'refer': np.round(refer + initial_capital, 2),
        'net': np.round(sumusd - refer - initial_capital, 2)
    })

# ==============================================================================
# 3. Strategy Action Generation
# ==============================================================================

# 3.1 Benchmark & Basic Strategies
def generate_actions_rebalance_daily(num_days: int) -> np.ndarray:
    """สร้าง Action สำหรับกลยุทธ์ Rebalance ทุกวัน (Min Performance)"""
    return np.ones(num_days, dtype=np.int32)

def generate_actions_perfect_foresight(prices: List[float], fix: int = 1500) -> np.ndarray:
    """สร้าง Action สำหรับกลยุทธ์ Perfect Foresight (Max Performance) โดยใช้ Dynamic Programming"""
    price_arr = np.asarray(prices, dtype=np.float64)
    n = len(price_arr)
    if n < 2: return np.ones(n, dtype=int)
    
    dp = np.zeros(n, dtype=np.float64); path = np.zeros(n, dtype=int); dp[0] = float(fix * 2)
    for i in range(1, n):
        j_indices = np.arange(i)
        profits = fix * ((price_arr[i] / price_arr[j_indices]) - 1)
        current_sumusd = dp[j_indices] + profits
        best_idx = np.argmax(current_sumusd); dp[i] = current_sumusd[best_idx]; path[i] = j_indices[best_idx]
    actions = np.zeros(n, dtype=int); current_day = np.argmax(dp)
    while current_day > 0:
        actions[current_day] = 1; current_day = path[current_day]
    actions[0] = 1
    return actions

# ! NEW: 3.2 LFSR (Characteristic Polynomial) Strategy
@njit(cache=True)
def _generate_lfsr_actions_numba(length: int, seed: int, taps: np.ndarray) -> np.ndarray:
    """
    สร้าง Action Sequence โดยใช้หลักการ Linear Feedback Shift Register (LFSR)
    เร่งความเร็วด้วย Numba
    """
    if length == 0: return np.empty(0, dtype=np.int32)
    state_len = max(length, 32)
    state = np.zeros(state_len, dtype=np.int32)

    for i in range(32):
        if (seed >> i) & 1:
            state[i] = 1

    if length > 32:
        for n in range(32, length):
            nxt_bit = 0
            for k in taps:
                nxt_bit ^= state[n - k]
            state[n] = nxt_bit
    
    actions = state[:length].copy()
    if length > 0: actions[0] = 1
    return actions

def evaluate_lfsr_batch(prices_tuple: Tuple[float, ...], num_samples_in_batch: int, batch_seed: int) -> List[Tuple[float, np.ndarray, Dict]]:
    """ประมวลผลการสุ่มหา LFSR parameters ที่ดีที่สุดใน 1 batch (สำหรับ Parallel Processing)"""
    results = []
    rng = np.random.default_rng(batch_seed)
    possible_taps = np.arange(1, 33, dtype=np.int32)
    window_len = len(prices_tuple)

    for _ in range(num_samples_in_batch):
        seed_val = rng.integers(0, 2**32, dtype=np.uint32)
        num_taps = rng.integers(4, 9) # สุ่มจำนวน tap ระหว่าง 4-8
        taps_val = rng.choice(possible_taps, size=num_taps, replace=False)
        taps_val.sort()

        actions = _generate_lfsr_actions_numba(window_len, int(seed_val), taps_val)
        
        _, sumusd, _, _, _, refer = calculate_optimized_cached(tuple(actions), prices_tuple)
        net = sumusd[-1] - refer[-1] - sumusd[0] if len(sumusd) > 0 else -np.inf
        
        params = {'seed': seed_val, 'taps': taps_val.tolist()}
        results.append((net, actions, params))
        
    return results

def find_best_lfsr_for_window(prices_window: np.ndarray, num_samples: int, max_workers: int, window_seed: int) -> Tuple[float, np.ndarray, Dict]:
    """ค้นหา LFSR parameters (seed, taps) ที่ให้กำไรสูงสุดใน 1 window"""
    window_len = len(prices_window)
    if window_len < 2: return 0.0, np.ones(window_len, dtype=np.int32), {}

    best_net_for_window = -np.inf
    best_actions_for_window = np.ones(window_len, dtype=np.int32)
    best_params_for_window = {}
    
    rng_for_batches = np.random.default_rng(window_seed)
    batch_size = max(1, num_samples // (max_workers * 4 if max_workers > 0 else 1))
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        batch_seeds = rng_for_batches.integers(0, 2**32, size=num_batches)
        futures = {executor.submit(evaluate_lfsr_batch, tuple(prices_window), batch_size, seed) for seed in batch_seeds}
        
        for future in as_completed(futures):
            try:
                for net, actions, params in future.result():
                    if net > best_net_for_window:
                        best_net_for_window = net
                        best_actions_for_window = actions
                        best_params_for_window = params
            except Exception as exc:
                st.warning(f"A batch in LFSR search generated an exception: {exc}")

    return best_net_for_window, best_actions_for_window, best_params_for_window

def generate_actions_sliding_window_lfsr(ticker_data: pd.DataFrame, window_size: int, num_samples: int, master_seed: int, max_workers: int) -> Tuple[np.ndarray, pd.DataFrame]:
    """สร้าง Action Sequence ทั้งหมดโดยใช้กลยุทธ์ LFSR Sliding Window"""
    prices = ticker_data['Close'].to_numpy(); n = len(prices)
    final_actions = np.array([], dtype=np.int32); window_details_list = []
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="กำลังประมวลผล LFSR Search...")
    st.write(f"🧬 ค้นหา LFSR (Seed, Taps) ที่ดีที่สุดในแต่ละ Window | {num_samples} samples/window")
    st.write(f"⚡ ใช้ Parallel Processing: {max_workers} workers")

    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n); prices_window = prices[start_index:end_index]
        if len(prices_window) < 2: continue

        window_seed = master_seed + i
        best_net, best_actions, best_params = find_best_lfsr_for_window(prices_window, num_samples, max_workers, window_seed)
        final_actions = np.concatenate((final_actions, best_actions))
        
        start_date_str = ticker_data.index[start_index].strftime('%Y-%m-%d'); end_date_str = ticker_data.index[end_index-1].strftime('%Y-%m-%d')
        window_details_list.append({
            'window_number': i + 1, 'timeline': f"{start_date_str} ถึง {end_date_str}",
            'max_net': round(best_net, 2), 
            'best_seed': int(best_params.get('seed', 0)),
            'best_taps': str(best_params.get('taps', [])), 
            'action_count': int(np.sum(best_actions)),
            'window_size': len(prices_window), 'window_seed': window_seed
        })
        progress_bar.progress((i + 1) / num_windows, text=f"ประมวลผล Window {i+1}/{num_windows}")
        
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)


# ==============================================================================
# 4. UI Rendering Functions
# ==============================================================================
def render_settings_tab(config: Dict[str, Any]):
    """แสดงผล UI สำหรับ Tab การตั้งค่า (ปรับให้แสดงเฉพาะพารามิเตอร์ที่ใช้)"""
    st.write("⚙️ **การตั้งค่าพารามิเตอร์**")
    
    asset_list = config.get('assets', ['FFWM'])
    try:
        default_index = asset_list.index(st.session_state.test_ticker)
    except ValueError:
        default_index = 0
    st.session_state.test_ticker = st.selectbox("เลือก Ticker สำหรับทดสอบ", options=asset_list, index=default_index)

    st.write("📅 **ช่วงวันที่สำหรับการวิเคราะห์**")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.start_date = st.date_input("วันที่เริ่มต้น", value=st.session_state.start_date)
    with col2:
        st.session_state.end_date = st.date_input("วันที่สิ้นสุด", value=st.session_state.end_date)
    if st.session_state.start_date >= st.session_state.end_date:
        st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")

    st.divider()
    st.subheader("พารามิเตอร์สำหรับ Sliding Window")
    c1, c2, c3, c4 = st.columns(4)
    st.session_state.window_size = c1.number_input("ขนาด Window (วัน)", min_value=2, value=st.session_state.window_size)
    st.session_state.num_samples = c2.number_input("จำนวน Samples/Window", min_value=100, value=st.session_state.num_samples, format="%d", help="จำนวนการสุ่ม Parameter (Seed, Taps) ในแต่ละ Window สำหรับกลยุทธ์ LFSR")
    st.session_state.master_seed = c3.number_input("Master Seed", value=st.session_state.master_seed, format="%d", help="Seed หลักสำหรับกลยุทธ์ LFSR เพื่อผลลัพธ์ที่ทำซ้ำได้")
    st.session_state.max_workers = c4.number_input("จำนวน Workers (CPU)", min_value=1, max_value=16, value=st.session_state.max_workers, help="ใช้สำหรับเร่งความเร็วการค้นหาในกลยุทธ์ LFSR")


def display_comparison_charts(results: Dict[str, pd.DataFrame], chart_title: str = '📊 เปรียบเทียบกำไรสุทธิ (Net Profit)'):
    """แสดงผลกราฟเปรียบเทียบผลลัพธ์จากหลายๆ กลยุทธ์"""
    if not results:
        st.warning("ไม่มีข้อมูลสำหรับสร้างกราฟเปรียบเทียบ")
        return

    valid_dfs = {name: df for name, df in results.items() if not df.empty and 'net' in df.columns}
    if not valid_dfs:
        st.warning("ไม่มีข้อมูล 'net' สำหรับสร้างกราฟ")
        return
        
    longest_index = max((df.index for df in valid_dfs.values()), key=len, default=None)
    if longest_index is None:
        st.warning("ไม่มีข้อมูลสำหรับสร้างกราฟเปรียบเทียบ")
        return

    chart_data = pd.DataFrame(index=longest_index)
    for name, df in valid_dfs.items():
        chart_data[name] = df['net'].reindex(longest_index).ffill()
    
    st.write(chart_title)
    st.line_chart(chart_data)

# ! NEW: UI Tab for LFSR Strategy
def render_lfsr_tab():
    """แสดงผล UI สำหรับ Tab LFSR (Characteristic Polynomial)"""
    st.write("---")
    st.markdown(f"### 🧬 ทดสอบด้วย {Strategy.LFSR_SEARCH}")
    st.info("กลยุทธ์นี้จะค้นหา **Seed (สถานะเริ่มต้น 32-bit)** และ **Taps (สมการ Recurrence)** ของ LFSR ที่ให้ผลตอบแทนดีที่สุดในแต่ละ Window")

    if st.button(f"🚀 เริ่มทดสอบ {Strategy.LFSR_SEARCH}", type="primary", key="lfsr_test_button"):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้องในแท็บ 'การตั้งค่า'"); return
            
        ticker = st.session_state.test_ticker; start_date_str = st.session_state.start_date.strftime('%Y-%m-%d'); end_date_str = st.session_state.end_date.strftime('%Y-%m-%d')
        st.info(f"กำลังดึงข้อมูลสำหรับ **{ticker}** | {start_date_str} ถึง {end_date_str}")
        ticker_data = get_ticker_data(ticker, start_date_str, end_date_str)
        if ticker_data.empty: st.error("ไม่พบข้อมูลสำหรับ Ticker และช่วงวันที่ที่เลือก"); return
        
        prices = ticker_data['Close'].to_numpy(); num_days = len(prices)
        with st.spinner("กำลังคำนวณกลยุทธ์ต่างๆ (LFSR Search)..."):
            actions, df_windows = generate_actions_sliding_window_lfsr(
                ticker_data, st.session_state.window_size, st.session_state.num_samples,
                st.session_state.master_seed, st.session_state.max_workers
            )
            actions_min = generate_actions_rebalance_daily(num_days)
            actions_max = generate_actions_perfect_foresight(prices.tolist())
            
            results = {}; strategy_map = {
                Strategy.LFSR_SEARCH: actions.tolist(),
                Strategy.REBALANCE_DAILY: actions_min.tolist(),
                Strategy.PERFECT_FORESIGHT: actions_max.tolist()
            }
            for strategy_name, actions_list in strategy_map.items():
                df = run_simulation(prices.tolist(), actions_list)
                if not df.empty: df.index = ticker_data.index[:len(df)]
                results[strategy_name] = df
        
        st.success("การทดสอบเสร็จสมบูรณ์!")
        st.write("---")
        display_comparison_charts(results)
        
        st.write(f"📈 **สรุปผลการค้นหาด้วย {Strategy.LFSR_SEARCH}**")
        st.dataframe(df_windows[['window_number', 'timeline', 'max_net', 'best_seed', 'best_taps', 'action_count']], use_container_width=True)
        csv = df_windows.to_csv(index=False)
        st.download_button(label="📥 ดาวน์โหลด LFSR Details (CSV)", data=csv, file_name=f'lfsr_seq_{ticker}.csv', mime='text/csv')

# --- โค้ดสำหรับ Tab อื่นๆ ที่ยังคงใช้งานอยู่ ---

def render_analytics_tab():
    """แสดงผล UI สำหรับ Tab Advanced Analytics"""
    st.header("📊 Advanced Analytics Dashboard")
    
    with st.container(border=True):
        st.subheader("เลือกวิธีการนำเข้าข้อมูล:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 1. อัปโหลดไฟล์จากเครื่อง")
            uploaded_file = st.file_uploader("อัปโหลดไฟล์ CSV ของคุณ", type=['csv'], key="local_uploader")
            if uploaded_file is not None:
                try:
                    st.session_state.df_for_analysis = pd.read_csv(uploaded_file)
                    st.success("✅ โหลดไฟล์สำเร็จ!")
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")
                    st.session_state.df_for_analysis = None
        with col2:
            st.markdown("##### 2. หรือ โหลดจาก GitHub URL")
            default_github_url = f"https://raw.githubusercontent.com/firstnattapon/streamlit-example-1/refs/heads/master/Seed_Sliding_Window/{st.session_state.test_ticker}.csv"
            github_url = st.text_input("ป้อน GitHub URL ของไฟล์ CSV:", value=default_github_url, key="github_url_input")
            if st.button("📥 โหลดข้อมูลจาก GitHub"):
                if github_url:
                    try:
                        raw_url = github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
                        with st.spinner("กำลังดาวน์โหลดข้อมูล..."):
                            st.session_state.df_for_analysis = pd.read_csv(raw_url)
                        st.success("✅ โหลดข้อมูลจาก GitHub สำเร็จ!")
                    except Exception as e:
                        st.error(f"❌ ไม่สามารถโหลดข้อมูลจาก URL ได้: {e}")
                        st.session_state.df_for_analysis = None
                else:
                    st.warning("กรุณาป้อน URL ของไฟล์ CSV")
    st.divider()

    if st.session_state.df_for_analysis is not None:
        st.subheader("ผลการวิเคราะห์")
        df_to_analyze = st.session_state.df_for_analysis
        try:
            required_cols = ['window_number', 'timeline', 'max_net']
            if 'action_sequence' not in df_to_analyze.columns:
                st.info("คอลัมน์ 'action_sequence' ไม่พบในไฟล์, ฟีเจอร์ 'Stitched DNA Analysis' จะถูกปิดใช้งาน")
                df_to_analyze['action_sequence'] = [[] for _ in range(len(df_to_analyze))]


            if not all(col in df_to_analyze.columns for col in required_cols):
                st.error(f"ไฟล์ CSV ไม่สมบูรณ์! ต้องมีคอลัมน์พื้นฐาน: {', '.join(required_cols)}")
                return
            
            df = df_to_analyze.copy()
            if 'result' not in df.columns:
                df['result'] = np.where(df['max_net'] > 0, 'Win', 'Loss')
            
            overview_tab, stitched_dna_tab = st.tabs(["🔬 ภาพรวมและสำรวจราย Window", "🧬 Stitched DNA Analysis"])

            with overview_tab:
                st.subheader("ภาพรวมประสิทธิภาพ (Overall Performance)")
                gross_profit = df[df['max_net'] > 0]['max_net'].sum()
                gross_loss = abs(df[df['max_net'] < 0]['max_net'].sum())
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                win_rate = (df['result'] == 'Win').mean() * 100
                
                kpi_cols = st.columns(4)
                kpi_cols[0].metric("Total Net Profit", f"${df['max_net'].sum():,.2f}")
                kpi_cols[1].metric("Win Rate", f"{win_rate:.2f}%")
                kpi_cols[2].metric("Profit Factor", f"{profit_factor:.2f}")
                kpi_cols[3].metric("Total Windows", f"{df.shape[0]}")
                
                st.subheader("สำรวจข้อมูลราย Window")
                selected_window = st.selectbox(
                    'เลือก Window ที่ต้องการดูรายละเอียด:',
                    options=df['window_number'],
                    format_func=lambda x: f"Window #{x} (Timeline: {df.loc[df['window_number'] == x, 'timeline'].iloc[0]})"
                )
                if selected_window:
                    window_data = df[df['window_number'] == selected_window].iloc[0]
                    st.markdown(f"**รายละเอียดของ Window #{selected_window}**")
                    st.dataframe(window_data)

            def safe_literal_eval(val):
                if pd.isna(val): return []
                if isinstance(val, list): return val
                if isinstance(val, str) and val.strip().startswith('[') and val.strip().endswith(']'):
                    try: return ast.literal_eval(val)
                    except: return []
                return []

            with stitched_dna_tab:
                st.subheader("ทดสอบกลยุทธ์จาก 'Stitched' DNA")
                st.markdown("จำลองการเทรดจริงโดยนำ **`action_sequence`** จากแต่ละ Window มา 'เย็บ' ต่อกัน และเปรียบเทียบกับ Benchmark")
                
                if 'action_sequence' not in df.columns:
                     st.warning("ไม่สามารถทำการวิเคราะห์ Stitched DNA ได้เนื่องจากไม่มีคอลัมน์ 'action_sequence' ในข้อมูลที่อัปโหลด")
                else:
                    df['action_sequence_list'] = df['action_sequence'].apply(safe_literal_eval)
                    df_sorted = df.sort_values('window_number')
                    stitched_actions = [action for seq in df_sorted['action_sequence_list'] for action in seq]
                    
                    dna_cols = st.columns(2)
                    stitch_ticker = dna_cols[0].text_input("Ticker สำหรับจำลอง", value=st.session_state.test_ticker, key='stitch_ticker_input')
                    stitch_start_date = dna_cols[1].date_input("วันที่เริ่มต้นจำลอง", value=datetime.now().date() - pd.Timedelta(days=365), key='stitch_date_input')
                    
                    if st.button("🧬 เริ่มการวิเคราะห์ Stitched DNA แบบเปรียบเทียบ", type="primary", key='stitch_dna_btn'):
                        if not stitched_actions:
                            st.error("ไม่สามารถสร้าง Action Sequence จากข้อมูลที่โหลดได้")
                        else:
                            with st.spinner(f"กำลังจำลองกลยุทธ์สำหรับ {stitch_ticker}..."):
                                sim_data = get_ticker_data(stitch_ticker, str(stitch_start_date), str(datetime.now().date()))
                                if sim_data.empty:
                                    st.error("ไม่สามารถดึงข้อมูลสำหรับจำลองได้")
                                else:
                                    prices = sim_data['Close'].to_numpy()
                                    n_total = len(prices)
                                    
                                    final_actions_dna = stitched_actions[:n_total]
                                    df_dna = run_simulation(prices[:len(final_actions_dna)].tolist(), final_actions_dna)
                                    df_max = run_simulation(prices.tolist(), generate_actions_perfect_foresight(prices.tolist()).tolist())
                                    df_min = run_simulation(prices.tolist(), generate_actions_rebalance_daily(n_total).tolist())
                                    
                                    results_dna = {}
                                    if not df_dna.empty:
                                        df_dna.index = sim_data.index[:len(df_dna)]
                                        results_dna['Stitched DNA'] = df_dna
                                    if not df_max.empty:
                                        df_max.index = sim_data.index[:len(df_max)]
                                        results_dna[Strategy.PERFECT_FORESIGHT] = df_max
                                    if not df_min.empty:
                                        df_min.index = sim_data.index[:len(df_min)]
                                        results_dna[Strategy.REBALANCE_DAILY] = df_min
                                        
                                    st.subheader("Performance Comparison (Net Profit)")
                                    display_comparison_charts(results_dna)
                                    
                                    st.subheader("สรุปผลลัพธ์สุดท้าย (Final Net Profit)")
                                    metric_cols = st.columns(3)
                                    final_net_max = results_dna.get(Strategy.PERFECT_FORESIGHT, pd.DataFrame({'net': [0]}))['net'].iloc[-1]
                                    final_net_dna = results_dna.get('Stitched DNA', pd.DataFrame({'net': [0]}))['net'].iloc[-1]
                                    final_net_min = results_dna.get(Strategy.REBALANCE_DAILY, pd.DataFrame({'net': [0]}))['net'].iloc[-1]
                                    
                                    metric_cols[0].metric("Max Performance", f"${final_net_max:,.2f}")
                                    metric_cols[1].metric("Stitched DNA Strategy", f"${final_net_dna:,.2f}", delta=f"{final_net_dna - final_net_min:,.2f} vs Min", delta_color="normal")
                                    metric_cols[2].metric("Min Performance", f"${final_net_min:,.2f}")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการวิเคราะห์ข้อมูล: {e}")
            st.exception(e)

def render_manual_seed_tab(config: Dict[str, Any]):
    """แสดงผล UI สำหรับ Tab Manual Seed Comparator"""
    st.header("🌱 Manual Seed Strategy Comparator")
    st.markdown("สร้างและเปรียบเทียบ Action Sequences โดยการตัดส่วนท้าย (`tail`) จาก Seed ที่กำหนด")
    
    with st.container(border=True):
        st.subheader("1. กำหนดค่า Input สำหรับการทดสอบ")
        col1, col2 = st.columns([1, 2])
        with col1:
            asset_list = config.get('assets', ['FFWM'])
            try:
                default_index = asset_list.index(st.session_state.get('manual_ticker_key', st.session_state.test_ticker))
            except (ValueError, KeyError):
                default_index = 0
            manual_ticker = st.selectbox("เลือก Ticker", options=asset_list, index=default_index, key="manual_ticker_key", on_change=on_ticker_change_callback, args=(config,))
        with col2:
            c1, c2 = st.columns(2)
            manual_start_date = c1.date_input("วันที่เริ่มต้น (Start Date)", value=st.session_state.start_date, key="manual_start_compare_tail")
            manual_end_date = c2.date_input("วันที่สิ้นสุด (End Date)", value=st.session_state.end_date, key="manual_end_compare_tail")
        
        if manual_start_date >= manual_end_date: st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
        st.divider()
        st.write("**กำหนดกลยุทธ์ (Seed/Size/Tail) ที่ต้องการเปรียบเทียบ:**")
        
        for i, line in enumerate(st.session_state.manual_seed_lines):
            cols = st.columns([1, 2, 2, 2])
            cols[0].write(f"**Line {i+1}**")
            line['seed'] = cols[1].number_input("Input Seed", value=line.get('seed', 1), min_value=0, key=f"seed_compare_tail_{i}")
            line['size'] = cols[2].number_input("Size (ขนาด Sequence เริ่มต้น)", value=line.get('size', 60), min_value=1, key=f"size_compare_tail_{i}")
            line['tail'] = cols[3].number_input("Tail (ส่วนท้ายที่จะใช้)", value=line.get('tail', 10), min_value=1, max_value=line.get('size', 60), key=f"tail_compare_tail_{i}")
            
        b_col1, b_col2, _ = st.columns([1,1,4])
        if b_col1.button("➕ เพิ่ม Line เปรียบเทียบ"):
            st.session_state.manual_seed_lines.append({'seed': np.random.randint(1, 10000), 'size': 50, 'tail': 20})
            st.rerun()
        if b_col2.button("➖ ลบ Line สุดท้าย"):
            if len(st.session_state.manual_seed_lines) > 1:
                st.session_state.manual_seed_lines.pop()
                st.rerun()
            else:
                st.warning("ต้องมีอย่างน้อย 1 line")
    
    st.write("---")
    if st.button("📈 เปรียบเทียบประสิทธิภาพ Seeds", type="primary", key="compare_manual_seeds_btn"):
        if manual_start_date >= manual_end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้อง"); return
            
        with st.spinner("กำลังดึงข้อมูลและจำลองการเทรด..."):
            start_str, end_str = manual_start_date.strftime('%Y-%m-%d'), manual_end_date.strftime('%Y-%m-%d')
            ticker_data = get_ticker_data(manual_ticker, start_str, end_str)
            if ticker_data.empty: st.error(f"ไม่พบข้อมูลสำหรับ {manual_ticker} ในช่วงวันที่ที่เลือก"); return

            prices, num_trading_days = ticker_data['Close'].to_numpy(), len(ticker_data)
            st.info(f"📊 พบข้อมูลราคา {num_trading_days} วันทำการในช่วงที่เลือก")
            
            results, max_sim_len = {}, 0
            
            for i, line_info in enumerate(st.session_state.manual_seed_lines):
                input_seed, size_seed, tail_seed = line_info['seed'], line_info['size'], line_info['tail']
                if tail_seed > size_seed:
                    st.error(f"Line {i+1}: Tail ({tail_seed}) ต้องไม่มากกว่า Size ({size_seed})"); return
                
                rng_best = np.random.default_rng(input_seed)
                full_actions = rng_best.integers(0, 2, size=size_seed)
                actions_from_tail = full_actions[-tail_seed:].tolist()
                
                sim_len = min(num_trading_days, len(actions_from_tail))
                if sim_len == 0: continue
                
                df_line = run_simulation(prices[:sim_len].tolist(), actions_from_tail[:sim_len])
                if not df_line.empty:
                    df_line.index = ticker_data.index[:sim_len]
                    results[f"Seed {input_seed} (Tail {tail_seed})"] = df_line
                    max_sim_len = max(max_sim_len, sim_len)

            if not results: st.error("ไม่สามารถสร้างผลลัพธ์จาก Seed ที่กำหนดได้"); return
            
            if max_sim_len > 0:
                prices_for_benchmark = prices[:max_sim_len].tolist()
                df_max = run_simulation(prices_for_benchmark, generate_actions_perfect_foresight(prices_for_benchmark).tolist())
                df_min = run_simulation(prices_for_benchmark, generate_actions_rebalance_daily(max_sim_len).tolist())
                if not df_max.empty:
                    df_max.index = ticker_data.index[:max_sim_len]
                    results[Strategy.PERFECT_FORESIGHT] = df_max
                if not df_min.empty:
                    df_min.index = ticker_data.index[:max_sim_len]
                    results[Strategy.REBALANCE_DAILY] = df_min
            
            st.success("การเปรียบเทียบเสร็จสมบูรณ์!")
            display_comparison_charts(results, chart_title="📊 Performance Comparison (Net Profit)")
            
            st.subheader("สรุปผลลัพธ์สุดท้าย (Final Net Profit)")
            
            sorted_names = [name for name in results.keys() if name not in [Strategy.PERFECT_FORESIGHT, Strategy.REBALANCE_DAILY]]
            display_order = [Strategy.PERFECT_FORESIGHT] + sorted(sorted_names) + [Strategy.REBALANCE_DAILY]
            
            final_results_list = [{'name': name, 'net': results[name]['net'].iloc[-1]} for name in display_order if name in results and not results[name].empty]
            
            if final_results_list:
                final_metrics_cols = st.columns(len(final_results_list))
                for idx, item in enumerate(final_results_list):
                    final_metrics_cols[idx].metric(item['name'], f"${item['net']:,.2f}")


# ==============================================================================
# 5. Main Application
# ==============================================================================
def main():
    """
    ฟังก์ชันหลักในการรัน Streamlit Application
    """
    st.set_page_config(page_title="Best Seed Sliding Window", page_icon="🎯", layout="wide")
    st.markdown("### 🎯 Best Seed Sliding Window Tester (LFSR Strategy)")
    st.caption("เครื่องมือทดสอบการหา Best Seed และ Sequence โดยใช้กลยุทธ์ LFSR (Core Calculation เร่งความเร็วด้วย Numba)")

    config = load_config()
    initialize_session_state(config)

    # ! UI: ปรับแก้ Tab ให้แสดงเฉพาะที่ต้องการ
    tab_list = [
        "⚙️ การตั้งค่า",
        "🧬 LFSR (Characteristic Poly)",
        "📊 Advanced Analytics",
        "🌱 Forward Rolling Comparator"
    ]
    tabs = st.tabs(tab_list)

    with tabs[0]: render_settings_tab(config)
    with tabs[1]: render_lfsr_tab() # ! UI: เรียกใช้ฟังก์ชันสำหรับ Tab ใหม่
    with tabs[2]: render_analytics_tab()
    with tabs[3]: render_manual_seed_tab(config)

    with st.expander("📖 คำอธิบายวิธีการทำงานและแนวคิด (LFSR Strategy)"):
        st.markdown("""
        ### หลักการทำงานของเวอร์ชันนี้:

        **เป้าหมาย: ประยุกต์ใช้หลักการ Characteristic Polynomial (LFSR) ในการสร้าง Action Sequence**

        1.  **Linear Feedback Shift Register (LFSR)**:
            - กลยุทธ์นี้ไม่ได้ใช้การสุ่ม `Action` โดยตรง แต่สร้าง `Action Sequence` จากกระบวนการทางคณิตศาสตร์ที่เรียกว่า LFSR ซึ่งเป็นพื้นฐานของ "Characteristic Polynomial"
            - LFSR จะสร้างลำดับของบิต (0 หรือ 1) โดยอาศัยสถานะก่อนหน้า (State) และสมการเวียนเกิด (Recurrence Relation) ที่กำหนดโดย "Taps"
        
        2.  **พารามิเตอร์ที่ค้นหา**:
            - **Seed (32-bit integer)**: คือสถานะเริ่มต้น 32 บิตของ Register
            - **Taps**: คือตำแหน่งของบิตใน Register ที่จะถูกนำมา XOR กันเพื่อสร้างบิตถัดไป (เช่น `taps = {1, 5, 8, 11, ...}`)
            - ในแต่ละ Window โปรแกรมจะทำการ **สุ่มหาคู่ (Seed, Taps) ที่ให้ผลตอบแทน (Net Profit) สูงที่สุด**

        3.  **กระบวนการสร้าง Action**:
            - `a = seed` (แปลง integer 32-bit เป็น state เริ่มต้น)
            - สำหรับ `n` ตั้งแต่ 32 เป็นต้นไป:
              `a[n] = a[n-k1] XOR a[n-k2] XOR ...` โดยที่ `k1, k2, ...` คือตำแหน่งใน `taps`
            - ลำดับ `a` ที่ได้จะถูกนำไปใช้เป็น `Action Sequence` (0=Hold, 1=Rebalance)

        4.  **⚡ Core Acceleration**:
            - ฟังก์ชันการคำนวณผลการเทรด (`_calculate_simulation_numba`) และฟังก์ชันสร้างลำดับ LFSR (`_generate_lfsr_actions_numba`) ถูกเร่งความเร็วด้วย **Numba (`@njit`)** ทำให้การค้นหาที่ต้องทำซ้ำนับแสนครั้งในแต่ละ Window สามารถทำได้อย่างรวดเร็ว
            - ใช้ `ThreadPoolExecutor` เพื่อกระจายงานค้นหาในแต่ละ Window ไปยังหลาย CPU Core พร้อมกัน
        
        5.  **ผลลัพธ์**:
            - ได้ `Action Sequence` ที่มีโครงสร้างและรูปแบบที่ซับซ้อนกว่าการสุ่มทั่วไป ซึ่งอาจจับรูปแบบของราคาได้ดีกว่า
            - ได้ผลการทดสอบที่รวดเร็วและน่าเชื่อถือจากพลังของ Numba และ Parallel Processing
        """)

if __name__ == "__main__":
    main()
