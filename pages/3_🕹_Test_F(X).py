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
    REBALANCE_DAILY = "Rebalance Daily (Min)"
    PERFECT_FORESIGHT = "Perfect Foresight (Max)"
    SLIDING_WINDOW = "Best Seed Sliding Window"
    # ! NEW MODEL: เพิ่มชื่อกลยุทธ์ใหม่
    PROBABILISTIC_GREEDY = "Probabilistic Greedy (Seed-Based)"

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
            "assets": ["FFWM", "NEGG", "RIVN", "BTC-USD", "NVDA"],
            "default_settings": {
                "selected_ticker": "FFWM", "start_date": "2024-01-01", "window_size": 30,
                "num_seeds": 10000, "max_workers": 8,
                "prob_lookback": 20, "prob_sensitivity": 5, "prob_master_seed": 42
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
    if not selected_ticker: return
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
        st.session_state.start_date = datetime.strptime(defaults.get('start_date', '2024-01-01'), '%Y-%m-%d').date()
    if 'end_date' not in st.session_state:
        st.session_state.end_date = datetime.now().date()
    if 'window_size' not in st.session_state:
        st.session_state.window_size = defaults.get('window_size', 30)
    if 'num_seeds' not in st.session_state:
        st.session_state.num_seeds = defaults.get('num_seeds', 10000)
    if 'max_workers' not in st.session_state:
        st.session_state.max_workers = defaults.get('max_workers', 8)
        
    # ! NEW MODEL: เพิ่มพารามิเตอร์สำหรับ Probabilistic Greedy
    if 'prob_lookback' not in st.session_state:
        st.session_state.prob_lookback = defaults.get('prob_lookback', 20)
    if 'prob_sensitivity' not in st.session_state:
        st.session_state.prob_sensitivity = defaults.get('prob_sensitivity', 5)
    if 'prob_master_seed' not in st.session_state:
        st.session_state.prob_master_seed = defaults.get('prob_master_seed', 42)

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
    """ดึงข้อมูลราคาหุ้น/สินทรัพย์จาก Yahoo Finance และ Cache ผลลัพธ์ไว้ 1 ชั่วโมง"""
    try:
        data = yf.Ticker(ticker).history(start=start_date, end=end_date)[['Close']]
        if data.empty: return pd.DataFrame()
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
    คำนวณผลลัพธ์การจำลองการเทรด (หัวใจของการคำนวณ) - เร่งความเร็วด้วย Numba
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

@lru_cache(maxsize=8192)
def calculate_optimized_cached(action_tuple: Tuple[int, ...], price_tuple: Tuple[float, ...], fix: int = 1500) -> Tuple:
    """Wrapper function สำหรับเรียกฟังก์ชัน Numba โดยใช้ Cache"""
    action_array = np.asarray(action_tuple, dtype=np.int32)
    price_array = np.asarray(price_tuple, dtype=np.float64)
    return _calculate_simulation_numba(action_array, price_array, fix)

def run_simulation(prices: List[float], actions: List[int], fix: int = 1500) -> pd.DataFrame:
    """สร้าง DataFrame ผลลัพธ์จากการจำลองการเทรด"""
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

# 3.1 Standard & Benchmark Strategies
def generate_actions_rebalance_daily(num_days: int) -> np.ndarray:
    """สร้าง Action สำหรับกลยุทธ์ Rebalance ทุกวัน (Min Performance)"""
    return np.ones(num_days, dtype=np.int32)

def generate_actions_perfect_foresight(prices: List[float], fix: int = 1500) -> np.ndarray:
    """สร้าง Action สำหรับกลยุทธ์ Perfect Foresight (Max Performance) โดยใช้ Dynamic Programming"""
    price_arr = np.asarray(prices, dtype=np.float64)
    n = len(price_arr)
    if n < 2: return np.ones(n, dtype=int)
    dp = np.zeros(n, dtype=np.float64); path = np.zeros(n, dtype=int)
    dp[0] = float(fix * 2)
    for i in range(1, n):
        j_indices = np.arange(i)
        profits = fix * ((price_arr[i] / price_arr[j_indices]) - 1)
        current_sumusd = dp[j_indices] + profits
        best_idx = np.argmax(current_sumusd)
        dp[i] = current_sumusd[best_idx]; path[i] = j_indices[best_idx]
    actions = np.zeros(n, dtype=int)
    current_day = np.argmax(dp)
    while current_day > 0:
        actions[current_day] = 1; current_day = path[current_day]
    actions[0] = 1
    return actions

# ! NEW MODEL LOGIC
def generate_actions_dynamic_probability(prices: np.ndarray, lookback_period: int, sensitivity: float, seed: int) -> np.ndarray:
    """
    สร้าง Action Sequence โดยใช้ความน่าจะเป็นที่แปรผันตามราคา
    - เป็นโมเดลที่พยายามเลียนแบบ Perfect Foresight โดยใช้ข้อมูลปัจจุบัน
    - มีความสง่างามทางคณิตศาสตร์, ไม่เป็น Black Box, และ Traceable ผ่าน Seed
    """
    n = len(prices)
    if n < lookback_period:
        return np.ones(n, dtype=np.int32) # ถ้าข้อมูลน้อยไป ให้ rebalance ทุกวัน

    # 1. สร้าง "เครื่องปั่นตัวเลข" จาก Seed เพื่อให้คำนวณซ้ำได้
    rng = np.random.default_rng(seed)
    random_draws = rng.random(size=n)

    # 2. คำนวณราคาอ้างอิง (Simple Moving Average)
    price_series = pd.Series(prices)
    reference_prices = price_series.rolling(window=lookback_period, min_periods=1).mean().to_numpy()

    # 3. คำนวณ "คะแนนความน่าสนใจ" (Vectorized for Speed)
    #    คะแนนเป็นบวกเมื่อราคาปัจจุบันต่ำกว่าราคาอ้างอิง (น่าซื้อ)
    score = (reference_prices - prices) / reference_prices

    # 4. แปลงคะแนนเป็น "ความน่าจะเป็น" โดยใช้ Sigmoid function
    #    'sensitivity' ควบคุมความชันของโค้ง (ยิ่งมาก ยิ่งตอบสนองรุนแรง)
    scaled_score = score * sensitivity
    probability = 1 / (1 + np.exp(-scaled_score))

    # 5. ตัดสินใจ Action โดยเทียบเลขสุ่มกับความน่าจะเป็น
    actions = (random_draws < probability).astype(np.int32)
    actions[0] = 1  # บังคับให้วันแรกซื้อเสมอ

    return actions


# 3.2 Standard Seed Generation (Original Logic)
def find_best_seed_for_window(prices_window: np.ndarray, num_seeds_to_try: int, max_workers: int) -> Tuple[int, float, np.ndarray]:
    """ค้นหา Seed ที่ให้ค่า Net Profit สูงสุดสำหรับ Price Window ที่กำหนด (Parallel Processing)"""
    window_len = len(prices_window)
    if window_len < 2: return 1, 0.0, np.ones(window_len, dtype=int)

    def evaluate_seed_batch(seed_batch: np.ndarray) -> List[Tuple[int, float]]:
        results = []; prices_tuple = tuple(prices_window)
        for seed in seed_batch:
            rng = np.random.default_rng(seed)
            actions_window = rng.integers(0, 2, size=window_len)
            _, sumusd, _, _, _, refer = calculate_optimized_cached(tuple(actions_window), prices_tuple)
            net = sumusd[-1] - refer[-1] - sumusd[0] if len(sumusd) > 0 else -np.inf
            results.append((seed, net))
        return results

    best_seed_for_window = -1; max_net_for_window = -np.inf
    random_seeds = np.arange(num_seeds_to_try)
    batch_size = max(1, num_seeds_to_try // (max_workers * 4 if max_workers > 0 else 1))
    seed_batches = [random_seeds[j:j+batch_size] for j in range(0, len(random_seeds), batch_size)]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_seed_batch, batch) for batch in seed_batches]
        for future in as_completed(futures):
            for seed, final_net in future.result():
                if final_net > max_net_for_window:
                    max_net_for_window = final_net; best_seed_for_window = seed

    if best_seed_for_window >= 0:
        rng_best = np.random.default_rng(best_seed_for_window)
        best_actions = rng_best.integers(0, 2, size=window_len)
    else:
        best_seed_for_window = 1; best_actions = np.ones(window_len, dtype=int); max_net_for_window = 0.0
    
    best_actions[0] = 1
    return best_seed_for_window, max_net_for_window, best_actions

def generate_actions_sliding_window(ticker_data: pd.DataFrame, window_size: int, num_seeds_to_try: int, max_workers: int) -> Tuple[np.ndarray, pd.DataFrame]:
    """สร้าง Action Sequence ทั้งหมดโดยใช้กลยุทธ์ Sliding Window"""
    prices = ticker_data['Close'].to_numpy(); n = len(prices)
    final_actions = np.array([], dtype=int); window_details_list = []
    
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="กำลังประมวลผล Sliding Windows...")
    st.write(f"📊 ข้อมูล: {n} วัน | Window: {window_size} วัน | Windows: {num_windows} | Seeds/Window: {num_seeds_to_try}")
    
    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n); prices_window = prices[start_index:end_index]
        window_len = len(prices_window)
        if window_len == 0: continue
        
        best_seed, max_net, best_actions = find_best_seed_for_window(prices_window, num_seeds_to_try, max_workers)
        final_actions = np.concatenate((final_actions, best_actions))
        
        start_date_str = ticker_data.index[start_index].strftime('%Y-%m-%d'); end_date_str = ticker_data.index[end_index-1].strftime('%Y-%m-%d')
        detail = {
            'window_number': i + 1, 'timeline': f"{start_date_str} ถึง {end_date_str}",
            'best_seed': best_seed, 'max_net': round(max_net, 2),
            'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2),
            'action_count': int(np.sum(best_actions)), 'window_size': window_len,
            'action_sequence': best_actions.tolist()
        }
        window_details_list.append(detail)
        progress_bar.progress((i + 1) / num_windows, text=f"ประมวลผล Window {i+1}/{num_windows}")
        
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)


# ==============================================================================
# 4. UI Rendering Functions
# ==============================================================================
def render_settings_tab(config: Dict[str, Any]):
    """แสดงผล UI สำหรับ Tab การตั้งค่า"""
    st.write("⚙️ **การตั้งค่าพารามิเตอร์**")
    
    asset_list = config.get('assets', ['FFWM'])
    try: default_index = asset_list.index(st.session_state.test_ticker)
    except ValueError: default_index = 0
    st.session_state.test_ticker = st.selectbox("เลือก Ticker สำหรับทดสอบ", options=asset_list, index=default_index)

    col1, col2 = st.columns(2)
    with col1: st.session_state.start_date = st.date_input("วันที่เริ่มต้น", value=st.session_state.start_date)
    with col2: st.session_state.end_date = st.date_input("วันที่สิ้นสุด", value=st.session_state.end_date)
    if st.session_state.start_date >= st.session_state.end_date:
        st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")

    st.divider()
    st.subheader("พารามิเตอร์สำหรับ 'Best Seed Sliding Window'")
    c1, c2, c3 = st.columns(3)
    st.session_state.window_size = c1.number_input("ขนาด Window (วัน)", min_value=2, value=st.session_state.window_size)
    st.session_state.num_seeds = c2.number_input("จำนวน Seeds ต่อ Window", min_value=100, value=st.session_state.num_seeds, format="%d")
    st.session_state.max_workers = c3.number_input("จำนวน Workers (CPU Cores)", min_value=1, max_value=16, value=st.session_state.max_workers)
    
    st.divider()
    # ! NEW MODEL: UI สำหรับ Probabilistic Greedy
    st.subheader("พารามิเตอร์สำหรับ 'Probabilistic Greedy Model'")
    p1, p2, p3 = st.columns(3)
    st.session_state.prob_lookback = p1.number_input("ช่วงวันดูย้อนหลัง (Lookback)", min_value=2, value=st.session_state.prob_lookback, help="จำนวนวันที่ใช้คำนวณราคาอ้างอิง (Moving Average)")
    st.session_state.prob_sensitivity = p2.number_input("ความไวต่อราคา (Sensitivity)", min_value=0.1, max_value=20.0, value=st.session_state.prob_sensitivity, format="%.1f", help="ยิ่งค่าสูง ยิ่งตอบสนองต่อการเปลี่ยนแปลงของราคาแรงขึ้น")
    st.session_state.prob_master_seed = p3.number_input("Master Seed", value=st.session_state.prob_master_seed, format="%d", help="Seed หลักสำหรับควบคุม 'เครื่องปั่นตัวเลข' เพื่อให้ผลลัพธ์เหมือนเดิมทุกครั้งที่รันด้วย Seed เดียวกัน")

def display_comparison_charts(results: Dict[str, pd.DataFrame], chart_title: str = '📊 เปรียบเทียบกำไรสุทธิ (Net Profit)'):
    """แสดงผลกราฟเปรียบเทียบผลลัพธ์จากหลายๆ กลยุทธ์"""
    if not results: return
    valid_dfs = {name: df for name, df in results.items() if not df.empty and 'net' in df.columns}
    if not valid_dfs: return
    try: longest_index = max((df.index for df in valid_dfs.values()), key=len, default=None)
    except ValueError: longest_index = None
    if longest_index is None: return

    chart_data = pd.DataFrame(index=longest_index)
    for name, df in valid_dfs.items():
        chart_data[name] = df['net'].reindex(longest_index).ffill()
    
    st.write(chart_title)
    st.line_chart(chart_data)

# ! NEW MODEL: UI Tab
def render_probabilistic_greedy_tab():
    """แสดงผล UI สำหรับ Tab Probabilistic Greedy Model"""
    st.header("🎯 Probabilistic Greedy Model")
    st.markdown("""
    โมเดลนี้พยายามเข้าใกล้ **Perfect Foresight** โดยใช้หลักการที่สง่างามทางคณิตศาสตร์และโปร่งใส:
    1.  **ประเมินราคา:** เทียบราคาปัจจุบันกับราคาเฉลี่ยย้อนหลัง (`Lookback`) เพื่อดูว่า "น่าสนใจ" แค่ไหน
    2.  **สร้างความน่าจะเป็น:** แปลงค่าความน่าสนใจเป็น "โอกาสในการซื้อ" ผ่านสมการ Sigmoid
    3.  **ตัดสินใจด้วย Seed:** ใช้ "เครื่องปั่นตัวเลข" ที่ควบคุมด้วย `Master Seed` เพื่อตัดสินใจว่าจะซื้อหรือไม่
    
    **ผลลัพธ์:** กลยุทธ์ที่ฉลาด, เร็ว, และสามารถคำนวณย้อนกลับได้ (Traceable)
    """)
    st.info("คุณสามารถปรับพารามิเตอร์ของโมเดลนี้ได้ในแท็บ **'⚙️ การตั้งค่า'**")

    if st.button("🚀 เริ่มทดสอบ Probabilistic Greedy Model", type="primary"):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้องในแท็บ 'การตั้งค่า'"); return
        
        ticker, start_date_str, end_date_str = st.session_state.test_ticker, st.session_state.start_date.strftime('%Y-%m-%d'), st.session_state.end_date.strftime('%Y-%m-%d')
        st.info(f"กำลังดึงข้อมูลสำหรับ **{ticker}** | {start_date_str} ถึง {end_date_str}")
        ticker_data = get_ticker_data(ticker, start_date_str, end_date_str)
        if ticker_data.empty: st.error("ไม่พบข้อมูล"); return

        prices = ticker_data['Close'].to_numpy(); num_days = len(prices)
        
        with st.spinner("กำลังคำนวณและเปรียบเทียบกลยุทธ์..."):
            # ! NEW MODEL: สร้าง Actions จากโมเดลใหม่
            actions_prob = generate_actions_dynamic_probability(
                prices, 
                st.session_state.prob_lookback, 
                st.session_state.prob_sensitivity, 
                st.session_state.prob_master_seed
            )
            # สร้าง Actions สำหรับ Benchmark
            actions_max = generate_actions_perfect_foresight(prices.tolist())
            actions_min = generate_actions_rebalance_daily(num_days)

            # รัน Simulation
            results = {}
            strategy_map = {
                Strategy.PROBABILISTIC_GREEDY: actions_prob.tolist(),
                Strategy.PERFECT_FORESIGHT: actions_max.tolist(),
                Strategy.REBALANCE_DAILY: actions_min.tolist()
            }
            for name, actions in strategy_map.items():
                df = run_simulation(prices.tolist(), actions)
                if not df.empty:
                    df.index = ticker_data.index[:len(df)]
                results[name] = df
        
        st.success("การทดสอบเสร็จสมบูรณ์!")
        display_comparison_charts(results)

        st.write("📈 **สรุปผลลัพธ์สุดท้าย (Final Net Profit)**")
        c1, c2, c3 = st.columns(3)
        final_net_max = results.get(Strategy.PERFECT_FORESIGHT, pd.DataFrame({'net': [0]}))['net'].iloc[-1]
        final_net_prob = results.get(Strategy.PROBABILISTIC_GREEDY, pd.DataFrame({'net': [0]}))['net'].iloc[-1]
        final_net_min = results.get(Strategy.REBALANCE_DAILY, pd.DataFrame({'net': [0]}))['net'].iloc[-1]

        c1.metric(Strategy.PERFECT_FORESIGHT, f"${final_net_max:,.2f}")
        c2.metric(Strategy.PROBABILISTIC_GREEDY, f"${final_net_prob:,.2f}", delta=f"{final_net_prob - final_net_min:,.2f} vs Min", delta_color="normal")
        c3.metric(Strategy.REBALANCE_DAILY, f"${final_net_min:,.2f}")

        st.write("---")
        st.write("📋 **Action Sequence ที่สร้างโดย Probabilistic Greedy Model**")
        st.dataframe(pd.DataFrame({'date': ticker_data.index[:len(actions_prob)], 'action': actions_prob}))

def render_test_tab():
    """แสดงผล UI สำหรับ Tab Best Seed (Random)"""
    st.header("🔍 Best Seed Sliding Window (Brute-Force Search)")
    st.markdown("กลยุทธ์ดั้งเดิมที่ใช้การ **'ค้นหาอย่างละเอียด'** (Brute-Force) เพื่อหา `seed` ที่ให้ผลตอบแทนดีที่สุดในแต่ละช่วงเวลา (`Window`) โดยใช้การประมวลผลแบบขนาน (Parallel Processing)")
    
    if st.button("🚀 เริ่มทดสอบ Best Seed (Accelerated)", type="primary"):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้องในแท็บ 'การตั้งค่า'"); return
            
        ticker, start_date_str, end_date_str = st.session_state.test_ticker, st.session_state.start_date.strftime('%Y-%m-%d'), st.session_state.end_date.strftime('%Y-%m-%d')
        st.info(f"กำลังดึงข้อมูลสำหรับ **{ticker}** | {start_date_str} ถึง {end_date_str}")
        ticker_data = get_ticker_data(ticker, start_date_str, end_date_str)
        if ticker_data.empty: st.error("ไม่พบข้อมูล"); return
            
        prices = ticker_data['Close'].to_numpy(); num_days = len(prices)

        with st.spinner("กำลังคำนวณกลยุทธ์ต่างๆ (Core calculation is Numba-accelerated)..."):
            actions_sliding, df_windows = generate_actions_sliding_window(
                ticker_data, st.session_state.window_size, st.session_state.num_seeds, st.session_state.max_workers
            )
            actions_min = generate_actions_rebalance_daily(num_days)
            actions_max = generate_actions_perfect_foresight(prices.tolist())
            
            results = {}; strategy_map = {
                Strategy.SLIDING_WINDOW: actions_sliding.tolist(),
                Strategy.PERFECT_FORESIGHT: actions_max.tolist(),
                Strategy.REBALANCE_DAILY: actions_min.tolist()
            }
            for name, actions in strategy_map.items():
                df = run_simulation(prices.tolist(), actions)
                if not df.empty: df.index = ticker_data.index[:len(df)]
                results[name] = df
        
        st.success("การทดสอบเสร็จสมบูรณ์!")
        display_comparison_charts(results)
        
        st.write("📈 **สรุปผลการค้นหา Best Seed**")
        total_actions = df_windows['action_count'].sum(); total_net = df_windows['max_net'].sum()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Windows", df_windows.shape[0])
        col2.metric("Total Actions", f"{total_actions}/{num_days}")
        col3.metric("Total Net (Sum)", f"${total_net:,.2f}")
        
        st.dataframe(df_windows[['window_number', 'timeline', 'best_seed', 'max_net', 'price_change_pct', 'action_count']], use_container_width=True)
        csv = df_windows.to_csv(index=False)
        st.download_button(label="📥 ดาวน์โหลด Window Details (CSV)", data=csv, file_name=f'best_seed_{ticker}_{st.session_state.window_size}w.csv', mime='text/csv')

# --- (The rest of the UI functions like render_analytics_tab and render_manual_seed_tab remain largely unchanged) ---
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
            # Check for necessary columns
            required_cols = ['window_number', 'timeline', 'max_net', 'action_sequence']
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
        
        if manual_start_date >= manual_end_date:
            st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
        
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
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้อง")
            return
            
        with st.spinner("กำลังดึงข้อมูลและจำลองการเทรด..."):
            start_str, end_str = manual_start_date.strftime('%Y-%m-%d'), manual_end_date.strftime('%Y-%m-%d')
            ticker_data = get_ticker_data(manual_ticker, start_str, end_str)
            if ticker_data.empty:
                st.error(f"ไม่พบข้อมูลสำหรับ {manual_ticker} ในช่วงวันที่ที่เลือก")
                return

            prices = ticker_data['Close'].to_numpy()
            num_trading_days = len(prices)
            st.info(f"📊 พบข้อมูลราคา {num_trading_days} วันทำการในช่วงที่เลือก")
            
            results = {}
            max_sim_len = 0
            
            for i, line_info in enumerate(st.session_state.manual_seed_lines):
                input_seed, size_seed, tail_seed = line_info['seed'], line_info['size'], line_info['tail']
                if tail_seed > size_seed:
                    st.error(f"Line {i+1}: Tail ({tail_seed}) ต้องไม่มากกว่า Size ({size_seed})")
                    return
                
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

            if not results:
                st.error("ไม่สามารถสร้างผลลัพธ์จาก Seed ที่กำหนดได้")
                return
            
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
            
            sorted_names = [name for name in results.keys() if name not in [Strategy.PERFECT_FORESIGHT, Strategy.REBALANCE_DAILY]]
            display_order = [Strategy.PERFECT_FORESIGHT] + sorted(sorted_names) + [Strategy.REBALANCE_DAILY]
            
            final_results_list = [
                {'name': name, 'net': results[name]['net'].iloc[-1]}
                for name in display_order if name in results and not results[name].empty
            ]
            
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
    st.set_page_config(page_title="Dynamic Strategy Tester", page_icon="🎯", layout="wide")
    st.markdown("### 🎯 Dynamic Strategy Tester (Numba Accelerated)")
    st.caption("เครื่องมือทดสอบกลยุทธ์การลงทุนที่เน้นความเร็ว, ความโปร่งใส, และประสิทธิภาพ")

    config = load_config()
    initialize_session_state(config)

    # ! UPDATED: ปรับปรุงรายการ Tab
    tab_list = [
        "⚙️ การตั้งค่า", 
        "🎯 Probabilistic Greedy Model",
        "🔍 Best Seed (Sliding Window)", 
        "📊 Advanced Analytics", 
        "🌱 Forward Rolling Comparator"
    ]
    tabs = st.tabs(tab_list)

    with tabs[0]: render_settings_tab(config)
    with tabs[1]: render_probabilistic_greedy_tab() # ! NEW
    with tabs[2]: render_test_tab()
    with tabs[3]: render_analytics_tab()
    with tabs[4]: render_manual_seed_tab(config)

    with st.expander("📖 คำอธิบายหลักการทำงานและปรัชญาของโมเดล (v.Refactored)"):
        st.markdown("""
        ### ปรัชญาการออกแบบ:
        แอปพลิเคชันนี้ถูกออกแบบใหม่โดยเน้นที่ **ความสง่างามทางคณิตศาสตร์, ความโปร่งใส, และประสิทธิภาพ** โดยมีเป้าหมายเพื่อสร้างกลยุทธ์ที่เข้าใกล้ผลลัพธ์ในอุดมคติ (`Perfect Foresight`) ให้ได้มากที่สุด

        ---
        
        ### 🎯 Probabilistic Greedy Model (โมเดลหลัก)
        นี่คือหัวใจของเวอร์ชันนี้ เป็นโมเดลที่ **ไม่ใช่ Black Box** และทำงานอย่างมีเหตุผล:
        1.  **มองราคาเทียบกับอดีต:** โมเดลจะคำนวณราคาอ้างอิง (เช่น Moving Average) เพื่อประเมินว่าราคาปัจจุบัน "ถูก" หรือ "แพง"
        2.  **สร้างความน่าจะเป็น:** หากราคาปัจจุบัน "ถูก" เมื่อเทียบกับราคาอ้างอิง "ความน่าจะเป็น" ที่จะเข้าซื้อ (Rebalance) ก็จะสูงขึ้นตามหลักคณิตศาสตร์ (ผ่าน Sigmoid Function)
        3.  **ใช้ Seed ตัดสินใจ:** โมเดลจะใช้ "เครื่องปั่นตัวเลข" (PRNG) ที่ควบคุมด้วย `Master Seed` เพื่อสร้างเลขสุ่ม ถ้าเลขสุ่มที่ได้ต่ำกว่าความน่าจะเป็น ก็จะเกิด Action ขึ้น
        
        **ข้อดี:**
        -   **Traceable:** สามารถคำนวณย้อนกลับได้ 100% เพียงใช้ Seed เดิม
        -   **Fast:** การคำนวณทั้งหมดใช้ NumPy Vectorization ทำให้เร็วมาก
        -   **Elegant:** ตรรกะเบื้องหลังเรียบง่ายและเข้าใจได้ ไม่ซับซ้อน
        -   **Effective:** มีเป้าหมายชัดเจนในการพยายามซื้อเมื่อราคาได้เปรียบ ซึ่งเป็นหัวใจของการทำกำไร

        ### 🔍 Best Seed Sliding Window (โมเดลดั้งเดิม)
        เป็นโมเดลที่ใช้วิธี **Brute-Force Search** คือการลองผิดลองถูกกับ Seed จำนวนมากในแต่ละช่วงเวลา (`Window`) เพื่อหา Seed ที่ให้ผลลัพธ์ดีที่สุด
        - **ข้อดี:** สามารถหาผลลัพธ์ที่ดีที่สุดในแต่ละ Window ได้จริง (ถ้ามีจำนวน Seed มากพอ)
        - **ข้อเสีย:** เป็น Black Box (เราไม่รู้ว่าทำไม Seed นี้ถึงดี), ช้ากว่า, และผลลัพธ์ที่ได้จากการ "เย็บ" แต่ละ Window อาจไม่ดีเท่าที่ควรในภาพรวม

        ### การเร่งความเร็วด้วย Numba
        ไม่ว่าจะเป็นโมเดลไหน ฟังก์ชันแกนกลางที่คำนวณผลกำไร/ขาดทุน (`_calculate_simulation_numba`) ถูกเร่งความเร็วด้วย **Numba (`@njit`)** ทำให้การจำลองแต่ละครั้งทำงานเร็วเทียบเท่าภาษา C ซึ่งเป็นกุญแจสำคัญที่ทำให้แอปพลิเคชันนี้ตอบสนองได้อย่างรวดเร็ว
        """)

if __name__ == "__main__":
    main()
