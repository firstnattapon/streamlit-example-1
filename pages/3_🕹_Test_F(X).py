import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import json
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
    REBALANCE_DAILY = "Rebalance Daily (Benchmark Min)"
    SLIDING_WINDOW = "Best Seed Sliding Window (Original)"
    FORWARD_ROLLING_FORESIGHT = "Forward Rolling Foresight (New Model)"

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
                "selected_ticker": "FFWM", "start_date": "2024-01-01", 
                "sliding_window_size": 30, "num_seeds": 10000, "max_workers": 8,
                "lookback_window": 60, "forward_window": 30
            }
        }

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
    
    # Sliding Window params
    if 'sliding_window_size' not in st.session_state:
        st.session_state.sliding_window_size = defaults.get('sliding_window_size', 30)
    if 'num_seeds' not in st.session_state:
        st.session_state.num_seeds = defaults.get('num_seeds', 10000)
    if 'max_workers' not in st.session_state:
        st.session_state.max_workers = defaults.get('max_workers', 8)
        
    # Forward Rolling Foresight params
    if 'lookback_window' not in st.session_state:
        st.session_state.lookback_window = defaults.get('lookback_window', 60)
    if 'forward_window' not in st.session_state:
        st.session_state.forward_window = defaults.get('forward_window', 30)


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
        return data
    except Exception as e:
        st.error(f"❌ ไม่สามารถดึงข้อมูล {ticker} ได้: {str(e)}")
        return pd.DataFrame()

# ! ACCELERATED: ฟังก์ชันแกนกลางที่ถูกเร่งความเร็วด้วย Numba
@njit(cache=True)
def _calculate_simulation_numba(action_array: np.ndarray, price_array: np.ndarray, fix: int = 1500) -> Tuple:
    """
    คำนวณผลลัพธ์การจำลองการเทรด (หัวใจของการคำนวณ)
    - ใช้ Numba @njit(cache=True) เพื่อคอมไพล์เป็น Machine Code ทำให้ทำงานเร็วมาก
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
        if action_array_calc[i] == 0:  # Hold
            amount[i] = amount[i-1]
            buffer[i] = 0.0
        else:  # Rebalance
            amount[i] = fix / curr_price
            buffer[i] = amount[i-1] * curr_price - fix

        cash[i] = cash[i-1] + buffer[i]
        asset_value[i] = amount[i] * curr_price
        sumusd[i] = cash[i] + asset_value[i]

    return buffer, sumusd, cash, asset_value, amount, refer

@lru_cache(maxsize=8192)
def calculate_optimized_cached(action_tuple: Tuple[int, ...], price_tuple: Tuple[float, ...], fix: int = 1500) -> Tuple:
    action_array = np.asarray(action_tuple, dtype=np.int32)
    price_array = np.asarray(price_tuple, dtype=np.float64)
    return _calculate_simulation_numba(action_array, price_array, fix)

def run_simulation(prices: List[float], actions: List[int], fix: int = 1500) -> pd.DataFrame:
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
        'refer': np.round(refer + initial_capital, 2), 'net': np.round(sumusd - refer - initial_capital, 2)
    })

# ==============================================================================
# 3. Strategy Action Generation
# ==============================================================================

# 3.1 Benchmark Strategy
def generate_actions_rebalance_daily(num_days: int) -> np.ndarray:
    """สร้าง Action สำหรับกลยุทธ์ Rebalance ทุกวัน (Min Performance)"""
    return np.ones(num_days, dtype=np.int32)

# 3.2 Perfect Foresight (Internal use for the new model)
def _generate_actions_perfect_foresight(prices: List[float], fix: int = 1500) -> np.ndarray:
    """สร้าง Action สำหรับกลยุทธ์ Perfect Foresight โดยใช้ Dynamic Programming (สำหรับใช้ภายใน)"""
    price_arr = np.asarray(prices, dtype=np.float64)
    n = len(price_arr)
    if n < 2: return np.ones(n, dtype=int)
    
    dp = np.zeros(n, dtype=np.float64)
    path = np.zeros(n, dtype=int)
    dp[0] = float(fix * 2)

    for i in range(1, n):
        j_indices = np.arange(i)
        profits = fix * ((price_arr[i] / price_arr[j_indices]) - 1)
        current_sumusd = dp[j_indices] + profits
        best_idx = np.argmax(current_sumusd)
        dp[i] = current_sumusd[best_idx]
        path[i] = j_indices[best_idx]
        
    actions = np.zeros(n, dtype=int)
    current_day = np.argmax(dp)
    while current_day > 0:
        actions[current_day] = 1
        current_day = path[current_day]
    actions[0] = 1
    return actions

# 3.3 Best Seed Sliding Window (Original Method)
def find_best_seed_for_window(prices_window: np.ndarray, num_seeds_to_try: int, max_workers: int) -> Tuple[int, float, np.ndarray]:
    window_len = len(prices_window)
    if window_len < 2: return 1, 0.0, np.ones(window_len, dtype=int)

    def evaluate_seed_batch(seed_batch: np.ndarray) -> List[Tuple[int, float]]:
        results = []
        prices_tuple = tuple(prices_window)
        for seed in seed_batch:
            rng = np.random.default_rng(seed)
            actions_window = rng.integers(0, 2, size=window_len)
            _, sumusd, _, _, _, refer = calculate_optimized_cached(tuple(actions_window), prices_tuple)
            net = sumusd[-1] - refer[-1] - sumusd[0] if len(sumusd) > 0 else -np.inf
            results.append((seed, net))
        return results

    best_seed, max_net = -1, -np.inf
    random_seeds = np.arange(num_seeds_to_try)
    batch_size = max(1, num_seeds_to_try // (max_workers * 4 if max_workers > 0 else 1))
    seed_batches = [random_seeds[j:j+batch_size] for j in range(0, len(random_seeds), batch_size)]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_seed_batch, batch) for batch in seed_batches}
        for future in as_completed(futures):
            for seed, final_net in future.result():
                if final_net > max_net:
                    max_net, best_seed = final_net, seed

    if best_seed >= 0:
        best_actions = np.random.default_rng(best_seed).integers(0, 2, size=window_len)
    else:
        best_seed, best_actions, max_net = 1, np.ones(window_len, dtype=int), 0.0
    
    best_actions[0] = 1
    return best_seed, max_net, best_actions

def generate_actions_sliding_window(ticker_data: pd.DataFrame, window_size: int, num_seeds_to_try: int, max_workers: int) -> Tuple[np.ndarray, pd.DataFrame]:
    prices = ticker_data['Close'].to_numpy()
    n = len(prices)
    final_actions = np.array([], dtype=int)
    window_details_list = []
    
    st.write(f"**กำลังประมวลผล: {Strategy.SLIDING_WINDOW}**")
    progress_bar = st.progress(0, text=f"ประมวลผล Sliding Windows (Random Seed)...")
    
    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        if len(prices_window) == 0: continue
        
        best_seed, max_net, best_actions = find_best_seed_for_window(prices_window, num_seeds_to_try, max_workers)
        final_actions = np.concatenate((final_actions, best_actions))
        
        start_date = ticker_data.index[start_index]; end_date = ticker_data.index[end_index-1]
        detail = {'window': i + 1, 'timeline': f"{start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}", 'best_seed': best_seed, 'max_net': round(max_net, 2)}
        window_details_list.append(detail)
        progress_bar.progress((i * window_size + len(prices_window)) / n)
        
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)

# 3.4 Forward Rolling Foresight (New Model)
def generate_actions_forward_rolling_foresight(ticker_data: pd.DataFrame, lookback_window: int, forward_window: int) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    สร้าง Action Sequence โดยใช้กลยุทธ์ Forward Rolling Foresight
    - ใช้ Perfect Foresight strategy จาก `lookback_window` วันก่อนหน้า
    - นำ Action sequence ที่ได้ (DNA) มาใช้กับ `forward_window` วันถัดไป
    """
    prices = ticker_data['Close'].to_numpy()
    n = len(prices)
    final_actions = np.array([], dtype=int)
    window_details_list = []

    st.write(f"**กำลังประมวลผล: {Strategy.FORWARD_ROLLING_FORESIGHT}**")
    progress_bar = st.progress(0, text="ประมวลผล Forward Rolling Windows...")

    for i in range(0, n, forward_window):
        # กำหนดช่วงเวลา Lookback และ Forward
        lookback_start = max(0, i - lookback_window)
        lookback_end = i
        forward_start = i
        forward_end = min(i + forward_window, n)

        if lookback_start >= lookback_end: # กรณีเริ่มต้นที่ยังไม่มีข้อมูล lookback พอ
            # ใช้ Rebalance Daily เป็นค่าเริ่มต้นสำหรับช่วงแรก
            learned_actions = np.ones(forward_end - forward_start, dtype=int)
            dna_source = "Initial (Rebalance Daily)"
        else:
            prices_lookback = prices[lookback_start:lookback_end]
            # เรียนรู้ Action ที่ดีที่สุดจากช่วง Lookback
            learned_actions = _generate_actions_perfect_foresight(prices_lookback.tolist())
            lookback_s_date = ticker_data.index[lookback_start]
            lookback_e_date = ticker_data.index[lookback_end-1]
            dna_source = f"Lookback {lookback_s_date:%Y-%m-%d} to {lookback_e_date:%Y-%m-%d} ({len(learned_actions)} days)"

        # นำ Action ที่เรียนรู้มาใช้กับช่วง Forward
        actions_to_apply = learned_actions[:(forward_end - forward_start)]
        final_actions = np.concatenate((final_actions, actions_to_apply))
        
        # บันทึกรายละเอียด
        forward_s_date = ticker_data.index[forward_start]
        forward_e_date = ticker_data.index[forward_end-1]
        detail = {
            'window': (i // forward_window) + 1,
            'timeline': f"{forward_s_date:%Y-%m-%d} to {forward_e_date:%Y-%m-%d}",
            'dna_source': dna_source,
            'actions_applied': len(actions_to_apply)
        }
        window_details_list.append(detail)
        progress_bar.progress(forward_end / n, text=f"Rolling Window {(i // forward_window) + 1}...")
    
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)


# ==============================================================================
# 4. UI Rendering Functions
# ==============================================================================
def render_main_tab():
    """แสดงผล UI สำหรับ Tab การตั้งค่าและการทดสอบหลัก"""
    
    # --- Settings ---
    st.subheader("1. กำหนดค่าพารามิเตอร์")
    with st.container(border=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            asset_list = config.get('assets', ['FFWM'])
            try: default_index = asset_list.index(st.session_state.test_ticker)
            except ValueError: default_index = 0
            st.session_state.test_ticker = st.selectbox("เลือก Ticker", options=asset_list, index=default_index)
        
        with col2:
            c1, c2 = st.columns(2)
            st.session_state.start_date = c1.date_input("วันที่เริ่มต้น", value=st.session_state.start_date)
            st.session_state.end_date = c2.date_input("วันที่สิ้นสุด", value=st.session_state.end_date)
            if st.session_state.start_date >= st.session_state.end_date:
                st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")

        st.divider()
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.markdown(f"**พารามิเตอร์สำหรับ `{Strategy.SLIDING_WINDOW}`**")
            st.session_state.sliding_window_size = st.number_input("ขนาด Window (วัน)", min_value=2, value=st.session_state.sliding_window_size, key="sw_size")
            st.session_state.num_seeds = st.number_input("จำนวน Seeds ต่อ Window", min_value=100, value=st.session_state.num_seeds, format="%d", key="sw_seeds")
            st.session_state.max_workers = st.number_input("จำนวน Workers (CPU Cores)", min_value=1, max_value=16, value=st.session_state.max_workers, key="sw_workers")

        with p_col2:
            st.markdown(f"**พารามิเตอร์สำหรับ `{Strategy.FORWARD_ROLLING_FORESIGHT}`**")
            st.session_state.lookback_window = st.number_input("Lookback Window (วัน)", min_value=5, value=st.session_state.lookback_window, help="จำนวนวันที่ใช้มองย้อนกลับไปเพื่อเรียนรู้รูปแบบที่ดีที่สุด")
            st.session_state.forward_window = st.number_input("Forward Window (วัน)", min_value=1, value=st.session_state.forward_window, help="จำนวนวันที่นำรูปแบบที่เรียนรู้มาใช้")

    # --- Run Button & Results ---
    st.subheader("2. เริ่มการทดสอบและดูผลลัพธ์")
    if st.button("🚀 เริ่มทดสอบและเปรียบเทียบกลยุทธ์", type="primary", use_container_width=True):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้อง")
            return
            
        ticker = st.session_state.test_ticker
        start_str, end_str = st.session_state.start_date.strftime('%Y-%m-%d'), st.session_state.end_date.strftime('%Y-%m-%d')
        
        st.info(f"กำลังดึงข้อมูลสำหรับ **{ticker}** | {start_str} ถึง {end_str}")
        ticker_data = get_ticker_data(ticker, start_str, end_str)
        
        if ticker_data.empty:
            st.error("ไม่พบข้อมูลสำหรับ Ticker และช่วงวันที่ที่เลือก")
            return
            
        prices_np = ticker_data['Close'].to_numpy()
        prices_list = prices_np.tolist()
        num_days = len(prices_np)
        st.success(f"ดึงข้อมูลสำเร็จ! พบข้อมูล {num_days} วันทำการ")

        with st.spinner("กำลังคำนวณกลยุทธ์ต่างๆ (Core calculation is Numba-accelerated)..."):
            # Generate actions for all strategies
            actions_min = generate_actions_rebalance_daily(num_days)
            actions_sliding, df_sliding_details = generate_actions_sliding_window(
                ticker_data, st.session_state.sliding_window_size, st.session_state.num_seeds, st.session_state.max_workers
            )
            actions_forward, df_forward_details = generate_actions_forward_rolling_foresight(
                ticker_data, st.session_state.lookback_window, st.session_state.forward_window
            )

            # Run simulations
            results = {}
            strategy_map = {
                Strategy.FORWARD_ROLLING_FORESIGHT: actions_forward.tolist(),
                Strategy.SLIDING_WINDOW: actions_sliding.tolist(),
                Strategy.REBALANCE_DAILY: actions_min.tolist()
            }
            
            for name, actions in strategy_map.items():
                df = run_simulation(prices_list, actions)
                if not df.empty:
                    df.index = ticker_data.index[:len(df)]
                results[name] = df
        
        st.success("การทดสอบเสร็จสมบูรณ์!")
        st.divider()

        # --- Display Results ---
        st.subheader("📈 ผลการเปรียบเทียบกำไรสุทธิ (Net Profit)")
        
        # Chart
        chart_data = pd.DataFrame(index=ticker_data.index)
        for name, df in results.items():
            if not df.empty and 'net' in df.columns:
                chart_data[name] = df['net']
        chart_data.ffill(inplace=True)
        st.line_chart(chart_data)

        # Final Metrics
        st.subheader("📊 สรุปผลลัพธ์สุดท้าย")
        final_metrics_cols = st.columns(len(results))
        sorted_results = sorted(results.items(), key=lambda item: item[1]['net'].iloc[-1] if not item[1].empty else -np.inf, reverse=True)

        for i, (name, df) in enumerate(sorted_results):
            final_net = df['net'].iloc[-1] if not df.empty else 0
            final_metrics_cols[i].metric(name, f"${final_net:,.2f}")
        
        st.divider()
        # Details Tables
        st.subheader("📄 รายละเอียดการทำงานของแต่ละ Window")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.markdown(f"**`{Strategy.FORWARD_ROLLING_FORESIGHT}`**")
            st.dataframe(df_forward_details, use_container_width=True)
        with res_col2:
            st.markdown(f"**`{Strategy.SLIDING_WINDOW}`**")
            st.dataframe(df_sliding_details, use_container_width=True)

def render_explanation_tab():
    """แสดงผล UI สำหรับ Tab คำอธิบายโมเดลใหม่"""
    st.header("📖 แนวคิดของโมเดลใหม่: Forward Rolling Foresight")
    st.markdown("""
    โมเดล **Forward Rolling Foresight** ถูกออกแบบมาเพื่อแก้ปัญหาของกลยุทธ์แบบเดิมๆ โดยมีเป้าหมายเพื่อสร้างกลยุทธ์ที่ **มีประสิทธิภาพสูง, โปร่งใส, และลดปัญหา Overfitting**

    #### ปัญหาของกลยุทธ์เดิม
    1.  **Perfect Foresight (Max)**: ให้ผลตอบแทนสูงสุดในทางทฤษฎี แต่ **เป็นไปไม่ได้ในโลกจริง** เพราะมัน "รู้อนาคต" ทำให้เกิด Overfitting กับข้อมูลที่ใช้ทดสอบ 100%
    2.  **Best Seed Sliding Window**: เป็นการ "เดาสุ่ม" (Brute-force) หา `seed` ที่ดีที่สุดในแต่ละช่วงเวลาสั้นๆ ซึ่งเป็น **"กล่องดำ" (Black Box)** ที่เราไม่เข้าใจว่าทำไม seed นั้นถึงดี และยังคงเสี่ยงต่อการ Overfitting ในแต่ละ window เล็กๆ

    ---

    ### หลักการทำงานของ Forward Rolling Foresight

    โมเดลนี้ทำงานแบบ Walk-Forward ซึ่งเป็นวิธีมาตรฐานในวงการ Quantitative Finance เพื่อทดสอบความทนทานของกลยุทธ์ โดยมีขั้นตอนดังนี้:

    1.  **แบ่งข้อมูล**: แบ่งช่วงเวลาทั้งหมดออกเป็นส่วนๆ ที่ไม่ซ้อนทับกัน เรียกว่า **Forward Window** (เช่น ทุกๆ 30 วัน)
    2.  **เรียนรู้จากอดีต (Lookback)**: ณ จุดเริ่มต้นของแต่ละ `Forward Window`, โมเดลจะมองย้อนกลับไปในอดีตเป็นระยะเวลาที่กำหนด เรียกว่า **Lookback Window** (เช่น 60 วันที่ผ่านมา)
    3.  **ค้นหากลยุทธ์ที่ดีที่สุด**: ในช่วง `Lookback Window` นี้ โมเดลจะใช้ Dynamic Programming เพื่อคำนวณหากลยุทธ์ที่สมบูรณ์แบบที่สุด (Perfect Foresight) สำหรับ *ข้อมูลในอดีต* นั้น ผลลัพธ์ที่ได้คือ **Action Sequence (หรือ "DNA")** ที่ดีที่สุดสำหรับช่วงเวลาที่ผ่านมา
    4.  **นำไปใช้กับอนาคต (Forward Rolling)**: โมเดลจะนำ `Action Sequence` ที่เรียนรู้จากข้อ 3 มาใช้กับการเทรดในช่วง `Forward Window` ถัดไป
    5.  **ทำซ้ำ**: ทำซ้ำขั้นตอนที่ 2-4 ไปเรื่อยๆ จนสิ้นสุดช่วงเวลาทดสอบ

    

    #### ทำไมวิธีนี้ถึงดีกว่า?
    -   **สง่างามทางคณิตศาสตร์ (Mathematically Elegant)**: ไม่มีการสุ่มเดา แต่ใช้หลักการที่ชัดเจนและคำนวณย้อนกลับได้ คือ "การเลียนแบบพฤติกรรมที่ดีที่สุดในอดีตอันใกล้"
    -   **ไม่ใช่ Black Box**: เราสามารถตรวจสอบได้เสมอว่าในแต่ละช่วงเวลา โมเดลเรียนรู้ "DNA" หน้าตาแบบไหนมาจากข้อมูลช่วงใด
    -   **ลด Overfitting**: การตัดสินใจเทรดในแต่ละ `Forward Window` มาจากการเรียนรู้ข้อมูล `Lookback Window` ที่เกิดขึ้นก่อนหน้าเท่านั้น ซึ่งจำลองสถานการณ์การเทรดจริงที่ต้องใช้ข้อมูลในอดีตมาคาดการณ์อนาคต
    -   **มีเสถียรภาพ**: ให้ผลลัพธ์เหมือนเดิมทุกครั้งที่รันบนข้อมูลชุดเดียวกัน ไม่ขึ้นอยู่กับค่า `seed` แบบสุ่ม
    """)

# ==============================================================================
# 5. Main Application
# ==============================================================================
def main():
    """
    ฟังก์ชันหลักในการรัน Streamlit Application
    """
    st.set_page_config(page_title="Forward Rolling Foresight", page_icon="🎯", layout="wide")
    st.title("🎯 Forward Rolling Foresight vs. Best Seed")
    st.caption("เครื่องมือเปรียบเทียบกลยุทธ์ใหม่ (Forward Rolling) กับกลยุทธ์ดั้งเดิม (Best Seed) ที่เร่งความเร็วด้วย Numba")

    # โหลดการตั้งค่าและเตรียม Session State
    global config
    config = load_config()
    initialize_session_state(config)

    # สร้าง Tabs
    tab1, tab2 = st.tabs(["⚙️ การตั้งค่าและการทดสอบ", "📖 คำอธิบายโมเดลใหม่"])

    with tab1:
        render_main_tab()
    
    with tab2:
        render_explanation_tab()

if __name__ == "__main__":
    main()
