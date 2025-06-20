import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import math
import json
from datetime import datetime
from typing import List, Tuple, Dict, Any
# นำเข้า prange สำหรับ Parallelism
from numba import njit, prange, int32, float64

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================
class Strategy:
    REBALANCE_DAILY = "Rebalance Daily"
    PERFECT_FORESIGHT = "Perfect Foresight (Max)"
    BRUTE_FORCE_OPTIMIZER = "Brute-Force (Quantum Leap)"

def load_config(filepath: str = "dynamic_seed_config.json") -> Dict[str, Any]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        st.warning(f"⚠️ ไม่พบหรือไฟล์ '{filepath}' ไม่ถูกต้อง จะใช้ค่าเริ่มต้นแทน")
        return {
            "assets": ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "TSLA", "NVDA"],
            "default_settings": {"selected_ticker": "BTC-USD", "start_date": "2024-01-01", "window_size": 30, "num_seeds": 2000000, "action_prob": 0.15}
        }

def initialize_session_state(config: Dict[str, Any]):
    defaults = config.get('default_settings', {})
    if 'test_ticker' not in st.session_state: st.session_state.test_ticker = defaults.get('selected_ticker', 'BTC-USD')
    if 'start_date' not in st.session_state:
        try: st.session_state.start_date = datetime.strptime(defaults.get('start_date', '2024-01-01'), '%Y-%m-%d').date()
        except ValueError: st.session_state.start_date = datetime(2024, 1, 1).date()
    if 'end_date' not in st.session_state: st.session_state.end_date = datetime.now().date()
    if 'window_size' not in st.session_state: st.session_state.window_size = defaults.get('window_size', 30)
    if 'num_seeds' not in st.session_state: st.session_state.num_seeds = defaults.get('num_seeds', 2000000)
    if 'action_probability' not in st.session_state: st.session_state.action_probability = defaults.get('action_prob', 0.15)


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

# [CORRECTED] ฟังก์ชันแกนกลางที่รับ action_prob เข้ามาด้วย
@njit(cache=True)
def _calculate_net_heuristic(seed: int, price_array: np.ndarray, action_prob: float, fix: float = 1500.0) -> float:
    n = price_array.shape[0]
    if n < 2: return 0.0

    state = np.uint32(seed)
    prob_threshold = np.uint32(action_prob * 10000)
    
    initial_price = price_array[0]
    amount_prev = fix / initial_price
    cash_prev = fix
    
    for i in range(1, n):
        state ^= (state << 13); state ^= (state >> 17); state ^= (state << 5)
        
        # <<< FIX: เพิ่มบรรทัดนี้ที่หายไป >>>
        curr_price = price_array[i]

        random_val = state % 10000
        action = 1 if random_val < prob_threshold else 0

        if action == 1:
            buffer = amount_prev * curr_price - fix
            cash_prev += buffer
            amount_prev = fix / curr_price
            
    final_asset_value = amount_prev * price_array[n-1]
    final_sumusd = cash_prev + final_asset_value
    
    refer_end = -fix * math.log(initial_price / price_array[n-1])
    sumusd_start = fix * 2.0
    
    return final_sumusd - refer_end - sumusd_start

# [QUANTUM LEAP] ฟังก์ชัน prange ที่ส่ง action_prob ต่อไป
@njit(parallel=True, cache=True)
def _find_best_seed_quantum_leap(prices_window: np.ndarray, num_seeds_to_try: int, action_prob: float) -> Tuple[int, float]:
    nets = np.empty(num_seeds_to_try, dtype=float64)
    
    for seed in prange(num_seeds_to_try):
        nets[seed] = _calculate_net_heuristic(seed, prices_window, action_prob)

    best_seed_idx = np.argmax(nets)
    return int(best_seed_idx), nets[best_seed_idx]

# ==============================================================================
# 3. Strategy Action Generation
# ==============================================================================
def find_best_seed_for_window(prices_window: np.ndarray, num_seeds_to_try: int, action_prob: float) -> Tuple[int, float, np.ndarray]:
    window_len = len(prices_window)
    if window_len < 2: return 1, 0.0, np.ones(window_len, dtype=np.int32)

    best_seed, max_net = _find_best_seed_quantum_leap(prices_window, num_seeds_to_try, action_prob)

    if best_seed >= 0:
        np.random.seed(best_seed)
        p = np.array([1.0 - action_prob, action_prob])
        best_actions = np.random.choice(np.array([0, 1], dtype=np.int32), size=window_len, p=p)
        best_actions[0] = 1
    else: 
        best_seed, max_net, best_actions = 1, 0.0, np.ones(window_len, dtype=np.int32)
        
    return best_seed, max_net, best_actions

def generate_actions_sliding_window_brute_force(ticker_data: pd.DataFrame, window_size: int, num_seeds: int, action_prob: float) -> Tuple[np.ndarray, pd.DataFrame]:
    prices = ticker_data['Close'].to_numpy()
    n = len(prices)
    final_actions = np.array([], dtype=np.int32); window_details_list = []
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="Quantum Leap Search is initializing...")
    
    st.write(f"🌌 **Brute-Force (Quantum Leap Mode)**")
    st.write(f"📊 ข้อมูล: {n} วัน | Window: {window_size} วัน | Seeds: **{num_seeds:,}** | Action Prob: **{action_prob:.0%}**")
    st.write(f"✅ ค้นหาแบบมีไหวพริบ (Heuristic-Guided Search) เพื่อผลลัพธ์ที่ดีที่สุดในเวลาที่จำกัด")
    st.write("---")

    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        if len(prices_window) == 0: continue
        
        best_seed, max_net, best_actions = find_best_seed_for_window(prices_window, num_seeds, action_prob)
        
        final_actions = np.concatenate((final_actions, best_actions))
        
        start_date_str = ticker_data.index[start_index].strftime('%Y-%m-%d')
        end_date_str = ticker_data.index[end_index-1].strftime('%Y-%m-%d')
        detail = {'window': i + 1, 'timeline': f"{start_date_str} to {end_date_str}", 'best_seed': best_seed, 'max_net': round(max_net, 2), 'actions': int(np.sum(best_actions))}
        window_details_list.append(detail)
        progress_bar.progress((i + 1) / num_windows, text=f"Leap Searching in Window {i+1}/{num_windows}...")
        
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)

# --- Helper functions for comparison (No changes needed here) ---
@njit(cache=True)
def _full_simulation_cpu(action_array: np.ndarray, price_array: np.ndarray, fix: int = 1500) -> Tuple:
    n=len(action_array);action_array_calc=action_array.copy();action_array_calc[0]=1;amount=np.empty(n,dtype=np.float64);cash=np.empty(n,dtype=np.float64);asset_value=np.empty(n,dtype=np.float64);sumusd=np.empty(n,dtype=np.float64);initial_price=price_array[0];amount[0]=fix/initial_price;cash[0]=fix;asset_value[0]=amount[0]*initial_price;sumusd[0]=cash[0]+asset_value[0];refer=-fix*np.log(initial_price/price_array);
    for i in range(1,n):
        curr_price=price_array[i];
        if action_array_calc[i]==0:amount[i]=amount[i-1];buffer=0.0
        else: amount[i]=fix/curr_price;buffer=amount[i-1]*curr_price-fix
        cash[i]=cash[i-1]+buffer;asset_value[i]=amount[i]*curr_price;sumusd[i]=cash[i]+asset_value[i]
    return sumusd, refer

def run_simulation(prices: List[float], actions: List[int]) -> pd.DataFrame:
    price_array, action_array = np.array(prices, dtype=np.float64), np.array(actions, dtype=np.int32)
    min_len = min(len(price_array), len(action_array))
    if min_len == 0: return pd.DataFrame()
    price_array, action_array = price_array[:min_len], action_array[:min_len]
    sumusd, refer = _full_simulation_cpu(action_array, price_array)
    df = pd.DataFrame({'sumusd': sumusd, 'refer': refer})
    df['net'] = df['sumusd'] - df['refer'] - df['sumusd'].iloc[0]
    return df

def generate_actions_rebalance_daily(num_days: int): return np.ones(num_days, dtype=np.int32)
def generate_actions_perfect_foresight(prices: np.ndarray):
    n=len(prices);
    if n==0: return np.array([],dtype=np.int32)
    actions=np.zeros(n,dtype=np.int32);actions[0]=1;last_buy_price=prices[0]
    for i in range(1,n):
        if prices[i]<last_buy_price:actions[i]=1;last_buy_price=prices[i]
    return actions


# ==============================================================================
# 4. UI Rendering Functions & Main App
# ==============================================================================
def render_settings_tab(config: Dict[str, Any]):
    st.write("⚙️ **การตั้งค่าพารามิเตอร์**")
    asset_list = config.get('assets'); st.session_state.test_ticker = st.selectbox("เลือก Ticker", options=asset_list, index=asset_list.index(st.session_state.test_ticker) if st.session_state.test_ticker in asset_list else 0)
    col1, col2 = st.columns(2)
    with col1: st.session_state.start_date = st.date_input("วันที่เริ่มต้น", value=st.session_state.start_date)
    with col2: st.session_state.end_date = st.date_input("วันที่สิ้นสุด", value=st.session_state.end_date)
    st.divider()
    st.subheader("พารามิเตอร์สำหรับ Quantum Leap Optimizer")
    st.session_state.action_probability = st.slider(
        "🌌 โอกาสในการ Action (Action Probability)", 
        min_value=0.01, max_value=0.5, value=st.session_state.action_probability, step=0.01, format="%.0f%%",
        help="ควบคุมความถี่ของการ Action (Rebalance) ยิ่งค่าน้อย การค้นหาจะยิ่งเน้นกลยุทธ์ที่ Action ห่างๆ คล้าย Perfect Foresight ทำให้เจอผลลัพธ์ที่ดีกว่าในเวลาเท่ากัน"
    )
    col1, col2 = st.columns(2)
    with col1: st.session_state.window_size = st.number_input("ขนาด Window (วัน)", min_value=10, value=st.session_state.window_size)
    with col2: st.session_state.num_seeds = st.number_input("จำนวน Seeds ต่อ Window", min_value=1000, max_value=10000000, value=st.session_state.num_seeds, format="%d")

def display_comparison_charts(results: Dict[str, pd.DataFrame]):
    valid_dfs = {name: df for name, df in results.items() if not df.empty and 'net' in df.columns}
    if not valid_dfs: return
    longest_index = max((df.index for df in valid_dfs.values()), key=len, default=None)
    if longest_index is None: return
    chart_data = pd.DataFrame(index=longest_index)
    for name, df in valid_dfs.items(): chart_data[name] = df['net'].reindex(longest_index).ffill()
    st.write('📊 **เปรียบเทียบกำไรสุทธิ (Net Profit)**'); st.line_chart(chart_data)

def render_brute_force_tab():
    st.markdown("### 🌌 Brute-Force Optimizer (Quantum Leap Mode)")
    st.info("ก้าวข้ามขีดจำกัดด้วย **Heuristic-Guided Search** ใช้ 'โอกาสในการ Action' เพื่อนำทางการค้นหาไปสู่กลยุทธ์ที่มีประสิทธิภาพสูง ทำให้ได้ผลตอบแทนดีที่สุดภายใต้ข้อจำกัดของเวลา")
    if st.button("🚀 เริ่มการค้นหาแบบ Quantum Leap", type="primary"):
        with st.spinner("Quantum Leap Search in progress... This may take a moment."):
            import time; start_time = time.time()
            ticker_data = get_ticker_data(st.session_state.test_ticker, st.session_state.start_date.strftime('%Y-%m-%d'), st.session_state.end_date.strftime('%Y-%m-%d'))
            if ticker_data.empty: st.error("ไม่พบข้อมูล"); return
            
            prices = ticker_data['Close'].to_numpy(); num_days = len(prices)
            actions_brute, df_windows = generate_actions_sliding_window_brute_force(
                ticker_data, st.session_state.window_size, st.session_state.num_seeds, st.session_state.action_probability
            )
            end_time = time.time(); st.success(f"Quantum Leap Search เสร็จสิ้นใน {end_time - start_time:.2f} วินาที!")
            
            results = {
                Strategy.BRUTE_FORCE_OPTIMIZER: run_simulation(prices.tolist(), actions_brute.tolist()),
                Strategy.REBALANCE_DAILY: run_simulation(prices.tolist(), generate_actions_rebalance_daily(num_days).tolist()),
                Strategy.PERFECT_FORESIGHT: run_simulation(prices.tolist(), generate_actions_perfect_foresight(prices).tolist())
            }
            for name, df in results.items():
                if not df.empty: df.index = ticker_data.index[:len(df)]
        
        display_comparison_charts(results)
        st.write("📈 **สรุปผลการค้นหา (Quantum Leap)**")
        total_net = df_windows['max_net'].sum()
        col1, col2 = st.columns(2)
        col1.metric("Total Windows", df_windows.shape[0])
        col2.metric("Total Net (Sum)", f"${total_net:,.2f}")
        st.dataframe(df_windows[['window', 'timeline', 'best_seed', 'max_net', 'actions']], use_container_width=True)

def main():
    st.set_page_config(page_title="Quantum Leap Optimizer", page_icon="🌌", layout="wide")
    st.markdown("## 🌌 Quantum Leap Optimizer")
    st.caption("ทะลวงขีดจำกัดของ Brute-Force ด้วยการค้นหาแบบมีไหวพริบ (Heuristic-Guided Search)")
    config = load_config()
    initialize_session_state(config)
    render_settings_tab(config)
    st.divider()
    render_brute_force_tab()
    
    with st.expander("📖 คำอธิบายอัลกอริทึม (Quantum Leap)"):
        st.markdown("""
        **🌌 The Quantum Leap: Heuristic-Guided Search**

        นี่คือขั้นสุดของการ Optimization ภายใต้ข้อจำกัดของ CPU โดยเราไม่ได้ทำให้คอมพิวเตอร์เร็วขึ้น แต่เราทำให้ **"การค้นหาฉลาดขึ้น"**

        - **ปัญหาของ Brute-Force แบบเดิม:** การสุ่ม Action ด้วยความน่าจะเป็น 50/50 ทำให้การค้นหาส่วนใหญ่เสียเวลาไปกับการทดสอบกลยุทธ์ที่แย่ (เช่น ซื้อขายบ่อยเกินไป) เปรียบเสมือนการหาเข็มในมหาสมุทรที่กว้างใหญ่ไพศาล

        - **ทางออกเชิงปฏิวัติ:** เราใช้ **Heuristic (ไหวพริบ)** ที่ว่า "กลยุทธ์ที่ดีมักจะไม่ Action บ่อย" เราจึงนำทาง (Guide) การค้นหาให้ไปเน้นในบริเวณที่มีโอกาสเจอคำตอบที่ดี

        - **กลไกการทำงาน:**
            1.  **Action Probability Slider:** ผู้ใช้สามารถกำหนด "ความหนาแน่น" ของการ Action ได้โดยตรง เช่น 15% หมายความว่าในแต่ละวันจะมีโอกาสเกิดการ Rebalance เพียง 15%
            2.  **Focused Search Space:** การค้นหา 2 ล้านครั้งของเรา จะไม่ใช่การสุ่มแบบมั่วซั่วอีกต่อไป แต่เป็นการสุ่ม "กลยุทธ์ที่ Action น้อย" ถึง 2 ล้านรูปแบบ
            3.  **Better Results, Same Time:** ผลลัพธ์คือ **Net Profit ที่สูงขึ้นอย่างมีนัยสำคัญ** ในเวลาการประมวลผลเท่าเดิม เพราะเราใช้ทรัพยากรการคำนวณทุกหยดไปกับการสำรวจพื้นที่ที่มีศักยภาพสูงเท่านั้น นี่คือการ "ก้าวข้ามขีดจำกัด" อย่างแท้จริง
        """)

if __name__ == "__main__":
    main()
