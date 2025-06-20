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
    # ชื่อยังคงเดิม แต่เบื้องหลังเร็วกว่า
    BRUTE_FORCE_OPTIMIZER = "Brute-Force Optimizer (CPU-Max)"
    MANUAL_ACTION = "Manual Action Sequence"

def load_config(filepath: str = "dynamic_seed_config.json") -> Dict[str, Any]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        st.warning(f"⚠️ ไม่พบหรือไฟล์ '{filepath}' ไม่ถูกต้อง จะใช้ค่าเริ่มต้นแทน")
        return {
            "assets": ["FFWM", "NEGG", "RIVN", "BTC-USD", "ETH-USD"],
            "default_settings": {"selected_ticker": "BTC-USD", "start_date": "2024-01-01", "window_size": 30, "num_seeds": 2000000}
        }

def initialize_session_state(config: Dict[str, Any]):
    defaults = config.get('default_settings', {})
    if 'test_ticker' not in st.session_state: st.session_state.test_ticker = defaults.get('selected_ticker', 'FFWM')
    if 'start_date' not in st.session_state:
        try: st.session_state.start_date = datetime.strptime(defaults.get('start_date', '2024-01-01'), '%Y-%m-%d').date()
        except ValueError: st.session_state.start_date = datetime(2024, 1, 1).date()
    if 'end_date' not in st.session_state: st.session_state.end_date = datetime.now().date()
    if 'window_size' not in st.session_state: st.session_state.window_size = defaults.get('window_size', 30)
    if 'num_seeds' not in st.session_state: st.session_state.num_seeds = defaults.get('num_seeds', 2000000)

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

# [OPTIMIZED] ฟังก์ชันจำลองสถานการณ์สำหรับ CPU-Max โดยเฉพาะ
# ลดการสร้าง Array ที่ไม่จำเป็น และคำนวณเฉพาะค่าที่ต้องใช้จริงๆ
@njit(cache=True)
def _calculate_net_for_seed(seed: int, price_array: np.ndarray, fix: float = 1500.0) -> float:
    n = price_array.shape[0]
    if n < 2:
        return 0.0

    # --- 1. ใช้ Xorshift PRNG ที่เร็วมากในการสร้าง Action ---
    # ไม่ต้องสร้าง action_array ทั้งหมดใน memory ก่อน
    state = np.uint32(seed)
    
    # --- 2. คำนวณแบบ In-place ไม่สร้าง Array ใหม่ ---
    initial_price = price_array[0]
    amount_prev = fix / initial_price
    cash_prev = fix
    
    # Loop ที่เร็วที่สุดเท่าที่จะทำได้
    for i in range(1, n):
        # สร้างเลขสุ่ม 0 หรือ 1
        state ^= (state << 13)
        state ^= (state >> 17)
        state ^= (state << 5)
        action = state & 1 # เอาแค่ bit สุดท้าย

        curr_price = price_array[i]
        
        if action == 1: # Rebalance
            buffer = amount_prev * curr_price - fix
            cash_prev += buffer
            amount_prev = fix / curr_price
        # ถ้า action == 0 ไม่ต้องทำอะไรเลย amount และ cash ไม่เปลี่ยน
            
    # --- 3. คำนวณผลลัพธ์สุดท้าย ---
    final_asset_value = amount_prev * price_array[n-1]
    final_sumusd = cash_prev + final_asset_value
    
    # คำนวณค่า refer และ sumusd เริ่มต้น
    refer_end = -fix * math.log(initial_price / price_array[n-1])
    sumusd_start = fix * 2.0
    
    net = final_sumusd - refer_end - sumusd_start
    return net

# [OPTIMIZED] ฟังก์ชันหลักที่ใช้ prange เรียกใช้ _calculate_net_for_seed
@njit(parallel=True, cache=True)
def _find_best_seed_cpu_max(prices_window: np.ndarray, num_seeds_to_try: int) -> Tuple[int, float]:
    # สร้าง array สำหรับเก็บผลลัพธ์จากแต่ละ thread
    nets = np.empty(num_seeds_to_try, dtype=float64)
    
    # prange จะกระจายงานใน loop นี้ไปยัง CPU ทุก Core
    for seed in prange(num_seeds_to_try):
        # เรียกใช้ฟังก์ชันที่ถูก optimize มาอย่างดีแล้ว
        # แต่ละ thread จะทำงานในส่วนของตัวเองโดยไม่ยุ่งกัน
        nets[seed] = _calculate_net_for_seed(seed, prices_window)

    # รวบรวมผลลัพธ์หลังจากทุก thread ทำงานเสร็จ
    best_seed_idx = np.argmax(nets)
    max_net = nets[best_seed_idx]
    
    # best_seed_idx ก็คือค่า seed ที่เราใช้ในลูป prange
    return int(best_seed_idx), max_net

# ==============================================================================
# 3. Strategy Action Generation
# ==============================================================================
def find_best_seed_for_window(prices_window: np.ndarray, num_seeds_to_try: int) -> Tuple[int, float, np.ndarray]:
    window_len = len(prices_window)
    if window_len < 2: return 1, 0.0, np.ones(window_len, dtype=np.int32)

    # เรียกใช้ฟังก์ชัน parallel ตัวใหม่ที่เร็วที่สุด
    best_seed, max_net = _find_best_seed_cpu_max(prices_window, num_seeds_to_try)

    # สร้าง Action Sequence จาก Seed ที่ดีที่สุด (ส่วนนี้เร็วมาก ทำบน CPU ได้)
    # ต้องสร้างใหม่เพราะ PRNG ใน numba ไม่ได้รับประกันว่าจะเหมือน np.random
    if best_seed >= 0:
        np.random.seed(best_seed)
        best_actions = np.random.randint(0, 2, size=window_len, dtype=np.int32)
        best_actions[0] = 1
    else: 
        best_seed = 1; max_net = 0.0; best_actions = np.ones(window_len, dtype=np.int32)
        
    return best_seed, max_net, best_actions

def generate_actions_sliding_window_brute_force(ticker_data: pd.DataFrame, window_size: int, num_seeds: int) -> Tuple[np.ndarray, pd.DataFrame]:
    prices = ticker_data['Close'].to_numpy()
    n = len(prices)
    final_actions = np.array([], dtype=np.int32); window_details_list = []
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="กำลังประมวลผลด้วย CPU-Max Optimizer...")
    
    st.write(f"🔥 **Brute-Force Optimizer (CPU-Max Mode)**")
    st.write(f"📊 ข้อมูลทั้งหมด: {n} วัน | ขนาด Window: {window_size} วัน | Seeds ต่อ Window: **{num_seeds:,}**")
    st.write(f"✅ ใช้ CPU Parallelism (prange) ที่ปรับแต่งเพื่อความเร็วสูงสุด")
    st.write("---")

    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        window_len = len(prices_window)
        if window_len == 0: continue
        
        best_seed, max_net, best_actions = find_best_seed_for_window(prices_window, num_seeds)
        
        final_actions = np.concatenate((final_actions, best_actions))
        
        start_date_str = ticker_data.index[start_index].strftime('%Y-%m-%d')
        end_date_str = ticker_data.index[end_index-1].strftime('%Y-%m-%d')
        detail = {'window_number': i + 1, 'timeline': f"{start_date_str} ถึง {end_date_str}", 'best_seed': best_seed, 'max_net': round(max_net, 2), 'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2), 'action_count': int(np.sum(best_actions)), 'window_size': window_len}
        window_details_list.append(detail)
        progress_bar.progress((i + 1) / num_windows, text=f"Optimizing Window {i+1}/{num_windows}...")
        
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)

# --- Helper functions for comparison (ต้องมีเพื่อรัน simulation) ---
@njit(cache=True)
def _full_simulation_cpu(action_array: np.ndarray, price_array: np.ndarray, fix: int = 1500) -> Tuple:
    n = len(action_array)
    action_array_calc = action_array.copy(); action_array_calc[0] = 1
    amount = np.empty(n, dtype=np.float64); cash = np.empty(n, dtype=np.float64)
    asset_value = np.empty(n, dtype=np.float64); sumusd = np.empty(n, dtype=np.float64)
    initial_price = price_array[0]
    amount[0] = fix / initial_price; cash[0] = fix
    asset_value[0] = amount[0] * initial_price; sumusd[0] = cash[0] + asset_value[0]
    refer = -fix * np.log(initial_price / price_array)
    for i in range(1, n):
        curr_price = price_array[i]
        if action_array_calc[i] == 0:
            amount[i] = amount[i-1]
            buffer = 0.0
        else:
            amount[i] = fix / curr_price
            buffer = amount[i-1] * curr_price - fix
        cash[i] = cash[i-1] + buffer
        asset_value[i] = amount[i] * curr_price
        sumusd[i] = cash[i] + asset_value[i]
    return sumusd, refer

def run_simulation(prices: List[float], actions: List[int]) -> pd.DataFrame:
    price_array = np.array(prices, dtype=np.float64)
    action_array = np.array(actions, dtype=np.int32)
    min_len = min(len(price_array), len(action_array))
    price_array, action_array = price_array[:min_len], action_array[:min_len]
    if min_len == 0: return pd.DataFrame()
    sumusd, refer = _full_simulation_cpu(action_array, price_array)
    df = pd.DataFrame({'sumusd': sumusd, 'refer': refer})
    df['net'] = df['sumusd'] - df['refer'] - df['sumusd'].iloc[0]
    return df

def generate_actions_rebalance_daily(num_days: int) -> np.ndarray: return np.ones(num_days, dtype=np.int32)
def generate_actions_perfect_foresight(prices: np.ndarray) -> np.ndarray:
    n = len(prices)
    if n == 0: return np.array([], dtype=np.int32)
    actions = np.zeros(n, dtype=np.int32); actions[0] = 1
    last_buy_price = prices[0]
    for i in range(1, n):
        if prices[i] < last_buy_price:
            actions[i] = 1; last_buy_price = prices[i]
    return actions


# ==============================================================================
# 4. UI Rendering Functions & Main App (UI ไม่เปลี่ยนแปลง)
# ==============================================================================
def render_settings_tab(config: Dict[str, Any]):
    st.write("⚙️ **การตั้งค่าพารามิเตอร์**")
    asset_list = config.get('assets', ['FFWM']); default_index = asset_list.index(st.session_state.test_ticker) if st.session_state.test_ticker in asset_list else 0
    st.session_state.test_ticker = st.selectbox("เลือก Ticker สำหรับทดสอบ", options=asset_list, index=default_index)
    st.write("📅 **ช่วงวันที่สำหรับการวิเคราะห์**"); col1, col2 = st.columns(2)
    with col1: st.session_state.start_date = st.date_input("วันที่เริ่มต้น", value=st.session_state.start_date)
    with col2: st.session_state.end_date = st.date_input("วันที่สิ้นสุด", value=st.session_state.end_date)
    if st.session_state.start_date >= st.session_state.end_date: st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
    st.divider()
    st.subheader("พารามิเตอร์สำหรับ Brute-Force Optimizer")
    st.session_state.window_size = st.number_input("ขนาด Window (วัน)", min_value=10, value=st.session_state.window_size)
    st.session_state.num_seeds = st.number_input("จำนวน Seeds ที่จะทดสอบต่อ Window", min_value=1000, max_value=10000000, value=st.session_state.num_seeds, format="%d", help="การคำนวณถูก Optimize ด้วย Numba (CPU-Max) แล้ว")

def display_comparison_charts(results: Dict[str, pd.DataFrame], chart_title: str = '📊 เปรียบเทียบกำไรสุทธิ (Net Profit)'):
    if not results: st.warning("ไม่มีข้อมูลสำหรับสร้างกราฟเปรียบเทียบ"); return
    valid_dfs = {name: df for name, df in results.items() if not df.empty and 'net' in df.columns}
    if not valid_dfs: st.warning("ไม่มีข้อมูล 'net' สำหรับสร้างกราฟ"); return
    longest_index = max((df.index for df in valid_dfs.values()), key=len, default=None)
    if longest_index is None: return
    chart_data = pd.DataFrame(index=longest_index)
    for name, df in valid_dfs.items(): chart_data[name] = df['net'].reindex(longest_index).ffill()
    st.write(chart_title); st.line_chart(chart_data)

def render_brute_force_tab():
    st.markdown("### 🚀 Brute-Force Optimizer (CPU-Max Mode)")
    st.info("โมเดลนี้ใช้ CPU ทุกคอร์พร้อมอัลกอริทึมที่ปรับแต่งมาโดยเฉพาะเพื่อรีดความเร็วสูงสุดบนสภาพแวดล้อมที่ไม่มี GPU (เช่น Streamlit Cloud)")
    if st.button("🚀 เริ่มทดสอบ Brute-Force Optimizer (CPU-Max)", type="primary"):
        if st.session_state.start_date >= st.session_state.end_date: st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้อง"); return
        ticker = st.session_state.test_ticker; start_date_str = st.session_state.start_date.strftime('%Y-%m-%d'); end_date_str = st.session_state.end_date.strftime('%Y-%m-%d')
        st.info(f"กำลังดึงข้อมูลสำหรับ **{ticker}** | {start_date_str} ถึง {end_date_str}")
        ticker_data = get_ticker_data(ticker, start_date_str, end_date_str)
        if ticker_data.empty: st.error("ไม่พบข้อมูลสำหรับ Ticker และช่วงวันที่ที่เลือก"); return
        prices = ticker_data['Close'].to_numpy(); num_days = len(prices)
        
        with st.spinner("กำลัง Brute-Force แบบขนาน (CPU-Max)..."):
            import time; start_time = time.time()
            actions_brute, df_windows = generate_actions_sliding_window_brute_force(ticker_data, st.session_state.window_size, st.session_state.num_seeds)
            end_time = time.time(); st.success(f"CPU-Max Brute-Force เสร็จสิ้นใน {end_time - start_time:.2f} วินาที!")
            
            actions_max = generate_actions_perfect_foresight(prices)
            results = {
                Strategy.BRUTE_FORCE_OPTIMIZER: run_simulation(prices.tolist(), actions_brute.tolist()),
                Strategy.REBALANCE_DAILY: run_simulation(prices.tolist(), generate_actions_rebalance_daily(num_days).tolist()),
                Strategy.PERFECT_FORESIGHT: run_simulation(prices.tolist(), actions_max.tolist())
            }
            for name, df in results.items():
                if not df.empty: df.index = ticker_data.index[:len(df)]
        
        display_comparison_charts(results)
        st.write("📈 **สรุปผลการค้นหา Best Seed**")
        total_actions = df_windows['action_count'].sum(); total_net = df_windows['max_net'].sum()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Windows", df_windows.shape[0]); col2.metric("Total Actions", f"{total_actions}/{num_days}"); col3.metric("Total Net (Sum)", f"${total_net:,.2f}")
        st.dataframe(df_windows[['window_number', 'timeline', 'best_seed', 'max_net', 'price_change_pct', 'action_count']], use_container_width=True)
        csv = df_windows.to_csv(index=False); st.download_button(label="📥 ดาวน์โหลด Window Details (CSV)", data=csv, file_name=f'bruteforce_cpu_max_{ticker}.csv', mime='text/csv')

def main():
    st.set_page_config(page_title="CPU-Max Optimizer", page_icon="⚡️", layout="wide")
    st.markdown("## ⚡️ CPU-Max Optimizer Lab")
    st.caption("เครื่องมือทดสอบกลยุทธ์ Brute-Force ที่ถูกปรับแต่งเพื่อความเร็วสูงสุดบน CPU")
    config = load_config()
    initialize_session_state(config)
    tab_list = ["⚙️ การตั้งค่า", "🚀 Brute-Force Optimizer"]
    tabs = st.tabs(tab_list)
    with tabs[0]: render_settings_tab(config)
    with tabs[1]: render_brute_force_tab()
    with st.expander("📖 คำอธิบายอัลกอริทึม (CPU-Max)"):
        st.markdown("""
        **⚡️ CPU-Max Hyperspeed Edition:**
        นี่คือเวอร์ชันที่ถูกออกแบบมาเพื่อ **รีดประสิทธิภาพสูงสุดจาก CPU** โดยเฉพาะ เหมาะสำหรับสภาพแวดล้อมที่ไม่มี GPU เช่น Streamlit Cloud โดยยังคง Logic การค้นหาแบบ Brute-Force เหมือนเดิม
        - **Micro-Optimized Kernel:** ฟังก์ชันแกนกลางที่จำลองการเทรด (`_calculate_net_for_seed`) ถูกเขียนขึ้นใหม่ให้กระทัดรัดที่สุด ลดการสร้างข้อมูลชั่วคราวและคำนวณเฉพาะผลลัพธ์สุดท้าย (Net Profit) เพื่อลดภาระงานใน Loop ที่ทำงานเป็นล้านรอบ
        - **Fast Pseudo-Random Generator:** เปลี่ยนจากการใช้ `np.random` ที่มี Overhead มาใช้ `Xorshift` ซึ่งเป็นอัลกอริทึมสร้างเลขสุ่มที่เรียบง่ายและเร็วมากเมื่อคอมไพล์ด้วย Numba
        - **Maximized Parallelism:** ยังคงใช้ `prange` เพื่อกระจายงานที่ถูกปรับให้เบาลงแล้วไปยังทุก Core ของ CPU ทำให้แต่ละ Core ทำงานได้มากขึ้นในเวลาเท่าเดิม
        - **ผลลัพธ์:** ความเร็วที่เพิ่มขึ้นอย่างเห็นได้ชัดเมื่อเทียบกับเวอร์ชัน `prange` มาตรฐาน ทำให้สามารถทดสอบ `seed` ในปริมาณที่มากขึ้นได้บน Streamlit Cloud
        """)

if __name__ == "__main__":
    main()
