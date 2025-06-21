import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
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
    WALK_FORWARD_DP = "Walk-Forward Dynamic Programming"

def initialize_session_state():
    """ตั้งค่าเริ่มต้นสำหรับ Streamlit session state"""
    if 'test_ticker' not in st.session_state:
        st.session_state.test_ticker = 'NVDA'
    if 'start_date' not in st.session_state:
        st.session_state.start_date = datetime(2023, 1, 1).date()
    if 'end_date' not in st.session_state:
        st.session_state.end_date = datetime.now().date()
    if 'window_size' not in st.session_state:
        st.session_state.window_size = 30
    if 'fix_capital' not in st.session_state:
        st.session_state.fix_capital = 1500

# ==============================================================================
# 2. Core Calculation & Data Functions (With Numba Acceleration)
# ==============================================================================
@st.cache_data(ttl=3600)
def get_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """ดึงข้อมูลราคาหุ้น/สินทรัพย์จาก Yahoo Finance และ Cache ผลลัพธ์ไว้ 1 ชั่วโมง"""
    try:
        data = yf.Ticker(ticker).history(start=start_date, end=end_date)[['Close']]
        if data.empty: return pd.DataFrame()
        return data
    except Exception as e:
        st.error(f"❌ ไม่สามารถดึงข้อมูล {ticker} ได้: {str(e)}")
        return pd.DataFrame()

@njit(cache=True)
def _calculate_simulation_numba(action_array: np.ndarray, price_array: np.ndarray, fix: int) -> Tuple:
    """
    คำนวณผลลัพธ์การจำลองการเทรด (หัวใจของการคำนวณ) เร่งความเร็วด้วย Numba
    """
    n = len(action_array)
    if n == 0 or len(price_array) == 0:
        empty_arr = np.empty(0, dtype=np.float64)
        return (empty_arr, empty_arr, empty_arr)

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

    net = sumusd - refer - sumusd[0]
    return sumusd, refer, net

@lru_cache(maxsize=16384)
def calculate_optimized_cached(action_tuple: Tuple[int, ...], price_tuple: Tuple[float, ...], fix: int) -> Tuple:
    """Wrapper function สำหรับเรียกฟังก์ชัน Numba โดยใช้ Cache"""
    action_array = np.asarray(action_tuple, dtype=np.int32)
    price_array = np.asarray(price_tuple, dtype=np.float64)
    return _calculate_simulation_numba(action_array, price_array, fix)

def run_simulation(prices: List[float], actions: List[int], fix: int) -> pd.DataFrame:
    """สร้าง DataFrame ผลลัพธ์จากการจำลองการเทรด"""
    if not prices or not actions: return pd.DataFrame()
    min_len = min(len(prices), len(actions))
    prices, actions = prices[:min_len], actions[:min_len]

    sumusd, refer, net = calculate_optimized_cached(tuple(actions), tuple(prices), fix)
    if len(sumusd) == 0: return pd.DataFrame()

    initial_capital = sumusd[0]
    return pd.DataFrame({
        'price': prices,
        'action': actions,
        'sumusd': np.round(sumusd, 2),
        'refer': np.round(refer + initial_capital, 2),
        'net': np.round(net, 2)
    })

# ==============================================================================
# 3. Strategy Action Generation
# ==============================================================================

def generate_actions_rebalance_daily(num_days: int) -> np.ndarray:
    """สร้าง Action สำหรับกลยุทธ์ Rebalance ทุกวัน (Min Performance)"""
    return np.ones(num_days, dtype=np.int32)

@njit(cache=True)
def _generate_perfect_foresight_numba(price_arr: np.ndarray, fix: int) -> np.ndarray:
    """
    คำนวณ Perfect Foresight โดยใช้ Dynamic Programming (เร่งความเร็วด้วย Numba)
    """
    n = len(price_arr)
    if n < 2: return np.ones(n, dtype=np.int32)

    dp = np.zeros(n, dtype=np.float64)
    path = np.zeros(n, dtype=np.int32)
    dp[0] = float(fix * 2)

    for i in range(1, n):
        j_indices = np.arange(i)
        profits = fix * ((price_arr[i] / price_arr[j_indices]) - 1)
        current_sumusd = dp[j_indices] + profits

        best_j_idx = np.argmax(current_sumusd)
        dp[i] = current_sumusd[best_j_idx]
        path[i] = best_j_idx

    actions = np.zeros(n, dtype=np.int32)
    current_day = np.argmax(dp)
    while current_day > 0:
        actions[current_day] = 1
        current_day = path[current_day]
    actions[0] = 1
    return actions

def generate_actions_perfect_foresight(prices: List[float], fix: int) -> np.ndarray:
    """Wrapper สำหรับเรียกใช้ Perfect Foresight"""
    price_arr = np.array(prices, dtype=np.float64)
    return _generate_perfect_foresight_numba(price_arr, fix)

@njit(cache=True)
def actions_to_seed(actions: np.ndarray) -> int:
    """
    แปลง Action Sequence (เลขฐานสอง) ให้เป็น Seed (เลขจำนวนเต็ม)
    นี่คือหัวใจของ "Reversible Seed" ที่สง่างามทางคณิตศาสตร์
    """
    seed = 0
    for i, action in enumerate(actions):
        if action == 1:
            seed += 1 << i # ใช้ Bitwise operation เพื่อความเร็ว
    return seed

def generate_walk_forward_strategy(ticker_data: pd.DataFrame, window_size: int, fix: int) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    สร้าง Action Sequence และผลวิเคราะห์ด้วยกลยุทธ์ Walk-Forward Dynamic Programming
    """
    prices = ticker_data['Close'].to_numpy()
    n = len(prices)
    final_actions = np.array([], dtype=int)
    window_details_list = []
    
    num_windows = (n // window_size) # คำนวณจำนวน window เต็มๆ
    progress_bar = st.progress(0, text="กำลังประมวลผล Walk-Forward Strategy...")
    
    oracle_dna_for_next_window = np.ones(window_size, dtype=int) # DNA เริ่มต้น

    for i in range(num_windows):
        # Window ปัจจุบันสำหรับ "เรียนรู้" (Learning Window)
        learn_start = i * window_size
        learn_end = learn_start + window_size
        learn_prices = prices[learn_start:learn_end]
        learn_dates = ticker_data.index[learn_start:learn_end]
        
        # Window ถัดไปสำหรับ "ทดสอบ" (Testing Window)
        test_start = (i + 1) * window_size
        test_end = test_start + window_size
        if test_end > n: continue # หยุดถ้า window ทดสอบไม่ครบ
        
        test_prices = prices[test_start:test_end]
        test_dates = ticker_data.index[test_start:test_end]
        
        # 1. ค้นหา "Oracle DNA" จาก Learning Window
        current_oracle_dna = generate_actions_perfect_foresight(learn_prices.tolist(), fix)
        
        # 2. เข้ารหัส DNA เป็น Seed
        oracle_seed = actions_to_seed(current_oracle_dna)
        
        # 3. นำ DNA ที่เรียนรู้จาก window ก่อนหน้า มาทดสอบกับ Testing Window
        # โดยใช้ `oracle_dna_for_next_window` ที่คำนวณไว้ใน loop ที่แล้ว
        actions_for_this_test_window = oracle_dna_for_next_window
        final_actions = np.concatenate((final_actions, actions_for_this_test_window))

        # 4. คำนวณผลลัพธ์ของการทดสอบ
        _, _, net_array = calculate_optimized_cached(tuple(actions_for_this_test_window), tuple(test_prices), fix)
        forward_test_net_profit = net_array[-1] if len(net_array) > 0 else 0
        
        # 5. บันทึกรายละเอียด
        detail = {
            'window_number': i + 1,
            'learn_period': f"{learn_dates[0]:%Y-%m-%d} ถึง {learn_dates[-1]:%Y-%m-%d}",
            'test_period': f"{test_dates[0]:%Y-%m-%d} ถึง {test_dates[-1]:%Y-%m-%d}",
            'oracle_seed': oracle_seed,
            'forward_test_net_profit': round(forward_test_net_profit, 2),
            'action_sequence_used': actions_for_this_test_window.tolist()
        }
        window_details_list.append(detail)
        
        # 6. เตรียม DNA สำหรับ Window ถัดไป
        oracle_dna_for_next_window = current_oracle_dna
        
        progress_bar.progress((i + 1) / num_windows, text=f"ประมวลผล Window {i+1}/{num_windows}")
        
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)

# ==============================================================================
# 4. UI Rendering Functions
# ==============================================================================
def render_settings_tab():
    """แสดงผล UI สำหรับ Tab การตั้งค่า"""
    st.write("⚙️ **การตั้งค่าพารามิเตอร์**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.test_ticker = st.text_input("เลือก Ticker สำหรับทดสอบ", value=st.session_state.test_ticker)
    with col2:
        st.session_state.window_size = st.number_input("ขนาด Window (วัน)", min_value=5, max_value=252, value=st.session_state.window_size)
    with col3:
        st.session_state.fix_capital = st.number_input("เงินลงทุนต่อครั้ง (Fix)", min_value=100, value=st.session_state.fix_capital, step=100)

    st.write("📅 **ช่วงวันที่สำหรับการวิเคราะห์**")
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.session_state.start_date = st.date_input("วันที่เริ่มต้น", value=st.session_state.start_date)
    with d_col2:
        st.session_state.end_date = st.date_input("วันที่สิ้นสุด", value=st.session_state.end_date)
    
    if st.session_state.start_date >= st.session_state.end_date:
        st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")

def display_comparison_charts(results: Dict[str, pd.DataFrame]):
    """แสดงผลกราฟเปรียบเทียบผลลัพธ์จากหลายๆ กลยุทธ์"""
    if not results:
        st.warning("ไม่มีข้อมูลสำหรับสร้างกราฟเปรียบเทียบ")
        return

    chart_data = pd.DataFrame()
    for name, df in results.items():
        if not df.empty and 'net' in df.columns:
            # Use a consistent name for the series for better tooltips
            chart_data[name] = df['net']
    
    st.write("📊 **เปรียบเทียบกำไรสุทธิสะสม (Net Profit)**")
    st.line_chart(chart_data)

def render_walk_forward_dp_tab():
    """แสดงผล UI สำหรับ Tab กลยุทธ์หลัก Walk-Forward Dynamic Programming"""
    st.markdown("### 🧠 Walk-Forward Dynamic Programming")
    st.info("""
    กลยุทธ์นี้ทำงานโดยการหา **"รูปแบบการเทรดที่สมบูรณ์แบบ" (Oracle DNA)** จากข้อมูลในอดีต (Learning Window)
    แล้วนำรูปแบบนั้นมา **ทดสอบกับข้อมูลในอนาคตทันที (Testing Window)** เพื่อจำลองการเทรดจริงและลดการ Overfitting
    """)
    if st.button("🚀 เริ่มการวิเคราะห์ Walk-Forward", type="primary"):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้องในแท็บ 'การตั้งค่า'")
            return
            
        ticker = st.session_state.test_ticker
        start_date_str = st.session_state.start_date.strftime('%Y-%m-%d')
        end_date_str = st.session_state.end_date.strftime('%Y-%m-%d')
        fix_capital = st.session_state.fix_capital
        
        with st.spinner(f"กำลังดึงข้อมูลสำหรับ **{ticker}**..."):
            ticker_data = get_ticker_data(ticker, start_date_str, end_date_str)
        
        if ticker_data.empty:
            st.error("ไม่พบข้อมูลสำหรับ Ticker และช่วงวันที่ที่เลือก")
            return
            
        prices = ticker_data['Close'].tolist()
        num_days = len(prices)

        with st.spinner("กำลังคำนวณกลยุทธ์ต่างๆ... (Core calculation is Numba-accelerated)"):
            # 1. คำนวณกลยุทธ์หลัก
            actions_walk_forward, df_windows = generate_walk_forward_strategy(
                ticker_data, st.session_state.window_size, fix_capital
            )
            
            # 2. คำนวณ Benchmarks
            actions_max = generate_actions_perfect_foresight(prices, fix_capital)
            actions_min = generate_actions_rebalance_daily(num_days)
            
            # 3. รัน Simulation
            results = {}
            strategy_map = {
                Strategy.WALK_FORWARD_DP: actions_walk_forward.tolist(),
                Strategy.PERFECT_FORESIGHT: actions_max.tolist(),
                Strategy.REBALANCE_DAILY: actions_min.tolist()
            }
            
            for strategy_name, actions in strategy_map.items():
                # จำลองเฉพาะช่วงเวลาที่มี action จริงๆ
                sim_prices = prices[:len(actions)]
                df = run_simulation(sim_prices, actions, fix_capital)
                if not df.empty:
                    df.index = ticker_data.index[:len(df)]
                results[strategy_name] = df
        
        st.success("การวิเคราะห์เสร็จสมบูรณ์!")
        st.write("---")
        
        # แสดงผลลัพธ์
        display_comparison_charts(results)
        
        st.write("📈 **สรุปผลการดำเนินงาน**")
        
        # Metrics
        final_net_max = results[Strategy.PERFECT_FORESIGHT]['net'].iloc[-1]
        final_net_walk_forward = results[Strategy.WALK_FORWARD_DP]['net'].iloc[-1] if not results[Strategy.WALK_FORWARD_DP].empty else 0
        final_net_min = results[Strategy.REBALANCE_DAILY]['net'].iloc[-1]

        col1, col2, col3 = st.columns(3)
        col1.metric(f"🥇 {Strategy.PERFECT_FORESIGHT}", f"${final_net_max:,.2f}")
        col2.metric(f"🧠 {Strategy.WALK_FORWARD_DP}", f"${final_net_walk_forward:,.2f}",
                    delta=f"{final_net_walk_forward - final_net_min:,.2f} vs Min", delta_color="normal")
        col3.metric(f"🥉 {Strategy.REBALANCE_DAILY}", f"${final_net_min:,.2f}")

        st.write("---")
        st.write("🔍 **รายละเอียดการทดสอบแบบ Walk-Forward ราย Window**")
        st.dataframe(df_windows[['window_number', 'learn_period', 'test_period', 'oracle_seed', 'forward_test_net_profit']], use_container_width=True)
        csv = df_windows.to_csv(index=False)
        st.download_button(label="📥 ดาวน์โหลดผลวิเคราะห์ (CSV)", data=csv, file_name=f'walk_forward_dp_{ticker}.csv', mime='text/csv')

# ==============================================================================
# 5. Main Application
# ==============================================================================
def main():
    """
    ฟังก์ชันหลักในการรัน Streamlit Application
    """
    st.set_page_config(page_title="Walk-Forward DP", page_icon="🧠", layout="wide")
    st.markdown("### 🧠 Walk-Forward Dynamic Programming Tester")
    st.caption("โมเดลใหม่ที่เข้าใกล้ Perfect Foresight โดยใช้หลักการเรียนรู้จากอดีตและทดสอบไปข้างหน้า (Walk-Forward)")

    # โหลดการตั้งค่าและเตรียม Session State
    initialize_session_state()

    # สร้าง Tabs
    tab_settings, tab_model = st.tabs(["⚙️ การตั้งค่า", "🚀 วิเคราะห์ด้วย Walk-Forward DP"])

    with tab_settings:
        render_settings_tab()
        
    with tab_model:
        render_walk_forward_dp_tab()
    
    with st.expander("📖 คำอธิบายหลักการทำงานของโมเดล (Walk-Forward Dynamic Programming)"):
        st.markdown("""
        ### แนวคิดหลัก: "นำบทเรียนที่สมบูรณ์แบบจากอดีต มาทดสอบกับอนาคต"

        โมเดลนี้ถูกออกแบบมาให้ **สง่างามทางคณิตศาสตร์, โปร่งใส, และลดปัญหาการ Overfitting** โดยมีขั้นตอนดังนี้:

        1.  **แบ่งข้อมูลเป็นหน้าต่าง (Window):** เราแบ่งข้อมูลราคาในอดีตออกเป็นส่วนๆ ที่ไม่ทับซ้อนกัน เรียกว่า Window (เช่น Window ขนาด 30 วัน).

        2.  **หา 'Oracle DNA' (Learning):**
            - ในแต่ละ Window (เช่น Window ที่ 1), เราใช้ **Dynamic Programming** เพื่อคำนวณหา `Action Sequence` ที่ให้ผลกำไร **สูงสุดที่เป็นไปได้ในทางทฤษฎี** สำหรับ Window นั้นๆ.
            - เราเรียก Sequence ที่สมบูรณ์แบบนี้ว่า **"Oracle DNA"** เพราะมันเหมือนการหยั่งรู้อนาคต... เฉพาะใน Window นั้น.
            - DNA นี้จะถูกแปลงเป็น **"Oracle Seed"** ซึ่งเป็นตัวเลข Integer เพียงตัวเดียว ทำให้สามารถคำนวณย้อนกลับและตรวจสอบได้.

        3.  **ทดสอบไปข้างหน้า (Walk-Forward Testing):**
            - นี่คือหัวใจสำคัญที่ทำให้โมเดลนี้ทรงพลังและเป็นอิสระจาก Overfitting.
            - เราจะไม่นำ `Oracle DNA` ที่หาได้จาก Window ที่ 1 มาคิดกำไรใน Window ที่ 1 (เพราะนั่นคือการโกง).
            - แต่เราจะนำ `Oracle DNA` จาก Window ที่ 1 **ไปใช้เทรดใน Window ที่ 2** แทน.
            - เช่นเดียวกัน `Oracle DNA` ที่หาได้จาก Window ที่ 2 ก็จะถูกนำไปใช้เทรดใน Window ที่ 3.
            - กระบวนการนี้เรียกว่า **Walk-Forward** ซึ่งจำลองสถานการณ์จริงที่เราต้องตัดสินใจโดยใช้ข้อมูลจากอดีตเพื่อเทรดในอนาคต.

        4.  **ผลลัพธ์:**
            - เส้นกราฟของกลยุทธ์ `Walk-Forward DP` ที่เห็น คือผลรวมจากการนำ "บทเรียนที่ดีที่สุด" จากแต่ละช่วงเวลา มาทดสอบไปข้างหน้าอย่างต่อเนื่อง.
            - ทำให้เราเห็นประสิทธิภาพที่แท้จริงของ "รูปแบบ" ที่เรียนรู้ได้จากตลาด โดยเทียบกับ **Best Case (Perfect Foresight)** และ **Worst Case (Rebalance Daily)**.
        """)

if __name__ == "__main__":
    main()
