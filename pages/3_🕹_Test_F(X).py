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
    """คลาสสำหรับเก็บชื่อกลยุทธ์ต่างๆ"""
    REBALANCE_DAILY = "Rebalance Daily (Min)"
    PERFECT_FORESIGHT = "Perfect Foresight (Max)"
    OPTIMAL_MRC = "Optimal Momentum-Reversion Crossover"

def initialize_session_state():
    """ตั้งค่าเริ่มต้นสำหรับ Streamlit session state"""
    if 'test_ticker' not in st.session_state: st.session_state.test_ticker = 'TSLA'
    if 'start_date' not in st.session_state: st.session_state.start_date = datetime(2022, 1, 1).date()
    if 'end_date' not in st.session_state: st.session_state.end_date = datetime.now().date()
    if 'fix_capital' not in st.session_state: st.session_state.fix_capital = 1500
    # Search range for the new model
    if 'fast_ma_range' not in st.session_state: st.session_state.fast_ma_range = (5, 50)
    if 'slow_ma_range' not in st.session_state: st.session_state.slow_ma_range = (20, 200)

# ==============================================================================
# 2. Core Calculation & Data Functions
# ==============================================================================
@st.cache_data(ttl=3600)
def get_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """ดึงข้อมูลราคาหุ้น/สินทรัพย์จาก Yahoo Finance"""
    try:
        data = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=True)[['Close']]
        if data.empty: return pd.DataFrame()
        return data
    except Exception as e:
        st.error(f"❌ ไม่สามารถดึงข้อมูล {ticker} ได้: {str(e)}")
        return pd.DataFrame()

@njit(cache=True)
def _calculate_simulation_numba(action_array: np.ndarray, price_array: np.ndarray, fix: int) -> np.ndarray:
    """คำนวณผลลัพธ์การจำลองการเทรด (หัวใจของการคำนวณ) เร่งความเร็วด้วย Numba"""
    n = len(action_array)
    if n == 0 or len(price_array) == 0: return np.empty(0, dtype=np.float64)

    action_array_calc = action_array.copy()
    if n > 0: action_array_calc[0] = 1 # บังคับซื้อวันแรกเสมอ

    cash, sumusd = np.empty(n, dtype=np.float64), np.empty(n, dtype=np.float64)
    amount, asset_value = np.empty(n, dtype=np.float64), np.empty(n, dtype=np.float64)

    initial_price = price_array[0]
    amount[0] = fix / initial_price
    cash[0] = fix
    asset_value[0] = amount[0] * initial_price
    sumusd[0] = cash[0] + asset_value[0]

    refer = -fix * np.log(initial_price / price_array)

    for i in range(1, n):
        curr_price = price_array[i]
        prev_amount = amount[i-1]
        
        if action_array_calc[i] == 0: # Hold
            amount[i] = prev_amount
            buffer = 0.0
        else: # Rebalance
            amount[i] = fix / curr_price
            buffer = prev_amount * curr_price - fix

        cash[i] = cash[i-1] + buffer
        asset_value[i] = amount[i] * curr_price
        sumusd[i] = cash[i] + asset_value[i]

    net = sumusd - refer - sumusd[0]
    return net

def run_simulation(prices: List[float], actions: List[int], fix: int) -> pd.DataFrame:
    """สร้าง DataFrame ผลลัพธ์จากการจำลองการเทรด"""
    if not prices or not actions: return pd.DataFrame()
    min_len = min(len(prices), len(actions))
    prices, actions = prices[:min_len], actions[:min_len]

    net = _calculate_simulation_numba(np.array(actions, dtype=np.int32), np.array(prices, dtype=np.float64), fix)
    if len(net) == 0: return pd.DataFrame()

    return pd.DataFrame({'price': prices, 'action': actions, 'net': np.round(net, 2)})

# ==============================================================================
# 3. Strategy Action Generation (NEW MODEL: MRC)
# ==============================================================================

# --- Helper functions for our Reversible Seed ---
def params_to_seed(fast: int, slow: int) -> int:
    """เข้ารหัสพารามิเตอร์ (10, 50) -> 10050"""
    return fast * 1000 + slow

def seed_to_params(seed: int) -> Tuple[int, int]:
    """ถอดรหัส Seed 10050 -> (10, 50)"""
    fast = seed // 1000
    slow = seed % 1000
    return fast, slow

# --- Core Model Logic (Numba Accelerated) ---
@njit(cache=True)
def _generate_mrc_actions_numba(prices: np.ndarray, fast_period: int, slow_period: int) -> np.ndarray:
    """สร้าง Action Sequence จากหลักการ Momentum-Reversion Crossover"""
    n = len(prices)
    actions = np.zeros(n, dtype=np.int32)
    if n < slow_period: return actions

    # Calculate SMAs (can't use pandas inside njit, so do it manually)
    fast_sma = np.empty(n, dtype=np.float64)
    slow_sma = np.empty(n, dtype=np.float64)
    fast_sum = 0.0
    slow_sum = 0.0
    
    for i in range(n):
        fast_sum += prices[i]
        slow_sum += prices[i]
        if i >= fast_period: fast_sum -= prices[i - fast_period]
        if i >= slow_period: slow_sum -= prices[i - slow_period]
        
        if i >= fast_period - 1: fast_sma[i] = fast_sum / fast_period
        else: fast_sma[i] = np.nan
        
        if i >= slow_period - 1: slow_sma[i] = slow_sum / slow_period
        else: slow_sma[i] = np.nan

    # Generate actions based on the crossover logic
    for i in range(slow_period - 1, n):
        # Regime Filter: Is the market in a Bullish Trend?
        is_bullish_regime = fast_sma[i] > slow_sma[i]
        
        if is_bullish_regime:
            # Reversion Signal: Buy the dip within the uptrend
            if prices[i] < fast_sma[i]:
                actions[i] = 1
            else:
                actions[i] = 0 # Hold, don't rebalance
        else:
            # Bearish Regime: Stay out
            actions[i] = 0
            
    return actions

# --- Optimization Function to find the best parameters ---
def find_best_mrc_params(prices: np.ndarray, fast_range: Tuple[int, int], slow_range: Tuple[int, int], fix: int) -> Dict:
    """ค้นหาพารามิเตอร์ (fast, slow) ที่ให้ Net Profit สูงสุด"""
    
    param_pairs = []
    for fast in range(fast_range[0], fast_range[1] + 1):
        for slow in range(slow_range[0], slow_range[1] + 1):
            if slow > fast * 1.5: # Ensure slow is significantly slower than fast
                param_pairs.append((fast, slow))

    if not param_pairs:
        st.warning("ช่วงการค้นหาที่กำหนดไม่สร้างคู่พารามิเตอร์ที่เหมาะสม")
        return {'best_net': -np.inf, 'best_params': (0,0), 'best_actions': np.array([])}

    best_net = -np.inf
    best_params = (0, 0)
    
    progress_bar = st.progress(0, text=f"กำลังทดสอบพารามิเตอร์ {len(param_pairs)} คู่...")
    
    # Using ThreadPoolExecutor for parallel evaluation
    with ThreadPoolExecutor() as executor:
        future_to_params = {
            executor.submit(_evaluate_mrc_params, prices, p, fix): p for p in param_pairs
        }
        
        completed_count = 0
        for future in as_completed(future_to_params):
            params = future_to_params[future]
            try:
                final_net = future.result()
                if final_net > best_net:
                    best_net = final_net
                    best_params = params
            except Exception as exc:
                print(f'{params} generated an exception: {exc}') # log to console
            
            completed_count += 1
            progress_bar.progress(completed_count / len(param_pairs), text=f"ทดสอบแล้ว {completed_count}/{len(param_pairs)} คู่...")

    progress_bar.empty()
    
    # Generate final actions with the best parameters found
    best_actions = _generate_mrc_actions_numba(prices, best_params[0], best_params[1])
    
    return {'best_net': best_net, 'best_params': best_params, 'best_actions': best_actions}

def _evaluate_mrc_params(prices: np.ndarray, params: Tuple[int, int], fix: int) -> float:
    """Helper function for parallel execution"""
    fast, slow = params
    actions = _generate_mrc_actions_numba(prices, fast, slow)
    net_array = _calculate_simulation_numba(actions, prices, fix)
    return net_array[-1] if len(net_array) > 0 else -np.inf


# --- Benchmark Strategies ---
@njit(cache=True)
def _generate_perfect_foresight_numba(price_arr: np.ndarray, fix: int) -> np.ndarray:
    """คำนวณ Perfect Foresight (Numba-accelerated)"""
    n = len(price_arr)
    if n < 2: return np.ones(n, dtype=np.int32)
    dp, path = np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.int32)
    dp[0] = float(fix * 2)

    for i in range(1, n):
        j_indices = np.arange(i)
        profits = fix * ((price_arr[i] / price_arr[j_indices]) - 1)
        current_sumusd = dp[j_indices] + profits
        best_j_idx = np.argmax(current_sumusd)
        dp[i] = current_sumusd[best_j_idx]
        path[i] = best_j_idx

    actions, current_day = np.zeros(n, dtype=np.int32), np.argmax(dp)
    while current_day > 0:
        actions[current_day], current_day = 1, path[current_day]
    actions[0] = 1
    return actions

# ==============================================================================
# 4. UI Rendering Functions
# ==============================================================================
def render_settings_tab():
    """แสดงผล UI สำหรับ Tab การตั้งค่า"""
    st.write("⚙️ **พารามิเตอร์พื้นฐาน**")
    
    c1, c2, c3 = st.columns(3)
    c1.text_input("Ticker สำหรับทดสอบ", key="test_ticker")
    c2.date_input("วันที่เริ่มต้น", key="start_date")
    c3.date_input("วันที่สิ้นสุด", key="end_date")
    
    st.divider()
    st.write("🧠 **พารามิเตอร์สำหรับโมเดล Momentum-Reversion Crossover (MRC)**")
    st.info("กำหนดช่วงของเส้นค่าเฉลี่ยที่ต้องการให้ระบบค้นหาค่าที่ดีที่สุด")
    
    s_c1, s_c2 = st.columns(2)
    s_c1.slider("ช่วงค้นหาเส้นค่าเฉลี่ยเร็ว (Fast MA)", 1, 100, key="fast_ma_range")
    s_c2.slider("ช่วงค้นหาเส้นค่าเฉลี่ยช้า (Slow MA)", 10, 252, key="slow_ma_range")
    
    if st.session_state.fast_ma_range[1] >= st.session_state.slow_ma_range[0]:
        st.warning("เพื่อให้ได้ผลดีที่สุด ควรตั้งค่าสูงสุดของ Fast MA ให้น้อยกว่าค่าต่ำสุดของ Slow MA")

def display_comparison_charts(results: Dict[str, pd.DataFrame]):
    """แสดงผลกราฟเปรียบเทียบผลลัพธ์จากหลายๆ กลยุทธ์"""
    if not results: return
    chart_data = pd.DataFrame({name: df['net'] for name, df in results.items() if not df.empty})
    st.write("📊 **เปรียบเทียบกำไรสุทธิสะสม (Net Profit)**")
    st.line_chart(chart_data)

def render_model_tab():
    """แสดงผล UI สำหรับ Tab กลยุทธ์หลัก MRC"""
    st.markdown("### 🧠 Momentum-Reversion Crossover (MRC) Optimizer")
    
    if st.button("🚀 ค้นหาพารามิเตอร์ที่ดีที่สุดและเริ่มวิเคราะห์", type="primary"):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้อง"); return
            
        ticker = st.session_state.test_ticker
        start_date_str, end_date_str = st.session_state.start_date.strftime('%Y-%m-%d'), st.session_state.end_date.strftime('%Y-%m-%d')
        fix_capital = st.session_state.fix_capital
        
        with st.spinner(f"กำลังดึงข้อมูล **{ticker}**..."):
            ticker_data = get_ticker_data(ticker, start_date_str, end_date_str)
        
        if ticker_data.empty: st.error("ไม่พบข้อมูล"); return
        
        prices_np = ticker_data['Close'].to_numpy()
        prices_list = ticker_data['Close'].tolist()
        num_days = len(prices_list)

        # 1. ค้นหาพารามิเตอร์ที่ดีที่สุดสำหรับ MRC
        mrc_results = find_best_mrc_params(prices_np, st.session_state.fast_ma_range, st.session_state.slow_ma_range, fix_capital)
        best_params = mrc_results['best_params']
        
        st.success(f"ค้นพบพารามิเตอร์ที่ดีที่สุด! Fast MA: **{best_params[0]}**, Slow MA: **{best_params[1]}** (Reversible Seed: {params_to_seed(best_params[0], best_params[1])})")

        with st.spinner("กำลังจำลองกลยุทธ์ต่างๆ..."):
            # 2. สร้าง Actions จากผลลัพธ์และ Benchmarks
            actions_mrc = mrc_results['best_actions'].tolist()
            actions_max = _generate_perfect_foresight_numba(prices_np, fix_capital).tolist()
            actions_min = np.ones(num_days, dtype=np.int32).tolist()
            
            # 3. รัน Simulation
            results = {}
            strategy_map = {
                Strategy.OPTIMAL_MRC: actions_mrc,
                Strategy.PERFECT_FORESIGHT: actions_max,
                Strategy.REBALANCE_DAILY: actions_min,
            }
            
            for name, actions in strategy_map.items():
                df = run_simulation(prices_list, actions, fix_capital)
                if not df.empty: df.index = ticker_data.index[:len(df)]
                results[name] = df
        
        st.success("การวิเคราะห์เสร็จสมบูรณ์!")
        st.write("---")
        
        # 4. แสดงผลลัพธ์
        display_comparison_charts(results)
        
        final_net_max = results[Strategy.PERFECT_FORESIGHT]['net'].iloc[-1]
        final_net_mrc = results[Strategy.OPTIMAL_MRC]['net'].iloc[-1]
        final_net_min = results[Strategy.REBALANCE_DAILY]['net'].iloc[-1]

        col1, col2, col3 = st.columns(3)
        col1.metric(f"🥇 {Strategy.PERFECT_FORESIGHT}", f"${final_net_max:,.2f}")
        col2.metric(f"🧠 {Strategy.OPTIMAL_MRC}", f"${final_net_mrc:,.2f}", f"Fast={best_params[0]}, Slow={best_params[1]}")
        col3.metric(f"🥉 {Strategy.REBALANCE_DAILY}", f"${final_net_min:,.2f}")

# ==============================================================================
# 5. Main Application
# ==============================================================================
def main():
    st.set_page_config(page_title="MRC Optimizer", page_icon="🧠", layout="wide")
    st.markdown("### 🧠 Momentum-Reversion Crossover (MRC) Optimizer")
    st.caption("โมเดลวิเคราะห์หาพารามิเตอร์เส้นค่าเฉลี่ยที่ดีที่สุด เพื่อสร้างกลยุทธ์ 'ซื้อเมื่อย่อในตลาดขาขึ้น'")

    initialize_session_state()

    tab_settings, tab_model = st.tabs(["⚙️ การตั้งค่า", "🚀 วิเคราะห์และแสดงผล"])
    with tab_settings: render_settings_tab()
    with tab_model: render_model_tab()
    
    with st.expander("📖 คำอธิบายหลักการทำงานของโมเดล (Momentum-Reversion Crossover)"):
        st.markdown("""
        ### แนวคิดหลัก: "กรองแนวโน้มใหญ่ รอจังหวะย่อตัว"

        โมเดลนี้ผสมผสานสองแนวคิดการเทรดที่ทรงพลัง เพื่อสร้างกลยุทธ์ที่แข็งแกร่งและลดสัญญาณรบกวน:

        1.  **ตัวกรองสภาวะตลาด (Regime Filter):**
            - เราใช้เส้นค่าเฉลี่ย 2 เส้น (Fast และ Slow) เพื่อระบุว่าตลาดโดยรวมอยู่ใน **"แนวโน้มขาขึ้น" (Bullish Regime)** หรือ **"แนวโน้มขาลง" (Bearish Regime)**.
            - **เงื่อนไข:** `Fast MA > Slow MA` หมายถึง ตลาดเป็นขาขึ้น เราจะพิจารณา "ซื้อ".
            - **เงื่อนไข:** `Fast MA < Slow MA` หมายถึง ตลาดเป็นขาลง เราจะ "อยู่เฉยๆ" เพื่อป้องกันความเสี่ยง.

        2.  **จังหวะเข้าซื้อ (Reversion Entry):**
            - เราจะไม่ซื้อทันทีที่ตลาดเป็นขาขึ้น!
            - แต่ภายใน **Bullish Regime** เราจะรอให้ราคา **"ย่อตัว"** ลงมาต่ำกว่าเส้นค่าเฉลี่ยเร็ว (Fast MA) ก่อน.
            - นี่คือการ **"ซื้อเมื่อย่อในตลาดกระทิง" (Buy the dip in an uptrend)** ซึ่งเป็นกลยุทธ์ที่ช่วยให้ได้ราคาเข้าที่ดีกว่า และหลีกเลี่ยงการไล่ราคาที่จุดสูงสุด.

        ### กระบวนการทำงาน:
        1.  **กำหนดช่วงค้นหา:** คุณกำหนดช่วงของ `Fast MA` และ `Slow MA` ที่ต้องการทดสอบในแท็บ 'การตั้งค่า'.
        2.  **ค้นหาคู่ที่ดีที่สุด:** ระบบจะทดสอบพารามิเตอร์ทุกคู่ที่เป็นไปได้แบบคู่ขนาน (Parallel Processing) เพื่อหาว่าคู่ไหน (`Fast`, `Slow`) ให้ผลกำไรสุทธิ (Net Profit) สูงสุดตลอดช่วงข้อมูลที่เลือก.
        3.  **จำลองและเปรียบเทียบ:** ระบบจะนำพารามิเตอร์คู่ที่ดีที่สุดมาสร้างกลยุทธ์ `Optimal MRC` และเปรียบเทียบประสิทธิภาพกับกลยุทธ์ Best-Case (`Perfect Foresight`) และ Worst-Case (`Rebalance Daily`).
        """)

if __name__ == "__main__":
    main()
