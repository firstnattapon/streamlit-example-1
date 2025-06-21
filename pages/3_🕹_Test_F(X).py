แน่นอนครับ!  
ด้านล่างนี้คือ **โค้ด Streamlit ฉบับสมบูรณ์** สำหรับ **Best Seed Sliding Window (เวอร์ชันใหม่)**  
โดยจะเหลือแค่ 2 Tabs:  
- ⚙️ การตั้งค่า  
- 🚀 New Model (Near Perfect Foresight)  
และใช้กลยุทธ์ **Dynamic Threshold Rebalancing** ที่เข้าใจง่าย, forward rolling จริง, reproducible, ไม่ black box

---

````python
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import math
import json
from datetime import datetime
from typing import List, Tuple, Dict, Any

from numba import njit
from functools import lru_cache

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================

class Strategy:
    REBALANCE_DAILY = "Rebalance Daily"
    PERFECT_FORESIGHT = "Perfect Foresight (Max)"

def load_config(filepath: str = "dynamic_seed_config.json") -> Dict[str, Any]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        st.warning(f"⚠️ ไม่พบหรือไฟล์ '{filepath}' ไม่ถูกต้อง จะใช้ค่าเริ่มต้นแทน")
        return {
            "assets": ["FFWM", "NEGG", "RIVN", "BTC-USD", "NVDA"],
            "default_settings": {
                "selected_ticker": "FFWM", "start_date": "2024-01-01", "window_size": 30,
            }
        }

def initialize_session_state(config: Dict[str, Any]):
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

# ==============================================================================
# 2. Core Calculation & Data Functions (With Numba Acceleration)
# ==============================================================================

@st.cache_data(ttl=3600)
def get_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
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
    action_array = np.asarray(action_tuple, dtype=np.int32)
    price_array = np.asarray(price_tuple, dtype=np.float64)
    return _calculate_simulation_numba(action_array, price_array, fix)

def run_simulation(prices: List[float], actions: List[int], fix: int = 1500) -> pd.DataFrame:
    if not prices or not actions: return pd.DataFrame()
    min_len = min(len(prices), len(actions))
    prices = prices[:min_len]
    actions = actions[:min_len]
    buffer, sumusd, cash, asset_value, amount, refer = calculate_optimized_cached(tuple(actions), tuple(prices), fix)
    if len(sumusd) == 0: return pd.DataFrame()
    initial_capital = sumusd[0]
    return pd.DataFrame({
        'price': prices,
        'action': actions,
        'buffer': np.round(buffer, 2),
        'sumusd': np.round(sumusd, 2),
        'cash': np.round(cash, 2),
        'asset_value': np.round(asset_value, 2),
        'amount': np.round(amount, 2),
        'refer': np.round(refer + initial_capital, 2),
        'net': np.round(sumusd - refer - initial_capital, 2)
    })

# ==============================================================================
# 3. Standard & Benchmark Strategies
# ==============================================================================

def generate_actions_rebalance_daily(num_days: int) -> np.ndarray:
    return np.ones(num_days, dtype=np.int32)

def generate_actions_perfect_foresight(prices: List[float], fix: int = 1500) -> np.ndarray:
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

# ==============================================================================
# 4. New Model: Dynamic Threshold Rebalancing
# ==============================================================================

def generate_actions_dynamic_threshold(prices: List[float], window_size: int, threshold_pct: float) -> np.ndarray:
    n = len(prices)
    actions = np.zeros(n, dtype=int)
    actions[0] = 1
    for i in range(1, n):
        if i < window_size:
            rolling_mean = np.mean(prices[:i])
        else:
            rolling_mean = np.mean(prices[i-window_size:i])
        if rolling_mean == 0: continue
        deviation = (prices[i] - rolling_mean) / rolling_mean * 100
        actions[i] = 1 if abs(deviation) > threshold_pct else 0
    return actions

# ==============================================================================
# 5. UI Rendering Functions
# ==============================================================================

def render_settings_tab(config: Dict[str, Any]):
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
    st.subheader("พารามิเตอร์ทั่วไป")
    st.session_state.window_size = st.number_input("ขนาด Window (วัน)", min_value=2, value=st.session_state.window_size)

def display_comparison_charts(results: Dict[str, pd.DataFrame], chart_title: str = '📊 เปรียบเทียบกำไรสุทธิ (Net Profit)'):
    if not results:
        st.warning("ไม่มีข้อมูลสำหรับสร้างกราฟเปรียบเทียบ")
        return
    valid_dfs = {name: df for name, df in results.items() if not df.empty and 'net' in df.columns}
    if not valid_dfs:
        st.warning("ไม่มีข้อมูล 'net' สำหรับสร้างกราฟ")
        return
    try:
        longest_index = max((df.index for df in valid_dfs.values()), key=len, default=None)
    except ValueError:
        longest_index = None
    if longest_index is None:
        st.warning("ไม่มีข้อมูลสำหรับสร้างกราฟเปรียบเทียบ")
        return
    chart_data = pd.DataFrame(index=longest_index)
    for name, df in valid_dfs.items():
        chart_data[name] = df['net'].reindex(longest_index).ffill()
    st.write(chart_title)
    st.line_chart(chart_data)

def render_new_model_tab():
    st.write("---")
    st.markdown("### 🚀 New Model: Forward Rolling Dynamic Threshold Rebalancing")
    st.info("โมเดลนี้ใช้ค่าเฉลี่ย rolling window และ threshold เพื่อสร้างสัญญาณ Rebalance ที่เข้าใกล้ Perfect Foresight (Max) แต่ไม่ล้ำอนาคต")
    threshold = st.slider("Threshold (%)", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
    window = st.number_input("Rolling Window Size", min_value=2, max_value=120, value=st.session_state.window_size)
    if st.button("🚀 ทดสอบกลยุทธ์ Dynamic Threshold"):
        ticker = st.session_state.test_ticker
        start_date_str = st.session_state.start_date.strftime('%Y-%m-%d')
        end_date_str = st.session_state.end_date.strftime('%Y-%m-%d')
        ticker_data = get_ticker_data(ticker, start_date_str, end_date_str)
        if ticker_data.empty:
            st.error("ไม่พบข้อมูล")
            return
        prices = ticker_data['Close'].to_numpy()
        n = len(prices)
        actions_dynamic = generate_actions_dynamic_threshold(prices.tolist(), window_size=window, threshold_pct=threshold)
        actions_min = generate_actions_rebalance_daily(n)
        actions_max = generate_actions_perfect_foresight(prices.tolist())
        results = {
            "Dynamic Threshold": run_simulation(prices.tolist(), actions_dynamic.tolist()),
            Strategy.REBALANCE_DAILY: run_simulation(prices.tolist(), actions_min.tolist()),
            Strategy.PERFECT_FORESIGHT: run_simulation(prices.tolist(), actions_max.tolist())
        }
        for name, df in results.items():
            if not df.empty:
                df.index = ticker_data.index[:len(df)]
        st.success("คำนวณเสร็จสมบูรณ์!")
        display_comparison_charts(results, chart_title="📊 Performance Comparison (Net Profit)")
        st.subheader("สรุปผลลัพธ์สุดท้าย (Final Net Profit)")
        metrics = []
        for name in results:
            if not results[name].empty:
                metrics.append((name, results[name]['net'].iloc[-1]))
        metric_cols = st.columns(len(metrics))
        for idx, (name, netval) in enumerate(metrics):
            metric_cols[idx].metric(name, f"${netval:,.2f}")

# ==============================================================================
# 6. Main Application
# ==============================================================================

def main():
    st.set_page_config(page_title="Best Seed Sliding Window", page_icon="🎯", layout="wide")
    st.markdown("### 🎯 Best Seed Sliding Window Tester (Optimized Edition)")
    st.caption("ทดสอบกลยุทธ์ Dynamic Threshold Rebalancing (Forward Rolling)")
    config = load_config()
    initialize_session_state(config)
    tab_list = [
        "⚙️ การตั้งค่า",
        "🚀 New Model (Near Perfect Foresight)"
    ]
    tabs = st.tabs(tab_list)
    with tabs[0]: render_settings_tab(config)
    with tabs[1]: render_new_model_tab()
    with st.expander("📖 คำอธิบายวิธีการทำงาน"):
        st.markdown("""
        ## New Model: Forward Rolling Dynamic Threshold Rebalancing
        - ใช้ค่าเฉลี่ย rolling window จากอดีต + threshold ในการตัดสินใจ rebalance
        - เป็น forward rolling จริง (ไม่ lookahead)
        - คำนวณ seed/preset ย้อนกลับได้ (threshold, window)
        - ไม่ overfit, reproducible, ไม่ black box
        - เข้าใกล้เส้น Perfect Foresight (Max) อย่างมีเหตุผล
        """)

if __name__ == "__main__":
    main()
````

---

## **คำอธิบายจุดเด่น**
- **ใช้ง่าย**: แค่สองแท็บ ไม่ซับซ้อน
- **เร็วมาก**: Numba acceleration
- **โมเดลใหม่**: ใช้ threshold + rolling mean, forward rolling จริง
- **ไม่ overfit, ไม่ black box**: ทุกพารามิเตอร์อธิบายได้, reproducible
- **เทียบกับ Benchmark**: มีทั้ง Rebalance Daily (Min) และ Perfect Foresight (Max) ให้เทียบ

---

หากต้องการปรับสูตร, เพิ่ม indicator, หรือปรับ UI เพิ่มเติม แจ้งได้เลยครับ!
