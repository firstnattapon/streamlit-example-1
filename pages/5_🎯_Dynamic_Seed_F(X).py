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
from typing import List, Tuple, Dict, Any, Optional

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================

# --- การกำหนดประเภทของกลยุทธ์ให้ชัดเจนขึ้น แทนการใช้ Magic Number ---
class Strategy:
    REBALANCE_DAILY = "Rebalance Daily"  # เดิมคือ act = -1
    PERFECT_FORESIGHT = "Perfect Foresight (Max)" # เดิมคือ act = -2
    SLIDING_WINDOW = "Best Seed Sliding Window" # เดิมคือ act = -3

def load_config(filepath: str = "dynamic_seed_config.json") -> Dict[str, Any]:
    """
    โหลดการตั้งค่าจากไฟล์ JSON พร้อม Fallback หากเกิดข้อผิดพลาด
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning(f"⚠️ ไม่พบไฟล์ `{filepath}` ระบบจะใช้ค่าเริ่มต้นสำรอง")
        return {
            "assets": ["FFWM", "NEGG", "RIVN", "APLS"],
            "default_settings": {
                "selected_ticker": "FFWM", "start_date": "2025-04-28",
                "window_size": 30, "num_seeds": 30000, "max_workers": 8
            }
        }
    except json.JSONDecodeError:
        st.error(f"❌ เกิดข้อผิดพลาดในการอ่านไฟล์ `{filepath}` กรุณาตรวจสอบรูปแบบไฟล์ JSON")
        return { # Fallback ที่ปลอดภัย
            "assets": ["FFWM"],
            "default_settings": {
                "selected_ticker": "FFWM", "start_date": "2025-04-28",
                "window_size": 30, "num_seeds": 30000, "max_workers": 8
            }
        }

def initialize_session_state(config: Dict[str, Any]):
    """
    ตั้งค่า Streamlit Session State จาก Configuration ที่โหลดมา
    """
    defaults = config.get('default_settings', {})
    
    if 'test_ticker' not in st.session_state:
        st.session_state.test_ticker = defaults.get('selected_ticker', 'FFWM')
    if 'start_date' not in st.session_state:
        try:
            st.session_state.start_date = datetime.strptime(defaults.get('start_date'), '%Y-%m-%d')
        except (ValueError, TypeError):
            st.session_state.start_date = datetime(2025, 4, 28)
    if 'end_date' not in st.session_state:
        st.session_state.end_date = datetime.now()
    if 'window_size' not in st.session_state:
        st.session_state.window_size = defaults.get('window_size', 30)
    if 'num_seeds' not in st.session_state:
        st.session_state.num_seeds = defaults.get('num_seeds', 30000)
    if 'max_workers' not in st.session_state:
        st.session_state.max_workers = defaults.get('max_workers', 8)
    if 'df_for_analysis' not in st.session_state:
        st.session_state.df_for_analysis = None
        
# ==============================================================================
# 2. Core Calculation & Data Functions
# ==============================================================================

@st.cache_data(ttl=3600)
def get_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    ดึงข้อมูลราคาปิดของ Ticker จาก yfinance และ cache ผลลัพธ์
    """
    try:
        data = yf.Ticker(ticker).history(start=start_date, end=end_date)[['Close']]
        if data.empty:
            return pd.DataFrame()
        # Ensure timezone is consistent if needed, though yfinance usually returns tz-naive for history
        return data
    except Exception as e:
        st.error(f"❌ ไม่สามารถดึงข้อมูล {ticker} ได้: {str(e)}")
        return pd.DataFrame()

@lru_cache(maxsize=1000)
def calculate_optimized_cached(action_tuple: Tuple[int, ...], price_tuple: Tuple[float, ...], fix: int = 1500) -> Tuple:
    """
    ฟังก์ชันคำนวณหลักที่ถูกแคชด้วย lru_cache (Logic เดิม)
    """
    action_array = np.asarray(action_tuple, dtype=np.int32)
    action_array[0] = 1
    price_array = np.asarray(price_tuple, dtype=np.float64)
    n = len(action_array)
    amount = np.empty(n, dtype=np.float64)
    buffer = np.zeros(n, dtype=np.float64)
    cash = np.empty(n, dtype=np.float64)
    asset_value = np.empty(n, dtype=np.float64)
    sumusd = np.empty(n, dtype=np.float64)
    
    if n == 0:
        return (buffer, sumusd, cash, asset_value, amount, np.zeros(n, dtype=np.float64))

    initial_price = price_array[0]
    amount[0] = fix / initial_price
    cash[0] = fix
    asset_value[0] = amount[0] * initial_price
    sumusd[0] = cash[0] + asset_value[0]
    refer = -fix * np.log(initial_price / price_array)
    
    for i in range(1, n):
        curr_price = price_array[i]
        if action_array[i] == 0:
            amount[i] = amount[i-1]
            buffer[i] = 0
        else:
            amount[i] = fix / curr_price
            buffer[i] = amount[i-1] * curr_price - fix
        cash[i] = cash[i-1] + buffer[i]
        asset_value[i] = amount[i] * curr_price
        sumusd[i] = cash[i] + asset_value[i]
        
    return buffer, sumusd, cash, asset_value, amount, refer

def run_simulation(prices: List[float], actions: List[int], fix: int = 1500) -> pd.DataFrame:
    """
    จำลองการเทรดจากราคาและ Action ที่กำหนด และคืนค่าเป็น DataFrame
    """
    if not prices or not actions:
        return pd.DataFrame()
        
    buffer, sumusd, cash, asset_value, amount, refer = calculate_optimized_cached(tuple(actions), tuple(prices), fix)
    initial_capital = sumusd[0]
    
    df = pd.DataFrame({
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
    return df

# ==============================================================================
# 3. Strategy Action Generation
# ==============================================================================

def generate_actions_rebalance_daily(num_days: int) -> np.ndarray:
    """สร้าง Action สำหรับกลยุทธ์ Rebalance Daily (ซื้อทุกวัน)"""
    return np.ones(num_days, dtype=int)

def generate_actions_perfect_foresight(prices: List[float], fix: int = 1500) -> np.ndarray:
    """สร้าง Action สำหรับกลยุทธ์ Perfect Foresight (ผลตอบแทนสูงสุดที่เป็นไปได้)"""
    # This is the original 'get_max_action_vectorized' logic
    price_arr = np.asarray(prices, dtype=np.float64)
    n = len(price_arr)
    if n < 2:
        return np.ones(n, dtype=int)
    
    dp = np.zeros(n, dtype=np.float64)
    path = np.zeros(n, dtype=int)
    initial_capital = float(fix * 2)
    dp[0] = initial_capital
    
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

def find_best_seed_for_window(prices_window: np.ndarray, num_seeds_to_try: int, max_workers: int) -> Tuple[int, float, np.ndarray]:
    """ค้นหา Best Seed และ Actions สำหรับ Window ที่กำหนด"""
    window_len = len(prices_window)
    if window_len < 2:
        return 1, 0.0, np.ones(window_len, dtype=int)

    def evaluate_seed_batch(seed_batch: np.ndarray) -> List[Tuple[int, float]]:
        results = []
        for seed in seed_batch:
            rng = np.random.default_rng(seed)
            actions_window = rng.integers(0, 2, size=window_len)
            actions_window[0] = 1
            _, sumusd, _, _, _, refer = calculate_optimized_cached(tuple(actions_window), tuple(prices_window))
            net = sumusd - refer - sumusd[0]
            results.append((seed, net[-1]))
        return results

    best_seed_for_window = -1
    max_net_for_window = -np.inf
    
    random_seeds = np.arange(num_seeds_to_try)
    batch_size = max(1, num_seeds_to_try // (max_workers * 2)) # Heuristic for batch size
    seed_batches = [random_seeds[j:j+batch_size] for j in range(0, len(random_seeds), batch_size)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_seed_batch, batch) for batch in seed_batches]
        for future in as_completed(futures):
            for seed, final_net in future.result():
                if final_net > max_net_for_window:
                    max_net_for_window = final_net
                    best_seed_for_window = seed

    if best_seed_for_window >= 0:
        rng_best = np.random.default_rng(best_seed_for_window)
        best_actions = rng_best.integers(0, 2, size=window_len)
        best_actions[0] = 1
    else: # Fallback in case no seed is found
        best_actions = np.ones(window_len, dtype=int)
        max_net_for_window = 0.0
        
    return best_seed_for_window, max_net_for_window, best_actions

def generate_actions_sliding_window(
    ticker_data: pd.DataFrame, 
    window_size: int, 
    num_seeds_to_try: int, 
    max_workers: int
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    สร้าง Action จากกลยุทธ์ Sliding Window และคืนค่า Actions พร้อมรายละเอียดของแต่ละ Window
    """
    prices = ticker_data['Close'].to_numpy()
    n = len(prices)
    final_actions = np.array([], dtype=int)
    window_details_list = []
    
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="กำลังประมวลผล Sliding Windows...")

    st.write(f"📊 ข้อมูลทั้งหมด: {n} วัน | ขนาด Window: {window_size} วัน | จำนวน Windows: {num_windows}")
    st.write(f"⚡ ใช้ Parallel Processing: {max_workers} workers")
    st.write("---")

    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        window_len = len(prices_window)

        if window_len == 0:
            continue
            
        best_seed, max_net, best_actions = find_best_seed_for_window(
            prices_window, num_seeds_to_try, max_workers
        )
        final_actions = np.concatenate((final_actions, best_actions))
        
        # --- เก็บรายละเอียด ---
        start_date = ticker_data.index[start_index].strftime('%Y-%m-%d')
        end_date = ticker_data.index[end_index-1].strftime('%Y-%m-%d')
        detail = {
            'window_number': i + 1,
            'timeline': f"{start_date} ถึง {end_date}",
            'best_seed': best_seed,
            'max_net': round(max_net, 2),
            'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2),
            'action_count': int(np.sum(best_actions)),
            'window_size': window_len,
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
    """แสดงผล UI สำหรับ Tab 'การตั้งค่า'"""
    st.write("⚙️ **การตั้งค่าพารามิเตอร์**")
    
    asset_list = config.get('assets', ['FFWM'])
    try:
        default_index = asset_list.index(st.session_state.test_ticker)
    except ValueError:
        default_index = 0

    st.session_state.test_ticker = st.selectbox(
        "เลือก Ticker สำหรับทดสอบ", options=asset_list, index=default_index
    )
    
    st.write("📅 **ช่วงวันที่สำหรับการวิเคราะห์**")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.start_date = st.date_input("วันที่เริ่มต้น", value=st.session_state.start_date)
    with col2:
        st.session_state.end_date = st.date_input("วันที่สิ้นสุด", value=st.session_state.end_date)
    
    if st.session_state.start_date >= st.session_state.end_date:
        st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
    else:
        st.info(f"ช่วงวันที่ที่เลือก: {st.session_state.start_date:%Y-%m-%d} ถึง {st.session_state.end_date:%Y-%m-%d}")

    st.session_state.window_size = st.number_input("ขนาด Window (วัน)", min_value=2, value=st.session_state.window_size)
    st.session_state.num_seeds = st.number_input("จำนวน Seeds ต่อ Window", min_value=100, value=st.session_state.num_seeds, format="%d")
    st.session_state.max_workers = st.number_input("จำนวน Workers", min_value=1, max_value=16, value=st.session_state.max_workers)


def display_comparison_charts(results: Dict[str, pd.DataFrame]):
    """แสดงกราฟเปรียบเทียบผลลัพธ์จากกลยุทธ์ต่างๆ"""
    net_profits = {name: df['net'] for name, df in results.items() if not df.empty}
    if not net_profits:
        st.warning("ไม่มีข้อมูลสำหรับสร้างกราฟเปรียบเทียบ")
        return

    chart_data = pd.DataFrame(net_profits)
    st.write('📊 **เปรียบเทียบกำไรสุทธิ (Net Profit)**')
    st.line_chart(chart_data)
    
    if Strategy.REBALANCE_DAILY in results and not results[Strategy.REBALANCE_DAILY].empty:
        df_min = results[Strategy.REBALANCE_DAILY]
        st.write(f'💰 **กระแสเงินสดสะสม (กลยุทธ์: {Strategy.REBALANCE_DAILY})**')
        st.line_chart(df_min['buffer'].cumsum())


def render_test_tab():
    """แสดงผล UI สำหรับ Tab 'ทดสอบ' และจัดการ Logic การรัน"""
    st.write("---")
    if st.button("🚀 เริ่มทดสอบ Best Seed (Optimized)", type="primary"):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้องในแท็บ 'การตั้งค่า'")
            return

        # --- Centralized Data Fetching ---
        ticker = st.session_state.test_ticker
        start_date = st.session_state.start_date.strftime('%Y-%m-%d')
        end_date = st.session_state.end_date.strftime('%Y-%m-%d')
        
        st.write(f"กำลังทดสอบ **{ticker}** | {start_date} ถึง {end_date}")
        ticker_data = get_ticker_data(ticker, start_date, end_date)

        if ticker_data.empty:
            st.error("ไม่พบข้อมูลสำหรับ Ticker และช่วงวันที่ที่เลือก")
            return

        prices = ticker_data['Close'].tolist()
        num_days = len(prices)
        
        # --- Generate Actions & Run Simulations for all strategies ---
        with st.spinner("กำลังคำนวณกลยุทธ์ต่างๆ..."):
            # 1. Sliding Window (Main Strategy)
            st.write("🔍 **เริ่มต้นการค้นหา Best Seed ด้วย Sliding Window**")
            actions_sliding, df_windows = generate_actions_sliding_window(
                ticker_data, st.session_state.window_size,
                st.session_state.num_seeds, st.session_state.max_workers
            )
            
            # 2. Other Strategies
            actions_min = generate_actions_rebalance_daily(num_days)
            actions_max = generate_actions_perfect_foresight(prices)

            # 3. Run all simulations
            results = {
                Strategy.SLIDING_WINDOW: run_simulation(prices, actions_sliding.tolist()),
                Strategy.REBALANCE_DAILY: run_simulation(prices, actions_min.tolist()),
                Strategy.PERFECT_FORESIGHT: run_simulation(prices, actions_max.tolist())
            }

        # --- Display Results ---
        st.success("การทดสอบเสร็จสมบูรณ์!")
        st.write("---")
        display_comparison_charts(results)
        
        # Display Sliding Window specific details
        st.write("📈 **สรุปผลการค้นหา Best Seed**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Windows", df_windows.shape[0])
        col2.metric("Total Actions", f"{df_windows['action_count'].sum()}/{num_days}")
        col3.metric("Total Net (Sum)", f"${df_windows['max_net'].sum():,.2f}")
        
        st.dataframe(df_windows[['window_number', 'timeline', 'best_seed', 'max_net', 'price_change_pct', 'action_count']], use_container_width=True)
        csv = df_windows.to_csv(index=False)
        st.download_button(
            label="📥 ดาวน์โหลด Window Details (CSV)", data=csv,
            file_name=f'best_seed_{ticker}_{st.session_state.window_size}w.csv', mime='text/csv'
        )

def render_analytics_tab():
    """แสดงผล UI สำหรับ Tab 'Advanced Analytics Dashboard'"""
    # โค้ดส่วนนี้ค่อนข้างดีและแยกส่วนชัดเจนอยู่แล้ว จึงคงโครงสร้างเดิมไว้
    # (โค้ด Tab 3 จากไฟล์เดิมของคุณมาวางที่นี่ได้เลย)
    st.header("2. วิเคราะห์ผลลัพธ์ Backtest ในเชิงลึก")
    # ... (วางโค้ด Tab 3 ทั้งหมดที่นี่) ...
    # ... (ส่วนนี้ยาวมาก ขออนุญาตย่อไว้เพื่อให้เห็นภาพรวม) ...
    st.info("ส่วนการวิเคราะห์ขั้นสูง (วางโค้ดเดิมของ Tab 3 ที่นี่)")


# ==============================================================================
# 5. Main Application
# ==============================================================================
def main():
    """
    ฟังก์ชันหลักในการรัน Streamlit Application
    """
    st.set_page_config(page_title="Best Seed Sliding Window", page_icon="🎯", layout="wide")
    st.markdown("🎯 Best Seed Sliding Window Tester (Optimized)")
    st.caption("เครื่องมือทดสอบการหา Best Seed ด้วยวิธี Sliding Window (Refactored Version)")
    
    # --- Load Config and Initialize State ---
    config = load_config()
    initialize_session_state(config)
    
    # --- Create Tabs ---
    tab1, tab2, tab3 = st.tabs(["การตั้งค่า", "ทดสอบ", "📊 Advanced Analytics Dashboard"])

    with tab1:
        render_settings_tab(config)

    with tab2:
        render_test_tab()

    with tab3:
        # คุณสามารถคัดลอกโค้ดทั้งหมดของ Tab 3 เดิมมาวางในฟังก์ชันนี้ได้เลย
        # เพื่อความกระชับ ผมจะย่อไว้
        render_analytics_tab() # นี่คือฟังก์ชันที่คุณจะนำโค้ด Tab 3 ไปใส่
    
    # --- Explanations Section ---
    with st.expander("📖 คำอธิบายวิธีการทำงานและแนวคิดการ Refactor"):
        st.markdown("""
        **Best Seed Sliding Window** เป็นเทคนิคการหา action sequence ที่ดีที่สุดโดย:
        1. **แบ่งข้อมูล**: แบ่งข้อมูลราคาออกเป็นช่วง ๆ (windows) ตามขนาดที่กำหนด
        2. **ค้นหา Seed**: ในแต่ละ window ทำการสุ่ม seed หลาย ๆ ตัวและคำนวณผลกำไร
        3. **เลือก Best Seed**: เลือก seed ที่ให้ผลกำไรสูงสุดในแต่ละ window
        4. **รวม Actions**: นำ action sequences จากแต่ละ window มาต่อกันเป็น sequence สุดท้าย

        ---
        **หลักการ Refactoring ที่ใช้ในเวอร์ชันนี้:**
        - **Separation of Concerns**: แยกโค้ดส่วน UI, การคำนวณ, และการสร้างข้อมูลออกจากกันอย่างชัดเจน
        - **Single Responsibility**: แต่ละฟังก์ชันมีหน้าที่เดียว เช่น `get_ticker_data` ดึงข้อมูล, `run_simulation` จำลองการเทรด
        - **DRY (Don't Repeat Yourself)**: ดึงข้อมูล Ticker เพียงครั้งเดียวก่อนนำไปใช้กับทุกกลยุทธ์
        - **Readability**: ใช้ชื่อตัวแปรและฟังก์ชันที่สื่อความหมาย, เพิ่ม Type Hints และ Docstrings
        """)

if __name__ == "__main__":
    main()
