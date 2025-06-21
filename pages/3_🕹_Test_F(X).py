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
    """คลาสสำหรับเก็บชื่อกลยุทธ์"""
    REBALANCE_DAILY = "Rebalance Daily (Min)"
    PERFECT_FORESIGHT = "Perfect Foresight (Max)"
    ADAPTIVE_SEED_SEARCH = "Adaptive Seed Search (Walk-Forward)"

def initialize_session_state():
    """ตั้งค่าเริ่มต้นสำหรับ Streamlit session state"""
    if 'test_ticker' not in st.session_state: st.session_state.test_ticker = 'BTC-USD'
    if 'start_date' not in st.session_state: st.session_state.start_date = datetime(2023, 1, 1).date()
    if 'end_date' not in st.session_state: st.session_state.end_date = datetime.now().date()
    if 'fix_capital' not in st.session_state: st.session_state.fix_capital = 1500
    if 'window_size' not in st.session_state: st.session_state.window_size = 30
    if 'total_seeds_per_window' not in st.session_state: st.session_state.total_seeds_per_window = 10000
    if 'exploration_ratio' not in st.session_state: st.session_state.exploration_ratio = 0.2 # 20% for exploration

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
    if n > 0: action_array_calc[0] = 1

    cash, sumusd = np.empty(n, dtype=np.float64), np.empty(n, dtype=np.float64)
    amount, asset_value = np.empty(n, dtype=np.float64), np.empty(n, dtype=np.float64)

    initial_price = price_array[0]
    amount[0] = fix / initial_price; cash[0] = fix
    asset_value[0] = amount[0] * initial_price; sumusd[0] = cash[0] + asset_value[0]
    refer = -fix * np.log(initial_price / price_array)

    for i in range(1, n):
        curr_price, prev_amount = price_array[i], amount[i-1]
        if action_array_calc[i] == 0: # Hold
            amount[i], buffer = prev_amount, 0.0
        else: # Rebalance
            amount[i], buffer = fix / curr_price, prev_amount * curr_price - fix
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

    return pd.DataFrame({'net': np.round(net, 2)})

@lru_cache(maxsize=32768)
def get_actions_and_net(seed: int, length: int, price_tuple: Tuple[float, ...], fix: int) -> Tuple[float, np.ndarray]:
    """สร้าง actions และคำนวณ net profit สำหรับ seed ที่กำหนด (Cached)"""
    rng = np.random.default_rng(seed)
    actions = rng.integers(0, 2, size=length, dtype=np.int32)
    actions[0] = 1
    net_array = _calculate_simulation_numba(actions, np.array(price_tuple, dtype=np.float64), fix)
    final_net = net_array[-1] if len(net_array) > 0 else -np.inf
    return final_net, actions

# ==============================================================================
# 3. Strategy Action Generation (NEW MODEL: Adaptive Seed Search)
# ==============================================================================

def _evaluate_seed_batch(seeds_batch: np.ndarray, prices_tuple: Tuple[float, ...], fix: int, length: int) -> List[Tuple[int, float]]:
    """Helper function for parallel execution, returns (seed, net_profit)"""
    results = []
    for seed in seeds_batch:
        final_net, _ = get_actions_and_net(seed, length, prices_tuple, fix)
        results.append((seed, final_net))
    return results

def find_best_seed_adaptively(prices_window: np.ndarray, total_seeds: int, exploration_ratio: float, fix: int) -> Dict[str, Any]:
    """ค้นหา Seed ที่ดีที่สุดด้วยกระบวนการ Adaptive Search"""
    window_len = len(prices_window)
    if window_len < 2: return {'best_seed': 1, 'best_net': 0, 'focused_range': 'N/A'}

    num_exploration_seeds = int(total_seeds * exploration_ratio)
    num_exploitation_seeds = total_seeds - num_exploration_seeds
    prices_tuple = tuple(prices_window)

    # 1. Exploration Phase
    exploration_results = []
    exploration_seeds = np.random.randint(0, 2**32, size=num_exploration_seeds, dtype=np.uint32)

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(_evaluate_seed_batch, np.array_split(exploration_seeds, executor._max_workers)[i], prices_tuple, fix, window_len) for i in range(executor._max_workers)}
        for future in as_completed(futures):
            exploration_results.extend(future.result())

    if not exploration_results: return {'best_seed': 1, 'best_net': 0, 'focused_range': 'N/A'}

    # 2. Analyze & Focus
    df_explore = pd.DataFrame(exploration_results, columns=['seed', 'net']).sort_values('net', ascending=False).reset_index(drop=True)
    top_10_percent_cutoff = df_explore['net'].quantile(0.9)
    top_seeds = df_explore[df_explore['net'] >= top_10_percent_cutoff]['seed']

    if len(top_seeds) < 2:
        best_seed = int(df_explore.iloc[0]['seed'])
        best_net = float(df_explore.iloc[0]['net'])
        return {'best_seed': best_seed, 'best_net': best_net, 'focused_range': 'Fallback'}

    min_promising_seed, max_promising_seed = int(top_seeds.min()), int(top_seeds.max())

    # 3. Exploitation Phase
    exploitation_results = []
    exploitation_seeds = np.random.randint(min_promising_seed, max_promising_seed + 1, size=num_exploitation_seeds, dtype=np.uint32)

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(_evaluate_seed_batch, np.array_split(exploitation_seeds, executor._max_workers)[i], prices_tuple, fix, window_len) for i in range(executor._max_workers)}
        for future in as_completed(futures):
            exploitation_results.extend(future.result())

    # 4. Final Result (SAFE VERSION)
    all_results_df = pd.concat([df_explore, pd.DataFrame(exploitation_results, columns=['seed', 'net'])]).drop_duplicates(subset=['seed'])
    best_result_df = all_results_df.sort_values('net', ascending=False).head(1)

    if best_result_df.empty:
        return {'best_seed': 1, 'best_net': 0, 'focused_range': 'N/A'}

    best_seed = int(best_result_df['seed'].iloc[0])
    best_net = float(best_result_df['net'].iloc[0])

    return {
        'best_seed': best_seed,
        'best_net': best_net,
        'exploration_best_net': df_explore.iloc[0]['net'],
        'focused_range': f"{min_promising_seed:,} - {max_promising_seed:,}"
    }


def generate_adaptive_walk_forward_strategy(ticker_data: pd.DataFrame, window_size: int, total_seeds: int, exploration_ratio: float, fix: int) -> Tuple[np.ndarray, pd.DataFrame]:
    """สร้าง Action Sequence โดยใช้ Adaptive Seed Search และ Walk-Forward Validation"""
    prices = ticker_data['Close'].to_numpy()
    n = len(prices)
    final_actions = np.array([], dtype=int)
    window_details_list = []
    
    num_windows = n // window_size
    progress_bar = st.progress(0, text="เริ่มต้นการวิเคราะห์แบบ Walk-Forward...")
    
    best_actions_for_next_window = np.ones(window_size, dtype=np.int32)
    
    for i in range(num_windows - 1): # We need a learning AND a testing window
        learn_start, learn_end = i * window_size, (i + 1) * window_size
        test_start, test_end = learn_end, learn_end + window_size
        
        learn_prices = prices[learn_start:learn_end]
        learn_dates = ticker_data.index[learn_start:learn_end]
        search_result = find_best_seed_adaptively(learn_prices, total_seeds, exploration_ratio, fix)
        best_seed_found = search_result['best_seed']
        
        test_prices = prices[test_start:test_end]
        test_dates = ticker_data.index[test_start:test_end]
        
        final_actions = np.concatenate((final_actions, best_actions_for_next_window))
        
        test_net_array = _calculate_simulation_numba(best_actions_for_next_window, test_prices, fix)
        walk_forward_net = test_net_array[-1] if len(test_net_array) > 0 else 0
        
        detail = {
            'window_num': i + 1,
            'learn_period': f"{learn_dates[0]:%Y-%m-%d} to {learn_dates[-1]:%Y-%m-%d}",
            'best_seed_found': best_seed_found,
            'test_period': f"{test_dates[0]:%Y-%m-%d} to {test_dates[-1]:%Y-%m-%d}",
            'walk_forward_net': round(walk_forward_net, 2),
            'focused_range': search_result.get('focused_range', 'N/A')
        }
        window_details_list.append(detail)
        
        _, best_actions_for_next_window = get_actions_and_net(best_seed_found, window_size, tuple(learn_prices), fix)
        
        progress_bar.progress((i + 1) / (num_windows - 1), text=f"ประมวลผล Window {i+1}/{num_windows -1}...")
        
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)


# --- Benchmark Strategies ---
@njit(cache=True)
def _generate_perfect_foresight_numba(price_arr: np.ndarray, fix: int) -> np.ndarray:
    n = len(price_arr); actions = np.zeros(n, dtype=np.int32)
    if n < 2: return np.ones(n, dtype=np.int32)
    dp, path = np.zeros(n), np.zeros(n, dtype=np.int32)
    dp[0] = float(fix * 2)
    for i in range(1, n):
        profits = fix * ((price_arr[i] / price_arr[:i]) - 1)
        current_sumusd = dp[:i] + profits
        best_j_idx = np.argmax(current_sumusd)
        dp[i], path[i] = current_sumusd[best_j_idx], best_j_idx
    current_day = np.argmax(dp)
    while current_day > 0: actions[current_day], current_day = 1, path[current_day]
    actions[0] = 1
    return actions

# ==============================================================================
# 4. UI Rendering Functions
# ==============================================================================
def render_settings_tab():
    st.write("⚙️ **พารามิเตอร์พื้นฐาน**")
    c1, c2, c3 = st.columns(3)
    c1.text_input("Ticker", key="test_ticker")
    c2.date_input("วันที่เริ่มต้น", key="start_date")
    c3.date_input("วันที่สิ้นสุด", key="end_date")
    
    st.divider()
    st.write("🧠 **พารามิเตอร์สำหรับโมเดล Adaptive Seed Search (ASS)**")
    s_c1, s_c2, s_c3 = st.columns(3)
    s_c1.number_input("ขนาด Window (วัน)", min_value=10, max_value=252, key="window_size")
    s_c2.number_input("จำนวน Seed ทั้งหมดต่อ Window", min_value=1000, max_value=100000, step=1000, key="total_seeds_per_window")
    s_c3.slider("สัดส่วนการสำรวจ (Exploration Ratio)", 0.05, 0.5, key="exploration_ratio", format="%.2f", help="สัดส่วนของ Seed ที่จะใช้ในการ 'สำรวจ' เพื่อหา 'ย่าน' ที่ดี ก่อนจะใช้ที่เหลือ 'เจาะลึก'")
    
def display_comparison_charts(results: Dict[str, pd.DataFrame]):
    if not results: return
    chart_data = pd.DataFrame({name: df['net'] for name, df in results.items() if not df.empty})
    st.write("📊 **เปรียบเทียบกำไรสุทธิสะสม (Net Profit)**")
    st.line_chart(chart_data)

def render_model_tab():
    st.markdown("### 🧠 Adaptive Seed Search (ASS) with Walk-Forward Validation")
    
    if st.button("🚀 เริ่มการค้นหาแบบอัจฉริยะและวิเคราะห์", type="primary"):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้อง"); return
            
        with st.spinner(f"กำลังดึงข้อมูล **{st.session_state.test_ticker}**..."):
            ticker_data = get_ticker_data(st.session_state.test_ticker, str(st.session_state.start_date), str(st.session_state.end_date))
        if ticker_data.empty: st.error("ไม่พบข้อมูล"); return
        
        prices_np = ticker_data['Close'].to_numpy()
        prices_list = ticker_data['Close'].tolist()
        num_days = len(prices_list)

        actions_ass, df_windows = generate_adaptive_walk_forward_strategy(
            ticker_data, st.session_state.window_size, st.session_state.total_seeds_per_window,
            st.session_state.exploration_ratio, st.session_state.fix_capital
        )
        st.success("ค้นพบกลยุทธ์และทำการตรวจสอบความเสถียร (Walk-Forward) เรียบร้อยแล้ว!")
        
        with st.spinner("กำลังจำลองกลยุทธ์เพื่อเปรียบเทียบ..."):
            actions_max = _generate_perfect_foresight_numba(prices_np, st.session_state.fix_capital).tolist()
            actions_min = np.ones(num_days, dtype=np.int32).tolist()
            
            results = {}
            strategy_map = {
                Strategy.ADAPTIVE_SEED_SEARCH: actions_ass.tolist(),
                Strategy.PERFECT_FORESIGHT: actions_max,
                Strategy.REBALANCE_DAILY: actions_min,
            }
            
            for name, actions in strategy_map.items():
                sim_prices = prices_list[:len(actions)]
                df = run_simulation(sim_prices, actions, st.session_state.fix_capital)
                if not df.empty: df.index = ticker_data.index[:len(df)]
                results[name] = df
        
        st.write("---")
        display_comparison_charts(results)
        
        # Check if results are available before accessing them
        if not results[Strategy.ADAPTIVE_SEED_SEARCH].empty:
            final_net_max = results[Strategy.PERFECT_FORESIGHT]['net'].iloc[-1]
            final_net_ass = results[Strategy.ADAPTIVE_SEED_SEARCH]['net'].iloc[-1]
            final_net_min = results[Strategy.REBALANCE_DAILY]['net'].iloc[-1]
            
            col1, col2, col3 = st.columns(3)
            col1.metric(f"🥇 {Strategy.PERFECT_FORESIGHT}", f"${final_net_max:,.2f}")
            col2.metric(f"🧠 {Strategy.ADAPTIVE_SEED_SEARCH}", f"${final_net_ass:,.2f}", delta=f"{final_net_ass - final_net_min:,.2f} vs Min")
            col3.metric(f"🥉 {Strategy.REBALANCE_DAILY}", f"${final_net_min:,.2f}")
        else:
            st.warning("ไม่สามารถคำนวณผลลัพธ์ของ Adaptive Seed Search ได้ อาจเนื่องมาจากข้อมูลไม่เพียงพอสำหรับ Walk-Forward")

        st.write("---")
        st.write("🔍 **รายละเอียดการค้นหาและทดสอบแบบ Walk-Forward ราย Window**")
        st.dataframe(df_windows[['window_num', 'learn_period', 'best_seed_found', 'test_period', 'walk_forward_net', 'focused_range']], use_container_width=True)
        csv = df_windows.to_csv(index=False)
        st.download_button(label="📥 ดาวน์โหลดผลวิเคราะห์ (CSV)", data=csv, file_name=f'adaptive_seed_search_{st.session_state.test_ticker}.csv', mime='text/csv')

# ==============================================================================
# 5. Main Application
# ==============================================================================
def main():
    st.set_page_config(page_title="Adaptive Seed Search", page_icon="🧠", layout="wide")
    st.markdown("### 🧠 Adaptive Seed Search (ASS) Optimizer")
    st.caption("โมเดลค้นหา Best Seed อย่างชาญฉลาดโดยใช้หลักการ Explore/Exploit และตรวจสอบด้วย Walk-Forward Validation")

    initialize_session_state()

    tab_settings, tab_model = st.tabs(["⚙️ การตั้งค่า", "🚀 วิเคราะห์และแสดงผล"])
    with tab_settings: render_settings_tab()
    with tab_model: render_model_tab()
    
    with st.expander("📖 คำอธิบายหลักการทำงานของโมเดล (Adaptive Seed Search)"):
        st.markdown("""
        ### แนวคิดหลัก: "ค้นหาอย่างฉลาด ไม่ใช่แค่สุ่มไปเรื่อยๆ"
        
        โมเดลนี้เลิกการค้นหาแบบ Brute-Force ที่ไร้ทิศทาง และเปลี่ยนเป็นการค้นหาแบบมีเป้าหมาย โดยเลียนแบบกระบวนการเรียนรู้ของมนุษย์:

        1.  **สำรวจ (Exploration Phase):**
            - ในช่วงแรก ระบบจะใช้ Seed จำนวนหนึ่ง (ตาม `Exploration Ratio`) เพื่อสุ่มหาในพื้นที่กว้างๆ
            - **เปรียบเหมือน:** การเดินสำรวจป่าเพื่อดูว่าบริเวณไหนมีเห็ดเยอะที่สุด

        2.  **วิเคราะห์และเจาะจง (Analyze & Focus):**
            - ระบบจะนำผลลัพธ์จากการสำรวจมาวิเคราะห์ ว่า Seed ที่ให้ผลตอบแทนดี (Top 10%) มักจะอยู่ "ย่าน" ไหน
            - **เปรียบเหมือน:** เมื่อรู้ว่าเห็ดเยอะทางทิศเหนือ เราก็จะมุ่งหน้าไปทางนั้น

        3.  **เจาะหาผลประโยชน์ (Exploitation Phase):**
            - ระบบจะใช้ Seed ที่เหลือทั้งหมด ค้นหาอย่างละเอียด **เฉพาะในย่านที่พบว่าดีที่สุด**
            - **เปรียบเหมือน:** เมื่อไปถึงทิศเหนือแล้ว ก็จะค่อยๆ เดินหาเห็ดอย่างละเอียดในบริเวณนั้น ไม่เสียเวลาไปเดินที่อื่นอีก

        4.  **ตรวจสอบความเสถียร (Walk-Forward Validation):**
            - เพื่อป้องกันไม่ให้โมเดล "ท่องจำ" อดีตได้ดีเกินไป (Overfitting) เราจะนำ Seed ที่ดีที่สุดที่หาได้จาก **Window ที่ 1** ไปใช้เทรดใน **Window ที่ 2**
            - ผลลัพธ์ที่แสดงในกราฟ `Adaptive Seed Search` คือผลลัพธ์จากการทดสอบแบบ Walk-Forward นี้ ซึ่งสะท้อนประสิทธิภาพที่น่าจะเกิดขึ้นจริงได้ดีกว่า
            - **เปรียบเหมือน:** การทดสอบว่า "วิธีการหาเห็ดที่ดีที่สุดของเมื่อวาน" ยังใช้ได้ผลกับ "ป่าของวันนี้" หรือไม่
        """)

if __name__ == "__main__":
    main()
