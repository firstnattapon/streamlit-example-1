# เริ่ม
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import math
import json
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================

class Strategy:
    REBALANCE_DAILY = "Rebalance Daily (Min)"
    PERFECT_FORESIGHT = "Perfect Foresight (Max)"
    SLIDING_WINDOW = "Best Seed Sliding Window"
    MANUAL_SEED = "Manual Seed Strategy"

def load_config(filepath: str = "dynamic_seed_config.json") -> Dict[str, Any]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "assets": ["FFWM", "NEGG", "RIVN", "APLS", "NVTS", "QXO", "RXRX", "AGL"],
            "default_settings": {
                "selected_ticker": "FFWM", "start_date": "2025-04-28",
                "window_size": 30, "num_seeds": 30000, "max_workers": 8
            }
        }

def initialize_session_state(config: Dict[str, Any]):
    defaults = config.get('default_settings', {})
    if 'test_ticker' not in st.session_state:
        st.session_state.test_ticker = defaults.get('selected_ticker', 'FFWM')
    if 'start_date' not in st.session_state:
        st.session_state.start_date = datetime.strptime(defaults.get('start_date', '2025-04-28'), '%Y-%m-%d')
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
    if 'manual_seed_lines' not in st.session_state:
        st.session_state.manual_seed_lines = [
            {'seed': 1234, 'size': 60, 'tail': 30},
            {'seed': 7777, 'size': 30, 'tail': 10}
        ]

# ==============================================================================
# 2. Core Calculation & Data Functions (ไม่มีการแก้ไข)
# ==============================================================================

@st.cache_data(ttl=3600)
def get_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
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

@lru_cache(maxsize=2048)
def calculate_optimized_cached(action_tuple: Tuple[int, ...], price_tuple: Tuple[float, ...], fix: int = 1500) -> Tuple:
    action_array = np.asarray(action_tuple, dtype=np.int32)
    price_array = np.asarray(price_tuple, dtype=np.float64)
    n = len(action_array)
    if n == 0: return (np.array([]),) * 6
    action_array_calc = action_array.copy(); action_array_calc[0] = 1
    amount = np.empty(n, dtype=np.float64); buffer = np.zeros(n, dtype=np.float64)
    cash = np.empty(n, dtype=np.float64); asset_value = np.empty(n, dtype=np.float64)
    sumusd = np.empty(n, dtype=np.float64)
    initial_price = price_array[0]; amount[0] = fix / initial_price; cash[0] = fix
    asset_value[0] = amount[0] * initial_price; sumusd[0] = cash[0] + asset_value[0]
    refer = -fix * np.log(initial_price / price_array)
    for i in range(1, n):
        curr_price = price_array[i]
        if action_array_calc[i] == 0: amount[i] = amount[i-1]; buffer[i] = 0
        else: amount[i] = fix / curr_price; buffer[i] = amount[i-1] * curr_price - fix
        cash[i] = cash[i-1] + buffer[i]; asset_value[i] = amount[i] * curr_price
        sumusd[i] = cash[i] + asset_value[i]
    return buffer, sumusd, cash, asset_value, amount, refer

def run_simulation(prices: List[float], actions: List[int], fix: int = 1500) -> pd.DataFrame:
    if not prices or not actions: return pd.DataFrame()
    buffer, sumusd, cash, asset_value, amount, refer = calculate_optimized_cached(tuple(actions), tuple(prices), fix)
    if len(sumusd) == 0: return pd.DataFrame()
    initial_capital = sumusd[0]
    df = pd.DataFrame({
        'price': prices, 'action': actions, 'buffer': np.round(buffer, 2),
        'sumusd': np.round(sumusd, 2), 'cash': np.round(cash, 2),
        'asset_value': np.round(asset_value, 2), 'amount': np.round(amount, 2),
        'refer': np.round(refer + initial_capital, 2),
        'net': np.round(sumusd - refer - initial_capital, 2)
    })
    return df

# ==============================================================================
# 3. Strategy Action Generation (ไม่มีการแก้ไข)
# ==============================================================================

def generate_actions_rebalance_daily(num_days: int) -> np.ndarray:
    return np.ones(num_days, dtype=int)

def generate_actions_perfect_foresight(prices: List[float], fix: int = 1500) -> np.ndarray:
    price_arr = np.asarray(prices, dtype=np.float64); n = len(price_arr)
    if n < 2: return np.ones(n, dtype=int)
    dp = np.zeros(n, dtype=np.float64); path = np.zeros(n, dtype=int); dp[0] = float(fix * 2)
    for i in range(1, n):
        j_indices = np.arange(i); profits = fix * ((price_arr[i] / price_arr[j_indices]) - 1)
        current_sumusd = dp[j_indices] + profits
        best_idx = np.argmax(current_sumusd); dp[i] = current_sumusd[best_idx]; path[i] = j_indices[best_idx]
    actions = np.zeros(n, dtype=int); current_day = np.argmax(dp)
    while current_day > 0:
        actions[current_day] = 1; current_day = path[current_day]
    actions[0] = 1
    return actions

def find_best_seed_for_window(prices_window: np.ndarray, num_seeds_to_try: int, max_workers: int) -> Tuple[int, float, np.ndarray]:
    window_len = len(prices_window)
    if window_len < 2: return 1, 0.0, np.ones(window_len, dtype=int)
    def evaluate_seed_batch(seed_batch: np.ndarray) -> List[Tuple[int, float]]:
        results = []
        for seed in seed_batch:
            rng = np.random.default_rng(seed)
            actions_window = rng.integers(0, 2, size=window_len)
            _, sumusd, _, _, _, refer = calculate_optimized_cached(tuple(actions_window), tuple(prices_window))
            net = sumusd - refer - sumusd[0]
            results.append((seed, net[-1]))
        return results
    best_seed_for_window = -1; max_net_for_window = -np.inf
    random_seeds = np.arange(num_seeds_to_try)
    batch_size = max(1, num_seeds_to_try // (max_workers * 4))
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
    else: best_seed_for_window = 1; best_actions = np.ones(window_len, dtype=int); max_net_for_window = 0.0
    best_actions[0] = 1
    return best_seed_for_window, max_net_for_window, best_actions

def generate_actions_sliding_window(ticker_data: pd.DataFrame, window_size: int, num_seeds_to_try: int, max_workers: int) -> Tuple[np.ndarray, pd.DataFrame]:
    prices = ticker_data['Close'].to_numpy(); n = len(prices)
    final_actions = np.array([], dtype=int); window_details_list = []
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="กำลังประมวลผล Sliding Windows...")
    st.write(f"📊 ข้อมูลทั้งหมด: {n} วัน | ขนาด Window: {window_size} วัน | จำนวน Windows: {num_windows}")
    st.write(f"⚡ ใช้ Parallel Processing: {max_workers} workers"); st.write("---")
    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n); prices_window = prices[start_index:end_index]; window_len = len(prices_window)
        if window_len == 0: continue
        best_seed, max_net, best_actions = find_best_seed_for_window(prices_window, num_seeds_to_try, max_workers)
        final_actions = np.concatenate((final_actions, best_actions))
        start_date_str = ticker_data.index[start_index].strftime('%Y-%m-%d'); end_date_str = ticker_data.index[end_index-1].strftime('%Y-%m-%d')
        detail = {'window_number': i + 1, 'timeline': f"{start_date_str} ถึง {end_date_str}", 'best_seed': best_seed, 'max_net': round(max_net, 2),
                  'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2), 'action_count': int(np.sum(best_actions)), 
                  'window_size': window_len, 'action_sequence': best_actions.tolist()}
        window_details_list.append(detail)
        progress_bar.progress((i + 1) / num_windows, text=f"ประมวลผล Window {i+1}/{num_windows}")
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)

# ==============================================================================
# 4. UI Rendering Functions
# ==============================================================================

def render_settings_tab(config: Dict[str, Any]):
    st.write("⚙️ **การตั้งค่าพารามิเตอร์**")
    asset_list = config.get('assets', ['FFWM']); default_index = asset_list.index(st.session_state.test_ticker) if st.session_state.test_ticker in asset_list else 0
    st.session_state.test_ticker = st.selectbox("เลือก Ticker สำหรับทดสอบ", options=asset_list, index=default_index)
    st.write("📅 **ช่วงวันที่สำหรับการวิเคราะห์**"); col1, col2 = st.columns(2)
    with col1: st.session_state.start_date = st.date_input("วันที่เริ่มต้น", value=st.session_state.start_date)
    with col2: st.session_state.end_date = st.date_input("วันที่สิ้นสุด", value=st.session_state.end_date)
    if st.session_state.start_date >= st.session_state.end_date: st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
    else: st.info(f"ช่วงวันที่ที่เลือก: {st.session_state.start_date:%Y-%m-%d} ถึง {st.session_state.end_date:%Y-%m-%d}")
    st.session_state.window_size = st.number_input("ขนาด Window (วัน)", min_value=2, value=st.session_state.window_size)
    st.session_state.num_seeds = st.number_input("จำนวน Seeds ต่อ Window", min_value=100, value=st.session_state.num_seeds, format="%d")
    st.session_state.max_workers = st.number_input("จำนวน Workers", min_value=1, max_value=16, value=st.session_state.max_workers)

def display_comparison_charts(results: Dict[str, pd.DataFrame], chart_title: str = '📊 เปรียบเทียบกำไรสุทธิ (Net Profit)'):
    if not results:
        st.warning("ไม่มีข้อมูลสำหรับสร้างกราฟเปรียบเทียบ"); return
    
    try:
        longest_index = max((df.index for df in results.values() if not df.empty), key=len, default=None)
    except ValueError:
        longest_index = None

    if longest_index is None:
        st.warning("ไม่มีข้อมูลสำหรับสร้างกราฟเปรียบเทียบ"); return
    
    chart_data_dict = {}
    for name, df in results.items():
        if not df.empty:
            chart_data_dict[name] = df['net'].reindex(longest_index)
            
    chart_data = pd.DataFrame(chart_data_dict)
    
    st.write(chart_title)
    st.line_chart(chart_data)
    # --- ส่วนของกราฟกระแสเงินสดถูกลบออกไปตามคำขอ ---

def render_test_tab():
    st.write("---")
    if st.button("🚀 เริ่มทดสอบ Best Seed (Optimized)", type="primary"):
        if st.session_state.start_date >= st.session_state.end_date: st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้องในแท็บ 'การตั้งค่า'"); return
        ticker = st.session_state.test_ticker
        start_date_str = st.session_state.start_date.strftime('%Y-%m-%d'); end_date_str = st.session_state.end_date.strftime('%Y-%m-%d')
        st.info(f"กำลังดึงข้อมูลสำหรับ **{ticker}** | {start_date_str} ถึง {end_date_str}")
        ticker_data = get_ticker_data(ticker, start_date_str, end_date_str)
        if ticker_data.empty: st.error("ไม่พบข้อมูลสำหรับ Ticker และช่วงวันที่ที่เลือก"); return
        prices = ticker_data['Close'].tolist(); num_days = len(prices)
        with st.spinner("กำลังคำนวณกลยุทธ์ต่างๆ..."):
            st.write("🔍 **เริ่มต้นการค้นหา Best Seed ด้วย Sliding Window**")
            actions_sliding, df_windows = generate_actions_sliding_window(ticker_data, st.session_state.window_size, st.session_state.num_seeds, st.session_state.max_workers)
            actions_min = generate_actions_rebalance_daily(num_days)
            actions_max = generate_actions_perfect_foresight(prices)
            results = {}
            for strategy_name, actions in {Strategy.SLIDING_WINDOW: actions_sliding.tolist(), Strategy.REBALANCE_DAILY: actions_min.tolist(), Strategy.PERFECT_FORESIGHT: actions_max.tolist()}.items():
                df = run_simulation(prices, actions)
                if not df.empty: df.index = ticker_data.index
                results[strategy_name] = df
        st.success("การทดสอบเสร็จสมบูรณ์!")
        st.write("---"); display_comparison_charts(results)
        st.write("📈 **สรุปผลการค้นหา Best Seed**")
        total_actions = df_windows['action_count'].sum(); total_net = df_windows['max_net'].sum()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Windows", df_windows.shape[0]); col2.metric("Total Actions", f"{total_actions}/{num_days}"); col3.metric("Total Net (Sum)", f"${total_net:,.2f}")
        st.dataframe(df_windows[['window_number', 'timeline', 'best_seed', 'max_net', 'price_change_pct', 'action_count']], use_container_width=True)
        csv = df_windows.to_csv(index=False)
        st.download_button(label="📥 ดาวน์โหลด Window Details (CSV)", data=csv, file_name=f'best_seed_{ticker}_{st.session_state.window_size}w.csv', mime='text/csv')

def render_analytics_tab():
    st.markdown("วิเคราะห์ผลลัพธ์ Backtest ในเชิงลึก")
    # ... โค้ดส่วนนี้ยาวและไม่มีการเปลี่ยนแปลง จึงขอละไว้เพื่อความกระชับ ...
    # (วางโค้ดเดิมของ render_analytics_tab ทั้งหมดที่นี่ได้เลย)
    
# ==============================================================================
# 4.5. <<< CORRECTED FUNCTION FOR TAB 4 (Final Version) >>>
# ==============================================================================

def render_manual_seed_tab(config: Dict[str, Any]):
    """
    แสดงผล UI และจัดการ Logic สำหรับ Tab 'Manual Seed'
    Final Version: ใช้หลักการ 'tail' และเปรียบเทียบแต่ละ line เป็นเส้นกราฟแยกกัน
    """
    st.header("🌱 Manual Seed Strategy Comparator")
    st.markdown("สร้างและเปรียบเทียบ Action Sequences โดยการตัดส่วนท้าย (`tail`) จาก Seed ที่กำหนด")

    with st.container(border=True):
        st.subheader("1. กำหนดค่า Input สำหรับการทดสอบ")
        
        # --- Input สำหรับ Ticker และ Date Range ---
        col1, col2 = st.columns([1, 2])
        with col1:
            asset_list = config.get('assets', ['FFWM'])
            default_index = asset_list.index(st.session_state.test_ticker) if st.session_state.test_ticker in asset_list else 0
            manual_ticker = st.selectbox("เลือก Ticker", options=asset_list, index=default_index, key="manual_ticker_compare_tail")

        with col2:
            c1, c2 = st.columns(2)
            manual_start_date = c1.date_input("วันที่เริ่มต้น (Start Date)", value=datetime(2025, 4, 28), key="manual_start_compare_tail")
            manual_end_date = c2.date_input("วันที่สิ้นสุด (End Date)", value=datetime(2025, 6, 9), key="manual_end_compare_tail")

        if manual_start_date >= manual_end_date:
            st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
        
        st.divider()

        # --- Input สำหรับ Seed Lines (Dynamic) ---
        st.write("**กำหนดกลยุทธ์ (Seed/Size/Tail) ที่ต้องการเปรียบเทียบ:**")

        for i, line in enumerate(st.session_state.manual_seed_lines):
            cols = st.columns([1, 2, 2, 2])
            cols[0].write(f"**Line {i+1}**")
            line['seed'] = cols[1].number_input("Input Seed", value=line.get('seed', 1), min_value=1, key=f"seed_compare_tail_{i}")
            line['size'] = cols[2].number_input("Size (ขนาด Sequence เริ่มต้น)", value=line.get('size', 60), min_value=1, key=f"size_compare_tail_{i}")
            line['tail'] = cols[3].number_input("Tail (ส่วนท้ายที่จะใช้)", value=line.get('tail', 10), min_value=1, max_value=line.get('size', 60), key=f"tail_compare_tail_{i}")
        
        b_col1, b_col2, _ = st.columns([1,1,4])
        if b_col1.button("➕ เพิ่ม Line เปรียบเทียบ"):
            st.session_state.manual_seed_lines.append({'seed': np.random.randint(1, 10000), 'size': 50, 'tail': 20})
            st.rerun()

        if b_col2.button("➖ ลบ Line สุดท้าย"):
            if len(st.session_state.manual_seed_lines) > 1: st.session_state.manual_seed_lines.pop(); st.rerun()
            else: st.warning("ต้องมีอย่างน้อย 1 line")
    
    st.write("---")

    if st.button("📈 เปรียบเทียบประสิทธิภาพ Seeds", type="primary"):
        if manual_start_date >= manual_end_date: st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้อง"); return
        
        with st.spinner("กำลังดึงข้อมูลและจำลองการเทรด..."):
            start_str = manual_start_date.strftime('%Y-%m-%d'); end_str = manual_end_date.strftime('%Y-%m-%d')
            ticker_data = get_ticker_data(manual_ticker, start_str, end_str)

            if ticker_data.empty: st.error(f"ไม่พบข้อมูลสำหรับ {manual_ticker} ในช่วงวันที่ที่เลือก"); return
            
            prices = ticker_data['Close'].tolist(); num_trading_days = len(prices)
            st.info(f"📊 พบข้อมูลราคา {num_trading_days} วันทำการในช่วงที่เลือก")

            results = {}
            max_sim_len = 0 

            for i, line_info in enumerate(st.session_state.manual_seed_lines):
                input_seed = line_info['seed']; size_seed = line_info['size']; tail_seed = line_info['tail']
                
                if tail_seed > size_seed:
                    st.error(f"Line {i+1}: Tail ({tail_seed}) ต้องไม่มากกว่า Size ({size_seed})"); return

                # สร้าง Action Sequence เต็ม -> แล้วตัดเอาเฉพาะส่วนท้าย
                rng_best = np.random.default_rng(input_seed)
                full_actions = rng_best.integers(0, 2, size=size_seed)
                actions_from_tail = full_actions[-tail_seed:].tolist()
                
                sim_len = min(num_trading_days, len(actions_from_tail))
                if sim_len == 0: continue

                prices_to_sim = prices[:sim_len]; actions_to_sim = actions_from_tail[:sim_len]
                
                df_line = run_simulation(prices_to_sim, actions_to_sim)
                if not df_line.empty:
                    df_line.index = ticker_data.index[:sim_len]
                    strategy_name = f"Seed {input_seed} (Tail {tail_seed})"
                    results[strategy_name] = df_line
                    max_sim_len = max(max_sim_len, sim_len)

            if not results: st.error("ไม่สามารถสร้างผลลัพธ์จาก Seed ที่กำหนดได้"); return
                
            if max_sim_len > 0:
                prices_for_benchmark = prices[:max_sim_len]
                df_max = run_simulation(prices_for_benchmark, generate_actions_perfect_foresight(prices_for_benchmark).tolist())
                df_min = run_simulation(prices_for_benchmark, generate_actions_rebalance_daily(max_sim_len).tolist())
                if not df_max.empty:
                    df_max.index = ticker_data.index[:max_sim_len]; results[Strategy.PERFECT_FORESIGHT] = df_max
                if not df_min.empty:
                    df_min.index = ticker_data.index[:max_sim_len]; results[Strategy.REBALANCE_DAILY] = df_min

            st.success("การเปรียบเทียบเสร็จสมบูรณ์!")
            display_comparison_charts(results, chart_title="📊 Performance Comparison (Net Profit)")

            st.subheader("สรุปผลลัพธ์สุดท้าย (Final Net Profit)")
            sorted_names = [name for name in results.keys() if name not in [Strategy.PERFECT_FORESIGHT, Strategy.REBALANCE_DAILY]]
            display_order = [Strategy.PERFECT_FORESIGHT] + sorted(sorted_names) + [Strategy.REBALANCE_DAILY]
            
            num_cols = len(results)
            if num_cols > 0:
                final_metrics_cols = st.columns(num_cols)
                col_idx = 0
                for name in display_order:
                    if name in results and not results[name].empty:
                        df_result = results[name]; final_net = df_result['net'].iloc[-1]
                        final_metrics_cols[col_idx].metric(name, f"${final_net:,.2f}")
                        col_idx += 1

# ==============================================================================
# 5. Main Application
# ==============================================================================
def main():
    st.set_page_config(page_title="Best Seed Sliding Window", page_icon="🎯", layout="wide")
    st.markdown("🎯 Best Seed Sliding Window Tester (Optimized)")
    st.caption("เครื่องมือทดสอบการหา Best Seed ด้วยวิธี Sliding Window (Refactored Version)")

    config = load_config()
    initialize_session_state(config)

    tab1, tab2, tab3, tab4 = st.tabs(["⚙️ การตั้งค่า", "🚀 ทดสอบ", "📊 Advanced Analytics", "🌱 Manual Seed"])

    with tab1: render_settings_tab(config)
    with tab2: render_test_tab()
    with tab3: render_analytics_tab() # ใส่โค้ดส่วนนี้กลับเข้าไปถ้าต้องการ
    with tab4: render_manual_seed_tab(config)

if __name__ == "__main__":
    main()
