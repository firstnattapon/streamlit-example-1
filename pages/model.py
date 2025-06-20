# ก่อนรัน ต้องติดตั้ง: pip install noise
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
from numba import njit
import noise #! Perlin Noise: Import library

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================

class Strategy:
    REBALANCE_DAILY = "Rebalance Daily"
    PERFECT_FORESIGHT = "Perfect Foresight (Max)"
    SLIDING_WINDOW = "Best Seed Sliding Window"
    CHAOTIC_SLIDING_WINDOW = "Chaotic Seed Sliding Window"
    # เพิ่มกลยุทธ์ใหม่ที่เป็น Generator
    PERLIN_NOISE_GENERATOR = "Perlin Noise Generator"
    MANUAL_SEED = "Manual Seed Strategy" # ย้ายไปไว้ท้ายสุดเพื่อความสอดคล้อง

def load_config(filepath: str = "dynamic_seed_config.json") -> Dict[str, Any]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        st.warning(f"⚠️ ไม่พบหรือไฟล์ '{filepath}' ไม่ถูกต้อง จะใช้ค่าเริ่มต้นแทน")
        return {
            "assets": ["FFWM", "NEGG", "RIVN", "BTC-USD"],
            "default_settings": {
                "selected_ticker": "FFWM", "start_date": "2024-01-01", "window_size": 30, 
                "num_seeds": 10000, "max_workers": 8,
                # เพิ่มค่าเริ่มต้นสำหรับ Perlin Noise
                "perlin_num_params": 5000 
            },
            "manual_seed_by_asset": {
                "default": [{'seed': 999, 'size': 50, 'tail': 15}],
                "FFWM": [{'seed': 1234, 'size': 60, 'tail': 30}, {'seed': 7777, 'size': 30, 'tail': 10}]
            }
        }

def initialize_session_state(config: Dict[str, Any]):
    defaults = config.get('default_settings', {})
    if 'test_ticker' not in st.session_state: st.session_state.test_ticker = defaults.get('selected_ticker', 'FFWM')
    if 'start_date' not in st.session_state:
        try: st.session_state.start_date = datetime.strptime(defaults.get('start_date', '2024-01-01'), '%Y-%m-%d').date()
        except ValueError: st.session_state.start_date = datetime(2024, 1, 1).date()
    if 'end_date' not in st.session_state: st.session_state.end_date = datetime.now().date()
    if 'window_size' not in st.session_state: st.session_state.window_size = defaults.get('window_size', 30)
    if 'num_seeds' not in st.session_state: st.session_state.num_seeds = defaults.get('num_seeds', 10000)
    if 'max_workers' not in st.session_state: st.session_state.max_workers = defaults.get('max_workers', 8)
    # เพิ่ม state สำหรับ Perlin Noise
    if 'perlin_num_params' not in st.session_state: st.session_state.perlin_num_params = defaults.get('perlin_num_params', 5000)
    if 'df_for_analysis' not in st.session_state: st.session_state.df_for_analysis = None
    if 'manual_action_sequence' not in st.session_state: st.session_state.manual_action_sequence = "[1, 0, 1, 0, 1, 1, 0, 0, 1, 0]"

# ==============================================================================
# 2. Core Calculation & Data Functions (No Changes)
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

@njit(cache=True)
def _calculate_simulation_numba(action_array: np.ndarray, price_array: np.ndarray, fix: int = 1500) -> Tuple:
    n = len(action_array)
    if n == 0 or len(price_array) == 0:
        empty_arr = np.empty(0, dtype=np.float64); return (empty_arr, empty_arr, empty_arr, empty_arr, empty_arr, empty_arr)
    action_array_calc = action_array.copy()
    if n > 0: action_array_calc[0] = 1
    amount = np.empty(n, dtype=np.float64); buffer = np.zeros(n, dtype=np.float64)
    cash = np.empty(n, dtype=np.float64); asset_value = np.empty(n, dtype=np.float64)
    sumusd = np.empty(n, dtype=np.float64); initial_price = price_array[0]
    amount[0] = fix / initial_price; cash[0] = fix
    asset_value[0] = amount[0] * initial_price; sumusd[0] = cash[0] + asset_value[0]
    refer = -fix * np.log(initial_price / price_array)
    for i in range(1, n):
        curr_price = price_array[i]
        if action_array_calc[i] == 0: amount[i] = amount[i-1]; buffer[i] = 0.0
        else: amount[i] = fix / curr_price; buffer[i] = amount[i-1] * curr_price - fix
        cash[i] = cash[i-1] + buffer[i]
        asset_value[i] = amount[i] * curr_price; sumusd[i] = cash[i] + asset_value[i]
    return buffer, sumusd, cash, asset_value, amount, refer

@lru_cache(maxsize=4096)
def calculate_optimized_cached(action_tuple: Tuple[int, ...], price_tuple: Tuple[float, ...], fix: int = 1500) -> Tuple:
    action_array = np.asarray(action_tuple, dtype=np.int32)
    price_array = np.asarray(price_tuple, dtype=np.float64)
    return _calculate_simulation_numba(action_array, price_array, fix)

def run_simulation(prices: List[float], actions: List[int], fix: int = 1500) -> pd.DataFrame:
    if not prices or not actions: return pd.DataFrame()
    min_len = min(len(prices), len(actions)); prices = prices[:min_len]; actions = actions[:min_len]
    action_array = np.asarray(actions, dtype=np.int32); price_array = np.asarray(prices, dtype=np.float64)
    buffer, sumusd, cash, asset_value, amount, refer = _calculate_simulation_numba(action_array, price_array, fix)
    if len(sumusd) == 0: return pd.DataFrame()
    initial_capital = sumusd[0]
    return pd.DataFrame({'price': prices, 'action': actions, 'buffer': np.round(buffer, 2),'sumusd': np.round(sumusd, 2), 'cash': np.round(cash, 2), 'asset_value': np.round(asset_value, 2), 'amount': np.round(amount, 2), 'refer': np.round(refer + initial_capital, 2), 'net': np.round(sumusd - refer - initial_capital, 2)})

# ==============================================================================
# 3. Strategy Action Generation
# ==============================================================================
# --- OMITTED FOR BREVITY: Benchmarks, Random Seed, Chaotic Seed are here ---

# 3.4 [NEW] Perlin Noise Generator
def generate_actions_from_perlin_noise(length: int, scale: float, octaves: int, persistence: float, lacunarity: float, seed: int) -> np.ndarray:
    """สร้าง Action Sequence จาก Perlin Noise Generator"""
    if length == 0: return np.array([], dtype=np.int32)
    
    actions = np.zeros(length, dtype=np.int32)
    for i in range(length):
        # สร้างค่า noise สำหรับแต่ละตำแหน่ง
        # เราใช้ seed เป็น offset (base) เพื่อให้ผลลัพธ์ทำซ้ำได้
        noise_val = noise.pnoise1(i / scale,
                                  octaves=octaves,
                                  persistence=persistence,
                                  lacunarity=lacunarity,
                                  repeat=length, # ไม่จำเป็นแต่ช่วยให้ขอบดูดีขึ้น
                                  base=seed)
        actions[i] = 1 if noise_val > 0 else 0
    
    if length > 0: actions[0] = 1
    return actions

def find_best_perlin_params(prices_window: np.ndarray, num_params_to_try: int, max_workers: int) -> Tuple[dict, float, np.ndarray]:
    """ค้นหาพารามิเตอร์ Perlin Noise ที่ดีที่สุดสำหรับ Window"""
    window_len = len(prices_window)
    if window_len < 2: return {'seed': 1, 'scale': 10.0, 'octaves': 1}, 0.0, np.ones(window_len, dtype=np.int32)

    def evaluate_perlin_batch(param_batch: List[Dict]) -> List[Tuple[dict, float]]:
        results = []
        for params in param_batch:
            actions_window = generate_actions_from_perlin_noise(window_len, **params)
            _, sumusd, _, _, _, refer = calculate_optimized_cached(tuple(actions_window), tuple(prices_window))
            net = sumusd[-1] - refer[-1] - sumusd[0] if len(sumusd) > 0 else -np.inf
            results.append((params, net))
        return results

    # สร้างชุดพารามิเตอร์แบบสุ่ม
    rng = np.random.default_rng()
    param_list = []
    for _ in range(num_params_to_try):
        params = {
            'seed': rng.integers(0, 1_000_000),
            'scale': rng.uniform(5.0, 50.0), # scale น้อย = หยักมาก, scale มาก = หยักน้อย
            'octaves': rng.integers(1, 6),
            'persistence': rng.uniform(0.3, 0.7),
            'lacunarity': rng.uniform(1.8, 2.2)
        }
        param_list.append(params)

    best_params = None; max_net = -np.inf
    batch_size = max(1, num_params_to_try // (max_workers * 4 if max_workers > 0 else 1))
    param_batches = [param_list[j:j + batch_size] for j in range(0, len(param_list), batch_size)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_perlin_batch, batch) for batch in param_batches]
        for future in as_completed(futures):
            for params, final_net in future.result():
                if final_net > max_net:
                    max_net = final_net
                    best_params = params

    if best_params:
        best_actions = generate_actions_from_perlin_noise(window_len, **best_params)
    else:
        best_params = {'seed': 1, 'scale': 10.0, 'octaves': 1, 'persistence': 0.5, 'lacunarity': 2.0}
        best_actions = np.ones(window_len, dtype=np.int32)
        max_net = 0.0
        
    return best_params, max_net, best_actions

def generate_actions_sliding_window_perlin(ticker_data: pd.DataFrame, window_size: int, num_params: int, max_workers: int) -> Tuple[np.ndarray, pd.DataFrame]:
    prices = ticker_data['Close'].to_numpy(); n = len(prices)
    final_actions = np.array([], dtype=np.int32); window_details_list = []
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="กำลังประมวลผล Perlin Noise Generator...")
    st.write(f"🌊 Generating coherent sequences with Perlin Noise")
    st.write(f"📊 ข้อมูล: {n} วัน | Window: {window_size} วัน | จำนวนพารามิเตอร์ที่ทดสอบ: {num_params}")
    st.write("---")
    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n); prices_window = prices[start_index:end_index]
        window_len = len(prices_window)
        if window_len < 2: continue
        
        best_params, max_net, best_actions = find_best_perlin_params(prices_window, num_params, max_workers)
        
        final_actions = np.concatenate((final_actions, best_actions))
        start_date_str = ticker_data.index[start_index].strftime('%Y-%m-%d'); end_date_str = ticker_data.index[end_index-1].strftime('%Y-%m-%d')
        detail = {
            'window_number': i + 1, 'timeline': f"{start_date_str} ถึง {end_date_str}", 
            'max_net': round(max_net, 2),
            'seed': best_params['seed'],
            'scale': round(best_params['scale'], 2),
            'octaves': best_params['octaves'],
            'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2),
            'action_count': int(np.sum(best_actions)), 'window_size': window_len,
            'action_sequence': best_actions.tolist()
        }
        window_details_list.append(detail)
        progress_bar.progress((i + 1) / num_windows, text=f"Generating for Window {i+1}/{num_windows}")
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
    st.divider()
    st.subheader("พารามิเตอร์ทั่วไป")
    st.session_state.window_size = st.number_input("ขนาด Window (วัน)", min_value=2, value=st.session_state.window_size)
    st.subheader("พารามิเตอร์สำหรับ Random/Chaotic Seed")
    c1, c2 = st.columns(2)
    st.session_state.num_seeds = c1.number_input("จำนวน Seeds/Params ต่อ Window", min_value=100, value=st.session_state.num_seeds, format="%d", help="สำหรับ Random และ Chaotic Seed")
    st.session_state.max_workers = c2.number_input("จำนวน Workers", min_value=1, max_value=16, value=st.session_state.max_workers)
    st.subheader("พารามิเตอร์สำหรับ Perlin Noise Generator")
    st.session_state.perlin_num_params = st.number_input("จำนวนชุดพารามิเตอร์ที่ทดสอบ", min_value=100, value=st.session_state.perlin_num_params, format="%d", help="จำนวนการสุ่มพารามิเตอร์ (seed, scale, etc.) เพื่อหาชุดที่ดีที่สุด")


def display_comparison_charts(results: Dict[str, pd.DataFrame], chart_title: str = '📊 เปรียบเทียบกำไรสุทธิ (Net Profit)'):
    if not results: st.warning("ไม่มีข้อมูลสำหรับสร้างกราฟเปรียบเทียบ"); return
    valid_dfs = {name: df for name, df in results.items() if not df.empty and 'net' in df.columns}
    if not valid_dfs: st.warning("ไม่มีข้อมูล 'net' สำหรับสร้างกราฟ"); return
    try: longest_index = max((df.index for df in valid_dfs.values()), key=len, default=None)
    except ValueError: longest_index = None
    if longest_index is None: st.warning("ไม่มีข้อมูลสำหรับสร้างกราฟเปรียบเทียบ"); return
    chart_data = pd.DataFrame(index=longest_index)
    for name, df in valid_dfs.items():
        chart_data[name] = df['net'].reindex(longest_index).ffill()
    st.write(chart_title); st.line_chart(chart_data)

def render_perlin_noise_tab():
    st.write("---")
    st.markdown("### 🌊 ทดสอบด้วย Perlin Noise Generator")
    st.info("กลยุทธ์นี้ใช้ Perlin Noise เพื่อสร้าง Action Sequence ที่มีความต่อเนื่องและเป็นธรรมชาติ จากนั้นจะค้นหาพารามิเตอร์ (seed, scale, etc.) ที่ให้ผลลัพธ์ดีที่สุดในแต่ละ Window")
    if st.button("🚀 เริ่มทดสอบ Best Perlin Noise", type="primary", key="perlin_test_button"):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้องในแท็บ 'การตั้งค่า'"); return
        ticker = st.session_state.test_ticker; start_date_str = st.session_state.start_date.strftime('%Y-%m-%d'); end_date_str = st.session_state.end_date.strftime('%Y-%m-%d')
        st.info(f"กำลังดึงข้อมูลสำหรับ **{ticker}** | {start_date_str} ถึง {end_date_str}")
        ticker_data = get_ticker_data(ticker, start_date_str, end_date_str)
        if ticker_data.empty: st.error("ไม่พบข้อมูลสำหรับ Ticker และช่วงวันที่ที่เลือก"); return
        prices = ticker_data['Close'].to_numpy(); num_days = len(prices)
        with st.spinner("กำลังคำนวณกลยุทธ์ต่างๆ (Perlin Noise Search)..."):
            actions_perlin, df_windows = generate_actions_sliding_window_perlin(
                ticker_data, st.session_state.window_size, 
                st.session_state.perlin_num_params, st.session_state.max_workers
            )
            actions_min = generate_actions_rebalance_daily(num_days)
            actions_max = generate_actions_perfect_foresight(prices)
            results = {}; strategy_map = {Strategy.PERLIN_NOISE_GENERATOR: actions_perlin.tolist(), Strategy.REBALANCE_DAILY: actions_min.tolist(), Strategy.PERFECT_FORESIGHT: actions_max.tolist()}
            for strategy_name, actions in strategy_map.items():
                df = run_simulation(prices.tolist(), actions)
                if not df.empty: df.index = ticker_data.index[:len(df)]
                results[strategy_name] = df
        st.success("การทดสอบเสร็จสมบูรณ์!"); st.write("---"); display_comparison_charts(results)
        st.write("📈 **สรุปผลการค้นหา Best Perlin Noise Parameters**")
        if not df_windows.empty:
            total_net = df_windows['max_net'].sum(); total_actions = df_windows['action_count'].sum()
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Net (Sum)", f"${total_net:,.2f}"); col2.metric("Total Actions", f"{total_actions}/{num_days}"); col3.metric("Total Windows", df_windows.shape[0])
        else:
            col1, col2, col3 = st.columns(3); col1.metric("Total Net (Sum)", "$0.00"); col2.metric("Total Actions", "0/0"); col3.metric("Total Windows", "0")
        st.dataframe(df_windows[['window_number', 'timeline', 'seed', 'scale', 'octaves', 'max_net', 'price_change_pct', 'action_count']], use_container_width=True)
        csv = df_windows.to_csv(index=False); st.download_button(label="📥 ดาวน์โหลด Perlin Noise Details (CSV)", data=csv, file_name=f'best_perlin_{ticker}_{st.session_state.window_size}w.csv', mime='text/csv')

# --- Other render functions are omitted for brevity, but should be included in the final file ---
def render_test_tab(): st.info("This is the Random Seed Tab.")
def render_chaotic_test_tab(): st.info("This is the Chaotic Seed Tab.")
def render_analytics_tab(): st.info("This is the Analytics Tab.")
def render_manual_seed_tab(config): st.info("This is the Manual/Forward Test Tab.")

# ==============================================================================
# 5. Main Application
# ==============================================================================
def main():
    st.set_page_config(page_title="Strategy Optimization Lab", page_icon="🎯", layout="wide")
    st.markdown("🎯 Best Seed / Parameter Finder")
    st.caption("เครื่องมือทดสอบการหาค่าที่ดีที่สุดด้วยวิธี Sliding Window")

    config = load_config()
    initialize_session_state(config)

    tab_list = ["⚙️ การตั้งค่า", "🎲 Random Seed", "🌀 Chaotic Seed", "🌊 Perlin Noise", "📊 Analytics", "🌱 Manual / Forward Test"]
    tabs = st.tabs(tab_list)

    with tabs[0]: render_settings_tab(config)
    with tabs[1]: render_test_tab()
    with tabs[2]: render_chaotic_test_tab()
    with tabs[3]: render_perlin_noise_tab()
    with tabs[4]: render_analytics_tab()
    with tabs[5]: render_manual_seed_tab(config)

    with st.expander("📖 คำอธิบายกลยุทธ์"):
        st.markdown("""
        **หลักการพื้นฐาน:** กลยุทธ์ทั้งหมดทำงานบนหลักการ "Sliding Window"
        
        **กลยุทธ์แบบ "เครื่องปั่นตัวเลข" (Generators):**
        - **🎲 Random Seed:** สร้างลำดับแบบสุ่มสมบูรณ์ (แต่ละวันไม่เกี่ยวข้องกัน)
        - **🌀 Chaotic Seed:** ใช้ Logistic Map สร้างลำดับที่ดูเหมือนสุ่มแต่มีรูปแบบซับซ้อน
        - **🌊 Perlin Noise:** ใช้ Perlin Noise สร้างลำดับที่มีความต่อเนื่องและเป็นธรรมชาติ (ค่าที่อยู่ใกล้กันมีแนวโน้มคล้ายกัน) ทำให้เกิด "ช่วง" ของการซื้อ/ไม่ซื้อ
        """)

if __name__ == "__main__":
    main()
