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

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================

class Strategy:
    REBALANCE_DAILY = "Rebalance Daily (Min)"
    PERFECT_FORESIGHT = "Perfect Foresight (Max)"
    MODEL_REFINEMENT_SLIDING_WINDOW = "Model Refinement (Sliding Window)" # CHANGED
    MANUAL_MODEL = "Manual Model Strategy" # CHANGED

def load_config(filepath: str = "model_refinement_config.json") -> Dict[str, Any]: # CHANGED Filename
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
            "assets": ["FFWM", "NEGG", "RIVN"],
            "default_settings": {"selected_ticker": "FFWM", "start_date": "2025-06-10", "window_size": 30, "num_models_to_test": 30000, "max_workers": 8},
            "manual_model_by_asset": {
                "default": [{'model_id': 999, 'size': 50, 'tail': 15}],
                "FFWM": [{'model_id': 1234, 'size': 60, 'tail': 30}, {'model_id': 7777, 'size': 30, 'tail': 10}]
            }
        }

def on_ticker_change_callback(config: Dict[str, Any]):
    """
    Callback ที่จะถูกเรียกเมื่อ Ticker ใน Tab Manual Model เปลี่ยน
    ทำการอัปเดต st.session_state.manual_model_lines ให้ตรงกับ Ticker ที่เลือก
    """
    selected_ticker = st.session_state.get("manual_ticker_key")
    if not selected_ticker:
        return

    presets_by_asset = config.get("manual_model_by_asset", {}) # CHANGED
    default_presets = presets_by_asset.get("default", [{'model_id': 999, 'size': 50, 'tail': 15}]) # CHANGED
    
    new_presets = presets_by_asset.get(selected_ticker, default_presets)
    st.session_state.manual_model_lines = new_presets # CHANGED

def initialize_session_state(config: Dict[str, Any]):
    """
    ตั้งค่าเริ่มต้นสำหรับ Streamlit session state โดยใช้ค่าจาก config
    """
    defaults = config.get('default_settings', {})
    
    if 'test_ticker' not in st.session_state:
        st.session_state.test_ticker = defaults.get('selected_ticker', 'FFWM')
    if 'start_date' not in st.session_state:
        try:
            st.session_state.start_date = datetime.strptime(defaults.get('start_date', '2025-01-01'), '%Y-%m-%d').date()
        except ValueError:
            st.session_state.start_date = datetime(2025, 1, 1).date()
    if 'end_date' not in st.session_state:
        st.session_state.end_date = datetime.now().date()
    if 'window_size' not in st.session_state:
        st.session_state.window_size = defaults.get('window_size', 30)
    # CHANGED variable name
    if 'num_models_to_test' not in st.session_state:
        st.session_state.num_models_to_test = defaults.get('num_models_to_test', 30000)
    if 'max_workers' not in st.session_state:
        st.session_state.max_workers = defaults.get('max_workers', 8)
    
    if 'df_for_analysis' not in st.session_state:
        st.session_state.df_for_analysis = None
    
    # CHANGED to manual_model_lines
    if 'manual_model_lines' not in st.session_state:
        initial_ticker = defaults.get('selected_ticker', 'FFWM')
        presets_by_asset = config.get("manual_model_by_asset", {})
        default_presets = presets_by_asset.get("default", [{'model_id': 999, 'size': 50, 'tail': 15}])
        st.session_state.manual_model_lines = presets_by_asset.get(initial_ticker, default_presets)

# ==============================================================================
# 2. Core Calculation & Data Functions
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
# 3. Strategy Action Generation
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

def find_best_model_for_window(prices_window: np.ndarray, num_models_to_test: int, max_workers: int) -> Tuple[int, float, np.ndarray]: # CHANGED
    window_len = len(prices_window)
    if window_len < 2: return 1, 0.0, np.ones(window_len, dtype=int)
    
    def evaluate_model_batch(model_id_batch: np.ndarray) -> List[Tuple[int, float]]: # CHANGED
        results = []
        for model_id in model_id_batch: # A seed is now a model_id
            rng = np.random.default_rng(model_id)
            actions_window = rng.integers(0, 2, size=window_len)
            _, sumusd, _, _, _, refer = calculate_optimized_cached(tuple(actions_window), tuple(prices_window))
            net = sumusd[-1] - refer[-1] - sumusd[0]
            results.append((model_id, net))
        return results
        
    best_model_id_for_window = -1; max_net_for_window = -np.inf # CHANGED
    model_ids = np.arange(num_models_to_test) # CHANGED
    batch_size = max(1, num_models_to_test // (max_workers * 4))
    model_id_batches = [model_ids[j:j+batch_size] for j in range(0, len(model_ids), batch_size)] # CHANGED
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_model_batch, batch) for batch in model_id_batches]
        for future in as_completed(futures):
            for model_id, final_net in future.result():
                if final_net > max_net_for_window:
                    max_net_for_window = final_net; best_model_id_for_window = model_id # CHANGED
                    
    if best_model_id_for_window >= 0:
        rng_best = np.random.default_rng(best_model_id_for_window)
        best_actions = rng_best.integers(0, 2, size=window_len)
    else: 
        best_model_id_for_window = 1; best_actions = np.ones(window_len, dtype=int); max_net_for_window = 0.0
        
    best_actions[0] = 1
    return best_model_id_for_window, max_net_for_window, best_actions

def generate_actions_sliding_window(ticker_data: pd.DataFrame, window_size: int, num_models_to_test: int, max_workers: int) -> Tuple[np.ndarray, pd.DataFrame]: # CHANGED
    prices = ticker_data['Close'].to_numpy(); n = len(prices)
    final_actions = np.array([], dtype=int); window_details_list = []
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text="กำลังประมวลผล Sliding Windows (Model Refinement)...") # CHANGED
    st.write(f"📊 ข้อมูลทั้งหมด: {n} วัน | ขนาด Window: {window_size} วัน | จำนวน Windows: {num_windows}")
    st.write(f"⚡ ใช้ Parallel Processing: {max_workers} workers"); st.write("---")
    
    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n); prices_window = prices[start_index:end_index]; window_len = len(prices_window)
        if window_len == 0: continue
        
        best_model_id, max_net, best_actions = find_best_model_for_window(prices_window, num_models_to_test, max_workers) # CHANGED
        
        final_actions = np.concatenate((final_actions, best_actions))
        start_date_str = ticker_data.index[start_index].strftime('%Y-%m-%d'); end_date_str = ticker_data.index[end_index-1].strftime('%Y-%m-%d')
        
        detail = {'window_number': i + 1, 'timeline': f"{start_date_str} ถึง {end_date_str}", 'best_model_id': best_model_id, 'max_net': round(max_net, 2), # CHANGED
                  'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2), 'action_count': int(np.sum(best_actions)),
                  'window_size': window_len, 'action_sequence': best_actions.tolist()}
        window_details_list.append(detail)
        
        progress_bar.progress((i + 1) / num_windows, text=f"กำลังปรับแต่ง Model Window {i+1}/{num_windows}") # CHANGED
        
    progress_bar.empty()
    return final_actions, pd.DataFrame(window_details_list)

# ==============================================================================
# 4. UI Rendering Functions
# ==============================================================================
def render_settings_tab(config: Dict[str, Any]):
    st.write("⚙️ **การตั้งค่าพารามิเตอร์**")

    asset_list = config.get('assets', ['FFWM'])
    try:
        default_index = asset_list.index(st.session_state.test_ticker)
    except ValueError:
        default_index = 0

    st.session_state.test_ticker = st.selectbox(
        "เลือก Ticker สำหรับทดสอบ",
        options=asset_list,
        index=default_index
    )

    st.write("📅 **ช่วงวันที่สำหรับการวิเคราะห์**"); col1, col2 = st.columns(2)
    with col1:
        st.session_state.start_date = st.date_input("วันที่เริ่มต้น", value=st.session_state.start_date)
    with col2:
        st.session_state.end_date = st.date_input("วันที่สิ้นสุด", value=st.session_state.end_date)

    if st.session_state.start_date >= st.session_state.end_date:
        st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
    else:
        st.info(f"ช่วงวันที่ที่เลือก: {st.session_state.start_date:%Y-%m-%d} ถึง {st.session_state.end_date:%Y-%m-%d}")

    st.session_state.window_size = st.number_input("ขนาด Window (วัน)", min_value=2, value=st.session_state.window_size)
    st.session_state.num_models_to_test = st.number_input("จำนวน Models ที่จะทดสอบต่อ Window", min_value=100, value=st.session_state.num_models_to_test, format="%d") # CHANGED
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

def render_test_tab():
    st.write("---")
    if st.button("🚀 เริ่มการปรับแต่งโมเดล (Model Refinement)", type="primary"): # CHANGED
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้องในแท็บ 'การตั้งค่า'"); return

        ticker = st.session_state.test_ticker
        start_date_str = st.session_state.start_date.strftime('%Y-%m-%d')
        end_date_str = st.session_state.end_date.strftime('%Y-%m-%d')

        st.info(f"กำลังดึงข้อมูลสำหรับ **{ticker}** | {start_date_str} ถึง {end_date_str}")
        ticker_data = get_ticker_data(ticker, start_date_str, end_date_str)

        if ticker_data.empty:
            st.error("ไม่พบข้อมูลสำหรับ Ticker และช่วงวันที่ที่เลือก"); return

        prices = ticker_data['Close'].tolist()
        num_days = len(prices)

        with st.spinner("กำลังคำนวณและปรับแต่งโมเดล..."): # CHANGED
            st.write("🔍 **เริ่มต้นการปรับแต่งโมเดลด้วย Sliding Window**") # CHANGED
            actions_sliding, df_windows = generate_actions_sliding_window(
                ticker_data, st.session_state.window_size,
                st.session_state.num_models_to_test, st.session_state.max_workers # CHANGED
            )
            actions_min = generate_actions_rebalance_daily(num_days)
            actions_max = generate_actions_perfect_foresight(prices)

            results = {}
            strategy_map = {
                Strategy.MODEL_REFINEMENT_SLIDING_WINDOW: actions_sliding.tolist(), # CHANGED
                Strategy.REBALANCE_DAILY: actions_min.tolist(),
                Strategy.PERFECT_FORESIGHT: actions_max.tolist()
            }

            for strategy_name, actions in strategy_map.items():
                df = run_simulation(prices, actions)
                if not df.empty:
                    df.index = ticker_data.index[:len(df)]
                results[strategy_name] = df

        st.success("การปรับแต่งโมเดลเสร็จสมบูรณ์!") # CHANGED
        st.write("---");
        display_comparison_charts(results)

        st.write("📈 **สรุปผลการปรับแต่งโมเดล**") # CHANGED
        total_actions = df_windows['action_count'].sum()
        total_net = df_windows['max_net'].sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Windows", df_windows.shape[0])
        col2.metric("Total Actions", f"{total_actions}/{num_days}")
        col3.metric("Total Net (Sum)", f"${total_net:,.2f}")

        st.dataframe(df_windows[['window_number', 'timeline', 'best_model_id', 'max_net', 'price_change_pct', 'action_count']], use_container_width=True) # CHANGED
        csv = df_windows.to_csv(index=False)
        st.download_button(
            label="📥 ดาวน์โหลด Window Details (CSV)",
            data=csv,
            file_name=f'model_refinement_{ticker}_{st.session_state.window_size}w.csv', # CHANGED
            mime='text/csv'
        )

def render_analytics_tab():
    st.header("📊 Advanced Analytics Dashboard")

    with st.container():
        st.subheader("เลือกวิธีการนำเข้าข้อมูล:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 1. อัปโหลดไฟล์จากเครื่อง")
            uploaded_file = st.file_uploader("อัปโหลดไฟล์ CSV ผลลัพธ์ Model Refinement", type=['csv'], key="local_uploader") # CHANGED
            if uploaded_file is not None:
                try:
                    st.session_state.df_for_analysis = pd.read_csv(uploaded_file)
                    st.success("✅ โหลดไฟล์สำเร็จ!")
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")
                    st.session_state.df_for_analysis = None

        with col2:
            st.markdown("##### 2. หรือ โหลดจาก GitHub URL")
            # This URL part needs to be updated manually if the repo structure changes
            default_github_url = f"https://raw.githubusercontent.com/firstnattapon/streamlit-example-1/refs/heads/master/Model_Refinement_Results/{st.session_state.test_ticker}.csv" # CHANGED
            github_url = st.text_input("ป้อน GitHub URL ของไฟล์ CSV:", value=default_github_url, key="github_url_input")
            if st.button("📥 โหลดข้อมูลจาก GitHub"):
                if github_url:
                    try:
                        raw_url = github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
                        with st.spinner("กำลังดาวน์โหลดข้อมูล..."):
                            st.session_state.df_for_analysis = pd.read_csv(raw_url)
                        st.success("✅ โหลดข้อมูลจาก GitHub สำเร็จ!")
                    except Exception as e:
                        st.error(f"❌ ไม่สามารถโหลดข้อมูลจาก URL ได้: {e}")
                        st.session_state.df_for_analysis = None
                else:
                    st.warning("กรุณาป้อน URL ของไฟล์ CSV")

    st.divider()

    if st.session_state.df_for_analysis is not None:
        st.subheader("ผลการวิเคราะห์")
        df_to_analyze = st.session_state.df_for_analysis
        try:
            # CHANGED best_seed to best_model_id
            required_cols = ['window_number', 'timeline', 'max_net', 'best_model_id', 'price_change_pct', 'action_sequence', 'window_size']
            if not all(col in df_to_analyze.columns for col in required_cols):
                st.error(f"ไฟล์ CSV ไม่สมบูรณ์! กรุณาตรวจสอบว่ามีคอลัมน์เหล่านี้ทั้งหมด: {', '.join(required_cols)}")
                return

            df = df_to_analyze.copy()

            if 'result' not in df.columns:
                df['result'] = np.where(df['max_net'] > 0, 'Win', 'Loss')

            overview_tab, stitched_model_tab = st.tabs([ # CHANGED
                "🔬 ภาพรวมและสำรวจราย Window",
                "🧬 Stitched Model Sequence Analysis"
            ])

            with overview_tab:
                st.subheader("ภาพรวมประสิทธิภาพ (Overall Performance)")
                gross_profit = df[df['max_net'] > 0]['max_net'].sum()
                gross_loss = abs(df[df['max_net'] < 0]['max_net'].sum())
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                win_rate = (df['result'] == 'Win').mean() * 100

                kpi_cols = st.columns(4)
                kpi_cols[0].metric("Total Net Profit", f"${df['max_net'].sum():,.2f}")
                kpi_cols[1].metric("Win Rate", f"{win_rate:.2f}%")
                kpi_cols[2].metric("Profit Factor", f"{profit_factor:.2f}")
                kpi_cols[3].metric("Total Windows", f"{df.shape[0]}")

                st.subheader("สำรวจข้อมูลราย Window")
                selected_window = st.selectbox(
                    'เลือก Window ที่ต้องการดูรายละเอียด:',
                    options=df['window_number'],
                    format_func=lambda x: f"Window #{x} (Timeline: {df.loc[df['window_number'] == x, 'timeline'].iloc[0]})"
                )
                if selected_window:
                    window_data = df[df['window_number'] == selected_window].iloc[0]
                    st.markdown(f"**รายละเอียดของ Window #{selected_window}**")
                    w_cols = st.columns(3)
                    w_cols[0].metric("Net Profit", f"${window_data['max_net']:.2f}")
                    w_cols[1].metric("Best Model ID", f"{window_data['best_model_id']}") # CHANGED
                    w_cols[2].metric("Price Change", f"{window_data['price_change_pct']:.2f}%")
                    st.markdown(f"**Action Sequence (ผลลัพธ์จาก Model):**") # CHANGED
                    st.code(window_data['action_sequence'], language='json')

            def safe_literal_eval(val):
                if pd.isna(val): return []
                if isinstance(val, list): return val
                if isinstance(val, str) and val.strip().startswith('[') and val.strip().endswith(']'):
                    try: return ast.literal_eval(val)
                    except: return []
                return []

            with stitched_model_tab: # CHANGED
                st.subheader("ทดสอบกลยุทธ์จาก 'Stitched' Model Sequence") # CHANGED
                st.markdown("จำลองการเทรดจริงโดยนำ **`action_sequence`** (ผลลัพธ์ของ Best Model จากแต่ละ Window) มา 'เย็บ' ต่อกัน และเปรียบเทียบกับ Benchmark") # CHANGED

                df['action_sequence_list'] = [safe_literal_eval(val) for val in df['action_sequence']]
                df_sorted = df.sort_values('window_number')
                stitched_actions = [action for seq in df_sorted['action_sequence_list'] for action in seq]

                dna_cols = st.columns(2)
                stitch_ticker = dna_cols[0].text_input(
                    "Ticker สำหรับจำลอง",
                    value=st.session_state.test_ticker,
                    key='stitch_ticker_input'
                )
                stitch_start_date = dna_cols[1].date_input("วันที่เริ่มต้นจำลอง", value=datetime(2024, 1, 1), key='stitch_date_input')

                if st.button("🧬 เริ่มการวิเคราะห์ Stitched Model Sequence", type="primary", key='stitch_model_btn'): # CHANGED
                    if not stitched_actions:
                        st.error("ไม่สามารถสร้าง Action Sequence จากข้อมูลที่โหลดได้")
                    else:
                        with st.spinner(f"กำลังจำลองกลยุทธ์สำหรับ {stitch_ticker}..."):
                            sim_data = get_ticker_data(stitch_ticker, str(stitch_start_date), str(datetime.now()))
                            if sim_data.empty:
                                st.error("ไม่สามารถดึงข้อมูลสำหรับจำลองได้")
                            else:
                                prices = sim_data['Close'].tolist()
                                n_total = len(prices)

                                final_actions_model = stitched_actions[:n_total] # CHANGED
                                df_model = run_simulation(prices[:len(final_actions_model)], final_actions_model) # CHANGED
                                df_max = run_simulation(prices, generate_actions_perfect_foresight(prices).tolist())
                                df_min = run_simulation(prices, generate_actions_rebalance_daily(n_total).tolist())

                                results_model = {} # CHANGED
                                if not df_model.empty:
                                    df_model.index = sim_data.index[:len(df_model)]
                                    results_model['Stitched Model Sequence'] = df_model # CHANGED
                                if not df_max.empty:
                                    df_max.index = sim_data.index[:len(df_max)]
                                    results_model[Strategy.PERFECT_FORESIGHT] = df_max
                                if not df_min.empty:
                                    df_min.index = sim_data.index[:len(df_min)]
                                    results_model[Strategy.REBALANCE_DAILY] = df_min

                                st.subheader("Performance Comparison (Net Profit)")
                                display_comparison_charts(results_model) # CHANGED

                                st.subheader("สรุปผลลัพธ์สุดท้าย (Final Net Profit)")
                                metric_cols = st.columns(3)

                                final_net_max = results_model.get(Strategy.PERFECT_FORESIGHT, pd.DataFrame({'net': [0]}))['net'].iloc[-1]
                                final_net_model = results_model.get('Stitched Model Sequence', pd.DataFrame({'net': [0]}))['net'].iloc[-1] # CHANGED
                                final_net_min = results_model.get(Strategy.REBALANCE_DAILY, pd.DataFrame({'net': [0]}))['net'].iloc[-1]

                                metric_cols[0].metric("Max Performance", f"${final_net_max:,.2f}")
                                metric_cols[1].metric("Stitched Model Strategy", f"${final_net_model:,.2f}", delta=f"{final_net_model - final_net_min:,.2f} vs Min", delta_color="normal") # CHANGED
                                metric_cols[2].metric("Min Performance", f"${final_net_min:,.2f}")

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการวิเคราะห์ข้อมูล: {e}")
            st.exception(e)

def render_manual_model_tab(config: Dict[str, Any]): # CHANGED
    st.header("🌱 Manual Model Comparator") # CHANGED
    st.markdown("สร้างและเปรียบเทียบ Action Sequences โดยใช้ **Model ID** ที่กำหนด และตัดส่วนท้าย (`tail`) ของผลลัพธ์มาใช้งาน") # CHANGED

    with st.container(border=True):
        st.subheader("1. กำหนดค่า Input สำหรับการทดสอบ")

        col1, col2 = st.columns([1, 2])
        with col1:
            asset_list = config.get('assets', ['FFWM'])
            try:
                default_index = asset_list.index(st.session_state.get('manual_ticker_key', st.session_state.test_ticker))
            except (ValueError, KeyError):
                default_index = 0
            
            manual_ticker = st.selectbox(
                "เลือก Ticker",
                options=asset_list,
                index=default_index,
                key="manual_ticker_key",
                on_change=on_ticker_change_callback,
                args=(config,)
            )

        with col2:
            c1, c2 = st.columns(2)
            manual_start_date = c1.date_input("วันที่เริ่มต้น (Start Date)", value= st.session_state.start_date , key="manual_start_compare_tail")
            manual_end_date = c2.date_input("วันที่สิ้นสุด (End Date)", value=datetime(2025, 7, 24).date(), key="manual_end_compare_tail")

        if manual_start_date >= manual_end_date:
            st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")

        st.divider()

        st.write("**กำหนด Model (ID/Size/Tail) ที่ต้องการเปรียบเทียบ:**") # CHANGED

        for i, line in enumerate(st.session_state.manual_model_lines): # CHANGED
            cols = st.columns([1, 2, 2, 2])
            cols[0].write(f"**Line {i+1}**")
            line['model_id'] = cols[1].number_input("Model ID", value=line.get('model_id', 1), min_value=0, key=f"model_id_compare_tail_{i}") # CHANGED
            line['size'] = cols[2].number_input("Size (ขนาด Sequence เริ่มต้น)", value=line.get('size', 60), min_value=1, key=f"size_compare_tail_{i}")
            line['tail'] = cols[3].number_input("Tail (ส่วนท้ายที่จะใช้)", value=line.get('tail', 10), min_value=1, max_value=line.get('size', 60), key=f"tail_compare_tail_{i}")

        b_col1, b_col2, _ = st.columns([1,1,4])
        if b_col1.button("➕ เพิ่ม Line เปรียบเทียบ"):
            st.session_state.manual_model_lines.append({'model_id': np.random.randint(1, 10000), 'size': 50, 'tail': 20}) # CHANGED
            st.rerun()

        if b_col2.button("➖ ลบ Line สุดท้าย"):
            if len(st.session_state.manual_model_lines) > 1: # CHANGED
                st.session_state.manual_model_lines.pop() # CHANGED
                st.rerun()
            else:
                st.warning("ต้องมีอย่างน้อย 1 line")

    st.write("---")

    if st.button("📈 เปรียบเทียบประสิทธิภาพ Models", type="primary", key="compare_manual_models_btn"): # CHANGED
        if manual_start_date >= manual_end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้อง"); return

        with st.spinner("กำลังดึงข้อมูลและจำลองการเทรด..."):
            start_str = manual_start_date.strftime('%Y-%m-%d'); end_str = manual_end_date.strftime('%Y-%m-%d')
            ticker_data = get_ticker_data(manual_ticker, start_str, end_str)

            if ticker_data.empty:
                st.error(f"ไม่พบข้อมูลสำหรับ {manual_ticker} ในช่วงวันที่ที่เลือก"); return

            prices = ticker_data['Close'].tolist(); num_trading_days = len(prices)
            st.info(f"📊 พบข้อมูลราคา {num_trading_days} วันทำการในช่วงที่เลือก")

            results = {}
            max_sim_len = 0

            for i, line_info in enumerate(st.session_state.manual_model_lines): # CHANGED
                input_model_id, size_model, tail_model = line_info['model_id'], line_info['size'], line_info['tail'] # CHANGED

                if tail_model > size_model:
                    st.error(f"Line {i+1}: Tail ({tail_model}) ต้องไม่มากกว่า Size ({size_model})"); return

                rng_best = np.random.default_rng(input_model_id)
                full_actions = rng_best.integers(0, 2, size=size_model)
                actions_from_tail = full_actions[-tail_model:].tolist()

                sim_len = min(num_trading_days, len(actions_from_tail))
                if sim_len == 0: continue

                prices_to_sim, actions_to_sim = prices[:sim_len], actions_from_tail[:sim_len]

                df_line = run_simulation(prices_to_sim, actions_to_sim)
                if not df_line.empty:
                    df_line.index = ticker_data.index[:sim_len]
                    strategy_name = f"Model {input_model_id} (Tail {tail_model})" # CHANGED
                    results[strategy_name] = df_line
                    max_sim_len = max(max_sim_len, sim_len)

            if not results:
                st.error("ไม่สามารถสร้างผลลัพธ์จาก Model ที่กำหนดได้"); return # CHANGED

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

            final_results_list = [{'name': name, 'net': results[name]['net'].iloc[-1]}
                                  for name in display_order if name in results and not results[name].empty]

            if final_results_list:
                final_metrics_cols = st.columns(len(final_results_list))
                for idx, item in enumerate(final_results_list):
                    final_metrics_cols[idx].metric(item['name'], f"${item['net']:,.2f}")

# ==============================================================================
# 5. Main Application
# ==============================================================================
def main():
    st.set_page_config(page_title="Model Refinement ML", page_icon="🤖", layout="wide") # CHANGED
    st.markdown("🤖 Model Refinement ML Tester (Optimized)") # CHANGED
    st.caption("เครื่องมือทดสอบการปรับแต่งโมเดล (Model Refinement) ด้วยวิธี Sliding Window") # CHANGED

    config = load_config()
    initialize_session_state(config)

    # CHANGED Tab Names
    tab1, tab2, tab3, tab4 = st.tabs([
        "⚙️ การตั้งค่า",
        "🚀 Model Refinement",
        "📊 Advanced Analytics",
        "🌱 Manual Model Comparison"
    ])

    with tab1:
        render_settings_tab(config)
    with tab2:
        render_test_tab()
    with tab3:
        render_analytics_tab()
    with tab4:
        render_manual_model_tab(config) # CHANGED

    with st.expander("📖 คำอธิบายวิธีการทำงานและแนวคิดการ Refactor"): # CHANGED
        st.markdown("""
        **แนวคิดหลัก: จาก 'Best Seed' สู่ 'Model Refinement'**
        
        เวอร์ชันนี้ได้เปลี่ยนมุมมอง (Reframe) จากการ "ค้นหา Seed ที่ดีที่สุด" ไปสู่กระบวนการที่เรียกว่า **"การปรับแต่งโมเดล (Model Refinement)"** ซึ่งเป็นแนวคิดพื้นฐานใน Machine Learning:
        - **Model ID**: แต่ละ `seed` ในเวอร์ชันก่อนหน้า ตอนนี้เปรียบเสมือน `Model ID` ที่ไม่ซ้ำกัน ซึ่งใช้สร้าง "โมเดล" เริ่มต้นขึ้นมา
        - **Action Sequence**: คือ "ผลลัพธ์" หรือ "Prediction" ที่ได้จากโมเดลนั้นๆ (สร้างจาก `np.random.default_rng(model_id)`)
        - **Model Refinement**: กระบวนการ `Sliding Window` คือการทดสอบโมเดลจำนวนมาก (`num_models_to_test`) เพื่อ "คัดเลือก" และ "ปรับแต่ง" หาโมเดลที่ให้ผลลัพธ์ดีที่สุดสำหรับข้อมูลในแต่ละช่วงเวลา (Window)
        - **Stitched Model Sequence**: คือการนำผลลัพธ์จากโมเดลที่ดีที่สุดของแต่ละช่วงเวลามาประกอบกันเป็นกลยุทธ์สุดท้าย เปรียบเสมือนการสร้าง Ensemble Model
        
        **หลักการ Refactoring ที่ใช้:**
        - **Dynamic Configuration**: ตั้งค่าผ่านไฟล์ `model_refinement_config.json` ทำให้ปรับเปลี่ยน Asset และค่าเริ่มต้นของแต่ละ Model ได้สะดวก
        - **Callback-Driven UI**: ใช้ `on_change` เพื่ออัปเดตค่า Preset ของ Manual Model ทันทีที่ผู้ใช้เปลี่ยน Ticker
        - **Separation of Concerns**: แยกโค้ดส่วน UI, การคำนวณ, และการสร้างข้อมูลออกจากกันชัดเจน
        - **Centralized Initialization**: รวมการตั้งค่า `session_state` ไว้ที่เดียวเพื่อง่ายต่อการจัดการ
        """)

if __name__ == "__main__":
    main()
