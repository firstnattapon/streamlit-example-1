# v3: Combined Monitor (v1) with Advanced Analytics (v2) into a single app with multiple tabs

import streamlit as st
import numpy as np
import datetime
import thingspeak
import pandas as pd
import yfinance as yf
import json
from functools import lru_cache
import concurrent.futures
from threading import Lock
import os
import ast
from typing import List, Tuple, Dict, Any

st.set_page_config(page_title="Total Asset Monitor", page_icon="🏦", layout="wide")

# ==============================================================================
# 1. GLOBAL CONFIGURATION & CACHE SETUP
# ==============================================================================

# --- Session State Initialization for Analytics Tab ---
# This ensures that variables persist across reruns for the analytics tab
if 'df_for_analysis' not in st.session_state:
    st.session_state.df_for_analysis = None
if 'analytics_ticker' not in st.session_state:
    st.session_state.analytics_ticker = 'FFWM' # Default ticker for analytics tab

# --- Cache for Real-time Prices ---
_cache_lock = Lock()
_price_cache = {}
_cache_timestamp = {}

@st.cache_data
def load_monitor_config(file_path='monitor_config.json'):
    """Load asset configuration for the real-time monitor from a JSON file."""
    if not os.path.exists(file_path):
        st.error(f"Configuration file not found: {file_path}")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        st.error(f"Error loading {file_path}: {e}")
        return []

ASSET_CONFIGS = load_monitor_config()

@st.cache_resource
def get_thingspeak_clients():
    """Initialize and cache ThingSpeak clients."""
    try:
        channel_id_1, key_1 = 2528199, '2E65V8XEIPH9B2VV'
        channel_id_2, key_2 = 2385118, 'IPSG3MMMBJEB9DY8'
        return (
            thingspeak.Channel(channel_id_1, key_1, fmt='json'),
            thingspeak.Channel(channel_id_2, key_2, fmt='json')
        )
    except Exception as e:
        st.error(f"Failed to initialize ThingSpeak clients: {e}")
        return None, None

client, client_2 = get_thingspeak_clients()

def clear_all_caches():
    """Clear all Streamlit and custom caches."""
    st.cache_data.clear()
    st.cache_resource.clear()
    sell.cache_clear()
    buy.cache_clear()
    calculate_optimized_cached.cache_clear() # Clear cache for analytics function
    with _cache_lock:
        _price_cache.clear()
        _cache_timestamp.clear()
    st.success("🗑️ Clear ALL caches complete!")
    # No st.rerun() here to allow the message to be seen before the rerun from the button.

# ==============================================================================
# 2. HELPER FUNCTIONS (From v1 - Monitor)
# ==============================================================================

@lru_cache(maxsize=128)
def sell(asset, fix_c=1500, Diff=60):
    if asset == 0: return 0, 0, 0
    unit_price = round((fix_c - Diff) / asset, 2)
    adjust_qty = round(abs(asset * unit_price - fix_c) / unit_price) if unit_price != 0 else 0
    total = round(asset * unit_price + adjust_qty * unit_price, 2)
    return unit_price, adjust_qty, total

@lru_cache(maxsize=128)
def buy(asset, fix_c=1500, Diff=60):
    if asset == 0: return 0, 0, 0
    unit_price = round((fix_c + Diff) / asset, 2)
    adjust_qty = round(abs(asset * unit_price - fix_c) / unit_price) if unit_price != 0 else 0
    total = round(asset * unit_price - adjust_qty * unit_price, 2)
    return unit_price, adjust_qty, total

def get_cached_price(ticker, max_age=30):
    """Get price from a time-sensitive cache or fetch from yfinance."""
    now = datetime.datetime.now()
    with _cache_lock:
        if (ticker in _price_cache and
            ticker in _cache_timestamp and
            (now - _cache_timestamp[ticker]).seconds < max_age):
            return _price_cache[ticker]
    try:
        price = yf.Ticker(ticker).fast_info['lastPrice']
        with _cache_lock:
            _price_cache[ticker] = price
            _cache_timestamp[ticker] = now
        return price
    except Exception:
        return 0.0

@st.cache_data(ttl=60)
def get_all_assets_from_thingspeak(configs):
    """Fetch all asset values from ThingSpeak concurrently."""
    if not client or not configs: return {}
    assets = {}
    def fetch_asset(field):
        try:
            data = client.get_field_last(field=field)
            return eval(json.loads(data)[field])
        except Exception:
            return 0.0

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(configs)) as executor:
        future_to_field = {executor.submit(fetch_asset, asset['asset_field']): asset for asset in configs}
        for future in concurrent.futures.as_completed(future_to_field):
            config = future_to_field[future]
            try:
                assets[config['ticker']] = future.result()
            except Exception as e:
                st.error(f"Error fetching asset for {config['ticker']} (field {config['asset_field']}): {e}")
    return assets


# ==============================================================================
# 3. HELPER FUNCTIONS (From v2 - Analytics)
# ==============================================================================

@st.cache_data(ttl=3600)
def get_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """ดึงข้อมูลราคาปิดของ Ticker จาก yfinance และ cache ผลลัพธ์."""
    try:
        data = yf.Ticker(ticker).history(start=start_date, end=end_date)[['Close']]
        if data.empty:
            return pd.DataFrame()
        # Ensure timezone is consistent
        if data.index.tz is None:
            data = data.tz_localize('UTC').tz_convert('Asia/Bangkok')
        else:
            data = data.tz_convert('Asia/Bangkok')
        return data
    except Exception as e:
        st.error(f"❌ ไม่สามารถดึงข้อมูล {ticker} ได้: {e}")
        return pd.DataFrame()

@lru_cache(maxsize=2048)
def calculate_optimized_cached(action_tuple: Tuple[int, ...], price_tuple: Tuple[float, ...], fix: int = 1500) -> Tuple:
    """ฟังก์ชันคำนวณหลักที่ถูกแคชด้วย lru_cache."""
    action_array = np.asarray(action_tuple, dtype=np.int32)
    price_array = np.asarray(price_tuple, dtype=np.float64)
    n = len(action_array)
    if n == 0: return (np.array([]),) * 6
    action_array_calc = action_array.copy()
    action_array_calc[0] = 1
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
            buffer[i] = 0
        else:
            amount[i] = fix / curr_price
            buffer[i] = amount[i-1] * curr_price - fix
        cash[i] = cash[i-1] + buffer[i]
        asset_value[i] = amount[i] * curr_price
        sumusd[i] = cash[i] + asset_value[i]
    return buffer, sumusd, cash, asset_value, amount, refer

def run_simulation(prices: List[float], actions: List[int], fix: int = 1500) -> pd.DataFrame:
    """จำลองการเทรดจากราคาและ Action ที่กำหนด และคืนค่าเป็น DataFrame."""
    if not prices or not actions: return pd.DataFrame()
    buffer, sumusd, cash, asset_value, amount, refer = calculate_optimized_cached(tuple(actions), tuple(prices), fix)
    if len(sumusd) == 0: return pd.DataFrame()
    initial_capital = sumusd[0]
    return pd.DataFrame({
        'price': prices, 'action': actions, 'buffer': np.round(buffer, 2),
        'sumusd': np.round(sumusd, 2), 'cash': np.round(cash, 2),
        'asset_value': np.round(asset_value, 2), 'amount': np.round(amount, 2),
        'refer': np.round(refer + initial_capital, 2),
        'net': np.round(sumusd - refer - initial_capital, 2)
    })

def generate_actions_rebalance_daily(num_days: int) -> np.ndarray:
    return np.ones(num_days, dtype=int)

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
# 4. UI RENDERING FUNCTIONS
# ==============================================================================

def render_monitor_tab(configs, last_assets_all):
    """Renders the main real-time monitoring dashboard."""
    st.header("Real-time Asset Monitor & Trading Signals")
    
    if not configs:
        st.warning("Monitor configuration is empty. Please check `monitor_config.json`.")
        return

    # --- Controls ---
    st.markdown("##### Controls")
    control_cols = st.columns(4)
    x_2 = control_cols[0].number_input('Difference (Diff)', step=1, value=60, help="ค่าส่วนต่างสำหรับคำนวณราคาเป้าหมาย")
    
    # --- Asset Inputs & Calculations ---
    st.markdown("##### Asset Values")
    asset_inputs = {}
    cols = st.columns(len(configs))
    for i, config in enumerate(configs):
        with cols[i]:
            ticker = config['ticker']
            last_asset_val = last_assets_all.get(ticker, 0.0)
            label = f'{ticker} Asset'
            asset_inputs[ticker] = st.number_input(label, step=0.001, value=last_asset_val, key=f"input_{ticker}_asset")

    st.divider()

    # --- Trading Sections ---
    st.markdown("##### Trading Actions")
    trading_cols = st.columns(len(configs))
    
    for i, config in enumerate(configs):
        with trading_cols[i]:
            ticker = config['ticker']
            asset_val = asset_inputs.get(ticker, 0.0)
            fix_c = config['fix_c']
            
            st.subheader(f"{ticker}")

            # Calculations
            sell_calc = sell(asset_val, fix_c=fix_c, Diff=x_2)
            buy_calc = buy(asset_val, fix_c=fix_c, Diff=x_2)
            
            # Display Price Info
            try:
                current_price = get_cached_price(ticker)
                if current_price > 0:
                    pv = current_price * asset_val
                    pl_value = pv - fix_c
                    pl_color = "green" if pl_value >= 0 else "red"
                    st.metric(
                        label="Current Price",
                        value=f"{current_price:,.3f}",
                        delta=f"P/L: {pl_value:,.2f}",
                        delta_color=pl_color
                    )
                    st.write(f"Portfolio Value: **{pv:,.2f}**")
                else:
                    st.info(f"Price for {ticker} unavailable.")
            except Exception:
                st.warning(f"Could not get price for {ticker}.")

            st.write("---")
            
            # Sell Action
            st.markdown(f"**SELL** at **{buy_calc[0]}**")
            st.write(f"(Qty: `{buy_calc[1]}`, Total: `{buy_calc[2]}`)")
            if st.button(f"SELL MATCH {ticker}", key=f'btn_sell_{ticker}'):
                field_num = int(config['asset_field'].replace('field', ''))
                new_asset = last_assets_all.get(ticker, 0.0) - buy_calc[1]
                client.update({f'field{field_num}': new_asset})
                st.success(f"SELL order for {ticker} sent! New asset: {new_asset}")
                st.toast(f"Updated {ticker} on ThingSpeak!", icon="📉")
                # No clear cache here to avoid disrupting other users, will be cleared on manual rerun

            # Buy Action
            st.markdown(f"**BUY** at **{sell_calc[0]}**")
            st.write(f"(Qty: `{sell_calc[1]}`, Total: `{sell_calc[2]}`)")
            if st.button(f"BUY MATCH {ticker}", key=f'btn_buy_{ticker}'):
                field_num = int(config['asset_field'].replace('field', ''))
                new_asset = last_assets_all.get(ticker, 0.0) + sell_calc[1]
                client.update({f'field{field_num}': new_asset})
                st.success(f"BUY order for {ticker} sent! New asset: {new_asset}")
                st.toast(f"Updated {ticker} on ThingSpeak!", icon="📈")

def render_update_assets_tab(configs):
    """Renders the tab for manually updating asset values on ThingSpeak."""
    st.header("Manual Asset Update on ThingSpeak")
    st.warning("การเปลี่ยนแปลงค่าในหน้านี้จะเขียนทับข้อมูลบน ThingSpeak โดยตรง", icon="⚠️")
    
    if not client or not configs:
        st.error("ThingSpeak client or configuration not available.")
        return

    for config in configs:
        with st.container(border=True):
            ticker = config['ticker']
            field = config['asset_field']
            
            st.markdown(f"#### Update Asset: `{ticker}` (Field: {field})")
            new_val = st.number_input(f"Enter new value for {ticker}", step=0.001, value=0.0, key=f'input_update_{ticker}')
            if st.button(f"UPDATE {ticker} ON THINGSPEAK", key=f'btn_update_{ticker}', type="primary"):
                try:
                    client.update({field: new_val})
                    st.success(f"Successfully updated {ticker} to: {new_val}")
                    st.toast(f"Sent update for {ticker}!", icon="🛰️")
                    # It's good practice to clear cache after manual update
                    if st.checkbox("Clear cache after update?", key=f"clear_cache_{ticker}", value=True):
                        clear_all_caches()
                        st.rerun()

                except Exception as e:
                    st.error(f"Failed to update {ticker}: {e}")

def render_analytics_tab():
    """Renders the UI for the 'Advanced Analytics Dashboard' tab, ported from v2."""
    st.header("Stitched DNA Backtest Analysis")
    st.markdown("วิเคราะห์ผลลัพธ์ Backtest ในเชิงลึกจากไฟล์ CSV ที่ได้จาก `Best Seed Sliding Window`")

    # --- Data Loading Section ---
    with st.container(border=True):
        st.markdown("##### 1. นำเข้าข้อมูล")
        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader("อัปโหลดไฟล์ CSV จากเครื่อง", type=['csv'], key="local_uploader")
            if uploaded_file:
                try:
                    st.session_state.df_for_analysis = pd.read_csv(uploaded_file)
                    st.success("✅ โหลดไฟล์สำเร็จ!")
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")
                    st.session_state.df_for_analysis = None
        
        with col2:
            st.session_state.analytics_ticker = st.text_input(
                "หรือ ป้อน Ticker เพื่อโหลดจาก GitHub",
                value=st.session_state.analytics_ticker,
                key="github_ticker_input"
            )
            github_url = f"https://raw.githubusercontent.com/firstnattapon/streamlit-example-1/main/Seed_Sliding_Window/{st.session_state.analytics_ticker}.csv"
            
            if st.button("📥 โหลดข้อมูลจาก GitHub"):
                st.info(f"พยายามโหลดจาก: `{github_url}`")
                try:
                    with st.spinner("กำลังดาวน์โหลดข้อมูล..."):
                        st.session_state.df_for_analysis = pd.read_csv(github_url)
                    st.success(f"✅ โหลดข้อมูล {st.session_state.analytics_ticker} จาก GitHub สำเร็จ!")
                except Exception as e:
                    st.error(f"❌ ไม่สามารถโหลดข้อมูลจาก URL ได้: {e}")
                    st.session_state.df_for_analysis = None

    st.divider()

    # --- Analysis Section ---
    if st.session_state.df_for_analysis is not None:
        st.markdown("##### 2. ผลการวิเคราะห์")
        df_to_analyze = st.session_state.df_for_analysis

        try:
            # Check for required columns
            required_cols = ['window_number', 'timeline', 'max_net', 'best_seed', 'action_sequence']
            if not all(col in df_to_analyze.columns for col in required_cols):
                st.error(f"ไฟล์ CSV ไม่สมบูรณ์! กรุณาตรวจสอบว่ามีคอลัมน์เหล่านี้: {', '.join(required_cols)}")
                return

            st.markdown("ทดสอบกลยุทธ์จาก 'Stitched' DNA")
            st.markdown("จำลองการเทรดจริงโดยนำ `action_sequence` จากแต่ละ Window มา 'เย็บ' ต่อกัน และเปรียบเทียบกับ Benchmark")

            def safe_literal_eval(val):
                if pd.isna(val) or val is None: return []
                if isinstance(val, list): return val
                if isinstance(val, str) and val.strip().startswith('[') and val.strip().endswith(']'):
                    try: return ast.literal_eval(val)
                    except (ValueError, SyntaxError): return []
                return []

            # Create the full 'stitched' action sequence
            df_to_analyze['action_sequence_list'] = df_to_analyze['action_sequence'].apply(safe_literal_eval)
            stitched_actions = [action for seq in df_to_analyze.sort_values('window_number')['action_sequence_list'] for action in seq]

            dna_cols = st.columns(2)
            stitch_ticker = dna_cols[0].text_input("Ticker สำหรับจำลอง", value=st.session_state.analytics_ticker, key='stitch_ticker_input')
            stitch_start_date = dna_cols[1].date_input("วันที่เริ่มต้นจำลอง", value=datetime.date(2024, 1, 1), key='stitch_date_input')

            if st.button("🧬 เริ่มการวิเคราะห์ Stitched DNA", type="primary", use_container_width=True):
                if not stitched_actions:
                    st.error("ไม่สามารถสร้าง Action Sequence จากข้อมูลที่โหลดได้")
                else:
                    with st.spinner(f"กำลังจำลองกลยุทธ์สำหรับ {stitch_ticker}..."):
                        end_date = datetime.datetime.now()
                        sim_data = get_ticker_data(stitch_ticker, stitch_start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

                        if sim_data.empty:
                            st.error("ไม่สามารถดึงข้อมูลสำหรับจำลองได้")
                        else:
                            prices = sim_data['Close'].tolist()
                            n_total = len(prices)
                            final_actions_dna = stitched_actions[:n_total]

                            df_dna = run_simulation(prices[:len(final_actions_dna)], final_actions_dna)
                            df_max = run_simulation(prices, generate_actions_perfect_foresight(prices).tolist())
                            df_min = run_simulation(prices, generate_actions_rebalance_daily(n_total).tolist())

                            plot_len = len(df_dna)
                            if plot_len > 0:
                                # !!! THE BUG FIX IS APPLIED HERE !!!
                                # Using .values ensures we get the raw data without trying to align mismatched pandas indexes.
                                plot_df = pd.DataFrame({
                                    'Max Performance (Perfect)': df_max['net'].values[:plot_len],
                                    'Stitched DNA Strategy': df_dna['net'].values,
                                    'Min Performance (Rebalance Daily)': df_min['net'].values[:plot_len]
                                }, index=sim_data.index[:plot_len])

                                st.subheader("Performance Comparison (Net Profit)")
                                st.line_chart(plot_df)

                                st.subheader("สรุปผลลัพธ์สุดท้าย (Final Net Profit)")
                                final_net_dna = df_dna['net'].iloc[-1]
                                final_net_min = df_min['net'].iloc[plot_len-1]
                                metric_cols = st.columns(3)
                                metric_cols[0].metric("Max Performance (at DNA End)", f"${df_max['net'].iloc[plot_len-1]:,.2f}")
                                metric_cols[1].metric("Stitched DNA Strategy", f"${final_net_dna:,.2f}", delta=f"{final_net_dna - final_net_min:,.2f} vs Min", delta_color="off")
                                metric_cols[2].metric("Min Performance (at DNA End)", f"${final_net_min:,.2f}")
                            else:
                                st.warning("ไม่มีข้อมูล DNA ที่จะแสดงผลได้")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการวิเคราะห์ข้อมูล: {e}")
            st.exception(e) # Show full traceback for debugging

# ==============================================================================
# 5. MAIN APPLICATION LOGIC
# ==============================================================================
def main():
    st.title("🏦 Total Asset Management Dashboard")

    # --- Data Fetching for Monitor Tab ---
    # This runs regardless of the selected tab to keep data fresh,
    # but st.cache_data prevents re-fetching too often.
    last_assets_all = get_all_assets_from_thingspeak(ASSET_CONFIGS) if ASSET_CONFIGS else {}
    if not client or not client_2:
        st.error("ThingSpeak client is not available. Some features will be disabled.")
        return
    if not ASSET_CONFIGS:
        st.stop()

    # --- Main Tab Structure ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Real-time Monitor",
        "📝 Update Assets",
        "🧬 Advanced Analytics",
        "⚙️ System Control"
    ])

    with tab1:
        render_monitor_tab(ASSET_CONFIGS, last_assets_all)

    with tab2:
        render_update_assets_tab(ASSET_CONFIGS)
        
    with tab3:
        render_analytics_tab()

    with tab4:
        st.header("System Control")
        st.warning("การกระทำในหน้านี้มีผลต่อทั้งแอปพลิเคชัน")
        if st.button("🔄 RERUN & REFETCH DATA", use_container_width=True):
            st.rerun()
        if st.button("🗑️ CLEAR ALL CACHES & RERUN", type="primary", use_container_width=True):
            clear_all_caches()
            st.rerun()

if __name__ == "__main__":
    main()
