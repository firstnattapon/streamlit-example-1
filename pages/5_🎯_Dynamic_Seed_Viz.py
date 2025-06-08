import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import math
import ast
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from datetime import datetime

# --- Page Configuration (ต้องเรียกเป็นคำสั่งแรก) ---
st.set_page_config(
    page_title="Unified Backtest & Analysis Suite",
    page_icon="🧩",
    layout="wide"
)


# ===================================================================
# SECTION 1: CORE BACKTESTING & CALCULATION FUNCTIONS
# (ฟังก์ชันหลักจากสคริปต์ Best Seed Tester)
# ===================================================================

@lru_cache(maxsize=1000)
def calculate_optimized_cached(action_tuple, price_tuple, fix=1500):
    """
    ฟังก์ชันคำนวณผลตอบแทนแบบ Cached เพื่อความเร็ว
    """
    action_array = np.asarray(action_tuple, dtype=np.int32)
    action_array[0] = 1  # Action แรกเป็นการซื้อเสมอ
    price_array = np.asarray(price_tuple, dtype=np.float64)
    n = len(action_array)
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
    refer = -fix * np.log(initial_price / price_array)  # Logarithmic Buy & Hold reference

    for i in range(1, n):
        curr_price = price_array[i]
        if action_array[i] == 0:  # Hold
            amount[i] = amount[i-1]
            buffer[i] = 0
        else:  # Buy/Rebalance
            amount[i] = fix / curr_price
            buffer[i] = amount[i-1] * curr_price - fix
        cash[i] = cash[i-1] + buffer[i]
        asset_value[i] = amount[i] * curr_price
        sumusd[i] = cash[i] + asset_value[i]

    return buffer, sumusd, cash, asset_value, amount, refer

def calculate_optimized(action_list, price_list, fix=1500):
    """Wrapper function to use the cached version."""
    return calculate_optimized_cached(tuple(action_list), tuple(price_list), fix)

def evaluate_seed_batch(seed_batch, prices_window, window_len):
    """
    ประเมินผลกำไรสำหรับกลุ่มของ Seeds (สำหรับ Parallel Processing)
    """
    results = []
    for seed in seed_batch:
        try:
            rng = np.random.default_rng(seed)
            actions_window = rng.integers(0, 2, size=window_len)
            actions_window[0] = 1
            if window_len < 2:
                final_net = 0
            else:
                _, sumusd, _, _, _, refer = calculate_optimized(actions_window.tolist(), prices_window.tolist())
                initial_capital = sumusd[0]
                net = sumusd - refer - initial_capital
                final_net = net[-1]
            results.append((seed, final_net))
        except Exception:
            results.append((seed, -np.inf))
            continue
    return results

def find_best_seed_sliding_window_optimized(price_list, ticker_data_with_dates, window_size, num_seeds_to_try, max_workers):
    """
    ฟังก์ชันหลักในการค้นหา Best Seed ด้วยวิธี Sliding Window โดยใช้ Parallel Processing
    """
    prices = np.asarray(price_list)
    n = len(prices)
    window_details = []
    num_windows = (n + window_size - 1) // window_size
    progress_bar = st.progress(0, text=f"Initializing for {num_windows} windows...")

    for i, start_index in enumerate(range(0, n, window_size)):
        progress_text = f"Processing Window {i+1}/{num_windows}..."
        progress_bar.progress((i + 1) / num_windows, text=progress_text)
        
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        window_len = len(prices_window)
        if window_len < 2: continue

        start_date = ticker_data_with_dates.index[start_index].strftime('%Y-%m-%d')
        end_date = ticker_data_with_dates.index[end_index-1].strftime('%Y-%m-%d')
        timeline_info = f"{start_date} to {end_date}"

        # Parallel processing
        best_seed_for_window = -1
        max_net_for_window = -np.inf
        random_seeds = np.arange(num_seeds_to_try)
        batch_size = max(1, num_seeds_to_try // (max_workers * 4)) # Fine-tune batch size
        seed_batches = [random_seeds[i:i+batch_size] for i in range(0, len(random_seeds), batch_size)]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(evaluate_seed_batch, batch, prices_window, window_len) for batch in seed_batches]
            for future in as_completed(futures):
                for seed, final_net in future.result():
                    if final_net > max_net_for_window:
                        max_net_for_window = final_net
                        best_seed_for_window = seed

        # Reconstruct best action sequence
        if best_seed_for_window != -1:
            rng_best = np.random.default_rng(best_seed_for_window)
            best_actions_for_window = rng_best.integers(0, 2, size=window_len)
            best_actions_for_window[0] = 1
        else: # Fallback
            best_actions_for_window = np.ones(window_len, dtype=int)
            max_net_for_window = 0

        window_detail = {
            'window_number': i + 1, 'timeline': timeline_info,
            'start_index': start_index, 'end_index': end_index - 1,
            'window_size': window_len, 'best_seed': best_seed_for_window,
            'max_net': round(max_net_for_window, 2),
            'start_price': round(prices_window[0], 2), 'end_price': round(prices_window[-1], 2),
            'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2),
            'action_count': int(np.sum(best_actions_for_window)),
            'action_sequence': best_actions_for_window.tolist()
        }
        window_details.append(window_detail)
    
    progress_bar.progress(1.0, text="Completed!")
    return window_details

@st.cache_data(ttl=3600)
def get_ticker_data(ticker, start_date_str, end_date_str):
    """
    ดึงข้อมูลหุ้นจาก yfinance และ cache ผลลัพธ์
    """
    try:
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)
        
        # ดึงข้อมูลเผื่อไว้เล็กน้อยสำหรับวันหยุด
        history_start = start_date - pd.Timedelta(days=7)
        history_end = end_date + pd.Timedelta(days=1)
        
        tickerData = yf.Ticker(ticker).history(start=history_start, end=history_end)[['Close']]
        
        # กรองข้อมูลให้ตรงกับช่วงวันที่ที่ต้องการหลังจากดึงมาแล้ว
        tickerData = tickerData[(tickerData.index.date >= start_date.date()) & (tickerData.index.date <= end_date.date())]

        return tickerData
    except Exception as e:
        st.error(f"❌ ไม่สามารถดึงข้อมูล {ticker} ได้: {str(e)}")
        return pd.DataFrame()


# ===================================================================
# SECTION 2: STREAMLIT APP LAYOUT & LOGIC
# ===================================================================

# --- Initialize Session State ---
if 'analysis_df' not in st.session_state:
    st.session_state.analysis_df = None
if 'seed_list_from_file' not in st.session_state:
    st.session_state.seed_list_from_file = "[]"

# --- Main App Title ---
st.title("🧩 Unified Backtest & Analysis Suite")
st.markdown("เครื่องมือครบวงจรสำหรับ **สร้าง** ผลการทดสอบกลยุทธ์แบบ Sliding Window และ **วิเคราะห์** ผลลัพธ์ในเชิงลึก")

# --- Create Main Tabs ---
tab_generator, tab_analyzer = st.tabs(["🚀 Backtest & Generate Seeds", "📊 Advanced Analytics Dashboard"])


# --- TAB 1: BACKTEST & GENERATE SEEDS ---
with tab_generator:
    st.header("1. ตั้งค่าและรัน Backtest")
    st.markdown("ในส่วนนี้, เราจะทำการค้นหา `Best Seed` สำหรับแต่ละช่วงเวลา (Window) ของข้อมูลราคาหุ้นที่คุณเลือก")

    with st.container(border=True):
        st.subheader("⚙️ พารามิเตอร์การทดสอบ")
        col1, col2 = st.columns(2)
        with col1:
            test_ticker = st.selectbox(
                "เลือก Ticker สำหรับทดสอบ",
                ['FFWM', 'NEGG', 'RIVN', 'APLS', 'NVTS', 'QXO', 'RXRX', 'SPY', 'QQQ', 'TSLA'],
                index=0, key='gen_ticker'
            )
            start_date = st.date_input(
                "วันที่เริ่มต้น", datetime(2023, 1, 1), key='gen_start'
            )
            window_size = st.number_input(
                "ขนาด Window (วัน)", min_value=5, max_value=120, value=30, step=5, key='gen_window'
            )
        with col2:
            max_workers = st.number_input(
                "จำนวน Workers (Parallel Processing)", min_value=1, max_value=16, value=8,
                help="เพิ่มจำนวนเพื่อความเร็ว (แนะนำ 4-8 สำหรับ CPU ส่วนใหญ่)", key='gen_workers'
            )
            end_date = st.date_input("วันที่สิ้นสุด", datetime.now(), key='gen_end')
            num_seeds = st.number_input(
                "จำนวน Seeds ต่อ Window", min_value=100, max_value=100000, value=10000, step=1000, key='gen_seeds'
            )
    
    if st.button("🚀 เริ่มการค้นหา Best Seeds!", type="primary", use_container_width=True):
        if start_date >= end_date:
            st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
        else:
            with st.spinner(f"กำลังดึงข้อมูลสำหรับ {test_ticker}..."):
                ticker_data = get_ticker_data(test_ticker, str(start_date), str(end_date))

            if ticker_data.empty or len(ticker_data) < window_size:
                st.warning(f"ไม่พบข้อมูลที่เพียงพอสำหรับ Ticker '{test_ticker}' ในช่วงวันที่ที่เลือก (ต้องการอย่างน้อย {window_size} วัน)")
            else:
                st.success(f"ดึงข้อมูลสำเร็จ: {len(ticker_data)} วันทำการ")
                
                with st.status(f"กำลังรัน Backtest สำหรับ {test_ticker}...", expanded=True) as status:
                    window_details = find_best_seed_sliding_window_optimized(
                        ticker_data['Close'].tolist(),
                        ticker_data,
                        window_size=window_size,
                        num_seeds_to_try=num_seeds,
                        max_workers=max_workers
                    )
                    status.update(label="Backtest เสร็จสิ้น!", state="complete")

                if window_details:
                    df_results = pd.DataFrame(window_details)
                    # เก็บผลลัพธ์ใน session_state เพื่อให้แท็บอื่นใช้ได้
                    st.session_state.analysis_df = df_results
                    
                    st.header("📈 สรุปผลการ Backtest")
                    total_net = df_results['max_net'].sum()
                    win_rate = (df_results['max_net'] > 0).mean() * 100
                    
                    res_col1, res_col2, res_col3 = st.columns(3)
                    res_col1.metric("Total Net Profit (Sum of Windows)", f"${total_net:,.2f}")
                    res_col2.metric("Win Rate", f"{win_rate:.2f}%")
                    res_col3.metric("Total Windows Found", len(df_results))
                    
                    st.dataframe(df_results.drop('action_sequence', axis=1), use_container_width=True)
                    
                    csv = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 ดาวน์โหลดผลลัพธ์ (CSV)",
                        data=csv,
                        file_name=f'best_seed_results_{test_ticker}_{window_size}d_{num_seeds}s.csv',
                        mime='text/csv',
                    )
                    st.success("ผลลัพธ์ถูกเก็บไว้แล้ว สามารถไปที่แท็บ 'Advanced Analytics Dashboard' เพื่อวิเคราะห์ต่อได้ทันที!")

# --- TAB 2: ADVANCED ANALYTICS DASHBOARD ---
with tab_analyzer:
    st.header("2. วิเคราะห์ผลลัพธ์ Backtest ในเชิงลึก")

    # --- Data Source Selection ---
    source_option = st.radio(
        "เลือกแหล่งข้อมูลเพื่อการวิเคราะห์:",
        ["ใช้ผลลัพธ์จากการ Backtest ล่าสุด", "อัปโหลดไฟล์ CSV"],
        horizontal=True, key='data_source'
    )

    df_to_analyze = None
    if source_option == "ใช้ผลลัพธ์จากการ Backtest ล่าสุด":
        if st.session_state.analysis_df is not None:
            df_to_analyze = st.session_state.analysis_df
            st.success("โหลดข้อมูลจากการ Backtest ล่าสุดเรียบร้อยแล้ว")
        else:
            st.info("ยังไม่มีผลการ Backtest ใน Session นี้ กรุณากลับไปที่แท็บแรกเพื่อรัน Backtest หรือเลือกอัปโหลดไฟล์ CSV")
    
    else: # Upload a CSV file
        uploaded_file = st.file_uploader(
            "อัปโหลดไฟล์ CSV ผลลัพธ์ 'best_seed' ของคุณ", type=['csv']
        )
        if uploaded_file:
            try:
                df_to_analyze = pd.read_csv(uploaded_file)
                st.success(f"ไฟล์ '{uploaded_file.name}' ถูกประมวลผลเรียบร้อยแล้ว")
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")
                df_to_analyze = None

    # --- Analysis Section (runs only if data is available) ---
    if df_to_analyze is not None:
        try:
            # Pre-process data
            df = df_to_analyze.copy()
            if 'result' not in df.columns:
                df['result'] = np.where(df['max_net'] > 0, 'Win', 'Loss')
            total_windows = df.shape[0]

            # --- Create Sub-tabs for different analysis types ---
            overview_tab, stitched_dna_tab, insights_tab = st.tabs([
                "🔬 ภาพรวมและสำรวจราย Window", 
                "🧬 Stitched DNA Analysis",
                "💡 Insights & Correlations"
            ])

            with overview_tab:
                # --- Key Performance Indicators ---
                st.subheader("ภาพรวมประสิทธิภาพ (Overall Performance)")
                gross_profit = df[df['max_net'] > 0]['max_net'].sum()
                gross_loss = abs(df[df['max_net'] < 0]['max_net'].sum())
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
                win_rate = (df['result'] == 'Win').mean() * 100

                kpi_cols = st.columns(4)
                kpi_cols[0].metric("Total Net Profit", f"${df['max_net'].sum():,.2f}")
                kpi_cols[1].metric("Win Rate", f"{win_rate:.2f}%")
                kpi_cols[2].metric("Profit Factor", f"{profit_factor:.2f}")
                kpi_cols[3].metric("Total Windows", f"{total_windows}")

                # --- Individual Window Explorer ---
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
                    w_cols[1].metric("Best Seed", f"{window_data['best_seed']}")
                    w_cols[2].metric("Price Change", f"{window_data['price_change_pct']:.2f}%")
                    
                    st.markdown(f"**Action Sequence:**")
                    st.code(window_data['action_sequence'], language='json')

            with stitched_dna_tab:
                st.subheader("ทดสอบกลยุทธ์จาก 'Stitched' DNA")
                st.markdown("จำลองการเทรดจริงโดยใช้ `best_seed` ที่ได้จากแต่ละ Window มา 'เย็บ' ต่อกันเป็นกลยุทธ์เดียว")

                # Auto-populate seeds from the loaded data
                if 'best_seed' in df.columns:
                    extracted_seeds = df.sort_values('window_number')['best_seed'].tolist()
                    st.session_state.seed_list_from_file = str(extracted_seeds)
                
                seed_list_input = st.text_area(
                    "DNA Seed List (แก้ไขได้):",
                    value=st.session_state.seed_list_from_file,
                    height=100,
                    help="รายการ seed ที่ดึงมาจากข้อมูลที่โหลด แก้ไขได้หากต้องการทดลอง"
                )

                dna_cols = st.columns(2)
                stitch_ticker = dna_cols[0].text_input("Ticker สำหรับจำลอง", value=st.session_state.get('gen_ticker', 'FFWM'))
                stitch_start_date = dna_cols[1].date_input("วันที่เริ่มต้นจำลอง", value=st.session_state.get('gen_start', datetime(2023, 1, 1)))

                if st.button("🧬 เริ่มการวิเคราะห์ Stitched DNA"):
                    try:
                        seeds_for_ticker = ast.literal_eval(seed_list_input)
                        if not isinstance(seeds_for_ticker, list):
                            st.error("รูปแบบ Seed List ไม่ถูกต้อง")
                        else:
                            with st.spinner(f"กำลังจำลองกลยุทธ์ Stitched DNA สำหรับ {stitch_ticker}..."):
                                # Fetch full price data for simulation
                                sim_data = get_ticker_data(stitch_ticker, str(stitch_start_date), str(datetime.now()))
                                if sim_data.empty:
                                    st.error("ไม่สามารถดึงข้อมูลสำหรับจำลองได้")
                                else:
                                    prices = sim_data['Close'].values
                                    n_total = len(prices)
                                    window_size_sim = int(n_total / len(seeds_for_ticker)) if len(seeds_for_ticker) > 0 else 30
                                    
                                    # Create stitched action sequence
                                    final_actions, seed_index = [], 0
                                    for i in range(0, n_total, window_size_sim):
                                        current_seed = seeds_for_ticker[min(seed_index, len(seeds_for_ticker)-1)]
                                        rng = np.random.default_rng(current_seed)
                                        actions_for_window = rng.integers(0, 2, min(window_size_sim, n_total - i))
                                        final_actions.extend(actions_for_window)
                                        seed_index += 1
                                    
                                    # Calculate results
                                    _, sumusd, _, _, _, refer = calculate_optimized(final_actions, prices)
                                    net = sumusd - refer - sumusd[0]
                                    
                                    # Plot
                                    plot_df = pd.DataFrame({'Stitched DNA Net Profit': net}, index=sim_data.index)
                                    st.subheader("Performance of Stitched DNA Strategy")
                                    st.line_chart(plot_df)
                                    st.metric("Final Net Profit", f"${net[-1]:,.2f}")

                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดในการวิเคราะห์: {e}")

            with insights_tab:
                st.subheader("ค้นหา Insights และความสัมพันธ์")
                
                st.markdown("**ความสัมพันธ์ระหว่างกำไร (Net Profit) และการเปลี่ยนแปลงราคา (Price Change)**")
                fig = px.scatter(
                    df, x='price_change_pct', y='max_net', color='result',
                    color_discrete_map={'Win': 'green', 'Loss': 'red'},
                    labels={'price_change_pct': 'Price Change (%)', 'max_net': 'Net Profit ($)'},
                    title='Net Profit vs. Price Change in each Window',
                    hover_data=['window_number', 'best_seed']
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**การกระจายตัวของ Net Profit**")
                fig2 = px.histogram(
                    df, x='max_net', color='result',
                    marginal='box', nbins=50,
                    title='Distribution of Net Profit per Window'
                )
                st.plotly_chart(fig2, use_container_width=True)


        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการวิเคราะห์ข้อมูล: {e}")
            st.warning("กรุณาตรวจสอบว่าไฟล์ CSV ของคุณมีคอลัมน์ที่จำเป็น เช่น 'window_number', 'timeline', 'max_net', 'best_seed', 'price_change_pct', 'action_sequence'")
