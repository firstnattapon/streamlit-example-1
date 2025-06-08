import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import ast # เพิ่ม import สำหรับแปลง string เป็น list

# ตรวจสอบว่า statsmodels ถูกติดตั้งแล้วหรือไม่ (สำหรับให้คำแนะนำที่เป็นมิตร)
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# ===================================================================
# Dummy Functions (ฟังก์ชันจำลองเพื่อให้โค้ดรันได้)
# ในการใช้งานจริง ให้แทนที่ส่วนนี้ด้วยฟังก์ชัน back-end ของคุณ
# ===================================================================
def Limit_fx(Ticker, act=-1, n_points=1000):
    """
    ฟังก์ชันจำลองสำหรับดึงข้อมูลราคาหุ้น
    """
    np.random.seed(42) # เพื่อให้ผลลัพธ์เหมือนเดิมทุกครั้งที่รัน
    price = 100 + np.random.randn(n_points).cumsum()
    
    # สร้าง net profit จำลอง
    if act == -1: # min_net
        net = (price - price[0]) + np.random.randn(n_points) * 5
    elif act == -2: # max_net
        net = (price - price[0]) + np.abs(np.random.randn(n_points).cumsum()) * 2
    else:
        net = np.zeros(n_points)
        
    return pd.DataFrame({'price': price, 'net': net})

def calculate_optimized(actions, prices):
    """
    ฟังก์ชันจำลองสำหรับคำนวณผลการ backtest
    """
    n = len(prices)
    initial_capital = 10000
    sumusd = np.zeros(n)
    refer = (prices - prices[0]) * (initial_capital / prices[0]) # Buy and Hold
    
    # สร้างผลลัพธ์จำลองที่ดูสมเหตุสมผล
    # ผลลัพธ์จะขึ้นๆ ลงๆ ตาม actions และ price
    simulated_profit = np.cumsum((actions - 0.5) * np.diff(prices, prepend=prices[0]) * 10)
    sumusd = initial_capital + refer + simulated_profit
    
    buffer = np.random.rand(n)
    cash = np.random.rand(n) * initial_capital
    asset_value = sumusd - cash
    amount = np.random.rand(n) * 100

    return buffer, sumusd, cash, asset_value, amount, refer

# ===================================================================
# สิ้นสุดส่วนของ Dummy Functions
# ===================================================================

# ตั้งค่าธีมของ Plotly เพื่อความสวยงามและสอดคล้องกัน
px.defaults.template = "plotly_white"

def advanced_analytics_dashboard():
    """
    ฟังก์ชันหลักสำหรับสร้าง Advanced Analytics Dashboard
    """
    st.set_page_config(layout="wide", page_title="Advanced Backtest Analytics")

    st.title("🚀 Advanced Backtest Analytics Dashboard")
    st.markdown("""
    Dashboard นี้ถูกออกแบบมาเพื่อวิเคราะห์ผลลัพธ์การ Backtest ในเชิงลึก 
    โดยเน้นการค้นหา Insights ผ่านเทคนิคทาง Data Science และ Data Visualization
    """)

    uploaded_file = st.file_uploader(
        "อัปโหลดไฟล์ CSV ผลลัพธ์ 'best_seed' ของคุณ",
        type=['csv']
    )

    if uploaded_file is None:
        st.info("กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มต้นการวิเคราะห์")
        return

    try:
        # ใช้ session state เพื่อเก็บ df ไว้ ทำให้ไม่ต้องโหลดใหม่ทุกครั้งที่สลับแท็บ
        if 'main_df' not in st.session_state or st.session_state.uploader_key != uploaded_file.name:
            df = pd.read_csv(uploaded_file)
            df['result'] = np.where(df['max_net'] > 0, 'Win', 'Loss')
            df['result_color'] = df['result'].apply(lambda x: '#2ca02c' if x == 'Win' else '#d62728')
            st.session_state.main_df = df
            st.session_state.uploader_key = uploaded_file.name
        
        df = st.session_state.main_df

        gross_profit = df[df['max_net'] > 0]['max_net'].sum()
        gross_loss = abs(df[df['max_net'] < 0]['max_net'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        total_net_profit = df['max_net'].sum()
        winning_windows = df[df['result'] == 'Win'].shape[0]
        total_windows = df.shape[0]
        win_rate = (winning_windows / total_windows) * 100 if total_windows > 0 else 0
        avg_win = df[df['result'] == 'Win']['max_net'].mean()
        avg_loss = df[df['result'] == 'Loss']['max_net'].mean()
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")
        return

    st.success(f"ไฟล์ '{uploaded_file.name}' ถูกประมวลผลเรียบร้อยแล้ว มีข้อมูลทั้งหมด {total_windows} windows")

    overview_tab, stitched_dna_tab = st.tabs(["📊 ภาพรวมและสำรวจราย Window", "🧬 Stitched DNA Analysis"])
    
    with overview_tab:
        st.header("🔬 สำรวจข้อมูลราย Window")
        selected_window = st.selectbox(
            'เลือก Window ที่ต้องการดูรายละเอียด:',
            options=df['window_number'],
            format_func=lambda x: f"Window #{x} (Timeline: {df.loc[df['window_number'] == x, 'timeline'].iloc[0]})"
        )
        if selected_window:
            window_data = df[df['window_number'] == selected_window].iloc[0]
            st.subheader(f"รายละเอียดของ Window #{selected_window}")
            
            w_cols = st.columns(3)
            w_cols[0].metric("Net Profit", f"{window_data['max_net']:.2f}")
            w_cols[0].metric("Price Change", f"{window_data['price_change_pct']:.2f}%")
            w_cols[1].metric("Start Price", f"{window_data['start_price']:.2f}")
            w_cols[1].metric("End Price", f"{window_data['end_price']:.2f}")
            w_cols[2].metric("Best Seed", f"{window_data['best_seed']}")
            w_cols[2].metric("Action Count", f"{window_data['action_count']}")

            st.markdown(f"**Action Sequence:**")
            st.code(window_data['action_sequence'], language='json')
            
            st.markdown("**ข้อมูลดิบสำหรับ Window นี้:**")
            st.dataframe(window_data.to_frame().T.set_index('window_number'))

    # ===================================================================
    # โค้ดสำหรับแท็บ Stitched DNA (เวอร์ชันรวมและปรับปรุง)
    # ===================================================================
    with stitched_dna_tab:
        st.header("Stitched DNA Analysis")
        st.markdown("วิเคราะห์กลยุทธ์ที่สร้างจากการ 'เย็บ' DNA (best seeds) ของแต่ละ window เข้าด้วยกัน")
        st.info("ข้อมูล Best Seed จะถูกดึงมาจากไฟล์ที่คุณอัปโหลดในหน้าหลักโดยอัตโนมัติ")

        # --- Input Section ---
        col1, col2 = st.columns(2)
        with col1:
            ticker_for_stitching = st.selectbox(
                "1. เลือก Ticker เป้าหมาย",
                ['FFWM', 'NEGG', 'RIVN', 'APLS', 'NVTS', 'QXO', 'RXRX'],
                key='stitched_ticker_upload'
            )
        with col2:
            window_size_from_file = st.number_input("2. Window Size (ที่ใช้ในการวิเคราะห์)", value=30, key='window_upload')

        # Text area สำหรับแสดง/แก้ไข seed list
        if 'seed_list_from_file' not in st.session_state:
            st.session_state.seed_list_from_file = "[]" # เริ่มต้นด้วย list ว่าง

        # ปุ่มสำหรับดึงข้อมูล seed จากไฟล์ที่อัปโหลด
        if st.button("3. ดึงค่า Best Seeds จากไฟล์ที่อัปโหลด"):
            if 'best_seed' in df.columns:
                extracted_seeds = df['best_seed'].tolist()
                # อัปเดตค่าใน text area และ session_state
                st.session_state.seed_list_from_file = str(extracted_seeds)
                st.success(f"ดึงข้อมูล {len(extracted_seeds)} seeds เรียบร้อยแล้ว!")
            else:
                st.error("ไม่พบคอลัมน์ 'best_seed' ในไฟล์ CSV ที่อัปโหลด")

        seed_list_input = st.text_area(
            "4. ตรวจสอบหรือแก้ไข DNA Seed List:",
            value=st.session_state.seed_list_from_file,
            height=150,
            key='seed_list_area',
            help="แก้ไข list ของ seed ได้โดยตรงในรูปแบบ Python list เช่น [1, 2, 3]"
        )

        if st.button(f"5. เริ่มการวิเคราะห์ Stitched DNA สำหรับ {ticker_for_stitching}"):
            try:
                # ใช้ ast.literal_eval เพื่อแปลง string จาก text area เป็น list อย่างปลอดภัย
                seeds_for_ticker = ast.literal_eval(seed_list_input)
                
                if not isinstance(seeds_for_ticker, list):
                    st.error("รูปแบบข้อมูลใน Text Area ไม่ถูกต้อง กรุณาใส่ข้อมูลในรูปแบบ Python list เช่น `[1, 2, 3]`")
                elif not seeds_for_ticker:
                    st.warning("Seed List ว่างเปล่า กรุณาดึงข้อมูลหรือใส่ค่า Seed ก่อนทำการวิเคราะห์")
                else:
                    with st.spinner(f"กำลังวิเคราะห์ {ticker_for_stitching}..."):
                        # 1. ดึงข้อมูลราคาหุ้นทั้งหมด (ใช้ Dummy Function)
                        full_data_df = Limit_fx(Ticker=ticker_for_stitching, act=-1)
                        prices = full_data_df['price'].values
                        n_total = len(prices)

                        if n_total == 0:
                            st.error(f"ไม่พบข้อมูลราคาสำหรับ {ticker_for_stitching}")
                        else:
                            # 2. สร้าง Stitched Action Sequence
                            final_actions = []
                            seed_index = 0
                            
                            for i in range(0, n_total, window_size_from_file):
                                current_window_size = min(window_size_from_file, n_total - i)
                                
                                if seed_index >= len(seeds_for_ticker):
                                    st.warning(f"Seed หมดที่ window {seed_index + 1} จะใช้ Seed สุดท้าย ({seeds_for_ticker[-1]}) ต่อไป")
                                    current_seed = seeds_for_ticker[-1]
                                else:
                                    current_seed = seeds_for_ticker[seed_index]

                                rng = np.random.default_rng(current_seed)
                                actions_for_window = rng.integers(0, 2, current_window_size)
                                if len(actions_for_window) > 0:
                                    actions_for_window[0] = 1 # สมมติว่า action แรกคือการซื้อเสมอ
                                
                                final_actions.extend(actions_for_window)
                                seed_index += 1

                            stitched_actions = np.array(final_actions, dtype=np.int32)[:n_total]

                            # 3. คำนวณผลลัพธ์ (ใช้ Dummy Function)
                            min_net = full_data_df['net']
                            max_net = Limit_fx(Ticker=ticker_for_stitching, act=-2)['net']
                            buffer, sumusd, cash, asset_value, amount, refer = calculate_optimized(stitched_actions, prices)
                            initial_capital = sumusd[0]
                            stitched_net = np.round(sumusd - refer - initial_capital, 2)
                            
                            # 4. สร้าง DataFrame สำหรับพล็อตกราฟ
                            plot_df = pd.DataFrame({
                                'min_net (benchmark)': min_net.values,
                                'max_net (benchmark)': max_net.values,
                                'stitched_dna_net': stitched_net
                            }, index=full_data_df.index)

                            # 5. แสดงผล
                            st.subheader("Performance Comparison (Net Profit)")
                            st.line_chart(plot_df)

                            with st.expander("แสดงข้อมูลผลลัพธ์ (Data)"):
                                st.dataframe(plot_df.round(2))
                            with st.expander("รายละเอียดของ Seed ที่ใช้"):
                                st.write("จำนวน Seed ทั้งหมด:", len(seeds_for_ticker))
                                st.write("จำนวน Window ที่สร้าง:", seed_index)
                                st.dataframe({'used_seeds': seeds_for_ticker})
            
            except (ValueError, SyntaxError) as e:
                st.error(f"รูปแบบข้อมูลใน Text Area ไม่ถูกต้อง: {e}. กรุณาตรวจสอบว่าเป็น Python list ที่ถูกต้อง เช่น `[28834, 1408, 9009]`")
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดที่ไม่คาดคิด: {e}")
                st.exception(e)

if __name__ == "__main__":
    advanced_analytics_dashboard()
