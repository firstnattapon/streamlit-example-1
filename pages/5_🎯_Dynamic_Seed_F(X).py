# ==============================================================================
# ===== Tab 3: Advanced Analytics Dashboard (ฉบับแก้ไขตามหลักการ) =====
# ==============================================================================
with tab3:
    st.header("2. วิเคราะห์ผลลัพธ์ Backtest ในเชิงลึก")
    
    # --- ใช้ Container เพื่อจัดกลุ่ม UI ของการโหลดข้อมูล ---
    with st.container():
        st.subheader("เลือกวิธีการนำเข้าข้อมูล:")
        
        # --- เริ่มต้น State สำหรับ Tab 3 ---
        if 'df_for_analysis' not in st.session_state:
            st.session_state.df_for_analysis = None

        # ส่วนสำหรับโหลดข้อมูล
        col1, col2 = st.columns(2)
        with col1:
            # --- อัปโหลดจากเครื่อง ---
            st.markdown("##### 1. อัปโหลดไฟล์จากเครื่อง")
            uploaded_file = st.file_uploader(
                "อัปโหลดไฟล์ CSV ของคุณ", type=['csv'], key="local_uploader"
            )
            if uploaded_file is not None:
                try:
                    st.session_state.df_for_analysis = pd.read_csv(uploaded_file)
                    st.success("✅ โหลดไฟล์สำเร็จ!")
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")
                    st.session_state.df_for_analysis = None
        
        with col2:
            # --- โหลดจาก GitHub ---
            st.markdown("##### 2. หรือ โหลดจาก GitHub URL")
            
            # <--- แก้ไขจุดที่ 1: สร้าง URL เริ่มต้นจาก st.session_state.test_ticker
            # หมายเหตุ: URL ของท่านใช้ .../refs/heads/master/... ซึ่งอาจไม่เป็น public URL มาตรฐาน
            # URL ที่ใช้งานได้ทั่วไปมักจะเป็น .../main/... หรือ .../master/...
            # แต่จะใช้โครงสร้างตามที่ท่านให้มาเป็นตัวอย่าง
            default_github_url = f"https://raw.githubusercontent.com/firstnattapon/streamlit-example-1/refs/heads/master/Seed_Sliding_Window/{st.session_state.test_ticker}.csv"

            github_url = st.text_input(
                "ป้อน GitHub URL ของไฟล์ CSV:", 
                value=default_github_url, # <--- แก้ไขจุดที่ 1: ใช้ URL ที่สร้างขึ้นเป็นค่าเริ่มต้น
                key="github_url_input"
            )
            if st.button("📥 โหลดข้อมูลจาก GitHub"):
                if github_url:
                    try:
                        # ตรวจสอบและแก้ไข URL ให้เป็น raw content URL โดยอัตโนมัติ
                        raw_url = github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
                        with st.spinner(f"กำลังดาวน์โหลดข้อมูล..."):
                            st.session_state.df_for_analysis = pd.read_csv(raw_url)
                        st.success("✅ โหลดข้อมูลจาก GitHub สำเร็จ!")
                    except Exception as e:
                        st.error(f"❌ ไม่สามารถโหลดข้อมูลจาก URL ได้: {e}")
                        st.session_state.df_for_analysis = None
                else:
                    st.warning("กรุณาป้อน URL ของไฟล์ CSV")
    
    st.divider() # เส้นคั่น

    # --- ส่วนของการวิเคราะห์ (จะทำงานเมื่อมีข้อมูลใน state เท่านั้น) ---
    if st.session_state.df_for_analysis is not None:
        st.subheader("ผลการวิเคราะห์")
        df_to_analyze = st.session_state.df_for_analysis

        try:
            required_cols = ['window_number', 'timeline', 'max_net', 'best_seed', 'price_change_pct', 'action_sequence', 'window_size']
            if not all(col in df_to_analyze.columns for col in required_cols):
                st.error(f"ไฟล์ CSV ไม่สมบูรณ์! กรุณาตรวจสอบว่ามีคอลัมน์เหล่านี้ทั้งหมด: {', '.join(required_cols)}")
            else:
                df = df_to_analyze.copy()
                if 'result' not in df.columns:
                    df['result'] = np.where(df['max_net'] > 0, 'Win', 'Loss')

                overview_tab, stitched_dna_tab = st.tabs([
                    "🔬 ภาพรวมและสำรวจราย Window",
                    "🧬 Stitched DNA Analysis"
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
                    selected_window = st.selectbox('เลือก Window ที่ต้องการดูรายละเอียด:', options=df['window_number'], format_func=lambda x: f"Window #{x} (Timeline: {df.loc[df['window_number'] == x, 'timeline'].iloc[0]})")
                    if selected_window:
                        window_data = df[df['window_number'] == selected_window].iloc[0]
                        st.markdown(f"**รายละเอียดของ Window #{selected_window}**")
                        w_cols = st.columns(3)
                        w_cols[0].metric("Net Profit", f"${window_data['max_net']:.2f}")
                        w_cols[1].metric("Best Seed", f"{window_data['best_seed']}")
                        w_cols[2].metric("Price Change", f"{window_data['price_change_pct']:.2f}%")
                        st.markdown(f"**Action Sequence:**")
                        st.code(window_data['action_sequence'], language='json')
                
                def safe_literal_eval(val):
                    if pd.isna(val): return []
                    if isinstance(val, list): return val
                    if isinstance(val, str) and val.strip().startswith('[') and val.strip().endswith(']'):
                        try: return ast.literal_eval(val)
                        except: return []
                    return []

                with stitched_dna_tab:
                    st.subheader("ทดสอบกลยุทธ์จาก 'Stitched' DNA")
                    st.markdown("จำลองการเทรดจริงโดยนำ **`action_sequence`** จากแต่ละ Window มา 'เย็บ' ต่อกัน และเปรียบเทียบกับ Benchmark")

                    df['action_sequence_list'] = [safe_literal_eval(val) for val in df['action_sequence']]
                    
                    df_sorted = df.sort_values('window_number')
                    stitched_actions = [action for seq in df_sorted['action_sequence_list'] for action in seq]
                    
                    dna_cols = st.columns(2)
                    
                    # <--- แก้ไขจุดที่ 2: ใช้ st.session_state.test_ticker เป็นค่าเริ่มต้น
                    stitch_ticker = dna_cols[0].text_input(
                        "Ticker สำหรับจำลอง", 
                        value=st.session_state.test_ticker, # <--- แก้ไข
                        key='stitch_ticker_input'
                    )
                    stitch_start_date = dna_cols[1].date_input("วันที่เริ่มต้นจำลอง", value=datetime(2024, 1, 1), key='stitch_date_input')

                    if st.button("🧬 เริ่มการวิเคราะห์ Stitched DNA แบบเปรียบเทียบ", type="primary", key='stitch_dna_btn'):
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
                                    final_actions_dna = stitched_actions[:n_total]
                                    _, sumusd_dna, _, _, _, refer_dna = calculate_optimized(final_actions_dna, prices[:len(final_actions_dna)])
                                    stitched_net = sumusd_dna - refer_dna - sumusd_dna[0]
                                    max_actions = get_max_action(prices)
                                    _, sumusd_max, _, _, _, refer_max = calculate_optimized(max_actions, prices)
                                    max_net = sumusd_max - refer_max - sumusd_max[0]
                                    min_actions = np.ones(n_total, dtype=int).tolist()
                                    _, sumusd_min, _, _, _, refer_min = calculate_optimized(min_actions, prices)
                                    min_net = sumusd_min - refer_min - sumusd_min[0]
                                    plot_len = len(stitched_net)
                                    plot_df = pd.DataFrame({
                                        'Max Performance (Perfect)': max_net[:plot_len],
                                        'Stitched DNA Strategy': stitched_net,
                                        'Min Performance (Rebalance Daily)': min_net[:plot_len]
                                    }, index=sim_data.index[:plot_len])
                                    st.subheader("Performance Comparison (Net Profit)")
                                    st.line_chart(plot_df)
                                    st.subheader("สรุปผลลัพธ์สุดท้าย (Final Net Profit)")
                                    metric_cols = st.columns(3)
                                    metric_cols[0].metric("Max Performance (at DNA End)", f"${max_net[plot_len-1]:,.2f}")
                                    metric_cols[1].metric("Stitched DNA Strategy", f"${stitched_net[-1]:,.2f}", delta=f"{stitched_net[-1] - min_net[plot_len-1]:,.2f} vs Min", delta_color="normal")
                                    metric_cols[2].metric("Min Performance (at DNA End)", f"${min_net[plot_len-1]:,.2f}")

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการวิเคราะห์ข้อมูล: {e}")
            st.exception(e)



# --- ส่วนคำอธิบายท้ายหน้า ---
st.write("---")
st.write("📖 คำอธิบายวิธีการทำงาน")
with st.expander("🔍 Best Seed Sliding Window คืออะไร?"):
    st.write("""
    **Best Seed Sliding Window** เป็นเทคนิคการหา action sequence ที่ดีที่สุดโดย:
    1. **แบ่งข้อมูล**: แบ่งข้อมูลราคาออกเป็นช่วง ๆ (windows) ตามขนาดที่กำหนด
    2. **ค้นหา Seed**: ในแต่ละ window ทำการสุ่ม seed หลาย ๆ ตัวและคำนวณผลกำไร
    3. **เลือก Best Seed**: เลือก seed ที่ให้ผลกำไรสูงสุดในแต่ละ window
    4. **รวม Actions**: นำ action sequences จากแต่ละ window มาต่อกันเป็น sequence สุดท้าย
    """)
with st.expander("⚙️ การตั้งค่าพารามิเตอร์"):
    st.write("""
    **Window Size (ขนาด Window):**
    - ขนาดเล็ก (10-20 วัน): ปรับตัวเร็ว แต่อาจมีความผันผวนสูง
    - ขนาดกลาง (20-50 วัน): สมดุลระหว่างการปรับตัวและเสถียรภาพ
    **จำนวน Seeds ต่อ Window:**
    - น้อย (100-500): เร็วแต่อาจไม่ได้ seed ที่ดีที่สุด
    - มาก (2000+): ได้ผลลัพธ์ดีแต่ใช้เวลานาน
    """)
with st.expander("⚡ การปรับปรุงความเร็ว"):
    st.write("""
    **การปรับปรุงที่ทำ:**
    1. **Parallel Processing**: ใช้ ThreadPoolExecutor เพื่อประเมิน seeds หลายตัวพร้อมกัน
    2. **Caching**: ใช้ @lru_cache สำหรับฟังก์ชันคำนวณและ @st.cache_data สำหรับข้อมูล ticker
    3. **Vectorization**: ใช้ NumPy operations แทน Python loops
    """)
