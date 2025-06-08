with stitched_dna_tab:
    st.subheader("ทดสอบกลยุทธ์จาก 'Stitched' DNA")
    st.markdown("จำลองการเทรดจริงโดยใช้ `best_seed` ที่ได้จากแต่ละ Window มา 'เย็บ' ต่อกัน และเปรียบเทียบกับ Benchmark (Min/Max Performance)")

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

    if st.button("🧬 เริ่มการวิเคราะห์ Stitched DNA แบบเปรียบเทียบ", type="primary"):
        try:
            seeds_for_ticker = ast.literal_eval(seed_list_input)
            if not isinstance(seeds_for_ticker, list) or not seeds_for_ticker:
                st.error("รูปแบบ Seed List ไม่ถูกต้อง หรือเป็น List ว่าง")
            else:
                with st.spinner(f"กำลังจำลองกลยุทธ์สำหรับ {stitch_ticker} และคำนวณ Benchmark..."):
                    # 1. Fetch full price data for simulation
                    sim_data = get_ticker_data(stitch_ticker, str(stitch_start_date), str(datetime.now()))
                    if sim_data.empty:
                        st.error("ไม่สามารถดึงข้อมูลสำหรับจำลองได้")
                    else:
                        prices = sim_data['Close'].tolist()
                        n_total = len(prices)
                        
                        # --- 2. Calculate for all 3 scenarios ---

                        # a) Stitched DNA
                        window_size_sim = int(n_total / len(seeds_for_ticker)) if len(seeds_for_ticker) > 0 else 30
                        final_actions, seed_index = [], 0
                        for i in range(0, n_total, window_size_sim):
                            current_seed = seeds_for_ticker[min(seed_index, len(seeds_for_ticker)-1)]
                            rng = np.random.default_rng(current_seed)
                            win_len = min(window_size_sim, n_total - i)
                            if win_len > 0:
                                actions_for_window = rng.integers(0, 2, win_len).tolist()
                                if actions_for_window:
                                    actions_for_window[0] = 1
                                final_actions.extend(actions_for_window)
                            seed_index += 1
                        
                        _, sumusd_dna, _, _, _, refer_dna = calculate_optimized(final_actions, prices)
                        stitched_net = sumusd_dna - refer_dna - sumusd_dna[0]

                        # b) Max Performance (Perfect Foresight)
                        max_actions = get_max_action_vectorized(prices)
                        _, sumusd_max, _, _, _, refer_max = calculate_optimized(max_actions, prices)
                        max_net = sumusd_max - refer_max - sumusd_max[0]

                        # c) Min Performance (Buy & Hold) -> Net profit is 0 by definition
                        min_net = np.zeros(n_total)
                        
                        # --- 3. Plotting ---
                        plot_df = pd.DataFrame({
                            'Max Performance (Perfect)': max_net,
                            'Stitched DNA Strategy': stitched_net,
                            'Min Performance (Buy & Hold)': min_net
                        }, index=sim_data.index)
                        
                        st.subheader("Performance Comparison (Net Profit)")
                        st.line_chart(plot_df)

                        # --- 4. Metrics ---
                        st.subheader("สรุปผลลัพธ์สุดท้าย (Final Net Profit)")
                        metric_cols = st.columns(3)
                        metric_cols[0].metric("Max Performance", f"${max_net[-1]:,.2f}", help="ผลตอบแทนสูงสุดตามทฤษฎี (Perfect Foresight)")
                        metric_cols[1].metric("Stitched DNA Strategy", f"${stitched_net[-1]:,.2f}", delta=f"{stitched_net[-1]:,.2f}", delta_color="normal")
                        metric_cols[2].metric("Buy & Hold", f"${min_net[-1]:,.2f}", help="ผลตอบแทนของกลยุทธ์ Buy and Hold (เป็นเส้นฐาน)")

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการวิเคราะห์: {e}")
            st.exception(e) # แสดง traceback เพื่อ debug
