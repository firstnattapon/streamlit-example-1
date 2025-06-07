import pandas as pd
import numpy as np
from numba import njit
import yfinance as yf
import streamlit as st

st.set_page_config(page_title="Best Seed Sliding Window", page_icon="🎯", layout="wide")

@njit(fastmath=True)
def calculate_optimized(action_list, price_list, fix=1500):
    # แปลงเป็น numpy array และกำหนด dtype ให้ชัดเจน
    action_array = np.asarray(action_list, dtype=np.int32)
    action_array[0] = 1
    price_array = np.asarray(price_list, dtype=np.float64)
    n = len(action_array)
    
    # Pre-allocate arrays ด้วย dtype ที่เหมาะสม
    amount = np.empty(n, dtype=np.float64)
    buffer = np.zeros(n, dtype=np.float64)
    cash = np.empty(n, dtype=np.float64)
    asset_value = np.empty(n, dtype=np.float64)
    sumusd = np.empty(n, dtype=np.float64)
    
    # คำนวณค่าเริ่มต้นที่ index 0
    initial_price = price_array[0]
    amount[0] = fix / initial_price
    cash[0] = fix
    asset_value[0] = amount[0] * initial_price
    sumusd[0] = cash[0] + asset_value[0]
    
    # คำนวณ refer ทั้งหมดในครั้งเดียว (แยกออกมาจาก loop หลัก)
    # เปลี่ยนเป็นสูตรที่ 2: refer = -fix * ln(t0/tn)
    refer = -fix * np.log(initial_price / price_array)
    
    # Main loop with minimal operations
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

def find_best_seed_sliding_window(price_list, ticker_data_with_dates=None, window_size=30, num_seeds_to_try=1000, progress_bar=None):
    """
    ค้นหาลำดับ action ที่ดีที่สุดโดยการหา seed ที่ให้ผลกำไรสูงสุดในแต่ละช่วงเวลา (sliding window)
    """
    prices = np.asarray(price_list)
    n = len(prices)
    final_actions = np.array([], dtype=int)
    window_details = []
    
    num_windows = (n + window_size - 1) // window_size
    
    # วนลูปตามข้อมูลราคาในแต่ละช่วง (window)
    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        window_len = len(prices_window)

        if window_len == 0:
            continue

        # ดึงวันที่สำหรับแสดง timeline
        if ticker_data_with_dates is not None:
            start_date = ticker_data_with_dates.index[start_index].strftime('%Y-%m-%d')
            end_date = ticker_data_with_dates.index[end_index-1].strftime('%Y-%m-%d')
            timeline_info = f"{start_date} ถึง {end_date}"
        else:
            timeline_info = f"Index {start_index} ถึง {end_index-1}"

        best_seed_for_window = -1
        max_net_for_window = -np.inf  

        # --- ค้นหา Seed ที่ดีที่สุดสำหรับ Window ปัจจุบัน ---
        random_seeds = np.random.randint(0, 10_000_000, size=num_seeds_to_try)

        for seed in random_seeds:
            rng = np.random.default_rng(seed)
            actions_window = rng.integers(0, 2, size=window_len)
            actions_window[0] = 1  # บังคับให้ action แรกของ window เป็น 1 เสมอ

            # ประเมินผลเฉพาะใน window นี้
            if window_len < 2:
                final_net = 0
            else:
                _, sumusd, _, _, _, refer = calculate_optimized(actions_window.tolist(), prices_window.tolist())
                initial_capital = sumusd[0]
                net = sumusd - refer - initial_capital
                final_net = net[-1]

            if final_net > max_net_for_window:
                max_net_for_window = final_net
                best_seed_for_window = seed

        # --- สร้าง Action ที่ดีที่สุดสำหรับ Window นี้จาก Seed ที่พบ ---
        rng_best = np.random.default_rng(best_seed_for_window)
        best_actions_for_window = rng_best.integers(0, 2, size=window_len)
        best_actions_for_window[0] = 1

        # เก็บรายละเอียดของ window
        window_detail = {
            'window_number': i + 1,
            'timeline': timeline_info,
            'start_index': start_index,
            'end_index': end_index - 1,
            'window_size': window_len,
            'best_seed': best_seed_for_window,
            'max_net': round(max_net_for_window, 2),
            'start_price': round(prices_window[0], 2),
            'end_price': round(prices_window[-1], 2),
            'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2),
            'action_count': int(np.sum(best_actions_for_window)),
            'action_sequence': best_actions_for_window.tolist()
        }
        window_details.append(window_detail)

        # นำ action ที่ดีที่สุดของ window นี้ไปต่อท้าย action หลัก
        final_actions = np.concatenate((final_actions, best_actions_for_window))
        
        # Update progress bar if provided
        if progress_bar:
            progress_bar.progress((i + 1) / num_windows)

    return final_actions, window_details

def get_max_action(price_list, fix=1500):
    """
    คำนวณหาลำดับ action (0, 1) ที่ให้ผลตอบแทนสูงสุดทางทฤษฎี
    โดยใช้ Dynamic Programming ร่วมกับการย้อนรอย (Backtracking)
    """
    prices = np.asarray(price_list, dtype=np.float64)
    n = len(prices)

    if n < 2:
        return np.ones(n, dtype=int)

    dp = np.zeros(n, dtype=np.float64)
    path = np.zeros(n, dtype=int) 
    
    initial_capital = float(fix * 2)
    dp[0] = initial_capital
    
    for i in range(1, n):
        max_prev_sumusd = 0
        best_j = 0
        
        for j in range(i):
            profit_from_j_to_i = fix * ((prices[i] / prices[j]) - 1)
            current_sumusd = dp[j] + profit_from_j_to_i
            
            if current_sumusd > max_prev_sumusd:
                max_prev_sumusd = current_sumusd
                best_j = j
        
        dp[i] = max_prev_sumusd
        path[i] = best_j

    actions = np.zeros(n, dtype=int)
    last_action_day = np.argmax(dp)
    
    current_day = last_action_day
    while current_day > 0:
        actions[current_day] = 1
        current_day = path[current_day]
        
    actions[0] = 1
    
    return actions

def Limit_fx(Ticker='', act=-1, window_size=30, num_seeds_to_try=1000):
    filter_date = '2023-01-01 12:00:00+07:00'
    tickerData = yf.Ticker(Ticker)
    tickerData = tickerData.history(period='max')[['Close']]
    tickerData.index = tickerData.index.tz_convert(tz='Asia/Bangkok')
    tickerData = tickerData[tickerData.index >= filter_date]
    
    prices = np.array(tickerData.Close.values, dtype=np.float64)

    if act == -1:  # min
        actions = np.array(np.ones(len(prices)), dtype=np.int64)
    elif act == -2:  # max  
        actions = get_max_action(prices)
    elif act == -3:  # best_seed sliding window
        progress_bar = st.progress(0)
        actions, window_details = find_best_seed_sliding_window(
            prices, tickerData, window_size=window_size, 
            num_seeds_to_try=num_seeds_to_try, progress_bar=progress_bar
        )
        st.session_state[f'window_details_{Ticker}'] = window_details
    else:
        rng = np.random.default_rng(act)
        actions = rng.integers(0, 2, len(prices))
    
    buffer, sumusd, cash, asset_value, amount, refer = calculate_optimized(actions, prices)
    
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

def plot_comparison(Ticker='', act=-1, window_size=30, num_seeds_to_try=1000):
    all = []
    all_id = []
    
    # min
    all.append(Limit_fx(Ticker, act=-1).net)
    all_id.append('min')

    # fx (best_seed or other)
    if act == -3:  # best_seed
        all.append(Limit_fx(Ticker, act=act, window_size=window_size, num_seeds_to_try=num_seeds_to_try).net)
        all_id.append('best_seed')
    else:
        all.append(Limit_fx(Ticker, act=act).net)
        all_id.append('fx_{}'.format(act))

    # max
    all.append(Limit_fx(Ticker, act=-2).net)
    all_id.append('max')
    
    chart_data = pd.DataFrame(np.array(all).T, columns=np.array(all_id))
    return chart_data

# Main Streamlit App
def main():
    st.title("🎯 Best Seed Sliding Window Tester")
    st.write("เครื่องมือทดสอบการหา Best Seed ด้วยวิธี Sliding Window สำหรับการเทรด")
    
    # สร้าง tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "⚙️ การตั้งค่า", 
        "📊 กราฟเปรียบเทียบ", 
        "📋 รายละเอียด Windows", 
        "🌱 Seeds ที่เลือกใช้",
        "📈 การวิเคราะห์",
        "📖 คำอธิบาย"
    ])
    
    # Tab 1: การตั้งค่า
    with tab1:
        st.header("⚙️ การตั้งค่าพารามิเตอร์")
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_ticker = st.selectbox(
                "🏢 เลือก Ticker สำหรับทดสอบ", 
                ['FFWM', 'NEGG', 'RIVN', 'APLS', 'NVTS', 'QXO', 'RXRX'],
                index=0
            )
            
            window_size = st.number_input(
                "📅 ขนาด Window (วัน)", 
                min_value=10, max_value=100, value=30,
                help="ขนาดของแต่ละช่วงเวลาที่จะค้นหา Best Seed"
            )
        
        with col2:
            num_seeds = st.number_input(
                "🌱 จำนวน Seeds ต่อ Window", 
                min_value=100, max_value=5000, value=1000,
                help="จำนวน random seeds ที่จะทดสอบในแต่ละ window"
            )
            
            st.write("")
            st.write("")
            
            start_test = st.button("🚀 เริ่มทดสอบ Best Seed", type="primary", use_container_width=True)
        
        # แสดงข้อมูลการตั้งค่าปัจจุบัน
        st.write("---")
        st.write("📋 **ข้อมูลการตั้งค่าปัจจุบัน:**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ticker", test_ticker)
        with col2:
            st.metric("Window Size", f"{window_size} วัน")
        with col3:
            st.metric("Seeds per Window", f"{num_seeds:,}")
        
        # เมื่อกดปุ่มเริ่มทดสอบ
        if start_test:
            st.write("---")
            st.write(f"🔄 **กำลังประมวลผล {test_ticker}...**")
            
            # บันทึกการตั้งค่าใน session state
            st.session_state['test_ticker'] = test_ticker
            st.session_state['window_size'] = window_size
            st.session_state['num_seeds'] = num_seeds
            st.session_state['test_completed'] = False
            
            # รันการทดสอบ
            try:
                chart_data = plot_comparison(
                    Ticker=test_ticker, 
                    act=-3, 
                    window_size=window_size, 
                    num_seeds_to_try=num_seeds
                )
                
                # บันทึกผลลัพธ์
                st.session_state['chart_data'] = chart_data
                st.session_state['test_completed'] = True
                
                st.success("✅ การทดสอบเสร็จสิ้น! ดูผลลัพธ์ใน tabs อื่น ๆ")
                
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
    
    # Tab 2: กราฟเปรียบเทียบ
    with tab2:
        st.header("📊 กราฟเปรียบเทียบ")
        
        if 'test_completed' in st.session_state and st.session_state['test_completed']:
            chart_data = st.session_state.get('chart_data')
            test_ticker = st.session_state.get('test_ticker')
            
            if chart_data is not None:
                st.write(f"**📈 Refer_Log Comparison สำหรับ {test_ticker}**")
                st.line_chart(chart_data)
                
                # แสดง Burn Cash
                df_plot = Limit_fx(test_ticker, act=-1)
                df_plot = df_plot[['buffer']].cumsum()
                st.write("**💰 Burn_Cash**")
                st.line_chart(df_plot)
                
                # แสดงตารางข้อมูลดิบ
                st.write("**📋 ข้อมูลดิบ**")
                raw_data = Limit_fx(test_ticker, act=-1)
                st.dataframe(raw_data, use_container_width=True)
            else:
                st.warning("⚠️ ไม่พบข้อมูลกราฟ กรุณาทำการทดสอบในแท็บ 'การตั้งค่า' ก่อน")
        else:
            st.info("ℹ️ กรุณาทำการทดสอบในแท็บ 'การตั้งค่า' ก่อนเพื่อดูกราฟเปรียบเทียบ")
    
    # Tab 3: รายละเอียด Windows
    with tab3:
        st.header("📋 รายละเอียดแต่ละ Window")
        
        if 'test_completed' in st.session_state and st.session_state['test_completed']:
            test_ticker = st.session_state.get('test_ticker')
            
            if f'window_details_{test_ticker}' in st.session_state:
                window_details = st.session_state[f'window_details_{test_ticker}']
                
                # แสดงสรุปผลรวม
                st.write("📈 **สรุปผลการค้นหา Best Seed**")
                total_windows = len(window_details)
                total_actions = sum([w['action_count'] for w in window_details])
                total_net = sum([w['max_net'] for w in window_details])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Windows", total_windows)
                with col2:
                    st.metric("Total Actions", f"{total_actions}")
                with col3:
                    st.metric("Total Net (Sum)", f"{total_net:.2f}")
                
                st.write("---")
                
                # แสดงตารางรายละเอียด
                df_details = pd.DataFrame(window_details)
                df_display = df_details[['window_number', 'timeline', 'best_seed', 'max_net', 
                                       'price_change_pct', 'action_count', 'window_size']].copy()
                df_display.columns = ['Window', 'Timeline', 'Best Seed', 'Net Profit', 
                                    'Price Change %', 'Actions', 'Window Size']
                st.dataframe(df_display, use_container_width=True)
                
                # ดาวน์โหลดผลลัพธ์
                st.write("---")
                st.write("💾 **ดาวน์โหลดผลลัพธ์**")
                csv = df_details.to_csv(index=False)
                st.download_button(
                    label="📥 ดาวน์โหลด Window Details (CSV)",
                    data=csv,
                    file_name=f'best_seed_results_{test_ticker}_{st.session_state.get("window_size")}w_{st.session_state.get("num_seeds")}s.csv',
                    mime='text/csv'
                )
            else:
                st.warning("⚠️ ไม่พบรายละเอียด Windows กรุณาทำการทดสอบในแท็บ 'การตั้งค่า' ก่อน")
        else:
            st.info("ℹ️ กรุณาทำการทดสอบในแท็บ 'การตั้งค่า' ก่อนเพื่อดูรายละเอียด Windows")
    
    # Tab 4: Seeds ที่เลือกใช้
    with tab4:
        st.header("🌱 Seeds ที่เลือกใช้ในแต่ละ Window")
        
        if 'test_completed' in st.session_state and st.session_state['test_completed']:
            test_ticker = st.session_state.get('test_ticker')
            
            if f'window_details_{test_ticker}' in st.session_state:
                window_details = st.session_state[f'window_details_{test_ticker}']
                df_windows = pd.DataFrame(window_details)
                
                # แสดง Seeds ที่ใช้
                seeds_df = df_windows[['window_number', 'timeline', 'best_seed', 'max_net', 'action_count']].copy()
                seeds_df.columns = ['Window', 'Timeline', 'Selected Seed', 'Net Profit', 'Actions']
                st.dataframe(seeds_df, use_container_width=True)
                
                # สถิติของ Seeds
                st.write("---")
                st.write("📊 **สถิติ Seeds**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Seed ต่ำสุด", f"{df_windows['best_seed'].min():,}")
                with col2:
                    st.metric("Seed สูงสุด", f"{df_windows['best_seed'].max():,}")
                with col3:
                    st.metric("Seed เฉลี่ย", f"{df_windows['best_seed'].mean():.0f}")
                
                # กราฟแสดง Seeds
                st.write("📈 **กราฟ Seeds ที่เลือกใช้**")
                st.bar_chart(df_windows.set_index('window_number')['best_seed'])
            else:
                st.warning("⚠️ ไม่พบข้อมูล Seeds กรุณาทำการทดสอบในแท็บ 'การตั้งค่า' ก่อน")
        else:
            st.info("ℹ️ กรุณาทำการทดสอบในแท็บ 'การตั้งค่า' ก่อนเพื่อดู Seeds ที่เลือกใช้")
    
    # Tab 5: การวิเคราะห์
    with tab5:
        st.header("📈 การวิเคราะห์เพิ่มเติม")
        
        if 'test_completed' in st.session_state and st.session_state['test_completed']:
            test_ticker = st.session_state.get('test_ticker')
            
            if f'window_details_{test_ticker}' in st.session_state:
                window_details = st.session_state[f'window_details_{test_ticker}']
                df_windows = pd.DataFrame(window_details)
                
                # กราฟแสดง Net Profit ของแต่ละ Window
                st.write("📊 **Net Profit แต่ละ Window**")
                st.bar_chart(df_windows.set_index('window_number')['max_net'])
                
                # กราฟแสดง Price Change % ของแต่ละ Window
                st.write("📈 **Price Change % แต่ละ Window**")
                st.bar_chart(df_windows.set_index('window_number')['price_change_pct'])
                
                # กราฟแสดง Action Count ของแต่ละ Window
                st.write("🎯 **Action Count แต่ละ Window**")
                st.bar_chart(df_windows.set_index('window_number')['action_count'])
                
                # สถิติเพิ่มเติม
                st.write("---")
                st.write("📊 **สถิติโดยรวม**")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Net เฉลี่ย", f"{df_windows['max_net'].mean():.2f}")
                with col2:
                    st.metric("Net รวม", f"{df_windows['max_net'].sum():.2f}")
                with col3:
                    st.metric("Actions เฉลี่ย", f"{df_windows['action_count'].mean():.1f}")
                with col4:
                    st.metric("Price Change เฉลี่ย", f"{df_windows['price_change_pct'].mean():.2f}%")
            else:
                st.warning("⚠️ ไม่พบข้อมูลการวิเคราะห์ กรุณาทำการทดสอบในแท็บ 'การตั้งค่า' ก่อน")
        else:
            st.info("ℹ️ กรุณาทำการทดสอบในแท็บ 'การตั้งค่า' ก่อนเพื่อดูการวิเคราะห์")
    
    # Tab 6: คำอธิบาย
    with tab6:
        st.header("📖 คำอธิบายวิธีการทำงาน")
        
        with st.expander("🔍 Best Seed Sliding Window คืออะไร?", expanded=True):
            st.write("""
            **Best Seed Sliding Window** เป็นเทคนิคการหา action sequence ที่ดีที่สุดโดย:
            
            1. **แบ่งข้อมูล**: แบ่งข้อมูลราคาออกเป็นช่วง ๆ (windows) ตามขนาดที่กำหนด
            2. **ค้นหา Seed**: ในแต่ละ window ทำการสุ่ม seed หลาย ๆ ตัวและคำนวณผลกำไร
            3. **เลือก Best Seed**: เลือก seed ที่ให้ผลกำไรสูงสุดในแต่ละ window
            4. **รวม Actions**: นำ action sequences จากแต่ละ window มาต่อกันเป็น sequence สุดท้าย
            
            **ข้อดี:**
            - ปรับตัวได้ตามสภาวะตลาดในแต่ละช่วงเวลา
            - ลดความเสี่ยงจากการใช้ seed เดียวตลอดทั้งช่วง
            - สามารถควบคุมพารามิเตอร์ได้ (window size, จำนวน seeds)
            """)
        
        with st.expander("⚙️ การตั้งค่าพารามิเตอร์"):
            st.write("""
            **Window Size (ขนาด Window):**
            - ขนาดเล็ก (10-20 วัน): ปรับตัวเร็ว แต่อาจมีความผันผวนสูง
            - ขนาดกลาง (20-50 วัน): สมดุลระหว่างการปรับตัวและเสถียรภาพ
            - ขนาดใหญ่ (50+ วัน): เสถียรแต่ปรับตัวช้า
            
            **จำนวน Seeds ต่อ Window:**
            - น้อย (100-500): เร็วแต่อาจไม่ได้ seed ที่ดีที่สุด
            - กลาง (500-2000): สมดุลระหว่างเวลาและคุณภาพ
            - มาก (2000+): ได้ผลลัพธ์ดีแต่ใช้เวลานาน
            """)
        
        with st.expander("📊 การอ่านผลลัพธ์"):
            st.write("""
            **กราฟเปรียบเทียบ:**
            - **min**: ผลลัพธ์จากการ action ทุกวัน (baseline
