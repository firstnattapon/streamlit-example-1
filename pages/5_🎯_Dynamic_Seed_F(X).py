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

    Args:
        price_list (list or np.array): รายการราคาของสินทรัพย์
        ticker_data_with_dates: DataFrame ที่มี index เป็นวันที่สำหรับแสดง timeline
        window_size (int): ขนาดของแต่ละช่วงเวลาที่จะค้นหา (เช่น 30 วัน)
        num_seeds_to_try (int): จำนวน seed ที่จะลองสุ่มในแต่ละ window
        progress_bar: (Optional) Streamlit progress bar object to update.

    Returns:
        tuple: (final_actions, window_details) 
               - final_actions: ลำดับ action ที่ดีที่สุด
               - window_details: รายละเอียดของแต่ละ window
    """
    prices = np.asarray(price_list)
    n = len(prices)
    final_actions = np.array([], dtype=int)
    window_details = []
    
    num_windows = (n + window_size - 1) // window_size # Calculate total number of windows
    
    st.write("🔍 **เริ่มต้นการค้นหา Best Seed ด้วย Sliding Window**")
    st.write(f"📊 ข้อมูลทั้งหมด: {n} วัน | ขนาด Window: {window_size} วัน | จำนวน Windows: {num_windows}")
    st.write("---")
    
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
        # ใช้ -np.inf เพื่อให้แน่ใจว่าค่า net แรกที่เจอมักจะสูงกว่าเสมอ
        max_net_for_window = -np.inf  

        # --- ค้นหา Seed ที่ดีที่สุดสำหรับ Window ปัจจุบัน ---
        # ใช้ np.random.randint เพื่อให้ได้ seed ที่หลากหลายมากขึ้น
        random_seeds = np.random.randint(0, 10_000_000, size=num_seeds_to_try)

        for seed in random_seeds:
            rng = np.random.default_rng(seed)
            actions_window = rng.integers(0, 2, size=window_len)
            actions_window[0] = 1  # บังคับให้ action แรกของ window เป็น 1 เสมอ

            # ประเมินผลเฉพาะใน window นี้ (เริ่มนับ 1 ใหม่)
            if window_len < 2:
                # ถ้าข้อมูลไม่พอคำนวณ net ให้ใช้ action ที่สร้างได้เลย
                final_net = 0
            else:
                _, sumusd, _, _, _, refer = calculate_optimized(actions_window.tolist(), prices_window.tolist())
                initial_capital = sumusd[0]
                net = sumusd - refer - initial_capital
                final_net = net[-1] # เอาผลกำไร ณ วันสุดท้ายของ window

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

        # แสดงผลแต่ละ window
        st.write(f"**🎯 Window {i+1}/{num_windows}** | {timeline_info}")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Best Seed", f"{best_seed_for_window:,}")
        with col2:
            st.metric("Net Profit", f"{max_net_for_window:.2f}")
        with col3:
            st.metric("Price Change", f"{window_detail['price_change_pct']:.2f}%")
        with col4:
            st.metric("Actions Count", f"{window_detail['action_count']}/{window_len}")

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
        return np.ones(n, dtype=int) # ถ้ามีข้อมูลไม่พอ ก็ action ไปเลย

    # --- ส่วนที่ 1: คำนวณไปข้างหน้า (เหมือนเดิม แต่เพิ่มการเก็บ path) ---
    
    dp = np.zeros(n, dtype=np.float64)
    # path[i] จะเก็บ index 'j' ของ action ครั้งก่อนหน้าที่ดีที่สุดสำหรับวัน 'i'
    path = np.zeros(n, dtype=int) 
    
    initial_capital = float(fix * 2)
    dp[0] = initial_capital
    
    for i in range(1, n):
        max_prev_sumusd = 0
        best_j = 0 # ตัวแปรสำหรับเก็บ j ที่ดีที่สุดสำหรับ i นี้
        
        for j in range(i):
            profit_from_j_to_i = fix * ((prices[i] / prices[j]) - 1)
            current_sumusd = dp[j] + profit_from_j_to_i
            
            if current_sumusd > max_prev_sumusd:
                max_prev_sumusd = current_sumusd
                best_j = j # เจอทางเลือกที่ดีกว่า ก็จำไว้ว่ามันมาจาก j ไหน
        
        dp[i] = max_prev_sumusd
        path[i] = best_j # บันทึกเส้นทางที่ดีที่สุดสำหรับวัน i

    # --- ส่วนที่ 2: การย้อนรอย (Backtracking) เพื่อสร้าง action array ---

    actions = np.zeros(n, dtype=int)
    
    # 1. หาจุดสิ้นสุดของเส้นทาง (วันที่ให้ sumusd สูงที่สุด)
    last_action_day = np.argmax(dp)
    
    # 2. เริ่มย้อนรอยจากจุดสิ้นสุดกลับไปหาจุดเริ่มต้น
    current_day = last_action_day
    while current_day > 0:
        # ทุกจุดที่เราเหยียบในการย้อนรอย คือวันที่ควรมี action
        actions[current_day] = 1
        # กระโดดกลับไปยัง action ครั้งก่อนหน้า
        current_day = path[current_day]
        
    # 3. กำหนดให้ action แรกสุดเป็น 1 เสมอ (เป็นจุดเริ่มต้นของทุกเส้นทาง)
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
        
        # แสดงสรุปผลรวม
        st.write("---")
        st.write("📈 **สรุปผลการค้นหา Best Seed**")
        total_windows = len(window_details)
        total_actions = sum([w['action_count'] for w in window_details])
        total_net = sum([w['max_net'] for w in window_details])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Windows", total_windows)
        with col2:
            st.metric("Total Actions", f"{total_actions}/{len(actions)}")
        with col3:
            st.metric("Total Net (Sum)", f"{total_net:.2f}")
        
        # แสดงตารางรายละเอียด
        st.write("📋 **รายละเอียดแต่ละ Window**")
        df_details = pd.DataFrame(window_details)
        df_display = df_details[['window_number', 'timeline', 'best_seed', 'max_net', 
                               'price_change_pct', 'action_count', 'window_size']].copy()
        df_display.columns = ['Window', 'Timeline', 'Best Seed', 'Net Profit', 
                            'Price Change %', 'Actions', 'Window Size']
        st.dataframe(df_display, use_container_width=True)
        
        # เก็บ window_details ใน session state เพื่อใช้ในการแสดงผลอื่น ๆ
        st.session_state[f'window_details_{Ticker}'] = window_details
      
    else:
        rng = np.random.default_rng(act)
        actions = rng.integers(0, 2, len(prices))
    
    buffer, sumusd, cash, asset_value, amount, refer = calculate_optimized(actions, prices)
    
    # ใช้ sumusd[0] แทนค่าคงที่ 3000
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
    
    st.write('📊 **Refer_Log Comparison**')
    st.line_chart(chart_data)

    df_plot = Limit_fx(Ticker, act=-1)
    df_plot = df_plot[['buffer']].cumsum()
    st.write('💰 **Burn_Cash**')
    st.line_chart(df_plot)
    st.write(Limit_fx(Ticker, act=-1))

# Main Streamlit App
def main():
    tab1, tab2, = st.tabs([ "การตั้งค่า", "ทดสอบ" ])
    with tab1:

        st.write("🎯 Best Seed Sliding Window Tester")
        st.write("เครื่องมือทดสอบการหา Best Seed ด้วยวิธี Sliding Window สำหรับการเทรด")
        
        # Sidebar สำหรับการตั้งค่า
        st.header("⚙️ การตั้งค่า")
        
        # เลือก ticker สำหรับทดสอบ
        test_ticker = st.selectbox(
            "เลือก Ticker สำหรับทดสอบ", 
            ['FFWM', 'NEGG', 'RIVN', 'APLS', 'NVTS', 'QXO', 'RXRX']
        )
        
        # ตั้งค่าพารามิเตอร์
        window_size = st.number_input(
            "ขนาด Window (วัน)", 
            min_value=10, max_value=100, value=30
        )
        
        num_seeds = st.number_input(
            "จำนวน Seeds ต่อ Window", 
            min_value=100, max_value=10000, value=1000
        )

    with tab2:
        # ปุ่มเริ่มทดสอบ
        if st.button("🚀 เริ่มทดสอบ Best Seed", type="primary"):
            st.write(f"กำลังทดสอบ Best Seed สำหรับ **{test_ticker}** 📊")
            st.write(f"⚙️ พารามิเตอร์: Window Size = {window_size}, Seeds per Window = {num_seeds}")
            st.write("---")
            
            # เรียกใช้ plot comparison
            plot_comparison(Ticker=test_ticker, act=-3, window_size=window_size, num_seeds_to_try=num_seeds)
            
            # แสดงข้อมูลเพิ่มเติมถ้ามี
            if f'window_details_{test_ticker}' in st.session_state:
                st.write("---")
                st.write("🔍 **การวิเคราะห์เพิ่มเติม**")
                
                window_details = st.session_state[f'window_details_{test_ticker}']
                
                # กราฟแสดง Net Profit ของแต่ละ Window
                df_windows = pd.DataFrame(window_details)
                st.write("📊 **Net Profit แต่ละ Window**")
                st.bar_chart(df_windows.set_index('window_number')['max_net'])
                
                # กราฟแสดง Price Change % ของแต่ละ Window
                st.write("📈 **Price Change % แต่ละ Window**")
                st.bar_chart(df_windows.set_index('window_number')['price_change_pct'])
                
                # แสดง Seeds ที่ใช้
                st.write("🌱 **Seeds ที่เลือกใช้ในแต่ละ Window**")
                seeds_df = df_windows[['window_number', 'timeline', 'best_seed', 'max_net']].copy()
                seeds_df.columns = ['Window', 'Timeline', 'Selected Seed', 'Net Profit']
                st.dataframe(seeds_df, use_container_width=True)
                
                # ดาวน์โหลดผลลัพธ์
                st.write("💾 **ดาวน์โหลดผลลัพธ์**")
                csv = df_windows.to_csv(index=False)
                st.download_button(
                    label="📥 ดาวน์โหลด Window Details (CSV)",
                    data=csv,
                    file_name=f'best_seed_results_{test_ticker}_{window_size}w_{num_seeds}s.csv',
                    mime='text/csv'
                )

    # คำอธิบายวิธีการทำงาน
    st.write("---")
    st.write("# 📖 คำอธิบายวิธีการทำงาน")

    
    with st.expander("🔍 Best Seed Sliding Window คืออะไร?"):
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

if __name__ == "__main__":
    main()
