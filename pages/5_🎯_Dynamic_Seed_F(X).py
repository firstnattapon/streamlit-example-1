import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
  
st.set_page_config(page_title="Best Seed Sliding Window", page_icon="🎯", layout="wide")

@lru_cache(maxsize=1000)
def calculate_optimized_cached(action_tuple, price_tuple, fix=1500):
    """
    ฟังก์ชันคำนวณที่ปรับปรุงแล้วพร้อม caching
    """
    action_array = np.asarray(action_tuple, dtype=np.int32)
    action_array[0] = 1
    price_array = np.asarray(price_tuple, dtype=np.float64)
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

def calculate_optimized(action_list, price_list, fix=1500):
    """
    Wrapper function สำหรับ cached version
    """
    return calculate_optimized_cached(tuple(action_list), tuple(price_list), fix)

def evaluate_seed_batch(seed_batch, prices_window, window_len):
    """
    ประเมิน seed หลายตัวพร้อมกัน
    """
    results = []
    
    for seed in seed_batch:
        try:
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
                final_net = net[-1] # เอาผลกำไร ณ วันสุดท้ายของ window

            results.append((seed, final_net))
                    
        except Exception as e:
            # ถ้ามี error ใน seed นี้ ให้ข้ามไป
            results.append((seed, -np.inf))
            continue
    
    return results

def find_best_seed_sliding_window_optimized(price_list, ticker_data_with_dates=None, window_size=30, num_seeds_to_try=1000, progress_bar=None, max_workers=4):
    """
    ค้นหาลำดับ action ที่ดีที่สุดโดยการหา seed ที่ให้ผลกำไรสูงสุดในแต่ละช่วงเวลา (sliding window)
    ปรับปรุงด้วย parallel processing และ vectorization
    """
    prices = np.asarray(price_list)
    n = len(prices)
    final_actions = np.array([], dtype=int)
    window_details = []
    
    num_windows = (n + window_size - 1) // window_size
    
    st.write("🔍 **เริ่มต้นการค้นหา Best Seed ด้วย Sliding Window (Optimized)**")
    st.write(f"📊 ข้อมูลทั้งหมด: {n} วัน | ขนาด Window: {window_size} วัน | จำนวน Windows: {num_windows}")
    st.write(f"⚡ ใช้ Parallel Processing: {max_workers} workers")
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
        max_net_for_window = -np.inf

        # --- ค้นหา Seed ที่ดีที่สุดสำหรับ Window ปัจจุบัน (Parallel Processing) ---
        random_seeds = np.arange(num_seeds_to_try)
        
        # แบ่ง seeds เป็น batches สำหรับ parallel processing
        batch_size = max(1, num_seeds_to_try // max_workers)
        seed_batches = [random_seeds[i:i+batch_size] for i in range(0, len(random_seeds), batch_size)]
        
        # ใช้ ThreadPoolExecutor สำหรับ parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_batch = {
                executor.submit(evaluate_seed_batch, batch, prices_window, window_len): batch 
                for batch in seed_batches
            }
            
            # Collect results
            all_results = []
            for future in as_completed(future_to_batch):
                batch_results = future.result()
                all_results.extend(batch_results)
        
        # หา seed ที่ดีที่สุดจากผลลัพธ์ทั้งหมด
        for seed, final_net in all_results:
            if final_net > max_net_for_window:
                max_net_for_window = final_net
                best_seed_for_window = seed

        # --- สร้าง Action ที่ดีที่สุดสำหรับ Window นี้จาก Seed ที่พบ ---
        if best_seed_for_window >= 0:
            rng_best = np.random.default_rng(best_seed_for_window)
            best_actions_for_window = rng_best.integers(0, 2, size=window_len)
            best_actions_for_window[0] = 1
        else:
            # ถ้าไม่เจอ seed ที่ดี ให้ใช้ action ทั้งหมดเป็น 1
            best_actions_for_window = np.ones(window_len, dtype=int)
            max_net_for_window = 0

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

# ใช้ numpy vectorization สำหรับ get_max_action
def get_max_action_vectorized(price_list, fix=1500):
    """
    คำนวณหาลำดับ action (0, 1) ที่ให้ผลตอบแทนสูงสุดทางทฤษฎี
    ปรับปรุงด้วย vectorization
    """
    prices = np.asarray(price_list, dtype=np.float64)
    n = len(prices)

    if n < 2:
        return np.ones(n, dtype=int)

    # --- ส่วนที่ 1: คำนวณไปข้างหน้า (ปรับปรุงด้วย vectorization) ---
    
    dp = np.zeros(n, dtype=np.float64)
    path = np.zeros(n, dtype=int) 
    
    initial_capital = float(fix * 2)
    dp[0] = initial_capital
    
    # Vectorized computation
    for i in range(1, n):
        # คำนวณ profit สำหรับทุก j ที่เป็นไปได้ในครั้งเดียว
        j_indices = np.arange(i)
        profits = fix * ((prices[i] / prices[j_indices]) - 1)
        current_sumusd = dp[j_indices] + profits
        
        # หาค่าสูงสุดและ index
        best_idx = np.argmax(current_sumusd)
        dp[i] = current_sumusd[best_idx]
        path[i] = j_indices[best_idx]

    # --- ส่วนที่ 2: การย้อนรอย (เหมือนเดิม) ---
    actions = np.zeros(n, dtype=int)
    
    last_action_day = np.argmax(dp)
    
    current_day = last_action_day
    while current_day > 0:
        actions[current_day] = 1
        current_day = path[current_day]
        
    actions[0] = 1
    
    return actions

def get_max_action(price_list, fix=1500):
    """
    Wrapper function สำหรับ vectorized version
    """
    return get_max_action_vectorized(price_list, fix)

@st.cache_data(ttl=3600)  # Cache ข้อมูล ticker เป็นเวลา 1 ชั่วโมง
def get_ticker_data(ticker, filter_date='2023-01-01 12:00:00+07:00'):
    """
    ดึงข้อมูล ticker พร้อม caching
    """
    tickerData = yf.Ticker(ticker)
    tickerData = tickerData.history(period='max')[['Close']]
    tickerData.index = tickerData.index.tz_convert(tz='Asia/Bangkok')
    tickerData = tickerData[tickerData.index >= filter_date]
    return tickerData

def Limit_fx(Ticker='', act=-1, window_size=30, num_seeds_to_try=1000, max_workers=4):
    tickerData = get_ticker_data(Ticker)
    prices = np.array(tickerData.Close.values, dtype=np.float64)

    if act == -1:  # min
        actions = np.array(np.ones(len(prices)), dtype=np.int64)

    elif act == -2:  # max  
        actions = get_max_action(prices)
    
    elif act == -3:  # best_seed sliding window
        progress_bar = st.progress(0)
        actions, window_details = find_best_seed_sliding_window_optimized(
            prices, tickerData, window_size=window_size, 
            num_seeds_to_try=num_seeds_to_try, progress_bar=progress_bar,
            max_workers=max_workers
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

def plot_comparison(Ticker='', act=-1, window_size=30, num_seeds_to_try=1000, max_workers=4):
    all = []
    all_id = []
    
    # min
    all.append(Limit_fx(Ticker, act=-1).net)
    all_id.append('min')

    # fx (best_seed or other)
    if act == -3:  # best_seed
        all.append(Limit_fx(Ticker, act=act, window_size=window_size, 
                           num_seeds_to_try=num_seeds_to_try, max_workers=max_workers).net)
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

        st.write("🎯 Best Seed Sliding Window Tester (Optimized)")
        st.write("เครื่องมือทดสอบการหา Best Seed ด้วยวิธี Sliding Window สำหรับการเทรด (ปรับปรุงความเร็ว)")
        
        # Sidebar สำหรับการตั้งค่า
        st.write("⚙️ การตั้งค่า")
        
        # เลือก ticker สำหรับทดสอบ
        test_ticker = st.selectbox(
            "เลือก Ticker สำหรับทดสอบ", 
            ['FFWM', 'NEGG', 'RIVN', 'APLS', 'NVTS', 'QXO', 'RXRX']
        )
        
        # ตั้งค่าพารามิเตอร์
        window_size = st.number_input(
            "ขนาด Window (วัน)", 
            min_value=2, max_value=730, value=30
        )
        
        num_seeds = st.number_input(
            "จำนวน Seeds ต่อ Window", 
            min_value=100, max_value=10000, value=1000
        )
        
        # เพิ่มการตั้งค่า parallel processing
        max_workers = st.number_input(
            "จำนวน Workers สำหรับ Parallel Processing", 
            min_value=1, max_value=16, value=4,
            help="เพิ่มจำนวน workers เพื่อความเร็วมากขึ้น (แนะนำ 4-8)"
        )

    with tab2:
        # ปุ่มเริ่มทดสอบ
        st.write("---")
        if st.button("🚀 เริ่มทดสอบ Best Seed (Optimized)", type="primary"):
            st.write(f"กำลังทดสอบ Best Seed สำหรับ **{test_ticker}** 📊")
            st.write(f"⚙️ พารามิเตอร์: Window Size = {window_size}, Seeds per Window = {num_seeds}, Workers = {max_workers}")
            st.write("---")
            
            try:
                # เรียกใช้ plot comparison
                plot_comparison(Ticker=test_ticker, act=-3, window_size=window_size, 
                              num_seeds_to_try=num_seeds, max_workers=max_workers)
                
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
                    
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
                st.write("กรุณาลองปรับพารามิเตอร์หรือเลือก ticker อื่น")

    # คำอธิบายวิธีการทำงาน
    st.write("---")
    st.write("📖 คำอธิบายวิธีการทำงาน")

    
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
        
        **จำนวน Workers:**
        - 1-2: สำหรับเครื่องที่มี CPU น้อย
        - 4: เหมาะสำหรับเครื่องทั่วไป
        - 6-8: สำหรับเครื่องที่มี CPU หลายคอร์
        """)
    
    with st.expander("⚡ การปรับปรุงความเร็ว"):
        st.write("""
        **การปรับปรุงที่ทำ:**
        
        1. **Parallel Processing**: ใช้ ThreadPoolExecutor เพื่อประเมิน seeds หลายตัวพร้อมกัน
        2. **Caching**: ใช้ @lru_cache สำหรับฟังก์ชันคำนวณและ @st.cache_data สำหรับข้อมูล ticker
        3. **Vectorization**: ใช้ NumPy operations แทน Python loops ในส่วนที่เป็นไปได้
        4. **Batch Processing**: แบ่ง seeds เป็น batches เพื่อลด overhead ของ threading
        5. **Memory Optimization**: ใช้ dtype ที่เหมาะสมและ pre-allocate arrays
        
        **ผลลัพธ์ที่คาดหวัง:**
        - เร็วขึ้น 2-4 เท่า (ขึ้นอยู่กับจำนวน CPU cores)
        - ใช้ memory มีประสิทธิภาพมากขึ้น
        - ผลลัพธ์เหมือนเดิมทุกประการ
        """)

if __name__ == "__main__":
    main()
