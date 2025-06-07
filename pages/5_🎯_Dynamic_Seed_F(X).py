import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(page_title="Best Seed Sliding Window", page_icon="🎯", layout="wide")

# --- ฟังก์ชันคำนวณและฟังก์ชันเสริม (ใช้โค้ดต้นฉบับ ไม่มีการเปลี่ยนแปลง) ---

@lru_cache(maxsize=1000)
def calculate_optimized_cached(action_tuple, price_tuple, fix=1500):
    action_array = np.asarray(action_tuple, dtype=np.int32)
    action_array[0] = 1
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
    refer = -fix * np.log(initial_price / price_array)
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
    return calculate_optimized_cached(tuple(action_list), tuple(price_list), fix)

def evaluate_seed_batch(seed_batch, prices_window, window_len):
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
    return results

@st.cache_data(ttl=3600)
def get_ticker_data(ticker, filter_date='2023-01-01 12:00:00+07:00'):
    tickerData = yf.Ticker(ticker)
    tickerData = tickerData.history(period='max')[['Close']]
    tickerData.index = tickerData.index.tz_convert(tz='Asia/Bangkok')
    tickerData = tickerData[tickerData.index >= filter_date]
    return tickerData

def get_max_action_vectorized(price_list, fix=1500):
    prices = np.asarray(price_list, dtype=np.float64)
    n = len(prices)
    if n < 2: return np.ones(n, dtype=int)
    dp = np.zeros(n, dtype=np.float64)
    path = np.zeros(n, dtype=int)
    dp[0] = float(fix * 2)
    for i in range(1, n):
        j_indices = np.arange(i)
        profits = fix * ((prices[i] / prices[j_indices]) - 1)
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

def get_max_action(price_list, fix=1500):
    return get_max_action_vectorized(price_list, fix)


# --- ฟังก์ชันหลักที่แก้ไขใหม่ (Corrected Version) ---

def find_best_seed_sliding_window_optimized(price_list, ticker_data_with_dates=None, window_size=30, num_seeds_to_try=1000, progress_bar=None, max_workers=4, seed_shift=0):
    prices = np.asarray(price_list)
    n = len(prices)
    
    windows_range = list(range(0, n, window_size))
    num_windows = len(windows_range)

    # --- ขั้นตอนที่ 1: ค้นหา Best Seed ดั้งเดิมสำหรับแต่ละ Window ---
    st.write("🔍 **ขั้นตอนที่ 1: กำลังค้นหา Best Seed ดั้งเดิมสำหรับแต่ละ Window...**")
    original_best_seeds = []

    for i, start_index in enumerate(windows_range):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        window_len = len(prices_window)

        if window_len < 2:
            original_best_seeds.append(-1) # ไม่มี seed ที่ดีสำหรับ window ที่สั้นไป
            continue

        best_seed_for_window = -1
        max_net_for_window = -np.inf
        
        # ใช้โครงสร้าง Parallel Processing เดิมที่ทำงานได้ดี
        random_seeds = np.arange(num_seeds_to_try)
        batch_size = max(1, num_seeds_to_try // max_workers)
        seed_batches = [random_seeds[j:j+batch_size] for j in range(0, len(random_seeds), batch_size)]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(evaluate_seed_batch, batch, prices_window, window_len): batch 
                for batch in seed_batches
            }
            all_results = []
            for future in as_completed(future_to_batch):
                all_results.extend(future.result())
        
        for seed, final_net in all_results:
            if final_net > max_net_for_window:
                max_net_for_window = final_net
                best_seed_for_window = seed
        
        original_best_seeds.append(best_seed_for_window)
        if progress_bar:
            progress_bar.progress((i + 1) / num_windows, text=f"ขั้นตอนที่ 1: ค้นหา Seed ของ Window {i+1}/{num_windows}")

    # --- ขั้นตอนที่ 2: "Shift" ลำดับของ Best Seeds ---
    st.write(f"⚙️ **ขั้นตอนที่ 2: ทำการ Shift ลำดับ Seed ไป {seed_shift} ตำแหน่ง...**")
    
    if seed_shift == 0:
        shifted_seeds = original_best_seeds
    elif seed_shift > 0:
        shifted_seeds = ([-1] * seed_shift) + original_best_seeds[:-seed_shift]
    else: 
        shift_abs = abs(seed_shift)
        shifted_seeds = original_best_seeds[shift_abs:] + ([-1] * shift_abs)

    # --- ขั้นตอนที่ 3: คำนวณ Actions และผลลัพธ์ใหม่โดยใช้ Shifted Seeds ---
    st.write("🔄 **ขั้นตอนที่ 3: คำนวณผลลัพธ์ใหม่จาก Shifted Seeds...**")
    final_actions = np.array([], dtype=int)
    window_details = []

    for i, start_index in enumerate(windows_range):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        window_len = len(prices_window)

        if window_len == 0:
            continue

        current_seed = shifted_seeds[i]

        if current_seed >= 0:
            rng_best = np.random.default_rng(current_seed)
            actions_for_window = rng_best.integers(0, 2, size=window_len)
            actions_for_window[0] = 1
        else:
            actions_for_window = np.ones(window_len, dtype=int)
            current_seed = np.nan

        if window_len < 2:
            final_net_for_window = 0
        else:
            _, sumusd, _, _, _, refer = calculate_optimized(actions_for_window.tolist(), prices_window.tolist())
            initial_capital = sumusd[0]
            net = sumusd - refer - initial_capital
            final_net_for_window = net[-1]
            
        final_actions = np.concatenate((final_actions, actions_for_window))

        timeline_info = f"Index {start_index} ถึง {end_index-1}"
        if ticker_data_with_dates is not None:
            start_date = ticker_data_with_dates.index[start_index].strftime('%Y-%m-%d')
            end_date = ticker_data_with_dates.index[end_index-1].strftime('%Y-%m-%d')
            timeline_info = f"{start_date} ถึง {end_date}"

        window_detail = {
            'window_number': i + 1, 'timeline': timeline_info,
            'best_seed': current_seed, 'max_net': round(final_net_for_window, 2),
            'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2) if window_len > 0 else 0,
            'action_count': int(np.sum(actions_for_window)), 'window_size': window_len,
        }
        window_details.append(window_detail)
        
        if progress_bar:
            progress_bar.progress((i + 1) / num_windows, text=f"ขั้นตอนที่ 3: คำนวณผลลัพธ์ Window {i+1}/{num_windows}")

    return final_actions, window_details


# --- ฟังก์ชันที่เรียกใช้หลัก (แก้ไขให้รับและส่ง seed_shift) ---

def Limit_fx(Ticker='', act=-1, window_size=30, num_seeds_to_try=1000, max_workers=4, seed_shift=0):
    tickerData = get_ticker_data(Ticker)
    prices = np.array(tickerData.Close.values, dtype=np.float64)

    actions = np.array([], dtype=int)
    window_details = []

    if act == -1:
        actions = np.array(np.ones(len(prices)), dtype=np.int64)
    elif act == -2:
        actions = get_max_action(prices)
    elif act == -3:
        progress_bar = st.progress(0, text="เริ่มต้น...")
        actions, window_details = find_best_seed_sliding_window_optimized(
            prices, tickerData, window_size=window_size, 
            num_seeds_to_try=num_seeds_to_try, progress_bar=progress_bar,
            max_workers=max_workers, seed_shift=seed_shift
        )
        
        st.write("---")
        st.write("📈 **สรุปผลการคำนวณ (ตาม Seed Shift ที่กำหนด)**")
        total_windows = len(window_details)
        total_actions = sum([w['action_count'] for w in window_details])
        total_net = sum([w['max_net'] for w in window_details])
        
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Total Windows", total_windows)
        with col2: st.metric("Total Actions", f"{total_actions}/{len(actions)}")
        with col3: st.metric("Total Net (Sum)", f"{total_net:.2f}")
        
        st.write("📋 **รายละเอียดแต่ละ Window (ใช้ Shifted Seed)**")
        df_details = pd.DataFrame(window_details)
        df_display = df_details[['window_number', 'timeline', 'best_seed', 'max_net', 
                               'price_change_pct', 'action_count', 'window_size']].copy()
        df_display.columns = ['Window', 'Timeline', 'Used Seed', 'Net Profit', 
                            'Price Change %', 'Actions', 'Window Size']
        st.dataframe(df_display, use_container_width=True)
        st.session_state[f'window_details_{Ticker}'] = window_details
    else:
        rng = np.random.default_rng(act)
        actions = rng.integers(0, 2, len(prices))
    
    buffer, sumusd, cash, asset_value, amount, refer = calculate_optimized(actions, prices)
    initial_capital = sumusd[0]
    
    df = pd.DataFrame({
        'price': prices, 'action': actions, 'buffer': np.round(buffer, 2),
        'sumusd': np.round(sumusd, 2), 'cash': np.round(cash, 2),
        'asset_value': np.round(asset_value, 2), 'amount': np.round(amount, 2),
        'refer': np.round(refer + initial_capital, 2),
        'net': np.round(sumusd - refer - initial_capital, 2)
    })
    return df

def plot_comparison(Ticker='', act=-1, window_size=30, num_seeds_to_try=1000, max_workers=4, seed_shift=0):
    all_data = []
    all_id = []
    
    all_data.append(Limit_fx(Ticker, act=-1).net)
    all_id.append('min')

    if act == -3:
        all_data.append(Limit_fx(Ticker, act=act, window_size=window_size, 
                                 num_seeds_to_try=num_seeds_to_try, max_workers=max_workers, 
                                 seed_shift=seed_shift).net)
        all_id.append(f'best_seed_shift({seed_shift})')
    else:
        all_data.append(Limit_fx(Ticker, act=act).net)
        all_id.append(f'fx_{act}')

    all_data.append(Limit_fx(Ticker, act=-2).net)
    all_id.append('max')
    
    chart_data = pd.DataFrame(np.array(all_data).T, columns=np.array(all_id))
    st.write('📊 **Refer_Log Comparison**')
    st.line_chart(chart_data)

    df_plot = Limit_fx(Ticker, act=-1)
    df_plot = df_plot[['buffer']].cumsum()
    st.write('💰 **Burn_Cash**')
    st.line_chart(df_plot)
    st.write("---")
    st.write("### ผลลัพธ์ของ 'min' strategy (Baseline)")
    st.dataframe(Limit_fx(Ticker, act=-1))

# --- Main Streamlit App (แก้ไขให้มี input seed_shift) ---

def main():
    tab1, tab2 = st.tabs(["การตั้งค่า", "ทดสอบ"])
    
    with tab1:
        st.write("🎯 **Best Seed Sliding Window Tester (with Seed Shift)**")
        st.write("เครื่องมือทดสอบการหา Best Seed และความสามารถในการ 'Shift' ลำดับของ Seed เพื่อวิเคราะห์ผลกระทบ")
        st.write("---")
        st.write("⚙️ **การตั้งค่า**")
        
        test_ticker = st.selectbox(
            "เลือก Ticker สำหรับทดสอบ", 
            ['FFWM', 'NEGG', 'RIVN', 'APLS', 'NVTS', 'QXO', 'RXRX', 'BTC-USD', 'ETH-USD']
        )
        
        col1, col2 = st.columns(2)
        with col1:
            window_size = st.number_input("ขนาด Window (วัน)", min_value=2, max_value=365, value=30)
            num_seeds = st.number_input("จำนวน Seeds ต่อ Window", min_value=100, max_value=10000, value=1000)
        with col2:
            seed_shift = st.number_input(
                "ตำแหน่งการเลื่อน Seed (Seed Shift)", 
                min_value=-50, max_value=50, value=0,
                help="เลื่อนลำดับของ Best Seed ที่หาได้ (0=ไม่เลื่อน, 1=เลื่อนไป 1 ตำแหน่ง, -1=เลื่อนกลับ 1 ตำแหน่ง)"
            )
            max_workers = st.number_input("จำนวน Workers (Parallel Processing)", min_value=1, max_value=16, value=4)

    with tab2:
        st.write("---")
        if st.button("🚀 เริ่มทดสอบ Best Seed", type="primary"):
            st.write(f"กำลังทดสอบ Best Seed สำหรับ **{test_ticker}** 📊")
            st.write(f"⚙️ พารามิเตอร์: Window Size = {window_size}, Seeds = {num_seeds}, Workers = {max_workers}, Seed Shift = {seed_shift}")
            
            try:
                plot_comparison(Ticker=test_ticker, act=-3, window_size=window_size, 
                              num_seeds_to_try=num_seeds, max_workers=max_workers, 
                              seed_shift=seed_shift)
                
                if f'window_details_{test_ticker}' in st.session_state:
                    st.write("---")
                    st.write("🔍 **การวิเคราะห์เพิ่มเติม**")
                    window_details = st.session_state[f'window_details_{test_ticker}']
                    df_windows = pd.DataFrame(window_details)
                    
                    st.write("📊 **Net Profit แต่ละ Window (คำนวณจาก Used Seed)**")
                    st.bar_chart(df_windows.set_index('window_number')['max_net'])
                    
                    st.write("📈 **Price Change % แต่ละ Window**")
                    st.bar_chart(df_windows.set_index('window_number')['price_change_pct'])
                    
                    csv = df_windows.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 ดาวน์โหลด Window Details (CSV)",
                        data=csv,
                        file_name=f'best_seed_results_{test_ticker}_w{window_size}_s{num_seeds}_shift{seed_shift}.csv',
                        mime='text/csv'
                    )
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
                st.exception(e)

    with st.expander("📖 คำอธิบายวิธีการทำงาน (เวอร์ชันแก้ไข)", expanded=True):
        st.markdown("""
        **หลักการทำงานของ Seed Shift:**
        
        1.  **ขั้นตอนที่ 1: ค้นหา Best Seed ดั้งเดิม**
            - โปรแกรมจะทำงานตามปกติก่อน คือแบ่งข้อมูลราคาออกเป็นช่วงๆ (Windows)
            - จากนั้นใช้ Parallel Processing เพื่อค้นหา `Seed` ที่ให้ผลกำไรสูงสุดสำหรับ **แต่ละ Window** แยกจากกัน
            - ผลลัพธ์ของขั้นตอนนี้คือ "ลำดับของ Seed ที่ดีที่สุด" เช่น `[Seed_W1, Seed_W2, Seed_W3, ...]`
        
        2.  **ขั้นตอนที่ 2: เลื่อนลำดับ (Shift)**
            - นำลำดับ Seed ที่ได้จากข้อ 1 มาเลื่อนตามค่า `Seed Shift` ที่คุณกำหนด:
            - **Shift = 0 (ค่าเริ่มต้น):** ไม่มีการเลื่อน ใช้ `[Seed_W1, Seed_W2, Seed_W3]`
            - **Shift = 1:** เลื่อนไปทางขวา 1 ตำแหน่ง ผลลัพธ์คือ `[ว่าง, Seed_W1, Seed_W2]`
            - **Shift = -1:** เลื่อนไปทางซ้าย 1 ตำแหน่ง ผลลัพธ์คือ `[Seed_W2, Seed_W3, ว่าง]`
            - (ตำแหน่ง `ว่าง` จะถูกแทนที่ด้วยค่า -1 และจะใช้ Action เริ่มต้นในการคำนวณ)

        3.  **ขั้นตอนที่ 3: คำนวณผลลัพธ์ใหม่**
            - โปรแกรมจะวนลูปผ่านแต่ละ Window อีกครั้ง แต่คราวนี้จะใช้ Seed จาก "ลำดับที่ถูกเลื่อนแล้ว" เพื่อสร้าง Action และคำนวณกำไรใหม่
            - **ตัวอย่าง (Shift=1):**
                - `Window 1` จะใช้ Seed `ว่าง` (ใช้ Action เริ่มต้น)
                - `Window 2` จะใช้ Seed เดิมของ `Window 1`
                - `Window 3` จะใช้ Seed เดิมของ `Window 2`
            - ผลลัพธ์สุดท้ายที่แสดงในตารางและกราฟจะมาจากการคำนวณในขั้นตอนนี้
        
        **ประโยชน์:** ทำให้สามารถวิเคราะห์ได้ว่า "กลยุทธ์ (Seed) ที่เคยดีที่สุดในอดีต จะยังทำงานได้ดีอยู่หรือไม่ในอนาคตอันใกล้"
        """)

if __name__ == "__main__":
    main()
