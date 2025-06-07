# --- START OF FILE 2025-06-07T12-07_export.csv --- is not needed for the code to run
# It's an example of the output data, which the new code will now generate.

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
    
    if n == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

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
            actions_window[0] = 1

            if window_len < 2:
                final_net = 0
            else:
                _, sumusd, _, _, _, refer = calculate_optimized(actions_window.tolist(), prices_window.tolist())
                initial_capital = sumusd[0]
                net = sumusd - refer - initial_capital
                final_net = net[-1]

            results.append((seed, final_net))
                    
        except Exception as e:
            results.append((seed, -np.inf))
            continue
    
    return results

def find_best_seed_sliding_window_optimized(price_list, ticker_data_with_dates=None, window_size=30, num_seeds_to_try=1000, progress_bar=None, max_workers=4):
    """
    ค้นหาลำดับ action ที่ดีที่สุดโดยการหา seed ที่ให้ผลกำไรสูงสุดในแต่ละช่วงเวลา (sliding window)
    """
    prices = np.asarray(price_list)
    n = len(prices)
    final_actions = np.array([], dtype=int)
    window_details = []
    
    # แก้ไขการคำนวณจำนวน windows ให้ถูกต้อง
    num_windows = math.ceil(n / window_size) if window_size > 0 else 0

    st.write("---")
    st.write("### 🔬 Step 1: ค้นหา Best Seed สำหรับแต่ละ Window (In-Sample Testing)")
    st.write(f"ข้อมูลทั้งหมด: {n} วัน | ขนาด Window: {window_size} วัน | จำนวน Windows: {num_windows}")
    
    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        window_len = len(prices_window)

        if window_len == 0:
            continue

        start_date = ticker_data_with_dates.index[start_index].strftime('%Y-%m-%d')
        end_date = ticker_data_with_dates.index[end_index-1].strftime('%Y-%m-%d')
        timeline_info = f"{start_date} ถึง {end_date}"

        best_seed_for_window = -1
        max_net_for_window = -np.inf

        random_seeds = np.arange(num_seeds_to_try)
        batch_size = max(1, num_seeds_to_try // max_workers)
        seed_batches = [random_seeds[j:j+batch_size] for j in range(0, len(random_seeds), batch_size)]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {executor.submit(evaluate_seed_batch, batch, prices_window, window_len): batch for batch in seed_batches}
            all_results = []
            for future in as_completed(future_to_batch):
                all_results.extend(future.result())
        
        for seed, final_net in all_results:
            if final_net > max_net_for_window:
                max_net_for_window = final_net
                best_seed_for_window = seed

        if best_seed_for_window >= 0:
            rng_best = np.random.default_rng(best_seed_for_window)
            best_actions_for_window = rng_best.integers(0, 2, size=window_len)
            best_actions_for_window[0] = 1
        else:
            best_actions_for_window = np.ones(window_len, dtype=int)
            max_net_for_window = 0

        window_detail = {
            'Window': i + 1,
            'Timeline': timeline_info,
            'start_index': start_index,
            'end_index': end_index - 1,
            'Window Size': window_len,
            'Best Seed': best_seed_for_window,
            'Net Profit': round(max_net_for_window, 2),
            'Price Change %': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2),
            'Actions': int(np.sum(best_actions_for_window)),
            'action_sequence': best_actions_for_window.tolist()
        }
        window_details.append(window_detail)
        final_actions = np.concatenate((final_actions, best_actions_for_window))
        
        if progress_bar:
            progress_bar.progress((i + 1) / num_windows)

    df_details = pd.DataFrame(window_details)
    st.dataframe(df_details[['Window', 'Timeline', 'Best Seed', 'Net Profit', 'Price Change %', 'Actions', 'Window Size']], use_container_width=True)
    return final_actions, df_details

# >>> NEW FUNCTION TO HANDLE SHIFTED CALCULATION <<<
def calculate_walk_forward_profit(original_details_df, prices, ticker_data_with_dates, shift_periods=1, fix=1500):
    """
    คำนวณ Net Profit โดยใช้ Action จาก Window ก่อนหน้า (Shifted / Walk-Forward)
    """
    st.write("---")
    st.write(f"### 🧬 Step 2: ทดสอบแบบ Walk-Forward (Shift = {shift_periods} Window)")
    st.write("นำ `Action Sequence` (DNA) ที่ดีที่สุดจาก Window ก่อนหน้า มาคำนวณผลกำไรใน Window ปัจจุบัน")

    shifted_results = []
    final_shifted_actions = np.array([], dtype=int)

    # เราจะเริ่มคำนวณจาก window ที่ `shift_periods` เป็นต้นไป
    for i in range(shift_periods, len(original_details_df)):
        current_window_info = original_details_df.iloc[i]
        # ดึง "DNA" จาก window ก่อนหน้า
        previous_window_info = original_details_df.iloc[i - shift_periods]

        # Actions ที่จะใช้ (จาก window ก่อนหน้า)
        actions_to_apply = np.array(previous_window_info['action_sequence'])
        
        # ราคาของ window ปัจจุบัน
        start_idx = current_window_info['start_index']
        end_idx = current_window_info['end_index'] + 1
        prices_current_window = prices[start_idx:end_idx]
        
        # จัดการกรณีที่ขนาด window ไม่เท่ากัน (เช่น window สุดท้าย)
        # โดยการตัด/เติม actions ให้มีขนาดเท่ากับราคาใน window ปัจจุบัน
        len_actions = len(actions_to_apply)
        len_prices = len(prices_current_window)

        if len_actions > len_prices:
            actions_to_apply = actions_to_apply[:len_prices]
        elif len_actions < len_prices:
            # เติม action สุดท้ายไปจนครบ
            padding = np.full(len_prices - len_actions, actions_to_apply[-1])
            actions_to_apply = np.concatenate([actions_to_apply, padding])

        # บังคับให้ action แรกเป็น 1 เสมอเพื่อเริ่มการลงทุนใน window
        actions_to_apply[0] = 1

        # คำนวณผลลัพธ์สำหรับ window นี้ด้วย action ที่ shifted มา
        if len(prices_current_window) < 2:
            net_profit_shifted = 0
        else:
            _, sumusd, _, _, _, refer = calculate_optimized(actions_to_apply.tolist(), prices_current_window.tolist(), fix=fix)
            initial_capital = sumusd[0]
            net = sumusd - refer - initial_capital
            net_profit_shifted = net[-1]

        # บันทึกผล
        result = {
            'Window': current_window_info['Window'],
            'Timeline': current_window_info['Timeline'],
            'Used Seed (From Window {target})'.format(target=previous_window_info['Window']): previous_window_info['Best Seed'],
            'Net Profit': round(net_profit_shifted, 2),
            'Price Change %': current_window_info['Price Change %'],
            'Actions': int(np.sum(actions_to_apply)),
            'Window Size': len(prices_current_window)
        }
        shifted_results.append(result)
        
        # เติม actions ที่ไม่ได้ใช้ในตอนต้นด้วย 0 (ไม่มีการเทรด)
        if i == shift_periods:
             num_initial_days = original_details_df.iloc[0]['start_index']
             for j in range(shift_periods):
                 initial_days_len = len(original_details_df.iloc[j]['action_sequence'])
                 num_initial_days += initial_days_len
             
             final_shifted_actions = np.zeros(num_initial_days, dtype=int)


        final_shifted_actions = np.concatenate((final_shifted_actions, actions_to_apply))
    
    if not shifted_results:
        st.warning("ไม่สามารถคำนวณ Walk-Forward ได้ จำนวนข้อมูลน้อยเกินไปเทียบกับ Shift Periods")
        return None, None

    df_shifted = pd.DataFrame(shifted_results)
    st.dataframe(df_shifted, use_container_width=True)
    return final_shifted_actions, df_shifted


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

@st.cache_data(ttl=3600)
def get_ticker_data(ticker, filter_date='2023-01-01 12:00:00+07:00'):
    tickerData = yf.Ticker(ticker)
    tickerData = tickerData.history(period='max')[['Close']]
    tickerData.index = tickerData.index.tz_convert(tz='Asia/Bangkok')
    tickerData = tickerData[tickerData.index >= filter_date]
    return tickerData

def calculate_full_period_net(actions, prices, fix=1500):
    """คำนวณ Net Profit สำหรับข้อมูลทั้งช่วง"""
    if len(actions) == 0 or len(prices) == 0:
        return pd.Series(dtype=np.float64)
        
    # ทำให้ length เท่ากัน
    min_len = min(len(actions), len(prices))
    actions = actions[:min_len]
    prices = prices[:min_len]

    buffer, sumusd, cash, asset_value, amount, refer = calculate_optimized(actions, prices)
    initial_capital = sumusd[0]
    net_series = pd.Series(sumusd - refer - initial_capital, name='net')
    return net_series

# --- RESTRUCTURED MAIN APP ---
def main():
    st.title("🧬 Best Seed Walk-Forward Tester")
    st.write("เครื่องมือทดสอบกลยุทธ์ Best Seed ด้วยวิธี Sliding Window และการประเมินผลแบบ Walk-Forward Analysis")
    
    with st.sidebar:
        st.header("⚙️ การตั้งค่า")
        test_ticker = st.selectbox(
            "เลือก Ticker", 
            ['FFWM', 'NEGG', 'RIVN', 'APLS', 'NVTS', 'QXO', 'RXRX', 'BTC-USD', 'ETH-USD'],
            index=0
        )
        window_size = st.number_input(
            "ขนาด Window (วัน)", 
            min_value=2, max_value=120, value=30, step=1
        )
        num_seeds = st.number_input(
            "จำนวน Seeds ต่อ Window", 
            min_value=100, max_value=10000, value=1000, step=100
        )
        max_workers = st.slider(
            "จำนวน Workers (Parallel CPU)", 
            min_value=1, max_value=16, value=4
        )
        shift_periods = st.number_input(
            "Walk-Forward Shift (Windows)",
            min_value=1, max_value=5, value=1, step=1,
            help="จำนวน window ที่จะเลื่อน 'DNA' ไปข้างหน้าเพื่อทดสอบ (1 = ใช้ DNA จาก window ที่แล้วมาเทรด window ปัจจุบัน)"
        )

    if st.sidebar.button("🚀 เริ่มการทดสอบและวิเคราะห์", type="primary", use_container_width=True):
        
        ticker_data = get_ticker_data(test_ticker)
        prices = ticker_data.Close.values
        
        if len(prices) < window_size * 2:
            st.error(f"ข้อมูลไม่เพียงพอสำหรับ Ticker: {test_ticker}. ต้องการข้อมูลอย่างน้อย {window_size*2} วัน แต่มีเพียง {len(prices)} วัน.")
            return

        progress_bar = st.progress(0, text="กำลังค้นหา Best Seed (In-Sample)...")
        
        # --- STEP 1: FIND BEST SEED (IN-SAMPLE) ---
        best_seed_actions, original_details = find_best_seed_sliding_window_optimized(
            prices, ticker_data, window_size=window_size, 
            num_seeds_to_try=num_seeds, progress_bar=progress_bar,
            max_workers=max_workers
        )
        progress_bar.empty()

        # --- STEP 2: CALCULATE WALK-FORWARD (SHIFTED) ---
        walk_forward_actions, shifted_details = calculate_walk_forward_profit(
            original_details, prices, ticker_data, shift_periods=shift_periods
        )

        # --- STEP 3: PREPARE DATA FOR COMPARISON PLOT ---
        st.write("---")
        st.write("### 📊 Step 3: สรุปและเปรียบเทียบผลลัพธ์ Net Profit")

        # Calculate Net series for each strategy
        net_comparison = pd.DataFrame(index=ticker_data.index)
        
        # Min Strategy (Buy & Hold DCA-like)
        min_actions = np.ones(len(prices), dtype=int)
        net_comparison['Min (DCA)'] = calculate_full_period_net(min_actions, prices)

        # Max Strategy (Theoretical Best)
        max_actions = get_max_action_vectorized(prices)
        net_comparison['Max (Theoretical)'] = calculate_full_period_net(max_actions, prices)
        
        # Best Seed Strategy (In-Sample)
        net_comparison['Best Seed (In-Sample)'] = calculate_full_period_net(best_seed_actions, prices)
        
        # Walk-Forward Strategy (Shifted)
        if walk_forward_actions is not None:
             net_comparison[f'Walk-Forward (Shift={shift_periods})'] = calculate_full_period_net(walk_forward_actions, prices)

        st.line_chart(net_comparison)

        # --- Display Summary Metrics ---
        st.write("#### สรุปผลกำไรสุดท้าย (Final Net Profit)")
        final_profits = {
            "กลยุทธ์": list(net_comparison.columns),
            "ผลกำไรสุดท้าย": [f"{net_comparison[col].iloc[-1]:,.2f}" for col in net_comparison.columns]
        }
        st.dataframe(pd.DataFrame(final_profits), use_container_width=True)

        st.write("---")
        st.info("💡 **In-Sample** คือผลลัพธ์จากการใช้ข้อมูลในอนาคต (ของ Window นั้นๆ) เพื่อหา Seed ที่ดีที่สุด | **Walk-Forward** คือผลลัพธ์ที่สมจริงกว่า โดยใช้ Seed ที่หาได้จากข้อมูลในอดีตมาเทรด")

    with st.expander("📖 คำอธิบายวิธีการทำงาน"):
        st.markdown("""
        **Best Seed Sliding Window** เป็นเทคนิคการหา action sequence ที่ดีที่สุดโดย:
        
        1.  **แบ่งข้อมูล**: แบ่งข้อมูลราคาออกเป็นช่วง ๆ (windows) ตามขนาดที่กำหนด
        2.  **ค้นหา Seed (In-Sample)**: ในแต่ละ window ทำการสุ่ม seed หลาย ๆ ตัวและคำนวณผลกำไรเพื่อหา **Best Seed** ที่ให้ผลกำไรสูงสุดสำหรับ window *นั้นๆ* ผลลัพธ์นี้คือ **In-Sample** ซึ่งมักจะดูดีเกินจริงเพราะมัน "รู้อนาคต" ภายใน window ของมัน
        3.  **ทดสอบแบบ Walk-Forward**: เพื่อการประเมินที่สมจริงขึ้น เราจะทำการ **"Shift"** หรือเลื่อน `action sequence` (DNA) ที่ได้จาก Best Seed ของ **Window ก่อนหน้า** มาใช้คำนวณผลกำไรกับราคาใน **Window ปัจจุบัน** นี่คือการจำลองว่าเรานำกลยุทธ์ที่เรียนรู้จากอดีตมาใช้กับอนาคตที่ไม่เคยเห็น
        4.  **เปรียบเทียบผล**: สุดท้ายเราจะเปรียบเทียบผลกำไรของกลยุทธ์ต่างๆ:
            *   **Min (DCA)**: ซื้อทุกวัน (เปรียบเทียบฐาน)
            *   **Max (Theoretical)**: ผลตอบแทนสูงสุดทางทฤษฎี (เป้าหมายสูงสุด)
            *   **Best Seed (In-Sample)**: ผลลัพธ์จากการหา Seed ที่ดีที่สุดในแต่ละ Window (มักจะดีเกินจริง)
            *   **Walk-Forward (Shifted)**: ผลลัพธ์ที่คาดหวังได้จริงมากกว่า จากการนำกลยุทธ์ในอดีตมาใช้
        """)

if __name__ == "__main__":
    main()
