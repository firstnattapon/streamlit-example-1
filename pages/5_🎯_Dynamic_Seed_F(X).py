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
    ฟังก์ชันคำนวณผลตอบแทนแบบสะสมต่อเนื่อง
    """
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
        except Exception as e:
            results.append((seed, -np.inf))
            continue
    return results

def find_best_seed_sliding_window_optimized(price_list, ticker_data_with_dates=None, window_size=30, num_seeds_to_try=1000, progress_bar=None, max_workers=4):
    prices = np.asarray(price_list)
    n = len(prices)
    final_actions = np.array([], dtype=int)
    window_details = []
    num_windows = (n + window_size - 1) // window_size
    
    st.info("ℹ️ **ตรรกะ 'Prev.seed Competition' ทำงานอยู่:** Seed ของ Window ก่อนหน้าจะถูกนำมาแข่งขันใน Window ปัจจุบัน")
    st.write("🔍 **เริ่มต้นการค้นหา Best Seed...**")
    st.write(f"📊 ข้อมูล: {n} วัน | ขนาด Window: {window_size} วัน | จำนวน Windows: {num_windows}")
    st.write("---")
    
    previous_best_seed = -1
    
    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        window_len = len(prices_window)
        if window_len == 0: continue

        timeline_info = f"Index {start_index} ถึง {end_index-1}"
        if ticker_data_with_dates is not None:
            start_date = ticker_data_with_dates.index[start_index].strftime('%Y-%m-%d')
            end_date = ticker_data_with_dates.index[end_index-1].strftime('%Y-%m-%d')
            timeline_info = f"{start_date} ถึง {end_date}"

        best_seed_for_window = -1
        max_net_for_window = -np.inf

        candidate_seeds = set(np.arange(num_seeds_to_try))
        if previous_best_seed >= 0:
            candidate_seeds.add(previous_best_seed)
        seeds_to_evaluate = list(candidate_seeds)

        batch_size = max(1, len(seeds_to_evaluate) // max_workers)
        seed_batches = [seeds_to_evaluate[j:j+batch_size] for j in range(0, len(seeds_to_evaluate), batch_size)]
        
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
            'window_number': i + 1, 'timeline': timeline_info,
            'previous_best_seed': previous_best_seed if previous_best_seed >= 0 else np.nan,
            'best_seed': best_seed_for_window, 'max_net': round(max_net_for_window, 2),
            'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2),
            'action_count': int(np.sum(best_actions_for_window)),
        }
        window_details.append(window_detail)

        st.write(f"**🎯 Window {i+1}/{num_windows}** | {timeline_info}")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            is_retained = "👑 (Retained)" if best_seed_for_window == previous_best_seed and previous_best_seed != -1 else ""
            st.metric("Best Seed", f"{best_seed_for_window:,} {is_retained}")
        with col2: st.metric("Net Profit", f"{max_net_for_window:.2f}")
        with col3: st.metric("Price Change", f"{window_detail['price_change_pct']:.2f}%")
        with col4: st.metric("Actions Count", f"{window_detail['action_count']}/{window_len}")

        final_actions = np.concatenate((final_actions, best_actions_for_window))
        if progress_bar: progress_bar.progress((i + 1) / num_windows)
        previous_best_seed = best_seed_for_window

    return final_actions, window_details

def get_max_action(price_list, fix=1500):
    # ... (no change needed here) ...
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
    last_action_day = np.argmax(dp)
    current_day = last_action_day
    while current_day > 0:
        actions[current_day] = 1
        current_day = path[current_day]
    actions[0] = 1
    return actions


@st.cache_data(ttl=3600)
def get_ticker_data(ticker, filter_date='2023-01-01 12:00:00+07:00'):
    # ... (no change needed here) ...
    tickerData = yf.Ticker(ticker)
    tickerData = tickerData.history(period='max')[['Close']]
    tickerData.index = tickerData.index.tz_convert(tz='Asia/Bangkok')
    tickerData = tickerData[tickerData.index >= filter_date]
    return tickerData


# <<< START OF MODIFICATION >>>
def Limit_fx(Ticker='', act=-1, window_size=30, num_seeds_to_try=1000, max_workers=4):
    """
    แก้ไข: ให้ฟังก์ชันนี้ return cả df และ window_details ออกมา
    """
    tickerData = get_ticker_data(Ticker)
    prices = np.array(tickerData.Close.values, dtype=np.float64)
    actions = None
    window_details = None  # ค่าเริ่มต้น

    if act == -1:
        actions = np.ones(len(prices), dtype=np.int64)
    elif act == -2:
        actions = get_max_action(prices)
    elif act == -3:
        progress_bar = st.progress(0)
        actions, window_details = find_best_seed_sliding_window_optimized(
            prices, tickerData, window_size=window_size, 
            num_seeds_to_try=num_seeds_to_try, progress_bar=progress_bar,
            max_workers=max_workers)
    else:
        rng = np.random.default_rng(act)
        actions = rng.integers(0, 2, len(prices))
    
    buffer, sumusd, cash, asset_value, amount, refer = calculate_optimized(actions.tolist(), prices.tolist())
    initial_capital = sumusd[0]
    
    df = pd.DataFrame({
        'price': prices, 'action': actions, 'buffer': np.round(buffer, 2),
        'sumusd': np.round(sumusd, 2), 'cash': np.round(cash, 2),
        'asset_value': np.round(asset_value, 2), 'amount': np.round(amount, 2),
        'refer': np.round(refer + initial_capital, 2),
        'net': np.round(sumusd - refer - initial_capital, 2)})
        
    return df, window_details # ส่งคืน 2 ค่า
# <<< END OF MODIFICATION >>>


# <<< START OF MODIFICATION >>>
def main():
    tab1, tab2, = st.tabs(["การตั้งค่า", "ทดสอบ"])
    with tab1:
        st.write("🎯 Best Seed Sliding Window (Logic: Prev.seed Competition)")
        st.write("เครื่องมือทดสอบการหา Best Seed โดยที่ Seed ที่ดีที่สุดจาก Window ก่อนหน้าจะถูกนำมาแข่งขันใน Window ปัจจุบันด้วย")
        st.write("---")
        st.write("⚙️ **การตั้งค่า**")
        test_ticker = st.selectbox("เลือก Ticker", ['FFWM', 'NEGG', 'RIVN', 'APLS', 'NVTS', 'QXO', 'RXRX'])
        window_size = st.number_input("ขนาด Window (วัน)", min_value=2, max_value=100, value=30)
        num_seeds = st.number_input("จำนวน Seeds ใหม่ต่อ Window", min_value=100, max_value=10000, value=1000)
        max_workers = st.number_input("จำนวน Workers", min_value=1, max_value=16, value=4, help="เพิ่มเพื่อความเร็ว (แนะนำ 4-8)")

    with tab2:
        if st.button("🚀 เริ่มทดสอบ (Logic: Prev.seed Competition)", type="primary", use_container_width=True):
            st.session_state.clear()
            try:
                st.write("---")
                st.write("1️⃣ **คำนวณ: Min (Action=1 ตลอด)**")
                df_min, _ = Limit_fx(Ticker=test_ticker, act=-1)
                
                st.write("2️⃣ **คำนวณ: Max (Theoretical Best)**")
                df_max, _ = Limit_fx(Ticker=test_ticker, act=-2)

                st.write("3️⃣ **คำนวณ: Best Seed (Prev.seed Competition)**")
                df_best_seed, window_details = Limit_fx(
                    Ticker=test_ticker, act=-3, window_size=window_size, 
                    num_seeds_to_try=num_seeds, max_workers=max_workers)

                st.write("---")
                st.write('📊 **Refer_Log Comparison**')
                tickerData = get_ticker_data(test_ticker)
                
                best_seed_label = "Best Seed"
                seed_sequence_str = "ไม่พบข้อมูล Seed"
                
                if window_details: 
                    all_seeds = [w['best_seed'] for w in window_details]
                    
                    if len(all_seeds) > 1:
                        best_seed_label = f"Best Seed[{all_seeds[0]}...{all_seeds[-1]}]"
                    else:
                        best_seed_label = f"Best Seed[{all_seeds[0]}]"

                    # สร้าง DNA String ตาม Format ที่ต้องการ
                    dna_parts = [f"NAN [Window 0]"]
                    for detail in window_details:
                        dna_parts.append(f"{detail['best_seed']} [Window {detail['window_number']}]")
                    seed_sequence_str = " -> ".join(dna_parts)
                
                chart_data = pd.DataFrame({
                    'min': df_min['net'],
                    best_seed_label: df_best_seed['net'],
                    'max': df_max['net']
                })
                chart_data.index = tickerData.index
                st.line_chart(chart_data)
                
                st.markdown("---")
                st.markdown(f"🧬 **Calculation DNA**")
                st.markdown(f"เส้น Net Profit ของ `{best_seed_label}` ถูกคำนวณแบบ **สะสมต่อเนื่อง** โดยใช้ Action ที่มาจากลำดับ Seed ดังนี้:")
                st.code(seed_sequence_str, language='text')
                st.markdown("---")

                st.write('💰 **Burn_Cash (จากกรณี Min)**')
                df_min.index = tickerData.index
                st.line_chart(df_min[['buffer']].cumsum())
                
                if window_details:
                    st.write("🔍 **การวิเคราะห์เพิ่มเติมจาก Best Seed Run**")
                    df_windows = pd.DataFrame(window_details)
                    st.bar_chart(df_windows.set_index('window_number')['max_net'], y_label="Net Profit", use_container_width=True)
                    st.bar_chart(df_windows.set_index('window_number')['price_change_pct'], y_label="Price Change %", use_container_width=True)
                    
                    st.write("🌱 **Seeds ที่เลือกใช้ในแต่ละ Window**")
                    seeds_df = df_windows[['window_number', 'timeline', 'previous_best_seed', 'best_seed', 'max_net']].copy()
                    seeds_df['previous_best_seed'] = seeds_df['previous_best_seed'].apply(lambda x: 'N/A' if pd.isna(x) else int(x))
                    seeds_df.columns = ['Window', 'Timeline', 'Prev. Seed', 'Selected Seed', 'Net Profit']
                    st.dataframe(seeds_df, use_container_width=True)
                    
                    csv = df_windows.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Window Details (CSV)", csv, f'best_seed_results_{test_ticker}.csv', 'text/csv')
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    main()
# <<< END OF MODIFICATION >>>
