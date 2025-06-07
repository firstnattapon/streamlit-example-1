import pandas as pd
import numpy as np
from numba import njit
import yfinance as yf
import streamlit as st

# --- Page Configuration ---
st.set_page_config(page_title="Best Seed Sliding Window", page_icon="🎯", layout="wide")

# --- Core Calculation Functions (Optimized) ---

@njit(fastmath=True)
def calculate_optimized(action_list: np.ndarray, price_list: np.ndarray, fix: float = 1500.0):
    """
    Optimized calculation engine using Numba.
    Receives and returns NumPy arrays for maximum speed.
    """
    # Ensure action_list is a mutable NumPy array of the correct type
    action_array = action_list.copy().astype(np.int32)
    action_array[0] = 1 # Force first action to be 1
    price_array = price_list.astype(np.float64)
    
    n = len(action_array)
    if n == 0:
        # Handle empty arrays to avoid errors
        return (np.empty(0, dtype=np.float64),) * 6

    # Pre-allocate arrays with the correct dtype
    amount = np.empty(n, dtype=np.float64)
    buffer = np.empty(n, dtype=np.float64)
    cash = np.empty(n, dtype=np.float64)
    asset_value = np.empty(n, dtype=np.float64)
    sumusd = np.empty(n, dtype=np.float64)

    # Calculate initial values at index 0
    initial_price = price_array[0]
    amount[0] = fix / initial_price
    buffer[0] = 0.0
    cash[0] = fix
    asset_value[0] = fix # amount[0] * initial_price is just 'fix'
    sumusd[0] = 2 * fix # cash[0] + asset_value[0]

    # Vectorized calculation for 'refer'
    # Formula: refer = -fix * ln(p0/pi) = fix * ln(pi/p0)
    refer = fix * np.log(price_array / initial_price)

    # Main loop with streamlined logic
    for i in range(1, n):
        # First, copy the previous state
        amount[i] = amount[i-1]
        cash[i] = cash[i-1]
        
        # Then, update if action is 1
        if action_array[i] == 1:
            buffer[i] = amount[i-1] * price_array[i] - fix
            cash[i] += buffer[i]
            amount[i] = fix / price_array[i]
        else: # action is 0
            buffer[i] = 0.0
            
        # These calculations are the same for both actions
        asset_value[i] = amount[i] * price_array[i]
        sumusd[i] = cash[i] + asset_value[i]

    return buffer, sumusd, cash, asset_value, amount, refer

@njit(fastmath=True)
def get_max_action(prices: np.ndarray, fix: float = 1500.0):
    """
    Calculates the theoretical maximum profit action sequence using Dynamic Programming.
    Optimized with Numba JIT for significant speedup on the O(n^2) complexity.
    """
    n = len(prices)
    if n < 2:
        return np.ones(n, dtype=np.int32)

    # --- Part 1: Forward calculation (DP) ---
    dp = np.zeros(n, dtype=np.float64)
    path = np.zeros(n, dtype=np.int32) # Stores the index 'j' of the best previous action
    
    initial_capital = 2.0 * fix
    dp[0] = initial_capital
    
    for i in range(1, n):
        max_prev_sumusd = -np.inf # Use -inf to correctly find the max
        best_j = 0
        for j in range(i):
            # Profit from a single trade from j to i
            profit_from_j_to_i = fix * (prices[i] / prices[j] - 1.0)
            current_sumusd = dp[j] + profit_from_j_to_i
            
            if current_sumusd > max_prev_sumusd:
                max_prev_sumusd = current_sumusd
                best_j = j
        
        dp[i] = max_prev_sumusd
        path[i] = best_j

    # --- Part 2: Backtracking to build the action array ---
    actions = np.zeros(n, dtype=np.int32)
    
    # 1. Find the end of the best path (day with the highest sumusd)
    last_action_day = np.argmax(dp)
    
    # 2. Backtrack from the end to the beginning
    current_day = last_action_day
    while current_day > 0:
        actions[current_day] = 1
        current_day = path[current_day]
        
    # 3. The first action is always 1
    actions[0] = 1
    
    return actions


def find_best_seed_sliding_window(prices: np.ndarray, dates_index: pd.DatetimeIndex, window_size: int, num_seeds_to_try: int, progress_bar=None):
    """
    Finds the best action sequence by finding the best seed for each sliding window.
    """
    n = len(prices)
    final_actions = np.array([], dtype=np.int32)
    window_details = []
    
    # Use integer division for correct window count
    num_windows = (n + window_size - 1) // window_size
    st.write(f"🔍 Starting Best Seed Search (Sliding Window)")
    st.write(f"📊 Total Data: {n} days | Window Size: {window_size} days | Number of Windows: {num_windows}")
    st.write("---")

    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        window_len = len(prices_window)
        
        if window_len == 0:
            continue

        # Get dates for timeline display
        start_date = dates_index[start_index].strftime('%Y-%m-%d')
        end_date = dates_index[end_index-1].strftime('%Y-%m-%d')
        timeline_info = f"{start_date} to {end_date}"

        best_seed_for_window = -1
        max_net_for_window = -np.inf

        # Generate all random seeds at once
        random_seeds = np.random.randint(0, 10_000_000, size=num_seeds_to_try)

        for seed in random_seeds:
            rng = np.random.default_rng(seed)
            actions_window = rng.integers(0, 2, size=window_len, dtype=np.int32)
            actions_window[0] = 1

            if window_len < 2:
                final_net = 0.0
            else:
                _, sumusd, _, _, _, refer = calculate_optimized(actions_window, prices_window)
                initial_capital = sumusd[0]
                net = sumusd - refer - initial_capital
                final_net = net[-1]

            if final_net > max_net_for_window:
                max_net_for_window = final_net
                best_seed_for_window = seed

        # Re-generate the best action sequence for the window
        rng_best = np.random.default_rng(best_seed_for_window)
        best_actions_for_window = rng_best.integers(0, 2, size=window_len, dtype=np.int32)
        best_actions_for_window[0] = 1

        window_detail = {
            'window_number': i + 1, 'timeline': timeline_info, 'start_index': start_index,
            'end_index': end_index - 1, 'window_size': window_len, 'best_seed': best_seed_for_window,
            'max_net': round(max_net_for_window, 2), 'start_price': round(prices_window[0], 2),
            'end_price': round(prices_window[-1], 2),
            'price_change_pct': round(((prices_window[-1] / prices_window[0]) - 1) * 100, 2),
            'action_count': int(np.sum(best_actions_for_window)),
        }
        window_details.append(window_detail)
        
        # Display progress for the current window
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Best Seed", f"{best_seed_for_window:,}")
        col2.metric("Net Profit", f"{max_net_for_window:,.2f}")
        col3.metric("Price Change", f"{window_detail['price_change_pct']:.2f}%")
        col4.metric("Actions Count", f"{window_detail['action_count']}/{window_len}")

        final_actions = np.concatenate((final_actions, best_actions_for_window))

        if progress_bar:
            progress_bar.progress((i + 1) / num_windows, text=f"Processing Window {i+1}/{num_windows}")

    return final_actions, window_details

# --- Data Fetching and Simulation Runner ---

@st.cache_data(ttl="1h") # Cache data for 1 hour
def get_yfinance_data(ticker_symbol):
    """
    Fetches and prepares historical data from yfinance.
    Results are cached to prevent re-downloading on every run.
    """
    try:
        tickerData = yf.Ticker(ticker_symbol)
        df = tickerData.history(period='max')[['Close']]
        df.index = df.index.tz_convert(tz='Asia/Bangkok')
        filter_date = '2023-01-01 12:00:00+07:00'
        df = df[df.index >= filter_date]
        if df.empty:
            st.error(f"No data found for {ticker_symbol} after {filter_date}.")
            return None
        return df
    except Exception as e:
        st.error(f"Failed to download data for {ticker_symbol}: {e}")
        return None

def run_simulation(prices: np.ndarray, dates_index: pd.DatetimeIndex, actions: np.ndarray):
    """
    Runs the core calculation and returns a results DataFrame.
    """
    buffer, sumusd, cash, asset_value, amount, refer = calculate_optimized(actions, prices)
    initial_capital = sumusd[0] if len(sumusd) > 0 else 0
    
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
    }, index=dates_index)
    return df

# --- Main Streamlit App ---

def main():
    st.title("🎯 Best Seed Sliding Window Tester")
    st.write("A tool to test and analyze trading strategies using the Best Seed Sliding Window technique.")

    with st.sidebar:
        st.header("⚙️ Configuration")
        test_ticker = st.selectbox(
            "Select Ticker",
            ('FFWM', 'NEGG', 'RIVN', 'APLS', 'NVTS', 'QXO', 'RXRX', 'SOUN', 'META', 'TSLA')
        )
        window_size = st.number_input("Window Size (days)", min_value=10, max_value=120, value=30, step=5)
        num_seeds = st.number_input("Seeds per Window", min_value=100, max_value=10000, value=1000, step=100)

    if st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True):
        
        # 1. Get Data (from cache or download)
        ticker_df = get_yfinance_data(test_ticker)
        
        if ticker_df is not None:
            prices = ticker_df.Close.to_numpy(dtype=np.float64)
            dates = ticker_df.index
            st.session_state['last_run_ticker'] = test_ticker # Save for reuse

            # --- Run all simulations ---
            # Min Actions (always trade)
            actions_min = np.ones(len(prices), dtype=np.int32)
            df_min = run_simulation(prices, dates, actions_min)

            # Max Actions (theoretical best)
            with st.spinner("Calculating theoretical max profit (DP)..."):
                actions_max = get_max_action(prices)
            df_max = run_simulation(prices, dates, actions_max)

            # Best Seed Sliding Window
            progress_bar = st.progress(0, text="Starting Best Seed search...")
            actions_best_seed, window_details = find_best_seed_sliding_window(
                prices, dates, 
                window_size=window_size, 
                num_seeds_to_try=num_seeds, 
                progress_bar=progress_bar
            )
            progress_bar.empty()
            df_best_seed = run_simulation(prices, dates, actions_best_seed)
            st.session_state[f'window_details_{test_ticker}'] = window_details
            
            # --- Display Results ---
            st.divider()
            st.header("📈 Performance Comparison")
            
            # Net Profit Comparison Chart
            net_comparison_df = pd.DataFrame({
                'Min Actions (Buy Every Day)': df_min['net'],
                'Best Seed Sliding Window': df_best_seed['net'],
                'Max Actions (Theoretical)': df_max['net']
            })
            st.line_chart(net_comparison_df)
            
            # Burn Cash Comparison Chart
            burn_cash_df = pd.DataFrame({
                'Min Actions (Buy Every Day)': df_min['buffer'].cumsum(),
                'Best Seed Sliding Window': df_best_seed['buffer'].cumsum(),
                'Max Actions (Theoretical)': df_max['buffer'].cumsum()
            })
            st.write("💰 **Cumulative Buffer (Burn Cash)**")
            st.line_chart(burn_cash_df)
            
            # --- Detailed Analysis for Best Seed ---
            st.divider()
            st.header(f"🔬 Detailed Analysis for Best Seed ({test_ticker})")

            # Summary Metrics
            total_net = df_best_seed['net'].iloc[-1]
            total_actions = actions_best_seed.sum()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Final Net Profit", f"${total_net:,.2f}")
            col2.metric("Total Actions", f"{total_actions} / {len(actions_best_seed)}")
            col3.metric("Final SumUSD", f"${df_best_seed['sumusd'].iloc[-1]:,.2f}")
            
            # Window Details Table
            st.write("📋 **Window-by-Window Breakdown**")
            df_details = pd.DataFrame(window_details)
            df_display = df_details[['window_number', 'timeline', 'best_seed', 'max_net', 
                                   'price_change_pct', 'action_count', 'window_size']].copy()
            df_display.columns = ['Window', 'Timeline', 'Best Seed', 'Net Profit', 'Price Change %', 'Actions', 'Size']
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # Download Button
            csv = df_details.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Window Details (CSV)",
                data=csv,
                file_name=f'best_seed_{test_ticker}_{window_size}w_{num_seeds}s.csv',
                mime='text/csv'
            )

    # Display explanation expanders
    st.divider()
    with st.expander("📖 How does 'Best Seed Sliding Window' work?"):
        st.markdown("""
        It's a technique to find a robust action sequence by:
        1.  **Divide & Conquer**: The entire price history is divided into smaller, consecutive periods (e.g., 30 days), called "windows".
        2.  **Local Optimization**: For each individual window, the algorithm exhaustively searches for the best random `seed`. It does this by generating thousands of different action sequences (using different seeds) and calculating the profit for each one *only within that window*.
        3.  **Select Best Seed**: The `seed` that yields the highest net profit for that specific window is chosen as the "best seed" for that period.
        4.  **Combine Actions**: The optimal action sequence generated from the best seed of each window is appended together to form the final, full action sequence.

        **Why is this good?**
        -   **Adaptability**: The market behaves differently over time. This method allows the strategy to adapt to local trends (e.g., be more active in a volatile window and less active in a sideways window).
        -   **Robustness**: It avoids relying on a single "lucky" seed that might work well for the entire period but perform poorly in specific segments.
        """)

    with st.expander("⚙️ What do the parameters mean?"):
        st.markdown("""
        -   **Window Size**: Controls how long each "local" period is.
            -   *Smaller (e.g., 15-20 days)*: Reacts very quickly to market changes but might lead to over-trading.
            -   *Larger (e.g., 60-90 days)*: Creates a more stable, slower-adapting strategy.
        -   **Seeds per Window**: Controls the thoroughness of the search within each window.
            -   *Fewer (e.g., 100-500)*: Faster calculation but might miss the truly optimal seed.
            -   *More (e.g., 2000+)*: A more exhaustive search that is more likely to find the best seed, but takes longer to compute.
        """)


if __name__ == "__main__":
    main()
