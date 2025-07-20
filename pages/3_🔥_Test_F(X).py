import pandas as pd
import numpy as np
from numba import njit
import yfinance as yf
import streamlit as st
import thingspeak
import json
import streamlit.components.v1 as components

st.set_page_config(page_title="Limit_F(X)", page_icon="✈️", layout="wide")

# === CONFIG LOADING ===
@st.cache_data
def load_config(path='limit_fx_config.json'):
    """Loads the asset configuration from a JSON file."""
    try:
        with open(path, 'r') as f:
            config = json.load(f)
        return config['assets']
    except FileNotFoundError:
        st.error(f"Configuration file '{path}' not found. Please ensure it exists in the correct directory.")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        st.error(f"Error reading or parsing '{path}': {e}")
        return None

ASSETS = load_config()
if not ASSETS:
    st.stop()

TICKERS = [a['symbol'] for a in ASSETS]


# === DATA FETCHING & CALCULATION FUNCTIONS ===

@st.cache_data(ttl=600)
def get_prices(tickers, start_date):
    """Fetches historical price data for a list of tickers."""
    df_list = []
    for ticker in tickers:
        try:
            tickerData = yf.Ticker(ticker)
            tickerHist = tickerData.history(period='max')[['Close']]
            if not tickerHist.empty:
                tickerHist.index = tickerHist.index.tz_convert(tz='Asia/Bangkok')
                tickerHist = tickerHist[tickerHist.index >= start_date]
                tickerHist = tickerHist.rename(columns={'Close': f"{ticker}_price"})
                df_list.append(tickerHist)
        except Exception as e:
            st.warning(f"Could not fetch data for {ticker}: {e}")
    if not df_list:
        return pd.DataFrame()
    return pd.concat(df_list, axis=1)

@st.cache_data(ttl=300)
def get_act_from_thingspeak(channel_id, api_key, field):
    """Fetches the last value from a specific field in a specific ThingSpeak channel."""
    try:
        client = thingspeak.Channel(channel_id, api_key, fmt='json')
        act_json = client.get_field_last(field=str(field))
        value = json.loads(act_json).get(f"field{field}")
        if value is None:
            st.warning(f"Field {field} on channel {channel_id} returned null. Using default value 0.")
            return 0
        return int(value)
    except Exception as e:
        st.error(f"Could not fetch data from ThingSpeak (Channel: {channel_id}, Field: {field}). Error: {e}")
        return 0

@njit(fastmath=True)
def calculate_optimized(action_list, price_list, fix=1500):
    action_array = np.asarray(action_list, dtype=np.int32)
    action_array[0] = 1
    price_array = np.asarray(price_list, dtype=np.float64)
    n = len(action_array)
    amount = np.empty(n, dtype=np.float64)
    buffer = np.zeros(n, dtype=np.float64)
    cash = np.empty(n, dtype=np.float64)
    asset_value = np.empty(n, dtype=np.float64)
    sumusd = np.empty(n, dtype=np.float64)
    if price_array.shape[0] == 0: # Guard against empty price array
        return buffer, sumusd, cash, asset_value, amount, np.empty(0, dtype=np.float64)
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


def get_max_action(price_list, fix=1500):
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

@st.cache_data(ttl=600)
def Limit_fx(Ticker, act=-1):
    filter_date = '2024-01-01 12:00:00+07:00'
    try:
        tickerData = yf.Ticker(Ticker)
        tickerData = tickerData.history(period='max')[['Close']]
        if tickerData.empty:
            return pd.DataFrame()
        tickerData.index = tickerData.index.tz_convert(tz='Asia/Bangkok')
        tickerData = tickerData[tickerData.index >= filter_date]
        prices = np.array(tickerData.Close.values, dtype=np.float64)
    except Exception as e:
        st.warning(f"Could not get yfinance data for {Ticker}: {e}")
        return pd.DataFrame()

    if len(prices) == 0:
        return pd.DataFrame()

    if act == -1:
        actions = np.ones(len(prices), dtype=np.int64)
    elif act == -2:
        actions = get_max_action(prices)
    else:
        rng = np.random.default_rng(act)
        actions = rng.integers(0, 2, len(prices))

    buffer, sumusd, cash, asset_value, amount, refer = calculate_optimized(actions, prices)
    initial_capital = sumusd[0]
    df = pd.DataFrame({
        'price': prices,
        'action': actions,
        'buffer': buffer,
        'sumusd': sumusd,
        'cash': cash,
        'asset_value': asset_value,
        'amount': amount,
        'refer': refer + initial_capital,
        'net': sumusd - refer - initial_capital
    }, index=tickerData.index)
    return df

# === UI FUNCTIONS ===
def plot(Ticker, act):
    df_min = Limit_fx(Ticker, act=-1)
    df_fx = Limit_fx(Ticker, act=act)
    df_max = Limit_fx(Ticker, act=-2)

    if df_min.empty or df_fx.empty or df_max.empty:
        st.error(f"Could not generate plot for {Ticker} due to missing data.")
        return

    chart_data = pd.DataFrame({
        'min': df_min.net,
        f'fx_{act}': df_fx.net,
        'max': df_max.net
    }, index=df_min.index)
    st.write('Refer_Log')
    st.line_chart(chart_data)

    df_plot_burn = df_min[['buffer']].cumsum()
    st.write('Burn_Cash (Cumulative)')
    st.line_chart(df_plot_burn)

    with st.expander("Detailed Data (Min Action)"):
        st.dataframe(df_min)

def iframe(frame='', width=1500, height=800):
    components.iframe(frame, width=width, height=height, scrolling=True)

# === MAIN APP LAYOUT ===
tab_names = TICKERS + ['Burn_Cash', 'Ref_index_Log', 'cf_log']
tabs = st.tabs(tab_names)
tab_dict = dict(zip(tab_names, tabs))

# === MAIN ASSET TABS ===
for asset in ASSETS:
    symbol = asset['symbol']
    with tab_dict[symbol]:
        act = get_act_from_thingspeak(
            channel_id=asset['channel_id'],
            api_key=asset['write_api_key'],
            field=asset['field']
        )
        plot(symbol, act)

# === REF_INDEX_LOG TAB (FIXED) ===
with tab_dict['Ref_index_Log']:
    filter_date = '2023-01-01 12:00:00+07:00'
    prices_df = get_prices(TICKERS, filter_date)

    if not prices_df.empty:
        # Create uniquely named dataframes for concatenation
        dfs_to_align = []
        for symbol in TICKERS:
            df_temp = Limit_fx(symbol, act=-1)
            if not df_temp.empty:
                renamed_df = df_temp[['sumusd']].rename(columns={'sumusd': f'sumusd_{symbol}'})
                dfs_to_align.append(renamed_df)
        
        if dfs_to_align:
            aligned_dfs = [prices_df] + dfs_to_align
            df_sumusd_ = pd.concat(aligned_dfs, axis=1).ffill().dropna()

            price_cols = [col for col in df_sumusd_.columns if '_price' in col]
            sumusd_cols = [col for col in df_sumusd_.columns if 'sumusd_' in col]
            
            if not price_cols or not sumusd_cols:
                 st.warning("Could not find price or sumusd columns after alignment.")
            else:
                int_st = np.prod(df_sumusd_.iloc[0][price_cols])
                initial_capital_per_stock = 3000
                initial_capital_Ref_index_Log = initial_capital_per_stock * len(TICKERS)

                def calculate_ref_log(row):
                    int_end = np.prod(row[price_cols])
                    if int_st == 0 or int_end == 0: return initial_capital_Ref_index_Log
                    return initial_capital_Ref_index_Log + (-1500 * np.log(int_st / int_end))

                df_sumusd_['ref_log'] = df_sumusd_.apply(calculate_ref_log, axis=1)
                df_sumusd_['daily_sumusd'] = df_sumusd_[sumusd_cols].sum(axis=1)

                total_initial_capital = sum([Limit_fx(symbol, act=-1).sumusd.iloc[0] for symbol in TICKERS if not Limit_fx(symbol, act=-1).empty])
                net_raw = df_sumusd_['daily_sumusd'] - df_sumusd_['ref_log'] - total_initial_capital
                net_at_index_0 = net_raw.iloc[0] if not net_raw.empty else 0
                df_sumusd_['net'] = net_raw - net_at_index_0
                
                # <<<--- START OF MODIFICATION ---<<<
                st.header("Net Performance Analysis (vs. Reference)")
                st.info("Performance analysis of the portfolio's net value against the logarithmic reference index. 'Worst Case' indicates maximum losses, while 'Average Case' shows the mean performance. 'Trough-to-Peak' shows the maximum possible gain from a low point.")

                net_series = df_sumusd_['net']
                
                # Pre-calculate change series to avoid re-computation
                daily_changes = net_series.diff()
                
                rolling_30_day_change = pd.Series(dtype=np.float64)
                if len(net_series) >= 30:
                    rolling_30_day_change = net_series.rolling(window=30).apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)

                rolling_90_day_change = pd.Series(dtype=np.float64)
                if len(net_series) >= 90:
                    rolling_90_day_change = net_series.rolling(window=90).apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)

                # Add the radio button for selection
                analysis_type = st.radio(
                    "Select Cash Flow (CF) Analysis Type:",
                    ('Worst Case', 'Average Case'),
                    horizontal=True,
                    key='cf_analysis_type'
                )

                # --- Metric Calculation based on selection ---
                if analysis_type == 'Worst Case':
                    metric_1d_val = daily_changes.min()
                    metric_30d_val = rolling_30_day_change.min() if not rolling_30_day_change.empty else 0
                    metric_90d_val = rolling_90_day_change.min() if not rolling_90_day_change.empty else 0
                    
                    label_1d = "📉 1-Day CF (Worst Day)"
                    label_30d = "📉 30-Day CF (Worst Month)"
                    label_90d = "📉 90-Day CF (Worst Quarter)"
                else: # Average Case
                    metric_1d_val = daily_changes.mean()
                    metric_30d_val = rolling_30_day_change.mean() if not rolling_30_day_change.empty else 0
                    metric_90d_val = rolling_90_day_change.mean() if not rolling_90_day_change.empty else 0

                    label_1d = "📊 1-Day CF (Average Day)"
                    label_30d = "📊 30-Day CF (Average Month)"
                    label_90d = "📊 90-Day CF (Average Quarter)"

                # Handle potential NaN values
                if pd.isna(metric_1d_val): metric_1d_val = 0
                if pd.isna(metric_30d_val): metric_30d_val = 0
                if pd.isna(metric_90d_val): metric_90d_val = 0

                # Trough-to-Peak Gain (Max Run-up) - This is a "max" metric and remains constant
                trough_to_peak_gain = 0
                if not net_series.empty:
                    trough_index = net_series.idxmin()
                    peak_after_trough = net_series.loc[trough_index:].max()
                    trough_value = net_series.loc[trough_index]
                    if pd.notna(peak_after_trough) and pd.notna(trough_value):
                            trough_to_peak_gain = peak_after_trough - trough_value
                    else:
                            trough_to_peak_gain = 0

                # --- Display Metrics ---
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Short-Term CF")
                    st.metric(label=label_1d, value=f"{metric_1d_val:,.2f} USD")
                    st.metric(label=label_30d, value=f"{metric_30d_val:,.2f} USD")

                with col2:
                    st.subheader("Medium to Long-Term CF")
                    st.metric(label=label_90d, value=f"{metric_90d_val:,.2f} USD")
                    st.metric(label="📈 Trough-to-Peak Gain (Max Run-up)", value=f"{trough_to_peak_gain:,.2f} USD")

                st.markdown("---")
                
                st.subheader("Net Performance Over Time")
                st.line_chart(df_sumusd_['net'])
                with st.expander("View Data"):
                    st.dataframe(df_sumusd_)
                # >>>--- END OF MODIFICATION ---<<<
        else:
             st.warning("Could not align dataframes. Not enough data available for the selected assets.")
    else:
        st.warning("Could not fetch sufficient price data for Ref_index_Log.")

# === BURN_CASH TAB ===
with tab_dict['Burn_Cash']:
    # Create uniquely named dataframes for concatenation
    dfs_to_align = []
    for symbol in TICKERS:
        df_temp = Limit_fx(symbol, act=-1)
        if not df_temp.empty:
            renamed_df = df_temp[['buffer']].rename(columns={'buffer': f'buffer_{symbol}'})
            dfs_to_align.append(renamed_df)
    
    if not dfs_to_align:
        st.error("Cannot calculate burn cash due to missing data for all assets.")
    else:
        df_burn_cash = pd.concat(dfs_to_align, axis=1).ffill().dropna()

        df_burn_cash['daily_burn'] = df_burn_cash.sum(axis=1)
        df_burn_cash['cumulative_burn'] = df_burn_cash['daily_burn'].cumsum()
        
        st.header("Cash Burn Risk Analysis")
        st.info("Based on a backtest using an 'always buy' strategy (act=-1) to assess maximum potential risk.")
        
        # --- Risk Calculation ---
        max_daily_burn = df_burn_cash['daily_burn'].min()
        cumulative_burn_series = df_burn_cash['cumulative_burn']
        
        peak_to_trough_burn = 0
        if not cumulative_burn_series.empty:
            peak_index = cumulative_burn_series.idxmax()
            peak_to_trough_burn = cumulative_burn_series.loc[peak_index] - cumulative_burn_series.loc[peak_index:].min()

        max_30_day_burn = 0
        if len(cumulative_burn_series) >= 30:
            rolling_30_day_change = cumulative_burn_series.rolling(window=30).apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)
            max_30_day_burn = rolling_30_day_change.min()
        
        max_90_day_burn = 0
        if len(cumulative_burn_series) >= 90:
            rolling_90_day_change = cumulative_burn_series.rolling(window=90).apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)
            max_90_day_burn = rolling_90_day_change.min()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Short-Term Risk")
            st.metric(label="🔥 1-Day Burn (Worst Day)", value=f"{max_daily_burn:,.2f} USD")
            st.metric(label="🔥 30-Day Burn (Worst Month)", value=f"{max_30_day_burn:,.2f} USD")
        
        with col2:
            st.subheader("Medium to Long-Term Risk")
            st.metric(label="🔥 90-Day Burn (Worst Quarter)", value=f"{max_90_day_burn:,.2f} USD")
            st.metric(label="🏔️ Peak-to-Trough Burn (Max Drawdown)", value=f"{peak_to_trough_burn:,.2f} USD")

        st.markdown("---")
        
        st.subheader("Cumulative Cash Burn Over Time")
        st.line_chart(df_burn_cash['cumulative_burn'])
        
        with st.expander("View Detailed Burn Data"):
            st.dataframe(df_burn_cash)

# === CF_LOG TAB ===
with tab_dict['cf_log']:
    st.markdown("""
    - **Rebalance**: `-fix * ln(t0 / tn)`
    - **Net Profit**: `sumusd - refer - sumusd[0]` (ต้นทุนเริ่มต้น)
    - **Ref_index_Log**: `initial_capital_Ref_index_Log + (-1500 * ln(int_st / int_end))`
    - **Net in Ref_index_Log**: `(daily_sumusd - ref_log - total_initial_capital) - net_at_index_0`
    - **Option P/L**: `(max(0, ราคาหุ้นปัจจุบัน - Strike) * contracts_or_shares) - (contracts_or_shares * premium_paid_per_share)`
    ---
    """)
    iframe("https://monica.im/share/artifact?id=Su47FeHfaWtyXmqDqmqp9W")
    st.markdown("---")
    iframe("https://monica.im/share/artifact?id=qpAkuKjBpuVz2cp9nNFRs3")
    st.markdown("---")
    iframe("https://monica.im/share/artifact?id=wEjeaMxVW6MgDDm3xAZatX")
    st.markdown("---")
    iframe("https://monica.im/share/artifact?id=ZfHT5iDP2Ypz82PCRw9nEK")
    st.markdown("---")
    iframe("https://monica.im/share/chat?shareId=SUsEYhzSMwqIq3Cx")
