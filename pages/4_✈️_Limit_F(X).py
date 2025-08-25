import pandas as pd
import numpy as np
from numba import njit
import yfinance as yf
import streamlit as st
import thingspeak
import json
import streamlit.components.v1 as components

# === CONFIG & INITIAL SETUP ===
st.set_page_config(page_title="Limit_F(X)", page_icon="✈️", layout="wide")

@st.cache_data
def load_config(path='limit_fx_config.json'):
    """Loads the asset configuration from a JSON file."""
    try:
        with open(path, 'r') as f:
            config = json.load(f)
        return config['assets']
    except FileNotFoundError:
        st.error(f"Configuration file '{path}' not found. Please ensure it exists.")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        st.error(f"Error reading or parsing '{path}': {e}")
        return None

ASSETS = load_config()
if not ASSETS:
    st.stop()

TICKERS = [a['symbol'] for a in ASSETS]
FILTER_DATE = '2024-01-01 12:00:00+07:00'
FIX_VALUE = 1500.0

# === DATA FETCHING & CORE CALCULATION FUNCTIONS ===

@st.cache_data(ttl=600)
def get_prices(tickers, start_date):
    """Fetches historical price data for a list of tickers."""
    df_list = []
    for ticker in tickers:
        try:
            data = yf.Ticker(ticker).history(period='max')[['Close']]
            if not data.empty:
                data.index = data.index.tz_convert(tz='Asia/Bangkok')
                data = data[data.index >= start_date]
                data = data.rename(columns={'Close': f"{ticker}_price"})
                df_list.append(data)
        except Exception as e:
            st.warning(f"Could not fetch data for {ticker}: {e}")
    if not df_list:
        return pd.DataFrame()
    return pd.concat(df_list, axis=1).ffill()

@st.cache_data(ttl=300)
def get_act_from_thingspeak(channel_id, api_key, field):
    """Fetches the last value from a ThingSpeak channel field."""
    try:
        client = thingspeak.Channel(channel_id, api_key, fmt='json')
        act_json = client.get_field_last(field=str(field))
        value = json.loads(act_json).get(f"field{field}")
        if value is None:
            st.warning(f"Field {field} on channel {channel_id} returned null. Using 0.")
            return 0
        return int(value)
    except Exception as e:
        st.error(f"ThingSpeak Error (Channel: {channel_id}, Field: {field}): {e}")
        return 0

@njit(fastmath=True)
def calculate_optimized(action_list, price_list, fix=FIX_VALUE):
    action_array = np.asarray(action_list, dtype=np.int32)
    action_array[0] = 1
    price_array = np.asarray(price_list, dtype=np.float64)
    n = len(action_array)
    amount = np.empty(n, dtype=np.float64)
    buffer = np.zeros(n, dtype=np.float64)
    cash = np.empty(n, dtype=np.float64)
    asset_value = np.empty(n, dtype=np.float64)
    sumusd = np.empty(n, dtype=np.float64)
    if price_array.shape[0] == 0:
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

def get_max_action(price_list, fix=FIX_VALUE):
    prices = np.asarray(price_list, dtype=np.float64)
    n = len(prices)
    if n < 2:
        return np.ones(n, dtype=int)
    dp = np.zeros(n, dtype=np.float64)
    path = np.zeros(n, dtype=int)
    initial_capital = float(fix * 2)
    dp[0] = initial_capital
    for i in range(1, n):
        max_prev_sumusd = 0.0
        best_j = 0
        for j in range(i):
            profit_from_j_to_i = fix * ((prices[i] / prices[j]) - 1.0)
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

# Perfect Foresight envelope (max at each t) — wealth series
@njit(fastmath=True)
def perfect_foresight_wealth_series(prices, fix=FIX_VALUE):
    """
    Compute W_i via DP:
      W_i = max_{j < i} [ W_j + fix*(P_i/P_j - 1) ], with W_0 = 2*fix
    Then for each t:
      Wealth_pf[t] = max_{i <= t} [ W_i + fix*(P_t/P_i - 1) ]
    """
    n = len(prices)
    wealth_after_reb = np.empty(n, dtype=np.float64)
    wealth_after_reb[0] = 2.0 * fix
    for i in range(1, n):
        best = 0.0
        for j in range(i):
            val = wealth_after_reb[j] + fix * (prices[i] / prices[j] - 1.0)
            if val > best:
                best = val
        wealth_after_reb[i] = best
    wealth_pf = np.empty(n, dtype=np.float64)
    for t in range(n):
        best_t = 0.0
        for i in range(t + 1):
            val = wealth_after_reb[i] + fix * (prices[t] / prices[i] - 1.0)
            if val > best_t:
                best_t = val
        wealth_pf[t] = best_t
    return wealth_pf

@st.cache_data(ttl=600)
def Limit_fx(Ticker, start_date=FILTER_DATE, act=-1):
    """Main calculation engine for a single ticker."""
    try:
        tickerData = yf.Ticker(Ticker).history(period='max')[['Close']]
        if tickerData.empty: return pd.DataFrame()
        tickerData.index = tickerData.index.tz_convert(tz='Asia/Bangkok')
        tickerData = tickerData[tickerData.index >= start_date]
        prices = tickerData.Close.values.astype(np.float64)
    except Exception as e:
        st.warning(f"Could not get yfinance data for {Ticker}: {e}")
        return pd.DataFrame()

    if len(prices) == 0: return pd.DataFrame()

    if act == -1:
        # always buy
        actions = np.ones(len(prices), dtype=np.int64)
        buffer, sumusd, cash, asset_value, amount, refer = calculate_optimized(actions, prices)
        initial_capital = sumusd[0]
        df = pd.DataFrame({
            'price': prices, 'action': actions, 'buffer': buffer, 'sumusd': sumusd,
            'cash': cash, 'asset_value': asset_value, 'amount': amount,
            'refer': refer + initial_capital,
        }, index=tickerData.index)
        # FIX: start net at 0
        df['net'] = df['sumusd'] - df['refer']
        return df

    elif act == -2:
        # legacy "max actions to end"
        actions = get_max_action(prices)
        buffer, sumusd, cash, asset_value, amount, refer = calculate_optimized(actions, prices)
        initial_capital = sumusd[0] if len(sumusd) > 0 else 0.0
        df = pd.DataFrame({
            'price': prices, 'action': actions, 'buffer': buffer, 'sumusd': sumusd,
            'cash': cash, 'asset_value': asset_value, 'amount': amount,
            'refer': refer + initial_capital,
        }, index=tickerData.index)
        # FIX: start net at 0
        df['net'] = df['sumusd'] - df['refer']
        return df

    elif act == -3:
        # Perfect Foresight envelope (true max_net each day)
        wealth_pf = perfect_foresight_wealth_series(prices, FIX_VALUE)
        initial_price = prices[0]
        refer = -FIX_VALUE * np.log(initial_price / prices)
        initial_capital = wealth_pf[0]  # 2*fix
        n = len(prices)
        df = pd.DataFrame({
            'price': prices,
            'action': np.zeros(n, dtype=np.int64),
            'buffer': np.zeros(n, dtype=np.float64),
            'sumusd': wealth_pf,
            'cash': np.nan,
            'asset_value': np.nan,
            'amount': np.nan,
            'refer': refer + initial_capital
        }, index=tickerData.index)
        # FIX: start net at 0
        df['net'] = df['sumusd'] - df['refer']
        return df

    else:
        # random seed mode
        rng = np.random.default_rng(act)
        actions = rng.integers(0, 2, len(prices))
        buffer, sumusd, cash, asset_value, amount, refer = calculate_optimized(actions, prices)
        initial_capital = sumusd[0] if len(sumusd) > 0 else 0.0
        df = pd.DataFrame({
            'price': prices, 'action': actions, 'buffer': buffer, 'sumusd': sumusd,
            'cash': cash, 'asset_value': asset_value, 'amount': amount,
            'refer': refer + initial_capital,
        }, index=tickerData.index)
        # FIX: start net at 0
        df['net'] = df['sumusd'] - df['refer']
        return df

# === ANALYSIS & DATA PREPARATION FUNCTIONS ===

@st.cache_data
def prepare_base_data(tickers):
    """Pre-calculates and caches base dataframes (min, max) for all tickers."""
    base_dfs = {}
    for ticker in tickers:
        df_min = Limit_fx(ticker, act=-1)
        if not df_min.empty:
            base_dfs[ticker] = {
                'min': df_min,
                'max_legacy': Limit_fx(ticker, act=-2),
                'max_pf': Limit_fx(ticker, act=-3),
            }
    return base_dfs

def align_ticker_data(base_dfs, column_name):
    """Aligns a specific column from all base dataframes for combined analysis."""
    dfs_to_align = []
    for symbol, data in base_dfs.items():
        if not data['min'].empty:
            renamed_df = data['min'][[column_name]].rename(columns={column_name: f'{column_name}_{symbol}'})
            dfs_to_align.append(renamed_df)
    if not dfs_to_align:
        return pd.DataFrame()
    return pd.concat(dfs_to_align, axis=1).ffill().dropna()

def calculate_max_drawdown(series: pd.Series) -> float:
    """Calculates the maximum drawdown (peak-to-trough)."""
    if series.empty or series.isnull().all():
        return 0.0
    running_max = series.cummax()
    drawdown = series - running_max  # negative or zero
    return abs(drawdown.min())

@st.cache_data
def generate_ref_log_data(base_dfs, prices_df):
    """Generates the combined dataframe for the Ref_index_Log tab."""
    if prices_df.empty: return pd.DataFrame()

    sumusd_aligned_df = align_ticker_data(base_dfs, 'sumusd')
    if sumusd_aligned_df.empty: return pd.DataFrame()
    
    combined_df = pd.concat([prices_df, sumusd_aligned_df], axis=1).ffill().dropna()

    price_cols = [col for col in combined_df.columns if '_price' in col]
    sumusd_cols = [col for col in combined_df.columns if 'sumusd_' in col]
    
    if not price_cols or not sumusd_cols: return pd.DataFrame()

    initial_capital_per_asset = FIX_VALUE * 2
    initial_capital_ref_index = initial_capital_per_asset * len(price_cols)

    # Vectorized calculation for 'ref_log'
    price_prod = combined_df[price_cols].prod(axis=1)
    int_st = price_prod.iloc[0]
    safe_ratio = int_st / price_prod.replace(0, np.nan)
    combined_df['ref_log'] = initial_capital_ref_index + (-FIX_VALUE * np.log(safe_ratio))
    combined_df['ref_log'] = combined_df['ref_log'].fillna(initial_capital_ref_index)
    
    combined_df['daily_sumusd'] = combined_df[sumusd_cols].sum(axis=1)
    
    total_initial_capital = sum(
        base_dfs[s]['min'].sumusd.iloc[0] for s in base_dfs if not base_dfs[s]['min'].empty
    )

    net_raw = combined_df['daily_sumusd'] - combined_df['ref_log'] - total_initial_capital
    net_at_index_0 = net_raw.iloc[0] if not net_raw.empty else 0
    combined_df['net'] = net_raw - net_at_index_0
    
    return combined_df

@st.cache_data
def generate_burn_cash_data(base_dfs):
    """Generates the combined dataframe for the Burn_Cash tab."""
    buffer_aligned_df = align_ticker_data(base_dfs, 'buffer')
    if buffer_aligned_df.empty:
        return pd.DataFrame()
    
    buffer_aligned_df['daily_burn'] = buffer_aligned_df.sum(axis=1)
    buffer_aligned_df['cumulative_burn'] = buffer_aligned_df['daily_burn'].cumsum()
    return buffer_aligned_df

# === UI FUNCTIONS ===
def plot_individual_asset(symbol, act, base_dfs):
    """Displays charts and data for a single asset."""
    df_min = base_dfs[symbol]['min']
    df_max_pf = base_dfs[symbol]['max_pf']   # true PF envelope
    df_fx = Limit_fx(symbol, act=act)

    if df_min.empty or df_fx.empty or df_max_pf.empty:
        st.error(f"Could not generate plot for {symbol} due to missing data.")
        return

    chart_data = pd.DataFrame({
        'min_net': df_min.net,
        f'fx_{act}_net': df_fx.net,
        'max_net': df_max_pf.net
    })
    st.write('Refer_Log Net Performance')
    st.line_chart(chart_data)

    df_plot_burn = df_min[['buffer']].cumsum()
    st.write('Burn_Cash (Cumulative)')
    st.line_chart(df_plot_burn)

    with st.expander("Detailed Data (Min Action)"):
        st.dataframe(df_min)

def iframe(frame='', width=1500, height=800):
    """Embeds an iframe component."""
    components.iframe(frame, width=width, height=height, scrolling=True)

# === MAIN APP LAYOUT ===

# --- Pre-calculate all base data ---
base_dataframes = prepare_base_data(TICKERS)
if not base_dataframes:
    st.error("Failed to fetch or calculate data for all assets. Stopping.")
    st.stop()
    
# --- Create Tabs ---
tab_names = TICKERS + ['Burn_Cash', 'Ref_index_Log', 'cf_log']
tabs = st.tabs(tab_names)
tab_dict = dict(zip(tab_names, tabs))

# --- Individual Asset Tabs ---
for asset in ASSETS:
    symbol = asset['symbol']
    if symbol in tab_dict and symbol in base_dataframes:
        with tab_dict[symbol]:
            act = get_act_from_thingspeak(
                channel_id=asset['channel_id'],
                api_key=asset['write_api_key'],
                field=asset['field']
            )
            plot_individual_asset(symbol, act, base_dataframes)

# --- Ref_index_Log Tab ---
with tab_dict['Ref_index_Log']:
    prices_df = get_prices(list(base_dataframes.keys()), FILTER_DATE)
    ref_log_df = generate_ref_log_data(base_dataframes, prices_df)

    if not ref_log_df.empty:
        st.header("Net Performance Analysis (vs. Reference)")
        st.info("Performance analysis of the portfolio's net value against the logarithmic reference index.")

        net_series = ref_log_df['net']
        daily_changes = net_series.diff()
        
        analysis_type = st.radio(
            "Select Cash Flow (CF) Analysis Type:",
            ('Worst Case', 'Average Case'),
            horizontal=True,
            key='cf_analysis_type'
        )

        if analysis_type == 'Worst Case':
            metric_1d_val = daily_changes.min()
            label_1d = "📉 1-Day CF (Worst Day)"
        else:
            metric_1d_val = daily_changes.mean()
            label_1d = "📊 1-Day CF (Average Day)"
        
        trough_to_peak_gain = 0
        if not net_series.empty:
            trough_index = net_series.idxmin()
            peak_after_trough = net_series.loc[trough_index:].max()
            trough_value = net_series.loc[trough_index]
            trough_to_peak_gain = peak_after_trough - trough_value

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label=label_1d, value=f"{metric_1d_val:,.2f} USD")
        with col2:
            st.metric(label="📈 Trough-to-Peak Gain (Max Run-up)", value=f"{trough_to_peak_gain:,.2f} USD")

        st.markdown("---")
        st.subheader("Net Performance Over Time")
        st.line_chart(ref_log_df['net'])
        with st.expander("View Data"):
            st.dataframe(ref_log_df)
    else:
        st.warning("Could not generate Ref_index_Log data.")

# --- Burn_Cash Tab ---
with tab_dict['Burn_Cash']:
    burn_cash_df = generate_burn_cash_data(base_dataframes)
    
    if not burn_cash_df.empty:
        st.header("Cash Burn Risk Analysis")
        st.info("Based on a backtest using an 'always buy' strategy (act=-1) to assess maximum potential risk.")
        
        cumulative_burn_series = burn_cash_df['cumulative_burn']
        max_daily_burn = burn_cash_df['daily_burn'].min()
        max_drawdown_burn = calculate_max_drawdown(cumulative_burn_series)
        
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
            st.metric(
                label="🏔️ Max Drawdown (Peak to Trough)", 
                value=f"{-max_drawdown_burn:,.2f} USD",
                help="The largest drop in cumulative cash from any peak to a subsequent trough."
            )

        st.markdown("---")
        st.subheader("Cumulative Cash Burn Over Time")
        st.line_chart(cumulative_burn_series)
        
        with st.expander("View Detailed Burn Data"):
            st.dataframe(burn_cash_df)
    else:
        st.error("Cannot calculate burn cash due to missing data.")

# --- cf_log Tab ---
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
