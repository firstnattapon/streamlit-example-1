import pandas as pd
import numpy as np
from numba import njit
import yfinance as yf
import streamlit as st
import thingspeak
import json
import streamlit.components.v1 as components

st.set_page_config(page_title="Limit_F(X)", page_icon="✈️", layout="wide")

# === CONFIG & CONSTANTS ===
@st.cache_data
def load_config(path='limit_fx_config.json'):
    """Loads the entire configuration from a JSON file."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Configuration file '{path}' not found. Please ensure it exists.")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        st.error(f"Error reading or parsing '{path}': {e}")
        return None

CONFIG = load_config()
if not CONFIG:
    st.stop()

ASSETS = CONFIG.get('assets', [])
GLOBAL_SETTINGS = CONFIG.get('global_settings', {})
CF_LOG_SETTINGS = CONFIG.get('cf_log_settings', {})

if not ASSETS:
    st.error("No 'assets' found in the configuration file.")
    st.stop()

TICKERS = [a['symbol'] for a in ASSETS]
FILTER_DATE = GLOBAL_SETTINGS.get('filter_date', '2023-01-01 12:00:00+07:00')
FIX_VALUE = GLOBAL_SETTINGS.get('fix_value', 1500)
INITIAL_CAPITAL_PER_STOCK = GLOBAL_SETTINGS.get('initial_capital_per_stock', 3000)
IFRAME_URLS = CF_LOG_SETTINGS.get('iframe_urls', [])

# === CORE FUNCTIONS ===

@st.cache_data(ttl=600)
def get_prices(tickers, start_date):
    """Fetches historical price data for a list of tickers."""
    df_list = []
    for ticker in tickers:
        try:
            data = yf.Ticker(ticker).history(period='max', auto_adjust=True)[['Close']]
            if not data.empty:
                data.index = data.index.tz_convert(tz='Asia/Bangkok')
                data = data[data.index >= start_date]
                data = data.rename(columns={'Close': f"{ticker}_price"})
                df_list.append(data)
        except Exception as e:
            st.warning(f"Could not fetch data for {ticker}: {e}")
    return pd.concat(df_list, axis=1) if df_list else pd.DataFrame()

@st.cache_data(ttl=300)
def get_act_from_thingspeak(channel_id, api_key, field):
    """Fetches the last value from a specific field in a specific ThingSpeak channel."""
    try:
        client = thingspeak.Channel(channel_id, api_key, fmt='json')
        response = client.get_field_last(field=str(field))
        value = json.loads(response).get(f"field{field}")
        return int(value) if value is not None else 0
    except Exception:
        return 0

# <<<--- START OF FIX ---<<<
@njit(fastmath=True)
def calculate_optimized(action_list, price_list, fix):
    """
    Numba-optimized calculation function.
    Assumes price_list is not empty.
    """
    action_array = np.asarray(action_list, dtype=np.int32)
    action_array[0] = 1
    price_array = np.asarray(price_list, dtype=np.float64)
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
# >>>--- END OF FIX ---<<<


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
def Limit_fx(Ticker, act, fix_value):
    try:
        data = yf.Ticker(Ticker).history(period='max', auto_adjust=True)[['Close']]
        if data.empty: return pd.DataFrame()
        data.index = data.index.tz_convert(tz='Asia/Bangkok')
        data = data[data.index >= FILTER_DATE]
        prices = np.array(data.Close.values, dtype=np.float64)
    except Exception as e:
        st.warning(f"Could not get yfinance data for {Ticker}: {e}")
        return pd.DataFrame()

    # <<<--- START OF FIX ---<<<
    # Guard against empty prices *before* calling the Numba function
    if len(prices) == 0:
        return pd.DataFrame()

    if act == -1:
        actions = np.ones(len(prices), dtype=np.int64)
    else:
        rng = np.random.default_rng(act)
        actions = rng.integers(0, 2, len(prices))

    # Now it's safe to call the Numba function
    buffer, sumusd, cash, asset_value, amount, refer = calculate_optimized(actions, prices, fix_value)
    # >>>--- END OF FIX ---<<<

    initial_capital = sumusd[0] if len(sumusd) > 0 else 0
    return pd.DataFrame({
        'buffer': buffer, 'sumusd': sumusd, 'net': sumusd - refer - initial_capital
    }, index=data.index)


# === UI COMPONENTS ===
def plot(Ticker, act, fix_value):
    df_min = Limit_fx(Ticker, act=-1, fix_value=fix_value)
    df_fx = Limit_fx(Ticker, act=act, fix_value=fix_value)
    
    if df_min.empty or df_fx.empty:
        st.error(f"Could not generate plot for {Ticker} due to missing data.")
        return

    chart_data = pd.DataFrame({'min': df_min.net, f'fx_{act}': df_fx.net}, index=df_min.index)
    st.write('Refer_Log')
    st.line_chart(chart_data)

    df_plot_burn = df_min[['buffer']].cumsum()
    st.write('Burn_Cash (Cumulative)')
    st.line_chart(df_plot_burn)

    with st.expander("Detailed Data (Min Action)"):
        st.dataframe(Limit_fx(Ticker, act=-1, fix_value=fix_value)) # Recalculate full df for display

def iframe(frame='', width=1500, height=800):
    components.iframe(frame, width=width, height=height, scrolling=True)

def render_risk_analysis(df_burn_cash):
    st.header("Cash Burn Risk Analysis")
    st.info("Based on a backtest using an 'always buy' strategy to assess maximum potential risk.")
    
    max_daily_burn = df_burn_cash['daily_burn'].min()
    cumulative_burn_series = df_burn_cash['cumulative_burn']
    
    peak_to_trough_burn = 0
    if not cumulative_burn_series.empty:
        peak_index = cumulative_burn_series.idxmax()
        peak_to_trough_burn = cumulative_burn_series.loc[peak_index] - cumulative_burn_series.loc[peak_index:].min()

    max_30_day_burn = cumulative_burn_series.rolling(window=30).apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False).min() if len(cumulative_burn_series) >= 30 else 0
    max_90_day_burn = cumulative_burn_series.rolling(window=90).apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False).min() if len(cumulative_burn_series) >= 90 else 0
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Short-Term Risk")
        st.metric(label="🔥 1-Day Burn (Worst Day)", value=f"{max_daily_burn:,.2f} USD")
        st.metric(label="🔥 30-Day Burn (Worst Month)", value=f"{max_30_day_burn:,.2f} USD")
    with col2:
        st.subheader("Medium to Long-Term Risk")
        st.metric(label="🔥 90-Day Burn (Worst Quarter)", value=f"{max_90_day_burn:,.2f} USD")
        st.metric(label="🏔️ Peak-to-Trough Burn (Max Drawdown)", value=f"{peak_to_trough_burn:,.2f} USD")

# === MAIN APP EXECUTION ===
tab_names = TICKERS + ['Burn_Cash', 'Ref_index_Log', 'cf_log']
tabs = st.tabs(tab_names)
tab_dict = dict(zip(tab_names, tabs))

# Render asset tabs
for asset in ASSETS:
    symbol = asset['symbol']
    with tab_dict[symbol]:
        act = get_act_from_thingspeak(asset['channel_id'], asset['write_api_key'], asset['field'])
        plot(symbol, act, FIX_VALUE)

# Render Burn_Cash tab
with tab_dict['Burn_Cash']:
    buffer_dfs = [Limit_fx(symbol, act=-1, fix_value=FIX_VALUE)[['buffer']].rename(columns={'buffer': f'buffer_{symbol}'}) for symbol in TICKERS]
    valid_dfs = [df for df in buffer_dfs if not df.empty]
    
    if not valid_dfs:
        st.error("Cannot calculate burn cash due to missing data for all assets.")
    else:
        df_burn_cash = pd.concat(valid_dfs, axis=1).ffill().dropna()
        df_burn_cash['daily_burn'] = df_burn_cash.sum(axis=1)
        df_burn_cash['cumulative_burn'] = df_burn_cash['daily_burn'].cumsum()
        
        render_risk_analysis(df_burn_cash)
        st.markdown("---")
        st.subheader("Cumulative Cash Burn Over Time")
        st.line_chart(df_burn_cash['cumulative_burn'])
        
        with st.expander("View Detailed Burn Data"):
            st.dataframe(df_burn_cash)

# Render Ref_index_Log tab
with tab_dict['Ref_index_Log']:
    prices_df = get_prices(TICKERS, FILTER_DATE)
    if not prices_df.empty:
        sumusd_dfs = [Limit_fx(symbol, act=-1, fix_value=FIX_VALUE)[['sumusd']].rename(columns={'sumusd': f'sumusd_{symbol}'}) for symbol in TICKERS]
        valid_sumusd_dfs = [df for df in sumusd_dfs if not df.empty]
        
        if valid_sumusd_dfs:
            df_sumusd_ = pd.concat([prices_df] + valid_sumusd_dfs, axis=1).ffill().dropna()
            
            price_cols = [col for col in df_sumusd_.columns if '_price' in col]
            sumusd_cols = [col for col in df_sumusd_.columns if 'sumusd_' in col]
            
            if price_cols and sumusd_cols:
                int_st = np.prod(df_sumusd_.iloc[0][price_cols])
                capital_ref_log = INITIAL_CAPITAL_PER_STOCK * len(TICKERS)

                def calculate_ref_log(row):
                    int_end = np.prod(row[price_cols])
                    return capital_ref_log + (-FIX_VALUE * np.log(int_end / int_st)) if int_st > 0 and int_end > 0 else capital_ref_log

                df_sumusd_['ref_log'] = df_sumusd_.apply(calculate_ref_log, axis=1)
                df_sumusd_['daily_sumusd'] = df_sumusd_[sumusd_cols].sum(axis=1)

                initial_cap_list = [Limit_fx(s, -1, FIX_VALUE).sumusd.iloc[0] for s in TICKERS if not Limit_fx(s, -1, FIX_VALUE).empty]
                total_initial_capital = sum(initial_cap_list)

                net_raw = df_sumusd_['daily_sumusd'] - df_sumusd_['ref_log'] - total_initial_capital
                df_sumusd_['net'] = net_raw - net_raw.iloc[0] if not net_raw.empty else 0
                
                st.line_chart(df_sumusd_['net'])
                with st.expander("View Data"):
                    st.dataframe(df_sumusd_)
    else:
        st.warning("Could not fetch sufficient price data for Ref_index_Log.")

# Render cf_log tab
with tab_dict['cf_log']:
    st.markdown("""
    - **Rebalance**: `-fix * ln(t0 / tn)`
    - **Net Profit**: `sumusd - refer - sumusd[0]` (ต้นทุนเริ่มต้น)
    - **Ref_index_Log**: `initial_capital_Ref_index_Log + (-1500 * ln(int_st / int_end))`
    - **Net in Ref_index_Log**: `(daily_sumusd - ref_log - total_initial_capital) - net_at_index_0`
    - **Option P/L**: `(max(0, ราคาหุ้นปัจจุบัน - Strike) * contracts_or_shares) - (contracts_or_shares * premium_paid_per_share)`
    ---
    """)
    for url in IFRAME_URLS:
        iframe(url)
        st.markdown("---")
