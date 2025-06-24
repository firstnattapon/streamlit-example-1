import pandas as pd
import numpy as np
from numba import njit
import yfinance as yf
import streamlit as st
import thingspeak
import json
import streamlit.components.v1 as components
from typing import List, Union

st.set_page_config(page_title="Limit_F(X)", page_icon="✈️", layout="wide")

# --- START: SimulationTracer Class ---
class SimulationTracer:
    """
    คลาสสำหรับห่อหุ้มกระบวนการทั้งหมด ตั้งแต่การถอดรหัสพารามิเตอร์
    ไปจนถึงการจำลองกระบวนการกลายพันธุ์ของ action sequence
    """
    def __init__(self, encoded_string: str):
        self.encoded_string: str = encoded_string
        if not isinstance(self.encoded_string, str):
            self.encoded_string = str(self.encoded_string)
        self._decode_and_set_attributes()

    def _decode_and_set_attributes(self):
        encoded_string = self.encoded_string
        if not encoded_string.isdigit():
            self._reset_attributes()
            return

        decoded_numbers = []
        idx = 0
        try:
            while idx < len(encoded_string):
                length_of_number = int(encoded_string[idx])
                idx += 1
                number_str = encoded_string[idx : idx + length_of_number]
                idx += length_of_number
                decoded_numbers.append(int(number_str))
        except (IndexError, ValueError):
            self._reset_attributes()
            return

        if len(decoded_numbers) < 3:
            self._reset_attributes()
            return

        self.action_length: int = decoded_numbers[0]
        self.mutation_rate: int = decoded_numbers[1]
        self.dna_seed: int = decoded_numbers[2]
        self.mutation_seeds: List[int] = decoded_numbers[3:]
        self.mutation_rate_float: float = self.mutation_rate / 100.0

    def _reset_attributes(self):
        self.action_length: int = 0
        self.mutation_rate: int = 0
        self.dna_seed: int = 0
        self.mutation_seeds: List[int] = []
        self.mutation_rate_float: float = 0.0

    def run(self) -> np.ndarray:
        if self.action_length <= 0:
            return np.array([])
            
        dna_rng = np.random.default_rng(seed=self.dna_seed)
        current_actions = dna_rng.integers(0, 2, size=self.action_length)
        if self.action_length > 0:
            current_actions[0] = 1
        for m_seed in self.mutation_seeds:
            mutation_rng = np.random.default_rng(seed=m_seed)
            mutation_mask = mutation_rng.random(self.action_length) < self.mutation_rate_float
            current_actions[mutation_mask] = 1 - current_actions[mutation_mask]
            if self.action_length > 0:
                current_actions[0] = 1
        return current_actions

# --- END: SimulationTracer Class ---

# === CONFIG LOADING ===
@st.cache_data
def load_config(path='limit_fx_config.json'):
    try:
        with open(path, 'r') as f:
            config = json.load(f)
        return config['assets']
    except FileNotFoundError:
        st.error(f"Configuration file '{path}' not found.")
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
def get_act_from_thingspeak(channel_id, api_key, field) -> Union[str, np.ndarray]:
    """
    Fetches data from ThingSpeak. It can now handle both encoded strings and full action arrays.
    Returns:
        - np.ndarray if the value is a full action array string.
        - str for encoded strings or fallback values.
    """
    try:
        client = thingspeak.Channel(channel_id, api_key, fmt='json')
        act_json = client.get_field_last(field=str(field))
        value_str = json.loads(act_json).get(f"field{field}")
        
        if value_str is None:
            st.warning(f"Field {field} on channel {channel_id} is null. Using fallback (always buy).")
            return "-1"

        value_str = value_str.strip()
        
        # ** NEW LOGIC: Check if it's a full action array **
        if value_str.startswith('[') and value_str.endswith(']'):
            try:
                # Convert string like '[1 0 1 ... 1]' to numpy array
                cleaned_str = value_str.strip('[]').replace('\n', ' ')
                actions_array = np.fromstring(cleaned_str, dtype=int, sep=' ')
                st.success(f"Successfully parsed a full action array of length {len(actions_array)} from ThingSpeak.")
                return actions_array
            except Exception as e:
                st.error(f"Failed to parse action array from ThingSpeak: {e}. Using fallback.")
                return "-1"
        
        # If not an array, return it as a string (for SimulationTracer or fallback)
        return value_str
        
    except Exception as e:
        st.error(f"Could not fetch data from ThingSpeak (Channel: {channel_id}, Field: {field}). Error: {e}")
        return "-1"

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

def get_max_action(price_list, fix=1500):
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

@st.cache_data(ttl=600)
def Limit_fx(Ticker: str, act: Union[str, np.ndarray] = "-1"):
    filter_date = '2023-01-01 12:00:00+07:00'
    try:
        tickerData = yf.Ticker(Ticker)
        history = tickerData.history(period='max')[['Close']]
        if history.empty: return pd.DataFrame()
        history.index = history.index.tz_convert(tz='Asia/Bangkok')
        history = history[history.index >= filter_date]
        prices = np.array(history.Close.values, dtype=np.float64)
    except Exception as e:
        st.warning(f"Could not get yfinance data for {Ticker}: {e}")
        return pd.DataFrame()

    if len(prices) == 0: return pd.DataFrame()

    num_prices = len(prices)
    actions = np.ones(num_prices, dtype=np.int64) # Default

    if isinstance(act, np.ndarray):
        # ** NEW: Directly use the full action array if provided **
        actions = act
    elif isinstance(act, str):
        if act == "-1":
            actions = np.ones(num_prices, dtype=np.int64)
        elif act == "-2":
            actions = get_max_action(prices)
        else:
            # Use SimulationTracer for encoded string actions
            try:
                tracer = SimulationTracer(encoded_string=act)
                generated_actions = tracer.run()
                if generated_actions.size > 0:
                    actions = generated_actions
                else:
                    st.warning(f"Invalid action string '{act}'. Defaulting to 'always buy'.")
            except Exception as e:
                st.error(f"Error during SimulationTracer for '{act}': {e}. Defaulting to 'always buy'.")

    # --- Final check for length mismatch and padding ---
    if len(actions) != num_prices:
        st.warning(f"Action sequence length ({len(actions)}) does not match price history length ({num_prices}). Truncating or padding actions.")
        final_actions = np.ones(num_prices, dtype=np.int64) # Pad with 'buy'
        copy_len = min(len(actions), num_prices)
        final_actions[:copy_len] = actions[:copy_len]
    else:
        final_actions = actions

    buffer, sumusd, cash, asset_value, amount, refer = calculate_optimized(final_actions, prices)
    if sumusd.size == 0: return pd.DataFrame()
    
    initial_capital = sumusd[0]
    df = pd.DataFrame({
        'price': prices,
        'action': final_actions,
        'buffer': buffer, 'sumusd': sumusd, 'cash': cash,
        'asset_value': asset_value, 'amount': amount,
        'refer': refer + initial_capital,
        'net': sumusd - refer - initial_capital
    }, index=history.index)
    return df

# === UI FUNCTIONS ===
def plot(Ticker, act):
    # Determine label for the main strategy line
    if isinstance(act, np.ndarray):
        fx_label = 'fx_Hybrid_Full'
    elif isinstance(act, str) and len(act) > 10:
        fx_label = f'fx_{act[:10]}...'
    else:
        fx_label = f'fx_{act}'
        
    df_min = Limit_fx(Ticker, act="-1")
    df_fx = Limit_fx(Ticker, act=act)
    df_max = Limit_fx(Ticker, act="-2")

    if df_min.empty or df_fx.empty or df_max.empty:
        st.error(f"Could not generate plot for {Ticker} due to missing data.")
        return

    chart_data = pd.DataFrame({
        'min': df_min.net,
        fx_label: df_fx.net,
        'max': df_max.net
    }, index=df_min.index)
    st.write('Refer_Log')
    st.line_chart(chart_data)

    df_plot_burn = df_min[['buffer']].cumsum()
    st.write('Burn_Cash (Cumulative)')
    st.line_chart(df_plot_burn)

    with st.expander("Detailed Data (Full Hybrid Strategy)"):
        st.dataframe(df_fx)

def iframe(frame='', width=1500, height=800):
    components.iframe(frame, width=width, height=height, scrolling=True)

# === MAIN APP LAYOUT (UNCHANGED) ===
tab_names = TICKERS + ['Burn_Cash', 'Ref_index_Log', 'cf_log']
tabs = st.tabs(tab_names)
tab_dict = dict(zip(tab_names, tabs))

for asset in ASSETS:
    symbol = asset['symbol']
    with tab_dict[symbol]:
        act = get_act_from_thingspeak(
            channel_id=asset['channel_id'],
            api_key=asset['write_api_key'],
            field=asset['field']
        )
        plot(symbol, act)

# ... (The rest of the code for 'Ref_index_Log', 'Burn_Cash', and 'cf_log' tabs remains exactly the same) ...
# === REF_INDEX_LOG TAB (UNCHANGED) ===
with tab_dict['Ref_index_Log']:
    filter_date = '2023-01-01 12:00:00+07:00'
    prices_df = get_prices(TICKERS, filter_date)

    if not prices_df.empty:
        dfs_to_align = []
        for symbol in TICKERS:
            df_temp = Limit_fx(symbol, act="-1")
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

                total_initial_capital = sum([Limit_fx(symbol, act="-1").sumusd.iloc[0] for symbol in TICKERS if not Limit_fx(symbol, act="-1").empty])
                net_raw = df_sumusd_['daily_sumusd'] - df_sumusd_['ref_log'] - total_initial_capital
                net_at_index_0 = net_raw.iloc[0] if not net_raw.empty else 0
                df_sumusd_['net'] = net_raw - net_at_index_0
                
                st.header("Net Performance Analysis (vs. Reference)")
                st.info("Performance analysis of the portfolio's net value against the logarithmic reference index. 'Worst' periods indicate maximum losses, while 'Trough-to-Peak' shows the maximum possible gain from a low point.")

                net_series = df_sumusd_['net']
                min_daily_cf = net_series.diff().min()
                if pd.isna(min_daily_cf): min_daily_cf = 0

                trough_to_peak_gain = 0
                if not net_series.empty:
                    trough_index = net_series.idxmin()
                    peak_after_trough = net_series.loc[trough_index:].max()
                    trough_value = net_series.loc[trough_index]
                    if pd.notna(peak_after_trough) and pd.notna(trough_value):
                         trough_to_peak_gain = peak_after_trough - trough_value

                min_30_day_cf = 0
                if len(net_series) >= 30:
                    rolling_30_day_change = net_series.rolling(window=30).apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)
                    if rolling_30_day_change.notna().any():
                        min_30_day_cf = rolling_30_day_change.min()
                if pd.isna(min_30_day_cf): min_30_day_cf = 0
                
                min_90_day_cf = 0
                if len(net_series) >= 90:
                    rolling_90_day_change = net_series.rolling(window=90).apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)
                    if rolling_90_day_change.notna().any():
                        min_90_day_cf = rolling_90_day_change.min()
                if pd.isna(min_90_day_cf): min_90_day_cf = 0

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Short-Term CF")
                    st.metric(label="📉 1-Day CF (Worst Day)", value=f"{min_daily_cf:,.2f} USD")
                    st.metric(label="📉 30-Day CF (Worst Month)", value=f"{min_30_day_cf:,.2f} USD")

                with col2:
                    st.subheader("Medium to Long-Term CF")
                    st.metric(label="📉 90-Day CF (Worst Quarter)", value=f"{min_90_day_cf:,.2f} USD")
                    st.metric(label="📈 Trough-to-Peak Gain (Max Run-up)", value=f"{trough_to_peak_gain:,.2f} USD")

                st.markdown("---")
                st.subheader("Net Performance Over Time")
                st.line_chart(df_sumusd_['net'])
                with st.expander("View Data"):
                    st.dataframe(df_sumusd_)
        else:
             st.warning("Could not align dataframes. Not enough data available for the selected assets.")
    else:
        st.warning("Could not fetch sufficient price data for Ref_index_Log.")

# === BURN_CASH TAB (UNCHANGED) ===
with tab_dict['Burn_Cash']:
    dfs_to_align = []
    for symbol in TICKERS:
        df_temp = Limit_fx(symbol, act="-1")
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
        
        max_daily_burn = df_burn_cash['daily_burn'].min()
        cumulative_burn_series = df_burn_cash['cumulative_burn']
        
        peak_to_trough_burn = 0
        if not cumulative_burn_series.empty:
            peak_index = cumulative_burn_series.idxmax()
            peak_to_trough_burn = cumulative_burn_series.loc[peak_index] - cumulative_burn_series.loc[peak_index:].min()

        max_30_day_burn = 0
        if len(cumulative_burn_series) >= 30:
            rolling_30_day_change = cumulative_burn_series.rolling(window=30).apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)
            if rolling_30_day_change.notna().any():
                max_30_day_burn = rolling_30_day_change.min()
        
        max_90_day_burn = 0
        if len(cumulative_burn_series) >= 90:
            rolling_90_day_change = cumulative_burn_series.rolling(window=90).apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)
            if rolling_90_day_change.notna().any():
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

# === CF_LOG TAB (UNCHANGED) ===
with tab_dict['cf_log']:
    st.markdown("""
    - **Rebalance**: `-fix * ln(t0 / tn)`
    - **Net Profit**: `sumusd - refer - sumusd[0]` (ต้นทุนเริ่มต้น)
    - **Ref_index_Log**: `initial_capital_Ref_index_Log + (-1500 * ln(int_st / int_end))`
    - **Net in Ref_index_Log**: `(daily_sumusd - ref_log - total_initial_capital) - net_at_index_0`
    - **Option P/L**: `(max(0, ราคาหุ้นปัจจุบัน - Strike) * contracts_or_shares) - (contracts_or_shares * premium_paid_per_share)`
    ---
    """)
    iframe("https://monica.im/share/artifact?id=qpAkuKjBpuVz2cp9nNFRs3")
    st.markdown("---")
    iframe("https://monica.im/share/artifact?id=wEjeaMxVW6MgDDm3xAZatX")
    st.markdown("---")
    iframe("https://monica.im/share/artifact?id=ZfHT5iDP2Ypz82PCRw9nEK")
    st.markdown("---")
    iframe("https://monica.im/share/chat?shareId=SUsEYhzSMwqIq3Cx")
