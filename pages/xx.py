import streamlit as st
import numpy as np
import datetime
import thingspeak
import pandas as pd
import yfinance as yf
import json

# --- Page Configuration ---
st.set_page_config(page_title="Monitor", page_icon="📈", layout="wide")

# --- Initialize Session State ---
if 'nex' not in st.session_state:
    st.session_state.nex = 0
if 'Nex_day_sell' not in st.session_state:
    st.session_state.Nex_day_sell = 0

# --- ThingSpeak Configuration ---
# Primary Channel for Asset Data
PRIMARY_CHANNEL_ID = 2528199
PRIMARY_WRITE_API_KEY = '2E65V8XEIPH9B2VV' # IMPORTANT: Use your actual Write API Key
client = thingspeak.Channel(PRIMARY_CHANNEL_ID, PRIMARY_WRITE_API_KEY, fmt='json')

# Secondary Channel for Monitor Seeds
SECONDARY_CHANNEL_ID = 2385118
SECONDARY_WRITE_API_KEY = 'IPSG3MMMBJEB9DY8' # IMPORTANT: Use your actual Write API Key
client_2 = thingspeak.Channel(SECONDARY_CHANNEL_ID, SECONDARY_WRITE_API_KEY, fmt='json')

# --- Helper Functions ---
def sell(asset=0, fix_c=1500, Diff=60):
    if asset == 0: return 0, 0, 0
    s1 = (fix_c - Diff) / asset
    s2 = round(s1, 2)
    if s2 == 0: return 0, 0, 0
    s3 = s2 * asset
    s4 = abs(s3 - fix_c)
    s5 = round(s4 / s2) if s2 != 0 else 0
    s6 = s5 * s2
    s7 = (asset * s2) + s6
    return s2, s5, round(s7, 2)

def buy(asset=0, fix_c=1500, Diff=60):
    if asset == 0: return 0, 0, 0
    b1 = (fix_c + Diff) / asset
    b2 = round(b1, 2)
    if b2 == 0: return 0, 0, 0
    b3 = b2 * asset
    b4 = abs(b3 - fix_c)
    b5 = round(b4 / b2) if b2 != 0 else 0
    b6 = b5 * b2
    b7 = (asset * b2) - b6
    return b2, b5, round(b7, 2)

def Monitor(Ticker='FFWM', field=2):
    try:
        ticker_obj = yf.Ticker(Ticker)
        ticker_data_history = ticker_obj.history(period='2y')[['Close']] # Shorter period
        ticker_data_history = round(ticker_data_history, 3)

        if ticker_data_history.index.tz is None:
            ticker_data_history.index = ticker_data_history.index.tz_localize('UTC')
        ticker_data_history.index = ticker_data_history.index.tz_convert(tz='Asia/Bangkok')
        
        # Filter from the start of the current year
        now_bangkok = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7)))
        current_year_start_dt = datetime.datetime(now_bangkok.year, 1, 1, 12, 0, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=7)))
        filter_date_str = current_year_start_dt.strftime('%Y-%m-%d %H:%M:%S%z')
        
        ticker_data_history = ticker_data_history[ticker_data_history.index >= filter_date_str]

        fx_seed_value = 0 # Default seed
        try:
            fx_raw = client_2.get_field_last(field=f'{field}')
            if fx_raw:
                fx_data = json.loads(fx_raw)
                if f"field{field}" in fx_data and fx_data[f"field{field}"] is not None:
                    fx_seed_value = int(fx_data[f"field{field}"])
                else:
                    st.warning(f"Seed field{field} not found or null in Channel {SECONDARY_CHANNEL_ID}. Using default seed 0.")
            else:
                st.warning(f"Could not retrieve seed for field{field} from Channel {SECONDARY_CHANNEL_ID}. Using default seed 0.")
        except Exception as e_seed:
            st.warning(f"Error getting seed for field{field} (Channel {SECONDARY_CHANNEL_ID}): {e_seed}. Using default seed 0.")

        rng = np.random.default_rng(fx_seed_value)
        if not ticker_data_history.empty:
            action_data = rng.integers(2, size=len(ticker_data_history))
            ticker_data_history['action'] = action_data
            ticker_data_history['index'] = [i + 1 for i in range(len(ticker_data_history))]
        else: # Handle empty historical data
            ticker_data_history['action'] = []
            ticker_data_history['index'] = []


        placeholder_df = pd.DataFrame(columns=(ticker_data_history.columns))
        placeholder_df['action'] = [i % 2 for i in range(5)] # Example placeholder actions
        placeholder_df.index = ['+0', "+1", "+2", "+3", "+4"]
        
        combined_df = pd.concat([ticker_data_history, placeholder_df], axis=0).fillna("")
        
        # Re-apply random action to the combined DataFrame if necessary, or ensure 'action' is consistent
        if not combined_df.empty:
             # Ensure action column is numeric before re-assigning random integers
            combined_df['action'] = pd.to_numeric(combined_df['action'], errors='coerce').fillna(0).astype(int)
            combined_df['action'] = rng.integers(2, size=len(combined_df))


        return combined_df.tail(7), fx_seed_value
    except Exception as e:
        st.error(f"Error in Monitor for {Ticker} (Seed Field {field}): {e}")
        default_cols = ['Close', 'action', 'index']
        empty_df = pd.DataFrame(0, index=pd.MultiIndex.from_tuples([tuple(f'+{i}' for _ in default_cols) for i in range(7)]), columns=default_cols).tail(7)
        empty_df['action'] = 0
        return empty_df, 0

def get_thingspeak_asset_value(ts_client, field_id_str):
    raw_data = ts_client.get_field_last(field=field_id_str)
    if raw_data:
        try:
            loaded_json = json.loads(raw_data)
            if field_id_str in loaded_json and loaded_json[field_id_str] is not None:
                return float(loaded_json[field_id_str])
            else:
                # st.warning(f"Field {field_id_str} is null or not in response. Defaulting to 0.0.")
                return 0.0
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            st.error(f"Error processing data for {field_id_str}: {e}. Raw: '{raw_data}'. Defaulting to 0.0.")
            return 0.0
    # st.warning(f"No data retrieved for {field_id_str}. Defaulting to 0.0.")
    return 0.0

def get_current_price(ticker_symbol):
    try:
        return yf.Ticker(ticker_symbol).fast_info.get('lastPrice', 0)
    except Exception:
        # st.warning(f"Could not fetch price for {ticker_symbol}")
        return 0

toggle = lambda x: 1 - x if x in [0, 1, '0', '1'] else 0

def get_action_value(df_action_series, index, default_value=0):
    try:
        val = df_action_series.values[index]
        return int(val) if pd.notna(val) and val != "" else default_value
    except (IndexError, ValueError):
        return default_value

# --- Data Loading ---
asset_configs = [
    {'ticker': 'FFWM', 'seed_field': 2, 'data_field': 'field1', 'label': 'FFWM'},
    {'ticker': 'NEGG', 'seed_field': 3, 'data_field': 'field2', 'label': 'NEGG'},
    {'ticker': 'RIVN', 'seed_field': 4, 'data_field': 'field3', 'label': 'RIVN'},
    {'ticker': 'APLS', 'seed_field': 5, 'data_field': 'field4', 'label': 'APLS'},
    {'ticker': 'NVTS', 'seed_field': 6, 'data_field': 'field5', 'label': 'NVTS'},
    {'ticker': 'QXO',  'seed_field': 7, 'data_field': 'field6', 'label': 'QXO', 'option_val': 79.0},
    {'ticker': 'RXRX', 'seed_field': 8, 'data_field': 'field7', 'label': 'RXRX'} # RXRX Added
]

monitor_data = {}
for asset in asset_configs:
    monitor_data[asset['ticker']] = {}
    monitor_data[asset['ticker']]['df'], monitor_data[asset['ticker']]['fx_js'] = Monitor(Ticker=asset['ticker'], field=asset['seed_field'])

# --- UI Layout ---
st.markdown("### Next Day Settings")
nex_day_check = st.checkbox('Advance to Next Day calculation period', key='cb_nex_day_check')
if nex_day_check:
    st.write(f"Current 'nex' offset: {st.session_state.nex}, Sell toggle: {'ON' if st.session_state.Nex_day_sell else 'OFF'}")
    nex_col1, nex_col2 = st.columns(2)
    if nex_col1.button("Set `nex` to 1 (Tomorrow)", key='btn_set_nex_1'):
        st.session_state.nex = 1
        st.rerun()
    if nex_col2.button("Toggle Sell Logic for Next Day & Set `nex` to 1", key='btn_toggle_sell_nex_1'):
        st.session_state.nex = 1
        st.session_state.Nex_day_sell = 1 - st.session_state.Nex_day_sell # Toggle
        st.rerun()
else: # Reset if unchecked
    if st.session_state.nex != 0 or st.session_state.Nex_day_sell != 0:
        st.session_state.nex = 0
        st.session_state.Nex_day_sell = 0
        st.rerun()


st.divider()
st.markdown("### Asset Configuration & Initial Values")

# Setup Column and Diff input
setup_col, diff_col_main, _, _, _, _, _, _, _ = st.columns(9) # Keep 9 for visual balance if needed

with diff_col_main:
    diff_input_val = st.number_input('Difference (Diff)', step=1, value=60, key='numin_diff_main')

with setup_col:
    start_config = st.checkbox('Enable Initial Asset Setup', key='cb_start_config')
    if start_config:
        for asset in asset_configs:
            if st.checkbox(f"Configure @_{asset['label']}_ASSET", key=f"cb_setup_{asset['ticker']}"):
                add_val = st.number_input(f"@_{asset['label']}_ASSET Value", step=0.001, value=0.0, key=f"numin_add_{asset['ticker']}")
                if st.button(f"Set {asset['label']} Initial", key=f"btn_set_initial_{asset['ticker']}"):
                    try:
                        client.update({asset['data_field']: add_val})
                        st.success(f"{asset['label']} initial asset set to {add_val} on ThingSpeak {asset['data_field']}.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to update {asset['label']}: {e}")
st.divider()

# --- Current Asset Values & Calculations ---
st.markdown("### Current Asset Status & Trade Simulation")
asset_cols = st.columns(len(asset_configs))
asset_data_live = {}

for i, asset_cfg in enumerate(asset_configs):
    with asset_cols[i]:
        asset_last_val_ts = get_thingspeak_asset_value(client, asset_cfg['data_field'])
        
        input_label = f"{asset_cfg['label']}_ASSET"
        current_asset_val_input = asset_last_val_ts

        if asset_cfg['ticker'] == 'QXO':
            input_label = f"{asset_cfg['label']} (Real, TS Val)" # Clarify QXO input is the ThingSpeak part
            # QXO_REAL is what's stored on ThingSpeak
        
        # Number input for live adjustment or display
        # This value is read and used, but not automatically written back to ThingSpeak unless a trade happens
        asset_data_live[asset_cfg['ticker']] = {}
        asset_data_live[asset_cfg['ticker']]['current_ts_value'] = asset_last_val_ts # Value from ThingSpeak
        
        # The x_val is what's used for buy/sell calculations
        x_val_for_calc = st.number_input(input_label, step=0.001, value=current_asset_val_input, key=f"numin_live_{asset_cfg['ticker']}")
        
        if asset_cfg['ticker'] == 'QXO':
             asset_data_live[asset_cfg['ticker']]['x_calc_val'] = asset_cfg.get('option_val', 0) + x_val_for_calc
             st.caption(f"QXO Option Fixed: {asset_cfg.get('option_val', 0)}")
             st.caption(f"QXO Total for Calc: {asset_data_live[asset_cfg['ticker']]['x_calc_val']:.3f}")
        else:
            asset_data_live[asset_cfg['ticker']]['x_calc_val'] = x_val_for_calc


# Calculate buy/sell parameters based on 'x_calc_val'
for asset_cfg in asset_configs:
    x_val = asset_data_live[asset_cfg['ticker']]['x_calc_val']
    asset_data_live[asset_cfg['ticker']]['sell_params'] = sell(asset=x_val, Diff=diff_input_val)
    asset_data_live[asset_cfg['ticker']]['buy_params'] = buy(asset=x_val, Diff=diff_input_val)

st.divider()

# --- Limit Order Sections ---
for asset_cfg in asset_configs:
    ticker = asset_cfg['ticker']
    label = asset_cfg['label']
    data_field_ts = asset_cfg['data_field']
    
    st.subheader(f"Limit Orders: {label} ({ticker})")

    # Get action from monitor data
    action_val_raw = get_action_value(monitor_data[ticker]['df']['action'], 1 + st.session_state.nex)
    
    # Determine if the order checkbox should be checked based on toggle and action
    is_order_active = bool(np.where(st.session_state.Nex_day_sell == 1, toggle(action_val_raw), action_val_raw))
    
    limit_order_checkbox = st.checkbox(f"Activate Limit Order Logic for {label}", value=is_order_active, key=f"cb_limit_order_{ticker}")

    if limit_order_checkbox:
        # Sell Side
        sell_price, sell_amount, sell_cost = asset_data_live[ticker]['buy_params'] # sell uses 'buy' function's output for amount calculation logic
        st.write(f"**Proposed Sell Action for {label}:** Amount: `{sell_amount:.2f}` at Price: `{sell_price:.2f}` (Cost: `{sell_cost:.2f}`)")
        
        s_col1, s_col2 = st.columns([3,1])
        with s_col1:
            sell_match_check = st.checkbox(f"Confirm Sell Match for {label}", key=f"cb_sell_match_{ticker}")
        with s_col2:
            if sell_match_check:
                if st.button(f"Execute Sell {label}", key=f"btn_exec_sell_{ticker}"):
                    current_ts_val = asset_data_live[ticker]['current_ts_value']
                    new_asset_val = current_ts_val - sell_amount
                    try:
                        client.update({data_field_ts: new_asset_val})
                        st.success(f"SELL {label}: Updated asset to {new_asset_val:.3f} on ThingSpeak {data_field_ts}.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"ThingSpeak Update Failed for SELL {label}: {e}")
        
        # Current Price Info
        last_price = get_current_price(ticker)
        current_portfolio_value = last_price * asset_data_live[ticker]['x_calc_val'] # Use x_calc_val for PV
        st.write(f"{label} Last Price: `{last_price:.3f}` | Portfolio Value (based on x_calc_val): `{current_portfolio_value:.2f}` (Diff from 1500: `{current_portfolio_value - 1500:.2f}`)")

        # Buy Side
        buy_price, buy_amount, buy_cost = asset_data_live[ticker]['sell_params'] # buy uses 'sell' function's output for amount calculation logic
        st.write(f"**Proposed Buy Action for {label}:** Amount: `{buy_amount:.2f}` at Price: `{buy_price:.2f}` (Cost: `{buy_cost:.2f}`)")
        
        b_col1, b_col2 = st.columns([3,1])
        with b_col1:
            buy_match_check = st.checkbox(f"Confirm Buy Match for {label}", key=f"cb_buy_match_{ticker}")
        with b_col2:
            if buy_match_check:
                if st.button(f"Execute Buy {label}", key=f"btn_exec_buy_{ticker}"):
                    current_ts_val = asset_data_live[ticker]['current_ts_value']
                    new_asset_val = current_ts_val + buy_amount
                    try:
                        client.update({data_field_ts: new_asset_val})
                        st.success(f"BUY {label}: Updated asset to {new_asset_val:.3f} on ThingSpeak {data_field_ts}.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"ThingSpeak Update Failed for BUY {label}: {e}")
    st.divider()

# --- Manual Rerun ---
if st.button("Refresh Data & UI (Manual Rerun)", key='btn_manual_rerun'):
    st.rerun()
