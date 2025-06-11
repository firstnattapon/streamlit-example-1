import streamlit as st
import numpy as np
import datetime
import thingspeak
import pandas as pd
import yfinance as yf
import json
from functools import lru_cache
import concurrent.futures
from threading import Lock
import os

st.set_page_config(page_title="Monitor", page_icon="📈", layout="wide")

# --- START: CONFIGURATION LOADING ---
@st.cache_data
def load_config(file_path='config.json'):
    """Loads asset configuration from a JSON file."""
    if not os.path.exists(file_path):
        st.error(f"Configuration file not found: {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Load configuration at the beginning
ASSET_CONFIGS = load_config()
if not ASSET_CONFIGS:
    st.stop()
# --- END: CONFIGURATION LOADING ---

_cache_lock = Lock()
_price_cache = {}
_cache_timestamp = {}

@st.cache_resource
def get_clients():
    # These should ideally be in st.secrets, but keeping as is for now.
    channel_id = 2528199
    write_api_key = '2E65V8XEIPH9B2VV'
    client = thingspeak.Channel(channel_id, write_api_key, fmt='json')
    
    channel_id_2 = 2385118
    write_api_key_2 = 'IPSG3MMMBJEB9DY8'
    client_2 = thingspeak.Channel(channel_id_2, write_api_key_2, fmt='json')
    
    return client, client_2

client, client_2 = get_clients()

def clear_all_caches():
    st.cache_data.clear()
    st.cache_resource.clear()
    sell.cache_clear()
    buy.cache_clear()
    with _cache_lock:
        _price_cache.clear()
        _cache_timestamp.clear()
    st.success("🗑️ Clear ALL caches complete!")
    st.rerun()

@lru_cache(maxsize=128)
def sell(asset, fix_c=1500, Diff=60):
    if asset == 0: return 0, 0, 0
    s1 = (fix_c - Diff) / asset
    s2 = round(s1, 2)
    s3 = s2 * asset
    s4 = abs(s3 - fix_c)
    s5 = round(s4 / s2) if s2 != 0 else 0
    s6 = s5 * s2
    s7 = (asset * s2) + s6
    return s2, s5, round(s7, 2)

@lru_cache(maxsize=128)
def buy(asset, fix_c=1500, Diff=60):
    if asset == 0: return 0, 0, 0
    b1 = (fix_c + Diff) / asset
    b2 = round(b1, 2)
    b3 = b2 * asset
    b4 = abs(b3 - fix_c)
    b5 = round(b4 / b2) if b2 != 0 else 0
    b6 = b5 * b2
    b7 = (asset * b2) - b6
    return b2, b5, round(b7, 2)

def get_cached_price(ticker, max_age=30):
    current_time = datetime.datetime.now()
    with _cache_lock:
        if (ticker in _price_cache and 
            ticker in _cache_timestamp and 
            (current_time - _cache_timestamp[ticker]).seconds < max_age):
            return _price_cache[ticker]
    try:
        price = yf.Ticker(ticker).fast_info['lastPrice']
        with _cache_lock:
            _price_cache[ticker] = price
            _cache_timestamp[ticker] = current_time
        return price
    except:
        return 0.0

@st.cache_data(ttl=300)
def Monitor(Ticker='FFWM', field=2):
    try:
        tickerData = yf.Ticker(Ticker)
        tickerData = round(tickerData.history(period='max')[['Close']], 3)
        tickerData.index = tickerData.index.tz_convert(tz='Asia/bangkok')
        filter_date = '2023-01-01 12:00:00+07:00'
        tickerData = tickerData[tickerData.index >= filter_date]

        fx = client_2.get_field_last(field='{}'.format(field))
        fx_js = int(json.loads(fx)["field{}".format(field)])
        rng = np.random.default_rng(fx_js)
        data = rng.integers(2, size=len(tickerData))
        tickerData['action'] = data
        tickerData['index'] = [i+1 for i in range(len(tickerData))]

        tickerData_1 = pd.DataFrame(columns=(tickerData.columns))
        tickerData_1['action'] = [i for i in range(5)]
        tickerData_1.index = ['+0', "+1", "+2", "+3", "+4"]
        df = pd.concat([tickerData, tickerData_1], axis=0).fillna("")
        rng = np.random.default_rng(fx_js)
        df['action'] = rng.integers(2, size=len(df))
        return df.tail(7), fx_js
    except Exception as e:
        st.error(f"Error in Monitor function for {Ticker}: {str(e)}")
        return pd.DataFrame(), 0

# --- DYNAMIC DATA FETCHING based on config ---
@st.cache_data(ttl=300)
def fetch_all_monitor_data(configs):
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(configs)) as executor:
        future_to_ticker = {
            executor.submit(Monitor, asset['ticker'], asset['monitor_field']): asset['ticker']
            for asset in configs
        }
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                results[ticker] = future.result()
            except Exception as e:
                st.error(f"Error fetching monitor data for {ticker}: {str(e)}")
                results[ticker] = (pd.DataFrame(), 0)
    return results

@st.cache_data(ttl=60)
def get_all_assets_from_thingspeak(configs):
    assets = {}
    asset_fields = [asset['asset_field'] for asset in configs]
    
    def fetch_asset(field):
        try:
            data = client.get_field_last(field=field)
            return eval(json.loads(data)[field])
        except:
            return 0.0

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(asset_fields)) as executor:
        future_to_field = {executor.submit(fetch_asset, field): field for field in asset_fields}
        for future in concurrent.futures.as_completed(future_to_field):
            field = future_to_field[future]
            try:
                ticker = next((c['ticker'] for c in configs if c['asset_field'] == field), None)
                if ticker:
                    assets[ticker] = future.result()
            except Exception as e:
                st.error(f"Error fetching asset for field {field}: {str(e)}")
    return assets

# Execute data fetching
monitor_data_all = fetch_all_monitor_data(ASSET_CONFIGS)
last_assets_all = get_all_assets_from_thingspeak(ASSET_CONFIGS)

# --- UI SECTION ---
nex = 0
Nex_day_sell = 0
toggle = lambda x: 1 - x

Nex_day_ = st.checkbox('nex_day')
if Nex_day_:
    st.write("value = ", nex)
    nex_col, Nex_day_sell_col, _, _, _ = st.columns(5)
    if nex_col.button("Nex_day"): nex = 1
    if Nex_day_sell_col.button("Nex_day_sell"):
        nex = 1
        Nex_day_sell = 1
    st.write("value = ", nex)
    if Nex_day_sell: st.write("Nex_day_sell = ", Nex_day_sell)

st.write("_____")

# --- DYNAMIC ASSET INPUTS based on config ---
control_cols = st.columns(4) 
x_2 = control_cols[0].number_input('Diff', step=1, value=60)
Start = control_cols[0].checkbox('start')

if Start:
    with control_cols[0].expander("Update Assets on ThingSpeak"):
        for config in ASSET_CONFIGS:
            ticker = config['ticker']
            field = config['asset_field']
            
            checkbox_key = f'@_{ticker}_ASSET'
            if st.checkbox(checkbox_key, key=f'check_{ticker}'):
                add_val = st.number_input(f"New Value for {ticker}", step=0.001, value=0.0, key=f'input_{ticker}')
                if st.button(f"GO_{ticker}", key=f'btn_{ticker}'):
                    client.update({field: add_val})
                    st.write(f"Updated {ticker} to: {add_val}")
                    clear_all_caches()

asset_input_cols = st.columns(len(ASSET_CONFIGS))
asset_inputs = {} 

for i, config in enumerate(ASSET_CONFIGS):
    with asset_input_cols[i]:
        ticker = config['ticker']
        last_asset_val = last_assets_all.get(ticker, 0.0)
        
        if config['option_config']:
            option_val = config['option_config']['base_value']
            label = config['option_config']['label']
            real_val = st.number_input(label, step=0.001, value=last_asset_val, key=f"input_{ticker}_real")
            asset_inputs[ticker] = option_val + real_val
        else:
            label = f'{ticker}_ASSET'
            asset_val = st.number_input(label, step=0.001, value=last_asset_val, key=f"input_{ticker}_asset")
            asset_inputs[ticker] = asset_val
            
st.write("_____")

# --- DYNAMIC CALCULATIONS based on config ---
calculations = {}
for config in ASSET_CONFIGS:
    ticker = config['ticker']
    asset_value = asset_inputs.get(ticker, 0.0)
    fix_c = config['fix_c']
    calculations[ticker] = {
        'sell': sell(asset_value, fix_c=fix_c, Diff=x_2),
        'buy': buy(asset_value, fix_c=fix_c, Diff=x_2)
    }

# --- DYNAMIC TRADING SECTIONS based on config ---
def create_trading_section(config, asset_val, asset_last, df_data, calculations_data, nex, Nex_day_sell):
    ticker = config['ticker']
    asset_field = config['asset_field']
    field_num_for_update = int(asset_field.replace('field', ''))

    try:
        action_val = np.where(
            Nex_day_sell == 1,
            toggle(df_data.action.values[1+nex]),
            df_data.action.values[1+nex]
        ) if len(df_data) > 1+nex else 0
    except:
        action_val = 0
    
    limit_order = st.checkbox(f'Limit_Order_{ticker}', value=bool(action_val), key=f'limit_order_{ticker}')
    
    if limit_order:
        sell_calc = calculations_data['sell']
        buy_calc = calculations_data['buy']
        
        st.write('sell', '    ', 'A', buy_calc[1], 'P', buy_calc[0], 'C', buy_calc[2])
        
        col1, col2, col3 = st.columns(3)
        if col3.checkbox(f'sell_match_{ticker}'):
            if col3.button(f"GO_SELL_{ticker}"):
                client.update({f'field{field_num_for_update}': asset_last - buy_calc[1]})
                col3.write(asset_last - buy_calc[1])
                clear_all_caches()
        
        # --- START: Compact display with labels (MODIFIED PART) ---
        try:
            current_price = get_cached_price(ticker)
            if current_price > 0:
                pv = current_price * asset_val
                fix_value = config['fix_c']
                pl_value = pv - fix_value
                
                pl_color = "green" if pl_value >= 0 else "red"

                st.markdown(
                    f"Price: **{current_price:,.3f}**  |  Portfolio Value: **{pv:,.2f}**  |  P/L (vs {fix_value:,}) : <span style='color:{pl_color}; font-weight:bold;'>{pl_value:,.2f}</span>",
                    unsafe_allow_html=True
                )
            else:
                st.info(f"Price data for {ticker} is currently unavailable.")
        except Exception as e:
            st.warning(f"Could not retrieve price data for {ticker}.")
        # --- END: Compact display with labels ---
        
        col4, col5, col6 = st.columns(3)
        st.write('buy', '   ', 'A', sell_calc[1], 'P', sell_calc[0], 'C', sell_calc[2])
        if col6.checkbox(f'buy_match_{ticker}'):
            if col6.button(f"GO_BUY_{ticker}"):
                client.update({f'field{field_num_for_update}': asset_last + sell_calc[1]})
                col6.write(asset_last + sell_calc[1])
                clear_all_caches()

# Loop to create each trading section
for config in ASSET_CONFIGS:
    ticker = config['ticker']
    df_data, _ = monitor_data_all.get(ticker, (pd.DataFrame(), 0))
    asset_last = last_assets_all.get(ticker, 0.0)
    asset_val = asset_inputs.get(ticker, 0.0)
    calculations_data = calculations.get(ticker, {})

    create_trading_section(config, asset_val, asset_last, df_data, calculations_data, nex, Nex_day_sell)
    st.write("_____")

if st.sidebar.button("RERUN"):
    clear_all_caches()
