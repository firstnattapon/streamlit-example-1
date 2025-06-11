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

# ---------- CONFIGURATION ----------
@st.cache_data
def load_config(file_path='monitor_config.json'):
    """Load asset configuration from a JSON file."""
    if not os.path.exists(file_path):
        st.error(f"Configuration file not found: {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

ASSET_CONFIGS = load_config()
if not ASSET_CONFIGS:
    st.stop()

# ---------- GLOBAL CACHE ----------
_cache_lock = Lock()
_price_cache = {}
_cache_timestamp = {}

@st.cache_resource
def get_clients():
    channel_id_1, key_1 = 2528199, '2E65V8XEIPH9B2VV'
    channel_id_2, key_2 = 2385118, 'IPSG3MMMBJEB9DY8'
    return (
        thingspeak.Channel(channel_id_1, key_1, fmt='json'),
        thingspeak.Channel(channel_id_2, key_2, fmt='json')
    )

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

# ---------- CALCULATION UTILS ----------
@lru_cache(maxsize=128)
def sell(asset, fix_c=1500, Diff=60):
    if asset == 0: return 0, 0, 0
    unit_price = round((fix_c - Diff) / asset, 2)
    adjust_qty = round(abs(asset * unit_price - fix_c) / unit_price) if unit_price != 0 else 0
    total = round(asset * unit_price + adjust_qty * unit_price, 2)
    return unit_price, adjust_qty, total

@lru_cache(maxsize=128)
def buy(asset, fix_c=1500, Diff=60):
    if asset == 0: return 0, 0, 0
    unit_price = round((fix_c + Diff) / asset, 2)
    adjust_qty = round(abs(asset * unit_price - fix_c) / unit_price) if unit_price != 0 else 0
    total = round(asset * unit_price - adjust_qty * unit_price, 2)
    return unit_price, adjust_qty, total

def get_cached_price(ticker, max_age=30):
    now = datetime.datetime.now()
    with _cache_lock:
        if (ticker in _price_cache and 
            ticker in _cache_timestamp and 
            (now - _cache_timestamp[ticker]).seconds < max_age):
            return _price_cache[ticker]
    try:
        price = yf.Ticker(ticker).fast_info['lastPrice']
        with _cache_lock:
            _price_cache[ticker] = price
            _cache_timestamp[ticker] = now
        return price
    except:
        return 0.0

# ---------- DATA FETCHING ----------
@st.cache_data(ttl=300)
def Monitor(Ticker='FFWM', field=2):
    try:
        tickerData = yf.Ticker(Ticker).history(period='max')[['Close']].round(3)
        tickerData.index = tickerData.index.tz_convert(tz='Asia/bangkok')
        tickerData = tickerData[tickerData.index >= '2023-01-01 12:00:00+07:00']
        fx = client_2.get_field_last(field=str(field))
        fx_js = int(json.loads(fx)[f"field{field}"])
        rng = np.random.default_rng(fx_js)
        tickerData['action'] = rng.integers(2, size=len(tickerData))
        tickerData['index'] = [i+1 for i in range(len(tickerData))]
        # Add 5 dummy rows for future
        tickerData_1 = pd.DataFrame({'action': [i for i in range(5)]}, index=[f'+{i}' for i in range(5)])
        df = pd.concat([tickerData, tickerData_1], axis=0).fillna("")
        df['action'] = rng.integers(2, size=len(df))
        return df.tail(7), fx_js
    except Exception as e:
        st.error(f"Error in Monitor function for {Ticker}: {str(e)}")
        return pd.DataFrame(), 0

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
    def fetch_asset(field):
        try:
            data = client.get_field_last(field=field)
            return eval(json.loads(data)[field])
        except:
            return 0.0

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(configs)) as executor:
        future_to_field = {
            executor.submit(fetch_asset, asset['asset_field']): asset['asset_field']
            for asset in configs
        }
        for future in concurrent.futures.as_completed(future_to_field):
            field = future_to_field[future]
            try:
                ticker = next((c['ticker'] for c in configs if c['asset_field'] == field), None)
                if ticker:
                    assets[ticker] = future.result()
            except Exception as e:
                st.error(f"Error fetching asset for field {field}: {str(e)}")
    return assets

# ---------- UI SECTION ----------
def render_asset_inputs(configs, last_assets):
    cols = st.columns(len(configs))
    asset_inputs = {}
    for i, config in enumerate(configs):
        with cols[i]:
            ticker = config['ticker']
            last_asset_val = last_assets.get(ticker, 0.0)
            if config.get('option_config'):
                option_val = config['option_config']['base_value']
                label = config['option_config']['label']
                real_val = st.number_input(label, step=0.001, value=last_asset_val, key=f"input_{ticker}_real")
                asset_inputs[ticker] = option_val + real_val
            else:
                label = f'{ticker}_ASSET'
                asset_val = st.number_input(label, step=0.001, value=last_asset_val, key=f"input_{ticker}_asset")
                asset_inputs[ticker] = asset_val
    return asset_inputs

def render_asset_update_controls(configs):
    with st.expander("Update Assets on ThingSpeak"):
        for config in configs:
            ticker = config['ticker']
            field = config['asset_field']
            if st.checkbox(f'@_{ticker}_ASSET', key=f'check_{ticker}'):
                add_val = st.number_input(f"New Value for {ticker}", step=0.001, value=0.0, key=f'input_{ticker}')
                if st.button(f"GO_{ticker}", key=f'btn_{ticker}'):
                    client.update({field: add_val})
                    st.write(f"Updated {ticker} to: {add_val}")
                    clear_all_caches()

def trading_section(config, asset_val, asset_last, df_data, calc, nex, Nex_day_sell):
    ticker = config['ticker']
    asset_field = config['asset_field']
    field_num = int(asset_field.replace('field', ''))

    def get_action_val():
        try:
            if len(df_data) > 1 + nex:
                val = df_data.action.values[1 + nex]
                return 1 - val if Nex_day_sell == 1 else val
            return 0
        except Exception:
            return 0

    action_val = get_action_val()
    limit_order = st.checkbox(f'Limit_Order_{ticker}', value=bool(action_val), key=f'limit_order_{ticker}')
    if not limit_order:
        return

    sell_calc, buy_calc = calc['sell'], calc['buy']
    st.write('sell', '    ', 'A', buy_calc[1], 'P', buy_calc[0], 'C', buy_calc[2])
    col1, col2, col3 = st.columns(3)
    if col3.checkbox(f'sell_match_{ticker}'):
        if col3.button(f"GO_SELL_{ticker}"):
            client.update({f'field{field_num}': asset_last - buy_calc[1]})
            col3.write(asset_last - buy_calc[1])
            clear_all_caches()

    # Compact price info
    try:
        current_price = get_cached_price(ticker)
        if current_price > 0:
            pv = current_price * asset_val
            fix_value = config['fix_c']
            pl_value = pv - fix_value
            st.markdown(
                f"Price: **{current_price:,.3f}**  |  Portfolio Value: **{pv:,.2f}**  |  P/L (vs {fix_value:,}) :  <span style='color:alpha; font-weight:bold;'>{pl_value:,.2f}</span>",
                unsafe_allow_html=True
            )
        else:
            st.info(f"Price data for {ticker} is currently unavailable.")
    except Exception:
        st.warning(f"Could not retrieve price data for {ticker}.")

    col4, col5, col6 = st.columns(3)
    st.write('buy', '   ', 'A', sell_calc[1], 'P', sell_calc[0], 'C', sell_calc[2])
    if col6.checkbox(f'buy_match_{ticker}'):
        if col6.button(f"GO_BUY_{ticker}"):
            client.update({f'field{field_num}': asset_last + sell_calc[1]})
            col6.write(asset_last + sell_calc[1])
            clear_all_caches()

# ---------- MAIN LOGIC ----------
monitor_data_all = fetch_all_monitor_data(ASSET_CONFIGS)
last_assets_all = get_all_assets_from_thingspeak(ASSET_CONFIGS)

# Nex day toggle
nex, Nex_day_sell = 0, 0
Nex_day_ = st.checkbox('nex_day')
if Nex_day_:
    st.write("value = ", nex)
    nex_col, Nex_day_sell_col, *_ = st.columns(5)
    if nex_col.button("Nex_day"): nex = 1
    if Nex_day_sell_col.button("Nex_day_sell"):
        nex, Nex_day_sell = 1, 1
    st.write("value = ", nex)
    if Nex_day_sell: st.write("Nex_day_sell = ", Nex_day_sell)
st.write("_____")

# Asset controls
control_cols = st.columns(8)
x_2 = control_cols[7].number_input('Diff', step=1, value=60)
Start = control_cols[0].checkbox('start')
if Start:
    render_asset_update_controls(ASSET_CONFIGS)

asset_inputs = render_asset_inputs(ASSET_CONFIGS, last_assets_all)
st.write("_____")

# Calculations
calculations = {}
for config in ASSET_CONFIGS:
    ticker = config['ticker']
    asset_value = asset_inputs.get(ticker, 0.0)
    fix_c = config['fix_c']
    calculations[ticker] = {
        'sell': sell(asset_value, fix_c=fix_c, Diff=x_2),
        'buy': buy(asset_value, fix_c=fix_c, Diff=x_2)
    }

# Trading sections
for config in ASSET_CONFIGS:
    ticker = config['ticker']
    df_data, _ = monitor_data_all.get(ticker, (pd.DataFrame(), 0))
    asset_last = last_assets_all.get(ticker, 0.0)
    asset_val = asset_inputs.get(ticker, 0.0)
    calc = calculations.get(ticker, {})
    trading_section(config, asset_val, asset_last, df_data, calc, nex, Nex_day_sell)
    st.write("_____")

if st.sidebar.button("RERUN"):
    clear_all_caches()
