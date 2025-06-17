# 📈_Monitor.py (Updated)
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
    """Load configuration from a JSON file."""
    if not os.path.exists(file_path):
        st.error(f"Configuration file not found: {file_path}")
        return None #<-- แก้ไข: คืนค่า None หากไม่พบไฟล์
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

#<-- แก้ไข: โหลดและแยกส่วนคอนฟิก
CONFIG_DATA = load_config()
if not CONFIG_DATA:
    st.stop()

ASSET_CONFIGS = CONFIG_DATA.get('assets', [])
GLOBAL_START_DATE = CONFIG_DATA.get('global_settings', {}).get('start_date')

if not ASSET_CONFIGS:
    st.error("No 'assets' list found in monitor_config.json")
    st.stop()
#<-- สิ้นสุดส่วนที่แก้ไข

# ---------- GLOBAL CACHE & CLIENT MANAGEMENT ----------
_cache_lock = Lock()
_price_cache = {}
_cache_timestamp = {}

@st.cache_resource
def get_thingspeak_clients(configs):
    """
    Creates and caches a dictionary of ThingSpeak clients based on the config.
    Returns a dict mapping {channel_id: client_object}.
    """
    clients = {}
    unique_channels = set()
    for config in configs:
        # Add monitor channel details
        mon_conf = config['monitor_field']
        unique_channels.add((mon_conf['channel_id'], mon_conf['api_key']))
        # Add asset channel details
        asset_conf = config['asset_field']
        unique_channels.add((asset_conf['channel_id'], asset_conf['api_key']))

    for channel_id, api_key in unique_channels:
        try:
            clients[channel_id] = thingspeak.Channel(channel_id, api_key, fmt='json')
        except Exception as e:
            st.error(f"Failed to create client for Channel ID {channel_id}: {e}")
    return clients

# Get all required clients once
THINGSPEAK_CLIENTS = get_thingspeak_clients(ASSET_CONFIGS)

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
#<-- แก้ไข: รับพารามิเตอร์ start_date
def Monitor(asset_config, _clients_ref, start_date):
    """Fetches monitor data for a single ticker using its specific channel."""
    ticker = asset_config['ticker']
    try:
        monitor_field_config = asset_config['monitor_field']
        
        client = _clients_ref[monitor_field_config['channel_id']]
        field_num = monitor_field_config['field']
        
        tickerData = yf.Ticker(ticker).history(period='max')[['Close']].round(3)
        tickerData.index = tickerData.index.tz_convert(tz='Asia/bangkok')
        
        #<-- แก้ไข: ใช้ start_date ส่วนกลางที่รับเข้ามา
        if start_date:
            tickerData = tickerData[tickerData.index >= start_date]
        
        fx_raw = client.get_field_last(field=str(field_num))
        fx_js = int(json.loads(fx_raw)[f"field{field_num}"])
        
        rng = np.random.default_rng(fx_js)
        
        tickerData['index'] = [i+1 for i in range(len(tickerData))]
        dummy_df = pd.DataFrame(index=[f'+{i}' for i in range(5)])
        
        df = pd.concat([tickerData, dummy_df], axis=0).fillna("")
        
        df['action'] = rng.integers(2, size=len(df))
        
        return df.tail(7), fx_js
    except Exception as e:
        st.error(f"Error in Monitor function for {ticker}: {str(e)}")
        return pd.DataFrame(), 0

@st.cache_data(ttl=300)
#<-- แก้ไข: รับพารามิเตอร์ start_date
def fetch_all_monitor_data(configs, _clients_ref, start_date):
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(configs)) as executor:
        #<-- แก้ไข: ส่ง start_date ไปยังฟังก์ชัน Monitor
        future_to_ticker = {
            executor.submit(Monitor, asset, _clients_ref, start_date): asset['ticker']
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

def fetch_asset(asset_config, client):
    """Helper to fetch a single asset value."""
    try:
        field_name = asset_config['field']
        data = client.get_field_last(field=field_name)
        return float(json.loads(data)[field_name])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return 0.0

@st.cache_data(ttl=60)
def get_all_assets_from_thingspeak(configs, _clients_ref):
    assets = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(configs)) as executor:
        future_to_ticker = {
            executor.submit(
                fetch_asset,
                asset['asset_field'],
                _clients_ref[asset['asset_field']['channel_id']]
            ): asset['ticker']
            for asset in configs if asset['asset_field']['channel_id'] in _clients_ref
        }
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                assets[ticker] = future.result()
            except Exception as e:
                st.error(f"Error fetching asset for ticker {ticker}: {str(e)}")
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

def render_asset_update_controls(configs, clients):
    with st.expander("Update Assets on ThingSpeak"):
        for config in configs:
            ticker = config['ticker']
            asset_conf = config['asset_field']
            field_name = asset_conf['field']
            
            if st.checkbox(f'@_{ticker}_ASSET', key=f'check_{ticker}'):
                add_val = st.number_input(f"New Value for {ticker}", step=0.001, value=0.0, key=f'input_{ticker}')
                if st.button(f"GO_{ticker}", key=f'btn_{ticker}'):
                    try:
                        client = clients[asset_conf['channel_id']]
                        client.update({field_name: add_val})
                        st.write(f"Updated {ticker} to: {add_val} on Channel {asset_conf['channel_id']}")
                        clear_all_caches()
                    except Exception as e:
                        st.error(f"Failed to update {ticker}: {e}")

def trading_section(config, asset_val, asset_last, df_data, calc, nex, Nex_day_sell, clients):
    ticker = config['ticker']
    asset_conf = config['asset_field']
    field_name = asset_conf['field']

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
            try:
                client = clients[asset_conf['channel_id']]
                new_asset_val = asset_last - buy_calc[1]
                client.update({field_name: new_asset_val})
                col3.write(f"Updated: {new_asset_val}")
                clear_all_caches()
            except Exception as e:
                st.error(f"Failed to SELL {ticker}: {e}")

    try:
        current_price = get_cached_price(ticker)
        if current_price > 0:
            pv = current_price * asset_val
            fix_value = config['fix_c']
            pl_value = pv - fix_value
            pl_color = "#a8d5a2" if pl_value >= 0 else "#fbb"
            st.markdown(
                f"Price: **{current_price:,.3f}** | Portfolio Value: **{pv:,.2f}** | P/L (vs {fix_value:,}) : <span style='color:{pl_color}; font-weight:bold;'>{pl_value:,.2f}</span>",
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
            try:
                client = clients[asset_conf['channel_id']]
                new_asset_val = asset_last + sell_calc[1]
                client.update({field_name: new_asset_val})
                col6.write(f"Updated: {new_asset_val}")
                clear_all_caches()
            except Exception as e:
                st.error(f"Failed to BUY {ticker}: {e}")

# ---------- MAIN LOGIC ----------
#<-- แก้ไข: ส่ง GLOBAL_START_DATE ไปยังฟังก์ชัน
monitor_data_all = fetch_all_monitor_data(ASSET_CONFIGS, THINGSPEAK_CLIENTS, GLOBAL_START_DATE)
last_assets_all = get_all_assets_from_thingspeak(ASSET_CONFIGS, THINGSPEAK_CLIENTS)

nex, Nex_day_sell = 0, 0
Nex_day_ = st.checkbox('nex_day')
if Nex_day_:
    nex_col, Nex_day_sell_col, *_ = st.columns(5)
    if nex_col.button("Nex_day"): nex = 1
    if Nex_day_sell_col.button("Nex_day_sell"):
        nex, Nex_day_sell = 1, 1
    st.write(f"nex value = {nex}", f" | Nex_day_sell = {Nex_day_sell}" if Nex_day_sell else "")
st.write("_____")

control_cols = st.columns(8)
x_2 = control_cols[7].number_input('Diff', step=1, value=60)
Start = control_cols[0].checkbox('start')
if Start:
    render_asset_update_controls(ASSET_CONFIGS, THINGSPEAK_CLIENTS)

asset_inputs = render_asset_inputs(ASSET_CONFIGS, last_assets_all)
st.write("_____")

calculations = {}
for config in ASSET_CONFIGS:
    ticker = config['ticker']
    asset_value = asset_inputs.get(ticker, 0.0)
    fix_c = config['fix_c']
    calculations[ticker] = {
        'sell': sell(asset_value, fix_c=fix_c, Diff=x_2),
        'buy': buy(asset_value, fix_c=fix_c, Diff=x_2)
    }

for config in ASSET_CONFIGS:
    ticker = config['ticker']
    df_data, _ = monitor_data_all.get(ticker, (pd.DataFrame(), 0))
    asset_last = last_assets_all.get(ticker, 0.0)
    asset_val = asset_inputs.get(ticker, 0.0)
    calc = calculations.get(ticker, {})
    
    trading_section(config, asset_val, asset_last, df_data, calc, nex, Nex_day_sell, THINGSPEAK_CLIENTS)
    st.write("_____")

if st.sidebar.button("RERUN"):
    clear_all_caches()
