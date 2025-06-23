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

# ============== 1. PAGE CONFIGURATION ==============
st.set_page_config(
    page_title="Asset Monitor",
    page_icon="📈",
    layout="wide"
)

# ============== 2. CONFIGURATION LOADING ==============
@st.cache_data
def load_config(file_path='monitor_config.json'):
    """Load configuration from a JSON file."""
    if not os.path.exists(file_path):
        st.error(f"Configuration file not found: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        st.error(f"Error reading or parsing config file: {e}")
        return None

CONFIG_DATA = load_config()
if not CONFIG_DATA:
    st.error("Stopping execution due to configuration error.")
    st.stop()

ASSET_CONFIGS = CONFIG_DATA.get('assets', [])
GLOBAL_START_DATE = CONFIG_DATA.get('global_settings', {}).get('start_date')

if not ASSET_CONFIGS:
    st.error("No 'assets' list found in monitor_config.json")
    st.stop()

# ============== 3. CACHE & CLIENT MANAGEMENT ==============
_cache_lock = Lock()
_price_cache = {}
_cache_timestamp = {}

@st.cache_resource
def get_thingspeak_clients(configs):
    """Creates and caches a dictionary of ThingSpeak clients."""
    clients = {}
    unique_channels = set()
    for config in configs:
        mon_conf = config['monitor_field']
        unique_channels.add((mon_conf['channel_id'], mon_conf['api_key']))
        asset_conf = config['asset_field']
        unique_channels.add((asset_conf['channel_id'], asset_conf['api_key']))

    for channel_id, api_key in unique_channels:
        try:
            clients[channel_id] = thingspeak.Channel(channel_id, api_key, fmt='json')
        except Exception as e:
            st.error(f"Failed to create client for Channel ID {channel_id}: {e}")
    return clients

THINGSPEAK_CLIENTS = get_thingspeak_clients(ASSET_CONFIGS)

def clear_all_caches():
    """Clears all streamlit and custom caches, then reruns the app."""
    st.cache_data.clear()
    st.cache_resource.clear()
    sell.cache_clear()
    buy.cache_clear()
    with _cache_lock:
        _price_cache.clear()
        _cache_timestamp.clear()
    st.toast("🗑️ All caches cleared! Rerunning...")
    st.rerun()

# ============== 4. CORE LOGIC & CALCULATIONS (Unchanged) ==============
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

def get_cached_price(ticker, max_age_seconds=30):
    now = datetime.datetime.now()
    with _cache_lock:
        if (ticker in _price_cache and
            ticker in _cache_timestamp and
            (now - _cache_timestamp[ticker]).seconds < max_age_seconds):
            return _price_cache[ticker]
    try:
        price = yf.Ticker(ticker).fast_info.get('lastPrice')
        if price:
            with _cache_lock:
                _price_cache[ticker] = price
                _cache_timestamp[ticker] = now
            return price
        return 0.0
    except Exception: return 0.0

# ============== 5. DATA FETCHING (Unchanged) ==============
@st.cache_data(ttl=300)
def Monitor(asset_config, _clients_ref, start_date):
    ticker = asset_config['ticker']
    try:
        monitor_field_config = asset_config['monitor_field']
        client = _clients_ref[monitor_field_config['channel_id']]
        field_num = monitor_field_config['field']
        tickerData = yf.Ticker(ticker).history(period='max')[['Close']].round(3)
        tickerData.index = tickerData.index.tz_convert(tz='Asia/bangkok')
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
    except: return pd.DataFrame(), 0

@st.cache_data(ttl=300)
def fetch_all_monitor_data(configs, _clients_ref, start_date):
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(configs)) as executor:
        future_to_ticker = {executor.submit(Monitor, c, _clients_ref, start_date): c['ticker'] for c in configs}
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                results[ticker] = future.result()
            except: results[ticker] = (pd.DataFrame(), 0)
    return results

def fetch_asset(asset_config, client):
    try:
        field_name = asset_config['field']
        data = client.get_field_last(field=field_name)
        return float(json.loads(data)[field_name])
    except: return 0.0

@st.cache_data(ttl=60)
def get_all_assets_from_thingspeak(configs, _clients_ref):
    assets = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(configs)) as executor:
        future_to_ticker = {
            executor.submit(fetch_asset, asset['asset_field'], _clients_ref[asset['asset_field']['channel_id']]): asset['ticker']
            for asset in configs if asset['asset_field']['channel_id'] in _clients_ref
        }
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                assets[ticker] = future.result()
            except: assets[ticker] = 0.0
    return assets

# ============== 6. MAIN APP UI & EXECUTION ==============
st.title("📈 Asset Monitor")
st.markdown("---")

# --- Global Control Panel ---
with st.container():
    col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1.5])
    with col1:
        diff_value = st.number_input('Difference (Diff)', step=1, value=60, label_visibility="collapsed")
    with col2:
        use_nex_day = st.toggle('Next Day', help="Use signal for the next day.")
    with col3:
        invert_signal = st.toggle('Invert', disabled=not use_nex_day, help="Invert the next day's signal (SELL becomes BUY and vice-versa).")
    with col4:
        if st.button("🔄 Refresh Data", use_container_width=True):
            clear_all_caches()

nex = 1 if use_nex_day else 0
nex_day_sell = 1 if invert_signal else 0

st.markdown("---")


# --- Fetch all data once ---
monitor_data_all = fetch_all_monitor_data(ASSET_CONFIGS, THINGSPEAK_CLIENTS, GLOBAL_START_DATE)
last_assets_all = get_all_assets_from_thingspeak(ASSET_CONFIGS, THINGSPEAK_CLIENTS)


# --- Loop and Display Each Asset Card ---
for config in ASSET_CONFIGS:
    ticker = config['ticker']
    asset_name = config.get('name', ticker)
    
    with st.container(border=True):
        # --- Get data for this specific asset ---
        df_data, _ = monitor_data_all.get(ticker, (pd.DataFrame(), 0))
        last_asset = last_assets_all.get(ticker, 0.0)
        current_price = get_cached_price(ticker)
        
        # --- Calculations for this asset ---
        portfolio_value = current_price * last_asset
        fix_value = config['fix_c']
        pl_value = portfolio_value - fix_value if last_asset > 0 else 0
        
        # --- Header with Manual Update Popover ---
        header_cols = st.columns([0.8, 0.2])
        with header_cols[0]:
            st.subheader(f"{asset_name} ({ticker})")
        with header_cols[1]:
            # <<< นี่คือส่วนที่เพิ่มเข้ามา: Manual Update Popover >>>
            with st.popover("📝 Edit", use_container_width=True):
                st.markdown(f"**Update {ticker} Asset**")
                new_val = st.number_input(
                    "New asset value",
                    value=last_asset,
                    step=0.001,
                    key=f"manual_input_{ticker}",
                    label_visibility="collapsed"
                )
                if st.button("Update on ThingSpeak", key=f"manual_btn_{ticker}"):
                    try:
                        client = THINGSPEAK_CLIENTS[config['asset_field']['channel_id']]
                        client.update({config['asset_field']['field']: new_val})
                        st.toast(f"Updated {ticker} to {new_val}!", icon="✅")
                        clear_all_caches()
                    except Exception as e:
                        st.error(f"Update failed: {e}")
        
        # --- Status Metrics ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Asset", f"{last_asset:,.3f}")
        m2.metric("Current Price", f"{current_price:,.3f}")
        m3.metric("Portfolio Value", f"{portfolio_value:,.2f}")
        m4.metric(f"P/L", f"{pl_value:,.2f}", delta=f"{pl_value:,.2f}" if pl_value != 0 else None)
        
        st.divider()

        # --- Trading Action Section ---
        show_actions = False
        try:
            action_val = df_data.action.values[1 + nex]
            show_actions = bool(1 - action_val if nex_day_sell == 1 else action_val)
        except:
            show_actions = False

        if not show_actions:
            st.info("Signal: **HOLD** (No action required)")
        else:
            sell_calc = sell(last_asset, fix_c=fix_value, Diff=diff_value)
            buy_calc = buy(last_asset, fix_c=fix_value, Diff=diff_value)
            
            sell_col, buy_col = st.columns(2)
            
            with sell_col:
                st.markdown("**SELL Opportunity**")
                st.markdown(f"**Price:** `{buy_calc[0]:,.2f}` | **Qty:** `{buy_calc[1]:,.3f}` | **Total:** `{buy_calc[2]:,.2f}`")
                if st.button(f"🔴 SELL {ticker}", key=f"go_sell_{ticker}", use_container_width=True):
                    try:
                        client = THINGSPEAK_CLIENTS[config['asset_field']['channel_id']]
                        new_asset_val = last_asset - buy_calc[1]
                        client.update({config['asset_field']['field']: new_asset_val})
                        st.success(f"SELL order sent! New asset: {new_asset_val:.3f}")
                        st.toast(f"SELL {ticker} Successful!", icon="⬇️")
                        clear_all_caches()
                    except Exception as e:
                        st.error(f"SELL failed: {e}")

            with buy_col:
                st.markdown("**BUY Opportunity**")
                st.markdown(f"**Price:** `{sell_calc[0]:,.2f}` | **Qty:** `{sell_calc[1]:,.3f}` | **Total:** `{sell_calc[2]:,.2f}`")
                if st.button(f"🟢 BUY {ticker}", key=f"go_buy_{ticker}", use_container_width=True):
                    try:
                        client = THINGSPEAK_CLIENTS[config['asset_field']['channel_id']]
                        new_asset_val = last_asset + sell_calc[1]
                        client.update({config['asset_field']['field']: new_asset_val})
                        st.success(f"BUY order sent! New asset: {new_asset_val:.3f}")
                        st.toast(f"BUY {ticker} Successful!", icon="⬆️")
                        clear_all_caches()
                    except Exception as e:
                        st.error(f"BUY failed: {e}")
