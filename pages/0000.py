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
    if not os.path.exists(file_path):
        st.error(f"Config file not found: {file_path}"); return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f: return json.load(f)
    except Exception as e:
        st.error(f"Error reading config: {e}"); return None

CONFIG_DATA = load_config()
if not CONFIG_DATA: st.stop()
ASSET_CONFIGS = CONFIG_DATA.get('assets', [])
GLOBAL_START_DATE = CONFIG_DATA.get('global_settings', {}).get('start_date')
if not ASSET_CONFIGS: st.error("No 'assets' in config"); st.stop()

# ============== 3. CACHE & CLIENT MANAGEMENT ==============
_cache_lock = Lock()
_price_cache = {}
_cache_timestamp = {}

@st.cache_resource
def get_thingspeak_clients(configs):
    clients = {}
    unique_channels = set()
    for config in configs:
        unique_channels.add((config['monitor_field']['channel_id'], config['monitor_field']['api_key']))
        unique_channels.add((config['asset_field']['channel_id'], config['asset_field']['api_key']))
    for channel_id, api_key in unique_channels:
        try:
            clients[channel_id] = thingspeak.Channel(channel_id, api_key, fmt='json')
        except Exception as e:
            st.error(f"Client fail for Chan ID {channel_id}: {e}")
    return clients

THINGSPEAK_CLIENTS = get_thingspeak_clients(ASSET_CONFIGS)

def clear_all_caches():
    st.cache_data.clear(); st.cache_resource.clear()
    sell.cache_clear(); buy.cache_clear()
    with _cache_lock: _price_cache.clear(); _cache_timestamp.clear()
    st.toast("🗑️ Caches cleared!", icon="✅")
    st.rerun()

# ============== 4. CORE LOGIC & CALCULATIONS ==============
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
        if ticker in _price_cache and (now - _cache_timestamp.get(ticker, now)).seconds < max_age_seconds:
            return _price_cache[ticker]
    try:
        price = yf.Ticker(ticker).fast_info.get('lastPrice')
        if price:
            with _cache_lock: _price_cache[ticker] = price; _cache_timestamp[ticker] = now
            return price
    except: pass
    return 0.0

# ============== 5. DATA FETCHING ==============
@st.cache_data(ttl=300)
def Monitor(asset_config, _clients_ref, start_date):
    try:
        client = _clients_ref[asset_config['monitor_field']['channel_id']]
        field_num = asset_config['monitor_field']['field']
        tickerData = yf.Ticker(asset_config['ticker']).history(period='max')[['Close']].round(3)
        tickerData.index = tickerData.index.tz_convert(tz='Asia/bangkok')
        if start_date: tickerData = tickerData[tickerData.index >= start_date]
        fx_raw = client.get_field_last(field=str(field_num))
        fx_js = int(json.loads(fx_raw)[f"field{field_num}"])
        rng = np.random.default_rng(fx_js)
        df = pd.concat([tickerData, pd.DataFrame(index=[f'+{i}' for i in range(5)])], axis=0).fillna("")
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
            try: results[ticker] = future.result()
            except: results[ticker] = (pd.DataFrame(), 0)
    return results

@st.cache_data(ttl=60)
def get_all_assets_from_thingspeak(configs, _clients_ref):
    assets = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(configs)) as executor:
        future_to_ticker = {
            executor.submit(
                lambda conf, cli: float(json.loads(cli.get_field_last(field=conf['field']))[conf['field']]),
                asset['asset_field'], _clients_ref[asset['asset_field']['channel_id']]
            ): asset['ticker']
            for asset in configs if asset['asset_field']['channel_id'] in _clients_ref
        }
        for future in concurrent.futures.as_completed(future_to_ticker):
            try: assets[future_to_ticker[future]] = future.result()
            except: assets[future_to_ticker[future]] = 0.0
    return assets

# ============== 6. MAIN APP UI & EXECUTION ==============
st.title("📈 Asset Monitor")

# --- Global Control Panel ---
c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1.5])
diff_value = c1.number_input('Diff', value=60, label_visibility="collapsed")
use_nex_day = c2.toggle('Next Day', help="Use signal for the next day.")
invert_signal = c3.toggle('Invert', disabled=not use_nex_day, help="Invert the signal.")
if c4.button("🔄 Refresh", use_container_width=True): clear_all_caches()
nex = 1 if use_nex_day else 0
nex_day_sell = 1 if invert_signal else 0
st.markdown("---",-1)

# --- Fetch all data once ---
monitor_data_all = fetch_all_monitor_data(ASSET_CONFIGS, THINGSPEAK_CLIENTS, GLOBAL_START_DATE)
last_assets_all = get_all_assets_from_thingspeak(ASSET_CONFIGS, THINGSPEAK_CLIENTS)

# --- Loop and Display Each Asset Card ---
for config in ASSET_CONFIGS:
    ticker = config['ticker']
    asset_name = config.get('name', ticker)
    
    with st.container(border=True):
        # --- Get & Calculate Data ---
        df_data, _ = monitor_data_all.get(ticker, (pd.DataFrame(), 0))
        last_asset = last_assets_all.get(ticker, 0.0)
        current_price = get_cached_price(ticker)
        portfolio_value = current_price * last_asset
        fix_value = config['fix_c']
        pl_value = portfolio_value - fix_value if last_asset > 0 else 0
        
        # --- Row 1: Info Metrics & Edit Button ---
        r1_cols = st.columns([3, 2, 2, 2, 0.8])
        r1_cols[0].subheader(f"{asset_name}")
        r1_cols[1].metric("Asset", f"{last_asset:,.3f}")
        r1_cols[2].metric("Price", f"{current_price:,.3f}")
        r1_cols[3].metric("P/L", f"{pl_value:,.2f}", delta=f"{pl_value:,.2f}" if pl_value != 0 else None)
        with r1_cols[4].popover("📝", help=f"Edit {ticker} Asset", use_container_width=True):
            new_val = st.number_input("New Value", value=last_asset, step=0.001, key=f"m_{ticker}", label_visibility="collapsed")
            if st.button("Update", key=f"b_{ticker}"):
                try:
                    client = THINGSPEAK_CLIENTS[config['asset_field']['channel_id']]
                    client.update({config['asset_field']['field']: new_val})
                    st.toast(f"Updated {ticker}!", icon="✅")
                    clear_all_caches()
                except Exception as e:
                    st.error(f"Update failed: {e}")
        
        # --- Row 2: Trading Actions ---
        show_actions = False
        try:
            action_val = df_data.action.values[1 + nex]
            show_actions = bool(1 - action_val if nex_day_sell == 1 else action_val)
        except: pass

        if not show_actions:
            st.caption("Signal: HOLD")
        else:
            sell_calc = sell(last_asset, fix_c=fix_value, Diff=diff_value)
            buy_calc = buy(last_asset, fix_c=fix_value, Diff=diff_value)
            
            action_cols = st.columns(2)
            
            with action_cols[0]:
                sell_text = f"🔴 SELL {ticker}\nQty: {buy_calc[1]:.3f} @ {buy_calc[0]:.2f}"
                if st.button(sell_text, key=f"sell_{ticker}", use_container_width=True):
                    try:
                        client = THINGSPEAK_CLIENTS[config['asset_field']['channel_id']]
                        new_asset = last_asset - buy_calc[1]
                        client.update({config['asset_field']['field']: new_asset})
                        st.toast(f"SELL {ticker} executed!", icon="⬇️")
                        clear_all_caches()
                    except Exception as e: st.error(f"SELL failed: {e}")

            with action_cols[1]:
                buy_text = f"🟢 BUY {ticker}\nQty: {sell_calc[1]:.3f} @ {sell_calc[0]:.2f}"
                if st.button(buy_text, key=f"buy_{ticker}", use_container_width=True):
                    try:
                        client = THINGSPEAK_CLIENTS[config['asset_field']['channel_id']]
                        new_asset = last_asset + sell_calc[1]
                        client.update({config['asset_field']['field']: new_asset})
                        st.toast(f"BUY {ticker} executed!", icon="⬆️")
                        clear_all_caches()
                    except Exception as e: st.error(f"BUY failed: {e}")
