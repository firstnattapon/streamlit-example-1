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
    layout="wide",
    initial_sidebar_state="expanded"
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
    st.success("🗑️ All caches cleared! Rerunning...")
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
    """Gets price from cache or fetches if stale/missing."""
    now = datetime.datetime.now()
    with _cache_lock:
        if (ticker in _price_cache and
            ticker in _cache_timestamp and
            (now - _cache_timestamp[ticker]).seconds < max_age_seconds):
            return _price_cache[ticker]
    try:
        # Use fast_info for quicker price retrieval
        price = yf.Ticker(ticker).fast_info.get('lastPrice')
        if price:
            with _cache_lock:
                _price_cache[ticker] = price
                _cache_timestamp[ticker] = now
            return price
        return 0.0
    except Exception:
        return 0.0

# ============== 5. DATA FETCHING (Unchanged logic, but Monitor returns full df) ==============
@st.cache_data(ttl=300)
def Monitor(asset_config, _clients_ref, start_date):
    """
    Fetches monitor data for a single ticker using its specific channel.
    NOW RETURNS THE FULL DATAFRAME.
    """
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
        
        # MODIFICATION: Return the full DataFrame, not just the tail.
        return df, fx_js
    except Exception as e:
        st.error(f"Error in Monitor({ticker}): {e}")
        return pd.DataFrame(), 0

@st.cache_data(ttl=300)
def fetch_all_monitor_data(configs, _clients_ref, start_date):
    """Fetches all monitor data concurrently."""
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(configs)) as executor:
        future_to_ticker = {
            executor.submit(Monitor, asset, _clients_ref, start_date): asset['ticker']
            for asset in configs
        }
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                results[ticker] = future.result()
            except Exception as e:
                st.error(f"Error fetching monitor data for {ticker}: {e}")
                results[ticker] = (pd.DataFrame(), 0)
    return results

def fetch_asset(asset_config, client):
    """Helper to fetch a single asset value from ThingSpeak."""
    try:
        field_name = asset_config['field']
        data = client.get_field_last(field=field_name)
        return float(json.loads(data)[field_name])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return 0.0

@st.cache_data(ttl=60)
def get_all_assets_from_thingspeak(configs, _clients_ref):
    """Fetches all asset quantities from ThingSpeak concurrently."""
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
                st.error(f"Error fetching asset for ticker {ticker}: {e}")
    return assets

# ============== 6. UI RENDERING FUNCTIONS ==============
def render_asset_update_controls(configs, clients):
    """UI for manually updating asset values on ThingSpeak."""
    st.subheader("Manual Asset Update")
    for config in configs:
        ticker = config['ticker']
        asset_conf = config['asset_field']
        field_name = asset_conf['field']
        
        with st.container(border=True):
            st.markdown(f"**Update {ticker}**")
            new_val = st.number_input(f"New Value for {ticker}", step=0.001, value=0.0, key=f'input_{ticker}', label_visibility="collapsed")
            if st.button(f"Update {ticker} on ThingSpeak", key=f'btn_{ticker}', use_container_width=True):
                try:
                    client = clients[asset_conf['channel_id']]
                    client.update({field_name: new_val})
                    st.success(f"Updated {ticker} to: {new_val} on Channel {asset_conf['channel_id']}")
                    st.toast(f"Success! {ticker} updated.", icon="✅")
                    # Clear asset cache to reflect the update immediately
                    get_all_assets_from_thingspeak.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to update {ticker}: {e}")

def render_asset_tab(config, monitor_data, last_asset, nex, nex_day_sell, diff_val, clients):
    """Renders the content for a single asset tab."""
    ticker = config['ticker']
    df_data, _ = monitor_data
    
    # --- Column for Status & Inputs ---
    col_status, col_input = st.columns([2, 1])

    with col_status:
        st.header(f"{config.get('name', ticker)}")
        current_price = get_cached_price(ticker)
        
        # Use last asset value from ThingSpeak for P/L calculation
        portfolio_value = current_price * last_asset
        fix_value = config['fix_c']
        pl_value = portfolio_value - fix_value if last_asset > 0 else 0
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Current Price", f"{current_price:,.3f}")
        m2.metric("Portfolio Value", f"{portfolio_value:,.2f}")
        m3.metric(f"P/L (vs {fix_value:,})", f"{pl_value:,.2f}", delta=f"{pl_value:,.2f}" if pl_value != 0 else None)
        
    with col_input:
        # Asset input for simulation/calculation
        st.subheader("Calculation Input")
        asset_input_val = st.number_input(
            "Asset Quantity for Calculation",
            step=0.001,
            value=last_asset,
            key=f"input_{ticker}_asset",
            help="This value is used for BUY/SELL calculations below. It defaults to the last value from ThingSpeak."
        )

    st.divider()

    # --- Trading Section ---
    st.subheader("Trading Actions")
    
    # Determine if trading is active
    try:
        # MODIFICATION: The logic now looks at the tail of the full df_data to get the future signals,
        # preserving the original behavior while allowing the full dataframe to be displayed.
        action_val = df_data.tail(7).action.values[1 + nex]
        limit_order_default = bool(1 - action_val if nex_day_sell == 1 else action_val)
    except Exception:
        limit_order_default = False

    if not st.toggle(f'Activate Limit Order for {ticker}', value=limit_order_default, key=f'limit_order_{ticker}'):
        st.info("Limit Order is not active for this asset based on the current signal.")
        return

    # Calculate Buy/Sell based on user input
    sell_calc = sell(asset_input_val, fix_c=config['fix_c'], Diff=diff_val)
    buy_calc = buy(asset_input_val, fix_c=config['fix_c'], Diff=diff_val)

    sell_col, buy_col = st.columns(2)

    with sell_col:
        with st.container(border=True):
            st.markdown("<h5 style='text-align: center; color: #ff4b4b;'>SELL Action</h5>", unsafe_allow_html=True)
            st.markdown(f"""
            - **Target Price:** `{buy_calc[0]:,.2f}`
            - **Quantity:** `{buy_calc[1]:,.3f}`
            - **Total Cost:** `{buy_calc[2]:,.2f}`
            """)
            if st.button(f"⬇️ Execute SELL", key=f"go_sell_{ticker}", use_container_width=True):
                try:
                    client = clients[config['asset_field']['channel_id']]
                    new_asset_val = last_asset - buy_calc[1]
                    client.update({config['asset_field']['field']: new_asset_val})
                    st.success(f"SELL order sent! New asset value for {ticker}: {new_asset_val:.3f}")
                    st.toast(f"SELL {ticker} Successful!", icon="⬇️")
                    clear_all_caches()
                except Exception as e:
                    st.error(f"Failed to execute SELL for {ticker}: {e}")

    with buy_col:
        with st.container(border=True):
            st.markdown("<h5 style='text-align: center; color: #26a32c;'>BUY Action</h5>", unsafe_allow_html=True)
            st.markdown(f"""
            - **Target Price:** `{sell_calc[0]:,.2f}`
            - **Quantity:** `{sell_calc[1]:,.3f}`
            - **Total Cost:** `{sell_calc[2]:,.2f}`
            """)
            if st.button(f"⬆️ Execute BUY", key=f"go_buy_{ticker}", use_container_width=True):
                try:
                    client = clients[config['asset_field']['channel_id']]
                    new_asset_val = last_asset + sell_calc[1]
                    client.update({config['asset_field']['field']: new_asset_val})
                    st.success(f"BUY order sent! New asset value for {ticker}: {new_asset_val:.3f}")
                    st.toast(f"BUY {ticker} Successful!", icon="⬆️")
                    clear_all_caches()
                except Exception as e:
                    st.error(f"Failed to execute BUY for {ticker}: {e}")
    
    # MODIFICATION: This expander now correctly shows the full dataframe.
    with st.expander("Show Raw Monitor Data"):
        st.dataframe(df_data, use_container_width=True)


# ============== 7. MAIN APP EXECUTION ==============
# --- Sidebar Controls ---
with st.sidebar:
    st.title("⚙️ Controls")
    st.divider()
    diff_value = st.number_input('Difference (Diff)', step=1, value=60, help="Value used in BUY/SELL price calculation.")
    st.divider()
    if st.button("🔄 Force Refresh & Clear Cache", use_container_width=True):
        clear_all_caches()
    st.caption(f"Last data refresh will be timestamped in cache.")

# --- Fetch all data once ---
monitor_data_all = fetch_all_monitor_data(ASSET_CONFIGS, THINGSPEAK_CLIENTS, GLOBAL_START_DATE)
last_assets_all = get_all_assets_from_thingspeak(ASSET_CONFIGS, THINGSPEAK_CLIENTS)

# --- Main Page Title ---
st.title("📈 Asset Trading Monitor")
st.markdown("Monitor assets and execute trades based on calculated signals.")

# --- Global Controls Expander ---
with st.expander("🛠️ Global Settings & Manual Updates"):
    st.subheader("Signal Timing Control")
    nex, nex_day_sell = 0, 0
    if st.checkbox('Use Next Day Signal (nex_day)'):
        nex = 1
        if st.checkbox('Invert Next Day Signal (Nex_day_sell)'):
            nex_day_sell = 1
    st.info(f"Current Signal Mode: `nex` = {nex}, `Nex_day_sell` = {nex_day_sell}")
    
    st.divider()
    render_asset_update_controls(ASSET_CONFIGS, THINGSPEAK_CLIENTS)

# --- Create a tab for each asset ---
asset_tickers = [config['ticker'] for config in ASSET_CONFIGS]
asset_tabs = st.tabs(asset_tickers)

# --- Render each tab with its data ---
for i, tab in enumerate(asset_tabs):
    with tab:
        config = ASSET_CONFIGS[i]
        ticker = config['ticker']
        
        monitor_data = monitor_data_all.get(ticker, (pd.DataFrame(), 0))
        last_asset = last_assets_all.get(ticker, 0.0)
        
        render_asset_tab(
            config,
            monitor_data,
            last_asset,
            nex,
            nex_day_sell,
            diff_value,
            THINGSPEAK_CLIENTS
        )
