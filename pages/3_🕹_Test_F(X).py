# 📈_Monitor.py
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
from typing import List

st.set_page_config(page_title="Monitor", page_icon="📈", layout="wide" , initial_sidebar_state = "expanded")

# --- START: โค้ดจาก action_simulationTracer.py ---
class SimulationTracer:
    """
    คลาสสำหรับห่อหุ้มกระบวนการทั้งหมด ตั้งแต่การถอดรหัสพารามิเตอร์
    ไปจนถึงการจำลองกระบวนการกลายพันธุ์ของ action sequence
    """
    def __init__(self, encoded_string: str):
        self.encoded_string: str = encoded_string
        # ทำให้แน่ใจว่าค่าที่เข้ามาเป็นสตริงเสมอ
        if not isinstance(self.encoded_string, str):
            self.encoded_string = str(self.encoded_string)

        self._decode_and_set_attributes()

    def _decode_and_set_attributes(self):
        encoded_string = self.encoded_string
        if not encoded_string.isdigit():
            self.action_length: int = 0
            self.mutation_rate: int = 0
            self.dna_seed: int = 0
            self.mutation_seeds: List[int] = []
            self.mutation_rate_float: float = 0.0
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
            pass

        if len(decoded_numbers) < 3:
            self.action_length: int = 0
            self.mutation_rate: int = 0
            self.dna_seed: int = 0
            self.mutation_seeds: List[int] = []
            self.mutation_rate_float: float = 0.0
            return

        self.action_length: int = decoded_numbers[0]
        self.mutation_rate: int = decoded_numbers[1]
        self.dna_seed: int = decoded_numbers[2]
        self.mutation_seeds: List[int] = decoded_numbers[3:]
        self.mutation_rate_float: float = self.mutation_rate / 100.0

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

# --- END: โค้ดจาก action_simulationTracer.py ---


# ---------- CONFIGURATION ----------
@st.cache_data
def load_config(file_path='monitor_config.json'):
    """Load configuration from a JSON file."""
    if not os.path.exists(file_path):
        st.error(f"Configuration file not found: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from {file_path}: {e}")
        return None
    except Exception as e:
        st.error(f"Error reading configuration file {file_path}: {e}")
        return None


CONFIG_DATA = load_config()
if not CONFIG_DATA:
    st.stop()

ASSET_CONFIGS = CONFIG_DATA.get('assets', [])
GLOBAL_START_DATE = CONFIG_DATA.get('global_settings', {}).get('start_date')

if not ASSET_CONFIGS:
    st.error("No 'assets' list found in monitor_config.json")
    st.stop()

# ---------- GLOBAL CACHE & CLIENT MANAGEMENT ----------
_cache_lock = Lock()
_price_cache = {}
_cache_timestamp = {}

@st.cache_resource
def get_thingspeak_clients(configs):
    """Creates and caches a dictionary of ThingSpeak clients based on the config."""
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
    except Exception:
        return 0.0

# ---------- DATA FETCHING ----------
@st.cache_data(ttl=300)
def Monitor(asset_config, _clients_ref, start_date):
    """
    Fetches monitor data and generates actions using SimulationTracer.
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
        
        fx_js_str = "0"
        try:
            field_data = client.get_field_last(field=str(field_num))
            retrieved_val = json.loads(field_data)[f"field{field_num}"]
            if retrieved_val is not None:
                fx_js_str = str(retrieved_val)
        except (json.JSONDecodeError, KeyError, TypeError, thingspeak.ThingSpeakError):
            fx_js_str = "0"

        tickerData['index'] = list(range(len(tickerData)))

        dummy_df = pd.DataFrame(index=[f'+{i}' for i in range(5)])
        df = pd.concat([tickerData, dummy_df], axis=0).fillna("")
        df['action'] = ""

        try:
            tracer = SimulationTracer(encoded_string=fx_js_str)
            final_actions = tracer.run()
            
            num_to_assign = min(len(df), len(final_actions))
            if num_to_assign > 0:
                action_col_idx = df.columns.get_loc('action')
                df.iloc[0:num_to_assign, action_col_idx] = final_actions[0:num_to_assign]
        
        except ValueError as e:
            st.warning(f"Tracer Error for {ticker} (input: '{fx_js_str}'): {e}")
        except Exception as e:
            st.error(f"Unexpected Tracer Error for {ticker}: {e}")

        return df.tail(7), fx_js_str
    
    except Exception as e:
        st.error(f"Error in Monitor function for {ticker}: {str(e)}")
        return pd.DataFrame(), "0"

@st.cache_data(ttl=300)
def fetch_all_monitor_data(configs, _clients_ref, start_date):
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
                st.error(f"Error fetching monitor data for {ticker}: {str(e)}")
                results[ticker] = (pd.DataFrame(), "0")
    return results

def fetch_asset(asset_config, client):
    """Helper to fetch a single asset value."""
    try:
        field_name = asset_config['field']
        data = client.get_field_last(field=field_name)
        return float(json.loads(data)[field_name])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError, thingspeak.ThingSpeakError):
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
                assets[ticker] = 0.0
    return assets

# ---------- UI SECTION ----------
def render_asset_inputs(configs, last_assets):
    num_cols = len(configs)
    cols = st.columns(num_cols)
    asset_inputs = {}
    for i, config in enumerate(configs):
        with cols[i]:
            ticker = config['ticker']
            last_asset_val = last_assets.get(ticker, 0.0)
            
            if config.get('option_config'):
                option_val = config['option_config']['base_value']
                label = config['option_config']['label']
                real_val = st.number_input(label, step=0.001, value=last_asset_val, key=f"input_{ticker}_real", format="%.2f")
                asset_inputs[ticker] = option_val + real_val
            else:
                label = f'{ticker}_ASSET'
                asset_val = st.number_input(label, step=0.001, value=last_asset_val, key=f"input_{ticker}_asset", format="%.2f")
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
                        st.success(f"Updated {ticker} to: {add_val} on Channel {asset_conf['channel_id']}")
                        clear_all_caches()
                    except Exception as e:
                        st.error(f"Failed to update {ticker}: {e}")

def trading_section(config, asset_val, asset_last, df_data, calc, nex, Nex_day_sell, clients):
    ticker = config['ticker']
    asset_conf = config['asset_field']
    field_name = asset_conf['field']

    def get_action_val():
        try:
            if df_data.empty or df_data.action.values[1 + nex] == "":
                return 0
            val = int(df_data.action.values[1 + nex])
            return 1 - val if Nex_day_sell == 1 else val
        except (IndexError, ValueError):
            return 0

    action_val = get_action_val()
    limit_order = st.checkbox(f'Limit_Order_{ticker}', value=bool(action_val), key=f'limit_order_{ticker}')
    if not limit_order:
        return

    buy_calc, sell_calc = calc['buy'], calc['sell'] # Swapped logic to match display
    
    # Sell section (uses buy_calc logic from original code)
    st.write('sell', '    ', 'A', buy_calc[1], 'P', buy_calc[0], 'C', buy_calc[2])
    _, _, col3 = st.columns(3)
    if col3.checkbox(f'sell_match_{ticker}', key=f"sell_match_check_{ticker}"):
        if col3.button(f"GO_SELL_{ticker}", key=f"go_sell_btn_{ticker}"):
            try:
                client = clients[asset_conf['channel_id']]
                new_asset_val = asset_last - buy_calc[1]
                client.update({field_name: new_asset_val})
                col3.success(f"Updated: {new_asset_val}")
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
                f"Price: **{current_price:,.3f}** | Value: **{pv:,.2f}** | P/L (vs {fix_value:,}) : <span style='color:{pl_color}; font-weight:bold;'>{pl_value:,.2f}</span>",
                unsafe_allow_html=True
            )
        else:
            st.info(f"Price data for {ticker} is currently unavailable.")
    except Exception:
        st.warning(f"Could not retrieve price data for {ticker}.")

    # Buy section (uses sell_calc logic from original code)
    st.write('buy', '   ', 'A', sell_calc[1], 'P', sell_calc[0], 'C', sell_calc[2])
    _, _, col6 = st.columns(3)
    if col6.checkbox(f'buy_match_{ticker}', key=f"buy_match_check_{ticker}"):
        if col6.button(f"GO_BUY_{ticker}", key=f"go_buy_btn_{ticker}"):
            try:
                client = clients[asset_conf['channel_id']]
                new_asset_val = asset_last + sell_calc[1]
                client.update({field_name: new_asset_val})
                col6.success(f"Updated: {new_asset_val}")
                clear_all_caches()
            except Exception as e:
                st.error(f"Failed to BUY {ticker}: {e}")

# ---------- MAIN LOGIC ----------
monitor_data_all = fetch_all_monitor_data(ASSET_CONFIGS, THINGSPEAK_CLIENTS, GLOBAL_START_DATE)
last_assets_all = get_all_assets_from_thingspeak(ASSET_CONFIGS, THINGSPEAK_CLIENTS)

# --- START: REORDERED CONTROL SECTION ---
nex, Nex_day_sell = 0, 0

with st.expander("⚙️ Controls & Asset Setup", expanded=False):
    
    # --- 1. nex_day section ---
    Nex_day_ = st.checkbox('nex_day')
    if Nex_day_:
        nex_col, sell_col, _ = st.columns([1, 1, 6]) 
        if nex_col.button("Nex_day"):
            nex = 1
        if sell_col.button("Nex_day_sell"):
            nex, Nex_day_sell = 1, 1
        
        status_text = f"nex value = {nex}"
        if Nex_day_sell:
            status_text += f" | Nex_day_sell = {Nex_day_sell}"
        st.write(status_text)
    
    st.write("---") 

    # --- 2. start section ---
    control_cols = st.columns(8)
    Start = control_cols[0].checkbox('start')
    x_2 = control_cols[7].number_input('Diff', step=1, value=60) 
    
    if Start:
        render_asset_update_controls(ASSET_CONFIGS, THINGSPEAK_CLIENTS)

    # --- 3. Asset Holdings section ---
    with st.expander("Asset Holdings", expanded=True):
        asset_inputs = render_asset_inputs(ASSET_CONFIGS, last_assets_all)

st.write("_____")
# --- END: REORDERED CONTROL SECTION ---


calculations = {}
for config in ASSET_CONFIGS:
    ticker = config['ticker']
    asset_value = asset_inputs.get(ticker, 0.0)
    fix_c = config['fix_c']
    calculations[ticker] = {
        'sell': sell(asset_value, fix_c=fix_c, Diff=x_2),
        'buy': buy(asset_value, fix_c=fix_c, Diff=x_2)
    }


# --- START: ASSET TABS WRAPPED IN EXPANDER ---
with st.expander("📈 Trading Dashboard", expanded=True):
    # 1. Pre-calculate data needed for tab labels
    tab_labels = []
    for config in ASSET_CONFIGS:
        ticker = config['ticker']
        asset_val = asset_inputs.get(ticker, 0.0)
        fix_c = config['fix_c']
        df_data, _ = monitor_data_all.get(ticker, (pd.DataFrame(), "0"))

        # --- Calculate Action and determine its Emoji color ---
        action_emoji = "⚪"
        try:
            if not df_data.empty and df_data.action.values[1 + nex] != "":
                raw_action = int(df_data.action.values[1 + nex])
                final_action = 1 - raw_action if Nex_day_sell == 1 else raw_action
                if final_action == 1:
                    action_emoji = "🟢" # Buy
                elif final_action == 0:
                    action_emoji = "🔴" # Sell
        except (IndexError, ValueError):
            pass

        # --- Calculate P/L ---
        pl_value = 0.0
        try:
            current_price = get_cached_price(ticker)
            if current_price > 0 and asset_val > 0:
                pv = current_price * asset_val
                pl_value = pv - fix_c
        except Exception:
            pass

        # 2. Format the label string
        label = f"{ticker} {action_emoji} | P/L: {pl_value:,.2f}"
        tab_labels.append(label)

    # 3. Create tabs with the new, informative labels
    tabs = st.tabs(tab_labels)

    # 4. Iterate through tabs and populate content
    for i, config in enumerate(ASSET_CONFIGS):
        st.write("_____")
        with tabs[i]:
            ticker = config['ticker']
            df_data, fx_js_str = monitor_data_all.get(ticker, (pd.DataFrame(), "0"))
            asset_last = last_assets_all.get(ticker, 0.0)
            asset_val = asset_inputs.get(ticker, 0.0)
            calc = calculations.get(ticker, {})
            
            st.write(f"**{ticker}** (f(x): `{fx_js_str}`)")
            trading_section(config, asset_val, asset_last, df_data, calc, nex, Nex_day_sell, THINGSPEAK_CLIENTS)
            
            st.write("_____")
            with st.expander("Show Raw Data Action"):
                st.dataframe(df_data, use_container_width=True)
                
if st.sidebar.button("RERUN"):
    clear_all_caches()
