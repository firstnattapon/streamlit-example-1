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
from typing import List, Dict, Any

st.set_page_config(page_title="Monitor", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# --- START: โค้ดจาก action_simulationTracer.py (Unchanged) ---
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
            self.action_length, self.mutation_rate, self.dna_seed = 0, 0, 0
            self.mutation_seeds, self.mutation_rate_float = [], 0.0
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
            self.action_length, self.mutation_rate, self.dna_seed = 0, 0, 0
            self.mutation_seeds, self.mutation_rate_float = [], 0.0
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


# ---------- CONFIGURATION & SETUP ----------
@st.cache_data
def load_config(file_path: str = 'monitor_config.json') -> Dict[str, Any]:
    """Load and validate configuration from a JSON file."""
    if not os.path.exists(file_path):
        st.error(f"Configuration file not found: {file_path}")
        st.stop()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        if 'assets' not in config or not config['assets']:
            st.error("No 'assets' list found or it is empty in monitor_config.json")
            st.stop()
        return config
    except (json.JSONDecodeError, Exception) as e:
        st.error(f"Error reading or parsing configuration file {file_path}: {e}")
        st.stop()

# ---------- GLOBAL CACHE & CLIENT MANAGEMENT ----------
_cache_lock = Lock()
_price_cache = {}
_cache_timestamp = {}

@st.cache_resource
def get_thingspeak_clients(configs: List[Dict]) -> Dict[int, thingspeak.Channel]:
    """Creates and caches a dictionary of ThingSpeak clients based on the config."""
    clients = {}
    unique_channels = set()
    for config in configs:
        for key in ['monitor_field', 'asset_field']:
            if key in config:
                conf = config[key]
                unique_channels.add((conf['channel_id'], conf['api_key']))

    for channel_id, api_key in unique_channels:
        try:
            clients[channel_id] = thingspeak.Channel(channel_id, api_key, fmt='json')
        except Exception as e:
            st.warning(f"Failed to create client for Channel ID {channel_id}: {e}")
    return clients

def clear_all_caches():
    """Clears all Streamlit and custom caches."""
    st.cache_data.clear()
    st.cache_resource.clear()
    sell.cache_clear()
    buy.cache_clear()
    with _cache_lock:
        _price_cache.clear()
        _cache_timestamp.clear()
    st.success("🗑️ All caches cleared!")
    st.rerun()

# ---------- CALCULATION UTILITIES ----------
@lru_cache(maxsize=128)
def sell(asset: float, fix_c: int = 1500, Diff: int = 60):
    if asset == 0: return 0, 0, 0
    unit_price = round((fix_c - Diff) / asset, 2)
    adjust_qty = round(abs(asset * unit_price - fix_c) / unit_price) if unit_price != 0 else 0
    total = round(asset * unit_price + adjust_qty * unit_price, 2)
    return unit_price, adjust_qty, total

@lru_cache(maxsize=128)
def buy(asset: float, fix_c: int = 1500, Diff: int = 60):
    if asset == 0: return 0, 0, 0
    unit_price = round((fix_c + Diff) / asset, 2)
    adjust_qty = round(abs(asset * unit_price - fix_c) / unit_price) if unit_price != 0 else 0
    total = round(asset * unit_price - adjust_qty * unit_price, 2)
    return unit_price, adjust_qty, total

def get_cached_price(ticker: str, max_age_seconds: int = 30) -> float:
    """Gets a ticker's price from a time-sensitive cache or fetches it."""
    now = datetime.datetime.now()
    with _cache_lock:
        if (ticker in _price_cache and (now - _cache_timestamp.get(ticker, now)).seconds < max_age_seconds):
            return _price_cache[ticker]
    try:
        price = yf.Ticker(ticker).fast_info['lastPrice']
        with _cache_lock:
            _price_cache[ticker] = price
            _cache_timestamp[ticker] = now
        return price
    except Exception:
        return 0.0

# ---------- DATA FETCHING LOGIC ----------
@st.cache_data(ttl=300)
def fetch_monitor_data(asset_config: Dict, _clients_ref: Dict, start_date: str) -> (pd.DataFrame, str):
    """Fetches monitor data and generates actions using SimulationTracer for a single asset."""
    ticker = asset_config['ticker']
    try:
        monitor_field_config = asset_config['monitor_field']
        client = _clients_ref[monitor_field_config['channel_id']]
        field_num = monitor_field_config['field']

        ticker_data = yf.Ticker(ticker).history(period='max')[['Close']].round(3)
        ticker_data.index = ticker_data.index.tz_convert(tz='Asia/bangkok')

        if start_date:
            ticker_data = ticker_data[ticker_data.index >= start_date]
        
        fx_js_str = "0"
        try:
            field_data = client.get_field_last(field=str(field_num))
            retrieved_val = json.loads(field_data).get(f"field{field_num}")
            if retrieved_val is not None:
                fx_js_str = str(retrieved_val)
        except (json.JSONDecodeError, KeyError, TypeError, thingspeak.ThingSpeakError):
            pass # Keep default "0"

        ticker_data['index'] = range(len(ticker_data))
        dummy_df = pd.DataFrame(index=[f'+{i}' for i in range(5)])
        df = pd.concat([ticker_data, dummy_df]).fillna("")
        df['action'] = ""

        try:
            tracer = SimulationTracer(encoded_string=fx_js_str)
            final_actions = tracer.run()
            num_to_assign = min(len(df), len(final_actions))
            if num_to_assign > 0:
                df.iloc[:num_to_assign, df.columns.get_loc('action')] = final_actions[:num_to_assign]
        except Exception as e:
            st.warning(f"Tracer Error for {ticker} (input: '{fx_js_str}'): {e}")

        return df.tail(7), fx_js_str
    
    except Exception as e:
        st.error(f"Error in fetch_monitor_data for {ticker}: {e}")
        return pd.DataFrame(), "0"

def fetch_all_data_concurrently(fetch_function, configs, clients_ref, *args):
    """Generic concurrent executor for fetching data."""
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(configs)) as executor:
        future_to_ticker = {
            executor.submit(fetch_function, asset, clients_ref, *args): asset['ticker']
            for asset in configs
        }
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                results[ticker] = future.result()
            except Exception as e:
                st.error(f"Error in concurrent task for {ticker}: {e}")
                results[ticker] = (pd.DataFrame(), "0") if fetch_function == fetch_monitor_data else 0.0
    return results

def _fetch_asset_helper(asset_config, clients_ref):
    """Helper to fetch a single asset value for concurrency."""
    asset_field_conf = asset_config['asset_field']
    client = clients_ref.get(asset_field_conf['channel_id'])
    if not client: return 0.0
    try:
        field_name = asset_field_conf['field']
        data = client.get_field_last(field=field_name)
        return float(json.loads(data)[field_name])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError, thingspeak.ThingSpeakError):
        return 0.0

# ---------- UI COMPONENTS ----------
def render_asset_inputs(configs, last_assets):
    """Renders the number input widgets for asset holdings."""
    cols = st.columns(len(configs))
    asset_inputs = {}
    for i, config in enumerate(configs):
        with cols[i]:
            ticker = config['ticker']
            last_asset_val = last_assets.get(ticker, 0.0)
            if config.get('option_config'):
                option_conf = config['option_config']
                real_val = st.number_input(option_conf['label'], step=0.001, value=last_asset_val, key=f"input_{ticker}_real", format="%.2f")
                asset_inputs[ticker] = option_conf['base_value'] + real_val
            else:
                asset_inputs[ticker] = st.number_input(f'{ticker}_ASSET', step=0.001, value=last_asset_val, key=f"input_{ticker}_asset", format="%.2f")
    return asset_inputs

def render_asset_update_controls(configs, clients):
    """Renders the UI for updating asset values on ThingSpeak."""
    with st.expander("Update Assets on ThingSpeak"):
        for config in configs:
            ticker, asset_conf = config['ticker'], config['asset_field']
            field_name = asset_conf['field']
            if st.checkbox(f'@_{ticker}_ASSET', key=f'check_{ticker}'):
                new_val = st.number_input(f"New Value for {ticker}", step=0.001, value=0.0, key=f'input_{ticker}')
                if st.button(f"GO_{ticker}", key=f'btn_{ticker}'):
                    try:
                        client = clients[asset_conf['channel_id']]
                        client.update({field_name: new_val})
                        st.success(f"Updated {ticker} to: {new_val} on Channel {asset_conf['channel_id']}")
                        clear_all_caches()
                    except Exception as e:
                        st.error(f"Failed to update {ticker}: {e}")

def trading_section(asset_data: Dict, nex: int, nex_day_sell: int, clients: Dict):
    """Renders the main trading interface for a single asset."""
    config = asset_data['config']
    ticker = config['ticker']
    asset_conf = config['asset_field']
    df_data = asset_data['df_data']
    
    # Determine the action for the limit order checkbox
    try:
        if df_data.empty or df_data.action.values[1 + nex] == "":
            action_val = 0
        else:
            raw_action = int(df_data.action.values[1 + nex])
            action_val = 1 - raw_action if nex_day_sell == 1 else raw_action
    except (IndexError, ValueError):
        action_val = 0

    if not st.checkbox(f'Limit_Order_{ticker}', value=bool(action_val), key=f'limit_order_{ticker}'):
        return

    # Unpack pre-calculated values
    # NOTE: The original logic uses 'buy_calc' for the sell section and vice-versa.
    # This behavior is preserved for logical consistency with the original script.
    sell_calc, buy_calc = asset_data['calculations']['buy'], asset_data['calculations']['sell']
    asset_last, asset_val = asset_data['asset_last'], asset_data['asset_val']
    
    # --- Sell Section ---
    st.write('sell', '    ', 'A', sell_calc[1], 'P', sell_calc[0], 'C', sell_calc[2])
    _, _, col3 = st.columns(3)
    if col3.checkbox(f'sell_match_{ticker}', key=f"sell_match_check_{ticker}"):
        if col3.button(f"GO_SELL_{ticker}", key=f"go_sell_btn_{ticker}"):
            try:
                client = clients[asset_conf['channel_id']]
                new_asset_val = asset_last - sell_calc[1]
                client.update({asset_conf['field']: new_asset_val})
                col3.success(f"Updated: {new_asset_val}")
                clear_all_caches()
            except Exception as e:
                st.error(f"Failed to SELL {ticker}: {e}")

    # --- P/L Display ---
    try:
        current_price = get_cached_price(ticker)
        if current_price > 0:
            pv = current_price * asset_val
            fix_value = config['fix_c']
            pl_value = pv - fix_value
            pl_color = "#a8d5a2" if pl_value >= 0 else "#fbb"
            st.markdown(f"Price: **{current_price:,.3f}** | Value: **{pv:,.2f}** | P/L (vs {fix_value:,}) : <span style='color:{pl_color}; font-weight:bold;'>{pl_value:,.2f}</span>", unsafe_allow_html=True)
        else:
            st.info(f"Price data for {ticker} is currently unavailable.")
    except Exception:
        st.warning(f"Could not retrieve price data for {ticker}.")

    # --- Buy Section ---
    st.write('buy', '   ', 'A', buy_calc[1], 'P', buy_calc[0], 'C', buy_calc[2])
    _, _, col6 = st.columns(3)
    if col6.checkbox(f'buy_match_{ticker}', key=f"buy_match_check_{ticker}"):
        if col6.button(f"GO_BUY_{ticker}", key=f"go_buy_btn_{ticker}"):
            try:
                client = clients[asset_conf['channel_id']]
                new_asset_val = asset_last + buy_calc[1]
                client.update({asset_conf['field']: new_asset_val})
                col6.success(f"Updated: {new_asset_val}")
                clear_all_caches()
            except Exception as e:
                st.error(f"Failed to BUY {ticker}: {e}")

# ---------- MAIN APPLICATION LOGIC ----------
def main():
    """Main function to run the Streamlit application."""
    config_data = load_config()
    asset_configs = config_data['assets']
    global_start_date = config_data.get('global_settings', {}).get('start_date')
    
    # Initialize Thingspeak Clients
    thingspeak_clients = get_thingspeak_clients(asset_configs)
    
    # --- 1. RENDER CONTROLS AND GET USER INPUTS ---
    nex, nex_day_sell = 0, 0
    with st.expander("⚙️ Controls & Asset Setup", expanded=False):
        if st.checkbox('nex_day'):
            nex_col, sell_col, _ = st.columns([1, 1, 6])
            if nex_col.button("Nex_day"): nex = 1
            if sell_col.button("Nex_day_sell"): nex, nex_day_sell = 1, 1
            st.write(f"nex value = {nex}" + (f" | Nex_day_sell = {nex_day_sell}" if nex_day_sell else ""))
        
        st.write("---")
        
        control_cols = st.columns(8)
        start_checked = control_cols[0].checkbox('start')
        diff_val = control_cols[7].number_input('Diff', step=1, value=60)
        
        if start_checked:
            render_asset_update_controls(asset_configs, thingspeak_clients)

        with st.expander("Asset Holdings", expanded=True):
            # Fetch last known assets to pre-fill inputs
            last_assets_all = fetch_all_data_concurrently(_fetch_asset_helper, asset_configs, thingspeak_clients)
            asset_inputs = render_asset_inputs(asset_configs, last_assets_all)

    st.write("_____")

    # --- 2. FETCH & PROCESS ALL DATA ---
    monitor_data_all = fetch_all_data_concurrently(fetch_monitor_data, asset_configs, thingspeak_clients, global_start_date)
    
    processed_assets = []
    for config in asset_configs:
        ticker = config['ticker']
        df_data, fx_js_str = monitor_data_all.get(ticker, (pd.DataFrame(), "0"))
        asset_val = asset_inputs.get(ticker, 0.0)
        
        # Determine action emoji for tab label
        action_emoji = "⚪"
        try:
            if not df_data.empty and df_data.action.values[1 + nex] != "":
                raw_action = int(df_data.action.values[1 + nex])
                final_action = 1 - raw_action if nex_day_sell == 1 else raw_action
                if final_action == 1: action_emoji = "🟢"  # Buy
                elif final_action == 0: action_emoji = "🔴"  # Sell
        except (IndexError, ValueError): pass

        # Calculate P/L for tab label
        pl_value = 0.0
        current_price = get_cached_price(ticker)
        if current_price > 0 and asset_val > 0:
            pl_value = (current_price * asset_val) - config['fix_c']
        
        processed_assets.append({
            "config": config,
            "ticker": ticker,
            "df_data": df_data,
            "fx_js_str": fx_js_str,
            "asset_last": last_assets_all.get(ticker, 0.0),
            "asset_val": asset_val,
            "calculations": {
                'buy': buy(asset_val, config['fix_c'], diff_val),
                'sell': sell(asset_val, config['fix_c'], diff_val)
            },
            "action_emoji": action_emoji,
            "pl_value": pl_value
        })

    # --- 3. RENDER MAIN DASHBOARD ---
    with st.expander("📈 Trading Dashboard", expanded=True):
        tab_labels = [f"{asset['ticker']} {asset['action_emoji']} |" for asset in processed_assets]
        tabs = st.tabs(tab_labels)
        
        for i, asset_data in enumerate(processed_assets):
            with tabs[i]:
                st.write(f"**{asset_data['ticker']}** (f(x): `{asset_data['fx_js_str']}`)")
                trading_section(asset_data, nex, nex_day_sell, thingspeak_clients)
                
                st.write("_____")
                with st.expander("Show Raw Data Action"):
                    st.dataframe(asset_data['df_data'], use_container_width=True)

    if st.sidebar.button("RERUN"):
        clear_all_caches()

if __name__ == "__main__":
    main()
