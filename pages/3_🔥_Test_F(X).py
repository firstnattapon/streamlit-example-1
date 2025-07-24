import streamlit as st
import numpy as np
import datetime
import thingspeak
import pandas as pd
import yfinance as yf
import json
from functools import lru_cache
import concurrent.futures
import os
from typing import List, Dict
import tenacity  # เพิ่ม library นี้สำหรับ retry (pip install tenacity)

st.set_page_config(page_title="Monitor", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# --- SimulationTracer Class ( unchanged, but added caching ) ---
class SimulationTracer:
    def __init__(self, encoded_string: str):
        self.encoded_string: str = str(encoded_string)
        self._decode_and_set_attributes()

    def _decode_and_set_attributes(self):
        # (unchanged code here)
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

    @lru_cache(maxsize=128)  # เพิ่ม caching สำหรับ run เพื่อความเร็ว
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

# --- Configuration Loading (unchanged, but added error handling) ---
@st.cache_data
def load_config(file_path='monitor_config.json') -> Dict:
    """Load configuration from a JSON file with error handling."""
    if not os.path.exists(file_path):
        st.error(f"Configuration file not found: {file_path}")
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON in config file: {e}")
        return {}

CONFIG_DATA = load_config()
if not CONFIG_DATA:
    st.stop()

ASSET_CONFIGS = CONFIG_DATA.get('assets', [])
GLOBAL_START_DATE = CONFIG_DATA.get('global_settings', {}).get('start_date')

if not ASSET_CONFIGS:
    st.error("No 'assets' list found in monitor_config.json")
    st.stop()

# --- ThingSpeak Clients (unchanged) ---
@st.cache_resource
def get_thingspeak_clients(configs: List[Dict]) -> Dict[int, thingspeak.Channel]:
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

# --- Clear Caches Function (modified: no auto rerun, only clear) ---
def clear_all_caches():
    st.cache_data.clear()
    st.cache_resource.clear()
    sell.cache_clear()
    buy.cache_clear()
    ui_state_keys_to_preserve = ['select_key', 'nex', 'Nex_day_sell']
    keys_to_delete = [k for k in st.session_state.keys() if k not in ui_state_keys_to_preserve]
    for key in keys_to_delete:
        del st.session_state[key]
    st.success("🗑️ Data caches cleared! UI state preserved.")

# --- Calculation Utils (unchanged) ---
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

# --- Price Fetching with Retry (modified: added retry and caching) ---
@st.cache_data(ttl=300)  # เพิ่ม ttl ยาวขึ้น
@tenacity.retry(wait=tenacity.wait_fixed(2), stop=tenacity.stop_after_attempt(3))
def get_cached_price(ticker: str) -> float:
    try:
        return yf.Ticker(ticker).fast_info['lastPrice']
    except Exception:
        return 0.0

# --- Data Fetching (optimized: combined into one cached function) ---
@st.cache_data(ttl=300)
def fetch_all_data(configs: List[Dict], _clients_ref: Dict, start_date: str) -> Dict[str, tuple]:
    """Fetch all monitor and asset data in one go with concurrency."""
    monitor_results = {}
    asset_results = {}
    
    def fetch_monitor(asset_config):
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
            except Exception:
                pass

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
            
            except Exception as e:
                st.warning(f"Tracer Error for {ticker}: {e}")

            return ticker, (df.tail(7), fx_js_str)
        except Exception as e:
            st.error(f"Error in Monitor for {ticker}: {str(e)}")
            return ticker, (pd.DataFrame(), "0")

    def fetch_asset(asset_config):
        ticker = asset_config['ticker']
        try:
            asset_conf = asset_config['asset_field']
            client = _clients_ref[asset_conf['channel_id']]
            field_name = asset_conf['field']
            data = client.get_field_last(field=field_name)
            return ticker, float(json.loads(data)[field_name])
        except Exception:
            return ticker, 0.0

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(configs)) as executor:
        # Fetch monitors
        monitor_futures = [executor.submit(fetch_monitor, asset) for asset in configs]
        for future in concurrent.futures.as_completed(monitor_futures):
            ticker, result = future.result()
            monitor_results[ticker] = result
        
        # Fetch assets
        asset_futures = [executor.submit(fetch_asset, asset) for asset in configs]
        for future in concurrent.futures.as_completed(asset_futures):
            ticker, result = future.result()
            asset_results[ticker] = result

    return {'monitors': monitor_results, 'assets': asset_results}

# --- UI Components (simplified) ---
def render_asset_inputs(configs: List[Dict], last_assets: Dict) -> Dict[str, float]:
    asset_inputs = {}
    cols = st.columns(len(configs))
    for i, config in enumerate(configs):
        with cols[i]:
            ticker = config['ticker']
            last_val = last_assets.get(ticker, 0.0)
            if config.get('option_config'):
                option_val = config['option_config']['base_value']
                label = config['option_config']['label']
                real_val = st.number_input(label, step=0.001, value=last_val, key=f"input_{ticker}_real")
                asset_inputs[ticker] = option_val + real_val
            else:
                label = f'{ticker}_ASSET'
                val = st.number_input(label, step=0.001, value=last_val, key=f"input_{ticker}_asset")
                asset_inputs[ticker] = val
    return asset_inputs

def render_asset_update_controls(configs: List[Dict], clients: Dict):
    with st.expander("Update Assets on ThingSpeak"):
        for config in configs:
            ticker = config['ticker']
            asset_conf = config['asset_field']
            field_name = asset_conf['field']
            with st.form(key=f'form_{ticker}'):  # ใช้ form เพื่อ compact
                add_val = st.number_input(f"New Value for {ticker}", step=0.001, value=0.0)
                submit = st.form_submit_button(f"Update {ticker}")
                if submit:
                    try:
                        client = clients[asset_conf['channel_id']]
                        client.update({field_name: add_val})
                        st.success(f"Updated {ticker} to: {add_val}")
                        clear_all_caches()
                    except Exception as e:
                        st.error(f"Failed to update {ticker}: {e}")

def trading_section(config: Dict, asset_val: float, asset_last: float, df_data: pd.DataFrame, calc: Dict, nex: int, Nex_day_sell: int, clients: Dict):
    ticker = config['ticker']
    asset_conf = config['asset_field']
    field_name = asset_conf['field']

    def get_action_val():
        try:
            if df_data.empty or df_data.action.values[1 + nex] == "":
                return 0
            val = df_data.action.values[1 + nex]
            return 1 - val if Nex_day_sell == 1 else val
        except Exception:
            return 0

    action_val = get_action_val()
    limit_order = st.checkbox(f'Limit_Order_{ticker}', value=bool(action_val), key=f'limit_order_{ticker}')
    if not limit_order:
        return

    sell_calc = calc['sell']
    buy_calc = calc['buy']

    # Compact display
    st.markdown(f"**Sell:** A {buy_calc[1]} | P {buy_calc[0]} | C {buy_calc[2]}")
    st.markdown(f"**Buy:** A {sell_calc[1]} | P {sell_calc[0]} | C {sell_calc[2]}")

    with st.form(key=f'trade_form_{ticker}'):  # ใช้ form สำหรับ buy/sell เพื่อลด buttons
        trade_type = st.radio("Trade Action", ["Sell", "Buy"])
        submit = st.form_submit_button("Execute Trade")
        if submit:
            try:
                client = clients[asset_conf['channel_id']]
                if trade_type == "Sell":
                    new_val = asset_last - buy_calc[1]
                else:
                    new_val = asset_last + sell_calc[1]
                client.update({field_name: new_val})
                st.success(f"{trade_type} executed: New value {new_val}")
                clear_all_caches()
            except Exception as e:
                st.error(f"Failed to {trade_type} {ticker}: {e}")

    try:
        current_price = get_cached_price(ticker)
        if current_price > 0:
            pv = current_price * asset_val
            fix_value = config['fix_c']
            pl_value = pv - fix_value
            pl_color = "#a8d5a2" if pl_value >= 0 else "#fbb"
            st.markdown(
                f"Price: **{current_price:,.3f}** | Value: **{pv:,.2f}** | P/L (vs {fix_value:,}) : <span style='color:{pl_color};'>{pl_value:,.2f}</span>",
                unsafe_allow_html=True
            )
        else:
            st.info(f"Price data for {ticker} unavailable.")
    except Exception:
        st.warning(f"Could not retrieve price for {ticker}.")

# --- Main Logic ---
all_data = fetch_all_data(ASSET_CONFIGS, THINGSPEAK_CLIENTS, GLOBAL_START_DATE)
monitor_data_all = all_data['monitors']
last_assets_all = all_data['assets']

# Stable Session State (unchanged)
if 'select_key' not in st.session_state:
    st.session_state.select_key = ""
if 'nex' not in st.session_state:
    st.session_state.nex = 0
if 'Nex_day_sell' not in st.session_state:
    st.session_state.Nex_day_sell = 0

tab1, tab2 = st.tabs(["📈 Monitor", "⚙️ Controls"])

with tab2:
    Nex_day_ = st.checkbox('nex_day', value=(st.session_state.nex == 1))
    if Nex_day_:
        col1, col2 = st.columns(2)
        if col1.button("Nex_day"):
            st.session_state.nex = 1
            st.session_state.Nex_day_sell = 0
        if col2.button("Nex_day_sell"):
            st.session_state.nex = 1
            st.session_state.Nex_day_sell = 1
    else:
        st.session_state.nex = 0
        st.session_state.Nex_day_sell = 0

    nex = st.session_state.nex
    Nex_day_sell = st.session_state.Nex_day_sell
    if Nex_day_:
        st.write(f"nex = {nex} | Nex_day_sell = {Nex_day_sell}" if Nex_day_sell else f"nex = {nex}")

    st.write("---")
    x_2 = st.number_input('Diff', step=1, value=60)
    st.write("---")
    asset_inputs = render_asset_inputs(ASSET_CONFIGS, last_assets_all)
    st.write("---")
    if st.checkbox('Start Updates'):
        render_asset_update_controls(ASSET_CONFIGS, THINGSPEAK_CLIENTS)

with tab1:
    # Selectbox Logic (simplified)
    selectbox_labels = {}
    ticker_actions = {}
    for config in ASSET_CONFIGS:
        ticker = config['ticker']
        df_data, fx_js_str = monitor_data_all.get(ticker, (pd.DataFrame(), "0"))
        action_emoji = ""
        try:
            if not df_data.empty and df_data.action.values[1 + nex] != "":
                raw_action = int(df_data.action.values[1 + nex])
                final_action = 1 - raw_action if Nex_day_sell == 1 else raw_action
                action_emoji = "🟢 " if final_action == 1 else "🔴 "
                ticker_actions[ticker] = final_action
        except Exception:
            pass
        selectbox_labels[ticker] = f"{action_emoji}{ticker} (f(x): {fx_js_str})"

    all_tickers = [config['ticker'] for config in ASSET_CONFIGS]
    selectbox_options = [""] + (["Filter Buy Tickers", "Filter Sell Tickers"] if nex == 1 else []) + all_tickers

    if st.session_state.select_key not in selectbox_options:
        st.session_state.select_key = ""

    def format_option(option):
        if option == "": return "Show All"
        if option in ["Filter Buy Tickers", "Filter Sell Tickers"]: return option
        return selectbox_labels.get(option, option).split(' (f(x):')[0]

    st.selectbox("Select Ticker:", options=selectbox_options, format_func=format_option, key="select_key")
    st.write("_____")

    selected = st.session_state.select_key
    if selected == "":
        configs_to_display = ASSET_CONFIGS
    elif selected == "Filter Buy Tickers":
        configs_to_display = [c for c in ASSET_CONFIGS if ticker_actions.get(c['ticker'], None) == 1]
    elif selected == "Filter Sell Tickers":
        configs_to_display = [c for c in ASSET_CONFIGS if ticker_actions.get(c['ticker'], None) == 0]
    else:
        configs_to_display = [c for c in ASSET_CONFIGS if c['ticker'] == selected]

    # Pre-calculate all
    calculations = {}
    for config in ASSET_CONFIGS:
        ticker = config['ticker']
        asset_value = asset_inputs.get(ticker, 0.0)
        fix_c = config['fix_c']
        calculations[ticker] = {
            'sell': sell(asset_value, fix_c=fix_c, Diff=x_2),
            'buy': buy(asset_value, fix_c=fix_c, Diff=x_2)
        }

    # เพิ่ม Summary Table สำหรับ usability
    summary_data = []
    for config in configs_to_display:
        ticker = config['ticker']
        action = ticker_actions.get(ticker, None)
        summary_data.append({"Ticker": ticker, "Action": "Buy" if action == 1 else "Sell" if action == 0 else "N/A"})
    st.dataframe(summary_data, use_container_width=True)

    for config in configs_to_display:
        ticker = config['ticker']
        df_data, fx_js_str = monitor_data_all.get(ticker, (pd.DataFrame(), "0"))
        asset_last = last_assets_all.get(ticker, 0.0)
        asset_val = asset_inputs.get(ticker, 0.0)
        calc = calculations.get(ticker, {})
        
        st.write(selectbox_labels.get(ticker, ticker))
        trading_section(config, asset_val, asset_last, df_data, calc, nex, Nex_day_sell, THINGSPEAK_CLIENTS)
        
        with st.expander("Show Raw Data"):
            st.dataframe(df_data, use_container_width=True)
            
        st.write("_____")

# Sidebar Button (unchanged)
if st.sidebar.button("RERUN"):
    clear_all_caches()
    st.rerun()
