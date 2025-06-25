# 📈_Monitor.py (Async Performance - Final Corrected Version)
import streamlit as st
import numpy as np
import datetime
import pandas as pd
import yfinance as yf
import json
from functools import lru_cache
from threading import Lock
import os
from typing import List, Dict, Any
import asyncio
import aiohttp
import thingspeak

st.set_page_config(page_title="Monitor", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# --- START: SimulationTracer Class (Unchanged) ---
class SimulationTracer:
    def __init__(self, encoded_string: str):
        self.encoded_string: str = str(encoded_string) if not isinstance(encoded_string, str) else encoded_string
        self._decode_and_set_attributes()
    def _decode_and_set_attributes(self):
        if not self.encoded_string.isdigit(): self.action_length, self.mutation_rate, self.dna_seed, self.mutation_seeds, self.mutation_rate_float = 0, 0, 0, [], 0.0; return
        decoded_numbers, idx = [], 0
        try:
            while idx < len(self.encoded_string):
                length_of_number = int(self.encoded_string[idx]); idx += 1; number_str = self.encoded_string[idx : idx + length_of_number]; idx += length_of_number; decoded_numbers.append(int(number_str))
        except (IndexError, ValueError): pass
        if len(decoded_numbers) < 3: self.action_length, self.mutation_rate, self.dna_seed, self.mutation_seeds, self.mutation_rate_float = 0, 0, 0, [], 0.0; return
        self.action_length, self.mutation_rate, self.dna_seed, self.mutation_seeds, self.mutation_rate_float = decoded_numbers[0], decoded_numbers[1], decoded_numbers[2], decoded_numbers[3:], self.mutation_rate / 100.0
    def run(self) -> np.ndarray:
        if self.action_length <= 0: return np.array([])
        dna_rng = np.random.default_rng(seed=self.dna_seed); current_actions = dna_rng.integers(0, 2, size=self.action_length)
        if self.action_length > 0: current_actions[0] = 1
        for m_seed in self.mutation_seeds:
            mutation_rng = np.random.default_rng(seed=m_seed); mutation_mask = mutation_rng.random(self.action_length) < self.mutation_rate_float
            current_actions[mutation_mask] = 1 - current_actions[mutation_mask]
            if self.action_length > 0: current_actions[0] = 1
        return current_actions
# --- END: SimulationTracer Class ---

# ---------- CONFIGURATION & SETUP ----------
@st.cache_data
def load_config(file_path: str = 'monitor_config.json') -> Dict[str, Any]:
    if not os.path.exists(file_path): st.error(f"Config not found: {file_path}"); st.stop()
    try:
        with open(file_path, 'r', encoding='utf-8') as f: config = json.load(f)
        if 'assets' not in config or not config['assets']: st.error("No 'assets' in config"); st.stop()
        return config
    except Exception as e: st.error(f"Error reading config: {e}"); st.stop()

# ---------- GLOBAL CACHE & CLIENT MANAGEMENT ----------
_cache_lock = Lock()
_price_cache, _cache_timestamp = {}, {}

@st.cache_resource
def get_thingspeak_clients(configs: List[Dict]):
    clients = {}
    for conf in configs:
        if 'asset_field' in conf:
            asset_conf = conf['asset_field']; channel_id, api_key = asset_conf['channel_id'], asset_conf['api_key']
            if channel_id not in clients:
                try: clients[channel_id] = thingspeak.Channel(channel_id, api_key, fmt='json')
                except Exception as e: st.warning(f"Failed to create sync client for {channel_id}: {e}")
    return clients

def clear_all_caches():
    st.cache_data.clear(); st.cache_resource.clear(); sell.cache_clear(); buy.cache_clear()
    with _cache_lock: _price_cache.clear(); _cache_timestamp.clear()
    st.success("🗑️ All caches cleared!"); st.rerun()

# ---------- CALCULATION UTILITIES (Unchanged) ----------
@lru_cache(maxsize=128)
def sell(asset: float, fix_c: int = 1500, Diff: int = 60):
    if asset == 0: return 0, 0, 0
    unit_price = round((fix_c - Diff) / asset, 2)
    adjust_qty = round(abs(asset * unit_price - fix_c) / unit_price) if unit_price != 0 else 0
    return unit_price, adjust_qty, round(asset * unit_price + adjust_qty * unit_price, 2)

@lru_cache(maxsize=128)
def buy(asset: float, fix_c: int = 1500, Diff: int = 60):
    if asset == 0: return 0, 0, 0
    unit_price = round((fix_c + Diff) / asset, 2)
    adjust_qty = round(abs(asset * unit_price - fix_c) / unit_price) if unit_price != 0 else 0
    return unit_price, adjust_qty, round(asset * unit_price - adjust_qty * unit_price, 2)

def get_cached_price(ticker: str, max_age_seconds: int = 30) -> float:
    now = datetime.datetime.now()
    with _cache_lock:
        if (ticker in _price_cache and (now - _cache_timestamp.get(ticker, now)).seconds < max_age_seconds):
            return _price_cache[ticker]
    try:
        price = yf.Ticker(ticker).fast_info['lastPrice']
        with _cache_lock: _price_cache[ticker], _cache_timestamp[ticker] = price, now
        return price
    except Exception: return 0.0

# ---------- ASYNCHRONOUS DATA FETCHING LOGIC (Corrected for Hashing) ----------

# --- STEP 1: INNER, UNCACHED functions that accept the unhashable `session` ---
async def _internal_fetch_thingspeak_field(session: aiohttp.ClientSession, channel_id, api_key, field) -> str:
    url = f"https://api.thingspeak.com/channels/{channel_id}/fields/{field}/last.json?api_key={api_key}"
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                data = await response.json(); key = f"field{field}" if isinstance(field, int) else field
                return str(data.get(key)) if data.get(key) is not None else "0"
            return "0"
    except Exception: return "0"

async def _internal_fetch_yfinance_history(ticker: str, start_date: str) -> pd.DataFrame:
    loop = asyncio.get_event_loop()
    try:
        ticker_data = await loop.run_in_executor(None, lambda: yf.Ticker(ticker).history(period='max')[['Close']].round(3))
        ticker_data.index = ticker_data.index.tz_convert(tz='Asia/bangkok')
        if start_date: ticker_data = ticker_data[ticker_data.index >= start_date]
        return ticker_data
    except Exception: return pd.DataFrame(columns=['Close'])

# --- STEP 2: OUTER, CACHED functions that accept only hashable parameters ---
@st.cache_data(ttl=300)
async def get_cached_yfinance_history(ticker: str, start_date: str) -> pd.DataFrame:
    return await _internal_fetch_yfinance_history(ticker, start_date)

@st.cache_data(ttl=300)
async def get_cached_thingspeak_field(channel_id, api_key, field) -> str:
    # This function is now the one being cached. It takes hashable args.
    # It will be called inside a running event loop that has a session.
    # We will pass the session to the internal function from the main gathering point.
    # This is a bit of a workaround to fit Streamlit's cache model.
    # The actual fetching logic will be called from a place that has `session`.
    # Let's adjust the structure.

    # A better structure: Cache the data, not the function call.
    # We will call the internal functions directly and cache their results manually if needed.
    # Let's simplify. We will cache the final processed data instead.

    # Let's try another, simpler structure.
    # The main gathering function will NOT be cached.
    # The individual data fetching functions WILL be cached.
    pass # This structure is getting complex. Let's simplify below.

# --- REVISED AND SIMPLIFIED ASYNC STRUCTURE ---
@st.cache_data(ttl=300)
async def fetch_and_process_monitor_data(ticker: str, mon_channel_id, mon_api_key, mon_field, start_date: str) -> (pd.DataFrame, str):
    """This function is fully hashable and cachable. It creates its own session."""
    async with aiohttp.ClientSession() as session:
        # Fetch data concurrently
        results = await asyncio.gather(
            _internal_fetch_yfinance_history(ticker, start_date),
            _internal_fetch_thingspeak_field(session, mon_channel_id, mon_api_key, mon_field)
        )
    ticker_data, fx_js_str = results[0], results[1]
    
    # Process data (this part is fast, no need to cache separately)
    ticker_data['index'] = range(len(ticker_data))
    df = pd.concat([ticker_data, pd.DataFrame(index=[f'+{i}' for i in range(5)])]).fillna("")
    df['action'] = ""
    try:
        tracer = SimulationTracer(encoded_string=fx_js_str)
        final_actions = tracer.run()
        num_to_assign = min(len(df), len(final_actions))
        if num_to_assign > 0: df.iloc[:num_to_assign, df.columns.get_loc('action')] = final_actions[:num_to_assign]
    except Exception as e: st.warning(f"Tracer Error for {ticker}: {e}")
    return df.tail(7), fx_js_str

@st.cache_data(ttl=60)
async def fetch_asset_data(channel_id, api_key, field) -> float:
    """This function is also fully hashable and cachable."""
    async with aiohttp.ClientSession() as session:
        result_str = await _internal_fetch_thingspeak_field(session, channel_id, api_key, field)
    try:
        return float(result_str)
    except (ValueError, TypeError):
        return 0.0

async def gather_all_data(configs: List[Dict], start_date: str):
    """Gathers all cachable async tasks."""
    monitor_tasks, asset_tasks = [], []
    for config in configs:
        mon_conf = config['monitor_field']
        monitor_tasks.append(fetch_and_process_monitor_data(
            config['ticker'], mon_conf['channel_id'], mon_conf['api_key'], mon_conf['field'], start_date
        ))
        asset_conf = config['asset_field']
        asset_tasks.append(fetch_asset_data(
            asset_conf['channel_id'], asset_conf['api_key'], asset_conf['field']
        ))

    all_results = await asyncio.gather(*monitor_tasks, *asset_tasks, return_exceptions=True)
    
    num_configs = len(configs)
    monitor_data_all = {configs[i]['ticker']: res if not isinstance(res, Exception) else (pd.DataFrame(), "0") for i, res in enumerate(all_results[:num_configs])}
    last_assets_all = {configs[i]['ticker']: res if not isinstance(res, Exception) else 0.0 for i, res in enumerate(all_results[num_configs:])}
    return monitor_data_all, last_assets_all

# ---------- UI COMPONENTS (Unchanged) ----------
def render_asset_inputs(configs, last_assets):
    cols, asset_inputs = st.columns(len(configs)), {}
    for i, config in enumerate(configs):
        with cols[i]:
            ticker, last_asset_val = config['ticker'], last_assets.get(ticker, 0.0)
            if config.get('option_config'):
                option_conf = config['option_config']
                real_val = st.number_input(option_conf['label'], step=0.001, value=last_asset_val, key=f"input_{ticker}_real", format="%.3f")
                asset_inputs[ticker] = option_conf['base_value'] + real_val
            else:
                asset_inputs[ticker] = st.number_input(f'{ticker}_ASSET', step=0.001, value=last_asset_val, key=f"input_{ticker}_asset", format="%.3f")
    return asset_inputs

def render_asset_update_controls(configs, clients):
    with st.expander("Update Assets on ThingSpeak"):
        for config in configs:
            ticker, asset_conf = config['ticker'], config['asset_field']
            if st.checkbox(f'@_{ticker}_ASSET', key=f'check_{ticker}'):
                new_val = st.number_input(f"New Value for {ticker}", step=0.001, value=0.0, key=f'input_{ticker}')
                if st.button(f"GO_{ticker}", key=f'btn_{ticker}'):
                    try: clients[asset_conf['channel_id']].update({asset_conf['field']: new_val}); st.success(f"Updated {ticker}"); clear_all_caches()
                    except Exception as e: st.error(f"Failed to update {ticker}: {e}")

def trading_section(asset_data: Dict, nex: int, nex_day_sell: int, clients: Dict):
    config, ticker, df_data = asset_data['config'], asset_data['ticker'], asset_data['df_data']
    try: raw_action = int(df_data.action.values[1 + nex]); action_val = 1 - raw_action if nex_day_sell == 1 else raw_action
    except: action_val = 0
    if not st.checkbox(f'Limit_Order_{ticker}', value=bool(action_val), key=f'limit_order_{ticker}'): return
    buy_calc, sell_calc = asset_data['calculations']['buy'], asset_data['calculations']['sell']
    asset_last, asset_val, asset_conf = asset_data['asset_last'], asset_data['asset_val'], config['asset_field']
    
    st.write('sell', 'A', sell_calc[1], 'P', sell_calc[0], 'C', sell_calc[2])
    _, _, col3 = st.columns(3)
    if col3.checkbox(f'sell_match_{ticker}', key=f"sell_match_check_{ticker}"):
        if col3.button(f"GO_SELL_{ticker}", key=f"go_sell_btn_{ticker}"):
            try: new_asset_val = asset_last - sell_calc[1]; clients[asset_conf['channel_id']].update({asset_conf['field']: new_asset_val}); col3.success(f"Updated: {new_asset_val:.3f}"); clear_all_caches()
            except Exception as e: st.error(f"Failed to SELL {ticker}: {e}")
    try:
        current_price = get_cached_price(ticker)
        if current_price > 0:
            pv, fix_value = current_price * asset_val, config['fix_c']; pl_value, pl_color = pv - fix_value, "#a8d5a2" if pv - fix_value >= 0 else "#fbb"
            st.markdown(f"Price: **{current_price:,.3f}** | Value: **{pv:,.2f}** | P/L (vs {fix_value:,}) : <span style='color:{pl_color}; font-weight:bold;'>{pl_value:,.2f}</span>", unsafe_allow_html=True)
    except: pass
    st.write('buy', 'A', buy_calc[1], 'P', buy_calc[0], 'C', buy_calc[2])
    _, _, col6 = st.columns(3)
    if col6.checkbox(f'buy_match_{ticker}', key=f"buy_match_check_{ticker}"):
        if col6.button(f"GO_BUY_{ticker}", key=f"go_buy_btn_{ticker}"):
            try: new_asset_val = asset_last + buy_calc[1]; clients[asset_conf['channel_id']].update({asset_conf['field']: new_asset_val}); col6.success(f"Updated: {new_asset_val:.3f}"); clear_all_caches()
            except Exception as e: st.error(f"Failed to BUY {ticker}: {e}")

# ---------- MAIN APPLICATION LOGIC ----------
def main():
    config_data = load_config()
    asset_configs = config_data['assets']
    global_start_date = config_data.get('global_settings', {}).get('start_date')
    thingspeak_clients = get_thingspeak_clients(asset_configs)

    with st.spinner("Fetching all data..."):
        monitor_data_all, last_assets_all = asyncio.run(gather_all_data(asset_configs, global_start_date))

    with st.expander("⚙️ Controls & Asset Setup", expanded=True):
        nex, nex_day_sell = 0, 0
        if st.checkbox('nex_day'): nex_col, sell_col, _ = st.columns([1, 1, 6]); nex = 1 if nex_col.button("Nex_day") else nex; nex, nex_day_sell = (1, 1) if sell_col.button("Nex_day_sell") else (nex, nex_day_sell)
        control_cols = st.columns(8); start_checked = control_cols[0].checkbox('start'); diff_val = control_cols[7].number_input('Diff', step=1, value=60)
        if start_checked: render_asset_update_controls(asset_configs, thingspeak_clients)
        asset_inputs = render_asset_inputs(asset_configs, last_assets_all)

    st.write("_____")
    
    processed_assets = []
    for config in asset_configs:
        ticker, (df_data, fx_js_str), asset_val = config['ticker'], monitor_data_all.get(ticker, (pd.DataFrame(), "0")), asset_inputs.get(ticker, 0.0)
        action_emoji = "⚪"
        try: raw_action = int(df_data.action.values[1 + nex]); final_action = 1 - raw_action if nex_day_sell == 1 else raw_action; action_emoji = "🟢" if final_action == 1 else "🔴"
        except: pass
        current_price = get_cached_price(ticker); pl_value = (current_price * asset_val) - config['fix_c'] if current_price > 0 and asset_val > 0 else 0.0
        processed_assets.append({
            "config": config, "ticker": ticker, "df_data": df_data, "fx_js_str": fx_js_str, "asset_last": last_assets_all.get(ticker, 0.0), "asset_val": asset_val,
            "calculations": {'buy': buy(asset_val, config['fix_c'], diff_val), 'sell': sell(asset_val, config['fix_c'], diff_val)}, "action_emoji": action_emoji, "pl_value": pl_value
        })

    with st.expander("📈 Trading Dashboard", expanded=True):
        tabs = st.tabs([f"{a['ticker']} {a['action_emoji']} | P/L: {a['pl_value']:,.2f}" for a in processed_assets])
        for i, asset_data in enumerate(processed_assets):
            with tabs[i]:
                st.write(f"**{asset_data['ticker']}** (f(x): `{asset_data['fx_js_str']}`)")
                trading_section(asset_data, nex, nex_day_sell, thingspeak_clients)
                st.write("_____")
                with st.expander("Show Raw Data Action"): st.dataframe(asset_data['df_data'], use_container_width=True)

    if st.sidebar.button("RERUN"): clear_all_caches()

if __name__ == "__main__":
    main()
