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

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Monitor", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# --- SIMULATION TRACER CLASS (from refactored version) ---
class SimulationTracer:
    """
    คลาสสำหรับห่อหุ้มกระบวนการทั้งหมด ตั้งแต่การถอดรหัสพารามิเตอร์
    ไปจนถึงการจำลองกระบวนการกลายพันธุ์ของ action sequence
    """
    def __init__(self, encoded_string: str):
        self.encoded_string = str(encoded_string)
        self._decode()

    def _decode(self):
        s = self.encoded_string
        if not s.isdigit():
            self.action_length = self.mutation_rate = self.dna_seed = 0
            self.mutation_seeds, self.mutation_rate_float = [], 0.0
            return
        nums, i = [], 0
        try:
            while i < len(s):
                ln = int(s[i]); i += 1
                nums.append(int(s[i:i + ln])); i += ln
        except (ValueError, IndexError):
            pass # Fail silently
        if len(nums) < 3:
            self.action_length = self.mutation_rate = self.dna_seed = 0
            self.mutation_seeds, self.mutation_rate_float = [], 0.0
            return
        self.action_length, self.mutation_rate, self.dna_seed = nums[:3]
        self.mutation_seeds = nums[3:]
        self.mutation_rate_float = self.mutation_rate / 100.0

    def run(self) -> np.ndarray:
        if self.action_length <= 0:
            return np.array([])
        rng = np.random.default_rng(self.dna_seed)
        act = rng.integers(0, 2, size=self.action_length)
        if self.action_length > 0:
            act[0] = 1 # Force first action to be buy
        for seed in self.mutation_seeds:
            m_rng = np.random.default_rng(seed)
            mask = m_rng.random(self.action_length) < self.mutation_rate_float
            act[mask] = 1 - act[mask]
            if self.action_length > 0:
                act[0] = 1 # Ensure first action is always buy
        return act

# --- CONFIGURATION & UTILITIES ---
@st.cache_data
def load_config(path='monitor_config.json') -> Dict[str, Any]:
    if not os.path.exists(path):
        st.error(f"Config file not found: {path}"); st.stop()
    try:
        with open(path, 'r', encoding='utf-8') as fp:
            cfg = json.load(fp)
        if not cfg.get('assets'):
            raise ValueError("'assets' list not found or empty in config.")
        return cfg
    except Exception as e:
        st.error(f"Error reading or parsing config file: {e}"); st.stop()

# --- CACHE MANAGEMENT ---
_price_lock = Lock()
_price_cache, _price_ts = {}, {}

def get_cached_price(ticker: str, max_age: int = 30) -> float:
    """Gets price from a local cache or yfinance, with a max_age lock."""
    now = datetime.datetime.now()
    with _price_lock:
        price = _price_cache.get(ticker)
        timestamp = _price_ts.get(ticker, datetime.datetime.min)
        if price is not None and (now - timestamp).seconds < max_age:
            return price
    try:
        new_price = yf.Ticker(ticker).fast_info['lastPrice']
        with _price_lock:
            _price_cache[ticker] = new_price
            _price_ts[ticker] = now
        return new_price
    except Exception:
        return 0.0

@lru_cache(maxsize=128)
def sell(asset: float, fix_c: int = 1500, Diff: int = 60):
    if asset == 0: return 0, 0, 0
    unit_price = round((fix_c - Diff) / asset, 2) if asset != 0 else 0
    adjust_qty = round(abs(asset * unit_price - fix_c) / unit_price) if unit_price != 0 else 0
    total = round(asset * unit_price + adjust_qty * unit_price, 2)
    return unit_price, adjust_qty, total

@lru_cache(maxsize=128)
def buy(asset: float, fix_c: int = 1500, Diff: int = 60):
    if asset == 0: return 0, 0, 0
    unit_price = round((fix_c + Diff) / asset, 2) if asset != 0 else 0
    adjust_qty = round(abs(asset * unit_price - fix_c) / unit_price) if unit_price != 0 else 0
    total = round(asset * unit_price - adjust_qty * unit_price, 2)
    return unit_price, adjust_qty, total

def clear_all_caches():
    """Flush all caches and rerun the app."""
    st.cache_data.clear()
    st.cache_resource.clear()
    fetch_all_data.cache_clear()
    sell.cache_clear()
    buy.cache_clear()
    with _price_lock:
        _price_cache.clear()
        _price_ts.clear()
    st.success("🗑️ All caches cleared!")
    st.rerun()

# --- DATA FETCHING (Integrated logic) ---
@st.cache_resource
def get_thingspeak_clients(configs):
    """Creates and caches a dictionary of ThingSpeak clients."""
    clients = {}
    unique_channels = set()
    for config in configs:
        if 'monitor_field' in config:
            mon_conf = config['monitor_field']
            unique_channels.add((mon_conf['channel_id'], mon_conf['api_key']))
        if 'asset_field' in config:
            asset_conf = config['asset_field']
            unique_channels.add((asset_conf['channel_id'], asset_conf['api_key']))
    for channel_id, api_key in unique_channels:
        try:
            clients[channel_id] = thingspeak.Channel(channel_id, api_key, fmt='json')
        except Exception as e:
            st.error(f"Failed to create client for Channel ID {channel_id}: {e}")
    return clients

def _fetch_monitor_data(asset_config, client, start_date):
    """Helper to fetch history and f(x) string for one asset."""
    ticker = asset_config['ticker']
    try:
        tickerData = yf.Ticker(ticker).history(period='max')[['Close']].round(3)
        tickerData.index = tickerData.index.tz_convert(tz='Asia/bangkok')
        if start_date:
            tickerData = tickerData[tickerData.index >= start_date]

        fx_js_str = "0"
        try:
            field_num = asset_config['monitor_field']['field']
            field_data = client.get_field_last(field=str(field_num))
            retrieved_val = json.loads(field_data)[f"field{field_num}"]
            if retrieved_val is not None:
                fx_js_str = str(retrieved_val)
        except (json.JSONDecodeError, KeyError, TypeError, thingspeak.ThingSpeakError):
            pass # Keep default fx_js_str="0"
        return ticker, {'history': tickerData, 'fx_str': fx_js_str}
    except Exception as e:
        st.warning(f"Could not fetch monitor data for {ticker}: {e}")
        return ticker, {'history': pd.DataFrame(), 'fx_str': "0"}

def _fetch_asset_value(asset_config, client):
    """Helper to fetch a single asset value."""
    ticker = asset_config['ticker']
    try:
        field_name = asset_config['asset_field']['field']
        data = client.get_field_last(field=field_name)
        return ticker, float(json.loads(data)[field_name])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError, thingspeak.ThingSpeakError):
        return ticker, 0.0

@st.cache_data(ttl=60)
def fetch_all_data(configs, _clients, start_date):
    """Fetches all monitor and asset data concurrently and merges them."""
    results = {asset['ticker']: {} for asset in configs}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(configs) * 2) as executor:
        # Submit monitor data fetch jobs
        future_to_ticker_mon = {
            executor.submit(_fetch_monitor_data, asset, _clients[asset['monitor_field']['channel_id']], start_date): asset['ticker']
            for asset in configs if asset['monitor_field']['channel_id'] in _clients
        }
        # Submit asset value fetch jobs
        future_to_ticker_asset = {
            executor.submit(_fetch_asset_value, asset, _clients[asset['asset_field']['channel_id']]): asset['ticker']
            for asset in configs if asset['asset_field']['channel_id'] in _clients
        }

        # Process monitor results
        for future in concurrent.futures.as_completed(future_to_ticker_mon):
            ticker, data = future.result()
            results[ticker].update(data)
        # Process asset results
        for future in concurrent.futures.as_completed(future_to_ticker_asset):
            ticker, value = future.result()
            results[ticker]['asset_val'] = value

    return results

# --- UI HELPER FUNCTIONS ---
def render_asset_inputs(configs, last_assets):
    cols = st.columns(len(configs))
    asset_inputs = {}
    for i, c in enumerate(configs):
        with cols[i]:
            tk = c['ticker']
            last = last_assets.get(tk, 0.0)
            if (opt := c.get('option_config')):
                real = st.number_input(opt['label'], step=0.001, value=last,
                                       key=f"inp_{tk}_real", format="%.2f")
                asset_inputs[tk] = opt['base_value'] + real
            else:
                asset_inputs[tk] = st.number_input(f'{tk} Asset', step=0.001,
                                                   value=last, key=f"inp_{tk}_asset",
                                                   format="%.2f")
    return asset_inputs

def render_asset_update_controls(cfgs, clients):
    with st.expander("Update Assets on ThingSpeak"):
        for c in cfgs:
            tk, a_conf = c['ticker'], c['asset_field']
            field = a_conf['field']
            if st.checkbox(f'@ Update {tk} Asset', key=f'chk_update_{tk}'):
                val = st.number_input(f"New value for {tk}", step=0.001, value=0.0,
                                      key=f'new_val_{tk}')
                if st.button(f"GO Update {tk}", key=f"btn_go_{tk}"):
                    try:
                        client = clients.get(a_conf['channel_id'])
                        if client:
                            client.update({field: val})
                            st.success(f"Updated {tk} to {val}")
                            clear_all_caches()
                        else:
                            st.error(f"Client for channel {a_conf['channel_id']} not found.")
                    except Exception as e:
                        st.error(f"Update failed for {tk}: {e}")

def trading_section(data: Dict, nex: int, nex_day_sell: int, clients):
    cfg, tk = data['config'], data['ticker']
    df = data['df_data']
    a_last, a_val = data['asset_last'], data['asset_val']
    buy_calc, sell_calc = data['calcs']['buy'], data['calcs']['sell']

    try:
        action_index = 1 + nex
        if df.empty or action_index >= len(df) or df.action.values[action_index] == "":
            act_val = 0
        else:
            raw = int(df.action.values[action_index])
            act_val = 1 - raw if nex_day_sell else raw
    except (ValueError, IndexError):
        act_val = 0

    if not st.checkbox(f'Enable Limit Order for {tk}', value=bool(act_val), key=f'limit_order_{tk}'):
        return

    st.write(f"**Sell:** Amount `{buy_calc[1]}` | Price `{buy_calc[0]}` | Cost `{buy_calc[2]}`") # Logic from original
    if st.checkbox(f'Match SELL order for {tk}', key=f'sell_match_{tk}'):
        if st.button(f"EXECUTE SELL {tk}", key=f'go_sell_{tk}'):
            try:
                new_val = a_last - buy_calc[1]
                a_conf = cfg['asset_field']
                client = clients.get(a_conf['channel_id'])
                if client:
                    client.update({a_conf['field']: new_val})
                    st.success(f"SELL successful. New asset value: {new_val}")
                    clear_all_caches()
                else:
                    st.error(f"Client for channel {a_conf['channel_id']} not found.")
            except Exception as e:
                st.error(f"SELL execution failed for {tk}: {e}")

    p = get_cached_price(tk)
    if p > 0:
        pv = p * a_val
        fix_c = cfg['fix_c']
        pl = pv - fix_c
        col = "#a8d5a2" if pl >= 0 else "#fbb"
        st.markdown(f"**Price:** `{p:,.3f}` | **Value:** `{pv:,.2f}` | **P/L** (vs {fix_c:,}): <span style='color:{col};font-weight:bold;'>`{pl:,.2f}`</span>", unsafe_allow_html=True)
    else:
        st.info(f"Price for {tk} is currently unavailable.")

    st.write(f"**Buy:** Amount `{sell_calc[1]}` | Price `{sell_calc[0]}` | Cost `{sell_calc[2]}`") # Logic from original
    if st.checkbox(f'Match BUY order for {tk}', key=f'buy_match_{tk}'):
        if st.button(f"EXECUTE BUY {tk}", key=f'go_buy_{tk}'):
            try:
                new_val = a_last + sell_calc[1]
                a_conf = cfg['asset_field']
                client = clients.get(a_conf['channel_id'])
                if client:
                    client.update({a_conf['field']: new_val})
                    st.success(f"BUY successful. New asset value: {new_val}")
                    clear_all_caches()
                else:
                    st.error(f"Client for channel {a_conf['channel_id']} not found.")
            except Exception as e:
                st.error(f"BUY execution failed for {tk}: {e}")

# --- MAIN APPLICATION LOGIC ---
def main():
    st.title("📈 Trading Monitor")
    
    cfg = load_config()
    assets_cfg = cfg['assets']
    start_date = cfg.get('global_settings', {}).get('start_date')

    # Get ThingSpeak clients (cached resource)
    thingspeak_clients = get_thingspeak_clients(assets_cfg)

    with st.expander("⚙️ Controls & Asset Setup", expanded=True):
        nex = nex_day_sell = 0
        if st.checkbox('Nex Day Simulation'):
            c1, c2, _ = st.columns([1, 1, 4])
            if c1.button("Nex Day"): nex = 1
            if c2.button("Nex Day (Sell)"): nex, nex_day_sell = 1, 1
            st.info(f"Simulation Mode: nex = {nex}, nex_day_sell = {nex_day_sell}")
        
        st.divider()
        
        cols = st.columns(2)
        start_chk = cols[0].checkbox('Enable Asset Updates on ThingSpeak')
        diff_val = cols[1].number_input('Diff Parameter', step=1, value=60)
        
        if start_chk:
            render_asset_update_controls(assets_cfg, thingspeak_clients)

    # --- Single point of data fetching ---
    bundle = fetch_all_data(assets_cfg, thingspeak_clients, start_date)
    if not bundle:
        st.error("Failed to fetch data. Please check logs and try again.")
        st.stop()
        
    last_assets = {t: data.get('asset_val', 0.0) for t, data in bundle.items() if data}
    with st.expander("Asset Holdings (Manual Override for Simulation)", expanded=True):
        asset_inputs = render_asset_inputs(assets_cfg, last_assets)

    st.divider()

    # --- Process each asset's data for UI display ---
    processed_assets = []
    for cfg_a in assets_cfg:
        tk = cfg_a['ticker']
        info = bundle.get(tk)
        if not info or info.get('history') is None:
            st.warning(f"Data for {tk} could not be loaded. Skipping.")
            continue

        # Prepare DataFrame with actions
        df = info['history'].copy()
        df['index'] = range(len(df))
        dummy = pd.DataFrame(index=[f'+{i}' for i in range(5)])
        df = pd.concat([df, dummy]).fillna("")
        df['action'] = ""

        tracer = SimulationTracer(info['fx_str'])
        acts = tracer.run()
        # FIX: Check if acts is not empty before assigning to prevent ValueError
        if len(acts) > 0:
            num_to_assign = min(len(df), len(acts))
            df.iloc[:num_to_assign, df.columns.get_loc('action')] = acts[:num_to_assign]

        # Get asset value (from UI override or fetched data)
        asset_val = asset_inputs.get(tk, info.get('asset_val', 0.0))
        
        # Calculate P/L and determine action emoji for tab label
        pl_val = 0.0
        pr = get_cached_price(tk)
        if pr and asset_val:
            pl_val = pr * asset_val - cfg_a['fix_c']

        act_emoji = "⚪" # Default
        try:
            action_index = 1 + nex
            if action_index < len(df) and df.action.values[action_index] != "":
                raw_act = int(df.action.values[action_index])
                final_act = 1 - raw_act if nex_day_sell else raw_act
                act_emoji = "🟢" if final_act == 1 else "🔴"
        except (ValueError, IndexError):
            pass # Keep default emoji

        processed_assets.append({
            "config": cfg_a, "ticker": tk, "df_data": df, "fx_js_str": info['fx_str'],
            "asset_last": last_assets.get(tk, 0.0), "asset_val": asset_val,
            "calcs": {
                'buy': buy(asset_val, cfg_a['fix_c'], diff_val),
                'sell': sell(asset_val, cfg_a['fix_c'], diff_val)
            },
            "act_emoji": act_emoji, "pl_value": pl_val,
        })

    # --- Render UI Tabs ---
    if processed_assets:
        with st.expander("📈 Trading Dashboard", expanded=True):
            tab_labels = [f"{d['ticker']} {d['act_emoji']} | P/L: {d['pl_value']:,.2f}" for d in processed_assets]
            tabs = st.tabs(tab_labels)
            for i, d in enumerate(processed_assets):
                with tabs[i]:
                    st.subheader(f"{d['ticker']} (f(x): `{d['fx_js_str']}`)")
                    trading_section(d, nex, nex_day_sell, thingspeak_clients)
                    st.divider()
                    with st.expander("Show Raw Data & Actions"):
                        st.dataframe(d['df_data'], use_container_width=True)

    if st.sidebar.button("RERUN & Clear Caches"):
        clear_all_caches()

if __name__ == "__main__":
    main()
