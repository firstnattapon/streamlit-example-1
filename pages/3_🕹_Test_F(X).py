# pages/3_🕹_Test_F(X).py

import streamlit as st
import numpy as np
import datetime
import pandas as pd
import yfinance as yf
import json
from functools import lru_cache
from threading import Lock
import os
from typing import Dict, Any, List

# ----------  🆕  FAST data layer  ----------
# Import a function and a cache object from our helper module
from helpers.data_hub import (
    fetch_monitor_and_asset,
    MEM as DATA_MEM,
)

st.set_page_config(page_title="Monitor", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------
#                     SimulationTracer (unchanged)
# --------------------------------------------------------------------
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


# --------------------------------------------------------------------
#                 CONFIG  /  UTILITIES
# --------------------------------------------------------------------
@st.cache_data
def load_config(path='monitor_config.json') -> Dict[str, Any]:
    if not os.path.exists(path):
        st.error(f"Config not found : {path}"); st.stop()
    try:
        with open(path, 'r', encoding='utf-8') as fp:
            cfg = json.load(fp)
        if not cfg.get('assets'): raise ValueError("assets list in config is empty")
        return cfg
    except Exception as e:
        st.error(f"Read config error : {e}"); st.stop()

# Local cache for prices to avoid hitting yfinance API on every UI interaction
_price_lock = Lock()
_price_cache, _price_ts = {}, {}

def get_cached_price(ticker: str, max_age=30) -> float:
    now = datetime.datetime.now()
    with _price_lock:
        price = _price_cache.get(ticker)
        timestamp = _price_ts.get(ticker, datetime.datetime.min)
        if price and (now - timestamp).seconds < max_age:
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
def sell(asset: float, fix_c: int = 1500, diff: int = 60):
    if asset == 0: return 0, 0, 0
    up = round((fix_c - diff) / asset, 2) if asset != 0 else 0
    adj = round(abs(asset * up - fix_c) / up) if up != 0 else 0
    tot = round(asset * up + adj * up, 2)
    return up, adj, tot

@lru_cache(maxsize=128)
def buy(asset: float, fix_c: int = 1500, diff: int = 60):
    if asset == 0: return 0, 0, 0
    up = round((fix_c + diff) / asset, 2) if asset != 0 else 0
    adj = round(abs(asset * up - fix_c) / up) if up != 0 else 0
    tot = round(asset * up - adj * up, 2)
    return up, adj, tot

def clear_all_caches():
    """Flush all caches and rerun the app."""
    st.cache_data.clear()
    st.cache_resource.clear()
    DATA_MEM.clear() # Clears the custom cache in data_hub
    sell.cache_clear(); buy.cache_clear()
    with _price_lock:
        _price_cache.clear(); _price_ts.clear()
    st.success("🗑️ All caches cleared!")
    st.rerun()

# --------------------------------------------------------------------
#                        UI HELPER FUNCTIONS
# --------------------------------------------------------------------
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

def render_asset_update_controls(cfgs):
    with st.expander("Update Assets on ThingSpeak"):
        # We need the ThingSpeak clients to update values
        ts_clients = DATA_MEM.ttl_get("ts_clients", 3600)
        if not ts_clients:
            st.warning("ThingSpeak clients not initialized. Please refresh.")
            return

        for c in cfgs:
            tk, a_conf = c['ticker'], c['asset_field']
            field = a_conf['field']
            if st.checkbox(f'@ Update {tk} Asset', key=f'chk_update_{tk}'):
                val = st.number_input(f"New value for {tk}", step=0.001, value=0.0,
                                      key=f'new_val_{tk}')
                if st.button(f"GO Update {tk}", key=f"btn_go_{tk}"):
                    try:
                        client = ts_clients.get(a_conf['channel_id'])
                        if client:
                            client.update({field: val})
                            st.success(f"Updated {tk} to {val}")
                            clear_all_caches()
                        else:
                            st.error(f"Client for channel {a_conf['channel_id']} not found.")
                    except Exception as e:
                        st.error(f"Update failed for {tk}: {e}")

def trading_section(data: Dict, nex: int, nex_day_sell: int):
    cfg, tk = data['config'], data['ticker']
    df = data['df_data']
    a_last, a_val = data['asset_last'], data['asset_val']
    buy_calc, sell_calc = data['calcs']['buy'], data['calcs']['sell']
    ts_clients = DATA_MEM.ttl_get("ts_clients", 3600)

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

    st.write(f"**Sell:** Amount `{sell_calc[1]}` | Price `{sell_calc[0]}` | Cost `{sell_calc[2]}`")
    if st.checkbox(f'Match SELL order for {tk}', key=f'sell_match_{tk}'):
        if st.button(f"EXECUTE SELL {tk}", key=f'go_sell_{tk}'):
            try:
                new_val = a_last - sell_calc[1]
                a_conf = cfg['asset_field']
                client = ts_clients.get(a_conf['channel_id'])
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

    st.write(f"**Buy:** Amount `{buy_calc[1]}` | Price `{buy_calc[0]}` | Cost `{buy_calc[2]}`")
    if st.checkbox(f'Match BUY order for {tk}', key=f'buy_match_{tk}'):
        if st.button(f"EXECUTE BUY {tk}", key=f'go_buy_{tk}'):
            try:
                new_val = a_last + buy_calc[1]
                a_conf = cfg['asset_field']
                client = ts_clients.get(a_conf['channel_id'])
                if client:
                    client.update({a_conf['field']: new_val})
                    st.success(f"BUY successful. New asset value: {new_val}")
                    clear_all_caches()
                else:
                    st.error(f"Client for channel {a_conf['channel_id']} not found.")
            except Exception as e:
                st.error(f"BUY execution failed for {tk}: {e}")

# --------------------------------------------------------------------
#                               MAIN
# --------------------------------------------------------------------
def main():
    st.title("📈 F(X) Trading Monitor")
    
    cfg = load_config()
    assets_cfg = cfg['assets']
    start_date = cfg.get('global_settings', {}).get('start_date')

    with st.expander("⚙️ Controls & Asset Setup"):
        nex = nex_day_sell = 0
        if st.checkbox('Nex Day Simulation'):
            c1, c2, _ = st.columns([1, 1, 4])
            if c1.button("Nex Day"): nex = 1
            if c2.button("Nex Day (Sell)"): nex = nex_day_sell = 1
            st.info(f"Simulation Mode: nex = {nex}, nex_day_sell = {nex_day_sell}")
        
        st.divider()
        
        cols = st.columns(2)
        start_chk = cols[0].checkbox('Enable Asset Updates on ThingSpeak')
        diff_val = cols[1].number_input('Diff Parameter', step=1, value=60)
        
        if start_chk:
            render_asset_update_controls(assets_cfg)

    # --- Single point of data fetching ---
    bundle = fetch_monitor_and_asset(assets_cfg, start_date)
    if not bundle:
        st.error("Failed to fetch data. Please check logs and try again.")
        st.stop()
        
    last_assets = {t: data.get('asset_val', 0.0) for t, data in bundle.items()}
    with st.expander("Asset Holdings (Manual Override for Simulation)", expanded=True):
        asset_inputs = render_asset_inputs(assets_cfg, last_assets)

    st.divider()

    # --- Process each asset's data ---
    processed_assets = []
    for cfg_a in assets_cfg:
        tk = cfg_a['ticker']
        info = bundle.get(tk)
        if not info or info.get('history') is None:
            st.warning(f"Data for {tk} could not be loaded. Skipping.")
            continue

        # Prepare DataFrame
        df = info['history'].copy()
        df['index'] = range(len(df))
        dummy = pd.DataFrame(index=[f'+{i}' for i in range(5)])
        df = pd.concat([df, dummy]).fillna("")
        df['action'] = ""

        # Run tracer and safely assign actions
        tracer = SimulationTracer(info['fx_str'])
        acts = tracer.run()
        
        # ✅✅✅ THIS IS THE FIX FOR THE ValueError ✅✅✅
        if len(acts) > 0:
            num_to_assign = min(len(df), len(acts))
            df.iloc[:num_to_assign, df.columns.get_loc('action')] = acts[:num_to_assign]

        # Get asset value (from UI override or fetched data)
        asset_val = asset_inputs.get(tk, info['asset_val'])
        
        # Calculate P/L and determine action emoji
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
            "config": cfg_a,
            "ticker": tk,
            "df_data": df,
            "fx_js_str": info['fx_str'],
            "asset_last": last_assets.get(tk, 0.0),
            "asset_val": asset_val,
            "calcs": {
                'buy': buy(asset_val, cfg_a['fix_c'], diff_val),
                'sell': sell(asset_val, cfg_a['fix_c'], diff_val)
            },
            "act_emoji": act_emoji,
            "pl_value": pl_val,
        })

    # --- Render UI Tabs ---
    if processed_assets:
        tab_labels = [f"{d['ticker']} {d['act_emoji']} | P/L: {d['pl_value']:,.2f}" for d in processed_assets]
        tabs = st.tabs(tab_labels)
        for i, d in enumerate(processed_assets):
            with tabs[i]:
                st.subheader(f"{d['ticker']} (f(x): `{d['fx_js_str']}`)")
                trading_section(d, nex, nex_day_sell)
                st.divider()
                with st.expander("Show Raw Data & Actions"):
                    st.dataframe(d['df_data'], use_container_width=True)

    if st.sidebar.button("Force Rerun & Clear Caches"):
        clear_all_caches()

if __name__ == "__main__":
    main()
