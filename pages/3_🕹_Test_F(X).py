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
from helpers.data_hub import (
    fetch_monitor_and_asset,
    get_ts_clients,
    MEM as DATA_MEM,
)

st.set_page_config(page_title="Monitor", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------
#                     SimulationTracer (unchanged)
# --------------------------------------------------------------------
class SimulationTracer:
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
            pass
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
        act[0] = 1
        for seed in self.mutation_seeds:
            m_rng = np.random.default_rng(seed)
            mask = m_rng.random(self.action_length) < self.mutation_rate_float
            act[mask] = 1 - act[mask]
            act[0] = 1
        return act


# --------------------------------------------------------------------
#                 CONFIG  /  UTILITIES  (almost unchanged)
# --------------------------------------------------------------------
@st.cache_data
def load_config(path='monitor_config.json') -> Dict[str, Any]:
    if not os.path.exists(path):
        st.error(f"Config not found : {path}"); st.stop()
    try:
        with open(path, 'r', encoding='utf-8') as fp:
            cfg = json.load(fp)
        if not cfg.get('assets'): raise ValueError("assets empty")
        return cfg
    except Exception as e:
        st.error(f"Read config error : {e}"); st.stop()


_cache_lock = Lock()
_price_cache, _price_ts = {}, {}


def get_cached_price(ticker: str, max_age=30) -> float:
    now = datetime.datetime.now()
    with _cache_lock:
        if (p := _price_cache.get(ticker)) and \
           (now - _price_ts.get(ticker, now)).seconds < max_age:
            return p
    try:
        p = yf.Ticker(ticker).fast_info['lastPrice']
        with _cache_lock:
            _price_cache[ticker] = p
            _price_ts[ticker] = now
        return p
    except Exception:
        return 0.0


@lru_cache(maxsize=128)
def sell(asset: float, fix_c: int = 1500, diff: int = 60):
    if asset == 0: return 0, 0, 0
    up = round((fix_c - diff) / asset, 2)
    adj = round(abs(asset * up - fix_c) / up) if up else 0
    tot = round(asset * up + adj * up, 2)
    return up, adj, tot


@lru_cache(maxsize=128)
def buy(asset: float, fix_c: int = 1500, diff: int = 60):
    if asset == 0: return 0, 0, 0
    up = round((fix_c + diff) / asset, 2)
    adj = round(abs(asset * up - fix_c) / up) if up else 0
    tot = round(asset * up - adj * up, 2)
    return up, adj, tot


def clear_all_caches():
    """Flush ทุก cache แล้ว rerun"""
    st.cache_data.clear()
    st.cache_resource.clear()
    DATA_MEM.clear()
    sell.cache_clear(); buy.cache_clear()
    with _cache_lock:
        _price_cache.clear(); _price_ts.clear()
    st.success("🗑️ Cache cleared!")
    st.rerun()


# --------------------------------------------------------------------
#                        UI helpers  (unchanged)
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
                asset_inputs[tk] = st.number_input(f'{tk}_ASSET', step=0.001,
                                                   value=last, key=f"inp_{tk}",
                                                   format="%.2f")
    return asset_inputs


def render_asset_update_controls(cfgs):
    with st.expander("Update Assets on ThingSpeak"):
        ts_clients: Dict[int, thingspeak.Channel] = DATA_MEM.ttl_get("ts_clients", 3600) or {}
        for c in cfgs:
            tk, a_conf = c['ticker'], c['asset_field']
            field = a_conf['field']
            if st.checkbox(f'@_{tk}_ASSET', key=f'chk_{tk}'):
                val = st.number_input(f"New value {tk}", step=0.001, value=0.0,
                                      key=f'new_{tk}')
                if st.button(f"GO_{tk}", key=f"go_{tk}"):
                    try:
                        ts_clients[a_conf['channel_id']].update({field: val})
                        st.success(f"Updated {tk} to {val}")
                        clear_all_caches()
                    except Exception as e:
                        st.error(f"Update fail {tk}: {e}")


def trading_section(data: Dict, nex: int, nex_day_sell: int):
    cfg, tk = data['config'], data['ticker']
    df = data['df_data']
    a_last, a_val = data['asset_last'], data['asset_val']
    buy_calc, sell_calc = data['calcs']['buy'], data['calcs']['sell']
    ts_clients: Dict[int, thingspeak.Channel] = DATA_MEM.ttl_get("ts_clients", 3600) or {}

    try:
        if df.empty or df.action.values[1 + nex] == "":
            act_val = 0
        else:
            raw = int(df.action.values[1 + nex])
            act_val = 1 - raw if nex_day_sell else raw
    except Exception:
        act_val = 0

    if not st.checkbox(f'Limit_Order_{tk}', value=bool(act_val),
                       key=f'limit_{tk}'):
        return

    # -------- SELL ----------
    st.write('sell   ', 'A', sell_calc[1], 'P', sell_calc[0], 'C', sell_calc[2])
    if st.checkbox(f'sell_match_{tk}', key=f'sm_{tk}'):
        if st.button(f"GO_SELL_{tk}", key=f'gos_{tk}'):
            try:
                new_val = a_last - sell_calc[1]
                a_conf = cfg['asset_field']
                ts_clients[a_conf['channel_id']].update({a_conf['field']: new_val})
                st.success(f"Updated : {new_val}")
                clear_all_caches()
            except Exception as e:
                st.error(f"SELL fail {tk}: {e}")

    # -------- price / PL ----------
    p = get_cached_price(tk)
    if p:
        pv = p * a_val
        fix_c = cfg['fix_c']
        pl = pv - fix_c
        col = "#a8d5a2" if pl >= 0 else "#fbb"
        st.markdown(
            f"Price: **{p:,.3f}** | Value: **{pv:,.2f}** | "
            f"P/L (vs {fix_c:,}): "
            f"<span style='color:{col};font-weight:bold;'>{pl:,.2f}</span>",
            unsafe_allow_html=True,
        )
    else:
        st.info("price unavailable")

    # -------- BUY ----------
    st.write('buy    ', 'A', buy_calc[1], 'P', buy_calc[0], 'C', buy_calc[2])
    if st.checkbox(f'buy_match_{tk}', key=f'bm_{tk}'):
        if st.button(f"GO_BUY_{tk}", key=f'gob_{tk}'):
            try:
                new_val = a_last + buy_calc[1]
                a_conf = cfg['asset_field']
                ts_clients[a_conf['channel_id']].update({a_conf['field']: new_val})
                st.success(f"Updated : {new_val}")
                clear_all_caches()
            except Exception as e:
                st.error(f"BUY fail {tk}: {e}")


# --------------------------------------------------------------------
#                               MAIN
# --------------------------------------------------------------------
def main():
    cfg = load_config()
    assets_cfg = cfg['assets']
    start_date = cfg.get('global_settings', {}).get('start_date')

    # ---------------- Controls ----------------
    nex = nex_day_sell = 0
    with st.expander("⚙️ Controls & Asset Setup", expanded=False):
        if st.checkbox('nex_day'):
            c1, c2, _ = st.columns([1, 1, 6])
            if c1.button("Nex_day"): nex = 1
            if c2.button("Nex_day_sell"): nex = nex_day_sell = 1
            st.write(f"nex = {nex}" + (f" | nex_day_sell = {nex_day_sell}" if nex_day_sell else ""))
        st.write("---")
        cols = st.columns(8)
        start_chk = cols[0].checkbox('start')
        diff_val = cols[7].number_input('Diff', step=1, value=60)
        if start_chk:
            render_asset_update_controls(assets_cfg)

    # --------------  Data download  (จุดเดียว) --------------
    bundle = fetch_monitor_and_asset(assets_cfg, start_date)

    # asset from TS (ล่าสุด)  &  UI input
    last_assets = {t: bundle[t]['asset_val'] for t in bundle}
    with st.expander("Asset Holdings", expanded=True):
        asset_inputs = render_asset_inputs(assets_cfg, last_assets)

    st.write("_____")

    # --------------  processing per-asset  --------------
    processed = []
    for cfg_a in assets_cfg:
        tk = cfg_a['ticker']
        info = bundle[tk]

        df = info['history'].copy()
        df['index'] = range(len(df))
        dummy = pd.DataFrame(index=[f'+{i}' for i in range(5)])
        df = pd.concat([df, dummy]).fillna("")
        df['action'] = ""

        tracer = SimulationTracer(info['fx_str'])
        acts = tracer.run()
        if len(acts):
            df.iloc[: len(acts), df.columns.get_loc('action')] = acts

        asset_val = asset_inputs.get(tk, info['asset_val'])

        pl_val = 0.0
        pr = get_cached_price(tk)
        if pr and asset_val:
            pl_val = pr * asset_val - cfg_a['fix_c']

        act_emoji = "⚪"
        if not df.empty and df.action.values[1] != "":
            raw = int(df.action.values[1])
            raw = 1 - raw if nex_day_sell else raw
            act_emoji = "🟢" if raw == 1 else "🔴"

        processed.append({
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

    # --------------  UI Tabs  --------------
    with st.expander("📈 Trading Dashboard", expanded=True):
        labels = [f"{d['ticker']} {d['act_emoji']} | P/L: {d['pl_value']:,.2f}" for d in processed]
        tabs = st.tabs(labels)
        for i, d in enumerate(processed):
            with tabs[i]:
                st.write(f"**{d['ticker']}**  (f(x): `{d['fx_js_str']}`)")
                trading_section(d, nex, nex_day_sell)
                st.write("_____")
                with st.expander("Show Raw Data Action"):
                    st.dataframe(d['df_data'], use_container_width=True)

    if st.sidebar.button("RERUN"):
        clear_all_caches()


if __name__ == "__main__":
    main()
