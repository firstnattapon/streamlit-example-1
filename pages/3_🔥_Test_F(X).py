# 📈_Monitor.py — Pro Optimistic UI + Smart Dashboard + Clean Cards
# ========================= ENHANCED PERFORMANCE & UX =======================

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
from typing import List, Dict, Optional, Tuple
import tenacity
import pytz
import re
from urllib.parse import urlencode 
from urllib.request import urlopen
import time
import math

# ---------------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------------
st.set_page_config(page_title="Monitor Pro", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------------------
# Custom CSS for Better Cards
# ---------------------------------------------------------------------------------
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .metric-card {
        background-color: #262730;
        border: 1px solid #464b5c;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .profit { color: #00ff7f; font-weight: bold; }
    .loss { color: #ff6347; font-weight: bold; }
    .highlight-action { border: 2px solid #f0ad4e !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------------
# Globals & Utils
# ---------------------------------------------------------------------------------
_EPS = 1e-12
TZ_BKK = pytz.timezone('Asia/Bangkok')
TZ_NY = pytz.timezone('America/New_York')
_FIELD_NUM_RE = re.compile(r'(\d+)')

def r2(x: float) -> float:
    try: return round(float(x), 2)
    except: return 0.0

def safe_float(x, default: float = 0.0) -> float:
    try: return float(x)
    except: return float(default)

def xor01(a: int, b: int) -> int:
    try: return (int(a) ^ int(b)) & 1
    except: return 0

# ---------------------------------------------------------------------------------
# Logic: Simulation & Config (Keep Core Logic Efficient)
# ---------------------------------------------------------------------------------
class SimulationTracer:
    def __init__(self, encoded_string: str):
        self.encoded_string = str(encoded_string)
        self._decode()

    def _decode(self):
        s = self.encoded_string
        if not s.isdigit():
            self.al, self.mr, self.ds, self.ms, self.mrf = 0, 0, 0, [], 0.0
            return
        
        nums, idx = [], 0
        try:
            while idx < len(s):
                length = int(s[idx])
                idx += 1
                nums.append(int(s[idx : idx + length]))
                idx += length
        except: pass

        if len(nums) < 3:
            self.al, self.mr, self.ds, self.ms, self.mrf = 0, 0, 0, [], 0.0
            return

        self.al = nums[0]
        self.mr = nums[1]
        self.ds = nums[2]
        self.ms = nums[3:]
        self.mrf = self.mr / 100.0

    @lru_cache(maxsize=128)
    def run(self) -> np.ndarray:
        if self.al <= 0: return np.array([])
        rng = np.random.default_rng(seed=self.ds)
        acts = rng.integers(0, 2, size=self.al)
        if self.al > 0: acts[0] = 1
        for m_seed in self.ms:
            mrng = np.random.default_rng(seed=m_seed)
            mask = mrng.random(self.al) < self.mrf
            acts[mask] = 1 - acts[mask]
            if self.al > 0: acts[0] = 1
        return acts

@st.cache_data
def load_config(file_path='monitor_config.json') -> Dict:
    if not os.path.exists(file_path): return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except: return {}

CONFIG_DATA = load_config()
if not CONFIG_DATA: st.stop()
ASSET_CONFIGS = CONFIG_DATA.get('assets', [])
ALL_TICKERS = [c['ticker'] for c in ASSET_CONFIGS]

# ---------------------------------------------------------------------------------
# Data Fetching (Optimized)
# ---------------------------------------------------------------------------------
@st.cache_resource
def get_thingspeak_clients(configs):
    clients = {}
    unique = set()
    for c in configs:
        m = c['monitor_field']
        a = c['asset_field']
        unique.add((m['channel_id'], m['api_key']))
        unique.add((a['channel_id'], a['api_key']))
    for cid, key in unique:
        try: clients[int(cid)] = thingspeak.Channel(int(cid), key, fmt='json')
        except: pass
    return clients

THINGSPEAK_CLIENTS = get_thingspeak_clients(ASSET_CONFIGS)

@st.cache_data(ttl=120, show_spinner=False)
def get_prices_map(tickers: List[str]) -> Dict[str, float]:
    """Batch fetch prices"""
    tickers = list(set([t for t in tickers if t]))
    out = {}
    
    def _fetch(t):
        try:
            tk = yf.Ticker(t)
            # Try fast info first
            p = tk.fast_info.get('lastPrice', 0.0)
            if p > 0: return t, p
            # Fallback to history
            hist = tk.history(period='1d')
            if not hist.empty: return t, float(hist['Close'].iloc[-1])
        except: pass
        return t, 0.0

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(_fetch, t) for t in tickers]
        for f in concurrent.futures.as_completed(futs):
            t, p = f.result()
            out[t] = p
    return out

# ---------------------------------------------------------------------------------
# Trading Math
# ---------------------------------------------------------------------------------
@lru_cache(maxsize=2048)
def _trade_math(asset: float, fix_c: float, Diff: float, side: int) -> Tuple[float, int, float]:
    a = float(asset)
    if abs(a) <= _EPS: return 0.0, 0, 0.0
    up = r2((float(fix_c) + float(side)*float(Diff)) / max(abs(a), _EPS))
    if abs(up) <= _EPS: return 0.0, 0, 0.0
    delta_qty = int(round(abs(a*up - float(fix_c)) / max(abs(up), _EPS)))
    total = r2(a*up - float(side)*delta_qty*up)
    return float(up), int(delta_qty), float(total)

def calculate_trade_logic(ticker, asset_val, fix_c, diff_val):
    return {
        'sell': _trade_math(asset_val, fix_c, diff_val, side=-1),
        'buy': _trade_math(asset_val, fix_c, diff_val, side=1)
    }

# ---------------------------------------------------------------------------------
# Optimistic UI & Queue
# ---------------------------------------------------------------------------------
def ts_update_via_http(write_key, field, value):
    fnum = _FIELD_NUM_RE.search(str(field))
    if not fnum: return "0"
    fn = fnum.group(1)
    url = f"https://api.thingspeak.com/update?api_key={write_key}&field{fn}={value}"
    try:
        with urlopen(url, timeout=4) as resp:
            return resp.read().decode('utf-8').strip()
    except: return "0"

def _optimistic_update(ticker, new_val, prev_val, asset_conf, op):
    st.session_state.setdefault('_overrides', {})[ticker] = float(new_val)
    st.session_state.setdefault('_queue', []).append({
        'ticker': ticker, 'cid': int(asset_conf['channel_id']),
        'field': asset_conf['field'], 
        'key': asset_conf.get('write_api_key') or asset_conf.get('api_key'),
        'new': float(new_val), 'prev': float(prev_val), 'op': op,
        'ts': time.time()
    })
    st.session_state['_cache_bump'] = st.session_state.get('_cache_bump', 0) + 1
    st.rerun()

def process_queue():
    q = st.session_state.get('_queue', [])
    if not q: return
    
    # Simple rate limit tracking
    last_ts = st.session_state.get('_last_ts_call', {})
    
    rem = []
    for job in q:
        cid = job['cid']
        now = time.time()
        # Ensure 15s gap per channel
        if now - last_ts.get(cid, 0) < 16:
            rem.append(job)
            continue
            
        res = ts_update_via_http(job['key'], job['field'], job['new'])
        if res == "0":
            # Fail: Rollback override
            st.toast(f"❌ Update failed for {job['ticker']}. Rolling back.")
            st.session_state.setdefault('_overrides', {})[job['ticker']] = job['prev']
        else:
            st.toast(f"✅ Updated {job['ticker']} ({job['op']}) -> Entry {res}")
            last_ts[cid] = now
            
    st.session_state['_last_ts_call'] = last_ts
    st.session_state['_queue'] = rem
    
    if rem: # If items remain, rerun to process them after delay
        time.sleep(1)
        st.rerun()

# ---------------------------------------------------------------------------------
# Data Loader Wrapper
# ---------------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_all_data(configs, _clients, cache_bump):
    # Simplified fetch logic focusing on critical data
    results = {'monitors': {}, 'assets': {}}
    
    def _task(conf):
        tkr = conf['ticker']
        try:
            # Asset
            ac = conf['asset_field']
            cli = _clients.get(int(ac['channel_id']))
            if cli:
                f = ac['field']
                d = cli.get_field_last(field=f)
                val = json.loads(d)[f]
                results['assets'][tkr] = float(val)
            
            # Monitor Action
            mc = conf['monitor_field']
            mcli = _clients.get(int(mc['channel_id']))
            if mcli:
                mf = mc['field']
                md = mcli.get_field_last(field=str(mf))
                js_str = json.loads(md)[f"field{mf}"]
                
                # Decode tracer
                tracer = SimulationTracer(str(js_str))
                acts = tracer.run()
                results['monitors'][tkr] = (acts, str(js_str))
        except:
            results['assets'][tkr] = 0.0
            results['monitors'][tkr] = (np.array([]), "0")

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(_task, configs))
    
    return results

# ---------------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------------
def render_portfolio_summary(configs, assets_map, prices_map):
    total_value = 0.0
    total_fix = 0.0
    
    for c in configs:
        t = c['ticker']
        qty = assets_map.get(t, 0.0)
        price = prices_map.get(t, 0.0)
        fix = float(c['fix_c'])
        
        total_value += (qty * price)
        total_fix += fix
        
    diff = total_value - total_fix
    color = "green" if diff >= 0 else "red"
    
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Portfolio Value", f"${total_value:,.2f}")
        c2.metric("Target Fix Capital", f"${total_fix:,.2f}")
        c3.markdown(f"**Net P/L:** <span style='color:{color};font-size:1.2em'>${diff:,.2f}</span>", unsafe_allow_html=True)

def render_ticker_card(idx, config, price, asset_qty, action_signal, calc, nex_state):
    t = config['ticker']
    fix = float(config['fix_c'])
    val = asset_qty * price
    pl = val - fix
    
    # Visual styles
    card_border = "1px solid #444"
    bg_color = "#262730"
    
    # Logic for signal
    signal_text = "HOLD"
    signal_color = "#888"
    if action_signal is not None:
        if action_signal == 1: 
            signal_text = "BUY SIGNAL"
            signal_color = "#00ff7f" # Green
            card_border = "2px solid #00ff7f"
        elif action_signal == 0:
            signal_text = "SELL SIGNAL"
            signal_color = "#ff6347" # Red
            card_border = "2px solid #ff6347"

    with st.container():
        st.markdown(f"""
        <div style="background:{bg_color}; padding:15px; border-radius:10px; border:{card_border}; margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h3 style="margin:0;">{t} <span style="font-size:0.6em; color:#aaa;">Target: {fix:,.0f}</span></h3>
                <span style="color:{signal_color}; font-weight:bold; border:1px solid {signal_color}; padding:2px 8px; border-radius:4px;">{signal_text}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:10px; font-size:0.9em;">
                <div>Price: <b>{price:.2f}</b></div>
                <div>Qty: <b>{asset_qty:.4f}</b></div>
                <div>Val: <b>{val:,.2f}</b></div>
                <div style="color:{'#0f0' if pl>=0 else '#f00'}">P/L: <b>{pl:+,.2f}</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Actions Area
        c1, c2 = st.columns(2)
        
        # SELL SIDE
        with c1:
            sc = calc['sell']
            if sc[1] > 0: # If actionable
                st.info(f"Sell {sc[1]} @ {sc[0]:.2f} (Total: {sc[2]:.2f})")
                if st.button(f"📉 SELL {t}", key=f"sell_{t}_{idx}", type="secondary"):
                    new_val = asset_qty - sc[1]
                    _optimistic_update(t, new_val, asset_qty, config['asset_field'], "SELL")
            else:
                st.caption(f"Sell Trigger: {sc[0]:.2f}")

        # BUY SIDE
        with c2:
            bc = calc['buy']
            if bc[1] > 0: # If actionable
                st.success(f"Buy {bc[1]} @ {bc[0]:.2f} (Total: {bc[2]:.2f})")
                if st.button(f"📈 BUY {t}", key=f"buy_{t}_{idx}", type="primary"):
                    new_val = asset_qty + bc[1]
                    _optimistic_update(t, new_val, asset_qty, config['asset_field'], "BUY")
            else:
                st.caption(f"Buy Trigger: {bc[0]:.2f}")

# ---------------------------------------------------------------------------------
# Main Layout
# ---------------------------------------------------------------------------------
# Init State
if '_cache_bump' not in st.session_state: st.session_state['_cache_bump'] = 0
if '_overrides' not in st.session_state: st.session_state['_overrides'] = {}
if 'nex' not in st.session_state: st.session_state.nex = 0
if 'Nex_day_sell' not in st.session_state: st.session_state.Nex_day_sell = 0
if 'diff_value' not in st.session_state: st.session_state.diff_value = 60.0

# Top Control Bar
c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    st.title("🚀 Monitor Pro")
with c2:
    st.caption("Config Control")
    nex_mode = st.checkbox("Enable Next Day Logic", value=False)
    if nex_mode:
        n1, n2 = st.columns(2)
        if n1.button("Set Next Day"): 
            st.session_state.nex = 1
            st.session_state.Nex_day_sell = 0
        if n2.button("Set Next Day Sell"):
            st.session_state.nex = 1
            st.session_state.Nex_day_sell = 1
    else:
        st.session_state.nex = 0
        st.session_state.Nex_day_sell = 0
with c3:
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

# Fetch Data
DATA = fetch_all_data(ASSET_CONFIGS, THINGSPEAK_CLIENTS, st.session_state['_cache_bump'])
PRICES = get_prices_map(ALL_TICKERS)

# Apply Overrides
ASSETS_MAP = DATA['assets'].copy()
ASSETS_MAP.update(st.session_state.get('_overrides', {}))

# Render Summary
render_portfolio_summary(ASSET_CONFIGS, ASSETS_MAP, PRICES)

st.divider()

# Filter & Sort Controls
fc1, fc2 = st.columns([3, 1])
with fc1:
    sort_actionable = st.checkbox("⚡ Prioritize Actionable Tickers", value=True)
with fc2:
    diff_input = st.number_input("Global Diff", value=st.session_state.diff_value, step=10.0)
    st.session_state.diff_value = diff_input

# Prepare List
tickers_data = []
for conf in ASSET_CONFIGS:
    t = conf['ticker']
    p = PRICES.get(t, 0.0)
    qty = ASSETS_MAP.get(t, 0.0)
    
    # Action Signal Logic
    acts, js = DATA['monitors'].get(t, (np.array([]), "0"))
    action_signal = None
    if len(acts) > 0:
        idx = 1 + st.session_state.nex
        if idx < len(acts):
            raw = int(acts[idx])
            flip = int(st.session_state.Nex_day_sell)
            action_signal = xor01(raw, flip)
            
    # Calc Trade Math
    diff = conf.get('diff', st.session_state.diff_value)
    calc = calculate_trade_logic(t, qty, float(conf['fix_c']), diff)
    
    # Determine "Actionable" score for sorting
    # Score 2 = Signal + Price matches trigger
    # Score 1 = Signal only
    # Score 0 = None
    score = 0
    if action_signal == 1 and calc['buy'][1] > 0: score = 2
    elif action_signal == 0 and calc['sell'][1] > 0: score = 2
    elif action_signal is not None: score = 1
    
    tickers_data.append({
        'conf': conf, 'price': p, 'qty': qty, 
        'sig': action_signal, 'calc': calc, 'score': score
    })

# Sort
if sort_actionable:
    tickers_data.sort(key=lambda x: x['score'], reverse=True)

# Render Grid
grid_cols = st.columns(3)
for i, data in enumerate(tickers_data):
    with grid_cols[i % 3]:
        render_ticker_card(
            i, data['conf'], data['price'], data['qty'], 
            data['sig'], data['calc'], st.session_state.nex
        )

# Process Queue at the end
process_queue()
