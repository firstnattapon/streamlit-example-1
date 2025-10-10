# 📈_Monitor.py — Pro Optimistic UI (2-phase queue) + Min_Rebalance (clean UI)
# ------------------------------------------------------------

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
import time  # RATE-LIMIT

# ---------------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------------
st.set_page_config(page_title="Monitor", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------------
# Small math helpers (0/1 logic)
# ---------------------------------------------------------------------------------
def xor01(a: int, b: int) -> int:
    """XOR สำหรับบิต 0/1 โดยใช้ bitwise operation ที่เร็วและปลอดภัย [SIMPLE/STABLE]"""
    try:
        return (int(a) ^ int(b)) & 1
    except (ValueError, TypeError):
        return 0

def safe_float(x, default: float = 0.0) -> float:
    """แปลงค่าเป็น float อย่างปลอดภัย ถ้าแปลงไม่ได้จะคืนค่า default [SIMPLE/STABLE]"""
    try:
        return float(x)
    except (ValueError, TypeError, AttributeError):
        return float(default)

# ---------------------------------------------------------------------------------
# SimulationTracer (เดิม)
# ---------------------------------------------------------------------------------
class SimulationTracer:
    def __init__(self, encoded_string: str):
        self.encoded_string: str = str(encoded_string)
        self._decode_and_set_attributes()

    def _decode_and_set_attributes(self) -> None:
        encoded_string = self.encoded_string
        if not encoded_string.isdigit():
            self.action_length = 0
            self.mutation_rate = 0
            self.dna_seed = 0
            self.mutation_seeds = []
            self.mutation_rate_float = 0.0
            return

        decoded_numbers: List[int] = []
        idx = 0
        try:
            while idx < len(encoded_string):
                length_of_number = int(encoded_string[idx])
                idx += 1
                number_str = encoded_string[idx: idx + length_of_number]
                idx += length_of_number
                decoded_numbers.append(int(number_str))
        except (IndexError, ValueError):
            pass

        if len(decoded_numbers) < 3:
            self.action_length = 0
            self.mutation_rate = 0
            self.dna_seed = 0
            self.mutation_seeds = []
            self.mutation_rate_float = 0.0
            return

        self.action_length = decoded_numbers[0]
        self.mutation_rate = decoded_numbers[1]
        self.dna_seed = decoded_numbers[2]
        self.mutation_seeds = decoded_numbers[3:]
        self.mutation_rate_float = self.mutation_rate / 100.0

    @lru_cache(maxsize=128)
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

# ---------------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------------
@st.cache_data
def load_config(file_path: str = 'monitor_config.json') -> Dict:
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

ASSET_CONFIGS: List[Dict] = CONFIG_DATA.get('assets', [])
GLOBAL_START_DATE: Optional[str] = CONFIG_DATA.get('global_settings', {}).get('start_date')

if not ASSET_CONFIGS:
    st.error("No 'assets' list found in monitor_config.json")
    st.stop()

# ---------------------------------------------------------------------------------
# ThingSpeak Clients
# ---------------------------------------------------------------------------------
@st.cache_resource
def get_thingspeak_clients(configs: List[Dict]) -> Dict[int, thingspeak.Channel]:
    clients: Dict[int, thingspeak.Channel] = {}
    unique_channels = set()
    for config in configs:
        unique_channels.add((config['monitor_field']['channel_id'], config['monitor_field']['api_key']))
        unique_channels.add((config['asset_field']['channel_id'], config['asset_field']['api_key']))
    
    for channel_id, api_key in unique_channels:
        try:
            clients[int(channel_id)] = thingspeak.Channel(int(channel_id), api_key, fmt='json')
        except Exception as e:
            st.error(f"Failed to create client for Channel ID {channel_id}: {e}")
    return clients

THINGSPEAK_CLIENTS = get_thingspeak_clients(ASSET_CONFIGS)

# ---------------------------------------------------------------------------------
# Cache / Rerun Management
# ---------------------------------------------------------------------------------
def clear_all_caches() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()
    sell.cache_clear()
    buy.cache_clear()
    ui_state_keys_to_preserve = {
        'select_key', 'nex', 'Nex_day_sell',
        '_cache_bump', '_last_assets_overrides',
        '_all_data_cache', '_skip_refresh_on_rerun',
        '_ts_last_update_at',
        '_pending_ts_update', '_ts_entry_ids',
        '_widget_shadow',
        'min_rebalance',
        'diff_value', '_last_selected_ticker'
    }
    for key in list(st.session_state.keys()):
        if key not in ui_state_keys_to_preserve:
            try:
                del st.session_state[key]
            except Exception:
                pass
    st.success("🗑️ Data caches cleared! UI state preserved.")

def rerun_keep_selection(ticker: str) -> None:
    st.session_state["_pending_select_key"] = ticker
    st.rerun()

# ---------------------------------------------------------------------------------
# Calc Utils
# ---------------------------------------------------------------------------------
@lru_cache(maxsize=128)
def sell(asset: float, fix_c: float = 1500, Diff: float = 60) -> Tuple[float, int, float]:
    if asset == 0:
        return 0.0, 0, 0.0
    unit_price = round((fix_c - Diff) / asset, 2)
    adjust_qty = round(abs(asset * unit_price - fix_c) / unit_price) if unit_price != 0 else 0
    total = round(asset * unit_price + adjust_qty * unit_price, 2)
    return unit_price, adjust_qty, total

@lru_cache(maxsize=128)
def buy(asset: float, fix_c: float = 1500, Diff: float = 60) -> Tuple[float, int, float]:
    if asset == 0:
        return 0.0, 0, 0.0
    unit_price = round((fix_c + Diff) / asset, 2)
    adjust_qty = round(abs(asset * unit_price - fix_c) / unit_price) if unit_price != 0 else 0
    total = round(asset * unit_price - adjust_qty * unit_price, 2)
    return unit_price, adjust_qty, total

# ---------------------------------------------------------------------------------
# Price & Time
# ---------------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
@tenacity.retry(wait=tenacity.wait_fixed(2), stop=tenacity.stop_after_attempt(3))
def get_cached_price(ticker: str) -> float:
    try:
        tk = yf.Ticker(ticker)
        # 1: fast_info
        p = safe_float(tk.fast_info.get('lastPrice', 0.0))
        if p > 0: return p
        # 2: info
        inf = getattr(tk, 'info', {}) or {}
        p = safe_float(inf.get('regularMarketPrice', 0.0))
        if p > 0: return p
        # 3: history
        df = tk.history(period='5d')
        if isinstance(df, pd.DataFrame) and not df.empty and 'Close' in df:
            p = safe_float(df['Close'].iloc[-1])
            if p > 0: return p
        return 0.0
    except Exception:
        return 0.0

@st.cache_data(ttl=3600, show_spinner=False)
def get_history_df_max_close_bkk(ticker: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period='max')[['Close']].round(3)
    try:
        df.index = df.index.tz_convert('Asia/Bangkok')
    except TypeError:
        df.index = df.index.tz_localize('UTC').tz_convert('Asia/Bangkok')
    return df

@st.cache_data(ttl=60, show_spinner=False)
def get_current_ny_date() -> datetime.date:
    ny_tz = pytz.timezone('America/New_York')
    return datetime.datetime.now(ny_tz).date()

def _previous_weekday(d: datetime.date) -> datetime.date:
    wd = d.weekday()
    if wd == 0:  # Monday
        return d - datetime.timedelta(days=3)
    elif wd == 6:  # Sunday
        return d - datetime.timedelta(days=2)
    else:
        return d - datetime.timedelta(days=1)

@st.cache_data(ttl=600, show_spinner=False)
def get_latest_us_premarket_open_bkk() -> datetime.datetime:
    tz_ny, tz_bkk = pytz.timezone('America/New_York'), pytz.timezone('Asia/Bangkok')
    now_ny = datetime.datetime.now(tz_ny)
    date_ny = now_ny.date()

    def make_open(dt_date: datetime.date) -> datetime.datetime:
        dt_naive = datetime.datetime(dt_date.year, dt_date.month, dt_date.day, 4, 0, 0)
        return tz_ny.localize(dt_naive)

    candidate = make_open(date_ny)
    while candidate.weekday() >= 5: # Saturday or Sunday
        date_ny = _previous_weekday(date_ny)
        candidate = make_open(date_ny)

    if now_ny < candidate:
        date_ny = _previous_weekday(date_ny)
        candidate = make_open(date_ny)
        while candidate.weekday() >= 5:
            date_ny = _previous_weekday(date_ny)
            candidate = make_open(date_ny)

    return candidate.astimezone(tz_bkk)

# ---------------------------------------------------------------------------------
# ThingSpeak helpers
# ---------------------------------------------------------------------------------
def _field_number(field_value) -> Optional[int]:
    if isinstance(field_value, int):
        return field_value
    m = re.search(r'(\d+)', str(field_value))
    return int(m.group(1)) if m else None

def _http_get_json(url: str, params: Dict) -> Dict:
    try:
        full = f"{url}?{urlencode(params)}" if params else url
        with urlopen(full, timeout=5) as resp:
            payload = resp.read().decode('utf-8', errors='ignore')
            return json.loads(payload)
    except Exception:
        return {}

def ts_update_via_http(write_api_key: str, field_name: str, value, timeout_sec: float = 5.0) -> str:
    fnum = _field_number(field_name)
    if fnum is None:
        return "0"
    params = {"api_key": write_api_key, f"field{fnum}": value}
    try:
        full = "https://api.thingspeak.com/update?" + urlencode(params)
        with urlopen(full, timeout=timeout_sec) as resp:
            return resp.read().decode("utf-8", errors="ignore").strip()
    except Exception:
        return "0"

def _now_ts() -> float:
    return time.time()

def _ensure_rate_limit_and_maybe_wait(channel_id: int, min_interval: float = 16.0, max_wait: float = 8.0) -> Tuple[bool, float]:
    last_map: Dict[int, float] = st.session_state.get('_ts_last_update_at', {})
    last = safe_float(last_map.get(int(channel_id), 0.0))
    now = _now_ts()
    elapsed = now - last

    if elapsed >= min_interval:
        return True, 0.0

    remaining = max(0.0, min_interval - elapsed)
    if remaining <= max_wait:
        with st.spinner(f"Waiting {remaining:.1f}s for ThingSpeak cooldown..."):
            time.sleep(remaining + 0.3)
        return True, remaining
    else:
        return False, remaining

# ---------------------------------------------------------------------------------
# Optimistic queue: apply & process
# ---------------------------------------------------------------------------------
def _optimistic_apply_asset(*, ticker: str, new_value: float, prev_value: float, asset_conf: Dict, op_label: str = "SET") -> None:
    st.session_state.setdefault('_last_assets_overrides', {})[ticker] = float(new_value)
    st.session_state.setdefault('_pending_ts_update', []).append({
        'ticker': ticker,
        'channel_id': int(asset_conf['channel_id']),
        'field_name': asset_conf['field'],
        'write_key': asset_conf.get('write_api_key') or asset_conf.get('api_key'),
        'new_value': float(new_value),
        'prev_value': float(prev_value),
        'op': str(op_label),
        'queued_at': _now_ts(),
    })
    st.session_state['_cache_bump'] = st.session_state.get('_cache_bump', 0) + 1
    st.session_state["_pending_select_key"] = ticker
    st.session_state["_skip_refresh_on_rerun"] = True
    st.rerun()

def process_pending_updates(min_interval: float = 16.0, max_wait: float = 8.0) -> None:
    q = list(st.session_state.get('_pending_ts_update', []))
    if not q:
        return

    remaining = []
    for job in q:
        ticker = job.get('ticker')
        write_key = job.get('write_key')
        channel_id = int(job.get('channel_id', 0))
        new_val, prev_val = job.get('new_value'), job.get('prev_value')
        op, field_name = job.get('op', 'SET'), job.get('field_name')

        if not write_key:
            st.error(f"[{ticker}] No write_api_key found — rolling back.")
            st.session_state.setdefault('_last_assets_overrides', {})[ticker] = float(prev_val)
            continue

        allowed, remaining_sec = _ensure_rate_limit_and_maybe_wait(channel_id, min_interval=min_interval, max_wait=max_wait)
        if not allowed:
            st.info(f"[{ticker}] API call queued, waiting ~{remaining_sec:.1f}s for cooldown.")
            remaining.append(job)
            continue

        resp = ts_update_via_http(write_key, field_name, new_val, timeout_sec=5.0)
        # Retry once on failure
        if str(resp).strip() == "0":
            time.sleep(1.8)
            resp = ts_update_via_http(write_key, field_name, new_val, timeout_sec=5.0)

        if str(resp).strip() == "0":
            st.error(f"[{ticker}] {op} failed (response=0) — rolling back to {prev_val}")
            st.session_state.setdefault('_last_assets_overrides', {})[ticker] = float(prev_val)
        else:
            st.sidebar.success(f"[{ticker}] {op} successful (entry #{resp})")
            st.session_state.setdefault('_ts_entry_ids', {}).setdefault(ticker, []).append(resp)
            st.session_state.setdefault('_ts_last_update_at', {})[channel_id] = _now_ts()

    st.session_state['_pending_ts_update'] = remaining

# ---------------------------------------------------------------------------------
# Net stats
# ---------------------------------------------------------------------------------
EMPTY_STATS_RESULT = dict(buy_count=0, sell_count=0, net_count=0, buy_units=0.0, sell_units=0.0, net_units=0.0)

@st.cache_data(ttl=180, show_spinner=False)
def _fetch_and_parse_ts_feed(asset_field_conf: Dict, cache_bump: int) -> List[Tuple[datetime.datetime, Optional[str]]]:
    try:
        channel_id, fnum = int(asset_field_conf['channel_id']), _field_number(asset_field_conf['field'])
        if fnum is None: return []
        
        params = {'results': 8000}
        if asset_field_conf.get('api_key'):
            params['api_key'] = asset_field_conf.get('api_key')

        url = f"https://api.thingspeak.com/channels/{channel_id}/fields/{fnum}.json"
        data = _http_get_json(url, params)
        feeds = data.get('feeds', [])
        if not feeds: return []

        tz_bkk = pytz.timezone('Asia/Bangkok')
        rows: List[Tuple[datetime.datetime, Optional[str]]] = []
        for r in feeds:
            try:
                dt_utc = datetime.datetime.fromisoformat(str(r.get('created_at', '')).replace('Z', '+00:00'))
                v = r.get(f"field{fnum}")
                if v is not None:
                    rows.append((dt_utc.astimezone(tz_bkk), v))
            except Exception:
                continue
        rows.sort(key=lambda x: x[0])
        return rows
    except Exception:
        return []

def _calculate_detailed_stats(
    rows: List[Tuple[datetime.datetime, Optional[str]]],
    window_start_local: datetime.datetime
) -> Dict[str, float]:
    baseline: Optional[float] = None
    for dt_local, v in rows:
        if dt_local < window_start_local:
            baseline = safe_float(v, baseline)
        else:
            break

    buy_count = sell_count = 0
    buy_units = sell_units = 0.0
    last_in_window: Optional[float] = None

    prev: Optional[float] = baseline
    for dt_local, v_str in rows:
        if dt_local < window_start_local:
            prev = safe_float(v_str, prev)
            continue
        
        curr = safe_float(v_str)
        if prev is not None:
            step = curr - prev
            if step > 0:
                buy_count += 1
                buy_units += step
            elif step < 0:
                sell_count += 1
                sell_units += abs(step)
        prev = curr
        last_in_window = curr

    net_units = 0.0
    if last_in_window is not None:
        ref = baseline if baseline is not None else (safe_float(rows[0][1]) if rows else 0.0)
        net_units = float(last_in_window - ref)

    return dict(
        buy_count=int(buy_count), sell_count=int(sell_count),
        net_count=int(buy_count - sell_count),
        buy_units=float(buy_units), sell_units=float(sell_units),
        net_units=float(net_units)
    )

def _get_tz_aware_datetime(iso_str: str, tz: datetime.tzinfo) -> datetime.datetime:
    try:
        dt = datetime.datetime.fromisoformat(iso_str)
        return tz.localize(dt) if dt.tzinfo is None else dt.astimezone(tz)
    except Exception:
        return datetime.datetime.now(tz)

@st.cache_data(ttl=180, show_spinner=False)
def fetch_net_detailed_stats_since(asset_field_conf: Dict, window_start_bkk_iso: str, cache_bump: int = 0) -> Dict[str, float]:
    rows = _fetch_and_parse_ts_feed(asset_field_conf, cache_bump)
    if not rows: return EMPTY_STATS_RESULT.copy()
    
    start_dt = _get_tz_aware_datetime(window_start_bkk_iso, pytz.timezone('Asia/Bangkok'))
    return _calculate_detailed_stats(rows, start_dt)

# ---------------------------------------------------------------------------------
# Data Fetching
# ---------------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_all_data(configs: List[Dict], _clients_ref: Dict, start_date: Optional[str],
                   window_start_bkk_iso: str, cache_bump: int = 0) -> Dict[str, dict]:
    results = {'monitors': {}, 'assets': {}, 'nets': {}, 'trade_stats': {}}

    def fetch_monitor(asset_config: Dict):
        ticker = asset_config['ticker']
        try:
            client = _clients_ref[int(asset_config['monitor_field']['channel_id'])]
            field_num = asset_config['monitor_field']['field']

            tickerData = get_history_df_max_close_bkk(ticker)
            if start_date:
                tickerData = tickerData[tickerData.index >= start_date]

            last_data_date = tickerData.index[-1].date() if not tickerData.empty else None

            fx_js_str = "0"
            try:
                field_data = client.get_field_last(field=str(field_num))
                retrieved_val = json.loads(field_data).get(f"field{field_num}")
                if retrieved_val is not None: fx_js_str = str(retrieved_val)
            except Exception: pass

            df = pd.concat([tickerData, pd.DataFrame(index=['+' + str(i) for i in range(5)])], axis=0).fillna("")
            df['action'] = ""

            tracer = SimulationTracer(encoded_string=fx_js_str)
            final_actions = tracer.run()
            num_to_assign = min(len(df), len(final_actions))
            if num_to_assign > 0:
                action_col_idx = df.columns.get_loc('action')
                df.iloc[0:num_to_assign, action_col_idx] = final_actions[0:num_to_assign]
            
            results['monitors'][ticker] = (df.tail(7), fx_js_str, last_data_date)
        except Exception as e:
            st.error(f"Error in Monitor for {ticker}: {str(e)}")
            results['monitors'][ticker] = (pd.DataFrame(), "0", None)

    def fetch_asset_and_stats(asset_config: Dict):
        ticker = asset_config['ticker']
        asset_conf = asset_config['asset_field']
        # Asset
        try:
            client = _clients_ref[int(asset_conf['channel_id'])]
            field_name = asset_conf['field']
            data = client.get_field_last(field=field_name)
            results['assets'][ticker] = float(json.loads(data)[field_name])
        except Exception:
            results['assets'][ticker] = 0.0
        # Stats
        try:
            stats = fetch_net_detailed_stats_since(asset_conf, window_start_bkk_iso, cache_bump=cache_bump)
            results['trade_stats'][ticker] = stats
            results['nets'][ticker] = int(stats.get('net_count', 0))
        except Exception:
            results['trade_stats'][ticker] = EMPTY_STATS_RESULT.copy()
            results['nets'][ticker] = 0

    workers = max(1, min(len(configs) * 2, 10))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_monitor, a) for a in configs]
        futures += [executor.submit(fetch_asset_and_stats, a) for a in configs]
        concurrent.futures.wait(futures)

    return results

# ---------------------------------------------------------------------------------
# Optimistic Net String
# ---------------------------------------------------------------------------------
def get_pending_net_delta_for_ticker(ticker: str) -> int:
    q = st.session_state.get('_pending_ts_update', [])
    delta = 0
    for job in q:
        if job.get('ticker') != ticker: continue
        op = str(job.get('op', '')).upper()
        if op == 'BUY': delta += 1
        elif op == 'SELL': delta -= 1
        else: # Fallback for 'SET'
            nv, pv = safe_float(job.get('new_value')), safe_float(job.get('prev_value'))
            if nv > pv: delta += 1
            elif nv < pv: delta -= 1
    return int(delta)

def make_net_str_with_optimism(ticker: str, base_net: int) -> str:
    try:
        pend = get_pending_net_delta_for_ticker(ticker)
        if pend == 0: return str(int(base_net))
        sign = '+' if pend > 0 else ''
        preview = int(base_net) + int(pend)
        return f"{int(base_net)}&nbsp;&nbsp;→&nbsp;&nbsp;{preview}&nbsp;&nbsp;(⏳{sign}{int(pend)})"
    except Exception:
        return str(int(base_net))

# ---------------------------------------------------------------------------------
# UI Helpers
# ---------------------------------------------------------------------------------
def render_asset_inputs(configs: List[Dict], last_assets: Dict[str, float], net_since_open_map: Dict[str, int]) -> Dict[str, float]:
    asset_inputs: Dict[str, float] = {}
    cols = st.columns(len(configs)) if configs else [st]
    overrides = st.session_state.get('_last_assets_overrides', {})
    shadow: Dict[str, float] = st.session_state.setdefault('_widget_shadow', {})

    for i, config in enumerate(configs):
        with cols[i]:
            ticker = config['ticker']
            last_val = safe_float(last_assets.get(ticker, 0.0))
            opt = config.get('option_config', {}) # ✅ แก้ไข Attribute Error ที่นี่
            raw_label = opt.get('label', ticker)
            
            display_label, base_help = raw_label, ""
            if '(' in raw_label:
                split_pos = raw_label.find('(')
                display_label = raw_label[:split_pos].strip()
                base_help = raw_label[split_pos:].strip()

            help_text = base_help or f"net_since_us_premarket_open = {net_since_open_map.get(ticker, 0)}"
            is_option = 'base_value' in opt
            key_name = f"input_{ticker}_{'real' if is_option else 'asset'}"

            # Sync session_state with override or last_val
            if ticker in overrides:
                if key_name not in st.session_state or abs(shadow.get(key_name, float('nan')) - last_val) > 1e-9:
                    st.session_state[key_name] = last_val
                    shadow[key_name] = last_val
            elif key_name not in st.session_state:
                 st.session_state[key_name] = last_val
                 shadow[key_name] = last_val

            input_val = st.number_input(
                label=display_label, help=help_text, step=0.001,
                value=safe_float(st.session_state.get(key_name, last_val)), key=key_name,
            )
            
            if is_option:
                option_base = safe_float(opt.get('base_value', 0.0))
                delta_factor = safe_float(opt.get('delta_factor', 1.0))
                asset_inputs[ticker] = float(option_base * delta_factor + input_val)
            else:
                asset_inputs[ticker] = float(input_val)
                
    return asset_inputs

def render_asset_update_controls(configs: List[Dict], last_assets: Dict[str, float]) -> None:
    with st.expander("Update Assets on ThingSpeak"):
        for config in configs:
            ticker, asset_conf = config['ticker'], config['asset_field']
            if st.checkbox(f'@_{ticker}_ASSET', key=f'check_{ticker}'):
                current_val = safe_float(last_assets.get(ticker, 0.0))
                new_val = st.number_input(f"New Value for {ticker}", step=0.001, value=current_val, key=f'input_{ticker}')
                if st.button(f"GO_{ticker}", key=f'btn_{ticker}'):
                    if not (asset_conf.get('write_api_key') or asset_conf.get('api_key')):
                        st.error(f"[{ticker}] No write_api_key for writing.")
                    else:
                        _optimistic_apply_asset(
                            ticker=ticker, new_value=float(new_val), prev_value=float(current_val),
                            asset_conf=asset_conf, op_label="SET"
                        )

def trading_section(
    config: Dict, asset_val: float, asset_last: float,
    action_val: Optional[int], calc: Dict[str, Tuple[float, int, float]],
    diff: float, min_rebalance: float
) -> None:
    ticker, asset_conf = config['ticker'], config['asset_field']
    limit_order_checked = st.checkbox(f'Limit_Order_{ticker}', value=(action_val is not None), key=f'limit_order_{ticker}')
    if not limit_order_checked: return

    sell_calc, buy_calc = calc['sell'], calc['buy']

    def make_trade_html(label: str, color: str, trade_calc: Tuple[float, int, float]) -> str:
        """Helper to generate consistent HTML for buy/sell rows [SIMPLE/STABLE]"""
        a, p, c = trade_calc[1], trade_calc[0], trade_calc[2]
        return (f"<span style='color:white;'>{label}</span>&nbsp;&nbsp;"
                f"A:&nbsp;<span style='color:{color}; font-weight:600'>{a}</span>&nbsp;&nbsp;"
                f"P:&nbsp;<span style='color:{color}; font-weight:600'>{p}</span>&nbsp;&nbsp;"
                f"C:&nbsp;<span style='color:{color}; font-weight:600'>{c}</span>")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(make_trade_html('sell', '#fbb', buy_calc), unsafe_allow_html=True)
    with col2:
        if st.checkbox(f's_match', key=f'sell_match_{ticker}', label_visibility="collapsed"):
            if st.button(f"GO SELL", key=f"GO_SELL_{ticker}", use_container_width=True):
                _optimistic_apply_asset(
                    ticker=ticker, new_value=float(asset_last) - float(buy_calc[1]),
                    prev_value=float(asset_last), asset_conf=asset_conf, op_label="SELL"
                )

    try:
        current_price = get_cached_price(ticker)
        if current_price > 0:
            pv = current_price * float(asset_val)
            fix_value = float(config['fix_c'])
            pl_value = pv - fix_value
            pl_color = "#a8d5a2" if pl_value >= 0 else "#fbb"
            trade_only_when = float(fix_value) * float(min_rebalance)
            st.markdown(
                f"Price: **{current_price:,.3f}** | Value: **{pv:,.2f}** | P/L (vs {fix_value:,.0f}) | "
                f"Min ({trade_only_when:,.0f} vs {float(diff):,.0f}) | <span style='color:{pl_color}; font-weight:bold;'>{pl_value:,.2f}</span>",
                unsafe_allow_html=True
            )
        else:
            st.info(f"Price data for {ticker} is currently unavailable.")
    except Exception:
        st.warning(f"Could not retrieve price data for {ticker}.")

    col3, col4 = st.columns([3, 1])
    with col3:
        st.markdown(make_trade_html('buy', '#a8d5a2', sell_calc), unsafe_allow_html=True)
    with col4:
        if st.checkbox(f'b_match', key=f'buy_match_{ticker}', label_visibility="collapsed"):
            if st.button(f"GO BUY", key=f"GO_BUY_{ticker}", use_container_width=True):
                 _optimistic_apply_asset(
                    ticker=ticker, new_value=float(asset_last) + float(sell_calc[1]),
                    prev_value=float(asset_last), asset_conf=asset_conf, op_label="BUY"
                )

# ---------------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------------
# Session State Initialization
DEFAULTS = {
    'select_key': "", 'nex': 0, 'Nex_day_sell': 0, '_cache_bump': 0,
    '_last_assets_overrides': {}, '_skip_refresh_on_rerun': False,
    '_all_data_cache': None, '_ts_last_update_at': {},
    '_pending_ts_update': [], '_ts_entry_ids': {},
    '_widget_shadow': {}, 'min_rebalance': 0.04,
    'diff_value': ASSET_CONFIGS[0].get('diff', 60) if ASSET_CONFIGS else 60,
    '_last_selected_ticker': ""
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Bootstrap selection from pending actions (e.g., after button clicks)
if pending := st.session_state.pop("_pending_select_key", None):
    st.session_state.select_key = pending

# Time & Data Fetching
latest_us_premarket_open_bkk = get_latest_us_premarket_open_bkk()
window_start_bkk_iso = latest_us_premarket_open_bkk.isoformat()

CACHE_BUMP = st.session_state.get('_cache_bump', 0)
if st.session_state.get('_skip_refresh_on_rerun', False) and st.session_state.get('_all_data_cache'):
    all_data = st.session_state['_all_data_cache']
    st.session_state['_skip_refresh_on_rerun'] = False
else:
    with st.spinner("Fetching latest data..."):
        all_data = fetch_all_data(ASSET_CONFIGS, THINGSPEAK_CLIENTS, GLOBAL_START_DATE, window_start_bkk_iso, cache_bump=CACHE_BUMP)
    st.session_state['_all_data_cache'] = all_data

# Apply optimistic overrides to data
monitor_data_all = all_data['monitors']
last_assets_all = {**all_data['assets'], **st.session_state.get('_last_assets_overrides', {})}
trade_nets_all = all_data['nets']

# --- UI Tabs ---
tab1, tab2 = st.tabs(["📈 Monitor", "⚙️ Controls"])

with tab2:
    left, right = st.columns([2, 1])
    with left:
        Nex_day_ = st.checkbox('nex_day', value=(st.session_state.nex == 1))
        if Nex_day_:
            c1, c2, _ = st.columns([1, 1, 3])
            if c1.button("Nex_day"): st.session_state.update(nex=1, Nex_day_sell=0)
            if c2.button("Nex_day_sell"): st.session_state.update(nex=1, Nex_day_sell=1)
        else:
            st.session_state.update(nex=0, Nex_day_sell=0)

        if st.session_state.nex == 1:
            st.write(f"nex value = 1" + (f" | Nex_day_sell = 1" if st.session_state.Nex_day_sell else ""))

    with right:
        st.number_input('Min_Rebalance', min_value=0.0, max_value=1.0, step=0.01, key='min_rebalance',
                        help="ลิมิตโซนสำหรับคอมโพเนนต์ Min ในบรรทัด P/L (เช่น 0.04 = 4%)")
    st.divider()

    # Dynamic Diff Logic
    selected_ticker = st.session_state.get('select_key', "")
    if selected_ticker and selected_ticker != st.session_state.get('_last_selected_ticker'):
        matching_config = next((c for c in ASSET_CONFIGS if c['ticker'] == selected_ticker), None)
        if matching_config:
            st.session_state.diff_value = matching_config.get('diff', 60)
        st.session_state._last_selected_ticker = selected_ticker
    
    x_2_from_state = st.sidebar.number_input('Diff', step=1, key='diff_value')

    asset_inputs = render_asset_inputs(ASSET_CONFIGS, last_assets_all, trade_nets_all)
    st.divider()
    if st.checkbox('start'):
        render_asset_update_controls(ASSET_CONFIGS, last_assets_all)

with tab1:
    current_ny_date = get_current_ny_date()
    selectbox_labels: Dict[str, str] = {}
    ticker_actions: Dict[str, Optional[int]] = {}

    for config in ASSET_CONFIGS:
        ticker = config['ticker']
        df_data, fx_js_str, last_data_date = monitor_data_all.get(ticker, (pd.DataFrame(), "0", None))
        
        action_emoji, final_action_val = "", None
        is_stale = st.session_state.nex == 0 and last_data_date and last_data_date < current_ny_date
        
        if is_stale:
            action_emoji = "🟡 "
        else:
            try:
                action_value_str = df_data.action.values[1 + st.session_state.nex]
                if action_value_str != "":
                    raw_action = int(action_value_str) & 1
                    flip = int(st.session_state.Nex_day_sell) & 1
                    final_action_val = xor01(raw_action, flip)
                    action_emoji = "🟢 " if final_action_val == 1 else "🔴 "
            except (IndexError, ValueError): pass
        
        ticker_actions[ticker] = final_action_val
        base_net = int(trade_nets_all.get(ticker, 0))
        net_str = make_net_str_with_optimism(ticker, base_net)
        selectbox_labels[ticker] = f"{action_emoji}{ticker} (f(x): {fx_js_str})&nbsp;&nbsp;{net_str}"

    all_tickers = [c['ticker'] for c in ASSET_CONFIGS]
    selectbox_options = [""] + (["Filter Buy Tickers", "Filter Sell Tickers"] if st.session_state.nex == 1 else []) + all_tickers

    def format_selectbox_options(option_name: str) -> str:
        if option_name in ["", "Filter Buy Tickers", "Filter Sell Tickers"]:
            return "Show All" if option_name == "" else option_name
        return selectbox_labels.get(option_name, option_name).split(' (f(x):')[0]

    st.selectbox("Select Ticker to View:", options=selectbox_options, format_func=format_selectbox_options, key="select_key")
    st.divider()

    selected_option = st.session_state.select_key
    if selected_option == "":
        configs_to_display = ASSET_CONFIGS
    elif selected_option == "Filter Buy Tickers":
        configs_to_display = [c for c in ASSET_CONFIGS if ticker_actions.get(c['ticker']) == 1]
    elif selected_option == "Filter Sell Tickers":
        configs_to_display = [c for c in ASSET_CONFIGS if ticker_actions.get(c['ticker']) == 0]
    else:
        configs_to_display = [c for c in ASSET_CONFIGS if c['ticker'] == selected_option]

    calculations = {
        c['ticker']: {
            'sell': sell(float(asset_inputs.get(c['ticker'], 0.0)), float(c['fix_c']), float(x_2_from_state)),
            'buy': buy(float(asset_inputs.get(c['ticker'], 0.0)), float(c['fix_c']), float(x_2_from_state)),
        } for c in ASSET_CONFIGS
    }

    for config in configs_to_display:
        ticker = config['ticker']
        df_data, _, _ = monitor_data_all.get(ticker, (pd.DataFrame(), "0", None))

        st.markdown(selectbox_labels.get(ticker, ticker), unsafe_allow_html=True)
        trading_section(
            config=config,
            asset_val=float(asset_inputs.get(ticker, 0.0)),
            asset_last=float(last_assets_all.get(ticker, 0.0)),
            action_val=ticker_actions.get(ticker),
            calc=calculations.get(ticker, {}),
            diff=float(x_2_from_state),
            min_rebalance=float(st.session_state['min_rebalance'])
        )
        with st.expander("Show Raw Data Action"):
            st.dataframe(df_data, use_container_width=True)
        st.divider()

# --- Sidebar ---
if st.sidebar.button("RERUN"):
    current_selection = st.session_state.get("select_key", "")
    clear_all_caches()
    if current_selection in [c['ticker'] for c in ASSET_CONFIGS]:
        rerun_keep_selection(current_selection)
    else:
        st.rerun()

# --- Process Pending API Calls at the End ---
process_pending_updates(min_interval=16.0, max_wait=8.0)
