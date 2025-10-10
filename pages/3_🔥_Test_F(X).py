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
# Small math helpers (0/1 logic)  [SIMPLE/STABLE]
# ---------------------------------------------------------------------------------
def xor01(a: int, b: int) -> int:
    """XOR สำหรับบิต 0/1 (เลี่ยง if-else)"""
    try:
        return (int(a) ^ int(b)) & 1
    except Exception:
        return 0

def safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
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
# ThingSpeak Clients (อ่านอย่างเดียว)
# ---------------------------------------------------------------------------------
@st.cache_resource
def get_thingspeak_clients(configs: List[Dict]) -> Dict[int, thingspeak.Channel]:
    clients: Dict[int, thingspeak.Channel] = {}
    unique_channels = set()
    for config in configs:
        mon_conf = config['monitor_field']
        unique_channels.add((mon_conf['channel_id'], mon_conf['api_key']))
        asset_conf = config['asset_field']
        unique_channels.add((asset_conf['channel_id'], asset_conf['api_key']))
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
        'min_rebalance', 'global_diff'
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
# Calc Utils (เดิม)
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
    """คืนราคาที่ดีที่สุดแบบมี fallback หลายชั้น  [SIMPLE/STABLE]"""
    try:
        tk = yf.Ticker(ticker)
        # ชั้น 1: fast_info
        try:
            p = float(tk.fast_info.get('lastPrice', 0.0))
            if p > 0:
                return p
        except Exception:
            pass
        # ชั้น 2: info (บางตัวใช้ regularMarketPrice)
        try:
            inf = getattr(tk, 'info', {}) or {}
            p = float(inf.get('regularMarketPrice', 0.0))
            if p > 0:
                return p
        except Exception:
            pass
        # ชั้น 3: ปิดล่าสุด
        try:
            df = tk.history(period='5d')
            if isinstance(df, pd.DataFrame) and not df.empty and 'Close' in df:
                p = float(df['Close'].iloc[-1])
                if p > 0:
                    return p
        except Exception:
            pass
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
    if wd == 0:
        return d - datetime.timedelta(days=3)
    elif wd == 6:
        return d - datetime.timedelta(days=2)
    else:
        return d - datetime.timedelta(days=1)

@st.cache_data(ttl=600, show_spinner=False)
def get_latest_us_premarket_open_bkk() -> datetime.datetime:
    tz_ny = pytz.timezone('America/New_York')
    tz_bkk = pytz.timezone('Asia/Bangkok')
    now_ny = datetime.datetime.now(tz_ny)
    date_ny = now_ny.date()

    def make_open(dt_date: datetime.date) -> datetime.datetime:
        dt_naive = datetime.datetime(dt_date.year, dt_date.month, dt_date.day, 4, 0, 0)
        return tz_ny.localize(dt_naive)

    candidate = make_open(date_ny)
    while candidate.weekday() >= 5:
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
    """อัปเดต ThingSpeak ผ่าน HTTP GET; คืนค่า entry_id (string) หรือ '0' ถ้าล้มเหลว"""
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

# RATE-LIMIT helpers
def _now_ts() -> float:
    return time.time()

def _ensure_rate_limit_and_maybe_wait(channel_id: int, min_interval: float = 16.0, max_wait: float = 8.0) -> Tuple[bool, float]:
    """
    ตรวจคูลดาวน์ต่อช่อง:
    - ถ้าเหลือเวลาน้อยกว่าหรือเท่ากับ max_wait → รอให้ครบและอนุญาตอัปเดต
    - ถ้าเหลือเวลามากกว่า max_wait → ไม่อนุญาต (คิวไว้รอบถัดไป)
    คืนค่า (allowed, remaining_seconds)
    """
    try:
        last_map: Dict[int, float] = st.session_state.get('_ts_last_update_at', {})
        last = float(last_map.get(int(channel_id), 0.0))
    except Exception:
        last = 0.0

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
# ✅ Optimistic queue: apply & process
# ---------------------------------------------------------------------------------
def _optimistic_apply_asset(*, ticker: str, new_value: float, prev_value: float, asset_conf: Dict, op_label: str = "SET") -> None:
    """เฟสที่ 1: อัปเดต UI ทันที + เข้าคิว API"""
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
    """เฟสที่ 2: ประมวลผลคิว → ยิง API; สำเร็จ=คง override, ล้มเหลว=rollback"""
    q = list(st.session_state.get('_pending_ts_update', []))
    if not q:
        return

    remaining = []
    for job in q:
        ticker = job.get('ticker')
        field_name = job.get('field_name')
        write_key = job.get('write_key')
        channel_id = int(job.get('channel_id', 0))
        new_val = job.get('new_value')
        prev_val = job.get('prev_value')
        op = job.get('op', 'SET')

        if not write_key:
            st.error(f"[{ticker}] ไม่มี write_api_key/api_key สำหรับเขียน — rollback แล้ว")
            st.session_state.setdefault('_last_assets_overrides', {})[ticker] = float(prev_val)
            continue

        allowed, remaining_sec = _ensure_rate_limit_and_maybe_wait(channel_id, min_interval=min_interval, max_wait=max_wait)
        if not allowed:
            st.info(f"[{ticker}] ต้องรอ ~{remaining_sec:.1f}s ก่อนยิง API (ThingSpeak ~15s/ช่อง) → จะลองใหม่อัตโนมัติ")
            remaining.append(job)
            continue

        resp = ts_update_via_http(write_key, field_name, new_val, timeout_sec=5.0)
        if str(resp).strip() == "0":
            time.sleep(1.8)
            resp = ts_update_via_http(write_key, field_name, new_val, timeout_sec=5.0)

        if str(resp).strip() == "0":
            st.error(f"[{ticker}] {op} ล้มเหลว (resp=0) — rollback เป็น {prev_val}")
            st.session_state.setdefault('_last_assets_overrides', {})[ticker] = float(prev_val)
        else:
            st.sidebar.success(f"[{ticker}] {op} สำเร็จ (entry #{resp})")
            st.session_state.setdefault('_ts_entry_ids', {}).setdefault(ticker, []).append(resp)
            st.session_state.setdefault('_ts_last_update_at', {})[channel_id] = _now_ts()

    st.session_state['_pending_ts_update'] = remaining

# ---------------------------------------------------------------------------------
# Net stats (เดิม)
# ---------------------------------------------------------------------------------
@st.cache_data(ttl=180, show_spinner=False)
def fetch_net_trades_since(asset_field_conf: Dict, window_start_bkk_iso: str, cache_bump: int = 0) -> int:
    try:
        channel_id = int(asset_field_conf['channel_id'])
        fnum = _field_number(asset_field_conf['field'])
        if fnum is None:
            return 0
        field_key = f"field{fnum}"

        params = {'results': 8000}
        if asset_field_conf.get('api_key'):
            params['api_key'] = asset_field_conf.get('api_key')

        url = f"https://api.thingspeak.com/channels/{channel_id}/fields/{fnum}.json"
        data = _http_get_json(url, params)
        feeds = data.get('feeds', [])
        if not feeds:
            return 0

        tz = pytz.timezone('Asia/Bangkok')
        try:
            window_start_local = datetime.datetime.fromisoformat(window_start_bkk_iso)
            if window_start_local.tzinfo is None:
                window_start_local = tz.localize(window_start_local)
            else:
                window_start_local = window_start_local.astimezone(tz)
        except Exception:
            window_start_local = datetime.datetime.now(tz)

        def _parse_row(r):
            try:
                dt_utc = datetime.datetime.fromisoformat(str(r.get('created_at', '')).replace('Z', '+00:00'))
            except Exception:
                return None, None
            return dt_utc.astimezone(tz), r.get(field_key)

        rows: List[Tuple[datetime.datetime, Optional[str]]] = []
        for r in feeds:
            dt_local, v = _parse_row(r)
            if dt_local is not None and v is not None:
                rows.append((dt_local, v))
        rows.sort(key=lambda x: x[0])

        prev_val: Optional[float] = None
        for dt_local, v in rows:
            if dt_local < window_start_local:
                try:
                    prev_val = float(v)
                except Exception:
                    continue
            else:
                break

        buys, sells = 0, 0
        for dt_local, v in rows:
            if dt_local < window_start_local:
                continue
            try:
                curr = float(v)
            except Exception:
                continue
            if prev_val is not None:
                if curr > prev_val:
                    buys += 1
                elif curr < prev_val:
                    sells += 1
            prev_val = curr

        return int(buys - sells)
    except Exception:
        return 0

@st.cache_data(ttl=180, show_spinner=False)
def fetch_net_detailed_stats_since(asset_field_conf: Dict, window_start_bkk_iso: str, cache_bump: int = 0) -> Dict[str, float]:
    try:
        channel_id = int(asset_field_conf['channel_id'])
        fnum = _field_number(asset_field_conf['field'])
        if fnum is None:
            return dict(buy_count=0, sell_count=0, net_count=0, buy_units=0.0, sell_units=0.0, net_units=0.0)
        field_key = f"field{fnum}"

        params = {'results': 8000}
        if asset_field_conf.get('api_key'):
            params['api_key'] = asset_field_conf.get('api_key')

        url = f"https://api.thingspeak.com/channels/{channel_id}/fields/{fnum}.json"
        data = _http_get_json(url, params)
        feeds = data.get('feeds', [])
        if not feeds:
            return dict(buy_count=0, sell_count=0, net_count=0, buy_units=0.0, sell_units=0.0, net_units=0.0)

        tz = pytz.timezone('Asia/Bangkok')
        try:
            window_start_local = datetime.datetime.fromisoformat(window_start_bkk_iso)
            if window_start_local.tzinfo is None:
                window_start_local = tz.localize(window_start_local)
            else:
                window_start_local = window_start_local.astimezone(tz)
        except Exception:
            window_start_local = datetime.datetime.now(tz)

        def _parse_row(r):
            try:
                dt_utc = datetime.datetime.fromisoformat(str(r.get('created_at', '')).replace('Z', '+00:00'))
            except Exception:
                return None, None
            return dt_utc.astimezone(tz), r.get(field_key)

        rows: List[Tuple[datetime.datetime, Optional[str]]] = []
        append = rows.append
        for r in feeds:
            dt_local, v = _parse_row(r)
            if dt_local is not None and v is not None:
                append((dt_local, v))
        rows.sort(key=lambda x: x[0])

        baseline: Optional[float] = None
        for dt_local, v in rows:
            if dt_local < window_start_local:
                try:
                    baseline = float(v)
                except Exception:
                    continue
            else:
                break

        buy_count = sell_count = 0
        buy_units = sell_units = 0.0
        first_after: Optional[float] = None
        last_after: Optional[float] = None

        prev: Optional[float] = baseline
        for dt_local, v in rows:
            try:
                curr = float(v)
            except Exception:
                continue

            if dt_local < window_start_local:
                prev = curr
                continue

            if first_after is None:
                first_after = curr
            if prev is not None:
                step = curr - prev
                if step > 0:
                    buy_count += 1
                    buy_units += step
                elif step < 0:
                    sell_count += 1
                    sell_units += (-step)
            prev = curr
            last_after = curr

        if last_after is None:
            net_units = 0.0
        else:
            ref = baseline if baseline is not None else (first_after if first_after is not None else last_after)
            net_units = float(last_after - ref)

        return dict(
            buy_count=int(buy_count),
            sell_count=int(sell_count),
            net_count=int(buy_count - sell_count),
            buy_units=float(buy_units),
            sell_units=float(sell_units),
            net_units=float(net_units)
        )
    except Exception:
        return dict(buy_count=0, sell_count=0, net_count=0, buy_units=0.0, sell_units=0.0, net_units=0.0)

@st.cache_data(ttl=180, show_spinner=False)
def fetch_net_detailed_stats_between(asset_field_conf: Dict, window_start_bkk_iso: str, window_end_bkk_iso: str, cache_bump: int = 0) -> Dict[str, float]:
    try:
        channel_id = int(asset_field_conf['channel_id'])
        fnum = _field_number(asset_field_conf['field'])
        if fnum is None:
            return dict(buy_count=0, sell_count=0, net_count=0, buy_units=0.0, sell_units=0.0, net_units=0.0)
        field_key = f"field{fnum}"

        params = {'results': 8000}
        if asset_field_conf.get('api_key'):
            params['api_key'] = asset_field_conf.get('api_key')

        url = f"https://api.thingspeak.com/channels/{channel_id}/fields/{fnum}.json"
        data = _http_get_json(url, params)
        feeds = data.get('feeds', [])
        if not feeds:
            return dict(buy_count=0, sell_count=0, net_count=0, buy_units=0.0, sell_units=0.0, net_units=0.0)

        tz = pytz.timezone('Asia/Bangkok')
        try:
            window_start_local = datetime.datetime.fromisoformat(window_start_bkk_iso)
            if window_start_local.tzinfo is None:
                window_start_local = tz.localize(window_start_local)
            else:
                window_start_local = window_start_local.astimezone(tz)
        except Exception:
            window_start_local = datetime.datetime.now(tz)

        try:
            window_end_local = datetime.datetime.fromisoformat(window_end_bkk_iso)
            if window_end_local.tzinfo is None:
                window_end_local = tz.localize(window_end_local)
            else:
                window_end_local = window_end_local.astimezone(tz)
        except Exception:
            window_end_local = datetime.datetime.now(tz)

        def _parse_row(r):
            try:
                dt_utc = datetime.datetime.fromisoformat(str(r.get('created_at', '')).replace('Z', '+00:00'))
            except Exception:
                return None, None
            return dt_utc.astimezone(tz), r.get(field_key)

        rows: List[Tuple[datetime.datetime, Optional[str]]] = []
        append = rows.append
        for r in feeds:
            dt_local, v = _parse_row(r)
            if dt_local is not None and v is not None:
                append((dt_local, v))
        rows.sort(key=lambda x: x[0])

        baseline: Optional[float] = None
        for dt_local, v in rows:
            if dt_local < window_start_local:
                try:
                    baseline = float(v)
                except Exception:
                    continue
            else:
                break

        buy_count = sell_count = 0
        buy_units = sell_units = 0.0
        first_after: Optional[float] = None
        last_within: Optional[float] = None

        prev: Optional[float] = baseline
        for dt_local, v in rows:
            try:
                curr = float(v)
            except Exception:
                continue

            if dt_local < window_start_local:
                prev = curr
                continue
            if dt_local > window_end_local:
                break

            if first_after is None:
                first_after = curr
            if prev is not None:
                step = curr - prev
                if step > 0:
                    buy_count += 1
                    buy_units += step
                elif step < 0:
                    sell_count += 1
                    sell_units += (-step)
            prev = curr
            last_within = curr

        if last_within is None:
            net_units = 0.0
        else:
            ref = baseline if baseline is not None else (first_after if first_after is not None else last_within)
            net_units = float(last_within - ref)

        return dict(
            buy_count=int(buy_count),
            sell_count=int(sell_count),
            net_count=int(buy_count - sell_count),
            buy_units=float(buy_units),
            sell_units=float(sell_units),
            net_units=float(net_units)
        )
    except Exception:
        return dict(buy_count=0, sell_count=0, net_count=0, buy_units=0.0, sell_units=0.0, net_units=0.0)

# ---------------------------------------------------------------------------------
# Fetch all data — รองรับ fast rerun
# ---------------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_all_data(configs: List[Dict], _clients_ref: Dict, start_date: Optional[str],
                   window_start_bkk_iso: str, cache_bump: int = 0) -> Dict[str, dict]:
    monitor_results: Dict[str, Tuple[pd.DataFrame, str, Optional[datetime.date]]] = {}
    asset_results: Dict[str, float] = {}
    nets_results: Dict[str, int] = {}
    trade_stats_results: Dict[str, Dict[str, float]] = {}

    def fetch_monitor(asset_config: Dict) -> Tuple[str, Tuple[pd.DataFrame, str, Optional[datetime.date]]]:
        ticker = asset_config['ticker']
        try:
            monitor_field_config = asset_config['monitor_field']
            client = _clients_ref[int(monitor_field_config['channel_id'])]
            field_num = monitor_field_config['field']

            tickerData = get_history_df_max_close_bkk(ticker)
            if start_date:
                tickerData = tickerData[tickerData.index >= start_date]

            last_data_date: Optional[datetime.date] = tickerData.index[-1].date() if not tickerData.empty else None

            fx_js_str = "0"
            try:
                field_data = client.get_field_last(field=str(field_num))
                retrieved_val = json.loads(field_data)[f"field{field_num}"]
                if retrieved_val is not None:
                    fx_js_str = str(retrieved_val)
            except Exception:
                pass

            tickerData = tickerData.copy()
            tickerData['index'] = list(range(len(tickerData)))

            dummy_df = pd.DataFrame(index=['+' + str(i) for i in range(5)])
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

            return ticker, (df.tail(7), fx_js_str, last_data_date)
        except Exception as e:
            st.error(f"Error in Monitor for {ticker}: {str(e)}")
            return ticker, (pd.DataFrame(), "0", None)

    def fetch_asset(asset_config: Dict) -> Tuple[str, float]:
        ticker = asset_config['ticker']
        try:
            asset_conf = asset_config['asset_field']
            client = _clients_ref[int(asset_conf['channel_id'])]
            field_name = asset_conf['field']
            data = client.get_field_last(field=field_name)
            return ticker, float(json.loads(data)[field_name])
        except Exception:
            return ticker, 0.0

    def fetch_trade_stats(asset_config: Dict) -> Tuple[str, Dict[str, float]]:
        ticker = asset_config['ticker']
        try:
            stats = fetch_net_detailed_stats_since(asset_config['asset_field'], window_start_bkk_iso, cache_bump=cache_bump)
            return ticker, stats
        except Exception:
            return ticker, dict(buy_count=0, sell_count=0, net_count=0, buy_units=0.0, sell_units=0.0, net_units=0.0)

    workers = max(1, min(len(configs), 8))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for future in concurrent.futures.as_completed([executor.submit(fetch_monitor, a) for a in configs]):
            ticker, result = future.result()
            monitor_results[ticker] = result

        for future in concurrent.futures.as_completed([executor.submit(fetch_asset, a) for a in configs]):
            ticker, result = future.result()
            asset_results[ticker] = result

        for future in concurrent.futures.as_completed([executor.submit(fetch_trade_stats, a) for a in configs]):
            ticker, stats = future.result()
            trade_stats_results[ticker] = stats
            nets_results[ticker] = int(stats.get('net_count', 0))

    return {'monitors': monitor_results, 'assets': asset_results, 'nets': nets_results, 'trade_stats': trade_stats_results}

# ---------------------------------------------------------------------------------
# [OPT-NET] — Pending delta → optimistic net_str
# ---------------------------------------------------------------------------------
def get_pending_net_delta_for_ticker(ticker: str) -> int:
    """รวม delta net จากคิวงาน BUY/SELL ที่ยัง pending สำหรับ ticker นั้น ๆ"""
    q = st.session_state.get('_pending_ts_update', [])
    delta = 0
    for job in q:
        if job.get('ticker') != ticker:
            continue
        op = str(job.get('op', '')).upper()
        if op == 'BUY':
            delta += 1
        elif op == 'SELL':
            delta -= 1
        else:
            # เผื่อกรณีไม่มี op ให้เทียบค่า
            try:
                nv = float(job.get('new_value', 0.0))
                pv = float(job.get('prev_value', 0.0))
                if nv > pv:
                    delta += 1
                elif nv < pv:
                    delta -= 1
            except Exception:
                pass
    return int(delta)

def make_net_str_with_optimism(ticker: str, base_net: int) -> str:
    """คืนสตริง net พร้อมแสดงผล optimistic (ถ้ามี) เช่น '2  →  3  (⏳+1)' """
    try:
        pend = get_pending_net_delta_for_ticker(ticker)
        if pend == 0:
            return str(int(base_net))
        sign = '+' if pend > 0 else ''
        preview = int(base_net) + int(pend)
        return f
