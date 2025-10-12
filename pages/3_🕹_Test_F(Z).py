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
import math ## <-- [EDIT] Goal 1: Import math for sqrt function

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
        'min_rebalance',
        'diff_value', '_last_selected_ticker' # <-- รักษา state ใหม่
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
# Net stats (ปรับปรุงใหม่)
# ---------------------------------------------------------------------------------
EMPTY_STATS_RESULT = dict(buy_count=0, sell_count=0, net_count=0, buy_units=0.0, sell_units=0.0, net_units=0.0)

@st.cache_data(ttl=180, show_spinner=False)
def _fetch_and_parse_ts_feed(asset_field_conf: Dict, cache_bump: int) -> List[Tuple[datetime.datetime, Optional[str]]]:
    """Helper: ดึงข้อมูลและแปลงเป็น list ของ (datetime, value) ที่จัดเรียงแล้ว"""
    try:
        channel_id = int(asset_field_conf['channel_id'])
        fnum = _field_number(asset_field_conf['field'])
        if fnum is None:
            return []
        field_key = f"field{fnum}"

        params = {'results': 8000}
        if asset_field_conf.get('api_key'):
            params['api_key'] = asset_field_conf.get('api_key')

        url = f"https://api.thingspeak.com/channels/{channel_id}/fields/{fnum}.json"
        data = _http_get_json(url, params)
        feeds = data.get('feeds', [])
        if not feeds:
            return []

        tz_bkk = pytz.timezone('Asia/Bangkok')

        def _parse_row(r):
            try:
                dt_utc = datetime.datetime.fromisoformat(str(r.get('created_at', '')
