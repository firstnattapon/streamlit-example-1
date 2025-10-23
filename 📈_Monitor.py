# 📈_Monitor.py — Pro Optimistic UI (2-phase queue) + Min_Rebalance (clean UI)
# ========================= SIMPLE–STABLE REFACTOR (FAST) =======================

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
import math  # ใช้ sqrt ฯลฯ

# ---------------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------------
st.set_page_config(page_title="Monitor", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------------
# Globals (ลดงานจุกจิก/เร่งความเร็ว)
# ---------------------------------------------------------------------------------
_EPS = 1e-12
TZ_BKK = pytz.timezone('Asia/Bangkok')
TZ_NY = pytz.timezone('America/New_York')
_FIELD_NUM_RE = re.compile(r'(\d+)')

# ---------------------------------------------------------------------------------
# Math utils [SIMPLE/STABLE]
# ---------------------------------------------------------------------------------
def r2(x: float) -> float:
    """Stable 2-dec rounding (เหมือนเดิมเชิงผลลัพธ์)"""
    try:
        return round(float(x), 2)
    except Exception:
        return 0.0

def heaviside(x: float) -> int:
    """H(x): 1 หาก x>0, 0 หาก x<=0 (no branch side-effect)"""
    return 1 if x > 0 else 0

def sgn(x: float) -> int:
    """sign function -> {-1,0,1}"""
    return (x > 0) - (x < 0)

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
# SimulationTracer (คงเดิม)
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

ALL_TICKERS: List[str] = [c['ticker'] for c in ASSET_CONFIGS]

# ---------------------------------------------------------------------------------
# ThingSpeak Clients
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
# TRADE CORE [SIMPLE/STABLE]
# ---------------------------------------------------------------------------------
@lru_cache(maxsize=1024)
def _trade_math(asset: float, fix_c: float = 1500.0, Diff: float = 60.0, side: int = +1) -> Tuple[float, int, float]:
    """
    ฟิสิกส์เดียวกันสำหรับ buy/sell:
      unit_price = round2( (fix_c + side*Diff) / max(|asset|, ε) )
      Δqty       = round( |asset*unit_price - fix_c| / max(unit_price, ε) )
      total      = round2( asset*unit_price - side*Δqty*unit_price )

    side = +1 => BUY (เหมือนเดิม)
    side = -1 => SELL (เหมือนเดิม)
    """
    a = float(asset)
    if abs(a) <= _EPS:
        return 0.0, 0, 0.0

    up = r2( (float(fix_c) + float(side)*float(Diff)) / max(abs(a), _EPS) )
    if abs(up) <= _EPS:
        return 0.0, 0, 0.0

    delta_qty = int( round( abs(a*up - float(fix_c)) / max(abs(up), _EPS) ) )
    total = r2( a*up - float(side)*delta_qty*up )
    return float(up), int(delta_qty), float(total)

@lru_cache(maxsize=128)
def sell(asset: float, fix_c: float = 1500, Diff: float = 60) -> Tuple[float, int, float]:
    if asset == 0:
        return 0.0, 0, 0.0
    return _trade_math(asset, fix_c, Diff, side=-1)

@lru_cache(maxsize=128)
def buy(asset: float, fix_c: float = 1500, Diff: float = 60) -> Tuple[float, int, float]:
    if asset == 0:
        return 0.0, 0, 0.0
    return _trade_math(asset, fix_c, Diff, side=+1)

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
        # ชั้น 2: info
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

@st.cache_data(ttl=120, show_spinner=False)
def get_prices_map(tickers: List[str]) -> Dict[str, float]:
    """Prefetch ราคาทั้งชุดแบบขนาน เพื่อเร่งตอนแสดงหลายตัวพร้อมกัน"""
    tickers = list(dict.fromkeys([t for t in tickers if isinstance(t, str) and t]))  # unique & clean
    out: Dict[str, float] = {}
    if not tickers:
        return out
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(tickers))) as ex:
        futs = {ex.submit(get_cached_price, t): t for t in tickers}
        for f in concurrent.futures.as_completed(futs):
            t = futs[f]
            try:
                out[t] = float(f.result() or 0.0)
            except Exception:
                out[t] = 0.0
    return out

@st.cache_data(ttl=3600, show_spinner=False)
def get_history_df_max_close_bkk(ticker: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period='max')[['Close']].round(3)
    try:
        df.index = df.index.tz_convert(TZ_BKK)
    except TypeError:
        df.index = df.index.tz_localize('UTC').tz_convert(TZ_BKK)
    return df

@st.cache_data(ttl=60, show_spinner=False)
def get_current_ny_date() -> datetime.date:
    return datetime.datetime.now(TZ_NY).date()

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
    now_ny = datetime.datetime.now(TZ_NY)
    date_ny = now_ny.date()

    def make_open(dt_date: datetime.date) -> datetime.datetime:
        dt_naive = datetime.datetime(dt_date.year, dt_date.month, dt_date.day, 4, 0, 0)
        return TZ_NY.localize(dt_naive)

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

    return candidate.astimezone(TZ_BKK)

# ---------------------------------------------------------------------------------
# ThingSpeak helpers
# ---------------------------------------------------------------------------------
def _field_number(field_value) -> Optional[int]:
    if isinstance(field_value, int):
        return field_value
    m = _FIELD_NUM_RE.search(str(field_value))
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
# Net stats — เวกเตอร์ไรซ์ [SIMPLE/STABLE] + WINDOW-AWARE THINGSPEAK FETCH
# ---------------------------------------------------------------------------------
EMPTY_STATS_RESULT = dict(buy_count=0, sell_count=0, net_count=0, buy_units=0.0, sell_units=0.0, net_units=0.0)

@st.cache_data(ttl=180, show_spinner=False)
def _fetch_and_parse_ts_feed(asset_field_conf: Dict, cache_bump: int, window_start_bkk_iso: Optional[str] = None) -> List[Tuple[datetime.datetime, Optional[str]]]:
    """
    ดึงฟีดแบบ 'รู้หน้าต่างเวลา':
      - ถ้าให้ window_start_bkk_iso → ขอด้วย 'start' เป็น UTC (บัฟเฟอร์ 36 ชม.) ลด payload มหาศาล
      - ถ้าล้มเหลว/ได้น้อยเกิน → fallback mode (results=8000) เพื่อคงผลลัพธ์เดิม
    """
    try:
        channel_id = int(asset_field_conf['channel_id'])
        fnum = _field_number(asset_field_conf['field'])
        if fnum is None:
            return []
        field_key = f"field{fnum}"

        # --- primary: windowed fetch ---
        feeds: List[Dict] = []
        if window_start_bkk_iso:
            try:
                start_local = datetime.datetime.fromisoformat(window_start_bkk_iso)
                if start_local.tzinfo is None:
                    start_local = TZ_BKK.localize(start_local)
                # บัฟเฟอร์ถอยหลัง 36 ชั่วโมง เพื่อให้มี baseline ก่อนเริ่ม window
                start_with_buffer_utc = (start_local - datetime.timedelta(hours=36)).astimezone(pytz.UTC)
                start_param = start_with_buffer_utc.strftime('%Y-%m-%dT%H:%M:%SZ')

                params = {'start': start_param}
                if asset_field_conf.get('api_key'):
                    params['api_key'] = asset_field_conf.get('api_key')

                url = f"https://api.thingspeak.com/channels/{channel_id}/fields/{fnum}.json"
                data = _http_get_json(url, params)
                feeds = data.get('feeds', []) or []
            except Exception:
                feeds = []

        # --- fallback: wide fetch ---
        if not feeds:
            params = {'results': 8000}
            if asset_field_conf.get('api_key'):
                params['api_key'] = asset_field_conf.get('api_key')
            url = f"https://api.thingspeak.com/channels/{channel_id}/fields/{fnum}.json"
            data = _http_get_json(url, params)
            feeds = data.get('feeds', []) or []

        if not feeds:
            return []

        def _parse_row(r):
            try:
                dt_utc = datetime.datetime.fromisoformat(str(r.get('created_at', '')).replace('Z', '+00:00'))
                return dt_utc.astimezone(TZ_BKK), r.get(field_key)
            except Exception:
                return None, None

        rows: List[Tuple[datetime.datetime, Optional[str]]] = []
        for r in feeds:
            dt_local, v = _parse_row(r)
            if dt_local is not None and v is not None:
                rows.append((dt_local, v))
        rows.sort(key=lambda x: x[0])
        return rows
    except Exception:
        return []

def _calc_stats_vectorized(
    rows: List[Tuple[datetime.datetime, Optional[str]]],
    window_start_local: datetime.datetime,
    window_end_local: Optional[datetime.datetime] = None
) -> Dict[str, float]:
    if not rows:
        return EMPTY_STATS_RESULT.copy()

    t = np.array([r[0] for r in rows], dtype='datetime64[ns]')
    v = np.array([safe_float(r[1], np.nan) for r in rows], dtype=float)
    mask_valid = ~np.isnan(v)
    if not mask_valid.any():
        return EMPTY_STATS_RESULT.copy()

    t = t[mask_valid]
    v = v[mask_valid]

    # baseline = ค่าสุดท้ายก่อนเริ่มหน้าต่าง
    before_mask = (t < np.datetime64(window_start_local))
    baseline = None
    if before_mask.any():
        baseline = float(v[before_mask][-1])

    # หน้าต่าง
    inside_mask = (t >= np.datetime64(window_start_local))
    if window_end_local is not None:
        inside_mask &= (t <= np.datetime64(window_end_local))

    v_inside = v[inside_mask]
    if v_inside.size == 0:
        return EMPTY_STATS_RESULT.copy()

    # เริ่มจากค่า ref_prev = baseline หรือ ค่าตัวแรกในหน้าต่าง
    ref_prev = baseline if baseline is not None else float(v_inside[0])

    # sequence สำหรับ diff = [ref_prev, v_inside...]
    seq = np.concatenate([[ref_prev], v_inside.astype(float)])
    steps = np.diff(seq)

    buy_steps = steps[steps > 0]
    sell_steps = steps[steps < 0]
    buy_units = float(buy_steps.sum()) if buy_steps.size else 0.0
    sell_units = float((-sell_steps).sum()) if sell_steps.size else 0.0

    buy_count = int(np.count_nonzero(steps > 0))
    sell_count = int(np.count_nonzero(steps < 0))
    last_in_window = float(seq[-1])
    ref_for_net = baseline if baseline is not None else float(v_inside[0])
    net_units = float(last_in_window - ref_for_net)

    return dict(
        buy_count=buy_count,
        sell_count=sell_count,
        net_count=int(buy_count - sell_count),
        buy_units=float(buy_units),
        sell_units=float(sell_units),
        net_units=float(net_units)
    )

@st.cache_data(ttl=180, show_spinner=False)
def fetch_net_detailed_stats_since(asset_field_conf: Dict, window_start_bkk_iso: str, cache_bump: int = 0) -> Dict[str, float]:
    rows = _fetch_and_parse_ts_feed(asset_field_conf, cache_bump, window_start_bkk_iso=window_start_bkk_iso)
    if not rows:
        return EMPTY_STATS_RESULT.copy()
    start_dt = _get_tz_aware_datetime(window_start_bkk_iso, TZ_BKK)
    return _calc_stats_vectorized(rows, start_dt, None)

@st.cache_data(ttl=180, show_spinner=False)
def fetch_net_detailed_stats_between(asset_field_conf: Dict, window_start_bkk_iso: str, window_end_bkk_iso: str, cache_bump: int = 0) -> Dict[str, float]:
    rows = _fetch_and_parse_ts_feed(asset_field_conf, cache_bump, window_start_bkk_iso=window_start_bkk_iso)
    if not rows:
        return EMPTY_STATS_RESULT.copy()
    start_dt = _get_tz_aware_datetime(window_start_bkk_iso, TZ_BKK)
    end_dt = _get_tz_aware_datetime(window_end_bkk_iso, TZ_BKK)
    return _calc_stats_vectorized(rows, start_dt, end_dt)

def _get_tz_aware_datetime(iso_str: str, tz: datetime.tzinfo) -> datetime.datetime:
    """แปลง ISO string เป็น datetime ที่มี timezone ถูกต้อง"""
    try:
        dt = datetime.datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            return tz.localize(dt)
        return dt.astimezone(tz)
    except Exception:
        return datetime.datetime.now(tz)

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
            return ticker, EMPTY_STATS_RESULT.copy()

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
    try:
        pend = get_pending_net_delta_for_ticker(ticker)
        if pend == 0:
            return str(int(base_net))
        sign = '+' if pend > 0 else ''
        preview = int(base_net) + int(pend)
        return f"{int(base_net)}  →  {preview}  ({sign}{int(pend)})"
    except Exception:
        return str(int(base_net))

# ---------------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------------
def render_asset_inputs(configs: List[Dict], last_assets: Dict[str, float], net_since_open_map: Dict[str, int]) -> Dict[str, float]:
    asset_inputs: Dict[str, float] = {}
    cols = st.columns(len(configs)) if configs else [st]

    overrides = st.session_state.get('_last_assets_overrides', {})
    shadow: Dict[str, float] = st.session_state.setdefault('_widget_shadow', {})

    for i, config in enumerate(configs):
        with cols[i]:
            ticker = config['ticker']
            last_val = safe_float(last_assets.get(ticker, 0.0), 0.0)

            opt = config.get('option_config') or {}
            raw_label = opt.get('label', ticker)
            base_help = ""
            display_label = raw_label
            split_pos = raw_label.find('(')
            if split_pos != -1:
                display_label = raw_label[:split_pos].strip()
                base_help = raw_label[split_pos:].strip()

            delta_factor = safe_float(opt.get('delta_factor', 1.0), 1.0)
            help_text_final = base_help if base_help else f"net_since_us_premarket_open = {net_since_open_map.get(ticker, 0)}"

            if opt:
                option_base = safe_float(opt.get('base_value', 0.0), 0.0)
                effective_option = option_base * delta_factor
                key_name = f"input_{ticker}_real"

                if ticker in overrides:
                    if (key_name not in st.session_state) or (abs(shadow.get(key_name, float('nan')) - last_val) > 1e-12):
                        st.session_state[key_name] = float(last_val)
                        shadow[key_name] = float(last_val)
                else:
                    if key_name not in st.session_state:
                        st.session_state[key_name] = float(last_val)
                        shadow[key_name] = float(last_val)

                safe_val = safe_float(st.session_state.get(key_name, last_val), last_val)
                real_val = st.number_input(
                    label=display_label,
                    help=help_text_final,
                    step=0.001,
                    value=safe_val,
                    key=key_name,
                )
                asset_inputs[ticker] = float(effective_option + float(real_val))
            else:
                key_name = f"input_{ticker}_asset"

                if ticker in overrides:
                    if (key_name not in st.session_state) or (abs(shadow.get(key_name, float('nan')) - last_val) > 1e-12):
                        st.session_state[key_name] = float(last_val)
                        shadow[key_name] = float(last_val)
                else:
                    if key_name not in st.session_state:
                        st.session_state[key_name] = float(last_val)
                        shadow[key_name] = float(last_val)

                safe_val = safe_float(st.session_state.get(key_name, last_val), last_val)
                val = st.number_input(
                    label=display_label,
                    help=help_text_final,
                    step=0.001,
                    value=safe_val,
                    key=key_name,
                )
                asset_inputs[ticker] = float(val)

    return asset_inputs

def safe_ts_update(client: thingspeak.Channel, payload: Dict, timeout_sec: float = 10.0):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(client.update, payload)
        return fut.result(timeout=timeout_sec)

def render_asset_update_controls(configs: List[Dict], clients: Dict[int, thingspeak.Channel], last_assets: Dict[str, float]) -> None:
    """ใช้เส้นทางเดียวกับ GO_SELL/GO_BUY: optimistic → queue"""
    with st.expander("Update Assets on ThingSpeak"):
        for config in configs:
            ticker = config['ticker']
            asset_conf = config['asset_field']
            field_name = asset_conf['field']

            if st.checkbox(f'@_{ticker}_ASSET', key=f'check_{ticker}'):
                current_val = safe_float(last_assets.get(ticker, 0.0), 0.0)
                add_val = st.number_input(
                    f"New Value for {ticker}",
                    step=0.001,
                    value=current_val,
                    key=f'input_{ticker}'
                )
                if st.button(f"GO_{ticker}", key=f'btn_{ticker}'):
                    write_key = asset_conf.get('write_api_key') or asset_conf.get('api_key')
                    if not write_key:
                        st.error(f"[{ticker}] ไม่มี write_api_key/api_key สำหรับเขียน")
                    else:
                        _optimistic_apply_asset(
                            ticker=ticker,
                            new_value=float(add_val),
                            prev_value=float(current_val),
                            asset_conf=asset_conf,
                            op_label="SET"
                        )

def trading_section(
    config: Dict,
    asset_val: float,
    asset_last: float,
    df_data: pd.DataFrame,
    calc: Dict[str, Tuple[float, int, float]],
    nex: int,
    Nex_day_sell: int,
    clients: Dict[int, thingspeak.Channel],
    diff: float,
    min_rebalance: float,
    price_hint: Optional[float] = None
) -> None:
    ticker = config['ticker']
    asset_conf = config['asset_field']

    def get_action_val() -> Optional[int]:
        """ใช้ XOR แทน if: final = (raw&1) ^ (flip&1)  [SIMPLE/STABLE]"""
        try:
            if df_data.empty or df_data.action.values[1 + nex] == "":
                return None
            raw_action = int(df_data.action.values[1 + nex]) & 1
            flip = int(Nex_day_sell) & 1
            return xor01(raw_action, flip)
        except Exception:
            return None

    action_val = get_action_val()
    has_signal = action_val is not None
    limit_order_checked = st.checkbox(f'Limit_Order_{ticker}', value=has_signal, key=f'limit_order_{ticker}')
    if not limit_order_checked:
        return

    sell_calc = calc['sell']
    buy_calc  = calc['buy']

    # SELL — คง UI/สี/ตัวหนา ตามที่คุณตั้งไว้ (ใช้ buy_calc ตามพฤติกรรมเดิม)
    sell_html = (
        f"<span style='color:#ffffff;'>sell</span>&nbsp;&nbsp;"
        f"<span style='color:#ffffff;'>A</span>&nbsp;"
        f"<span style='color:#fbb; font-size:0.9em; font-weight:600'>{buy_calc[1]}</span> "
        f"<span style='color:#ffffff;'>P</span>&nbsp;"
        f"<span style='color:#fbb; font-size:0.9em; font-weight:600'>{buy_calc[0]}</span> "
        f"<span style='color:#ffffff;'>C</span>&nbsp;"
        f"<span style='color:#fbb; font-size:0.9em; font-weight:600'>{buy_calc[2]}</span>"
    )
    st.markdown(sell_html, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    if col3.checkbox(f'sell_match_{ticker}'):
        if col3.button(f"GO_SELL_{ticker}"):
            try:
                new_asset_val = float(asset_last) - float(buy_calc[1])  # ใช้ buy_calc ตามพฤติกรรมเดิม
                _optimistic_apply_asset(
                    ticker=ticker,
                    new_value=float(new_asset_val),
                    prev_value=float(asset_last),
                    asset_conf=asset_conf,
                    op_label="SELL"
                )
            except Exception as e:
                st.error(f"SELL {ticker} error: {e}")

    # Price & P/L (ใช้ price_hint ถ้ามี เพื่อไม่ต้องเรียกซ้ำ)
    try:
        current_price = float(price_hint) if (price_hint is not None and price_hint > 0) else get_cached_price(ticker)
        if current_price > 0:
            pv = current_price * float(asset_val)
            fix_value = float(config['fix_c'])
            pl_value = pv - fix_value
            pl_color = "#a8d5a2" if pl_value >= 0 else "#fbb"

            # trade_only_when และ % แสดงผล
            trade_only_when = math.sqrt(float(fix_value) * float(min_rebalance))
            new_pct = (trade_only_when / fix_value) * 100.0 if fix_value > 0 else 0.0
            new_pct_str = f"{new_pct:.2f}%" if new_pct < 1 else f"{new_pct:.0f}%"

            st.markdown(
                (
                    f"Price: **{current_price:,.3f}** | "
                    f"Value: **{pv:,.2f}** | "
                    f"P/L (vs {fix_value:,.0f}) | "
                    f"Min ({trade_only_when:,.0f}:{new_pct_str} vs {float(diff):,.0f}) | "
                    f"<span style='color:{pl_color}; font-weight:bold;'>{pl_value:,.2f}</span>"
                ),
                unsafe_allow_html=True
            )
        else:
            st.info(f"Price data for {ticker} is currently unavailable.")
    except Exception:
        st.warning(f"Could not retrieve price data for {ticker}.")

    # BUY — คง UI เดิม (ใช้ sell_calc ตามพฤติกรรมเดิม)
    col4, col5, col6 = st.columns(3)
    st.write('buy', '    ', 'A', sell_calc[1], 'P', sell_calc[0], 'C', sell_calc[2])
    if col6.checkbox(f'buy_match_{ticker}'):
        if col6.button(f"GO_BUY_{ticker}"):
            try:
                new_asset_val = float(asset_last) + float(sell_calc[1])
                _optimistic_apply_asset(
                    ticker=ticker,
                    new_value=float(new_asset_val),
                    prev_value=float(asset_last),
                    asset_conf=asset_conf,
                    op_label="BUY"
                )
            except Exception as e:
                st.error(f"BUY {ticker} error: {e}")

# ---------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------
# Session State init
if 'select_key' not in st.session_state:
    st.session_state.select_key = ""
if 'nex' not in st.session_state:
    st.session_state.nex = 0
if 'Nex_day_sell' not in st.session_state:
    st.session_state.Nex_day_sell = 0
if '_cache_bump' not in st.session_state:
    st.session_state['_cache_bump'] = 0
if '_last_assets_overrides' not in st.session_state:
    st.session_state['_last_assets_overrides'] = {}
if '_skip_refresh_on_rerun' not in st.session_state:
    st.session_state['_skip_refresh_on_rerun'] = False
if '_all_data_cache' not in st.session_state:
    st.session_state['_all_data_cache'] = None
if '_ts_last_update_at' not in st.session_state:
    st.session_state['_ts_last_update_at'] = {}
if '_pending_ts_update' not in st.session_state:
    st.session_state['_pending_ts_update'] = []
if '_ts_entry_ids' not in st.session_state:
    st.session_state['_ts_entry_ids'] = {}
if '_widget_shadow' not in st.session_state:
    st.session_state['_widget_shadow'] = {}
if 'min_rebalance' not in st.session_state:
    st.session_state['min_rebalance'] = 2.4  # คงสเปค

# === 💡 GOAL_1: DYNAMIC DIFF LOGIC START ===
if 'diff_value' not in st.session_state:
    st.session_state.diff_value = ASSET_CONFIGS[0].get('diff', 60) if ASSET_CONFIGS else 60
if '_last_selected_ticker' not in st.session_state:
    st.session_state._last_selected_ticker = ""
# === 💡 GOAL_1: DYNAMIC DIFF LOGIC END ===

pending = st.session_state.pop("_pending_select_key", None)
if pending:
    st.session_state.select_key = pending

latest_us_premarket_open_bkk = get_latest_us_premarket_open_bkk()
window_start_bkk_iso = latest_us_premarket_open_bkk.isoformat()

CACHE_BUMP = st.session_state.get('_cache_bump', 0)
if st.session_state.get('_skip_refresh_on_rerun', False) and st.session_state.get('_all_data_cache'):
    all_data = st.session_state['_all_data_cache']
    st.session_state['_skip_refresh_on_rerun'] = False
else:
    all_data = fetch_all_data(ASSET_CONFIGS, THINGSPEAK_CLIENTS, GLOBAL_START_DATE, window_start_bkk_iso, cache_bump=CACHE_BUMP)
    st.session_state['_all_data_cache'] = all_data

monitor_data_all = all_data['monitors']
last_assets_all = all_data['assets']

# optimistic overrides
if st.session_state.get('_last_assets_overrides'):
    last_assets_all = {**last_assets_all, **st.session_state['_last_assets_overrides']}

trade_nets_all = all_data['nets']
trade_stats_all = all_data['trade_stats']

# Tabs
tab1, tab2 = st.tabs(["📈 Monitor", "⚙️ Controls"])

with tab2:
    left, right = st.columns([2, 1])
    with left:
        Nex_day_ = st.checkbox('nex_day', value=(st.session_state.nex == 1))
        if Nex_day_:
            nex_col, Nex_day_sell_col, *_ = st.columns([1, 1, 3])
            if nex_col.button("Nex_day"):
                st.session_state.nex = 1
                st.session_state.Nex_day_sell = 0
            if Nex_day_sell_col.button("Nex_day_sell"):
                st.session_state.nex = 1
                st.session_state.Nex_day_sell = 1
        else:
            st.session_state.nex = 0
            st.session_state.Nex_day_sell = 0

        nex = st.session_state.nex
        Nex_day_sell = st.session_state.Nex_day_sell
        if Nex_day_:
            st.write(f"nex value = {nex}", f" | Nex_day_sell = {Nex_day_sell}" if Nex_day_sell else "")
    with right:
        st.session_state['min_rebalance'] = st.number_input(
            'Min_Rebalance',
            min_value=0.0,
            step=0.1,
            value=float(st.session_state.get('min_rebalance', 2.4)),
            help="แฟกเตอร์สำหรับคำนวณ trade_only_when ด้วยสมการ sqrt(Min_Rebalance * fix_value)"
        )

    st.write("---")

    # === 💡 GOAL_1: DYNAMIC DIFF LOGIC START ===
    selected_ticker = st.session_state.get('select_key', "")
    if selected_ticker != st.session_state.get('_last_selected_ticker'):
        new_diff = None
        if selected_ticker and selected_ticker in [c['ticker'] for c in ASSET_CONFIGS]:
            for config in ASSET_CONFIGS:
                if config['ticker'] == selected_ticker:
                    new_diff = config.get('diff', 60)
                    break
        if new_diff is not None:
            st.session_state.diff_value = new_diff
        st.session_state._last_selected_ticker = selected_ticker

    x_2_from_state = st.sidebar.number_input('Diff', step=1, key='diff_value')
    # === 💡 GOAL_1: DYNAMIC DIFF LOGIC END ===

    asset_inputs = render_asset_inputs(ASSET_CONFIGS, last_assets_all, trade_nets_all)

    st.write("_____")
    Start = st.checkbox('start')
    if Start:
        render_asset_update_controls(ASSET_CONFIGS, THINGSPEAK_CLIENTS, last_assets_all)

with tab1:
    current_ny_date = get_current_ny_date()

    selectbox_labels: Dict[str, str] = {}
    ticker_actions: Dict[str, Optional[int]] = {}

    for config in ASSET_CONFIGS:
        ticker = config['ticker']
        df_data, fx_js_str, last_data_date = monitor_data_all.get(ticker, (pd.DataFrame(), "0", None))
        action_emoji, final_action_val = "", None

        if st.session_state.nex == 0 and last_data_date and last_data_date < current_ny_date:
            action_emoji = "🟡 "
        else:
            try:
                if not df_data.empty and df_data.action.values[1 + st.session_state.nex] != "":
                    raw_action = int(df_data.action.values[1 + st.session_state.nex]) & 1
                    flip = int(st.session_state.Nex_day_sell) & 1
                    final_action_val = xor01(raw_action, flip)
                    if final_action_val == 1:
                        action_emoji = "🟢 "
                    elif final_action_val == 0:
                        action_emoji = "🔴 "
            except Exception:
                pass

        ticker_actions[ticker] = final_action_val

        base_net = int(trade_nets_all.get(ticker, 0))
        net_str = make_net_str_with_optimism(ticker, base_net)
        selectbox_labels[ticker] = f"{action_emoji}{ticker} (f(x): {fx_js_str})  {net_str}"

    buy_set  = {t for t, a in ticker_actions.items() if a == 1}
    sell_set = {t for t, a in ticker_actions.items() if a == 0}
    buy_list  = [t for t in ALL_TICKERS if t in buy_set]
    sell_list = [t for t in ALL_TICKERS if t in sell_set]

    all_tickers = [c['ticker'] for c in ASSET_CONFIGS]
    selectbox_options: List[str] = [""]
    if st.session_state.nex == 1:
        selectbox_options.extend(["Filter Buy Tickers", "Filter Sell Tickers"])
    selectbox_options.extend(all_tickers)

    if st.session_state.select_key not in selectbox_options:
        st.session_state.select_key = ""

    def format_selectbox_options(option_name: str) -> str:
        if option_name in ["", "Filter Buy Tickers", "Filter Sell Tickers"]:
            return "Show All" if option_name == "" else option_name
        return selectbox_labels.get(option_name, option_name).split(' (f(x):')[0]

    st.selectbox(
        "Select Ticker to View:",
        options=selectbox_options,
        format_func=format_selectbox_options,
        key="select_key"
    )
    st.write("_____")

    selected_option = st.session_state.select_key
    if selected_option == "":
        configs_to_display = ASSET_CONFIGS
    elif selected_option == "Filter Buy Tickers":
        configs_to_display = [c for c in ASSET_CONFIGS if c['ticker'] in buy_set]
    elif selected_option == "Filter Sell Tickers":
        configs_to_display = [c for c in ASSET_CONFIGS if c['ticker'] in sell_set]
    else:
        configs_to_display = [c for c in ASSET_CONFIGS if c['ticker'] == selected_option]

    # ✅ Prefetch ราคาทั้งชุดในหน้า Monitor รอบเดียว
    display_tickers = [c['ticker'] for c in configs_to_display]
    prices_hint_map = get_prices_map(display_tickers)

    calculations: Dict[str, Dict[str, Tuple[float, int, float]]] = {}
    for config in ASSET_CONFIGS:
        ticker = config['ticker']
        asset_value = float(asset_inputs.get(ticker, 0.0))
        fix_c = float(config['fix_c'])
        calculations[ticker] = {
            'sell': sell(asset_value, fix_c=fix_c, Diff=float(st.session_state.diff_value)),
            'buy':  buy(asset_value,  fix_c=fix_c, Diff=float(st.session_state.diff_value)),
        }

    for config in configs_to_display:
        ticker = config['ticker']
        df_data, fx_js_str, _ = monitor_data_all.get(ticker, (pd.DataFrame(), "0", None))
        asset_last = float(last_assets_all.get(ticker, 0.0))
        asset_val = float(asset_inputs.get(ticker, 0.0))  # delta-equivalent
        calc = calculations.get(ticker, {})

        title_label = selectbox_labels.get(ticker, ticker)
        st.write(title_label)

        trading_section(
            config=config,
            asset_val=asset_val,
            asset_last=asset_last,
            df_data=df_data,
            calc=calc,
            nex=st.session_state.nex,
            Nex_day_sell=st.session_state.Nex_day_sell,
            clients=THINGSPEAK_CLIENTS,
            diff=float(st.session_state.diff_value),
            min_rebalance=float(st.session_state['min_rebalance']),
            price_hint=prices_hint_map.get(ticker)
        )

        with st.expander("Show Raw Data Action"):
            st.dataframe(df_data, use_container_width=True)
        st.write("_____")

# === 🧭 Sidebar Ticker Navigator
with st.sidebar:
    st.write("_____")
    sel = st.session_state.get("select_key", "")

    if sel == "Filter Buy Tickers" and buy_list:
        idx = int(st.session_state.get('_filter_nav_idx_buy', 0)) % len(buy_list)
        current_preview = buy_list[idx]
        st.markdown(f"**Buy Navigator**  \n{idx+1}/{len(buy_list)} · `{current_preview}`")
        c1, c2 = st.columns(2)
        if c1.button("◀ Prev", use_container_width=True, key="__nav_prev_buy"):
            st.session_state['_filter_nav_idx_buy'] = (idx - 1) % len(buy_list)
            st.session_state["_pending_select_key"] = buy_list[st.session_state['_filter_nav_idx_buy']]
            st.rerun()
        if c2.button("Next ▶", use_container_width=True, key="__nav_next_buy"):
            st.session_state['_filter_nav_idx_buy'] = (idx + 1) % len(buy_list)
            st.session_state["_pending_select_key"] = buy_list[st.session_state['_filter_nav_idx_buy']]
            st.rerun()

    elif sel == "Filter Sell Tickers" and sell_list:
        idx = int(st.session_state.get('_filter_nav_idx_sell', 0)) % len(sell_list)
        current_preview = sell_list[idx]
        st.markdown(f"**Sell Navigator**  \n{idx+1}/{len(sell_list)} · `{current_preview}`")
        c1, c2 = st.columns(2)
        if c1.button("◀ Prev", use_container_width=True, key="__nav_prev_sell"):
            st.session_state['_filter_nav_idx_sell'] = (idx - 1) % len(sell_list)
            st.session_state["_pending_select_key"] = sell_list[st.session_state['_filter_nav_idx_sell']]
            st.rerun()
        if c2.button("Next ▶", use_container_width=True, key="__nav_next_sell"):
            st.session_state['_filter_nav_idx_sell'] = (idx + 1) % len(sell_list)
            st.session_state["_pending_select_key"] = sell_list[st.session_state['_filter_nav_idx_sell']]
            st.rerun()

    elif sel in ALL_TICKERS:
        act = locals().get('ticker_actions', {}).get(sel, None)
        if act == 1 and buy_list:
            nav_list = buy_list
            title = "Buy Navigator"
        elif act == 0 and sell_list:
            nav_list = sell_list
            title = "Sell Navigator"
        else:
            nav_list = ALL_TICKERS
            title = "Ticker Navigator"

        if nav_list:
            idx = nav_list.index(sel) if sel in nav_list else 0
            st.markdown(f"**{title}**  \n{idx+1}/{len(nav_list)} · `{sel}`")
            c1, c2 = st.columns(2)
            if c1.button("◀ Prev", use_container_width=True, key="__nav_prev_single"):
                new_idx = (idx - 1) % len(nav_list)
                st.session_state["_pending_select_key"] = nav_list[new_idx]
                st.rerun()
            if c2.button("Next ▶", use_container_width=True, key="__nav_next_single"):
                new_idx = (idx + 1) % len(nav_list)
                st.session_state["_pending_select_key"] = nav_list[new_idx]
                st.rerun()

# Sidebar Rerun (Hard Reload)
if st.sidebar.button("RERUN"):
    current_selection = st.session_state.get("select_key", "")
    clear_all_caches()
    if current_selection in [c['ticker'] for c in ASSET_CONFIGS]:
        rerun_keep_selection(current_selection)
    else:
        st.rerun()

# ✅ PROCESS PENDING UPDATES AT THE END
process_pending_updates(min_interval=16.0, max_wait=8.0)
