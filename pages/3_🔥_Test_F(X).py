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

# ---------------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------------
st.set_page_config(page_title="Monitor", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------------
# SimulationTracer (พฤติกรรมเดิม)
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
# Cache Management (Hard Reload ใช้เฉพาะปุ่ม Sidebar)
# ---------------------------------------------------------------------------------
def clear_all_caches() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()
    sell.cache_clear()
    buy.cache_clear()

    ui_state_keys_to_preserve = {'select_key', 'nex', 'Nex_day_sell', '_cache_bump',
                                 '_last_assets_overrides', '_all_data_cache',
                                 '_fast_focus_enabled', '_skip_refresh_on_rerun'}
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
# Calculation Utils (เดิม)
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
# Price & Time helpers
# ---------------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
@tenacity.retry(wait=tenacity.wait_fixed(2), stop=tenacity.stop_after_attempt(3))
def get_cached_price(ticker: str) -> float:
    try:
        return float(yf.Ticker(ticker).fast_info['lastPrice'])
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

# ------------------- HTTP GET update (เขียน ThingSpeak ด้วย Write Key) -------------------
def ts_update_via_http(write_api_key: str, field_name: str, value, timeout_sec: float = 5.0) -> str:
    """
    เรียก ThingSpeak update ผ่าน HTTP GET
    Return: entry_id string ถ้าสำเร็จ, "0" ถ้าไม่สำเร็จ
    """
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

# ---------------------------------------------------------------------------------
# Incremental cache-busting (เดิม)
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
# Fetch all data — โหมดปกติ + รองรับ fast rerun (ข้าม fetch)
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
# UI helpers
# ---------------------------------------------------------------------------------
def render_asset_inputs(configs: List[Dict], last_assets: Dict[str, float], net_since_open_map: Dict[str, int]) -> Dict[str, float]:
    asset_inputs: Dict[str, float] = {}
    cols = st.columns(len(configs)) if configs else [st]
    for i, config in enumerate(configs):
        with cols[i]:
            ticker = config['ticker']
            last_val = float(last_assets.get(ticker, 0.0))

            # SAFE read for option_config (กัน null/ชนิดอื่น)
            opt = config.get('option_config')
            if not isinstance(opt, dict):
                opt = {}

            raw_label = opt.get('label', ticker)
            display_label = raw_label
            base_help = ""
            split_pos = raw_label.find('(')
            if split_pos != -1:
                display_label = raw_label[:split_pos].strip()
                base_help = raw_label[split_pos:].strip()
            help_text_final = base_help if base_help else f"net_since_us_premarket_open = {net_since_open_map.get(ticker, 0)}"

            if opt:  # มี option_config เป็น dict จริง
                option_val = float(opt.get('base_value', 0.0))
                real_val = st.number_input(
                    label=display_label, help=help_text_final,
                    step=0.001, value=last_val, key=f"input_{ticker}_real"
                )
                asset_inputs[ticker] = option_val + float(real_val)
            else:
                val = st.number_input(
                    label=display_label, help=help_text_final,
                    step=0.001, value=last_val, key=f"input_{ticker}_asset"
                )
                asset_inputs[ticker] = float(val)
    return asset_inputs

# --- ตัวช่วย update แบบมี timeout โดยไม่เปลี่ยนไลบรารี (ใช้ใน Controls; ตัด rerun ออก) -------------
def safe_ts_update(client: thingspeak.Channel, payload: Dict, timeout_sec: float = 10.0):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(client.update, payload)
        return fut.result(timeout=timeout_sec)

def render_asset_update_controls(configs: List[Dict], clients: Dict[int, thingspeak.Channel]) -> None:
    with st.expander("Update Assets on ThingSpeak"):
        for config in configs:
            ticker = config['ticker']
            asset_conf = config['asset_field']
            field_name = asset_conf['field']
            if st.checkbox(f'@_{ticker}_ASSET', key=f'check_{ticker}'):
                add_val = st.number_input(f"New Value for {ticker}", step=0.001, value=0.0, key=f'input_{ticker}')
                if st.button(f"GO_{ticker}", key=f'btn_{ticker}'):
                    try:
                        client = clients[int(asset_conf['channel_id'])]
                        safe_ts_update(client, {field_name: add_val}, timeout_sec=10.0)
                        st.write(f"Updated {ticker} to: {add_val} on Channel {asset_conf['channel_id']}")
                        # optimistic override แทน rerun
                        st.session_state.setdefault('_last_assets_overrides', {})[ticker] = float(add_val)
                        st.session_state['_cache_bump'] = st.session_state.get('_cache_bump', 0) + 1
                    except concurrent.futures.TimeoutError:
                        st.error(f"Update {ticker} timed out (>10s).")
                    except Exception as e:
                        st.error(f"Failed to update {ticker}: {e}")

def trading_section(
    config: Dict,
    asset_val: float,
    asset_last: float,
    df_data: pd.DataFrame,
    calc: Dict[str, Tuple[float, int, float]],
    nex: int,
    Nex_day_sell: int,
    clients: Dict[int, thingspeak.Channel]
) -> None:
    ticker = config['ticker']
    asset_conf = config['asset_field']
    field_name = asset_conf['field']

    def get_action_val() -> Optional[int]:
        try:
            if df_data.empty or df_data.action.values[1 + nex] == "":
                return None
            raw_action = int(df_data.action.values[1 + nex])
            final_action = 1 - raw_action if Nex_day_sell == 1 else raw_action
            return final_action
        except (IndexError, ValueError, TypeError):
            return None

    action_val = get_action_val()
    has_signal = action_val is not None
    limit_order_checked = st.checkbox(f'Limit_Order_{ticker}', value=has_signal, key=f'limit_order_{ticker}')
    if not limit_order_checked:
        return

    sell_calc = calc['sell']
    buy_calc = calc['buy']

    # SELL line (UI/ตรรกะเดิม) — เปลี่ยนเฉพาะการ "อัปเดต" → HTTP GET + Optimistic UI (+โหมดโฟกัสเร็ว)
    st.write('sell', '    ', 'A', buy_calc[1], 'P', buy_calc[0], 'C', buy_calc[2])
    col1, col2, col3 = st.columns(3)
    if col3.checkbox(f'sell_match_{ticker}'):
        if col3.button(f"GO_SELL_{ticker}"):
            try:
                new_asset_val = asset_last - buy_calc[1]  # คำนวณเดิม
                write_key = asset_conf.get('write_api_key') or asset_conf.get('api_key')
                resp = ts_update_via_http(write_key, field_name, new_asset_val, timeout_sec=5.0)

                if resp.strip() == "0":
                    st.error("ThingSpeak ไม่บันทึกค่า (resp=0): ตรวจ Write API Key/ชื่อฟิลด์ และเว้น ~15s ต่อการเขียน/ช่อง")
                else:
                    col3.write(f"Updated: {new_asset_val} (entry #{resp})")
                    # Optimistic UI
                    st.session_state.setdefault('_last_assets_overrides', {})[ticker] = float(new_asset_val)
                    st.session_state['_cache_bump'] = st.session_state.get('_cache_bump', 0) + 1
                    # โหมด B (ถ้าผู้ใช้เปิด): โฟกัส ticker + rerun แบบเบา
                    if st.session_state.get('_fast_focus_enabled', False):
                        st.session_state["_pending_select_key"] = ticker
                        st.session_state["_skip_refresh_on_rerun"] = True
                        st.rerun()
            except Exception as e:
                st.error(f"SELL {ticker} error: {e}")

    # ราคาปัจจุบัน & P/L (เดิม)
    try:
        current_price = get_cached_price(ticker)
        if current_price > 0:
            pv = current_price * asset_val
            fix_value = float(config['fix_c'])
            pl_value = pv - fix_value
            pl_color = "#a8d5a2" if pl_value >= 0 else "#fbb"
            st.markdown(
                f"Price: **{current_price:,.3f}** | Value: **{pv:,.2f}** | P/L (vs {fix_value:,.0f}) : "
                f"<span style='color:{pl_color}; font-weight:bold;'>{pl_value:,.2f}</span>",
                unsafe_allow_html=True
            )
        else:
            st.info(f"Price data for {ticker} is currently unavailable.")
    except Exception:
        st.warning(f"Could not retrieve price data for {ticker}.")

    # BUY line (UI/ตรรกะเดิม) — เปลี่ยนเฉพาะการ "อัปเดต" → HTTP GET + Optimistic UI (+โหมดโฟกัสเร็ว)
    col4, col5, col6 = st.columns(3)
    st.write('buy', '   ', 'A', sell_calc[1], 'P', sell_calc[0], 'C', sell_calc[2])
    if col6.checkbox(f'buy_match_{ticker}'):
        if col6.button(f"GO_BUY_{ticker}"):
            try:
                new_asset_val = asset_last + sell_calc[1]
                write_key = asset_conf.get('write_api_key') or asset_conf.get('api_key')
                resp = ts_update_via_http(write_key, field_name, new_asset_val, timeout_sec=5.0)

                if resp.strip() == "0":
                    st.error("ThingSpeak ไม่บันทึกค่า (resp=0): ตรวจ Write API Key/ชื่อฟิลด์ และเว้น ~15s ต่อการเขียน/ช่อง")
                else:
                    col6.write(f"Updated: {new_asset_val} (entry #{resp})")
                    st.session_state.setdefault('_last_assets_overrides', {})[ticker] = float(new_asset_val)
                    st.session_state['_cache_bump'] = st.session_state.get('_cache_bump', 0) + 1
                    if st.session_state.get('_fast_focus_enabled', False):
                        st.session_state["_pending_select_key"] = ticker
                        st.session_state["_skip_refresh_on_rerun"] = True
                        st.rerun()
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
# โหมดผสม: ตัวเลือก & ธง fast rerun
if '_fast_focus_enabled' not in st.session_state:
    st.session_state['_fast_focus_enabled'] = False
if '_skip_refresh_on_rerun' not in st.session_state:
    st.session_state['_skip_refresh_on_rerun'] = False
if '_all_data_cache' not in st.session_state:
    st.session_state['_all_data_cache'] = None

# Bootstrap selection BEFORE widgets (สำหรับโหมด B)
pending = st.session_state.pop("_pending_select_key", None)
if pending:
    st.session_state.select_key = pending

# เวลาเปิดตลาดสหรัฐ (Pre-Market) ล่าสุด → จุดอ้างอิง window_start
latest_us_premarket_open_bkk = get_latest_us_premarket_open_bkk()
window_start_bkk_iso = latest_us_premarket_open_bkk.isoformat()

# ดึงข้อมูลหลัก — รองรับ fast rerun (ข้าม fetch หนัก)
CACHE_BUMP = st.session_state.get('_cache_bump', 0)
if st.session_state.get('_skip_refresh_on_rerun', False) and st.session_state.get('_all_data_cache'):
    all_data = st.session_state['_all_data_cache']
    st.session_state['_skip_refresh_on_rerun'] = False  # ใช้ครั้งเดียว
else:
    all_data = fetch_all_data(ASSET_CONFIGS, THINGSPEAK_CLIENTS, GLOBAL_START_DATE, window_start_bkk_iso, cache_bump=CACHE_BUMP)
    st.session_state['_all_data_cache'] = all_data  # เก็บไว้ใช้รอบถัดไป

monitor_data_all = all_data['monitors']
last_assets_all = all_data['assets']

# apply optimistic overrides (ค่าที่เพิ่งอัปเดต)
if st.session_state.get('_last_assets_overrides'):
    last_assets_all = {**last_assets_all, **st.session_state['_last_assets_overrides']}

trade_nets_all = all_data['nets']          # สำหรับ label/help เดิม
trade_stats_all = all_data['trade_stats']  # แยก buy/sell ตั้งแต่ premarket

tab1, tab2 = st.tabs(["📈 Monitor", "⚙️ Controls"])

with tab2:
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

    st.write("---")
    x_2 = st.number_input('Diff', step=1, value=60)
    st.write("---")
    asset_inputs = render_asset_inputs(ASSET_CONFIGS, last_assets_all, trade_nets_all)
    st.write("---")

    # METRICS (พฤติกรรมเดิม)
    with st.expander("METRICS"):
        tz_bkk = pytz.timezone('Asia/Bangkok')
        now_bkk = datetime.datetime.now(tz_bkk)

        def _parse_global_start_date_to_date(s: Optional[str]) -> Optional[datetime.date]:
            if not s:
                return None
            try:
                return datetime.datetime.fromisoformat(s).date()
            except Exception:
                pass
            m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
            if m:
                y, mo, d = map(int, m.groups())
                try:
                    return datetime.date(y, mo, d)
                except Exception:
                    return None
            return None

        min_candidate = _parse_global_start_date_to_date(GLOBAL_START_DATE)
        min_date = min_candidate or (now_bkk.date() - datetime.timedelta(days=30))
        max_date = now_bkk.date()
        default_start = max(min_date, latest_us_premarket_open_bkk.date())
        default_end = max_date

        date_start, date_end = st.slider(
            "ช่วงวันที่ (Asia/Bangkok)",
            min_value=min_date,
            max_value=max_date,
            value=(default_start, default_end),
            format="YYYY-MM-DD"
        )

        start_dt = tz_bkk.localize(datetime.datetime.combine(date_start, datetime.time.min))
        end_dt = tz_bkk.localize(datetime.datetime.combine(date_end, datetime.time.max))

        total_buy_orders = 0
        total_sell_orders = 0
        total_buy_usd = 0.0
        total_sell_usd = 0.0

        rows = []
        for cfg in ASSET_CONFIGS:
            t = cfg['ticker']
            stats = fetch_net_detailed_stats_between(
                cfg['asset_field'],
                start_dt.isoformat(),
                end_dt.isoformat(),
                cache_bump=CACHE_BUMP
            )
            b_cnt = int(stats.get('buy_count', 0))
            s_cnt = int(stats.get('sell_count', 0))
            b_units = float(stats.get('buy_units', 0.0))
            s_units = float(stats.get('sell_units', 0.0))
            net_cnt = int(stats.get('net_count', 0))
            net_units = float(stats.get('net_units', 0.0))

            px = float(get_cached_price(t))
            buy_usd = b_units * px
            sell_usd = - s_units * px
            net_usd = buy_usd + sell_usd

            total_buy_orders += b_cnt
            total_sell_orders += s_cnt
            total_buy_usd += buy_usd
            total_sell_usd += sell_usd

            rows.append({
                "Ticker": t,
                "Buy_Orders": b_cnt,
                "Sell_Orders": s_cnt,
                "Total_Orders(Net)": net_cnt,
                "Buy_Units": b_units,
                "Sell_Units": s_units,
                "Net_Units": net_units,
                "Price": px,
                "Buy_USD": buy_usd,
                "Sell_USD": sell_usd,
                "Net_USD": net_usd
            })

        net_orders_total = total_buy_orders - total_sell_orders
        net_usd_total = total_buy_usd + total_sell_usd

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Orders (Buy - Sell)", f"{net_orders_total}")
        c2.metric("Buy_Orders", f"{total_buy_orders}")
        c3.metric("Sell_Orders", f"{total_sell_orders}")

        d1, d2, d3 = st.columns(3)
        d1.metric("Net USD Flow (ช่วงที่เลือก)", f"${net_usd_total:,.2f}")
        d2.metric("Buy_USD", f"${total_buy_usd:,.2f}")
        d3.metric("Sell_USD", f"${total_sell_usd:,.2f}")

        with st.expander("Per-ticker detail"):
            df_metrics = pd.DataFrame(rows).set_index("Ticker")
            st.dataframe(df_metrics, use_container_width=True)

    st.write("_____")

    Start = st.checkbox('start')
    if Start:
        render_asset_update_controls(ASSET_CONFIGS, THINGSPEAK_CLIENTS)

    # โหมดผสม: สวิตช์ Fast focus
    st.checkbox(
        "Fast focus after update",
        value=st.session_state['_fast_focus_enabled'],
        key='_fast_focus_enabled',
        help="เปิดไว้ถ้าต้องการให้ selectbox โฟกัส ticker หลังอัปเดต โดยทำ rerun แบบเบา (ข้ามการดึงข้อมูลหนัก ๆ)"
    )

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
                    raw_action = int(df_data.action.values[1 + st.session_state.nex])
                    final_action_val = 1 - raw_action if st.session_state.Nex_day_sell == 1 else raw_action
                    if final_action_val == 1:
                        action_emoji = "🟢 "
                    elif final_action_val == 0:
                        action_emoji = "🔴 "
            except (IndexError, ValueError, TypeError):
                pass

        ticker_actions[ticker] = final_action_val
        net_str = trade_nets_all.get(ticker, 0)
        selectbox_labels[ticker] = f"{action_emoji}{ticker} (f(x): {fx_js_str})  {net_str}"

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
        buy_tickers = {t for t, action in ticker_actions.items() if action == 1}
        configs_to_display = [c for c in ASSET_CONFIGS if c['ticker'] in buy_tickers]
    elif selected_option == "Filter Sell Tickers":
        sell_tickers = {t for t, action in ticker_actions.items() if action == 0}
        configs_to_display = [c for c in ASSET_CONFIGS if c['ticker'] in sell_tickers]
    else:
        configs_to_display = [c for c in ASSET_CONFIGS if c['ticker'] == selected_option]

    calculations: Dict[str, Dict[str, Tuple[float, int, float]]] = {}
    for config in ASSET_CONFIGS:
        ticker = config['ticker']
        asset_value = float(asset_inputs.get(ticker, 0.0))
        fix_c = float(config['fix_c'])

        calculations[ticker] = {
            'sell': sell(asset_value, fix_c=fix_c, Diff=float(x_2)),
            'buy': buy(asset_value, fix_c=fix_c, Diff=float(x_2)),
        }

    for config in configs_to_display:
        ticker = config['ticker']
        df_data, fx_js_str, _ = monitor_data_all.get(ticker, (pd.DataFrame(), "0", None))
        asset_last = float(last_assets_all.get(ticker, 0.0))
        asset_val = float(asset_inputs.get(ticker, 0.0))
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
            clients=THINGSPEAK_CLIENTS
        )

        with st.expander("Show Raw Data Action"):
            st.dataframe(df_data, use_container_width=True)
        st.write("_____")

# Sidebar Rerun (Hard Reload)
if st.sidebar.button("RERUN"):
    current_selection = st.session_state.get("select_key", "")
    clear_all_caches()
    if current_selection in [c['ticker'] for c in ASSET_CONFIGS]:
        rerun_keep_selection(current_selection)
    else:
        st.rerun()
