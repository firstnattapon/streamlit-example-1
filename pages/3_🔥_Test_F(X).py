import streamlit as st
import numpy as np
import datetime as dt
import thingspeak
import pandas as pd
import yfinance as yf
import json
from functools import lru_cache
import concurrent.futures
import os
from typing import List, Dict, Tuple, Any
import tenacity

# --- CHANGE START: timezone helper (stdlib) ---
try:
    from zoneinfo import ZoneInfo
except Exception:
    # Python <3.9 fallback (ถ้าจำเป็นจริง ๆ ให้ติดตั้ง pytz แล้วใช้ pytz.timezone)
    ZoneInfo = None
# --- CHANGE END ---

st.set_page_config(page_title="Monitor", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# --- SimulationTracer Class (unchanged) ---
class SimulationTracer:
    def __init__(self, encoded_string: str):
        self.encoded_string: str = str(encoded_string)
        self._decode_and_set_attributes()

    def _decode_and_set_attributes(self):
        encoded_string = self.encoded_string
        if not encoded_string.isdigit():
            self.action_length: int = 0
            self.mutation_rate: int = 0
            self.dna_seed: int = 0
            self.mutation_seeds: List[int] = []
            self.mutation_rate_float: float = 0.0
            return

        decoded_numbers = []
        idx = 0
        try:
            while idx < len(encoded_string):
                length_of_number = int(encoded_string[idx])
                idx += 1
                number_str = encoded_string[idx : idx + length_of_number]
                idx += length_of_number
                decoded_numbers.append(int(number_str))
        except (IndexError, ValueError):
            pass

        if len(decoded_numbers) < 3:
            self.action_length: int = 0
            self.mutation_rate: int = 0
            self.dna_seed: int = 0
            self.mutation_seeds: List[int] = []
            self.mutation_rate_float: float = 0.0
            return

        self.action_length: int = decoded_numbers[0]
        self.mutation_rate: int = decoded_numbers[1]
        self.dna_seed: int = decoded_numbers[2]
        self.mutation_seeds: List[int] = decoded_numbers[3:]
        self.mutation_rate_float: float = self.mutation_rate / 100.0

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

# --- Configuration Loading (unchanged) ---
@st.cache_data
def load_config(file_path='monitor_config.json') -> Dict:
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

ASSET_CONFIGS = CONFIG_DATA.get('assets', [])
GLOBAL_START_DATE = CONFIG_DATA.get('global_settings', {}).get('start_date')

if not ASSET_CONFIGS:
    st.error("No 'assets' list found in monitor_config.json")
    st.stop()

# --- ThingSpeak Clients (unchanged) ---
@st.cache_resource
def get_thingspeak_clients(configs: List[Dict]) -> Dict[int, thingspeak.Channel]:
    clients = {}
    unique_channels = set()
    for config in configs:
        mon_conf = config['monitor_field']
        unique_channels.add((mon_conf['channel_id'], mon_conf['api_key']))
        asset_conf = config['asset_field']
        unique_channels.add((asset_conf['channel_id'], asset_conf['api_key']))

    for channel_id, api_key in unique_channels:
        try:
            clients[channel_id] = thingspeak.Channel(channel_id, api_key, fmt='json')
        except Exception as e:
            st.error(f"Failed to create client for Channel ID {channel_id}: {e}")
    return clients

THINGSPEAK_CLIENTS = get_thingspeak_clients(ASSET_CONFIGS)

# --- Clear Caches Function (unchanged) ---
def clear_all_caches():
    st.cache_data.clear()
    st.cache_resource.clear()
    sell.cache_clear()
    buy.cache_clear()
    ui_state_keys_to_preserve = ['select_key', 'nex', 'Nex_day_sell']
    keys_to_delete = [k for k in st.session_state.keys() if k not in ui_state_keys_to_preserve]
    for key in keys_to_delete:
        del st.session_state[key]
    st.success("🗑️ Data caches cleared! UI state preserved.")

# --- Calculation Utils (unchanged) ---
@lru_cache(maxsize=128)
def sell(asset, fix_c=1500, Diff=60):
    if asset == 0: return 0, 0, 0
    unit_price = round((fix_c - Diff) / asset, 2)
    adjust_qty = round(abs(asset * unit_price - fix_c) / unit_price) if unit_price != 0 else 0
    total = round(asset * unit_price + adjust_qty * unit_price, 2)
    return unit_price, adjust_qty, total

@lru_cache(maxsize=128)
def buy(asset, fix_c=1500, Diff=60):
    if asset == 0: return 0, 0, 0
    unit_price = round((fix_c + Diff) / asset, 2)
    adjust_qty = round(abs(asset * unit_price - fix_c) / unit_price) if unit_price != 0 else 0
    total = round(asset * unit_price - adjust_qty * unit_price, 2)
    return unit_price, adjust_qty, total

# --- CHANGE START: Market session helpers + synced quote with fallback chain ---
def _now_ny() -> dt.datetime:
    if ZoneInfo is None:
        # Fallback: assume UTC then subtract 4/5 hours is unsafe; keep UTC and treat as closed
        return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    return dt.datetime.now(ZoneInfo("America/New_York"))

def _is_weekday(d: dt.date) -> bool:
    return d.weekday() < 5  # Mon-Fri

def is_us_market_open(now_ny: dt.datetime | None = None) -> bool:
    """Approximate regular session (ไม่รวมวันหยุด): 09:30-16:00 ET, Mon-Fri."""
    if now_ny is None:
        now_ny = _now_ny()
    if ZoneInfo is None:
        return False
    if not _is_weekday(now_ny.date()):
        return False
    t = now_ny.time()
    return (t >= dt.time(9, 30)) and (t <= dt.time(16, 0))

@st.cache_data(ttl=60)  # live path refresh เร็วหน่อย
def get_synced_quote(ticker: str, want_live: bool) -> Tuple[float, str]:
    """
    ถ้า want_live=True → พยายามใช้ 1m intraday; ถ้าไม่ได้ค่อยตกลง daily; ถ้ายังไม่ได้ค่อย fast_info
    คืนค่า: (price, asof_label)
    """
    # 1) Intraday 1m (เมื่อเปิดตลาด)
    if want_live:
        try:
            df_1m = yf.Ticker(ticker).history(period="1d", interval="1m", prepost=True, auto_adjust=True)
            if not df_1m.empty:
                # เอาบาร์สุดท้ายที่เป็นตัวเลขจริง
                last_row = df_1m["Close"].dropna().tail(1)
                if not last_row.empty:
                    ts = last_row.index[-1]
                    # ทำให้แสดงเวลา ET สวย ๆ
                    if ZoneInfo is not None:
                        ts_et = (ts.tz_convert(ZoneInfo("America/New_York"))
                                 if ts.tzinfo else ts.tz_localize(ZoneInfo("America/New_York")))
                        label = f"LIVE {ts_et.strftime('%Y-%m-%d %H:%M ET')}"
                    else:
                        label = "LIVE"
                    return float(last_row.iloc[0]), label
        except Exception:
            pass

    # 2) Daily EOD
    try:
        df_1d = yf.Ticker(ticker).history(period="10d", interval="1d", auto_adjust=True)
        if not df_1d.empty:
            price = float(df_1d["Close"].dropna().iloc[-1])
            last_idx = df_1d["Close"].dropna().index[-1]
            if ZoneInfo is not None:
                dt_et = (last_idx.tz_convert(ZoneInfo("America/New_York"))
                         if last_idx.tzinfo else last_idx.tz_localize(ZoneInfo("America/New_York")))
                label = f"EOD {dt_et.strftime('%Y-%m-%d')}"
            else:
                label = "EOD"
            return price, label
    except Exception:
        pass

    # 3) Fallback: fast_info (สุดท้ายจริง ๆ)
    try:
        p = yf.Ticker(ticker).fast_info.get("lastPrice", 0.0)
        return float(p) if p else 0.0, "SNAPSHOT"
    except Exception:
        return 0.0, "N/A"
# --- CHANGE END ---

# --- Data Fetching (combined & synced) ---
@st.cache_data(ttl=60)  # ลด TTL เพื่อให้ราคา LIVE สดขึ้น แต่ยัง cache พอสมควร
def fetch_all_data(configs: List[Dict], _clients_ref: Dict, start_date: str) -> Dict[str, tuple]:
    monitor_results: Dict[str, Tuple[pd.DataFrame, str, float, str]] = {}
    asset_results: Dict[str, float] = {}

    # สรุปสถานะตลาดหนึ่งครั้ง/ต่อรอบ
    now_et = _now_ny() if ZoneInfo is not None else dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    live_mode = is_us_market_open(now_et)

    def _safe_tz_to_bkk(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        try:
            if idx.tz is None:
                if ZoneInfo is not None:
                    return idx.tz_localize(dt.timezone.utc).tz_convert(ZoneInfo("Asia/Bangkok"))
                else:
                    return idx  # ถ้าไม่มี zoneinfo ก็ปล่อยแบบเดิม
            else:
                if ZoneInfo is not None:
                    return idx.tz_convert(ZoneInfo("Asia/Bangkok"))
                else:
                    return idx
        except Exception:
            return idx

    def fetch_monitor(asset_config: Dict[str, Any]):
        ticker = asset_config['ticker']
        try:
            monitor_field_config = asset_config['monitor_field']
            client = _clients_ref[monitor_field_config['channel_id']]
            field_num = monitor_field_config['field']

            # ดึง daily history เพื่อทำ action (ยังคงเหมือนเดิม)
            # ใช้ period 'max' ได้แต่หนัก เปลี่ยนเป็นเริ่มจาก start_date ถ้ามี
            tkr = yf.Ticker(ticker)
            if start_date:
                tickerData = tkr.history(start=start_date, auto_adjust=True)[['Close']].round(3)
            else:
                tickerData = tkr.history(period='5y', auto_adjust=True)[['Close']].round(3)

            # timezone → Asia/Bangkok เพื่อให้ UI สม่ำเสมอ
            if not tickerData.empty:
                tickerData.index = _safe_tz_to_bkk(tickerData.index)

            fx_js_str = "0"
            try:
                field_data = client.get_field_last(field=str(field_num))
                retrieved_val = json.loads(field_data).get(f"field{field_num}", None)
                if retrieved_val is not None:
                    fx_js_str = str(retrieved_val)
            except Exception:
                pass

            # เตรียม df action + dummy 5 แถว
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

            # --- CHANGE START: ใช้ synced quote + label ---
            price, asof_label = get_synced_quote(ticker, want_live=live_mode)
            # --- CHANGE END ---

            return ticker, (df.tail(7), fx_js_str, price, asof_label)
        except Exception as e:
            st.error(f"Error in Monitor for {ticker}: {str(e)}")
            return ticker, (pd.DataFrame(), "0", 0.0, "N/A")

    def fetch_asset(asset_config):
        ticker = asset_config['ticker']
        try:
            asset_conf = asset_config['asset_field']
            client = _clients_ref[asset_conf['channel_id']]
            field_name = asset_conf['field']
            data = client.get_field_last(field=field_name)
            return ticker, float(json.loads(data)[field_name])
        except Exception:
            return ticker, 0.0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(4, len(configs)*2)) as executor:
        monitor_futures = [executor.submit(fetch_monitor, asset) for asset in configs]
        for future in concurrent.futures.as_completed(monitor_futures):
            ticker, result = future.result()
            monitor_results[ticker] = result

        asset_futures = [executor.submit(fetch_asset, asset) for asset in configs]
        for future in concurrent.futures.as_completed(asset_futures):
            ticker, result = future.result()
            asset_results[ticker] = result

    return {'monitors': monitor_results, 'assets': asset_results}

# --- UI Components (unchanged) ---
def render_asset_inputs(configs: List[Dict], last_assets: Dict) -> Dict[str, float]:
    asset_inputs = {}
    cols = st.columns(len(configs))
    for i, config in enumerate(configs):
        with cols[i]:
            ticker = config['ticker']
            last_val = last_assets.get(ticker, 0.0)
            if config.get('option_config'):
                option_val = config['option_config']['base_value']
                label = config['option_config']['label']
                real_val = st.number_input(label, step=0.001, value=last_val, key=f"input_{ticker}_real")
                asset_inputs[ticker] = option_val + real_val
            else:
                label = f'{ticker}_ASSET'
                val = st.number_input(label, step=0.001, value=last_val, key=f"input_{ticker}_asset")
                asset_inputs[ticker] = val
    return asset_inputs

def render_asset_update_controls(configs: List[Dict], clients: Dict):
    with st.expander("Update Assets on ThingSpeak"):
        for config in configs:
            ticker = config['ticker']
            asset_conf = config['asset_field']
            field_name = asset_conf['field']

            if st.checkbox(f'@_{ticker}_ASSET', key=f'check_{ticker}'):
                add_val = st.number_input(f"New Value for {ticker}", step=0.001, value=0.0, key=f'input_{ticker}')
                if st.button(f"GO_{ticker}", key=f'btn_{ticker}'):
                    try:
                        client = clients[asset_conf['channel_id']]
                        client.update({field_name: add_val})
                        st.write(f"Updated {ticker} to: {add_val} on Channel {asset_conf['channel_id']}")
                        clear_all_caches()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to update {ticker}: {e}")

# --- CHANGE START: trading_section รับ asof_label ด้วย ---
def trading_section(config: Dict, asset_val: float, asset_last: float, df_data: pd.DataFrame,
                    current_price: float, asof_label: str, calc: Dict, nex: int, Nex_day_sell: int, clients: Dict):
# --- CHANGE END ---
    ticker = config['ticker']
    asset_conf = config['asset_field']
    field_name = asset_conf['field']

    def get_action_val():
        try:
            if df_data.empty or df_data.action.values[1 + nex] == "":
                return 0
            val = df_data.action.values[1 + nex]
            return 1 - val if Nex_day_sell == 1 else val
        except Exception:
            return 0

    action_val = get_action_val()
    limit_order = st.checkbox(f'Limit_Order_{ticker}', value=bool(action_val), key=f'limit_order_{ticker}')
    if not limit_order:
        return

    sell_calc, buy_calc = calc['sell'], calc['buy']
    st.write('sell', '    ', 'A', buy_calc[1], 'P', buy_calc[0], 'C', buy_calc[2])
    col1, col2, col3 = st.columns(3)
    if col3.checkbox(f'sell_match_{ticker}'):
        if col3.button(f"GO_SELL_{ticker}"):
            try:
                client = clients[asset_conf['channel_id']]
                new_asset_val = asset_last - buy_calc[1]
                client.update({field_name: new_asset_val})
                col3.write(f"Updated: {new_asset_val}")
                clear_all_caches()
                st.rerun()
            except Exception as e:
                st.error(f"Failed to SELL {ticker}: {e}")

    # ใช้ราคาที่ sync มาแล้ว + แสดง as-of
    try:
        if current_price > 0:
            pv = current_price * asset_val
            fix_value = config['fix_c']
            pl_value = pv - fix_value
            pl_color = "#a8d5a2" if pl_value >= 0 else "#fbb"
            st.markdown(
                f"Price: **{current_price:,.3f}** | Value: **{pv:,.2f}** | P/L (vs {fix_value:,}) : "
                f"<span style='color:{pl_color}; font-weight:bold;'>{pl_value:,.2f}</span>",
                unsafe_allow_html=True
            )
            st.caption(f"As of: {asof_label}")
        else:
            st.info(f"Price data for {ticker} is currently unavailable.")
    except Exception:
        st.warning(f"Could not process price data for {ticker}.")

    col4, col5, col6 = st.columns(3)
    st.write('buy', '   ', 'A', sell_calc[1], 'P', sell_calc[0], 'C', sell_calc[2])
    if col6.checkbox(f'buy_match_{ticker}'):
        if col6.button(f"GO_BUY_{ticker}"):
            try:
                client = clients[asset_conf['channel_id']]
                new_asset_val = asset_last + sell_calc[1]
                client.update({field_name: new_asset_val})
                col6.write(f"Updated: {new_asset_val}")
                clear_all_caches()
                st.rerun()
            except Exception as e:
                st.error(f"Failed to BUY {ticker}: {e}")

# --- Main Logic ---
all_data = fetch_all_data(ASSET_CONFIGS, THINGSPEAK_CLIENTS, GLOBAL_START_DATE)
monitor_data_all = all_data['monitors']
last_assets_all = all_data['assets']

# Stable Session State (unchanged)
if 'select_key' not in st.session_state:
    st.session_state.select_key = ""
if 'nex' not in st.session_state:
    st.session_state.nex = 0
if 'Nex_day_sell' not in st.session_state:
    st.session_state.Nex_day_sell = 0

tab1, tab2 = st.tabs(["📈 Monitor", "⚙️ Controls"])

with tab2: # (unchanged)
    Nex_day_ = st.checkbox('nex_day', value=(st.session_state.nex == 1))

    if Nex_day_:
        nex_col, Nex_day_sell_col, *_ = st.columns([1,1,3])
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

    asset_inputs = render_asset_inputs(ASSET_CONFIGS, last_assets_all)

    st.write("---")

    Start = st.checkbox('start')
    if Start:
        render_asset_update_controls(ASSET_CONFIGS, THINGSPEAK_CLIENTS)

with tab1: # (Main logic loop)
    selectbox_labels = {}
    ticker_actions = {}
    for config in ASSET_CONFIGS:
        ticker = config['ticker']
        # unpack 4 values: df, fx_js, price, asof
        df_data, fx_js_str, _, _ = monitor_data_all.get(ticker, (pd.DataFrame(), "0", 0.0, "N/A"))
        action_emoji, final_action_val = "", None
        try:
            if not df_data.empty and df_data.action.values[1 + nex] != "":
                raw_action = int(df_data.action.values[1 + nex])
                final_action_val = 1 - raw_action if Nex_day_sell == 1 else raw_action
                if final_action_val == 1: action_emoji = "🟢 "
                elif final_action_val == 0: action_emoji = "🔴 "
        except (IndexError, ValueError, TypeError):
            pass
        ticker_actions[ticker] = final_action_val
        selectbox_labels[ticker] = f"{action_emoji}{ticker} (f(x): {fx_js_str})"

    all_tickers = [config['ticker'] for config in ASSET_CONFIGS]
    selectbox_options = [""]
    if nex == 1:
        selectbox_options.extend(["Filter Buy Tickers", "Filter Sell Tickers"])
    selectbox_options.extend(all_tickers)

    if st.session_state.select_key not in selectbox_options:
        st.session_state.select_key = ""

    def format_selectbox_options(option_name):
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
        configs_to_display = [config for config in ASSET_CONFIGS if config['ticker'] in buy_tickers]
    elif selected_option == "Filter Sell Tickers":
        sell_tickers = {t for t, action in ticker_actions.items() if action == 0}
        configs_to_display = [config for config in ASSET_CONFIGS if config['ticker'] in sell_tickers]
    else:
        configs_to_display = [config for config in ASSET_CONFIGS if config['ticker'] == selected_option]

    calculations = {}
    for config in ASSET_CONFIGS:
        ticker = config['ticker']
        asset_value = asset_inputs.get(ticker, 0.0)
        fix_c = config['fix_c']
        calculations[ticker] = {
            'sell': sell(asset_value, fix_c=fix_c, Diff=x_2),
            'buy': buy(asset_value, fix_c=fix_c, Diff=x_2)
        }

    for config in configs_to_display:
        ticker = config['ticker']
        df_data, fx_js_str, last_price, asof_label = monitor_data_all.get(ticker, (pd.DataFrame(), "0", 0.0, "N/A"))
        asset_last = last_assets_all.get(ticker, 0.0)
        asset_val = asset_inputs.get(ticker, 0.0)
        calc = calculations.get(ticker, {})

        title_label = selectbox_labels.get(ticker, ticker)
        st.write(title_label)

        trading_section(config, asset_val, asset_last, df_data, last_price, asof_label, calc, nex, Nex_day_sell, THINGSPEAK_CLIENTS)

        with st.expander("Show Raw Data Action"):
            st.dataframe(df_data, use_container_width=True)

        st.write("_____")

if st.sidebar.button("RERUN"):
    clear_all_caches()
    st.rerun()
