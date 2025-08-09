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
from collections import Counter
from zoneinfo import ZoneInfo

st.set_page_config(page_title="Monitor", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

MARKET_TZ = ZoneInfo("America/New_York")
LOCAL_TZ  = ZoneInfo("Asia/Bangkok")

# --- SimulationTracer Class (unchanged except annotation) ---
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

# --- Configuration Loading ---
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

# --- ThingSpeak Clients ---
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

# --- Clear Caches ---
def clear_all_caches():
    st.cache_data.clear()
    st.cache_resource.clear()
    sell.cache_clear()
    buy.cache_clear()
    ui_state_keys_to_preserve = ['select_key', 'nex', 'Nex_day_sell', 'session_override_mode']
    keys_to_delete = [k for k in list(st.session_state.keys()) if k not in ui_state_keys_to_preserve]
    for key in keys_to_delete:
        del st.session_state[key]
    st.success("🗑️ Data caches cleared! UI state preserved.")

# --- Calculation Utils ---
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

# --- Price Fetching with Retry ---
@st.cache_data(ttl=300)
@tenacity.retry(wait=tenacity.wait_fixed(2), stop=tenacity.stop_after_attempt(3))
def get_cached_price(ticker: str) -> Tuple[float, dt.datetime]:
    try:
        fi = yf.Ticker(ticker).fast_info
        price = float(fi.get('lastPrice', 0.0) or 0.0)
        # best-effort timestamp
        rmt = fi.get('regularMarketTime')  # epoch seconds (YZ), may be None
        if rmt is not None:
            asof = dt.datetime.fromtimestamp(int(rmt), tz=MARKET_TZ)
        else:
            asof = dt.datetime.now(tz=MARKET_TZ)
        return price, asof
    except Exception:
        return 0.0, dt.datetime.now(tz=MARKET_TZ)

# --- Data Fetching (return full DF + last session date in ET) ---
def _safe_set_tz_index_to_et(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    if idx.tz is None:
        df.index = idx.tz_localize(dt.timezone.utc).tz_convert(MARKET_TZ)
    else:
        df.index = idx.tz_convert(MARKET_TZ)
    return df

@st.cache_data(ttl=300)
def fetch_all_data(configs: List[Dict], _clients_ref: Dict, start_date: str) -> Dict[str, Any]:
    monitor_results: Dict[str, Dict[str, Any]] = {}
    asset_results: Dict[str, float] = {}

    def fetch_monitor(asset_config: Dict) -> Tuple[str, Dict[str, Any]]:
        ticker = asset_config['ticker']
        try:
            monitor_field_config = asset_config['monitor_field']
            client = _clients_ref[monitor_field_config['channel_id']]
            field_num = str(monitor_field_config['field'])

            # 1) Pull daily history
            hist = yf.Ticker(ticker).history(period='max', interval='1d', auto_adjust=False)[['Close']].round(3)
            if hist.empty:
                return ticker, {'df': pd.DataFrame(), 'fx': "0", 'last_et_date': None}

            hist = _safe_set_tz_index_to_et(hist)

            if start_date:
                # interpret start_date as local day; convert to ET midnight
                try:
                    sd = pd.to_datetime(start_date).tz_localize(LOCAL_TZ).astimezone(MARKET_TZ)
                    hist = hist[hist.index >= sd]
                except Exception:
                    pass

            # 2) Read fx string (SimulationTracer code)
            fx_js_str = "0"
            try:
                field_data = client.get_field_last(field=field_num)
                j = json.loads(field_data)
                retrieved_val = j.get(f"field{int(field_num)}")
                if retrieved_val is not None:
                    fx_js_str = str(retrieved_val)
            except Exception:
                pass

            df = hist.copy()
            df['DateET'] = df.index
            df['DateLocal'] = df.index.tz_convert(LOCAL_TZ)
            df['index'] = range(len(df))
            df['action'] = ""

            # 3) Fill actions
            try:
                tracer = SimulationTracer(encoded_string=fx_js_str)
                final_actions = tracer.run()
                num_to_assign = min(len(df), len(final_actions))
                if num_to_assign > 0:
                    df.loc[df.index[:num_to_assign], 'action'] = final_actions[:num_to_assign]
            except Exception as e:
                st.warning(f"Tracer Error for {ticker}: {e}")

            last_et_date = df['DateET'].dt.date.iloc[-1]

            return ticker, {'df': df, 'fx': fx_js_str, 'last_et_date': last_et_date}
        except Exception as e:
            st.error(f"Error in Monitor for {ticker}: {str(e)}")
            return ticker, {'df': pd.DataFrame(), 'fx': "0", 'last_et_date': None}

    def fetch_asset(asset_config: Dict) -> Tuple[str, float]:
        ticker = asset_config['ticker']
        try:
            asset_conf = asset_config['asset_field']
            client = _clients_ref[asset_conf['channel_id']]
            field_name = asset_conf['field']
            data = client.get_field_last(field=field_name)
            return ticker, float(json.loads(data)[field_name])
        except Exception:
            return ticker, 0.0

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, max(1, len(configs)))) as executor:
        monitor_futures = [executor.submit(fetch_monitor, asset) for asset in configs]
        for future in concurrent.futures.as_completed(monitor_futures):
            ticker, result = future.result()
            monitor_results[ticker] = result

        asset_futures = [executor.submit(fetch_asset, asset) for asset in configs]
        for future in concurrent.futures.as_completed(asset_futures):
            ticker, result = future.result()
            asset_results[ticker] = result

    return {'monitors': monitor_results, 'assets': asset_results}

# --- Session Alignment Helpers ---
def compute_consensus_session_date(monitor_meta: Dict[str, Dict[str, Any]]) -> Tuple[dt.date, float]:
    dates = [v['last_et_date'] for v in monitor_meta.values() if v.get('last_et_date') is not None]
    if not dates:
        # fallback: “yesterday” in ET
        return (dt.datetime.now(tz=MARKET_TZ) - dt.timedelta(days=1)).date(), 0.0
    cnt = Counter(dates)
    consensus_date, freq = cnt.most_common(1)[0]
    coverage = freq / len(dates)
    return consensus_date, coverage

def filter_df_to_session(df: pd.DataFrame, session_date_et: dt.date) -> pd.DataFrame:
    if df.empty:
        return df
    mask = df['DateET'].dt.date <= session_date_et
    out = df.loc[mask].copy()
    if out.empty:
        return df.tail(7).copy()  # safety fallback
    out = out.tail(7).copy()
    out['index'] = range(len(out))  # keep numeric index used by UI
    return out

# --- UI Components (reused) ---
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

def trading_section(config: Dict, asset_val: float, asset_last: float, df_data: pd.DataFrame,
                    calc: Dict, nex: int, Nex_day_sell: int, clients: Dict, stale_note: str):
    ticker = config['ticker']
    asset_conf = config['asset_field']
    field_name = asset_conf['field']

    def get_action_val():
        try:
            if df_data.empty or df_data.action.values[1 + nex] == "":
                return 0
            val = int(df_data.action.values[1 + nex])
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

    try:
        current_price, asof = get_cached_price(ticker)
        if current_price > 0:
            pv = current_price * asset_val
            fix_value = config['fix_c']
            pl_value = pv - fix_value
            pl_color = "#a8d5a2" if pl_value >= 0 else "#fbb"
            st.markdown(
                f"Price: **{current_price:,.3f}** "
                f"| As-of (ET): **{asof.strftime('%Y-%m-%d %H:%M')}** "
                f"{stale_note}"
                f"| Value: **{pv:,.2f}** "
                f"| P/L (vs {fix_value:,}) : <span style='color:{pl_color}; font-weight:bold;'>{pl_value:,.2f}</span>",
                unsafe_allow_html=True
            )
        else:
            st.info(f"Price data for {ticker} is currently unavailable.")
    except Exception:
        st.warning(f"Could not retrieve price data for {ticker}.")

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
monitor_data_all: Dict[str, Dict[str, Any]] = all_data['monitors']
last_assets_all = all_data['assets']

# Stable Session State
st.session_state.setdefault('select_key', "")
st.session_state.setdefault('nex', 0)
st.session_state.setdefault('Nex_day_sell', 0)
st.session_state.setdefault('session_override_mode', 'Auto (Consensus)')  # Auto | Prev Close | Latest

tab1, tab2 = st.tabs(["📈 Monitor", "⚙️ Controls"])

with tab2:
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

    # Session Lock UI
    consensus_date, coverage = compute_consensus_session_date(monitor_data_all)
    st.markdown(f"**Auto Session (ET):** {consensus_date}  ·  Coverage: {coverage:.0%}")

    mode = st.selectbox(
        "Session Lock Override",
        ["Auto (Consensus)", "Prev Close (All)", "Latest Available"],
        index=["Auto (Consensus)", "Prev Close (All)", "Latest Available"].index(st.session_state.session_override_mode)
    )
    st.session_state.session_override_mode = mode

    st.caption("Auto = ใช้วันทีมีจำนวน ticker ตรงกันมากสุด / Prev Close = ใช้วันต่ำสุดร่วม / Latest = ใช้วันสูงสุดร่วม")

    asset_inputs = render_asset_inputs(ASSET_CONFIGS, last_assets_all)

    st.write("---")
    Start = st.checkbox('start')
    if Start:
        render_asset_update_controls(ASSET_CONFIGS, THINGSPEAK_CLIENTS)

with tab1:
    # Decide session_date per override
    cons_date, coverage = compute_consensus_session_date(monitor_data_all)
    all_dates = [m['last_et_date'] for m in monitor_data_all.values() if m['last_et_date'] is not None]
    if all_dates:
        min_date = min(all_dates)
        max_date = max(all_dates)
    else:
        now_et = dt.datetime.now(tz=MARKET_TZ).date()
        min_date = max_date = cons_date if cons_date else now_et

    if st.session_state.session_override_mode == "Prev Close (All)":
        session_date = min_date
    elif st.session_state.session_override_mode == "Latest Available":
        session_date = max_date
    else:
        session_date = cons_date

    # Build select labels with session-aligned data
    selectbox_labels = {}
    ticker_actions = {}
    prepared_df_by_ticker: Dict[str, pd.DataFrame] = {}

    for config in ASSET_CONFIGS:
        ticker = config['ticker']
        meta = monitor_data_all.get(ticker, {'df': pd.DataFrame(), 'fx': "0", 'last_et_date': None})
        df_full = meta['df']
        fx_js_str = meta['fx']
        last_date = meta['last_et_date']

        df_view = filter_df_to_session(df_full, session_date)
        prepared_df_by_ticker[ticker] = df_view

        # compute action after alignment
        action_emoji, final_action_val = "", None
        try:
            # same position as original: row 1+nex from tail(7)
            if not df_view.empty and df_view.action.values[1 + st.session_state.nex] != "":
                raw_action = int(df_view.action.values[1 + st.session_state.nex])
                final_action_val = 1 - raw_action if st.session_state.Nex_day_sell == 1 else raw_action
                if final_action_val == 1: action_emoji = "🟢 "
                elif final_action_val == 0: action_emoji = "🔴 "
        except Exception:
            pass

        stale_tag = ""
        if last_date is not None and last_date < session_date:
            stale_tag = f" 🟡 (lag {last_date})"

        ticker_actions[ticker] = final_action_val
        selectbox_labels[ticker] = f"{action_emoji}{ticker} (f(x): {fx_js_str}){stale_tag}"

    all_tickers = [config['ticker'] for config in ASSET_CONFIGS]
    selectbox_options = [""]
    if st.session_state.nex == 1:
        selectbox_options.extend(["Filter Buy Tickers", "Filter Sell Tickers"])
    selectbox_options.extend(all_tickers)

    if st.session_state.select_key not in selectbox_options:
        st.session_state.select_key = ""

    def format_selectbox_options(option_name):
        if option_name in ["", "Filter Buy Tickers", "Filter Sell Tickers"]:
            return "Show All" if option_name == "" else option_name
        return selectbox_labels.get(option_name, option_name).split(' (f(x):')[0]

    st.selectbox(
        f"Select Ticker to View  ·  Session(ET) = {session_date}",
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

    # Pre-calc buy/sell per ticker (unchanged)
    calculations = {}
    for config in ASSET_CONFIGS:
        ticker = config['ticker']
        asset_value = asset_inputs.get(ticker, 0.0)
        fix_c = config['fix_c']
        calculations[ticker] = {
            'sell': sell(asset_value, fix_c=fix_c, Diff=x_2),
            'buy': buy(asset_value, fix_c=fix_c, Diff=x_2)
        }

    # Render each selected
    for config in configs_to_display:
        ticker = config['ticker']
        df_data = prepared_df_by_ticker.get(ticker, pd.DataFrame())
        asset_last = last_assets_all.get(ticker, 0.0)
        asset_val = asset_inputs.get(ticker, 0.0)
        calc = calculations.get(ticker, {})
        title_label = selectbox_labels.get(ticker, ticker)

        # stale note for price line
        last_date = monitor_data_all.get(ticker, {}).get('last_et_date')
        stale_note = ""
        if last_date is not None and last_date < session_date:
            stale_note = f"| <span style='color:#E0A800;font-weight:bold;'>STALE vs session</span> "

        st.write(title_label)
        trading_section(config, asset_val, asset_last, df_data, calc, st.session_state.nex, st.session_state.Nex_day_sell, THINGSPEAK_CLIENTS, stale_note)

        with st.expander("Show Raw Data Action (Session-aligned)"):
            st.dataframe(df_data, use_container_width=True)

        st.write("_____")

if st.sidebar.button("RERUN"):
    clear_all_caches()
    st.rerun()
