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
from typing import List, Dict, Optional
import tenacity
import pytz
from datetime import time as dt_time

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

# --- (MODIFIED) Clear Caches Function (PRESERVE UI KEYS) ---
def clear_all_caches():
    # Clear caches
    st.cache_data.clear()
    st.cache_resource.clear()
    sell.cache_clear()
    buy.cache_clear()

    # Preserve key UI states to prevent reset on data refresh
    ui_state_keys_to_preserve = {'select_key', 'nex', 'Nex_day_sell'}
    keys_to_delete = [k for k in list(st.session_state.keys()) if k not in ui_state_keys_to_preserve]
    for key in keys_to_delete:
        try:
            del st.session_state[key]
        except Exception:
            pass

    st.success("🗑️ Data caches cleared! UI state preserved.")

# --- (NEW) Helper function to rerun app while keeping the selectbox selection ---
def rerun_keep_selection(ticker: str):
    """
    Sets a temporary key in session_state and reruns the app.
    This ensures the selectbox maintains its selection on the next run.
    """
    st.session_state["_pending_select_key"] = ticker
    st.rerun()

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

# --- Price Fetching with Retry (unchanged) ---
@st.cache_data(ttl=300)
@tenacity.retry(wait=tenacity.wait_fixed(2), stop=tenacity.stop_after_attempt(3))
def get_cached_price(ticker: str) -> float:
    try:
        return float(yf.Ticker(ticker).fast_info['lastPrice'])
    except Exception:
        return 0.0

# --- Helper: current date/time in New York (EXTENDED) ---
@st.cache_data(ttl=60)
def get_current_ny_date() -> datetime.date:
    ny_tz = pytz.timezone('America/New_York')
    return datetime.datetime.now(ny_tz).date()

@st.cache_data(ttl=5)
def get_current_ny_dt() -> datetime.datetime:
    ny_tz = pytz.timezone('America/New_York')
    return datetime.datetime.now(ny_tz)

def market_is_open(ny_dt: datetime.datetime) -> bool:
    # Simple regular-hours check (excludes holidays)
    if ny_dt.weekday() >= 5:  # 5=Sat, 6=Sun
        return False
    t = ny_dt.time()
    return dt_time(9, 30) <= t <= dt_time(16, 0)

# --- Data Fetching (unchanged, with minor robust tz fix) ---
@st.cache_data(ttl=300)
def fetch_all_data(configs: List[Dict], _clients_ref: Dict, start_date: str) -> Dict[str, tuple]:
    monitor_results = {}
    asset_results = {}

    def fetch_monitor(asset_config):
        ticker = asset_config['ticker']
        try:
            monitor_field_config = asset_config['monitor_field']
            client = _clients_ref[monitor_field_config['channel_id']]
            field_num = monitor_field_config['field']

            tickerData = yf.Ticker(ticker).history(period='max')[['Close']].round(3)
            try:
                tickerData.index = tickerData.index.tz_convert('Asia/Bangkok')
            except TypeError:
                tickerData.index = tickerData.index.tz_localize('UTC').tz_convert('Asia/Bangkok')

            if start_date:
                tickerData = tickerData[tickerData.index >= start_date]

            last_data_date = tickerData.index[-1].date() if not tickerData.empty else None

            fx_js_str = "0"
            try:
                field_data = client.get_field_last(field=str(field_num))
                retrieved_val = json.loads(field_data)[f"field{field_num}"]
                if retrieved_val is not None:
                    fx_js_str = str(retrieved_val)
            except Exception:
                pass

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

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(configs)) as executor:
        monitor_futures = [executor.submit(fetch_monitor, asset) for asset in configs]
        for future in concurrent.futures.as_completed(monitor_futures):
            ticker, result = future.result()
            monitor_results[ticker] = result

        asset_futures = [executor.submit(fetch_asset, asset) for asset in configs]
        for future in concurrent.futures.as_completed(asset_futures):
            ticker, result = future.result()
            asset_results[ticker] = result

    return {'monitors': monitor_results, 'assets': asset_results}

# --- Initialize persistent state for Semi-Auto ---
if 'trade_log' not in st.session_state:
    st.session_state.trade_log: List[Dict] = []
if 'auto_exec_memory' not in st.session_state:
    # per-ticker memory: {ticker: {last_signature: str, last_time: datetime}}, plus '_daily': {(date,ticker,side): notional}
    st.session_state.auto_exec_memory = {}

# --- UI Components ---

def render_asset_inputs(configs: List[Dict], last_assets: Dict) -> Dict[str, float]:
    asset_inputs = {}
    cols = st.columns(len(configs))
    for i, config in enumerate(configs):
        with cols[i]:
            ticker = config['ticker']
            last_val = last_assets.get(ticker, 0.0)
            if config.get('option_config'):
                raw_label = config['option_config']['label']
            else:
                raw_label = ticker
            display_label = raw_label
            help_text = ""
            split_pos = raw_label.find('(')
            if split_pos != -1:
                display_label = raw_label[:split_pos].strip()
                help_text = raw_label[split_pos:].strip()
            else:
                help_text = "(NULL)"
            if config.get('option_config'):
                option_val = config['option_config']['base_value']
                real_val = st.number_input(
                    label=display_label, help=help_text,
                    step=0.001, value=last_val, key=f"input_{ticker}_real"
                )
                asset_inputs[ticker] = option_val + real_val
            else:
                val = st.number_input(
                    label=display_label, help=help_text,
                    step=0.001, value=last_val, key=f"input_{ticker}_asset"
                )
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
                        # keep selection after update
                        rerun_keep_selection(st.session_state.get("select_key", ""))
                    except Exception as e:
                        st.error(f"Failed to update {ticker}: {e}")

# --- (MODIFIED) trading_section with Semi-Auto Pro Mode ---
# NOTE: UI/outputs are preserved; we only add optional auto-execution with guardrails

def trading_section(config: Dict, asset_val: float, asset_last: float, df_data: pd.DataFrame, calc: Dict, nex: int, Nex_day_sell: int, clients: Dict, last_data_date: Optional[datetime.date], auto_settings: Dict):
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

    sell_calc, buy_calc = calc['sell'], calc['buy']
    st.write('sell', '    ', 'A', buy_calc[1], 'P', buy_calc[0], 'C', buy_calc[2])

    # --- GO_SELL manual control (unchanged) ---
    col1, col2, col3 = st.columns(3)
    if col3.checkbox(f'sell_match_{ticker}'):
        if col3.button(f"GO_SELL_{ticker}"):
            try:
                client = clients[asset_conf['channel_id']]
                new_asset_val = asset_last - buy_calc[1]
                client.update({field_name: new_asset_val})
                col3.write(f"Updated: {new_asset_val}")
                # log
                st.session_state.trade_log.append({
                    'ts_ny': get_current_ny_dt().strftime('%Y-%m-%d %H:%M:%S'),
                    'ticker': ticker,
                    'side': 'SELL',
                    'qty': int(buy_calc[1]),
                    'price': float(buy_calc[0]),
                    'new_asset_val': float(new_asset_val),
                    'mode': 'manual'
                })
                clear_all_caches()
                rerun_keep_selection(ticker)
            except Exception as e:
                st.error(f"Failed to SELL {ticker}: {e}")

    # --- price / P&L display (unchanged) ---
    try:
        current_price = get_cached_price(ticker)
        if current_price > 0:
            pv = current_price * asset_val
            fix_value = config['fix_c']
            pl_value = pv - fix_value
            pl_color = "#a8d5a2" if pl_value >= 0 else "#fbb"
            st.markdown(
                f"Price: **{current_price:,.3f}** | Value: **{pv:,.2f}** | P/L (vs {fix_value:,}) : <span style='color:{pl_color}; font-weight:bold;'>{pl_value:,.2f}</span>",
                unsafe_allow_html=True
            )
        else:
            st.info(f"Price data for {ticker} is currently unavailable.")
    except Exception:
        st.warning(f"Could not retrieve price data for {ticker}.")

    # --- BUY manual control (unchanged) ---
    col4, col5, col6 = st.columns(3)
    st.write('buy', '   ', 'A', sell_calc[1], 'P', sell_calc[0], 'C', sell_calc[2])
    if col6.checkbox(f'buy_match_{ticker}'):
        if col6.button(f"GO_BUY_{ticker}"):
            try:
                client = clients[asset_conf['channel_id']]
                new_asset_val = asset_last + sell_calc[1]
                client.update({field_name: new_asset_val})
                col6.write(f"Updated: {new_asset_val}")
                # log
                st.session_state.trade_log.append({
                    'ts_ny': get_current_ny_dt().strftime('%Y-%m-%d %H:%M:%S'),
                    'ticker': ticker,
                    'side': 'BUY',
                    'qty': int(sell_calc[1]),
                    'price': float(sell_calc[0]),
                    'new_asset_val': float(new_asset_val),
                    'mode': 'manual'
                })
                clear_all_caches()
                rerun_keep_selection(ticker)
            except Exception as e:
                st.error(f"Failed to BUY {ticker}: {e}")

    # --- PRO MODE: Optional Semi-Auto execution with guardrails ---
    if auto_settings.get('enabled', False) and has_signal:
        ny_now = get_current_ny_dt()
        if auto_settings.get('market_hours_only', True) and not market_is_open(ny_now):
            st.caption("⏸️ Pro Mode armed, waiting for regular US market hours (NY 09:30–16:00).")
            return

        # Build an idempotent signature for this signal
        signature = f"{ticker}|{str(last_data_date)}|{nex}|{Nex_day_sell}|{action_val}"
        mem = st.session_state.auto_exec_memory.get(ticker, {})

        # Cooldown check
        last_time = mem.get('last_time')
        cooldown_min = int(auto_settings.get('cooldown_minutes', 15))
        if last_time and (ny_now - last_time).total_seconds() < cooldown_min * 60:
            st.caption(f"⏳ Pro Mode cooldown ({cooldown_min}m) active; skipping auto-exec.")
            return

        # Already executed this exact signal recently?
        if mem.get('last_signature') == signature:
            st.caption("✅ Pro Mode: this signal already executed; skipping duplicate.")
            return

        # Decide side/qty/price exactly as manual controls
        side = 'SELL' if action_val == 0 else 'BUY'
        if side == 'SELL':
            qty = int(buy_calc[1])
            price = float(buy_calc[0])
            new_asset_val = asset_last - qty
        else:
            qty = int(sell_calc[1])
            price = float(sell_calc[0])
            new_asset_val = asset_last + qty

        # Apply max_qty guardrail
        max_qty = int(auto_settings.get('max_qty', 0))
        if max_qty > 0 and qty > max_qty:
            qty = max_qty
            # Recompute new_asset_val according to reduced qty
            if side == 'SELL':
                new_asset_val = asset_last - qty
            else:
                new_asset_val = asset_last + qty

        if qty <= 0:
            st.caption("⚠️ Pro Mode: qty <= 0 after guardrails; skipping.")
            return

        # Daily notional guardrail
        notional = float(qty) * float(price)
        daily_limit = float(auto_settings.get('daily_max_notional', 0.0))
        if daily_limit > 0:
            day_key = ny_now.strftime('%Y-%m-%d')
            daily_map = st.session_state.auto_exec_memory.get('_daily', {})
            used = float(daily_map.get((day_key, ticker, side), 0.0))
            if used + notional > daily_limit:
                st.caption(f"🧯 Pro Mode: daily {side} notional limit reached ({used:,.0f}/{daily_limit:,.0f}); skipping.")
                return

        # Execute
        executed = False
        err_msg = None
        try:
            if not auto_settings.get('dry_run', True):
                client = clients[asset_conf['channel_id']]
                client.update({field_name: new_asset_val})
            executed = True
        except Exception as e:
            err_msg = str(e)
            st.error(f"Auto-exec failed for {ticker}: {err_msg}")

        # Log attempt
        st.session_state.trade_log.append({
            'ts_ny': ny_now.strftime('%Y-%m-%d %H:%M:%S'),
            'ticker': ticker,
            'side': side,
            'qty': int(qty),
            'price': float(price),
            'new_asset_val': float(new_asset_val),
            'mode': 'auto_pro',
            'dry_run': bool(auto_settings.get('dry_run', True)),
            'status': 'executed' if executed else f'error: {err_msg}'
        })

        if executed:
            # update memory & daily map
            mem['last_signature'] = signature
            mem['last_time'] = ny_now
            st.session_state.auto_exec_memory[ticker] = mem
            if daily_limit > 0:
                daily_map = st.session_state.auto_exec_memory.get('_daily', {})
                day_key = ny_now.strftime('%Y-%m-%d')
                used = float(daily_map.get((day_key, ticker, side), 0.0))
                daily_map[(day_key, ticker, side)] = used + notional
                st.session_state.auto_exec_memory['_daily'] = daily_map

            note = "(DRY RUN) " if auto_settings.get('dry_run', True) else ""
            st.success(f"🤖 Pro Mode {note}{side} {ticker} qty={qty} @ {price:.2f} → asset={new_asset_val}")

            # On real execution, refresh & keep selection
            if not auto_settings.get('dry_run', True):
                clear_all_caches()
                rerun_keep_selection(ticker)

# --- Main Logic ---
all_data = fetch_all_data(ASSET_CONFIGS, THINGSPEAK_CLIENTS, GLOBAL_START_DATE)
monitor_data_all = all_data['monitors']
last_assets_all = all_data['assets']

# Stable Session State Initialization
if 'select_key' not in st.session_state:
    st.session_state.select_key = ""
if 'nex' not in st.session_state:
    st.session_state.nex = 0
if 'Nex_day_sell' not in st.session_state:
    st.session_state.Nex_day_sell = 0

# --- (NEW) BOOTSTRAP selection BEFORE creating any widgets ---
pending = st.session_state.pop("_pending_select_key", None)
if pending:
    st.session_state.select_key = pending

# Tabs
tab1, tab2 = st.tabs(["📈 Monitor", "⚙️ Controls"])

with tab2:
    st.subheader("Pro Mode — Semi-Automatic Execution")
    left, right = st.columns([2,1])
    with left:
        auto_enabled = st.checkbox('Enable Semi‑Auto (Pro Mode)', value=False, key='auto_enabled')
        dry_run = st.checkbox('Dry Run (no ThingSpeak updates)', value=True, key='auto_dry_run')
        mkt_only = st.checkbox('Restrict to US market hours (NY 09:30–16:00)', value=True, key='auto_market_hours_only')
        cooldown_min = st.number_input('Cooldown minutes per ticker', min_value=0, max_value=10080, value=15, step=1, key='auto_cooldown')
        max_qty = st.number_input('Max quantity per trade (0 = unlimited)', min_value=0, value=0, step=1, key='auto_max_qty')
        daily_notional = st.number_input('Daily max notional per ticker (0 = unlimited)', min_value=0.0, value=0.0, step=100.0, key='auto_daily_notional')
        st.caption("Guardrails help prevent repeat/oversized orders. Pro Mode mirrors the existing BUY/SELL logic — UI and calculations remain unchanged.")

    with right:
        if st.button('Reset Pro Mode memory'):
            st.session_state.auto_exec_memory = {}
            st.success('Cleared Pro Mode memory (cooldowns, duplicates, daily counters).')

    st.write("---")

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

    st.write("---")
    with st.expander("Audit Log (CSV download)", expanded=False):
        if st.session_state.trade_log:
            df_log = pd.DataFrame(st.session_state.trade_log)
            st.dataframe(df_log, use_container_width=True)
            csv = df_log.to_csv(index=False).encode('utf-8')
            st.download_button("Download trade_log.csv", data=csv, file_name="trade_log.csv", mime="text/csv")
        else:
            st.caption("No trades logged yet.")

with tab1:
    current_ny_date = get_current_ny_date()

    selectbox_labels = {}
    ticker_actions = {}
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
                    if final_action_val == 1: action_emoji = "🟢 "
                    elif final_action_val == 0: action_emoji = "🔴 "
            except (IndexError, ValueError, TypeError):
                pass
        ticker_actions[ticker] = final_action_val
        selectbox_labels[ticker] = f"{action_emoji}{ticker} (f(x): {fx_js_str})"

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

    # Prepare calc map once
    calculations = {}
    for config in ASSET_CONFIGS:
        ticker = config['ticker']
        asset_value = asset_inputs.get(ticker, 0.0)
        fix_c = config['fix_c']
        calculations[ticker] = {
            'sell': sell(asset_value, fix_c=fix_c, Diff=x_2),
            'buy': buy(asset_value, fix_c=fix_c, Diff=x_2)
        }

    # Bundle auto settings
    auto_settings = {
        'enabled': st.session_state.get('auto_enabled', False),
        'dry_run': st.session_state.get('auto_dry_run', True),
        'market_hours_only': st.session_state.get('auto_market_hours_only', True),
        'cooldown_minutes': st.session_state.get('auto_cooldown', 15),
        'max_qty': st.session_state.get('auto_max_qty', 0),
        'daily_max_notional': st.session_state.get('auto_daily_notional', 0.0),
    }

    for config in configs_to_display:
        ticker = config['ticker']
        df_data, fx_js_str, last_data_date = monitor_data_all.get(ticker, (pd.DataFrame(), "0", None))
        asset_last = last_assets_all.get(ticker, 0.0)
        asset_val = asset_inputs.get(ticker, 0.0)
        calc = calculations.get(ticker, {})

        title_label = selectbox_labels.get(ticker, ticker)
        st.write(title_label)

        trading_section(
            config,
            asset_val,
            asset_last,
            df_data,
            calc,
            st.session_state.nex,
            st.session_state.Nex_day_sell,
            THINGSPEAK_CLIENTS,
            last_data_date,
            auto_settings,
        )

        with st.expander("Show Raw Data Action"):
            st.dataframe(df_data, use_container_width=True)
        st.write("_____")

# Sidebar rerun (unchanged + keep selection)
if st.sidebar.button("RERUN"):
    current_selection = st.session_state.get("select_key", "")
    clear_all_caches()
    if current_selection in [c['ticker'] for c in ASSET_CONFIGS]:
        rerun_keep_selection(current_selection)
    else:
        st.rerun()
