# app_semi_auto_minimal.py
# Minimal Patch: UI/logic/output เหมือนเดิม 100% + (Confirm 2 steps, Ticket .txt, Dry-Run)
# ใช้งานได้ทั้งมี/ไม่มี thingspeak (optional)

import streamlit as st
import numpy as np
import datetime
import pandas as pd
import yfinance as yf
import json
from functools import lru_cache
import concurrent.futures
import os
from typing import List, Dict, Optional, Tuple
import tenacity
import pytz
import uuid

# --- Optional dependency (ThingSpeak) ---------------------------------------------------------
try:
    import thingspeak  # type: ignore
    THINGSPEAK_AVAILABLE = True
except Exception:
    thingspeak = None  # type: ignore
    THINGSPEAK_AVAILABLE = False

# ---------------------------------- App Setup -------------------------------------------------
st.set_page_config(page_title="Monitor (Semi-Auto Minimal)", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

# --- Minimal guardrails (ปรับได้ในไฟล์ config) -----------------------------------------------
DEFAULT_GUARDS = {
    "dry_run": True,        # ไม่อัปเดต ThingSpeak จนกว่าจะปิดเอง
    "require_confirm": True # ต้องกด Confirm Executed ก่อน จึงจะ Apply Local ได้
}

# ---------------------------------- Ticket helper ---------------------------------------------
def build_ticket_text(*, ticker: str, side: str, qty: int, limit_price: float, note: str) -> str:
    tid = str(uuid.uuid4())[:8].upper()
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    txt = (
        f"TICKET {tid}\n"
        f"TS     {ts}\n"
        f"TICKER {ticker}\n"
        f"SIDE   {side}\n"
        f"QTY    {qty}\n"
        f"LIMIT  {limit_price:.4f}\n"
        f"NOTE   {note}\n"
    )
    return tid, txt

# ---------------------------------- SimulationTracer ------------------------------------------
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
                length_of_number = int(encoded_string[idx]); idx += 1
                number_str = encoded_string[idx: idx + length_of_number]; idx += length_of_number
                decoded_numbers.append(int(number_str))
        except (IndexError, ValueError):
            pass

        if len(decoded_numbers) < 3:
            self.action_length = 0; self.mutation_rate = 0; self.dna_seed = 0
            self.mutation_seeds = []; self.mutation_rate_float = 0.0
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

# ---------------------------------- Config -----------------------------------------------------
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
GUARDS = {**DEFAULT_GUARDS, **CONFIG_DATA.get('order_guardrails', {})}

if not ASSET_CONFIGS:
    st.error("No 'assets' list found in monitor_config.json")
    st.stop()

# ---------------------------------- ThingSpeak Clients ----------------------------------------
@st.cache_resource
def get_thingspeak_clients(configs: List[Dict]) -> Dict[int, "thingspeak.Channel"]:
    clients = {}
    if not THINGSPEAK_AVAILABLE:
        return clients
    uniq = set()
    for cfg in configs:
        m = cfg['monitor_field']; uniq.add((m['channel_id'], m['api_key']))
        a = cfg['asset_field'];   uniq.add((a['channel_id'], a['api_key']))
    for channel_id, api_key in uniq:
        try:
            clients[channel_id] = thingspeak.Channel(channel_id, api_key, fmt='json')
        except Exception as e:
            st.error(f"Failed to create client for Channel ID {channel_id}: {e}")
    return clients

THINGSPEAK_CLIENTS = get_thingspeak_clients(ASSET_CONFIGS)

# ---------------------------------- Cache mgmt ------------------------------------------------
def clear_all_caches():
    st.cache_data.clear(); st.cache_resource.clear()
    sell.cache_clear(); buy.cache_clear()
    preserve = {'select_key', 'nex', 'Nex_day_sell'}
    for k in list(st.session_state.keys()):
        if k not in preserve:
            try: del st.session_state[k]
            except: pass
    st.success("🗑️ Cleared caches (UI preserved)")

def rerun_keep_selection(ticker: str):
    st.session_state["_pending_select_key"] = ticker
    st.rerun()

# ---------------------------------- Calcs (เดิม) ----------------------------------------------
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

# ---------------------------------- Price / Date ----------------------------------------------
@st.cache_data(ttl=300)
@tenacity.retry(wait=tenacity.wait_fixed(2), stop=tenacity.stop_after_attempt(3))
def get_cached_price(ticker: str) -> float:
    try:
        return float(yf.Ticker(ticker).fast_info['lastPrice'])
    except Exception:
        return 0.0

@st.cache_data(ttl=60)
def get_current_ny_date() -> datetime.date:
    ny_tz = pytz.timezone('America/New_York')
    return datetime.datetime.now(ny_tz).date()

# ---------------------------------- Fetching (เดิม) -------------------------------------------
@st.cache_data(ttl=300)
def fetch_all_data(configs: List[Dict], _clients_ref: Dict, start_date: str) -> Dict[str, tuple]:
    monitor_results, asset_results = {}, {}

    def fetch_monitor(asset_config):
        ticker = asset_config['ticker']
        try:
            monitor_field_config = asset_config['monitor_field']
            client = _clients_ref.get(monitor_field_config['channel_id'])
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
                if client is not None:
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
                n = min(len(df), len(final_actions))
                if n > 0:
                    action_col_idx = df.columns.get_loc('action')
                    df.iloc[0:n, action_col_idx] = final_actions[0:n]
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
            client = _clients_ref.get(asset_conf['channel_id'])
            field_name = asset_conf['field']
            if client is None:
                return ticker, 0.0
            data = client.get_field_last(field=field_name)
            return ticker, float(json.loads(data)[field_name])
        except Exception:
            return ticker, 0.0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(configs))) as ex:
        for fut in concurrent.futures.as_completed([ex.submit(fetch_monitor, a) for a in configs]):
            t, res = fut.result(); monitor_results[t] = res
        for fut in concurrent.futures.as_completed([ex.submit(fetch_asset, a) for a in configs]):
            t, res = fut.result(); asset_results[t] = res

    return {'monitors': monitor_results, 'assets': asset_results}

# ---------------------------------- UI helpers (เดิม) -----------------------------------------
def render_asset_inputs(configs: List[Dict], last_assets: Dict) -> Dict[str, float]:
    asset_inputs = {}; cols = st.columns(len(configs))
    for i, config in enumerate(configs):
        with cols[i]:
            ticker = config['ticker']
            last_val = last_assets.get(ticker, 0.0)
            raw_label = config['option_config']['label'] if config.get('option_config') else ticker
            display_label, help_text = raw_label, ""
            split_pos = raw_label.find('(')
            if split_pos != -1:
                display_label = raw_label[:split_pos].strip()
                help_text = raw_label[split_pos:].strip()
            else:
                help_text = "(NULL)"
            if config.get('option_config'):
                option_val = config['option_config']['base_value']
                real_val = st.number_input(label=display_label, help=help_text, step=0.001,
                                           value=float(last_val), key=f"input_{ticker}_real")
                asset_inputs[ticker] = option_val + real_val
            else:
                val = st.number_input(label=display_label, help=help_text, step=0.001,
                                      value=float(last_val), key=f"input_{ticker}_asset")
                asset_inputs[ticker] = val
    return asset_inputs

def render_asset_update_controls(configs: List[Dict], clients: Dict):
    with st.expander("Update Assets on ThingSpeak"):
        st.toggle("Dry Run (ไม่อัปเดต ThingSpeak)", value=GUARDS["dry_run"], key="dry_run_toggle")
        GUARDS["dry_run"] = st.session_state["dry_run_toggle"]
        for config in configs:
            ticker = config['ticker']
            asset_conf = config['asset_field']
            field_name = asset_conf['field']
            if st.checkbox(f'@_{ticker}_ASSET', key=f'check_{ticker}'):
                add_val = st.number_input(f"New Value for {ticker}", step=0.001, value=0.0, key=f'input_{ticker}')
                if st.button(f"GO_{ticker}", key=f'btn_{ticker}'):
                    try:
                        client = clients.get(asset_conf['channel_id'])
                        if client is not None and not GUARDS["dry_run"]:
                            client.update({field_name: add_val})
                        st.write(f"Updated {ticker} to: {add_val} (DryRun={GUARDS['dry_run']})")
                        clear_all_caches(); rerun_keep_selection(st.session_state.get("select_key", ""))
                    except Exception as e:
                        st.error(f"Failed to update {ticker}: {e}")

# ---------------------------------- trading_section (Minimal Patch) ----------------------------
def trading_section(config: Dict, asset_val: float, asset_last: float, df_data: pd.DataFrame,
                    calc: Dict, nex: int, Nex_day_sell: int, clients: Dict):
    ticker = config['ticker']
    asset_conf = config['asset_field']
    field_name = asset_conf['field']
    client = clients.get(asset_conf['channel_id'])

    # ดึงสัญญาณตามของเดิม
    def get_action_val() -> Optional[int]:
        try:
            if df_data.empty or df_data.action.values[1 + nex] == "":
                return None
            raw_action = int(df_data.action.values[1 + nex])
            return 1 - raw_action if Nex_day_sell == 1 else raw_action
        except (IndexError, ValueError, TypeError):
            return None

    action_val = get_action_val()
    has_signal = action_val is not None
    limit_order_checked = st.checkbox(f'Limit_Order_{ticker}', value=has_signal, key=f'limit_order_{ticker}')
    if not limit_order_checked:
        return

    # แสดงผลแบบเดิม
    sell_calc, buy_calc = calc['sell'], calc['buy']
    st.write('sell', '    ', 'A', buy_calc[1], 'P', buy_calc[0], 'C', buy_calc[2])
    col1, col2, col3 = st.columns(3)

    # ราคาปัจจุบัน + P/L (เดิม)
    try:
        current_price = get_cached_price(ticker)
        if current_price > 0:
            pv = current_price * asset_val
            fix_value = config['fix_c']
            pl_value = pv - fix_value
            pl_color = "#a8d5a2" if pl_value >= 0 else "#fbb"
            st.markdown(
                f"Price: **{current_price:,.3f}** | Value: **{pv:,.2f}** | "
                f"P/L (vs {fix_value:,}) : "
                f"<span style='color:{pl_color}; font-weight:bold;'>{pl_value:,.2f}</span>",
                unsafe_allow_html=True
            )
        else:
            st.info(f"Price data for {ticker} is currently unavailable.")
    except Exception:
        st.warning(f"Could not retrieve price data for {ticker}.")

    # ----------------- NEW: Ticket + Confirm (แต่ UI หลักเดิมคงไว้) -----------------
    st.markdown("**🧾 Ticket (ไม่ยิงคำสั่งโบรกเกอร์)**")
    # ใช้ราคาจากคอลัมน์เดิมเป็น limit แนะนำ
    suggested_side = "BUY" if action_val == 1 else ("SELL" if action_val == 0 else "BUY")
    suggested_qty = sell_calc[1] if suggested_side == "BUY" else buy_calc[1]
    suggested_limit = sell_calc[0] if suggested_side == "BUY" else buy_calc[0]

    cA, cB, cC, cD = st.columns([1,1,1,3])
    side = cA.selectbox("Side", ["BUY", "SELL"], index=0 if suggested_side=="BUY" else 1, key=f"side_{ticker}")
    qty = int(cB.number_input("Qty", min_value=0, step=1, value=int(suggested_qty), key=f"qty_{ticker}"))
    limit_price = float(cC.number_input("Limit", step=0.01, value=float(suggested_limit), key=f"limit_{ticker}"))
    note = cD.text_input("Note", value="", key=f"note_{ticker}")

    # สร้าง ticket (ไฟล์ txt)
    if st.button(f"Build Ticket ({ticker})", key=f"build_{ticker}"):
        tid, txt = build_ticket_text(ticker=ticker, side=side, qty=qty, limit_price=limit_price, note=note)
        st.session_state[f"ticket_txt_{ticker}"] = txt
        st.success(f"Ticket built: {tid}")

    ticket_txt = st.session_state.get(f"ticket_txt_{ticker}")
    if ticket_txt:
        st.code(ticket_txt)
        st.download_button("⬇️ Download Ticket (.txt)",
                           data=ticket_txt.encode("utf-8"),
                           file_name=f"ticket_{ticker}.txt",
                           mime="text/plain",
                           key=f"dl_{ticker}")

    # ต้อง Confirm ก่อน ถึงจะ Apply ได้ (Minimal patch)
    if GUARDS["require_confirm"]:
        st.checkbox("✅ I executed this order at my broker (Confirm Executed)",
                    key=f"confirm_exec_{ticker}")

    # ----------------- ปุ่มเดิม (GO_SELL/GO_BUY) แต่ล็อกด้วย Confirm -----------------
    if col3.checkbox(f'sell_match_{ticker}'):
        if col3.button(f"GO_SELL_{ticker}", key=f"go_sell_{ticker}"):
            if GUARDS["require_confirm"] and not st.session_state.get(f"confirm_exec_{ticker}", False):
                st.error("กรุณา Confirm Executed ก่อน (เพื่อกันพลาด)")
            else:
                try:
                    new_asset_val = asset_last - buy_calc[1]
                    if not GUARDS["dry_run"] and client is not None:
                        client.update({field_name: new_asset_val})
                    col3.write(f"Updated: {new_asset_val} (DryRun={GUARDS['dry_run']})")
                    clear_all_caches(); rerun_keep_selection(ticker)
                except Exception as e:
                    st.error(f"Failed to SELL {ticker}: {e}")

    st.write('buy', '   ', 'A', sell_calc[1], 'P', sell_calc[0], 'C', sell_calc[2])
    col4, col5, col6 = st.columns(3)
    if col6.checkbox(f'buy_match_{ticker}'):
        if col6.button(f"GO_BUY_{ticker}", key=f"go_buy_{ticker}"):
            if GUARDS["require_confirm"] and not st.session_state.get(f"confirm_exec_{ticker}", False):
                st.error("กรุณา Confirm Executed ก่อน (เพื่อกันพลาด)")
            else:
                try:
                    new_asset_val = asset_last + sell_calc[1]
                    if not GUARDS["dry_run"] and client is not None:
                        client.update({field_name: new_asset_val})
                    col6.write(f"Updated: {new_asset_val} (DryRun={GUARDS['dry_run']})")
                    clear_all_caches(); rerun_keep_selection(ticker)
                except Exception as e:
                    st.error(f"Failed to BUY {ticker}: {e}")

# ---------------------------------- Main (เดิม) -----------------------------------------------
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

# BOOTSTRAP selection
pending = st.session_state.pop("_pending_select_key", None)
if pending:
    st.session_state.select_key = pending

tab1, tab2 = st.tabs(["📈 Monitor", "⚙️ Controls"])

with tab2:
    Nex_day_ = st.checkbox('nex_day', value=(st.session_state.nex == 1))
    if Nex_day_:
        nex_col, Nex_day_sell_col, *_ = st.columns([1,1,3])
        if nex_col.button("Nex_day"):
            st.session_state.nex = 1; st.session_state.Nex_day_sell = 0
        if Nex_day_sell_col.button("Nex_day_sell"):
            st.session_state.nex = 1; st.session_state.Nex_day_sell = 1
    else:
        st.session_state.nex = 0; st.session_state.Nex_day_sell = 0

    st.write("---")
    x_2 = st.number_input('Diff', step=1, value=60)
    st.write("---")
    asset_inputs = render_asset_inputs(ASSET_CONFIGS, last_assets_all)
    st.write("---")
    Start = st.checkbox('start')
    if Start:
        render_asset_update_controls(ASSET_CONFIGS, THINGSPEAK_CLIENTS)

with tab1:
    current_ny_date = get_current_ny_date()

    selectbox_labels = {}; ticker_actions = {}
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

    all_tickers = [cfg['ticker'] for cfg in ASSET_CONFIGS]
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

    st.selectbox("Select Ticker to View:", options=selectbox_options,
                 format_func=format_selectbox_options, key="select_key")
    st.write("_____")

    selected_option = st.session_state.select_key
    if selected_option == "":
        configs_to_display = ASSET_CONFIGS
    elif selected_option == "Filter Buy Tickers":
        buy_tickers = {t for t, a in ticker_actions.items() if a == 1}
        configs_to_display = [cfg for cfg in ASSET_CONFIGS if cfg['ticker'] in buy_tickers]
    elif selected_option == "Filter Sell Tickers":
        sell_tickers = {t for t, a in ticker_actions.items() if a == 0}
        configs_to_display = [cfg for cfg in ASSET_CONFIGS if cfg['ticker'] in sell_tickers]
    else:
        configs_to_display = [cfg for cfg in ASSET_CONFIGS if cfg['ticker'] == selected_option]

    calculations = {}
    for cfg in ASSET_CONFIGS:
        t = cfg['ticker']; asset_value = asset_inputs.get(t, 0.0); fix_c = cfg['fix_c']
        calculations[t] = {'sell': sell(asset_value, fix_c=fix_c, Diff=x_2),
                           'buy':  buy(asset_value, fix_c=fix_c, Diff=x_2)}

    for cfg in configs_to_display:
        t = cfg['ticker']
        df_data, fx_js_str, _ = monitor_data_all.get(t, (pd.DataFrame(), "0", None))
        asset_last = last_assets_all.get(t, 0.0)
        asset_val = asset_inputs.get(t, 0.0)
        calc = calculations.get(t, {})

        title_label = selectbox_labels.get(t, t)
        st.write(title_label)

        trading_section(cfg, asset_val, asset_last, df_data, calc,
                        st.session_state.nex, st.session_state.Nex_day_sell, THINGSPEAK_CLIENTS)

        with st.expander("Show Raw Data Action"):
            st.dataframe(df_data, use_container_width=True)
        st.write("_____")

# Sidebar RERUN (เดิม)
if st.sidebar.button("RERUN"):
    current_selection = st.session_state.get("select_key", "")
    clear_all_caches()
    tickers = [c['ticker'] for c in ASSET_CONFIGS]
    rerun_keep_selection(current_selection) if current_selection in tickers else st.rerun()
