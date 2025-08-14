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

try:
    import thingspeak
    THINGSPEAK_AVAILABLE = True
except Exception:
    thingspeak = None
    THINGSPEAK_AVAILABLE = False

st.set_page_config(page_title="Monitor (Semi‑Auto Pro)", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

STATE_FILE = "portfolio_state.json"
BLOTTER_FILE = "trade_blotter.csv"
CONFIG_FILE = "monitor_config.json"

class SimulationTracer:
    def __init__(self, encoded_string: str):
        self.encoded_string: str = str(encoded_string)
        self._decode_and_set_attributes()

    def _decode_and_set_attributes(self):
        encoded_string = self.encoded_string
        if not encoded_string.isdigit():
            self.action_length = 0
            self.mutation_rate = 0
            self.dna_seed = 0
            self.mutation_seeds = []
            self.mutation_rate_float = 0.0
            return
        decoded_numbers = []
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

def save_blotter(df: pd.DataFrame):
    try:
        df.to_csv(BLOTTER_FILE, index=False)
        st.toast("🧾 Blotter saved")
    except Exception as e:
        st.error(f"Failed to save blotter: {e}")


# --------------------------------- Mode & external clients -----------------------------------
MODE = st.sidebar.radio("Mode", ["Offline (no‑API)", "Online (ThingSpeak)"] if THINGSPEAK_AVAILABLE else ["Offline (no‑API)"])

@st.cache_resource(show_spinner=False)
def get_thingspeak_clients(configs: List[Dict]) -> Dict[int, "thingspeak.Channel"]:
    clients: Dict[int, "thingspeak.Channel"] = {}
    if not THINGSPEAK_AVAILABLE:
        return clients
    unique_channels = set()
    for config in configs:
        mon_conf = config["monitor_field"]
        unique_channels.add((mon_conf["channel_id"], mon_conf["api_key"]))
        asset_conf = config["asset_field"]
        unique_channels.add((asset_conf["channel_id"], asset_conf["api_key"]))
    for channel_id, api_key in unique_channels:
        try:
            clients[channel_id] = thingspeak.Channel(channel_id, api_key, fmt="json")
        except Exception as e:
            st.error(f"Failed to create ThingSpeak client for Channel {channel_id}: {e}")
    return clients

THINGSPEAK_CLIENTS = get_thingspeak_clients(ASSET_CONFIGS)

# ----------------------------------- Cached helpers ------------------------------------------
@st.cache_data(ttl=60)
def get_current_ny_date() -> datetime.date:
    ny_tz = pytz.timezone("America/New_York")
    return datetime.datetime.now(ny_tz).date()

@st.cache_data(ttl=300)
@tenacity.retry(wait=tenacity.wait_fixed(2), stop=tenacity.stop_after_attempt(3))
def get_cached_price(ticker: str) -> float:
    try:
        return float(yf.Ticker(ticker).fast_info["lastPrice"])  # type: ignore
    except Exception:
        return 0.0

@st.cache_data(ttl=300)
def fetch_all_data(configs: List[Dict], _clients_ref: Dict, start_date: Optional[str]) -> Dict[str, Dict[str, Tuple]]:
    monitor_results: Dict[str, Tuple[pd.DataFrame, str, Optional[datetime.date]]] = {}
    asset_results: Dict[str, float] = {}

    def fetch_monitor(asset_config):
        ticker = asset_config["ticker"]
        try:
            monitor_field_config = asset_config["monitor_field"]
            client = _clients_ref.get(monitor_field_config["channel_id"]) if THINGSPEAK_AVAILABLE else None
            field_num = monitor_field_config["field"]

            tickerData = yf.Ticker(ticker).history(period="max")[['Close']].round(3)
            try:
                tickerData.index = tickerData.index.tz_convert('Asia/Bangkok')
            except TypeError:
                tickerData.index = tickerData.index.tz_localize('UTC').tz_convert('Asia/Bangkok')
            if start_date:
                tickerData = tickerData[tickerData.index >= start_date]
            last_data_date = tickerData.index[-1].date() if not tickerData.empty else None

            fx_js_str = "0"
            if client is not None:
                try:
                    field_data = client.get_field_last(field=str(field_num))  # type: ignore
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

    def fetch_asset_online(asset_config):
        ticker = asset_config['ticker']
        try:
            if not THINGSPEAK_AVAILABLE:
                return ticker, 0.0
            asset_conf = asset_config['asset_field']
            client = _clients_ref[asset_conf['channel_id']]
            field_name = asset_conf['field']
            data = client.get_field_last(field=field_name)
            return ticker, float(json.loads(data)[field_name])
        except Exception:
            return ticker, 0.0

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(configs)) as ex:
        # monitors
        for f in concurrent.futures.as_completed([ex.submit(fetch_monitor, a) for a in configs]):
            ticker, result = f.result()
            monitor_results[ticker] = result
        # assets
        if MODE.startswith("Online"):
            for f in concurrent.futures.as_completed([ex.submit(fetch_asset_online, a) for a in configs]):
                ticker, v = f.result()
                asset_results[ticker] = v
        else:
            asset_results = load_portfolio_state()

    return {"monitors": monitor_results, "assets": asset_results}

# --------------------- Basic sizing functions (unchanged math) --------------------------------
@lru_cache(maxsize=128)
def sell(asset, fix_c: float = 1500, Diff: float = 60):
    if asset == 0:
        return 0.0, 0, 0.0
    unit_price = round((fix_c - Diff) / asset, 2)
    adjust_qty = round(abs(asset * unit_price - fix_c) / unit_price) if unit_price != 0 else 0
    total = round(asset * unit_price + adjust_qty * unit_price, 2)
    return unit_price, adjust_qty, total

@lru_cache(maxsize=128)
def buy(asset, fix_c: float = 1500, Diff: float = 60):
    if asset == 0:
        return 0.0, 0, 0.0
    unit_price = round((fix_c + Diff) / asset, 2)
    adjust_qty = round(abs(asset * unit_price - fix_c) / unit_price) if unit_price != 0 else 0
    total = round(asset * unit_price - adjust_qty * unit_price, 2)
    return unit_price, adjust_qty, total

# --------------------------- Session State (stable) -------------------------------------------
if 'select_key' not in st.session_state:
    st.session_state.select_key = ""
if 'nex' not in st.session_state:
    st.session_state.nex = 0
if 'Nex_day_sell' not in st.session_state:
    st.session_state.Nex_day_sell = 0
if 'tickets' not in st.session_state:
    st.session_state.tickets: List[Dict] = []

# rerun helper that preserves selection
def rerun_keep_selection(ticker: str):
    st.session_state["_pending_select_key"] = ticker
    st.rerun()

# bootstrap selection on rerun
pending = st.session_state.pop("_pending_select_key", None)
if pending:
    st.session_state.select_key = pending

# ------------------------------------ Layout --------------------------------------------------
all_data = fetch_all_data(ASSET_CONFIGS, THINGSPEAK_CLIENTS, GLOBAL_START_DATE)
monitor_data_all = all_data['monitors']
last_assets_all = all_data['assets']

# ===== Sidebar Controls =====
st.sidebar.markdown("### ⚙️ Controls")
Nex_day_ = st.sidebar.checkbox('nex_day', value=(st.session_state.nex == 1))
if Nex_day_:
    c1, c2 = st.sidebar.columns(2)
    if c1.button("Nex_day"):
        st.session_state.nex = 1
        st.session_state.Nex_day_sell = 0
    if c2.button("Nex_day_sell"):
        st.session_state.nex = 1
        st.session_state.Nex_day_sell = 1
else:
    st.session_state.nex = 0
    st.session_state.Nex_day_sell = 0

x_2 = st.sidebar.number_input('Diff', step=1, value=60)

# offline asset editing
if MODE.startswith("Offline"):
    st.sidebar.markdown("### 🧮 Asset values (local)")
    new_state = {}
    for cfg in ASSET_CONFIGS:
        t = cfg['ticker']
        last = float(last_assets_all.get(t, 0.0))
        new_state[t] = st.sidebar.number_input(f"{t}", value=float(last), step=1.0, key=f"asset_{t}")
    csave, cload = st.sidebar.columns(2)
    if csave.button("Save state"):
        save_portfolio_state(new_state)
        st.cache_data.clear()
        st.rerun()
    if cload.button("Reload state"):
        st.cache_data.clear()
        st.rerun()
else:
    # online update controls (ThingSpeak) kept for backwards‑compat
    if THINGSPEAK_AVAILABLE:
        with st.sidebar.expander("Update Assets on ThingSpeak"):
            for config in ASSET_CONFIGS:
                ticker = config['ticker']
                asset_conf = config['asset_field']
                field_name = asset_conf['field']
                if st.checkbox(f'@_{ticker}_ASSET', key=f'check_{ticker}'):
                    add_val = st.number_input(f"New Value for {ticker}", step=0.001, value=0.0, key=f'input_{ticker}')
                    if st.button(f"GO_{ticker}", key=f'btn_{ticker}'):
                        try:
                            client = THINGSPEAK_CLIENTS[asset_conf['channel_id']]
                            client.update({field_name: add_val})
                            st.success(f"Updated {ticker} to: {add_val} on Channel {asset_conf['channel_id']}")
                            st.cache_data.clear(); st.cache_resource.clear()
                            rerun_keep_selection(st.session_state.get("select_key", ""))
                        except Exception as e:
                            st.error(f"Failed to update {ticker}: {e}")

# ===== Main Tabs =====
tab1, tab2 = st.tabs(["📈 Monitor", "🧾 Blotter & Tickets"])

with tab1:
    current_ny_date = get_current_ny_date()

    # build labels with signal emoji
    selectbox_labels: Dict[str, str] = {}
    ticker_actions: Dict[str, Optional[int]] = {}
    nex = st.session_state.nex
    Nex_day_sell = st.session_state.Nex_day_sell

    for config in ASSET_CONFIGS:
        ticker = config['ticker']
        df_data, fx_js_str, last_data_date = monitor_data_all.get(ticker, (pd.DataFrame(), "0", None))
        action_emoji, final_action_val = "", None
        if nex == 0 and last_data_date and last_data_date < current_ny_date:
            action_emoji = "🟡 "  # stale (yesterday)
        else:
            try:
                if not df_data.empty and df_data.action.values[1 + nex] != "":
                    raw_action = int(df_data.action.values[1 + nex])
                    final_action_val = 1 - raw_action if Nex_day_sell == 1 else raw_action
                    if final_action_val == 1:
                        action_emoji = "🟢 "
                    elif final_action_val == 0:
                        action_emoji = "🔴 "
            except (IndexError, ValueError, TypeError):
                pass
        ticker_actions[ticker] = final_action_val
        selectbox_labels[ticker] = f"{action_emoji}{ticker} (f(x): {fx_js_str})"

    all_tickers = [c['ticker'] for c in ASSET_CONFIGS]
    selectbox_options = [""]
    if nex == 1:
        selectbox_options.extend(["Filter Buy Tickers", "Filter Sell Tickers"])
    selectbox_options.extend(all_tickers)

    if st.session_state.select_key not in selectbox_options:
        st.session_state.select_key = ""

    def _fmt(opt: str):
        if opt in ["", "Filter Buy Tickers", "Filter Sell Tickers"]:
            return "Show All" if opt == "" else opt
        return selectbox_labels.get(opt, opt).split(' (f(x):')[0]

    st.selectbox("Select Ticker to View:", options=selectbox_options, format_func=_fmt, key="select_key")
    st.write("___")

    selected = st.session_state.select_key
    if selected == "":
        configs_to_display = ASSET_CONFIGS
    elif selected == "Filter Buy Tickers":
        buy_tickers = {t for t, a in ticker_actions.items() if a == 1}
        configs_to_display = [cfg for cfg in ASSET_CONFIGS if cfg['ticker'] in buy_tickers]
    elif selected == "Filter Sell Tickers":
        sell_tickers = {t for t, a in ticker_actions.items() if a == 0}
        configs_to_display = [cfg for cfg in ASSET_CONFIGS if cfg['ticker'] in sell_tickers]
    else:
        configs_to_display = [cfg for cfg in ASSET_CONFIGS if cfg['ticker'] == selected]

    # precompute calculations for all tickers
    calculations = {}
    for cfg in ASSET_CONFIGS:
        t = cfg['ticker']
        asset_value = float(last_assets_all.get(t, 0.0)) if MODE.startswith("Online") else float(load_portfolio_state().get(t, 0.0))
        fix_c = cfg['fix_c']
        calculations[t] = {
            'sell': sell(asset_value, fix_c=fix_c, Diff=x_2),
            'buy': buy(asset_value, fix_c=fix_c, Diff=x_2),
        }

    # --- Trading card per ticker (semi‑auto, no broker API) ---
    for cfg in configs_to_display:
        tkr = cfg['ticker']
        df_data, fx_js_str, _ = monitor_data_all.get(tkr, (pd.DataFrame(), "0", None))
        asset_val = float(last_assets_all.get(tkr, 0.0)) if MODE.startswith("Online") else float(load_portfolio_state().get(tkr, 0.0))
        calc = calculations.get(tkr, {})
        title_label = selectbox_labels.get(tkr, tkr)
        st.subheader(title_label)

        # Determine signal
        def get_action_val() -> Optional[int]:
            try:
                if df_data.empty or df_data.action.values[1 + nex] == "":
                    return None
                raw_action = int(df_data.action.values[1 + nex])
                return 1 - raw_action if Nex_day_sell == 1 else raw_action
            except Exception:
                return None
        action_val = get_action_val()
        has_signal = action_val is not None

        # Price / PV / P&L
        try:
            px = get_cached_price(tkr)
            if px > 0:
                fix_value = cfg['fix_c']
                pv = px * asset_val
                pl_value = pv - fix_value
                pl_color = "#a8d5a2" if pl_value >= 0 else "#fbb"
                st.markdown(
                    f"Price: **{px:,.3f}** | Value: **{pv:,.2f}** | P/L (vs {fix_value:,}) : "
                    f"<span style='color:{pl_color}; font-weight:bold;'>{pl_value:,.2f}</span>",
                    unsafe_allow_html=True,
                )
        except Exception:
            st.info("Price data currently unavailable.")

        sell_calc, buy_calc = calc.get('sell', (0.0, 0, 0.0)), calc.get('buy', (0.0, 0, 0.0))
        st.caption(f"Sizing → SELL path: A {buy_calc[1]} | P {buy_calc[0]} | C {buy_calc[2]}  ·  BUY path: A {sell_calc[1]} | P {sell_calc[0]} | C {sell_calc[2]}")

        # Semi‑auto ticket creation (no broker API)
        create = st.checkbox(f"Create Ticket for {tkr}", value=has_signal, key=f"limit_order_{tkr}")
        if create:
            suggested_side = "BUY" if (action_val == 1) else ("SELL" if action_val == 0 else "BUY")
            if suggested_side == "BUY":
                suggested_qty = int(sell_calc[1]); suggested_price = float(sell_calc[0])
            else:
                suggested_qty = int(buy_calc[1]); suggested_price = float(buy_calc[0])

            with st.form(f"ticket_form_{tkr}", clear_on_submit=False):
                c1, c2, c3, c4 = st.columns([1,1,1,2])
                side = c1.selectbox("Side", ["BUY", "SELL"], index=["BUY","SELL"].index(suggested_side))
                qty = int(c2.number_input("Qty", value=max(0, suggested_qty)))
                limit_price = float(c3.number_input("Limit", value=max(0.0, suggested_price), format="%.2f"))
                note = c4.text_input("Note (rule/why)", value=f"f(x)={fx_js_str}; nex={nex}; flip={Nex_day_sell}")
                c5, c6 = st.columns([1,1])
                signal = c5.text_input("Signal tag", value="auto")
                good_till = c6.date_input("Good‑till", value=datetime.date.today())
                # sanity guardrails
                max_notional = float(st.number_input("Max notional (guard)", value=100000.0, help="Prevent fat‑finger orders"))
                notional = qty * limit_price
                if notional > max_notional:
                    st.warning(f"Notional {notional:,.2f} exceeds guard {max_notional:,.2f}")
                submitted = st.form_submit_button("➕ Add Ticket to Blotter")
                if submitted:
                    now = datetime.datetime.now().isoformat(timespec='seconds')
                    row = {
                        "ts": now,
                        "ticker": tkr,
                        "side": side,
                        "qty": qty,
                        "limit_price": limit_price,
                        "status": "NEW",
                        "note": note,
                        "signal": signal,
                        "suggested_by": "MonitorApp",
                        "exec_ts": "",
                        "exec_price": np.nan,
                        "fees": 0.0,
                        "slippage": 0.0,
                    }
                    blotter = load_blotter()
                    blotter = pd.concat([blotter, pd.DataFrame([row])], ignore_index=True)
                    save_blotter(blotter)
                    st.success(f"Ticket added: {side} {qty} {tkr} @ {limit_price}")

        with st.expander("Show Raw Data Action"):
            st.dataframe(df_data, use_container_width=True)
        st.write("___")

with tab2:
    st.markdown("#### Trade Blotter")
    blotter_df = load_blotter()
    st.caption("Edit status/exec fields below; click **Save changes** to persist.")

    editable = st.data_editor(
        blotter_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "status": st.column_config.SelectboxColumn(options=["NEW","WORKING","CANCELLED","FILLED"], required=False),
            "exec_ts": st.column_config.DatetimeColumn(step=60),
            "exec_price": st.column_config.NumberColumn(format="%.4f"),
            "fees": st.column_config.NumberColumn(format="%.4f"),
            "slippage": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    c1, c2, c3 = st.columns([1,1,2])
    if c1.button("💾 Save changes"):
        save_blotter(editable)
        st.cache_data.clear()
        st.rerun()
    ticket_txt = editable.tail(1).to_csv(index=False)
    c2.download_button("⬇️ Download blotter CSV", data=editable.to_csv(index=False), file_name=BLOTTER_FILE, mime="text/csv")
    c3.download_button("⬇️ Download latest ticket (.csv)", data=ticket_txt, file_name="latest_ticket.csv", mime="text/csv")

# --- Global Rerun ---
if st.sidebar.button("RERUN"):
    st.cache_data.clear(); st.cache_resource.clear(); st.rerun()
