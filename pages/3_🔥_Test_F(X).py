# streamlit_app_semi_auto_pro.py
# Semi-Auto Trading Monitor (Pro) — no broker API, with guardrails, tickets, logs, and local sync
# Requirements: streamlit, numpy, pandas, yfinance, tenacity, pytz
# Optional: thingspeak (if you use remote fields). If unavailable, app runs in "Offline" mode.

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
import time
import uuid
import hashlib

# ------------------------------ Optional ThingSpeak -------------------------------------------
try:
    import thingspeak  # type: ignore
    THINGSPEAK_AVAILABLE = True
except Exception:
    thingspeak = None  # type: ignore
    THINGSPEAK_AVAILABLE = False

# ------------------------------ App Setup ------------------------------------------------------
st.set_page_config(page_title="Monitor (Semi-Auto Pro)", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

STATE_FILE = "portfolio_state.json"          # local persistence (if you want later)
ORDERS_DIR = "orders"
ORDERS_LOG_CSV = os.path.join(ORDERS_DIR, "orders_log.csv")

# ------------------------------ Guardrails (defaults; overridable by config) -------------------
DEFAULT_GUARDS = {
    "dry_run": True,             # ไม่ยิง update ไป ThingSpeak จนกว่าจะปิด
    "max_order_value": 50_000,   # USD (or quote ccy) ต่อคำสั่ง
    "max_qty_change": 100_000,   # จำนวนสูงสุดต่อคำสั่ง
    "max_position_value": 250_000,  # มูลค่า position สูงสุดหลังทำรายการ
    "slip_bps": 20,              # สร้าง Limit จากราคาตลาด: 20 bps = 0.20%
    "cooldown_sec": 4,           # กันดับเบิลคลิก “Confirm”
    "price_sanity_pct": 30,      # ป้องกันส่งคำสั่งหากราคาหลุดจากวันก่อนเกิน X%
    "require_two_clicks": True   # ต้อง Build Ticket -> Confirm ถึงจะ Apply ได้
}

# ------------------------------ Utils: Files/Dirs ---------------------------------------------
def ensure_dirs():
    os.makedirs(ORDERS_DIR, exist_ok=True)
    if not os.path.exists(ORDERS_LOG_CSV):
        pd.DataFrame(columns=[
            "id", "ts", "ticker", "side", "qty", "limit_price",
            "source_price", "slip_bps", "note", "confirmed", "applied_locally"
        ]).to_csv(ORDERS_LOG_CSV, index=False)

ensure_dirs()

def append_order_log(row: Dict):
    df = pd.DataFrame([row])
    header = not os.path.exists(ORDERS_LOG_CSV) or os.path.getsize(ORDERS_LOG_CSV) == 0
    df.to_csv(ORDERS_LOG_CSV, mode="a", index=False, header=header)

def checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]

# ------------------------------ SimulationTracer (unchanged) -----------------------------------
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

# ------------------------------ Configuration Loading ------------------------------------------
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

# ------------------------------ ThingSpeak Clients ---------------------------------------------
@st.cache_resource
def get_thingspeak_clients(configs: List[Dict]) -> Dict[int, "thingspeak.Channel"]:
    clients = {}
    if not THINGSPEAK_AVAILABLE:
        return clients
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

# ------------------------------ Clear Caches (preserve UI) ------------------------------------
def clear_all_caches():
    st.cache_data.clear()
    st.cache_resource.clear()
    sell.cache_clear()
    buy.cache_clear()
    ui_state_keys_to_preserve = {'select_key', 'nex', 'Nex_day_sell'}
    keys_to_delete = [k for k in list(st.session_state.keys()) if k not in ui_state_keys_to_preserve]
    for key in keys_to_delete:
        try:
            del st.session_state[key]
        except Exception:
            pass
    st.success("🗑️ Data caches cleared! UI state preserved.")

def rerun_keep_selection(ticker: str):
    st.session_state["_pending_select_key"] = ticker
    st.rerun()

# ------------------------------ Calc Utils (unchanged) ----------------------------------------
@lru_cache(maxsize=128)
def sell(asset, fix_c=1500, Diff=60):
    if asset == 0: return 0, 0, 0
    unit_price = round((fix_c - Diff) / asset, 2)
    adjust_qty = round(abs(asset * unit_price - fix_c) / unit_price) if unit_price != 0 else 0
    total = round(asset * unit_price + adjust_qty * unit_price, 2)
    return unit_price, adjust_qty, total  # (price, qty, total)

@lru_cache(maxsize=128)
def buy(asset, fix_c=1500, Diff=60):
    if asset == 0: return 0, 0, 0
    unit_price = round((fix_c + Diff) / asset, 2)
    adjust_qty = round(abs(asset * unit_price - fix_c) / unit_price) if unit_price != 0 else 0
    total = round(asset * unit_price - adjust_qty * unit_price, 2)
    return unit_price, adjust_qty, total  # (price, qty, total)

# ------------------------------ Price / Date Helpers -------------------------------------------
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

# ------------------------------ Data Fetching --------------------------------------------------
@st.cache_data(ttl=300)
def fetch_all_data(configs: List[Dict], _clients_ref: Dict, start_date: str) -> Dict[str, tuple]:
    monitor_results = {}
    asset_results = {}

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
            client = _clients_ref.get(asset_conf['channel_id'])
            field_name = asset_conf['field']
            if client is None:
                return ticker, 0.0
            data = client.get_field_last(field=field_name)
            return ticker, float(json.loads(data)[field_name])
        except Exception:
            return ticker, 0.0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(configs))) as executor:
        monitor_futures = [executor.submit(fetch_monitor, asset) for asset in configs]
        for future in concurrent.futures.as_completed(monitor_futures):
            ticker, result = future.result()
            monitor_results[ticker] = result

        asset_futures = [executor.submit(fetch_asset, asset) for asset in configs]
        for future in concurrent.futures.as_completed(asset_futures):
            ticker, result = future.result()
            asset_results[ticker] = result

    return {'monitors': monitor_results, 'assets': asset_results}

# ------------------------------ Order Ticket (NEW) ---------------------------------------------
def build_ticket(ticker: str, side: str, qty: int, limit_price: float,
                 source_price: float, slip_bps: int, note: str = "") -> Dict:
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    tid = str(uuid.uuid4())[:8].upper()
    text = (
        f"TICKET {tid}\n"
        f"TS     {ts}\n"
        f"TICKER {ticker}\n"
        f"SIDE   {side}\n"
        f"QTY    {qty}\n"
        f"LIMIT  {limit_price:.4f}\n"
        f"PRICE* {source_price:.4f}\n"
        f"SLIP   {slip_bps} bps\n"
        f"NOTE   {note}\n"
    )
    sig = checksum(text)
    text += f"CHECK  {sig}\n"
    return {
        "id": tid, "ts": ts, "ticker": ticker, "side": side, "qty": int(qty),
        "limit_price": float(limit_price), "source_price": float(source_price),
        "slip_bps": int(slip_bps), "note": note, "text": text, "checksum": sig
    }

def guard_validate(
    *, ticker: str, side: str, qty: int, price_now: float, price_yclose: float,
    max_order_value: float, max_qty_change: int, max_position_value: float,
    current_position_qty: float
) -> Tuple[bool, List[str]]:
    errs = []
    if qty <= 0:
        errs.append("Qty ต้องมากกว่า 0")
    if qty > max_qty_change:
        errs.append(f"Qty เกินลิมิตต่อคำสั่ง ({max_qty_change})")
    order_value = qty * price_now
    if order_value > max_order_value:
        errs.append(f"มูลค่าคำสั่งเกินลิมิตต่อคำสั่ง: {order_value:,.2f} > {max_order_value:,.2f}")
    # position after
    next_qty = current_position_qty + (qty if side == "BUY" else -qty)
    pos_value = abs(next_qty) * price_now
    if pos_value > max_position_value:
        errs.append(f"มูลค่า Position หลังทำรายการเกินลิมิต: {pos_value:,.2f} > {max_position_value:,.2f}")
    if price_yclose > 0:
        move = abs(price_now - price_yclose) / price_yclose * 100
        if move > GUARDS["price_sanity_pct"]:
            errs.append(f"ราคาปัจจุบันหลุดจากวันก่อน {move:.1f}% > {GUARDS['price_sanity_pct']}%")
    return (len(errs) == 0), errs

def apply_local_sync(*, side: str, qty: int, asset_last: float, client, field_name: str) -> float:
    """Return new asset value (only local/ThingSpeak sync)."""
    if side.upper() == "BUY":
        new_asset_val = asset_last + qty
    else:
        new_asset_val = asset_last - qty
    if client is not None and not GUARDS["dry_run"]:
        client.update({field_name: new_asset_val})
    return new_asset_val

# ------------------------------ UI: Inputs -----------------------------------------------------
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
                    step=0.001, value=float(last_val), key=f"input_{ticker}_real"
                )
                asset_inputs[ticker] = option_val + real_val
            else:
                val = st.number_input(
                    label=display_label, help=help_text,
                    step=0.001, value=float(last_val), key=f"input_{ticker}_asset"
                )
                asset_inputs[ticker] = val
    return asset_inputs

def render_asset_update_controls(configs: List[Dict], clients: Dict):
    st.caption("อัปเดตค่า Assets ไปยัง ThingSpeak (ใช้เมื่อซิงก์พอร์ตภายใน)")
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
                    st.success(f"Updated {ticker} to: {add_val} (DryRun={GUARDS['dry_run']})")
                    clear_all_caches()
                    rerun_keep_selection(st.session_state.get("select_key", ""))
                except Exception as e:
                    st.error(f"Failed to update {ticker}: {e}")

# ------------------------------ NEW: trading_section with tickets ------------------------------
def trading_section(config: Dict, asset_val: float, asset_last: float, df_data: pd.DataFrame,
                    calc: Dict, nex: int, Nex_day_sell: int, clients: Dict):
    ticker = config['ticker']
    asset_conf = config['asset_field']
    field_name = asset_conf['field']
    client = clients.get(asset_conf['channel_id'])

    # --- derive action from tracer ---
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

    # --- Original displays (unchanged) ---
    sell_calc, buy_calc = calc['sell'], calc['buy']
    st.write('sell', '    ', 'A', buy_calc[1], 'P', buy_calc[0], 'C', buy_calc[2])
    col1, col2, col3 = st.columns(3)
    # GO_SELL/GO_BUY จะถูก “ปลดล็อก” หลัง Confirm Ticket เท่านั้น
    ticket_confirmed_key = f"ticket_confirmed_{ticker}"
    st.session_state.setdefault(ticket_confirmed_key, False)

    # --- Price & P/L info ---
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

    # --- Ticket Builder (NEW) ---
    st.markdown("**🧾 Trade Ticket Builder** — ยืนยัน 2 ชั้น + กฎความปลอดภัย")
    # side suggestion from signal
    suggested_side = "BUY" if action_val == 1 else ("SELL" if action_val == 0 else "BUY")
    suggested_qty = sell_calc[1] if suggested_side == "BUY" else buy_calc[1]  # ตามของเดิมในโค้ด
    colA, colB, colC, colD, colE = st.columns(5)
    side = colA.selectbox("Side", options=["BUY", "SELL"], index=0 if suggested_side=="BUY" else 1, key=f"side_{ticker}")
    qty  = int(colB.number_input("Qty", min_value=0, step=1, value=int(suggested_qty), key=f"qty_{ticker}"))
    slip_bps = int(colC.number_input("Slippage (bps)", min_value=0, max_value=500, step=5,
                                     value=int(GUARDS["slip_bps"]), key=f"slip_{ticker}"))
    # Limit from current price with slip (+ for BUY, - for SELL)
    base_price = current_price if current_price > 0 else 0.0
    limit_price = base_price * (1 + slip_bps/10_000) if side == "BUY" else base_price * (1 - slip_bps/10_000)
    limit_price = round(limit_price, 4)
    colD.metric("Limit Price", f"{limit_price:,.4f}")
    note = colE.text_input("Note", value="", key=f"note_{ticker}")

    # Sanity ref: yesterday close (for price_sanity_pct)
    try:
        hist = yf.Ticker(ticker).history(period="5d")["Close"]
        yclose = float(hist[-2]) if len(hist) >= 2 else base_price
    except Exception:
        yclose = base_price

    ok, errs = guard_validate(
        ticker=ticker, side=side, qty=qty, price_now=base_price, price_yclose=yclose,
        max_order_value=GUARDS["max_order_value"], max_qty_change=GUARDS["max_qty_change"],
        max_position_value=GUARDS["max_position_value"], current_position_qty=float(asset_last)
    )
    if not ok:
        st.error("⚠️ ไม่ผ่านกฎ: " + " | ".join(errs))

    # Build ticket (freeze)
    ticket_key = f"ticket_obj_{ticker}"
    if st.button(f"Build Ticket ({ticker})", key=f"build_{ticker}") and ok:
        ticket = build_ticket(ticker, side, qty, limit_price, base_price, slip_bps, note)
        st.session_state[ticket_key] = ticket
        st.session_state[ticket_confirmed_key] = False
        st.session_state[f"cooldown_{ticker}"] = time.time() + GUARDS["cooldown_sec"]

    ticket = st.session_state.get(ticket_key)
    if ticket:
        st.code(ticket["text"])
        st.download_button("⬇️ Download Ticket (.txt)",
                           data=ticket["text"].encode("utf-8"),
                           file_name=f"{ticket['id']}_{ticker}.txt",
                           mime="text/plain",
                           key=f"dl_{ticker}")

        # Confirm layer
        cd_until = st.session_state.get(f"cooldown_{ticker}", 0)
        remain = max(0, int(cd_until - time.time()))
        disabled = GUARDS["require_two_clicks"] and remain > 0
        colX, colY = st.columns([1,3])
        with colX:
            st.button(f"Confirm Ticket ({remain}s)" if disabled else "Confirm Ticket",
                      key=f"confirm_{ticker}", disabled=disabled,
                      on_click=lambda: None if disabled else st.session_state.__setitem__(ticket_confirmed_key, True))
        with colY:
            st.caption("เมื่อ Confirm แล้ว จะบันทึก log และปลดล็อกปุ่ม Apply Locally")

        # Log on confirm
        if st.session_state.get(ticket_confirmed_key):
            append_order_log({
                "id": ticket["id"], "ts": ticket["ts"], "ticker": ticket["ticker"],
                "side": ticket["side"], "qty": ticket["qty"], "limit_price": ticket["limit_price"],
                "source_price": ticket["source_price"], "slip_bps": ticket["slip_bps"],
                "note": ticket["note"], "confirmed": True, "applied_locally": False
            })
            st.success(f"✅ Ticket Confirmed & Logged: {ticket['id']}")

    # ---------- Original SELL/BUY buttons now visible only after confirmation ----------
    if st.session_state.get(ticket_confirmed_key):
        col4, col5, col6 = st.columns(3)
        st.write('buy', '   ', 'A', sell_calc[1], 'P', sell_calc[0], 'C', sell_calc[2])
        if col3.checkbox(f'sell_match_{ticker}'):
            if col3.button(f"Apply SELL Locally ({ticker})"):
                try:
                    new_val = apply_local_sync(side="SELL", qty=buy_calc[1], asset_last=asset_last,
                                               client=client, field_name=field_name)
                    st.success(f"SELL applied locally. New asset={new_val} (DryRun={GUARDS['dry_run']})")
                    # mark applied in log if current ticket is SELL with same qty
                    if ticket and ticket["side"] == "SELL" and ticket["qty"] == int(buy_calc[1]):
                        append_order_log({
                            "id": ticket["id"], "ts": ticket["ts"], "ticker": ticket["ticker"],
                            "side": ticket["side"], "qty": ticket["qty"],
                            "limit_price": ticket["limit_price"], "source_price": ticket["source_price"],
                            "slip_bps": ticket["slip_bps"], "note": ticket["note"],
                            "confirmed": True, "applied_locally": True
                        })
                    clear_all_caches()
                    rerun_keep_selection(ticker)
                except Exception as e:
                    st.error(f"Failed to SELL {ticker}: {e}")

        if col6.checkbox(f'buy_match_{ticker}'):
            if col6.button(f"Apply BUY Locally ({ticker})"):
                try:
                    new_val = apply_local_sync(side="BUY", qty=sell_calc[1], asset_last=asset_last,
                                               client=client, field_name=field_name)
                    st.success(f"BUY applied locally. New asset={new_val} (DryRun={GUARDS['dry_run']})")
                    if ticket and ticket["side"] == "BUY" and ticket["qty"] == int(sell_calc[1]):
                        append_order_log({
                            "id": ticket["id"], "ts": ticket["ts"], "ticker": ticket["ticker"],
                            "side": ticket["side"], "qty": ticket["qty"],
                            "limit_price": ticket["limit_price"], "source_price": ticket["source_price"],
                            "slip_bps": ticket["slip_bps"], "note": ticket["note"],
                            "confirmed": True, "applied_locally": True
                        })
                    clear_all_caches()
                    rerun_keep_selection(ticker)
                except Exception as e:
                    st.error(f"Failed to BUY {ticker}: {e}")

        # Revert last local change (simple undo)
        with st.expander("♻️ Revert Local Adjustment (Undo 1 step)"):
            undo_side = st.selectbox("Undo Side", ["BUY", "SELL"], key=f"undo_side_{ticker}")
            undo_qty = int(st.number_input("Undo Qty", min_value=0, step=1, value=int(sell_calc[1]), key=f"undo_qty_{ticker}"))
            if st.button("UNDO (apply opposite locally)", key=f"undo_btn_{ticker}"):
                try:
                    opp = "SELL" if undo_side == "BUY" else "BUY"
                    new_val = apply_local_sync(side=opp, qty=undo_qty, asset_last=asset_last,
                                               client=client, field_name=field_name)
                    st.success(f"UNDO applied (side={opp}, qty={undo_qty}). New asset={new_val} (DryRun={GUARDS['dry_run']})")
                    clear_all_caches()
                    rerun_keep_selection(ticker)
                except Exception as e:
                    st.error(f"Undo failed: {e}")
    else:
        st.info("สร้างและยืนยัน Ticket ก่อน จึงจะปลดล็อกปุ่ม Apply Locally (ซิงก์พอร์ตภายใน)")

# ------------------------------ Main -----------------------------------------------------------
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

# Bootstrap selection persistence
pending = st.session_state.pop("_pending_select_key", None)
if pending:
    st.session_state.select_key = pending

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

    st.selectbox("Select Ticker to View:", options=selectbox_options,
                 format_func=format_selectbox_options, key="select_key")
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
        df_data, fx_js_str, _ = monitor_data_all.get(ticker, (pd.DataFrame(), "0", None))
        asset_last = last_assets_all.get(ticker, 0.0)
        asset_val = asset_inputs.get(ticker, 0.0)
        calc = calculations.get(ticker, {})

        title_label = selectbox_labels.get(ticker, ticker)
        st.write(title_label)
        trading_section(config, asset_val, asset_last, df_data, calc,
                        st.session_state.nex, st.session_state.Nex_day_sell, THINGSPEAK_CLIENTS)

        with st.expander("Show Raw Data Action"):
            st.dataframe(df_data, use_container_width=True)
        st.write("_____")

# Sidebar quick rerun (preserve selection)
if st.sidebar.button("RERUN"):
    current_selection = st.session_state.get("select_key", "")
    clear_all_caches()
    if current_selection in [c['ticker'] for c in ASSET_CONFIGS]:
        rerun_keep_selection(current_selection)
    else:
        st.rerun()
