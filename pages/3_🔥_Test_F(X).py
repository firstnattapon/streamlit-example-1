#main code
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import thingspeak
import json
import streamlit.components.v1 as components
from typing import Dict, Any, Tuple, List

# --- Page Configuration ---
st.set_page_config(page_title="Add_CF_V2_Show_Work", page_icon="🧮", layout="centered")

# ### MODIFIED PART 1: Initialize Session State ###
if 'portfolio_cash' not in st.session_state:
    st.session_state.portfolio_cash = 0.00

# --- 1. CONFIGURATION & INITIALIZATION FUNCTIONS ---

@st.cache_data
def load_config(filename: str = "add_cf_config.json") -> Dict[str, Any]:
    """Loads and parses the JSON configuration file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.error(f"Error loading or parsing {filename}: {e}")
        st.stop()

def save_config(config: Dict[str, Any], filename: str = "add_cf_config.json") -> None:
    """Persist config (including per-asset b_offset) to disk."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        st.success("💾 Saved config (with updated b_offset / reference_price).")
    except Exception as e:
        st.error(f"Failed to save config: {e}")

@st.cache_resource
def initialize_thingspeak_clients(config: Dict[str, Any],
                                  stock_assets: List[Dict[str, Any]],
                                  option_assets: List[Dict[str, Any]]) -> Tuple[thingspeak.Channel, Dict[str, thingspeak.Channel]]:
    """Initializes ThingSpeak clients for the main channel and individual asset channels."""
    main_channel_config = config.get('thingspeak_channels', {}).get('main_output', {})
    try:
        client_main = thingspeak.Channel(main_channel_config['channel_id'], main_channel_config['write_api_key'])
        asset_clients = {}
        for asset in stock_assets:
            ticker = asset['ticker']
            channel_info = asset.get('holding_channel', {})
            if channel_info.get('channel_id'):
                asset_clients[ticker] = thingspeak.Channel(channel_info['channel_id'], channel_info['write_api_key'])

        num_asset_clients = len(asset_clients)
        num_option_assets = len(option_assets)
        st.success(f"Initialized main client and {num_asset_clients} asset holding clients, {num_option_assets} option legs.")
        return client_main, asset_clients
    except Exception as e:
        st.error(f"Failed to initialize ThingSpeak clients: {e}")
        st.stop()

def fetch_initial_data(stock_assets: List[Dict[str, Any]],
                       option_assets: List[Dict[str, Any]],
                       asset_clients: Dict[str, thingspeak.Channel]) -> Dict[str, Dict[str, Any]]:
    """Fetches initial prices from yfinance and last holdings from ThingSpeak."""
    initial_data = {}
    tickers_to_fetch = {asset['ticker'].strip() for asset in stock_assets}
    tickers_to_fetch.update({opt.get('underlying_ticker').strip() for opt in option_assets if opt.get('underlying_ticker')})

    for ticker in tickers_to_fetch:
        initial_data[ticker] = {}
        try:
            last_price = yf.Ticker(ticker).fast_info['lastPrice']
            initial_data[ticker]['last_price'] = last_price
        except Exception:
            ref_price = next((a.get('reference_price', 0.0) for a in stock_assets if a['ticker'].strip() == ticker), 0.0)
            initial_data[ticker]['last_price'] = ref_price
            st.warning(f"Could not fetch price for {ticker}. Defaulting to reference price {ref_price}.")

    for asset in stock_assets:
        ticker = asset["ticker"].strip()
        last_holding = 0.0
        if ticker in asset_clients:
            try:
                client = asset_clients[ticker]
                field = asset['holding_channel']['field']
                last_asset_json_string = client.get_field_last(field=field)
                if last_asset_json_string:
                    data_dict = json.loads(last_asset_json_string)
                    last_holding = float(data_dict[field])
            except Exception as e:
                st.warning(f"Could not fetch holding for {ticker}. Defaulting to 0. Error: {e}")
        initial_data[ticker]['last_holding'] = last_holding
    return initial_data

# --- 2. UI & DISPLAY FUNCTIONS ---

def render_ui_and_get_inputs(stock_assets: List[Dict[str, Any]],
                             option_assets: List[Dict[str, Any]],
                             initial_data: Dict[str, Dict[str, Any]],
                             product_cost_default: float) -> Dict[str, Any]:
    """Renders all UI components and collects user inputs into a dictionary."""
    user_inputs = {}
    st.write("📊 Current Asset Prices")
    current_prices = {}
    all_tickers = {asset['ticker'].strip() for asset in stock_assets}
    all_tickers.update({opt['underlying_ticker'].strip() for opt in option_assets if opt.get('underlying_ticker')})

    for ticker in sorted(list(all_tickers)):
        label = f"ราคา_{ticker}"
        price_value = initial_data.get(ticker, {}).get('last_price', 0.0)
        current_prices[ticker] = st.number_input(label, value=float(price_value), key=f"price_{ticker}", format="%.2f")
    user_inputs['current_prices'] = current_prices

    st.divider()
    st.write("📦 Stock Holdings")
    current_holdings = {}
    total_stock_value = 0.0
    for asset in stock_assets:
        ticker = asset["ticker"].strip()
        holding_value = initial_data.get(ticker, {}).get('last_holding', 0.0)

        asset_holding = st.number_input(
            f"{ticker}_asset",
            value=float(holding_value),
            key=f"holding_{ticker}",
            format="%.2f"
        )

        current_holdings[ticker] = asset_holding
        individual_asset_value = asset_holding * current_prices.get(ticker, 0.0)
        st.write(f"มูลค่า {ticker}: **{individual_asset_value:,.2f}**")
        total_stock_value += individual_asset_value

    user_inputs['current_holdings'] = current_holdings
    user_inputs['total_stock_value'] = total_stock_value

    st.divider()
    st.write("⚙️ Calculation Parameters")
    user_inputs['product_cost'] = st.number_input('Product_cost', value=float(product_cost_default), format="%.2f")

    # Link to session state for portfolio_cash
    st.number_input('Portfolio_cash', key='portfolio_cash', format="%.2f")
    user_inputs['portfolio_cash'] = float(st.session_state.portfolio_cash)

    return user_inputs

def render_rollover_ui(stock_assets: List[Dict[str, Any]], config: Dict[str, Any]) -> Tuple[Dict[str, float], bool]:
    """
    UI: Per-asset Rollover t0 with b compensation.
    Returns (pending_rolls, apply_flag). pending_rolls[ticker] = new_t0 (float)
    """
    pending_rolls: Dict[str, float] = {}
    with st.expander("🔁 Rollover t₀ (per-asset, with b compensation)", expanded=False):
        st.caption("เมื่อ roll t₀ จะปรับ b_offset อัตโนมัติ: b ← b + fix*ln(t0_new/t0_old) และอัปเดต reference_price = t0_new")
        for asset in stock_assets:
            ticker = asset['ticker'].strip()
            fix_c = float(asset.get('fix_c', 1500.0))
            ref_price = float(asset.get('reference_price', 0.0))
            b_off = float(asset.get('b_offset', 0.0))
            cols = st.columns([1.2, 1, 1, 1.2])
            with cols[0]:
                st.write(f"**{ticker}**")
                st.write(f"fix={fix_c:.0f}")
            with cols[1]:
                st.number_input(f"t0 (current) - {ticker}", value=ref_price, key=f"t0_current_{ticker}", format="%.4f", disabled=True)
            with cols[2]:
                st.number_input(f"b_offset - {ticker}", value=b_off, key=f"b_offset_view_{ticker}", format="%.4f", disabled=True)
            with cols[3]:
                do_roll = st.checkbox(f"Roll {ticker}", key=f"roll_do_{ticker}", value=False)
                new_t0 = st.number_input(f"t0' (new) - {ticker}", value=ref_price, key=f"roll_to_{ticker}", format="%.4f")
                if do_roll:
                    if new_t0 <= 0:
                        st.warning(f"{ticker}: t0' ต้อง > 0")
                    elif abs(new_t0 - ref_price) < 1e-12:
                        st.info(f"{ticker}: t0' = t0 (ไม่เปลี่ยน)")
                    else:
                        pending_rolls[ticker] = float(new_t0)

        apply = st.button("✅ Apply rolls & Save config")
    return pending_rolls, apply

def apply_rolls_and_update_config(config: Dict[str, Any],
                                  stock_assets: List[Dict[str, Any]],
                                  pending_rolls: Dict[str, float]) -> None:
    """Apply t0 rolls to each selected asset, adjust b_offset, and persist to config."""
    if not pending_rolls:
        st.info("No rolls selected.")
        return

    changes = []
    for asset in stock_assets:
        ticker = asset['ticker'].strip()
        if ticker in pending_rolls:
            old_t0 = float(asset.get('reference_price', 0.0))
            new_t0 = float(pending_rolls[ticker])
            if old_t0 <= 0 or new_t0 <= 0:
                st.warning(f"Skip {ticker}: invalid t0 (old={old_t0}, new={new_t0})")
                continue
            fix_c = float(asset.get('fix_c', 1500.0))
            old_b = float(asset.get('b_offset', 0.0))
            delta_b = fix_c * np.log(new_t0 / old_t0)
            asset['b_offset'] = float(old_b + delta_b)
            asset['reference_price'] = new_t0
            changes.append((ticker, old_t0, new_t0, fix_c, old_b, delta_b, asset['b_offset']))

    if changes:
        # Reflect back in the full config (stock_assets are dicts from config['assets'])
        save_config(config)
        st.success("Applied rolls. Details:")
        for tck, old_t0, new_t0, fix_c, old_b, delta_b, new_b in changes:
            st.code(f"{tck}: t0 {old_t0:.4f} → {new_t0:.4f} | Δb = {fix_c:.0f}*ln({new_t0:.4f}/{old_t0:.4f}) = {delta_b:+.4f} | b: {old_b:.4f} → {new_b:.4f}")

# --- 3. DISPLAY & CHARTING FUNCTIONS ---
def display_results(metrics: Dict[str, float], options_pl: float, total_option_cost: float, config: Dict[str, Any]):
    """Displays all calculated metrics, including a detailed breakdown with b_offset."""
    st.divider()
    with st.expander("📈 Results", expanded=True):
        metric_label = (f"Current Total Value (Stocks + Cash + Current_Options P/L: {options_pl:,.2f}) "
                        f"| Max_Roll_Over: ({-total_option_cost:,.2f})")
        st.metric(label=metric_label, value=f"{metrics['now_pv']:,.2f}")

        col1, col2, col3 = st.columns(3)
        col1.metric('log_pv Baseline (Sum of fix_c)', f"{metrics.get('log_pv_baseline', 0.0):,.2f}")
        col2.metric('b_total (roll compensation)', f"{metrics.get('b_total', 0.0):,.2f}")
        col3.metric('ln_weighted', f"{metrics.get('ln_weighted', 0.0):,.2f}")

        st.metric(
            f"Log PV (Calculated: {metrics.get('log_pv_baseline', 0.0):,.2f} + {metrics.get('b_total', 0.0):,.2f} + {metrics.get('ln_weighted', 0.0):,.2f})",
            f"{metrics['log_pv']:,.2f}"
        )

        st.metric(label="💰 Net Cashflow (Combined)", value=f"{metrics['net_cf']:,.2f}")

        offset_display_val = -config.get('cashflow_offset', 0.0)
        baseline_val = metrics.get('log_pv_baseline', 0.0)
        product_cost = config.get('product_cost_default', 0)
        baseline_label = f"💰 Baseline_T0 | {baseline_val:,.1f}(Control) = {product_cost} (Cost)  + {offset_display_val:.0f} (Lv) "
        st.metric(label=baseline_label, value=f"{metrics['net_cf'] - config.get('cashflow_offset', 0.0):,.2f}")

        baseline_target = config.get('baseline_target', 0.0)
        adjusted_cf = metrics['net_cf'] - config.get('cashflow_offset', 0.0)
        final_value = baseline_target - adjusted_cf
        st.metric(label=f"💰 Net_Zero @ {config.get('cashflow_offset_comment', '')}", value=f"( {final_value*(-1):,.2f} )")

    with st.expander("Show 'per-asset' Calculation Breakdown"):
        st.write("ต่อหุ้น: Fᵢ = bᵢ + fixᵢ * ln(liveᵢ / t0ᵢ)")
        ln_breakdown_data = metrics.get('ln_breakdown', [])
        for item in ln_breakdown_data:
            if item['ref_price'] > 0:
                st.code(
                    f"{item['ticker']:<6}: F_contrib = [{item['fix_c']} * ln({item['live_price']:.4f} / {item['ref_price']:.4f})] "
                    f"+ b ({item['b_offset']:+.4f}) = {item['F_contribution']:+.4f}"
                )
            else:
                st.code(f"{item['ticker']:<6}: (skip; ref_price is zero)")
        st.code("----------------------------------------------------------------")
        st.code(f"Σ ln-term = {metrics.get('ln_weighted', 0.0):+,.4f} | Σ b = {metrics.get('b_total', 0.0):+,.4f} | Total F = {metrics.get('F_total', 0.0):+,.4f}")

def render_charts(config: Dict[str, Any]):
    """Renders ThingSpeak charts using iframe components in a specific order."""
    st.write("📊 ThingSpeak Charts")
    main_channel_config = config.get('thingspeak_channels', {}).get('main_output', {})
    main_channel_id = main_channel_config.get('channel_id')
    main_fields_map = main_channel_config.get('fields', {})

    def create_chart_iframe(channel_id, field_name, chart_title):
        if channel_id and field_name:
            chart_number = field_name.replace('field', '')
            url = f'https://thingspeak.com/channels/{channel_id}/charts/{chart_number}?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&type=line&update=15'
            st.write(f"**{chart_title}**")
            components.iframe(url, width=800, height=200)
            st.divider()

    create_chart_iframe(main_channel_id, main_fields_map.get('net_cf'), 'Cashflow')
    create_chart_iframe(main_channel_id, main_fields_map.get('now_pv'), 'Current Total Value')
    create_chart_iframe(main_channel_id, main_fields_map.get('pure_alpha'), 'Pure_Alpha')
    create_chart_iframe(main_channel_id, main_fields_map.get('cost_minus_cf'), 'Product_cost - CF')
    create_chart_iframe(main_channel_id, main_fields_map.get('buffer'), 'Buffer')

# --- 4. CALCULATION FUNCTION (with b_offset support) ---
def calculate_metrics(stock_assets: List[Dict[str, Any]],
                      option_assets: List[Dict[str, Any]],
                      user_inputs: Dict[str, Any],
                      config: Dict[str, Any]) -> Tuple[Dict[str, float], float, float]:
    """Calculates all core metrics including b_offset and saves per-asset breakdown."""
    metrics = {}
    portfolio_cash = float(user_inputs['portfolio_cash'])
    current_prices = user_inputs['current_prices']
    total_stock_value = float(user_inputs['total_stock_value'])

    # Options P/L (unchanged)
    total_options_pl, total_option_cost = 0.0, 0.0
    for option in option_assets:
        underlying_ticker = option.get("underlying_ticker", "").strip()
        if not underlying_ticker:
            continue
        last_price = float(current_prices.get(underlying_ticker, 0.0))
        strike = float(option.get("strike", 0.0))
        contracts = float(option.get("contracts_or_shares", 0.0))
        premium = float(option.get("premium_paid_per_share", 0.0))
        total_cost_basis = contracts * premium
        total_option_cost += total_cost_basis
        intrinsic_value = max(0.0, last_price - strike) * contracts
        total_options_pl += intrinsic_value - total_cost_basis

    metrics['now_pv'] = total_stock_value + portfolio_cash + total_options_pl

    # Per-asset calc
    log_pv_baseline, ln_weighted, b_total, F_total = 0.0, 0.0, 0.0, 0.0
    ln_breakdown = []

    for asset in stock_assets:
        fix_c = float(asset.get('fix_c', 1500.0))
        ticker = asset['ticker'].strip()
        ref_price = float(asset.get('reference_price', 0.0))
        live_price = float(current_prices.get(ticker, 0.0))
        b_off = float(asset.get('b_offset', 0.0))  # NEW

        log_pv_baseline += fix_c
        b_total += b_off

        ln_term = 0.0
        if ref_price > 0.0 and live_price > 0.0:
            ln_term = fix_c * np.log(live_price / ref_price)

        ln_weighted += ln_term
        F_contrib = b_off + ln_term
        F_total += F_contrib

        ln_breakdown.append({
            "ticker": ticker,
            "fix_c": fix_c,
            "live_price": live_price,
            "ref_price": ref_price,
            "b_offset": b_off,
            "ln_term": ln_term,
            "F_contribution": F_contrib
        })

    metrics['log_pv_baseline'] = log_pv_baseline
    metrics['b_total'] = b_total
    metrics['ln_weighted'] = ln_weighted
    metrics['F_total'] = F_total
    metrics['log_pv'] = log_pv_baseline + b_total + ln_weighted  # NEW: include b_total
    metrics['net_cf'] = metrics['now_pv'] - metrics['log_pv']
    metrics['ln_breakdown'] = ln_breakdown

    return metrics, total_options_pl, total_option_cost

# --- 5. THINGSPEAK UPDATE FUNCTION (Unchanged) ---
def handle_thingspeak_update(config: Dict[str, Any], clients: Tuple,
                             stock_assets: List[Dict[str, Any]],
                             metrics: Dict[str, float],
                             user_inputs: Dict[str, Any]):
    """Handles the UI for confirming and sending data to ThingSpeak."""
    client_main, asset_clients = clients
    with st.expander("⚠️ Confirm to Add Cashflow and Update Holdings", expanded=False):
        if st.button("Confirm and Send All Data"):
            diff = metrics['net_cf'] - config.get('cashflow_offset', 0.0)
            try:
                fields_map = config.get('thingspeak_channels', {}).get('main_output', {}).get('fields', {})
                payload = {
                    fields_map.get('net_cf', 'field1'): diff,
                    fields_map.get('pure_alpha', 'field2'): diff / user_inputs['product_cost'] if user_inputs['product_cost'] != 0 else 0,
                    fields_map.get('buffer', 'field3'): user_inputs['portfolio_cash'],
                    fields_map.get('cost_minus_cf', 'field4'): user_inputs['product_cost'] - diff,
                    fields_map.get('now_pv', 'field5'): metrics.get('now_pv', 0.0)
                }
                client_main.update(payload)
                st.success("✅ Successfully updated Main Channel on Thingspeak!")
            except Exception as e:
                st.error(f"❌ Failed to update Main Channel on Thingspeak: {e}")

            st.divider()
            for asset in stock_assets:
                ticker = asset['ticker'].strip()
                if ticker in asset_clients:
                    try:
                        current_holding = user_inputs['current_holdings'][ticker]
                        field_to_update = asset['holding_channel']['field']
                        asset_clients[ticker].update({field_to_update: current_holding})
                        st.success(f"✅ Successfully updated holding for {ticker}.")
                    except Exception as e:
                        st.error(f"❌ Failed to update holding for {ticker}: {e}")

# --- main() FUNCTION ---
def main():
    """Main function to run the Streamlit application."""
    config = load_config()
    if not config:
        return

    # Ensure b_offset exists per stock
    for a in config.get('assets', []):
        if a.get('type', 'stock') == 'stock':
            a.setdefault('b_offset', 0.0)

    all_assets = config.get('assets', [])
    stock_assets = [item for item in all_assets if item.get('type', 'stock') == 'stock']
    option_assets = [item for item in all_assets if item.get('type') == 'option']

    clients = initialize_thingspeak_clients(config, stock_assets, option_assets)
    initial_data = fetch_initial_data(stock_assets, option_assets, clients[1])

    # --- NEW: Rollover UI (apply updates to config before calc) ---
    pending_rolls, apply = render_rollover_ui(stock_assets, config)
    if apply:
        apply_rolls_and_update_config(config, stock_assets, pending_rolls)
        # Re-load to ensure we pick up persisted values (optional)
        config = load_config()
        all_assets = config.get('assets', [])
        stock_assets = [item for item in all_assets if item.get('type', 'stock') == 'stock']
        option_assets = [item for item in all_assets if item.get('type') == 'option']

    user_inputs = render_ui_and_get_inputs(
        stock_assets,
        option_assets,
        initial_data,
        config.get('product_cost_default', 0.0)
    )

    if st.button("Recalculate"):
        pass

    # 1. Initial Calculation
    metrics, options_pl, total_option_cost = calculate_metrics(stock_assets, option_assets, user_inputs, config)

    # 2. Dynamic Cashflow Offset Calculation (UNCHANGED logic)
    log_pv_baseline = metrics.get('log_pv_baseline', 0.0)
    product_cost = user_inputs.get('product_cost', 0.0)
    dynamic_offset = product_cost - log_pv_baseline
    config['cashflow_offset'] = dynamic_offset  # override in-memory only

    # 3. Display and Update
    display_results(metrics, options_pl, total_option_cost, config)
    handle_thingspeak_update(config, clients, stock_assets, metrics, user_inputs)
    render_charts(config)

if __name__ == "__main__":
    main()
