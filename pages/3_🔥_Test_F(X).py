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
st.set_page_config(page_title="Add_CF_V2_Show_Work", page_icon="🧮", layout= "centered" )

# --- Initialize Session State ---
# This runs only once at the beginning of the session.
if 'portfolio_cash' not in st.session_state:
    st.session_state.portfolio_cash = 0.00
if 'product_cost' not in st.session_state:
    st.session_state.product_cost = 0.00 # Initialize product_cost as well

# --- 1. CONFIGURATION & INITIALIZATION FUNCTIONS (Unchanged) ---

@st.cache_data
def load_config(filename: str = "add_cf_config.json") -> Dict[str, Any]:
    """Loads and parses the JSON configuration file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.error(f"Error loading or parsing {filename}: {e}")
        st.stop()

@st.cache_resource
def initialize_thingspeak_clients(config: Dict[str, Any], stock_assets: List[Dict[str, Any]], option_assets: List[Dict[str, Any]]) -> Tuple[thingspeak.Channel, Dict[str, thingspeak.Channel]]:
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
        st.success(f"Initialized main client and {num_asset_clients} asset {num_option_assets} option holding clients.")

        return client_main, asset_clients
    except Exception as e:
        st.error(f"Failed to initialize ThingSpeak clients: {e}")
        st.stop()

def fetch_initial_data(stock_assets: List[Dict[str, Any]], option_assets: List[Dict[str, Any]], asset_clients: Dict[str, thingspeak.Channel]) -> Dict[str, Dict[str, Any]]:
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

# --- 2. DISPLAY & CHARTING FUNCTIONS (Unchanged, as they only display data) ---
def display_results(metrics: Dict[str, float], options_pl: float, total_option_cost: float, config: Dict[str, Any]):
    """Displays all calculated metrics, including a detailed breakdown of ln_weighted."""
    st.divider()
    with st.expander("📈 Results", expanded=True):
        metric_label = (f"Current Total Value (Stocks + Cash + Current_Options P/L: {options_pl:,.2f}) "
                        f"| Max_Roll_Over: ({-total_option_cost:,.2f})")
        st.metric(label=metric_label, value=f"{metrics['now_pv']:,.2f}")

        col1, col2 = st.columns(2)
        col1.metric('log_pv Baseline (Sum of fix_c)', f"{metrics.get('log_pv_baseline', 0.0):,.2f}")
        col2.metric('log_pv Adjustment (ln_weighted)', f"{metrics.get('ln_weighted', 0.0):,.2f}")

        st.metric(f"Log PV (Calculated: {metrics.get('log_pv_baseline', 0.0):,.2f} + {metrics.get('ln_weighted', 0.0):,.2f})",
                  f"{metrics['log_pv']:,.2f}")

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

    with st.expander("Show 'ln_weighted' Calculation Breakdown"):
        st.write("ค่า `ln_weighted` คำนวณมาจากผลรวมของหุ้นแต่ละตัว:")
        ln_breakdown_data = metrics.get('ln_breakdown', [])

        for item in ln_breakdown_data:
            if item['ref_price'] > 0:
                formula_string = (
                    f"{item['ticker']:<6}: {item['contribution']:+9.4f} = "
                    f"[ {item['fix_c']} * ln( {item['live_price']:.2f} / {item['ref_price']:.2f} ) ]"
                )
            else:
                formula_string = f"{item['ticker']:<6}: {item['contribution']:+9.4f}   (Calculation skipped: ref_price is zero)"

            st.code(formula_string, language='text')

        st.code("----------------------------------------------------------------")
        st.code(f"Total Sum = {metrics.get('ln_weighted', 0.0):+51.4f}")

def render_charts(config: Dict[str, Any]):
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

# --- 3. CALCULATION FUNCTION (Unchanged) ---
def calculate_metrics(stock_assets: List[Dict[str, Any]], option_assets: List[Dict[str, Any]], user_inputs: Dict[str, Any], config: Dict[str, Any]) -> Tuple[Dict[str, float], float, float]:
    metrics = {}
    portfolio_cash = user_inputs['portfolio_cash']
    current_prices = user_inputs['current_prices']
    total_stock_value = user_inputs['total_stock_value']

    total_options_pl, total_option_cost = 0.0, 0.0
    for option in option_assets:
        underlying_ticker = option.get("underlying_ticker", "").strip()
        if not underlying_ticker: continue
        last_price = current_prices.get(underlying_ticker, 0.0)
        strike = option.get("strike", 0.0)
        contracts = option.get("contracts_or_shares", 0.0)
        premium = option.get("premium_paid_per_share", 0.0)
        total_cost_basis = contracts * premium
        total_option_cost += total_cost_basis
        intrinsic_value = max(0, last_price - strike) * contracts
        total_options_pl += intrinsic_value - total_cost_basis

    metrics['now_pv'] = total_stock_value + portfolio_cash + total_options_pl

    log_pv_baseline, ln_weighted = 0.0, 0.0
    ln_breakdown = []
    for asset in stock_assets:
        fix_c = asset.get('fix_c', 1500)
        ticker = asset['ticker'].strip()
        ref_price = asset.get('reference_price', 0.0)
        live_price = current_prices.get(ticker, 0.0)
        log_pv_baseline += fix_c
        contribution = 0.0
        if ref_price > 0 and live_price > 0:
            contribution = fix_c * np.log(live_price / ref_price)
        ln_weighted += contribution
        ln_breakdown.append({"ticker": ticker, "fix_c": fix_c, "live_price": live_price, "ref_price": ref_price, "contribution": contribution})

    metrics['log_pv_baseline'] = log_pv_baseline
    metrics['ln_weighted'] = ln_weighted
    metrics['log_pv'] = log_pv_baseline + ln_weighted
    metrics['net_cf'] = metrics['now_pv'] - metrics['log_pv']
    metrics['ln_breakdown'] = ln_breakdown
    return metrics, total_options_pl, total_option_cost

# --- 4. THINGSPEAK UPDATE FUNCTION (Unchanged) ---
def handle_thingspeak_update(config: Dict[str, Any], clients: Tuple, stock_assets: List[Dict[str, Any]], metrics: Dict[str, float], user_inputs: Dict[str, Any]):
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

# --- 5. RESTRUCTURED main() FUNCTION ---
def main():
    """Main function to run the Streamlit application using st.form for robust state handling."""
    config = load_config()
    if not config: return

    all_assets = config.get('assets', [])
    stock_assets = [item for item in all_assets if item.get('type', 'stock') == 'stock']
    option_assets = [item for item in all_assets if item.get('type') == 'option']

    clients = initialize_thingspeak_clients(config, stock_assets, option_assets)
    initial_data = fetch_initial_data(stock_assets, option_assets, clients[1])

    # The user_inputs dictionary will be populated inside the form
    user_inputs = {}

    # Create a form to group all user inputs
    with st.form(key="calculation_form"):
        st.write("📊 Current Asset Prices")
        current_prices = {}
        all_tickers = {asset['ticker'].strip() for asset in stock_assets}
        all_tickers.update({opt['underlying_ticker'].strip() for opt in option_assets if opt.get('underlying_ticker')})

        for ticker in sorted(list(all_tickers)):
            label = f"ราคา_{ticker}"
            price_value = initial_data.get(ticker, {}).get('last_price', 0.0)
            # Use a unique key for each widget to avoid conflicts
            current_prices[ticker] = st.number_input(label, value=price_value, key=f"price_{ticker}", format="%.2f")
        user_inputs['current_prices'] = current_prices

        st.divider()
        st.write("📦 Stock Holdings")
        current_holdings = {}
        total_stock_value = 0.0
        for asset in stock_assets:
            ticker = asset["ticker"].strip()
            holding_value = initial_data.get(ticker, {}).get('last_holding', 0.0)
            asset_holding = st.number_input(f"{ticker}_asset", value=holding_value, key=f"holding_{ticker}", format="%.2f")
            current_holdings[ticker] = asset_holding
            individual_asset_value = asset_holding * current_prices.get(ticker, 0.0)
            st.write(f"มูลค่า {ticker}: **{individual_asset_value:,.2f}**")
            total_stock_value += individual_asset_value
        user_inputs['current_holdings'] = current_holdings
        user_inputs['total_stock_value'] = total_stock_value

        st.divider()
        st.write("⚙️ Calculation Parameters")
        
        # We now use a key and assign the returned value to the dictionary
        # The key ensures the state is managed correctly by the form
        user_inputs['product_cost'] = st.number_input(
            'Product_cost',
            key='product_cost_input',
            value=config.get('product_cost_default', 0.0),
            format="%.2f"
        )
        user_inputs['portfolio_cash'] = st.number_input(
            'Portfolio_cash',
            key='portfolio_cash_input',
            value=st.session_state.portfolio_cash, # Read initial value from session_state
            format="%.2f"
        )

        # The one and only button to trigger the calculation
        submitted = st.form_submit_button("Calculate / Refresh")

    # --- Calculation and display logic only runs AFTER the form is submitted ---
    if submitted:
        # Update session state with the value from the form just before calculation
        st.session_state.portfolio_cash = user_inputs['portfolio_cash']
        
        metrics, options_pl, total_option_cost = calculate_metrics(stock_assets, option_assets, user_inputs, config)

        log_pv_baseline = metrics.get('log_pv_baseline', 0.0)
        product_cost = user_inputs.get('product_cost', 0.0)
        dynamic_offset = product_cost - log_pv_baseline

        # Create a temporary config copy for this run to hold the dynamic offset
        run_config = config.copy()
        run_config['cashflow_offset'] = dynamic_offset
        run_config['product_cost_default'] = product_cost # Use the submitted product cost

        display_results(metrics, options_pl, total_option_cost, run_config)
        handle_thingspeak_update(run_config, clients, stock_assets, metrics, user_inputs)
        render_charts(config)


if __name__ == "__main__":
    main()
