# code
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import thingspeak
import json
import streamlit.components.v1 as components
from typing import Dict, Any, Tuple, List

# --- Page Configuration ---
st.set_page_config(page_title="Add_CF_V3_MultiChannel (Refactored)", page_icon="🚀")

# --- 1. CONFIGURATION & INITIALIZATION FUNCTIONS ---

@st.cache_data
def load_config(filename: str = "add_cf_config.json") -> Dict[str, Any]:
    """Loads the entire configuration from a JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.error(f"Error loading or parsing {filename}: {e}")
        st.stop()

# ... (initialize_thingspeak_clients and fetch_initial_data remain the same) ...
@st.cache_resource
def initialize_thingspeak_clients(config: Dict[str, Any]) -> Tuple[thingspeak.Channel, Dict[str, thingspeak.Channel]]:
    """Initializes and returns the main and asset-specific ThingSpeak clients."""
    main_channel_config = config.get('thingspeak_channels', {}).get('main_output', {})
    assets_config = config.get('assets', [])
    
    try:
        # Client for Main Output (Net CF, etc.)
        client_main = thingspeak.Channel(
            main_channel_config['channel_id'],
            main_channel_config['write_api_key']
        )
        
        # Create a client for each individual asset
        asset_clients = {}
        for asset in assets_config:
            ticker = asset['ticker']
            channel_info = asset.get('holding_channel', {})
            if channel_info.get('channel_id') and channel_info.get('write_api_key'):
                asset_clients[ticker] = thingspeak.Channel(
                    channel_info['channel_id'],
                    channel_info['write_api_key']
                )
            else:
                st.warning(f"Missing 'holding_channel' config for {ticker}. It won't be updated on ThingSpeak.")

        st.success(f"Initialized main client and {len(asset_clients)} asset clients.")
        return client_main, asset_clients

    except (KeyError, Exception) as e:
        st.error(f"Failed to initialize ThingSpeak clients from config. Check your config file. Error: {e}")
        st.stop()

def fetch_initial_data(assets_config: List[Dict[str, Any]], asset_clients: Dict[str, thingspeak.Channel]) -> Dict[str, Dict[str, float]]:
    """Fetches last holding from ThingSpeak and last price from yfinance for each asset."""
    initial_data = {}
    for asset in assets_config:
        ticker = asset["ticker"]
        initial_data[ticker] = {}

        # Fetch last holding from ThingSpeak
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
                st.warning(f"Could not fetch last holding for {ticker}. Defaulting to 0. Error: {e}")
        initial_data[ticker]['last_holding'] = last_holding

        # Fetch last price from yfinance
        try:
            last_price = yf.Ticker(ticker).fast_info['lastPrice']
            initial_data[ticker]['last_price'] = last_price
        except Exception as e:
            st.warning(f"Could not fetch price for {ticker}. Defaulting to reference price. Error: {e}")
            initial_data[ticker]['last_price'] = asset.get('reference_price', 0.0)
            
    return initial_data
# --- 2. UI & DISPLAY FUNCTIONS ---
# ... (render_ui_and_get_inputs remains the same) ...
def render_ui_and_get_inputs(assets_config: List[Dict[str, Any]], initial_data: Dict[str, Dict[str, float]], product_cost_default: float) -> Dict[str, Any]:
    """Renders all Streamlit input widgets and returns their current values."""
    user_inputs = {}
    
    # --- Price Inputs ---
    st.write("📊 Current Asset Prices")
    current_prices = {}
    for asset in assets_config:
        ticker = asset["ticker"]
        label = f"ราคา_{ticker}_{asset['reference_price']}"
        price_value = initial_data[ticker].get('last_price', 0.0)
        current_prices[ticker] = st.number_input(label, step=0.01, value=price_value, key=f"price_{ticker}", format="%.2f")
    user_inputs['current_prices'] = current_prices

    st.divider()

    # --- Holding Inputs ---
    st.write("📦 Asset Holdings")
    current_holdings = {}
    total_asset_value = 0.0
    for asset in assets_config:
        ticker = asset["ticker"]
        holding_value = initial_data[ticker].get('last_holding', 0.0)
        asset_holding = st.number_input(f"{ticker}_asset", step=0.01, value=holding_value, key=f"holding_{ticker}", format="%.2f")
        current_holdings[ticker] = asset_holding
        
        individual_asset_value = asset_holding * current_prices[ticker]
        st.write(f"มูลค่า {ticker}: **{individual_asset_value:,.2f}**")
        total_asset_value += individual_asset_value
    user_inputs['current_holdings'] = current_holdings
    user_inputs['total_asset_value'] = total_asset_value

    st.divider()
    
    # --- Final Calculation Inputs ---
    st.write("⚙️ Calculation Parameters")
    user_inputs['product_cost'] = st.number_input('Product_cost', step=0.01, value=product_cost_default, format="%.2f")
    user_inputs['portfolio_cash'] = st.number_input('Portfolio_cash', step=0.01, value=0.00, format="%.2f")
    
    return user_inputs

def display_results(metrics: Dict[str, float], cashflow_offset: float): # <--- MODIFIED: Pass in offset
    """Displays all the calculated metrics."""
    
    st.divider()
    with st.expander("📈 Results", expanded=False):
        st.write('Current Portfolio Value (Assets + Cash):', f"**{metrics['now_pv']:,.2f}**")
        
        col1, col2 = st.columns(2)
        col1.metric('t_0 (Product of Reference Prices)', f"{metrics['t_0']:,.2f}")
        col2.metric('t_n (Product of Live Prices)', f"{metrics['t_n']:,.2f}")
    
        st.metric('Fix Component (ln)', f"{metrics['ln']:,.2f}")
        st.metric('Log PV (Calculated Cost)', f"{metrics['log_pv']:,.2f}")
        
        # <--- NEW: Display the offset for transparency --->
        st.metric('Cashflow Offset (Reset Baseline)', f"{cashflow_offset:,.2f}", 
                  help="This value is subtracted from the raw cashflow to get the final Net Cashflow. Use the 'Reset' button to update it.")
    
    st.metric(label="💰 Net Cashflow", value=f"{metrics['net_cf']:,.2f}")

# ... (render_charts remains the same) ...
def render_charts(config: Dict[str, Any]):
    """Renders all ThingSpeak charts using iframes."""
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
        else:
            st.warning(f"Chart for '{chart_title}' cannot be displayed. Missing config for field '{field_name}'.")

    create_chart_iframe(main_channel_id, main_fields_map.get('net_cf'), 'Cashflow')
    create_chart_iframe(main_channel_id, main_fields_map.get('pure_alpha'), 'Pure_Alpha')
    create_chart_iframe(main_channel_id, main_fields_map.get('cost_minus_cf'), 'Product_cost - CF')
    create_chart_iframe(main_channel_id, main_fields_map.get('buffer'), 'Buffer')

# --- 3. CORE LOGIC & UPDATE FUNCTIONS ---

def calculate_metrics(assets_config: List[Dict[str, Any]], user_inputs: Dict[str, Any], cashflow_offset: float) -> Dict[str, float]: # <--- MODIFIED: Pass in offset
    """Performs all financial calculations."""
    metrics = {}
    
    # Unpack user inputs for clarity
    total_asset_value = user_inputs['total_asset_value']
    portfolio_cash = user_inputs['portfolio_cash']
    product_cost = user_inputs['product_cost']
    current_prices = user_inputs['current_prices']

    metrics['now_pv'] = total_asset_value + portfolio_cash
    
    reference_prices = [asset['reference_price'] for asset in assets_config]
    live_prices = [current_prices[asset['ticker']] for asset in assets_config]

    metrics['t_0'] = np.prod(reference_prices) if reference_prices else 0
    metrics['t_n'] = np.prod(live_prices) if live_prices else 0
    
    t_0, t_n = metrics['t_0'], metrics['t_n']
    metrics['ln'] = -1500 * np.log(t_0 / t_n) if t_0 > 0 and t_n > 0 else 0
    
    number_of_assets = len(assets_config)
    metrics['log_pv'] = (number_of_assets * 1500) + metrics['ln']
    
    # <--- MODIFIED SECTION: Calculate raw CF, then apply the offset --->
    raw_net_cf = metrics['now_pv'] - metrics['log_pv']
    metrics['raw_net_cf'] = raw_net_cf # Store the raw value for the reset logic
    metrics['net_cf'] = raw_net_cf - cashflow_offset # This is the final, displayed value
    
    return metrics

# <--- NEW FUNCTION: Handles the entire reset process --->
def handle_cashflow_reset(config: Dict[str, Any], metrics: Dict[str, float]):
    """Renders UI for resetting the cashflow and handles the config file update."""
    st.divider()
    with st.expander("⚙️ Reset Net Cashflow Baseline"):
        st.write(f"Current Raw Net Cashflow: **{metrics.get('raw_net_cf', 0):,.2f}**")
        st.write(f"Current Offset: **{config.get('cashflow_offset', 0.0):,.2f}**")

# ... (handle_thingspeak_update remains mostly the same, just a logic change) ...
def handle_thingspeak_update(config: Dict[str, Any], clients: Tuple, metrics: Dict[str, float], user_inputs: Dict[str, Any]):
    """Handles the expander and button logic for updating all ThingSpeak channels."""
    client_main, asset_clients = clients
    main_channel_config = config.get('thingspeak_channels', {}).get('main_output', {})
    
    with st.expander("⚠️ Confirm to Add Cashflow and Update Holdings", expanded=False):
        st.write("Click the button below to confirm and send data to all ThingSpeak channels.")
        
        if st.button("Confirm and Send All Data"):
            if user_inputs['product_cost'] == 0:
                st.error("Product_cost cannot be zero. Update failed.")
                return

            try:
                main_fields_map = main_channel_config.get('fields', {})
                # IMPORTANT: We send the final, offset-adjusted 'net_cf' to ThingSpeak
                payload = {
                    main_fields_map.get('net_cf'): metrics['net_cf'],
                    main_fields_map.get('pure_alpha'): metrics['net_cf'] / user_inputs['product_cost'] if user_inputs['product_cost'] != 0 else 0,
                    main_fields_map.get('buffer'): user_inputs['portfolio_cash'],
                    main_fields_map.get('cost_minus_cf'): user_inputs['product_cost'] - metrics['net_cf']
                }
                payload = {k: v for k, v in payload.items() if k}
                client_main.update(payload)
                st.success("✅ Successfully updated Main Channel on Thingspeak!")
            except Exception as e:
                st.error(f"❌ Failed to update Main Channel on Thingspeak: {e}")

            st.divider()
            
            st.write("Updating individual asset holdings...")
            for asset in config.get('assets', []):
                ticker = asset['ticker']
                if ticker in asset_clients:
                    try:
                        current_holding = user_inputs['current_holdings'][ticker]
                        field_to_update = asset['holding_channel']['field']
                        client_to_update = asset_clients[ticker]
                        client_to_update.update({field_to_update: current_holding})
                        st.success(f"✅ Successfully updated holding for {ticker}.")
                    except Exception as e:
                        st.error(f"❌ Failed to update holding for {ticker}: {e}")
                else:
                    st.info(f"ℹ️ Skipping update for {ticker} (no client configured).")

# --- 4. MAIN APPLICATION FLOW ---

def main():
    """Main function to run the Streamlit application."""
    st.write("🔥 Add Cashflow V3 - MultiChannel (Refactored)")

    config = load_config()
    if not config: return
    
    clients = initialize_thingspeak_clients(config)
    
    assets_config = config.get('assets', [])
    product_cost_default = config.get('product_cost_default', 0.0)
    # <--- NEW: Get the offset from the config --->
    cashflow_offset = config.get('cashflow_offset', 0.0) 

    initial_data = fetch_initial_data(assets_config, clients[1])
    user_inputs = render_ui_and_get_inputs(assets_config, initial_data, product_cost_default)

    if st.button("Recalculate"):
        pass 
        
    # <--- MODIFIED: Pass the offset into the calculation --->
    metrics = calculate_metrics(assets_config, user_inputs, cashflow_offset)
    
    # <--- MODIFIED: Pass the offset for display --->
    display_results(metrics, cashflow_offset)
    
    # <--- NEW: Call the reset handler function --->
    handle_cashflow_reset(config, metrics)
    
    handle_thingspeak_update(config, clients, metrics, user_inputs)
    
    render_charts(config)


if __name__ == "__main__":
    main()
