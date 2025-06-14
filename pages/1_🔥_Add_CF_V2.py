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

# MODIFIED: Now returns a third client for the config channel
@st.cache_resource
def initialize_thingspeak_clients(config: Dict[str, Any]) -> Tuple[thingspeak.Channel, Dict[str, thingspeak.Channel], thingspeak.Channel]:
    """Initializes and returns the main, asset-specific, and config ThingSpeak clients."""
    channels_config = config.get('thingspeak_channels', {})
    assets_config = config.get('assets', [])
    
    try:
        # Client for Main Output
        main_cfg = channels_config.get('main_output', {})
        client_main = thingspeak.Channel(main_cfg['channel_id'], main_cfg['write_api_key'])
        
        # Clients for individual assets
        asset_clients = {}
        for asset in assets_config:
            ticker = asset['ticker']
            channel_info = asset.get('holding_channel', {})
            if channel_info.get('channel_id') and channel_info.get('write_api_key'):
                asset_clients[ticker] = thingspeak.Channel(
                    channel_info['channel_id'],
                    channel_info['write_api_key']
                )
        
        # NEW: Client for Configuration (offset, etc.)
        client_config = None
        config_cfg = channels_config.get('config_channel', {})
        if config_cfg.get('channel_id') and config_cfg.get('write_api_key'):
            client_config = thingspeak.Channel(
                config_cfg['channel_id'],
                write_key=config_cfg['write_api_key'],
                read_key=config_cfg.get('read_api_key') # Read key is important!
            )
            st.success(f"Initialized main, {len(asset_clients)} asset, and config clients.")
        else:
            st.warning("Missing 'config_channel' config. Reset functionality will be disabled.")

        return client_main, asset_clients, client_config

    except (KeyError, Exception) as e:
        st.error(f"Failed to initialize ThingSpeak clients. Check your config file. Error: {e}")
        st.stop()

# NEW: Function to fetch the offset from ThingSpeak
def fetch_initial_offset(client_config: thingspeak.Channel, config_details: Dict[str, str]) -> float:
    """Fetches the last cashflow offset value from the config ThingSpeak channel."""
    if not client_config or not config_details:
        return 0.0

    try:
        field = config_details['offset_field']
        last_offset_str = client_config.get_field_last(field=field)
        if last_offset_str:
            data_dict = json.loads(last_offset_str)
            last_offset = float(data_dict[field])
            st.info(f"Successfully fetched last cashflow offset from ThingSpeak: {last_offset:,.2f}")
            return last_offset
        return 0.0
    except Exception as e:
        st.warning(f"Could not fetch last offset from ThingSpeak. Defaulting to 0. Error: {e}")
        return 0.0

def fetch_initial_data(assets_config: List[Dict[str, Any]], asset_clients: Dict[str, thingspeak.Channel]) -> Dict[str, Dict[str, float]]:
    """Fetches last holding from ThingSpeak and last price from yfinance for each asset."""
    initial_data = {}
    for asset in assets_config:
        ticker = asset["ticker"]
        initial_data[ticker] = {}
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
        try:
            last_price = yf.Ticker(ticker).fast_info['lastPrice']
            initial_data[ticker]['last_price'] = last_price
        except Exception as e:
            st.warning(f"Could not fetch price for {ticker}. Defaulting to reference price. Error: {e}")
            initial_data[ticker]['last_price'] = asset.get('reference_price', 0.0)
    return initial_data

# --- 2. UI & DISPLAY FUNCTIONS (No changes needed here) ---
def render_ui_and_get_inputs(assets_config: List[Dict[str, Any]], initial_data: Dict[str, Dict[str, float]], product_cost_default: float) -> Dict[str, Any]:
    user_inputs = {}
    st.write("📊 Current Asset Prices")
    current_prices = {}
    for asset in assets_config:
        ticker = asset["ticker"]
        label = f"ราคา_{ticker}_{asset['reference_price']}"
        price_value = initial_data[ticker].get('last_price', 0.0)
        current_prices[ticker] = st.number_input(label, step=0.01, value=price_value, key=f"price_{ticker}", format="%.2f")
    user_inputs['current_prices'] = current_prices
    st.divider()
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
    st.write("⚙️ Calculation Parameters")
    user_inputs['product_cost'] = st.number_input('Product_cost', step=0.01, value=product_cost_default, format="%.2f")
    user_inputs['portfolio_cash'] = st.number_input('Portfolio_cash', step=0.01, value=0.00, format="%.2f")
    return user_inputs

def display_results(metrics: Dict[str, float], cashflow_offset: float):
    st.divider()
    with st.expander("📈 Results", expanded=False):
        st.write('Current Portfolio Value (Assets + Cash):', f"**{metrics['now_pv']:,.2f}**")
        col1, col2 = st.columns(2)
        col1.metric('t_0 (Product of Reference Prices)', f"{metrics['t_0']:,.2f}")
        col2.metric('t_n (Product of Live Prices)', f"{metrics['t_n']:,.2f}")
        st.metric('Fix Component (ln)', f"{metrics['ln']:,.2f}")
        st.metric('Log PV (Calculated Cost)', f"{metrics['log_pv']:,.2f}")
        st.metric('Cashflow Offset (Reset Baseline)', f"{cashflow_offset:,.2f}", 
                  help="This value is subtracted from the raw cashflow to get the final Net Cashflow. It is fetched from ThingSpeak on startup.")
    st.metric(label="💰 Net Cashflow", value=f"{metrics['net_cf']:,.2f}")

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
        else:
            st.warning(f"Chart for '{chart_title}' cannot be displayed. Missing config for field '{field_name}'.")
    create_chart_iframe(main_channel_id, main_fields_map.get('net_cf'), 'Cashflow')
    create_chart_iframe(main_channel_id, main_fields_map.get('pure_alpha'), 'Pure_Alpha')
    create_chart_iframe(main_channel_id, main_fields_map.get('cost_minus_cf'), 'Product_cost - CF')
    create_chart_iframe(main_channel_id, main_fields_map.get('buffer'), 'Buffer')

# --- 3. CORE LOGIC & UPDATE FUNCTIONS ---

def calculate_metrics(assets_config: List[Dict[str, Any]], user_inputs: Dict[str, Any], cashflow_offset: float) -> Dict[str, float]:
    metrics = {}
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
    raw_net_cf = metrics['now_pv'] - metrics['log_pv']
    metrics['raw_net_cf'] = raw_net_cf
    metrics['net_cf'] = raw_net_cf - cashflow_offset
    return metrics

# MODIFIED: This function now updates ThingSpeak instead of writing to a file.
def handle_cashflow_reset(client_config: thingspeak.Channel, config_details: Dict[str, str], metrics: Dict[str, float], current_offset: float):
    """Renders UI for resetting the cashflow and sends the new offset to ThingSpeak."""
    st.divider()
    with st.expander("⚙️ Reset Net Cashflow Baseline"):
        if not client_config:
            st.error("Config Channel not initialized. Cannot reset cashflow.")
            return

        st.warning("This will update the baseline (offset) on ThingSpeak, effectively making the displayed Net Cashflow zero on next run.")
        st.write(f"Current Raw Net Cashflow: **{metrics.get('raw_net_cf', 0):,.2f}**")
        st.write(f"Current Offset (from ThingSpeak): **{current_offset:,.2f}**")
        
        if st.button("RESET CASHFLOW TO ZERO (Update ThingSpeak)"):
            new_offset = metrics.get('raw_net_cf', 0.0)
            field_to_update = config_details['offset_field']
            
            try:
                # Update the offset value on the ThingSpeak channel
                client_config.update({field_to_update: new_offset})
                
                st.success(f"✅ Success! Sent new offset {new_offset:,.2f} to ThingSpeak.")
                st.info("🔄 Please press 'R' or rerun the app to fetch the new offset and see the change take effect.")
                
                # Clear cache to force re-fetch on the next run
                st.cache_data.clear()
                st.cache_resource.clear()
            except Exception as e:
                st.error(f"❌ Failed to update offset on ThingSpeak: {e}")

def handle_thingspeak_update(config: Dict[str, Any], clients: Tuple, metrics: Dict[str, float], user_inputs: Dict[str, Any]):
    client_main, asset_clients, _ = clients # Unpack the third client, but we don't use it here
    main_channel_config = config.get('thingspeak_channels', {}).get('main_output', {})
    with st.expander("⚠️ Confirm to Add Cashflow and Update Holdings", expanded=False):
        st.write("Click the button below to confirm and send data to all ThingSpeak channels.")
        if st.button("Confirm and Send All Data"):
            if user_inputs['product_cost'] == 0:
                st.error("Product_cost cannot be zero. Update failed.")
                return
            try:
                main_fields_map = main_channel_config.get('fields', {})
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
    st.write("🔥 Add Cashflow V3 - MultiChannel (Persistent Offset)")

    config = load_config()
    if not config: return
    
    # MODIFIED: Unpack three clients now
    client_main, asset_clients, client_config = initialize_thingspeak_clients(config)
    
    assets_config = config.get('assets', [])
    product_cost_default = config.get('product_cost_default', 0.0)
    
    # NEW: Fetch the offset from ThingSpeak at the start
    config_channel_details = config.get('thingspeak_channels', {}).get('config_channel', {})
    cashflow_offset = fetch_initial_offset(client_config, config_channel_details)

    initial_data = fetch_initial_data(assets_config, asset_clients)
    user_inputs = render_ui_and_get_inputs(assets_config, initial_data, product_cost_default)

    if st.button("Recalculate"):
        pass 
        
    metrics = calculate_metrics(assets_config, user_inputs, cashflow_offset)
    
    display_results(metrics, cashflow_offset)
    
    # MODIFIED: Call the reset handler with the new required arguments
    handle_cashflow_reset(client_config, config_channel_details, metrics, cashflow_offset)
    
    # MODIFIED: Pass all three clients to the update handler
    handle_thingspeak_update(config, (client_main, asset_clients, client_config), metrics, user_inputs)
    
    render_charts(config)

if __name__ == "__main__":
    main()
