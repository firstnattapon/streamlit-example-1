# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import thingspeak
import json
import streamlit.components.v1 as components

st.set_page_config(page_title="Add_CF_V3_MultiChannel", page_icon="🔥")

# =============================================================================
# 1. CONFIGURATION & CLIENT INITIALIZATION FUNCTIONS
# =============================================================================

@st.cache_data
def load_config(filename="add_cf_config.json"):
    """Loads the entire configuration from a JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.error(f"Error loading or parsing {filename}: {e}")
        return None

def initialize_thingspeak_clients(config):
    """Initializes and returns Thingspeak clients based on the config."""
    main_channel_config = config.get('thingspeak_channels', {}).get('main_output', {})
    assets_config = config.get('assets', [])
    
    try:
        # Client for Main Output (Net CF, etc.)
        client_main = thingspeak.Channel(
            main_channel_config.get('channel_id'),
            main_channel_config.get('write_api_key')
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
                st.warning(f"Missing 'holding_channel' config for {ticker}. It won't be updated on Thingspeak.")
        
        st.sidebar.success(f"Initialized main client and {len(asset_clients)} asset clients.")
        return client_main, asset_clients

    except Exception as e:
        st.error(f"Failed to initialize Thingspeak clients from config. Error: {e}")
        return None, None

# =============================================================================
# 2. DATA FETCHING FUNCTIONS
# =============================================================================

def get_last_holding_from_thingspeak(client, field_key):
    """Fetches and parses the last holding value from a Thingspeak field."""
    try:
        last_asset_json_string = client.get_field_last(field=field_key)
        if last_asset_json_string:
            data_dict = json.loads(last_asset_json_string)
            return float(data_dict[field_key])
        return 0.0  # Return 0 if the field is empty
    except Exception as e:
        st.warning(f"Could not fetch last holding from field '{field_key}'. Defaulting to 0. Error: {e}")
        return 0.0

def get_last_price_from_yfinance(ticker, reference_price):
    """Fetches the last price from yfinance, with a fallback to a reference price."""
    try:
        return yf.Ticker(ticker).fast_info['lastPrice']
    except Exception as e:
        st.warning(f"Could not fetch price for {ticker} from yfinance. Defaulting to reference price. Error: {e}")
        return reference_price

def fetch_initial_asset_data(assets_config, asset_clients):
    """Fetches initial holding and price data for all assets."""
    initial_data = {}
    for asset in assets_config:
        ticker = asset["ticker"]
        
        last_holding = 0.0
        if ticker in asset_clients:
            client = asset_clients[ticker]
            field_key = asset['holding_channel']['field']
            last_holding = get_last_holding_from_thingspeak(client, field_key)
            
        last_price = get_last_price_from_yfinance(ticker, asset.get('reference_price', 0.0))
        
        initial_data[ticker] = {
            'last_holding': last_holding,
            'last_price': last_price,
        }
    return initial_data
    
# =============================================================================
# 3. CHARTING FUNCTION
# =============================================================================

def create_chart_iframe(channel_id, field_name, chart_title):
    """Creates and displays a Thingspeak chart iframe."""
    if channel_id and field_name:
        chart_number = field_name.replace('field', '')
        url = f'https://thingspeak.com/channels/{channel_id}/charts/{chart_number}?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&type=line&update=15'
        st.write(f"**{chart_title}**")
        components.iframe(url, width=800, height=200)
        st.write("_____")
    else:
        st.warning(f"Chart for '{chart_title}' cannot be displayed. Missing config.")

# =============================================================================
# 4. MAIN APPLICATION LOGIC
# =============================================================================

def main():
    """Main function to run the Streamlit application."""
    # --- Load Config and Initialize Clients ---
    config = load_config()
    if not config:
        st.stop()

    client_main, asset_clients = initialize_thingspeak_clients(config)
    if not client_main:
        st.stop()

    assets_config = config.get('assets', [])
    product_cost_default = config.get('product_cost_default', 0.0)
    main_channel_config = config.get('thingspeak_channels', {}).get('main_output', {})
    
    # --- Fetch Initial Data (once per run) ---
    initial_asset_data = fetch_initial_asset_data(assets_config, asset_clients)

    # --- Streamlit UI Section ---
    # This section now builds the UI and gathers user inputs in a single loop
    current_prices = {}
    current_holdings = {}
    total_asset_value = 0.0

    st.write("### Prices")
    for asset in assets_config:
        ticker = asset["ticker"]
        label = f"ราคา_{ticker}_{asset['reference_price']}"
        price_value = initial_asset_data[ticker].get('last_price', 0.0)
        # The value from the number_input is the most current price for calculations
        current_prices[ticker] = st.number_input(label, step=0.01, value=price_value, key=f"price_{ticker}")

    st.write("_____")
    
    st.write("### Holdings")
    for asset in assets_config:
        ticker = asset["ticker"]
        holding_value = initial_asset_data[ticker].get('last_holding', 0.0)
        # Get user-adjusted holding value
        current_holdings[ticker] = st.number_input(f"{ticker}_asset", step=0.01, value=holding_value, key=f"holding_{ticker}")
        
        # Calculate and display individual asset value based on current UI values
        individual_asset_value = current_holdings[ticker] * current_prices[ticker]
        st.write(f"มูลค่า {ticker}: {individual_asset_value:,.2f}")
        total_asset_value += individual_asset_value

    st.write("_____")

    # --- Final Calculations ---
    Product_cost = st.number_input('Product_cost', step=0.01, value=product_cost_default)
    j_1 = st.number_input('Portfolio_cash', step=0.01, value=0.00)

    now_pv = total_asset_value + j_1
    st.write('now_pv:', f"{now_pv:,.2f}")
    st.write("_____")

    reference_prices = [asset['reference_price'] for asset in assets_config]
    live_prices_list = list(current_prices.values())

    t_0 = np.prod(reference_prices)
    t_n = np.prod(live_prices_list)
    
    ln = -1500 * np.log(t_0 / t_n) if t_0 > 0 and t_n > 0 else 0.0
    log_pv = Product_cost + ln
    net_cf = now_pv - log_pv

    st.write('t_0', f"{t_0:,.2f}")
    st.write('t_n', f"{t_n:,.2f}")
    st.write('fix', f"{ln:,.2f}")
    st.write('log_pv', f"{log_pv:,.2f}")
    st.metric(label="Net Cashflow", value=f"{net_cf:,.2f}")
    st.write('____')

    # --- Buttons and Thingspeak Update Section ---
    if st.button("Rerun Page"):
        st.rerun()
    st.write("_____")

    with st.expander("⚠️ Confirm to Add Cashflow and Update Holdings"):
        st.write("Click the button below to confirm and send data to all Thingspeak channels.")
        
        if st.button("Confirm and Send All Data"):
            # 1. Update Main Channel
            if Product_cost == 0:
                st.error("Product_cost cannot be zero.")
            else:
                try:
                    main_fields_map = main_channel_config.get('fields', {})
                    payload = {
                        main_fields_map.get('net_cf'): net_cf,
                        main_fields_map.get('pure_alpha'): net_cf / Product_cost,
                        main_fields_map.get('buffer'): j_1,
                        main_fields_map.get('cost_minus_cf'): Product_cost - net_cf
                    }
                    payload_cleaned = {f"field{k}": v for k, v in payload.items() if k is not None}
                    client_main.update(payload_cleaned)
                    st.success("✅ Successfully updated Main Channel on Thingspeak!")
                except Exception as e:
                    st.error(f"❌ Failed to update Main Channel on Thingspeak: {e}")

            # 2. Update individual asset channels
            st.write("---")
            st.write("Updating individual asset holdings...")
            for asset in assets_config:
                ticker = asset['ticker']
                if ticker in asset_clients:
                    try:
                        # Use the most recent holding value from the UI
                        holding_to_update = current_holdings[ticker]
                        field_to_update = asset['holding_channel']['field']
                        
                        client_to_update = asset_clients[ticker]
                        client_to_update.update({field_to_update: holding_to_update})
                        st.success(f"✅ Successfully updated holding for {ticker}.")
                    except Exception as e:
                        st.error(f"❌ Failed to update holding for {ticker}: {e}")
                else:
                    st.info(f"ℹ️ Skipping update for {ticker} (no client configured).")

    # --- Chart Display Section ---
    st.write("_____")
    main_channel_id = main_channel_config.get('channel_id')
    main_fields_map = main_channel_config.get('fields', {})

    create_chart_iframe(main_channel_id, f"field{main_fields_map.get('net_cf')}", 'Cashflow')
    create_chart_iframe(main_channel_id, f"field{main_fields_map.get('pure_alpha')}", 'Pure_Alpha')
    create_chart_iframe(main_channel_id, f"field{main_fields_map.get('cost_minus_cf')}", 'Product_cost - CF')
    create_chart_iframe(main_channel_id, f"field{main_fields_map.get('buffer')}", 'Buffer')

# This is a standard Python practice to run the main function
if __name__ == "__main__":
    main()
