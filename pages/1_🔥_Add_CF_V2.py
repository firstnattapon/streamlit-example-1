import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import thingspeak
import json
import streamlit.components.v1 as components

st.set_page_config(page_title="Add_CF_V3_MultiChannel", page_icon="🔥")

# --- Function to load configuration ---
@st.cache_data
def load_config(filename="add_cf_config.json"):
    """Loads the entire configuration from a JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        st.error(f"Error loading or parsing {filename}: {e}")
        st.stop()
        return None

# --- Load Entire Configuration ---
config = load_config()
if not config:
    st.stop()

assets_config = config.get('assets', [])
product_cost_default = config.get('product_cost_default', 0.0)
main_channel_config = config.get('thingspeak_channels', {}).get('main_output', {})

# --- Thingspeak Clients Initialization ---
asset_clients = {} # Dictionary to hold a client for each asset
try:
    # Client for Main Output (Net CF, etc.)
    client_main = thingspeak.Channel(
        main_channel_config.get('channel_id'),
        main_channel_config.get('write_api_key')
    )
    
    # Create a client for each individual asset
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

except Exception as e:
    st.error(f"Failed to initialize Thingspeak clients from config. Error: {e}")
    st.stop()


# --- Data Fetching and UI Generation ---
asset_data = {}
for asset in assets_config:
    ticker = asset["ticker"]
    asset_data[ticker] = {} # Initialize dictionary for the ticker
    
    # Fetch last holding only if a client was created for it
    if ticker in asset_clients:
        try:
            field = asset['holding_channel']['field']
            client = asset_clients[ticker]
            # This returns a JSON string like: '{"created_at":...,"field1":"82.0"}'
            last_asset_json_string = client.get_field_last(field=field)

            # --- BUG FIX IS HERE (v2) ---
            # We now correctly handle the JSON string response from the API.
            if last_asset_json_string:
                # 1. Parse the JSON string into a Python dictionary
                data_dict = json.loads(last_asset_json_string)
                # 2. Extract the value using the 'field' variable (e.g., "field1") as the key
                # 3. Convert the extracted value (e.g., "82.0") to a float
                last_holding = float(data_dict[field])
            else: # Handle case where the field is empty on ThingSpeak
                last_holding = 0.0
            
            asset_data[ticker]['last_holding'] = last_holding
        except Exception as e:
            st.warning(f"Could not fetch last holding for {ticker}. Defaulting to 0. Error: {e}")
            asset_data[ticker]['last_holding'] = 0.0
    else:
        asset_data[ticker]['last_holding'] = 0.0

    try:
        # Get last price from yfinance
        last_price = yf.Ticker(ticker).fast_info['lastPrice']
        asset_data[ticker]['last_price'] = last_price
    except Exception as e:
        st.warning(f"Could not fetch price for {ticker} from yfinance. Defaulting to reference price. Error: {e}")
        asset_data[ticker]['last_price'] = asset.get('reference_price', 0.0)

# --- Streamlit UI Section ---
current_prices = {}
for asset in assets_config:
    ticker = asset["ticker"]
    label = f"ราคา_{ticker}_{asset['reference_price']}"
    price_value = asset_data[ticker].get('last_price', 0.0)
    current_prices[ticker] = st.number_input(label, step=0.01, value=price_value, key=f"price_{ticker}")

st.write("_____")

total_asset_value = 0.0
# We will get holding values from st.session_state later
for asset in assets_config:
    ticker = asset["ticker"]
    holding_value = asset_data[ticker].get('last_holding', 0.0)
    # The key here is important for retrieving the value later
    asset_holding = st.number_input(f"{ticker}_asset", step=0.01, value=holding_value, key=f"holding_{ticker}")
    
    individual_asset_value = asset_holding * current_prices[ticker]
    st.write(f"มูลค่า {ticker}: {individual_asset_value:,.2f}")
    
    total_asset_value += individual_asset_value

st.write("_____")

# --- Final Calculations ---
Product_cost = st.number_input('Product_cost', step=0.01, value=product_cost_default)
j_1 = st.number_input('Portfolio_cash', step=0.01, value=0.00)

number = total_asset_value + j_1
st.write('now_pv:', f"{number:,.2f}")

st.write("_____")

reference_prices = [asset['reference_price'] for asset in assets_config]
live_prices = [current_prices[asset['ticker']] for asset in assets_config]

t_0 = np.prod(reference_prices)
t_n = np.prod(live_prices)

ln = -1500 * np.log(t_0 / t_n) if t_0 > 0 and t_n > 0 else 0
log_pv = Product_cost + ln
net_cf = number - log_pv

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
        try:
            if Product_cost == 0:
                st.error("Product_cost cannot be zero.")
            else:
                main_fields_map = main_channel_config.get('fields', {})
                payload = {
                    main_fields_map.get('net_cf'): net_cf,
                    main_fields_map.get('pure_alpha'): net_cf / Product_cost,
                    main_fields_map.get('buffer'): j_1,
                    main_fields_map.get('cost_minus_cf'): Product_cost - net_cf
                }
                payload = {k: v for k, v in payload.items() if k is not None}
                client_main.update(payload)
                st.success("✅ Successfully updated Main Channel on Thingspeak!")
        except Exception as e:
            st.error(f"❌ Failed to update Main Channel on Thingspeak: {e}")

        # 2. Loop through each asset and update its individual channel
        st.write("---")
        st.write("Updating individual asset holdings...")
        for asset in assets_config:
            ticker = asset['ticker']
            if ticker in asset_clients:
                try:
                    # Get the current holding from the number_input widget
                    current_holding = st.session_state[f'holding_{ticker}']
                    field_to_update = asset['holding_channel']['field']
                    
                    # Update the specific channel for this asset
                    client_to_update = asset_clients[ticker]
                    client_to_update.update({field_to_update: current_holding})
                    st.success(f"✅ Successfully updated holding for {ticker}.")
                except Exception as e:
                    st.error(f"❌ Failed to update holding for {ticker}: {e}")
            else:
                st.info(f"ℹ️ Skipping update for {ticker} (no client configured).")


# --- Chart Display Section ---
st.write("_____")
main_channel_id = main_channel_config.get('channel_id')
main_fields_map = main_channel_config.get('fields', {})

def create_chart_iframe(channel_id, field_name, chart_title):
    if channel_id and field_name:
        chart_number = field_name.replace('field', '')
        url = f'https://thingspeak.com/channels/{channel_id}/charts/{chart_number}?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&type=line&update=15'
        st.write(f"**{chart_title}**")
        components.iframe(url, width=800, height=200)
        st.write("_____")
    else:
        st.warning(f"Chart for '{chart_title}' cannot be displayed. Missing config.")

create_chart_iframe(main_channel_id, main_fields_map.get('net_cf'), 'Cashflow')
create_chart_iframe(main_channel_id, main_fields_map.get('pure_alpha'), 'Pure_Alpha')
create_chart_iframe(main_channel_id, main_fields_map.get('cost_minus_cf'), 'Product_cost - CF')
create_chart_iframe(main_channel_id, main_fields_map.get('buffer'), 'Buffer')
