#เริ่ม โค๊ด
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import thingspeak
import json
import streamlit.components.v1 as components

st.set_page_config(page_title="Add_CF_V3_Dynamic", page_icon="🔥")

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

# --- Load Configuration ---
config = load_config()
if not config:
    st.stop()

assets_config = config.get('assets', [])
product_cost_default = config.get('product_cost_default', 0.0)
primary_update_channel_config = config.get('primary_update_channel')


# --- Thingspeak Clients Initialization (Dynamic) ---
thingspeak_clients = {}

def get_thingspeak_client(channel_id, api_key):
    """Creates and caches Thingspeak client instances to avoid duplicates."""
    if channel_id not in thingspeak_clients:
        # We create clients with fmt='json' for easier parsing
        thingspeak_clients[channel_id] = thingspeak.Channel(channel_id, api_key, fmt='json')
    return thingspeak_clients[channel_id]

# Create the primary client for updating final results
if primary_update_channel_config:
    primary_client = get_thingspeak_client(
        primary_update_channel_config['channel_id'],
        primary_update_channel_config['write_api_key']
    )
else:
    st.error("`primary_update_channel` not found in config file.")
    st.stop()
    primary_client = None


# --- Data Fetching and UI Generation using Loops ---
# สร้าง Dictionary เพื่อเก็บข้อมูลต่างๆ ที่จะใช้
asset_data = {}

# ดึงข้อมูลล่าสุดจาก yfinance และ thingspeak
for asset in assets_config:
    ticker = asset["ticker"]
    field_num = asset["thingspeak_field"]
    
    # Initialize dictionary for the ticker
    asset_data[ticker] = {}

    try:
        # Get last asset holding from its specific Thingspeak channel
        data_source_client = get_thingspeak_client(
            asset['data_source_channel_id'],
            asset['data_source_api_key']
        )
        
        # The key in the returned JSON will be 'fieldX'
        field_key = f'field{field_num}'
        
        # get_field_last requires the field number as an integer
        last_asset_json = data_source_client.get_field_last(field=field_num) 
        
        # Parse the JSON response
        last_holding = float(json.loads(last_asset_json)[field_key])
        asset_data[ticker]['last_holding'] = last_holding

    except Exception as e:
        st.warning(f"Could not fetch last holding for {ticker} (Field {field_num}) from Channel {asset['data_source_channel_id']}. Defaulting to 0. Error: {e}")
        asset_data[ticker]['last_holding'] = 0.0

    try:
        # Get last price from yfinance
        last_price = yf.Ticker(ticker).fast_info['lastPrice']
        asset_data[ticker]['last_price'] = last_price
    except Exception as e:
        st.warning(f"Could not fetch price for {ticker} from yfinance. Defaulting to reference price. Error: {e}")
        asset_data[ticker]['last_price'] = asset.get('reference_price', 0.0)


# --- Streamlit UI Section ---

# แสดง Input field สำหรับราคา โดยใช้ข้อมูลที่ดึงมา
current_prices = {}
for asset in assets_config:
    ticker = asset["ticker"]
    label = f"ราคา_{ticker}_{asset['reference_price']}"
    price_value = asset_data[ticker].get('last_price', 0.0)
    current_prices[ticker] = st.number_input(label, step=0.01, value=price_value, key=f"price_{ticker}")

st.write("_____")

# แสดง Input field สำหรับจำนวนสินทรัพย์ และคำนวณมูลค่า
total_asset_value = 0.0
for asset in assets_config:
    ticker = asset["ticker"]
    holding_value = asset_data[ticker].get('last_holding', 0.0)
    asset_holding = st.number_input(f"{ticker}_asset", step=0.01, value=holding_value, key=f"holding_{ticker}")

    individual_asset_value = asset_holding * current_prices[ticker]
    st.write(f"มูลค่า {ticker}: {individual_asset_value:,.2f}")
    
    total_asset_value += individual_asset_value

st.write("_____")

# --- Final Calculations ---
Product_cost = st.number_input('Product_cost', step=0.01, value=product_cost_default)
j_1 = st.number_input('Portfolio_cash', step=0.01, value=0.00)

number = total_asset_value + j_1
st.write('now_pv:', number)

st.write("_____")

# คำนวณ t_0 และ t_n จาก config และราคาปัจจุบัน
reference_prices = [asset['reference_price'] for asset in assets_config]
live_prices = [current_prices[asset['ticker']] for asset in assets_config]

t_0 = np.prod(reference_prices)
t_n = np.prod(live_prices)

ln = -1500 * np.log(t_0 / t_n) if t_0 > 0 and t_n > 0 else 0

st.write('t_0', t_0)
st.write('t_n', t_n)
st.write('fix', ln)
st.write('log_pv', Product_cost + ln)
st.write('now_pv', number)
st.write('____')
net_cf = number - (Product_cost + ln)
st.write('net_cf', net_cf)
st.write('____')


# --- Buttons and Thingspeak Update Section ---
if st.button("rerun"):
    st.rerun()
st.write("_____")

with st.expander("⚠️ Confirm to Add Cashflow"):
    st.write("Click the button below to confirm and send data to the primary Thingspeak channel.")
    
    if st.button("Confirm and Send"):
        if primary_client:
            try:
                if Product_cost == 0:
                    st.error("Product_cost cannot be zero.")
                else:
                    # Update the primary channel with the calculated results
                    payload = {
                        'field1': net_cf,
                        'field2': net_cf / Product_cost,
                        'field3': j_1,
                        'field4': Product_cost - net_cf
                    }
                    primary_client.update(payload)
                    st.success(f"Successfully updated Thingspeak Channel {primary_update_channel_config['channel_id']}!")
                    st.write(payload)

            except Exception as e:
                st.error(f"Failed to update Thingspeak: {e}")
        else:
            st.error("Primary Thingspeak client is not configured. Cannot send data.")


# --- Chart Display Section ---
# Display charts from the primary update channel
if primary_update_channel_config:
    ch_id = primary_update_channel_config['channel_id']
    st.write("_____")
    st.write(f"### Cashflow (Channel: {ch_id})")
    components.iframe(f'https://thingspeak.com/channels/{ch_id}/charts/1?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&type=line&update=15', width=800, height=200)
    st.write("_____")
    st.write(f"### Pure_Alpha (Channel: {ch_id})")
    components.iframe(f'https://thingspeak.com/channels/{ch_id}/charts/2?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&type=line&update=15', width=800, height=200)
    st.write(f"### Product_cost (Channel: {ch_id})")
    components.iframe(f'https://thingspeak.mathworks.com/channels/{ch_id}/charts/4?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&type=line&update=15', width=800, height=200)
    st.write("_____")
    st.write(f"### Buffer (Channel: {ch_id})")
    components.iframe(f'https://thingspeak.com/channels/{ch_id}/charts/3?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&type=line&update=15', width=800, height=200)
    st.write("_____")
