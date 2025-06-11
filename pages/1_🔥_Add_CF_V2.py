import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import thingspeak
import json
import streamlit.components.v1 as components

st.set_page_config(page_title="Add_CF_V2", page_icon="🔥")

# --- Function to load configuration ---
@st.cache_data
def load_config(filename="add_cf_config.json"):
    """Loads complete configuration from a JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.error(f"Error loading or parsing {filename}: {e}")
        st.stop()

# --- Thingspeak Clients Initialization ---
channel_id_log = 2329127
write_api_key_log = 'V10DE0HKR4JKB014'
client_log = thingspeak.Channel(channel_id_log, write_api_key_log)

channel_id = 2394198
write_api_key = 'OVZNYQBL57GJW5JF'
client = thingspeak.Channel(channel_id, write_api_key)

channel_id_2 = 2528199
write_api_key_2 = '2E65V8XEIPH9B2VV'
client_2 = thingspeak.Channel(channel_id_2, write_api_key_2, fmt='json')

# --- Load Complete Configuration ---
config = load_config()
assets_config = config.get('assets', [])
product_cost_default = config.get('product_cost_default', 0.0)

# --- Data Fetching ---
asset_data = {}
for asset in assets_config:
    ticker = asset["ticker"]
    asset_data[ticker] = {}

for asset in assets_config:
    ticker = asset["ticker"]
    field = asset["thingspeak_field"]
    try:
        last_asset_json = client_2.get_field_last(field=field)
        last_holding = eval(json.loads(last_asset_json)[field])
        asset_data[ticker]['last_holding'] = last_holding
    except Exception as e:
        asset_data[ticker]['last_holding'] = 0.0

    try:
        last_price = yf.Ticker(ticker).fast_info['lastPrice']
        asset_data[ticker]['last_price'] = last_price
    except Exception as e:
        asset_data[ticker]['last_price'] = asset.get('reference_price', 0.0)

# --- Streamlit UI Section ---
current_prices = {}
for asset in assets_config:
    ticker = asset["ticker"]
    # *** แก้ไข Label ของ NEGG ให้ตรงกับ V1 เพื่อลดความสับสน ***
    if ticker == 'NEGG':
        label = f"ราคา_NEGG_1.26 , 25.20"
    else:
        label = f"ราคา_{ticker}_{asset['reference_price']}"
    
    price_value = asset_data[ticker].get('last_price', 0.0)
    current_prices[ticker] = st.number_input(label, step=0.01, value=price_value, key=f"price_{ticker}")

st.write("_____")

total_asset_value = 0.0
for asset in assets_config:
    ticker = asset["ticker"]
    holding_value = asset_data[ticker].get('last_holding', 0.0)
    asset_holding = st.number_input(f"{ticker}_asset", step=0.01, value=holding_value, key=f"holding_{ticker}")
    # ใช้ราคาจาก UI Input (`current_prices`) สำหรับการคำนวณมูลค่าพอร์ต (now_pv)
    individual_asset_value = asset_holding * current_prices[ticker]
    st.write(individual_asset_value)
    total_asset_value += individual_asset_value

st.write("_____")

# --- Final Calculations ---
Product_cost = st.number_input('Product_cost', step=0.01, value=product_cost_default)
j_1 = st.number_input('Portfolio_cash', step=0.01, value=0.00)

number = total_asset_value + j_1
st.write('now_pv:', number)

st.write("_____")

# *** จุดที่แก้ไข 1: การคำนวณ t_0 ใช้ reference_price จาก config ที่แก้ไขแล้ว ***
# (ต้องแน่ใจว่า NEGG ใน json เป็น 25.20)
reference_prices = [asset['reference_price'] for asset in assets_config]
t_0 = np.prod(reference_prices)

# *** จุดที่แก้ไข 2: การคำนวณ t_n ให้ดึงราคาใหม่เหมือน V1 ทุกประการ ***
st.write("Fetching latest prices for t_n calculation...") # เพิ่มข้อความให้ผู้ใช้รู้ว่ากำลังทำอะไร
live_prices_for_tn = []
for asset in assets_config:
    try:
        # ใช้ .info['currentPrice'] เหมือน V1
        price = yf.Ticker(asset['ticker']).info['currentPrice']
        live_prices_for_tn.append(price)
    except Exception as e:
        st.warning(f"Could not fetch .info['currentPrice'] for {asset['ticker']}. Using price from input box.")
        # หากดึง .info ไม่ได้ ให้ใช้ราคาจากหน้าจอแทนเพื่อป้องกัน error
        live_prices_for_tn.append(current_prices[asset['ticker']])

t_n = np.prod(live_prices_for_tn)


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

# --- ส่วนที่เหลือของโค้ดเหมือนเดิม ---
if st.button("rerun"):
    st.rerun()
# ... (โค้ดส่วนปุ่มและกราฟ)
