import pandas as pd
import numpy as np
# from numba import njit # ไม่ได้ถูกใช้งาน จึงเอาออกได้
import yfinance as yf
import streamlit as st
import thingspeak
import json
import streamlit.components.v1 as components

st.set_page_config(page_title="Add_CF_V2", page_icon="🔥")

# --- Function to load configuration ---
@st.cache_data
def load_config(filename="add_cf_config.json"):
    """Loads asset configuration from a JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)['assets']
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        st.error(f"Error loading or parsing {filename}: {e}")
        st.stop()
        return []

# --- Thingspeak Clients Initialization ---
# (ส่วนนี้ยังคงเหมือนเดิม)
channel_id_log = 2329127
write_api_key_log = 'V10DE0HKR4JKB014'
client_log = thingspeak.Channel(channel_id_log, write_api_key_log)

channel_id = 2394198
write_api_key = 'OVZNYQBL57GJW5JF'
client = thingspeak.Channel(channel_id, write_api_key)

channel_id_2 = 2528199
write_api_key_2 = '2E65V8XEIPH9B2VV'
client_2 = thingspeak.Channel(channel_id_2, write_api_key_2, fmt='json')

# --- Load Asset Configuration ---
assets_config = load_config()

# --- Data Fetching and UI Generation using Loops ---
# สร้าง Dictionary เพื่อเก็บข้อมูลต่างๆ ที่จะใช้
asset_data = {}
for asset in assets_config:
    ticker = asset["ticker"]
    asset_data[ticker] = {}

# ดึงข้อมูลล่าสุดจาก yfinance และ thingspeak
for asset in assets_config:
    ticker = asset["ticker"]
    field = asset["thingspeak_field"]
    try:
        # Get last asset holding from Thingspeak
        last_asset_json = client_2.get_field_last(field=field)
        last_holding = eval(json.loads(last_asset_json)[field])
        asset_data[ticker]['last_holding'] = last_holding
    except Exception as e:
        st.warning(f"Could not fetch last holding for {ticker} from Thingspeak. Defaulting to 0. Error: {e}")
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
    # สร้าง label ที่เหมือนของเดิม แต่ใช้ข้อมูลจาก config
    label = f"ราคา_{ticker}_{asset['reference_price']}"
    # ดึงราคาล่าสุดที่ดึงมาเป็นค่าเริ่มต้น
    price_value = asset_data[ticker].get('last_price', 0.0)
    # สร้าง st.number_input และเก็บค่าที่ผู้ใช้อาจแก้ไขไว้ใน dictionary
    current_prices[ticker] = st.number_input(label, step=0.01, value=price_value, key=f"price_{ticker}")

st.write("_____")

# แสดง Input field สำหรับจำนวนสินทรัพย์ และคำนวณมูลค่า
total_asset_value = 0.0
for asset in assets_config:
    ticker = asset["ticker"]
    # ดึงจำนวนที่ถือล่าสุดมาเป็นค่าเริ่มต้น
    holding_value = asset_data[ticker].get('last_holding', 0.0)
    # สร้าง st.number_input
    asset_holding = st.number_input(f"{ticker}_asset", step=0.01, value=holding_value, key=f"holding_{ticker}")

    # คำนวณมูลค่าของสินทรัพย์แต่ละตัว
    individual_asset_value = asset_holding * current_prices[ticker]
    st.write(individual_asset_value)
    
    # เพิ่มมูลค่าเข้ายอดรวม
    total_asset_value += individual_asset_value

st.write("_____")

# --- Final Calculations ---
Product_cost = st.number_input('Product_cost', step=0.01, value=10750.)
j_1 = st.number_input('Portfolio_cash', step=0.01, value=0.00)

# 'number' คือมูลค่าพอร์ตปัจจุบัน (เหมือนเดิม)
number = total_asset_value + j_1
st.write('now_pv:', number)

st.write("_____")

# คำนวณ t_0 และ t_n จาก config และราคาปัจจุบัน โดยใช้ Loop
reference_prices = [asset['reference_price'] for asset in assets_config]
# ใช้ราคาจาก input field เพื่อให้แน่ใจว่าค่าตรงกับที่ user เห็น
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
# (ส่วนนี้ยังคงเหมือนเดิมทุกประการ)
if st.button("rerun"):
    st.rerun()
st.write("_____")

Check_ADD = st.checkbox('ADD_CF ')
if Check_ADD:
    button_ADD = st.button("ADD_CF")
    if button_ADD:
        try:
            client.update({'field1': net_cf, 'field2': net_cf / Product_cost, 'field3': j_1, 'field4': Product_cost - net_cf})
            st.write({'Cashflow': net_cf, 'Pure_Alpha': net_cf / Product_cost, 'ฺBuffer': j_1})
        except Exception as e:
            st.error(f"Failed to update Thingspeak: {e}")

# --- Chart Display Section ---
# (ส่วนนี้ยังคงเหมือนเดิมทุกประการ)
st.write("_____")
st.write("Cashflow")
components.iframe('https://thingspeak.com/channels/2394198/charts/1?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&type=line&update=15', width=800, height=200)
st.write("_____")
st.write("Pure_Alpha")
components.iframe('https://thingspeak.com/channels/2394198/charts/2?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&type=line&update=15', width=800, height=200)
st.write("Product_cost")
components.iframe('https://thingspeak.mathworks.com/channels/2394198/charts/4?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&type=line&update=15', width=800, height=200)
st.write("_____")
st.write("ฺBuffer")
components.iframe('https://thingspeak.com/channels/2394198/charts/3?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&type=line&update=15', width=800, height=200)
st.write("_____")
