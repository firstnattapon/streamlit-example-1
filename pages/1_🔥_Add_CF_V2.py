import streamlit as st
import numpy as np
import yfinance as yf
import thingspeak
import json
import streamlit.components.v1 as components

st.set_page_config(page_title="Add_CF_V2", page_icon="🔥")

# --- การตั้งค่าและเชื่อมต่อ THING SPEAK ---
try:
    # Channel สำหรับบันทึกข้อมูลหลัก
    channel_id = 2394198
    write_api_key = 'OVZNYQBL57GJW5JF'
    client = thingspeak.Channel(channel_id, write_api_key)

    # Channel สำหรับดึงข้อมูล Asset ล่าสุด
    channel_id_2 = 2528199
    write_api_key_2 = '2E65V8XEIPH9B2VV'
    client_2 = thingspeak.Channel(channel_id_2, write_api_key_2, fmt='json')
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ ThingSpeak: {e}")
    st.stop()


# --- ตั้งค่าข้อมูลหุ้น (จัดการง่ายในที่เดียว) ---
STOCKS_CONFIG = {
    'FFWM': {'field': 'field1', 'initial_price': 6.88},
    'NEGG': {'field': 'field2', 'initial_price': 25.20},
    'RIVN': {'field': 'field3', 'initial_price': 10.07},
    'APLS': {'field': 'field4', 'initial_price': 39.61},
    'NVTS': {'field': 'field5', 'initial_price': 3.05},
    'QXO':  {'field': 'field6', 'initial_price': 19.00},
    'RXRX': {'field': 'field7', 'initial_price': 5.40},
    'AGL':  {'field': 'field8', 'initial_price': 3.00}
}
TICKERS = list(STOCKS_CONFIG.keys())

# --- ฟังก์ชันดึงข้อมูลหุ้น (มี Cache เพื่อประสิทธิภาพ) ---
@st.cache_data(ttl=60) # Cache data for 60 seconds
def get_stock_data(tickers):
    """ดึงข้อมูลราคาล่าสุดของหุ้นทั้งหมดในครั้งเดียว"""
    data = {}
    for ticker in tickers:
        try:
            # fast_info เร็วกว่า .info มาก
            price = yf.Ticker(ticker).fast_info.get('lastPrice', 0)
            data[ticker] = price
        except Exception as e:
            st.warning(f"ไม่สามารถดึงข้อมูลราคาของ {ticker} ได้: {e}")
            data[ticker] = 0
    return data

# --- ฟังก์ชันดึง Asset จาก ThingSpeak (ใช้ Loop) ---
@st.cache_data(ttl=300) # Cache data for 5 minutes
def get_last_assets():
    """ดึงข้อมูล Asset ล่าสุดจาก ThingSpeak ทั้งหมด"""
    last_assets = {}
    for ticker, config in STOCKS_CONFIG.items():
        try:
            # แก้ไขการใช้ eval() เป็น float() ที่ปลอดภัยกว่า
            response = client_2.get_field_last(field=config['field'])
            # ใช้ float() แทน eval()
            asset_value = float(json.loads(response)[config['field']])
            last_assets[ticker] = asset_value
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            st.warning(f"ไม่สามารถดึง Asset ของ {ticker} ได้: {e}")
            last_assets[ticker] = 0.0
    return last_assets

# --- ดึงข้อมูล ---
stock_prices = get_stock_data(TICKERS)
last_assets = get_last_assets()

# --- ส่วนแสดงผลของ STREAMLIT ---
st.header("Stock Prices (Live)")
prices_col = st.columns(4)
current_prices = {}
i = 0
for ticker in TICKERS:
    with prices_col[i % 4]:
        # ใช้ข้อมูลที่ดึงมาแล้ว ไม่ต้องเรียก API ซ้ำ
        price_label = f"ราคา_{ticker}_{STOCKS_CONFIG[ticker]['initial_price']}"
        current_prices[ticker] = st.number_input(price_label, step=0.01, value=stock_prices.get(ticker, 0.0), key=f"price_{ticker}")
    i += 1

st.write("---")

st.header("Asset Values")
assets_col = st.columns(4)
asset_values = {}
total_asset_value = 0
i = 0
for ticker in TICKERS:
    with assets_col[i % 4]:
        # ใช้ Loop ทำให้โค้ดสั้นลง
        asset_label = f"{ticker}_asset"
        shares = st.number_input(asset_label, step=0.01, value=last_assets.get(ticker, 0.0), key=f"asset_{ticker}")
        value = shares * current_prices[ticker]
        asset_values[ticker] = value
        st.write(f"Value: ${value:,.2f}")
        total_asset_value += value
    i += 1

st.write("---")

# --- การคำนวณ Portfolio ---
Product_cost = st.number_input('Product_cost', step=0.01, value=10750.0)
j_1 = st.number_input('Portfolio_cash', step=0.01, value=0.00)

now_pv = total_asset_value + j_1
st.metric("Current Portfolio Value (now_pv)", f"${now_pv:,.2f}")

st.write("---")

# --- การคำนวณ Log (Hedge/Fix) ---
# แก้ไขการคำนวณให้ถูกต้องและมีประสิทธิภาพ
t_0 = np.prod([config['initial_price'] for config in STOCKS_CONFIG.values()])
t_n = np.prod(list(current_prices.values()))

# ป้องกันการหารด้วยศูนย์
if t_n > 0:
    ln = -1500 * np.log(t_0 / t_n)
else:
    ln = 0
    st.warning("t_n เป็นศูนย์, ไม่สามารถคำนวณ log ได้")

st.metric("Log Adjustment (fix)", f"{ln:,.2f}")
log_pv = Product_cost + ln
st.metric("Adjusted Portfolio Value (log_pv)", f"${log_pv:,.2f}")

st.write("---")
net_cf = now_pv - log_pv
st.metric("Net Cash Flow (net_cf)", f"${net_cf:,.2f}", delta=f"{net_cf:,.2f}")
st.write("---")

# --- ปุ่มดำเนินการ ---
if st.button("Rerun Page"):
    st.rerun()

if st.checkbox('Confirm to ADD Cashflow'):
    if st.button("ADD_CF"):
        try:
            payload = {
                'field1': net_cf,
                'field2': net_cf / Product_cost if Product_cost != 0 else 0,
                'field3': j_1,
                'field4': Product_cost - net_cf
            }
            client.update(payload)
            st.success("ส่งข้อมูลไปที่ ThingSpeak สำเร็จ!")
            st.write(payload)
        except Exception as e:
            # แสดงข้อผิดพลาดให้ผู้ใช้เห็น ไม่ใช่ pass เฉยๆ
            st.error(f"ส่งข้อมูลไม่สำเร็จ: {e}")

# --- แสดงผลกราฟ ---
st.write("---")
st.header("ThingSpeak Charts")
components.iframe('https://thingspeak.com/channels/2394198/charts/1?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&type=line&update=15', width=800, height=250)
components.iframe('https://thingspeak.com/channels/2394198/charts/2?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&type=line&update=15', width=800, height=250)
components.iframe('https://thingspeak.mathworks.com/channels/2394198/charts/4?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&type=line&update=15', width=800, height=250)
components.iframe('https://thingspeak.com/channels/2394198/charts/3?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&type=line&update=15', width=800, height=250)
