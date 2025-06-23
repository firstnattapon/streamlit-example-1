import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import thingspeak
import json
from pathlib import Path
import math # ใช้ในฟังก์ชัน production_cost

# --- 1. การตั้งค่าและโหลด Configuration ---
st.set_page_config(page_title="Asset Monitor", page_icon="📈" , layout= "wide" )

@st.cache_data(ttl=300) # Cache config data for 5 minutes
def load_config(filepath="calculator_config.json"):
    """
    Loads the configuration from a JSON file with error handling.
    """
    config_path = Path(filepath)
    if not config_path.is_file():
        st.error(f"Error: Configuration file not found at '{filepath}'")
        st.stop()
    try:
        with config_path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from '{filepath}'. Please check for syntax errors.")
        st.stop()

# โหลด config
CONFIG = load_config()

# --- 2. ฟังก์ชันที่ปรับปรุงแล้ว (Refactored Functions) ---

@st.cache_data(ttl=600) # Cache a Ticker's history for 10 minutes
def get_ticker_history(ticker_symbol):
    """Fetches and processes historical data for a given ticker."""
    ticker = yf.Ticker(ticker_symbol)
    history = ticker.history(period='max')[['Close']]
    history.index = history.index.tz_convert(tz='Asia/Bangkok')
    return round(history, 3)

def average_cf(cf_config):
    """
    Calculates average CF. Uses .get() for safety to prevent KeyErrors.
    """
    history = get_ticker_history(cf_config['ticker'])
    default_date = "2024-01-01 12:00:00+07:00"
    filter_date = cf_config.get('filter_date', default_date)
    filtered_data = history[history.index >= filter_date]
    count_data = len(filtered_data)
    if count_data == 0:
        return 0
    client = thingspeak.Channel(
        id=cf_config['channel_id'],
        api_key=cf_config['write_api_key'],
        fmt='json'
    )
    field_data = client.get_field_last(field=f"{cf_config['field']}")
    value = int(eval(json.loads(field_data)[f"field{cf_config['field']}"]))
    adjusted_value = value - cf_config.get('offset', 0)
    return adjusted_value / count_data

@st.cache_data(ttl=60)
def production_cost(ticker, t0, fix):
    """
    Calculates Production based on the new formula:
    production = (fix * -1) * ln(t0 / current_price)
    Returns a tuple (max_production, now_production) or None on error.
    """
    if t0 <= 0 or fix == 0:
        return None
    try:
        ticker_info = yf.Ticker(ticker)
        current_price = ticker_info.fast_info['lastPrice']
        if current_price <= 0:
            st.warning(f"Cannot calculate production for {ticker}: Current price is {current_price}, which is invalid for the formula.")
            return None
        # สมมติราคาต่ำสุดที่เป็นไปได้คือ 0.01 เพื่อคำนวณค่า Max Production
        max_production_value = (fix * -1) * math.log(t0 / 0.01)
        now_production_value = (fix * -1) * math.log(t0 / current_price)
        return max_production_value, now_production_value
    except Exception as e:
        st.warning(f"Could not calculate Production for {ticker}: {e}")
        return None

def monitor(channel_id, api_key, ticker, field, filter_date):
    """Monitors an asset. Now robust to missing data."""
    thingspeak_client = thingspeak.Channel(id=channel_id, api_key=api_key, fmt='json')
    history = get_ticker_history(ticker)
    filtered_data = history[history.index >= filter_date].copy()
    try:
        field_data = thingspeak_client.get_field_last(field=f'{field}')
        fx_js = int(json.loads(field_data)[f"field{field}"])
    except (json.JSONDecodeError, KeyError, TypeError):
        fx_js = 0

    rng = np.random.default_rng(fx_js)
    # สร้าง DataFrame สำหรับแสดงผล โดยมี 5 แถวสำหรับข้อมูลคาดการณ์
    future_rows = pd.DataFrame(index=['+0', "+1", "+2", "+3", "+4"])
    combined_df = pd.concat([filtered_data, future_rows]).fillna("")
    # กำหนด 'action' ให้กับทุกแถว (ทั้งอดีตและอนาคต)
    combined_df['action'] = rng.integers(2, size=len(combined_df))
    # จัดการ index ให้แสดงเป็นลำดับตัวเลขสำหรับข้อมูลที่มีอยู่จริง
    if not filtered_data.empty:
        combined_df.loc[filtered_data.index, 'Row'] = range(1, len(filtered_data) + 1)
    # เปลี่ยนชื่อ Index ของ DataFrame เพื่อความชัดเจน
    combined_df.index.name = "Date"
    return combined_df.tail(7), fx_js

# --- 3. ส่วนแสดงผลหลัก (Main Display Logic) ---

def main():
    """Main function to run the Streamlit app."""
    st.title("📈 Asset Monitor Dashboard")

    if st.button("🔄️ Rerun & Fetch Latest Data"):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    # --- ส่วนแสดงผล Average CF ---
    st.header("Cost of Funds (CF)")
    avg_cf_config = CONFIG.get('average_cf_config')
    if avg_cf_config:
        cf_day = average_cf(avg_cf_config)
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Average CF (Daily)", value=f"{cf_day:.2f} USD")
        with col2:
            st.metric(label="Average CF (Monthly)", value=f"{cf_day * 30:.2f} USD")
    else:
        st.warning("`average_cf_config` not found in configuration file.")
    
    st.divider()

    # --- ส่วนแสดงผลของแต่ละ Asset ---
    st.header("Asset Details")
    monitor_config = CONFIG.get('monitor_config', {})
    default_monitor_date = "2025-04-28 12:00:00+07:00"
    monitor_filter_date = monitor_config.get('filter_date', default_monitor_date)

    for asset_config in CONFIG.get('assets', []):
        ticker = asset_config.get('ticker', 'N/A')
        monitor_field = asset_config.get('monitor_field')
        prod_params = asset_config.get('production_params', {})
        channel_id = asset_config.get('channel_id')
        api_key = asset_config.get('write_api_key')

        if not all([ticker, monitor_field, channel_id, api_key]):
            st.warning(f"Skipping an asset due to missing configuration: {asset_config}")
            continue
        
        # ใช้ st.expander เพื่อจัดกลุ่มข้อมูลของแต่ละ asset
        with st.expander(f"📊 {ticker}", expanded=True):
            df_7, fx_js = monitor(channel_id, api_key, ticker, monitor_field, monitor_filter_date)

            prod_cost = production_cost(
                ticker=ticker,
                t0=prod_params.get('t0', 0.0),
                fix=prod_params.get('fix', 0.0)
            )

            prod_cost_max = prod_cost[0] if prod_cost is not None else 0.0
            prod_cost_now = prod_cost[1] if prod_cost is not None else 0.0

            # ใช้ st.columns และ st.metric เพื่อการแสดงผลที่สวยงาม
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="f(x) from ThingSpeak", value=f"{fx_js}")
            with col2:
                st.metric(label="Max Production", value=f"{prod_cost_max:,.2f}")
            with col3:
                st.metric(label="Current Production", value=f"{prod_cost_now:,.2f}")
            
            st.write("Recent & Forecasted Data")
            
            # --- ใช้ st.dataframe พร้อมตกแต่งคอลัมน์ ---
            st.dataframe(
                df_7,
                use_container_width=True,
                column_config={
                    "Row": st.column_config.NumberColumn(
                        "Row",
                        help="ลำดับของข้อมูลที่มีอยู่จริง",
                        format="%d"
                    ),
                    "Close": st.column_config.NumberColumn(
                        "Close Price (USD)",
                        help="ราคาปิดของสินทรัพย์",
                        format="$%.3f",
                    ),
                    "action": st.column_config.SelectboxColumn(
                        "Action",
                        help="คำแนะนำการดำเนินการแบบสุ่ม (0=Hold/Sell, 1=Buy)",
                        options=[0, 1],
                    )
                }
            )

    st.divider()
    st.info("""
    **คำแนะนำการใช้งาน:**
    - **ก่อนตลาดเปิด:** ตรวจสอบข้อมูลล่าสุดจาก ThingSpeak
    - **เมื่อตลาดเปิด:** กดปุ่ม "Rerun" เพื่อดึงราคาและข้อมูลล่าสุด
    - **RE > 60 USD:** อาจเป็นเงื่อนไขเฉพาะสำหรับการตัดสินใจ
    """)

if __name__ == "__main__":
    main()
