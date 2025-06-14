import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import thingspeak
import json
from pathlib import Path
import math # <-- 1. เพิ่มการ import math สำหรับฟังก์ชัน log

# --- 1. การตั้งค่าและโหลด Configuration ---
st.set_page_config(page_title="Calculator", page_icon="⌨️")

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

if st.button("Rerun"):
    st.rerun()

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

# --- 2.1. แก้ไขฟังก์ชัน production_cost ---
@st.cache_data(ttl=60)
def production_cost(ticker, t0, fix):
    """
    Calculates Production based on the new formula:
    production = (fix * -1) * ln(t0 / current_price)
    """
    # ตรวจสอบค่า input พื้นฐาน
    if t0 <= 0 or fix == 0:
        return 0.0

    try:
        1. ดึงราคาปัจจุบัน
        ticker_info = yf.Ticker(ticker)
        current_price = ticker_info.fast_info['lastPrice']

        2. ป้องกันการหารด้วยศูนย์ หรือ log ของค่าที่ไม่ใช่บวก
        if current_price <= 0:
            st.warning(f"Cannot calculate production for {ticker}: Current price is {current_price}, which is invalid for the formula.")
            return None

        #3. คำนวณตามสมการใหม่
        max_production_value = (fix * -1) * math.log(t0 / 0.01)
        now_production_value = (fix * -1) * math.log(t0 /current_price)

        return max_production_value , now_production_value

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
    
    display_df = pd.DataFrame(index=['+0', "+1", "+2", "+3", "+4"])
    combined_df = pd.concat([filtered_data, display_df]).fillna("")
    combined_df['action'] = rng.integers(2, size=len(combined_df))
    
    if 'index' not in combined_df.columns:
        combined_df['index'] = ""
    
    if not filtered_data.empty:
        combined_df.loc[filtered_data.index, 'index'] = range(1, len(filtered_data) + 1)
        
    return combined_df.tail(7), fx_js

# --- 3. ส่วนแสดงผลหลัก (Main Display Logic) ---

def main():
    """Main function to run the Streamlit app."""
    st.write('____')
    
    avg_cf_config = CONFIG.get('average_cf_config')
    if avg_cf_config:
        cf_day = average_cf(avg_cf_config)
        st.write(f"average_cf_day: {cf_day:.2f} USD  :  average_cf_mo: {cf_day * 30:.2f} USD")
    else:
        st.warning("`average_cf_config` not found in configuration file.")
    st.write('____')
    
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

        df_7, fx_js = monitor(channel_id, api_key, ticker, monitor_field, monitor_filter_date)
        
        # --- 3.1. แก้ไขการเรียกใช้ฟังก์ชัน production_cost ---
        # อ่านค่า t0 และ fix จาก config แล้วส่งไปให้ฟังก์ชัน
        max , now = production_cost(
            ticker=ticker,
            t0=prod_params.get('t0', 0.0),      # ดึงค่า t0, ถ้าไม่มีให้เป็น 0.0
            fix=prod_params.get('fix', 0.0)     # ดึงค่า fix, ถ้าไม่มีให้เป็น 0.0
        )
        
        # prod_cost_display = f"{prod_cost[0]:.2f}" if prod_cost is not None else "N/A"
        
        st.write(ticker)
        st.write(f"f(x): {fx_js} ,  , Production: max")
        st.table(df_7)
        st.write("_____")

    st.write("***ก่อนตลาดเปิดตรวจสอบ TB ล่าสุด > RE เมื่อตลอดเปิด")
    st.write("***RE > 60 USD")
    st.stop()

if __name__ == "__main__":
    main()
