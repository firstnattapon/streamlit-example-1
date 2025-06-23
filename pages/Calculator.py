import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import thingspeak
import json
from pathlib import Path
import math

# --- 1. การตั้งค่าและโหลด Configuration ---
st.set_page_config(page_title="Calculator", page_icon="⌨️" , layout= "centered" )

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
        # 1. ดึงราคาปัจจุบัน
        ticker_info = yf.Ticker(ticker)
        current_price = ticker_info.fast_info['lastPrice']

        # 2. ป้องกันการหารด้วยศูนย์ หรือ log ของค่าที่ไม่ใช่บวก
        if current_price <= 0:
            st.warning(f"Cannot calculate production for {ticker}: Current price is {current_price}, which is invalid for the formula.")
            return None

        # 3. คำนวณตามสมการใหม่
        # สมมติราคาต่ำสุดที่เป็นไปได้คือ 0.01 เพื่อคำนวณค่า Max Production
        max_production_value = (fix * -1) * math.log(t0 / 0.01)
        now_production_value = (fix * -1) * math.log(t0 / current_price)

        return max_production_value, now_production_value

    except Exception as e:
        st.warning(f"Could not calculate Production for {ticker}: {e}")
        return None

# --- 2.1. แก้ไขฟังก์ชัน monitor เพื่อให้ทำงานกับ st.dataframe ได้ดีขึ้น ---
def monitor(channel_id, api_key, ticker, field, filter_date):
    """
    Monitors an asset. Creates a DataFrame suitable for st.dataframe.
    Now robust to missing data and uses a clearer column name.
    """
    thingspeak_client = thingspeak.Channel(id=channel_id, api_key=api_key, fmt='json')
    history = get_ticker_history(ticker)

    # กรองข้อมูล
    filtered_data = history[history.index >= filter_date].copy()

    # ดึงข้อมูลจาก Thingspeak
    try:
        field_data = thingspeak_client.get_field_last(field=f'{field}')
        fx_js = int(json.loads(field_data)[f"field{field}"])
    except (json.JSONDecodeError, KeyError, TypeError):
        fx_js = 0

    # สร้าง DataFrame สำหรับแสดงผล
    # 1. เพิ่มคอลัมน์ 'ลำดับ' สำหรับข้อมูลที่มีอยู่
    if not filtered_data.empty:
        filtered_data['ลำดับ'] = range(1, len(filtered_data) + 1)
    else:
        # ทำให้คอลัมน์ 'ลำดับ' มีอยู่เสมอแม้ไม่มีข้อมูล
        filtered_data['ลำดับ'] = pd.Series(dtype='object')

    # 2. สร้างแถวว่างสำหรับข้อมูลในอนาคต
    future_rows_index = [f"+{i}" for i in range(5)] # +0, +1, +2, +3, +4
    future_df = pd.DataFrame(index=future_rows_index, columns=filtered_data.columns)

    # 3. รวมข้อมูลอดีตและอนาคต
    combined_df = pd.concat([filtered_data, future_df])

    # 4. เพิ่มคอลัมน์ 'action'
    rng = np.random.default_rng(fx_js)
    combined_df['action'] = rng.integers(2, size=len(combined_df))

    # 5. เลือกเฉพาะ 7 แถวสุดท้ายและจัดเรียงคอลัมน์
    display_df = combined_df[['ลำดับ', 'Close', 'action']].tail(7).fillna("")
    
    # 6. จัดรูปแบบคอลัมน์ 'Close' ให้สวยงาม
    if 'Close' in display_df:
        # แปลงเป็นตัวเลข ถ้าแปลงไม่ได้ให้เป็น NaT แล้วจัดรูปแบบ
        display_df['Close'] = pd.to_numeric(display_df['Close'], errors='coerce').apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else ""
        )

    return display_df, fx_js


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

        prod_cost = production_cost(
            ticker=ticker,
            t0=prod_params.get('t0', 0.0),
            fix=prod_params.get('fix', 0.0)
        )

        prod_cost_max_display = f"{prod_cost[0]:.2f}" if prod_cost is not None else "N/A"
        prod_cost_now_display = f"{prod_cost[1]:.2f}" if prod_cost is not None else "N/A"

        st.write(ticker)
        st.write(f"f(x): {fx_js} ,   Production_max : {prod_cost_max_display}  , Production_now : {prod_cost_now_display}")
        
        # --- 3.1. เปลี่ยนจาก st.table เป็น st.dataframe ---
        st.dataframe(df_7, use_container_width=True)
        
        st.write("_____")

    st.write("***ก่อนตลาดเปิดตรวจสอบ TB ล่าสุด > RE เมื่อตลอดเปิด")
    st.write("***RE > 60 USD")

if __name__ == "__main__":
    main()
