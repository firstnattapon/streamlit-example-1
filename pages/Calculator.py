import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import thingspeak
import json
from pathlib import Path

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
    Calculates average CF based on parameters from the config dict.
    สร้าง Thingspeak client และใช้ filter_date จาก config
    """
    history = get_ticker_history(cf_config['ticker'])
    
    # อ่านค่า filter_date จาก config แทนการ hardcode
    filter_date = cf_config['filter_date'] # <--- เปลี่ยนแปลงตรงนี้
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
    adjusted_value = value - cf_config['offset']
    
    return adjusted_value / count_data

@st.cache_data(ttl=60) # Cache production cost calculation for 1 minute
def production_cost(ticker, fixed_asset_value, cash_balance):
    """Calculates Production Costs. Parameters are now passed in."""
    try:
        ticker_info = yf.Ticker(ticker)
        entry_price = ticker_info.fast_info['lastPrice']
        step = 0.01

        samples = np.arange(step, np.around(entry_price, 2) * 3 + step, step)
        
        df = pd.DataFrame({'Asset_Price': np.around(samples, 2)})
        df['Fixed_Asset_Value'] = fixed_asset_value
        df['Amount_Asset'] = df['Fixed_Asset_Value'] / df['Asset_Price']

        df_top = df[df.Asset_Price >= entry_price].copy()
        df_top['Cash_Balan'] = cash_balance + ((df_top['Amount_Asset'].shift(1) - df_top['Amount_Asset']) * df_top['Asset_Price']).cumsum().fillna(0)
        df_top = df_top.sort_values(by='Amount_Asset').iloc[:-1]

        df_down = df[df.Asset_Price < entry_price].copy().sort_values(by='Asset_Price', ascending=False)
        df_down['Cash_Balan'] = cash_balance + ((df_down['Amount_Asset'].shift(-1) - df_down['Amount_Asset']) * df_down['Asset_Price']).cumsum().fillna(0)
        
        combined_df = pd.concat([df_top, df_down], axis=0)
        if combined_df.empty:
            return None
            
        final_cash_balance = combined_df['Cash_Balan'].iloc[-1]
        return abs(final_cash_balance - cash_balance)
        
    except Exception as e:
        st.warning(f"Could not calculate Production for {ticker}: {e}")
        return None

# เพิ่ม parameter filter_date เข้าไปในฟังก์ชัน
def monitor(channel_id, api_key, ticker, field, filter_date): # <--- เปลี่ยนแปลงตรงนี้
    """
    Monitors an asset and returns the last 7 days of data and the function value.
    ฟังก์ชันนี้จะรับ filter_date มาจากภายนอก
    """
    thingspeak_client = thingspeak.Channel(id=channel_id, api_key=api_key, fmt='json')
    history = get_ticker_history(ticker)
    
    # ใช้ filter_date ที่รับเข้ามาแทนการ hardcode
    # filter_date = '2025-04-28 12:00:00+07:00' # <--- ลบบรรทัดนี้
    filtered_data = history[history.index >= filter_date].copy()
    
    field_data = thingspeak_client.get_field_last(field=f'{field}')
    fx_js = int(json.loads(field_data)[f"field{field}"])
    
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
    
    cf_day = average_cf(CONFIG['average_cf_config'])
    st.write(f"average_cf_day: {cf_day:.2f} USD  :  average_cf_mo: {cf_day * 30:.2f} USD")
    st.write('____')
    
    # ดึงค่า filter_date สำหรับ monitor จาก config มาเก็บไว้ในตัวแปร
    monitor_filter_date = CONFIG['monitor_config']['filter_date'] # <--- อ่านค่าจาก config

    # Loop through each asset in the config and display its monitor
    for asset_config in CONFIG['assets']:
        ticker = asset_config['ticker']
        monitor_field = asset_config['monitor_field']
        prod_params = asset_config['production_params']
        channel_id = asset_config['channel_id']
        api_key = asset_config['write_api_key']

        # ส่ง filter_date ที่อ่านจาก config เข้าไปในฟังก์ชัน monitor
        df_7, fx_js = monitor(channel_id, api_key, ticker, monitor_field, monitor_filter_date) # <--- เปลี่ยนแปลงตรงนี้
        
        prod_cost = production_cost(
            ticker=ticker,
            fixed_asset_value=prod_params['fixed_asset_value'],
            cash_balance=prod_params['cash_balance']
        )
        
        prod_cost_display = f"{prod_cost:.2f}" if prod_cost is not None else "N/A"
        
        st.write(ticker)
        st.write(f"f(x): {fx_js} ,  , Production: {prod_cost_display}")
        st.table(df_7)
        st.write("_____")

    # Footer
    st.write("***ก่อนตลาดเปิดตรวจสอบ TB ล่าสุด > RE เมื่อตลอดเปิด")
    st.write("***RE > 60 USD")
    st.stop()

if __name__ == "__main__":
    main()
