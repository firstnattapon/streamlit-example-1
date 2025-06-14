import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import thingspeak
import json
from pathlib import Path

# --- 1. การตั้งค่าและโหลด Configuration ---
st.set_page_config(page_title="Calculator", page_icon="⌨️")

@st.cache_data(ttl=300)
def load_config(filepath="calculator_config.json"):
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

CONFIG = load_config()

if st.button("Rerun"):
    st.rerun()

# --- 2. ฟังก์ชันที่ปรับปรุงแล้ว (Refactored Functions) ---

@st.cache_data(ttl=600)
def get_ticker_history(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    history = ticker.history(period='max')[['Close']]
    history.index = history.index.tz_convert(tz='Asia/Bangkok')
    return round(history, 3)

def average_cf(cf_config):
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
def production_cost(ticker, fixed_asset_value, cash_balance):
    """
    Calculates Production Costs.
    **UPDATED**: Returns None if fixed_asset_value is 0 to avoid misleading results.
    """
    # --- เพิ่มการตรวจสอบตรงนี้ ---
    if fixed_asset_value <= 0:
        st.warning(f"Production cost for {ticker} skipped: 'fixed_asset_value' is zero or missing in config.")
        return None

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

def monitor(channel_id, api_key, ticker, field, filter_date):
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

        if not all([ticker != 'N/A', monitor_field, channel_id, api_key]):
            st.warning(f"Skipping an asset due to missing configuration: {asset_config}")
            continue

        df_7, fx_js = monitor(channel_id, api_key, ticker, monitor_field, monitor_filter_date)
        
        prod_cost = production_cost(
            ticker=ticker,
            fixed_asset_value=prod_params.get('fixed_asset_value', 0),
            cash_balance=prod_params.get('cash_balance', 0)
        )
        
        prod_cost_display = f"{prod_cost:.2f}" if prod_cost is not None else "N/A"
        
        st.write(ticker)
        st.write(f"f(x): {fx_js} ,  , Production: {prod_cost_display}")
        st.table(df_7)
        st.write("_____")

    st.write("***ก่อนตลาดเปิดตรวจสอบ TB ล่าสุด > RE เมื่อตลอดเปิด")
    st.write("***RE > 60 USD")
    st.stop()

if __name__ == "__main__":
    main()
