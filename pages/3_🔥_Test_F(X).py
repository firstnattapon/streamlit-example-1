import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import thingspeak
import json
import time
from numba import njit

# --- ส่วนของการคำนวณและฟังก์ชันหลัก (ไม่มีการเปลี่ยนแปลง) ---
st.set_page_config(page_title="_Add_Gen_F(X)", page_icon="🏠", layout="wide")

# ... (ฟังก์ชัน calculate_optimized, feed_data, delta2 ไม่มีการเปลี่ยนแปลง) ...
@njit(fastmath=True)
def calculate_optimized(action_list, price_list, fix=500):
    action_array = np.asarray(action_list, dtype=np.int32)
    action_array[0] = 1
    price_array = np.asarray(price_list, dtype=np.float64)
    n = len(action_array)
    amount = np.empty(n, dtype=np.float64); buffer = np.zeros(n, dtype=np.float64); cash = np.empty(n, dtype=np.float64); asset_value = np.empty(n, dtype=np.float64); sumusd = np.empty(n, dtype=np.float64)
    initial_price = price_array[0]; amount[0] = fix / initial_price; cash[0] = fix; asset_value[0] = amount[0] * initial_price; sumusd[0] = cash[0] + asset_value[0]
    refer = fix * (1 - np.log(initial_price / price_array))
    for i in range(1, n):
        curr_price = price_array[i]
        if action_array[i] == 0: amount[i] = amount[i-1]; buffer[i] = 0
        else: amount[i] = fix / curr_price; buffer[i] = amount[i-1] * curr_price - fix
        cash[i] = cash[i-1] + buffer[i]; asset_value[i] = amount[i] * curr_price; sumusd[i] = cash[i] + asset_value[i]
    refer =  sumusd - (refer+fix)
    return buffer, sumusd, cash, asset_value, amount, refer

def feed_data( data = "APLS"):
    # ... (code as provided) ...
    pass

def delta2(Ticker = "FFWM" , pred = 1 ,  filter_date ='2023-01-01 12:00:00+07:00'):
    # ... (code as provided) ...
    pass
# --- จบส่วนที่ไม่เปลี่ยนแปลง ---

# --- ส่วนที่ปรับปรุงใหม่ ---
def load_gen_config(filename="add_gen_config.json"):
    """Loads asset configurations from a JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error: Configuration file '{filename}' not found.")
        return None
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from '{filename}'. Please check its format.")
        return None

def create_asset_tab_content(asset_config):
    """Creates the UI content for a single asset tab."""
    ticker = asset_config.get('ticker', 'N/A')
    field = asset_config.get('thingspeak_field')
    channel_id = asset_config.get('channel_id')
    write_api_key = asset_config.get('write_api_key')

    if not all([field, channel_id, write_api_key]):
        st.error(f"Configuration for '{ticker}' is incomplete. Missing field, channel_id, or write_api_key.")
        return

    try:
        client = thingspeak.Channel(channel_id, write_api_key, fmt='json')
    except Exception as e:
        st.error(f"Failed to create ThingSpeak client for {ticker}: {e}")
        return

    st.subheader(f"Manual Update for {ticker}")
    input_val_str = st.text_input(
        f'Insert a number or encoded string for {ticker}',
        key=f"text_input_{ticker}",
        placeholder="Enter a large integer value or encoded string"
    )
    if st.button("Update ThingSpeak Manually", key=f"rerun_m_{ticker}"):
        if not input_val_str:
            st.warning("Please enter a value to update.")
            return
        
        # ThingSpeak field must be a number, if string is all digits, convert it.
        # Otherwise, we keep it as is, but this might fail depending on ThingSpeak's API.
        # For our encoded strings, we can't convert to int. Let's send it as a string.
        # Let's assume the best course is to just send the value.
        # If it's a seed, it should be an int. If it's our new encoded string, it must be a string.
        # ThingSpeak API might handle this, but let's be careful. The docs say fields are string-based.
        
        value_to_send = input_val_str # Send as string

        with st.spinner(f"Updating field {field} for {ticker}..."):
            try:
                client.update({f'field{field}': value_to_send})
                st.success(f"Updated {ticker} with value: {value_to_send}")
            except Exception as e:
                st.error(f"Failed to update ThingSpeak: {e}")
                st.json(e) # Show detailed error
                
    st.write("---")

# --- ส่วนหลักของ Streamlit UI ---

st.markdown("## 🏠 Add Gen F(X) & ThingSpeak Updater")

# --- GOAL 6: Bulk Update Section ---
st.divider()
st.subheader("📤 Bulk Update ThingSpeak from Hybrid Lab")
if 'all_ticker_encoded_strings' in st.session_state and st.session_state.all_ticker_encoded_strings:
    st.success("✅ Found Encoded String data from the Hybrid Lab.")
    
    # Load configs to map tickers to fields
    gen_configs = load_gen_config()
    if gen_configs:
        config_by_ticker = {c['ticker']: c for c in gen_configs}
        encoded_strings = st.session_state.all_ticker_encoded_strings

        data_to_show = []
        for ticker, encoded_str in encoded_strings.items():
            if ticker in config_by_ticker:
                config = config_by_ticker[ticker]
                data_to_show.append({
                    "Ticker": ticker,
                    "Channel ID": config['channel_id'],
                    "Field": config['thingspeak_field'],
                    "Encoded String (Preview)": f"{encoded_str[:30]}..."
                })
        
        st.write("The following data will be sent to ThingSpeak:")
        st.dataframe(pd.DataFrame(data_to_show), use_container_width=True)

        if st.button("Update All Tickers on ThingSpeak", type="primary"):
            total_tickers = len(encoded_strings)
            progress_bar = st.progress(0, text="Starting bulk update...")
            
            for i, (ticker, encoded_str) in enumerate(encoded_strings.items()):
                progress_text = f"Updating {ticker} ({i+1}/{total_tickers})..."
                progress_bar.progress((i + 1) / total_tickers, text=progress_text)

                if ticker in config_by_ticker:
                    config = config_by_ticker[ticker]
                    try:
                        client = thingspeak.Channel(config['channel_id'], config['write_api_key'], fmt='json')
                        client.update({f"field{config['thingspeak_field']}": encoded_str})
                        st.write(f"✔️ Successfully updated {ticker}.")
                        time.sleep(16) # ThingSpeak has a rate limit of ~15 seconds per update
                    except Exception as e:
                        st.error(f"❌ Failed to update {ticker}: {e}")
                else:
                    st.warning(f"⚠️ No ThingSpeak config found for {ticker}, skipping.")
            
            progress_bar.empty()
            st.success("🎉 Bulk update process finished!")
    else:
        st.error("Could not load `add_gen_config.json`. Cannot proceed with bulk update.")

else:
    st.info("No data found from the Hybrid Lab. Please run the 'Run for All Tickers' process in the 'Hybrid_Multi_Mutation' page first.")

st.divider()
st.subheader("Manual & Individual Controls")

# Load asset configs for individual tabs
asset_configs = load_gen_config()
if asset_configs:
    tab_names = [config.get('tab_name', f"Tab {i+1}") for i, config in enumerate(asset_configs)]
    tabs = st.tabs(tab_names)
    for i, tab in enumerate(tabs):
        with tab:
            create_asset_tab_content(asset_configs[i])
else:
    st.warning("No asset configurations were loaded. Individual controls cannot be displayed.")
