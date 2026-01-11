import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import thingspeak
import json
import time
from numba import njit

# --- ส่วนของการคำนวณและฟังก์ชันหลัก (Original Logic - No Changes) ---
st.set_page_config(page_title="_Add_Gen_F(X)", page_icon="🏠", layout="wide")

@njit(fastmath=True)
def calculate_optimized(action_list, price_list, fix=500):
    action_array = np.asarray(action_list, dtype=np.int32)
    action_array[0] = 1
    price_array = np.asarray(price_list, dtype=np.float64)
    n = len(action_array)
    
    amount = np.empty(n, dtype=np.float64)
    buffer = np.zeros(n, dtype=np.float64)
    cash = np.empty(n, dtype=np.float64)
    asset_value = np.empty(n, dtype=np.float64)
    sumusd = np.empty(n, dtype=np.float64)
    
    initial_price = price_array[0]
    amount[0] = fix / initial_price
    cash[0] = fix
    asset_value[0] = amount[0] * initial_price
    sumusd[0] = cash[0] + asset_value[0]
    
    refer = fix * (1 - np.log(initial_price / price_array))
    
    for i in range(1, n):
        curr_price = price_array[i]
        if action_array[i] == 0:
            amount[i] = amount[i-1]
            buffer[i] = 0
        else:
            amount[i] = fix / curr_price
            buffer[i] = amount[i-1] * curr_price - fix
        
        cash[i] = cash[i-1] + buffer[i]
        asset_value[i] = amount[i] * curr_price
        sumusd[i] = cash[i] + asset_value[i]

    refer =  sumusd - (refer+fix)
    return buffer, sumusd, cash, asset_value, amount, refer

def feed_data(data="APLS"):
    Ticker = data
    filter_date = '2023-01-01 12:00:00+07:00'
    tickerData = yf.Ticker(Ticker)
    tickerData = tickerData.history(period='max')[['Close']]
    tickerData.index = tickerData.index.tz_convert(tz='Asia/bangkok')
    tickerData = tickerData[tickerData.index >= filter_date]
    
    prices = np.array(tickerData.Close.values, dtype=np.float64)
    actions = np.array(np.ones(len(prices)), dtype=np.int64)
    
    net_initial = 0.
    seed = 0
    # หมายเหตุ: loop 2,000,000 ครั้งอาจใช้เวลานานมาก
    for i in range(2000000): 
        rng = np.random.default_rng(i)
        actions = rng.integers(0, 2, len(prices))
        _, _, _, _, _ , net_cf = calculate_optimized(actions, prices)
        if net_cf[-1] > net_initial:
            net_initial = net_cf[-1]
            seed  = i 
    return seed

def delta2(Ticker="FFWM", pred=1, filter_date='2023-01-01 12:00:00+07:00'):
    # ... โค้ดของฟังก์ชัน delta2 เดิม (Original Logic) ...
    try:
        tickerData = yf.Ticker(Ticker)
        tickerData = tickerData.history(period='max')[['Close']]
        tickerData.index = tickerData.index.tz_convert(tz='Asia/bangkok')
        tickerData = tickerData[tickerData.index >= filter_date]
        entry  = tickerData.Close[0] ; step = 0.01 ;  Fixed_Asset_Value = 1500. ; Cash_Balan = 650.
        if entry < 10000 :
            samples = np.arange( 0  ,  np.around(entry, 2) * 3 + step  ,  step)
            df = pd.DataFrame()
            df['Asset_Price'] =   np.around(samples, 2)
            df['Fixed_Asset_Value'] = Fixed_Asset_Value
            df['Amount_Asset']  =   df['Fixed_Asset_Value']  / df['Asset_Price']
            df_top = df[df.Asset_Price >= np.around(entry, 2) ]
            df_top['Cash_Balan_top'] = (df_top['Amount_Asset'].shift(1) -  df_top['Amount_Asset']) * df_top['Asset_Price']
            df_top.fillna(0, inplace=True)
            np_Cash_Balan_top = df_top['Cash_Balan_top'].values
            xx = np.zeros(len(np_Cash_Balan_top)) ; y_0 = Cash_Balan
            for idx, v_0  in enumerate(np_Cash_Balan_top) :
                z_0 = y_0 + v_0; y_0 = z_0; xx[idx] = y_0
            df_top['Cash_Balan_top'] = xx
            df_top = df_top.rename(columns={'Cash_Balan_top': 'Cash_Balan'}); df_top  = df_top.sort_values(by='Amount_Asset'); df_top  = df_top[:-1]
            df_down = df[df.Asset_Price <= np.around(entry, 2) ]
            df_down['Cash_Balan_down'] = (df_down['Amount_Asset'].shift(-1) -  df_down['Amount_Asset']) * df_down['Asset_Price']
            df_down.fillna(0, inplace=True)
            df_down = df_down.sort_values(by='Asset_Price' , ascending=False)
            np_Cash_Balan_down = df_down['Cash_Balan_down'].values
            xxx= np.zeros(len(np_Cash_Balan_down)) ; y_1 = Cash_Balan
            for idx, v_1  in enumerate(np_Cash_Balan_down) :
                z_1 = y_1 + v_1; y_1 = z_1; xxx[idx] = y_1
            df_down['Cash_Balan_down'] = xxx
            df_down = df_down.rename(columns={'Cash_Balan_down': 'Cash_Balan'})
            df = pd.concat([df_top, df_down], axis=0)
            Production_Costs = (df['Cash_Balan'].values[-1]) -  Cash_Balan
            tickerData['Close'] = np.around(tickerData['Close'].values , 2)
            tickerData['pred'] = pred
            tickerData['Fixed_Asset_Value'] = Fixed_Asset_Value
            tickerData['Amount_Asset']  =  0.
            tickerData['Amount_Asset'][0]  =  tickerData['Fixed_Asset_Value'][0] / tickerData['Close'][0]
            tickerData['re']  =  0.
            tickerData['Cash_Balan'] = Cash_Balan
            Close =   tickerData['Close'].values; pred =  tickerData['pred'].values; Amount_Asset =  tickerData['Amount_Asset'].values; re = tickerData['re'].values; Cash_Balan = tickerData['Cash_Balan'].values
            for idx, x_0 in enumerate(Amount_Asset):
                if idx != 0:
                    if pred[idx] == 0: Amount_Asset[idx] = Amount_Asset[idx-1]
                    elif  pred[idx] == 1: Amount_Asset[idx] =   Fixed_Asset_Value / Close[idx]
            tickerData['Amount_Asset'] = Amount_Asset
            for idx, x_1 in enumerate(re):
                if idx != 0:
                    if pred[idx] == 0: re[idx] =  0
                    elif  pred[idx] == 1: re[idx] =  (Amount_Asset[idx-1] * Close[idx])  - Fixed_Asset_Value
            tickerData['re'] = re
            for idx, x_2 in enumerate(Cash_Balan):
                if idx != 0: Cash_Balan[idx] = Cash_Balan[idx-1] + re[idx]
            tickerData['Cash_Balan'] = Cash_Balan
            tickerData ['refer_model'] = 0.
            price = np.around(tickerData['Close'].values, 2); Cash  = tickerData['Cash_Balan'].values; refer_model =  tickerData['refer_model'].values
            for idx, x_3 in enumerate(price):
                try: refer_model[idx] = (df[df['Asset_Price'] == x_3]['Cash_Balan'].values[0])
                except: refer_model[idx] = np.nan
            tickerData['Production_Costs'] = abs(Production_Costs); tickerData['refer_model'] = refer_model; tickerData['pv'] =  tickerData['Cash_Balan'] + ( tickerData['Amount_Asset'] * tickerData['Close']  ); tickerData['refer_pv'] = tickerData['refer_model'] + Fixed_Asset_Value; tickerData['net_pv'] =   tickerData['pv'] - tickerData['refer_pv']  
            final = tickerData[['net_pv']]
            return  final
    except:pass

# --- ส่วนจัดการ Config และ Helper Functions ---

def load_config(filename="add_gen_config.json"):
    """Loads asset configurations from a JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error: Configuration file '{filename}' not found.")
        return []
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from '{filename}'.")
        return []

def get_config_by_ticker(configs, ticker_name):
    """Helper to find config dict for a specific ticker."""
    for config in configs:
        if config.get('ticker') == ticker_name:
            return config
    return None

# --- ส่วน UI และ Logic การทำงานหลัก ---

# โหลดการตั้งค่า Global
asset_configs = load_config()

# === SIDEBAR: One-Click Bulk Import System ===
with st.sidebar:
    st.header("📂 Bulk Data Import")
    st.write("Upload JSON to update all tickers.")
    
    uploaded_file = st.file_uploader("Choose a JSON file", type=['json'])
    
    if uploaded_file is not None:
        try:
            # Load JSON Data
            import_data = json.load(uploaded_file)
            
            # Show metadata info (Optional)
            if "metadata" in import_data:
                st.info(f"Exported at: {import_data['metadata'].get('exported_at', 'N/A')}")
            
            if st.button("🚀 Process & Update All", type="primary"):
                tickers_map = import_data.get("tickers", {})
                
                if not tickers_map:
                    st.warning("No tickers found in the file.")
                else:
                    st.write("---")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    total_items = len(tickers_map)
                    completed = 0
                    
                    # Create an Expander to show detailed logs
                    with st.status("Processing Tickers...", expanded=True) as status:
                        
                        for ticker, value_str in tickers_map.items():
                            status_text.text(f"Processing: {ticker}")
                            
                            # 1. Find Config
                            config = get_config_by_ticker(asset_configs, ticker)
                            
                            if config:
                                try:
                                    # 2. Prepare Data
                                    # Convert string to huge integer
                                    val_int = int(value_str) 
                                    
                                    channel_id = config.get('channel_id')
                                    write_key = config.get('write_api_key')
                                    field = config.get('thingspeak_field')
                                    
                                    # 3. Update ThingSpeak
                                    if channel_id and write_key and field:
                                        client = thingspeak.Channel(channel_id, write_key, fmt='json')
                                        
                                        # Update logic
                                        client.update({f'field{field}': val_int})
                                        
                                        status.write(f"✅ **{ticker}**: Updated field{field} with value ending in ...{value_str[-6:]}")
                                    else:
                                        status.write(f"⚠️ **{ticker}**: Missing config (Channel/Key/Field).")
                                        
                                except Exception as e:
                                    status.write(f"❌ **{ticker}**: Error - {e}")
                            else:
                                status.write(f"⚠️ **{ticker}**: No configuration found in 'add_gen_config.json'.")
                            
                            # Update Progress
                            completed += 1
                            progress_bar.progress(completed / total_items)
                            
                            # Slight delay to be gentle on API (optional, can be removed if keys allow high rate)
                            time.sleep(0.5) 
                        
                        status.update(label="Bulk Update Complete!", state="complete", expanded=False)
                    
                    st.success("All operations finished.")
                    
        except json.JSONDecodeError:
            st.error("Invalid JSON file.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

st.write("___")

# === MAIN CONTENT: Tabs for Individual Assets (Existing UI) ===

def Gen_fx(Ticker, field, client):
    """Runs the Gen_fx process and updates ThingSpeak."""
    container = st.container(border=True)
    fx = [0]
    progress_text = f"Processing {Ticker} iterations. Please wait."
    my_bar = st.progress(0, text=progress_text)
    
    # Initialize z
    pred_init = delta2(Ticker=Ticker)
    if pred_init is not None and not pred_init.empty:
        z = int(pred_init.net_pv.values[-1])
        container.write(f"Initial Value (x=0), Result: {z}")
    else:
        st.error(f"Could not get initial data for {Ticker}. Aborting Gen_fx.")
        my_bar.empty()
        return

    for i in range(1, 2000):
        rng = np.random.default_rng(i)
        siz = len(pred_init)
        pred_run = delta2(Ticker=Ticker, pred=rng.integers(2, size=siz))
        
        if pred_run is not None and not pred_run.empty:
            y = int(pred_run.net_pv.values[-1])
            if y > z:
                container.write(f"New Best Found! Seed: {i}, Result: {y}")
                z = y
                fx.append(i)
        
        percent_complete = (i + 1) / 2000
        my_bar.progress(percent_complete, text=progress_text)

    time.sleep(1)
    my_bar.empty()
    
    best_seed = fx[-1]
    st.write(f"Finished. Best seed found for {Ticker} is: {best_seed}")
    
    with st.spinner(f"Updating ThingSpeak field {field} for {Ticker}..."):
        try:
            client.update({f'field{field}': best_seed})
            st.success(f"Successfully updated ThingSpeak for {Ticker} with seed: {best_seed}")
        except Exception as e:
            st.error(f"Failed to update ThingSpeak: {e}")

def create_asset_tab_content(asset_config):
    """Creates the UI content for a single asset tab."""
    ticker = asset_config.get('ticker', 'N/A')
    field = asset_config.get('thingspeak_field')
    channel_id = asset_config.get('channel_id')
    write_api_key = asset_config.get('write_api_key')

    if not all([field, channel_id, write_api_key]):
        st.error(f"Configuration for '{ticker}' is incomplete.")
        return

    try:
        client = thingspeak.Channel(channel_id, write_api_key, fmt='json')
    except Exception as e:
        st.error(f"Failed to create ThingSpeak client for {ticker}: {e}")
        return

    # --- ส่วน Manual Add Gen (Tab Specific) ---
    st.subheader(f"Manage: {ticker}")
    gen_m_check = st.checkbox(f'Enable Manual Input for {ticker}', key=f"gen_m_{ticker}")
    
    if gen_m_check:
        input_val_str = st.text_input(
            f'Insert a seed/value for {ticker}',
            key=f"text_input_{ticker}",
            placeholder="Enter a large integer value"
        )
        if st.button("Update Manually", key=f"rerun_m_{ticker}"):
            try:
                input_val = int(input_val_str)
                with st.spinner(f"Updating field {field} for {ticker}..."):
                    try:
                        client.update({f'field{field}': input_val})
                        st.success(f"Updated {ticker} with value: {input_val}")
                    except Exception as e:
                        st.error(f"Failed to update ThingSpeak: {e}")
            except ValueError:
                st.error(f"Invalid input: '{input_val_str}'. Please enter a valid integer.")
    
    # --- ส่วน Gen_fx (Optional - Uncomment if needed) ---
    # if st.button(f"Run Optimizer Loop for {ticker}", key=f"run_gen_{ticker}"):
    #     Gen_fx(Ticker=ticker, field=field, client=client)

# Main UI Logic
if asset_configs:
    tab_names = [config.get('tab_name', f"Asset {i+1}") for i, config in enumerate(asset_configs)]
    tabs = st.tabs(tab_names)
    
    for i, tab in enumerate(tabs):
        with tab:
            create_asset_tab_content(asset_configs[i])
else:
    st.warning("No asset configurations were loaded. Please check 'add_gen_config.json'.")
