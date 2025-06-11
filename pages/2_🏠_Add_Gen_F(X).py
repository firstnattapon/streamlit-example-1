import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import thingspeak
import json
import time
from numba import njit

# --- ส่วนของการคำนวณและฟังก์ชันหลัก (ไม่มีการเปลี่ยนแปลง) ---

st.set_page_config(page_title="_Add_Gen_F(X)", page_icon="🏠")

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

def feed_data( data = "APLS"):
    Ticker = data
    filter_date = '2023-01-01 12:00:00+07:00'
    tickerData = yf.Ticker(Ticker)
    tickerData = tickerData.history(period= 'max' )[['Close']]
    tickerData.index = tickerData.index.tz_convert(tz='Asia/bangkok')
    filter_date = filter_date
    tickerData = tickerData[tickerData.index >= filter_date]
    
    prices = np.array( tickerData.Close.values , dtype=np.float64)
    actions = np.array( np.ones( len(prices) ) , dtype=np.int64)
    initial_cash = 500.0
    initial_asset_value = 500.0
    initial_price = prices[0]
    
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

def delta2(Ticker = "FFWM" , pred = 1 ,  filter_date ='2023-01-01 12:00:00+07:00'):
    # ... โค้ดของฟังก์ชัน delta2 เดิม (ยาวมาก จึงย่อไว้) ...
    # ... ไม่มีการเปลี่ยนแปลงในฟังก์ชันนี้ ...
    try:
        tickerData = yf.Ticker(Ticker)
        tickerData = tickerData.history(period= 'max' )[['Close']]
        tickerData.index = tickerData.index.tz_convert(tz='Asia/bangkok')
        filter_date = filter_date
        tickerData = tickerData[tickerData.index >= filter_date]
        entry  = tickerData.Close[0] ; step = 0.01 ;  Fixed_Asset_Value = 1500. ; Cash_Balan = 650.
        if entry < 10000 :
            samples = np.arange( 0  ,  np.around(entry, 2) * 3 + step  ,  step)
            df = pd.DataFrame()
            df['Asset_Price'] =   np.around(samples, 2)
            df['Fixed_Asset_Value'] = Fixed_Asset_Value
            df['Amount_Asset']  =   df['Fixed_Asset_Value']  / df['Asset_Price']
            df_top = df[df.Asset_Price >= np.around(entry, 2) ]
            df_top['Cash_Balan_top'] = (df_top['Amount_Asset'].shift(1) -  df_top['Amount_Asset']) *  df_top['Asset_Price']
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


def Gen_fx (Ticker =  'FFWM' ,  field = 2 ):
    # ... โค้ดของฟังก์ชัน Gen_fx เดิม ...
    # ... ไม่มีการเปลี่ยนแปลงในฟังก์ชันนี้ ...
    container = st.container(border=True)
    fx = [0]
    progress_text = "Processing iterations. Please wait."
    my_bar = st.progress(0, text=progress_text)
    for j in range(1):
        pred  = delta2(Ticker=Ticker)
        siz = len(pred)
        z = int( pred.net_pv.values[-1])
        container.write("x , {}".format(z))
        for i in range(2000):
            rng = np.random.default_rng(i)
            pred  = delta2(Ticker= Ticker , pred= rng.integers(2, size= siz) )
            y = int( pred.net_pv.values[-1])
            if  y > z :
                container.write("{} , {}".format(i,y))
                z = y
                fx.append(i)
            percent_complete = (i + 1) / 2000 * 100
            my_bar.progress(int(percent_complete), text=progress_text)
    time.sleep(1)
    my_bar.empty()
    client.update(  {'field{}'.format(field) : fx[-1] } )


# --- ส่วนของการเชื่อมต่อ ThingSpeak (ไม่มีการเปลี่ยนแปลง) ---
channel_id = 2385118
write_api_key = 'IPSG3MMMBJEB9DY8'
client = thingspeak.Channel(channel_id, write_api_key , fmt='json')


# --- ส่วนที่ปรับปรุงใหม่ ---

# 1. ฟังก์ชันสำหรับโหลดการตั้งค่าจาก JSON
def load_config(filename="add_gen_config.json"):
    """Loads asset configurations from a JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error: Configuration file '{filename}' not found.")
        return []
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from '{filename}'. Please check its format.")
        return []

# 2. ฟังก์ชันสำหรับสร้าง UI ภายในแต่ละแท็บ (เพื่อลดการเขียนโค้ดซ้ำ)
def create_asset_tab_content(asset_config):
    """Creates the UI content for a single asset tab."""
    ticker = asset_config['ticker']
    field = asset_config['thingspeak_field']

    # --- ส่วน Gen_fx (Commented out like original) ---
    # gen_fx_check = st.checkbox(f'{ticker}_Add_Gen', key=f"gen_fx_{ticker}")
    # if gen_fx_check:
    #     if st.button("Rerun_Gen", key=f"rerun_gen_{ticker}"):
    #         Gen_fx(Ticker=ticker, field=field)

    # --- ส่วน Manual Add Gen ---
    gen_m_check = st.checkbox(f'{ticker}_Add_Gen_M', key=f"gen_m_{ticker}")
    if gen_m_check:
        input_val = st.number_input(f'Insert a number for {ticker}', step=1, key=f"num_input_{ticker}")
        if st.button("Rerun_Gen_M", key=f"rerun_m_{ticker}"):
            with st.spinner(f"Updating field {field} for {ticker}..."):
                client.update({f'field{field}': input_val})
                st.success(f"Updated {ticker} with value: {input_val}")

    # --- ส่วน Njit ---
    njit_check = st.checkbox(f'{ticker}_njit', key=f"njit_{ticker}")
    if njit_check:
        if st.button(f"Run Njit for {ticker}", key=f"run_njit_{ticker}"):
            with st.spinner(f"Running NJIT optimization for {ticker}... This may take a long time."):
                ix = feed_data(data=ticker)
                client.update({f'field{field}': ix})
                st.success(f"Found optimal seed for {ticker}: {ix}")

    st.write("_____")

# --- ส่วนหลักของ Streamlit UI ---

# โหลดการตั้งค่า
asset_configs = load_config()

if asset_configs:
    # สร้างชื่อแท็บจากข้อมูลที่โหลดมา
    tab_names = [config['tab_name'] for config in asset_configs]
    
    # สร้างแท็บ
    tabs = st.tabs(tab_names)
    
    # วนลูปเพื่อสร้างเนื้อหาในแต่ละแท็บ
    for i, tab in enumerate(tabs):
        with tab:
            # เรียกใช้ฟังก์ชันเพื่อสร้าง UI โดยส่งค่า config ของ Asset นั้นๆ เข้าไป
            create_asset_tab_content(asset_configs[i])
