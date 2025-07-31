import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import json
import plotly.express as px

# ------------------- ส่วนที่ปรับปรุงใหม่ -------------------
 
# 1. ฟังก์ชันสำหรับโหลด Config จากไฟล์ JSON (ปรับปรุงให้อ่านค่า Default)
def load_config(filename="un15_fx_config.json"):
    """
    Loads configurations from a JSON file.
    It expects a special key '__DEFAULT_CONFIG__' for default values.
    Returns a tuple: (ticker_configs, default_config)
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        st.error(f"Error: Configuration file '{filename}' not found.")
        return {}, {} # คืนค่า dict ว่าง
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from '{filename}'. Please check its format.")
        return {}, {} # คืนค่า dict ว่าง

    # กำหนดค่า fallback เผื่อไม่มี '__DEFAULT_CONFIG__' ในไฟล์ JSON
    fallback_default = {
        "Fixed_Asset_Value": 1500.0, "Cash_Balan": 650.0, "step": 0.01,
        "filter_date": "2024-01-01 12:00:00+07:00", "pred": 1
    }

    # ดึงค่า default config ออกมา, ถ้าไม่เจอก็ใช้ fallback
    default_config = data.pop('__DEFAULT_CONFIG__', fallback_default)
    
    # ข้อมูลที่เหลือคือ config ของ Ticker แต่ละตัว
    ticker_configs = data
    
    return ticker_configs, default_config

# 2. ฟังก์ชันกลางที่รวม Logic ที่ซ้ำซ้อนกัน (เหมือนเดิม)
def calculate_cash_balance_model(entry, step, Fixed_Asset_Value, Cash_Balan):
    """Calculates the core cash balance model DataFrame. This logic was duplicated in the original code."""
    if entry >= 10000:
        return pd.DataFrame()

    samples = np.arange(0, np.around(entry, 2) * 3 + step, step)
    
    df = pd.DataFrame()
    df['Asset_Price'] = np.around(samples, 2)
    df['Fixed_Asset_Value'] = Fixed_Asset_Value
    df['Amount_Asset'] = df['Fixed_Asset_Value'] / df['Asset_Price']

    # --- Top part calculation ---
    df_top = df[df.Asset_Price >= np.around(entry, 2)].copy()
    df_top['Cash_Balan_top'] = (df_top['Amount_Asset'].shift(1) - df_top['Amount_Asset']) * df_top['Asset_Price']
    df_top.fillna(0, inplace=True)
    
    np_Cash_Balan_top = df_top['Cash_Balan_top'].values
    xx = np.zeros(len(np_Cash_Balan_top))
    y_0 = Cash_Balan
    for idx, v_0 in enumerate(np_Cash_Balan_top):
        z_0 = y_0 + v_0
        y_0 = z_0
        xx[idx] = y_0
        
    df_top['Cash_Balan_top'] = xx
    df_top.rename(columns={'Cash_Balan_top': 'Cash_Balan'}, inplace=True)
    df_top = df_top.sort_values(by='Amount_Asset')[:-1]

    # --- Down part calculation ---
    df_down = df[df.Asset_Price <= np.around(entry, 2)].copy()
    df_down['Cash_Balan_down'] = (df_down['Amount_Asset'].shift(-1) - df_down['Amount_Asset']) * df_down['Asset_Price']
    df_down.fillna(0, inplace=True)
    df_down = df_down.sort_values(by='Asset_Price', ascending=False)
    
    np_Cash_Balan_down = df_down['Cash_Balan_down'].values
    xxx = np.zeros(len(np_Cash_Balan_down))
    y_1 = Cash_Balan
    for idx, v_1 in enumerate(np_Cash_Balan_down):
        z_1 = y_1 + v_1
        y_1 = z_1
        xxx[idx] = y_1

    df_down['Cash_Balan_down'] = xxx
    df_down.rename(columns={'Cash_Balan_down': 'Cash_Balan'}, inplace=True)

    # --- Combine and return ---
    combined_df = pd.concat([df_top, df_down], axis=0)
    return combined_df

# ------------------- ฟังก์ชันหลัก (เหมือนเดิม) -------------------

def delta_1(asset_config):
    """Calculates Production_Costs based on asset configuration."""
    try:
        Ticker = asset_config['Ticker']
        Fixed_Asset_Value = asset_config['Fixed_Asset_Value']
        Cash_Balan = asset_config['Cash_Balan']
        step = asset_config['step']
        
        tickerData = yf.Ticker(Ticker)
        entry = tickerData.fast_info['lastPrice']
        
        df_model = calculate_cash_balance_model(entry, step, Fixed_Asset_Value, Cash_Balan)

        if not df_model.empty:
            Production_Costs = (df_model['Cash_Balan'].values[-1]) - Cash_Balan
            return abs(Production_Costs)
    except Exception as e:
        return None

def delta6(asset_config):
    """Performs historical simulation based on asset configuration."""
    try:
        Ticker = asset_config['Ticker']
        pred = asset_config['pred']
        filter_date = asset_config['filter_date']
        Fixed_Asset_Value = asset_config['Fixed_Asset_Value']
        Cash_Balan = asset_config['Cash_Balan']
        step = asset_config['step']

        ticker_hist = yf.Ticker(Ticker).history(period='max')
        ticker_hist.index = ticker_hist.index.tz_convert(tz='Asia/bangkok')
        ticker_hist = ticker_hist[ticker_hist.index >= filter_date][['Close']]
        
        entry = ticker_hist.Close[0]
        
        df_model = calculate_cash_balance_model(entry, step, Fixed_Asset_Value, Cash_Balan)
        
        if df_model.empty:
            return None

        tickerData = ticker_hist.copy()
        tickerData['Close'] = np.around(tickerData['Close'].values, 2)
        tickerData['pred'] = pred
        tickerData['Fixed_Asset_Value'] = Fixed_Asset_Value
        tickerData['Amount_Asset'] = 0.
        tickerData['Amount_Asset'][0] = tickerData['Fixed_Asset_Value'][0] / tickerData['Close'][0]
        tickerData['re'] = 0.
        tickerData['Cash_Balan'] = Cash_Balan

        Close = tickerData['Close'].values
        pred_vals = tickerData['pred'].values
        Amount_Asset = tickerData['Amount_Asset'].values
        re = tickerData['re'].values
        Cash_Balan_sim = tickerData['Cash_Balan'].values

        for idx in range(1, len(Amount_Asset)):
            if pred_vals[idx] == 1:
                Amount_Asset[idx] = Fixed_Asset_Value / Close[idx]
                re[idx] = (Amount_Asset[idx-1] * Close[idx]) - Fixed_Asset_Value
            else: 
                Amount_Asset[idx] = Amount_Asset[idx-1]
                re[idx] = 0
            Cash_Balan_sim[idx] = Cash_Balan_sim[idx-1] + re[idx]

        tickerData['Amount_Asset'] = Amount_Asset
        tickerData['re'] = re
        tickerData['Cash_Balan'] = Cash_Balan_sim
        
        tickerData['refer_model'] = 0.
        price = np.around(tickerData['Close'].values, 2)
        refer_model = tickerData['refer_model'].values

        for idx, x_3 in enumerate(price):
            try:
                refer_model[idx] = (df_model[df_model['Asset_Price'] == x_3]['Cash_Balan'].values[0])
            except IndexError:
                refer_model[idx] = np.nan
        
        tickerData['refer_model'].interpolate(method='linear', inplace=True)
        tickerData['refer_model'].fillna(method='bfill', inplace=True)
        tickerData['refer_model'].fillna(method='ffill', inplace=True)

        tickerData['pv'] = tickerData['Cash_Balan'] + (tickerData['Amount_Asset'] * tickerData['Close'])
        tickerData['refer_pv'] = tickerData['refer_model'] + Fixed_Asset_Value
        tickerData['net_pv'] = tickerData['pv'] - tickerData['refer_pv']
        
        final = tickerData[['net_pv', 'pred', 're', 'Cash_Balan', 'Close']]
        return final
        
    except Exception as e:
        return None

def un_16(active_configs):
    """Aggregates results from multiple assets specified in active_configs."""
    a_0 = pd.DataFrame()
    a_1 = pd.DataFrame()
    Max_Production = 0
    
    for ticker_name, config in active_configs.items():
        a_2 = delta6(config)
        if a_2 is not None:
            a_0 = pd.concat([a_0, a_2[['re']].rename(columns={"re": f"{ticker_name}_re"})], axis=1)
            a_1 = pd.concat([a_1, a_2[['net_pv']].rename(columns={"net_pv": f"{ticker_name}_net_pv"})], axis=1)
        
        prod_cost = delta_1(config)
        if prod_cost is not None:
            Max_Production += prod_cost
    
    if a_0.empty:
        return pd.DataFrame()
        
    net_dd = []
    net = 0
    for i in a_0.sum(axis=1, numeric_only=True).values:
        net = net + i
        net_dd.append(net)

    a_0['maxcash_dd'] = net_dd
    a_1['cf'] = a_1.sum(axis=1, numeric_only=True)
    a_x = pd.concat([a_0, a_1], axis=1)

    return a_x

# ------------------- ส่วนแสดงผล STREAMLIT (ปรับปรุงใหม่) -------------------
st.set_page_config(page_title="Exist_F(X)", page_icon="☀")

# 1. โหลด config ทั้งหมดจากไฟล์ (ตอนนี้จะคืนค่า 2 ตัว)
full_config, DEFAULT_CONFIG = load_config()

# ----> จุดนี้ ไม่ต้องมี DEFAULT_CONFIG แบบ Hardcode อีกต่อไป <----

if full_config or DEFAULT_CONFIG: # ตรวจสอบว่ามีข้อมูล config หรือ default config อย่างน้อยหนึ่งอย่าง
    # 2. ใช้ Session State เพื่อเก็บ Ticker ที่เพิ่มจาก UI
    if 'custom_tickers' not in st.session_state:
        st.session_state.custom_tickers = {}

    # 3. UI สำหรับเพิ่ม Ticker ใหม่
    st.subheader("เพิ่ม Ticker ใหม่")
    new_ticker = st.text_input("พิมพ์ Ticker ที่ต้องการเพิ่ม (เช่น AAPL):").upper()
    if st.button("เพิ่ม Ticker"):
        if new_ticker and new_ticker not in full_config and new_ticker not in st.session_state.custom_tickers:
            # สร้าง config สำหรับ Ticker ใหม่โดยใช้ default ที่อ่านมาจาก JSON
            st.session_state.custom_tickers[new_ticker] = {
                "Ticker": new_ticker,
                **DEFAULT_CONFIG
            }
            st.success(f"เพิ่ม {new_ticker} สำเร็จ! (ใช้ค่า default จากไฟล์ config)")
        elif new_ticker in full_config:
            st.warning(f"{new_ticker} มีอยู่ใน config จาก JSON แล้ว")
        elif new_ticker in st.session_state.custom_tickers:
            st.warning(f"{new_ticker} ถูกเพิ่มแล้ว")
        else:
            st.warning("กรุณาพิมพ์ชื่อ Ticker")

    # 4. รวม Ticker จาก JSON และจาก UI
    all_tickers = list(full_config.keys()) + list(st.session_state.custom_tickers.keys())

    # 5. สร้าง UI ให้ผู้ใช้เลือก Ticker จากทั้งหมด
    selected_tickers = st.multiselect(
        "Select Tickers to Analyze",
        options=all_tickers,
        default=list(full_config.keys())  # เลือก Ticker จากไฟล์ JSON เป็นค่าเริ่มต้น
    )

    # 6. สร้าง dict config เฉพาะ Ticker ที่ถูกเลือก (รวมจากทั้ง JSON และ custom)
    active_configs = {}
    for ticker in selected_tickers:
        if ticker in full_config:
            active_configs[ticker] = full_config[ticker]
        elif ticker in st.session_state.custom_tickers:
            active_configs[ticker] = st.session_state.custom_tickers[ticker]

    # 7. ตรวจสอบว่าผู้ใช้ได้เลือก Ticker หรือไม่
    if not active_configs:
        st.warning("Please select at least one ticker to start the analysis.")
    else:
        # 8. รันการคำนวณด้วย config ที่เลือก
        with st.spinner('Calculating... Please wait.'):
            data = un_16(active_configs)
        
        if data.empty:
            st.error("Failed to generate data for the selected tickers. Please check logs or try again.")
        else:
            # ------------------- ส่วนแสดงผล (เหมือนเดิม) -------------------
            for i in selected_tickers:
                col_name = f'{i}_re'
                if col_name in data.columns:
                    data[col_name] = np.cumsum(data[col_name].values)

            df_new = data

            roll_over = []
            max_dd = df_new.maxcash_dd.values
            for i in range(len(max_dd)):
                try:
                    roll = max_dd[:i]
                    if len(roll) > 0:
                        roll_min = np.min(roll)
                    else:
                        roll_min = 0
                    roll_max = 0
                    data_roll = roll_min - roll_max
                    roll_over.append(data_roll)
                except:
                    roll_over.append(0)

            min_sum_val = np.min(roll_over)
            if min_sum_val == 0:
                min_sum = 1 
            else:
                min_sum = abs(min_sum_val)
                
            sum_val = (df_new.cf.values / min_sum) * 100
            cf = df_new.cf.values

            df_all = pd.DataFrame(list(zip(cf, roll_over)), columns=['Sum_Delta', 'Max_Sum_Buffer'])
            df_all_2 = pd.DataFrame(sum_val, columns=['True_Alpha'])

            st.write('____')
            st.write(f"({df_all.Sum_Delta.values[-1]:.2f}, {df_all.Max_Sum_Buffer.values[-1]:.2f}) , {df_all_2.True_Alpha.values[-1]:.2f}")

            col1, col2 = st.columns(2)
            col1.plotly_chart(px.line(df_all, title="Sum Delta vs Max Sum Buffer"))
            col2.plotly_chart(px.line(df_all_2, title="True Alpha (%)"))
            st.write('____')
            st.plotly_chart(px.line(df_new, title="Detailed Portfolio Simulation"))
else:
    st.error("Could not load any configuration. Please check the 'un15_fx_config.json' file.")
