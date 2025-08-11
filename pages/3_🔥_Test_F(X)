import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import json
import plotly.express as px

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="CF_Graph", page_icon="🔥" , layout= "wide" )

# --- ฟังก์ชันหลักในการคำนวณ (เหมือนเดิมทุกประการ) ---
def CF_Graph(entry=1.26, ref=1.26, Fixed_Asset_Value=1500., Cash_Balan=650.):
    try:
        step = 0.01
        # ป้องกันกรณี entry เป็น 0 หรือค่าผิดปกติ
        if entry <= 0:
            return pd.DataFrame(), 0.0
            
        samples = np.arange(0, np.around(entry, 2) * 3 + step, step)

        df = pd.DataFrame()
        df['Asset_Price'] = np.around(samples, 2)
        # ป้องกันการหารด้วยศูนย์
        df = df[df['Asset_Price'] > 0]
        
        df['Fixed_Asset_Value'] = Fixed_Asset_Value
        df['Amount_Asset'] = df['Fixed_Asset_Value'] / df['Asset_Price']

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
        df_top = df_top.rename(columns={'Cash_Balan_top': 'Cash_Balan'})
        df_top = df_top.sort_values(by='Amount_Asset')
        if not df_top.empty:
            df_top = df_top[:-1]

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
        df_down = df_down.rename(columns={'Cash_Balan_down': 'Cash_Balan'})
        
        df_final = pd.concat([df_top, df_down], axis=0)
        df_final['net_pv'] = df_final['Fixed_Asset_Value'] + df_final['Cash_Balan']
        
        df_2 = df_final[df_final['Asset_Price'] == np.around(ref, 2)]['net_pv'].values
        
        # คืนค่า default ถ้าไม่เจอราคาที่ตรงกัน
        result_pv = df_2[-1] if len(df_2) > 0 else 0.0
        
        return df_final[['Asset_Price', 'Cash_Balan', 'net_pv', 'Fixed_Asset_Value']], result_pv
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการคำนวณกราฟ: {e}")
        return pd.DataFrame(), 0.0

# --- ส่วนหลักของแอปพลิเคชัน ---

# 1. โหลดการตั้งค่าจากไฟล์ JSON
try:
    with open('cf_graph_config.json', 'r', encoding='utf-8') as f:
        assets_config = json.load(f)
except FileNotFoundError:
    st.error("ไม่พบไฟล์ 'cf_graph_config.json'. กรุณาสร้างไฟล์และใส่ข้อมูล Asset ตามโครงสร้างใหม่")
    st.stop() # หยุดการทำงานถ้าไม่มีไฟล์ config

# 2. สร้างชื่อ Tab ทั้งหมด
tab_names = ['DATA', 'BATA'] + [asset['ticker'] for asset in assets_config]  
tabs = st.tabs(tab_names)

# Dictionary สำหรับเก็บค่าที่รับจาก input และค่าเฉพาะตัวของ Asset
current_prices = {}
asset_params = {}
results_rf = {}

# 3. สร้าง Tab "BATA" (หลักการ)
with tabs[1]:
    with st.expander("หลักการ BATA" , expanded=True):
        st.components.v1.iframe("https://monica.im/share/artifact?id=sK3gfn5iVhqdVMEgTNWHdP", width=1500 , height=1000  , scrolling=0)
    with st.expander("หลักการ Rollover"):
        st.components.v1.iframe("https://monica.im/share/artifact?id=E9Mg5JX9RaAcfssZsU7K3E", width=1500 , height=1000  , scrolling=0)

# 4. สร้าง Tab "DATA" สำหรับรับ Input และแสดงประวัติ (REWORKED SECTION)
with tabs[0]:
    st.write("⚙️ **ตั้งค่าและประวัติของแต่ละ Asset**")
    
    # Input ส่วนกลางเหลือแค่ Cash Balance
    x_6 = st.number_input('Cash_Balan เริ่มต้น (ใช้กับทุกตัว)', step=1.0, value=0.)
    st.write("---")
    
    # สร้าง Expander และ Input สำหรับแต่ละ Asset
    for asset in assets_config:
        ticker = asset['ticker']
        
        with st.expander(f"ประวัติและตั้งค่าสำหรับ: **{ticker}**"):
            try:
                # แยกค่า Final ออกเป็น Entry Price และ Fixed Asset Value
                final_parts = asset.get('Final', '0,0').split(',')
                final_price = float(final_parts[0].strip())
                final_fav = float(final_parts[1].strip())

                # เก็บค่าเฉพาะตัวของ asset นี้ไว้ใน Dictionary
                asset_params[ticker] = {
                    'entry': final_price,
                    'fav': final_fav
                }
                
                # --- แสดงข้อมูลประวัติ ---
                st.markdown(f"**สถานะสุดท้าย (Final):** ราคาอ้างอิง: `{final_price}`, ทุนคงที่: `{final_fav}`")
                
                if 'Original' in asset and asset['Original']:
                    st.text(f"Original: {asset['Original']}")
                if 'history_1' in asset and asset['history_1']:
                    st.text(f"History 1: {asset['history_1']}")
                if 'history_2' in asset and asset['history_2']:
                    st.text(f"History 2: {asset['history_2']}")
                if 'comment' in asset and asset['comment']:
                    st.info(f"Comment: {asset['comment']}")
                
                st.write("---")

                # --- Input สำหรับราคาปัจจุบัน ---
                try:
                    # ดึงราคาล่าสุดจาก yfinance เป็นค่าเริ่มต้น
                    last_price = yf.Ticker(ticker).fast_info.get('lastPrice', final_price)
                except Exception:
                    st.warning(f"ไม่สามารถดึงราคาล่าสุดของ {ticker} ได้, ใช้ราคา Final แทน")
                    last_price = final_price

                # เก็บราคาปัจจุบันที่ผู้ใช้กรอกลงใน Dictionary
                current_prices[ticker] = st.number_input(
                    f"ราคาปัจจุบันของ {ticker} (สำหรับคำนวณ)", 
                    step=0.01, 
                    value=float(last_price), 
                    key=f"price_{ticker}"
                )

            except Exception as e:
                st.error(f"ข้อมูลใน config ของ {ticker} ผิดพลาด: {e}")


# 5. สร้าง Tab ของแต่ละ Asset และแสดงกราฟ (REWORKED SECTION)
for i, asset in enumerate(assets_config):
    with tabs[i + 2]: # เริ่มจาก tab ที่ 2 (ถัดจาก DATA, BATA)
        ticker = asset['ticker']
        
        # ตรวจสอบว่ามีข้อมูลของ ticker นี้หรือไม่ (ป้องกัน error หาก config ผิด)
        if ticker in current_prices and ticker in asset_params:
            # ดึงค่าต่างๆ จาก Dictionaries ที่เราเตรียมไว้
            ref_price = current_prices[ticker]
            entry_price = asset_params[ticker]['entry']
            fixed_asset_value = asset_params[ticker]['fav']

            st.write(f"### กราฟแสดงความสัมพันธ์ของ {ticker}")
            
            # เรียกใช้ฟังก์ชันคำนวณด้วยค่าเฉพาะตัวของ Asset
            df, df_rf_value = CF_Graph(
                entry=entry_price, 
                ref=ref_price, 
                Fixed_Asset_Value=fixed_asset_value, 
                Cash_Balan=x_6 # ใช้ Cash Balance ร่วมกัน
            )
            
            # เก็บผลลัพธ์ net_pv สำหรับการสรุปผลรวม
            results_rf[ticker] = df_rf_value
            
            if not df.empty:
                # พล็อตกราฟ
                as_1 = df.set_index('Asset_Price')
                as_1_py = px.line(as_1, title=f"Analysis for {ticker} (Entry: {entry_price}, FAV: {fixed_asset_value:,.0f})")

                # เพิ่มเส้นแนวตั้ง
                as_1_py.add_vline(x=ref_price, line_width=1.5, line_dash="dash", line_color="red")
                as_1_py.add_vline(x=entry_price, line_width=1.5, line_dash="solid", line_color="green", opacity=0.6)

                # คำนวณตำแหน่งกึ่งกลางของแกน Y
                y_position = df['net_pv'].median() 

                # เพิ่มข้อความ (Annotation)
                as_1_py.add_annotation(x=ref_price, y=y_position, text=f"Current: {ref_price:.2f}", showarrow=False, yshift=15, font=dict(color="red", size=12), bgcolor="rgba(255, 255, 255, 0.6)")
                as_1_py.add_annotation(x=entry_price, y=y_position, text=f"Entry: {entry_price:.2f}", showarrow=False, yshift=-15, font=dict(color="green", size=12), bgcolor="rgba(255, 255, 255, 0.6)")

                st.plotly_chart(as_1_py, use_container_width=True)
                
                st.metric(label=f"Net PV ของ {ticker} ที่ราคาปัจจุบัน", value=f"${df_rf_value:,.2f}")
                st.write("_____") 

            else:
                st.warning("ไม่สามารถสร้างกราฟได้เนื่องจากไม่มีข้อมูล")
        else:
            st.warning(f"ไม่พบข้อมูลการตั้งค่าสำหรับ {ticker} ใน Tab DATA, กรุณาตรวจสอบไฟล์ config")

# 6. แสดงผลสรุปรวม (REWORKED SECTION)
st.write("_______")
st.header("สรุปผลรวมพอร์ต")

total_rf = sum(results_rf.values())
num_assets = len(assets_config)
# คำนวณ Fixed Asset รวมจากค่าเฉพาะตัวของแต่ละ Asset
total_fixed_asset_value = sum(params['fav'] for params in asset_params.values())
total_initial_cash = x_6 * num_assets

st.metric("✅ **SUM Net PV (ตามราคาอ้างอิงปัจจุบัน)**", f"${total_rf:,.2f}")

col1, col2, col3 = st.columns(3)
col1.metric("จำนวน Asset ทั้งหมด", f"{num_assets} ตัว")
col2.metric("มูลค่า Fixed Asset รวม", f"${total_fixed_asset_value:,.2f}")
col3.metric("เงินสดเริ่มต้นรวม", f"${total_initial_cash:,.2f}")

# แสดงค่าแต่ละตัวใน Expander เพื่อตรวจสอบ
with st.expander("ดู Net PV ของแต่ละตัว (ตามราคาอ้างอิง)"):
    for ticker, value in results_rf.items():
        st.write(f"{ticker}: ${value:,.2f}")
