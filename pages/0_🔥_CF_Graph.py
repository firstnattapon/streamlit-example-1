import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import json
import plotly.express as px

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="CF_Graph", page_icon="🔥" , layout= "wide" )

# # --- ฟังก์ชันหลักในการคำนวณ (เหมือนเดิมทุกประการ) ---
# def CF_Graph(entry=1.26, ref=1.26, Fixed_Asset_Value=1500., Cash_Balan=650.):
#     try:
#         step = 0.01
#         # ป้องกันกรณี entry เป็น 0 หรือค่าผิดปกติ
#         if entry <= 0:
#             return pd.DataFrame(), 0.0
            
#         samples = np.arange(0, np.around(entry, 2) * 3 + step, step)

#         df = pd.DataFrame()
#         df['Asset_Price'] = np.around(samples, 2)
#         # ป้องกันการหารด้วยศูนย์
#         df = df[df['Asset_Price'] > 0]
        
#         df['Fixed_Asset_Value'] = Fixed_Asset_Value
#         df['Amount_Asset'] = df['Fixed_Asset_Value'] / df['Asset_Price']

#         df_top = df[df.Asset_Price >= np.around(entry, 2)].copy()
#         df_top['Cash_Balan_top'] = (df_top['Amount_Asset'].shift(1) - df_top['Amount_Asset']) * df_top['Asset_Price']
#         df_top.fillna(0, inplace=True)
#         np_Cash_Balan_top = df_top['Cash_Balan_top'].values

#         xx = np.zeros(len(np_Cash_Balan_top))
#         y_0 = Cash_Balan
#         for idx, v_0 in enumerate(np_Cash_Balan_top):
#             z_0 = y_0 + v_0
#             y_0 = z_0
#             xx[idx] = y_0

#         df_top['Cash_Balan_top'] = xx
#         df_top = df_top.rename(columns={'Cash_Balan_top': 'Cash_Balan'})
#         df_top = df_top.sort_values(by='Amount_Asset')
#         if not df_top.empty:
#             df_top = df_top[:-1]

#         df_down = df[df.Asset_Price <= np.around(entry, 2)].copy()
#         df_down['Cash_Balan_down'] = (df_down['Amount_Asset'].shift(-1) - df_down['Amount_Asset']) * df_down['Asset_Price']
#         df_down.fillna(0, inplace=True)
#         df_down = df_down.sort_values(by='Asset_Price', ascending=False)
#         np_Cash_Balan_down = df_down['Cash_Balan_down'].values

#         xxx = np.zeros(len(np_Cash_Balan_down))
#         y_1 = Cash_Balan
#         for idx, v_1 in enumerate(np_Cash_Balan_down):
#             z_1 = y_1 + v_1
#             y_1 = z_1
#             xxx[idx] = y_1

#         df_down['Cash_Balan_down'] = xxx
#         df_down = df_down.rename(columns={'Cash_Balan_down': 'Cash_Balan'})
        
#         df_final = pd.concat([df_top, df_down], axis=0)
#         df_final['net_pv'] = df_final['Fixed_Asset_Value'] + df_final['Cash_Balan']
        
#         df_2 = df_final[df_final['Asset_Price'] == np.around(ref, 2)]['net_pv'].values
        
#         # คืนค่า default ถ้าไม่เจอราคาที่ตรงกัน
#         result_pv = df_2[-1] if len(df_2) > 0 else 0.0
        
#         return df_final[['Asset_Price', 'Cash_Balan', 'net_pv', 'Fixed_Asset_Value']], result_pv
#     except Exception as e:
#         st.error(f"เกิดข้อผิดพลาดในการคำนวณกราฟ: {e}")
#         return pd.DataFrame(), 0.0

# --- ส่วนหลักของแอปพลิเคชัน ---

# 1. โหลดการตั้งค่าจากไฟล์ JSON
try:
    with open('cf_graph_config.json', 'r', encoding='utf-8') as f:
        assets_config = json.load(f)
except FileNotFoundError:
    st.error("ไม่พบไฟล์ 'cf_graph_config.json'. กรุณาสร้างไฟล์และใส่ข้อมูล Asset ตามโครงสร้างใหม่")
    st.stop() 

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
        st.components.v1.iframe("https://monica.im/share/artifact?id=a3cRH7RVEHvBTjPhBC5Hd7", width=1500 , height=1000  , scrolling=0)
    with st.expander("หลักการ Rollover"):
        st.components.v1.iframe("https://monica.im/share/artifact?id=E9Mg5JX9RaAcfssZsU7K3E", width=1500 , height=1000  , scrolling=0)

# # 4. สร้าง Tab "DATA" สำหรับรับ Input และแสดงประวัติ (REWORKED SECTION - FINAL V2)
# with tabs[0]:
#     st.write("⚙️ **ตั้งค่าทั่วไปและประวัติ Asset**")
    
#     x_6 = st.number_input('Cash_Balan เริ่มต้น (ใช้กับทุกตัว)', step=1.0, value=0.)
#     st.write("---")
    
#     for asset in assets_config:
#         ticker = asset['ticker']
#         # ดึงค่า Final มาเพื่อใช้ใน Label ของ expander
#         final_str = asset.get('Final', 'N/A') 

#         # [แก้ไข] สร้าง expander label แบบไดนามิก
#         with st.expander(f"ตั้งค่าและดูประวัติ: **{ticker}** | \"{final_str}\""):
#             try:
#                 # --- ส่วนประมวลผลข้อมูล (เบื้องหลัง) ---
#                 # เรายังคงต้อง parse ค่า final_str เพื่อนำไปคำนวณ
#                 final_parts = final_str.split(',')
#                 final_price = float(final_parts[0].strip())
#                 final_fav = float(final_parts[1].strip())

#                 asset_params[ticker] = {'entry': final_price, 'fav': final_fav}

#                 # --- ส่วนแสดงผลบน UI (ข้างใน Expander) ---
#                 try:
#                     last_price = yf.Ticker(ticker).fast_info.get('lastPrice', final_price)
#                 except Exception:
#                     st.warning(f"ไม่สามารถดึงราคาล่าสุดของ {ticker} ได้, ใช้ราคา Final แทน")
#                     last_price = final_price

#                 current_prices[ticker] = st.number_input(
#                     "ราคาปัจจุบัน (สำหรับคำนวณ)", 
#                     step=0.01, 
#                     value=float(last_price), 
#                     key=f"price_{ticker}"
#                 )
                
#                 st.write("---")
#                 st.write("ข้อมูลดิบจาก Config:")
#                 st.json(asset)

#             except Exception as e:
#                 st.error(f"ข้อมูลใน config ของ {ticker} ผิดพลาด: {e}")
#                 # กำหนดค่า default ป้องกันแอปพัง
#                 asset_params[ticker] = {'entry': 0, 'fav': 0}
#                 current_prices[ticker] = 0.0

# # 5. สร้าง Tab ของแต่ละ Asset และแสดงกราฟ (ส่วนนี้ทำงานได้ปกติ ไม่ต้องแก้ไข)
# for i, asset in enumerate(assets_config):
#     with tabs[i + 2]: 
#         ticker = asset['ticker']
#         if ticker in current_prices and ticker in asset_params:
#             ref_price = current_prices.get(ticker, 0)
#             entry_price = asset_params[ticker].get('entry', 0)
#             fixed_asset_value = asset_params[ticker].get('fav', 0)

#             st.write(f"### กราฟแสดงความสัมพันธ์ของ {ticker}")
            
#             df, df_rf_value = CF_Graph(
#                 entry=entry_price, ref=ref_price, 
#                 Fixed_Asset_Value=fixed_asset_value, Cash_Balan=x_6
#             )
            
#             results_rf[ticker] = df_rf_value
            
#             if not df.empty:
#                 as_1 = df.set_index('Asset_Price')
#                 as_1_py = px.line(as_1, title=f"Analysis for {ticker} (Entry: {entry_price}, FAV: {fixed_asset_value:,.0f})")
#                 as_1_py.add_vline(x=ref_price, line_width=1.5, line_dash="dash", line_color="red")
#                 as_1_py.add_vline(x=entry_price, line_width=1.5, line_dash="solid", line_color="green", opacity=0.6)
#                 y_position = df['net_pv'].median() 
#                 as_1_py.add_annotation(x=ref_price, y=y_position, text=f"Current: {ref_price:.2f}", showarrow=False, yshift=15, font=dict(color="red", size=12), bgcolor="rgba(255, 255, 255, 0.6)")
#                 as_1_py.add_annotation(x=entry_price, y=y_position, text=f"Entry: {entry_price:.2f}", showarrow=False, yshift=-15, font=dict(color="green", size=12), bgcolor="rgba(255, 255, 255, 0.6)")
#                 st.plotly_chart(as_1_py, use_container_width=True)
#                 st.metric(label=f"Net PV ของ {ticker} ที่ราคาปัจจุบัน", value=f"${df_rf_value:,.2f}")
#                 st.write("_____") 
#             else:
#                 st.warning("ไม่สามารถสร้างกราฟได้เนื่องจากไม่มีข้อมูล")
#         else:
#             st.warning(f"ไม่พบข้อมูลการตั้งค่าสำหรับ {ticker} ใน Tab DATA, กรุณาตรวจสอบไฟล์ config")

# # 6. แสดงผลสรุปรวม (ส่วนนี้ทำงานได้ปกติ ไม่ต้องแก้ไข)
# st.write("_______")
# st.header("สรุปผลรวมพอร์ต")
# total_rf = sum(results_rf.values())
# num_assets = len(assets_config)
# total_fixed_asset_value = sum(params['fav'] for params in asset_params.values())
# total_initial_cash = x_6 * num_assets
# st.metric("✅ **SUM Net PV (ตามราคาอ้างอิงปัจจุบัน)**", f"${total_rf:,.2f}")
# col1, col2, col3 = st.columns(3)
# col1.metric("จำนวน Asset ทั้งหมด", f"{num_assets} ตัว")
# col2.metric("มูลค่า Fixed Asset รวม", f"${total_fixed_asset_value:,.2f}")
# col3.metric("เงินสดเริ่มต้นรวม", f"${total_initial_cash:,.2f}")
# with st.expander("ดู Net PV ของแต่ละตัว (ตามราคาอ้างอิง)"):
#     for ticker, value in results_rf.items():
#         st.write(f"{ticker}: ${value:,.2f}")
