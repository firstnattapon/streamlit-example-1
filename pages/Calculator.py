import streamlit as st
import numpy as np
import datetime
import thingspeak
import pandas as pd
import yfinance as yf





st.set_page_config(page_title="Calculator", page_icon="🔥")

x_1 = st.number_input('ราคา_NEGG_1.26' , step=0.01 , value =  yf.Ticker('NEGG').fast_info['lastPrice']   )
x_2 = st.number_input('ราคา_FFWM_6.88', step=0.01 ,   value = yf.Ticker('FFWM').fast_info['lastPrice']   )
st.write("_____") 

st.write(yf.Ticker('FFWM').fast_info['lastPrice']) 
st.write("_____") 

def sell (x_3):
  s1 = 1440/x_3
  s2 = 0
  

x_3 = st.number_input('NEGG_ASSET', step=0.01 ,   )
x_4 = st.number_input('FFWM_ASSET', step=0.01 ,   )
st.write("Limut_order_Sell_NEGG") 


a = 0.734588

st.write(round(a, 2))

  
st.write("""เป้าหมาย / asset
ปัดลงหาราคา 
ได้ราคาคูณของ
เอาไปลบ fix c หาส่วนต่าง
เอา ส่วนต่างไปหารราคา ปัดลง ได้ของ * ราคา 
ตรวจสอบ ของ * ราคาที่หาได้  + ของ * ราคาที่หาได้    น้อยกว่า fix c""") 
