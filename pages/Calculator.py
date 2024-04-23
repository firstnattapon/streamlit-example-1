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

def sell (x_3 , fix_c=1500):
  s1 =  1440/x_3
  s2 =  round(s1, 2)
  s3 =  s2*x_3
  s4 =  abs(s3 - fix_c)
  s5 =  int( s4 / s2 )  
  s6 =  s5*s2
  s7 =  (x_3 * s2) + s6
  return s2 , s5 , s7
  
  

x_3 = st.number_input('NEGG_ASSET', step=0.01 ,   )
x_4 = st.number_input('FFWM_ASSET', step=0.01 ,   )
st.write("Limut_order_Sell_NEGG") 
s8 , s9 , s10 =  sell(x_3)

st.write(s8 , s9 , s10 )

  
st.write("""เป้าหมาย / asset
ปัดลงหาราคา 
ได้ราคาคูณของ
เอาไปลบ fix c หาส่วนต่าง
เอา ส่วนต่างไปหารราคา ปัดลง ได้ของ * ราคา 
ตรวจสอบ ของ * ราคาที่หาได้  + ของ * ราคาที่หาได้    น้อยกว่า fix c""") 
