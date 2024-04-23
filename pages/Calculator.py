import streamlit as st
import numpy as np
import datetime
import thingspeak
import pandas as pd
import yfinance as yf





st.set_page_config(page_title="Calculator", page_icon="🔥")

st.write("_____") 

def sell (x_3 , fix_c=1500 , Diff=60):
  s1 =  (1500-Diff) /x_3
  s2 =  round(s1, 2)
  s3 =  s2*x_3
  s4 =  abs(s3 - fix_c)
  s5 =  int( s4 / s2 )  
  s6 =  s5*s2
  s7 =  (x_3 * s2) + s6
  return s2 , s5 , round(s7, 2)
  
  
x_2 = st.number_input('Diff', step=1 , value=40  )
x_3 = st.number_input('NEGG_ASSET', step=0.01 , value=1  )
x_4 = st.number_input('FFWM_ASSET', step=0.01 , value=1   )
st.write("Limut_order_Sell_NEGG") 
s8 , s9 , s10 =  sell(x_3 , Diff= x_2)

st.write('P' , s8 ,'A', s9 , 'C' ,s10 )

  
st.write("""เป้าหมาย / asset
ปัดลงหาราคา 
ได้ราคาคูณของ
เอาไปลบ fix c หาส่วนต่าง
เอา ส่วนต่างไปหารราคา ปัดลง ได้ของ * ราคา 
ตรวจสอบ ของ * ราคาที่หาได้  + ของ * ราคาที่หาได้    น้อยกว่า fix c""") 
