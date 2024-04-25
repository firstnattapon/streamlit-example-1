import streamlit as st
import numpy as np
import datetime
import thingspeak
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="Calculator", page_icon="⌨️")

def sell (x_3 , fix_c=1500 , Diff=60):
  s1 =  (1500-Diff) /x_3
  s2 =  round(s1, 2)
  s3 =  s2*x_3
  s4 =  abs(s3 - fix_c)
  s5 =  round( s4 / s2 )  
  s6 =  s5*s2
  s7 =  (x_3 * s2) + s6
  return s2 , s5 , round(s7, 2)

def buy (x_3 , fix_c=1500 , Diff=60):
  b1 =  (1500+Diff) /x_3
  b2 =  round(b1, 2)
  b3 =  b2*x_3
  b4 =  abs(b3 - fix_c)
  b5 =  round( b4 / b2 )  
  b6 =  b5*b2
  b7 =  (x_3 * b2) - b6
  return b2 , b5 , round(b7, 2)
  
x_2 = st.number_input('Diff', step=1 , value= 60  )
st.write("_____") 

x_3 = st.number_input('NEGG_ASSET', step=0.001 ,  value=np.nan )
x_4 = st.number_input('FFWM_ASSET', step=0.001  , value=np.nan  )
st.write("_____") 

try:
  s8 , s9 , s10 =  sell(x_3 , Diff= x_2)
  s11 , s12 , s13 =  sell(x_4 , Diff= x_2)
  b8 , b9 , b10 =  buy(x_3 , Diff= x_2)
  b11 , b12 , b13 =  buy(x_4 , Diff= x_2)

  st.write("Limut_Order_NEGG") 
  st.write( 'A', b9  , 'P' , b8 ,'C' ,b10 )
  st.write(yf.Ticker('NEGG').fast_info['lastPrice'])
  st.write('A',  s9  ,  'P' , s8 , 'C' ,s10 )
  st.write("_____") 
  
  st.write("Limut Order_FFWM") 
  st.write( 'A', b12 , 'P' , b11  , 'C' ,b13 )
  st.write(yf.Ticker('FFWM').fast_info['lastPrice'])
  st.write( 'A', s12 , 'P' , s11  , 'C' ,s13 )
  st.write("_____") 

except:pass
