import streamlit as st
import numpy as np
import datetime
import thingspeak
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="Calculator", page_icon="⌨️")

def sell (asset = 0 , fix_c=1500 , Diff=60):
  s1 =  (1500-Diff) /asset
  s2 =  round(s1, 2)
  s3 =  s2  *asset
  s4 =  abs(s3 - fix_c)
  s5 =  round( s4 / s2 )  
  s6 =  s5*s2
  s7 =  (asset * s2) + s6
  return s2 , s5 , round(s7, 2)

def buy (asset = 0 , fix_c=1500 , Diff=60):
  b1 =  (1500+Diff) /asset
  b2 =  round(b1, 2)
  b3 =  b2 *asset
  b4 =  abs(b3 - fix_c)
  b5 =  round( b4 / b2 )  
  b6 =  b5*b2
  b7 =  (asset * b2) - b6
  return b2 , b5 , round(b7, 2)
  
x_2 = st.number_input('Diff', step=1 , value= 60  )
st.write("_____") 
x_3 = st.number_input('NEGG_ASSET', step=0.001 ,  value= 1875.28 )
x_4 = st.number_input('FFWM_ASSET', step=0.001  , value=218.66  )
st.write("_____") 

try:
  s8 , s9 , s10 =  sell( asset = x_3 , Diff= x_2)
  s11 , s12 , s13 =  sell(asset = x_4 , Diff= x_2)
  b8 , b9 , b10 =  buy(asset = x_3 , Diff= x_2)
  b11 , b12 , b13 =  buy(asset = x_4 , Diff= x_2)

  st.write("Limut_Order_NEGG") 
  st.write( 'A', b9  , 'P' , b8 ,'C' ,b10 ,'' ,'' , 'sell' )
  st.write(yf.Ticker('NEGG').fast_info['lastPrice'])
  st.write('A',  s9  ,  'P' , s8 , 'C' ,s10 ,'' ,'' , 'buy' )
  st.write("_____") 
  
  st.write("Limut Order_FFWM") 
  st.write( 'A', b12 , 'P' , b11  , 'C' ,b13 )
  st.write(yf.Ticker('FFWM').fast_info['lastPrice'])
  st.write( 'A', s12 , 'P' , s11  , 'C' ,s13 )
  st.write("_____") 

except:pass
