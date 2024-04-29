import streamlit as st
import numpy as np
import datetime
import thingspeak
import pandas as pd
import yfinance as yf
import json

st.set_page_config(page_title="Calculator", page_icon="⌨️")

channel_id = 2528199
write_api_key = '2E65V8XEIPH9B2VV'
client = thingspeak.Channel(channel_id, write_api_key , fmt='json')

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

col13, col14 , col15  = st.columns(3)

Start = col13.checkbox('start')
if Start :
  thingspeak_1 = col13.checkbox('@_FFWM_ASSET')
  if thingspeak_1 :
    add_1 = col13.number_input('@_FFWM_ASSET', step=0.001 ,  value=0.)
    _FFWM_ASSET = col13.button("GO!")
    if _FFWM_ASSET :
      client.update(  {'field1': add_1 } )
      col13.write(add_1) 
      
  thingspeak_2 = col13.checkbox('@_NEGG_ASSET')
  if thingspeak_2 :
    add_2 = col13.number_input('@_NEGG_ASSET', step=0.001 ,  value=0.)
    _NEGG_ASSET = col13.button("GO!")
    if _NEGG_ASSET :
      client.update(  {'field2': add_2 }  )
      col13.write(add_2) 

FFWM_ASSET_LAST = client.get_field_last(field='field1')
FFWM_ASSET_LAST =  eval(json.loads(FFWM_ASSET_LAST)['field1'])

NEGG_ASSET_LAST = client.get_field_last(field='field2')
NEGG_ASSET_LAST = eval(json.loads(NEGG_ASSET_LAST)['field2'])

x_3 = col14.number_input('NEGG_ASSET', step=0.001 ,  value= NEGG_ASSET_LAST )
x_4 = col15.number_input('FFWM_ASSET', step=0.001  , value= FFWM_ASSET_LAST  )
st.write("_____") 

try:
  
  s8 , s9 , s10 =  sell( asset = x_3 , Diff= x_2)
  s11 , s12 , s13 =  sell(asset = x_4 , Diff= x_2)
  b8 , b9 , b10 =  buy(asset = x_3 , Diff= x_2)
  b11 , b12 , b13 =  buy(asset = x_4 , Diff= x_2)
  
  st.write("Limut_Order_NEGG") 
  st.write( 'sell' , '   ' ,'A', b9  , 'P' , b8 ,'C' ,b10  )
  
  col1, col2 , col3  = st.columns(3)
  sell_negg = col3.checkbox('sell_match_negg')
  if sell_negg :
    GO_NEGG_SELL = col3.button("GO!")
    if GO_NEGG_SELL :
      client.update(  {'field2': NEGG_ASSET_LAST - b9  } )
      col3.write(NEGG_ASSET_LAST - b9) 
    
  st.write(yf.Ticker('NEGG').fast_info['lastPrice'])
  
  col4, col5 , col6  = st.columns(3)
  st.write( 'buy' , '   ','A',  s9  ,  'P' , s8 , 'C' ,s10  )
  buy_negg = col6.checkbox('buy_match_negg')
  if buy_negg :
    GO_NEGG_Buy = col6.button("GO!")
    if GO_NEGG_Buy :
      client.update(  {'field2': NEGG_ASSET_LAST + s9  } )
      col6.write(NEGG_ASSET_LAST + s9) 
  
  st.write("_____") 
  
  
  st.write("Limut Order_FFWM") 
  st.write( 'sell' , '   ' , 'A', b12 , 'P' , b11  , 'C' , b13  )
  col7, col8 , col9  = st.columns(3)
  sell_ffwm = col9.checkbox('sell_match_ffwn')
  if sell_ffwm :
    GO_ffwm_sell = col9.button("GO!")
    if GO_ffwm_sell :
      client.update(  {'field1': FFWM_ASSET_LAST - b12  } )
      col9.write(FFWM_ASSET_LAST - b12) 
  
  st.write(yf.Ticker('FFWM').fast_info['lastPrice'])
  
  col10, col11 , col12  = st.columns(3)
  st.write(  'buy' , '   ', 'A', s12 , 'P' , s11  , 'C'  , s13  )
  buy_ffwm = col12.checkbox('buy_match_ffwm')
  if buy_ffwm :
    GO_ffwm_Buy = col12.button("GO!")
    if GO_ffwm_Buy :
      client.update(  {'field1': FFWM_ASSET_LAST + s12  } )
      col12.write(FFWM_ASSET_LAST + s12) 
  
  st.write("_____") 

except:pass
