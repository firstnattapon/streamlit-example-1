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

thingspeak_1 = st.checkbox('@_FFWM_ASSET')
if thingspeak_1 :
  add_1 = st.number_input('@_FFWM_ASSET', step=0.001 ,  value=0.)
  _FFWM_ASSET = st.button("GO!")
  if _FFWM_ASSET :
    client.update(  {'field1': add_1 } )
    st.write(add_1) 
    
thingspeak_2 = st.checkbox('@_NEGG_ASSET')
if thingspeak_2 :
  add_2 = st.number_input('@_NEGG_ASSET', step=0.001 ,  value=0.)
  _NEGG_ASSET = st.button("GO!")
  if _NEGG_ASSET :
    client.update(  {'field2': add_2 }  )
    st.write(add_2) 
st.write("_____") 

FFWM_ASSET_LAST = client.get_field_last(field='field1')
FFWM_ASSET_LAST =  json.loads(FFWM_ASSET_LAST)
st.write(FFWM_ASSET_LAST) 


# NEGG_ASSET_LAST = client.get_field_last(field='field2')
# NEGG_ASSET_LAST = int(eval(json.loads(NEGG_ASSET_LAST)['field2']))
# st.write(NEGG_ASSET_LAST) 


# x_3 = st.number_input('NEGG_ASSET', step=0.001 ,  value= NEGG_ASSET_LAST )
# x_4 = st.number_input('FFWM_ASSET', step=0.001  , value= FFWM_ASSET_LAST  )
# st.write("_____") 

# try:
#   s8 , s9 , s10 =  sell( asset = x_3 , Diff= x_2)
#   s11 , s12 , s13 =  sell(asset = x_4 , Diff= x_2)
#   b8 , b9 , b10 =  buy(asset = x_3 , Diff= x_2)
#   b11 , b12 , b13 =  buy(asset = x_4 , Diff= x_2)
  
#   st.write("Limut_Order_NEGG") 
#   st.write( 'sell' , '   ' ,'A', b9  , 'P' , b8 ,'C' ,b10  )
#   sell_negg = st.checkbox('sell_negg')
#   if sell_negg :
#     client.update(  {'NEGG_ASSET': x_3 - b9  }  )
    
    
#   st.write(yf.Ticker('NEGG').fast_info['lastPrice'])
#   st.write( 'buy' , '   ','A',  s9  ,  'P' , s8 , 'C' ,s10  )
#   st.write("_____") 
  
#   st.write("Limut Order_FFWM") 
#   st.write( 'sell' , '   ' , 'A', b12 , 'P' , b11  , 'C' , b13  )
#   st.write(yf.Ticker('FFWM').fast_info['lastPrice'])
#   st.write(  'buy' , '   ', 'A', s12 , 'P' , s11  , 'C'  , s13  )
#   st.write("_____") 

#   # client.update(  {'FFWM_ASSET': cf , 'NEGG_ASSET': cf / k_3   }  )

# except:pass
