import pandas as pd
import numpy as np
from numba import njit
import yfinance as yf
import streamlit as st
import thingspeak
import json
import streamlit.components.v1 as components

st.set_page_config(page_title="Add_CF_V2", page_icon="🔥")

 
channel_id_log = 2329127
write_api_key_log = 'V10DE0HKR4JKB014'
client_log = thingspeak.Channel(channel_id_log, write_api_key_log)
    
channel_id = 2394198
write_api_key = 'OVZNYQBL57GJW5JF'
client = thingspeak.Channel(channel_id, write_api_key)

channel_id_2 = 2528199
write_api_key_2 = '2E65V8XEIPH9B2VV'
client_2 = thingspeak.Channel(channel_id_2, write_api_key_2 , fmt='json')

FFWM_ASSET_LAST = client_2.get_field_last(field='field1')
FFWM_ASSET_LAST =  eval(json.loads(FFWM_ASSET_LAST)['field1'])

NEGG_ASSET_LAST = client_2.get_field_last(field='field2')
NEGG_ASSET_LAST = eval(json.loads(NEGG_ASSET_LAST)['field2'])

RIVN_ASSET_LAST = client_2.get_field_last(field='field3')
RIVN_ASSET_LAST = eval(json.loads(RIVN_ASSET_LAST)['field3'])

APLS_ASSET_LAST = client_2.get_field_last(field='field4')
APLS_ASSET_LAST = eval(json.loads(APLS_ASSET_LAST)['field4'])

NVTS_ASSET_LAST = client_2.get_field_last(field='field5')
NVTS_ASSET_LAST = eval(json.loads(NVTS_ASSET_LAST)['field5'])

QXO_ASSET_LAST = client_2.get_field_last(field='field6')
QXO_ASSET_LAST = eval(json.loads(QXO_ASSET_LAST)['field6'])

RXRX_ASSET_LAST = client_2.get_field_last(field='field7')
RXRX_ASSET_LAST = eval(json.loads(RXRX_ASSET_LAST)['field7'])

AGL_ASSET_LAST = client_2.get_field_last(field='field8')
AGL_ASSET_LAST = eval(json.loads(AGL_ASSET_LAST)['field8'])

x_1 = st.number_input('ราคา_NEGG_1.26 , 25.20' , step=0.01 ,  value =  yf.Ticker('NEGG').fast_info['lastPrice']   ) 
x_2 = st.number_input('ราคา_FFWM_6.88', step=0.01  ,  value = yf.Ticker('FFWM').fast_info['lastPrice']   ) 
x_3 = st.number_input('ราคา_RIVN_10.07', step=0.01 ,   value = yf.Ticker('RIVN').fast_info['lastPrice'] ) 
x_4 = st.number_input('ราคา_APLS_39.61', step=0.01 ,   value = yf.Ticker('APLS').fast_info['lastPrice'] ) 
x_5 = st.number_input('ราคา_NVTS_3.05', step=0.01, value=yf.Ticker('NVTS').fast_info['lastPrice'])
x_6 = st.number_input('ราคา_QXO_19.00', step=0.01, value=yf.Ticker('QXO').fast_info['lastPrice'])
x_7 = st.number_input('ราคา_RXRX_5.40', step=0.01, value=yf.Ticker('RXRX').fast_info['lastPrice'])
x_8 = st.number_input('ราคา_AGL_3.00', step=0.01, value=yf.Ticker('AGL').fast_info['lastPrice'])

st.write("_____") 

y_1 = st.number_input('FFWM_asset', step=0.01 , value = FFWM_ASSET_LAST ) 
y_1 = y_1*x_2
st.write(y_1) 

y_2 = st.number_input('NEGG_asset', step=0.01 , value = NEGG_ASSET_LAST  ) 
y_2 = y_2*x_1
st.write(y_2) 

y_3 = st.number_input('RIVN_asset', step=0.01 , value = RIVN_ASSET_LAST  ) 
y_3 = y_3*x_3
st.write(y_3)

y_4 = st.number_input('APLS_asset', step=0.01 , value = APLS_ASSET_LAST ) 
y_4 = y_4*x_4
st.write(y_4) 

y_5 = st.number_input('NVTS_asset', step=0.01, value= NVTS_ASSET_LAST)
y_5 = y_5 * x_5
st.write(y_5)

y_6 = st.number_input('QXO_asset', step=0.01, value= QXO_ASSET_LAST ) # LV
y_6 = y_6 * x_6
st.write(y_6)

y_7 = st.number_input('RXRX_asset', step=0.01, value= RXRX_ASSET_LAST ) # LV
y_7 = y_7 * x_7
st.write(y_7)

st.write("_____")

y_8 = st.number_input('AGL_asset', step=0.01, value= AGL_ASSET_LAST ) # LV
y_8 = y_8 * x_8
st.write(y_8)

st.write("_____")

Product_cost = st.number_input('Product_cost', step=0.01 , value = 10750. )
j_1 = st.number_input('Portfolio_cash', step=0.01 , value = 0.00 )
number = (y_1 + y_2 + y_3 + y_4 + y_5 + y_6 + y_7 + y_8 ) + j_1 # แก้
st.write('now_pv:' , number) 

st.write("_____")


t_0 = 25.20 * 6.88 * 10.07 * 39.61 * 3.05 * 19.00 * 5.40 * 3.00 # แก้

t_n = yf.Ticker('NEGG').info['currentPrice'] * yf.Ticker('FFWM').info['currentPrice'] *yf.Ticker('RIVN').info['currentPrice'] * yf.Ticker('APLS').info['currentPrice'] * yf.Ticker('NVTS').info['currentPrice'] * yf.Ticker('QXO').info['currentPrice']  * yf.Ticker('RXRX').info['currentPrice  * yf.Ticker('AGL').info['currentPrice']                      
ln =  -1500 * np.log ( t_0 / t_n)

st.write ('t_0' , t_0)
st.write ('t_n' , t_n)
st.write ('fix' , ln)
st.write ('log_pv' , Product_cost + ln) ### แก้
st.write ('now_pv' , number)
st.write ('____')
net_cf = number - (Product_cost + ln)
st.write ( 'net_cf' , net_cf ) ##แก้
st.write ('____')

if st.button("rerun"):
    st.rerun()
st.write("_____") 
    
Check_ADD = st.checkbox('ADD_CF ')
if Check_ADD :
    button_ADD = st.button("ADD_CF")
    if button_ADD:    
        try:
            client.update(  {'field1': net_cf , 'field2': net_cf / Product_cost , 'field3': j_1 , 'field4': Product_cost - net_cf }  )
            st.write({'Cashflow': net_cf , 'Pure_Alpha': net_cf / Product_cost ,  'ฺBuffer': j_1  }) 
        except:pass

st.write("_____")
st.write("Cashflow") 
components.iframe('https://thingspeak.com/channels/2394198/charts/1?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&type=line&update=15' , width=800, height=200)
st.write("_____")
st.write("Pure_Alpha")
components.iframe('https://thingspeak.com/channels/2394198/charts/2?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&type=line&update=15' , width=800, height=200)
st.write("Product_cost")
components.iframe('https://thingspeak.mathworks.com/channels/2394198/charts/4?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&type=line&update=15' , width=800, height=200)

st.write("_____") 
st.write("ฺBuffer")
components.iframe('https://thingspeak.com/channels/2394198/charts/3?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&type=line&update=15' , width=800, height=200)
st.write("_____") 
        
