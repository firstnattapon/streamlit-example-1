import streamlit as st
import numpy as np
import datetime
import thingspeak
import pandas as pd
import yfinance as yf
import json

st.set_page_config(page_title="Monitor", page_icon="📈" , layout="wide" )
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

channel_id_2 = 2385118
write_api_key_2 = 'IPSG3MMMBJEB9DY8'
client_2 = thingspeak.Channel(channel_id_2, write_api_key_2 , fmt='json' )

def Monitor (Ticker = 'FFWM' , field = 2  ):

    tickerData = yf.Ticker(Ticker)
    tickerData = round(tickerData.history(period= 'max' )[['Close']] , 3 )
    tickerData.index = tickerData.index.tz_convert(tz='Asia/bangkok')
    filter_date = '2023-01-01 12:00:00+07:00'
    tickerData = tickerData[tickerData.index >= filter_date]

    fx = client_2.get_field_last(field='{}'.format(field))
    fx_js = int(json.loads(fx)["field{}".format(field)])
    rng = np.random.default_rng(fx_js)
    data = rng.integers(2, size = len(tickerData))
    tickerData['action'] = data
    tickerData['index'] = [ i+1 for i in range(len(tickerData))]

    tickerData_1 = pd.DataFrame(columns=(tickerData.columns))
    tickerData_1['action'] =  [ i for i in range(5)]
    tickerData_1.index = ['+0' , "+1" , "+2" , "+3" , "+4"]
    df = pd.concat([tickerData , tickerData_1], axis=0).fillna("")
    rng = np.random.default_rng(fx_js)
    df['action'] = rng.integers(2, size = len(df))
    return df.tail(7) , fx_js

df_7   , fx_js    = Monitor(Ticker = 'FFWM', field = 2    )
df_7_1 , fx_js_1  = Monitor(Ticker = 'NEGG', field = 3    )
df_7_2 , fx_js_2  = Monitor(Ticker = 'RIVN', field = 4 )
df_7_3 , fx_js_3  = Monitor(Ticker = 'APLS', field = 5 )
df_7_4 , fx_js_4  = Monitor(Ticker = 'NVTS', field = 6 )
df_7_5 , fx_js_5  = Monitor(Ticker = 'QXO', field = 7 )

nex = 0
Nex_day_sell = 0
toggle = lambda x : 1 - x

Nex_day_ = st.checkbox('nex_day')
if Nex_day_ :
    st.write( "value = " , nex)
    nex_col , Nex_day_sell_col ,_,_,_  = st.columns(5)

    if nex_col.button("Nex_day"):
        nex = 1
        st.write( "value = " , nex)

    if Nex_day_sell_col.button("Nex_day_sell"):
        nex = 1
        Nex_day_sell = 1
        st.write( "value = " , nex)
        st.write( "Nex_day_sell = " , Nex_day_sell)

st.write("_____")

col13, col16, col14, col15, col17, col18, col19, col20 = st.columns(8)

x_2 = col16.number_input('Diff', step=1 , value= 60   )

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
        _NEGG_ASSET = col13.button("GO! ")
        if _NEGG_ASSET :
            client.update(  {'field2': add_2 }  )
            col13.write(add_2)

    thingspeak_3 = col13.checkbox('@_RIVN_ASSET')
    if thingspeak_3 :
        add_3 = col13.number_input('@_RIVN_ASSET', step=0.001 ,  value=0.)
        _RIVN_ASSET = col13.button("GO!  ")
        if _RIVN_ASSET :
            client.update(  {'field3': add_3 }  )
            col13.write(add_3)

    thingspeak_4 = col13.checkbox('@_APLS_ASSET')
    if thingspeak_4 :
        add_4 = col13.number_input('@_APLS_ASSET', step=0.001 ,  value=0.)
        _APLS_ASSET = col13.button("GO!   ")
        if _APLS_ASSET :
            client.update(  {'field4': add_4 }  )
            col13.write(add_4)

    thingspeak_5 = col13.checkbox('@_NVTS_ASSET')
    if thingspeak_5:
        add_5 = col13.number_input('@_NVTS_ASSET', step=0.001, value= 0.)
        _NVTS_ASSET = col13.button("GO!    ")
        if _NVTS_ASSET:
            client.update({'field5': add_5})
            col13.write(add_5)

    thingspeak_6 = col13.checkbox('@_QXO_ASSET')
    if thingspeak_6:
        add_6 = col13.number_input('@_QXO_ASSET', step=0.001, value=0.)
        _QXO_ASSET = col13.button("GO!     ")
        if _QXO_ASSET:
            client.update({'field6': add_6})
            col13.write(add_6)



FFWM_ASSET_LAST = client.get_field_last(field='field1')
FFWM_ASSET_LAST =  eval(json.loads(FFWM_ASSET_LAST)['field1'])

NEGG_ASSET_LAST = client.get_field_last(field='field2')
NEGG_ASSET_LAST = eval(json.loads(NEGG_ASSET_LAST)['field2'])

RIVN_ASSET_LAST = client.get_field_last(field='field3')
RIVN_ASSET_LAST = eval(json.loads(RIVN_ASSET_LAST)['field3'])

APLS_ASSET_LAST = client.get_field_last(field='field4')
APLS_ASSET_LAST = eval(json.loads(APLS_ASSET_LAST)['field4'])

NVTS_ASSET_LAST = client.get_field_last(field='field5')
NVTS_ASSET_LAST = eval(json.loads(NVTS_ASSET_LAST)['field5'])

QXO_ASSET_LAST = client.get_field_last(field='field6') ####
if 'field6' in json.loads(QXO_ASSET_LAST):
    QXO_ASSET_LAST = eval(json.loads(QXO_ASSET_LAST)['field6'])
else:
    QXO_ASSET_LAST = 0


x_3 = col14.number_input('NEGG_ASSET', step=0.001 ,  value= NEGG_ASSET_LAST )
x_4 = col15.number_input('FFWM_ASSET', step=0.001  , value= FFWM_ASSET_LAST    )
x_5 = col17.number_input('RIVN_ASSET', step=0.001  , value= RIVN_ASSET_LAST    )
x_6 = col18.number_input('APLS_ASSET', step=0.001  , value= APLS_ASSET_LAST    )
x_7 = col19.number_input('NVTS_ASSET', step=0.001  , value= NVTS_ASSET_LAST )

QXO_OPTION = 79.
QXO_REAL   =  col20.number_input('QXO_ASSET (LV:79@19.0)', step=0.001  , value=  QXO_ASSET_LAST)
x_8 =  QXO_OPTION  + QXO_REAL

st.write("_____")

# try:

s8 , s9 , s10 =  sell( asset = x_3 , Diff= x_2)
s11 , s12 , s13 =  sell(asset = x_4 , Diff= x_2)
b8 , b9 , b10 =  buy(asset = x_3 , Diff= x_2)
b11 , b12 , b13 =  buy(asset = x_4 , Diff= x_2)
u1 , u2 , u3 = sell( asset = x_5 , Diff= x_2)
u4 , u5 , u6 = buy( asset = x_5 , Diff= x_2)
p1 , p2 , p3 = sell( asset = x_6 , Diff= x_2)
p4 , p5 , p6 = buy( asset = x_6 , Diff= x_2)
u7 , u8 , u9 = sell( asset = x_7 , Diff= x_2)
p7 , p8 , p9 = buy( asset = x_7 , Diff= x_2)
q1, q2, q3 = sell(asset=x_8, Diff=x_2)
q4, q5, q6 = buy(asset=x_8, Diff=x_2)


Limut_Order_NEGG = st.checkbox('Limut_Order_NEGG', value =  np.where(  Nex_day_sell == 1 ,  toggle(  df_7_1.action.values[1+nex] )   ,
