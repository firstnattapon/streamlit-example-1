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
df_7_6 , fx_js_6  = Monitor(Ticker = 'RXRX', field = 8 )


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

col13, col16, col14, col15, col17, col18, col19, col20, col21 = st.columns(9)

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
    
    thingspeak_7 = col13.checkbox('@_RXRX_ASSET')
    if thingspeak_7:
        add_7 = col13.number_input('@_RXRX_ASSET', step=0.001, value=0.)
        _RXRX_ASSET = col13.button("GO!      ")
        if _RXRX_ASSET:
            client.update({'field7': add_7})
            col13.write(add_7)


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

QXO_ASSET_LAST = client.get_field_last(field='field6')
QXO_ASSET_LAST = eval(json.loads(QXO_ASSET_LAST)['field6'])

RXRX_ASSET_LAST = client.get_field_last(field='field7')
RXRX_ASSET_LAST = eval(json.loads(RXRX_ASSET_LAST)['field7'])


x_3 = col14.number_input('NEGG_ASSET', step=0.001 ,  value= NEGG_ASSET_LAST )
x_4 = col15.number_input('FFWM_ASSET', step=0.001  , value= FFWM_ASSET_LAST    )
x_5 = col17.number_input('RIVN_ASSET', step=0.001  , value= RIVN_ASSET_LAST    )
x_6 = col18.number_input('APLS_ASSET', step=0.001  , value= APLS_ASSET_LAST    )
x_7 = col19.number_input('NVTS_ASSET', step=0.001  , value= NVTS_ASSET_LAST )

QXO_OPTION = 79.
QXO_REAL   =  col20.number_input('QXO (LV:79@19.0)', step=0.001  , value=  QXO_ASSET_LAST)
x_8 =  QXO_OPTION  + QXO_REAL

RXRX_OPTION = 278.
RXRX_REAL   =  col21.number_input('RXRX (LV:278@5.4)', step=0.001  , value=  RXRX_ASSET_LAST)
x_9 =  RXRX_OPTION  + RXRX_REAL

st.write("_____")


s8 , s9 , s10 =  sell( asset = x_3 , Diff= x_2)
s11 , s12 , s13 =  sell(asset = x_4 , Diff= x_2)
b8 , b9 , b10 =  buy(asset = x_3 , Diff= x_2)
b11 , b12 , b13 =  buy(asset = x_4 , Diff= x_2)
u1 , u2 , u3 = sell( asset = x_5 , Diff= x_2)
u4 , u5 , u6 = buy( asset = x_5 , Diff= x_2)
p1 , p2 , p3 = sell( asset = x_6 , Diff= x_2)
p4 , p5 , p6 = buy( asset = x_6 , Diff= x_2)
u7 , u8 , u9 = sell( asset = x_7 , Diff= x_2)
p7 , p8 , p9 = buy( asset = x_7 ,   Diff= x_2)
q1, q2, q3 = sell(asset=x_8, Diff=x_2)
q4, q5, q6 = buy(asset=x_8, Diff=x_2)
rx1, rx2, rx3 = sell(asset=x_9, Diff=x_2)
rx4, rx5, rx6 = buy(asset=x_9, Diff=x_2)


Limut_Order_NEGG = st.checkbox('Limut_Order_NEGG', value =  np.where(  Nex_day_sell == 1 ,  toggle(  df_7_1.action.values[1+nex] )   ,  df_7_1.action.values[1+nex]   ))
if Limut_Order_NEGG :
    st.write( 'sell' , '    ' ,'A', b9  , 'P' , b8 ,'C' ,b10    )

    col1, col2 , col3  = st.columns(3)
    sell_negg = col3.checkbox('sell_match_NEGG')
    if sell_negg :
        GO_NEGG_SELL = col3.button("GO!   ")
        if GO_NEGG_SELL :
            client.update(  {'field2': NEGG_ASSET_LAST - b9  } )
            col3.write(NEGG_ASSET_LAST - b9)

    pv_negg =  yf.Ticker('NEGG').fast_info['lastPrice'] * x_3
    st.write(yf.Ticker('NEGG').fast_info['lastPrice'] , pv_negg  ,'(',  pv_negg - 1500 ,')',  )

    col4, col5 , col6  = st.columns(3)
    st.write( 'buy' , '   ','A',  s9  ,  'P' , s8 , 'C' ,s10    )
    buy_negg = col6.checkbox('buy_match_NEGG')
    if buy_negg :
        GO_NEGG_Buy = col6.button("GO!    ")
        if GO_NEGG_Buy :
            client.update(  {'field2': NEGG_ASSET_LAST + s9  } )
            col6.write(NEGG_ASSET_LAST + s9)

st.write("_____")

Limut_Order_FFWM = st.checkbox('Limut_Order_FFWM',  value =  np.where(  Nex_day_sell == 1 ,  toggle(  df_7.action.values[1+nex] )   ,  df_7.action.values[1+nex]   ))
if Limut_Order_FFWM :
    st.write( 'sell' , '    ' , 'A', b12 , 'P' , b11  , 'C' , b13    )

    col7, col8 , col9  = st.columns(3)
    sell_ffwm = col9.checkbox('sell_match_FFWM')
    if sell_ffwm :
        GO_ffwm_sell = col9.button("GO!    ")
        if GO_ffwm_sell :
            client.update(  {'field1': FFWM_ASSET_LAST - b12  } )
            col9.write(FFWM_ASSET_LAST - b12)

    pv_ffwm =   yf.Ticker('FFWM').fast_info['lastPrice'] * x_4
    st.write(yf.Ticker('FFWM').fast_info['lastPrice'] , pv_ffwm ,'(',  pv_ffwm - 1500 ,')', )

    col10, col11 , col12  = st.columns(3)
    st.write(  'buy' , '    ', 'A', s12 , 'P' , s11  , 'C'  , s13    )
    buy_ffwm = col12.checkbox('buy_match_FFWM')
    if buy_ffwm :
        GO_ffwm_Buy = col12.button("GO!     ")
        if GO_ffwm_Buy :
            client.update(  {'field1': FFWM_ASSET_LAST + s12  } )
            col12.write(FFWM_ASSET_LAST + s12)

st.write("_____")

Limut_Order_RIVN = st.checkbox('Limut_Order_RIVN',value =  np.where(  Nex_day_sell == 1 ,  toggle(  df_7_2.action.values[1+nex] )   ,  df_7_2.action.values[1+nex]   ))
if Limut_Order_RIVN :
    st.write( 'sell' , '    ' , 'A', u5 , 'P' , u4  , 'C' , u6    )

    col77, col88 , col99  = st.columns(3)
    sell_RIVN = col99.checkbox('sell_match_RIVN')
    if sell_RIVN :
        GO_RIVN_sell = col99.button("GO!     ")
        if GO_RIVN_sell :
            client.update(  {'field3': RIVN_ASSET_LAST - u5  } )
            col99.write(RIVN_ASSET_LAST - u5)

    pv_rivn =   yf.Ticker('RIVN').fast_info['lastPrice'] * x_5
    st.write(yf.Ticker('RIVN').fast_info['lastPrice'] , pv_rivn ,'(',  pv_rivn - 1500 ,')', )

    col100 , col111 , col122  = st.columns(3)
    st.write(  'buy' , '    ', 'A', u2 , 'P' , u1  , 'C'  , u3    )
    buy_RIVN = col122.checkbox('buy_match_RIVN')
    if buy_RIVN :
        GO_RIVN_Buy = col122.button("GO!      ")
        if GO_RIVN_Buy :
            client.update(  {'field3': RIVN_ASSET_LAST + u2  } )
            col122.write(RIVN_ASSET_LAST + u2)

st.write("_____")

Limut_Order_APLS = st.checkbox('Limut_Order_APLS',value =  np.where(  Nex_day_sell == 1 ,  toggle(  df_7_3.action.values[1+nex] )   ,  df_7_3.action.values[1+nex]   ))
if Limut_Order_APLS :
    st.write( 'sell' , '    ' , 'A', p5 , 'P' , p4  , 'C' , p6    )

    col7777, col8888 , col9999  = st.columns(3)
    sell_APLS = col9999.checkbox('sell_match_APLS')
    if sell_APLS :
        GO_APLS_sell = col9999.button("GO!      ")
        if GO_APLS_sell :
            client.update(  {'field4': APLS_ASSET_LAST - p5  } )
            col9999.write(APLS_ASSET_LAST - p5)

    pv_apls =   yf.Ticker('APLS').fast_info['lastPrice'] * x_6
    st.write(yf.Ticker('APLS' ).fast_info['lastPrice'] , pv_apls ,'(',  pv_apls - 1500 ,')', )

    col1000 , col1111 , col1222  = st.columns(3)
    st.write(  'buy' , '    ', 'A', p2 , 'P' , p1  , 'C'  , p3    )
    buy_APLS = col1222.checkbox('buy_match_APLS')
    if buy_APLS :
        GO_APLS_Buy = col1222.button("GO!       ")
        if GO_APLS_Buy :
            client.update(  {'field4': APLS_ASSET_LAST + p2  } )
            col1222.write(APLS_ASSET_LAST + p2)

st.write("_____")

Limut_Order_NVTS = st.checkbox('Limut_Order_NVTS', np.where(  Nex_day_sell == 1 ,  toggle(  df_7_4.action.values[1+nex] )   ,  df_7_4.action.values[1+nex]   ))
if Limut_Order_NVTS:
    st.write('sell', '    ', 'A', p8 , 'P', p7  , 'C', p9    )

    col_nvts1, col_nvts2, col_nvts3 = st.columns(3)
    sell_NVTS = col_nvts3.checkbox('sell_match_NVTS')

    if sell_NVTS:
        GO_NVTS_sell = col_nvts3.button("GO!       ")
        if GO_NVTS_sell:
            client.update({'field5': NVTS_ASSET_LAST - p8})
            col_nvts3.write(NVTS_ASSET_LAST - p8)

    pv_nvts = yf.Ticker('NVTS').fast_info['lastPrice'] * x_7
    st.write(yf.Ticker('NVTS').fast_info['lastPrice'], pv_nvts, '(', pv_nvts - 1500  , ')') 

    col_nvts4, col_nvts5, col_nvts6 = st.columns(3)
    st.write('buy', '    ', 'A', u8, 'P', u7  , 'C',u9 )
    buy_NVTS = col_nvts6.checkbox('buy_match_NVTS')
    if buy_NVTS:
        GO_NVTS_Buy = col_nvts6.button("GO!        ")
        if GO_NVTS_Buy:
            client.update({'field5': NVTS_ASSET_LAST + u8})
            col_nvts6.write(NVTS_ASSET_LAST  + u8)
    #
    st.write("_____")
    col_nvtsm1, col_nvtsm2, col_nvtsm3 = st.columns(3)
    asset_input =  NVTS_ASSET_LAST 
    fix = 2100 ; diff = {"buy": 60,"sell": -60} ; asset = asset_input ; fx = lambda fix, diff_value, asset: (fix + diff_value) / asset  
    nvts_M = { 'sell' :  round( fx(fix, diff['buy'], asset) , 2)  , 'Price':  yf.Ticker('NVTS').fast_info['lastPrice'] ,   'buy' : round(fx(fix, diff['sell'], asset), 2)   }
    nvts_MM =  {  'sell' : np.floor( diff['buy'] / fx(fix, diff['buy'], asset))* -1  , 'ASSET_LAST': asset_input   ,  'buy' :   np.floor( diff['sell'] / fx(fix, diff['sell'], asset))* -1 }
    col_nvtsm1.write(nvts_M )
    col_nvtsm2.write(nvts_MM)
    #
st.write("_____")

Limut_Order_QXO = st.checkbox('Limut_Order_QXO', np.where(Nex_day_sell == 1, toggle(df_7_5.action.values[1+nex]), df_7_5.action.values[1+nex]))
if Limut_Order_QXO:
    st.write('sell', '    ', 'A', q5, 'P', q4, 'C', q6)

    col_qxo1, col_qxo2, col_qxo3 = st.columns(3)
    sell_QXO = col_qxo3.checkbox('sell_match_QXO')

    if sell_QXO:
        GO_QXO_sell = col_qxo3.button("GO!        ")
        if GO_QXO_sell:
            client.update({'field6': QXO_ASSET_LAST - q5})
            col_qxo3.write(QXO_ASSET_LAST - q5)

    pv_qxo = yf.Ticker('QXO').fast_info['lastPrice'] * x_8
    st.write(yf.Ticker('QXO').fast_info['lastPrice'], pv_qxo, '(', pv_qxo - 1500, ')')

    col_qxo4, col_qxo5, col_qxo6 = st.columns(3)
    st.write('buy', '    ', 'A', q2, 'P', q1, 'C', q3)
    buy_QXO = col_qxo6.checkbox('buy_match_QXO')
    if buy_QXO:
        GO_QXO_Buy = col_qxo6.button("GO!         ")
        if GO_QXO_Buy:
            client.update({'field6': QXO_ASSET_LAST + q2})
            col_qxo6.write(QXO_ASSET_LAST + q2)

st.write("_____")

Limut_Order_RXRX = st.checkbox('Limut_Order_RXRX', value = np.where( Nex_day_sell == 1 , toggle( df_7_6.action.values[1+nex] ) , df_7_6.action.values[1+nex] ))
if Limut_Order_RXRX:
    st.write('sell', '    ', 'A', rx5, 'P', rx4, 'C', rx6)

    col_rxrx1, col_rxrx2, col_rxrx3 = st.columns(3)
    sell_RXRX = col_rxrx3.checkbox('sell_match_RXRX')

    if sell_RXRX:
        GO_RXRX_sell = col_rxrx3.button("GO!         ")
        if GO_RXRX_sell:
            client.update({'field7': RXRX_ASSET_LAST - rx5})
            col_rxrx3.write(RXRX_ASSET_LAST - rx5)
    
    pv_rxrx = yf.Ticker('RXRX').fast_info['lastPrice'] * x_9
    st.write(yf.Ticker('RXRX').fast_info['lastPrice'], pv_rxrx, '(', pv_rxrx - 1500, ')')

    col_rxrx4, col_rxrx5, col_rxrx6 = st.columns(3)
    st.write('buy', '    ', 'A', rx2, 'P', rx1, 'C', rx3)
    buy_RXRX = col_rxrx6.checkbox('buy_match_RXRX')
    if buy_RXRX:
        GO_RXRX_Buy = col_rxrx6.button("GO!          ")
        if GO_RXRX_Buy:
            client.update({'field7': RXRX_ASSET_LAST + rx2})
            col_rxrx6.write(RXRX_ASSET_LAST + rx2)

st.write("_____")


if st.button("RERUN"):
    st.sidebar.rerun()
    # st.rerun()
