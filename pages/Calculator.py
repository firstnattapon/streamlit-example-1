import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import thingspeak
import json


st.set_page_config( page_title="Calculator", page_icon="⌨️")

if st.button("rerun"):
    st.rerun()
    
channel_id = 2385118
write_api_key = 'IPSG3MMMBJEB9DY8'
client = thingspeak.Channel(channel_id, write_api_key , fmt='json')

def average_cf (Ticker = 'FFWM' , field = 1 ):
    tickerData = yf.Ticker( Ticker)
    tickerData = round(tickerData.history(period= 'max' )[['Close']] , 3 )
    tickerData.index = tickerData.index.tz_convert(tz='Asia/bangkok')
    filter_date = '2024-01-01 12:00:00+07:00'
    tickerData = tickerData[tickerData.index >= filter_date]
    tickerData = len(tickerData)
    
    client_2 = thingspeak.Channel(2394198 , 'OVZNYQBL57GJW5JF' , fmt='json')
    fx_2 = client_2.get_field_last(field='{}'.format(field))
    fx_js_2 = ( int(eval(json.loads(fx_2)["field{}".format(field)])) ) -  393
    return   fx_js_2 / tickerData 
    
st.write('____')
cf_day = average_cf()
st.write( 'average_cf_day:' ,  round(cf_day , 2 ), 'USD', " : " , 'average_cf_mo:' , round(cf_day*30 , 2) ,'USD'  )
st.write('____')

def Production(Ticker = "FFWM" ):
    try:
        tickerData = yf.Ticker(Ticker)
        entry  = tickerData.fast_info['lastPrice']  ; step = 0.01 ;  Fixed_Asset_Value = 1500. ; Cash_Balan = 650.
        if entry < 10000 :
            samples = np.arange( 0  ,  np.around(entry, 2) * 3 + step  ,  step)

            df = pd.DataFrame()
            df['Asset_Price'] =   np.around(samples, 2)
            df['Fixed_Asset_Value'] = Fixed_Asset_Value
            df['Amount_Asset']  =   df['Fixed_Asset_Value']  / df['Asset_Price']

            df_top = df[df.Asset_Price >= np.around(entry, 2) ]
            df_top['Cash_Balan_top'] = (df_top['Amount_Asset'].shift(1) -  df_top['Amount_Asset']) *  df_top['Asset_Price']
            df_top.fillna(0, inplace=True)
            np_Cash_Balan_top = df_top['Cash_Balan_top'].values

            xx = np.zeros(len(np_Cash_Balan_top)) ; y_0 = Cash_Balan
            for idx, v_0  in enumerate(np_Cash_Balan_top) :
                z_0 = y_0 + v_0
                y_0 = z_0
                xx[idx] = y_0

            df_top['Cash_Balan_top'] = xx
            df_top = df_top.rename(columns={'Cash_Balan_top': 'Cash_Balan'})
            df_top  = df_top.sort_values(by='Amount_Asset')
            df_top  = df_top[:-1]

            df_down = df[df.Asset_Price <= np.around(entry, 2) ]
            df_down['Cash_Balan_down'] = (df_down['Amount_Asset'].shift(-1) -  df_down['Amount_Asset'])     *  df_down['Asset_Price']
            df_down.fillna(0, inplace=True)
            df_down = df_down.sort_values(by='Asset_Price' , ascending=False)
            np_Cash_Balan_down = df_down['Cash_Balan_down'].values

            xxx= np.zeros(len(np_Cash_Balan_down)) ; y_1 = Cash_Balan
            for idx, v_1  in enumerate(np_Cash_Balan_down) :
                z_1 = y_1 + v_1
                y_1 = z_1
                xxx[idx] = y_1

            df_down['Cash_Balan_down'] = xxx
            df_down = df_down.rename(columns={'Cash_Balan_down': 'Cash_Balan'})

            df = pd.concat([df_top, df_down], axis=0)
            Production_Costs = (df['Cash_Balan'].values[-1]) -  Cash_Balan
            return   abs(Production_Costs)
    except:pass

def Monitor (Ticker = 'FFWM' , field = 2 ):
    tickerData = yf.Ticker( Ticker)
    tickerData = round(tickerData.history(period= 'max' )[['Close']] , 3 )
    tickerData.index = tickerData.index.tz_convert(tz='Asia/bangkok')
    filter_date = '2023-01-01 12:00:00+07:00'
    tickerData = tickerData[tickerData.index >= filter_date]
    
    fx = client.get_field_last(field='{}'.format(field))
    fx_js = int(json.loads(fx)["field{}".format(field)])
    rng = np.random.default_rng(fx_js)  # <-- แก้ตรงนี้
    data = rng.integers(2, size = len(tickerData)) # <-- แก้ตรงนี้
    tickerData['action'] = data
    tickerData['index'] = [ i+1 for i in range(len(tickerData))]
    
    tickerData_1 = pd.DataFrame(columns=(tickerData.columns))
    tickerData_1['action'] =  [ i for i in range(5)]
    tickerData_1.index = ['+0' , "+1" , "+2" , "+3" , "+4"]
    df = pd.concat([tickerData , tickerData_1], axis=0).fillna("")
    rng = np.random.default_rng(fx_js)   # <-- แก้ตรงนี้
    df['action'] = rng.integers(2, size=len(df))   # <-- แก้ตรงนี้
    return df.tail(7) , fx_js

df_7 , fx_js  = Monitor(Ticker = 'FFWM', field = 2)
st.write( 'FFWM')
st.write("f(x): {}".format(fx_js) ," , " , "Production: {}".format(    np.around(Production('FFWM'), 2) ))
st.table(df_7)
st.write("_____") 

df_7_1 , fx_js_1  = Monitor(Ticker = 'NEGG', field = 3)
st.write( 'NEGG')
st.write("f(x): {}".format(fx_js_1) ," , " , "Production: {}".format(    np.around(Production('NEGG'), 2) ))
st.table(df_7_1)
st.write("_____") 

df_7_2 , fx_js_2  = Monitor(Ticker = 'RIVN', field = 4)
st.write( 'RIVN')
st.write("f(x): {}".format(fx_js_2) ," , " , "Production: {}".format(    np.around(Production('RIVN'), 2) ))
st.table(df_7_2)
st.write("_____") 

df_7_3 , fx_js_3  = Monitor(Ticker = 'APLS', field = 5)
st.write( 'APLS')
st.write("f(x): {}".format(fx_js_3) ," , " , "Production: {}".format(    np.around(Production('APLS'), 2) ))
st.table(df_7_3)
st.write("_____") 

df_7_4 , fx_js_4  = Monitor(Ticker = 'NVTS', field = 6)
st.write( 'NVTS')
st.write("f(x): {}".format(fx_js_4) ," , " , "Production: {}".format(    np.around(Production('NVTS'), 2) ))
st.table(df_7_4)
st.write("_____")

st.write("***ก่อนตลาดเปิดตรวจสอบ TB ล่าสุด > RE เมื่อตลอดเปิด")
st.write("***RE > 60 USD")
st.stop()







