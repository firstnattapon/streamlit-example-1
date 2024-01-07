import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import thingspeak
import json


st.set_page_config( page_title="Monitor", page_icon="📈")

if st.button("rerun"):
    st.rerun()

channel_id = 2385118
write_api_key = 'IPSG3MMMBJEB9DY8'
client = thingspeak.Channel(channel_id, write_api_key , fmt='json')

client_2 = thingspeak.Channel(2394198 , 'OVZNYQBL57GJW5JF' , fmt='json')
fx_2 = client_2.get_field_last(field='{}'.format(1))
# fx_js_2 = int(json.loads(fx_2)["field{}".format(1)])
st.write(json.loads(fx_2)["field1"])

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
    filter_date = '2022-12-21 12:00:00+07:00'
    tickerData = tickerData[tickerData.index >= filter_date]
    
    fx = client.get_field_last(field='{}'.format(field))
    fx_js = int(json.loads(fx)["field{}".format(field)])
    np.random.seed(fx_js)
    data = np.random.randint(2, size = len(tickerData))
    tickerData['action'] = data
    tickerData['index'] = [ i+1 for i in range(len(tickerData))]
    
    tickerData_1 = pd.DataFrame(columns=(tickerData.columns))
    tickerData_1['action'] =  [ i for i in range(5)]
    tickerData_1.index = ['+0' , "+1" , "+2" , "+3" , "+4"]
    df = pd.concat([tickerData , tickerData_1], axis=0).fillna("")
    np.random.seed(fx_js)
    df['action'] = np.random.randint(2, size = len(df))
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

st.write("***ก่อนตลาดเปิดตรวจสอบ TB ล่าสุด > RE เมื่อตลอดเปิด")
st.write("***RE > 60 USD")
st.stop()
