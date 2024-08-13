import streamlit as st
import numpy as np
import datetime
import thingspeak
import pandas as pd
import yfinance as yf
import json
import plotly.express as px

st.set_page_config(page_title="CF_Graph", page_icon="🔥")


def CF_Graph(entry = 1.26 , ref = 1.26 , Fixed_Asset_Value =1500. , Cash_Balan = 650.   ):
    try:
        entry  = entry ; step = 0.01 ;  Fixed_Asset_Value = Fixed_Asset_Value ; Cash_Balan = Cash_Balan
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
            df['net_pv'] = df['Fixed_Asset_Value'] + df['Cash_Balan']
            df_2 =  df[df['Asset_Price'] == np.around(ref, 2) ]['net_pv'].values
            return   df[['Asset_Price', 'Cash_Balan' , 'net_pv' ,'Fixed_Asset_Value']] ,  df_2[-1]
    except:pass

x_1 = st.number_input('ราคา_NEGG_1.26' , step=0.01 ,  value =  yf.Ticker('NEGG').fast_info['lastPrice']   ) 
x_2 = st.number_input('ราคา_FFWM_6.88', step=0.01  ,  value = yf.Ticker('FFWM').fast_info['lastPrice']   ) 
x_3 = st.number_input('ราคา_RIVN_10.07', step=0.01 ,   value = yf.Ticker('RIVN').fast_info['lastPrice'] ) 
x_4 = st.number_input('ราคา_APLS_39.61', step=0.01 ,   value = yf.Ticker('APLS').fast_info['lastPrice'] ) 
st.write("_____") 

df ,  df_2 = CF_Graph(entry = 6.88, ref = x_2)
as_1 =  df.set_index('Asset_Price')
as_1_py = px.line( as_1 )
as_1_py.add_vline(x= x_2  , line_width=1 , line_dash="dash")
st.plotly_chart( as_1_py ) 
st.write( 'rf:' , df_2) 

