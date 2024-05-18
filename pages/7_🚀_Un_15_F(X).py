import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import thingspeak
import json
st.set_page_config(page_title="Exist_F(X)", page_icon="☀")

def delta2(Ticker = "FFWM" , pred = 1 ,  filter_date = '2022-12-21 12:00:00+07:00'):
    try:
        tickerData = yf.Ticker(Ticker)
        tickerData = tickerData.history(period= 'max' )[['Close']]
        tickerData.index = tickerData.index.tz_convert(tz='Asia/bangkok')
        filter_date = filter_date
        tickerData = tickerData[tickerData.index >= filter_date]
        entry  = tickerData.Close[0] ; step = 0.01 ;  Fixed_Asset_Value = 1500. ; Cash_Balan = 650.
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
            tickerData['Close'] = np.around(tickerData['Close'].values , 2)
            tickerData['pred'] = pred
            tickerData['Fixed_Asset_Value'] = Fixed_Asset_Value
            tickerData['Amount_Asset']  =  0.
            tickerData['Amount_Asset'][0]  =  tickerData['Fixed_Asset_Value'][0] / tickerData['Close'][0]
            tickerData['re']  =  0.
            tickerData['Cash_Balan'] = Cash_Balan
            Close =   tickerData['Close'].values
            pred =  tickerData['pred'].values
            Amount_Asset =  tickerData['Amount_Asset'].values
            re = tickerData['re'].values
            Cash_Balan = tickerData['Cash_Balan'].values
            for idx, x_0 in enumerate(Amount_Asset):
                if idx != 0:
                    if pred[idx] == 0:
                        Amount_Asset[idx] = Amount_Asset[idx-1]
                    elif  pred[idx] == 1:
                        Amount_Asset[idx] =   Fixed_Asset_Value / Close[idx]
            tickerData['Amount_Asset'] = Amount_Asset
            for idx, x_1 in enumerate(re):
                if idx != 0:
                    if pred[idx] == 0:
                        re[idx] =  0
                    elif  pred[idx] == 1:
                        re[idx] =  (Amount_Asset[idx-1] * Close[idx])  - Fixed_Asset_Value
            tickerData['re'] = re
            for idx, x_2 in enumerate(Cash_Balan):
                if idx != 0:
                    Cash_Balan[idx] = Cash_Balan[idx-1] + re[idx]
            tickerData['Cash_Balan'] = Cash_Balan
            tickerData ['refer_model'] = 0.
            price = np.around(tickerData['Close'].values, 2)
            Cash  = tickerData['Cash_Balan'].values
            refer_model =  tickerData['refer_model'].values
            for idx, x_3 in enumerate(price):
                try:
                    refer_model[idx] = (df[df['Asset_Price'] == x_3]['Cash_Balan'].values[0])
                except:
                    refer_model[idx] = np.nan
            tickerData['Production_Costs'] = abs(Production_Costs)
            tickerData['refer_model'] = refer_model
            tickerData['pv'] =  tickerData['Cash_Balan'] + ( tickerData['Amount_Asset'] * tickerData['Close']  )
            tickerData['refer_pv'] = tickerData['refer_model'] + Fixed_Asset_Value
            tickerData['net_pv'] =   tickerData['pv'] - tickerData['refer_pv']
            tickerData = tickerData.reset_index()
            final = tickerData[['re' , 'net_pv']]
            return  final
    except:pass

def Un_15 (Ticker = '' , seed = 36 ):
    a_0 = pd.DataFrame()
    a_1 = pd.DataFrame()
    
    for  x in Ticker :
      np.random.seed( seed[x])
      siz = len(delta2(Ticker = x))
      a_2 = delta2( x  , pred= np.random.randint(2, size=  siz )  )[['re' , 'net_pv'] ]
      a_0 = pd.concat([a_0 , a_2[['re']].rename( columns={"re": "{}_re".format(x) })   ], axis = 1)
      a_1 = pd.concat([a_1 , a_2[['net_pv']].rename(columns={"net_pv": "{}_net_pv".format(x) }) ], axis = 1)
    
    net_dd = []
    net = 0
    for i in  a_0.sum(axis=1 ,    numeric_only=True).values  :
      net = net+i
      net_dd.append(net)
    
    a_0['Sum_Buffer'] =    net_dd
    a_1['Sum_Delta'] =     a_1.sum(axis=1 ,    numeric_only=True )

    a_3 = pd.DataFrame()
    net_dd_1 = []
    net_1 = 0
    for i in   a_0.FFWM_re.values :
        net_1 = net_1+i
        net_dd_1.append(net_1)
    a_3['FFWM_Buffer'] =    net_dd_1
    
    net_dd_2 = []
    net_2 = 0
    for i in   a_0.NEGG_re.values :
        net_2 = net_2+i
        net_dd_2.append(net_2)
    a_3['NEGG_Buffer'] =  net_dd_2

    net_dd_3 = []
    net_3 = 0
    for i in   a_0.RIVN_re.values :
        net_3 = net_3+i
        net_dd_3.append(net_3)
    a_3['RIVN_Buffer'] =  net_dd_3

    return  a_1 , a_0 , a_3

Delta , Sum_Buffer , Buffer =  Un_15(Ticker = ['FFWM' , 'NEGG' ,'RIVN'] ,seed = { 'FFWM' :36 , 'NEGG' :553 ,'RIVN':144} )

checkbox2 = st.checkbox('Delta $' , value=1 )
if checkbox2 :
    st.line_chart(Delta)

checkbox3 = st.checkbox('Buffer $' , value=1 )
if checkbox3 :
    st.line_chart(Sum_Buffer)
    st.write( Sum_Buffer)
    st.line_chart(Buffer)

checkbox1 = st.checkbox('Delta / Survival ' , value=1 )
if checkbox1 :
    Delta_2 = Delta
    Delta_2['FFWM'] =  (Delta.FFWM_net_pv.values) / (float(1500 + (abs( np.min(Buffer.FFWM_Buffer.values))+ abs( np.max(Buffer.FFWM_Buffer.values))))) *100
    Delta_2['NEGG'] =  (Delta.NEGG_net_pv.values)  / (float(1500 + (abs( np.min(Buffer.NEGG_Buffer.values))+ abs( np.max(Buffer.NEGG_Buffer.values))))) *100
    Delta_2['RIVN'] =  (Delta.RIVN_net_pv.values)  / (float(1500 + (abs( np.min(Buffer.RIVN_Buffer.values))+ abs( np.max(Buffer.RIVN_Buffer.values))))) *100
    Delta_2['Sum.Delta/Max.Sum.Buffer'] = (Delta.Sum_Delta.values) / (float(1500 + (abs( np.min(Sum_Buffer.Sum_Buffer.values))+ abs( np.max(Sum_Buffer.Sum_Buffer.values))))) *100
    Delta_2 = Delta_2[['Sum.Delta/Max.Sum.Buffer' , 'FFWM' , 'NEGG' ,'RIVN' ]]
    st.line_chart(Delta_2)

