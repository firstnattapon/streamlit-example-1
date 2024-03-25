import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import thingspeak
import json
st.set_page_config(page_title="Benchmark_F(X)", page_icon="🛰️")

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

def Un_15 (Ticker = '' ):
    a_0 = pd.DataFrame()
    a_1 = pd.DataFrame()
    
    for  x in (Ticker) :
      a_2 = delta2( x  , pred= 1  )[['re' , 'net_pv'] ]
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
    net_dd_1 = [] #1
    net_1 = 0
    for i in   a_0['{}_re'.format(Ticker[0])].values :
        net_1 = net_1+i
        net_dd_1.append(net_1)
    a_3['{}_Buffer'.format(Ticker[0])] =    net_dd_1

    net_dd_2 = [] #2
    net_2 = 0
    for i in   a_0['{}_re'.format(Ticker[1])].values :
        net_2 = net_2+i
        net_dd_2.append(net_2)
    a_3['{}_Buffer'.format(Ticker[1])] =  net_dd_2

    net_dd_3 = [] #3
    net_3 = 0
    for i in   a_0['{}_re'.format(Ticker[2])].values :
        net_3 = net_3+i
        net_dd_3.append(net_3)
    a_3['{}_Buffer'.format(Ticker[2])] =  net_dd_3

    #diff
    di = a_2
    di['dif'] = di.net_pv.diff().fillna(0.0)
    di = di.dif.values
    return  a_1 , a_0 , a_3 , di

ans = ['RIVN',
 'GME',
 'FSLR',
 'YETI',
 'ALGM',
 'FLNC',
 'SPCE',
 'ENVX',
 'COHU',
 'ASTS',
 'INNV',
 'PHAT',
 'PLRX',
 'MRSN',
 'ETNB',
 'SMMT']

number = st.number_input('Ticker_Yahoo', value=0 , step =1 , min_value=0  ) 
title = st.text_input('Ticker_Yahoo', ans[number])

# try:
Ticker_s = ['SPY' , 'QQQM' , title ]
Delta , Sum_Buffer , Buffer , diff_fx =  Un_15(Ticker = Ticker_s )

checkbox1 = st.checkbox('Delta_Benchmark_F(X) / Max.Sum_Buffer %' , value=1 )
if checkbox1 :
    Delta_2 = Delta
    Delta_2['S&P_500_ETF'] =  (Delta['{}_net_pv'.format(Ticker_s[0])].values  /  abs(np.min( Buffer['{}_Buffer'.format(Ticker_s[0])].values)) ) *100
    Delta_2['NASDAQ_100_ETF'] =  (Delta['{}_net_pv'.format(Ticker_s[1])].values  /  abs(np.min( Buffer['{}_Buffer'.format(Ticker_s[1])].values)) ) *100
    Delta_2['{}'.format(Ticker_s[2])] =  (Delta['{}_net_pv'.format(Ticker_s[2])].values  /  abs(np.min( Buffer['{}_Buffer'.format(Ticker_s[2])].values)) ) *100
    Delta_2 = Delta_2[[ 'S&P_500_ETF' , 'NASDAQ_100_ETF' , '{}'.format(Ticker_s[2]) ]]

    tickerData = yf.Ticker(title)
    tickerData = tickerData.history(period= 'max' )[['Close']]
    tickerData.index = tickerData.index.tz_convert(tz='Asia/bangkok')
    filter_date_1 = '2020-12-21 12:00:00+07:00'
    tickerData_1 = tickerData[tickerData.index >= filter_date_1]
    filter_date_2 = '2022-12-21 12:00:00+07:00'
    tickerData_2 = tickerData[tickerData.index >= filter_date_2]

    st.line_chart(Delta_2)
    st.line_chart(Delta['{}_net_pv'.format(title)])
    
    tickerData_2['Diff'] = diff_fx
    # st.scatter_chart( tickerData_2 , y= 'Close' , size= 'Diff'  )
    st.scatter_chart( y= tickerData_2.values[0]   )

    # st.line_chart(tickerData_2.Diff.values)
    st.line_chart(tickerData_1.values)


# except:pass



st.write(""" 
{ การเกิด Cycle_Market ของระบบ }

Step1 . ถ้า Intrinsic_Value_Cf  {หนี} Benchmark_Cf  และ  Delta/Zone สูง &  Vo ปกติหรือต่ำลง

_____( สะสมดูดของ , แจกจ่ายทุ่มของ ) เกิด Cycle  >  {Timing Realize}

Step2 .ถ้า Intrinsic_Value_Cf  {หนี} Benchmark_Cf  และ Delta/Zone ต่ำ  &   Vo สูง

_____เจ็บปวด , คาดหวัง , เริ่มต้นวัฏจักร Cycle > {ตลาดไม่มีประสิทธิภาพ No_Realize}

Step3 .ถ้า Intrinsic_Value_Cf  {เท่ากับ}  Benchmark_Cf และ Delta/Zone สูง  &  Vo ปกติหรือต่ำลง

_____ไม่มี Premium กับ Discount ไม่มีช่องว่างให้เล่นสินทรัพย์สะท้อนมูลค่าที่แท้จริง > {Realize}
""")
