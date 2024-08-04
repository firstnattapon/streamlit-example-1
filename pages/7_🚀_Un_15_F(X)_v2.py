import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import thingspeak
import json
st.set_page_config(page_title="Exist_F(X)", page_icon="☀")

def delta_1(Ticker = "FFWM" ):
    try:
        tickerData = yf.Ticker(Ticker)
        entry  = tickerData.fast_info['lastPrice']  ; step = 0.01 ;  Fixed_Asset_Value = 1500. ; Cash_Balan = 650.
        # แก้ตรงentry -1
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

            # df_down =  df[:int(len(samples)/2+1)]
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


def delta6(Ticker = "FFWM" , pred = 1 ,  filter_date = '2022-12-21 12:00:00+07:00'):
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

            # df_down =  df[:int(len(samples)/2+1)]
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

            tickerData['refer_model'] = refer_model
            tickerData['pv'] =  tickerData['Cash_Balan'] + ( tickerData['Amount_Asset'] * tickerData['Close']  )
            tickerData['refer_pv'] = tickerData['refer_model'] + Fixed_Asset_Value
            tickerData['net_pv'] =   tickerData['pv'] - tickerData['refer_pv']
            final = tickerData[['net_pv' , 'pred' ,  're'  , 'Cash_Balan' ,'Close' ]]
            return  final
    except:pass


def un_16 (df_pc_pe =[]):
  a_0 = pd.DataFrame()
  a_1 = pd.DataFrame()
  Max_Production  = 0
  for x in df_pc_pe :
      a_2 = delta6( x )[['re' , 'net_pv'] ]
      a_0 = pd.concat([a_0 , a_2[['re']].rename( columns={"re": "{}_re".format(x) })   ], axis = 1)
      a_1 = pd.concat([a_1 , a_2[['net_pv']].rename(columns={"net_pv": "{}_net_pv".format(x) }) ], axis = 1)
      Max_Production = Max_Production + delta_1(x)

  net_dd = []
  net = 0
  for i in  a_0.sum(axis=1 ,    numeric_only=True).values  :
      net = net+i
      net_dd.append(net)

  a_0['maxcash_dd'] =    net_dd
  a_1['cf'] = a_1.sum(axis=1 ,    numeric_only=True )
  a_x = pd.concat([a_0 , a_1], axis = 1)

  return  a_x
  
Ticker_input  = st.text_input("Ticker", ['FFWM','NEGG','RIVN','APLS'])
list_from_string = eval(Ticker_input)
data = un_16(list_from_string)
drop_df =  [ '{}_re'.format(i) for i in list_from_string ]
df_new = data.drop( drop_df , axis=1)
st.line_chart(df_new)

roll_over = []
max_dd = df_new.maxcash_dd.values
for i in range(len(max_dd)):
    try:
        roll = max_dd[:i]
        roll_min = np.min(roll)
        roll_max = 0
        data_roll =  roll_min - roll_max  
        roll_over.append(data_roll)
    except:pass
# st.line_chart(roll_over)

min_sum =  abs(np.min(roll_over))
sum =    (df_new.cf.values   / min_sum ) * 100
# st.line_chart(sum)

cf =  df_new.cf.values
# st.line_chart(cf)

df_all = pd.DataFrame(list(zip(cf,   roll_over )) , columns =['Sum.Delta',   'Max.Sum.Buffer'] )
df_all_2 = pd.DataFrame( list(zip(sum,   roll_over )) , columns = ['True_Alpha']  )
col1, col2  = st.columns(2)

col1.line_chart(df_all)
col2.line_chart(df_all_2)
