import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import thingspeak
import json
st.set_page_config(page_title="Exist_F(X)", page_icon="☀")

def un_16 (df_pc_pe =['FFWM' , 'NEGG' , 'RIVN' ,'APLS' ]):
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
# st.line_chart(data)
