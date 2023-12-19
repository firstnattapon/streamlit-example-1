import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd

col1, col2, col3 = st.columns(3)
col2.write("                  FFWM")

re = st.button("Rerun")
if re :
       
       # np.random.seed(1074)
       # data = np.random.randint(2, size=500)[-251:]
       data = [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1,
              0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1,
              0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0,
              1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0,
              0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0,
              0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0,
              0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1,
              0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0,
              1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0,
              1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1,
              0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1,
              1, 1, 1, 0, 1, 0, 0, 1, 1]
       
       tickerData = yf.Ticker( 'FFWM')
       tickerData = tickerData.history(period= 'max' ,  start='2023-12-19')[['Close']]
       tickerData = round(tickerData , 3)
       tickerData.index = tickerData.index.tz_convert(tz='Asia/bangkok')
       tickerData['Action'] = data[: len(tickerData)]
       
       
       tickerData_1 = pd.DataFrame(columns=(tickerData.columns))
       tickerData_1['Action'] = data[len(tickerData) : len(tickerData)+3] 
       tickerData_1.index = ['+1' , "+2" , "+3"]
       
       df = pd.concat([tickerData, tickerData_1], axis=0).fillna("")
       st.table(df)
       st.stop()
