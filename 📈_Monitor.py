import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import thingspeak
import json


st.set_page_config( page_title="Monitor", page_icon="📈")

channel_id = 2385118
write_api_key = 'IPSG3MMMBJEB9DY8'
client = thingspeak.Channel(channel_id, write_api_key , fmt='json')

# col1, col2, col3,  col4, col5, col6 ,col7  = st.columns(7)
# col4.write("FFWM")

Check_ADD = st.checkbox('Add_Last.Re.Price')
if Check_ADD :
    x = st.number_input('Updated ')
    button_ADD = st.button("Updated")
    if button_ADD:
        client.update(  {'field1': x } )
        st.write(x)


col1, col2, col3,  col4, col5, col6 ,col7  = st.columns(7)
re = col7.button("Rerun")
if re :
        tickerData = yf.Ticker( 'FFWM')
        tickerData = tickerData.history(period= 'max' ,  start='2023-12-18')[['Close']]
        tickerData = round(tickerData , 3)
        tickerData.index = tickerData.index.tz_convert(tz='Asia/bangkok')

        np.random.seed(68)
        data = np.random.randint(2, size=248+len(tickerData)) 
    
        tickerData['Action'] = data[: len(tickerData)]
        
        tickerData_1 = pd.DataFrame(columns=(tickerData.columns))
        tickerData_1['Action'] = data[len(tickerData) : len(tickerData)+3] 
        tickerData_1.index = ['+1' , "+2" , "+3"  ]
       
        df = pd.concat([tickerData, tickerData_1], axis=0).fillna("")
        col1, col2, col3,  col4, col5, col6 , col7 , col8  = st.columns(8)
        last_v = tickerData.tail(1).Close.values[0]

        final = client.get_field_last(field='1')
        final_js = float(json.loads(final)["field1"])
        col8.write(round(((1500 * (last_v / final_js)) - 1500) , 2 ))
        st.table(df)
        st.stop()

