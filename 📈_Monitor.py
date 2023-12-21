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


col1, col2, col3,  col4, col5, col6   = st.columns(6)
re = col6.button("Rerun_TB")
if re :
        #1519
        tickerData = yf.Ticker( 'FFWM')
        tickerData = round(tickerData.history(period= 'max' )[['Close']] , 3 )
        tickerData.index = tickerData.index.tz_convert(tz='Asia/bangkok')
        filter_date = '2022-12-21 12:00:00+07:00'
        tickerData = tickerData[tickerData.index >= filter_date]

        fx = client.get_field_last(field='2')
        fx_js = int(json.loads(fx)["field2"])
        np.random.seed(fx_js)
        data = np.random.randint(2, size = len(tickerData))
        tickerData['action'] = data
        tickerData['index'] = [ i+1 for i in range(len(tickerData))]
        
        tickerData_1 = pd.DataFrame(columns=(tickerData.columns))
        tickerData_1['action'] = np.random.randint(2, size = len(tickerData)+5)[-5:]
        tickerData_1.index = ['+0' , "+1" , "+2" , "+3" , "+4"]
        df = pd.concat([tickerData.tail(5), tickerData_1], axis=0).fillna("")
        st.write("f(x) {}".format(fx_js))
        st.table(df.tail(7))
        st.write("***ก่อนตลาดเปิดตรวจสอบ TB ล่าสุด > RE เมื่อตลอดเปิด")
        # st.stop()

col1, col2, col3,  col4, col5, col6   = st.columns(6)
re_0 = col6.button("Rerun_RE")
if re_0 :
    tickerData = yf.Ticker( 'FFWM')
    tickerData = tickerData.history(period= 'max' )[['Close']]

    last_v = tickerData.tail(1).Close.values[0]
    final = client.get_field_last(field='1')
    final_js = float(json.loads(final)["field1"])
    col6.write(round(last_v , 3))
    col6.write(round(((1500 * (last_v / final_js)) - 1500) , 2 ))
    # st.stop()


st.stop()


