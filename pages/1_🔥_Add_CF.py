import streamlit as st
import numpy as np
import datetime
import thingspeak
import pandas as pd

st.set_page_config(page_title="Add_CF", page_icon="🔥")

channel_id = 2329127
write_api_key = 'V10DE0HKR4JKB014'
client = thingspeak.Channel(channel_id, write_api_key)

def NEGG(entry = 1.26 ,ref = 1.26  ):
    try:
        entry  = entry ; step = 0.01 ;  Fixed_Asset_Value = 1500. ; Cash_Balan = 650.
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
            df =  df [df['Asset_Price'] == np.around(ref, 2) ]['net_pv'].values
            return    df[-1]
    except:pass

def FFWM(entry = 6.88 ,ref = 6.88  ):
    try:
        entry  = entry ; step = 0.01 ;  Fixed_Asset_Value = 1500. ; Cash_Balan = 650.
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
            df =  df [df['Asset_Price'] == np.around(ref, 2) ]['net_pv'].values
            return    df[-1]
    except:pass

x_1 = st.number_input('ราคา_NEGG_1.26' , step=0.01 , value =  0.00  )
x_2 = st.number_input('ราคา_FFWM_6.88', step=0.01 ,   value = 0.00  )
st.write("_____") 

y_1 = st.number_input('FFWM_asset', step=0.01 , value = 0.00  )
y_2 = st.number_input('NEGG_asset', step=0.01 , value = 0.00  )
st.write("_____") 
j_1 = st.number_input('Portfolio_cash', step=0.01 , value = 0.00  )
st.write("_____") 
z_1 = st.number_input('Adjust', step=0.01 , value = -1000.00 )
st.write("_____") 

q_1 =  NEGG( ref = x_1 )
q_2 =  FFWM( ref = x_2 )

k_1 =  (y_1 + y_2) + j_1
k_2 = (q_1 + q_2) + z_1

st.write(x_1 , x_2 ,  y_1 , y_2 , j_1 ,z_1 , q_1 , q_2 , k_1)
st.write('ref:' , k_2) 
st.write('cf:' ,  k_1 - k_2 ) 

st.write("_____") 

Check_ADD = st.checkbox('ADD_CF ')
if Check_ADD :
    button_ADD = st.button("ADD_CF")
    if button_ADD:    
        try:
            client.update(  {'field1': p - y , 'field2':(p - y)/ 2150 } )
            st.write({'Cashflow': p - y , 'Yield':(p - y)/2150 }) 
        except:pass

st.write('https://thingspeak.com/channels/2329127')
import streamlit.components.v1 as components
components.iframe('https://thingspeak.com/channels/2329127/charts/1?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&type=line&update=15' , width=800, height=200)
components.iframe('https://thingspeak.com/channels/2329127/charts/2?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&type=line&update=15' , width=800, height=200)
