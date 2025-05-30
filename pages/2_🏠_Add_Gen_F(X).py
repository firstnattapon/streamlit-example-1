import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import thingspeak
import json
import time
from numba import njit
st.set_page_config(page_title="_Add_Gen_F(X)", page_icon="🏠")


@njit(fastmath=True)  # เพิ่ม fastmath=True เพื่อให้ compiler optimize มากขึ้น
def calculate_optimized(action_list, price_list, fix=500):
    # แปลงเป็น numpy array และกำหนด dtype ให้ชัดเจน
    action_array = np.asarray(action_list, dtype=np.int32)
    action_array[0] = 1
    price_array = np.asarray(price_list, dtype=np.float64)
    n = len(action_array)
    
    # Pre-allocate arrays ด้วย dtype ที่เหมาะสม
    amount = np.empty(n, dtype=np.float64)
    buffer = np.zeros(n, dtype=np.float64)
    cash = np.empty(n, dtype=np.float64)
    asset_value = np.empty(n, dtype=np.float64)
    sumusd = np.empty(n, dtype=np.float64)
    
    # คำนวณค่าเริ่มต้นที่ index 0
    initial_price = price_array[0]
    amount[0] = fix / initial_price
    cash[0] = fix
    asset_value[0] = amount[0] * initial_price
    sumusd[0] = cash[0] + asset_value[0]
    
    # คำนวณ refer ทั้งหมดในครั้งเดียว (แยกออกมาจาก loop หลัก)
    refer = fix * (1 - np.log(initial_price / price_array))
    
    # Main loop with minimal operations
    for i in range(1, n):
        curr_price = price_array[i]
        if action_array[i] == 0:
            amount[i] = amount[i-1]
            buffer[i] = 0
        else:
            amount[i] = fix / curr_price
            buffer[i] = amount[i-1] * curr_price - fix
        
        cash[i] = cash[i-1] + buffer[i]
        asset_value[i] = amount[i] * curr_price
        sumusd[i] = cash[i] + asset_value[i]

    refer =  sumusd - (refer+fix)
    return buffer, sumusd, cash, asset_value, amount, refer


def feed_data( data = "APLS"):
    Ticker = data
    filter_date = '2023-01-01 12:00:00+07:00'
    tickerData = yf.Ticker(Ticker)
    tickerData = tickerData.history(period= 'max' )[['Close']]
    tickerData.index = tickerData.index.tz_convert(tz='Asia/bangkok')
    filter_date = filter_date
    tickerData = tickerData[tickerData.index >= filter_date]
    
    prices = np.array( tickerData.Close.values , dtype=np.float64)
    actions = np.array( np.ones( len(prices) ) , dtype=np.int64)
    initial_cash = 500.0
    initial_asset_value = 500.0
    initial_price = prices[0]
    
    net_initial = 0.
    seed = 0
    for i in range(2000000):
        rng = np.random.default_rng(i)
        actions = rng.integers(0, 2, len(prices))

        _, _, _, _, _ , net_cf = calculate_optimized(actions, prices)

        if net_cf[-1] > net_initial:
            net_initial = net_cf[-1]
            seed  = i 
    return  seed

def delta2(Ticker = "FFWM" , pred = 1 ,  filter_date ='2023-01-01 12:00:00+07:00'):
    try:
        tickerData = yf.Ticker(Ticker)

        # tickerData = tickerData.history(period= '30m' ,  start='2000-01-01', end='2025-01-01')[-limit:].reset_index()[['Close']]
        # tickerData = tickerData.history(period= 'max' )[-limit:][['Close']]
        # tickerData = tickerData.history(period= '30m' ,  start='2000-01-01', end='2025-01-01')[['Close']]
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

            # df_top = df[int(len(samples)/2):]
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
            # df =  df[df['Cash_Balan'] > 0 ]

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
            # tickerData['delta'] = tickerData['Cash_Balan'] - tickerData['refer_model']
            # tickerData['P/E'] =  1 /  (tickerData['delta'] / tickerData['Production_Costs'] )
            # tickerData['y%']  =  (tickerData['delta'] / tickerData['Production_Costs'] ) * 100
            tickerData['pv'] =  tickerData['Cash_Balan'] + ( tickerData['Amount_Asset'] * tickerData['Close']  )
            tickerData['refer_pv'] = tickerData['refer_model'] + Fixed_Asset_Value
            tickerData['net_pv'] =   tickerData['pv'] - tickerData['refer_pv']  
            # final = tickerData[['delta' , 'Close' , 'pred' , 're' , 'Cash_Balan' , 'refer_model' , 'Amount_Asset' , 'pv' , 'refer_pv' , 'net_pv']]
            final = tickerData[['net_pv']]
            # final_1 = tickerData[['delta' , 'Close' , 'Production_Costs' ,'P/E' , 'y%' ]]
            return  final
    except:pass

channel_id = 2385118
write_api_key = 'IPSG3MMMBJEB9DY8'
client = thingspeak.Channel(channel_id, write_api_key , fmt='json')


def Gen_fx (Ticker =  'FFWM' ,  field = 2 ):
    container = st.container(border=True)
    fx = [0]

    progress_text = "Processing iterations. Please wait."
    my_bar = st.progress(0, text=progress_text)
    
    for j in range(1):
        pred  = delta2(Ticker=Ticker)
        siz = len(pred)
        z = int( pred.net_pv.values[-1])
        container.write("x , {}".format(z))
         
        for i in range(2000):
            rng = np.random.default_rng(i)  # <-- แก้ตรงนี้
            pred  = delta2(Ticker= Ticker , pred= rng.integers(2, size= siz) ) #เมธอด integers() ใน default_rng() ทำงานเหมือน randint() แต่มีประสิทธิภาพดีกว่า
            y = int( pred.net_pv.values[-1])
            if  y > z :
                container.write("{} , {}".format(i,y))
                z = y
                fx.append(i)
                
            percent_complete = (i + 1) / 2000 * 100
            my_bar.progress(int(percent_complete), text=progress_text)

    time.sleep(1)
    my_bar.empty()
    client.update(  {'field{}'.format(field) : fx[-1] } )



tab1, tab2, tab3, tab4, tab5 , tab6 , tab7  = st.tabs(["FFWM", "NEGG", "RIVN", "APLS", "NVTS" , 'QXO' ,  'RXRX'])

with tab1:
    # FFWM_Check_Gen = st.checkbox('FFWM_Add_Gen')
    # if FFWM_Check_Gen :
    #     re = st.button("Rerun_Gen_tab1")
    #     if re :
    #         Gen_fx (Ticker = 'FFWM' , field = 2 )
    
    FFWM_Check_Gen_M = st.checkbox('FFWM_Add_Gen_M')
    if FFWM_Check_Gen_M :    
        input = st.number_input('Insert a number{}'.format(1),step=1 ,  key=1 )
        re_ = st.button("Rerun_Gen_M")
        if re_ :
            client.update(  {'field2': input } )
            st.write(input)        
            
    FFWM_njit = st.checkbox('FFWM_njit')
    if FFWM_njit : 
        FFWM_njit_ = st.button("FFWM_njit_")
        if FFWM_njit_ :
            ix =  feed_data(data= 'FFWM')
            client.update(  {'field2': ix } )
            st.write(ix) 
    st.write("_____") 

with tab2:
    # NEGG_Check_Gen = st.checkbox('NEGG_Add_Gen')
    # if NEGG_Check_Gen :
    #     re = st.button("Rerun_Gen_tab2")
    #     if re :
    #         Gen_fx (Ticker = 'NEGG' , field = 3 )

    NEGG_Check_Gen_M = st.checkbox('NEGG_Add_Gen_M')
    if NEGG_Check_Gen_M :    
        input = st.number_input('Insert a number{}'.format(1),step=1 ,  key=1 )
        re_ = st.button("Rerun_Gen_M")
        if re_ :
            client.update(  {'field3': input } )
            st.write(input)    

    NEGG_njit = st.checkbox('NEGG_njit')
    if NEGG_njit : 
        NEGG_njit_ = st.button("NEGG_njit_")
        if NEGG_njit_ :
            ix =  feed_data(data= 'NEGG')
            client.update(  {'field3': ix } )
            st.write(ix)     
    st.write("_____") 

with tab3:
    # RIVN_Check_Gen = st.checkbox('RIVN_Add_Gen')
    # if RIVN_Check_Gen :
    #     re = st.button("Rerun_Gen_tab3")
    #     if re :
    #         Gen_fx (Ticker = 'RIVN' , field = 4)
    
    RIVN_Check_Gen_M = st.checkbox('RIVN_Add_Gen_M')
    if RIVN_Check_Gen_M :    
        input = st.number_input('Insert a number{}'.format(1),step=1 ,  key=1 )
        re_ = st.button("Rerun_Gen_M")
        if re_ :
            client.update(  {'field4': input } )
            st.write(input)    

    RIVN_njit = st.checkbox('RIVN_njit')
    if RIVN_njit : 
        RIVN_njit_ = st.button("RIVN_njit_")
        if RIVN_njit_ :
            ix =  feed_data(data= 'RIVN')
            client.update(  {'field4': ix } )
            st.write(ix)     
    st.write("_____") 
    

with tab4:
    # APLS_Check_Gen = st.checkbox('APLS_Add_Gen')
    # if APLS_Check_Gen :
    #     re = st.button("Rerun_Gen_tab4")
    #     if re :
    #         Gen_fx (Ticker = 'APLS' , field = 5)
    
    APLS_Check_Gen_M = st.checkbox('APLS_Add_Gen_M')
    if APLS_Check_Gen_M :    
        input = st.number_input('Insert a number{}'.format(1),step=1 ,  key=1 )
        re_ = st.button("Rerun_Gen_M")
        if re_ :
            client.update(  {'field5': input } )
            st.write(input)  

    APLS_njit = st.checkbox('APLS_njit')
    if APLS_njit : 
        APLS_njit_ = st.button("APLS_njit_")
        if APLS_njit_ :
            ix =  feed_data(data= 'APLS')
            client.update(  {'field5': ix } )
            st.write(ix)     
    st.write("_____") 


with tab5:
    # NVTS_Check_Gen = st.checkbox('NVTS_Add_Gen')
    # if NVTS_Check_Gen :
    #     re = st.button("Rerun_Gen_tab5")
    #     if re :
    #         Gen_fx (Ticker = 'NVTS' , field = 6)
    
    NVTS_Check_Gen_M = st.checkbox('NVTS_Add_Gen_M')
    if NVTS_Check_Gen_M :    
        input = st.number_input('Insert a number{}'.format(1),step=1 ,  key=1 )
        re_ = st.button("Rerun_Gen_M_tab5")
        if re_ :
            client.update(  {'field6': input } )
            st.write(input)  

    NVTS_njit = st.checkbox('NVTS_njit')
    if NVTS_njit : 
        NVTS_njit_ = st.button("NVTS_njit_")
        if NVTS_njit_ :
            ix =  feed_data(data= 'NVTS')
            client.update(  {'field6': ix } )
            st.write(ix)     
    st.write("_____")


with tab6:
    QXO_Check_Gen_M = st.checkbox('QXO_Add_Gen_M')
    if QXO_Check_Gen_M :    
        input = st.number_input('Insert a number{}'.format(1),step=1 ,  key=1 )
        re_ = st.button("Rerun_Gen_M_tab6")
        if re_ :
            client.update(  {'field7': input } )
            st.write(input)  

    QXO_njit = st.checkbox('QXO_njit')
    if QXO_njit : 
        QXO_njit_ = st.button("QXO_njit_")
        if QXO_njit_ :
            ix =  feed_data(data= 'QXO')
            client.update(  {'field7': ix } )
            st.write(ix)     
    st.write("_____")

with tab7:
    RXRX_Check_Gen_M = st.checkbox('RXRX_Add_Gen_M')
    if RXRX_Check_Gen_M :    
        input = st.number_input('Insert a number{}'.format(1),step=1 ,  key=1 )
        re_ = st.button("Rerun_Gen_M_tab7")
        if re_ :
            client.update(  {'field8': input } )
            st.write(input)  

    RXRX_njit = st.checkbox('RXRX_njit')
    if RXRX_njit : 
        RXRX_njit_ = st.button("RXRX_njit_")
        if RXRX_njit_ :
            ix =  feed_data(data= 'RXRX')
            client.update(  {'field8': ix } )
            st.write(ix)     
    st.write("_____")
