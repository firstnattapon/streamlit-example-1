import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import thingspeak
import json
st.set_page_config(page_title="Delta_F(X)", page_icon="🕹")

# @st.cache_data
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
            final = tickerData[['net_pv' , 'Close']]
            return  final
    except:pass
        
# @st.cache_data
def delta_z (Ticker = 'FFWM' , T = np.nan ) :
    up_dn = np.array([])
    for idX , v in enumerate(T)  :
        try :
            if  T[idX+1] > v :
                up_dn = np.append( up_dn , 1)
            elif T[idX+1] <  v :
                up_dn = np.append( up_dn , 0)
            elif  T[idX+1] ==  v:
                up_dn = np.append( up_dn , up_dn[-1])
        except :
            up_dn = np.append( up_dn , up_dn[-1])
    
        final_x = 0
        xl = np.array([])
        for  vv in  up_dn:
            if  vv  != final_x :
                xl = np.append( xl , 1)
                final_x = vv
            else:
                xl = np.append( xl , 0)
        
        delta_z_re = delta2(Ticker =  Ticker  , pred = xl)
    return  delta_z_re.net_pv.values

# @st.cache_data
def delta_x (Ticker = 'FFWM' , number = [68 , 36]):
    container_1 = st.container(border=True)
    for i in range(1):
        pred  = delta2(Ticker=Ticker)
        siz = len(pred)
        prd_x =  pred.net_pv.values
        z = int(prd_x[-1])
        all_m.append(prd_x)
        all_id_m.append('min')
        container_1.write("min , {}".format(z))
    
        for i in number:
            # np.random.seed(i)
            rng = np.random.default_rng(i)
            pred  = delta2(Ticker=Ticker , pred= rng.integers(2, size= siz))
            prd_y = pred.net_pv.values
            y = int(prd_y[-1])
            all_m.append(prd_y)
            all_id_m.append(i)
            container_1.write("{} , {} ".format(i , y ))
            
        for i in range(1):
            delta_q = delta_z( Ticker , pred.Close.values )
            j = int(delta_q[-1])
            all_m.append(delta_q)
            all_id_m.append('max')
            container_1.write("max , {}".format(j))
            
            Mean_min = prd_x
            Mean_max = delta_q 
            Mean_a = Mean_min + (( Mean_max - Mean_min) / 2 )
            all_m.append(Mean_a)
            all_id_m.append('mean')
            Mean_b = Mean_a * 0.9
            all_m.append(Mean_b)
            all_id_m.append('mean_0.9')            

        chart_data = pd.DataFrame(np.array(all_m).T , columns= np.array(all_id_m))
        st.line_chart(chart_data)
        st.stop()
        
# @st.cache_data
def delta_y (Ticker = 'FFWM' ):
    container = st.container(border=True)
    all = []
    all_id = []
    for i in range(1):
        pred  = delta2(Ticker=Ticker)
        siz = len(pred)
        prd_x =  pred.net_pv.values
        z = int(prd_x[-1])
        all.append(prd_x)
        all_id.append('min')
        container.write("min , {}".format(z))
        
        for i in range(2000):
            # np.random.seed(i)
            rng = np.random.default_rng(i)
            pred  = delta2(Ticker=Ticker , pred= rng.integers(2, size= siz))
            prd_y = pred.net_pv.values
            y = int(prd_y[-1])
            if  y > z :
                z = y
                all.append(prd_y)
                all_id.append(i)
                container.write("{} , {}".format(i,y))

        chart_data = pd.DataFrame(np.array(all).T , columns= np.array(all_id))
        st.line_chart(chart_data)

FFWM_Graph = st.checkbox('FFWM_Graph_F(X)')
if FFWM_Graph :
    re = st.button("Rerun_Graph")
    if re :
        delta_y('FFWM')
FFWM_Graph_M = st.checkbox('FFWM_Graph_F(X)_M')
if FFWM_Graph_M :
    number_1  = st.number_input('Insert a number{}'.format(1),step=1 , value=68  ,  key=1 )
    number_2 =  st.number_input('Insert a number{}'.format(2),step=1 , value=36   , key=2 )
    all_id_m = [] ; all_m = []
    number = [number_1 , number_2 ]
    delta_x( Ticker = 'FFWM'  , number = number)
st.write("_____") 
NEGG_Graph = st.checkbox('NEGG_Graph_F(X)')
if NEGG_Graph :
    re = st.button("Rerun_Graph")
    if re :
        delta_y('NEGG')

NEGG_Graph_M = st.checkbox('NEGG_Graph_F(X)_M')
if NEGG_Graph_M :
    number_1  = st.number_input('Insert a number{}'.format(1),step=1 , value=130     ,  key=1 )
    number_2 =  st.number_input('Insert a number{}'.format(2),step=1 , value=553    , key=2 )
    all_id_m = [] ; all_m = []
    number = [number_1 , number_2 ]
    delta_x( Ticker = 'NEGG'  , number = number)

st.write("_____") 
RIVN_Graph = st.checkbox('RIVN_Graph_F(X)')
if RIVN_Graph :
    re = st.button("Rerun_Graph")
    if re :
        delta_y('RIVN')

RIVN_Graph_M = st.checkbox('RIVN_Graph_F(X)_M')
if RIVN_Graph_M :
    number_1  = st.number_input('Insert a number{}'.format(1),step=1 , value=1075  ,  key=1 )
    number_2 =  st.number_input('Insert a number{}'.format(2),step=1 , value=144    , key=2 )
    all_id_m = [] ; all_m = []
    number = [number_1 , number_2 ]
    delta_x( Ticker = 'RIVN'  , number = number)
st.write("_____") 

#
APLS_Graph = st.checkbox('APLS_Graph_F(X)')
if APLS_Graph :
    re = st.button("Rerun_Graph")
    if re :
        delta_y('APLS')

APLS_Graph_M = st.checkbox('APLS_Graph_F(X)_M')
if APLS_Graph_M :
    number_1  = st.number_input('Insert a number{}'.format(1),step=1 , value=168  ,  key=1 )
    number_2 =  st.number_input('Insert a number{}'.format(2),step=1 , value=23    , key=2 )
    all_id_m = [] ; all_m = []
    number = [number_1 , number_2 ]
    delta_x( Ticker = 'APLS'  , number = number)
st.write("_____") 

st.stop()
