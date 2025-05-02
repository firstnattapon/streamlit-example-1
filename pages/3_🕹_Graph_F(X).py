import streamlit as st
import numpy as np
import datetime
import thingspeak
import pandas as pd
import json
from alpha_vantage.timeseries import TimeSeries
import time

st.set_page_config(page_title="Monitor", page_icon="📈")
channel_id = 2528199
write_api_key = '2E65V8XEIPH9B2VV'
client = thingspeak.Channel(channel_id, write_api_key , fmt='json')

# --------- Alpha Vantage API Key ---------
ALPHA_VANTAGE_API_KEY = 'QPYQL3VZEBDQ31RH'  # <--- ใส่ API KEY ที่นี่

# --------- ฟังก์ชันดึงข้อมูลหุ้น ---------
def get_stock_data_alpha_vantage(symbol, api_key, outputsize='full'):
    ts = TimeSeries(key=api_key, output_format='pandas')
    try:
        data, meta_data = ts.get_daily(symbol=symbol, outputsize=outputsize)
    except ValueError:
        st.error(f"ไม่พบข้อมูลหุ้น {symbol} หรือ API Key มีปัญหา")
        return pd.DataFrame()
    data = data.rename(columns={'4. close': 'Close'})
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    return data

def get_last_close(symbol, api_key):
    data = get_stock_data_alpha_vantage(symbol, api_key, outputsize='compact')
    if data.empty:
        return np.nan
    return data['Close'][-1]

def sell(asset=0, fix_c=1500, Diff=60):
    s1 = (1500 - Diff) / asset if asset != 0 else 0
    s2 = round(s1, 2)
    s3 = s2 * asset
    s4 = abs(s3 - fix_c)
    s5 = round(s4 / s2) if s2 != 0 else 0
    s6 = s5 * s2
    s7 = (asset * s2) + s6
    return s2, s5, round(s7, 2)

def buy(asset=0, fix_c=1500, Diff=60):
    b1 = (1500 + Diff) / asset if asset != 0 else 0
    b2 = round(b1, 2)
    b3 = b2 * asset
    b4 = abs(b3 - fix_c)
    b5 = round(b4 / b2) if b2 != 0 else 0
    b6 = b5 * b2
    b7 = (asset * b2) - b6
    return b2, b5, round(b7, 2)

channel_id_2 = 2385118
write_api_key_2 = 'IPSG3MMMBJEB9DY8'
client_2 = thingspeak.Channel(channel_id_2, write_api_key_2 , fmt='json' )

def Monitor(Ticker='FFWM', field=2):
    tickerData = get_stock_data_alpha_vantage(Ticker, ALPHA_VANTAGE_API_KEY)
    if tickerData.empty:
        return pd.DataFrame(), 0
    tickerData = round(tickerData[['Close']], 3)
    tickerData.index = tickerData.index.tz_localize('UTC').tz_convert('Asia/Bangkok')
    filter_date = '2023-01-01 12:00:00+07:00'
    tickerData = tickerData[tickerData.index >= filter_date]

    fx = client_2.get_field_last(field='{}'.format(field))
    fx_js = int(json.loads(fx)["field{}".format(field)])
    rng = np.random.default_rng(fx_js)
    data = rng.integers(2, size=len(tickerData))
    tickerData['action'] = data
    tickerData['index'] = [i + 1 for i in range(len(tickerData))]

    tickerData_1 = pd.DataFrame(columns=(tickerData.columns))
    tickerData_1['action'] = [i for i in range(5)]
    tickerData_1.index = ['+0', "+1", "+2", "+3", "+4"]
    df = pd.concat([tickerData, tickerData_1], axis=0).fillna("")
    rng = np.random.default_rng(fx_js)
    df['action'] = rng.integers(2, size=len(df))
    return df.tail(7), fx_js

# --------- เรียกข้อมูล Monitor ---------
df_7, fx_js = Monitor(Ticker='FFWM', field=2)
time.sleep(12)  # รอเพื่อไม่ให้เกิน limit
df_7_1, fx_js_1 = Monitor(Ticker='NEGG', field=3)
time.sleep(12)
df_7_2, fx_js_2 = Monitor(Ticker='RIVN', field=4)
time.sleep(12)
df_7_3, fx_js_3 = Monitor(Ticker='APLS', field=5)
time.sleep(12)
df_7_4, fx_js_4 = Monitor(Ticker='NVTS', field=6)

nex = 0
Nex_day_sell = 0
toggle = lambda x: 1 - x

Nex_day_ = st.checkbox('nex_day')
if Nex_day_:
    st.write("value = ", nex)
    nex_col, Nex_day_sell_col, _, _, _ = st.columns(5)

    if nex_col.button("Nex_day"):
        nex = 1
        st.write("value = ", nex)

    if Nex_day_sell_col.button("Nex_day_sell"):
        nex = 1
        Nex_day_sell = 1
        st.write("value = ", nex)
        st.write("Nex_day_sell = ", Nex_day_sell)

st.write("_____")

col13, col16, col14, col15, col17, col18, col19 = st.columns(7)

x_2 = col16.number_input('Diff', step=1, value=60)

Start = col13.checkbox('start')
if Start:
    thingspeak_1 = col13.checkbox('@_FFWM_ASSET')
    if thingspeak_1:
        add_1 = col13.number_input('@_FFWM_ASSET', step=0.001, value=0.)
        _FFWM_ASSET = col13.button("GO!")
        if _FFWM_ASSET:
            client.update({'field1': add_1})
            col13.write(add_1)

    thingspeak_2 = col13.checkbox('@_NEGG_ASSET')
    if thingspeak_2:
        add_2 = col13.number_input('@_NEGG_ASSET', step=0.001, value=0.)
        _NEGG_ASSET = col13.button("GO!")
        if _NEGG_ASSET:
            client.update({'field2': add_2})
            col13.write(add_2)

    thingspeak_3 = col13.checkbox('@_RIVN_ASSET')
    if thingspeak_3:
        add_3 = col13.number_input('@_RIVN_ASSET', step=0.001, value=0.)
        _RIVN_ASSET = col13.button("GO!")
        if _RIVN_ASSET:
            client.update({'field3': add_3})
            col13.write(add_3)

    thingspeak_4 = col13.checkbox('@_APLS_ASSET')
    if thingspeak_4:
        add_4 = col13.number_input('@_APLS_ASSET', step=0.001, value=0.)
        _APLS_ASSET = col13.button("GO!")
        if _APLS_ASSET:
            client.update({'field4': add_4})
            col13.write(add_4)

    thingspeak_5 = col13.checkbox('@_NVTS_ASSET')
    if thingspeak_5:
        add_5 = col13.number_input('@_NVTS_ASSET', step=0.001, value=0.)
        _NVTS_ASSET = col13.button("GO!")
        if _NVTS_ASSET:
            client.update({'field5': add_5})
            col13.write(add_5)

FFWM_ASSET_LAST = client.get_field_last(field='field1')
FFWM_ASSET_LAST = eval(json.loads(FFWM_ASSET_LAST)['field1'])

NEGG_ASSET_LAST = client.get_field_last(field='field2')
NEGG_ASSET_LAST = eval(json.loads(NEGG_ASSET_LAST)['field2'])

RIVN_ASSET_LAST = client.get_field_last(field='field3')
RIVN_ASSET_LAST = eval(json.loads(RIVN_ASSET_LAST)['field3'])

APLS_ASSET_LAST = client.get_field_last(field='field4')
APLS_ASSET_LAST = eval(json.loads(APLS_ASSET_LAST)['field4'])

NVTS_ASSET_LAST = client.get_field_last(field='field5')
NVTS_ASSET_LAST = eval(json.loads(NVTS_ASSET_LAST)['field5'])

x_3 = col14.number_input('NEGG_ASSET', step=0.001, value=NEGG_ASSET_LAST)
x_4 = col15.number_input('FFWM_ASSET', step=0.001, value=FFWM_ASSET_LAST)
x_5 = col17.number_input('RIVN_ASSET', step=0.001, value=RIVN_ASSET_LAST)
x_6 = col18.number_input('APLS_ASSET', step=0.001, value=APLS_ASSET_LAST)
x_7 = col19.number_input('NVTS_ASSET', step=0.001, value=NVTS_ASSET_LAST)

st.write("_____")

# ---- คำนวณซื้อขาย ----
s8, s9, s10 = sell(asset=x_3, Diff=x_2)
s11, s12, s13 = sell(asset=x_4, Diff=x_2)
b8, b9, b10 = buy(asset=x_3, Diff=x_2)
b11, b12, b13 = buy(asset=x_4, Diff=x_2)
u1, u2, u3 = sell(asset=x_5, Diff=x_2)
u4, u5, u6 = buy(asset=x_5, Diff=x_2)
p1, p2, p3 = sell(asset=x_6, Diff=x_2)
p4, p5, p6 = buy(asset=x_6, Diff=x_2)
u7, u8, u9 = sell(asset=x_7, Diff=x_2)
p7, p8, p9 = buy(asset=x_7, Diff=x_2)

# ---- ส่วนแสดงผลและอัปเดตข้อมูล ----
def show_order(symbol, df, asset_last, x, sA, sB, sC, bA, bB, bC, field, label):
    Limut_Order = st.checkbox(f'Limut_Order_{label}', value=np.where(Nex_day_sell == 1, toggle(df.action.values[1 + nex]), df.action.values[1 + nex]))
    if Limut_Order:
        st.write('sell', '   ', 'A', bB, 'P', bA, 'C', bC)
        col1, col2, col3 = st.columns(3)
        sell_match = col3.checkbox(f'sell_match_{label}')
        if sell_match:
            GO_sell = col3.button("GO!")
            if GO_sell:
                client.update({f'field{field}': asset_last - bB})
                col3.write(asset_last - bB)

        last_price = get_last_close(symbol, ALPHA_VANTAGE_API_KEY)
        pv = last_price * x if not np.isnan(last_price) else 0
        st.write(last_price, pv, '(', pv - 1500, ')')

        col4, col5, col6 = st.columns(3)
        st.write('buy', '   ', 'A', sB, 'P', sA, 'C', sC)
        buy_match = col6.checkbox(f'buy_match_{label}')
        if buy_match:
            GO_Buy = col6.button("GO!")
            if GO_Buy:
                client.update({f'field{field}': asset_last + sB})
                col6.write(asset_last + sB)
    st.write("_____")

show_order('NEGG', df_7_1, NEGG_ASSET_LAST, x_3, s8, s9, s10, b8, b9, b10, 2, 'NEGG')
show_order('FFWM', df_7, FFWM_ASSET_LAST, x_4, s11, s12, s13, b11, b12, b13, 1, 'FFWM')
show_order('RIVN', df_7_2, RIVN_ASSET_LAST, x_5, u1, u2, u3, u4, u5, u6, 3, 'RIVN')
show_order('APLS', df_7_3, APLS_ASSET_LAST, x_6, p1, p2, p3, p4, p5, p6, 4, 'APLS')
show_order('NVTS', df_7_4, NVTS_ASSET_LAST, x_7, u7, u8, u9, p7, p8, p9, 5, 'NVTS')

if st.button("RERUN"):
    st.rerun()













# import streamlit as st
# import numpy as np
# import datetime
# import thingspeak
# import pandas as pd
# import yfinance as yf
# import json
# import time
# import pytz

# from curl_cffi import requests
# session = requests.Session(impersonate="chrome")


# tickerData = yf.download('NVTS', period='1mo')
# tickerData = round(tickerData.history(period= '5y' )[['Close']] , 3 )
# tickerData.index = tickerData.index.tz_convert(tz='Asia/bangkok')
# filter_date = '2023-01-01 12:00:00+07:00'
# tickerData = tickerData[tickerData.index >= filter_date]
# st.write(tickerData) 

# yf.download(['MSFT', 'AAPL', 'GOOG'], period='1mo')


# import numpy as np
# import pandas as pd
# import yfinance as yf
# import streamlit as st
# import thingspeak
# import json
# st.set_page_config(page_title="Graph_F(X)", page_icon="🕹")
# def delta2(Ticker = "FFWM" , pred = 1 ,  filter_date = '2022-12-21 12:00:00+07:00'):
#     try:
#         tickerData = yf.Ticker(Ticker)
#         # tickerData = tickerData.history(period= '30m' ,  start='2000-01-01', end='2025-01-01')[-limit:].reset_index()[['Close']]
#         # tickerData = tickerData.history(period= 'max' )[-limit:][['Close']]
#         # tickerData = tickerData.history(period= '30m' ,  start='2000-01-01', end='2025-01-01')[['Close']]
#         tickerData = tickerData.history(period= 'max' )[['Close']]
#         tickerData.index = tickerData.index.tz_convert(tz='Asia/bangkok')
#         filter_date = filter_date
#         tickerData = tickerData[tickerData.index >= filter_date]
#         entry  = tickerData.Close[0] ; step = 0.01 ;  Fixed_Asset_Value = 1500. ; Cash_Balan = 650.
#         if entry < 10000 :
#             samples = np.arange( 0  ,  np.around(entry, 2) * 3 + step  ,  step)
#             df = pd.DataFrame()
#             df['Asset_Price'] =   np.around(samples, 2)
#             df['Fixed_Asset_Value'] = Fixed_Asset_Value
#             df['Amount_Asset']  =   df['Fixed_Asset_Value']  / df['Asset_Price']
#             # df_top = df[int(len(samples)/2):]
#             df_top = df[df.Asset_Price >= np.around(entry, 2) ]
#             df_top['Cash_Balan_top'] = (df_top['Amount_Asset'].shift(1) -  df_top['Amount_Asset']) *  df_top['Asset_Price']
#             df_top.fillna(0, inplace=True)
#             np_Cash_Balan_top = df_top['Cash_Balan_top'].values
#             xx = np.zeros(len(np_Cash_Balan_top)) ; y_0 = Cash_Balan
#             for idx, v_0  in enumerate(np_Cash_Balan_top) :
#                 z_0 = y_0 + v_0
#                 y_0 = z_0
#                 xx[idx] = y_0
#             df_top['Cash_Balan_top'] = xx
#             df_top = df_top.rename(columns={'Cash_Balan_top': 'Cash_Balan'})
#             df_top  = df_top.sort_values(by='Amount_Asset')
#             df_top  = df_top[:-1]
#             # df_down =  df[:int(len(samples)/2+1)]
#             df_down = df[df.Asset_Price <= np.around(entry, 2) ]
#             df_down['Cash_Balan_down'] = (df_down['Amount_Asset'].shift(-1) -  df_down['Amount_Asset'])     *  df_down['Asset_Price']
#             df_down.fillna(0, inplace=True)
#             df_down = df_down.sort_values(by='Asset_Price' , ascending=False)
#             np_Cash_Balan_down = df_down['Cash_Balan_down'].values
#             xxx= np.zeros(len(np_Cash_Balan_down)) ; y_1 = Cash_Balan
#             for idx, v_1  in enumerate(np_Cash_Balan_down) :
#                 z_1 = y_1 + v_1
#                 y_1 = z_1
#                 xxx[idx] = y_1
#             df_down['Cash_Balan_down'] = xxx
#             df_down = df_down.rename(columns={'Cash_Balan_down': 'Cash_Balan'})
#             df = pd.concat([df_top, df_down], axis=0)
#             Production_Costs = (df['Cash_Balan'].values[-1]) -  Cash_Balan
#             # df =  df[df['Cash_Balan'] > 0 ]
#             tickerData['Close'] = np.around(tickerData['Close'].values , 2)
#             tickerData['pred'] = pred
#             tickerData['Fixed_Asset_Value'] = Fixed_Asset_Value
#             tickerData['Amount_Asset']  =  0.
#             tickerData['Amount_Asset'][0]  =  tickerData['Fixed_Asset_Value'][0] / tickerData['Close'][0]
#             tickerData['re']  =  0.
#             tickerData['Cash_Balan'] = Cash_Balan
#             Close =   tickerData['Close'].values
#             pred =  tickerData['pred'].values
#             Amount_Asset =  tickerData['Amount_Asset'].values
#             re = tickerData['re'].values
#             Cash_Balan = tickerData['Cash_Balan'].values
#             for idx, x_0 in enumerate(Amount_Asset):
#                 if idx != 0:
#                     if pred[idx] == 0:
#                         Amount_Asset[idx] = Amount_Asset[idx-1]
#                     elif  pred[idx] == 1:
#                         Amount_Asset[idx] =   Fixed_Asset_Value / Close[idx]
#             tickerData['Amount_Asset'] = Amount_Asset
#             for idx, x_1 in enumerate(re):
#                 if idx != 0:
#                     if pred[idx] == 0:
#                         re[idx] =  0
#                     elif  pred[idx] == 1:
#                         re[idx] =  (Amount_Asset[idx-1] * Close[idx])  - Fixed_Asset_Value
#             tickerData['re'] = re
#             for idx, x_2 in enumerate(Cash_Balan):
#                 if idx != 0:
#                     Cash_Balan[idx] = Cash_Balan[idx-1] + re[idx]
#             tickerData['Cash_Balan'] = Cash_Balan
#             tickerData ['refer_model'] = 0.
#             price = np.around(tickerData['Close'].values, 2)
#             Cash  = tickerData['Cash_Balan'].values
#             refer_model =  tickerData['refer_model'].values
#             for idx, x_3 in enumerate(price):
#                 try:
#                     refer_model[idx] = (df[df['Asset_Price'] == x_3]['Cash_Balan'].values[0])
#                 except:
#                     refer_model[idx] = np.nan
#             tickerData['Production_Costs'] = abs(Production_Costs)
#             tickerData['refer_model'] = refer_model
#             # tickerData['delta'] = tickerData['Cash_Balan'] - tickerData['refer_model']
#             # tickerData['P/E'] =  1 /  (tickerData['delta'] / tickerData['Production_Costs'] )
#             # tickerData['y%']  =  (tickerData['delta'] / tickerData['Production_Costs'] ) * 100
#             tickerData['pv'] =  tickerData['Cash_Balan'] + ( tickerData['Amount_Asset'] * tickerData['Close']  )
#             tickerData['refer_pv'] = tickerData['refer_model'] + Fixed_Asset_Value
#             tickerData['net_pv'] =   tickerData['pv'] - tickerData['refer_pv']  
#             # final = tickerData[['delta' , 'Close' , 'pred' , 're' , 'Cash_Balan' , 'refer_model' , 'Amount_Asset' , 'pv' , 'refer_pv' , 'net_pv']]
#             final = tickerData[['net_pv']]
#             # final_1 = tickerData[['delta' , 'Close' , 'Production_Costs' ,'P/E' , 'y%' ]]
#             return  final
#     except:pass
# def delta_x (Ticker = 'FFWM' , number = [36 , 68]):
#     container_1 = st.container(border=True)
#     for i in range(1):
#         pred  = delta2(Ticker=Ticker)
#         siz = len(pred)
#         prd_x =  pred.net_pv.values
#         z = int(prd_x[-1])
#         all_m.append(prd_x)
#         all_id_m.append(i)
#         container_1.write("x , {}".format(z))
    
#         for i in number:
#             np.random.seed(i)
#             pred  = delta2(Ticker=Ticker , pred= np.random.randint(2, size= siz))
#             prd_y = pred.net_pv.values
#             y = int(prd_y[-1])
#             all_m.append(prd_y)
#             all_id_m.append(i)
#             container_1.write("{} , {} ".format(i , y ))
#         chart_data = pd.DataFrame(np.array(all_m).T , columns= np.array(all_id_m))
#         st.line_chart(chart_data)
#         st.stop()
# def delta_y (Ticker = 'FFWM' ):
#     container = st.container(border=True)
#     all = []
#     all_id = []
#     for i in range(1):
#         pred  = delta2(Ticker=Ticker)
#         siz = len(pred)
#         prd_x =  pred.net_pv.values
#         z = int(prd_x[-1])
#         all.append(prd_x)
#         all_id.append(i)
#         container.write("x , {}".format(z))
#         # print( 'x' ,  z )
        
#         for i in range(2000):
#             np.random.seed(i)
#             pred  = delta2(Ticker=Ticker , pred= np.random.randint(2, size= siz))
#             prd_y = pred.net_pv.values
#             y = int(prd_y[-1])
#             if  y > z :
#                 # print( i , y )
#                 z = y
#                 all.append(prd_y)
#                 all_id.append(i)
#                 container.write("{} , {}".format(i,y))
                    
#         chart_data = pd.DataFrame(np.array(all).T , columns= np.array(all_id))
#         st.line_chart(chart_data)
# FFWM_Graph = st.checkbox('FFWM_Graph_F(X)')
# if FFWM_Graph :
#     re = st.button("Rerun_Graph")
#     if re :
#         delta_y('FFWM')
# FFWM_Graph_M = st.checkbox('FFWM_Graph_F(X)_M')
# if FFWM_Graph_M :
#     number_1  = st.number_input('Insert a number{}'.format(1),step=1 , value=36  ,  key=1 )
#     number_2 =  st.number_input('Insert a number{}'.format(2),step=1 , value=68   , key=2 )
#     all_id_m = [] ; all_m = []
#     number = [number_1 , number_2 ]
#     delta_x( Ticker = 'FFWM'  , number = number)
# st.write("_____") 
# NEGG_Graph = st.checkbox('NEGG_Graph_F(X)')
# if NEGG_Graph :
#     re = st.button("Rerun_Graph")
#     if re :
#         delta_y('NEGG')

# NEGG_Graph_M = st.checkbox('NEGG_Graph_F(X)_M')
# if NEGG_Graph_M :
#     number_1  = st.number_input('Insert a number{}'.format(1),step=1 , value=130     ,  key=1 )
#     number_2 =  st.number_input('Insert a number{}'.format(2),step=1 , value=553    , key=2 )
#     all_id_m = [] ; all_m = []
#     number = [number_1 , number_2 ]
#     delta_x( Ticker = 'NEGG'  , number = number)

# st.stop()
