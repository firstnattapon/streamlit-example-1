import streamlit as st
import pandas as pd
import requests
import re
from datetime import datetime, date

# ฟังก์ชันดึงราคาปิดรายวันจาก Yahoo Finance
def fetch_yahoo_history(symbol: str, start_date: str, end_date: str):
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/91.0.4472.124 Safari/537.36',
        'Referer': f'https://finance.yahoo.com/quote/{symbol}/history'
    })

    html = session.get(f'https://finance.yahoo.com/quote/{symbol}/history').text
    m = re.search(r'CrumbStore":{"crumb":"(?P<crumb>[^"]+)"}', html)
    if not m:
        raise RuntimeError("ไม่พบ crumb ใน HTML")

    crumb = m.group('crumb').replace('\\u002F', '/')

    period1 = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    period2 = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

    params = {
        'period1': period1,
        'period2': period2,
        'interval': '1d',
        'events': 'history',
        'crumb': crumb
    }
    url = f'https://query2.finance.yahoo.com/v8/finance/chart/{symbol}'
    resp = session.get(url, params=params)

    if resp.status_code != 200:
        st.error(f"Error: Received status code {resp.status_code}")
        st.error(resp.text)
        return None

    try:
        data = resp.json()['chart']['result'][0]
        ts = data['timestamp']
        close = data['indicators']['quote'][0]['close']
        df = pd.DataFrame({
            'Date': [datetime.fromtimestamp(t) for t in ts],
            'Close': close
        }).set_index('Date')
        return df
    except Exception as e:
        st.error("ไม่สามารถแปลงข้อมูล JSON ได้")
        st.error(str(e))
        return None

# ส่วนของ Streamlit UI
st.set_page_config(page_title="Yahoo Finance Price Fetcher", page_icon="📈")
st.title("📈 ดึงราคาปิดรายวันจาก Yahoo Finance (requests + crumb)")

symbol = st.text_input('กรอกชื่อ Symbol (เช่น SPY, FFWM, NEGG, RIVN, APLS, NVTS)', 'SPY')
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input('เริ่มวันที่', value=date(2023, 1, 1))
with col2:
    end_date = st.date_input('ถึงวันที่', value=date(2023, 12, 31))

if st.button("ดึงข้อมูล"):
    try:
        df = fetch_yahoo_history(symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        if df is not None and not df.empty:
            st.success(f"ดึงข้อมูลสำเร็จ {len(df)} แถว")
            st.dataframe(df)
            st.line_chart(df['Close'])
        else:
            st.warning("ไม่พบข้อมูลหรือไม่มีข้อมูลในช่วงเวลานี้")
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")

st.markdown("""
---
**หมายเหตุ**
- หากดึงข้อมูลไม่ได้ อาจเกิดจาก Yahoo Finance เปลี่ยนโครงสร้างเว็บหรือ block bot
- Symbol ต้องตรงกับ Yahoo Finance (เช่น SPY, AAPL, TSLA, ฯลฯ)
- สามารถนำ DataFrame ที่ได้ไปประยุกต์ใช้ต่อกับระบบของคุณได้เลย
""")







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
