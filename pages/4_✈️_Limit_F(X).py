# import pandas as pd
# import numpy as np
# from numba import njit
# import yfinance as yf
# import streamlit as st
# import thingspeak
# import json

# st.set_page_config(page_title="Limit_F(X)", page_icon="✈️")

# @njit(fastmath=True)  # เพิ่ม fastmath=True เพื่อให้ compiler optimize มากขึ้น
# def calculate_optimized(action_list, price_list, fix=1500):
#     # แปลงเป็น numpy array และกำหนด dtype ให้ชัดเจน
#     action_array = np.asarray(action_list, dtype=np.int32)
#     action_array[0] = 1
#     price_array = np.asarray(price_list, dtype=np.float64)
#     n = len(action_array)
    
#     # Pre-allocate arrays ด้วย dtype ที่เหมาะสม
#     amount = np.empty(n, dtype=np.float64)
#     buffer = np.zeros(n, dtype=np.float64)
#     cash = np.empty(n, dtype=np.float64)
#     asset_value = np.empty(n, dtype=np.float64)
#     sumusd = np.empty(n, dtype=np.float64)
    
#     # คำนวณค่าเริ่มต้นที่ index 0
#     initial_price = price_array[0]
#     amount[0] = fix / initial_price
#     cash[0] = fix
#     asset_value[0] = amount[0] * initial_price
#     sumusd[0] = cash[0] + asset_value[0]
    
#     # คำนวณ refer ทั้งหมดในครั้งเดียว (แยกออกมาจาก loop หลัก)
#     # เปลี่ยนเป็นสูตรที่ 2: refer = -fix * ln(t0/tn)
#     refer = -fix * np.log(initial_price / price_array)
    
#     # Main loop with minimal operations
#     for i in range(1, n):
#         curr_price = price_array[i]
#         if action_array[i] == 0:
#             amount[i] = amount[i-1]
#             buffer[i] = 0
#         else:
#             amount[i] = fix / curr_price
#             buffer[i] = amount[i-1] * curr_price - fix
        
#         cash[i] = cash[i-1] + buffer[i]
#         asset_value[i] = amount[i] * curr_price
#         sumusd[i] = cash[i] + asset_value[i]
    
#     return buffer, sumusd, cash, asset_value, amount, refer


# def get_max_action(prices):
#     prices = np.array(prices, dtype=np.float64)
#     n = len(prices)
    
#     if n < 3:
#         return np.full(n, np.nan)
    
#     action = np.full(n, np.nan, dtype=np.float64)
#     diff = np.diff(prices)
#     action[1:-1] = np.where(diff[:-1] * diff[1:] < 0, 1, 0)
    
#     return action


# def Limit_fx (Ticker = '' , act = -1 ):
#     filter_date = '2023-01-01 12:00:00+07:00'
#     tickerData = yf.Ticker(Ticker)
#     tickerData = tickerData.history(period= 'max' )[['Close']]
#     tickerData.index = tickerData.index.tz_convert(tz='Asia/Bangkok')
#     filter_date = filter_date
#     tickerData = tickerData[tickerData.index >= filter_date]
    
#     prices = np.array( tickerData.Close.values , dtype=np.float64)

#     if  act == -1 : # min
#         actions = np.array( np.ones( len(prices) ) , dtype=np.int64) 

#     elif act == -2:  # max  
#         actions = get_max_action(prices) 
      
#     else :
#         rng = np.random.default_rng(act)
#         actions = rng.integers(0, 2, len(prices)) 
    
#     buffer, sumusd, cash, asset_value, amount , refer = calculate_optimized( actions ,  prices)
    
#     # ใช้ sumusd[0] แทนค่าคงที่ 3000
#     initial_capital = sumusd[0]
    
#     df = pd.DataFrame({
#         'price': prices,
#         'action': actions,
#         'buffer': np.round(buffer, 2),
#         'sumusd': np.round(sumusd, 2),
#         'cash': np.round(cash, 2),
#         'asset_value': np.round(asset_value, 2),
#         'amount': np.round(amount, 2),
#         'refer': np.round(refer, 2),
#         'net': np.round( sumusd - refer - initial_capital , 2) # ใช้ sumusd[0] แทน 3000
#     })
#     return df 


# def plot (Ticker = ''   ,  act = -1 ):
#     all = []
#     all_id = []
    
#     #min
#     all.append( Limit_fx(Ticker , act = -1 ).net  )
#     all_id.append('min')

#     #fx
#     all.append(Limit_fx( Ticker , act = act ).net )
#     all_id.append('fx_{}'.format(act))

#     #max
#     all.append(Limit_fx( Ticker , act = -2 ).net )
#     all_id.append('max')
    
#     chart_data = pd.DataFrame(np.array(all).T , columns= np.array(all_id))
    
#     st.write('Refer_Log')
#     st.line_chart(chart_data)

#     df_plot =  Limit_fx(Ticker , act = -1 )
#     df_plot = df_plot[['buffer']].cumsum()
#     st.write('Burn_Cash')
#     st.line_chart(df_plot)
#     st.write( Limit_fx(Ticker , act = -1 ) ) 


# channel_id = 2385118
# write_api_key = 'IPSG3MMMBJEB9DY8'
# client = thingspeak.Channel(channel_id, write_api_key , fmt='json')

# tab1, tab2, tab3, tab4, tab5 , tab6 , tab7 ,  Burn_Cash  , Ref_index_Log , cf_log   = st.tabs([ "FFWM", "NEGG", "RIVN" , 'APLS', 'NVTS', 'QXO(LV)' , 'RXRX(LV)' ,  'Burn_Cash' ,  'Ref_index_Log' , 'cf_log' ])

# with Ref_index_Log:
#     tickers = ['FFWM', 'NEGG', 'RIVN', 'APLS' , 'NVTS' , 'QXO' , 'RXRX' ]
#     def get_prices(tickers, start_date):
#         df_list = []
#         for ticker in tickers:
#             tickerData = yf.Ticker(ticker)
#             tickerHist = tickerData.history(period='max')[['Close']]
#             tickerHist.index = tickerHist.index.tz_convert(tz='Asia/Bangkok')
#             tickerHist = tickerHist[tickerHist.index >= start_date]
#             tickerHist = tickerHist.rename(columns={'Close': ticker})
#             df_list.append(tickerHist[[ticker]])
#         prices_df = pd.concat(df_list, axis=1)
#         return prices_df
    
#     filter_date = '2023-01-01 12:00:00+07:00'
#     prices_df = get_prices(tickers, filter_date)
#     prices_df = prices_df.dropna()
    
#     int_st_list = prices_df.iloc[0][tickers]  # ราคาปิดเริ่มต้นของแต่ละ Ticker
#     int_st = np.prod(int_st_list)
    
#     # คำนวณ initial_capital_Ref_index_Log
#     # สมมติว่า sumusd[0] ของแต่ละหุ้นคือ 3000 (fix * 2 = 1500 * 2)
#     initial_capital_per_stock = 3000  # หรือจะใช้ Limit_fx(tickers[0], act=-1).sumusd.iloc[0] ก็ได้
#     initial_capital_Ref_index_Log = initial_capital_per_stock * len(tickers)  # 3000 * 7 = 21000
    
#     # คำนวณ ref_log สำหรับแต่ละแถว ด้วยสูตรใหม่
#     def calculate_ref_log(row):
#         int_end = np.prod(row[tickers])  # ผลคูณของราคาปิดในแถวปัจจุบัน
#         ref_log = initial_capital_Ref_index_Log + (-1500 * np.log(int_st / int_end))
#         return ref_log
    
#     prices_df['ref_log'] = prices_df.apply(calculate_ref_log, axis=1)    
#     prices_df = prices_df.reset_index()
#     prices_df = prices_df.ref_log.values 

#     sumusd_ = {'sumusd_{}'.format(symbol) : Limit_fx(symbol, act=-1).sumusd for symbol in tickers }
#     df_sumusd_ = pd.DataFrame(sumusd_)
#     df_sumusd_['daily_sumusd'] = df_sumusd_.sum(axis=1)
#     df_sumusd_['ref_log'] = prices_df
    
#     # คำนวณต้นทุนรวมจาก sumusd[0] ของแต่ละหุ้น
#     total_initial_capital = sum([Limit_fx(symbol, act=-1).sumusd.iloc[0] for symbol in tickers])
    
#     # คำนวณ net แบบใหม่: ลบต้นทุนเริ่มต้นออกเพื่อแสดงเฉพาะส่วนเกินทุน
#     net_raw = df_sumusd_['daily_sumusd'] - df_sumusd_['ref_log'] - total_initial_capital
#     net_at_index_0 = net_raw.iloc[0]  # ค่า net ที่ index 0
#     df_sumusd_['net'] = net_raw - net_at_index_0  # ปรับให้ index 0 = 0
    
#     df_sumusd_ = df_sumusd_.reset_index().set_index('index')
#     st.line_chart(df_sumusd_.net)
#     st.dataframe(df_sumusd_)


# with Burn_Cash:
#     STOCK_SYMBOLS = ['FFWM', 'NEGG', 'RIVN', 'APLS' , 'NVTS' , 'QXO' , 'RXRX' ]
    
#     buffers = {
#         'buffer_{}'.format(symbol) : Limit_fx(symbol, act=-1).buffer
#         for symbol in STOCK_SYMBOLS}
    
#     df_burn_cash = pd.DataFrame(buffers)
#     # คำนวณผลรวมรายวัน
#     df_burn_cash['daily_burn'] = df_burn_cash.sum(axis=1)
#     # คำนวณผลรวมสะสม
#     df_burn_cash['cumulative_burn'] = df_burn_cash['daily_burn'].cumsum()
    
#     # 4. Visualization ด้วย Streamlit
#     st.line_chart(df_burn_cash['cumulative_burn'])

#     df_burn_cash = df_burn_cash.reset_index(drop=True)
#     # แสดงตารางข้อมูลแบบ expandable
#     st.dataframe(df_burn_cash) 
        
# with tab1:
#     FFWM_act = client.get_field_last(field='{}'.format(2))
#     FFWM_act_js = int(json.loads(FFWM_act)["field{}".format(2) ])
#     plot( Ticker = 'FFWM'  , act =  FFWM_act_js  )
    
# with tab2:
#     NEGG_act = client.get_field_last(field='{}'.format(3))
#     NEGG_act_js = int(json.loads(NEGG_act)["field{}".format(3) ])
#     plot( Ticker = 'NEGG'  , act =  NEGG_act_js  )

# with tab3:
#     RIVN_act = client.get_field_last(field='{}'.format(4))
#     RIVN_act_js = int(json.loads(RIVN_act)["field{}".format(4) ])
#     plot( Ticker = 'RIVN'  , act =  RIVN_act_js  )

# with tab4:
#     APLS_act = client.get_field_last(field='{}'.format(5))
#     APLS_act_js = int(json.loads(APLS_act)["field{}".format(5) ])
#     plot( Ticker = 'APLS'  , act =  APLS_act_js  )

# with tab5:
#     NVTS_act = client.get_field_last(field='{}'.format(6))
#     NVTS_act_js = int(json.loads(NVTS_act)["field{}".format(6) ])
#     plot( Ticker = 'NVTS'  , act =  NVTS_act_js  )

# with tab6:
#     QXO_act = client.get_field_last(field='{}'.format(7))
#     QXO_act_js = int(json.loads(QXO_act)["field{}".format(7) ]) 
#     plot( Ticker = 'QXO'  , act =  QXO_act_js  )


# with tab7:
#     RXRX_act = client.get_field_last(field='{}'.format(8))
#     RXRX_act_js = int(json.loads(RXRX_act)["field{}".format(8) ]) 
#     plot( Ticker = 'RXRX'  , act =  RXRX_act_js  )



# import streamlit.components.v1 as components
# # @st.cache_data
# def iframe ( frame = ''):
#   src = frame
#   st.components.v1.iframe(src, width=1500 , height=800  , scrolling=0)

# with cf_log: 
#     st.write('')
#     st.write(' Rebalance   =  -fix * ln( t0 / tn )')
#     st.write(' Net Profit  =  sumusd - refer - sumusd[0] (ต้นทุนเริ่มต้น)')
#     st.write(' Ref_index_Log = initial_capital_Ref_index_Log + (-1500 * ln(int_st / int_end))')
#     st.write(' Net in Ref_index_Log = (daily_sumusd - ref_log - total_initial_capital) - net_at_index_0')
    
#     st.write("""🎯 กฎสำคัญ(Max_Limit) \n
#     P[i   ]     คือ ราคาปัจจุบัน   \n
#     [P[i-1]     คือ ราคาของอดีต (จุดก่อนหน้า)  \n
#     [P[i+1]     คือ ราคาของอนาคต (จุดถัดไป)  \n
#     Action = 1 เมื่อ   (P[i] - P[i-1])   *  (P[i+1] - P[i])  < 0  \n
#     Action = 0 เมื่อ   (P[i] - P[i-1])   *  (P[i+1] - P[i])  ≥ 0 \n
#     Action = NaN สำหรับจุดแรกและจุดสุดท้าย (คำนวณไม่ได้)""") 

#     iframe(frame = "    https://monica.im/share/chat?shareId=8ThBItxou24MSEka")    



import pandas as pd
import numpy as np
from numba import njit
import yfinance as yf
import streamlit as st
import thingspeak
import json
from concurrent.futures import ThreadPoolExecutor
import functools

st.set_page_config(page_title="Limit_F(X)", page_icon="✈️")

# Cache สำหรับเก็บข้อมูลราคา
@st.cache_data(ttl=300)  # Cache 5 นาที
def get_ticker_data(ticker, filter_date='2023-01-01 12:00:00+07:00'):
    """ดึงข้อมูลราคาและ cache ไว้"""
    tickerData = yf.Ticker(ticker)
    tickerData = tickerData.history(period='max')[['Close']]
    tickerData.index = tickerData.index.tz_convert(tz='Asia/Bangkok')
    tickerData = tickerData[tickerData.index >= filter_date]
    return np.array(tickerData.Close.values, dtype=np.float64)

@njit(fastmath=True, cache=True)  # เพิ่ม cache=True
def calculate_optimized(action_list, price_list, fix=1500):
    """คำนวณหลักด้วย numba ที่เร็วขึ้น"""
    action_array = np.asarray(action_list, dtype=np.int32)
    action_array[0] = 1
    price_array = np.asarray(price_list, dtype=np.float64)
    n = len(action_array)
    
    # Pre-allocate arrays
    amount = np.empty(n, dtype=np.float64)
    buffer = np.zeros(n, dtype=np.float64)
    cash = np.empty(n, dtype=np.float64)
    asset_value = np.empty(n, dtype=np.float64)
    sumusd = np.empty(n, dtype=np.float64)
    
    # คำนวณค่าเริ่มต้น
    initial_price = price_array[0]
    amount[0] = fix / initial_price
    cash[0] = fix
    asset_value[0] = amount[0] * initial_price
    sumusd[0] = cash[0] + asset_value[0]
    
    # คำนวณ refer ทั้งหมดในครั้งเดียว
    refer = -fix * np.log(initial_price / price_array)
    
    # Main loop ที่เร็วขึ้น
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
    
    return buffer, sumusd, cash, asset_value, amount, refer

@njit(fastmath=True, cache=True)
def get_max_action_optimized(prices):
    """คำนวณ max action ด้วย numba"""
    n = len(prices)
    if n < 3:
        return np.full(n, np.nan)
    
    action = np.full(n, np.nan, dtype=np.float64)
    
    # คำนวณ diff ทีละขั้น 
    for i in range(1, n-1):
        diff_prev = prices[i] - prices[i-1]
        diff_next = prices[i+1] - prices[i]
        action[i] = 1.0 if diff_prev * diff_next < 0 else 0.0
    
    return action

# Cache ผลลัพธ์ของ Limit_fx
@st.cache_data(ttl=300)
def Limit_fx_cached(ticker, act=-1):
    """Limit_fx ที่มี cache"""
    prices = get_ticker_data(ticker)
    
    if act == -1:  # min
        actions = np.ones(len(prices), dtype=np.int64)
    elif act == -2:  # max  
        actions = get_max_action_optimized(prices)
    else:
        rng = np.random.default_rng(act)
        actions = rng.integers(0, 2, len(prices))
    
    buffer, sumusd, cash, asset_value, amount, refer = calculate_optimized(actions, prices)
    
    initial_capital = sumusd[0]
    
    df = pd.DataFrame({
        'price': prices,
        'action': actions,
        'buffer': np.round(buffer, 2),
        'sumusd': np.round(sumusd, 2),
        'cash': np.round(cash, 2),
        'asset_value': np.round(asset_value, 2),
        'amount': np.round(amount, 2),
        'refer': np.round(refer, 2),
        'net': np.round(sumusd - refer - initial_capital, 2)
    })
    return df

def Limit_fx(ticker='', act=-1):
    """Wrapper function สำหรับความเข้ากันได้"""
    return Limit_fx_cached(ticker, act)

# ใช้ ThreadPoolExecutor สำหรับคำนวณแบบขนาน
def calculate_parallel_plots(ticker, act):
    """คำนวณ plot ทั้ง 3 แบบแบบขนาน"""
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            'min': executor.submit(Limit_fx, ticker, -1),
            'fx': executor.submit(Limit_fx, ticker, act),
            'max': executor.submit(Limit_fx, ticker, -2)
        }
        
        results = {}
        for key, future in futures.items():
            results[key] = future.result()
    
    return results

def plot(ticker='', act=-1):
    """Plot function ที่เร็วขึ้น"""
    # คำนวณแบบขนาน
    results = calculate_parallel_plots(ticker, act)
    
    # สร้าง chart data
    chart_data = pd.DataFrame({
        'min': results['min'].net,
        'fx_{}'.format(act): results['fx'].net,
        'max': results['max'].net
    })
    
    st.write('Refer_Log')
    st.line_chart(chart_data)
    
    # Burn Cash chart
    df_plot = results['min'][['buffer']].cumsum()
    st.write('Burn_Cash')
    st.line_chart(df_plot)
    st.write(results['min'])

# Cache สำหรับ ThingSpeak data
@st.cache_data(ttl=60)  # Cache 1 นาที
def get_thingspeak_data(channel_id, write_api_key, field):
    """ดึงข้อมูล ThingSpeak และ cache"""
    client = thingspeak.Channel(channel_id, write_api_key, fmt='json')
    data = client.get_field_last(field=str(field))
    return int(json.loads(data)[f"field{field}"])

# Pre-calculate ข้อมูลที่ใช้บ่อย
@st.cache_data(ttl=300)
def get_all_ticker_data():
    """ดึงข้อมูลทุก ticker พร้อมกัน"""
    tickers = ['FFWM', 'NEGG', 'RIVN', 'APLS', 'NVTS', 'QXO', 'RXRX']
    
    with ThreadPoolExecutor(max_workers=len(tickers)) as executor:
        futures = {ticker: executor.submit(get_ticker_data, ticker) for ticker in tickers}
        ticker_data = {ticker: future.result() for ticker, future in futures.items()}
    
    return ticker_data, tickers

# Main execution
channel_id = 2385118
write_api_key = 'IPSG3MMMBJEB9DY8'

tab1, tab2, tab3, tab4, tab5, tab6, tab7, Burn_Cash, Ref_index_Log, cf_log = st.tabs([
    "FFWM", "NEGG", "RIVN", 'APLS', 'NVTS', 'QXO(LV)', 'RXRX(LV)', 
    'Burn_Cash', 'Ref_index_Log', 'cf_log'
])

# Pre-load ข้อมูล ThingSpeak
thingspeak_data = {}
for i, ticker in enumerate(['FFWM', 'NEGG', 'RIVN', 'APLS', 'NVTS', 'QXO', 'RXRX'], 2):
    thingspeak_data[ticker] = get_thingspeak_data(channel_id, write_api_key, i)

with Ref_index_Log:
    ticker_data, tickers = get_all_ticker_data()
    
    # สร้าง DataFrame จากข้อมูลที่ cache แล้ว
    min_length = min(len(data) for data in ticker_data.values())
    prices_df = pd.DataFrame({
        ticker: data[:min_length] for ticker, data in ticker_data.items()
    })
    
    int_st_list = prices_df.iloc[0][tickers]
    int_st = np.prod(int_st_list)
    
    initial_capital_per_stock = 3000
    initial_capital_Ref_index_Log = initial_capital_per_stock * len(tickers)
    
    # คำนวณ ref_log แบบ vectorized
    int_end_array = np.prod(prices_df[tickers].values, axis=1)
    ref_log_array = initial_capital_Ref_index_Log + (-1500 * np.log(int_st / int_end_array))
    
    # คำนวณ sumusd แบบขนาน
    with ThreadPoolExecutor(max_workers=len(tickers)) as executor:
        sumusd_futures = {
            f'sumusd_{symbol}': executor.submit(lambda s=symbol: Limit_fx(s, act=-1).sumusd)
            for symbol in tickers
        }
        sumusd_dict = {key: future.result() for key, future in sumusd_futures.items()}
    
    df_sumusd_ = pd.DataFrame(sumusd_dict)
    df_sumusd_['daily_sumusd'] = df_sumusd_.sum(axis=1)
    df_sumusd_['ref_log'] = ref_log_array
    
    total_initial_capital = sum([Limit_fx(symbol, act=-1).sumusd.iloc[0] for symbol in tickers])
    
    net_raw = df_sumusd_['daily_sumusd'] - df_sumusd_['ref_log'] - total_initial_capital
    net_at_index_0 = net_raw.iloc[0]
    df_sumusd_['net'] = net_raw - net_at_index_0
    
    df_sumusd_ = df_sumusd_.reset_index().set_index('index')
    st.line_chart(df_sumusd_.net)
    st.dataframe(df_sumusd_)

with Burn_Cash:
    # คำนวณ buffer แบบขนาน
    with ThreadPoolExecutor(max_workers=len(tickers)) as executor:
        buffer_futures = {
            f'buffer_{symbol}': executor.submit(lambda s=symbol: Limit_fx(s, act=-1).buffer)
            for symbol in tickers
        }
        buffers = {key: future.result() for key, future in buffer_futures.items()}
    
    df_burn_cash = pd.DataFrame(buffers)
    df_burn_cash['daily_burn'] = df_burn_cash.sum(axis=1)
    df_burn_cash['cumulative_burn'] = df_burn_cash['daily_burn'].cumsum()
    
    st.line_chart(df_burn_cash['cumulative_burn'])
    df_burn_cash = df_burn_cash.reset_index(drop=True)
    st.dataframe(df_burn_cash)

# แท็บต่างๆ ใช้ข้อมูลที่ pre-load แล้ว
with tab1:
    plot(Ticker='FFWM', act=thingspeak_data['FFWM'])

with tab2:
    plot(Ticker='NEGG', act=thingspeak_data['NEGG'])

with tab3:
    plot(Ticker='RIVN', act=thingspeak_data['RIVN'])

with tab4:
    plot(Ticker='APLS', act=thingspeak_data['APLS'])

with tab5:
    plot(Ticker='NVTS', act=thingspeak_data['NVTS'])

with tab6:
    plot(Ticker='QXO', act=thingspeak_data['QXO'])

with tab7:
    plot(Ticker='RXRX', act=thingspeak_data['RXRX'])

import streamlit.components.v1 as components

def iframe(frame=''):
    src = frame
    st.components.v1.iframe(src, width=1500, height=800, scrolling=0)

with cf_log: 
    st.write('')
    st.write(' Rebalance   =  -fix * ln( t0 / tn )')
    st.write(' Net Profit  =  sumusd - refer - sumusd[0] (ต้นทุนเริ่มต้น)')
    st.write(' Ref_index_Log = initial_capital_Ref_index_Log + (-1500 * ln(int_st / int_end))')
    st.write(' Net in Ref_index_Log = (daily_sumusd - ref_log - total_initial_capital) - net_at_index_0')
    
    st.write("""🎯 กฎสำคัญ(Max_Limit) \n
    P[i   ]     คือ ราคาปัจจุบัน   \n
    [P[i-1]     คือ ราคาของอดีต (จุดก่อนหน้า)  \n
    [P[i+1]     คือ ราคาของอนาคต (จุดถัดไป)  \n
    Action = 1 เมื่อ   (P[i] - P[i-1])   *  (P[i+1] - P[i])  < 0  \n
    Action = 0 เมื่อ   (P[i] - P[i-1])   *  (P[i+1] - P[i])  ≥ 0 \n
    Action = NaN สำหรับจุดแรกและจุดสุดท้าย (คำนวณไม่ได้)""") 

    iframe(frame="https://monica.im/share/chat?shareId=8ThBItxou24MSEka")
