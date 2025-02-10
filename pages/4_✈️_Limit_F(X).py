import pandas as pd
import numpy as np
from numba import njit
import yfinance as yf
import streamlit as st
import thingspeak
import json

st.set_page_config(page_title="Limit_F(X)", page_icon="✈️")
@njit(fastmath=True)  # เพิ่ม fastmath=True เพื่อให้ compiler optimize มากขึ้น
def calculate_optimized(action_list, price_list, fix=1500):
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
    
    return buffer, sumusd, cash, asset_value, amount, refer



def get_max_action(prices):
    prices = np.array(prices, dtype=np.float64)
    n = len(prices)
    action = np.empty(n, dtype=np.int64)
    action[0] = 0
    
    if n > 2:
        diff = np.diff(prices) 
        action[1:-1] = np.where(diff[:-1] * diff[1:] < 0, 1, 0)
    elif n == 2:
        action[1] = -1
    action[-1] = -1

    return action


def Limit_fx (Ticker = '' , act = -1 ):
    filter_date = '2023-01-01 12:00:00+07:00'
    tickerData = yf.Ticker(Ticker)
    tickerData = tickerData.history(period= 'max' )[['Close']]
    tickerData.index = tickerData.index.tz_convert(tz='Asia/Bangkok')
    filter_date = filter_date
    tickerData = tickerData[tickerData.index >= filter_date]
    
    prices = np.array( tickerData.Close.values , dtype=np.float64)

    if  act == -1 : # min
        actions = np.array( np.ones( len(prices) ) , dtype=np.int64) 

    elif act == -2:  # max  
        actions = get_max_action(prices) 
      
    else :
        rng = np.random.default_rng(act)
        actions = rng.integers(0, 2, len(prices)) 
    
    buffer, sumusd, cash, asset_value, amount , refer = calculate_optimized( actions ,  prices)
    
    df = pd.DataFrame({
        'price': prices,
        'action': actions,
        'buffer': np.round(buffer, 2),
        'sumusd': np.round(sumusd, 2),
        'cash': np.round(cash, 2),
        'asset_value': np.round(asset_value, 2),
        'amount': np.round(amount, 2),
        'refer': np.round(refer, 2),
        'net': np.round( sumusd -  (refer+1500) , 2) # add1500
    })
    return df 


def plot (Ticker = ''   ,  act = -1 ):
    all = []
    all_id = []
    
    #min
    all.append( Limit_fx(Ticker , act = -1 ).net  )
    all_id.append('min')

    #fx
    all.append(Limit_fx( Ticker , act = act ).net )
    all_id.append('fx_{}'.format(act))

    #max
    all.append(Limit_fx( Ticker , act = -2 ).net )
    all_id.append('max')
    
    chart_data = pd.DataFrame(np.array(all).T , columns= np.array(all_id))
    
    st.write('Refer_Log')
    st.line_chart(chart_data)

    df_plot =  Limit_fx(Ticker , act = -1 )
    df_plot = df_plot[['buffer']].cumsum()
    st.write('Burn_Cash')
    st.line_chart(df_plot)
    st.write( Limit_fx(Ticker , act = -1 ) ) 


channel_id = 2385118
write_api_key = 'IPSG3MMMBJEB9DY8'
client = thingspeak.Channel(channel_id, write_api_key , fmt='json')

Ref_index_Log ,  Burn_Cash , tab1, tab2, tab3, tab4, tab5 = st.tabs([ 'Ref_index_Log' ,'Burn_Cash' ,"FFWM", "NEGG", "RIVN" , 'APLS', 'NVTS' ])


with Ref_index_Log:
    tickers = ['FFWM', 'NEGG', 'RIVN', 'APLS' , 'NVTS' ]

    def get_prices(tickers, start_date):
        df_list = []
        for ticker in tickers:
            tickerData = yf.Ticker(ticker)
            tickerHist = tickerData.history(period='max')[['Close']]
            tickerHist.index = tickerHist.index.tz_convert(tz='Asia/Bangkok')
            tickerHist = tickerHist[tickerHist.index >= start_date]
            tickerHist = tickerHist.rename(columns={'Close': ticker})
            df_list.append(tickerHist[[ticker]])
        prices_df = pd.concat(df_list, axis=1)
        return prices_df
    
    # กำหนดวันที่เริ่มต้น
    filter_date = '2023-01-01 12:00:00+07:00'
        
    # ดึงข้อมูลราคาและสร้าง DataFrame
    prices_df = get_prices(tickers, filter_date)
    
    # ดรอปแถวที่มีค่าว่าง (ถ้ามี)
    prices_df = prices_df.dropna()
    
    # คำนวณ int_st: ผลคูณของราคาปิดเริ่มต้นของแต่ละ Ticker
    int_st_list = prices_df.iloc[0][tickers]  # ราคาปิดเริ่มต้นของแต่ละ Ticker
    int_st = np.prod(int_st_list)
    
    # คำนวณ ref_log สำหรับแต่ละแถว
    def calculate_ref_log(row):
        int_end = np.prod(row[tickers])  # ผลคูณของราคาปิดในแถวปัจจุบัน
        ref_log = 15000 + (-1500 * np.log(int_st / int_end))
        return ref_log
    
    prices_df['ref_log'] = prices_df.apply(calculate_ref_log, axis=1)    
    prices_df = prices_df.reset_index()
    st.write( prices_df )
    prices_df = prices_df.ref_log.values 

    sumusd_ = {
    'sumusd_{}'.format(symbol) : Limit_fx(symbol, act=-1).sumusd for symbol in tickers}

    df_sumusd_ = pd.DataFrame(sumusd_)
    df_sumusd_['daily_sumusd'] = df_sumusd_.sum(axis=1)
    df_sumusd_['ref_log'] = prices_df
    df_sumusd_['net'] = df_sumusd_['daily_sumusd'] - df_sumusd_['ref_log']
    st.write( df_sumusd_ )
    st.line_chart(df_sumusd_)


with Burn_Cash:
    STOCK_SYMBOLS = ['FFWM', 'NEGG', 'RIVN', 'APLS' , 'NVTS' ]
    
    buffers = {
        'buffer_{}'.format(symbol) : Limit_fx(symbol, act=-1).buffer
        for symbol in STOCK_SYMBOLS}
    
    df_burn_cash = pd.DataFrame(buffers)
    # คำนวณผลรวมรายวัน
    df_burn_cash['daily_burn'] = df_burn_cash.sum(axis=1)
    # คำนวณผลรวมสะสม
    df_burn_cash['cumulative_burn'] = df_burn_cash['daily_burn'].cumsum()
    
    # 4. Visualization ด้วย Streamlit
    st.line_chart(df_burn_cash['cumulative_burn'])

    df_burn_cash = df_burn_cash.reset_index(drop=True)
    # แสดงตารางข้อมูลแบบ expandable
    with st.expander("View Raw Data"):
        st.dataframe(df_burn_cash)
        
with tab1:
    FFWM_act = client.get_field_last(field='{}'.format(2))
    FFWM_act_js = int(json.loads(FFWM_act)["field{}".format(2) ])
    plot( Ticker = 'FFWM'  , act =  FFWM_act_js  )
    
with tab2:
    NEGG_act = client.get_field_last(field='{}'.format(3))
    NEGG_act_js = int(json.loads(NEGG_act)["field{}".format(3) ])
    plot( Ticker = 'NEGG'  , act =  NEGG_act_js  )

with tab3:
    RIVN_act = client.get_field_last(field='{}'.format(4))
    RIVN_act_js = int(json.loads(RIVN_act)["field{}".format(4) ])
    plot( Ticker = 'RIVN'  , act =  RIVN_act_js  )

with tab4:
    APLS_act = client.get_field_last(field='{}'.format(5))
    APLS_act_js = int(json.loads(APLS_act)["field{}".format(5) ])
    plot( Ticker = 'APLS'  , act =  APLS_act_js  )

with tab5:
    NVTS_act = client.get_field_last(field='{}'.format(6))
    NVTS_act_js = int(json.loads(NVTS_act)["field{}".format(6) ])
    plot( Ticker = 'NVTS'  , act =  NVTS_act_js  )
