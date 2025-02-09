import pandas as pd
import numpy as np
from numba import njit
import yfinance as yf
import streamlit as st
import thingspeak
import json

@njit
def calculate_optimized(actions, prices, cash_start, initial_asset_value, initial_price):
    n = len(actions)
    buffers = np.zeros(n)
    cash = np.zeros(n)
    sumusd = np.zeros(n)
    refer = np.zeros(n)

    # คำนวณค่า refer ด้วยสูตรใหม่
    for i in range(n):
        refer[i] = cash_start + (-initial_asset_value) * np.log(initial_price / prices[i])

    # คำนวณค่าเริ่มต้น
    current_amount = initial_asset_value / initial_price  # ใช้ initial_price แทน prices[0]
    cash[0] = cash_start
    sumusd[0] = cash[0] + (current_amount * prices[0])

    prev_amount = current_amount
    prev_cash = cash[0]

    for i in range(1, n):
        if actions[i] == 1:
            current_amount = (prev_amount * prices[i-1]) / prices[i]
        else:
            current_amount = prev_amount

        if actions[i] != 0:
            buffers[i] = prev_amount * (prices[i] - prices[i-1])
        else:
            buffers[i] = 0.0

        cash[i] = prev_cash + buffers[i]
        sumusd[i] = cash[i] + (current_amount * prices[i])

        prev_amount = current_amount
        prev_cash = cash[i]

    net_cf   =   cash   -    refer
    # net_cf   =   sumusd   -  (refer + initial_asset_value)
    
    return buffers, cash, sumusd, refer  , net_cf
 
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
        actions = np.full(len(prices), np.nan, dtype=np.float64)
        n = np.nan
        for idx in range(len(prices) - 1):  
            if  prices[idx+1] > prices[idx]:
                actions[idx] =  0
            elif prices[idx+1] < prices[idx]:
                actions[idx] =  1
                
    else :
        rng = np.random.default_rng(act)
        actions = rng.integers(0, 2, len(prices))
    
    initial_cash = 500.0
    initial_asset_value = 500.0
    initial_price = prices[0]
    
    buffers, cash, sumusd, refer , net_cf = calculate_optimized(actions, prices, initial_cash, initial_asset_value, initial_price)
    
    df = pd.DataFrame({
        'price': prices,
        'action': actions,
        'buffer': np.round(buffers, 2),
        'sumusd': np.round(sumusd, 2),
        'cash': np.round(cash, 2),
        'refer': np.round(refer, 2),
        'net_cf': np.round(net_cf, 2)
    })
    return df 

def plot (Ticker = ''   ,  act = -1 ):
    all = []
    all_id = []
    
    #min
    all.append( Limit_fx(Ticker , act = -1 ).net_cf )
    all_id.append('min')
    #fx
    all.append(Limit_fx( Ticker , act = act ).net_cf )
    all_id.append('fx')
    #max
    all.append(Limit_fx( Ticker , act = -2 ).net_cf )
    all_id.append('max')
    
    chart_data = pd.DataFrame(np.array(all).T , columns= np.array(all_id))
    st.line_chart(chart_data)

    st.write( Limit_fx(Ticker , act = -2 )  ) 


channel_id = 2385118
write_api_key = 'IPSG3MMMBJEB9DY8'
client = thingspeak.Channel(channel_id, write_api_key , fmt='json')

FFWM_act = client.get_field_last(field='{}'.format(2))
FFWM_act_js = int(json.loads(FFWM_act)["field{}".format(2) ])
plot( Ticker = 'FFWM'  , act =  FFWM_act_js  )

NEGG_act = client.get_field_last(field='{}'.format(3))
NEGG_act_js = int(json.loads(NEGG_act)["field{}".format(3) ])
plot( Ticker = 'NEGG'  , act =  NEGG_act_js  )

RIVN_act = client.get_field_last(field='{}'.format(4))
RIVN_act_js = int(json.loads(RIVN_act)["field{}".format(4) ])
plot( Ticker = 'RIVN'  , act =  RIVN_act_js  )

APLS_act = client.get_field_last(field='{}'.format(5))
APLS_act_js = int(json.loads(APLS_act)["field{}".format(5) ])
plot( Ticker = 'APLS'  , act =  APLS_act_js  )

NVTS_act = client.get_field_last(field='{}'.format(6))
NVTS_act_js = int(json.loads(NVTS_act)["field{}".format(6) ])
plot( Ticker = 'NVTS'  , act =  NVTS_act_js  )
