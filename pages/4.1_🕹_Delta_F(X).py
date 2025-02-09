import pandas as pd
import numpy as np
from numba import njit
import yfinance as yf
import streamlit as st
import thingspeak
import json


@njit
def calculate_optimized(action_list, price_list, fix =500):
    action_array = np.asarray(action_list)
    action_array[0] = 1
    price_array = np.asarray(price_list)
    n = len(action_array)
    refer = np.zeros(n) #
    
    # Preallocate arrays
    amount = np.zeros(n, dtype=np.float64)
    buffer = np.zeros(n, dtype=np.float64)
    cash = np.zeros(n, dtype=np.float64)
    asset_value = np.zeros(n, dtype=np.float64)
    sumusd = np.zeros(n, dtype=np.float64)
    
    # Initialize variables
    prev_amount = 0.0
    prev_cash = 0.0
    initial_price = price_array[0]
    
    for i in range(n):
        current_price = price_array[i]
        refer[i] =  fix + (- fix) * np.log(initial_price / price_array[i]) #

        
        if i == 0:
            if action_array[i] != 0:
                amount[i] = fix / current_price
                cash[i] = fix
            # else: default zeros
        else:
            if action_array[i] == 0:
                amount[i] = prev_amount
            else:
                amount[i] = fix / current_price
                buffer[i] = prev_amount * current_price - fix
                
            cash[i] = prev_cash + buffer[i]
            
        # Update tracking variables
        asset_value[i] = amount[i] * current_price
        sumusd[i] = cash[i] + asset_value[i]
        
        # Store previous values
        prev_amount = amount[i]
        prev_cash = cash[i]
    
    return buffer, sumusd, cash, asset_value, amount , refer

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
        'net': np.round( sumusd -  (refer+500) , 2)
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
    all_id.append('fx')
    #max
    all.append(Limit_fx( Ticker , act = -2 ).net )
    all_id.append('max')
    
    chart_data = pd.DataFrame(np.array(all).T , columns= np.array(all_id))
    st.line_chart(chart_data)

    st.write( Limit_fx(Ticker , act = -2 )  ) 


channel_id = 2385118
write_api_key = 'IPSG3MMMBJEB9DY8'
client = thingspeak.Channel(channel_id, write_api_key , fmt='json')

tab1, tab2, tab3, tab4, tab5 = st.tabs(["FFWM", "NEGG", "RIVN" , 'APLS', 'NVTS'])

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
