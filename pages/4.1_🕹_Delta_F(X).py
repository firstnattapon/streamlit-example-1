import pandas as pd
import numpy as np
from numba import njit
import yfinance as yf
import streamlit as st
import thingspeak
import json

@njit
def calculate_optimized(actions, prices, cash_start, asset_values_start, initial_price):
    n = len(actions)
    buffers = np.zeros(n)
    cash = np.zeros(n)
    sumusd = np.zeros(n)
    refer = np.zeros(n)

    # คำนวณค่า refer ด้วยสูตรใหม่
    for i in range(n):
        refer[i] = cash_start + (-asset_values_start) * np.log(initial_price / prices[i])

    # คำนวณค่าเริ่มต้น
    current_amount = asset_values_start / initial_price  # ใช้ initial_price แทน prices[0]
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

    net_cf =  sumusd  -  (refer + 500 )
    # net_cf =  cash   -  refer

    return buffers, cash, sumusd, refer , net_cf


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
        T = prices
        # สร้าง array เก็บค่าขึ้น-ลง
        up_dn = np.array([])
        # วิเคราะห์ค่าใน array T เพื่อหาทิศทาง
        for idX, v in enumerate(T):
            try:
                next_val = T[idX+1]
                if next_val > v:
                    up_dn = np.append(up_dn, 0)  # ค่าเพิ่มขึ้น
                elif next_val < v:
                    up_dn = np.append(up_dn, 1)  # ค่าลดลง
                else:
                    up_dn = np.append(up_dn, up_dn[-1])  # ค่าเท่าเดิม -> ใช้ค่าล่าสุด
            except IndexError:
                up_dn = np.append(up_dn, up_dn[-1])  # กรณีจบ array
        
        # สร้าง array เก็บตำแหน่งที่เปลี่ยนค่า
        final_x = 0
        xl = np.array([])
        
        # ตรวจสอบการเปลี่ยนแปลงค่าใน up_dn
        for vv in up_dn:
            if vv != final_x:
                xl = np.append(xl, 1 )  # มีการเปลี่ยนแปลง
                final_x = vv
            else:
                xl = np.append(xl, 0 )  # ไม่เปลี่ยนแปลง

        actions = up_dn

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
    
    max
    all.append(Limit_fx( Ticker , act = -2 ).net_cf )
    all_id.append('max')
    
    chart_data = pd.DataFrame(np.array(all).T , columns= np.array(all_id))
    st.line_chart(chart_data)

    st.write(Limit_fx( Ticker , act = -2 ))

channel_id = 2385118
write_api_key = 'IPSG3MMMBJEB9DY8'
client = thingspeak.Channel(channel_id, write_api_key , fmt='json')

FFWM_act = client.get_field_last(field='{}'.format(2))
FFWM_act_js = int(json.loads(FFWM_act)["field{}".format(2) ])
plot( Ticker = 'FFWM'  , act =  FFWM_act_js  )

NEGG_act = client.get_field_last(field='{}'.format(3))
NEGG_act_js = int(json.loads(NEGG_act)["field{}".format(3) ])
plot( Ticker = 'NEGG'  , act =  NEGG_act_js  )
