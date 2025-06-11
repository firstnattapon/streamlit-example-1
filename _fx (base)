import pandas as pd
import numpy as np
from numba import njit

def get_action(prices):
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



# V2
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


@njit
def compute_values_optimized_v2(action_list, price_list, fix =500):
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


@
filter_date = '2023-01-01 12:00:00+07:00'
tickerData = yf.Ticker('NVTS')
tickerData = tickerData.history(period= 'max' )[['Close']]
tickerData.index = tickerData.index.tz_convert(tz='Asia/Bangkok')
filter_date = filter_date
tickerData = tickerData[tickerData.index >= filter_date]
prices = np.array( tickerData.Close.values , dtype=np.float64)

buffer, sumusd, cash, asset_value, amount , refer  = compute_values_optimized_v2( actions ,  prices )

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
