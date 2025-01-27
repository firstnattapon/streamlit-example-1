import pandas as pd
import numpy as np
from numba import njit

data = [
    # index, action, buffer, sumusd, cash, asset-value, Amount, price
    [0.00, 1, 0.00, 1000.00, 500.00, 500.00, 5.05, 99.00],
    [1.00, 1, 5.05, 1005.05, 505.05, 500.00, 5.00, 100.00],
    [2.00, 1, -30.00, 975.05, 475.05, 500.00, 5.32, 94.00],
    [3.00, 1, 26.60, 1001.65, 501.65, 500.00, 5.05, 99.00],
    [4.00, 1, -10.10, 991.55, 491.55, 500.00, 5.15, 97.00],
    [5.00, 1, -30.93, 960.62, 460.62, 500.00, 5.49, 91.00],
    [6.00, 1, 49.45, 1010.07, 510.07, 500.00, 5.00, 100.00],
    [7.00, 0, -25.00, 985.07, 485.07, 500.00, 5.26, 95.00],
    [8.00, 1, 26.32, 1011.38, 511.38, 500.00, 5.00, 100.00],
    [9.00, 1, -30.00, 981.38, 481.38, 500.00, 5.32, 94.00],
    [10.00, 1, -37.23, 944.15, 444.15, 500.00, 5.75, 87.00],
    [11.00, 1, 28.74, 972.89, 472.89, 500.00, 5.43, 92.00],
    [12.00, 1, -5.43, 967.45, 467.45, 500.00, 5.49, 91.00],
    [13.00, 1, 16.48, 983.93, 483.93, 500.00, 5.32, 94.00],
    [14.00, 1, 10.64, 994.57, 494.57, 500.00, 5.21, 96.00],
    [15.00, 1, -10.42, 984.16, 484.16, 500.00, 5.32, 94.00],
    [16.00, 1, -15.96, 968.20, 468.20, 500.00, 5.49, 91.00],
    [17.00, 0, 21.98, 990.18, 490.18, 500.00, 5.26, 95.00],
    [18.00, 1, -21.05, 969.12, 469.12, 500.00, 5.49, 91.00],
    [19.00, 1, 21.98, 991.10, 491.10, 500.00, 5.26, 95.00],
    [20.00, 1, 26.32, 1017.42, 517.42, 500.00, 5.00, 100.00],
    [21.00, 1, -15.00, 1002.42, 502.42, 500.00, 5.15, 97.00],
    [22.00, 1, -25.77, 976.64, 476.64, 500.00, 5.43, 92.00]
]

columns = ['index', 'action', 'buffer', 'sumusd', 'cash', 'asset-value', 'Amount', 'price']
df = pd.DataFrame(data, columns=columns)
actions = df['action'].values.astype(np.int64)
prices = df['price'].values.astype(np.float64)
# amount = df['Amount'].values.astype(np.float64)

@njit
def calculate_optimized(actions, prices  , cash_start ,  asset_values_start ):
    n = len(actions)
    amounts = np.zeros(n)
    buffers = np.zeros(n)
    cash = np.zeros(n)
    asset_values = np.zeros(n)
    sumusd_arr = np.zeros(n)
    
    #input
    amounts[0] = cash_start / prices[0]
    cash[0] = cash_start
    asset_values[0] = asset_values_start
    sumusd_arr[0] = cash_start + asset_values_start
    
    prev_amount = amounts[0]
    prev_cash = cash[0]
    
    for i in range(1, n):
        if actions[i] == 1:
            current_amount = (prev_amount * prices[i-1]) / prices[i]
        else:
            current_amount = prev_amount
        
        current_asset_value = current_amount * prices[i]
        
        if actions[i] != 0:
            buffer_val = prev_amount * (prices[i] - prices[i-1])
        else:
            buffer_val = 0.0
        
        current_cash = prev_cash + buffer_val
        current_sumusd = current_cash + current_asset_value
        
        amounts[i] = current_amount
        buffers[i] = buffer_val
        cash[i] = current_cash
        asset_values[i] = current_asset_value
        sumusd_arr[i] = current_sumusd
        
        prev_amount = current_amount
        prev_cash = current_cash
    
    return amounts, buffers, cash, asset_values, sumusd_arr

cash_start = 1000 ; asset_values_start = 1000
amounts, buffers, cash, asset_values, sumusd_arr = calculate_optimized( actions, prices  , cash_start , asset_values_start)

df['Amount'] = np.round(amounts, 2)
df['buffer'] = np.round(buffers, 2)
df['cash'] = np.round(cash, 2)
df['asset-value'] = np.round(asset_values, 2)
df['sumusd'] = np.round(sumusd_arr, 2)

print(df[['index', 'action', 'buffer', 'sumusd', 'cash', 'asset-value', 'Amount', 'price']])
