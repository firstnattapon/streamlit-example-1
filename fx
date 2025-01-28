import pandas as pd
import numpy as np
from numba import njit

@njit
def calculate_optimized(actions, prices, cash_start, asset_values_start, initial_price):
    n = len(actions)
    buffers = np.zeros(n)
    cash = np.zeros(n)
    sumusd = np.zeros(n)
    refer = np.zeros(n)

    # คำนวณค่า refer
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

    net_cf =  cash   -  refer

    return buffers, cash, sumusd, refer , net_cf


Ticker = "APLS"
filter_date = '2023-01-01 12:00:00+07:00'
tickerData = yf.Ticker(Ticker)
tickerData = tickerData.history(period= 'max' )[['Close']]
tickerData.index = tickerData.index.tz_convert(tz='Asia/bangkok')
filter_date = filter_date
tickerData = tickerData[tickerData.index >= filter_date]

prices = np.array( tickerData.Close.values , dtype=np.float64)
actions = np.array( np.ones( len(prices) ) , dtype=np.int64)
initial_cash = 500.0
initial_asset_value = 500.0
initial_price = prices[0]
