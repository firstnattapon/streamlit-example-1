import pandas as pd
import numpy as np
from numba import njit
import yfinance as yf
import streamlit as st
import thingspeak
import json
  
st.set_page_config(page_title="Limit_F(X)", page_icon="✈️"  )

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
    # เปลี่ยนเป็นสูตรที่ 2: refer = -fix * ln(t0/tn)
    refer = -fix * np.log(initial_price / price_array)
    
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

def get_max_action(price_list, fix=1500):
    """
    คำนวณหาลำดับ action (0, 1) ที่ให้ผลตอบแทนสูงสุดทางทฤษฎี
    โดยใช้ Dynamic Programming ร่วมกับการย้อนรอย (Backtracking)
    """
    prices = np.asarray(price_list, dtype=np.float64)
    n = len(prices)

    if n < 2:
        return np.ones(n, dtype=int) # ถ้ามีข้อมูลไม่พอ ก็ action ไปเลย

    # --- ส่วนที่ 1: คำนวณไปข้างหน้า (เหมือนเดิม แต่เพิ่มการเก็บ path) ---
    
    dp = np.zeros(n, dtype=np.float64)
    # path[i] จะเก็บ index 'j' ของ action ครั้งก่อนหน้าที่ดีที่สุดสำหรับวัน 'i'
    path = np.zeros(n, dtype=int) 
    
    initial_capital = float(fix * 2)
    dp[0] = initial_capital
    
    for i in range(1, n):
        max_prev_sumusd = 0
        best_j = 0 # ตัวแปรสำหรับเก็บ j ที่ดีที่สุดสำหรับ i นี้
        
        for j in range(i):
            profit_from_j_to_i = fix * ((prices[i] / prices[j]) - 1)
            current_sumusd = dp[j] + profit_from_j_to_i
            
            if current_sumusd > max_prev_sumusd:
                max_prev_sumusd = current_sumusd
                best_j = j # เจอทางเลือกที่ดีกว่า ก็จำไว้ว่ามันมาจาก j ไหน
        
        dp[i] = max_prev_sumusd
        path[i] = best_j # บันทึกเส้นทางที่ดีที่สุดสำหรับวัน i

    # --- ส่วนที่ 2: การย้อนรอย (Backtracking) เพื่อสร้าง action array ---

    actions = np.zeros(n, dtype=int)
    
    # 1. หาจุดสิ้นสุดของเส้นทาง (วันที่ให้ sumusd สูงที่สุด)
    last_action_day = np.argmax(dp)
    
    # 2. เริ่มย้อนรอยจากจุดสิ้นสุดกลับไปหาจุดเริ่มต้น
    current_day = last_action_day
    while current_day > 0:
        # ทุกจุดที่เราเหยียบในการย้อนรอย คือวันที่ควรมี action
        actions[current_day] = 1
        # กระโดดกลับไปยัง action ครั้งก่อนหน้า
        current_day = path[current_day]
        
    # 3. กำหนดให้ action แรกสุดเป็น 1 เสมอ (เป็นจุดเริ่มต้นของทุกเส้นทาง)
    actions[0] = 1
    
    return actions


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
    
    # ใช้ sumusd[0] แทนค่าคงที่ 3000
    initial_capital = sumusd[0]
    
    df = pd.DataFrame({
        'price': prices,
        'action': actions,
        'buffer': np.round(buffer, 2),
        'sumusd': np.round(sumusd, 2),
        'cash': np.round(cash, 2),
        'asset_value': np.round(asset_value, 2),
        'amount': np.round(amount, 2),
        'refer': np.round(refer + initial_capital, 2),
        'net': np.round( sumusd - refer - initial_capital , 2) # ใช้ sumusd[0] แทน 3000
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

tab1, tab2, tab3, tab4, tab5 , tab6 , tab7 ,  Burn_Cash  , Ref_index_Log , cf_log   = st.tabs([ "FFWM", "NEGG", "RIVN" , 'APLS', 'NVTS', 'QXO(LV)' , 'RXRX(LV)' ,  'Burn_Cash' ,  'Ref_index_Log' , 'cf_log' ])

with Ref_index_Log:
    tickers = ['FFWM', 'NEGG', 'RIVN', 'APLS' , 'NVTS' , 'QXO' , 'RXRX' ]
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
    
    filter_date = '2023-01-01 12:00:00+07:00'
    prices_df = get_prices(tickers, filter_date)
    prices_df = prices_df.dropna()
    
    int_st_list = prices_df.iloc[0][tickers]  # ราคาปิดเริ่มต้นของแต่ละ Ticker
    int_st = np.prod(int_st_list)
    
    # คำนวณ initial_capital_Ref_index_Log
    # สมมติว่า sumusd[0] ของแต่ละหุ้นคือ 3000 (fix * 2 = 1500 * 2)
    initial_capital_per_stock = 3000  # หรือจะใช้ Limit_fx(tickers[0], act=-1).sumusd.iloc[0] ก็ได้
    initial_capital_Ref_index_Log = initial_capital_per_stock * len(tickers)  # 3000 * 7 = 21000
    
    # คำนวณ ref_log สำหรับแต่ละแถว ด้วยสูตรใหม่
    def calculate_ref_log(row):
        int_end = np.prod(row[tickers])  # ผลคูณของราคาปิดในแถวปัจจุบัน
        ref_log = initial_capital_Ref_index_Log + (-1500 * np.log(int_st / int_end))
        return ref_log
    
    prices_df['ref_log'] = prices_df.apply(calculate_ref_log, axis=1)    
    prices_df = prices_df.reset_index()
    prices_df = prices_df.ref_log.values 

    sumusd_ = {'sumusd_{}'.format(symbol) : Limit_fx(symbol, act=-1).sumusd for symbol in tickers }
    df_sumusd_ = pd.DataFrame(sumusd_)
    df_sumusd_['daily_sumusd'] = df_sumusd_.sum(axis=1)
    df_sumusd_['ref_log'] = prices_df
    
    # คำนวณต้นทุนรวมจาก sumusd[0] ของแต่ละหุ้น
    total_initial_capital = sum([Limit_fx(symbol, act=-1).sumusd.iloc[0] for symbol in tickers])
    
    # คำนวณ net แบบใหม่: ลบต้นทุนเริ่มต้นออกเพื่อแสดงเฉพาะส่วนเกินทุน
    net_raw = df_sumusd_['daily_sumusd'] - df_sumusd_['ref_log'] - total_initial_capital
    net_at_index_0 = net_raw.iloc[0]  # ค่า net ที่ index 0
    df_sumusd_['net'] = net_raw - net_at_index_0  # ปรับให้ index 0 = 0
    
    df_sumusd_ = df_sumusd_.reset_index().set_index('index')
    st.line_chart(df_sumusd_.net)
    st.dataframe(df_sumusd_)


with Burn_Cash:
    STOCK_SYMBOLS = ['FFWM', 'NEGG', 'RIVN', 'APLS' , 'NVTS' , 'QXO' , 'RXRX' ]
    
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

with tab6:
    QXO_act = client.get_field_last(field='{}'.format(7))
    QXO_act_js = int(json.loads(QXO_act)["field{}".format(7) ]) 
    plot( Ticker = 'QXO'  , act =  QXO_act_js  )


with tab7:
    RXRX_act = client.get_field_last(field='{}'.format(8))
    RXRX_act_js = int(json.loads(RXRX_act)["field{}".format(8) ]) 
    plot( Ticker = 'RXRX'  , act =  RXRX_act_js  )



import streamlit.components.v1 as components
# @st.cache_data
def iframe ( frame = ''):
  src = frame
  st.components.v1.iframe(src, width=1500 , height=800  , scrolling=0)

with cf_log: 
    st.write('')
    st.write(' Rebalance   =  -fix * ln( t0 / tn )')
    st.write(' Net Profit  =  sumusd - refer - sumusd[0] (ต้นทุนเริ่มต้น)')
    st.write(' Ref_index_Log = initial_capital_Ref_index_Log + (-1500 * ln(int_st / int_end))')
    st.write(' Net in Ref_index_Log = (daily_sumusd - ref_log - total_initial_capital) - net_at_index_0')
    st.write('________')
    iframe(frame = "https://monica.im/share/artifact?id=qpAkuKjBpuVz2cp9nNFRs3")  
    st.write('________')
    iframe(frame = "https://monica.im/share/artifact?id=wEjeaMxVW6MgDDm3xAZatX")    
    st.write('________')
    iframe(frame = "https://monica.im/share/chat?shareId=SUsEYhzSMwqIq3Cx")    



import pandas as pd
import numpy as np
from numba import njit
import yfinance as yf
import streamlit as st
import thingspeak
import json
 
st.set_page_config(page_title="Limit_F(X)", page_icon="✈️" , layout = "wide" )

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
    # เปลี่ยนเป็นสูตรที่ 2: refer = -fix * ln(t0/tn)
    refer = -fix * np.log(initial_price / price_array)
    
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

def find_best_seed_sliding_window(price_list, window_size=30, num_seeds_to_try=1000, progress_bar=None):
    """
    ค้นหาลำดับ action ที่ดีที่สุดโดยการหา seed ที่ให้ผลกำไรสูงสุดในแต่ละช่วงเวลา (sliding window)

    Args:
        price_list (list or np.array): รายการราคาของสินทรัพย์
        window_size (int): ขนาดของแต่ละช่วงเวลาที่จะค้นหา (เช่น 30 วัน)
        num_seeds_to_try (int): จำนวน seed ที่จะลองสุ่มในแต่ละ window
        progress_bar: (Optional) Streamlit progress bar object to update.

    Returns:
        np.array: ลำดับ action ที่ดีที่สุดที่ได้จากการผสมผสานของแต่ละ window
    """
    prices = np.asarray(price_list)
    n = len(prices)
    final_actions = np.array([], dtype=int)
    
    num_windows = (n + window_size - 1) // window_size # Calculate total number of windows
    
    print("Starting sliding window seed search...")
    # วนลูปตามข้อมูลราคาในแต่ละช่วง (window)
    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        window_len = len(prices_window)

        if window_len == 0:
            continue

        best_seed_for_window = -1
        # ใช้ -np.inf เพื่อให้แน่ใจว่าค่า net แรกที่เจอมักจะสูงกว่าเสมอ
        max_net_for_window = -np.inf  

        # --- ค้นหา Seed ที่ดีที่สุดสำหรับ Window ปัจจุบัน ---
        # ใช้ np.random.randint เพื่อให้ได้ seed ที่หลากหลายมากขึ้น
        random_seeds = np.random.randint(0, 10_000_000, size=num_seeds_to_try)

        for seed in random_seeds:
            rng = np.random.default_rng(seed)
            actions_window = rng.integers(0, 2, size=window_len)
            actions_window[0] = 1  # บังคับให้ action แรกของ window เป็น 1 เสมอ

            # ประเมินผลเฉพาะใน window นี้ (เริ่มนับ 1 ใหม่)
            if window_len < 2:
                # ถ้าข้อมูลไม่พอคำนวณ net ให้ใช้ action ที่สร้างได้เลย
                final_net = 0
            else:
                _, sumusd, _, _, _, refer = calculate_optimized(actions_window.tolist(), prices_window.tolist())
                initial_capital = sumusd[0]
                net = sumusd - refer - initial_capital
                final_net = net[-1] # เอาผลกำไร ณ วันสุดท้ายของ window

            if final_net > max_net_for_window:
                max_net_for_window = final_net
                best_seed_for_window = seed

        # --- สร้าง Action ที่ดีที่สุดสำหรับ Window นี้จาก Seed ที่พบ ---
        print(f"Window {start_index}-{end_index}: Best Seed = {best_seed_for_window} (Net: {max_net_for_window:.2f})")
        rng_best = np.random.default_rng(best_seed_for_window)
        best_actions_for_window = rng_best.integers(0, 2, size=window_len)
        best_actions_for_window[0] = 1

        # นำ action ที่ดีที่สุดของ window นี้ไปต่อท้าย action หลัก
        final_actions = np.concatenate((final_actions, best_actions_for_window))
        
        # Update progress bar if provided
        if progress_bar:
            progress_bar.progress((i + 1) / num_windows)

    return final_actions

def get_max_action(price_list, fix=1500):
    """
    คำนวณหาลำดับ action (0, 1) ที่ให้ผลตอบแทนสูงสุดทางทฤษฎี
    โดยใช้ Dynamic Programming ร่วมกับการย้อนรอย (Backtracking)
    """
    prices = np.asarray(price_list, dtype=np.float64)
    n = len(prices)

    if n < 2:
        return np.ones(n, dtype=int) # ถ้ามีข้อมูลไม่พอ ก็ action ไปเลย

    # --- ส่วนที่ 1: คำนวณไปข้างหน้า (เหมือนเดิม แต่เพิ่มการเก็บ path) ---
    
    dp = np.zeros(n, dtype=np.float64)
    # path[i] จะเก็บ index 'j' ของ action ครั้งก่อนหน้าที่ดีที่สุดสำหรับวัน 'i'
    path = np.zeros(n, dtype=int) 
    
    initial_capital = float(fix * 2)
    dp[0] = initial_capital
    
    for i in range(1, n):
        max_prev_sumusd = 0
        best_j = 0 # ตัวแปรสำหรับเก็บ j ที่ดีที่สุดสำหรับ i นี้
        
        for j in range(i):
            profit_from_j_to_i = fix * ((prices[i] / prices[j]) - 1)
            current_sumusd = dp[j] + profit_from_j_to_i
            
            if current_sumusd > max_prev_sumusd:
                max_prev_sumusd = current_sumusd
                best_j = j # เจอทางเลือกที่ดีกว่า ก็จำไว้ว่ามันมาจาก j ไหน
        
        dp[i] = max_prev_sumusd
        path[i] = best_j # บันทึกเส้นทางที่ดีที่สุดสำหรับวัน i

    # --- ส่วนที่ 2: การย้อนรอย (Backtracking) เพื่อสร้าง action array ---

    actions = np.zeros(n, dtype=int)
    
    # 1. หาจุดสิ้นสุดของเส้นทาง (วันที่ให้ sumusd สูงที่สุด)
    last_action_day = np.argmax(dp)
    
    # 2. เริ่มย้อนรอยจากจุดสิ้นสุดกลับไปหาจุดเริ่มต้น
    current_day = last_action_day
    while current_day > 0:
        # ทุกจุดที่เราเหยียบในการย้อนรอย คือวันที่ควรมี action
        actions[current_day] = 1
        # กระโดดกลับไปยัง action ครั้งก่อนหน้า
        current_day = path[current_day]
        
    # 3. กำหนดให้ action แรกสุดเป็น 1 เสมอ (เป็นจุดเริ่มต้นของทุกเส้นทาง)
    actions[0] = 1
    
    return actions


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
    
    elif act == -3:  # best_seed sliding window
        progress_bar = st.progress(0)
        st.write("กำลังค้นหา Best Seed ด้วย Sliding Window...")
        actions = find_best_seed_sliding_window(prices, window_size=30, num_seeds_to_try=1000, progress_bar=progress_bar)
        st.write("เสร็จสิ้นการค้นหา Best Seed!")
      
    else :
        rng = np.random.default_rng(act)
        actions = rng.integers(0, 2, len(prices)) 
    
    buffer, sumusd, cash, asset_value, amount , refer = calculate_optimized( actions ,  prices)
    
    # ใช้ sumusd[0] แทนค่าคงที่ 3000
    initial_capital = sumusd[0]
    
    df = pd.DataFrame({
        'price': prices,
        'action': actions,
        'buffer': np.round(buffer, 2),
        'sumusd': np.round(sumusd, 2),
        'cash': np.round(cash, 2),
        'asset_value': np.round(asset_value, 2),
        'amount': np.round(amount, 2),
        'refer': np.round(refer + initial_capital, 2),
        'net': np.round( sumusd - refer - initial_capital , 2) # ใช้ sumusd[0] แทน 3000
    })
    return df 


def plot (Ticker = ''   ,  act = -1 ):
    all = []
    all_id = []
    
    #min
    all.append( Limit_fx(Ticker , act = -1 ).net  )
    all_id.append('min')

    #fx
    if act == -3:  # best_seed
        all.append(Limit_fx( Ticker , act = act ).net )
        all_id.append('best_seed')
    else:
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

tab1, tab2, tab3, tab4, tab5 , tab6 , tab7 ,  Burn_Cash  , Ref_index_Log , cf_log , best_seed_tab   = st.tabs([ "FFWM", "NEGG", "RIVN" , 'APLS', 'NVTS', 'QXO(LV)' , 'RXRX(LV)' ,  'Burn_Cash' ,  'Ref_index_Log' , 'cf_log' , 'Best_Seed_Test' ])

with best_seed_tab:
    st.write("## ทดสอบ Best Seed Sliding Window")
    
    # เลือก ticker สำหรับทดสอบ
    test_ticker = st.selectbox("เลือก Ticker สำหรับทดสอบ", ['FFWM', 'NEGG', 'RIVN', 'APLS', 'NVTS', 'QXO', 'RXRX'])
    
    if st.button("ทดสอบ Best Seed"):
        st.write(f"กำลังทดสอบ Best Seed สำหรับ {test_ticker}...")
        plot(Ticker=test_ticker, act=-3)

with Ref_index_Log:
    tickers = ['FFWM', 'NEGG', 'RIVN', 'APLS' , 'NVTS' , 'QXO' , 'RXRX' ]
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
    
    filter_date = '2023-01-01 12:00:00+07:00'
    prices_df = get_prices(tickers, filter_date)
    prices_df = prices_df.dropna()
    
    int_st_list = prices_df.iloc[0][tickers]  # ราคาปิดเริ่มต้นของแต่ละ Ticker
    int_st = np.prod(int_st_list)
    
    # คำนวณ initial_capital_Ref_index_Log
    # สมมติว่า sumusd[0] ของแต่ละหุ้นคือ 3000 (fix * 2 = 1500 * 2)
    initial_capital_per_stock = 3000  # หรือจะใช้ Limit_fx(tickers[0], act=-1).sumusd.iloc[0] ก็ได้
    initial_capital_Ref_index_Log = initial_capital_per_stock * len(tickers)  # 3000 * 7 = 21000
    
    # คำนวณ ref_log สำหรับแต่ละแถว ด้วยสูตรใหม่
    def calculate_ref_log(row):
        int_end = np.prod(row[tickers])  # ผลคูณของราคาปิดในแถวปัจจุบัน
        ref_log = initial_capital_Ref_index_Log + (-1500 * np.log(int_st / int_end))
        return ref_log
    
    prices_df['ref_log'] = prices_df.apply(calculate_ref_log, axis=1)    
    prices_df = prices_df.reset_index()
    prices_df = prices_df.ref_log.values 

    sumusd_ = {'sumusd_{}'.format(symbol) : Limit_fx(symbol, act=-1).sumusd for symbol in tickers }
    df_sumusd_ = pd.DataFrame(sumusd_)
    df_sumusd_['daily_sumusd'] = df_sumusd_.sum(axis=1)
    df_sumusd_['ref_log'] = prices_df
    
    # คำนวณต้นทุนรวมจาก sumusd[0] ของแต่ละหุ้น
    total_initial_capital = sum([Limit_fx(symbol, act=-1).sumusd.iloc[0] for symbol in tickers])
    
    # คำนวณ net แบบใหม่: ลบต้นทุนเริ่มต้นออกเพื่อแสดงเฉพาะส่วนเกินทุน
    net_raw = df_sumusd_['daily_sumusd'] - df_sumusd_['ref_log'] - total_initial_capital
    net_at_index_0 = net_raw.iloc[0]  # ค่า net ที่ index 0
    df_sumusd_['net'] = net_raw - net_at_index_0  # ปรับให้ index 0 = 0
    
    df_sumusd_ = df_sumusd_.reset_index().set_index('index')
    st.line_chart(df_sumusd_.net)
    st.dataframe(df_sumusd_)


with Burn_Cash:
    STOCK_SYMBOLS = ['FFWM', 'NEGG', 'RIVN', 'APLS' , 'NVTS' , 'QXO' , 'RXRX' ]
    
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

with tab6:
    QXO_act = client.get_field_last(field='{}'.format(7))
    QXO_act_js = int(json.loads(QXO_act)["field{}".format(7) ]) 
    plot( Ticker = 'QXO'  , act =  QXO_act_js  )


with tab7:
    RXRX_act = client.get_field_last(field='{}'.format(8))
    RXRX_act_js = int(json.loads(RXRX_act)["field{}".format(8) ]) 
    plot( Ticker = 'RXRX'  , act =  RXRX_act_js  )



import streamlit.components.v1 as components
# @st.cache_data
def iframe ( frame = ''):
  src = frame
  st.components.v1.iframe(src, width=1500 , height=800  , scrolling=0)

with cf_log: 
    st.write('')
    st.write(' Rebalance   =  -fix * ln( t0 / tn )')
    st.write(' Net Profit  =  sumusd - refer - sumusd[0] (ต้นทุนเริ่มต้น)')
    st.write(' Ref_index_Log = initial_capital_Ref_index_Log + (-1500 * ln(int_st / int_end))')
    st.write(' Net in Ref_index_Log = (daily_sumusd - ref_log - total_initial_capital) - net_at_index_0')
    st.write(' Best Seed Sliding Window = หา seed ที่ดีที่สุดในแต่ละช่วง window แล้วนำมาต่อกัน')
    st.write('________')
    iframe(frame = "https://monica.im/share/artifact?id=qpAkuKjBpuVz2cp9nNFRs3")  
    st.write('________')
    iframe(frame = "https://monica.im/share/artifact?id=wEjeaMxVW6MgDDm3xAZatX")    
    st.write('________')
    iframe(frame = "https://monica.im/share/chat?shareId=SUsEYhzSMwqIq3Cx")    
