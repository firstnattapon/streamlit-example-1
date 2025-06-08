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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, Burn_Cash, Ref_index_Log, cf_log, dna_tab, stitched_dna_tab = st.tabs(["FFWM", "NEGG", "RIVN", 'APLS', 'NVTS', 'QXO(LV)', 'RXRX(LV)', 'Burn_Cash', 'Ref_index_Log', 'cf_log', 'DNA Analysis', 'Stitched DNA'])
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
    iframe(frame = "https://monica.im/share/artifact?id=ZfHT5iDP2Ypz82PCRw9nEK")    
    st.write('________')
    iframe(frame = "https://monica.im/share/chat?shareId=SUsEYhzSMwqIq3Cx")    


# ===================================================================
# โค้ดสำหรับแท็บ Stitched DNA Analysis (วางส่วนนี้ต่อท้ายไฟล์)
# ===================================================================
import io

@st.cache_data # Cache ผลลัพธ์เพื่อความรวดเร็วในการรันครั้งต่อไป
def build_stitched_sequence_from_csv(csv_data):
    """
    อ่านข้อมูล CSV, แปลง action_sequence ที่เป็น string,
    และประกอบร่างเป็น action sequence สุดท้าย
    """
    # ใช้ io.StringIO เพื่อให้ pandas อ่าน string ได้เหมือนไฟล์
    df = pd.read_csv(io.StringIO(csv_data))
    
    final_actions = []
    # วนลูปเพื่อนำ action จากแต่ละ window มาต่อกัน
    for index, row in df.iterrows():
        # คอลัมน์ action_sequence เป็น string -> แปลงเป็น list ของ int
        # ใช้ json.loads ซึ่งปลอดภัยและจัดการกับ string format นี้ได้ดี
        action_list_for_window = json.loads(row['action_sequence'])
        final_actions.extend(action_list_for_window)
        
    return np.array(final_actions, dtype=np.int32), df

with stitched_dna_tab:
    st.header("Stitched DNA Performance Analysis (Window-based Optimization)")
    st.info("กลยุทธ์นี้สร้างขึ้นจากการนำ 'Action Sequence' ที่ดีที่สุดจากแต่ละช่วงเวลา (Window) มาต่อกัน")

    # ข้อมูล CSV จากไฟล์ best_seed_results_FFWM_30w_30000s.csv
    csv_data_ffwm = """window_number,timeline,start_index,end_index,window_size,best_seed,max_net,start_price,end_price,price_change_pct,action_count,action_sequence
1,2023-01-03 ถึง 2023-02-14,0,29,30,28834,12.55,14.04,15.3,8.98,17,"[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1]"
2,2023-02-15 ถึง 2023-03-29,30,59,30,1408,382.37,15.58,7.85,-49.59,6,"[1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]"
3,2023-03-30 ถึง 2023-05-11,60,89,30,9009,207.97,7.34,4.03,-45.1,8,"[1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]"
4,2023-05-12 ถึง 2023-06-26,90,119,30,21238,187.14,3.85,4.22,9.54,14,"[1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0]"
5,2023-06-27 ถึง 2023-08-08,120,149,30,25558,353.46,4.12,7.41,80.02,11,"[1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1]"
6,2023-08-09 ถึง 2023-09-20,150,179,30,2396,38.18,7.21,7.44,3.17,16,"[1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1]"
7,2023-09-21 ถึง 2023-11-01,180,209,30,24599,88.19,6.9,4.6,-33.33,10,"[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1]"
8,2023-11-02 ถึง 2023-12-14,210,239,30,21590,251.56,5.22,8.99,72.14,11,"[1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]"
9,2023-12-15 ถึง 2024-01-30,240,269,30,15176,40.75,8.82,10.23,15.93,10,"[1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0]"
10,2024-01-31 ถึง 2024-03-13,270,299,30,19030,60.14,9.49,7.89,-16.92,10,"[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0]"
11,2024-03-14 ถึง 2024-04-25,300,329,30,5252,52.19,7.31,6.69,-8.47,15,"[1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0]"
12,2024-04-26 ถึง 2024-06-07,330,359,30,16872,60.81,6.07,5.69,-6.24,14,"[1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]"
13,2024-06-10 ถึง 2024-07-23,360,389,30,21590,186.0,5.69,6.95,22.14,11,"[1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]"
14,2024-07-24 ถึง 2024-09-04,390,419,30,23566,47.62,6.71,6.93,3.28,10,"[1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0]"
15,2024-09-05 ถึง 2024-10-16,420,449,30,25802,79.68,6.9,7.69,11.45,8,"[1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]"
16,2024-10-17 ถึง 2024-11-27,450,479,30,14998,85.98,7.71,8.08,4.8,12,"[1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]"
17,2024-11-29 ถึง 2025-01-14,480,509,30,18548,70.37,7.95,6.0,-24.53,11,"[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0]"
18,2025-01-15 ถึง 2025-02-27,510,539,30,29470,57.0,6.21,4.98,-19.81,19,"[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1]"
19,2025-02-28 ถึง 2025-04-10,540,569,30,15035,42.83,5.09,4.61,-9.43,11,"[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1]"
20,2025-04-11 ถึง 2025-05-23,570,599,30,17303,20.19,4.64,5.17,11.42,8,"[1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0]"
21,2025-05-27 ถึง 2025-06-06,600,608,9,3754,5.59,5.39,5.3,-1.67,2,"[1, 0, 0, 0, 1, 0, 0, 0, 0]"
"""

    if st.button("Analyze Stitched DNA for FFWM"):
        try:
            # 1. สร้าง 'action_sequence' ทั้งหมดจากข้อมูล CSV
            stitched_actions, details_df = build_stitched_sequence_from_csv(csv_data_ffwm)
            
            # 2. ดึงข้อมูลราคาหุ้น FFWM ทั้งหมด (ใช้ฟังก์ชัน Limit_fx เพื่อความสะดวก)
            # เราจะดึงข้อมูลมาแค่ครั้งเดียวแล้วใช้ซ้ำ
            full_data_df = Limit_fx(Ticker='FFWM', act=-1) # act=-1 แค่เพื่อให้ได้ข้อมูลมา
            prices = full_data_df['price'].values
            
            # 3. ตรวจสอบความสอดคล้องของข้อมูล
            if len(stitched_actions) != len(prices):
                st.error(f"Data Mismatch! Length of stitched actions ({len(stitched_actions)}) does not match length of price data ({len(prices)}). Please check the date range and CSV file.")
            else:
                st.success(f"Data check passed. Analyzing {len(prices)} data points.")
                
                # 4. คำนวณผลลัพธ์ของกลยุทธ์ต่างๆ
                # 'min' และ 'max' สามารถดึงมาจาก DataFrame ที่โหลดมาได้เลย
                min_net = full_data_df['net']
                max_net = Limit_fx(Ticker='FFWM', act=-2)['net']

                # คำนวณผลลัพธ์ของ Stitched DNA
                buffer, sumusd, cash, asset_value, amount, refer = calculate_optimized(stitched_actions, prices)
                initial_capital = sumusd[0]
                stitched_net = np.round(sumusd - refer - initial_capital, 2)
                
                # 5. สร้าง DataFrame สำหรับพล็อตกราฟ
                plot_df = pd.DataFrame({
                    'min': min_net.values,
                    'max': max_net.values,
                    'stitched_dna': stitched_net
                }, index=full_data_df.index)

                # 6. แสดงผล
                st.subheader("Performance Comparison (Net Profit)")
                st.line_chart(plot_df)

                with st.expander("Show Result Data"):
                    st.dataframe(plot_df)

                with st.expander("Show Details of Stitched Sequence"):
                    st.write("ตารางสรุปผลลัพธ์ที่ดีที่สุดในแต่ละ Window:")
                    st.dataframe(details_df)
                    st.write(f"Total actions in stitched sequence: {stitched_actions.sum()} out of {len(stitched_actions)}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
