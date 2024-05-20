# import streamlit as st
# import numpy as np
# import datetime
# import thingspeak
# import pandas as pd
# import yfinance as yf
# import json

# st.set_page_config(page_title="Calculator", page_icon="⌨️")

# channel_id = 2528199
# write_api_key = '2E65V8XEIPH9B2VV'
# client = thingspeak.Channel(channel_id, write_api_key , fmt='json')

# def sell (asset = 0 , fix_c=1500 , Diff=60):
#   s1 =  (1500-Diff) /asset
#   s2 =  round(s1, 2)
#   s3 =  s2  *asset
#   s4 =  abs(s3 - fix_c)
#   s5 =  round( s4 / s2 )  
#   s6 =  s5*s2
#   s7 =  (asset * s2) + s6
#   return s2 , s5 , round(s7, 2)

# def buy (asset = 0 , fix_c=1500 , Diff=60):
#   b1 =  (1500+Diff) /asset
#   b2 =  round(b1, 2)
#   b3 =  b2 *asset
#   b4 =  abs(b3 - fix_c)
#   b5 =  round( b4 / b2 )  
#   b6 =  b5*b2
#   b7 =  (asset * b2) - b6
#   return b2 , b5 , round(b7, 2)


# col13, col16 , col14 , col15 , col17   = st.columns(5)

# x_2 = col16.number_input('Diff', step=1 , value= 60  )

# Start = col13.checkbox('start')
# if Start :
#   thingspeak_1 = col13.checkbox('@_FFWM_ASSET')
#   if thingspeak_1 :
#     add_1 = col13.number_input('@_FFWM_ASSET', step=0.001 ,  value=0.)
#     _FFWM_ASSET = col13.button("GO!")
#     if _FFWM_ASSET :
#       client.update(  {'field1': add_1 } )
#       col13.write(add_1) 
      
#   thingspeak_2 = col13.checkbox('@_NEGG_ASSET')
#   if thingspeak_2 :
#     add_2 = col13.number_input('@_NEGG_ASSET', step=0.001 ,  value=0.)
#     _NEGG_ASSET = col13.button("GO!")
#     if _NEGG_ASSET :
#       client.update(  {'field2': add_2 }  )
#       col13.write(add_2) 
      
#   thingspeak_3 = col13.checkbox('@_RIVN_ASSET')
#   if thingspeak_3 :
#     add_3 = col13.number_input('@_RIVN_ASSET', step=0.001 ,  value=0.)
#     _RIVN_ASSET = col13.button("GO!")
#     if _RIVN_ASSET :
#       client.update(  {'field3': add_3 }  )
#       col13.write(add_3) 



# FFWM_ASSET_LAST = client.get_field_last(field='field1')
# FFWM_ASSET_LAST =  eval(json.loads(FFWM_ASSET_LAST)['field1'])

# NEGG_ASSET_LAST = client.get_field_last(field='field2')
# NEGG_ASSET_LAST = eval(json.loads(NEGG_ASSET_LAST)['field2'])

# RIVN_ASSET_LAST = client.get_field_last(field='field3')
# RIVN_ASSET_LAST = eval(json.loads(RIVN_ASSET_LAST)['field3'])

# x_3 = col14.number_input('NEGG_ASSET', step=0.001 ,  value= NEGG_ASSET_LAST )
# x_4 = col15.number_input('FFWM_ASSET', step=0.001  , value= FFWM_ASSET_LAST  )
# x_5 = col17.number_input('RIVN_ASSET', step=0.001  , value= RIVN_ASSET_LAST  )

# st.write("_____") 

# try:
#   s8 , s9 , s10 =  sell( asset = x_3 , Diff= x_2)
#   s11 , s12 , s13 =  sell(asset = x_4 , Diff= x_2)
#   b8 , b9 , b10 =  buy(asset = x_3 , Diff= x_2)
#   b11 , b12 , b13 =  buy(asset = x_4 , Diff= x_2)
#   u1 , u2 , u3 = sell( asset = x_5 , Diff= x_2)
#   u4 , u5 , u6 = buy( asset = x_5 , Diff= x_2)
  
#   st.write("Limut_Order_NEGG") 
#   st.write( 'sell' , '   ' ,'A', b9  , 'P' , b8 ,'C' ,b10  )
  
#   col1, col2 , col3  = st.columns(3)
#   sell_negg = col3.checkbox('sell_match_NEGG')
#   if sell_negg :
#     GO_NEGG_SELL = col3.button("GO!")
#     if GO_NEGG_SELL :
#       client.update(  {'field2': NEGG_ASSET_LAST - b9  } )
#       col3.write(NEGG_ASSET_LAST - b9) 
    
#   st.write(yf.Ticker('NEGG').fast_info['lastPrice'])
  
#   col4, col5 , col6  = st.columns(3)
#   st.write( 'buy' , '   ','A',  s9  ,  'P' , s8 , 'C' ,s10  )
#   buy_negg = col6.checkbox('buy_match_NEGG')
#   if buy_negg :
#     GO_NEGG_Buy = col6.button("GO!")
#     if GO_NEGG_Buy :
#       client.update(  {'field2': NEGG_ASSET_LAST + s9  } )
#       col6.write(NEGG_ASSET_LAST + s9) 
  
#   st.write("_____") 
  
  
#   st.write("Limut Order_FFWM") 
#   st.write( 'sell' , '   ' , 'A', b12 , 'P' , b11  , 'C' , b13  )
#   col7, col8 , col9  = st.columns(3)
#   sell_ffwm = col9.checkbox('sell_match_FFWM')
#   if sell_ffwm :
#     GO_ffwm_sell = col9.button("GO!")
#     if GO_ffwm_sell :
#       client.update(  {'field1': FFWM_ASSET_LAST - b12  } )
#       col9.write(FFWM_ASSET_LAST - b12) 
  
#   st.write(yf.Ticker('FFWM').fast_info['lastPrice'])
  
#   col10, col11 , col12  = st.columns(3)
#   st.write(  'buy' , '   ', 'A', s12 , 'P' , s11  , 'C'  , s13  )
#   buy_ffwm = col12.checkbox('buy_match_FFWM')
#   if buy_ffwm :
#     GO_ffwm_Buy = col12.button("GO!")
#     if GO_ffwm_Buy :
#       client.update(  {'field1': FFWM_ASSET_LAST + s12  } )
#       col12.write(FFWM_ASSET_LAST + s12) 
  
#   st.write("_____") 
  
# #
#   st.write("Limut Order_RIVN") 
#   st.write( 'sell' , '   ' , 'A', u5 , 'P' , u4  , 'C' , u6  )
#   col77, col88 , col99  = st.columns(3)
#   sell_RIVN = col99.checkbox('sell_match_RIVN')
#   if sell_RIVN :
#     GO_RIVN_sell = col99.button("GO!")
#     if GO_RIVN_sell :
#       client.update(  {'field3': RIVN_ASSET_LAST - u5  } )
#       col99.write(RIVN_ASSET_LAST - u5) 
  
#   st.write(yf.Ticker('RIVN').fast_info['lastPrice'])
  
#   col100 , col111 , col122  = st.columns(3)
#   st.write(  'buy' , '   ', 'A', u2 , 'P' , u1  , 'C'  , u3  )
#   buy_RIVN = col122.checkbox('buy_match_RIVN')
#   if buy_RIVN :
#     GO_RIVN_Buy = col122.button("GO!")
#     if GO_RIVN_Buy :
#       client.update(  {'field3': RIVN_ASSET_LAST + u2  } )
#       col122.write(RIVN_ASSET_LAST + u2) 
  
#   st.write("_____") 

# except:pass


import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import thingspeak
import json


st.set_page_config( page_title="Calculator", page_icon="⌨️")

if st.button("rerun"):
    st.rerun()

channel_id = 2385118
write_api_key = 'IPSG3MMMBJEB9DY8'
client = thingspeak.Channel(channel_id, write_api_key , fmt='json')

def average_cf (Ticker = 'FFWM' , field = 1 ):
    tickerData = yf.Ticker( Ticker)
    tickerData = round(tickerData.history(period= 'max' )[['Close']] , 3 )
    tickerData.index = tickerData.index.tz_convert(tz='Asia/bangkok')
    filter_date = '2024-01-01 12:00:00+07:00'
    tickerData = tickerData[tickerData.index >= filter_date]
    tickerData = len(tickerData)
    
    client_2 = thingspeak.Channel(2394198 , 'OVZNYQBL57GJW5JF' , fmt='json')
    fx_2 = client_2.get_field_last(field='{}'.format(field))
    fx_js_2 = ( int(eval(json.loads(fx_2)["field{}".format(field)])) ) -  393
    return   fx_js_2 / tickerData 
    
st.write('____')
cf_day = average_cf()
st.write( 'average_cf_day:' ,  round(cf_day , 2 ), 'USD', " : " , 'average_cf_mo:' , round(cf_day*30 , 2) ,'USD'  )
st.write('____')

def Production(Ticker = "FFWM" ):
    try:
        tickerData = yf.Ticker(Ticker)
        entry  = tickerData.fast_info['lastPrice']  ; step = 0.01 ;  Fixed_Asset_Value = 1500. ; Cash_Balan = 650.
        if entry < 10000 :
            samples = np.arange( 0  ,  np.around(entry, 2) * 3 + step  ,  step)

            df = pd.DataFrame()
            df['Asset_Price'] =   np.around(samples, 2)
            df['Fixed_Asset_Value'] = Fixed_Asset_Value
            df['Amount_Asset']  =   df['Fixed_Asset_Value']  / df['Asset_Price']

            df_top = df[df.Asset_Price >= np.around(entry, 2) ]
            df_top['Cash_Balan_top'] = (df_top['Amount_Asset'].shift(1) -  df_top['Amount_Asset']) *  df_top['Asset_Price']
            df_top.fillna(0, inplace=True)
            np_Cash_Balan_top = df_top['Cash_Balan_top'].values

            xx = np.zeros(len(np_Cash_Balan_top)) ; y_0 = Cash_Balan
            for idx, v_0  in enumerate(np_Cash_Balan_top) :
                z_0 = y_0 + v_0
                y_0 = z_0
                xx[idx] = y_0

            df_top['Cash_Balan_top'] = xx
            df_top = df_top.rename(columns={'Cash_Balan_top': 'Cash_Balan'})
            df_top  = df_top.sort_values(by='Amount_Asset')
            df_top  = df_top[:-1]

            df_down = df[df.Asset_Price <= np.around(entry, 2) ]
            df_down['Cash_Balan_down'] = (df_down['Amount_Asset'].shift(-1) -  df_down['Amount_Asset'])     *  df_down['Asset_Price']
            df_down.fillna(0, inplace=True)
            df_down = df_down.sort_values(by='Asset_Price' , ascending=False)
            np_Cash_Balan_down = df_down['Cash_Balan_down'].values

            xxx= np.zeros(len(np_Cash_Balan_down)) ; y_1 = Cash_Balan
            for idx, v_1  in enumerate(np_Cash_Balan_down) :
                z_1 = y_1 + v_1
                y_1 = z_1
                xxx[idx] = y_1

            df_down['Cash_Balan_down'] = xxx
            df_down = df_down.rename(columns={'Cash_Balan_down': 'Cash_Balan'})

            df = pd.concat([df_top, df_down], axis=0)
            Production_Costs = (df['Cash_Balan'].values[-1]) -  Cash_Balan
            return   abs(Production_Costs)
    except:pass

def Monitor (Ticker = 'FFWM' , field = 2 ):
    tickerData = yf.Ticker( Ticker)
    tickerData = round(tickerData.history(period= 'max' )[['Close']] , 3 )
    tickerData.index = tickerData.index.tz_convert(tz='Asia/bangkok')
    filter_date = '2022-12-21 12:00:00+07:00'
    tickerData = tickerData[tickerData.index >= filter_date]
    
    fx = client.get_field_last(field='{}'.format(field))
    fx_js = int(json.loads(fx)["field{}".format(field)])
    np.random.seed(fx_js)
    data = np.random.randint(2, size = len(tickerData))
    tickerData['action'] = data
    tickerData['index'] = [ i+1 for i in range(len(tickerData))]
    
    tickerData_1 = pd.DataFrame(columns=(tickerData.columns))
    tickerData_1['action'] =  [ i for i in range(5)]
    tickerData_1.index = ['+0' , "+1" , "+2" , "+3" , "+4"]
    df = pd.concat([tickerData , tickerData_1], axis=0).fillna("")
    np.random.seed(fx_js)
    df['action'] = np.random.randint(2, size = len(df))
    return df.tail(7) , fx_js

df_7 , fx_js  = Monitor(Ticker = 'FFWM', field = 2)
st.write( 'FFWM')
st.write("f(x): {}".format(fx_js) ," , " , "Production: {}".format(    np.around(Production('FFWM'), 2) ))
st.table(df_7)
st.write("_____") 

df_7_1 , fx_js_1  = Monitor(Ticker = 'NEGG', field = 3)
st.write( 'NEGG')
st.write("f(x): {}".format(fx_js_1) ," , " , "Production: {}".format(    np.around(Production('NEGG'), 2) ))
st.table(df_7_1)
st.write("_____") 

df_7_2 , fx_js_2  = Monitor(Ticker = 'RIVN', field = 4)
st.write( 'RIVN')
st.write("f(x): {}".format(fx_js_2) ," , " , "Production: {}".format(    np.around(Production('RIVN'), 2) ))
st.table(df_7_2)
st.write("_____") 


st.write("***ก่อนตลาดเปิดตรวจสอบ TB ล่าสุด > RE เมื่อตลอดเปิด")
st.write("***RE > 60 USD")
st.stop()







