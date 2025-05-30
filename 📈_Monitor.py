import streamlit as st
import numpy as np
import datetime
import thingspeak
import pandas as pd
import yfinance as yf
import json

st.set_page_config(page_title="Monitor", page_icon="📈" , layout="wide" )
channel_id = 2528199
write_api_key = '2E65V8XEIPH9B2VV' # Please ensure this is your correct Write API Key
client = thingspeak.Channel(channel_id, write_api_key , fmt='json')

def sell (asset = 0 , fix_c=1500 , Diff=60):
    s1 =  (1500-Diff) /asset if asset != 0 else 0
    s2 =  round(s1, 2)
    s3 =  s2  *asset
    s4 =  abs(s3 - fix_c)
    s5 =  round( s4 / s2 ) if s2 != 0 else 0
    s6 =  s5*s2
    s7 =  (asset * s2) + s6
    return s2 , s5 , round(s7, 2)

def buy (asset = 0 , fix_c=1500 , Diff=60):
    b1 =  (1500+Diff) /asset if asset != 0 else 0
    b2 =  round(b1, 2)
    b3 =  b2 *asset
    b4 =  abs(b3 - fix_c)
    b5 =  round( b4 / b2 ) if b2 != 0 else 0
    b6 =  b5*b2
    b7 =  (asset * b2) - b6
    return b2 , b5 , round(b7, 2)

channel_id_2 = 2385118
write_api_key_2 = 'IPSG3MMMBJEB9DY8' # Please ensure this is your correct Write API Key
client_2 = thingspeak.Channel(channel_id_2, write_api_key_2 , fmt='json' )

def Monitor (Ticker = 'FFWM' , field = 2  ):
    try:
        tickerData = yf.Ticker(Ticker)
        # Use a short period for history to avoid issues with delisted/problematic tickers for daily use
        # Max period can be slow and is often not needed for this kind of monitoring.
        # Consider '1y' or a relevant recent period. For this implementation, using 'max' as in original.
        tickerData = round(tickerData.history(period= 'max' )[['Close']] , 3 )
        if tickerData.empty: # Handle empty dataframe if ticker history is not available
            st.error(f"No data found for Ticker: {Ticker}. Please check the ticker symbol.")
            # Return a default DataFrame structure to prevent downstream errors
            default_cols = ['Close', 'action', 'index']
            return pd.DataFrame(columns=default_cols).tail(7), 0

        tickerData.index = tickerData.index.tz_convert(tz='Asia/Bangkok') # Original tz='Asia/bangkok'
        filter_date = '2023-01-01 12:00:00+07:00'
        tickerData = tickerData[tickerData.index >= filter_date]

        fx_data = client_2.get_field_last(field='{}'.format(field))
        fx_js = 0 # Default value
        if fx_data:
            try:
                loaded_fx = json.loads(fx_data)
                if isinstance(loaded_fx, dict) and "field{}".format(field) in loaded_fx and loaded_fx["field{}".format(field)] is not None:
                    fx_js = int(loaded_fx["field{}".format(field)])
                else: # Handle case where field might be missing or null after successful JSON load
                    fx_js = int(datetime.datetime.now().timestamp()) # Use timestamp as a fallback seed
            except (json.JSONDecodeError, ValueError, TypeError): # Handle errors during JSON parsing or int conversion
                st.warning(f"Could not parse ThingSpeak seed for {Ticker} (field {field}). Using timestamp as seed.")
                fx_js = int(datetime.datetime.now().timestamp()) # Use timestamp as a fallback seed
        else: # Handle case where get_field_last returns None or empty
            st.warning(f"No ThingSpeak seed data for {Ticker} (field {field}). Using timestamp as seed.")
            fx_js = int(datetime.datetime.now().timestamp())


        rng = np.random.default_rng(fx_js)
        data = rng.integers(0, 2, size = len(tickerData)) # integers are [low, high)
        tickerData['action'] = data
        tickerData['index'] = [ i+1 for i in range(len(tickerData))]

        tickerData_1 = pd.DataFrame(columns=(tickerData.columns))
        # Ensure 'action' column exists before assigning to it if tickerData was empty
        if 'action' not in tickerData_1.columns:
            tickerData_1['action'] = None
        tickerData_1['action'] =  [ i % 2 for i in range(5)] # Ensure values are 0 or 1
        tickerData_1.index = ['+0' , "+1" , "+2" , "+3" , "+4"]
        df = pd.concat([tickerData , tickerData_1], axis=0).fillna("")
        
        rng_df = np.random.default_rng(fx_js) # Re-seed for consistent full df manipulation if intended
        df['action'] = rng_df.integers(0, 2, size = len(df))
        return df.tail(7) , fx_js
    except Exception as e:
        st.error(f"Error in Monitor function for {Ticker}: {e}")
        # Return a default DataFrame structure
        default_cols = ['Close', 'action', 'index']
        return pd.DataFrame(columns=default_cols).tail(7), 0


df_7   , fx_js    = Monitor(Ticker = 'FFWM', field = 2    )
df_7_1 , fx_js_1  = Monitor(Ticker = 'NEGG', field = 3    )
df_7_2 , fx_js_2  = Monitor(Ticker = 'RIVN', field = 4  )
df_7_3 , fx_js_3  = Monitor(Ticker = 'APLS', field = 5  )
df_7_4 , fx_js_4  = Monitor(Ticker = 'NVTS', field = 6  )
df_7_5 , fx_js_5  = Monitor(Ticker = 'QXO', field = 7  )
df_7_6 , fx_js_6  = Monitor(Ticker = 'RXRX', field = 8 ) # Added for RXRX


nex = 0
Nex_day_sell = 0
toggle = lambda x : 1 - x if x in (0,1) else 0 # Ensure toggle works for 0 and 1

Nex_day_ = st.checkbox('nex_day')
if Nex_day_ :
    st.write( "value = " , nex)
    nex_col , Nex_day_sell_col ,_,_,_  = st.columns(5)

    if nex_col.button("Nex_day"):
        nex = 1
        # st.write( "value = " , nex) # Can be removed if redundant, or use st.experimental_rerun
        st.rerun()


    if Nex_day_sell_col.button("Nex_day_sell"):
        nex = 1
        Nex_day_sell = 1
        # st.write( "value = " , nex) # Can be removed
        # st.write( "Nex_day_sell = " , Nex_day_sell) # Can be removed
        st.rerun()


st.write("_____")

# Added col21 for RXRX
col13, col16, col14, col15, col17, col18, col19, col20, col21 = st.columns(9)

x_2 = col16.number_input('Diff', step=1 , value= 60      )

Start = col13.checkbox('start')
if Start :
    thingspeak_1 = col13.checkbox('@_FFWM_ASSET')
    if thingspeak_1 :
        add_1 = col13.number_input('@_FFWM_ASSET_input', step=0.001 ,  value=0., key='add_1_ffwm') # Unique key
        _FFWM_ASSET = col13.button("GO!")
        if _FFWM_ASSET :
            client.update(  {'field1': add_1 } )
            col13.write(add_1)
            st.rerun()


    thingspeak_2 = col13.checkbox('@_NEGG_ASSET')
    if thingspeak_2 :
        add_2 = col13.number_input('@_NEGG_ASSET_input', step=0.001 ,  value=0., key='add_2_negg') # Unique key
        _NEGG_ASSET = col13.button("GO! ") # Original key, be mindful of potential collisions
        if _NEGG_ASSET :
            client.update(  {'field2': add_2 }  )
            col13.write(add_2)
            st.rerun()

    thingspeak_3 = col13.checkbox('@_RIVN_ASSET')
    if thingspeak_3 :
        add_3 = col13.number_input('@_RIVN_ASSET_input', step=0.001 ,  value=0., key='add_3_rivn') # Unique key
        _RIVN_ASSET = col13.button("GO!  ") # Original key
        if _RIVN_ASSET :
            client.update(  {'field3': add_3 }  )
            col13.write(add_3)
            st.rerun()

    thingspeak_4 = col13.checkbox('@_APLS_ASSET')
    if thingspeak_4 :
        add_4 = col13.number_input('@_APLS_ASSET_input', step=0.001 ,  value=0., key='add_4_apls') # Unique key
        _APLS_ASSET = col13.button("GO!   ") # Original key
        if _APLS_ASSET :
            client.update(  {'field4': add_4 }  )
            col13.write(add_4)
            st.rerun()

    thingspeak_5 = col13.checkbox('@_NVTS_ASSET')
    if thingspeak_5:
        add_5 = col13.number_input('@_NVTS_ASSET_input', step=0.001, value= 0., key='add_5_nvts') # Unique key
        _NVTS_ASSET = col13.button("GO!    ") # Original key
        if _NVTS_ASSET:
            client.update({'field5': add_5})
            col13.write(add_5)
            st.rerun()
    
    thingspeak_6 = col13.checkbox('@_QXO_ASSET')
    if thingspeak_6:
        add_6 = col13.number_input('@_QXO_ASSET_input', step=0.001, value=0., key='add_6_qxo') # Unique key
        _QXO_ASSET = col13.button("GO!     ") # Original key
        if _QXO_ASSET:
            client.update({'field6': add_6})
            col13.write(add_6)
            st.rerun()

    # Added for RXRX
    thingspeak_7 = col13.checkbox('@_RXRX_ASSET')
    if thingspeak_7:
        add_7 = col13.number_input('@_RXRX_ASSET_input', step=0.001, value=0., key='add_7_rxrx') # Unique key
        _RXRX_ASSET_BTN = col13.button("GO_RXRX_INIT") # Unique key for RXRX
        if _RXRX_ASSET_BTN:
            client.update({'field7': add_7})
            col13.write(add_7)
            st.rerun()


def get_asset_value_from_thingspeak(client, field_id_str, field_key_str):
    asset_last_json_str = client.get_field_last(field=field_id_str)
    asset_last_val = 0.0
    if asset_last_json_str:
        try:
            loaded_json = json.loads(asset_last_json_str)
            if isinstance(loaded_json, dict) and field_key_str in loaded_json and loaded_json[field_key_str] is not None:
                asset_last_val = float(loaded_json[field_key_str]) # Using float for safety
            # else: asset_last_val remains 0.0
        except (json.JSONDecodeError, ValueError, TypeError):
            # Error in parsing or field not a number, asset_last_val remains 0.0
            pass # Optionally log error: st.warning(f"Could not parse data for {field_key_str}")
    return asset_last_val

FFWM_ASSET_LAST = get_asset_value_from_thingspeak(client, 'field1', 'field1')
NEGG_ASSET_LAST = get_asset_value_from_thingspeak(client, 'field2', 'field2')
RIVN_ASSET_LAST = get_asset_value_from_thingspeak(client, 'field3', 'field3')
APLS_ASSET_LAST = get_asset_value_from_thingspeak(client, 'field4', 'field4')
NVTS_ASSET_LAST = get_asset_value_from_thingspeak(client, 'field5', 'field5')
QXO_ASSET_LAST = get_asset_value_from_thingspeak(client, 'field6', 'field6')
RXRX_ASSET_LAST = get_asset_value_from_thingspeak(client, 'field7', 'field7') # Added for RXRX

x_3 = col14.number_input('NEGG_ASSET', step=0.001 ,  value= NEGG_ASSET_LAST, key='x_3_negg' )
x_4 = col15.number_input('FFWM_ASSET', step=0.001  , value= FFWM_ASSET_LAST, key='x_4_ffwm'  )
x_5 = col17.number_input('RIVN_ASSET', step=0.001  , value= RIVN_ASSET_LAST, key='x_5_rivn'  )
x_6 = col18.number_input('APLS_ASSET', step=0.001  , value= APLS_ASSET_LAST, key='x_6_apls'  )
x_7 = col19.number_input('NVTS_ASSET', step=0.001  , value= NVTS_ASSET_LAST, key='x_7_nvts' )

QXO_OPTION = 79.
QXO_REAL   =  col20.number_input('QXO_ASSET (LV:79@19.0)', step=0.001  , value=  QXO_ASSET_LAST, key='qxo_real')      
x_8 =  QXO_OPTION  + QXO_REAL

# Added for RXRX
RXRX_OPTION = 278.
RXRX_REAL   =  col21.number_input('RXRX_ASSET (LV:278@5.4)', step=0.001  , value=  RXRX_ASSET_LAST, key='rxrx_real')      
x_9 =  RXRX_OPTION  + RXRX_REAL

st.write("_____")

# try: # Original try was commented out, keeping it that way.

s8 , s9 , s10 =  sell( asset = x_3 , Diff= x_2)
s11 , s12 , s13 =  sell(asset = x_4 , Diff= x_2)
b8 , b9 , b10 =  buy(asset = x_3 , Diff= x_2)
b11 , b12 , b13 =  buy(asset = x_4 , Diff= x_2)
u1 , u2 , u3 = sell( asset = x_5 , Diff= x_2)
u4 , u5 , u6 = buy( asset = x_5 , Diff= x_2)
p1 , p2 , p3 = sell( asset = x_6 , Diff= x_2)
p4 , p5 , p6 = buy( asset = x_6 , Diff= x_2)
u7 , u8 , u9 = sell( asset = x_7 , Diff= x_2) # Note: u7,u8,u9 for NVTS sell
p7 , p8 , p9 = buy( asset = x_7 , Diff= x_2) # Note: p7,p8,p9 for NVTS buy
q1, q2, q3 = sell(asset=x_8, Diff=x_2)
q4, q5, q6 = buy(asset=x_8, Diff=x_2)

# Added for RXRX (using 'r' for RXRX variables)
r1, r2, r3 = sell(asset=x_9, Diff=x_2) # RXRX sell parameters
r4, r5, r6 = buy(asset=x_9, Diff=x_2)  # RXRX buy parameters


# Helper function to get current price, safely
def get_current_price(ticker_symbol, default_price=0.0):
    try:
        price = yf.Ticker(ticker_symbol).fast_info.get('lastPrice', default_price)
        if price is None: # If key exists but value is None
             return default_price
        return float(price)
    except Exception: # Catch any error during API call or data parsing
        return default_price

# --- NEGG ---
# Ensure df_7_1.action.values is not empty before accessing
negg_action_value = 0
if df_7_1 is not None and 'action' in df_7_1.columns and len(df_7_1.action.values) > (1+nex):
    action_val_negg = df_7_1.action.values[1+nex]
    if pd.isna(action_val_negg) or action_val_negg == "": # Handle potential empty string from fillna("")
        action_val_negg = 0 # Default to 0 if value is NaN or empty string
    negg_action_value = toggle(action_val_negg) if Nex_day_sell == 1 else action_val_negg
else:
    st.warning("NEGG action data not available for current/next day.")

Limut_Order_NEGG = st.checkbox('Limut_Order_NEGG', value = bool(negg_action_value), key='limit_negg')
if Limut_Order_NEGG :
    st.write( 'sell' , '     ' ,'A', b9  , 'P' , b8 ,'C' ,b10      ) # Buy params for UI "sell"

    col1, col2 , col3  = st.columns(3)
    sell_negg = col3.checkbox('sell_match_NEGG', key='sell_match_negg')
    if sell_negg :
        GO_NEGG_SELL = col3.button("GO!_SELL_NEGG") # Unique Key
        if GO_NEGG_SELL :
            client.update(  {'field2': NEGG_ASSET_LAST - b9  } )
            col3.write(f"Updated NEGG Asset: {NEGG_ASSET_LAST - b9}")
            st.rerun()

    negg_current_price = get_current_price('NEGG')
    pv_negg =  negg_current_price * x_3 if x_3 > 0 else 0
    st.write(negg_current_price , pv_negg  ,'(',  round(pv_negg - 1500,2) ,')',  )

    col4, col5 , col6  = st.columns(3)
    st.write( 'buy' , '    ','A',  s9  ,  'P' , s8 , 'C' ,s10      ) # Sell params for UI "buy"
    buy_negg = col6.checkbox('buy_match_NEGG', key='buy_match_negg')
    if buy_negg :
        GO_NEGG_Buy = col6.button("GO!_BUY_NEGG") # Unique Key
        if GO_NEGG_Buy :
            client.update(  {'field2': NEGG_ASSET_LAST + s9  } )
            col6.write(f"Updated NEGG Asset: {NEGG_ASSET_LAST + s9}")
            st.rerun()
st.write("_____")

# --- FFWM ---
ffwm_action_value = 0
if df_7 is not None and 'action' in df_7.columns and len(df_7.action.values) > (1+nex):
    action_val_ffwm = df_7.action.values[1+nex]
    if pd.isna(action_val_ffwm) or action_val_ffwm == "":
        action_val_ffwm = 0
    ffwm_action_value = toggle(action_val_ffwm) if Nex_day_sell == 1 else action_val_ffwm
else:
    st.warning("FFWM action data not available for current/next day.")

Limut_Order_FFWM = st.checkbox('Limut_Order_FFWM',  value = bool(ffwm_action_value), key='limit_ffwm')
if Limut_Order_FFWM :
    st.write( 'sell' , '     ' , 'A', b12 , 'P' , b11  , 'C' , b13      )

    col7, col8 , col9  = st.columns(3)
    sell_ffwm = col9.checkbox('sell_match_FFWM', key='sell_match_ffwm')
    if sell_ffwm :
        GO_ffwm_sell = col9.button("GO!_SELL_FFWM") # Unique Key
        if GO_ffwm_sell :
            client.update(  {'field1': FFWM_ASSET_LAST - b12  } )
            col9.write(f"Updated FFWM Asset: {FFWM_ASSET_LAST - b12}")
            st.rerun()

    ffwm_current_price = get_current_price('FFWM')
    pv_ffwm =  ffwm_current_price * x_4 if x_4 > 0 else 0
    st.write(ffwm_current_price , pv_ffwm ,'(',  round(pv_ffwm - 1500,2) ,')', )

    col10, col11 , col12  = st.columns(3)
    st.write(  'buy' , '    ', 'A', s12 , 'P' , s11  , 'C'  , s13      )
    buy_ffwm = col12.checkbox('buy_match_FFWM', key='buy_match_ffwm')
    if buy_ffwm :
        GO_ffwm_Buy = col12.button("GO!_BUY_FFWM") # Unique Key
        if GO_ffwm_Buy :
            client.update(  {'field1': FFWM_ASSET_LAST + s12  } )
            col12.write(f"Updated FFWM Asset: {FFWM_ASSET_LAST + s12}")
            st.rerun()
st.write("_____")

# --- RIVN ---
rivn_action_value = 0
if df_7_2 is not None and 'action' in df_7_2.columns and len(df_7_2.action.values) > (1+nex):
    action_val_rivn = df_7_2.action.values[1+nex]
    if pd.isna(action_val_rivn) or action_val_rivn == "":
        action_val_rivn = 0
    rivn_action_value = toggle(action_val_rivn) if Nex_day_sell == 1 else action_val_rivn
else:
    st.warning("RIVN action data not available for current/next day.")

Limut_Order_RIVN = st.checkbox('Limut_Order_RIVN',value = bool(rivn_action_value), key='limit_rivn')
if Limut_Order_RIVN :
    st.write( 'sell' , '     ' , 'A', u5 , 'P' , u4  , 'C' , u6      ) # Buy params (u4,u5,u6) for UI "sell"

    col77, col88 , col99  = st.columns(3)
    sell_RIVN = col99.checkbox('sell_match_RIVN', key='sell_match_rivn')
    if sell_RIVN :
        GO_RIVN_sell = col99.button("GO!_SELL_RIVN") # Unique Key
        if GO_RIVN_sell :
            client.update(  {'field3': RIVN_ASSET_LAST - u5  } )
            col99.write(f"Updated RIVN Asset: {RIVN_ASSET_LAST - u5}")
            st.rerun()

    rivn_current_price = get_current_price('RIVN')
    pv_rivn =  rivn_current_price * x_5 if x_5 > 0 else 0
    st.write(rivn_current_price , pv_rivn ,'(',  round(pv_rivn - 1500,2) ,')', )

    col100 , col111 , col122  = st.columns(3)
    st.write(  'buy' , '    ', 'A', u2 , 'P' , u1  , 'C'  , u3      ) # Sell params (u1,u2,u3) for UI "buy"
    buy_RIVN = col122.checkbox('buy_match_RIVN', key='buy_match_rivn')
    if buy_RIVN :
        GO_RIVN_Buy = col122.button("GO!_BUY_RIVN") # Unique Key
        if GO_RIVN_Buy :
            client.update(  {'field3': RIVN_ASSET_LAST + u2  } )
            col122.write(f"Updated RIVN Asset: {RIVN_ASSET_LAST + u2}")
            st.rerun()
st.write("_____")

# --- APLS ---
apls_action_value = 0
if df_7_3 is not None and 'action' in df_7_3.columns and len(df_7_3.action.values) > (1+nex):
    action_val_apls = df_7_3.action.values[1+nex]
    if pd.isna(action_val_apls) or action_val_apls == "":
        action_val_apls = 0
    apls_action_value = toggle(action_val_apls) if Nex_day_sell == 1 else action_val_apls
else:
    st.warning("APLS action data not available for current/next day.")

Limut_Order_APLS = st.checkbox('Limut_Order_APLS',value = bool(apls_action_value), key='limit_apls')
if Limut_Order_APLS :
    st.write( 'sell' , '     ' , 'A', p5 , 'P' , p4  , 'C' , p6      ) # Buy params (p4,p5,p6) for UI "sell"

    col7777, col8888 , col9999  = st.columns(3)
    sell_APLS = col9999.checkbox('sell_match_APLS', key='sell_match_apls')
    if sell_APLS :
        GO_APLS_sell = col9999.button("GO!_SELL_APLS") # Unique Key
        if GO_APLS_sell :
            client.update(  {'field4': APLS_ASSET_LAST - p5  } )
            col9999.write(f"Updated APLS Asset: {APLS_ASSET_LAST - p5}")
            st.rerun()

    apls_current_price = get_current_price('APLS')
    pv_apls =  apls_current_price * x_6 if x_6 > 0 else 0
    st.write(apls_current_price , pv_apls ,'(',  round(pv_apls - 1500,2) ,')', )

    col1000 , col1111 , col1222  = st.columns(3)
    st.write(  'buy' , '    ', 'A', p2 , 'P' , p1  , 'C'  , p3      ) # Sell params (p1,p2,p3) for UI "buy"
    buy_APLS = col1222.checkbox('buy_match_APLS', key='buy_match_apls')
    if buy_APLS :
        GO_APLS_Buy = col1222.button("GO!_BUY_APLS") # Unique Key
        if GO_APLS_Buy :
            client.update(  {'field4': APLS_ASSET_LAST + p2  } )
            col1222.write(f"Updated APLS Asset: {APLS_ASSET_LAST + p2}")
            st.rerun()
st.write("_____")

# --- NVTS ---
nvts_action_value = 0
if df_7_4 is not None and 'action' in df_7_4.columns and len(df_7_4.action.values) > (1+nex):
    action_val_nvts = df_7_4.action.values[1+nex]
    if pd.isna(action_val_nvts) or action_val_nvts == "":
        action_val_nvts = 0
    nvts_action_value = toggle(action_val_nvts) if Nex_day_sell == 1 else action_val_nvts
else:
    st.warning("NVTS action data not available for current/next day.")

Limut_Order_NVTS = st.checkbox('Limut_Order_NVTS', value=bool(nvts_action_value), key='limit_nvts')
if Limut_Order_NVTS:
    st.write('sell', '    ', 'A', p8 , 'P', p7  , 'C', p9      ) # Buy params (p7,p8,p9) for UI "sell"

    col_nvts1, col_nvts2, col_nvts3 = st.columns(3)
    sell_NVTS = col_nvts3.checkbox('sell_match_NVTS', key='sell_match_nvts')

    if sell_NVTS:
        GO_NVTS_sell = col_nvts3.button("GO!_SELL_NVTS") # Unique Key
        if GO_NVTS_sell:
            client.update({'field5': NVTS_ASSET_LAST - p8})
            col_nvts3.write(f"Updated NVTS Asset: {NVTS_ASSET_LAST - p8}")
            st.rerun()

    nvts_current_price = get_current_price('NVTS')
    pv_nvts = nvts_current_price * x_7 if x_7 > 0 else 0
    st.write(nvts_current_price, pv_nvts, '(', round(pv_nvts - 1500,2), ')')

    col_nvts4, col_nvts5, col_nvts6 = st.columns(3)
    st.write('buy', '    ', 'A', u8, 'P', u7  , 'C',u9 ) # Sell params (u7,u8,u9) for UI "buy"
    buy_NVTS = col_nvts6.checkbox('buy_match_NVTS', key='buy_match_nvts')
    if buy_NVTS:
        GO_NVTS_Buy = col_nvts6.button("GO!_BUY_NVTS") # Unique Key
        if GO_NVTS_Buy:
            client.update({'field5': NVTS_ASSET_LAST + u8})
            col_nvts6.write(f"Updated NVTS Asset: {NVTS_ASSET_LAST  + u8}")
            st.rerun()
st.write("_____")

# --- QXO ---
qxo_action_value = 0
if df_7_5 is not None and 'action' in df_7_5.columns and len(df_7_5.action.values) > (1+nex):
    action_val_qxo = df_7_5.action.values[1+nex]
    if pd.isna(action_val_qxo) or action_val_qxo == "":
        action_val_qxo = 0
    qxo_action_value = toggle(action_val_qxo) if Nex_day_sell == 1 else action_val_qxo
else:
    st.warning("QXO action data not available for current/next day.")

Limut_Order_QXO = st.checkbox('Limut_Order_QXO', value=bool(qxo_action_value), key='limit_qxo')
if Limut_Order_QXO:
    st.write('sell', '    ', 'A', q5, 'P', q4, 'C', q6) # Buy params (q4,q5,q6) for UI "sell"

    col_qxo1, col_qxo2, col_qxo3 = st.columns(3)
    sell_QXO = col_qxo3.checkbox('sell_match_QXO', key='sell_match_qxo')

    if sell_QXO:
        GO_QXO_sell = col_qxo3.button("GO!_SELL_QXO") # Unique Key
        if GO_QXO_sell:
            client.update({'field6': QXO_ASSET_LAST - q5})
            col_qxo3.write(f"Updated QXO Asset: {QXO_ASSET_LAST - q5}")
            st.rerun()
    
    qxo_current_price = get_current_price('QXO')
    pv_qxo = qxo_current_price * x_8 if x_8 > 0 else 0
    st.write(qxo_current_price, pv_qxo, '(', round(pv_qxo - 1500,2), ')')

    col_qxo4, col_qxo5, col_qxo6 = st.columns(3)
    st.write('buy', '    ', 'A', q2, 'P', q1, 'C', q3) # Sell params (q1,q2,q3) for UI "buy"
    buy_QXO = col_qxo6.checkbox('buy_match_QXO', key='buy_match_qxo')
    if buy_QXO:
        GO_QXO_Buy = col_qxo6.button("GO!_BUY_QXO") # Unique Key
        if GO_QXO_Buy:
            client.update({'field6': QXO_ASSET_LAST + q2})
            col_qxo6.write(f"Updated QXO Asset: {QXO_ASSET_LAST + q2}")
            st.rerun()
st.write("_____")

# --- RXRX (Added Section) ---
rxrx_action_value = 0
if df_7_6 is not None and 'action' in df_7_6.columns and len(df_7_6.action.values) > (1+nex):
    action_val_rxrx = df_7_6.action.values[1+nex]
    if pd.isna(action_val_rxrx) or action_val_rxrx == "": # Handle potential empty string
        action_val_rxrx = 0 # Default to 0 if value is NaN or empty
    rxrx_action_value = toggle(action_val_rxrx) if Nex_day_sell == 1 else action_val_rxrx
else:
    st.warning("RXRX action data not available for current/next day.")


Limut_Order_RXRX = st.checkbox('Limut_Order_RXRX', value=bool(rxrx_action_value), key='limit_rxrx')
if Limut_Order_RXRX:
    # For UI "sell" (target higher price), use parameters from buy() function: r4, r5, r6
    st.write('sell', '    ', 'A', r5, 'P', r4, 'C', r6)

    col_rxrx1, col_rxrx2, col_rxrx3 = st.columns(3)
    sell_RXRX = col_rxrx3.checkbox('sell_match_RXRX', key='sell_match_rxrx')

    if sell_RXRX:
        GO_RXRX_sell = col_rxrx3.button("GO_RXRX_SELL_MATCH") # Unique key
        if GO_RXRX_sell:
            client.update({'field7': RXRX_ASSET_LAST - r5}) # r5 is amount from buy()
            col_rxrx3.write(f"Updated RXRX Asset: {RXRX_ASSET_LAST - r5}")
            st.rerun()

    rxrx_current_price = get_current_price('RXRX')
    pv_rxrx = rxrx_current_price * x_9 if x_9 > 0 else 0
    st.write(rxrx_current_price, pv_rxrx, '(', round(pv_rxrx - 1500,2), ')')

    col_rxrx4, col_rxrx5, col_rxrx6 = st.columns(3)
    # For UI "buy" (target lower price), use parameters from sell() function: r1, r2, r3
    st.write('buy', '    ', 'A', r2, 'P', r1, 'C', r3)
    buy_RXRX = col_rxrx6.checkbox('buy_match_RXRX', key='buy_match_rxrx')
    if buy_RXRX:
        GO_RXRX_Buy = col_rxrx6.button("GO_RXRX_BUY_MATCH") # Unique key
        if GO_RXRX_Buy:
            client.update({'field7': RXRX_ASSET_LAST + r2}) # r2 is amount from sell()
            col_rxrx6.write(f"Updated RXRX Asset: {RXRX_ASSET_LAST + r2}")
            st.rerun()
st.write("_____")


if st.button("RERUN"):
    st.rerun()

# except:pass # Original except:pass was commented out
