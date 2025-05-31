import streamlit as st
import numpy as np
import datetime
import thingspeak
import pandas as pd
import yfinance as yf
import json

st.set_page_config(page_title="Monitor", page_icon="📈" , layout="wide" )
channel_id = 2528199
write_api_key = '2E65V8XEIPH9B2VV' # Replace with your actual write API key if necessary
client = thingspeak.Channel(channel_id, write_api_key , fmt='json')

def sell (asset = 0 , fix_c=1500 , Diff=60):
    if asset == 0: # Avoid division by zero
        return 0, 0, 0
    s1 =  (1500-Diff) /asset
    s2 =  round(s1, 2)
    if s2 == 0: # Avoid division by zero in subsequent steps
        return 0,0,0
    s3 =  s2  *asset
    s4 =  abs(s3 - fix_c)
    s5 =  round( s4 / s2 )
    s6 =  s5*s2
    s7 =  (asset * s2) + s6
    return s2 , s5 , round(s7, 2)

def buy (asset = 0 , fix_c=1500 , Diff=60):
    if asset == 0: # Avoid division by zero
        return 0, 0, 0
    b1 =  (1500+Diff) /asset
    b2 =  round(b1, 2)
    if b2 == 0: # Avoid division by zero in subsequent steps
        return 0,0,0
    b3 =  b2 *asset
    b4 =  abs(b3 - fix_c)
    b5 =  round( b4 / b2 )
    b6 =  b5*b2
    b7 =  (asset * b2) - b6
    return b2 , b5 , round(b7, 2)

channel_id_2 = 2385118
write_api_key_2 = 'IPSG3MMMBJEB9DY8' # Replace with your actual write API key if necessary
client_2 = thingspeak.Channel(channel_id_2, write_api_key_2 , fmt='json' )

def Monitor (Ticker = 'FFWM' , field = 2  ):
    try:
        tickerData = yf.Ticker(Ticker)
        # Use a short period for history to speed up, 'max' can be slow and unnecessary if only recent data is needed for 'action'
        tickerData_history = tickerData.history(period='1y')[['Close']] 
        tickerData_history = round(tickerData_history , 3 )
        # Ensure index is timezone-aware before converting
        if tickerData_history.index.tz is None:
            tickerData_history.index = tickerData_history.index.tz_localize('UTC') # Or appropriate original timezone
        tickerData_history.index = tickerData_history.index.tz_convert(tz='Asia/Bangkok') # Corrected tz name
        
        current_year_start = pd.Timestamp(datetime.datetime.now().year, 1, 1, 12, 0, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=7)))
        filter_date = current_year_start.strftime('%Y-%m-%d %H:%M:%S%z')
        tickerData_history = tickerData_history[tickerData_history.index >= filter_date]

        fx = client_2.get_field_last(field='{}'.format(field))
        if fx: # Check if fx is not None
            fx_data = json.loads(fx)
            if "field{}".format(field) in fx_data and fx_data["field{}".format(field)] is not None:
                 fx_js = int(fx_data["field{}".format(field)])
            else:
                 st.warning(f"Field {field} not found or is null in ThingSpeak channel 2. Using default seed 0.")
                 fx_js = 0 # Default seed
        else:
            st.warning(f"Could not retrieve data for field {field} from ThingSpeak channel 2. Using default seed 0.")
            fx_js = 0 # Default seed
            
        rng = np.random.default_rng(fx_js)
        data = rng.integers(2, size = len(tickerData_history))
        tickerData_history['action'] = data
        tickerData_history['index'] = [ i+1 for i in range(len(tickerData_history))]

        tickerData_1 = pd.DataFrame(columns=(tickerData_history.columns))
        # Ensure 'action' column exists before trying to assign to it if tickerData_history is empty
        if 'action' not in tickerData_1.columns:
            tickerData_1['action'] = None # Or some other default
            
        tickerData_1['action'] =  [ i for i in range(5)] # This seems to be placeholder, ensure it's intended
        tickerData_1.index = ['+0' , "+1" , "+2" , "+3" , "+4"]
        df = pd.concat([tickerData_history , tickerData_1], axis=0).fillna("")
        
        # Re-seed or use a new RNG if fresh random numbers are needed for the combined df
        rng_df = np.random.default_rng(fx_js) # Or a different seed/RNG
        df['action'] = rng_df.integers(2, size = len(df))
        return df.tail(7) , fx_js
    except Exception as e:
        st.error(f"Error in Monitor for {Ticker} (Field {field}): {e}")
        # Return a default DataFrame structure to prevent downstream errors
        default_cols = ['Close', 'action', 'index']
        empty_df_hist = pd.DataFrame(columns=default_cols, index=pd.to_datetime([]).tz_localize('UTC').tz_convert('Asia/Bangkok'))
        empty_df_placeholder = pd.DataFrame(columns=default_cols, index=['+0', '+1', '+2', '+3', '+4'])
        empty_df_placeholder['action'] = [0,0,0,0,0] # default action
        empty_df = pd.concat([empty_df_hist, empty_df_placeholder]).fillna("")
        empty_df['action'] = np.random.default_rng(0).integers(2, size=len(empty_df)) # default random action
        return empty_df.tail(7), 0


df_7   , fx_js   = Monitor(Ticker = 'FFWM', field = 2    )
df_7_1 , fx_js_1  = Monitor(Ticker = 'NEGG', field = 3    )
df_7_2 , fx_js_2  = Monitor(Ticker = 'RIVN', field = 4 )
df_7_3 , fx_js_3  = Monitor(Ticker = 'APLS', field = 5 )
df_7_4 , fx_js_4  = Monitor(Ticker = 'NVTS', field = 6 )
df_7_5 , fx_js_5  = Monitor(Ticker = 'QXO', field = 7 )
df_7_6 , fx_js_6  = Monitor(Ticker = 'RXRX', field = 8 ) # Added RXRX

nex = 0
Nex_day_sell = 0
toggle = lambda x : 1 - x if x in [0,1,'0','1'] else 0 # Make toggle more robust

Nex_day_ = st.checkbox('nex_day')
if Nex_day_ :
    st.write( "value = " , nex)
    nex_col , Nex_day_sell_col ,_,_,_  = st.columns(5)

    if nex_col.button("Nex_day"):
        nex = 1
        st.write( "value = " , nex) # This will only show if rerun happens, consider st.experimental_rerun or session state

    if Nex_day_sell_col.button("Nex_day_sell"):
        nex = 1
        Nex_day_sell = 1
        st.write( "value = " , nex) # Ditto
        st.write( "Nex_day_sell = " , Nex_day_sell) # Ditto

st.write("_____")

col13, col16, col14, col15, col17, col18, col19, col20, col21 = st.columns(9) # Increased to 9 columns

x_2 = col16.number_input('Diff', step=1 , value= 60   )

Start = col13.checkbox('start')
if Start :
    thingspeak_1 = col13.checkbox('@_FFWM_ASSET')
    if thingspeak_1 :
        add_1 = col13.number_input('@_FFWM_ASSET', step=0.001 ,  value=0., key='add_1_ffwm')
        _FFWM_ASSET = col13.button("GO!")
        if _FFWM_ASSET :
            client.update(  {'field1': add_1 } )
            col13.write(add_1)

    thingspeak_2 = col13.checkbox('@_NEGG_ASSET')
    if thingspeak_2 :
        add_2 = col13.number_input('@_NEGG_ASSET', step=0.001 ,  value=0., key='add_2_negg')
        _NEGG_ASSET = col13.button("GO! ") # 1 space
        if _NEGG_ASSET :
            client.update(  {'field2': add_2 }  )
            col13.write(add_2)

    thingspeak_3 = col13.checkbox('@_RIVN_ASSET')
    if thingspeak_3 :
        add_3 = col13.number_input('@_RIVN_ASSET', step=0.001 ,  value=0., key='add_3_rivn')
        _RIVN_ASSET = col13.button("GO!  ") # 2 spaces
        if _RIVN_ASSET :
            client.update(  {'field3': add_3 }  )
            col13.write(add_3)

    thingspeak_4 = col13.checkbox('@_APLS_ASSET')
    if thingspeak_4 :
        add_4 = col13.number_input('@_APLS_ASSET', step=0.001 ,  value=0., key='add_4_apls')
        _APLS_ASSET = col13.button("GO!   ") # 3 spaces
        if _APLS_ASSET :
            client.update(  {'field4': add_4 }  )
            col13.write(add_4)

    thingspeak_5 = col13.checkbox('@_NVTS_ASSET')
    if thingspeak_5:
        add_5 = col13.number_input('@_NVTS_ASSET', step=0.001, value= 0., key='add_5_nvts')
        _NVTS_ASSET = col13.button("GO!    ") # 4 spaces
        if _NVTS_ASSET:
            client.update({'field5': add_5})
            col13.write(add_5)

    thingspeak_6 = col13.checkbox('@_QXO_ASSET')
    if thingspeak_6:
        add_6 = col13.number_input('@_QXO_ASSET', step=0.001, value=0., key='add_6_qxo')
        _QXO_ASSET = col13.button("GO!     ") # 5 spaces
        if _QXO_ASSET:
            client.update({'field6': add_6})
            col13.write(add_6)
    
    thingspeak_7 = col13.checkbox('@_RXRX_ASSET') # Added for RXRX
    if thingspeak_7:
        add_7 = col13.number_input('@_RXRX_ASSET', step=0.001, value=0., key='add_7_rxrx')
        _RXRX_ASSET = col13.button("GO!      ") # 6 spaces
        if _RXRX_ASSET:
            client.update({'field7': add_7}) # Use field7 for RXRX
            col13.write(add_7)


# Helper function to safely get and eval asset data from ThingSpeak
def get_asset_value(client, field_name_str_num):
    raw_data = client.get_field_last(field=field_name_str_num)
    if raw_data:
        try:
            loaded_json = json.loads(raw_data)
            if field_name_str_num in loaded_json and loaded_json[field_name_str_num] is not None:
                return float(loaded_json[field_name_str_num]) # Using float instead of eval for safety
            else:
                st.warning(f"Field {field_name_str_num} is null or not in response: {raw_data}. Defaulting to 0.")
                return 0.0
        except json.JSONDecodeError:
            st.error(f"Failed to decode JSON for {field_name_str_num}: {raw_data}. Defaulting to 0.")
            return 0.0
        except (ValueError, TypeError) as e:
            st.error(f"Error converting value for {field_name_str_num} (value: {loaded_json.get(field_name_str_num)}): {e}. Defaulting to 0.")
            return 0.0
    st.warning(f"No data retrieved for {field_name_str_num}. Defaulting to 0.")
    return 0.0

FFWM_ASSET_LAST = get_asset_value(client, 'field1')
NEGG_ASSET_LAST = get_asset_value(client, 'field2')
RIVN_ASSET_LAST = get_asset_value(client, 'field3')
APLS_ASSET_LAST = get_asset_value(client, 'field4')
NVTS_ASSET_LAST = get_asset_value(client, 'field5')
QXO_ASSET_LAST = get_asset_value(client, 'field6')
RXRX_ASSET_LAST = get_asset_value(client, 'field7') # Added for RXRX


x_3 = col14.number_input('NEGG_ASSET', step=0.001 ,  value= NEGG_ASSET_LAST, key='x3_negg' )
x_4 = col15.number_input('FFWM_ASSET', step=0.001  , value= FFWM_ASSET_LAST, key='x4_ffwm'   )
x_5 = col17.number_input('RIVN_ASSET', step=0.001  , value= RIVN_ASSET_LAST, key='x5_rivn'   )
x_6 = col18.number_input('APLS_ASSET', step=0.001  , value= APLS_ASSET_LAST, key='x6_apls'   )
x_7 = col19.number_input('NVTS_ASSET', step=0.001  , value= NVTS_ASSET_LAST, key='x7_nvts' )

QXO_OPTION = 79.
QXO_REAL   =  col20.number_input('QXO_ASSET (LV:79@19.0)', step=0.001  , value=  QXO_ASSET_LAST, key='qxo_real')
x_8 =  QXO_OPTION  + QXO_REAL

x_9 = col21.number_input('RXRX_ASSET', step=0.001, value=RXRX_ASSET_LAST, key='x9_rxrx') # Added for RXRX

st.write("_____")

# try: # Consider removing try-except for debugging or making it more specific

s8 , s9 , s10 =  sell( asset = x_3 , Diff= x_2)
s11 , s12 , s13 =  sell(asset = x_4 , Diff= x_2)
b8 , b9 , b10 =  buy(asset = x_3 , Diff= x_2)
b11 , b12 , b13 =  buy(asset = x_4 , Diff= x_2)
u1 , u2 , u3 = sell( asset = x_5 , Diff= x_2)
u4 , u5 , u6 = buy( asset = x_5 , Diff= x_2)
p1 , p2 , p3 = sell( asset = x_6 , Diff= x_2)
p4 , p5 , p6 = buy( asset = x_6 , Diff= x_2)
u7 , u8 , u9 = sell( asset = x_7 , Diff= x_2)
p7 , p8 , p9 = buy( asset = x_7 , Diff= x_2)
q1, q2, q3 = sell(asset=x_8, Diff=x_2)
q4, q5, q6 = buy(asset=x_8, Diff=x_2)
r1, r2, r3 = sell(asset=x_9, Diff=x_2) # For RXRX
r4, r5, r6 = buy(asset=x_9, Diff=x_2)  # For RXRX


# Helper function to get action value safely
def get_action_value(df_action_series, index, default_value=0):
    try:
        val = df_action_series.values[index]
        if pd.isna(val) or val == "": # Check for NaN or empty string
             return default_value
        return int(val)
    except IndexError:
        st.warning(f"Index {index} out of bounds for action values. Defaulting to {default_value}.")
        return default_value
    except ValueError:
        st.warning(f"Could not convert action value '{val}' to int. Defaulting to {default_value}.")
        return default_value

# --- NEGG ---
action_negg = get_action_value(df_7_1.action, 1 + nex)
Limut_Order_NEGG = st.checkbox('Limut_Order_NEGG', value = bool(np.where( Nex_day_sell == 1 ,  toggle(action_negg) , action_negg )))
if Limut_Order_NEGG :
    st.write( 'sell NEGG:' , '   ' ,'A:', b9  , 'P:' , b8 ,'C:' ,b10   )
    col1, col2 , col3  = st.columns(3)
    sell_negg = col3.checkbox('sell_match_NEGG', key='sell_negg_cb')
    if sell_negg :
        GO_NEGG_SELL = col3.button("GO!   ", key='go_negg_sell') # 3 spaces
        if GO_NEGG_SELL :
            client.update(  {'field2': NEGG_ASSET_LAST - b9 } )
            col3.write(f"Updated NEGG Asset: {NEGG_ASSET_LAST - b9}")
            st.rerun()
    try:
        pv_negg_price = yf.Ticker('NEGG').fast_info.get('lastPrice', 0)
        pv_negg =  pv_negg_price * x_3
        st.write('NEGG Last Price:', pv_negg_price , 'PV:', pv_negg  ,'(',  pv_negg - 1500 ,')',  )
    except Exception as e: st.warning(f"Could not get NEGG price: {e}")
    st.write( 'buy NEGG:' , '    ','A:',  s9  ,  'P:' , s8 , 'C:' ,s10   )
    col4, col5 , col6  = st.columns(3)
    buy_negg = col6.checkbox('buy_match_NEGG', key='buy_negg_cb')
    if buy_negg :
        GO_NEGG_Buy = col6.button("GO!    ", key='go_negg_buy') # 4 spaces
        if GO_NEGG_Buy :
            client.update(  {'field2': NEGG_ASSET_LAST + s9  } )
            col6.write(f"Updated NEGG Asset: {NEGG_ASSET_LAST + s9}")
            st.rerun()
st.write("_____")

# --- FFWM ---
action_ffwm = get_action_value(df_7.action, 1 + nex)
Limut_Order_FFWM = st.checkbox('Limut_Order_FFWM',  value = bool(np.where(  Nex_day_sell == 1 ,  toggle(action_ffwm)  , action_ffwm  )))
if Limut_Order_FFWM :
    st.write( 'sell FFWM:' , '    ' , 'A:', b12 , 'P:' , b11  , 'C:' , b13   )
    col7, col8 , col9  = st.columns(3)
    sell_ffwm = col9.checkbox('sell_match_FFWM', key='sell_ffwm_cb')
    if sell_ffwm :
        GO_ffwm_sell = col9.button("GO!     ", key='go_ffwm_sell') # 5 spaces
        if GO_ffwm_sell :
            client.update(  {'field1': FFWM_ASSET_LAST - b12  } )
            col9.write(f"Updated FFWM Asset: {FFWM_ASSET_LAST - b12}")
            st.rerun()
    try:
        pv_ffwm_price = yf.Ticker('FFWM').fast_info.get('lastPrice', 0)
        pv_ffwm =   pv_ffwm_price * x_4
        st.write('FFWM Last Price:', pv_ffwm_price , 'PV:', pv_ffwm ,'(',  pv_ffwm - 1500 ,')', )
    except Exception as e: st.warning(f"Could not get FFWM price: {e}")
    col10, col11 , col12  = st.columns(3)
    st.write(  'buy FFWM:' , '    ', 'A:', s12 , 'P:' , s11  , 'C:'  , s13   )
    buy_ffwm = col12.checkbox('buy_match_FFWM', key='buy_ffwm_cb')
    if buy_ffwm :
        GO_ffwm_Buy = col12.button("GO!      ", key='go_ffwm_buy') # 6 spaces (Same as RXRX setup, ensure it's okay or change)
        if GO_ffwm_Buy :
            client.update(  {'field1': FFWM_ASSET_LAST + s12  } )
            col12.write(f"Updated FFWM Asset: {FFWM_ASSET_LAST + s12}")
            st.rerun()
st.write("_____")

# --- RIVN ---
action_rivn = get_action_value(df_7_2.action, 1 + nex)
Limut_Order_RIVN = st.checkbox('Limut_Order_RIVN',value = bool(np.where(  Nex_day_sell == 1 ,  toggle(action_rivn)  , action_rivn  )))
if Limut_Order_RIVN :
    st.write( 'sell RIVN:' , '    ' , 'A:', u5 , 'P:' , u4  , 'C:' , u6   )
    col77, col88 , col99  = st.columns(3)
    sell_RIVN = col99.checkbox('sell_match_RIVN', key='sell_rivn_cb')
    if sell_RIVN :
        GO_RIVN_sell = col99.button("GO!       ", key='go_rivn_sell') # 7 spaces
        if GO_RIVN_sell :
            client.update(  {'field3': RIVN_ASSET_LAST - u5  } )
            col99.write(f"Updated RIVN Asset: {RIVN_ASSET_LAST - u5}")
            st.rerun()
    try:
        pv_rivn_price = yf.Ticker('RIVN').fast_info.get('lastPrice', 0)
        pv_rivn =   pv_rivn_price * x_5
        st.write('RIVN Last Price:', pv_rivn_price , 'PV:', pv_rivn ,'(',  pv_rivn - 1500 ,')', )
    except Exception as e: st.warning(f"Could not get RIVN price: {e}")
    col100 , col111 , col122  = st.columns(3)
    st.write(  'buy RIVN:' , '    ', 'A:', u2 , 'P:' , u1  , 'C:'  , u3   )
    buy_RIVN = col122.checkbox('buy_match_RIVN', key='buy_rivn_cb')
    if buy_RIVN :
        GO_RIVN_Buy = col122.button("GO!        ", key='go_rivn_buy') # 8 spaces
        if GO_RIVN_Buy :
            client.update(  {'field3': RIVN_ASSET_LAST + u2  } )
            col122.write(f"Updated RIVN Asset: {RIVN_ASSET_LAST + u2}")
            st.rerun()
st.write("_____")

# --- APLS ---
action_apls = get_action_value(df_7_3.action, 1 + nex)
Limut_Order_APLS = st.checkbox('Limut_Order_APLS',value = bool(np.where(  Nex_day_sell == 1 ,  toggle(action_apls)  , action_apls )))
if Limut_Order_APLS :
    st.write( 'sell APLS:' , '    ' , 'A:', p5 , 'P:' , p4  , 'C:' , p6   )
    col7777, col8888 , col9999  = st.columns(3)
    sell_APLS = col9999.checkbox('sell_match_APLS', key='sell_apls_cb')
    if sell_APLS :
        GO_APLS_sell = col9999.button("GO!         ", key='go_apls_sell') # 9 spaces
        if GO_APLS_sell :
            client.update(  {'field4': APLS_ASSET_LAST - p5  } )
            col9999.write(f"Updated APLS Asset: {APLS_ASSET_LAST - p5}")
            st.rerun()
    try:
        pv_apls_price = yf.Ticker('APLS').fast_info.get('lastPrice', 0)
        pv_apls =   pv_apls_price * x_6
        st.write('APLS Last Price:', pv_apls_price , 'PV:', pv_apls ,'(',  pv_apls - 1500 ,')', )
    except Exception as e: st.warning(f"Could not get APLS price: {e}")
    col1000 , col1111 , col1222  = st.columns(3)
    st.write(  'buy APLS:' , '    ', 'A:', p2 , 'P:' , p1  , 'C:'  , p3   )
    buy_APLS = col1222.checkbox('buy_match_APLS', key='buy_apls_cb')
    if buy_APLS :
        GO_APLS_Buy = col1222.button("GO!          ", key='go_apls_buy') # 10 spaces
        if GO_APLS_Buy :
            client.update(  {'field4': APLS_ASSET_LAST + p2  } )
            col1222.write(f"Updated APLS Asset: {APLS_ASSET_LAST + p2}")
            st.rerun()
st.write("_____")

# --- NVTS ---
action_nvts = get_action_value(df_7_4.action, 1 + nex)
Limut_Order_NVTS = st.checkbox('Limut_Order_NVTS', value=bool(np.where(  Nex_day_sell == 1 ,  toggle(action_nvts)  , action_nvts  )))
if Limut_Order_NVTS:
    st.write('sell NVTS:', '    ', 'A:', p8 , 'P:', p7  , 'C:', p9   )
    col_nvts1, col_nvts2, col_nvts3 = st.columns(3)
    sell_NVTS = col_nvts3.checkbox('sell_match_NVTS', key='sell_nvts_cb')
    if sell_NVTS:
        GO_NVTS_sell = col_nvts3.button("GO!           ", key='go_nvts_sell') # 11 spaces
        if GO_NVTS_sell:
            client.update({'field5': NVTS_ASSET_LAST - p8})
            col_nvts3.write(f"Updated NVTS Asset: {NVTS_ASSET_LAST - p8}")
            st.rerun()
    try:
        pv_nvts_price = yf.Ticker('NVTS').fast_info.get('lastPrice', 0)
        pv_nvts = pv_nvts_price * x_7
        st.write('NVTS Last Price:', pv_nvts_price, 'PV:', pv_nvts, '(', pv_nvts - 1500, ')')
    except Exception as e: st.warning(f"Could not get NVTS price: {e}")
    col_nvts4, col_nvts5, col_nvts6 = st.columns(3)
    st.write('buy NVTS:', '    ', 'A:', u8, 'P:', u7  , 'C:',u9 )
    buy_NVTS = col_nvts6.checkbox('buy_match_NVTS', key='buy_nvts_cb')
    if buy_NVTS:
        GO_NVTS_Buy = col_nvts6.button("GO!            ", key='go_nvts_buy') # 12 spaces
        if GO_NVTS_Buy:
            client.update({'field5': NVTS_ASSET_LAST + u8})
            col_nvts6.write(f"Updated NVTS Asset: {NVTS_ASSET_LAST  + u8}")
            st.rerun()
st.write("_____")

# --- QXO ---
action_qxo = get_action_value(df_7_5.action, 1 + nex)
Limut_Order_QXO = st.checkbox('Limut_Order_QXO', value=bool(np.where(Nex_day_sell == 1, toggle(action_qxo), action_qxo)))
if Limut_Order_QXO:
    st.write('sell QXO:', '    ', 'A:', q5, 'P:', q4, 'C:', q6)
    col_qxo1, col_qxo2, col_qxo3 = st.columns(3)
    sell_QXO = col_qxo3.checkbox('sell_match_QXO', key='sell_qxo_cb')
    if sell_QXO:
        GO_QXO_sell = col_qxo3.button("GO!             ", key='go_qxo_sell') # 13 spaces
        if GO_QXO_sell:
            client.update({'field6': QXO_ASSET_LAST - q5}) # QXO uses QXO_ASSET_LAST (which is QXO_REAL effectively)
            col_qxo3.write(f"Updated QXO Asset: {QXO_ASSET_LAST - q5}")
            st.rerun()
    try:
        pv_qxo_price = yf.Ticker('QXO').fast_info.get('lastPrice', 0)
        pv_qxo = pv_qxo_price * x_8 # x_8 includes QXO_OPTION
        st.write('QXO Last Price:', pv_qxo_price, 'PV:', pv_qxo, '(', pv_qxo - 1500, ')')
    except Exception as e: st.warning(f"Could not get QXO price: {e}")
    col_qxo4, col_qxo5, col_qxo6 = st.columns(3)
    st.write('buy QXO:', '    ', 'A:', q2, 'P:', q1, 'C:', q3)
    buy_QXO = col_qxo6.checkbox('buy_match_QXO', key='buy_qxo_cb')
    if buy_QXO:
        GO_QXO_Buy = col_qxo6.button("GO!              ", key='go_qxo_buy') # 14 spaces
        if GO_QXO_Buy:
            client.update({'field6': QXO_ASSET_LAST + q2})
            col_qxo6.write(f"Updated QXO Asset: {QXO_ASSET_LAST + q2}")
            st.rerun()
st.write("_____")

# --- RXRX --- Added Section
action_rxrx = get_action_value(df_7_6.action, 1 + nex) # Use df_7_6 for RXRX
Limut_Order_RXRX = st.checkbox('Limut_Order_RXRX', value=bool(np.where(Nex_day_sell == 1, toggle(action_rxrx), action_rxrx)))
if Limut_Order_RXRX:
    st.write('sell RXRX:', '    ', 'A:', r5, 'P:', r4, 'C:', r6) # Use r-variables for RXRX
    col_rxrx1, col_rxrx2, col_rxrx3 = st.columns(3)
    sell_RXRX = col_rxrx3.checkbox('sell_match_RXRX', key='sell_rxrx_cb')
    if sell_RXRX:
        GO_RXRX_sell = col_rxrx3.button("GO!               ", key='go_rxrx_sell') # 15 spaces
        if GO_RXRX_sell:
            client.update({'field7': RXRX_ASSET_LAST - r5}) # Use field7 for RXRX
            col_rxrx3.write(f"Updated RXRX Asset: {RXRX_ASSET_LAST - r5}")
            st.rerun()
    try:
        pv_rxrx_price = yf.Ticker('RXRX').fast_info.get('lastPrice', 0)
        pv_rxrx = pv_rxrx_price * x_9 # x_9 is the asset value for RXRX
        st.write('RXRX Last Price:', pv_rxrx_price, 'PV:', pv_rxrx, '(', pv_rxrx - 1500, ')')
    except Exception as e: st.warning(f"Could not get RXRX price: {e}")

    col_rxrx4, col_rxrx5, col_rxrx6 = st.columns(3)
    st.write('buy RXRX:', '    ', 'A:', r2, 'P:', r1, 'C:', r3) # Use r-variables for RXRX
    buy_RXRX = col_rxrx6.checkbox('buy_match_RXRX', key='buy_rxrx_cb')
    if buy_RXRX:
        GO_RXRX_Buy = col_rxrx6.button("GO!                ", key='go_rxrx_buy') # 16 spaces
        if GO_RXRX_Buy:
            client.update({'field7': RXRX_ASSET_LAST + r2}) # Use field7 for RXRX
            col_rxrx6.write(f"Updated RXRX Asset: {RXRX_ASSET_LAST + r2}")
            st.rerun()
st.write("_____")


if st.button("RERUN Manual"): # Changed label slightly in case "RERUN" is a special keyword for Streamlit features
    st.rerun()

# except Exception as e: # General exception handling
#    st.error(f"An unexpected error occurred: {e}")
#    pass
