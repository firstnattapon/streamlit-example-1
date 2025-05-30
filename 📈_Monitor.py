import streamlit as st
import numpy as np
import datetime
import thingspeak
import pandas as pd
import yfinance as yf
import json

st.set_page_config(page_title="Monitor", page_icon="📈" , layout="wide" )
channel_id = 2528199
write_api_key = '2E65V8XEIPH9B2VV' # Keep your actual API keys private
client = thingspeak.Channel(channel_id, write_api_key , fmt='json')

def sell (asset = 0 , fix_c=1500 , Diff=60):
    s1 =  (1500-Diff) /asset if asset != 0 else 0 # Avoid division by zero
    s2 =  round(s1, 2)
    s3 =  s2  *asset
    s4 =  abs(s3 - fix_c)
    s5 =  round( s4 / s2 ) if s2 != 0 else 0 # Avoid division by zero
    s6 =  s5*s2
    s7 =  (asset * s2) + s6
    return s2 , s5 , round(s7, 2)

def buy (asset = 0 , fix_c=1500 , Diff=60):
    b1 =  (1500+Diff) /asset if asset != 0 else 0 # Avoid division by zero
    b2 =  round(b1, 2)
    b3 =  b2 *asset
    b4 =  abs(b3 - fix_c)
    b5 =  round( b4 / b2 ) if b2 != 0 else 0 # Avoid division by zero
    b6 =  b5*b2
    b7 =  (asset * b2) - b6
    return b2 , b5 , round(b7, 2)

channel_id_2 = 2385118
write_api_key_2 = 'IPSG3MMMBJEB9DY8' # Keep your actual API keys private
client_2 = thingspeak.Channel(channel_id_2, write_api_key_2 , fmt='json' )

def Monitor (Ticker = 'FFWM' , field = 2  ):
    try:
        tickerData = yf.Ticker(Ticker)
        tickerData = round(tickerData.history(period= 'max' )[['Close']] , 3 )
        tickerData.index = tickerData.index.tz_convert(tz='Asia/Bangkok') # Corrected tz
        filter_date = '2023-01-01 12:00:00+07:00'
        tickerData = tickerData[tickerData.index >= filter_date]

        fx = client_2.get_field_last(field='{}'.format(field))
        fx_js_data = json.loads(fx)
        # Check if field exists and is not None
        if fx_js_data and "field{}".format(field) in fx_js_data and fx_js_data["field{}".format(field)] is not None:
            fx_js = int(fx_js_data["field{}".format(field)])
        else:
            st.warning(f"Could not retrieve valid data for Ticker {Ticker}, field {field}. Using default seed 0.")
            fx_js = 0 # Default seed or handle error as appropriate

        rng = np.random.default_rng(fx_js)
        data = rng.integers(2, size = len(tickerData))
        tickerData['action'] = data
        tickerData['index'] = [ i+1 for i in range(len(tickerData))]

        tickerData_1 = pd.DataFrame(columns=(tickerData.columns))
        # Ensure 'action' column is present before assigning to it
        if 'action' not in tickerData_1.columns:
            tickerData_1['action'] = np.nan # or some other default
        tickerData_1['action'] =  [ i for i in range(5)] # This overrides the previous line, consider intent
        tickerData_1.index = ['+0' , "+1" , "+2" , "+3" , "+4"]
        df = pd.concat([tickerData , tickerData_1], axis=0).fillna("")
        rng = np.random.default_rng(fx_js) # Re-seeding, ensure this is intended
        df['action'] = rng.integers(2, size = len(df))
        return df.tail(7) , fx_js
    except Exception as e:
        st.error(f"Error in Monitor for {Ticker}: {e}")
        # Return a default DataFrame structure to avoid further errors downstream
        empty_cols = ['Close', 'action', 'index']
        empty_df = pd.DataFrame(columns=empty_cols, index=[f'+{i}' for i in range(5)] + ['dummy1', 'dummy2'])
        empty_df = empty_df.fillna("")
        empty_df['action'] = 0 # default action
        return empty_df.tail(7), 0


df_7   , fx_js    = Monitor(Ticker = 'FFWM', field = 2    )
df_7_1 , fx_js_1  = Monitor(Ticker = 'NEGG', field = 3    )
df_7_2 , fx_js_2  = Monitor(Ticker = 'RIVN', field = 4 )
df_7_3 , fx_js_3  = Monitor(Ticker = 'APLS', field = 5 )
df_7_4 , fx_js_4  = Monitor(Ticker = 'NVTS', field = 6 )
df_7_5 , fx_js_5  = Monitor(Ticker = 'QXO', field = 7 )
df_7_6 , fx_js_6  = Monitor(Ticker = 'RXRX', field = 8 ) # New Ticker RXRX

nex = 0
Nex_day_sell = 0
toggle = lambda x : 1 - x

Nex_day_ = st.checkbox('nex_day')
if Nex_day_ :
    st.write( "value = " , nex) # This will always show initial nex=0
    nex_col , Nex_day_sell_col ,_,_,_  = st.columns(5)

    if nex_col.button("Nex_day"):
        # nex = 1 # This change is local to this block and won't persist for checkbox logic unless using session state
        st.session_state.nex = 1 # Use session state to make it effective
        st.write( "value = " , st.session_state.nex)
        st.rerun()


    if Nex_day_sell_col.button("Nex_day_sell"):
        # nex = 1
        # Nex_day_sell = 1 # Similar to nex, this needs session state
        st.session_state.nex = 1
        st.session_state.Nex_day_sell = 1
        st.write( "value = " , st.session_state.nex)
        st.write( "Nex_day_sell = " , st.session_state.Nex_day_sell)
        st.rerun()

# Initialize session state variables if they don't exist
if 'nex' not in st.session_state:
    st.session_state.nex = 0
if 'Nex_day_sell' not in st.session_state:
    st.session_state.Nex_day_sell = 0

nex = st.session_state.nex
Nex_day_sell = st.session_state.Nex_day_sell


st.write("_____")

col13, col16, col14, col15, col17, col18, col19, col20, col21 = st.columns(9) # Added col21

x_2 = col16.number_input('Diff', step=1 , value= 60    )

Start = col13.checkbox('start')
if Start :
    thingspeak_1 = col13.checkbox('@_FFWM_ASSET')
    if thingspeak_1 :
        add_1 = col13.number_input('@_FFWM_ASSET_val', step=0.001 ,  value=0., key="add_1_val") # Unique key
        _FFWM_ASSET = col13.button("GO! FFWM") # More specific button label
        if _FFWM_ASSET :
            client.update(  {'field1': add_1 } )
            col13.write(f"FFWM Asset Updated: {add_1}")

    thingspeak_2 = col13.checkbox('@_NEGG_ASSET')
    if thingspeak_2 :
        add_2 = col13.number_input('@_NEGG_ASSET_val', step=0.001 ,  value=0., key="add_2_val")
        _NEGG_ASSET = col13.button("GO! NEGG")
        if _NEGG_ASSET :
            client.update(  {'field2': add_2 }  )
            col13.write(f"NEGG Asset Updated: {add_2}")

    thingspeak_3 = col13.checkbox('@_RIVN_ASSET')
    if thingspeak_3 :
        add_3 = col13.number_input('@_RIVN_ASSET_val', step=0.001 ,  value=0., key="add_3_val")
        _RIVN_ASSET = col13.button("GO! RIVN")
        if _RIVN_ASSET :
            client.update(  {'field3': add_3 }  )
            col13.write(f"RIVN Asset Updated: {add_3}")

    thingspeak_4 = col13.checkbox('@_APLS_ASSET')
    if thingspeak_4 :
        add_4 = col13.number_input('@_APLS_ASSET_val', step=0.001 ,  value=0., key="add_4_val")
        _APLS_ASSET = col13.button("GO! APLS")
        if _APLS_ASSET :
            client.update(  {'field4': add_4 }  )
            col13.write(f"APLS Asset Updated: {add_4}")

    thingspeak_5 = col13.checkbox('@_NVTS_ASSET')
    if thingspeak_5:
        add_5 = col13.number_input('@_NVTS_ASSET_val', step=0.001, value= 0., key="add_5_val")
        _NVTS_ASSET = col13.button("GO! NVTS")
        if _NVTS_ASSET:
            client.update({'field5': add_5})
            col13.write(f"NVTS Asset Updated: {add_5}")
    
    thingspeak_6 = col13.checkbox('@_QXO_ASSET')
    if thingspeak_6:
        add_6 = col13.number_input('@_QXO_ASSET_val', step=0.001, value=0., key="add_6_val")
        _QXO_ASSET = col13.button("GO! QXO")
        if _QXO_ASSET:
            client.update({'field6': add_6})
            col13.write(f"QXO Asset Updated: {add_6}")

    thingspeak_7 = col13.checkbox('@_RXRX_ASSET') # New Ticker RXRX
    if thingspeak_7:
        add_7 = col13.number_input('@_RXRX_ASSET_val', step=0.001, value=0., key="add_7_val")
        _RXRX_ASSET = col13.button("GO! RXRX")
        if _RXRX_ASSET:
            client.update({'field7': add_7}) # Assuming field7 for RXRX
            col13.write(f"RXRX Asset Updated: {add_7}")


def get_thingspeak_field_value(client, field_name, default_value=0.0):
    try:
        last_value_str = client.get_field_last(field=field_name)
        last_value_json = json.loads(last_value_str)
        if field_name in last_value_json and last_value_json[field_name] is not None:
            return float(last_value_json[field_name]) # Use float for asset values
        else:
            # st.warning(f"Field {field_name} not found or is null in ThingSpeak. Using default value {default_value}.")
            return default_value
    except Exception as e:
        # st.error(f"Error fetching ThingSpeak field {field_name}: {e}. Using default value {default_value}.")
        return default_value

FFWM_ASSET_LAST = get_thingspeak_field_value(client, 'field1')
NEGG_ASSET_LAST = get_thingspeak_field_value(client, 'field2')
RIVN_ASSET_LAST = get_thingspeak_field_value(client, 'field3')
APLS_ASSET_LAST = get_thingspeak_field_value(client, 'field4')
NVTS_ASSET_LAST = get_thingspeak_field_value(client, 'field5')
QXO_ASSET_LAST = get_thingspeak_field_value(client, 'field6')
RXRX_ASSET_LAST = get_thingspeak_field_value(client, 'field7') # New Ticker RXRX (using field7)


x_4 = col15.number_input('FFWM_ASSET', step=0.001  , value= FFWM_ASSET_LAST    , key="ffwm_asset_main")
x_3 = col14.number_input('NEGG_ASSET', step=0.001 ,  value= NEGG_ASSET_LAST , key="negg_asset_main")
x_5 = col17.number_input('RIVN_ASSET', step=0.001  , value= RIVN_ASSET_LAST    , key="rivn_asset_main")
x_6 = col18.number_input('APLS_ASSET', step=0.001  , value= APLS_ASSET_LAST    , key="apls_asset_main")
x_7 = col19.number_input('NVTS_ASSET', step=0.001  , value= NVTS_ASSET_LAST , key="nvts_asset_main")

QXO_OPTION = 79.
QXO_REAL   =  col20.number_input('QXO_ASSET (LV:79@19.0)', step=0.001  , value=  QXO_ASSET_LAST, key="qxo_asset_main")      
x_8 =  QXO_OPTION  + QXO_REAL

RXRX_OPTION = 278. # New Ticker RXRX
RXRX_REAL   =  col21.number_input('RXRX_ASSET (LV:278@5.4)', step=0.001  , value=  RXRX_ASSET_LAST, key="rxrx_asset_main") # New Ticker RXRX
x_9 =  RXRX_OPTION  + RXRX_REAL # New Ticker RXRX


st.write("_____")

# try:

s8 , s9 , s10 =  sell( asset = x_3 , Diff= x_2)
s11 , s12 , s13 =  sell(asset = x_4 , Diff= x_2)
b8 , b9 , b10 =  buy(asset = x_3 , Diff= x_2)
b11 , b12 , b13 =  buy(asset = x_4 , Diff= x_2)
u1 , u2 , u3 = sell( asset = x_5 , Diff= x_2)
u4 , u5 , u6 = buy( asset = x_5 , Diff= x_2)
p1 , p2 , p3 = sell( asset = x_6 , Diff= x_2)
p4 , p5 , p6 = buy( asset = x_6 , Diff= x_2)
u7 , u8 , u9 = sell( asset = x_7 , Diff= x_2) # Note: u7,u8,u9 for NVTS sell
p7 , p8 , p9 = buy( asset = x_7 , Diff= x_2)  # Note: p7,p8,p9 for NVTS buy

q1, q2, q3 = sell(asset=x_8, Diff=x_2)
q4, q5, q6 = buy(asset=x_8, Diff=x_2)

r1, r2, r3 = sell(asset=x_9, Diff=x_2) # New Ticker RXRX sell params
r4, r5, r6 = buy(asset=x_9, Diff=x_2)  # New Ticker RXRX buy params

def get_last_price(ticker_symbol):
    try:
        return yf.Ticker(ticker_symbol).fast_info['lastPrice']
    except Exception as e:
        # st.warning(f"Could not fetch last price for {ticker_symbol}: {e}")
        return 0 # Return a default value like 0 or None

# Function to safely get action value
def get_action_value(df_action_values, index, default_value=0):
    try:
        return df_action_values[index]
    except IndexError:
        # st.warning(f"Index {index} out of bounds for action values. Using default {default_value}.")
        return default_value


Limut_Order_NEGG = st.checkbox('Limut_Order_NEGG', value = bool(np.where( Nex_day_sell == 1 ,  toggle( get_action_value(df_7_1.action.values, 1+nex) )   ,  get_action_value(df_7_1.action.values, 1+nex)   )))
if Limut_Order_NEGG :
    st.write( 'sell NEGG:' , '    ' ,'A', b9  , 'P' , b8 ,'C' ,b10    )

    col1, col2 , col3  = st.columns(3)
    sell_negg = col3.checkbox('sell_match_NEGG', key="sell_negg_cb")
    if sell_negg :
        GO_NEGG_SELL = col3.button("GO! SELL NEGG", key="go_negg_sell")
        if GO_NEGG_SELL :
            new_asset_val = NEGG_ASSET_LAST - b9
            client.update(  {'field2': new_asset_val } )
            col3.write(f"NEGG Asset after sell: {new_asset_val}")
            st.rerun()


    pv_negg =  get_last_price('NEGG') * x_3
    st.write(get_last_price('NEGG') , pv_negg  ,'(',  pv_negg - 1500 ,')',  )

    st.write( 'buy NEGG:' , '    ','A',  s9  ,  'P' , s8 , 'C' ,s10    )
    col4, col5 , col6  = st.columns(3) # Re-define columns if they are to be separate
    buy_negg = col6.checkbox('buy_match_NEGG', key="buy_negg_cb")
    if buy_negg :
        GO_NEGG_Buy = col6.button("GO! BUY NEGG", key="go_negg_buy")
        if GO_NEGG_Buy :
            new_asset_val = NEGG_ASSET_LAST + s9
            client.update(  {'field2': new_asset_val  } )
            col6.write(f"NEGG Asset after buy: {new_asset_val}")
            st.rerun()

st.write("_____")

Limut_Order_FFWM = st.checkbox('Limut_Order_FFWM',  value = bool(np.where(  Nex_day_sell == 1 ,  toggle( get_action_value(df_7.action.values, 1+nex) )   ,  get_action_value(df_7.action.values, 1+nex)   )))
if Limut_Order_FFWM :
    st.write( 'sell FFWM:' , '    ' , 'A', b12 , 'P' , b11  , 'C' , b13    )

    col7, col8 , col9  = st.columns(3)
    sell_ffwm = col9.checkbox('sell_match_FFWM', key="sell_ffwm_cb")
    if sell_ffwm :
        GO_ffwm_sell = col9.button("GO! SELL FFWM", key="go_ffwm_sell")
        if GO_ffwm_sell :
            new_asset_val = FFWM_ASSET_LAST - b12
            client.update(  {'field1': new_asset_val } )
            col9.write(f"FFWM Asset after sell: {new_asset_val}")
            st.rerun()

    pv_ffwm =   get_last_price('FFWM') * x_4
    st.write(get_last_price('FFWM') , pv_ffwm ,'(',  pv_ffwm - 1500 ,')', )

    st.write(  'buy FFWM:' , '    ', 'A', s12 , 'P' , s11  , 'C'  , s13    )
    col10, col11 , col12  = st.columns(3)
    buy_ffwm = col12.checkbox('buy_match_FFWM', key="buy_ffwm_cb")
    if buy_ffwm :
        GO_ffwm_Buy = col12.button("GO! BUY FFWM", key="go_ffwm_buy")
        if GO_ffwm_Buy :
            new_asset_val = FFWM_ASSET_LAST + s12
            client.update(  {'field1': new_asset_val  } )
            col12.write(f"FFWM Asset after buy: {new_asset_val}")
            st.rerun()

st.write("_____")

Limut_Order_RIVN = st.checkbox('Limut_Order_RIVN',value = bool(np.where(  Nex_day_sell == 1 ,  toggle( get_action_value(df_7_2.action.values,1+nex) )   ,  get_action_value(df_7_2.action.values,1+nex)   )))
if Limut_Order_RIVN :
    st.write( 'sell RIVN:' , '    ' , 'A', u5 , 'P' , u4  , 'C' , u6    )

    col77, col88 , col99  = st.columns(3)
    sell_RIVN = col99.checkbox('sell_match_RIVN', key="sell_rivn_cb")
    if sell_RIVN :
        GO_RIVN_sell = col99.button("GO! SELL RIVN", key="go_rivn_sell")
        if GO_RIVN_sell :
            new_asset_val = RIVN_ASSET_LAST - u5
            client.update(  {'field3': new_asset_val } )
            col99.write(f"RIVN Asset after sell: {new_asset_val}")
            st.rerun()

    pv_rivn =   get_last_price('RIVN') * x_5
    st.write(get_last_price('RIVN') , pv_rivn ,'(',  pv_rivn - 1500 ,')', )

    st.write(  'buy RIVN:' , '    ', 'A', u2 , 'P' , u1  , 'C'  , u3    )
    col100 , col111 , col122  = st.columns(3)
    buy_RIVN = col122.checkbox('buy_match_RIVN', key="buy_rivn_cb")
    if buy_RIVN :
        GO_RIVN_Buy = col122.button("GO! BUY RIVN", key="go_rivn_buy")
        if GO_RIVN_Buy :
            new_asset_val = RIVN_ASSET_LAST + u2
            client.update(  {'field3': new_asset_val  } )
            col122.write(f"RIVN Asset after buy: {new_asset_val}")
            st.rerun()

st.write("_____")

Limut_Order_APLS = st.checkbox('Limut_Order_APLS',value = bool(np.where(  Nex_day_sell == 1 ,  toggle( get_action_value(df_7_3.action.values,1+nex) )   ,  get_action_value(df_7_3.action.values,1+nex)   )))
if Limut_Order_APLS :
    st.write( 'sell APLS:' , '    ' , 'A', p5 , 'P' , p4  , 'C' , p6    )

    col7777, col8888 , col9999  = st.columns(3)
    sell_APLS = col9999.checkbox('sell_match_APLS', key="sell_apls_cb")
    if sell_APLS :
        GO_APLS_sell = col9999.button("GO! SELL APLS", key="go_apls_sell")
        if GO_APLS_sell :
            new_asset_val = APLS_ASSET_LAST - p5
            client.update(  {'field4': new_asset_val } )
            col9999.write(f"APLS Asset after sell: {new_asset_val}")
            st.rerun()

    pv_apls =   get_last_price('APLS') * x_6
    st.write(get_last_price('APLS' ) , pv_apls ,'(',  pv_apls - 1500 ,')', )

    st.write(  'buy APLS:' , '    ', 'A', p2 , 'P' , p1  , 'C'  , p3    )
    col1000 , col1111 , col1222  = st.columns(3)
    buy_APLS = col1222.checkbox('buy_match_APLS', key="buy_apls_cb")
    if buy_APLS :
        GO_APLS_Buy = col1222.button("GO! BUY APLS", key="go_apls_buy")
        if GO_APLS_Buy :
            new_asset_val = APLS_ASSET_LAST + p2
            client.update(  {'field4': new_asset_val  } )
            col1222.write(f"APLS Asset after buy: {new_asset_val}")
            st.rerun()

st.write("_____")

Limut_Order_NVTS = st.checkbox('Limut_Order_NVTS', value=bool(np.where(  Nex_day_sell == 1 ,  toggle( get_action_value(df_7_4.action.values,1+nex) )   ,  get_action_value(df_7_4.action.values,1+nex)   )))
if Limut_Order_NVTS:
    st.write('sell NVTS:', '    ', 'A', p8 , 'P', p7  , 'C', p9    ) # Using p7,p8,p9 for NVTS buy based on your buy(asset=x_7) which is p7,p8,p9
                                                                    # If it should be u7,u8,u9 for sell, then buy uses different variables. Let's assume p values are for buy side based on how it was defined.
                                                                    # Let's correct variable usage: sell(asset=x_7) -> u7,u8,u9 and buy(asset=x_7) -> p7,p8,p9
                                                                    # So for sell NVTS, it should be buy variables (p8, p7, p9) - this seems counterintuitive as 'sell' usually means asset goes down.
                                                                    # Let's assume buy() returns (price, amount, cost) for a target buy, and sell() for a target sell.
                                                                    # If Limut_Order is True (meaning an indication to act)
                                                                    # 'sell' section: displays parameters for a potential sell order. This should use buy() function results. (b-variables)
                                                                    # 'buy' section: displays parameters for a potential buy order. This should use sell() function results. (s-variables)

    # For NVTS: sell(asset=x_7) is u7,u8,u9. buy(asset=x_7) is p7,p8,p9
    # If action is SELL (Limut_Order_NVTS is True and it's a sell signal based on df_7_4)
    # The app shows "sell A P C", these should be parameters to create a sell order.
    # sell() -> s2 (price), s5 (amount), s7 (cost)
    # buy() -> b2 (price), b5 (amount), b7 (cost)

    # Correcting the NVTS section logic display:
    # If the intent is to show parameters for a SELL action when Limut_Order_NVTS is checked:
    # Parameters for selling are typically from the buy() function in your logic (higher price target)
    # Parameters for buying are typically from the sell() function in your logic (lower price target)

    st.write('sell NVTS:', '    ', 'A', p8 , 'P', p7 , 'C', p9 ) # Sell uses buy() results: price p7, amount p8, cost p9

    col_nvts1, col_nvts2, col_nvts3 = st.columns(3)
    sell_NVTS = col_nvts3.checkbox('sell_match_NVTS', key="sell_nvts_cb")
    if sell_NVTS:
        GO_NVTS_sell = col_nvts3.button("GO! SELL NVTS", key="go_nvts_sell")
        if GO_NVTS_sell:
            new_asset_val = NVTS_ASSET_LAST - p8 # Amount to sell is p8
            client.update({'field5': new_asset_val})
            col_nvts3.write(f"NVTS Asset after sell: {new_asset_val}")
            st.rerun()

    pv_nvts = get_last_price('NVTS') * x_7
    st.write(get_last_price('NVTS'), pv_nvts, '(', pv_nvts - 1500, ')')

    st.write('buy NVTS:', '    ', 'A', u8, 'P', u7 , 'C', u9 ) # Buy uses sell() results: price u7, amount u8, cost u9
    col_nvts4, col_nvts5, col_nvts6 = st.columns(3)
    buy_NVTS = col_nvts6.checkbox('buy_match_NVTS', key="buy_nvts_cb")
    if buy_NVTS:
        GO_NVTS_Buy = col_nvts6.button("GO! BUY NVTS", key="go_nvts_buy")
        if GO_NVTS_Buy:
            new_asset_val = NVTS_ASSET_LAST + u8 # Amount to buy is u8
            client.update({'field5': new_asset_val})
            col_nvts6.write(f"NVTS Asset after buy: {new_asset_val}")
            st.rerun()

st.write("_____")

Limut_Order_QXO = st.checkbox('Limut_Order_QXO', value=bool(np.where(Nex_day_sell == 1, toggle(get_action_value(df_7_5.action.values,1+nex)), get_action_value(df_7_5.action.values,1+nex))))
if Limut_Order_QXO:
    # sell() -> q1(P), q2(A), q3(C)
    # buy()  -> q4(P), q5(A), q6(C)
    st.write('sell QXO:', '    ', 'A', q5, 'P', q4, 'C', q6) # Sell: Amount q5, Price q4, Cost q6

    col_qxo1, col_qxo2, col_qxo3 = st.columns(3)
    sell_QXO = col_qxo3.checkbox('sell_match_QXO', key="sell_qxo_cb")
    if sell_QXO:
        GO_QXO_sell = col_qxo3.button("GO! SELL QXO", key="go_qxo_sell")
        if GO_QXO_sell:
            new_asset_val = QXO_ASSET_LAST - q5
            client.update({'field6': new_asset_val}) # Field6 for QXO
            col_qxo3.write(f"QXO Asset after sell: {new_asset_val}")
            st.rerun()

    pv_qxo = get_last_price('QXO') * x_8
    st.write(get_last_price('QXO'), pv_qxo, '(', pv_qxo - 1500, ')')

    st.write('buy QXO:', '    ', 'A', q2, 'P', q1, 'C', q3) # Buy: Amount q2, Price q1, Cost q3
    col_qxo4, col_qxo5, col_qxo6 = st.columns(3)
    buy_QXO = col_qxo6.checkbox('buy_match_QXO', key="buy_qxo_cb")
    if buy_QXO:
        GO_QXO_Buy = col_qxo6.button("GO! BUY QXO", key="go_qxo_buy")
        if GO_QXO_Buy:
            new_asset_val = QXO_ASSET_LAST + q2
            client.update({'field6': new_asset_val}) # Field6 for QXO
            col_qxo6.write(f"QXO Asset after buy: {new_asset_val}")
            st.rerun()
st.write("_____")

# --- RXRX Section ---
Limut_Order_RXRX = st.checkbox('Limut_Order_RXRX', value=bool(np.where(Nex_day_sell == 1, toggle(get_action_value(df_7_6.action.values,1+nex)), get_action_value(df_7_6.action.values,1+nex))))
if Limut_Order_RXRX:
    # sell() for RXRX: r1(P), r2(A), r3(C)
    # buy() for RXRX:  r4(P), r5(A), r6(C)
    st.write('sell RXRX:', '    ', 'A', r5, 'P', r4, 'C', r6) # Sell: Amount r5, Price r4, Cost r6

    col_rxrx1, col_rxrx2, col_rxrx3 = st.columns(3)
    sell_RXRX = col_rxrx3.checkbox('sell_match_RXRX', key="sell_rxrx_cb")
    if sell_RXRX:
        GO_RXRX_sell = col_rxrx3.button("GO! SELL RXRX", key="go_rxrx_sell")
        if GO_RXRX_sell:
            new_asset_val = RXRX_ASSET_LAST - r5
            client.update({'field7': new_asset_val}) # Assuming field7 for RXRX
            col_rxrx3.write(f"RXRX Asset after sell: {new_asset_val}")
            st.rerun()

    pv_rxrx = get_last_price('RXRX') * x_9
    st.write(get_last_price('RXRX'), pv_rxrx, '(', pv_rxrx - 1500, ')')

    st.write('buy RXRX:', '    ', 'A', r2, 'P', r1, 'C', r3) # Buy: Amount r2, Price r1, Cost r3
    col_rxrx4, col_rxrx5, col_rxrx6 = st.columns(3)
    buy_RXRX = col_rxrx6.checkbox('buy_match_RXRX', key="buy_rxrx_cb")
    if buy_RXRX:
        GO_RXRX_Buy = col_rxrx6.button("GO! BUY RXRX", key="go_rxrx_buy")
        if GO_RXRX_Buy:
            new_asset_val = RXRX_ASSET_LAST + r2
            client.update({'field7': new_asset_val}) # Assuming field7 for RXRX
            col_rxrx6.write(f"RXRX Asset after buy: {new_asset_val}")
            st.rerun()

st.write("_____")


if st.button("RERUN App"): # Changed label slightly to avoid conflict if any other "RERUN" exists
    st.rerun()

# Removed the try-except pass for the whole block as it can hide errors.
# Individual error handling (like in get_last_price, Monitor, get_thingspeak_field_value) is usually better.
