import streamlit as st
import numpy as np
import datetime
import thingspeak
import pandas as pd
import yfinance as yf
import json

st.set_page_config(page_title="Monitor", page_icon="📈")
channel_id = 2528199
write_api_key = '2E65V8XEIPH9B2VV' # Make sure this is your actual write API key
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
write_api_key_2 = 'IPSG3MMMBJEB9DY8' # Make sure this is your actual write API key
client_2 = thingspeak.Channel(channel_id_2, write_api_key_2 , fmt='json' )

def Monitor (Ticker = 'FFWM' , field = 2  ):

    tickerData = yf.Ticker(Ticker)
    # Use a short period for history if 'max' is too slow or not needed for recent actions
    try:
        tickerData = round(tickerData.history(period= '1y' )[['Close']] , 3 ) # Changed period to '1y' for faster loading
        if tickerData.empty: # Fallback if '1y' returns empty
            tickerData = round(tickerData.history(period= 'max' )[['Close']] , 3 )
    except Exception as e:
        st.error(f"Error fetching history for {Ticker}: {e}")
        # Create a dummy dataframe to prevent downstream errors
        dummy_dates = pd.to_datetime([datetime.date.today() - datetime.timedelta(days=i) for i in range(7)])
        dummy_index = pd.DatetimeIndex(dummy_dates).tz_localize('UTC').tz_convert('Asia/Bangkok') # Ensure timezone aware
        tickerData = pd.DataFrame({'Close': [0]*7}, index=dummy_index)


    tickerData.index = tickerData.index.tz_convert(tz='Asia/Bangkok') # Ensure correct timezone
    filter_date_str = '2023-01-01 12:00:00+07:00'
    # Ensure filter_date is timezone-aware for comparison
    filter_date = pd.Timestamp(filter_date_str, tz='Asia/Bangkok')
    tickerData = tickerData[tickerData.index >= filter_date]

    fx = client_2.get_field_last(field='{}'.format(field))
    try:
        fx_js = int(json.loads(fx)["field{}".format(field)])
    except (json.JSONDecodeError, KeyError, TypeError) as e: # Handle potential errors from ThingSpeak
        st.warning(f"Could not fetch/parse field {field} for {Ticker} from ThingSpeak. Using default seed 0. Error: {e}")
        fx_js = 0 # Default seed

    rng = np.random.default_rng(fx_js)
    data = rng.integers(2, size = len(tickerData))
    tickerData['action'] = data
    tickerData['index'] = [ i+1 for i in range(len(tickerData))]

    tickerData_1 = pd.DataFrame(columns=(tickerData.columns))
    # tickerData_1['action'] =  [ i for i in range(5)] # This was creating actions 0,1,2,3,4
    # To match the pattern of random integers (0 or 1) like above:
    rng_fixed = np.random.default_rng(0) # Use a fixed seed for these dummy rows for consistency
    tickerData_1['action'] = rng_fixed.integers(2, size=5)

    tickerData_1.index = ['+0' , "+1" , "+2" , "+3" , "+4"]
    df = pd.concat([tickerData , tickerData_1], axis=0).fillna("")
    
    # Re-apply random numbers to 'action' for the whole combined df if that's the desired logic
    # Or, if the intent was for tickerData_1 to have fixed predictable actions, the above rng_fixed is better.
    # The original code re-randomizes the whole df['action'] column here:
    rng_final = np.random.default_rng(fx_js) # Using fx_js seed again as in original
    df['action'] = rng_final.integers(2, size = len(df))
    
    return df.tail(7) , fx_js

df_7   , fx_js   = Monitor(Ticker = 'FFWM', field = 2  )
df_7_1 , fx_js_1  = Monitor(Ticker = 'NEGG', field = 3  )
df_7_2 , fx_js_2  = Monitor(Ticker = 'RIVN', field = 4 )
df_7_3 , fx_js_3  = Monitor(Ticker = 'APLS', field = 5 )
df_7_4 , fx_js_4  = Monitor(Ticker = 'NVTS', field = 6 )
# --- ADDED MONITORS ---
df_7_5 , fx_js_5  = Monitor(Ticker = 'QXO', field = 7 )
df_7_6 , fx_js_6  = Monitor(Ticker = 'RXRX', field = 8 )

nex = 0
Nex_day_sell = 0
toggle = lambda x : 1 - x if x in (0,1) else 0 # Ensure toggle works for 0 and 1

Nex_day_ = st.checkbox('nex_day')
if Nex_day_ :
  st.write( "value = " , nex)
  nex_col , Nex_day_sell_col ,_,_,_  = st.columns(5)

  if nex_col.button("Nex_day"):
    nex = 1
    st.write( "value = " , nex) # This write might not show if rerun happens, consider st.rerun or session state
    st.rerun()

  if Nex_day_sell_col.button("Nex_day_sell"):
    nex = 1
    Nex_day_sell = 1
    st.write( "value = " , nex) # Same as above
    st.write( "Nex_day_sell = " , Nex_day_sell)
    st.rerun()

st.write("_____")

# --- ADJUSTED COLUMNS ---
# Original: col13, col16, col14, col15, col17, col18, col19 = st.columns(7)
# For 7 assets + Diff + Start column, we need 9 columns if Start has its own widgets.
# col13 is for 'Start' and initial asset settings.
# The other columns are for 'Diff' and asset value inputs.
# So, 1 column for Start, 1 for Diff, 7 for assets = 9 columns for main inputs.
# Let's define them more clearly.
# col_start_setup will be the first column.
# The rest (Diff + 7 assets) will be in subsequent columns.

# This definition assigns columns sequentially based on st.columns(N)
# col13 = 1st, col16 = 2nd, col14 = 3rd, col15 = 4th, col17 = 5th, col18 = 6th, col19 = 7th
# To add two more, we need 9 columns in total for this row.
col13, col16, col14, col15, col17, col18, col19, col20, col21 = st.columns(9)


x_2 = col16.number_input('Diff', step=1 , value= 60   ) # Diff Input

Start = col13.checkbox('start') # Start checkbox in the first column
if Start :
  # --- FFWM ---
  thingspeak_1 = col13.checkbox('@_FFWM_ASSET')
  if thingspeak_1 :
    add_1 = col13.number_input('@_FFWM_ASSET', step=0.001 ,  value=0., key='add_ffwm_asset')
    _FFWM_ASSET_BTN = col13.button("GO!", key='go_ffwm_asset')
    if _FFWM_ASSET_BTN :
      try:
        client.update(  {'field1': add_1 } )
        col13.write(f"FFWM Asset Updated: {add_1}")
      except Exception as e:
        col13.error(f"ThingSpeak Error: {e}")

  # --- NEGG ---
  thingspeak_2 = col13.checkbox('@_NEGG_ASSET')
  if thingspeak_2 :
    add_2 = col13.number_input('@_NEGG_ASSET', step=0.001 ,  value=0., key='add_negg_asset')
    _NEGG_ASSET_BTN = col13.button("GO!", key='go_negg_asset')
    if _NEGG_ASSET_BTN :
      try:
        client.update(  {'field2': add_2 }  )
        col13.write(f"NEGG Asset Updated: {add_2}")
      except Exception as e:
        col13.error(f"ThingSpeak Error: {e}")

  # --- RIVN ---
  thingspeak_3 = col13.checkbox('@_RIVN_ASSET')
  if thingspeak_3 :
    add_3 = col13.number_input('@_RIVN_ASSET', step=0.001 ,  value=0., key='add_rivn_asset')
    _RIVN_ASSET_BTN = col13.button("GO!", key='go_rivn_asset')
    if _RIVN_ASSET_BTN :
      try:
        client.update(  {'field3': add_3 }  )
        col13.write(f"RIVN Asset Updated: {add_3}")
      except Exception as e:
        col13.error(f"ThingSpeak Error: {e}")

  # --- APLS ---
  thingspeak_4 = col13.checkbox('@_APLS_ASSET')
  if thingspeak_4 :
    add_4 = col13.number_input('@_APLS_ASSET', step=0.001 ,  value=0., key='add_apls_asset')
    _APLS_ASSET_BTN = col13.button("GO!", key='go_apls_asset')
    if _APLS_ASSET_BTN :
      try:
        client.update(  {'field4': add_4 }  )
        col13.write(f"APLS Asset Updated: {add_4}")
      except Exception as e:
        col13.error(f"ThingSpeak Error: {e}")

  # --- NVTS ---
  thingspeak_5 = col13.checkbox('@_NVTS_ASSET')
  if thingspeak_5:
    add_5 = col13.number_input('@_NVTS_ASSET', step=0.001, value= 0., key='add_nvts_asset')
    _NVTS_ASSET_BTN = col13.button("GO!", key='go_nvts_asset')
    if _NVTS_ASSET_BTN:
      try:
        client.update({'field5': add_5})
        col13.write(f"NVTS Asset Updated: {add_5}")
      except Exception as e:
        col13.error(f"ThingSpeak Error: {e}")

  # --- ADDED: QXO ---
  thingspeak_6 = col13.checkbox('@_QXO_ASSET')
  if thingspeak_6:
    add_6 = col13.number_input('@_QXO_ASSET', step=0.001, value= 0., key='add_qxo_asset') # Unique label for number_input acts as key
    _QXO_ASSET_BTN = col13.button("GO!", key='go_qxo_asset') # Unique key for button
    if _QXO_ASSET_BTN:
      try:
        client.update({'field7': add_6})
        col13.write(f"QXO Asset Updated: {add_6}")
      except Exception as e:
        col13.error(f"ThingSpeak Error: {e}")

  # --- ADDED: RXRX ---
  thingspeak_7 = col13.checkbox('@_RXRX_ASSET')
  if thingspeak_7:
    add_7 = col13.number_input('@_RXRX_ASSET', step=0.001, value= 0., key='add_rxrx_asset') # Unique label for number_input acts as key
    _RXRX_ASSET_BTN = col13.button("GO!", key='go_rxrx_asset') # Unique key for button
    if _RXRX_ASSET_BTN:
      try:
        client.update({'field8': add_7})
        col13.write(f"RXRX Asset Updated: {add_7}")
      except Exception as e:
        col13.error(f"ThingSpeak Error: {e}")

def get_last_asset_value(client, field_name, default_value=0.0):
    try:
        raw_value = client.get_field_last(field=field_name)
        # Check if raw_value is not None and is a string before json.loads
        if raw_value is not None and isinstance(raw_value, str):
            loaded_json = json.loads(raw_value)
            # Check if the expected field exists in the loaded JSON
            if field_name in loaded_json:
                value_str = loaded_json[field_name]
                # Check if value_str is not None before eval
                if value_str is not None:
                    return eval(str(value_str)) # Use str() for safety if it's already a number
                else:
                    st.warning(f"ThingSpeak field '{field_name}' is null. Using default: {default_value}")
                    return default_value
            else:
                st.warning(f"ThingSpeak field '{field_name}' not found in JSON response. Using default: {default_value}")
                return default_value
        elif raw_value is None:
             st.warning(f"ThingSpeak returned None for field '{field_name}'. Using default: {default_value}")
             return default_value
        else: # If it's already a number or other type somehow (should be str from client)
            st.warning(f"ThingSpeak field '{field_name}' has unexpected type: {type(raw_value)}. Attempting to use as is or default.")
            return float(raw_value) if isinstance(raw_value, (int, float, str)) else default_value
    except json.JSONDecodeError:
        st.warning(f"ThingSpeak field '{field_name}' contained invalid JSON: '{raw_value}'. Using default: {default_value}")
        return default_value
    except Exception as e:
        st.error(f"Error getting/parsing ThingSpeak field '{field_name}': {e}. Using default: {default_value}")
        return default_value
    return default_value


FFWM_ASSET_LAST = get_last_asset_value(client, 'field1')
NEGG_ASSET_LAST = get_last_asset_value(client, 'field2')
RIVN_ASSET_LAST = get_last_asset_value(client, 'field3')
APLS_ASSET_LAST = get_last_asset_value(client, 'field4')
NVTS_ASSET_LAST = get_last_asset_value(client, 'field5')
# --- ADDED LAST ASSET FETCHES ---
QXO_ASSET_LAST = get_last_asset_value(client, 'field7')
RXRX_ASSET_LAST = get_last_asset_value(client, 'field8')


# Asset value inputs in their respective columns:
# col14 is 3rd col, col15 is 4th, col17 is 5th, col18 is 6th, col19 is 7th
# col20 is 8th, col21 is 9th
x_3 = col14.number_input('NEGG_ASSET', step=0.001 ,  value= float(NEGG_ASSET_LAST), key='negg_asset_val' )
x_4 = col15.number_input('FFWM_ASSET', step=0.001  , value= float(FFWM_ASSET_LAST), key='ffwm_asset_val'  )
x_5 = col17.number_input('RIVN_ASSET', step=0.001  , value= float(RIVN_ASSET_LAST), key='rivn_asset_val'  )
x_6 = col18.number_input('APLS_ASSET', step=0.001  , value= float(APLS_ASSET_LAST), key='apls_asset_val'  )
x_7 = col19.number_input('NVTS_ASSET', step=0.001  , value= float(NVTS_ASSET_LAST), key='nvts_asset_val'  )
# --- ADDED ASSET INPUTS ---
x_8 = col20.number_input('QXO_ASSET', step=0.001, value= float(QXO_ASSET_LAST), key='qxo_asset_val')
x_9 = col21.number_input('RXRX_ASSET', step=0.001, value= float(RXRX_ASSET_LAST), key='rxrx_asset_val')

st.write("_____")

# Calculate sell/buy parameters
s8 , s9 , s10 =  sell( asset = x_3 , Diff= x_2) # NEGG sell (price, amount, cost)
s11 , s12 , s13 =  sell(asset = x_4 , Diff= x_2) # FFWM sell
b8 , b9 , b10 =  buy(asset = x_3 , Diff= x_2)  # NEGG buy
b11 , b12 , b13 =  buy(asset = x_4 , Diff= x_2) # FFWM buy
u1 , u2 , u3 = sell( asset = x_5 , Diff= x_2) # RIVN sell
u4 , u5 , u6 = buy( asset = x_5 , Diff= x_2)  # RIVN buy
p1 , p2 , p3 = sell( asset = x_6 , Diff= x_2) # APLS sell
p4 , p5 , p6 = buy( asset = x_6 , Diff= x_2)  # APLS buy
u7 , u8 , u9 = sell( asset = x_7 , Diff= x_2) # NVTS sell
p7 , p8 , p9 = buy( asset = x_7 , Diff= x_2)  # NVTS buy

# --- ADDED CALCULATIONS FOR NEW ASSETS ---
# QXO: x_8
q_s_price, q_s_amount, q_s_cost = sell(asset=x_8, Diff=x_2) # QXO sell params (price, amount, cost)
q_b_price, q_b_amount, q_b_cost = buy(asset=x_8, Diff=x_2)  # QXO buy params (price, amount, cost)

# RXRX: x_9
r_s_price, r_s_amount, r_s_cost = sell(asset=x_9, Diff=x_2) # RXRX sell params
r_b_price, r_b_amount, r_b_cost = buy(asset=x_9, Diff=x_2)  # RXRX buy params


# --- Helper function to get yfinance data safely ---
def get_yf_price(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        price = ticker.fast_info.get('lastPrice')
        if price is None: # Fallback if lastPrice is not available
            hist = ticker.history(period="1d")
            if not hist.empty:
                price = hist['Close'].iloc[-1]
            else:
                return 0.0 # Or handle as an error
        return price
    except Exception as e:
        st.warning(f"Could not fetch price for {ticker_symbol}: {e}")
        return 0.0 # Default to 0 or some other indicator of failure

# --- NEGG Section ---
if len(df_7_1.action.values) > (1+nex): # Boundary check
    Limut_Order_NEGG_val = np.where( Nex_day_sell == 1 , toggle( df_7_1.action.values[1+nex] ) , df_7_1.action.values[1+nex] )
    Limut_Order_NEGG = st.checkbox('Limut_Order_NEGG', value = bool(Limut_Order_NEGG_val), key='limit_negg')
    if Limut_Order_NEGG :
      # Sell section uses buy() results: b9 (amount), b8 (price), b10 (cost)
      st.write( 'sell' , '    ' ,'A', b9  , 'P' , b8 ,'C' ,b10  )

      col1, col2 , col3  = st.columns(3)
      sell_negg = col3.checkbox('sell_match_NEGG', key='sell_match_negg')
      if sell_negg :
        GO_NEGG_SELL = col3.button("GO!", key='go_negg_sell')
        if GO_NEGG_SELL :
          try:
            client.update(  {'field2': NEGG_ASSET_LAST - b9 } )
            col3.write(f"NEGG Updated: {NEGG_ASSET_LAST - b9}")
            st.rerun()
          except Exception as e:
            col3.error(f"ThingSpeak Error: {e}")

      negg_price = get_yf_price('NEGG')
      pv_negg =  negg_price * x_3
      st.write(negg_price , pv_negg  ,'(',  pv_negg - 1500 ,')',  )

      col4, col5 , col6  = st.columns(3)
      # Buy section uses sell() results: s9 (amount), s8 (price), s10 (cost)
      st.write( 'buy' , '    ','A',  s9  ,  'P' , s8 , 'C' ,s10  )
      buy_negg = col6.checkbox('buy_match_NEGG', key='buy_match_negg')
      if buy_negg :
        GO_NEGG_Buy = col6.button("GO!", key='go_negg_buy')
        if GO_NEGG_Buy :
          try:
            client.update(  {'field2': NEGG_ASSET_LAST + s9  } )
            col6.write(f"NEGG Updated: {NEGG_ASSET_LAST + s9}")
            st.rerun()
          except Exception as e:
            col6.error(f"ThingSpeak Error: {e}")
else:
    st.warning("Not enough data for NEGG action.")
st.write("_____")

# --- FFWM Section ---
if len(df_7.action.values) > (1+nex): # Boundary check
    Limut_Order_FFWM_val = np.where( Nex_day_sell == 1 , toggle( df_7.action.values[1+nex] ) , df_7.action.values[1+nex] )
    Limut_Order_FFWM = st.checkbox('Limut_Order_FFWM',  value = bool(Limut_Order_FFWM_val), key='limit_ffwm')
    if Limut_Order_FFWM :
      # Sell section uses buy() results: b12 (amount), b11 (price), b13 (cost)
      st.write( 'sell' , '    ' , 'A', b12 , 'P' , b11  , 'C' , b13  )

      col7, col8 , col9  = st.columns(3)
      sell_ffwm = col9.checkbox('sell_match_FFWM', key='sell_match_ffwm')
      if sell_ffwm :
        GO_ffwm_sell = col9.button("GO!", key='go_ffwm_sell')
        if GO_ffwm_sell :
          try:
            client.update(  {'field1': FFWM_ASSET_LAST - b12  } )
            col9.write(f"FFWM Updated: {FFWM_ASSET_LAST - b12}")
            st.rerun()
          except Exception as e:
            col9.error(f"ThingSpeak Error: {e}")

      ffwm_price = get_yf_price('FFWM')
      pv_ffwm =   ffwm_price * x_4
      st.write(ffwm_price , pv_ffwm ,'(',  pv_ffwm - 1500 ,')', )

      col10, col11 , col12  = st.columns(3)
      # Buy section uses sell() results: s12 (amount), s11 (price), s13 (cost)
      st.write(  'buy' , '    ', 'A', s12 , 'P' , s11  , 'C'  , s13  )
      buy_ffwm = col12.checkbox('buy_match_FFWM', key='buy_match_ffwm')
      if buy_ffwm :
        GO_ffwm_Buy = col12.button("GO!", key='go_ffwm_buy')
        if GO_ffwm_Buy :
          try:
            client.update(  {'field1': FFWM_ASSET_LAST + s12  } )
            col12.write(f"FFWM Updated: {FFWM_ASSET_LAST + s12}")
            st.rerun()
          except Exception as e:
            col12.error(f"ThingSpeak Error: {e}")
else:
    st.warning("Not enough data for FFWM action.")
st.write("_____")

# --- RIVN Section ---
if len(df_7_2.action.values) > (1+nex): # Boundary check
    Limut_Order_RIVN_val = np.where( Nex_day_sell == 1 , toggle( df_7_2.action.values[1+nex] ) , df_7_2.action.values[1+nex] )
    Limut_Order_RIVN = st.checkbox('Limut_Order_RIVN',value = bool(Limut_Order_RIVN_val), key='limit_rivn')
    if Limut_Order_RIVN :
      # Sell section uses buy() results: u5 (amount), u4 (price), u6 (cost)
      st.write( 'sell' , '    ' , 'A', u5 , 'P' , u4  , 'C' , u6  )

      col77, col88 , col99  = st.columns(3)
      sell_RIVN = col99.checkbox('sell_match_RIVN', key='sell_match_rivn')
      if sell_RIVN :
        GO_RIVN_sell = col99.button("GO!", key='go_rivn_sell')
        if GO_RIVN_sell :
          try:
            client.update(  {'field3': RIVN_ASSET_LAST - u5  } )
            col99.write(f"RIVN Updated: {RIVN_ASSET_LAST - u5}")
            st.rerun()
          except Exception as e:
            col99.error(f"ThingSpeak Error: {e}")

      rivn_price = get_yf_price('RIVN')
      pv_rivn =   rivn_price * x_5
      st.write(rivn_price , pv_rivn ,'(',  pv_rivn - 1500 ,')', )

      col100 , col111 , col122  = st.columns(3)
      # Buy section uses sell() results: u2 (amount), u1 (price), u3 (cost)
      st.write(  'buy' , '    ', 'A', u2 , 'P' , u1  , 'C'  , u3  )
      buy_RIVN = col122.checkbox('buy_match_RIVN', key='buy_match_rivn')
      if buy_RIVN :
        GO_RIVN_Buy = col122.button("GO!", key='go_rivn_buy')
        if GO_RIVN_Buy :
          try:
            client.update(  {'field3': RIVN_ASSET_LAST + u2  } )
            col122.write(f"RIVN Updated: {RIVN_ASSET_LAST + u2}")
            st.rerun()
          except Exception as e:
            col122.error(f"ThingSpeak Error: {e}")
else:
    st.warning("Not enough data for RIVN action.")
st.write("_____")

# --- APLS Section ---
if len(df_7_3.action.values) > (1+nex): # Boundary check
    Limut_Order_APLS_val = np.where( Nex_day_sell == 1 , toggle( df_7_3.action.values[1+nex] ) , df_7_3.action.values[1+nex] )
    Limut_Order_APLS = st.checkbox('Limut_Order_APLS',value = bool(Limut_Order_APLS_val), key='limit_apls')
    if Limut_Order_APLS :
      # Sell section uses buy() results: p5 (amount), p4 (price), p6 (cost)
      st.write( 'sell' , '    ' , 'A', p5 , 'P' , p4  , 'C' , p6  )

      col7777, col8888 , col9999  = st.columns(3)
      sell_APLS = col9999.checkbox('sell_match_APLS', key='sell_match_apls')
      if sell_APLS :
        GO_APLS_sell = col9999.button("GO!", key='go_apls_sell')
        if GO_APLS_sell :
          try:
            client.update(  {'field4': APLS_ASSET_LAST - p5  } )
            col9999.write(f"APLS Updated: {APLS_ASSET_LAST - p5}")
            st.rerun()
          except Exception as e:
            col9999.error(f"ThingSpeak Error: {e}")

      apls_price = get_yf_price('APLS')
      pv_apls =   apls_price * x_6
      st.write(apls_price , pv_apls ,'(',  pv_apls - 1500 ,')', )

      col1000 , col1111 , col1222  = st.columns(3)
      # Buy section uses sell() results: p2 (amount), p1 (price), p3 (cost)
      st.write(  'buy' , '    ', 'A', p2 , 'P' , p1  , 'C'  , p3  )
      buy_APLS = col1222.checkbox('buy_match_APLS', key='buy_match_apls')
      if buy_APLS :
        GO_APLS_Buy = col1222.button("GO!", key='go_apls_buy')
        if GO_APLS_Buy :
          try:
            client.update(  {'field4': APLS_ASSET_LAST + p2  } )
            col1222.write(f"APLS Updated: {APLS_ASSET_LAST + p2}")
            st.rerun()
          except Exception as e:
            col1222.error(f"ThingSpeak Error: {e}")
else:
    st.warning("Not enough data for APLS action.")
st.write("_____")

# --- NVTS Section ---
# Original NVTS sell used p8, p7, p9 (from buy(x_7))
# Original NVTS buy used u8, u7, u9 (from sell(x_7))
if len(df_7_4.action.values) > (1+nex): # Boundary check
    Limut_Order_NVTS_val = np.where( Nex_day_sell == 1 , toggle( df_7_4.action.values[1+nex] ) , df_7_4.action.values[1+nex] )
    Limut_Order_NVTS = st.checkbox('Limut_Order_NVTS', value=bool(Limut_Order_NVTS_val), key='limit_nvts')
    if Limut_Order_NVTS:
      # Sell section uses buy() results: p8 (amount), p7 (price), p9 (cost)
      st.write('sell', '    ', 'A', p8 , 'P', p7  , 'C', p9  )

      col_nvts1, col_nvts2, col_nvts3 = st.columns(3)
      sell_NVTS = col_nvts3.checkbox('sell_match_NVTS', key='sell_match_nvts')

      if sell_NVTS:
          GO_NVTS_sell = col_nvts3.button("GO!", key='go_nvts_sell')
          if GO_NVTS_sell:
            try:
              client.update({'field5': NVTS_ASSET_LAST - p8})
              col_nvts3.write(f"NVTS Updated: {NVTS_ASSET_LAST - p8}")
              st.rerun()
            except Exception as e:
              col_nvts3.error(f"ThingSpeak Error: {e}")

      nvts_price = get_yf_price('NVTS')
      pv_nvts = nvts_price * x_7
      st.write(nvts_price, pv_nvts, '(', pv_nvts - 1500, ')')

      col_nvts4, col_nvts5, col_nvts6 = st.columns(3)
      # Buy section uses sell() results: u8 (amount), u7 (price), u9 (cost)
      st.write('buy', '    ', 'A', u8, 'P', u7  , 'C',u9 )
      buy_NVTS = col_nvts6.checkbox('buy_match_NVTS', key='buy_match_nvts')
      if buy_NVTS:
          GO_NVTS_Buy = col_nvts6.button("GO!", key='go_nvts_buy')
          if GO_NVTS_Buy:
            try:
              client.update({'field5': NVTS_ASSET_LAST + u8})
              col_nvts6.write(f"NVTS Updated: {NVTS_ASSET_LAST  + u8}")
              st.rerun()
            except Exception as e:
              col_nvts6.error(f"ThingSpeak Error: {e}")
else:
    st.warning("Not enough data for NVTS action.")
st.write("_____")


# --- ADDED: QXO Section ---
if len(df_7_5.action.values) > (1+nex): # Boundary check
    Limut_Order_QXO_val = np.where( Nex_day_sell == 1 , toggle( df_7_5.action.values[1+nex] ) , df_7_5.action.values[1+nex] )
    Limut_Order_QXO = st.checkbox('Limut_Order_QXO', value=bool(Limut_Order_QXO_val), key='limit_qxo')
    if Limut_Order_QXO:
        # Sell section uses buy() results: q_b_amount, q_b_price, q_b_cost
        st.write('sell', '   ', 'A', q_b_amount, 'P', q_b_price, 'C', q_b_cost)

        col_qxo_s1, col_qxo_s2, col_qxo_s3 = st.columns(3)
        sell_match_QXO = col_qxo_s3.checkbox('sell_match_QXO', key='sell_match_qxo')

        if sell_match_QXO:
            GO_QXO_sell = col_qxo_s3.button("GO!", key='go_qxo_sell')
            if GO_QXO_sell:
                try:
                    client.update({'field7': QXO_ASSET_LAST - q_b_amount}) # field7 for QXO
                    col_qxo_s3.write(f"QXO Updated: {QXO_ASSET_LAST - q_b_amount}")
                    st.rerun()
                except Exception as e:
                    col_qxo_s3.error(f"ThingSpeak Error: {e}")

        qxo_price = get_yf_price('QXO')
        pv_qxo = qxo_price * x_8 # x_8 is QXO_ASSET
        st.write(qxo_price, pv_qxo, '(', pv_qxo - 1500, ')')

        col_qxo_b1, col_qxo_b2, col_qxo_b3 = st.columns(3)
        # Buy section uses sell() results: q_s_amount, q_s_price, q_s_cost
        st.write('buy', '   ', 'A', q_s_amount, 'P', q_s_price, 'C', q_s_cost)
        buy_match_QXO = col_qxo_b3.checkbox('buy_match_QXO', key='buy_match_qxo')
        if buy_match_QXO:
            GO_QXO_Buy = col_qxo_b3.button("GO!", key='go_qxo_buy')
            if GO_QXO_Buy:
                try:
                    client.update({'field7': QXO_ASSET_LAST + q_s_amount}) # field7 for QXO
                    col_qxo_b3.write(f"QXO Updated: {QXO_ASSET_LAST + q_s_amount}")
                    st.rerun()
                except Exception as e:
                    col_qxo_b3.error(f"ThingSpeak Error: {e}")
else:
    st.warning("Not enough data for QXO action.")
st.write("_____")

# --- ADDED: RXRX Section ---
if len(df_7_6.action.values) > (1+nex): # Boundary check
    Limut_Order_RXRX_val = np.where( Nex_day_sell == 1 , toggle( df_7_6.action.values[1+nex] ) , df_7_6.action.values[1+nex] )
    Limut_Order_RXRX = st.checkbox('Limut_Order_RXRX', value=bool(Limut_Order_RXRX_val), key='limit_rxrx')
    if Limut_Order_RXRX:
        # Sell section uses buy() results: r_b_amount, r_b_price, r_b_cost
        st.write('sell', '   ', 'A', r_b_amount, 'P', r_b_price, 'C', r_b_cost)

        col_rxrx_s1, col_rxrx_s2, col_rxrx_s3 = st.columns(3)
        sell_match_RXRX = col_rxrx_s3.checkbox('sell_match_RXRX', key='sell_match_rxrx')

        if sell_match_RXRX:
            GO_RXRX_sell = col_rxrx_s3.button("GO!", key='go_rxrx_sell')
            if GO_RXRX_sell:
                try:
                    client.update({'field8': RXRX_ASSET_LAST - r_b_amount}) # field8 for RXRX
                    col_rxrx_s3.write(f"RXRX Updated: {RXRX_ASSET_LAST - r_b_amount}")
                    st.rerun()
                except Exception as e:
                    col_rxrx_s3.error(f"ThingSpeak Error: {e}")

        rxrx_price = get_yf_price('RXRX')
        pv_rxrx = rxrx_price * x_9 # x_9 is RXRX_ASSET
        st.write(rxrx_price, pv_rxrx, '(', pv_rxrx - 1500, ')')

        col_rxrx_b1, col_rxrx_b2, col_rxrx_b3 = st.columns(3)
        # Buy section uses sell() results: r_s_amount, r_s_price, r_s_cost
        st.write('buy', '   ', 'A', r_s_amount, 'P', r_s_price, 'C', r_s_cost)
        buy_match_RXRX = col_rxrx_b3.checkbox('buy_match_RXRX', key='buy_match_rxrx')
        if buy_match_RXRX:
            GO_RXRX_Buy = col_rxrx_b3.button("GO!", key='go_rxrx_buy')
            if GO_RXRX_Buy:
                try:
                    client.update({'field8': RXRX_ASSET_LAST + r_s_amount}) # field8 for RXRX
                    col_rxrx_b3.write(f"RXRX Updated: {RXRX_ASSET_LAST + r_s_amount}")
                    st.rerun()
                except Exception as e:
                    col_rxrx_b3.error(f"ThingSpeak Error: {e}")
else:
    st.warning("Not enough data for RXRX action.")
st.write("_____")


if st.button("RERUN"):
  st.rerun()
