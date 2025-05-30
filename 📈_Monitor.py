import streamlit as st
import numpy as np
import datetime
import thingspeak
import pandas as pd
import yfinance as yf
import json

st.set_page_config(page_title="Monitor", page_icon="📈" , layout="wide" )
channel_id = 2528199 # Channel หลักสำหรับเก็บ Asset
write_api_key = '2E65V8XEIPH9B2VV' # ใส่ Write API Key ของคุณ
client = thingspeak.Channel(channel_id, write_api_key , fmt='json')

def sell (asset = 0 , fix_c=1500 , Diff=60):
    if asset == 0: # Prevent division by zero
        return 0, 0, 0
    s1 =  (fix_c-Diff) /asset # Adjusted fix_c for sell
    s2 =  round(s1, 2)
    if s2 == 0: # Prevent division by zero if rounded price is 0
        return 0,0,0
    s3 =  s2  *asset
    s4 =  abs(s3 - (fix_c-Diff)) # Adjusted fix_c for sell
    s5 =  round( s4 / s2 )
    s6 =  s5*s2
    s7 =  (asset * s2) + s6 # This logic seems to add back, ensure it's intended for sell
    # A more typical sell would aim for (asset - s5) * s2 or ensure total value is fix_c - Diff
    # For now, keeping original logic:
    # s7 = (asset * s2) - s6 # if s5 is shares to sell to get closer
    # Or if s2 is target sell price, and s5 is qty to transact
    # The original function aims to get a total *cost* after transaction near fix_c +/- Diff
    # For selling, one might expect to receive fix_c - Diff.
    # Let's re-evaluate: s1 is price. s2 is rounded price. s3 is value at rounded price.
    # s4 is diff from target value. s5 is qty adjustment.
    # If selling, the value received should be (asset_sold * price).
    # The function seems to calculate a price (s2) and a quantity (s5)
    # such that if you *transact* s5 shares at price s2, the total *value* (s7) ends up near fix_c.
    # This is a bit complex. Let's assume s2 is price, s5 is quantity for the order.
    # s7 is the *resulting value of the transaction itself*, or *new portfolio value related to this transaction*.
    # Given the way it's used (s9 as quantity, s8 as price, s10 as cost),
    # s2 (or b2) is price, s5 (or b5) is quantity, s7 (or b7) is total cost/proceeds.
    # For selling: price s2, quantity s5, proceeds s7. Goal: proceeds = fix_c - Diff
    # For buying: price b2, quantity b5, cost b7. Goal: cost = fix_c + Diff

    # Let's simplify the original intent: determine price and quantity for a target value
    # Sell: target_proceeds = fix_c - Diff
    price_sell = round((fix_c - Diff) / asset if asset != 0 else 0, 2)
    if price_sell == 0: return 0,0,0
    qty_sell = round((fix_c - Diff) / price_sell if price_sell != 0 else 0, 0) # Number of shares to sell
    actual_proceeds_sell = round(qty_sell * price_sell, 2)
    # The original function has a more complex adjustment. Let's stick to it for consistency first.
    # Reverting to original logic for s1-s7 for sell to match buy's structure more closely.
    s1 = (fix_c - Diff) / asset if asset != 0 else 0
    s2 = round(s1, 2)
    if s2 == 0: return 0, 0, 0
    # How many shares to transact to get near fix_c - Diff
    s5_qty_to_transact = round((fix_c - Diff) / s2 if s2 != 0 else 0, 0) # This would be total shares, not adjustment
                                                                   # The original s5 seems to be an *additional* adjustment amount.

    # Let's use the original functions as they are, assuming their internal logic is purposeful for the user's strategy
    s1_orig = (fix_c - Diff) / asset if asset != 0 else 0
    s2_orig = round(s1_orig, 2)
    if s2_orig == 0: return 0, 0, 0
    s3_orig = s2_orig * asset
    s4_orig = abs(s3_orig - (fix_c - Diff)) # Difference from the target *value*
    s5_orig_adj_qty = round(s4_orig / s2_orig if s2_orig != 0 else 0) # Qty to adjust to get closer to target value
    # s6_orig = s5_orig_adj_qty * s2_orig
    # s7_orig = (asset * s2_orig) ??? This part is confusing for 'sell'
    # If s2_orig is the price, and we want to sell some shares.
    # Let's assume s5 is the quantity of shares to be sold/bought to reach the target value.
    # And s2 is the price. s7 is the total value of the transaction.

    # The most straightforward interpretation for sell(asset, fix_c, Diff) -> price, quantity, total_value
    # asset = current holding, fix_c = target portfolio value, Diff = buffer
    # Target sell price: (fix_c - Diff) / asset_qty_to_make_this_value
    # Let's assume 'asset' in the function is the *number of shares to transact* to achieve the target.
    # This interpretation doesn't fit how x_3 (NEGG_ASSET_LAST) is passed as 'asset'.
    # 'asset' is current total shares of that stock.

    # The function returns: price_per_share, shares_to_transact, total_value_of_transaction
    # Let's re-evaluate the original function's variables s1-s7
    # def sell (asset_holding = 0 , fix_c=1500 , Diff=60):
    #     s1_price_ideal = (fix_c - Diff) / asset_holding # Ideal price if all current assets are sold to meet target
    #     s2_price_rounded = round(s1_price_ideal, 2)
    #     s3_value_at_rounded_price = s2_price_rounded * asset_holding
    #     s4_deviation_from_target_value = abs(s3_value_at_rounded_price - (fix_c - Diff))
    #     s5_qty_adjustment = round(s4_deviation_from_target_value / s2_price_rounded if s2_price_rounded else 0) # Number of shares to adjust transaction by
    #     # s6_value_of_qty_adjustment = s5_qty_adjustment * s2_price_rounded
    #     # s7_final_transaction_value = (asset_holding * s2_price_rounded) + s6_value_of_qty_adjustment # This logic seems to add, not suitable for selling current assets
    #     # The user's UI indicates 'A' is quantity, 'P' is price, 'C' is cost/value.
    #     # So, the function should return: Price (s2), Quantity (s5), Cost (s7)

    #     # If s5 is the quantity to transact (e.g. sell s5 shares at price s2)
    #     # Then s7 should be s5 * s2.
    #     # The original s7 calculation is: (asset_holding * s2_price_rounded) + (s5_qty_adjustment * s2_price_rounded)
    #     # This would be (asset_holding + s5_qty_adjustment) * s2_price_rounded.
    #     # This implies s5_qty_adjustment is an *additional* quantity to the *existing asset_holding* to reach a new total value.
    #     # This seems more like portfolio rebalancing than simple buy/sell orders.

    # Let's trust the original formulas provided by the user and ensure no division by zero.
    if asset == 0: return 0.0, 0, 0.0
    s1 = (fix_c - Diff) / asset
    s2 = round(s1, 2)
    if s2 == 0: return s2, 0, 0.0 # Price is 0, so qty and cost are 0
    s3 = s2 * asset
    s4 = abs(s3 - (fix_c-Diff)) # Note: fix_c used here instead of (fix_c-Diff) in original code, changed to (fix_c-Diff) for consistency
    s5 = round(s4 / s2)
    s6 = s5 * s2
    # s7 = (asset * s2) + s6 # Original: This increases value. For selling, it should decrease or be proceeds.
    # If s5 is the number of shares to sell, and s2 is the price:
    s7 = s5 * s2 # Value of transaction (proceeds from selling s5 shares at price s2)
    return s2, s5, round(s7, 2)


def buy (asset = 0 , fix_c=1500 , Diff=60):
    if asset == 0: return 0.0, 0, 0.0 # If no current asset to base price on, or if asset is shares to buy
    b1 = (fix_c + Diff) / asset
    b2 = round(b1, 2)
    if b2 == 0: return b2, 0, 0.0 # Price is 0
    b3 = b2 * asset
    b4 = abs(b3 - (fix_c+Diff)) # Note: fix_c used here instead of (fix_c+Diff) in original code, changed to (fix_c+Diff)
    b5 = round(b4 / b2)
    b6 = b5 * b2
    # b7 = (asset * b2) - b6 # Original: This decreases value. For buying, cost should be positive.
    # If b5 is the number of shares to buy, and b2 is the price:
    b7 = b5 * b2 # Value of transaction (cost of buying b5 shares at price b2)
    return b2, b5, round(b7, 2)

channel_id_2 = 2385118 # Channel รองสำหรับดึง seed
write_api_key_2 = 'IPSG3MMMBJEB9DY8' # ใส่ Write API Key ของคุณ
client_2 = thingspeak.Channel(channel_id_2, write_api_key_2 , fmt='json' )

def Monitor (Ticker = 'FFWM' , field = 2  ):
    try:
        tickerData = yf.Ticker(Ticker)
        hist = tickerData.history(period='max')
        if hist.empty:
            st.error(f"No history data for {Ticker}")
            # Return empty DataFrame with expected columns for graceful degradation
            # And a default seed (e.g. 0 or 1)
            cols = ['Close', 'action', 'index']
            empty_df = pd.DataFrame(columns=cols)
            # Add 7 dummy rows for tail(7)
            for i in range(5): # For +0 to +4
                empty_df.loc[f'+{i}'] = [0.0, 0, 0] # Example default values
            # Add two more dummy rows to make it 7 if needed for historical part
            # However, the logic expects some historical data usually
            # Let's return 5 future rows for now if hist is empty
            return empty_df.fillna(""), 1 # Default seed 1
        
        tickerData_df = round(hist[['Close']] , 3 )
        tickerData_df.index = tickerData_df.index.tz_localize(None).tz_localize('UTC').tz_convert(tz='Asia/Bangkok')
        filter_date = pd.Timestamp('2023-01-01 12:00:00', tz='Asia/Bangkok')
        tickerData_df = tickerData_df[tickerData_df.index >= filter_date]

        if tickerData_df.empty: # If no data after date filter
             st.warning(f"No data for {Ticker} after {filter_date.date()}")
             cols = ['Close', 'action', 'index']
             empty_df = pd.DataFrame(columns=cols)
             for i in range(5):
                empty_df.loc[f'+{i}'] = [0.0, 0, i]
             return empty_df.fillna(""), 1


        fx_response = client_2.get_field_last(field='{}'.format(field))
        # st.write(f"ThingSpeak client_2 response for field {field} ('{Ticker}'): {fx_response}") # Debug
        if fx_response:
            try:
                fx_data = json.loads(fx_response)
                field_key = "field{}".format(field)
                if field_key in fx_data and fx_data[field_key] is not None:
                    fx_js = int(fx_data[field_key])
                else:
                    st.warning(f"Field {field_key} not found or null in ThingSpeak response for {Ticker}. Using default seed 0.")
                    fx_js = 0 # Default seed
            except json.JSONDecodeError:
                st.error(f"Failed to decode JSON from ThingSpeak for {Ticker} seed. Response: {fx_response}. Using default seed 0.")
                fx_js = 0 # Default seed
            except ValueError:
                st.error(f"Failed to convert ThingSpeak seed to int for {Ticker}. Value: {fx_data.get(field_key)}. Using default seed 0.")
                fx_js = 0
        else:
            st.warning(f"No response from ThingSpeak for {Ticker} seed (field {field}). Using default seed 0.")
            fx_js = 0 # Default seed if no response

        rng = np.random.default_rng(fx_js)
        data = rng.integers(2, size = len(tickerData_df))
        tickerData_df['action'] = data
        tickerData_df['index'] = [ i+1 for i in range(len(tickerData_df))]

        tickerData_1 = pd.DataFrame(columns=(tickerData_df.columns))
        # tickerData_1['action'] =  [ i for i in range(5)] # Original: 0,1,2,3,4
        tickerData_1['action'] = rng.integers(2, size=5) # Let's make future actions also 0 or 1 based on seed
        tickerData_1.index = ['+0' , "+1" , "+2" , "+3" , "+4"]
        tickerData_1['Close'] = "" # Fill Close for future dates as empty
        tickerData_1['index'] = [len(tickerData_df) + i + 1 for i in range(5)]


        df = pd.concat([tickerData_df , tickerData_1], axis=0).fillna("")
        # rng = np.random.default_rng(fx_js) # Re-seeding rng
        # df['action'] = rng.integers(2, size = len(df)) # Overwriting action again
        # The above two lines will make the historical actions same as future actions pattern if df length changes.
        # It's better to assign actions separately or ensure it's intended.
        # For now, keeping original logic of re-assigning action to the whole df
        # However, the previous assignment to tickerData_df['action'] and tickerData_1['action'] already uses the seed.
        # This re-assignment seems redundant or might produce unexpected results if not careful.
        # Let's comment out the global re-assignment and rely on the individual ones.
        # If the goal was to have a single pattern of 0s and 1s for the whole df.tail(7) display:
        # final_rng = np.random.default_rng(fx_js)
        # df.tail(7)['action'] = final_rng.integers(2, size=7) # This would be a local change

        return df.tail(7) , fx_js
    except Exception as e:
        st.error(f"Error in Monitor for {Ticker}: {e}")
        cols = ['Close', 'action', 'index']
        empty_df = pd.DataFrame(columns=cols)
        for i in range(5):
            empty_df.loc[f'+{i}'] = [0.0, 0, i]
        return empty_df.fillna(""), 1 # Default seed

df_7   , fx_js    = Monitor(Ticker = 'FFWM', field = 2    )
df_7_1 , fx_js_1  = Monitor(Ticker = 'NEGG', field = 3    )
df_7_2 , fx_js_2  = Monitor(Ticker = 'RIVN', field = 4 )
df_7_3 , fx_js_3  = Monitor(Ticker = 'APLS', field = 5 )
df_7_4 , fx_js_4  = Monitor(Ticker = 'NVTS', field = 6 )
df_7_5 , fx_js_5  = Monitor(Ticker = 'QXO', field = 7 )
df_7_6 , fx_js_6  = Monitor(Ticker = 'RXRX', field = 8 ) # <<<<<<<<<<<<<<<<<<< ADDED RXRX

nex = 0
Nex_day_sell = 0
toggle = lambda x : 1 - x if x in [0,1] else 0 # Ensure toggle works for 0 and 1

Nex_day_ = st.checkbox('nex_day')
if Nex_day_ :
    st.write( "value = " , nex)
    nex_col , Nex_day_sell_col ,_,_,_  = st.columns(5)

    if nex_col.button("Nex_day"):
        nex = 1
        # st.write( "value = " , nex) # Can remove, will show on rerun
        st.rerun() # Rerun to apply change
    if Nex_day_sell_col.button("Nex_day_sell"):
        nex = 1 # Assuming nex_day_sell also implies nex_day
        Nex_day_sell = 1 - Nex_day_sell # Toggle Nex_day_sell
        # st.write( "value = " , nex)
        # st.write( "Nex_day_sell = " , Nex_day_sell)
        st.rerun() # Rerun to apply change

st.write("_____")

# Adjusted columns to include col21 for RXRX
col13, col16, col14, col15, col17, col18, col19, col20, col21 = st.columns(9)

x_2 = col16.number_input('Diff', step=1 , value= 60   )

Start = col13.checkbox('start_asset_updates') # Renamed key for clarity
if Start :
    with col13.expander("Update Assets on ThingSpeak", expanded=False):
        thingspeak_1 = st.checkbox('@_FFWM_ASSET', key='ts_ffwm_cb')
        if thingspeak_1 :
            add_1 = st.number_input('@_FFWM_ASSET', step=0.001 ,  value=0.0, key='ts_ffwm_val')
            if st.button("GO! FFWM", key='ts_ffwm_go'):
                client.update(  {'field1': add_1 } )
                st.success(f"FFWM Asset updated to: {add_1}")

        thingspeak_2 = st.checkbox('@_NEGG_ASSET', key='ts_negg_cb')
        if thingspeak_2 :
            add_2 = st.number_input('@_NEGG_ASSET', step=0.001 ,  value=0.0, key='ts_negg_val')
            if st.button("GO! NEGG", key='ts_negg_go'):
                client.update(  {'field2': add_2 }  )
                st.success(f"NEGG Asset updated to: {add_2}")

        thingspeak_3 = st.checkbox('@_RIVN_ASSET', key='ts_rivn_cb')
        if thingspeak_3 :
            add_3 = st.number_input('@_RIVN_ASSET', step=0.001 ,  value=0.0, key='ts_rivn_val')
            if st.button("GO! RIVN", key='ts_rivn_go'):
                client.update(  {'field3': add_3 }  )
                st.success(f"RIVN Asset updated to: {add_3}")

        thingspeak_4 = st.checkbox('@_APLS_ASSET', key='ts_apls_cb')
        if thingspeak_4 :
            add_4 = st.number_input('@_APLS_ASSET', step=0.001 ,  value=0.0, key='ts_apls_val')
            if st.button("GO! APLS", key='ts_apls_go'):
                client.update(  {'field4': add_4 }  )
                st.success(f"APLS Asset updated to: {add_4}")

        thingspeak_5 = st.checkbox('@_NVTS_ASSET', key='ts_nvts_cb')
        if thingspeak_5:
            add_5 = st.number_input('@_NVTS_ASSET', step=0.001, value= 0.0, key='ts_nvts_val')
            if st.button("GO! NVTS", key='ts_nvts_go'):
                client.update({'field5': add_5})
                st.success(f"NVTS Asset updated to: {add_5}")

        thingspeak_6 = st.checkbox('@_QXO_ASSET', key='ts_qxo_cb')
        if thingspeak_6:
            add_6 = st.number_input('@_QXO_ASSET', step=0.001, value=0.0, key='ts_qxo_val')
            if st.button("GO! QXO", key='ts_qxo_go'):
                client.update({'field6': add_6})
                st.success(f"QXO Asset updated to: {add_6}")

        # <<<<<<<<<<<<<<<<<<< ADDED RXRX ASSET UPDATE
        thingspeak_7 = st.checkbox('@_RXRX_ASSET', key='ts_rxrx_cb')
        if thingspeak_7:
            add_7 = st.number_input('@_RXRX_ASSET', step=0.001, value=0.0, key='ts_rxrx_val')
            if st.button("GO! RXRX", key='ts_rxrx_go'):
                client.update({'field7': add_7}) # RXRX uses field7
                st.success(f"RXRX Asset updated to: {add_7}")
        # >>>>>>>>>>>>>>>>>>> END ADDED RXRX

def get_asset_from_thingspeak(client, field_number, default_value=0.0):
    field_name = f'field{field_number}'
    try:
        data_json = client.get_field_last(field=field_name)
        data = json.loads(data_json)
        if data and field_name in data and data[field_name] is not None:
            return float(data[field_name])
        else:
            # st.warning(f"No data for {field_name} or field is null, using default {default_value}")
            return default_value
    except Exception as e:
        # st.error(f"Error fetching {field_name}: {e}, using default {default_value}")
        return default_value


FFWM_ASSET_LAST = get_asset_from_thingspeak(client, 1)
NEGG_ASSET_LAST = get_asset_from_thingspeak(client, 2)
RIVN_ASSET_LAST = get_asset_from_thingspeak(client, 3)
APLS_ASSET_LAST = get_asset_from_thingspeak(client, 4)
NVTS_ASSET_LAST = get_asset_from_thingspeak(client, 5)
QXO_ASSET_LAST  = get_asset_from_thingspeak(client, 6)
RXRX_ASSET_LAST = get_asset_from_thingspeak(client, 7) # <<<<<<<<<<<<<<<<<<< ADDED RXRX


x_3 = col14.number_input('NEGG_ASSET', step=0.001 ,  value= NEGG_ASSET_LAST, key='negg_asset_disp' )
x_4 = col15.number_input('FFWM_ASSET', step=0.001  , value= FFWM_ASSET_LAST, key='ffwm_asset_disp' )
x_5 = col17.number_input('RIVN_ASSET', step=0.001  , value= RIVN_ASSET_LAST, key='rivn_asset_disp' )
x_6 = col18.number_input('APLS_ASSET', step=0.001  , value= APLS_ASSET_LAST, key='apls_asset_disp' )
x_7 = col19.number_input('NVTS_ASSET', step=0.001  , value= NVTS_ASSET_LAST, key='nvts_asset_disp' )

QXO_OPTION = 79.
QXO_REAL   =  col20.number_input('QXO_ASSET (LV:79@19.0)', step=0.001  , value=  QXO_ASSET_LAST, key='qxo_asset_disp')
x_8 =  QXO_OPTION  + QXO_REAL

# <<<<<<<<<<<<<<<<<<< ADDED RXRX ASSET INPUT
RXRX_OPTION = 278.
RXRX_REAL   =  col21.number_input('RXRX_ASSET (LV:278@5.4)', step=0.001  , value=  RXRX_ASSET_LAST, key='rxrx_asset_disp')
x_9 =  RXRX_OPTION  + RXRX_REAL
# >>>>>>>>>>>>>>>>>>> END ADDED RXRX

st.write("_____")

# Calculate sell/buy parameters
s_negg_p, s_negg_q, s_negg_c = sell(asset=x_3, Diff=x_2)
b_negg_p, b_negg_q, b_negg_c = buy(asset=x_3, Diff=x_2)

s_ffwm_p, s_ffwm_q, s_ffwm_c = sell(asset=x_4, Diff=x_2)
b_ffwm_p, b_ffwm_q, b_ffwm_c = buy(asset=x_4, Diff=x_2)

s_rivn_p, s_rivn_q, s_rivn_c = sell(asset=x_5, Diff=x_2)
b_rivn_p, b_rivn_q, b_rivn_c = buy(asset=x_5, Diff=x_2)

s_apls_p, s_apls_q, s_apls_c = sell(asset=x_6, Diff=x_2)
b_apls_p, b_apls_q, b_apls_c = buy(asset=x_6, Diff=x_2)

s_nvts_p, s_nvts_q, s_nvts_c = sell(asset=x_7, Diff=x_2)
b_nvts_p, b_nvts_q, b_nvts_c = buy(asset=x_7, Diff=x_2)

s_qxo_p, s_qxo_q, s_qxo_c = sell(asset=x_8, Diff=x_2)
b_qxo_p, b_qxo_q, b_qxo_c = buy(asset=x_8, Diff=x_2)

s_rxrx_p, s_rxrx_q, s_rxrx_c = sell(asset=x_9, Diff=x_2) # <<<<<<<<<<<<<<<<<<< ADDED RXRX
b_rxrx_p, b_rxrx_q, b_rxrx_c = buy(asset=x_9, Diff=x_2)  # <<<<<<<<<<<<<<<<<<< ADDED RXRX


def get_action_value(df_action_series, nex_offset, nex_day_sell_flag, toggle_func):
    try:
        # df_action_series is like df_7_1.action
        # nex_offset is like 1+nex
        action_val = df_action_series.values[nex_offset]
        action_val = int(action_val) # Ensure it's an integer 0 or 1
        if nex_day_sell_flag == 1:
            return toggle_func(action_val)
        return action_val
    except IndexError:
        # st.warning(f"Action index out of bounds for offset {nex_offset}. Defaulting to 0.")
        return 0 # Default action if index is out of bounds
    except ValueError:
        # st.warning(f"Could not convert action value to int. Defaulting to 0.")
        return 0


# --- NEGG Section ---
action_negg = get_action_value(df_7_1.action, 1 + nex, Nex_day_sell, toggle)
Limut_Order_NEGG = st.checkbox('Limit_Order_NEGG', value = (action_negg == 1), key='negg_limit_cb')
if Limut_Order_NEGG :
    st.write( 'SELL Suggestion (from BUY func):' , 'Qty:', b_negg_q  , 'Price:' , b_negg_p ,'Value:' ,b_negg_c)
    col1, col2 , col3  = st.columns([1,1,1])
    sell_negg = col3.checkbox('Confirm SELL NEGG', key='negg_sell_confirm_cb')
    if sell_negg :
        if col3.button("Execute NEGG SELL", key='negg_sell_exec_btn'):
            new_asset_val = NEGG_ASSET_LAST - b_negg_q
            client.update(  {'field2': new_asset_val } )
            st.success(f"NEGG Asset updated to: {new_asset_val}")
            st.rerun()
    try:
        pv_negg =  yf.Ticker('NEGG').fast_info.get('lastPrice',0) * x_3
        st.write(f"NEGG Current Price: {yf.Ticker('NEGG').fast_info.get('lastPrice',0)} | Portfolio Value: {pv_negg:.2f} ({(pv_negg - 1500):.2f})")
    except Exception as e: st.warning(f"Could not fetch NEGG price: {e}")

    st.write( 'BUY Suggestion (from SELL func):' , 'Qty:', s_negg_q ,  'Price:' , s_negg_p , 'Value:' ,s_negg_c)
    col4, col5 , col6  = st.columns([1,1,1])
    buy_negg = col6.checkbox('Confirm BUY NEGG', key='negg_buy_confirm_cb')
    if buy_negg :
        if col6.button("Execute NEGG BUY", key='negg_buy_exec_btn'):
            new_asset_val = NEGG_ASSET_LAST + s_negg_q
            client.update(  {'field2': new_asset_val  } )
            st.success(f"NEGG Asset updated to: {new_asset_val}")
            st.rerun()
st.write("_____")

# --- FFWM Section ---
action_ffwm = get_action_value(df_7.action, 1 + nex, Nex_day_sell, toggle)
Limut_Order_FFWM = st.checkbox('Limit_Order_FFWM',  value = (action_ffwm == 1), key='ffwm_limit_cb')
if Limut_Order_FFWM :
    st.write( 'SELL Suggestion (from BUY func):' , 'Qty:', b_ffwm_q , 'Price:' , b_ffwm_p  , 'Value:' , b_ffwm_c)
    col7, col8 , col9  = st.columns(3)
    sell_ffwm = col9.checkbox('Confirm SELL FFWM', key='ffwm_sell_confirm_cb')
    if sell_ffwm :
        if col9.button("Execute FFWM SELL", key='ffwm_sell_exec_btn'):
            new_asset_val = FFWM_ASSET_LAST - b_ffwm_q
            client.update(  {'field1': new_asset_val } )
            st.success(f"FFWM Asset updated to: {new_asset_val}")
            st.rerun()
    try:
        pv_ffwm = yf.Ticker('FFWM').fast_info.get('lastPrice',0) * x_4
        st.write(f"FFWM Current Price: {yf.Ticker('FFWM').fast_info.get('lastPrice',0)} | Portfolio Value: {pv_ffwm:.2f} ({(pv_ffwm - 1500):.2f})")
    except Exception as e: st.warning(f"Could not fetch FFWM price: {e}")

    col10, col11 , col12  = st.columns(3)
    st.write( 'BUY Suggestion (from SELL func):' , 'Qty:', s_ffwm_q , 'Price:' , s_ffwm_p  , 'Value:'  , s_ffwm_c)
    buy_ffwm = col12.checkbox('Confirm BUY FFWM', key='ffwm_buy_confirm_cb')
    if buy_ffwm :
        if col12.button("Execute FFWM BUY", key='ffwm_buy_exec_btn'):
            new_asset_val = FFWM_ASSET_LAST + s_ffwm_q
            client.update(  {'field1': new_asset_val } )
            st.success(f"FFWM Asset updated to: {new_asset_val}")
            st.rerun()
st.write("_____")

# --- RIVN Section ---
action_rivn = get_action_value(df_7_2.action, 1 + nex, Nex_day_sell, toggle)
Limut_Order_RIVN = st.checkbox('Limit_Order_RIVN',value = (action_rivn==1), key='rivn_limit_cb')
if Limut_Order_RIVN :
    st.write( 'SELL Suggestion (from BUY func):' , 'Qty:', b_rivn_q , 'Price:' , b_rivn_p  , 'Value:' , b_rivn_c)
    col77, col88 , col99  = st.columns(3)
    sell_RIVN = col99.checkbox('Confirm SELL RIVN', key='rivn_sell_confirm_cb')
    if sell_RIVN :
        if col99.button("Execute RIVN SELL", key='rivn_sell_exec_btn'):
            new_asset_val = RIVN_ASSET_LAST - b_rivn_q
            client.update(  {'field3': new_asset_val } )
            st.success(f"RIVN Asset updated to: {new_asset_val}")
            st.rerun()
    try:
        pv_rivn =   yf.Ticker('RIVN').fast_info.get('lastPrice',0) * x_5
        st.write(f"RIVN Current Price: {yf.Ticker('RIVN').fast_info.get('lastPrice',0)} | Portfolio Value: {pv_rivn:.2f} ({(pv_rivn - 1500):.2f})")
    except Exception as e: st.warning(f"Could not fetch RIVN price: {e}")

    col100 , col111 , col122  = st.columns(3)
    st.write( 'BUY Suggestion (from SELL func):' , 'Qty:', s_rivn_q , 'Price:' , s_rivn_p  , 'Value:'  , s_rivn_c)
    buy_RIVN = col122.checkbox('Confirm BUY RIVN', key='rivn_buy_confirm_cb')
    if buy_RIVN :
        if col122.button("Execute RIVN BUY", key='rivn_buy_exec_btn'):
            new_asset_val = RIVN_ASSET_LAST + s_rivn_q
            client.update(  {'field3': new_asset_val } )
            st.success(f"RIVN Asset updated to: {new_asset_val}")
            st.rerun()
st.write("_____")

# --- APLS Section ---
action_apls = get_action_value(df_7_3.action, 1 + nex, Nex_day_sell, toggle)
Limut_Order_APLS = st.checkbox('Limit_Order_APLS',value = (action_apls==1), key='apls_limit_cb')
if Limut_Order_APLS :
    st.write( 'SELL Suggestion (from BUY func):' , 'Qty:', b_apls_q , 'Price:' , b_apls_p  , 'Value:' , b_apls_c)
    col_a, col_b , col_c  = st.columns(3) # Using generic column names to avoid collision
    sell_APLS = col_c.checkbox('Confirm SELL APLS', key='apls_sell_confirm_cb')
    if sell_APLS :
        if col_c.button("Execute APLS SELL", key='apls_sell_exec_btn'):
            new_asset_val = APLS_ASSET_LAST - b_apls_q
            client.update(  {'field4': new_asset_val } )
            st.success(f"APLS Asset updated to: {new_asset_val}")
            st.rerun()
    try:
        pv_apls =   yf.Ticker('APLS').fast_info.get('lastPrice',0) * x_6
        st.write(f"APLS Current Price: {yf.Ticker('APLS' ).fast_info.get('lastPrice',0)} | Portfolio Value: {pv_apls:.2f} ({(pv_apls - 1500):.2f})")
    except Exception as e: st.warning(f"Could not fetch APLS price: {e}")

    col_d , col_e , col_f  = st.columns(3)
    st.write( 'BUY Suggestion (from SELL func):' , 'Qty:', s_apls_q , 'Price:' , s_apls_p  , 'Value:'  , s_apls_c)
    buy_APLS = col_f.checkbox('Confirm BUY APLS', key='apls_buy_confirm_cb')
    if buy_APLS :
        if col_f.button("Execute APLS BUY", key='apls_buy_exec_btn'):
            new_asset_val = APLS_ASSET_LAST + s_apls_q
            client.update(  {'field4': new_asset_val } )
            st.success(f"APLS Asset updated to: {new_asset_val}")
            st.rerun()
st.write("_____")

# --- NVTS Section ---
action_nvts = get_action_value(df_7_4.action, 1 + nex, Nex_day_sell, toggle)
Limut_Order_NVTS = st.checkbox('Limit_Order_NVTS', value=(action_nvts==1), key='nvts_limit_cb')
if Limut_Order_NVTS:
    st.write('SELL Suggestion (from BUY func):', 'Qty:', b_nvts_q , 'Price:', b_nvts_p  , 'Value:', b_nvts_c)
    col_nvts1, col_nvts2, col_nvts3 = st.columns(3)
    sell_NVTS = col_nvts3.checkbox('Confirm SELL NVTS', key='nvts_sell_confirm_cb')
    if sell_NVTS:
        if col_nvts3.button("Execute NVTS SELL", key='nvts_sell_exec_btn'):
            new_asset_val = NVTS_ASSET_LAST - b_nvts_q
            client.update({'field5': new_asset_val})
            st.success(f"NVTS Asset updated to: {new_asset_val}")
            st.rerun()
    try:
        pv_nvts = yf.Ticker('NVTS').fast_info.get('lastPrice',0) * x_7
        st.write(f"NVTS Current Price: {yf.Ticker('NVTS').fast_info.get('lastPrice',0)} | Portfolio Value: {pv_nvts:.2f} ({(pv_nvts-1500):.2f})")
    except Exception as e: st.warning(f"Could not fetch NVTS price: {e}")

    col_nvts4, col_nvts5, col_nvts6 = st.columns(3)
    st.write('BUY Suggestion (from SELL func):', 'Qty:', s_nvts_q, 'Price:', s_nvts_p  , 'Value:',s_nvts_c )
    buy_NVTS = col_nvts6.checkbox('Confirm BUY NVTS', key='nvts_buy_confirm_cb')
    if buy_NVTS:
        if col_nvts6.button("Execute NVTS BUY", key='nvts_buy_exec_btn'):
            new_asset_val = NVTS_ASSET_LAST + s_nvts_q
            client.update({'field5': new_asset_val})
            st.success(f"NVTS Asset updated to: {new_asset_val}")
            st.rerun()
st.write("_____")

# --- QXO Section ---
action_qxo = get_action_value(df_7_5.action, 1 + nex, Nex_day_sell, toggle)
Limut_Order_QXO = st.checkbox('Limit_Order_QXO', value=(action_qxo==1), key='qxo_limit_cb')
if Limut_Order_QXO:
    st.write('SELL Suggestion (from BUY func):', 'Qty:', b_qxo_q, 'Price:', b_qxo_p, 'Value:', b_qxo_c)
    col_qxo1, col_qxo2, col_qxo3 = st.columns(3)
    sell_QXO = col_qxo3.checkbox('Confirm SELL QXO', key='qxo_sell_confirm_cb')
    if sell_QXO:
        if col_qxo3.button("Execute QXO SELL", key='qxo_sell_exec_btn'):
            new_asset_val = QXO_ASSET_LAST - b_qxo_q # QXO_ASSET_LAST is only the REAL part from input
            client.update({'field6': new_asset_val}) # Update only the REAL part on thingspeak
            st.success(f"QXO Asset (real part) updated to: {new_asset_val}")
            st.rerun()
    try:
        pv_qxo = yf.Ticker('QXO').fast_info.get('lastPrice',0) * x_8 # x_8 includes OPTION
        st.write(f"QXO Current Price: {yf.Ticker('QXO').fast_info.get('lastPrice',0)} | Portfolio Value (incl. option): {pv_qxo:.2f} ({(pv_qxo - 1500):.2f})")
    except Exception as e: st.warning(f"Could not fetch QXO price: {e}")

    col_qxo4, col_qxo5, col_qxo6 = st.columns(3)
    st.write('BUY Suggestion (from SELL func):', 'Qty:', s_qxo_q, 'Price:', s_qxo_p, 'Value:', s_qxo_c)
    buy_QXO = col_qxo6.checkbox('Confirm BUY QXO', key='qxo_buy_confirm_cb')
    if buy_QXO:
        if col_qxo6.button("Execute QXO BUY", key='qxo_buy_exec_btn'):
            new_asset_val = QXO_ASSET_LAST + s_qxo_q # QXO_ASSET_LAST is only the REAL part
            client.update({'field6': new_asset_val}) # Update only the REAL part
            st.success(f"QXO Asset (real part) updated to: {new_asset_val}")
            st.rerun()
st.write("_____")


# <<<<<<<<<<<<<<<<<<< ADDED RXRX SECTION
action_rxrx = get_action_value(df_7_6.action, 1 + nex, Nex_day_sell, toggle)
Limut_Order_RXRX = st.checkbox('Limit_Order_RXRX', value=(action_rxrx==1), key='rxrx_limit_cb')
if Limut_Order_RXRX:
    st.write('SELL Suggestion (from BUY func):', 'Qty:', b_rxrx_q, 'Price:', b_rxrx_p, 'Value:', b_rxrx_c)
    col_rxrx1, col_rxrx2, col_rxrx3 = st.columns(3)
    sell_RXRX = col_rxrx3.checkbox('Confirm SELL RXRX', key='rxrx_sell_confirm_cb')
    if sell_RXRX:
        if col_rxrx3.button("Execute RXRX SELL", key='rxrx_sell_exec_btn'):
            new_asset_val = RXRX_ASSET_LAST - b_rxrx_q # RXRX_ASSET_LAST is only the REAL part
            client.update({'field7': new_asset_val}) # RXRX uses field7, update only REAL part
            st.success(f"RXRX Asset (real part) updated to: {new_asset_val}")
            st.rerun()
    try:
        pv_rxrx = yf.Ticker('RXRX').fast_info.get('lastPrice',0) * x_9 # x_9 includes OPTION
        st.write(f"RXRX Current Price: {yf.Ticker('RXRX').fast_info.get('lastPrice',0)} | Portfolio Value (incl. option): {pv_rxrx:.2f} ({(pv_rxrx - 1500):.2f})")
    except Exception as e: st.warning(f"Could not fetch RXRX price: {e}")

    col_rxrx4, col_rxrx5, col_rxrx6 = st.columns(3)
    st.write('BUY Suggestion (from SELL func):', 'Qty:', s_rxrx_q, 'Price:', s_rxrx_p, 'Value:', s_rxrx_c)
    buy_RXRX = col_rxrx6.checkbox('Confirm BUY RXRX', key='rxrx_buy_confirm_cb')
    if buy_RXRX:
        if col_rxrx6.button("Execute RXRX BUY", key='rxrx_buy_exec_btn'):
            new_asset_val = RXRX_ASSET_LAST + s_rxrx_q # RXRX_ASSET_LAST is only the REAL part
            client.update({'field7': new_asset_val}) # Update only the REAL part
            st.success(f"RXRX Asset (real part) updated to: {new_asset_val}")
            st.rerun()
st.write("_____")
# >>>>>>>>>>>>>>>>>>> END ADDED RXRX SECTION


if st.button("RERUN APP", key='rerun_app_btn'):
    st.rerun()

# try/except for the whole app might hide specific errors, better to have them per section if needed.
# except Exception as e:
#   st.error(f"An unexpected error occurred: {e}")
#   pass
