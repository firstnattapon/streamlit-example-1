import streamlit as st
import numpy as np
import datetime
import thingspeak
import pandas as pd
import yfinance as yf
import json

st.set_page_config(page_title="Monitor", page_icon="📈", layout="wide")
channel_id = 2528199
write_api_key = '2E65V8XEIPH9B2VV' # Replace with your actual ThingSpeak Write API Key
client = thingspeak.Channel(channel_id, write_api_key, fmt='json')

def sell(asset=0, fix_c=1500, Diff=60):
    if asset == 0: # Avoid division by zero
        return 0, 0, 0
    s1 = (1500 - Diff) / asset
    s2 = round(s1, 2)
    if s2 == 0: # Avoid division by zero in s5 if s2 is zero
        return 0,0,0
    s3 = s2 * asset
    s4 = abs(s3 - fix_c)
    s5 = round(s4 / s2)
    s6 = s5 * s2
    s7 = (asset * s2) + s6
    return s2, s5, round(s7, 2)

def buy(asset=0, fix_c=1500, Diff=60):
    if asset == 0: # Avoid division by zero
        return 0, 0, 0
    b1 = (1500 + Diff) / asset
    b2 = round(b1, 2)
    if b2 == 0: # Avoid division by zero in b5 if b2 is zero
        return 0,0,0
    b3 = b2 * asset
    b4 = abs(b3 - fix_c)
    b5 = round(b4 / b2)
    b6 = b5 * b2
    b7 = (asset * b2) - b6
    return b2, b5, round(b7, 2)

channel_id_2 = 2385118
write_api_key_2 = 'IPSG3MMMBJEB9DY8' # Replace with your actual ThingSpeak Write API Key for client_2
client_2 = thingspeak.Channel(channel_id_2, write_api_key_2, fmt='json')

def Monitor(Ticker='FFWM', field=2):
    try:
        tickerData = yf.Ticker(Ticker)
        tickerData = round(tickerData.history(period='max')[['Close']], 3)
        tickerData.index = tickerData.index.tz_convert(tz='Asia/Bangkok') # Corrected tz name
        filter_date = '2023-01-01 12:00:00+07:00'
        tickerData = tickerData[tickerData.index >= filter_date]

        fx_raw = client_2.get_field_last(field='{}'.format(field))
        fx_js = 0 # Default value
        if fx_raw:
            try:
                loaded_fx = json.loads(fx_raw)
                if isinstance(loaded_fx, dict) and "field{}".format(field) in loaded_fx and loaded_fx["field{}".format(field)] is not None:
                    fx_js = int(loaded_fx["field{}".format(field)])
                else: # Fallback if field is not in json or is null
                    fx_js = int(datetime.datetime.now().timestamp()) # Use timestamp as a fallback seed
            except (json.JSONDecodeError, TypeError, ValueError) :
                fx_js = int(datetime.datetime.now().timestamp()) # Use timestamp as a fallback seed on error
        else: # Fallback if fx_raw is None
             fx_js = int(datetime.datetime.now().timestamp())


        rng = np.random.default_rng(fx_js)
        data = rng.integers(2, size=len(tickerData))
        tickerData['action'] = data
        tickerData['index'] = [i + 1 for i in range(len(tickerData))]

        tickerData_1 = pd.DataFrame(columns=(tickerData.columns))
        # Ensure 'action' column exists before assigning to it
        if 'action' not in tickerData_1.columns:
            tickerData_1['action'] = pd.Series(dtype='int')

        tickerData_1['action'] = [i for i in range(5)] # This line seems to create actions 0,1,2,3,4 for 5 rows.
        tickerData_1.index = ['+0', "+1", "+2", "+3", "+4"]
        df = pd.concat([tickerData, tickerData_1], axis=0).fillna("")
        
        # Re-seed or use a different RNG if you want different random numbers here
        rng_df_action = np.random.default_rng(fx_js + 1) # Offset seed for potentially different random numbers
        df['action'] = rng_df_action.integers(2, size=len(df))
        
        if len(df) < 7: # Ensure we can always get tail(7)
            # Pad with empty rows if necessary, though concat logic should handle this.
            # This case should be rare if tickerData has enough history and tickerData_1 adds 5 rows.
            num_missing = 7 - len(df)
            empty_df = pd.DataFrame("", index=range(num_missing), columns=df.columns)
            df = pd.concat([df, empty_df], axis=0)


        return df.tail(7), fx_js
    except Exception as e:
        # st.error(f"Error in Monitor for {Ticker} (field {field}): {e}")
        # Return a default DataFrame and fx_js to prevent app crash
        default_cols = ['Close', 'action', 'index']
        default_df = pd.DataFrame(np.nan, index=pd.to_datetime([]), columns=default_cols)
        for i in range(5): # Adding the +0 to +4 rows manually for default
             default_df.loc[f'+{i}'] = [np.nan, np.random.randint(0,2), np.nan] # random action for default
        
        # Pad if still less than 7 rows
        if len(default_df) < 7:
            num_missing = 7 - len(default_df)
            padding_df = pd.DataFrame(np.nan, index=[f"pad_{j}" for j in range(num_missing)], columns=default_cols)
            padding_df['action'] = np.random.randint(0,2, size=num_missing)
            default_df = pd.concat([default_df.head(len(default_df)-num_missing), padding_df], axis=0)

        return default_df.tail(7) , int(datetime.datetime.now().timestamp())


df_7, fx_js = Monitor(Ticker='FFWM', field=2)
df_7_1, fx_js_1 = Monitor(Ticker='NEGG', field=3)
df_7_2, fx_js_2 = Monitor(Ticker='RIVN', field=4)
df_7_3, fx_js_3 = Monitor(Ticker='APLS', field=5)
df_7_4, fx_js_4 = Monitor(Ticker='NVTS', field=6)
df_7_5, fx_js_5 = Monitor(Ticker='QXO', field=7)
df_7_6, fx_js_6 = Monitor(Ticker='RXRX', field=8) # New asset RXRX

nex = 0
Nex_day_sell = 0
toggle = lambda x: 1 - x if x in (0,1) else 0 # Ensure toggle works for 0 and 1

Nex_day_ = st.checkbox('nex_day')
if Nex_day_:
    st.write("value = ", nex)
    nex_col, Nex_day_sell_col, _, _, _ = st.columns(5)

    if nex_col.button("Nex_day"):
        nex = 1
        # st.write("value = ", nex) # st.write invalidates button state, use session state or rerun for complex logic
        st.rerun()


    if Nex_day_sell_col.button("Nex_day_sell"):
        nex = 1
        Nex_day_sell = 1
        # st.write("value = ", nex)
        # st.write("Nex_day_sell = ", Nex_day_sell)
        st.rerun()


st.write("_____")

# Adjusted to 9 columns for the new asset RXRX
col13, col16, col14, col15, col17, col18, col19, col20, col21 = st.columns(9)

x_2 = col16.number_input('Diff', step=1, value=60)

Start = col13.checkbox('start')
if Start:
    thingspeak_1 = col13.checkbox('@_FFWM_ASSET')
    if thingspeak_1:
        add_1 = col13.number_input('@_FFWM_ASSET_val', step=0.001, value=0., key='add_1_ffwm_val') # Unique key
        _FFWM_ASSET = col13.button("GO!")
        if _FFWM_ASSET:
            client.update({'field1': add_1})
            col13.write(add_1)
            st.rerun()


    thingspeak_2 = col13.checkbox('@_NEGG_ASSET')
    if thingspeak_2:
        add_2 = col13.number_input('@_NEGG_ASSET_val', step=0.001, value=0., key='add_2_negg_val') # Unique key
        _NEGG_ASSET = col13.button("GO! ")
        if _NEGG_ASSET:
            client.update({'field2': add_2})
            col13.write(add_2)
            st.rerun()

    thingspeak_3 = col13.checkbox('@_RIVN_ASSET')
    if thingspeak_3:
        add_3 = col13.number_input('@_RIVN_ASSET_val', step=0.001, value=0., key='add_3_rivn_val') # Unique key
        _RIVN_ASSET = col13.button("GO!  ")
        if _RIVN_ASSET:
            client.update({'field3': add_3})
            col13.write(add_3)
            st.rerun()

    thingspeak_4 = col13.checkbox('@_APLS_ASSET')
    if thingspeak_4:
        add_4 = col13.number_input('@_APLS_ASSET_val', step=0.001, value=0., key='add_4_apls_val') # Unique key
        _APLS_ASSET = col13.button("GO!   ")
        if _APLS_ASSET:
            client.update({'field4': add_4})
            col13.write(add_4)
            st.rerun()

    thingspeak_5 = col13.checkbox('@_NVTS_ASSET')
    if thingspeak_5:
        add_5 = col13.number_input('@_NVTS_ASSET_val', step=0.001, value=0., key='add_5_nvts_val') # Unique key
        _NVTS_ASSET = col13.button("GO!    ")
        if _NVTS_ASSET:
            client.update({'field5': add_5})
            col13.write(add_5)
            st.rerun()

    thingspeak_6 = col13.checkbox('@_QXO_ASSET')
    if thingspeak_6:
        add_6 = col13.number_input('@_QXO_ASSET_val', step=0.001, value=0., key='add_6_qxo_val') # Unique key
        _QXO_ASSET = col13.button("GO!     ")
        if _QXO_ASSET:
            client.update({'field6': add_6})
            col13.write(add_6)
            st.rerun()

    thingspeak_7 = col13.checkbox('@_RXRX_ASSET') # New asset RXRX
    if thingspeak_7:
        add_7 = col13.number_input('@_RXRX_ASSET_val', step=0.001, value=0., key='add_7_rxrx_val') # Unique key
        _RXRX_ASSET = col13.button("GO!      ") # Unique button label
        if _RXRX_ASSET:
            client.update({'field7': add_7}) # RXRX to field7
            col13.write(add_7)
            st.rerun()


def get_asset_last_from_thingspeak(raw_json_string, field_key):
    if raw_json_string:
        try:
            data = json.loads(raw_json_string)
            if isinstance(data, dict) and field_key in data and data[field_key] is not None:
                evaluated_value = eval(str(data[field_key]))
                if isinstance(evaluated_value, (int, float)):
                    return float(evaluated_value)
                return 0.0
            return 0.0
        except (json.JSONDecodeError, SyntaxError, NameError, TypeError):
            return 0.0
    return 0.0

FFWM_ASSET_LAST_raw = client.get_field_last(field='field1')
FFWM_ASSET_LAST = get_asset_last_from_thingspeak(FFWM_ASSET_LAST_raw, 'field1')

NEGG_ASSET_LAST_raw = client.get_field_last(field='field2')
NEGG_ASSET_LAST = get_asset_last_from_thingspeak(NEGG_ASSET_LAST_raw, 'field2')

RIVN_ASSET_LAST_raw = client.get_field_last(field='field3')
RIVN_ASSET_LAST = get_asset_last_from_thingspeak(RIVN_ASSET_LAST_raw, 'field3')

APLS_ASSET_LAST_raw = client.get_field_last(field='field4')
APLS_ASSET_LAST = get_asset_last_from_thingspeak(APLS_ASSET_LAST_raw, 'field4')

NVTS_ASSET_LAST_raw = client.get_field_last(field='field5')
NVTS_ASSET_LAST = get_asset_last_from_thingspeak(NVTS_ASSET_LAST_raw, 'field5')

QXO_ASSET_LAST_raw = client.get_field_last(field='field6')
QXO_ASSET_LAST = get_asset_last_from_thingspeak(QXO_ASSET_LAST_raw, 'field6')

RXRX_ASSET_LAST_raw = client.get_field_last(field='field7') # New RXRX from field7
RXRX_ASSET_LAST = get_asset_last_from_thingspeak(RXRX_ASSET_LAST_raw, 'field7')


x_3 = col14.number_input('NEGG_ASSET', step=0.001, value=NEGG_ASSET_LAST, key='x3_negg')
x_4 = col15.number_input('FFWM_ASSET', step=0.001, value=FFWM_ASSET_LAST, key='x4_ffwm')
x_5 = col17.number_input('RIVN_ASSET', step=0.001, value=RIVN_ASSET_LAST, key='x5_rivn')
x_6 = col18.number_input('APLS_ASSET', step=0.001, value=APLS_ASSET_LAST, key='x6_apls')
x_7 = col19.number_input('NVTS_ASSET', step=0.001, value=NVTS_ASSET_LAST, key='x7_nvts')

QXO_OPTION = 79.
QXO_REAL = col20.number_input('QXO_ASSET (LV:79@19.0)', step=0.001, value=QXO_ASSET_LAST, key='qxo_real')
x_8 = QXO_OPTION + QXO_REAL

x_9 = col21.number_input('RXRX_ASSET', step=0.001, value=RXRX_ASSET_LAST, key='x9_rxrx') # New RXRX input

st.write("_____")

# try: # Original code had a try here, ensure all variables are defined before use
s8, s9, s10 = sell(asset=x_3, Diff=x_2)
s11, s12, s13 = sell(asset=x_4, Diff=x_2)
b8, b9, b10 = buy(asset=x_3, Diff=x_2)
b11, b12, b13 = buy(asset=x_4, Diff=x_2)
u1, u2, u3 = sell(asset=x_5, Diff=x_2)
u4, u5, u6 = buy(asset=x_5, Diff=x_2)
p1, p2, p3 = sell(asset=x_6, Diff=x_2)
p4, p5, p6 = buy(asset=x_6, Diff=x_2)
u7, u8, u9 = sell(asset=x_7, Diff=x_2) # NVTS sell uses u7,u8,u9
p7, p8, p9 = buy(asset=x_7, Diff=x_2)  # NVTS buy uses p7,p8,p9. Check variable names consistency
                                      # Original code has u7,u8,u9 for NVTS sell and p7,p8,p9 for NVTS buy.
                                      # However, the display for NVTS buy uses u8, u7, u9. This implies
                                      # p7,p8,p9 from buy() should map to u7,u8,u9 (price, amount, cost) for consistency
                                      # Let's keep original variable assignment and adjust display if needed.
                                      # For clarity, I'll rename NVTS buy vars: nvts_b_p, nvts_b_a, nvts_b_c
                                      # And NVTS sell vars: nvts_s_p, nvts_s_a, nvts_s_c
# For NVTS:
nvts_s_p, nvts_s_a, nvts_s_c = sell(asset=x_7, Diff=x_2) # (u7, u8, u9 in original)
nvts_b_p, nvts_b_a, nvts_b_c = buy(asset=x_7, Diff=x_2)  # (p7, p8, p9 in original)


q1, q2, q3 = sell(asset=x_8, Diff=x_2)
q4, q5, q6 = buy(asset=x_8, Diff=x_2)

rx_s_p, rx_s_a, rx_s_c = sell(asset=x_9, Diff=x_2) # New RXRX sell (price, amount, cost)
rx_b_p, rx_b_a, rx_b_c = buy(asset=x_9, Diff=x_2)  # New RXRX buy (price, amount, cost)


# Helper to safely get action value
def get_action_value(df_action_series, index):
    try:
        return df_action_series.values[index]
    except IndexError:
        return 0 # Default action if index is out of bounds


Limut_Order_NEGG = st.checkbox('Limut_Order_NEGG', value=np.where(Nex_day_sell == 1, toggle(get_action_value(df_7_1.action, 1 + nex)), get_action_value(df_7_1.action, 1 + nex)))
if Limut_Order_NEGG:
    st.write('sell', '     ', 'A', b9, 'P', b8, 'C', b10) # NEGG uses b9,b8,b10 for sell info (from buy function) - this is confusing. Assuming it's intended.
                                                       # buy() returns price, amount, cost. So b8 is price, b9 is amount.
                                                       # For 'sell' display, it usually means "target sell price".
                                                       # If b variables from buy() are used for a sell limit order:
                                                       # Amount b9 at Price b8 for Cost b10.

    col1, col2, col3 = st.columns(3)
    sell_negg = col3.checkbox('sell_match_NEGG')
    if sell_negg:
        GO_NEGG_SELL = col3.button("GO!   NEGG Sell") # Descriptive unique key
        if GO_NEGG_SELL:
            client.update({'field2': NEGG_ASSET_LAST - b9})
            col3.write(NEGG_ASSET_LAST - b9)
            st.rerun()

    try:
        pv_negg = yf.Ticker('NEGG').fast_info['lastPrice'] * x_3
        st.write(yf.Ticker('NEGG').fast_info['lastPrice'], pv_negg, '(', pv_negg - 1500, ')')
    except Exception as e:
        st.write("Could not fetch NEGG price.")


    col4, col5, col6 = st.columns(3)
    st.write('buy', '     ', 'A', s9, 'P', s8, 'C', s10) # NEGG uses s9,s8,s10 for buy info (from sell function)
    buy_negg = col6.checkbox('buy_match_NEGG')
    if buy_negg:
        GO_NEGG_Buy = col6.button("GO!    NEGG Buy") # Descriptive unique key
        if GO_NEGG_Buy:
            client.update({'field2': NEGG_ASSET_LAST + s9})
            col6.write(NEGG_ASSET_LAST + s9)
            st.rerun()
st.write("_____")


Limut_Order_FFWM = st.checkbox('Limut_Order_FFWM', value=np.where(Nex_day_sell == 1, toggle(get_action_value(df_7.action, 1 + nex)), get_action_value(df_7.action, 1 + nex)))
if Limut_Order_FFWM:
    st.write('sell', '     ', 'A', b12, 'P', b11, 'C', b13) # FFWM uses b12,b11,b13 for sell (from buy function)
    col7, col8, col9 = st.columns(3)
    sell_ffwm = col9.checkbox('sell_match_FFWM')
    if sell_ffwm:
        GO_ffwm_sell = col9.button("GO!     FFWM Sell") # Descriptive unique key
        if GO_ffwm_sell:
            client.update({'field1': FFWM_ASSET_LAST - b12})
            col9.write(FFWM_ASSET_LAST - b12)
            st.rerun()
    try:
        pv_ffwm = yf.Ticker('FFWM').fast_info['lastPrice'] * x_4
        st.write(yf.Ticker('FFWM').fast_info['lastPrice'], pv_ffwm, '(', pv_ffwm - 1500, ')')
    except Exception as e:
        st.write("Could not fetch FFWM price.")


    col10, col11, col12 = st.columns(3)
    st.write('buy', '     ', 'A', s12, 'P', s11, 'C', s13) # FFWM uses s12,s11,s13 for buy (from sell function)
    buy_ffwm = col12.checkbox('buy_match_FFWM')
    if buy_ffwm:
        GO_ffwm_Buy = col12.button("GO!      FFWM Buy") # Descriptive unique key
        if GO_ffwm_Buy:
            client.update({'field1': FFWM_ASSET_LAST + s12})
            col12.write(FFWM_ASSET_LAST + s12)
            st.rerun()
st.write("_____")


Limut_Order_RIVN = st.checkbox('Limut_Order_RIVN', value=np.where(Nex_day_sell == 1, toggle(get_action_value(df_7_2.action, 1 + nex)), get_action_value(df_7_2.action, 1 + nex)))
if Limut_Order_RIVN:
    st.write('sell', '     ', 'A', u5, 'P', u4, 'C', u6) # RIVN uses u5,u4,u6 for sell (from buy function, u4=price, u5=amount)
    col77, col88, col99 = st.columns(3)
    sell_RIVN = col99.checkbox('sell_match_RIVN')
    if sell_RIVN:
        GO_RIVN_sell = col99.button("GO!       RIVN Sell") # Descriptive unique key
        if GO_RIVN_sell:
            client.update({'field3': RIVN_ASSET_LAST - u5})
            col99.write(RIVN_ASSET_LAST - u5)
            st.rerun()
    try:
        pv_rivn = yf.Ticker('RIVN').fast_info['lastPrice'] * x_5
        st.write(yf.Ticker('RIVN').fast_info['lastPrice'], pv_rivn, '(', pv_rivn - 1500, ')')
    except Exception as e:
        st.write("Could not fetch RIVN price.")

    col100, col111, col122 = st.columns(3)
    st.write('buy', '     ', 'A', u2, 'P', u1, 'C', u3) # RIVN uses u2,u1,u3 for buy (from sell function, u1=price, u2=amount)
    buy_RIVN = col122.checkbox('buy_match_RIVN')
    if buy_RIVN:
        GO_RIVN_Buy = col122.button("GO!        RIVN Buy") # Descriptive unique key
        if GO_RIVN_Buy:
            client.update({'field3': RIVN_ASSET_LAST + u2})
            col122.write(RIVN_ASSET_LAST + u2)
            st.rerun()
st.write("_____")


Limut_Order_APLS = st.checkbox('Limut_Order_APLS', value=np.where(Nex_day_sell == 1, toggle(get_action_value(df_7_3.action, 1 + nex)), get_action_value(df_7_3.action, 1 + nex)))
if Limut_Order_APLS:
    st.write('sell', '     ', 'A', p5, 'P', p4, 'C', p6) # APLS uses p5,p4,p6 for sell (from buy function, p4=price, p5=amount)
    col7777, col8888, col9999 = st.columns(3)
    sell_APLS = col9999.checkbox('sell_match_APLS')
    if sell_APLS:
        GO_APLS_sell = col9999.button("GO!         APLS Sell") # Descriptive unique key
        if GO_APLS_sell:
            client.update({'field4': APLS_ASSET_LAST - p5})
            col9999.write(APLS_ASSET_LAST - p5)
            st.rerun()
    try:
        pv_apls = yf.Ticker('APLS').fast_info['lastPrice'] * x_6
        st.write(yf.Ticker('APLS').fast_info['lastPrice'], pv_apls, '(', pv_apls - 1500, ')')
    except Exception as e:
        st.write("Could not fetch APLS price.")

    col1000, col1111, col1222 = st.columns(3)
    st.write('buy', '     ', 'A', p2, 'P', p1, 'C', p3) # APLS uses p2,p1,p3 for buy (from sell function, p1=price, p2=amount)
    buy_APLS = col1222.checkbox('buy_match_APLS')
    if buy_APLS:
        GO_APLS_Buy = col1222.button("GO!          APLS Buy") # Descriptive unique key
        if GO_APLS_Buy:
            client.update({'field4': APLS_ASSET_LAST + p2})
            col1222.write(APLS_ASSET_LAST + p2)
            st.rerun()
st.write("_____")

# Corrected NVTS section based on new variables nvts_s_p, nvts_s_a, etc.
# sell() returns price, amount, cost. buy() returns price, amount, cost.
# Standard display: Price (P), Amount (A), Cost (C)
# For a sell limit order: you sell 'Amount' at 'Price', total 'Cost'
# For a buy limit order: you buy 'Amount' at 'Price', total 'Cost'

Limut_Order_NVTS = st.checkbox('Limut_Order_NVTS', value=np.where(Nex_day_sell == 1, toggle(get_action_value(df_7_4.action, 1 + nex)), get_action_value(df_7_4.action, 1 + nex)))
if Limut_Order_NVTS:
    # Display for SELL: using results from buy() function (nvts_b_p, nvts_b_a, nvts_b_c)
    # Original logic: st.write('sell', '     ', 'A', p8 , 'P', p7 , 'C', p9 )
    # p7, p8, p9 are from buy(asset=x_7). So p7=price, p8=amount, p9=cost.
    st.write('sell', '     ', 'A', nvts_b_a, 'P', nvts_b_p, 'C', nvts_b_c)

    col_nvts1, col_nvts2, col_nvts3 = st.columns(3)
    sell_NVTS = col_nvts3.checkbox('sell_match_NVTS')
    if sell_NVTS:
        GO_NVTS_sell = col_nvts3.button("GO!           NVTS Sell") # Descriptive unique key
        if GO_NVTS_sell:
            client.update({'field5': NVTS_ASSET_LAST - nvts_b_a}) # sell amount nvts_b_a
            col_nvts3.write(NVTS_ASSET_LAST - nvts_b_a)
            st.rerun()
    try:
        pv_nvts = yf.Ticker('NVTS').fast_info['lastPrice'] * x_7
        st.write(yf.Ticker('NVTS').fast_info['lastPrice'], pv_nvts, '(', pv_nvts - 1500, ')')
    except Exception as e:
        st.write("Could not fetch NVTS price.")

    col_nvts4, col_nvts5, col_nvts6 = st.columns(3)
    # Display for BUY: using results from sell() function (nvts_s_p, nvts_s_a, nvts_s_c)
    # Original logic: st.write('buy', '     ', 'A', u8, 'P', u7 , 'C',u9 )
    # u7, u8, u9 are from sell(asset=x_7). So u7=price, u8=amount, u9=cost.
    st.write('buy', '     ', 'A', nvts_s_a, 'P', nvts_s_p, 'C', nvts_s_c)
    buy_NVTS = col_nvts6.checkbox('buy_match_NVTS')
    if buy_NVTS:
        GO_NVTS_Buy = col_nvts6.button("GO!            NVTS Buy") # Descriptive unique key
        if GO_NVTS_Buy:
            client.update({'field5': NVTS_ASSET_LAST + nvts_s_a}) # buy amount nvts_s_a
            col_nvts6.write(NVTS_ASSET_LAST + nvts_s_a)
            st.rerun()
st.write("_____")


Limut_Order_QXO = st.checkbox('Limut_Order_QXO', value=np.where(Nex_day_sell == 1, toggle(get_action_value(df_7_5.action, 1 + nex)), get_action_value(df_7_5.action, 1 + nex)))
if Limut_Order_QXO:
    st.write('sell', '     ', 'A', q5, 'P', q4, 'C', q6) # QXO uses q5,q4,q6 for sell (from buy function, q4=price, q5=amount)
    col_qxo1, col_qxo2, col_qxo3 = st.columns(3)
    sell_QXO = col_qxo3.checkbox('sell_match_QXO')
    if sell_QXO:
        GO_QXO_sell = col_qxo3.button("GO!             QXO Sell") # Descriptive unique key
        if GO_QXO_sell:
            client.update({'field6': QXO_ASSET_LAST - q5})
            col_qxo3.write(QXO_ASSET_LAST - q5)
            st.rerun()
    try:
        pv_qxo = yf.Ticker('QXO').fast_info['lastPrice'] * x_8
        st.write(yf.Ticker('QXO').fast_info['lastPrice'], pv_qxo, '(', pv_qxo - 1500, ')')
    except Exception as e:
        st.write("Could not fetch QXO price.")

    col_qxo4, col_qxo5, col_qxo6 = st.columns(3)
    st.write('buy', '     ', 'A', q2, 'P', q1, 'C', q3) # QXO uses q2,q1,q3 for buy (from sell function, q1=price, q2=amount)
    buy_QXO = col_qxo6.checkbox('buy_match_QXO')
    if buy_QXO:
        GO_QXO_Buy = col_qxo6.button("GO!              QXO Buy") # Descriptive unique key
        if GO_QXO_Buy:
            client.update({'field6': QXO_ASSET_LAST + q2})
            col_qxo6.write(QXO_ASSET_LAST + q2)
            st.rerun()
st.write("_____")

# --- New Section for RXRX ---
Limut_Order_RXRX = st.checkbox('Limut_Order_RXRX', value=np.where(Nex_day_sell == 1, toggle(get_action_value(df_7_6.action, 1 + nex)), get_action_value(df_7_6.action, 1 + nex)))
if Limut_Order_RXRX:
    # For SELL display, using results from buy(x_9) -> rx_b_p, rx_b_a, rx_b_c
    # rx_b_p = price, rx_b_a = amount, rx_b_c = cost
    st.write('sell', '     ', 'A', rx_b_a, 'P', rx_b_p, 'C', rx_b_c)

    col_rxrx1, col_rxrx2, col_rxrx3 = st.columns(3)
    sell_RXRX = col_rxrx3.checkbox('sell_match_RXRX')
    if sell_RXRX:
        GO_RXRX_sell = col_rxrx3.button("GO!               RXRX Sell") # Descriptive unique key
        if GO_RXRX_sell:
            client.update({'field7': RXRX_ASSET_LAST - rx_b_a}) # Update field7 for RXRX
            col_rxrx3.write(RXRX_ASSET_LAST - rx_b_a)
            st.rerun()
    try:
        # Ensure Ticker is correct for yfinance
        pv_rxrx = yf.Ticker('RXRX').fast_info['lastPrice'] * x_9
        st.write(yf.Ticker('RXRX').fast_info['lastPrice'], pv_rxrx, '(', pv_rxrx - 1500, ')')
    except Exception as e:
        st.write(f"Could not fetch RXRX price: {e}")


    col_rxrx4, col_rxrx5, col_rxrx6 = st.columns(3)
    # For BUY display, using results from sell(x_9) -> rx_s_p, rx_s_a, rx_s_c
    # rx_s_p = price, rx_s_a = amount, rx_s_c = cost
    st.write('buy', '     ', 'A', rx_s_a, 'P', rx_s_p, 'C', rx_s_c)
    buy_RXRX = col_rxrx6.checkbox('buy_match_RXRX')
    if buy_RXRX:
        GO_RXRX_Buy = col_rxrx6.button("GO!                RXRX Buy") # Descriptive unique key
        if GO_RXRX_Buy:
            client.update({'field7': RXRX_ASSET_LAST + rx_s_a}) # Update field7 for RXRX
            col_rxrx6.write(RXRX_ASSET_LAST + rx_s_a)
            st.rerun()
st.write("_____")
# --- End of New Section for RXRX ---


if st.button("RERUN SCRIPT"): # Changed label slightly for clarity
    st.rerun()

# except:pass # Original code had a try-except pass here. It's generally better to handle specific exceptions.
