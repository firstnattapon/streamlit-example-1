import streamlit as st
import numpy as np
import datetime
import thingspeak
import pandas as pd
import yfinance as yf
import json
from functools import lru_cache
import concurrent.futures
from threading import Lock

st.set_page_config(page_title="Monitor", page_icon="📈", layout="wide")

_cache_lock = Lock()
_price_cache = {}
_cache_timestamp = {}

@st.cache_resource
def get_clients():
    channel_id = 2528199
    write_api_key = '2E65V8XEIPH9B2VV'
    client = thingspeak.Channel(channel_id, write_api_key, fmt='json')
    
    channel_id_2 = 2385118
    write_api_key_2 = 'IPSG3MMMBJEB9DY8'
    client_2 = thingspeak.Channel(channel_id_2, write_api_key_2, fmt='json')
    
    return client, client_2

client, client_2 = get_clients()

def clear_all_caches():
    st.cache_data.clear()
    st.cache_resource.clear()
    
    sell.cache_clear()
    buy.cache_clear()
    
    with _cache_lock:
        _price_cache.clear()
        _cache_timestamp.clear()
    
    st.success("🗑️ Clear ALL caches complete!")
    st.rerun()

@lru_cache(maxsize=128)
def sell(asset, fix_c=1500, Diff=60):
    if asset == 0:
        return 0, 0, 0
    s1 = (fix_c - Diff) / asset
    s2 = round(s1, 2)
    s3 = s2 * asset
    s4 = abs(s3 - fix_c)
    s5 = round(s4 / s2)
    s6 = s5 * s2
    s7 = (asset * s2) + s6
    return s2, s5, round(s7, 2)

@lru_cache(maxsize=128)
def buy(asset, fix_c=1500, Diff=60):
    if asset == 0:
        return 0, 0, 0
    b1 = (fix_c + Diff) / asset
    b2 = round(b1, 2)
    b3 = b2 * asset
    b4 = abs(b3 - fix_c)
    b5 = round(b4 / b2)
    b6 = b5 * b2
    b7 = (asset * b2) - b6
    return b2, b5, round(b7, 2)

def get_cached_price(ticker, max_age=30):
    current_time = datetime.datetime.now()
    
    with _cache_lock:
        if (ticker in _price_cache and 
            ticker in _cache_timestamp and 
            (current_time - _cache_timestamp[ticker]).seconds < max_age):
            return _price_cache[ticker]
    
    try:
        price = yf.Ticker(ticker).fast_info['lastPrice']
        with _cache_lock:
            _price_cache[ticker] = price
            _cache_timestamp[ticker] = current_time
        return price
    except:
        return 0.0

@st.cache_data(ttl=300)
def Monitor(Ticker='FFWM', field=2):
    try:
        tickerData = yf.Ticker(Ticker)
        tickerData = round(tickerData.history(period='max')[['Close']], 3)
        tickerData.index = tickerData.index.tz_convert(tz='Asia/bangkok')
        filter_date = '2023-01-01 12:00:00+07:00'
        tickerData = tickerData[tickerData.index >= filter_date]

        fx = client_2.get_field_last(field='{}'.format(field))
        fx_js = int(json.loads(fx)["field{}".format(field)])
        rng = np.random.default_rng(fx_js)
        data = rng.integers(2, size=len(tickerData))
        tickerData['action'] = data
        tickerData['index'] = [i+1 for i in range(len(tickerData))]

        tickerData_1 = pd.DataFrame(columns=(tickerData.columns))
        tickerData_1['action'] = [i for i in range(5)]
        tickerData_1.index = ['+0', "+1", "+2", "+3", "+4"]
        df = pd.concat([tickerData, tickerData_1], axis=0).fillna("")
        rng = np.random.default_rng(fx_js)
        df['action'] = rng.integers(2, size=len(df))
        return df.tail(7), fx_js
    except Exception as e:
        st.error(f"Error in Monitor function for {Ticker}: {str(e)}")
        return pd.DataFrame(), 0

def fetch_monitor_data():
    tickers_fields = [
        ('AGL', 1), ('APLS', 5), ('FFWM', 2), ('NEGG', 3), ('NVTS', 6), 
        ('QXO', 7), ('RIVN', 4), ('RXRX', 8)
    ]
    
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tickers_fields)) as executor:
        future_to_ticker = {
            executor.submit(Monitor, ticker, field): ticker 
            for ticker, field in tickers_fields
        }
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                results[ticker] = future.result()
            except Exception as e:
                st.error(f"Error fetching data for {ticker}: {str(e)}")
                results[ticker] = (pd.DataFrame(), 0)
    
    return results

monitor_results = fetch_monitor_data()
df_AGL, fx_js_AGL = monitor_results.get('AGL', (pd.DataFrame(), 0))
df_7_3, fx_js_3 = monitor_results.get('APLS', (pd.DataFrame(), 0))
df_7, fx_js = monitor_results.get('FFWM', (pd.DataFrame(), 0))
df_7_1, fx_js_1 = monitor_results.get('NEGG', (pd.DataFrame(), 0))
df_7_4, fx_js_4 = monitor_results.get('NVTS', (pd.DataFrame(), 0))
df_7_5, fx_js_5 = monitor_results.get('QXO', (pd.DataFrame(), 0))
df_7_2, fx_js_2 = monitor_results.get('RIVN', (pd.DataFrame(), 0))
df_7_6, fx_js_6 = monitor_results.get('RXRX', (pd.DataFrame(), 0))

@st.cache_data(ttl=60)
def get_all_assets():
    fields = ['field1', 'field2', 'field3', 'field4', 'field5', 'field6', 'field7', 'field8']
    assets = {}
    
    def fetch_asset(field):
        try:
            data = client.get_field_last(field=field)
            return eval(json.loads(data)[field])
        except:
            return 0.0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(fields)) as executor:
        future_to_field = {executor.submit(fetch_asset, field): field for field in fields}
        
        for future in concurrent.futures.as_completed(future_to_field):
            field = future_to_field[future]
            try:
                assets[field] = future.result()
            except:
                assets[field] = 0.0
    
    return assets

all_assets = get_all_assets()
AGL_ASSET_LAST = all_assets.get('field8', 0.0)
APLS_ASSET_LAST = all_assets.get('field4', 0.0)
FFWM_ASSET_LAST = all_assets.get('field1', 0.0)
NEGG_ASSET_LAST = all_assets.get('field2', 0.0)
NVTS_ASSET_LAST = all_assets.get('field5', 0.0)
QXO_ASSET_LAST = all_assets.get('field6', 0.0)
RIVN_ASSET_LAST = all_assets.get('field3', 0.0)
RXRX_ASSET_LAST = all_assets.get('field7', 0.0)

nex = 0
Nex_day_sell = 0
toggle = lambda x: 1 - x

Nex_day_ = st.checkbox('nex_day')
if Nex_day_:
    st.write("value = ", nex)
    nex_col, Nex_day_sell_col, _, _, _ = st.columns(5)

    if nex_col.button("Nex_day"):
        nex = 1
        st.write("value = ", nex)

    if Nex_day_sell_col.button("Nex_day_sell"):
        nex = 1
        Nex_day_sell = 1
        st.write("value = ", nex)
        st.write("Nex_day_sell = ", Nex_day_sell)

st.write("_____")

col13, col16, col22, col14, col15, col17, col18, col19, col20, col21 = st.columns(10)

x_2 = col16.number_input('Diff', step=1, value=60)

Start = col13.checkbox('start')
if Start:
    asset_configs = [
        ('AGL', 'field8', AGL_ASSET_LAST),
        ('APLS', 'field4', APLS_ASSET_LAST),
        ('FFWM', 'field1', FFWM_ASSET_LAST),
        ('NEGG', 'field2', NEGG_ASSET_LAST),
        ('NVTS', 'field5', NVTS_ASSET_LAST),
        ('QXO', 'field6', QXO_ASSET_LAST),
        ('RIVN', 'field3', RIVN_ASSET_LAST),
        ('RXRX', 'field7', RXRX_ASSET_LAST)
    ]
    
    for i, (name, field, default_val) in enumerate(asset_configs):
        checkbox_key = f'@_{name}_ASSET'
        thingspeak_checkbox = col13.checkbox(checkbox_key)
        if thingspeak_checkbox:
            add_val = col13.number_input(checkbox_key, step=0.001, value=0., key=f'input_{name}')
            button_key = f"GO_{name}"
            asset_button = col13.button(button_key, key=f'btn_{name}')
            if asset_button:
                client.update({field: add_val})
                col13.write(add_val)
                clear_all_caches()

AGL_OPTION = 500.
AGL_REAL = col22.number_input('AGL (LV:500@3.0)', step=0.001, value=AGL_ASSET_LAST)
x_10 = AGL_OPTION + AGL_REAL

x_6 = col14.number_input('APLS_ASSET', step=0.001, value=APLS_ASSET_LAST)

FFWM_OPTION = 200. 
FFWM_REAL = col15.number_input('FFWM (LV:200@6.88)', step=0.001, value=FFWM_ASSET_LAST)
x_4 = FFWM_OPTION + FFWM_REAL

x_3 = col17.number_input('NEGG_ASSET', step=0.001, value=NEGG_ASSET_LAST)
x_7 = col18.number_input('NVTS_ASSET', step=0.001, value=NVTS_ASSET_LAST)

QXO_OPTION = 79.
QXO_REAL = col19.number_input('QXO (LV:79@19.0)', step=0.001, value=QXO_ASSET_LAST)
x_8 = QXO_OPTION + QXO_REAL

x_5 = col20.number_input('RIVN_ASSET', step=0.001, value=RIVN_ASSET_LAST)

RXRX_OPTION = 200.
RXRX_REAL = col21.number_input('RXRX (LV:200@5.4)', step=0.001, value=RXRX_ASSET_LAST)
x_9 = RXRX_OPTION + RXRX_REAL

st.write("_____")

calculations = {}
assets = [x_10, x_6, x_4, x_3, x_7, x_8, x_5, x_9]
asset_names = ['AGL', 'APLS', 'FFWM', 'NEGG', 'NVTS', 'QXO', 'RIVN', 'RXRX']

for i, (asset, name) in enumerate(zip(assets, asset_names)):
    fix_c = 2100 if name == 'NVTS' else 1500
    calculations[name] = {
        'sell': sell(asset, fix_c=fix_c, Diff=x_2),
        'buy': buy(asset, fix_c=fix_c, Diff=x_2)
    }

def create_trading_section(ticker, asset_val, asset_last, df_data, field_num, calculations_key, nex, Nex_day_sell):
    try:
        action_val = np.where(
            Nex_day_sell == 1,
            toggle(df_data.action.values[1+nex]),
            df_data.action.values[1+nex]
        ) if len(df_data) > 1+nex else 0
    except:
        action_val = 0
    
    limit_order = st.checkbox(f'Limut_Order_{ticker}', value=bool(action_val), key=f'limit_order_{ticker}')
    
    if limit_order:
        sell_calc = calculations[calculations_key]['sell']
        buy_calc = calculations[calculations_key]['buy']
        
        st.write('sell', '    ', 'A', buy_calc[1], 'P', buy_calc[0], 'C', buy_calc[2])
        
        col1, col2, col3 = st.columns(3)
        sell_match = col3.checkbox(f'sell_match_{ticker}')
        if sell_match:
            go_sell = col3.button(f"GO_SELL_{ticker}")
            if go_sell:
                client.update({f'field{field_num}': asset_last - buy_calc[1]})
                col3.write(asset_last - buy_calc[1])
                clear_all_caches()
        
        try:
            current_price = get_cached_price(ticker)
            pv = current_price * asset_val
            fix_value = 2100 if ticker == 'NVTS' else 1500 
            st.write(current_price, pv, '(', pv - fix_value, ')')
        except:
            st.write("Price unavailable")
        
        col4, col5, col6 = st.columns(3)
        st.write('buy', '   ', 'A', sell_calc[1], 'P', sell_calc[0], 'C', sell_calc[2])
        buy_match = col6.checkbox(f'buy_match_{ticker}')
        if buy_match:
            go_buy = col6.button(f"GO_BUY_{ticker}")
            if go_buy:
                client.update({f'field{field_num}': asset_last + sell_calc[1]})
                col6.write(asset_last + sell_calc[1])
                clear_all_caches()

trading_configs = [
    ('AGL', x_10, AGL_ASSET_LAST, df_AGL, 8, 'AGL'),
    ('APLS', x_6, APLS_ASSET_LAST, df_7_3, 4, 'APLS'),
    ('FFWM', x_4, FFWM_ASSET_LAST, df_7, 1, 'FFWM'),
    ('NEGG', x_3, NEGG_ASSET_LAST, df_7_1, 2, 'NEGG'),
    ('NVTS', x_7, NVTS_ASSET_LAST, df_7_4, 5, 'NVTS'),
    ('QXO', x_8, QXO_ASSET_LAST, df_7_5, 6, 'QXO'),
    ('RIVN', x_5, RIVN_ASSET_LAST, df_7_2, 3, 'RIVN'),
    ('RXRX', x_9, RXRX_ASSET_LAST, df_7_6, 7, 'RXRX')
]

for config in trading_configs:
    create_trading_section(*config, nex, Nex_day_sell)
    st.write("_____")

if st.sidebar.button("RERUN"):
    clear_all_caches()
