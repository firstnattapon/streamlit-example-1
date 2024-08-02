import telebot
import streamlit as st
import time
import thingspeak
import json
import yfinance as yf

while True: 
    TOKEN = '7248188209:AAFhpA9doHJpESvb2BLkIz6sHTmUeUjOG6E'
    chat_id = 7355273754
    bot = telebot.TeleBot(TOKEN)
    
    channel_id = 2528199
    write_api_key = '2E65V8XEIPH9B2VV'
    client = thingspeak.Channel(channel_id, write_api_key , fmt='json')
    
    FFWM_ASSET_LAST = client.get_field_last(field='field1')
    FFWM_ASSET_LAST =  eval(json.loads(FFWM_ASSET_LAST)['field1'])
    
    NEGG_ASSET_LAST = client.get_field_last(field='field2')
    NEGG_ASSET_LAST = eval(json.loads(NEGG_ASSET_LAST)['field2'])
    
    RIVN_ASSET_LAST = client.get_field_last(field='field3')
    RIVN_ASSET_LAST = eval(json.loads(RIVN_ASSET_LAST)['field3'])
    
    APLS_ASSET_LAST = client.get_field_last(field='field4')
    APLS_ASSET_LAST = eval(json.loads(APLS_ASSET_LAST)['field4'])
    
    channel_id_2 = 2385118
    write_api_key_2 = 'IPSG3MMMBJEB9DY8'
    client_2 = thingspeak.Channel(channel_id_2, write_api_key_2 , fmt='json')
    def Monitor (Ticker = 'FFWM' , field = 2 ):
        tickerData = yf.Ticker( Ticker)
        tickerData = round(tickerData.history(period= 'max' )[['Close']] , 3 )
        tickerData.index = tickerData.index.tz_convert(tz='Asia/bangkok')
        filter_date = '2022-12-21 12:00:00+07:00'
        tickerData = tickerData[tickerData.index >= filter_date]
        
        fx = client_2.get_field_last(field='{}'.format(field))
        fx =  json.loads(fx)
        fx =  fx["field{}".format(field)] 
        fx_js = int(fx)
        
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
    
    FFWM_fx , _  = Monitor(Ticker = 'FFWM', field = 2)
    NEGG_fx , _  = Monitor(Ticker = 'NEGG', field = 3)
    RIVN_fx , _  = Monitor(Ticker = 'RIVN', field = 4)
    APLS_fx , _  = Monitor(Ticker = 'APLS', field = 5)
    Dict_fx = {'FFWM': FFWM_fx.action.values[1], 'NEGG': NEGG_fx.action.values[1] , 'RIVN': RIVN_fx.action.values[1] , 'APLS': APLS_fx.action.values[1]}

    def send_message(chat_id, text):
      bot.send_message(chat_id, text)
    
    Ticker = ['FFWM' , 'NEGG' , 'RIVN' , 'APLS']
    Ticker_data = [FFWM_ASSET_LAST , NEGG_ASSET_LAST , RIVN_ASSET_LAST , APLS_ASSET_LAST]
    text = ['....']
    for idx , v in enumerate(Ticker_data):
    diff = yf.Ticker(Ticker[idx]).fast_info['lastPrice'] * v
    if diff < 1440 or diff > 1560 :
      if Dict_fx[Ticker[idx]] == 1:
        text_add = '{} - {}'.format( Ticker[idx] , diff) 
        text.append(text_add)
    
    for i in text:
    send_message(chat_id, i)
    time.sleep(300)
