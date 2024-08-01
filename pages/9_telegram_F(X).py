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
  
  def send_message(chat_id, text):
      bot.send_message(chat_id, text)
  
  Ticker = ['FFWM' , 'NEGG' , 'RIVN' , 'APLS']
  Ticker_data = [FFWM_ASSET_LAST , NEGG_ASSET_LAST , RIVN_ASSET_LAST , APLS_ASSET_LAST]
  text = ['....']
  for idx , v in enumerate(Ticker_data):
    diff = yf.Ticker(Ticker[idx]).fast_info['lastPrice'] * v
    if diff < 1440 or diff > 1560 :
      text_add = '{} - {}'.format( Ticker[idx] , diff) 
      text.append(text_add)

  for i in text:
    send_message(chat_id, i)
  time.sleep(300)

