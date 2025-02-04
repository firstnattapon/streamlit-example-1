import telebot
import streamlit as st
import time
import thingspeak
import json
import yfinance as yf
import numpy as np
import pandas as pd

st.set_page_config(page_title="Book_F(X)", page_icon="📑" , layout="wide")
@st.cache_data
def iframe ( frame = ''):
  src = frame
  st.components.v1.iframe(src, width=1500 , height=800  , scrolling=0)

tab1,  tab2, tab3 , tab4 , tab5   = st.tabs(["บทที่_1", "บทที่_2" ,"บทที่_3", "บทที่_4" , "บทที่_5"  ])

with tab1:
  iframe(frame = "https://monica.im/share/chat?shareId=DeGdfM5eVeodP6Vn")
  st.write('____')
  
with tab2:
  iframe(frame = "https://monica.im/share/chat?shareId=yPPCO6zGemygtGg5")
  st.write('____')

with tab3:
  iframe(frame = "https://monica.im/share/chat?shareId=XUh5BcMKrOWcuczd")
  st.write('____')

with tab4:
  iframe(frame = "https://monica.im/share/chat?shareId=sXn5UKzj8lDE4N0y")
  st.write('____')

with tab5:
  iframe(frame = "https://monica.im/share/chat?shareId=pc9UBD2WxBh6tSi8")
  st.write('____')


