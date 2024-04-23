import streamlit as st
import numpy as np
import datetime
import thingspeak
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="Calculator", page_icon="🔥")

x_1 = st.number_input('ราคา_NEGG_1.26' , step=0.01 , value =  yf.Ticker('NEGG').fast_info['lastPrice']   )
x_2 = st.number_input('ราคา_FFWM_6.88', step=0.01 ,   value = yf.Ticker('FFWM').fast_info['lastPrice']   )
st.write("_____") 
x_3 = st.number_input('NEGG_ASSET', step=0.01 ,   )
x_4 = st.number_input('FFWM_ASSET', step=0.01 ,   )

st.write(yf.Ticker('FFWM').fast_info['lastPrice']) 

