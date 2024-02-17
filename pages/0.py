import streamlit as st
import numpy as np
import datetime
import thingspeak
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="0", page_icon="🔥")

url = "http://www.soccersuck.com/boards#5"  # Replace with your website URL

st.write(f'<iframe src="{url}" style="width:100%; height:500px;"></iframe>', unsafe_allow_html=True)
