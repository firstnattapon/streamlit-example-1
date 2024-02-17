import streamlit as st
import numpy as np
import datetime
import thingspeak
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="0", page_icon="🔥")

url = "https://www.mindmeister.com/app/map/3066443605"  # Replace with your website URL

st.write(f'<iframe src="{url}" style="width:100%; height:500px;"></iframe>', unsafe_allow_html=True)
