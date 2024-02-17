import streamlit as st
import numpy as np
import datetime
import thingspeak
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="0", page_icon="🔥")


map_url = "https://www.mindmeister.com/app/map/3066443605?t=dxos6u4HQQ&m=outline"
st.iframe(map_url)
