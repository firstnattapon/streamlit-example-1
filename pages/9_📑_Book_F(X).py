import telebot
import streamlit as st
import time
import thingspeak
import json
import yfinance as yf
import numpy as np
import pandas as pd

st.set_page_config(page_title="Book_F(X)", page_icon="📑" , layout="wide")

st.markdown('''
# สรุปบทที่ 1: การคิดระดับสอง
- Second-Level Thinking เป็นกุญแจสำคัญในการสร้างผลตอบแทนที่เหนือกว่าตลาด 
- โดยต้องคิดลึกกว่า มองต่างจาก และอดทนมากกว่านักลงทุนทั่วไป.
''')

st.markdown('''
# สรุปบทที่ 2: ประสิทธิภาพของตลาดและข้อจำกัด
- ตลาดมีประสิทธิภาพส่วนใหญ่ แต่ไม่สมบูรณ์แบบ 
- ความไม่สมบูรณ์แบบไร้ประสิทธิภาพเกิดจาก อารมณ์ของ มนุษย์ 
''')


@st.cache_data
def iframe ( frame = ''):
  src = frame
  st.components.v1.iframe(src, width=1500 , height=800  , scrolling=0)

tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10,tab11,tab12,tab13,tab14,tab15,tab16,tab17,tab18,tab19,tab20  = st.tabs(["บทที่_1", "บทที่_2" ,"บทที่_3", "บทที่_4" , "บทที่_5" ,"บทที่_6" , "บทที่_7" , "บทที่_8" , "บทที่_9"  ,
                                                                                       "บทที่_10"  , "บทที่_11", "บทที่_12", "บทที่_13" , "บทที่_14","บทที่_15","บทที่_16","บทที่_17","บทที่_18","บทที่_19" ,"บทที่_20" ])


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

with tab6:
  iframe(frame = "https://monica.im/share/chat?shareId=0QTT4X0ajaBB0VsB")
  st.write('____')
  
with tab7:
  iframe(frame = "https://monica.im/share/chat?shareId=Ai6I7TDCrdj55BkE")
  st.write('____')

with tab8:
  iframe(frame = "https://monica.im/share/chat?shareId=GzqDAt1PG5bHNcdz")
  st.write('____')

with tab9:
  iframe(frame = "https://monica.im/share/chat?shareId=L9w0lOu7dHuQgzuP")
  st.write('____')

with tab10:
  iframe(frame = "https://monica.im/share/chat?shareId=ejez2QpkAaw1n5Rn")
  st.write('____')

with tab11:
  iframe(frame = "https://monica.im/share/chat?shareId=ijSatOtgyJ06lElv")
  st.write('____')

with tab12:
  iframe(frame = "https://monica.im/share/chat?shareId=PRzdffXSMscrThUV")
  st.write('____')

with tab13:
  iframe(frame = "https://monica.im/share/chat?shareId=dUNOZrSFx6exT6bF")
  st.write('____')
  
with tab14:
  iframe(frame = "https://monica.im/share/chat?shareId=TVnfn8gPjit32jXi")
  st.write('____')

with tab15:
  iframe(frame = "https://monica.im/share/chat?shareId=ePDlpaLktq3OaJld")
  st.write('____')
  
with tab16:
  iframe(frame = "https://monica.im/share/chat?shareId=AlqLIalwMYWKxYgz")
  st.write('____')

with tab17:
  iframe(frame = "https://monica.im/share/chat?shareId=3va9N2nS9eifbmzi")
  st.write('____')

with tab18:
  iframe(frame = "https://monica.im/share/chat?shareId=3d9kQUZ98S1u6OxT")
  st.write('____')

with tab19:
  iframe(frame = "https://monica.im/share/chat?shareId=FEynUPq6rANTTjPU")
  st.write('____') 

with tab20:
  iframe(frame = "https://monica.im/share/chat?shareId=oznm4bVmlX7D61Rn")
  st.write('____')

