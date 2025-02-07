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
###### สรุปบทที่ 1: การคิดระดับสอง
- Second-Level Thinking (second opinion)? คิดซ้อน อย่าคิดแบบผิวเผินตรงไปตรงมา
''')

st.markdown('''
###### สรุปบทที่ 2: ประสิทธิภาพของตลาดและข้อจำกัด
- ตลาดมีประสิทธิภาพส่วนใหญ่แต่ไม่สมบูรณ์แบบเกิดจาก อารมณ์ของมนุษย์ เช่นความโลภ(Greed)และ ความกลัว (Fear)
''')

st.markdown('''
###### สรุปบทที่ 3: สรุปบทที่ 3: คุณค่า" (Value)
-  Value คือรากฐาน(หลักยืด)ของการลงทุน "การเข้าใจมูลค่าที่แท้จริง และซื้อในราคาที่ต่ำกว่านั้น"
''')

st.markdown('''
###### สรุปบทที่ 4 Value vs Price Relationships
- "การลงทุนที่ดีคือการเข้าใจความแตกต่างระหว่าง ราคา(Price) และ มูลค่า(Value)"
''')

st.markdown('''
###### สรุปบทที่ 5 Understanding Risk (เข้าใจความเสี่ยง)
- นิยามใหม่ของความเสี่ยงคือ "ความไม่แน่นอนของผลตอบแทน" & "โอกาสที่ผลลัพธ์จะไม่เป็นไปตามคาด"
''')

st.markdown('''
###### สรุปบทที่ 6 Recognizing Risk (ยอมรับความเสี่ยง)
- ความเสี่ยงไม่ได้มาจากตัวสินทรัพย์ แต่มาจากราคาที่คุณจ่ายไป และจิตวิทยาของคุณต่อมัน
- อนาคตไม่สามารถคาดเดาได้ และ ความไม่แน่นอนในตลาดเป็นสิ่งที่หลีกเลี่ยงไม่ได้ 
- ความเสี่ยงที่แท้จริงอันตรายสุด คือสิ่งที่มองไม่เห็นวัดไม่ได้ อย่าความมั่นใจเกินไป (Overconfidence)
''')

st.markdown('''
###### สรุปบทที่ 7 Control Risk (การควบคุมความเสี่ยง)
- ความสำคัญของการควบคุมความเสี่ยงเหนือการแสวงหากำไรสูงสุด  การควบคุมความเสี่ยงเป็นรากฐาน(หลักยืด)ของการลงทุน
- การวางแผนรับมือจึงสำคัญกว่าการพยายามทำนาย
- จัดการความเสี่ยงอย่างเป็นระบบ ไม่ใช่การพยายามกำจัดความเสี่ยงทั้งหมด โดนเน้นด้าน คุณภาพ,จิตวิทยา,และกลยุทธ์
''')

st.markdown('''
###### สรุปบทที่ 8 Price Cycle (การรับรู้ถึงวัฏจักร)
- Price Cycle คือ กฎธรรมชาติ
- โฟกัสที่การประเมินว่า "ตอนนี้เราอยู่ในช่วงไหนของวัฏจักร" สำคัญกว่าคาดเดาอนาคต
- วัฏจักรไม่ใช่ศัตรูของคุณ แต่ความไม่รู้เกี่ยวกับวัฏจักรต่างหากที่เป็นศัตรู
''')

st.markdown('''
###### สรุปบทที่ 9 Cycle is Pendulum (วัฏจักรเคลื่อนไหวแบบลูกตุ้ม)
- ไม่มีจุดสมดุลถาวร จุดกึ่งกลางของลูกตุ้ม เป็นเพียงสิ่งชั่วคราว
- เมื่อเห็น Cycle  ใช้สติการจัดการอารมณ์ด้วยความเป็นกลาง  เพื่อหลีกเลี่ยงการตัดสินใจผิดพลาดจากพฤติกรรมฝูงชน 
''')

st.markdown('''
###### สรุปบทที่ 10 Dark Side (การต่อสู้กับอิทธิพลด้านมืดในตัว)
- อย่าตามกลุ่ม ควบคุมจิตใจให้อยู่เหนือวงจรตลาด หลีกเลี่ยงการถูกควบคุมโดยอารมณ์"โลภ"และ"กลัว" อารมณ์ทั้งสองนี้เป็นศัตรูสำคัญของนักลงทุน
- แยกแยะระหว่าง "ความเสี่ยง" และ "ความรู้สึกไม่สบายใจ"
- สร้างกระบวนการตัดสินใจที่เป็นระบบ เพื่อลดอิทธิพลของอคติส่วนตัว
''')

st.markdown('''
###### สรุปบทที่ 11 Contrarianism (นักแทงสวนอย่างไตร่ตรอ)
- Contrarian ≠ การค้านกระแสทุกเรื่อง แต่คือ นักแทงสวนอย่างไตร่ตรอและมีหลักการรองรับ
- การ Contrarian แบบมีหลักการ จะผิดไม่สบายใจ..ชั่วคราว แต่ระยะยาวแล้วจะเป็นฝ่ายถูก
''')

st.write(' ')

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

