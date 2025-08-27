import telebot
import streamlit as st
import time
import thingspeak
import json
import yfinance as yf
import numpy as np
import pandas as pd

st.set_page_config(page_title="Book_F(X)", page_icon="📑" , layout="centered")


with st.expander("Book : The Most Important Thing : นักลงทุนเหนือชั้น & Mastering the Market Cycle : เหนือกว่าวัฏจักรการลงทุน " , expanded= 0  ):
  
  st.video('https://www.youtube.com/watch?v=APegVkFI39w')  
  st.video('https://www.youtube.com/watch?v=KWG5kKgBHxE') 
  st.video('https://www.youtube.com/watch?v=9neT0cTCbgY')

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
  
  st.markdown('''
  ###### สรุปบทที่ 12 หา DISCOUNT (หาส่วนลด)
  - "ของถูก ≠ ของดีเสมอไป" ต้องแยกความแตกต่างระหว่าง "ราคาถูก" และ "มูลค่าที่แท้จริง"
  - หลีกเลี่ยงกับดัก Value Traps สินทรัพย์ราคาถูกที่ไม่มีโอกาสฟื้นตัว เช่น หุ้นที่ราคาต่ำเรื้อรังเพราะปัญหาโครงสร้างที่แก้ไขไม่ได้
  - ไม่ใช่แค่การหาของถูก แต่ต้องเข้าใจปัจจัยที่ทำให้ราคาถูก
  ''')
  
  st.markdown('''
  ###### สรุปบทที่ 13 Patient Opportunism (รอคอยโอกาสอย่างอดทน)
  - การรอคอยอย่างชาญฉลาดเป็นคุณสมบัติสำคัญ ต้องพร้อมเเมื่อเวลานั้นมาถึง เช่นราคาสินทรัพย์ต่ำกว่ามูลค่าที่แท้จริง (Margin of Safety)
  - รอโอกาสที่ "เหมาะสม" แทนการต้องลงทุนตลอดเวลา หลีกเลี่ยงการตัดสินใจจากแรงกดดัน เช่นความโลภ(Greed)และ ความกลัว (Fear) 
  ''')
  
  st.markdown('''
  ###### สรุปบทที่ 14 การยอมรับข้อจำกัดของ "ความไม่รู้
  - การพยายามคาดการณ์อนาคตแบบแน่นอนเป็นเรื่องอันตราย ควรโฟกัสที่ "การจัดการความเสี่ยงมากกว่าการไล่หากำไร"   
  - การยอมรับ "ความไม่รู้" ช่วยสร้างภูมิคุ้มกันทางจิตใจ และออกแบบกลยุทธ์ที่ผลลัพธ์ที่อาจเกิดขึ้นหลายแบบแทนยึดติดกับสถานการณ์เดียว
  - การยอมรับว่า "เราอาจผิดเสมอ" เป็นหัวใจสำคัญของการลงทุนที่ยั่งยืนการจัดการความเสี่ยง
  ''')
  
  st.markdown('''
  ###### สรุปบทที่ 15 ตอนนี้เราอยู่จุดไหนของวัฏจักร
  - การรู้ตัวว่ายืนอยู่จุดไหนของวัฏจักร สำคัญกว่าการพยายามทำนายอนาคต
  - การควบคุมจิตใจให้เป็นกลางอยู่เหนืออารมณ์  มองตลาดอย่างมีแบบมีหลักการรองรับ
  ''')
  
  
  st.markdown('''
  ###### สรุปบทที่ 16 บทบาทของ "โชค
  - อย่าวัดทักษะจากผลลัพธ์ระยะสั้น เพราะโชคอาจบิดเบือนความจริง อย่าให้โชคดีหลอกให้คุณหลงตัวเอง
  - เตรียมพร้อมสำหรับโชคร้าย แม้คุณจะคิดว่าทำทุกอย่างถูกต้องตามหลัการแล้ว อย่าให้โชคร้ายทำลายความเชื่อมั่นในกระบวนการที่ถูกต้อง
  - โชคมีบทบาทลดลงในระยะยาว เวลาคือเกณฑ์วัดทักษะที่แท้จริง เพราะโชคจะถูกหักล้างด้วยเวลา
  - เน้นควบคุมความเสี่ยงแทนการไล่ตามโชค ไม่เดาอนาคต แต่กระจายความเสี่ยงเตรียมพร้อมสำหรับทุกสถานการณ์
  ''')
  
  st.markdown('''
  ###### สรุปบทที่ 17 เน้นเป็นนักลงทุนเชิงรับ (Risk Control)
  - การอยู่รอดสำคัญกว่าชัยชนะครั้งใหญ่ เน้นการหลีกเลี่ยงการสูญเสียมากกว่าการไล่ตามผลตอบแทนสูงสุด
  - ซื้อสินทรัพย์ในราคาที่ต่ำกว่ามูลค่าที่แท้จริง (Intrinsic Value) เพื่อสร้าง "เกราะป้องกัน" หากเกิดข้อผิดพลาด
  - การยอมรับความไม่แน่นอน ตลาดการเงินไม่สามารถคาดเดาได้ ใช้การกระจายความเสี่ยงและจำกัดขนาดการลงทุนในแต่ละสินทรัพย์
  - หลีกเลี่ยงกับดักทางจิตวิทยาตามกระแส การควบคุมจิตใจให้เป็นกลางอยู่เหนืออารมณ์ตลาด เป็นนักแทงสวนอย่างไตร่ตรอ
  ''')
  
  
  st.markdown('''
  ###### สรุปบทที่ 18 หลีกเลี่ยงหลุมพราง
  - หลุมพรางทางจิตวิทยา ศัตรูที่มองไม่เห็น 
  - จัดการความเสี่ยงก่อนคิดถึงกำไร สิ่งสำคัญไม่ใช่การคาดการณ์อนาคต แต่คือการเตรียมพร้อมสำหรับทุกสถานการณ์
  - เป้าหมายสูงสุดคือ การอยู่รอด (Survival) เพราะหากสูญเสียเงินทุน คุณจะไม่มีโอกาสกลับมาเล่นเกมนี้อีก
  ''')
  
  
  st.markdown('''
  ###### สรุปบทที่ 19 Adding Value (ทักษะการเพิ่มมูลค่า)
  - สร้างผลตอบแทนเหนือตลาด "ความได้เปรียบเชิงการแข่งขัน" ต้องชนะ Benchmark เฉลี่ยเช่น sp500
  - การคิดแบบ Contrarian มองหาความผิดพลาดของตลาด  เช่น ราคาสินทรัพย์สูงหรือต่ำเกินจริง
  - การสร้างมูลค่าไม่ใช่แค่การไล่ตามผลตอบแทนสูงสุด แต่ต้องควบคุมความเสี่ยงให้ดีกว่าคนอื่น 
  - การสร้างมูลค่าเพิ่มต้องอาศัยความอดทนและยึดมั่นในกลยุทธ์ แม้ตลาดจะผันผวน
  ''')
  
  st.markdown('''
  ###### สรุปจุดสำคัญจากบทที่ 20: "การประสานทุกสิ่งเข้าด้วยกัน" ของหนังสือ The Most Important Thing โดย Howard Marks
  - 1.การลงทุนบนพื้นฐานสำคัญคือ "คุณค่า" (Value Investing)
  - 2.การจัดการความเสี่ยงต้องมาก่อนการแสวงหาผลตอบแทน
  - 3.เข้าใจธรรมชาติของวัฏจักรตลาด (Market Cycles)
  - 4.การคิดเชิงลึก  (Second-Level Thinking)
  - 5.ผสมผสานความรู้ด้าน มูลค่า ความเสี่ยง และจิตวิทยาตลาด เข้าด้วยกันอย่างเป็นระบบสมดุลและยืดหยุ่น
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

with st.expander("Book : Principles : หลักการ (ทุกอย่างมีระบบของมัน)" , expanded = 0 ):
  st.video('https://www.youtube.com/watch?v=D6LRTghic7c') 
  st.write('____') 

with st.expander("Book : margin of safety : ส่วนเผื่อเพื่อความปลอดภัย " , expanded = 0 ):
  st.video('https://www.youtube.com/watch?v=sQgn7xb_TOo') 
  st.write('____') 

with st.expander("Book : The Intelligent Investor  : ลงทุนแบบเน้นคุณค่า " , expanded = 0 ):
  st.video('https://www.youtube.com/watch?v=PL2Ji-Wb7bc') 
  st.write('____') 

with st.expander("Book : The Alchemy of Finance  : จอร์จ โซรอส " , expanded = 0 ):
  st.video('https://www.youtube.com/watch?v=wek-SSkDxGg') 
  st.write('____') 

with st.expander("Book :Capital in the Twenty-First Century : ทุนนิยมในศตวรรษที่ 21" , expanded = 0 ):
  st.video('https://www.youtube.com/watch?v=4W7_Ei_EzQs') 
  st.video('https://www.youtube.com/watch?v=Fe42UWyqk-Y') 
  st.video('https://www.youtube.com/watch?v=DaDB-hLgakE') 
  st.write('____') 

with st.expander("Book : 1 The Bitcoin Standard & 2 The Fiat Standard : ประวัติศาสตร์เงิน" , expanded = 0 ):
  st.video('https://www.youtube.com/watch?v=_0dvh8D5OPg') 
  st.video('https://www.youtube.com/watch?v=pU1HSLhwhig') 
  st.video('https://www.youtube.com/watch?v=Mn4UACzMVa0') 
  st.write('____') 

with st.expander("Book :  Start with Why(How, What) : เริ่มต้นด้วยทำไม & Start with the End in Mind (Stephen Covey) : mental blueprint  " , expanded = 0 ):
  st.video('https://www.youtube.com/watch?v=Y-qJiiF0wPM')
  st.write('____')  

with st.expander("Book : the power of now : พลังแห่งจิตปัจจุบัน " , expanded = 0 ):
  st.video('https://www.youtube.com/watch?v=WfGfBFWPQKA') 
  st.video('https://www.youtube.com/watch?v=fhgc8FBPnA8') 
  st.write('____')  

with st.expander("Book : 5 key messages " , expanded = 0 ):
  st.components.v1.iframe("https://monica.im/share/chat?shareId=ZZYqC15w7HFYvGhW", width=1100 , height=1000  , scrolling=0)
  st.write('____')  

st.write('notebooklm')

with st.expander("Book : Mudley Live by Jatuphon vol.1 " , expanded = 0 ):
  st.write('https://notebooklm.google.com/notebook/7266636a-4bc4-45ae-816e-78521222f9e0')
  st.write('____')  

with st.expander("Book : Mudley Live by Jatuphon vol.2 " , expanded = 0 ):
  st.write('https://notebooklm.google.com/notebook/b0777e9a-2faf-4be0-9f66-2974633e7d1c')
  st.write('____')  

with st.expander("Book : Mudley Live by Jatuphon vol.3 (TFEX  PROFESSIONAL TRADER) " , expanded = 0 ):
  st.write('https://notebooklm.google.com/notebook/c1970908-007f-4a5c-a33e-774ab91acb1b')
  st.write('____')  

with st.expander("Book : The Most Important Thing : นักลงทุนเหนือชั้น & Mastering the Market Cycle : เหนือกว่าวัฏจักรการลงทุน " , expanded = 0 ):
  st.write('https://notebooklm.google.com/notebook/b59fa253-0e58-421e-b085-c8092aebd56c')
  st.write('____')  

with st.expander("Book : Capital in the Twenty-First Century : ทุนนิยมในศตวรรษที่ 21 " , expanded = 0 ):
  st.write('https://notebooklm.google.com/notebook/dbe6015c-5aa1-4c71-a8bd-1b3ac898f5ef')
  st.write('____')  

with st.expander("Book : การลงทุน 50 ปีของ Howard Marks" , expanded = 0 ):
  st.write('https://notebooklm.google.com/notebook/7e3b54f2-add3-40bb-9441-19c70a53654b')
  st.write('____')  

st.write('Podcast')
with st.expander("Podcast_notebook_lm " , expanded = 0 ):
  st.write('https://monica.im/share/artifact?id=QUH6h83XGQHJ4ot3gjEiXk')
  st.write('https://github.com/firstnattapon/streamlit-example-1/releases/edit/notebook_lm')
  st.write('____')  
