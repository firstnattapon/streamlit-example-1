import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="Principle_F(X)", page_icon="📕" , layout="wide")

# @st.cache_data
def iframe ( frame = ''):
  src = frame
  st.components.v1.iframe(src, width=1500 , height=800  , scrolling=0)

Main,  Control , Mind  ,   tab2, tab3 , tab4  , tab5 = st.tabs(["Main", "Control" , "Mind"  ,"tab2 (Value vs Time)", "tab3 (Fix_Asset vs Ratio_Asset)" , "tab4 (Principle)"  , "tab5 (Land Price Calculator)" ])

with Main:
  iframe(frame = "https://monica.im/share/artifact?id=nrFWXCsa3Bwf3EvM74DwU7")    
  st.link_button("(Price_Cycle) พื้นฐาน Global_macro ", "https://drive.google.com/file/d/1-bNM1gPEG7i-CW1TMd_6Cu6Z5132UjGZ/view?usp=sharing")
 
with tab2:
  st.image("https://img.soccersuck.com/images/2025/01/21/455599160_312374775229941_5381968498530104178_n.jpg", width=1000)

with tab3:
  st.image("https://img.soccersuck.com/images/2025/01/21/275206549_1046792219234327_607135450348777933_n.jpg", width=1000)
  st.link_button("Rebalance_Fix_Asset-Ratio_Clip-1", "https://drive.google.com/file/d/1x_zlS7y9tTxWxHqD_yWpqrvFkGH2J2pB/view?usp=sharing")
  st.link_button("Rebalance_Fix_Asset-Ratio_Clip-2", "https://drive.google.com/file/d/1DiySbVSV5MeTYktk03FpOEe3tmpZYGyl/view?usp=sharing")

with tab4:
  st.image("https://img.soccersuck.com/images/2025/01/21/407896427_1617027032037031_1189622303379814904_n.jpg", width=1000)

st.write('____')

with Control:
  st.link_button("1 .หลักการและคำตอบว่า = ทำไม FIX = LN(T0/TN) ใช้ได้", "https://g.co/gemini/share/b58a41a06475")
  iframe(frame = "https://monica.im/share/artifact?id=McLUV4GWRWenzAaWWx44qk")

with Mind:
  with st.expander("คลิป"):
    st.video('https://www.youtube.com/watch?v=fhgc8FBPnA8')
  with st.expander("Note"):
    st.components.v1.iframe('https://monica.im/share/artifact?id=dKModf53bUtZswz2SbrNoJ', width=1500 , height=800  , scrolling=0)

with tab5:
  iframe(frame = "https://monica.im/share/artifact?id=jaZw59TvuQ9kqBS3534m64") 

checkbox1 = st.checkbox(' mindmeister' , value=0 )
if checkbox1 :
  st.write('https://www.mindmeister.com/app/map/3178532454?m=outline')



