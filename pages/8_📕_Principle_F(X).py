import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="Principle_F(X)", page_icon="📕" , layout="wide")

@st.cache_data
def iframe ( frame = ''):
  src = frame
  st.components.v1.iframe(src, width=1500 , height=800  , scrolling=0)

tab1,  Control , Mind  ,   tab2, tab3 , tab4  = st.tabs(["tab1", "Control" , "Mind"  ,"tab2", "tab3" , "tab4"  ])

with tab1:
  iframe(frame = "https://monica.im/share/artifact?id=ybc5eexwsxVUDh4FbKSwjA")    
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
  iframe(frame = "https://monica.im/share/artifact?id=DF3TydyEV664kgsgbbySs3")

with Mind:
  Mind_col1, Mind_col2  = st.columns(2)
  Mind_col1.video('https://www.youtube.com/watch?v=fhgc8FBPnA8')
  Mind_col2.components.v1.iframe("https://monica.im/share/artifact?id=wBtgUhDcA94wLY7mD9PmRn")

checkbox1 = st.checkbox(' mindmeister' , value=0 )
if checkbox1 :
  st.write('https://www.mindmeister.com/app/map/3178532454?m=outline')



