import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="Principle_F(X)", page_icon="📕" , layout="wide")

@st.cache_data
def iframe ():
  # src="https://www.mindmeister.com/app/map/3178532454?fullscreen=1&v=embedded&m=outline" 
  # st.components.v1.iframe(src, width=1500 , height=2000, scrolling=1)
  src = "https://img.soccersuck.com/images/2025/01/21/mermaid-diagram-2025-01-12-143129.svgb333dee1071b110d.png"
  st.components.v1.iframe(src, width=1500 , height=800  , scrolling=0)
  
# iframe()
st.link_button("Principle", "https://www.mermaidchart.com/raw/e3003041-c706-467f-b732-e3b1754530d2?theme=light&version=v0.1&format=svg")
st.write('____')

checkbox1 = st.checkbox(' mindmeister' , value=0 )
if checkbox1 :
  st.write('https://www.mindmeister.com/app/map/3178532454?m=outline')
