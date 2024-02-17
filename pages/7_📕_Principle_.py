import streamlit as st
import streamlit.components.v1 as components
st.set_page_config(page_title="Principle", page_icon="📕")
st.set_page_config(layout="wide")

@st.cache_data
def iframe ():
  src="https://www.mindmeister.com/app/map/3066443605?fullscreen=1&v=embedded&m=outline" 
  st.components.v1.iframe(src, width=1500 , height=2000, scrolling=1)
  
iframe()
