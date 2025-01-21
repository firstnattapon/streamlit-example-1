import streamlit as st
import streamlit.components.v1 as components
import streamlit_mermaid as stmd


st.set_page_config(page_title="Principle_F(X)", page_icon="📕" , layout="wide")

# @st.cache_data
# def iframe ():
#   # src="https://www.mindmeister.com/app/map/3178532454?fullscreen=1&v=embedded&m=outline" 
#   # st.components.v1.iframe(src, width=1500 , height=2000, scrolling=1)

#   src="https://img.soccersuck.com/images/2025/01/21/brave_screenshot_img.soccersuck.com.png" 
#   st.components.v1.iframe(src, width=1500 , height=800  , scrolling=1)

# iframe()


st.markdown("""
```mermaid
graph TD;
    A-->B;
    A-->C;
    B-->D;
    C-->D;

""", unsafe_allow_html=True)



st.write('____')

checkbox1 = st.checkbox(' mindmeister' , value=0 )
if checkbox1 :
  st.write('https://www.mindmeister.com/app/map/3178532454?m=outline')
