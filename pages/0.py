import streamlit as st
import streamlit.components.v1 as components


src="https://www.mindmeister.com/maps/public_map_shell/3066443605/principle-by-first?width=600&height=1000&z=auto&no_share=1&no_logo=1" 


st.components.v1.iframe(src, width=600, height=1000, scrolling=False)

