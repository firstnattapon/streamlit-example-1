import streamlit as st
import numpy as np

x = st.number_input('ราคา')
y = -742+1500 * np.log(x)
st.write("สมการ = -742+1500") 
st.write( y) 


