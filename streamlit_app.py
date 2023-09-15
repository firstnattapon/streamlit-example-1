import streamlit as st
import numpy as np

st.write("สมการ" ,"  =  -742+1500"  , "fix 1500 : Initial Port" , "เริ่ม 6.88") 
st.write("")
x = st.number_input('ราคา')
y = -742+1500 * np.log(x)
st.write("Price", x ,"=" , y) 
