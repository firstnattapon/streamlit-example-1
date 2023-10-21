import streamlit as st
import numpy as np

st.write("สมการ ","  =  -742+1500ln x "  , "fix 1500 : Initial Port" , "เริ่ม 6.88") 
st.write("")
x = st.number_input('ราคา')
z = st.number_input('cash')
q = st.number_input('asset')
p = z+q
y = -742+1500 * np.log(x)

st.write("Price", x ,"=" , y) 
st.write( 'cf' ,"=", p - y) 


if st.button('ADD CF'):
  np.save("my_array.npy", p - y)

