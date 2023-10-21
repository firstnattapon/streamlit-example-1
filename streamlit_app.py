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

button_NEW = st.button("NEW_DATA")
if button_NEW:
    np.save('my_array.npy', np.array([]))
    st.write( "NEW_DATA")

button_LOAD = st.button("LOAD_DATA")
if button_LOAD:
    my_array = np.load('my_array.npy')
    st.write( my_array) 

button_ADD = st.button("ADD_CF")
if button_ADD:
    my_array_a = np.load('my_array.npy')
    my_array_b = np.append(my_array_a, p - y)
    np.save('my_array.npy', my_array_b)
    st.write( p - y)
    
button_DEL = st.button("DEL_CF")
if button_DEL:
    my_array_a = np.load('my_array.npy')
    my_array_b = np.delete(my_array_a, -1)
    np.save('my_array.npy', my_array_b)
    st.write( button_DEL)
