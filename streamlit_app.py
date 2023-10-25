import streamlit as st
import numpy as np
import datetime

st.write("สมการ ","  =  -742+1500ln x "  , "fix 1500 : Initial Port" , "เริ่ม 6.88") 
st.write("")
x = st.number_input('ราคา')
z = st.number_input('cash')
q = st.number_input('asset')
p = z+q
y = -742+1500 * np.log(x)

st.write("Price", x ,"=" , y) 
st.write( 'cf' ,"=", p - y) 

Check_NEW = st.checkbox('NEW_DATA ')
if Check_NEW :
    button_NEW = st.button("NEW_DATA")
    if button_NEW:
        try:
            np.save('my_array.npy', np.array([]))
            my_array_1 = np.load('my_array.npy')
            st.write(my_array_1) 
        except:pass

Check_LOAD = st.checkbox('LOAD_DATA ')
if Check_LOAD :
    button_LOAD = st.button("LOAD_DATA")
    if button_LOAD:
        try:
            my_array_2 = np.load('my_array.npy')
            st.write(my_array_2) 
        except:pass
        
Check_ADD = st.checkbox('ADD_CF ')
if Check_ADD :
    button_ADD = st.button("ADD_CF")
    if button_ADD:    
        try:
            my_array_a = np.load('my_array.npy')
            my_array_b = np.append(my_array_a, '({}):({:.2f}):({:.2f})'.format( datetime.datetime.now() , p - y , (p - y)/2150 ))
            np.save('my_array.npy', my_array_b)
            st.write( p - y) 
        except:pass
        
Check_DEL = st.checkbox('DEL_CF ')
if Check_DEL :
    button_DEL = st.button("DEL_CF")
    if button_DEL:
        try:
            my_array_c = np.load('my_array.npy')
            my_array_d = np.delete(my_array_c, -1)
            np.save('my_array.npy', my_array_d)
            my_array_3 = np.load('my_array.npy')
            st.write(my_array_3)     
        except:pass   
