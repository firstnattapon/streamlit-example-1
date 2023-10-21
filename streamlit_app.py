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


def save_numpy_array(array, filename):
    np.save(filename, array)

st.title("Save Numpy Array to File")

array = np.random.randint(0, 100, size=(100,))

button = st.button("Save Array")

if button:
    filename = "my_array.npy"
    save_numpy_array(array, filename)

    st.write("Array saved to file: {}".format(filename))


my_array = np.load('my_array.npy')

st.write( my_array) 
