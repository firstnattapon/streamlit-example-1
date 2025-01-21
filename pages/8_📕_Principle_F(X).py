import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="Principle_F(X)", page_icon="📕" , layout="wide")

@st.cache_data
def iframe ():
  # src="https://www.mindmeister.com/app/map/3178532454?fullscreen=1&v=embedded&m=outline" 
  # st.components.v1.iframe(src, width=1500 , height=2000, scrolling=1)
  src = "https://mermaid.live/view#pako:eNq1VltrE0EU_ivLgtJWG22rL0ULVusFWy1GFExCmG5mk6G7O3F2NiGI4KVQL48qpQpiLVIQfPA--TfzUzxzZi_NuqA-GMLOmZ0z33euM3vP9XiHuouuH_Ch1yNCOjcvNCMHfhcFCemQi01ndnbJ8XgkBQ8ac7XzVmpVaW1QSRrztWUYqtcT36eisVBbRqFSJ6ADKkiXNk7VVlOxUi9kUSemsnG6psePtXqi1YEew_OLVtt6vKXVHk7fa_VDq5davdLqGwgpln0aexGMxAZqrnbOjM5RZ13wTuLJVkltQIKEgoMGXI21-qDVVkb6EYUDrXb1-IFWD5FuV6sdrRQIE7xIh4gs8gMiGY_a-C4GG66UXrXKe2KPCI9mG-Zr9cPzCR60F_dAajtWo9F052qX8-mZDXFiacqYr_b1-GkaP3DLWP0i92-66bbKmDH3ZY4JZuTTFLNIwl-D22dhLPLQuwmTjJrYrKRiq1JRUBK0aSyJNEm6YWYrOJtAz-DS-EvBoph54MImHTlnHYjFc61-ormf0IeXKL9CW79iZtNKc65ku9u3TFQKPw5ZUrasDTQ52cJ6vqWIJu6IeNvjYZ-CK01Xj4H0EZJuG0F9xjLL4psZWQpjiRRh-wHxDOK6GY2vO8YnU7AfpjFtt01_HXNWmU9hqEsymlqJBkzwKKSRrPSwALddY-CtZAjyZL_D9L-1NL-_N9E2_4fo7Gv4Qx9iaz_D-G-j2u4fDGDQCGBGH21g1sV93LllCi3lLt44Z5wZMuhO5bXYnXbmTzpHZkqxzOuk4GkgQfv8yAuyApvUConosqixhoPDfadOfCpHE7iIZDuUdXtte8BAwsst88_nzaHK_Yh7t3OPrF0l0tQcLIvKlTSj1WtZ2CuX7bMoaHvM8-H_8Tbt01L-cjok73KoHXgFJ8olEFtVOiHxeiwyB8malSq1eCLBKwpX2nUrTbBmPDkpOGsIofO1emMvJKzDA3RyD32DWHzPrq4iaUVQc7DDTKm1uOazAW3HkpomOD1bB8HcZx6N4xwtNdueNHSIZ9I1GK_SUcs97oZUhIR14LvgntFvurJHQ9p0F0HsUJ8kgWy6zeg-qJJE8voo8txFKRJ63BU86fbcRZ8EMcySfgf68wIjXbi7M5U-ie5wHuZKtMMkF2v2QwS_R-7_AuIRs88"
  st.components.v1.iframe(src, width=1500 , height=800  , scrolling=0)
  
iframe()
st.write('____')

checkbox1 = st.checkbox(' mindmeister' , value=0 )
if checkbox1 :
  st.write('https://www.mindmeister.com/app/map/3178532454?m=outline')
