import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="Principle_F(X)", page_icon="📕" , layout="wide")

@st.cache_data
def iframe ():
  src = "https://kroki.io/mermaid/svg/eNq1VV1LFFEYvu9XHAYKNRxS886ENCtJS9qoi0GG43hmPTg7s82cXdm7LGH7uEwRCyITEYQu-j7zb85P6X3fc2Z2USEKWpY97_l6nuf9Ohsn2Va0wXPFHt26xOBzO-ctsZXlm4FXm94qGx-fZVGWqjxLAm_Cn7cmbFxha0LxwJv052C0C504FnngTflzZNFiIroi500ReNf9JWfTRkum64VQgTftm_KF0S-NPjEl_H41um_KHaMPafrJ6J9G7xq9Z_R3MLxV0ovspI4XhDLh30QDgFfybL0TKSLp8qQjUCSi6dLoY6N3KpZTMk6MPjDlM6O3Cf_A6H2jNRiOiPCJSaZxwpXM0pDWCiRdPLNGrEXE80jUpyb9xvCCwyVphAtpWA9rN-7Ws5m1fHYEZeojU75ygQH5qO5t7ceopcxiVWEAXz2zGINo_hmMxA0kkULxtCOVFOTxgrOJNhc8CUWhuKIoP8TpAk0dUHXTBVDlMi1kFHibosduMPDujdG_SNBnUrlL9h6p-UY5cUXBFqvL4WOMXKV0SAFxDM1DIKmYplbc-UGg6HiahVHWagvUb0qge050fTT0FyqNKnaVvPPESERo7YRHALSCAzq3j05gbR2PYiKeQE-xq2xJxgKGhuK9kYW0K_MsbYlU2US2bfUCiDUQps7SR8rbBwI7v4wxxO82OfIOvoBHvfWaotqnYweORkJBAlkbmaSVe0QHdrAOLMNggc2wMd5tjtSV0hxlk9fY5TEXjTq1Ng6I6IDD-V6UuJbneVOmgbdMI8ti1uCxUD2HQbdsR8jmRuh691zN_nUjDxXWKd3tO0Kr5wyjlUIZvGjDZeXCrSqkF-0OCo124fn9Lw66znEO1izE2cygYmEJX-gHHQVyqrREGzIV2Nx34AitZXYfm3rZblcvQ-1WjRl4eA2azOj39o2mmjkh6YekGDz8Ub3mg-gTLsHEsivCQon2P2I5ubalxZbt_OnxBiDi_0EkiurZrYJgj04F3n04fk9ADf4G1n08MQ"  
  st.components.v1.iframe(src, width=1500 , height=800  , scrolling=0)
iframe()

st.link_button("Principle", "https://claude.site/artifacts/f8e8bc35-1eed-4f5f-9545-2a8faf9a86dd")
st.write('____')

checkbox1 = st.checkbox(' mindmeister' , value=0 )
if checkbox1 :
  st.write('https://www.mindmeister.com/app/map/3178532454?m=outline')
