import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="Principle_F(X)", page_icon="📕" , layout="wide")

@st.cache_data
def iframe ():
  src = "https://monica.im/share/artifact?id=EoDyXJCkbU5U4G3dFvsAdm"
  st.components.v1.iframe(src, width=1500 , height=800  , scrolling=0)
iframe()

tab1, tab2, tab3 , tab4  = st.tabs(["tab1", "tab2", "tab3" , "tab4"])

with tab1:
  multi = "[![](https://mermaid.ink/img/pako:eNq1VVuLGzcU_iuDoGE33R18m72YNNAk2zY02yx1aaF1GZSxxhYZj1yNxlt3WWiShe3lsQlhWyhNQ1gI9KF3-d_op_ToaEZjkn1oAzXGPpKOvu87F0lHJBEjRvokzcRhMqFSBR_cGOYBfN6SdMoOhbz7yZB4e0g-DTY3rwaJyJUUGSy1w-vOtkuXgjtMUZjthNfAqKbKNGUSJrvhNTTddMbmTNIxg4VeeKsauKUpz0cFU7AShWb5wOivjD43S_j9zehTszwx-gkOfzb6L6MfGv3I6D_AgO1OvJWBQmnhgNrhm9YC8AMpRmWiHNOcZiVDvRZSL41-ZvRJTfUcjXOjz8zyS6PvIcmZ0Y-N1mB4NiRBOp6nGVVc5DHOFch884VJR10kVCas8euEg9UZD44SERzKM4qbgN7xwyt35NU1K1c_NcuvqyxBGFbldz6e9YpXpMqjAKkfOpQmuf8CzilsdKFM9lnJFWcu9r1q4Lglo1nMCkWVS_r7dryHY49Wb6_yqSTPC56A-122CN4IIM5vjf4bhf2Cah-i_QhV_Y5VqnoluFnvjj-0SWwUr-hAmpVxDDSerHvgtzRpwx25iBMxnTEMxCyB9D6SnlpD_4otU2eyFnkRvaVDwFlGE4t1YP9tlI9tNLbtnq3bynwEpy94PbjFUwZ_A0UXa3v5nEuRT1muqtrOXG9bGGdZIF-4n7CUPyLcy9M2nfZ7D6P5Hr4AiKfvG0zwKbqd1UQcOhXoZsjFneSn6HJim8NxNBPBleAynY_XfPuM14NOK3jtsk-KL7VLh8WssePriySr7wYqxzyHlX00ApEGA5oytfBAuNUdGD6exPURf6mh__N5X-m257j31HM6VS-QVnKwoBeuVDW6eK1O8IXLTfvhMlze_0-c1anycXoipB0L6GOYwrv9dqlAkq9SMuE5wyvgbfBys8K54NHfdx7NJeIj9NDgZ_fCOTT6B3fBYzOdYxBPUDvE-mf9FKyWA8ERKeVzFheKzV4drtLtDj47rK6IaHMAqPY9SVjR3Nd1Upx3Fxzfgy3vMtegZINMmZxSPoJH98juGBI1YTZvfTBzJoZkmB-DGy2VGCzyhPSVLNkGkaIcT0g_pVkBo3I2gtvjBqdjeJprlxnNPxbCD9mIKyH33QOP7zy6kP4R-Zz0O51wq9vudXtRN2q3OtvRBlmQ_lY73Nlpb0Xb7V7U6vWi4w3yBUK2wt3dVnd3p7OzG3W3tttR5_gfdlqidg?type=png)](https://mermaid.live/edit#pako:eNq1VVuLGzcU_iuDoGE33R18m72YNNAk2zY02yx1aaF1GZSxxhYZj1yNxlt3WWiShe3lsQlhWyhNQ1gI9KF3-d_op_ToaEZjkn1oAzXGPpKOvu87F0lHJBEjRvokzcRhMqFSBR_cGOYBfN6SdMoOhbz7yZB4e0g-DTY3rwaJyJUUGSy1w-vOtkuXgjtMUZjthNfAqKbKNGUSJrvhNTTddMbmTNIxg4VeeKsauKUpz0cFU7AShWb5wOivjD43S_j9zehTszwx-gkOfzb6L6MfGv3I6D_AgO1OvJWBQmnhgNrhm9YC8AMpRmWiHNOcZiVDvRZSL41-ZvRJTfUcjXOjz8zyS6PvIcmZ0Y-N1mB4NiRBOp6nGVVc5DHOFch884VJR10kVCas8euEg9UZD44SERzKM4qbgN7xwyt35NU1K1c_NcuvqyxBGFbldz6e9YpXpMqjAKkfOpQmuf8CzilsdKFM9lnJFWcu9r1q4Lglo1nMCkWVS_r7dryHY49Wb6_yqSTPC56A-122CN4IIM5vjf4bhf2Cah-i_QhV_Y5VqnoluFnvjj-0SWwUr-hAmpVxDDSerHvgtzRpwx25iBMxnTEMxCyB9D6SnlpD_4otU2eyFnkRvaVDwFlGE4t1YP9tlI9tNLbtnq3bynwEpy94PbjFUwZ_A0UXa3v5nEuRT1muqtrOXG9bGGdZIF-4n7CUPyLcy9M2nfZ7D6P5Hr4AiKfvG0zwKbqd1UQcOhXoZsjFneSn6HJim8NxNBPBleAynY_XfPuM14NOK3jtsk-KL7VLh8WssePriySr7wYqxzyHlX00ApEGA5oytfBAuNUdGD6exPURf6mh__N5X-m257j31HM6VS-QVnKwoBeuVDW6eK1O8IXLTfvhMlze_0-c1anycXoipB0L6GOYwrv9dqlAkq9SMuE5wyvgbfBys8K54NHfdx7NJeIj9NDgZ_fCOTT6B3fBYzOdYxBPUDvE-mf9FKyWA8ERKeVzFheKzV4drtLtDj47rK6IaHMAqPY9SVjR3Nd1Upx3Fxzfgy3vMtegZINMmZxSPoJH98juGBI1YTZvfTBzJoZkmB-DGy2VGCzyhPSVLNkGkaIcT0g_pVkBo3I2gtvjBqdjeJprlxnNPxbCD9mIKyH33QOP7zy6kP4R-Zz0O51wq9vudXtRN2q3OtvRBlmQ_lY73Nlpb0Xb7V7U6vWi4w3yBUK2wt3dVnd3p7OzG3W3tttR5_gfdlqidg)"
  st.markdown(multi)
  
  _, _, _, _ = st.columns(4)
  with _:
    st.image("https://img.soccersuck.com/images/2025/01/21/principle-5-1-1024x666.png", width=400)
    
  st.link_button("(Price_Cycle) พื้นฐาน Global_macro ", "https://drive.google.com/file/d/1-bNM1gPEG7i-CW1TMd_6Cu6Z5132UjGZ/view?usp=sharing")


with tab2:
  st.image("https://img.soccersuck.com/images/2025/01/21/455599160_312374775229941_5381968498530104178_n.jpg", width=1000)

with tab3:
  st.image("https://img.soccersuck.com/images/2025/01/21/275206549_1046792219234327_607135450348777933_n.jpg", width=1000)
  st.link_button("Rebalance_Fix_Asset-Ratio_Clip-1", "https://drive.google.com/file/d/1x_zlS7y9tTxWxHqD_yWpqrvFkGH2J2pB/view?usp=sharing")
  st.link_button("Rebalance_Fix_Asset-Ratio_Clip-2", "https://drive.google.com/file/d/1DiySbVSV5MeTYktk03FpOEe3tmpZYGyl/view?usp=sharing")

with tab4:
  st.image("https://img.soccersuck.com/images/2025/01/21/407896427_1617027032037031_1189622303379814904_n.jpg", width=1000)

st.write('____')
st.link_button("Principle", "https://claude.site/artifacts/f8e8bc35-1eed-4f5f-9545-2a8faf9a86dd")
st.write('____')

checkbox1 = st.checkbox(' mindmeister' , value=0 )
if checkbox1 :
  st.write('https://www.mindmeister.com/app/map/3178532454?m=outline')
