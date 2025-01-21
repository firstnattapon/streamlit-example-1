import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="Principle_F(X)", page_icon="📕" , layout="wide")

# @st.cache_data
# def iframe ():
#   src = "https://kroki.io/mermaid/svg/eNq1VV1LFFEYvu9XHAYKNRxS886ENCtJS9qoi0GG43hmPTg7s82cXdm7LGH7uEwRCyITEYQu-j7zb85P6X3fc2Z2USEKWpY97_l6nuf9Ohsn2Va0wXPFHt26xOBzO-ctsZXlm4FXm94qGx-fZVGWqjxLAm_Cn7cmbFxha0LxwJv052C0C504FnngTflzZNFiIroi500ReNf9JWfTRkum64VQgTftm_KF0S-NPjEl_H41um_KHaMPafrJ6J9G7xq9Z_R3MLxV0ovspI4XhDLh30QDgFfybL0TKSLp8qQjUCSi6dLoY6N3KpZTMk6MPjDlM6O3Cf_A6H2jNRiOiPCJSaZxwpXM0pDWCiRdPLNGrEXE80jUpyb9xvCCwyVphAtpWA9rN-7Ws5m1fHYEZeojU75ygQH5qO5t7ceopcxiVWEAXz2zGINo_hmMxA0kkULxtCOVFOTxgrOJNhc8CUWhuKIoP8TpAk0dUHXTBVDlMi1kFHibosduMPDujdG_SNBnUrlL9h6p-UY5cUXBFqvL4WOMXKV0SAFxDM1DIKmYplbc-UGg6HiahVHWagvUb0qge050fTT0FyqNKnaVvPPESERo7YRHALSCAzq3j05gbR2PYiKeQE-xq2xJxgKGhuK9kYW0K_MsbYlU2US2bfUCiDUQps7SR8rbBwI7v4wxxO82OfIOvoBHvfWaotqnYweORkJBAlkbmaSVe0QHdrAOLMNggc2wMd5tjtSV0hxlk9fY5TEXjTq1Ng6I6IDD-V6UuJbneVOmgbdMI8ti1uCxUD2HQbdsR8jmRuh691zN_nUjDxXWKd3tO0Kr5wyjlUIZvGjDZeXCrSqkF-0OCo124fn9Lw66znEO1izE2cygYmEJX-gHHQVyqrREGzIV2Nx34AitZXYfm3rZblcvQ-1WjRl4eA2azOj39o2mmjkh6YekGDz8Ub3mg-gTLsHEsivCQon2P2I5ubalxZbt_OnxBiDi_0EkiurZrYJgj04F3n04fk9ADf4G1n08MQ"  
#   st.components.v1.iframe(src, width=1500 , height=800  , scrolling=0)
# iframe()

tab1, tab2, tab3 = st.tabs(["Cat", "Dog", "Owl"])

with tab1:
  multi = "[![](https://mermaid.ink/img/pako:eNq1VVuLGzcU_iuDoGE33R18m72YNNAk2zY02yx1aaF1GZSxxhYZj1yNxlt3WWiShe3lsQlhWyhNQ1gI9KF3-d_op_ToaEZjkn1oAzXGPpKOvu87F0lHJBEjRvokzcRhMqFSBR_cGOYBfN6SdMoOhbz7yZB4e0g-DTY3rwaJyJUUGSy1w-vOtkuXgjtMUZjthNfAqKbKNGUSJrvhNTTddMbmTNIxg4VeeKsauKUpz0cFU7AShWb5wOivjD43S_j9zehTszwx-gkOfzb6L6MfGv3I6D_AgO1OvJWBQmnhgNrhm9YC8AMpRmWiHNOcZiVDvRZSL41-ZvRJTfUcjXOjz8zyS6PvIcmZ0Y-N1mB4NiRBOp6nGVVc5DHOFch884VJR10kVCas8euEg9UZD44SERzKM4qbgN7xwyt35NU1K1c_NcuvqyxBGFbldz6e9YpXpMqjAKkfOpQmuf8CzilsdKFM9lnJFWcu9r1q4Lglo1nMCkWVS_r7dryHY49Wb6_yqSTPC56A-122CN4IIM5vjf4bhf2Cah-i_QhV_Y5VqnoluFnvjj-0SWwUr-hAmpVxDDSerHvgtzRpwx25iBMxnTEMxCyB9D6SnlpD_4otU2eyFnkRvaVDwFlGE4t1YP9tlI9tNLbtnq3bynwEpy94PbjFUwZ_A0UXa3v5nEuRT1muqtrOXG9bGGdZIF-4n7CUPyLcy9M2nfZ7D6P5Hr4AiKfvG0zwKbqd1UQcOhXoZsjFneSn6HJim8NxNBPBleAynY_XfPuM14NOK3jtsk-KL7VLh8WssePriySr7wYqxzyHlX00ApEGA5oytfBAuNUdGD6exPURf6mh__N5X-m257j31HM6VS-QVnKwoBeuVDW6eK1O8IXLTfvhMlze_0-c1anycXoipB0L6GOYwrv9dqlAkq9SMuE5wyvgbfBys8K54NHfdx7NJeIj9NDgZ_fCOTT6B3fBYzOdYxBPUDvE-mf9FKyWA8ERKeVzFheKzV4drtLtDj47rK6IaHMAqPY9SVjR3Nd1Upx3Fxzfgy3vMtegZINMmZxSPoJH98juGBI1YTZvfTBzJoZkmB-DGy2VGCzyhPSVLNkGkaIcT0g_pVkBo3I2gtvjBqdjeJprlxnNPxbCD9mIKyH33QOP7zy6kP4R-Zz0O51wq9vudXtRN2q3OtvRBlmQ_lY73Nlpb0Xb7V7U6vWi4w3yBUK2wt3dVnd3p7OzG3W3tttR5_gfdlqidg?type=png)](https://mermaid.live/edit#pako:eNq1VVuLGzcU_iuDoGE33R18m72YNNAk2zY02yx1aaF1GZSxxhYZj1yNxlt3WWiShe3lsQlhWyhNQ1gI9KF3-d_op_ToaEZjkn1oAzXGPpKOvu87F0lHJBEjRvokzcRhMqFSBR_cGOYBfN6SdMoOhbz7yZB4e0g-DTY3rwaJyJUUGSy1w-vOtkuXgjtMUZjthNfAqKbKNGUSJrvhNTTddMbmTNIxg4VeeKsauKUpz0cFU7AShWb5wOivjD43S_j9zehTszwx-gkOfzb6L6MfGv3I6D_AgO1OvJWBQmnhgNrhm9YC8AMpRmWiHNOcZiVDvRZSL41-ZvRJTfUcjXOjz8zyS6PvIcmZ0Y-N1mB4NiRBOp6nGVVc5DHOFch884VJR10kVCas8euEg9UZD44SERzKM4qbgN7xwyt35NU1K1c_NcuvqyxBGFbldz6e9YpXpMqjAKkfOpQmuf8CzilsdKFM9lnJFWcu9r1q4Lglo1nMCkWVS_r7dryHY49Wb6_yqSTPC56A-122CN4IIM5vjf4bhf2Cah-i_QhV_Y5VqnoluFnvjj-0SWwUr-hAmpVxDDSerHvgtzRpwx25iBMxnTEMxCyB9D6SnlpD_4otU2eyFnkRvaVDwFlGE4t1YP9tlI9tNLbtnq3bynwEpy94PbjFUwZ_A0UXa3v5nEuRT1muqtrOXG9bGGdZIF-4n7CUPyLcy9M2nfZ7D6P5Hr4AiKfvG0zwKbqd1UQcOhXoZsjFneSn6HJim8NxNBPBleAynY_XfPuM14NOK3jtsk-KL7VLh8WssePriySr7wYqxzyHlX00ApEGA5oytfBAuNUdGD6exPURf6mh__N5X-m257j31HM6VS-QVnKwoBeuVDW6eK1O8IXLTfvhMlze_0-c1anycXoipB0L6GOYwrv9dqlAkq9SMuE5wyvgbfBys8K54NHfdx7NJeIj9NDgZ_fCOTT6B3fBYzOdYxBPUDvE-mf9FKyWA8ERKeVzFheKzV4drtLtDj47rK6IaHMAqPY9SVjR3Nd1Upx3Fxzfgy3vMtegZINMmZxSPoJH98juGBI1YTZvfTBzJoZkmB-DGy2VGCzyhPSVLNkGkaIcT0g_pVkBo3I2gtvjBqdjeJprlxnNPxbCD9mIKyH33QOP7zy6kP4R-Zz0O51wq9vudXtRN2q3OtvRBlmQ_lY73Nlpb0Xb7V7U6vWi4w3yBUK2wt3dVnd3p7OzG3W3tttR5_gfdlqidg)"
  st.markdown(multi)

with tab2:
  st.image("https://img.soccersuck.com/images/2025/01/21/455599160_312374775229941_5381968498530104178_n.jpg", width=1000)

with tab3:
  st.image("https://img.soccersuck.com/images/2025/01/21/275206549_1046792219234327_607135450348777933_n.jpg", width=200)


st.link_button("Principle", "https://claude.site/artifacts/f8e8bc35-1eed-4f5f-9545-2a8faf9a86dd")
st.write('____')

checkbox1 = st.checkbox(' mindmeister' , value=0 )
if checkbox1 :
  st.write('https://www.mindmeister.com/app/map/3178532454?m=outline')
