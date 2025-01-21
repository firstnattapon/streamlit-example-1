import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="Principle_F(X)", page_icon="📕" , layout="wide")

# @st.cache_data
# def iframe ():
#   src = "https://kroki.io/mermaid/svg/eNq1VV1LFFEYvu9XHAYKNRxS886ENCtJS9qoi0GG43hmPTg7s82cXdm7LGH7uEwRCyITEYQu-j7zb85P6X3fc2Z2USEKWpY97_l6nuf9Ohsn2Va0wXPFHt26xOBzO-ctsZXlm4FXm94qGx-fZVGWqjxLAm_Cn7cmbFxha0LxwJv052C0C504FnngTflzZNFiIroi500ReNf9JWfTRkum64VQgTftm_KF0S-NPjEl_H41um_KHaMPafrJ6J9G7xq9Z_R3MLxV0ovspI4XhDLh30QDgFfybL0TKSLp8qQjUCSi6dLoY6N3KpZTMk6MPjDlM6O3Cf_A6H2jNRiOiPCJSaZxwpXM0pDWCiRdPLNGrEXE80jUpyb9xvCCwyVphAtpWA9rN-7Ws5m1fHYEZeojU75ygQH5qO5t7ceopcxiVWEAXz2zGINo_hmMxA0kkULxtCOVFOTxgrOJNhc8CUWhuKIoP8TpAk0dUHXTBVDlMi1kFHibosduMPDujdG_SNBnUrlL9h6p-UY5cUXBFqvL4WOMXKV0SAFxDM1DIKmYplbc-UGg6HiahVHWagvUb0qge050fTT0FyqNKnaVvPPESERo7YRHALSCAzq3j05gbR2PYiKeQE-xq2xJxgKGhuK9kYW0K_MsbYlU2US2bfUCiDUQps7SR8rbBwI7v4wxxO82OfIOvoBHvfWaotqnYweORkJBAlkbmaSVe0QHdrAOLMNggc2wMd5tjtSV0hxlk9fY5TEXjTq1Ng6I6IDD-V6UuJbneVOmgbdMI8ti1uCxUD2HQbdsR8jmRuh691zN_nUjDxXWKd3tO0Kr5wyjlUIZvGjDZeXCrSqkF-0OCo124fn9Lw66znEO1izE2cygYmEJX-gHHQVyqrREGzIV2Nx34AitZXYfm3rZblcvQ-1WjRl4eA2azOj39o2mmjkh6YekGDz8Ub3mg-gTLsHEsivCQon2P2I5ubalxZbt_OnxBiDi_0EkiurZrYJgj04F3n04fk9ADf4G1n08MQ"  
#   st.components.v1.iframe(src, width=1500 , height=800  , scrolling=0)
# iframe()

tab1, tab2, tab3 , tab4  = st.tabs(["tab1", "tab2", "tab3" , "tab4"])

with tab1:
  multi = "[![](https://mermaid.ink/img/pako:eNq1VVuLGzcU_iuDoGE33R18m72YNNAk2zY02yx1aaF1GZSxxhYZj1yNxlt3WWiShe3lsQlhWyhNQ1gI9KF3-d_op_ToaEZjkn1oAzXGPpKOvu87F0lHJBEjRvokzcRhMqFSBR_cGOYBfN6SdMoOhbz7yZB4e0g-DTY3rwaJyJUUGSy1w-vOtkuXgjtMUZjthNfAqKbKNGUSJrvhNTTddMbmTNIxg4VeeKsauKUpz0cFU7AShWb5wOivjD43S_j9zehTszwx-gkOfzb6L6MfGv3I6D_AgO1OvJWBQmnhgNrhm9YC8AMpRmWiHNOcZiVDvRZSL41-ZvRJTfUcjXOjz8zyS6PvIcmZ0Y-N1mB4NiRBOp6nGVVc5DHOFch884VJR10kVCas8euEg9UZD44SERzKM4qbgN7xwyt35NU1K1c_NcuvqyxBGFbldz6e9YpXpMqjAKkfOpQmuf8CzilsdKFM9lnJFWcu9r1q4Lglo1nMCkWVS_r7dryHY49Wb6_yqSTPC56A-122CN4IIM5vjf4bhf2Cah-i_QhV_Y5VqnoluFnvjj-0SWwUr-hAmpVxDDSerHvgtzRpwx25iBMxnTEMxCyB9D6SnlpD_4otU2eyFnkRvaVDwFlGE4t1YP9tlI9tNLbtnq3bynwEpy94PbjFUwZ_A0UXa3v5nEuRT1muqtrOXG9bGGdZIF-4n7CUPyLcy9M2nfZ7D6P5Hr4AiKfvG0zwKbqd1UQcOhXoZsjFneSn6HJim8NxNBPBleAynY_XfPuM14NOK3jtsk-KL7VLh8WssePriySr7wYqxzyHlX00ApEGA5oytfBAuNUdGD6exPURf6mh__N5X-m257j31HM6VS-QVnKwoBeuVDW6eK1O8IXLTfvhMlze_0-c1anycXoipB0L6GOYwrv9dqlAkq9SMuE5wyvgbfBys8K54NHfdx7NJeIj9NDgZ_fCOTT6B3fBYzOdYxBPUDvE-mf9FKyWA8ERKeVzFheKzV4drtLtDj47rK6IaHMAqPY9SVjR3Nd1Upx3Fxzfgy3vMtegZINMmZxSPoJH98juGBI1YTZvfTBzJoZkmB-DGy2VGCzyhPSVLNkGkaIcT0g_pVkBo3I2gtvjBqdjeJprlxnNPxbCD9mIKyH33QOP7zy6kP4R-Zz0O51wq9vudXtRN2q3OtvRBlmQ_lY73Nlpb0Xb7V7U6vWi4w3yBUK2wt3dVnd3p7OzG3W3tttR5_gfdlqidg?type=png)](https://mermaid.live/edit#pako:eNq1VVuLGzcU_iuDoGE33R18m72YNNAk2zY02yx1aaF1GZSxxhYZj1yNxlt3WWiShe3lsQlhWyhNQ1gI9KF3-d_op_ToaEZjkn1oAzXGPpKOvu87F0lHJBEjRvokzcRhMqFSBR_cGOYBfN6SdMoOhbz7yZB4e0g-DTY3rwaJyJUUGSy1w-vOtkuXgjtMUZjthNfAqKbKNGUSJrvhNTTddMbmTNIxg4VeeKsauKUpz0cFU7AShWb5wOivjD43S_j9zehTszwx-gkOfzb6L6MfGv3I6D_AgO1OvJWBQmnhgNrhm9YC8AMpRmWiHNOcZiVDvRZSL41-ZvRJTfUcjXOjz8zyS6PvIcmZ0Y-N1mB4NiRBOp6nGVVc5DHOFch884VJR10kVCas8euEg9UZD44SERzKM4qbgN7xwyt35NU1K1c_NcuvqyxBGFbldz6e9YpXpMqjAKkfOpQmuf8CzilsdKFM9lnJFWcu9r1q4Lglo1nMCkWVS_r7dryHY49Wb6_yqSTPC56A-122CN4IIM5vjf4bhf2Cah-i_QhV_Y5VqnoluFnvjj-0SWwUr-hAmpVxDDSerHvgtzRpwx25iBMxnTEMxCyB9D6SnlpD_4otU2eyFnkRvaVDwFlGE4t1YP9tlI9tNLbtnq3bynwEpy94PbjFUwZ_A0UXa3v5nEuRT1muqtrOXG9bGGdZIF-4n7CUPyLcy9M2nfZ7D6P5Hr4AiKfvG0zwKbqd1UQcOhXoZsjFneSn6HJim8NxNBPBleAynY_XfPuM14NOK3jtsk-KL7VLh8WssePriySr7wYqxzyHlX00ApEGA5oytfBAuNUdGD6exPURf6mh__N5X-m257j31HM6VS-QVnKwoBeuVDW6eK1O8IXLTfvhMlze_0-c1anycXoipB0L6GOYwrv9dqlAkq9SMuE5wyvgbfBys8K54NHfdx7NJeIj9NDgZ_fCOTT6B3fBYzOdYxBPUDvE-mf9FKyWA8ERKeVzFheKzV4drtLtDj47rK6IaHMAqPY9SVjR3Nd1Upx3Fxzfgy3vMtegZINMmZxSPoJH98juGBI1YTZvfTBzJoZkmB-DGy2VGCzyhPSVLNkGkaIcT0g_pVkBo3I2gtvjBqdjeJprlxnNPxbCD9mIKyH33QOP7zy6kP4R-Zz0O51wq9vudXtRN2q3OtvRBlmQ_lY73Nlpb0Xb7V7U6vWi4w3yBUK2wt3dVnd3p7OzG3W3tttR5_gfdlqidg)"
  st.markdown(multi)
  
  _, _, _, col4 = st.columns(4)
  with col4:
    st.image("https://img.soccersuck.com/images/2025/01/21/principle-5-1-1024x666.png", width=400)
    
  st.link_button("พื้นฐาน Global_macro (Price_Cycle)", "https://drive.google.com/file/d/1-bNM1gPEG7i-CW1TMd_6Cu6Z5132UjGZ/view?usp=sharing")
  st.video("https://rr3---sn-npoeener.googlevideo.com/videoplayback?expire=1737494306&ei=wrqPZ6ruMqmJ6dsP2u-GoQg&ip=39.35.158.81&id=o-AKlhrqY-5jbYlVUm1J3b7RyIbCSuj8VFQWmvoCr_WJjW&itag=18&source=youtube&requiressl=yes&xpc=EgVo2aDSNQ%3D%3D&bui=AY2Et-M4OSCFueHBNE11XMNok8nGwjeSBmMsUOfNt2CugE63oAkb4XfFE1I-OJpcEQazBynrN5IFzZVe&spc=9kzgDWvMrWwpnpZzVSQfPPNodwf5sxQWqHeeUZFHpTGyvnUtNYTE8mn9XA4hc7Qmsg&vprv=1&svpuc=1&mime=video%2Fmp4&ns=TpeiXBLkSajX5DyEpSHzjpEQ&rqh=1&gir=yes&clen=1312973&ratebypass=yes&dur=209.165&lmt=1735792364772299&fexp=24350590,24350737,24350825,24350827,24350860,24350961,24350975,24351028,51326932,51331020,51335594,51353498,51371294,51384461&c=MWEB&sefc=1&txp=6209224&n=38tIP0ZqYprvTg&sparams=expire%2Cei%2Cip%2Cid%2Citag%2Csource%2Crequiressl%2Cxpc%2Cbui%2Cspc%2Cvprv%2Csvpuc%2Cmime%2Cns%2Crqh%2Cgir%2Cclen%2Cratebypass%2Cdur%2Clmt&sig=AJfQdSswRgIhAPtCAKqYmvEZEsKpRnikxbNSnq6e8jiu9WE_7fAan2zlAiEA2XUfglRui1_IN-pym0piiSM8HsXu1KQu_KF7ap3sQ1o%3D&title=%E0%B8%AA%E0%B8%A3%E0%B8%B8%E0%B8%9B%E0%B8%AB%E0%B8%A5%E0%B8%B1%E0%B8%81%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B8%A5%E0%B8%87%E0%B8%97%E0%B8%B8%E0%B8%99%E0%B8%AA%E0%B8%B3%E0%B8%84%E0%B8%B1%E0%B8%8D%205%20%E0%B8%82%E0%B9%89%E0%B8%AD%20%22The%20Most%20Important%20Thing%22&rm=sn-2uja-n3ue76,sn-hgne67l&rrc=79,104,80,80&req_id=cab99e2d495fa3ee&ipbypass=yes&cm2rm=sn-w5nuxa-c33ll7l,sn-30asz7l&redirect_counter=4&cms_redirect=yes&cmsv=e&met=1737472720,&mh=6l&mip=2403:6200:8853:5dd3:e9cd:9aca:849a:a128&mm=34&mn=sn-npoeener&ms=ltu&mt=1737472374&mv=m&mvi=3&pl=49&rms=ltu,au&lsparams=ipbypass,met,mh,mip,mm,mn,ms,mv,mvi,pl,rms&lsig=AGluJ3MwRQIhALUYtoVIkDYD7d5S8Vl4ZeUiUHcVBIMzl5nTfNs6lOGyAiB_GIXOqle9S2tVpnVI4Wa3tP22bEx0Tr8DvNna2fIyng%3D%3D
")


with tab2:
  st.image("https://img.soccersuck.com/images/2025/01/21/455599160_312374775229941_5381968498530104178_n.jpg", width=1000)

with tab3:
  st.image("https://img.soccersuck.com/images/2025/01/21/275206549_1046792219234327_607135450348777933_n.jpg", width=1000)
  st.link_button("Rebalance_Fix_Asset-Ratio_Clip-1", "https://drive.google.com/file/d/1x_zlS7y9tTxWxHqD_yWpqrvFkGH2J2pB/view?usp=sharing")
  st.link_button("Rebalance_Fix_Asset-Ratio_Clip-2", "https://drive.google.com/file/d/1DiySbVSV5MeTYktk03FpOEe3tmpZYGyl/view?usp=sharing")

with tab4:
  pass

st.write('____')
st.link_button("Principle", "https://claude.site/artifacts/f8e8bc35-1eed-4f5f-9545-2a8faf9a86dd")
st.write('____')

checkbox1 = st.checkbox(' mindmeister' , value=0 )
if checkbox1 :
  st.write('https://www.mindmeister.com/app/map/3178532454?m=outline')
