import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="Principle_F(X)", page_icon="📕" , layout="wide")

@st.cache_data
def iframe ():
  # src="https://www.mindmeister.com/app/map/3178532454?fullscreen=1&v=embedded&m=outline" 
  # st.components.v1.iframe(src, width=1500 , height=2000, scrolling=1)
  src=  "https://www.mermaidchart.com/play#pako:eNq1Vt9vG0UQ_ldWloiSkrNjx1GRSyNoCaWihahB8MAha323d15yvr3urW2sqhKFSIHySKsqICFKVUWqxAO_1__N_inMzt7tWUkeyAOSZc_O7H7fN7Ozu37QikTMWoNWEARhHok84ekgzAlRYzZhAzKJ7CCjCzFVAxLTVDJ0CHE4IFFGy5LDDFycZGIejalU5KN37BxC3pV0wuZCHn4atrwdtj4jQbBLgEtJkUGo277pbBtaIyOmKHh77RtgVK5pkjAJzu32DTSdO2MzJmnKINBv36kGLjTheVwyBZGdtll-bfQ3Rp-aJXz_bvSxWR4Z_RyHvxj9t9FPjH5q9J9gwHIn3spAoZAjAnXbb1sLwPeliKeRckwzmk0Z6rWQemn0S6OPaqpXaJwafWKWXxr9CElOjH5mtAbDsyEJ0nEoJFVc5EP0lch8-4zTUZcRlRFr5vXaB6seD44SERy2Jx42Cb3nh2-O5O66latfmOW3VZUgDavye5_PRsUrEuVRgNQPHUpT3P8A5xQ2ulAmuz_lijOX-141cNyS0WzISkWVK_o9O97DsUerl1f1VJLn0KYw_ZAtyHUCeX5n9D8o7FdU-wTtp6jqD9ylqlfI7Xr18GNbxEbxig6kWRkPgcaTbe_7JU3ZcEUuhpGYFAwTMUsg_QpJj62hf8OWqStZi7yI3tIhYJHRyGLt21-b5TObjW27lxt2Zz6B00deJ3d4wuDnQNHF-l4-41LkE5aram8L19sWxlkWyG_cz7iVPyHcebctp_08wmx-gA9Zo5PiGsET-BiLfIxTT2oyDt0KlAXycSf7BU45sg3ieBoHWcvUNXKFztJ130XpBultkdeu-Nr4HXdVsbA1_PDmIsrqK4LKlOcQuYsGEQk5oAlTCw-ES9254el4WJ_0c3196WO_0nSvcO2x53SqzpBWcnBfL4xUW3VxrK7xheGmCzEMd_j_k2d1uHyenghpUwHtDC684j-cKpDkdyka85y99QAeoRHLBsReCbdgOrQGKdUiY9fDMITnS82DkZAxk0FZ0IjnafDFwEZ2wxZ5CEDCoeKlcdeBNtePL4pXA_MsC5xgo390TwO24Cnm_RzThfL8VT8iqzuI4IiU8BkblooVq_ovh3upPJ2CKlV3y7B5dR_tBAcgxD5eEStLe7K85Lr8bsU2TP4Alr3PmqOQ98FJ6rc73zmHdzZ9SLgc0wL-QkgWqUrYajnOR_M-uPkktTUaK1WUg07ncypZHDNWtiGjzrwI7N8GuK060yITNC47va3uG52tbkfSRUwzLgLLEViOoHDKgm67yNOwtUnGA9Lr9zfJHH6vXt0khSgt1yhsgYLWw38B-D7sqg"
  st.components.v1.iframe(src, width=1500 , height=800  , scrolling=0)
  
iframe()
# st.link_button("Principle", "https://www.mermaidchart.com/raw/e3003041-c706-467f-b732-e3b1754530d2?theme=light&version=v0.1&format=svg")
st.write('____')

checkbox1 = st.checkbox(' mindmeister' , value=0 )
if checkbox1 :
  st.write('https://www.mindmeister.com/app/map/3178532454?m=outline')
