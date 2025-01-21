import streamlit as st
import streamlit.components.v1 as components
import streamlit_mermaid as stmd


st.set_page_config(page_title="Principle_F(X)", page_icon="📕" , layout="wide")

# @st.cache_data
# def iframe ():
#   # src="https://www.mindmeister.com/app/map/3178532454?fullscreen=1&v=embedded&m=outline" 
#   # st.components.v1.iframe(src, width=1500 , height=2000, scrolling=1)

#   src="https://img.soccersuck.com/images/2025/01/21/brave_screenshot_img.soccersuck.com.png" 
#   st.components.v1.iframe(src, width=1500 , height=800  , scrolling=1)

# iframe()


code ="""
flowchart TD
    Framework["Framework"] --> control["1.Control"] & beta["2.Beta"] & buffer["3.Buffer"] & leverage["4.Leverage"] & mindset["5.ใจว่างไม่ยึดติด"]
    beta --> asset["1.Asset & Product"] & value["2.มูลค่าส่วนเกินทุน"]
    asset --> inflation_assets["1.Inflation_assets"] & scarce_assets["2.Scarce_assets"]
    value --> hard_asset["1.Hard_asset<br>(สร้างคุณค่า)"] & soft_asset["2.Soft_asset<br>(ไม่สร้างคุณค่า)"]
    hard_asset --> equities["1.Equities"] & real_estate["2.Real_Estate"]
    equities --> intrinsic["key = (ซื้อได้ต่ำกว่า Intrinsic_Value)"]
    real_estate --> real_estate_key["key = 3P"]
    soft_asset --> no_compete["แข่งขันไม่ได้"]
    real_estate_key --> place["Place (ทำเล)<br>Work + Life + Stay(Environment)"] & product["Product (คุณภาพ)<br>คุณภาพออกแบบ &amp; ใช้งาน"] & price_prop["Price (ราคา)<br>ราคา &lt; *avg(ค่าg) 20 %*"]
    intrinsic --> price["Price_Cycle"] & margin["Margin of Safety"]
    price --> high_value["สร้างมูลค่าส่วนเกินทุนได้สูง"]
    margin --> high_value
    place --> high_value
    product --> high_value
    price_prop --> high_value
    no_compete --> low_value["สร้างมูลค่าส่วนเกินทุนได้ต่ำ"]
    low_value --> goal_low["3.Outcome"] & machine@{ label: "1.Goal<br style=\"--tw-border-spacing-x:\">" } & outcome["2.Machine"]
    high_value --> goal["Goal = ปิดความเสี่ยง"]
    machine --> five_step@{ label: "Goal = ปิดความเสี่ยง<br style=\"--tw-border-spacing-x:\">" }
    outcome --> new_key["5-Step Process<br>"]
    goal_low --> n3["New_Key"]"""
stmd.st_mermaid(code)



st.write('____')

checkbox1 = st.checkbox(' mindmeister' , value=0 )
if checkbox1 :
  st.write('https://www.mindmeister.com/app/map/3178532454?m=outline')
