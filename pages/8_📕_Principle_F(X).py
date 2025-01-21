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


code ="""flowchart TD
    Framework --> control[1.Control]
    Framework --> beta[2.Beta]
    Framework --> buffer[3.Buffer]
    Framework --> leverage[4.Leverage]
    Framework --> mindset[5.ใจว่างไม่ยึดติด]
    
    beta --> asset[1.Asset & Product]
    beta --> value[2.มูลค่าส่วนเกินทุน]
    
    asset --> inflation_assets[1.Inflation_assets]
    asset --> scarce_assets[2.Scarce_assets]
    
    value --> hard_asset["1.Hard_asset<br/>(สร้างคุณค่า)"]
    value --> soft_asset["2.Soft_asset<br/>(ไม่สร้างคุณค่า)"]
    
    hard_asset --> equities[1.Equities]
    hard_asset --> real_estate[2.Real_Estate]
    
    equities --> intrinsic["key = (ซื้อได้ต่ำกว่า Intrinsic_Value)"]
    real_estate --> real_estate_key["key = 3P"]
    soft_asset --> no_compete["แข่งขันไม่ได้"]
    
    real_estate_key --> place["Place (ทำเล)<br/>Work + Life + Stay(Environment)"]
    real_estate_key --> product["Product (คุณภาพ)<br/>คุณภาพออกแบบ & ใช้งาน"]
    real_estate_key --> price_prop["Price (ราคา)<br/>ราคา < *avg(ค่าg) 20 %*"]
    
    intrinsic --> price[Price_Cycle]
    intrinsic --> margin[Margin of Safety]
    
    price --> high_value["สร้างมูลค่าส่วนเกินทุนได้สูง"]
    margin --> high_value
    place --> high_value
    product --> high_value
    price_prop --> high_value
    
    no_compete --> low_value["สร้างมูลค่าส่วนเกินทุนได้ต่ำ"]
    
    low_value --> goal_low[1.Goal]
    low_value --> machine[2.Machine]
    low_value --> outcome[3.Outcome]
    
    goal_low --> goal["Goal = ปิดความเสี่ยง"]
    high_value --> goal
    
    machine --> five_step["5-Step Process"]
    outcome --> new_key[New_Key]"""
stmd.st_mermaid(code)



st.write('____')

checkbox1 = st.checkbox(' mindmeister' , value=0 )
if checkbox1 :
  st.write('https://www.mindmeister.com/app/map/3178532454?m=outline')
