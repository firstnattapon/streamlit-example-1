import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="Principle_F(X)", page_icon="📕" , layout="wide")

@st.cache_data
def iframe ():
  # src="https://www.mindmeister.com/app/map/3178532454?fullscreen=1&v=embedded&m=outline" 
  # st.components.v1.iframe(src, width=1500 , height=2000, scrolling=1)
  src = "https://mermaid.live/view#pako:eNq1Vm1v3EQQ_isrS0RJOftyl3t1aQVtA1Q0EBEEEhidNvb6ztTnNfY61yOKRCFSePlIqyogIUpVRarEB959_2Z_CrOz9t4pyYfmA9Ep3p3ZfZ5nZmfHPrR8HjDLtcKYz_wJzQT54I6XEPh7M6NTNuPZ_U88y4w961Ni2zeJzxOR8RhcLee2HivXGtlngoK17dyCQWUqwpBlYNxybuFQm2N2wDI6ZuDoOPeqiXZNoyTImQBP15GLr2X5jSzP5AL-_yHLE7k4luVTnP4qy39k-UiWj2X5FwxguxavZKBQmmuglvOGGgH4bsaDwhea6YDGBUO9CrJcyPK5LI9rqhc4OJPlqVx8KcuHSHIqyyeyLGFg2JAE6aIkjKmIeDJCW47Md88ZNXXu08xny3VtZ2_VYsBRIoLD8QSjZUBvm-lr-9nNdSW3fCYX31ZZgjCUyh9MPBsVLw-FQQFSM9Uoy-S-BJxWuNSFMtnnRSQipmPfriaaO2M0HrFcUKGT_r6ab-PcoNXbq3yKLEryyIfl99mc3CAQ5_ey_BeF_YZqH-H4Mar6E0-pqhVyt949-lAlcal4RQfSrMxHQGPItnbNlmXacEfCRz6fpgwDkQsg_QpJT9Sg_B1Lps5kLfIyekWHgGlMfYW1q54qyicqGlV2zzfUyXwEt4-8Su5FIYPHnqDz9e3kIMp4MmWJqM421bWtYPRIAZmD-wWP8meEu2hW6VS_hxjNj_Aja3SaXid4A7_DJJ_g0tOaLIJqBcoU-SIt-xkuOVYFonmWBrIWi-vkGj0Yr5sqGm-Q9iZ55ZrJjTlxnRUFW8OPbs_9uG4RNBtHCXh2cEB4SPZoyMTcAOFWfW-i8WRU3_QLdX3la79SdC9w74nh1KrOkVZy8Fwv9VRHdbmvzvGl7mUVoht6-P8TZ3W5TJyGCGnHHMoZTNji3ysESDKn5E-ihL1-SGK6z2KXqJbwFiyH0iC5mMfshud5lm2Lmb3Ps4Bldp5SP0rG9gNXeW56FjkCIK5RsWnsaNBl-zFJMWpgnWKBGyzLn_SrAUvwDON-iuFCev6uXyKrJ4jgiBRGB2yUC5au6r8a7pXi1AqqUHWXYbOqH3XtPRCiXl4-y3N1s4zkOv16xxYsfhe2vcOWVyHpgJGcjxGiyic0ZS70I19U7KsxX_QmHTBH07FKxESINHebzc9oxoKAsdwB2c1ZaqtvA2hJzSKNOQ3yZnuzNWhutpoZnQc0jritOGzFYac6HLvlpMnYsxpk4kI72GyQGTz7_QZJea649j3ryGpYU5ZNaRTA58qh0uNZYsJUqaklAQtpEcNr3UvUUloIvjdPfMsVWcEaVsaL8cRyQxrnMCvSAHrvnYiO4cOmXpLS5GPOzZQFkeDZjv48wq8kXGK5h9YDy-20t5xhp9tt9Qftdm_Q6zasueX22k673-n0h91ebzDoDYfdo4b1BYJuOn2wDwfD_nDQHrR6reHRf0pTC8k"
  st.components.v1.iframe(src, width=1500 , height=800  , scrolling=0)
iframe()

flowchart_code = """
flowchart TD Framework["Framework"] --> control["1.Control"] & beta["2.Beta"] & buffer["3.Buffer"] & leverage["4.Leverage"] & mindset["5.ใจว่างไม่ยึดติด"]
 beta --> asset["1.Asset & Product"] & value["2.มูลค่าส่วนเกินทุน"]
 asset --> inflation_assets["1.Inflation_assets"] & scarce_assets["2.Scarce_assets"]
 value --> hard_asset["1.Hard_asset<br>(สร้างคุณค่า)"] & soft_asset["2.Soft_asset<br>(ไม่สร้างคุณค่า)"]
 hard_asset --> equities["1.Equities"] & real_estate["2.Real_Estate"]
 equities --> intrinsic["key = (ซื้อได้ต่ำกว่า Intrinsic_Value)"]
 real_estate --> real_estate_key["key =3P"]
 soft_asset --> no_compete["แข่งขันไม่ได้"]
 real_estate_key --> place["Place (ทำเล)<br>Work + Life + Stay(Environment)"] & product["Product (คุณภาพ)<br>คุณภาพออกแบบ &amp; ใช้งาน"] & price_prop["Price (ราคา)<br>ราคา &lt; *avg(ค่าg)20 %*"]
 intrinsic --> price["Price_Cycle"] & margin["Margin of Safety"]
 price --> high_value["สร้างมูลค่าส่วนเกินทุนได้สูง"]
 margin --> high_value place --> high_value product --> high_value price_prop --> high_value no_compete --> low_value["สร้างมูลค่าส่วนเกินทุนได้ต่ำ"]
 low_value --> goal_low["3.Outcome"] & machine@{ label: "1.Goal<br style=\"--tw-border-spacing-x:\">" } & outcome["2.Machine"]
 high_value --> goal["Goal = ปิดความเสี่ยง"]
 machine --> five_step@{ label: "Goal = ปิดความเสี่ยง<br style=\"--tw-border-spacing-x:\">" }
 outcome --> new_key["5-Step Process<br>"]
 goal_low --> n3["New_Key"]
 n4[" "]
 machine@{ shape: rect}
 five_step@{ shape: rect}
 n4@{ img: "https://jareddees.com/wp-content/uploads/2018/01/raydalio-five-step-process-1.png", h:244, w:277, pos: "b"}
"""

st.graphviz_chart(flowchart_code)


st.write('____')

checkbox1 = st.checkbox(' mindmeister' , value=0 )
if checkbox1 :
  st.write('https://www.mindmeister.com/app/map/3178532454?m=outline')
