import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="Principle_F(X)", page_icon="📕" , layout="wide")

@st.cache_data
def iframe ():
  # src="https://www.mindmeister.com/app/map/3178532454?fullscreen=1&v=embedded&m=outline" 
  # st.components.v1.iframe(src, width=1500 , height=2000, scrolling=1)
  # src = "https://mermaid.live/view#pako:eNq1Vm1v3EQQ_isrS0RJOftyl3t1aQVtA1Q0EBEEEhidNvb6ztTnNfY61yOKRCFSePlIqyogIUpVRarEB959_2Z_CrOz9t4pyYfmA9Ep3p3ZfZ5nZmfHPrR8HjDLtcKYz_wJzQT54I6XEPh7M6NTNuPZ_U88y4w961Ni2zeJzxOR8RhcLee2HivXGtlngoK17dyCQWUqwpBlYNxybuFQm2N2wDI6ZuDoOPeqiXZNoyTImQBP15GLr2X5jSzP5AL-_yHLE7k4luVTnP4qy39k-UiWj2X5FwxguxavZKBQmmuglvOGGgH4bsaDwhea6YDGBUO9CrJcyPK5LI9rqhc4OJPlqVx8KcuHSHIqyyeyLGFg2JAE6aIkjKmIeDJCW47Md88ZNXXu08xny3VtZ2_VYsBRIoLD8QSjZUBvm-lr-9nNdSW3fCYX31ZZgjCUyh9MPBsVLw-FQQFSM9Uoy-S-BJxWuNSFMtnnRSQipmPfriaaO2M0HrFcUKGT_r6ab-PcoNXbq3yKLEryyIfl99mc3CAQ5_ey_BeF_YZqH-H4Mar6E0-pqhVyt949-lAlcal4RQfSrMxHQGPItnbNlmXacEfCRz6fpgwDkQsg_QpJT9Sg_B1Lps5kLfIyekWHgGlMfYW1q54qyicqGlV2zzfUyXwEt4-8Su5FIYPHnqDz9e3kIMp4MmWJqM421bWtYPRIAZmD-wWP8meEu2hW6VS_hxjNj_Aja3SaXid4A7_DJJ_g0tOaLIJqBcoU-SIt-xkuOVYFonmWBrIWi-vkGj0Yr5sqGm-Q9iZ55ZrJjTlxnRUFW8OPbs_9uG4RNBtHCXh2cEB4SPZoyMTcAOFWfW-i8WRU3_QLdX3la79SdC9w74nh1KrOkVZy8Fwv9VRHdbmvzvGl7mUVoht6-P8TZ3W5TJyGCGnHHMoZTNji3ysESDKn5E-ihL1-SGK6z2KXqJbwFiyH0iC5mMfshud5lm2Lmb3Ps4Bldp5SP0rG9gNXeW56FjkCIK5RsWnsaNBl-zFJMWpgnWKBGyzLn_SrAUvwDON-iuFCev6uXyKrJ4jgiBRGB2yUC5au6r8a7pXi1AqqUHWXYbOqH3XtPRCiXl4-y3N1s4zkOv16xxYsfhe2vcOWVyHpgJGcjxGiyic0ZS70I19U7KsxX_QmHTBH07FKxESINHebzc9oxoKAsdwB2c1ZaqtvA2hJzSKNOQ3yZnuzNWhutpoZnQc0jritOGzFYac6HLvlpMnYsxpk4kI72GyQGTz7_QZJea649j3ryGpYU5ZNaRTA58qh0uNZYsJUqaklAQtpEcNr3UvUUloIvjdPfMsVWcEaVsaL8cRyQxrnMCvSAHrvnYiO4cOmXpLS5GPOzZQFkeDZjv48wq8kXGK5h9YDy-20t5xhp9tt9Qftdm_Q6zasueX22k673-n0h91ebzDoDYfdo4b1BYJuOn2wDwfD_nDQHrR6reHRf0pTC8k"
  src = "https://claude.site/artifacts/f8e8bc35-1eed-4f5f-9545-2a8faf9a86dd"
  st.components.v1.iframe(src, width=1500 , height=800  , scrolling=1)
  
iframe()

st.link_button("Principle", "https://claude.site/artifacts/f8e8bc35-1eed-4f5f-9545-2a8faf9a86dd")

st.write('____')

checkbox1 = st.checkbox(' mindmeister' , value=0 )
if checkbox1 :
  st.write('https://www.mindmeister.com/app/map/3178532454?m=outline')
