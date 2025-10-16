import streamlit as st
import json
import yfinance as yf
import re

st.set_page_config(page_title="CF_Graph", page_icon="🔥", layout="wide")

try:
    with open("cf_graph_config.json", "r", encoding="utf-8") as f:
        assets_config = json.load(f)
except FileNotFoundError:
    st.error("ไม่พบไฟล์ 'cf_graph_config.json'. กรุณาสร้างไฟล์และใส่ข้อมูล Asset ตามโครงสร้างใหม่")
    st.stop()

tab_names = ["DATA", "BATA" , "All_Ticker" , "Option_Sum" , "Historical_Backtest_CF" , "Call_ratio_spread"]  
tabs = st.tabs(tab_names)

current_prices = {}
asset_params = {}

with tabs[1]:
    with st.expander("หลักการ BATA", expanded=True):
        st.components.v1.iframe("https://monica.im/share/artifact?id=a3cRH7RVEHvBTjPhBC5Hd7", width=1500, height=1000, scrolling=0)
    with st.expander("หลักการ Rollover"):
        st.components.v1.iframe("https://monica.im/share/artifact?id=E9Mg5JX9RaAcfssZsU7K3E", width=1500, height=1000, scrolling=0)

with tabs[2]:
    with st.expander("All_Ticker", expanded=True):
        st.components.v1.iframe("https://monica.im/share/artifact?id=z6iWHpc2rQjTTMriGBbthi", width=1500, height=1000, scrolling=0)

with tabs[3]:
    with st.expander("Option_Sum", expanded=True):
        st.components.v1.iframe("https://monica.im/share/artifact?id=aYyGrPmCRG24qMZiB4872K", width=1500, height=1000, scrolling=0)

with tabs[4]:
    with st.expander("Historical_Backtest_CF", expanded=True):
        st.components.v1.iframe("https://monica.im/share/artifact?id=DioBCRhLKWgMw2ey5YMRKJ", width=1500, height=1000, scrolling=0)
        with st.expander("Monthly", expanded=False):        
            st.components.v1.iframe("https://monica.im/share/artifact?id=cgYnircvQqM9YUf58VHd8a", width=1500, height=1000, scrolling=0)
        
with tabs[5]:
    with st.expander("3_piecewise_line", expanded=True):
        st.components.v1.iframe("https://monica.im/share/artifact?id=Y3MdBwfsaZRLxYyj9qZvGd", width=1500, height=1000, scrolling=0)
        with st.expander("3_piecewise_line_v1", expanded=False):
            st.components.v1.iframe("https://monica.im/share/artifact?id=t8qKXjs8Aywi3PcTf4pDZM", width=1500, height=1000, scrolling=0)

def parse_final_two_numbers(s):
    nums = re.findall(r"[-+]?\d*\.?\d+", str(s))
    a = float(nums[0]) if len(nums) > 0 else 0.0
    b = float(nums[1]) if len(nums) > 1 else 0.0
    return a, b

with tabs[0]:
    st.write("⚙️ ตั้งค่าทั่วไปและประวัติ Asset")

    with st.expander(f"pnl_tracking_strategy"):
        st.json({
        "pnl_tracking_strategy": {
        "description": "กลยุทธ์การติดตามกำไรและขาดทุน (P&L) เพื่อประเมินผลการเทรดแต่ละรอบอย่างแม่นยำ",
        "components": [
          {
            "variable_name": "accumulated_realized_pnl",
            "description": "กำไร/ขาดทุนที่เกิดขึ้นจริงสะสมจากอดีต",
            "purpose": "เก็บยอดสะสมจากรอบการเทรดที่ปิดไปแล้ว ไม่นำมาคำนวณในรอบปัจจุบัน",
            "initial_value": -313.79
          },
          {
            "variable_name": "current_unrealized_pnl",
            "description": "กำไร/ขาดทุนที่ยังไม่เกิดขึ้นจริงของรอบปัจจุบัน",
            "purpose": "ใช้ติดตามผลการดำเนินงานของรอบปัจจุบันเท่านั้น โดยจะเริ่มนับจาก 0",
            "formula": "5000 * ln(P / 20.0)",
            "note": "ค่า P คือราคาปัจจุบัน (Price) และ P&L จะเป็น 0 เมื่อ P = 20.0"
          },
          {
            "variable_name": "total_lifetime_pnl",
            "description": "กำไร/ขาดทุนรวมทั้งหมดตั้งแต่เริ่มต้น",
            "purpose": "ใช้สำหรับดูภาพรวมของผลการดำเนินงานทั้งหมด",
            "formula": "accumulated_realized_pnl + current_unrealized_pnl"
          }
        ],
        "summary": "สำหรับการเทรดรอบใหม่ ให้เริ่มนับ P&L ของรอบใหม่จากศูนย์เสมอ ส่วนผลลัพธ์ของรอบเก่าให้เก็บไว้เป็นยอดสะสมแยกต่างหาก วิธีนี้จะทำให้คุณประเมินกลยุทธ์ในแต่ละรอบการเทรดได้อย่างแม่นยำที่สุดครับ"
        }
        })
    
    st.write("---")
    for asset in assets_config:
        ticker = asset.get("ticker", "N/A")
        final_str = asset.get("Final", "N/A")
        with st.expander(f"ตั้งค่าและดูประวัติ: {ticker} | \"{final_str}\""):
            try:
                final_price, final_fav = parse_final_two_numbers(final_str)
                asset_params[ticker] = {"entry": final_price, "fav": final_fav}
                try:
                    last_price = yf.Ticker(ticker).fast_info.get("lastPrice", final_price)
                except Exception:
                    st.warning(f"ไม่สามารถดึงราคาล่าสุดของ {ticker} ได้, ใช้ราคา Final แทน")
                    last_price = final_price
                current_prices[ticker] = st.number_input("ราคาปัจจุบัน (สำหรับคำนวณ)", step=0.01, value=float(last_price), key=f"price_{ticker}")
                st.write("---")
                st.write("ข้อมูลดิบจาก Config:")
                st.json(asset)
            except Exception as e:
                st.error(f"ข้อมูลใน config ของ {ticker} ผิดพลาด: {e}")
                asset_params[ticker] = {"entry": 0.0, "fav": 0.0}
                current_prices[ticker] = 0.0
