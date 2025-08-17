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

tab_names = ["DATA", "BATA"] + [asset["ticker"] for asset in assets_config]
tabs = st.tabs(tab_names)

current_prices = {}
asset_params = {}

with tabs[1]:
    with st.expander("หลักการ BATA", expanded=True):
        st.components.v1.iframe("https://monica.im/share/artifact?id=a3cRH7RVEHvBTjPhBC5Hd7", width=1500, height=1000, scrolling=0)
    with st.expander("หลักการ Rollover"):
        st.components.v1.iframe("https://monica.im/share/artifact?id=E9Mg5JX9RaAcfssZsU7K3E", width=1500, height=1000, scrolling=0)

def parse_final_two_numbers(s):
    nums = re.findall(r"[-+]?\d*\.?\d+", str(s))
    a = float(nums[0]) if len(nums) > 0 else 0.0
    b = float(nums[1]) if len(nums) > 1 else 0.0
    return a, b

with tabs[0]:
    st.write("⚙️ ตั้งค่าทั่วไปและประวัติ Asset")
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
