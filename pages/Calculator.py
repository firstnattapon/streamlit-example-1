import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import thingspeak
import json
from pathlib import Path
import math
from typing import List, Dict, Any

# --- คลาส SimulationTracer และฟังก์ชันอื่นๆ เหมือนเดิม ---
class SimulationTracer:
    """
    คลาสสำหรับห่อหุ้มกระบวนการทั้งหมด ตั้งแต่การถอดรหัสพารามิเตอร์
    ไปจนถึงการจำลองกระบวนการกลายพันธุ์ของ action sequence
    """
    def __init__(self, encoded_string: str):
        self.encoded_string: str = encoded_string
        self._decode_and_set_attributes()

    def _decode_and_set_attributes(self):
        encoded_string = str(self.encoded_string)
        if not encoded_string.isdigit():
            raise ValueError("Input ต้องเป็นสตริงที่ประกอบด้วยตัวเลขเท่านั้น")
        decoded_numbers = []
        idx = 0
        while idx < len(encoded_string):
            try:
                length_of_number = int(encoded_string[idx])
                idx += 1
                number_str = encoded_string[idx : idx + length_of_number]
                idx += length_of_number
                decoded_numbers.append(int(number_str))
            except (IndexError, ValueError):
                raise ValueError(f"รูปแบบของสตริง '{encoded_string}' ไม่ถูกต้องที่ตำแหน่ง {idx}")
        if len(decoded_numbers) < 3:
            raise ValueError("ข้อมูลในสตริงไม่ครบถ้วน (ต้องการอย่างน้อย 3 ค่า)")
        self.action_length: int = decoded_numbers[0]
        self.mutation_rate: int = decoded_numbers[1]
        self.dna_seed: int = decoded_numbers[2]
        self.mutation_seeds: List[int] = decoded_numbers[3:]
        self.mutation_rate_float: float = self.mutation_rate / 100.0

    def run(self) -> np.ndarray:
        dna_rng = np.random.default_rng(seed=self.dna_seed)
        current_actions = dna_rng.integers(0, 2, size=self.action_length)
        if self.action_length > 0:
            current_actions[0] = 1
        for m_seed in self.mutation_seeds:
            mutation_rng = np.random.default_rng(seed=m_seed)
            mutation_mask = mutation_rng.random(self.action_length) < self.mutation_rate_float
            current_actions[mutation_mask] = 1 - current_actions[mutation_mask]
            if self.action_length > 0:
                current_actions[0] = 1
        return current_actions

st.set_page_config(page_title="Calculator", page_icon="⌨️" , layout= "centered" )

@st.cache_data(ttl=300)
def load_config(filepath="calculator_config.json"):
    config_path = Path(filepath)
    if not config_path.is_file():
        st.error(f"Error: Configuration file not found at '{filepath}'")
        st.stop()
    try:
        with config_path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from '{filepath}'. Please check for syntax errors.")
        st.stop()

CONFIG = load_config()

if st.button("Rerun"):
    st.rerun()

@st.cache_data(ttl=600)
def get_ticker_history(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    history = ticker.history(period='max')[['Close']]
    history.index = history.index.tz_convert(tz='Asia/Bangkok')
    return round(history, 3)

def average_cf(cf_config):
    history = get_ticker_history(cf_config['ticker'])
    default_date = "2024-06-18 12:00:00+07:00"
    filter_date = cf_config.get('filter_date_cf', default_date)
    filtered_data = history[history.index >= filter_date]
    count_data = len(filtered_data)
    if count_data == 0:
        return 0, 0, 0 # Return tuple of zeros
    client = thingspeak.Channel(id=cf_config['channel_id'], api_key=cf_config['write_api_key'], fmt='json')
    field_data = client.get_field_last(field=f"{cf_config['field']}")
    value = int(eval(json.loads(field_data)[f"field{cf_config['field']}"]))
    adjusted_value = value -  (cf_config.get('offset', 0))
    return (adjusted_value / count_data) , count_data , adjusted_value

@st.cache_data(ttl=60)
def production_cost(ticker, t0, fix):
    if t0 <= 0 or fix == 0:
        return None
    try:
        ticker_info = yf.Ticker(ticker)
        current_price = ticker_info.fast_info['lastPrice']
        if current_price <= 0:
            st.warning(f"Cannot calculate production for {ticker}: Current price is {current_price}, which is invalid for the formula.")
            return None
        max_production_value = (fix * -1) * math.log(t0 / 0.01)
        now_production_value = (fix * -1) * math.log(t0 / current_price)
        return max_production_value, now_production_value
    except Exception as e:
        st.warning(f"Could not calculate Production for {ticker}: {e}")
        return None

# ==============================================================================
# ฟังก์ชัน monitor ที่แก้ไขให้ action ถูกต้อง
# ==============================================================================
def monitor(channel_id, api_key, ticker, field, filter_date):
    thingspeak_client = thingspeak.Channel(id=channel_id, api_key=api_key, fmt='json')
    
    # 1. ดึงข้อมูลและเรียงลำดับจากใหม่ไปเก่า
    history = get_ticker_history(ticker)
    filtered_data = history[history.index >= filter_date].copy()
    history_desc = filtered_data.sort_index(ascending=False)

    # 2. เตรียม DataFrame สำหรับข้อมูลในอนาคต (placeholder)
    future_index = ['+4', '+3', '+2', '+1', '0']
    future_df = pd.DataFrame(index=future_index, columns=['Close'])

    # 3. รวม DataFrame และจัดเรียงคอลัมน์
    combined_df = pd.concat([future_df, history_desc])
    combined_df.fillna("", inplace=True)
    combined_df['index'] = ""
    combined_df['action'] = ""
    combined_df = combined_df[['index', 'Close', 'action']]
    
    # 4. สร้างค่า 'index' ที่เป็นตัวเลขเรียงจากมากไปน้อย
    combined_df['index'] = range(len(combined_df) - 1, -1, -1)

    # 5. ดึงค่า fx_js จาก ThingSpeak
    fx_js = "0"
    try:
        field_data = thingspeak_client.get_field_last(field=f'{field}')
        retrieved_val = json.loads(field_data)[f"field{field}"]
        if retrieved_val is not None:
            fx_js = str(retrieved_val)
    except (json.JSONDecodeError, KeyError, TypeError):
        fx_js = "0"

    # 6. สร้างและกำหนดค่า action โดยใช้ค่า 'index' เป็นตัวอ้างอิง
    try:
        tracer = SimulationTracer(encoded_string=fx_js)
        final_actions = tracer.run() # นี่คือ array ของ action ทั้งหมด

        # สร้างฟังก์ชันเพื่อดึง action ที่ถูกต้องสำหรับแต่ละแถว
        def get_action_for_row(row_index_val):
            # ตรวจสอบว่าค่า index อยู่ในขอบเขตของ array final_actions หรือไม่
            if 0 <= row_index_val < len(final_actions):
                return final_actions[row_index_val]
            return "" # ถ้าไม่มี action ที่สอดคล้อง ให้เป็นค่าว่าง

        # ใช้ .apply() เพื่อกำหนดค่า action ให้กับทุกแถวใน DataFrame
        combined_df['action'] = combined_df['index'].apply(get_action_for_row)

    except ValueError as e:
        st.warning(f"Error generating actions for {ticker} with input '{fx_js}': {e}")
        combined_df['action'] = "" # เคลียร์ค่า action ถ้ามีปัญหา
    except Exception as e:
        st.error(f"An unexpected error occurred during action generation for {ticker}: {e}")
        combined_df['action'] = ""

    # 7. ตั้งชื่อ index ของ DataFrame เพื่อให้แสดงผลในหัวตาราง
    combined_df.index.name = '↓ index'
    
    return combined_df, fx_js
# ==============================================================================


# --- 3. ส่วนแสดงผลหลัก (Main Display Logic) ---
def main():
    st.write('____')

    avg_cf_config = CONFIG.get('average_cf_config')
    if avg_cf_config:
        cf_day , count_data , adjusted_value = average_cf(avg_cf_config)
        # เปลี่ยนจำนวนคอลัมน์เป็น 5
        col1, col2, col3, col4, col5 = st.columns(5)
        
        col1.metric(label="Net (USD)", value=f"{adjusted_value:.2f}")
        col2.metric(label="Days", value=f"{count_data}")
        col3.metric(label="Avg CF/Day (USD)", value=f"{cf_day:.2f}")
        col4.metric(label="Avg CF/Month (USD)", value=f"{cf_day * 30:.2f}")

        # เพิ่มคอลัมน์ที่ 5
        max_roll = avg_cf_config.get('max_roll_over', 0.0)
        # คำนวณค่าผลต่าง
        diff_to_max_roll =  adjusted_value - max_roll 
        col5.metric(label="Max_roll - Net (USD)", value=f"{diff_to_max_roll:.2f}")
        
    else:
        st.warning("`average_cf_config` not found in configuration file.")
    st.write('____')

    monitor_config = CONFIG.get('monitor_config', {})
    default_monitor_date = "2025-04-28 12:00:00+07:00"
    monitor_filter_date = monitor_config.get('filter_date', default_monitor_date)

    for asset_config in CONFIG.get('assets', []):
        ticker = asset_config.get('ticker', 'N/A')
        monitor_field = asset_config.get('monitor_field')
        prod_params = asset_config.get('production_params', {})
        channel_id = asset_config.get('channel_id')
        api_key = asset_config.get('write_api_key')

        if not all([ticker, monitor_field, channel_id, api_key]):
            st.warning(f"Skipping an asset due to missing configuration: {asset_config}")
            continue

        asset_df, fx_js = monitor(channel_id, api_key, ticker, monitor_field, monitor_filter_date)
        
        prod_cost = production_cost(
            ticker=ticker,
            t0=prod_params.get('t0', 0.0),
            fix=prod_params.get('fix', 0.0)
        )

        prod_cost_max_display = f"{prod_cost[0]:.2f}" if prod_cost is not None else "N/A"
        prod_cost_now_display = f"{prod_cost[1]:.2f}" if prod_cost is not None else "N/A"

        st.write(ticker)
        st.write(f"f(x): {fx_js} ,   Production_max : {prod_cost_max_display}  , Production_now : {prod_cost_now_display}")
        
        st.dataframe(asset_df)

        st.write("_____")

    st.write("***ก่อนตลาดเปิดตรวจสอบ TB ล่าสุด > RE เมื่อตลอดเปิด")
    st.write("***RE > 60 USD")

if __name__ == "__main__":
    main()
