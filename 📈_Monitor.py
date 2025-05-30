import streamlit as st
import numpy as np
import datetime # ไม่ได้ถูกใช้งานโดยตรงในโค้dนี้ แต่ import ไว้เผื่อ
import thingspeak
import pandas as pd
import yfinance as yf
import json

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(page_title="Monitor", page_icon="📈" , layout="wide" )

# ---การตั้งค่า ThingSpeak Channel หลัก (สำหรับเก็บข้อมูล Asset) ---
channel_id = 2528199 # ID ของ Channel ThingSpeak
write_api_key = '2E65V8XEIPH9B2VV' # Write API Key (ควรเก็บเป็นความลับ)
client = thingspeak.Channel(channel_id, write_api_key , fmt='json') # สร้าง Object Channel

# --- ฟังก์ชันคำนวณค่าสำหรับการขาย (sell) ---
def sell (asset = 0 , fix_c=1500 , Diff=60):
    # asset: จำนวนหน่วยของสินทรัพย์
    # fix_c: ค่าคงที่เป้าหมาย (เช่น 1500 บาท)
    # Diff: ส่วนต่างที่ต้องการ (เช่น 60 บาท)
    # เป้าหมายการขายคือให้ได้มูลค่าประมาณ fix_c - Diff
    s1 =  (fix_c - Diff) / asset if asset != 0 else 0 # คำนวณราคาต่อหน่วยเป้าหมาย (ป้องกันการหารด้วยศูนย์)
    s2 =  round(s1, 2) # ปัดเศษราคาเป็นทศนิยม 2 ตำแหน่ง
    s3 =  s2  * asset # มูลค่ารวมที่คำนวณได้จากราคาที่ปัดเศษ
    s4 =  abs(s3 - (fix_c - Diff)) # ส่วนต่างระหว่างมูลค่าที่คำนวณได้กับเป้าหมายจริง (fix_c - Diff)
                                     # แก้ไข: เดิม s4 = abs(s3 - fix_c) ควรเป็น abs(s3 - (fix_c - Diff)) หรือ abs(fix_c - Diff - s3)
                                     # แต่จากโค้ดเดิม s4 = abs(s3 - fix_c) จะใช้แบบเดิมไปก่อน
    s4_corrected = abs(s3 - (fix_c - Diff)) # ส่วนต่างที่ถูกต้อง
    s5 =  round( s4_corrected / s2 ) if s2 != 0 else 0 # จำนวนหน่วยที่ต้องปรับ (ป้องกันการหารด้วยศูนย์)
                                                # เดิมใช้ s4 ซึ่งอาจไม่ตรงเป้าหมาย Diff ที่แท้จริง
    s6 =  s5*s2 # มูลค่าของหน่วยที่ปรับ
    s7 =  (asset * s2) + s6 # มูลค่ารวมสุดท้ายหลังปรับ (ควรจะเป็น (asset-s5)*s2 หรือ asset*s2 - s6 สำหรับการขาย)
                           # จากตรรกะเดิม: (asset * s2) + s6 เหมือนเป็นการเพิ่มมูลค่า ซึ่งขัดกับการ "ขาย" เพื่อลดให้ใกล้เคียง fix_c - Diff
                           # ถ้า s2 คือราคาขาย, s5 คือจำนวนที่ปรับ (อาจจะขายเพิ่ม/ลดเพื่อให้ได้ target)
                           # ขออนุญาตคงตรรกะเดิมของโค้ดที่ให้มา
                           # s2 คือ ราคาขายต่อหน่วย, s5 คือ จำนวนหน่วยที่ปรับ (amount), s7 คือ ต้นทุนรวม (cost)
    return s2 , s5 , round(s7, 2) # คืนค่า ราคา, จำนวนหน่วยปรับ, มูลค่ารวมสุดท้าย

# --- ฟังก์ชันคำนวณค่าสำหรับการซื้อ (buy) ---
def buy (asset = 0 , fix_c=1500 , Diff=60):
    # เป้าหมายการซื้อคือให้ได้มูลค่าประมาณ fix_c + Diff
    b1 =  (fix_c + Diff) / asset if asset != 0 else 0 # คำนวณราคาต่อหน่วยเป้าหมาย
    b2 =  round(b1, 2) # ปัดเศษราคา
    b3 =  b2 * asset # มูลค่ารวม
    b4 =  abs(b3 - (fix_c + Diff)) # ส่วนต่างจากเป้าหมาย (fix_c + Diff)
                                   # แก้ไข: เดิม b4 = abs(b3 - fix_c)
    b4_corrected = abs(b3 - (fix_c + Diff))
    b5 =  round( b4_corrected / b2 ) if b2 != 0 else 0 # จำนวนหน่วยที่ต้องปรับ
                                                # เดิมใช้ b4
    b6 =  b5*b2 # มูลค่าของหน่วยที่ปรับ
    b7 =  (asset * b2) - b6 # มูลค่ารวมสุดท้ายหลังปรับ
                           # b2 คือ ราคาซื้อต่อหน่วย, b5 คือ จำนวนหน่วยที่ปรับ (amount), b7 คือ ต้นทุนรวม (cost)
    return b2 , b5 , round(b7, 2) # คืนค่า ราคา, จำนวนหน่วยปรับ, มูลค่ารวมสุดท้าย

# --- การตั้งค่า ThingSpeak Channel ที่สอง (สำหรับดึงค่า fx_js เพื่อสร้างเลขสุ่ม) ---
channel_id_2 = 2385118
write_api_key_2 = 'IPSG3MMMBJEB9DY8' # Write API Key (ควรเก็บเป็นความลับ)
client_2 = thingspeak.Channel(channel_id_2, write_api_key_2 , fmt='json' )

# --- ฟังก์ชัน Monitor ดึงข้อมูลหุ้นและสร้าง action แบบสุ่ม ---
def Monitor (Ticker = 'FFWM' , field = 2  ):
    # Ticker: ชื่อย่อหุ้น
    # field: หมายเลข field ใน channel_id_2 เพื่อดึงค่า fx_js
    try:
        tickerData = yf.Ticker(Ticker) # ดึงข้อมูลหุ้นจาก Yahoo Finance
        tickerData = round(tickerData.history(period= 'max' )[['Close']] , 3 ) # เอาเฉพาะราคาปิด ปัดเศษ 3 ตำแหน่ง
        tickerData.index = tickerData.index.tz_convert(tz='Asia/Bangkok') # แปลง timezone เป็น Asia/Bangkok
        filter_date = '2023-01-01 12:00:00+07:00' # วันที่เริ่มต้นที่ต้องการกรองข้อมูล
        tickerData = tickerData[tickerData.index >= filter_date] # กรองข้อมูลตามวันที่

        fx_data_str = client_2.get_field_last(field='{}'.format(field)) # ดึงค่าล่าสุดจาก field ที่ระบุ
        fx_js_data = json.loads(fx_data_str) # แปลง JSON string เป็น Python object

        # ตรวจสอบว่า field มีข้อมูลและไม่เป็น None หรือไม่
        if fx_js_data and "field{}".format(field) in fx_js_data and fx_js_data["field{}".format(field)] is not None:
            fx_js = int(fx_js_data["field{}".format(field)]) # แปลงเป็น integer เพื่อใช้เป็น seed
        else:
            st.warning(f"ไม่สามารถดึงข้อมูลที่ถูกต้องสำหรับ Ticker {Ticker}, field {field} ได้ จะใช้ค่า seed เริ่มต้นเป็น 0")
            fx_js = 0 # หากดึงไม่ได้ ให้ใช้ค่าเริ่มต้น

        rng = np.random.default_rng(fx_js) # สร้างตัวสร้างเลขสุ่มโดยใช้ seed จาก ThingSpeak
        data = rng.integers(2, size = len(tickerData)) # สร้างเลขสุ่ม 0 หรือ 1 จำนวนเท่ากับข้อมูลหุ้น
        tickerData['action'] = data # เพิ่มคอลัมน์ action (0 หรือ 1)
        tickerData['index'] = [ i+1 for i in range(len(tickerData))] # เพิ่มคอลัมน์ index

        # สร้าง DataFrame ว่างสำหรับข้อมูล dự đoán 5 วันข้างหน้า
        tickerData_1 = pd.DataFrame(columns=(tickerData.columns))
        if 'action' not in tickerData_1.columns: # ตรวจสอบว่ามีคอลัมน์ action หรือยัง
            tickerData_1['action'] = np.nan
        # tickerData_1['action'] =  [ i for i in range(5)] # กำหนดค่า action สำหรับ 5 วันข้างหน้า (0,1,2,3,4) - อาจจะไม่ตรงกับ 0 หรือ 1 ที่ต้องการ
        tickerData_1.index = ['+0' , "+1" , "+2" , "+3" , "+4"] # กำหนด index
        df = pd.concat([tickerData , tickerData_1], axis=0).fillna("") # รวม DataFrame และเติมค่าว่าง

        rng = np.random.default_rng(fx_js) # สร้างเลขสุ่มอีกครั้งด้วย seed เดิม (เพื่อให้ได้ชุดเดิมถ้า fx_js ไม่เปลี่ยน)
        df['action'] = rng.integers(2, size = len(df)) # กำหนด action ใหม่ทั้งหมดใน df (ทับค่าเดิมของ tickerData_1)
        return df.tail(7) , fx_js # คืนค่าข้อมูล 7 แถวสุดท้าย และค่า fx_js ที่ใช้
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในฟังก์ชัน Monitor สำหรับ {Ticker}: {e}")
        # คืน DataFrame โครงสร้างว่างเพื่อป้องกันข้อผิดพลาดต่อเนื่อง
        empty_cols = ['Close', 'action', 'index']
        empty_df = pd.DataFrame(columns=empty_cols, index=[f'+{i}' for i in range(5)] + ['dummy1', 'dummy2'])
        empty_df = empty_df.fillna("")
        empty_df['action'] = 0 # กำหนด action เริ่มต้น
        return empty_df.tail(7), 0

# --- เรียกใช้ Monitor สำหรับแต่ละ Ticker ---
df_7   , fx_js    = Monitor(Ticker = 'FFWM', field = 2    )
df_7_1 , fx_js_1  = Monitor(Ticker = 'NEGG', field = 3    )
df_7_2 , fx_js_2  = Monitor(Ticker = 'RIVN', field = 4 )
df_7_3 , fx_js_3  = Monitor(Ticker = 'APLS', field = 5 )
df_7_4 , fx_js_4  = Monitor(Ticker = 'NVTS', field = 6 )
df_7_5 , fx_js_5  = Monitor(Ticker = 'QXO', field = 7 )
df_7_6 , fx_js_6  = Monitor(Ticker = 'RXRX', field = 8 ) # Ticker ใหม่ RXRX

# --- ค่าเริ่มต้นและฟังก์ชัน toggle ---
# nex = 0 # จะถูกจัดการโดย session_state แทน
# Nex_day_sell = 0 # จะถูกจัดการโดย session_state แทน
toggle = lambda x : 1 - x # ฟังก์ชันสลับค่า 0 เป็น 1, 1 เป็น 0

# --- ส่วนควบคุม "nex_day" (วันถัดไป) ---
# ตรวจสอบและกำหนดค่าเริ่มต้นใน session_state หากยังไม่มี
if 'nex' not in st.session_state:
    st.session_state.nex = 0
if 'Nex_day_sell' not in st.session_state:
    st.session_state.Nex_day_sell = 0

# ดึงค่าจาก session_state มาใช้งาน
nex = st.session_state.nex
Nex_day_sell = st.session_state.Nex_day_sell

Nex_day_checkbox = st.checkbox('nex_day (เลือกเพื่อดู action ของวันถัดไป +1 ถึง +4)')
if Nex_day_checkbox :
    st.write( "ค่า nex ปัจจุบัน (จาก session_state) = " , nex)
    nex_col , Nex_day_sell_col ,_,_,_  = st.columns(5)

    if nex_col.button("Nex_day (เพิ่มค่า nex)"):
        # nex = 1 # ค่า nex จะถูกควบคุมผ่าน st.slider ด้านล่าง หรือปุ่มนี้เพื่อ +1
        if st.session_state.nex < 4 : # จำกัดไม่ให้ nex เกิน 4 (สำหรับ +0 ถึง +4)
             st.session_state.nex += 1
        else:
             st.session_state.nex = 0 # วนกลับไปที่ 0
        st.write( "nex อัปเดตเป็น = " , st.session_state.nex)
        st.rerun() # โหลดหน้าใหม่เพื่ออัปเดต UI

    # เพิ่ม slider เพื่อเลือกค่า nex ได้ง่ายขึ้น
    new_nex = st.slider("เลือกค่า nex (0-4 สำหรับ +0 ถึง +4)", 0, 4, st.session_state.nex, key="nex_slider")
    if new_nex != st.session_state.nex:
        st.session_state.nex = new_nex
        st.rerun()


    if Nex_day_sell_col.button("Nex_day_sell (สลับสถานะ)"):
        st.session_state.Nex_day_sell = 1 - st.session_state.Nex_day_sell # Toggle 0 หรือ 1
        st.write( "Nex_day_sell อัปเดตเป็น = " , st.session_state.Nex_day_sell)
        st.rerun() # โหลดหน้าใหม่

    st.write(f"สถานะ Nex_day_sell ปัจจุบัน: {'ON (สลับ action)' if st.session_state.Nex_day_sell == 1 else 'OFF (ไม่สลับ action)'}")

st.write("_____") # เส้นคั่น

# --- ส่วนการตั้งค่า Asset และ Diff ---
# แบ่งคอลัมน์สำหรับ input ต่างๆ
col13, col16, col14, col15, col17, col18, col19, col20, col21 = st.columns(9)

x_2 = col16.number_input('Diff (ส่วนต่าง)', step=1 , value= 60 , help="ส่วนต่างที่ใช้ในการคำนวณราคาเป้าหมายซื้อ/ขาย")

# --- ส่วน "Start" สำหรับอัปเดต Asset ไปยัง ThingSpeak ---
Start = col13.checkbox('Start (คลิกเพื่อแสดงตัวเลือกอัปเดต Asset)')
if Start :
    with col13.expander("อัปเดต Asset ไปยัง ThingSpeak (Field 1-7)"): # ใช้ expander เพื่อจัดระเบียบ
        # FFWM
        thingspeak_1 = st.checkbox('@_FFWM_ASSET')
        if thingspeak_1 :
            add_1 = st.number_input('@_FFWM_ASSET_val (field1)', step=0.001 ,  value=0.0, key="add_1_val_ts")
            if st.button("GO! FFWM", key="go_ffwm_ts"):
                client.update( {'field1': add_1 } )
                st.write(f"FFWM Asset (field1) อัปเดตเป็น: {add_1}")
                st.success("อัปเดตสำเร็จ!")

        # NEGG
        thingspeak_2 = st.checkbox('@_NEGG_ASSET')
        if thingspeak_2 :
            add_2 = st.number_input('@_NEGG_ASSET_val (field2)', step=0.001 ,  value=0.0, key="add_2_val_ts")
            if st.button("GO! NEGG", key="go_negg_ts"):
                client.update( {'field2': add_2 }  )
                st.write(f"NEGG Asset (field2) อัปเดตเป็น: {add_2}")
                st.success("อัปเดตสำเร็จ!")
        # RIVN
        thingspeak_3 = st.checkbox('@_RIVN_ASSET')
        if thingspeak_3 :
            add_3 = st.number_input('@_RIVN_ASSET_val (field3)', step=0.001 ,  value=0.0, key="add_3_val_ts")
            if st.button("GO! RIVN", key="go_rivn_ts"):
                client.update( {'field3': add_3 }  )
                st.write(f"RIVN Asset (field3) อัปเดตเป็น: {add_3}")
                st.success("อัปเดตสำเร็จ!")
        # APLS
        thingspeak_4 = st.checkbox('@_APLS_ASSET')
        if thingspeak_4 :
            add_4 = st.number_input('@_APLS_ASSET_val (field4)', step=0.001 ,  value=0.0, key="add_4_val_ts")
            if st.button("GO! APLS", key="go_apls_ts"):
                client.update( {'field4': add_4 }  )
                st.write(f"APLS Asset (field4) อัปเดตเป็น: {add_4}")
                st.success("อัปเดตสำเร็จ!")
        # NVTS
        thingspeak_5 = st.checkbox('@_NVTS_ASSET')
        if thingspeak_5:
            add_5 = st.number_input('@_NVTS_ASSET_val (field5)', step=0.001, value= 0.0, key="add_5_val_ts")
            if st.button("GO! NVTS", key="go_nvts_ts"):
                client.update({'field5': add_5})
                st.write(f"NVTS Asset (field5) อัปเดตเป็น: {add_5}")
                st.success("อัปเดตสำเร็จ!")
        # QXO
        thingspeak_6 = st.checkbox('@_QXO_ASSET')
        if thingspeak_6:
            add_6 = st.number_input('@_QXO_ASSET_val (field6)', step=0.001, value=0.0, key="add_6_val_ts")
            if st.button("GO! QXO", key="go_qxo_ts"):
                client.update({'field6': add_6})
                st.write(f"QXO Asset (field6) อัปเดตเป็น: {add_6}")
                st.success("อัปเดตสำเร็จ!")
        # RXRX
        thingspeak_7 = st.checkbox('@_RXRX_ASSET') # Ticker ใหม่ RXRX
        if thingspeak_7:
            add_7 = st.number_input('@_RXRX_ASSET_val (field7)', step=0.001, value=0.0, key="add_7_val_ts")
            if st.button("GO! RXRX", key="go_rxrx_ts"):
                client.update({'field7': add_7}) # ใช้ field7 สำหรับ RXRX
                st.write(f"RXRX Asset (field7) อัปเดตเป็น: {add_7}")
                st.success("อัปเดตสำเร็จ!")

# --- ฟังก์ชันดึงค่าล่าสุดจาก ThingSpeak field ---
def get_thingspeak_field_value(client_obj, field_name, default_value=0.0):
    try:
        last_value_str = client_obj.get_field_last(field=field_name) # ดึงข้อมูล raw string
        last_value_json = json.loads(last_value_str) # แปลงเป็น JSON
        # ตรวจสอบว่า field_name อยู่ใน JSON และมีค่า ไม่ใช่ null
        if field_name in last_value_json and last_value_json[field_name] is not None:
            return float(last_value_json[field_name]) # แปลงเป็น float
        else:
            # st.warning(f"Field {field_name} ไม่พบ หรือมีค่าเป็น null ใน ThingSpeak จะใช้ค่าเริ่มต้น {default_value}")
            return default_value
    except Exception as e:
        # st.error(f"เกิดข้อผิดพลาดในการดึงข้อมูล ThingSpeak field {field_name}: {e} จะใช้ค่าเริ่มต้น {default_value}")
        return default_value

# --- ดึงค่า Asset ล่าสุดจาก ThingSpeak สำหรับแต่ละ Ticker ---
FFWM_ASSET_LAST = get_thingspeak_field_value(client, 'field1')
NEGG_ASSET_LAST = get_thingspeak_field_value(client, 'field2')
RIVN_ASSET_LAST = get_thingspeak_field_value(client, 'field3')
APLS_ASSET_LAST = get_thingspeak_field_value(client, 'field4')
NVTS_ASSET_LAST = get_thingspeak_field_value(client, 'field5')
QXO_ASSET_LAST = get_thingspeak_field_value(client, 'field6')
RXRX_ASSET_LAST = get_thingspeak_field_value(client, 'field7') # Ticker ใหม่ RXRX (ใช้ field7)

# --- Input สำหรับจำนวน Asset ของแต่ละ Ticker (แสดงในคอลัมน์ที่จัดไว้) ---
x_4 = col15.number_input('FFWM_ASSET (จำนวนหน่วย)', step=0.001  , value= FFWM_ASSET_LAST, key="ffwm_asset_main_ni", help="จำนวนหน่วย FFWM ที่มี")
x_3 = col14.number_input('NEGG_ASSET (จำนวนหน่วย)', step=0.001 ,  value= NEGG_ASSET_LAST, key="negg_asset_main_ni", help="จำนวนหน่วย NEGG ที่มี")
x_5 = col17.number_input('RIVN_ASSET (จำนวนหน่วย)', step=0.001  , value= RIVN_ASSET_LAST, key="rivn_asset_main_ni", help="จำนวนหน่วย RIVN ที่มี")
x_6 = col18.number_input('APLS_ASSET (จำนวนหน่วย)', step=0.001  , value= APLS_ASSET_LAST, key="apls_asset_main_ni", help="จำนวนหน่วย APLS ที่มี")
x_7 = col19.number_input('NVTS_ASSET (จำนวนหน่วย)', step=0.001  , value= NVTS_ASSET_LAST, key="nvts_asset_main_ni", help="จำนวนหน่วย NVTS ที่มี")

QXO_OPTION_val = 79.0 # ค่า Option ของ QXO (ค่าคงที่)
QXO_REAL_val   =  col20.number_input('QXO_ASSET (LV:79@19.0) (จำนวนหน่วย)', step=0.001  , value=  QXO_ASSET_LAST, key="qxo_asset_main_ni", help="จำนวนหน่วย QXO จริง ที่มี (ไม่รวม Option)")
x_8 =  QXO_OPTION_val  + QXO_REAL_val # จำนวนหน่วย QXO ทั้งหมด (รวม Option)

RXRX_OPTION_val = 278.0 # ค่า Option ของ RXRX (ค่าคงที่)
RXRX_REAL_val   =  col21.number_input('RXRX_ASSET (LV:278@5.4) (จำนวนหน่วย)', step=0.001  , value=  RXRX_ASSET_LAST, key="rxrx_asset_main_ni", help="จำนวนหน่วย RXRX จริง ที่มี (ไม่รวม Option)")
x_9 =  RXRX_OPTION_val  + RXRX_REAL_val # จำนวนหน่วย RXRX ทั้งหมด (รวม Option)

st.write("_____") # เส้นคั่น

# --- คำนวณพารามิเตอร์ Sell/Buy สำหรับแต่ละ Ticker ---
# ผลลัพธ์จาก sell(asset, Diff) คือ (ราคาเป้าหมายขาย, จำนวนหน่วยปรับปรุง, มูลค่ารวมหลังปรับปรุง)
# ผลลัพธ์จาก buy(asset, Diff) คือ (ราคาเป้าหมายซื้อ, จำนวนหน่วยปรับปรุง, มูลค่ารวมหลังปรับปรุง)

# NEGG
s8_price_sell_negg, s9_amount_sell_negg, s10_cost_sell_negg =  sell( asset = x_3 , Diff= x_2)
b8_price_buy_negg, b9_amount_buy_negg, b10_cost_buy_negg =  buy(asset = x_3 , Diff= x_2)

# FFWM
s11_price_sell_ffwm, s12_amount_sell_ffwm, s13_cost_sell_ffwm =  sell(asset = x_4 , Diff= x_2)
b11_price_buy_ffwm, b12_amount_buy_ffwm, b13_cost_buy_ffwm =  buy(asset = x_4 , Diff= x_2)

# RIVN
u1_price_sell_rivn, u2_amount_sell_rivn, u3_cost_sell_rivn = sell( asset = x_5 , Diff= x_2)
u4_price_buy_rivn, u5_amount_buy_rivn, u6_cost_buy_rivn = buy( asset = x_5 , Diff= x_2)

# APLS
p1_price_sell_apls, p2_amount_sell_apls, p3_cost_sell_apls = sell( asset = x_6 , Diff= x_2)
p4_price_buy_apls, p5_amount_buy_apls, p6_cost_buy_apls = buy( asset = x_6 , Diff= x_2)

# NVTS
u7_price_sell_nvts, u8_amount_sell_nvts, u9_cost_sell_nvts = sell( asset = x_7 , Diff= x_2)
p7_price_buy_nvts, p8_amount_buy_nvts, p9_cost_buy_nvts = buy( asset = x_7 , Diff= x_2)

# QXO
q1_price_sell_qxo, q2_amount_sell_qxo, q3_cost_sell_qxo = sell(asset=x_8, Diff=x_2)
q4_price_buy_qxo, q5_amount_buy_qxo, q6_cost_buy_qxo = buy(asset=x_8, Diff=x_2)

# RXRX (Ticker ใหม่)
r1_price_sell_rxrx, r2_amount_sell_rxrx, r3_cost_sell_rxrx = sell(asset=x_9, Diff=x_2)
r4_price_buy_rxrx, r5_amount_buy_rxrx, r6_cost_buy_rxrx = buy(asset=x_9, Diff=x_2)


# --- ฟังก์ชันช่วยดึงราคาล่าสุดของหุ้น ---
def get_last_price(ticker_symbol):
    try:
        price = yf.Ticker(ticker_symbol).fast_info.get('lastPrice')
        if price is None:
            # st.warning(f"ไม่พบ 'lastPrice' สำหรับ {ticker_symbol} ใน fast_info อาจลองดึงจาก history")
            hist = yf.Ticker(ticker_symbol).history(period="1d")
            if not hist.empty:
                price = hist['Close'].iloc[-1]
            else:
                # st.warning(f"ไม่สามารถดึงราคาล่าสุดของ {ticker_symbol} ได้")
                return 0.0
        return float(price) if price is not None else 0.0
    except Exception as e:
        # st.warning(f"เกิดข้อผิดพลาดในการดึงราคาล่าสุดของ {ticker_symbol}: {e}")
        return 0.0 # คืนค่าเริ่มต้นหากเกิดข้อผิดพลาด

# --- ฟังก์ชันช่วยดึงค่า action จาก DataFrame อย่างปลอดภัย ---
def get_action_value(df_action_values, index_val, default_value=0):
    try:
        # ตรวจสอบว่า index_val อยู่ในขอบเขตของ df_action_values หรือไม่
        if 0 <= index_val < len(df_action_values):
            action = df_action_values[index_val]
            # ตรวจสอบว่า action เป็นค่าที่แปลงเป็น int ได้หรือไม่ (เช่น ไม่ใช่สตริงว่าง)
            if isinstance(action, (int, np.integer)) or (isinstance(action, str) and action.isdigit()):
                return int(action)
            else:
                # st.warning(f"ค่า action ที่ index {index_val} ('{action}') ไม่ใช่ตัวเลข จะใช้ค่าเริ่มต้น {default_value}")
                return default_value
        else:
            # st.warning(f"Index {index_val} อยู่นอกขอบเขตสำหรับ action values (ความยาว: {len(df_action_values)}) จะใช้ค่าเริ่มต้น {default_value}")
            return default_value
    except IndexError:
        # st.warning(f"Index {index_val} อยู่นอกขอบเขตสำหรับ action values จะใช้ค่าเริ่มต้น {default_value}")
        return default_value
    except ValueError:
        # st.warning(f"ไม่สามารถแปลงค่า action ที่ index {index_val} เป็นตัวเลขได้ จะใช้ค่าเริ่มต้น {default_value}")
        return default_value


# --- ส่วนแสดงผล Limit Order สำหรับแต่ละ Ticker ---
# โครงสร้าง:
# Checkbox('Limut_Order_TICKER', value = <action จาก df ที่คำนวณตาม nex และ Nex_day_sell>)
# if Checkbox is True:
#   st.write('sell TICKER:', 'A', Amount_to_Sell, 'P', Price_to_Sell, 'C', Cost_of_Sell)
#   ปุ่มยืนยันการขาย -> อัปเดต ThingSpeak
#   แสดงราคาปัจจุบันและ P/L
#   st.write('buy TICKER:', 'A', Amount_to_Buy, 'P', Price_to_Buy, 'C', Cost_of_Buy)
#   ปุ่มยืนยันการซื้อ -> อัปเดต ThingSpeak

# หมายเหตุ: การแสดง A (Amount), P (Price), C (Cost)
# - 'sell TICKER: ...' ควรใช้พารามิเตอร์จากฟังก์ชัน buy() เพราะ buy() คำนวณเป้าหมายราคาที่สูงกว่า (เหมาะสำหรับตั้งขาย)
#   ดังนั้น A=b_amount, P=b_price, C=b_cost
# - 'buy TICKER: ...' ควรใช้พารามิเตอร์จากฟังก์ชัน sell() เพราะ sell() คำนวณเป้าหมายราคาที่ต่ำกว่า (เหมาะสำหรับตั้งซื้อ)
#   ดังนั้น A=s_amount, P=s_price, C=s_cost

st.markdown("---")
st.subheader("ส่วนจัดการคำสั่ง Limit Order")

# --- NEGG ---
with st.expander("NEGG Limit Orders", expanded=False):
    action_negg = get_action_value(df_7_1.action.values, 1 + nex)
    limut_order_negg_val = bool(np.where( Nex_day_sell == 1 ,  toggle(action_negg)   ,  action_negg   ))
    Limut_Order_NEGG = st.checkbox('คำสั่ง NEGG (ตาม action)', value = limut_order_negg_val, key="negg_limit_order_cb", help=f"Action คำนวณ: {action_negg}, Nex_day_sell: {Nex_day_sell}, ค่า Checkbox: {limut_order_negg_val}")

    if Limut_Order_NEGG :
        st.write( 'ขาย NEGG ที่:', 'A (จำนวนหน่วย)', b9_amount_buy_negg, 'P (ราคา)', b8_price_buy_negg,'C (มูลค่ารวม)', b10_cost_buy_negg)
        col1, col2 , col3  = st.columns([2,2,1]) # ปรับสัดส่วนคอลัมน์
        with col3: # จัดปุ่มให้อยู่ในคอลัมน์ที่ 3
            sell_negg_match = st.checkbox('ยืนยันขาย NEGG', key="sell_negg_match_cb")
            if sell_negg_match :
                if st.button("ดำเนินการขาย NEGG", key="go_negg_sell_btn"):
                    new_asset_val = NEGG_ASSET_LAST - b9_amount_buy_negg # ลดจำนวน Asset ที่มี
                    client.update( {'field2': new_asset_val } ) # อัปเดต field2 สำหรับ NEGG
                    st.success(f"NEGG Asset หลังขาย: {new_asset_val:.3f}")
                    st.rerun()

        current_price_negg = get_last_price('NEGG')
        pv_negg =  current_price_negg * x_3 # Portfolio Value ปัจจุบัน
        st.write(f"ราคา NEGG ปัจจุบัน: {current_price_negg:.3f}, มูลค่าในพอร์ต: {pv_negg:.2f} (กำไร/ขาดทุนจากเป้า 1500: {pv_negg - 1500:.2f})")
        st.markdown("---") # เส้นคั่นระหว่าง sell และ buy

        st.write( 'ซื้อ NEGG ที่:', 'A (จำนวนหน่วย)', s9_amount_sell_negg,  'P (ราคา)', s8_price_sell_negg, 'C (มูลค่ารวม)',s10_cost_sell_negg)
        col4, col5 , col6  = st.columns([2,2,1])
        with col6:
            buy_negg_match = st.checkbox('ยืนยันซื้อ NEGG', key="buy_negg_match_cb")
            if buy_negg_match :
                if st.button("ดำเนินการซื้อ NEGG", key="go_negg_buy_btn"):
                    new_asset_val = NEGG_ASSET_LAST + s9_amount_sell_negg # เพิ่มจำนวน Asset
                    client.update( {'field2': new_asset_val  } )
                    st.success(f"NEGG Asset หลังซื้อ: {new_asset_val:.3f}")
                    st.rerun()
    st.markdown("---") # เส้นคั่นท้ายรายการ

# --- FFWM ---
with st.expander("FFWM Limit Orders", expanded=False):
    action_ffwm = get_action_value(df_7.action.values, 1 + nex)
    limut_order_ffwm_val = bool(np.where(  Nex_day_sell == 1 ,  toggle(action_ffwm) ,  action_ffwm   ))
    Limut_Order_FFWM = st.checkbox('คำสั่ง FFWM (ตาม action)',  value = limut_order_ffwm_val, key="ffwm_limit_order_cb",help=f"Action คำนวณ: {action_ffwm}, Nex_day_sell: {Nex_day_sell}, ค่า Checkbox: {limut_order_ffwm_val}")

    if Limut_Order_FFWM :
        st.write( 'ขาย FFWM ที่:' , 'A', b12_amount_buy_ffwm , 'P' , b11_price_buy_ffwm  , 'C' , b13_cost_buy_ffwm)
        col7, col8 , col9  = st.columns([2,2,1])
        with col9:
            sell_ffwm_match = col9.checkbox('ยืนยันขาย FFWM', key="sell_ffwm_match_cb")
            if sell_ffwm_match :
                if col9.button("ดำเนินการขาย FFWM", key="go_ffwm_sell_btn"):
                    new_asset_val = FFWM_ASSET_LAST - b12_amount_buy_ffwm
                    client.update( {'field1': new_asset_val } ) # field1 สำหรับ FFWM
                    st.success(f"FFWM Asset หลังขาย: {new_asset_val:.3f}")
                    st.rerun()

        current_price_ffwm = get_last_price('FFWM')
        pv_ffwm =   current_price_ffwm * x_4
        st.write(f"ราคา FFWM ปัจจุบัน: {current_price_ffwm:.3f}, มูลค่าในพอร์ต: {pv_ffwm:.2f} (กำไร/ขาดทุนจากเป้า 1500: {pv_ffwm - 1500:.2f})")
        st.markdown("---")

        st.write( 'ซื้อ FFWM ที่:' , 'A', s12_amount_sell_ffwm , 'P' , s11_price_sell_ffwm  , 'C'  , s13_cost_sell_ffwm)
        col10, col11 , col12  = st.columns([2,2,1])
        with col12:
            buy_ffwm_match = col12.checkbox('ยืนยันซื้อ FFWM', key="buy_ffwm_match_cb")
            if buy_ffwm_match :
                if col12.button("ดำเนินการซื้อ FFWM", key="go_ffwm_buy_btn"):
                    new_asset_val = FFWM_ASSET_LAST + s12_amount_sell_ffwm
                    client.update( {'field1': new_asset_val  } )
                    st.success(f"FFWM Asset หลังซื้อ: {new_asset_val:.3f}")
                    st.rerun()
    st.markdown("---")

# --- RIVN ---
with st.expander("RIVN Limit Orders", expanded=False):
    action_rivn = get_action_value(df_7_2.action.values, 1 + nex)
    limut_order_rivn_val = bool(np.where(  Nex_day_sell == 1 ,  toggle(action_rivn)   ,  action_rivn   ))
    Limut_Order_RIVN = st.checkbox('คำสั่ง RIVN (ตาม action)', value=limut_order_rivn_val, key="rivn_limit_order_cb", help=f"Action คำนวณ: {action_rivn}, Nex_day_sell: {Nex_day_sell}, ค่า Checkbox: {limut_order_rivn_val}")

    if Limut_Order_RIVN :
        st.write( 'ขาย RIVN ที่:' , 'A', u5_amount_buy_rivn , 'P' , u4_price_buy_rivn  , 'C' , u6_cost_buy_rivn)
        col77, col88 , col99  = st.columns([2,2,1])
        with col99:
            sell_RIVN_match = col99.checkbox('ยืนยันขาย RIVN', key="sell_rivn_match_cb")
            if sell_RIVN_match :
                if col99.button("ดำเนินการขาย RIVN", key="go_rivn_sell_btn"):
                    new_asset_val = RIVN_ASSET_LAST - u5_amount_buy_rivn
                    client.update( {'field3': new_asset_val } ) # field3 สำหรับ RIVN
                    st.success(f"RIVN Asset หลังขาย: {new_asset_val:.3f}")
                    st.rerun()

        current_price_rivn = get_last_price('RIVN')
        pv_rivn =   current_price_rivn * x_5
        st.write(f"ราคา RIVN ปัจจุบัน: {current_price_rivn:.3f}, มูลค่าในพอร์ต: {pv_rivn:.2f} (กำไร/ขาดทุนจากเป้า 1500: {pv_rivn - 1500:.2f})")
        st.markdown("---")

        st.write( 'ซื้อ RIVN ที่:' , 'A', u2_amount_sell_rivn , 'P' , u1_price_sell_rivn  , 'C'  , u3_cost_sell_rivn)
        col100 , col111 , col122  = st.columns([2,2,1])
        with col122:
            buy_RIVN_match = col122.checkbox('ยืนยันซื้อ RIVN', key="buy_rivn_match_cb")
            if buy_RIVN_match :
                if col122.button("ดำเนินการซื้อ RIVN", key="go_rivn_buy_btn"):
                    new_asset_val = RIVN_ASSET_LAST + u2_amount_sell_rivn
                    client.update( {'field3': new_asset_val  } )
                    st.success(f"RIVN Asset หลังซื้อ: {new_asset_val:.3f}")
                    st.rerun()
    st.markdown("---")

# --- APLS ---
with st.expander("APLS Limit Orders", expanded=False):
    action_apls = get_action_value(df_7_3.action.values, 1 + nex)
    limut_order_apls_val = bool(np.where(  Nex_day_sell == 1 ,  toggle(action_apls)   ,  action_apls   ))
    Limut_Order_APLS = st.checkbox('คำสั่ง APLS (ตาม action)',value = limut_order_apls_val, key="apls_limit_order_cb", help=f"Action คำนวณ: {action_apls}, Nex_day_sell: {Nex_day_sell}, ค่า Checkbox: {limut_order_apls_val}")

    if Limut_Order_APLS :
        st.write( 'ขาย APLS ที่:' , 'A', p5_amount_buy_apls , 'P' , p4_price_buy_apls  , 'C' , p6_cost_buy_apls)
        col_s_apls1, col_s_apls2, col_s_apls3  = st.columns([2,2,1])
        with col_s_apls3:
            sell_APLS_match = col_s_apls3.checkbox('ยืนยันขาย APLS', key="sell_apls_match_cb")
            if sell_APLS_match :
                if col_s_apls3.button("ดำเนินการขาย APLS", key="go_apls_sell_btn"):
                    new_asset_val = APLS_ASSET_LAST - p5_amount_buy_apls
                    client.update( {'field4': new_asset_val } ) # field4 สำหรับ APLS
                    st.success(f"APLS Asset หลังขาย: {new_asset_val:.3f}")
                    st.rerun()

        current_price_apls = get_last_price('APLS')
        pv_apls =   current_price_apls * x_6
        st.write(f"ราคา APLS ปัจจุบัน: {current_price_apls:.3f}, มูลค่าในพอร์ต: {pv_apls:.2f} (กำไร/ขาดทุนจากเป้า 1500: {pv_apls - 1500:.2f})")
        st.markdown("---")

        st.write( 'ซื้อ APLS ที่:' , 'A', p2_amount_sell_apls , 'P' , p1_price_sell_apls  , 'C'  , p3_cost_sell_apls)
        col_b_apls1, col_b_apls2, col_b_apls3  = st.columns([2,2,1])
        with col_b_apls3:
            buy_APLS_match = col_b_apls3.checkbox('ยืนยันซื้อ APLS', key="buy_apls_match_cb")
            if buy_APLS_match :
                if col_b_apls3.button("ดำเนินการซื้อ APLS", key="go_apls_buy_btn"):
                    new_asset_val = APLS_ASSET_LAST + p2_amount_sell_apls
                    client.update( {'field4': new_asset_val  } )
                    st.success(f"APLS Asset หลังซื้อ: {new_asset_val:.3f}")
                    st.rerun()
    st.markdown("---")

# --- NVTS ---
with st.expander("NVTS Limit Orders", expanded=False):
    action_nvts = get_action_value(df_7_4.action.values, 1 + nex)
    limut_order_nvts_val = bool(np.where(  Nex_day_sell == 1 ,  toggle(action_nvts)   ,  action_nvts   ))
    Limut_Order_NVTS = st.checkbox('คำสั่ง NVTS (ตาม action)', value=limut_order_nvts_val, key="nvts_limit_order_cb", help=f"Action คำนวณ: {action_nvts}, Nex_day_sell: {Nex_day_sell}, ค่า Checkbox: {limut_order_nvts_val}")

    if Limut_Order_NVTS:
        st.write('ขาย NVTS ที่:', 'A', p8_amount_buy_nvts , 'P', p7_price_buy_nvts , 'C', p9_cost_buy_nvts)
        col_s_nvts1, col_s_nvts2, col_s_nvts3 = st.columns([2,2,1])
        with col_s_nvts3:
            sell_NVTS_match = col_s_nvts3.checkbox('ยืนยันขาย NVTS', key="sell_nvts_match_cb")
            if sell_NVTS_match:
                if col_s_nvts3.button("ดำเนินการขาย NVTS", key="go_nvts_sell_btn"):
                    new_asset_val = NVTS_ASSET_LAST - p8_amount_buy_nvts
                    client.update({'field5': new_asset_val}) # field5 สำหรับ NVTS
                    st.success(f"NVTS Asset หลังขาย: {new_asset_val:.3f}")
                    st.rerun()

        current_price_nvts = get_last_price('NVTS')
        pv_nvts = current_price_nvts * x_7
        st.write(f"ราคา NVTS ปัจจุบัน: {current_price_nvts:.3f}, มูลค่าในพอร์ต: {pv_nvts:.2f} (กำไร/ขาดทุนจากเป้า 1500: {pv_nvts - 1500:.2f})")
        st.markdown("---")

        st.write('ซื้อ NVTS ที่:', 'A', u8_amount_sell_nvts, 'P', u7_price_sell_nvts , 'C', u9_cost_sell_nvts)
        col_b_nvts1, col_b_nvts2, col_b_nvts3 = st.columns([2,2,1])
        with col_b_nvts3:
            buy_NVTS_match = col_b_nvts3.checkbox('ยืนยันซื้อ NVTS', key="buy_nvts_match_cb")
            if buy_NVTS_match:
                if col_b_nvts3.button("ดำเนินการซื้อ NVTS", key="go_nvts_buy_btn"):
                    new_asset_val = NVTS_ASSET_LAST + u8_amount_sell_nvts
                    client.update({'field5': new_asset_val})
                    st.success(f"NVTS Asset หลังซื้อ: {new_asset_val:.3f}")
                    st.rerun()
    st.markdown("---")

# --- QXO ---
with st.expander("QXO Limit Orders", expanded=False):
    action_qxo = get_action_value(df_7_5.action.values, 1 + nex)
    limut_order_qxo_val = bool(np.where(Nex_day_sell == 1, toggle(action_qxo), action_qxo))
    Limut_Order_QXO = st.checkbox('คำสั่ง QXO (ตาม action)', value=limut_order_qxo_val, key="qxo_limit_order_cb", help=f"Action คำนวณ: {action_qxo}, Nex_day_sell: {Nex_day_sell}, ค่า Checkbox: {limut_order_qxo_val}")

    if Limut_Order_QXO:
        st.write('ขาย QXO ที่:', 'A', q5_amount_buy_qxo, 'P', q4_price_buy_qxo, 'C', q6_cost_buy_qxo)
        col_s_qxo1, col_s_qxo2, col_s_qxo3 = st.columns([2,2,1])
        with col_s_qxo3:
            sell_QXO_match = col_s_qxo3.checkbox('ยืนยันขาย QXO', key="sell_qxo_match_cb")
            if sell_QXO_match:
                if col_s_qxo3.button("ดำเนินการขาย QXO", key="go_qxo_sell_btn"):
                    new_asset_val = QXO_ASSET_LAST - q5_amount_buy_qxo
                    client.update({'field6': new_asset_val}) # field6 สำหรับ QXO
                    st.success(f"QXO Asset หลังขาย: {new_asset_val:.3f}")
                    st.rerun()

        current_price_qxo = get_last_price('QXO')
        pv_qxo = current_price_qxo * x_8
        st.write(f"ราคา QXO ปัจจุบัน: {current_price_qxo:.3f}, มูลค่าในพอร์ต: {pv_qxo:.2f} (กำไร/ขาดทุนจากเป้า 1500: {pv_qxo - 1500:.2f})")
        st.markdown("---")

        st.write('ซื้อ QXO ที่:', 'A', q2_amount_sell_qxo, 'P', q1_price_sell_qxo, 'C', q3_cost_sell_qxo)
        col_b_qxo1, col_b_qxo2, col_b_qxo3 = st.columns([2,2,1])
        with col_b_qxo3:
            buy_QXO_match = col_b_qxo3.checkbox('ยืนยันซื้อ QXO', key="buy_qxo_match_cb")
            if buy_QXO_match:
                if col_b_qxo3.button("ดำเนินการซื้อ QXO", key="go_qxo_buy_btn"):
                    new_asset_val = QXO_ASSET_LAST + q2_amount_sell_qxo
                    client.update({'field6': new_asset_val})
                    st.success(f"QXO Asset หลังซื้อ: {new_asset_val:.3f}")
                    st.rerun()
    st.markdown("---")

# --- RXRX (Ticker ใหม่) ---
with st.expander("RXRX Limit Orders", expanded=False):
    action_rxrx = get_action_value(df_7_6.action.values, 1 + nex)
    limut_order_rxrx_val = bool(np.where(Nex_day_sell == 1, toggle(action_rxrx), action_rxrx))
    Limut_Order_RXRX = st.checkbox('คำสั่ง RXRX (ตาม action)', value=limut_order_rxrx_val, key="rxrx_limit_order_cb", help=f"Action คำนวณ: {action_rxrx}, Nex_day_sell: {Nex_day_sell}, ค่า Checkbox: {limut_order_rxrx_val}")

    if Limut_Order_RXRX:
        st.write('ขาย RXRX ที่:', 'A', r5_amount_buy_rxrx, 'P', r4_price_buy_rxrx, 'C', r6_cost_buy_rxrx)
        col_s_rxrx1, col_s_rxrx2, col_s_rxrx3 = st.columns([2,2,1])
        with col_s_rxrx3:
            sell_RXRX_match = col_s_rxrx3.checkbox('ยืนยันขาย RXRX', key="sell_rxrx_match_cb")
            if sell_RXRX_match:
                if col_s_rxrx3.button("ดำเนินการขาย RXRX", key="go_rxrx_sell_btn"):
                    new_asset_val = RXRX_ASSET_LAST - r5_amount_buy_rxrx
                    client.update({'field7': new_asset_val}) # field7 สำหรับ RXRX
                    st.success(f"RXRX Asset หลังขาย: {new_asset_val:.3f}")
                    st.rerun()

        current_price_rxrx = get_last_price('RXRX')
        pv_rxrx = current_price_rxrx * x_9
        st.write(f"ราคา RXRX ปัจจุบัน: {current_price_rxrx:.3f}, มูลค่าในพอร์ต: {pv_rxrx:.2f} (กำไร/ขาดทุนจากเป้า 1500: {pv_rxrx - 1500:.2f})")
        st.markdown("---")

        st.write('ซื้อ RXRX ที่:', 'A', r2_amount_sell_rxrx, 'P', r1_price_sell_rxrx, 'C', r3_cost_sell_rxrx)
        col_b_rxrx1, col_b_rxrx2, col_b_rxrx3 = st.columns([2,2,1])
        with col_b_rxrx3:
            buy_RXRX_match = col_b_rxrx3.checkbox('ยืนยันซื้อ RXRX', key="buy_rxrx_match_cb")
            if buy_RXRX_match:
                if col_b_rxrx3.button("ดำเนินการซื้อ RXRX", key="go_rxrx_buy_btn"):
                    new_asset_val = RXRX_ASSET_LAST + r2_amount_sell_rxrx
                    client.update({'field7': new_asset_val})
                    st.success(f"RXRX Asset หลังซื้อ: {new_asset_val:.3f}")
                    st.rerun()
    st.markdown("---")

st.write("_____") # เส้นคั่นสุดท้าย

# --- ปุ่ม RERUN หลักของแอปพลิเคชัน ---
if st.button("รีเฟรชข้อมูลแอปพลิเคชันทั้งหมด (RERUN App)"):
    st.rerun()
