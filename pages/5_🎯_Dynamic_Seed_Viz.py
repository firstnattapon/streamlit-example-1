import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

# ตรวจสอบว่า statsmodels ถูกติดตั้งแล้วหรือไม่ (สำหรับให้คำแนะนำที่เป็นมิตร)
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


# ตั้งค่าธีมของ Plotly เพื่อความสวยงามและสอดคล้องกัน
px.defaults.template = "plotly_white"

def advanced_analytics_dashboard():
    """
    ฟังก์ชันหลักสำหรับสร้าง Advanced Analytics Dashboard
    """
    st.set_page_config(layout="wide", page_title="Advanced Backtest Analytics")

    st.title("🚀 Advanced Backtest Analytics Dashboard")
    st.markdown("""
    Dashboard นี้ถูกออกแบบมาเพื่อวิเคราะห์ผลลัพธ์การ Backtest ในเชิงลึก 
    โดยเน้นการค้นหา Insights ผ่านเทคนิคทาง Data Science และ Data Visualization
    """)

    uploaded_file = st.file_uploader(
        "อัปโหลดไฟล์ CSV ผลลัพธ์ 'best_seed' ของคุณ",
        type=['csv']
    )

    if uploaded_file is None:
        st.info("กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มต้นการวิเคราะห์")
        return

    try:
        df = pd.read_csv(uploaded_file)
        df['result'] = np.where(df['max_net'] > 0, 'Win', 'Loss')
        df['result_color'] = df['result'].apply(lambda x: '#2ca02c' if x == 'Win' else '#d62728')
        gross_profit = df[df['max_net'] > 0]['max_net'].sum()
        gross_loss = abs(df[df['max_net'] < 0]['max_net'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        total_net_profit = df['max_net'].sum()
        winning_windows = df[df['result'] == 'Win'].shape[0]
        total_windows = df.shape[0]
        win_rate = (winning_windows / total_windows) * 100 if total_windows > 0 else 0
        avg_win = df[df['result'] == 'Win']['max_net'].mean()
        avg_loss = df[df['result'] == 'Loss']['max_net'].mean()
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")
        return

    st.success(f"ไฟล์ '{uploaded_file.name}' ถูกประมวลผลเรียบร้อยแล้ว มีข้อมูลทั้งหมด {total_windows} windows")

    tab4 = st.tabs(["🔬 สำรวจราย Window (Explorer)"])

    with tab4:
        # (โค้ดส่วนที่เหลือเหมือนเดิม)
        st.header("🔬 สำรวจข้อมูลราย Window")
        selected_window = st.selectbox(
            'เลือก Window ที่ต้องการดูรายละเอียด:',
            options=df['window_number'],
            format_func=lambda x: f"Window #{x} (Timeline: {df.loc[df['window_number'] == x, 'timeline'].iloc[0]})"
        )
        if selected_window:
            window_data = df[df['window_number'] == selected_window].iloc[0]
            st.subheader(f"รายละเอียดของ Window #{selected_window}")
            
            w_cols = st.columns(3)
            w_cols[0].metric("Net Profit", f"{window_data['max_net']:.2f}")
            w_cols[0].metric("Price Change", f"{window_data['price_change_pct']:.2f}%")
            w_cols[1].metric("Start Price", f"{window_data['start_price']:.2f}")
            w_cols[1].metric("End Price", f"{window_data['end_price']:.2f}")
            w_cols[2].metric("Best Seed", f"{window_data['best_seed']}")
            w_cols[2].metric("Action Count", f"{window_data['action_count']}")

            st.markdown(f"**Action Sequence:**")
            st.code(window_data['action_sequence'], language='json')
            
            st.markdown("**ข้อมูลดิบสำหรับ Window นี้:**")
            st.dataframe(window_data.to_frame().T.set_index('window_number'))


if __name__ == "__main__":
    advanced_analytics_dashboard()
