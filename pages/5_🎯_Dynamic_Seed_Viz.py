import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io

def professional_dashboard():
    """
    ฟังก์ชันหลักสำหรับสร้าง Dashboard
    """
    st.set_page_config(layout="wide", page_title="Backtest Results Dashboard")

    st.title("📊 Professional Backtest Results Dashboard")
    st.markdown("""
    อัปโหลดไฟล์ `best_seed_results.csv` ของคุณเพื่อวิเคราะห์และแสดงผลลัพธ์การทดสอบย้อนหลัง (Backtesting)
    ในรูปแบบของกราฟและข้อมูลเชิงลึกต่างๆ
    """)

    # 1. File Uploader ตามที่ร้องขอ
    uploaded_file = st.file_uploader(
        "อัปโหลดไฟล์ CSV ผลลัพธ์ 'best_seed' ของคุณที่นี่",
        type=['csv']
    )

    if uploaded_file is not None:
        # อ่านข้อมูลจากไฟล์ที่อัปโหลด
        # ใช้ io.StringIO เพื่อให้ pandas อ่านไฟล์ในหน่วยความจำได้
        df = pd.read_csv(uploaded_file)

        # --- ส่วนแสดงผลเมื่อมีไฟล์ ---
        st.success(f"ไฟล์ '{uploaded_file.name}' ถูกอัปโหลดและประมวลผลเรียบร้อยแล้ว")

        # 2. ภาพรวม (High-Level KPIs)
        st.header("📈 ภาพรวมประสิทธิภาพ (Overall Performance)")

        total_net_profit = df['max_net'].sum()
        winning_windows = df[df['max_net'] > 0].shape[0]
        total_windows = df.shape[0]
        win_rate = (winning_windows / total_windows) * 100 if total_windows > 0 else 0
        avg_action_count = df['action_count'].mean()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="กำไรสุทธิรวม (Total Net Profit)", value=f"{total_net_profit:,.2f}")
        with col2:
            st.metric(label="อัตราการชนะ (Win Rate)", value=f"{win_rate:.2f}%", help=f"ชนะ {winning_windows} จาก {total_windows} windows")
        with col3:
            st.metric(label="จำนวน Action เฉลี่ยต่อ Window", value=f"{avg_action_count:.2f}")

        # 3. กราฟหลัก: เปรียบเทียบ Net Profit กับ Price Change
        st.header("💰 กำไรสุทธิ (Net Profit) vs. การเปลี่ยนแปลงราคา (%)")
        
        # สร้าง Figure ด้วย Plotly Graph Objects เพื่อทำกราฟ 2 แกน
        fig_main = go.Figure()

        # เพิ่มกราฟแท่งสำหรับ Net Profit (แกน Y1 - ซ้าย)
        fig_main.add_trace(go.Bar(
            x=df['timeline'],
            y=df['max_net'],
            name='Net Profit',
            marker_color=['#2ca02c' if val > 0 else '#d62728' for val in df['max_net']], # เขียวสำหรับบวก, แดงสำหรับลบ
            yaxis='y1',
            hovertemplate='<b>Window %{customdata[0]}</b><br>' +
                          'Timeline: %{x}<br>' +
                          'Net Profit: %{y:,.2f}<extra></extra>',
            customdata=df[['window_number']]
        ))

        # เพิ่มกราฟเส้นสำหรับ Price Change % (แกน Y2 - ขวา)
        fig_main.add_trace(go.Scatter(
            x=df['timeline'],
            y=df['price_change_pct'],
            name='Price Change (%)',
            mode='lines+markers',
            marker=dict(color='#1f77b4'),
            line=dict(width=3),
            yaxis='y2',
            hovertemplate='<b>Window %{customdata[0]}</b><br>' +
                          'Timeline: %{x}<br>' +
                          'Price Change: %{y:.2f}%<extra></extra>',
            customdata=df[['window_number']]
        ))

        # ตั้งค่า Layout ของกราฟ
        fig_main.update_layout(
            xaxis_title='Timeline Window',
            yaxis=dict(
                title='Net Profit',
                titlefont=dict(color='#2ca02c'),
                tickfont=dict(color='#2ca02c')
            ),
            yaxis2=dict(
                title='Price Change (%)',
                titlefont=dict(color='#1f77b4'),
                tickfont=dict(color='#1f77b4'),
                overlaying='y',
                side='right'
            ),
            legend=dict(x=0, y=1.1, orientation='h'),
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig_main, use_container_width=True)


        # 4. กราฟวิเคราะห์เพิ่มเติม
        st.header("🔍 การวิเคราะห์เพิ่มเติม (Additional Analysis)")
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("จำนวน Action ในแต่ละ Window")
            fig_actions = px.bar(
                df,
                x='timeline',
                y='action_count',
                title="Action Count per Window",
                labels={'timeline': 'Timeline Window', 'action_count': 'Number of Actions'},
                color='action_count',
                color_continuous_scale=px.colors.sequential.Viridis
            )
            fig_actions.update_layout(xaxis_title=None)
            st.plotly_chart(fig_actions, use_container_width=True)

        with col_b:
            st.subheader("Best Seed ที่ใช้ในแต่ละ Window")
            fig_seeds = px.line(
                df,
                x='timeline',
                y='best_seed',
                title="Best Seed per Window",
                labels={'timeline': 'Timeline Window', 'best_seed': 'Best Seed Value'},
                markers=True
            )
            fig_seeds.update_traces(line=dict(color='#ff7f0e', width=2))
            fig_seeds.update_layout(xaxis_title=None)
            st.plotly_chart(fig_seeds, use_container_width=True)

        # 5. การวิเคราะห์ราย Window (Drill-Down)
        st.header("🔬 วิเคราะห์ราย Window (Detailed Window Analysis)")
        
        # สร้าง selectbox ให้ผู้ใช้เลือก
        options = df['window_number'].unique()
        selected_window = st.selectbox(
            'เลือก Window ที่ต้องการดูรายละเอียด:',
            options=options
        )

        if selected_window:
            # กรองข้อมูลเฉพาะ window ที่เลือก
            window_data = df[df['window_number'] == selected_window].iloc[0]

            st.subheader(f"รายละเอียดของ Window #{selected_window}")
            
            # แสดงข้อมูลในรูปแบบคอลัมน์
            w_col1, w_col2, w_col3 = st.columns(3)
            with w_col1:
                st.metric("Net Profit", f"{window_data['max_net']:.2f}")
                st.metric("Price Change", f"{window_data['price_change_pct']:.2f}%")
            with w_col2:
                st.metric("Start Price", f"{window_data['start_price']:.2f}")
                st.metric("End Price", f"{window_data['end_price']:.2f}")
            with w_col3:
                st.metric("Best Seed", f"{window_data['best_seed']}")
                st.metric("Action Count", f"{window_data['action_count']}")

            # แสดง Action Sequence
            st.markdown(f"**Timeline:** `{window_data['timeline']}`")
            st.markdown("**Action Sequence:**")
            st.code(window_data['action_sequence'], language='json')
            
            # แสดงตารางข้อมูลของแถวที่เลือก
            st.dataframe(window_data.to_frame().T.set_index('window_number'))

    else:
        # --- ส่วนแสดงผลเมื่อยังไม่มีไฟล์ ---
        st.info("กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มต้นการวิเคราะห์")
        st.markdown("### ตัวอย่างข้อมูลในไฟล์ CSV ที่ต้องการ:")
        # สร้างข้อมูลตัวอย่างเพื่อแสดงให้ผู้ใช้ดู
        sample_data = {
            'window_number': [1, 2],
            'timeline': ['2023-01-03 ถึง 2023-02-14', '2023-02-15 ถึง 2023-03-29'],
            'start_index': [0, 30],
            'end_index': [29, 59],
            'window_size': [30, 30],
            'best_seed': [17321, 28422],
            'max_net': [262.97, 90.22],
            'start_price': [26.6, 35.0],
            'end_price': [32.2, 26.2],
            'price_change_pct': [21.05, -25.14],
            'action_count': [13, 10],
            'action_sequence': ['[1, 1, 0, 1, ...]', '[1, 0, 0, 0, ...]']
        }
        st.dataframe(pd.DataFrame(sample_data))


if __name__ == "__main__":
    professional_dashboard()
