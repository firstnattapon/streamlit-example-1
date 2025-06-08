import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

# ตั้งค่าธีมของ Plotly เพื่อความสวยงามและสอดคล้องกัน
px.defaults.template = "plotly_white"

def advanced_analytics_dashboard():
    """
    ฟังก์ชันหลักสำหรับสร้าง Advanced Analytics Dashboard
    """
    st.set_page_config(layout="wide", page_title="Advanced Backtest Analytics")

    # --- ส่วนหัว ---
    st.title("🚀 Advanced Backtest Analytics Dashboard")
    st.markdown("""
    Dashboard นี้ถูกออกแบบมาเพื่อวิเคราะห์ผลลัพธ์การ Backtest ในเชิงลึก 
    โดยเน้นการค้นหา Insights ผ่านเทคนิคทาง Data Science และ Data Visualization
    """)

    # --- File Uploader ---
    uploaded_file = st.file_uploader(
        "อัปโหลดไฟล์ CSV ผลลัพธ์ 'best_seed' ของคุณ",
        type=['csv']
    )

    if uploaded_file is None:
        st.info("กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มต้นการวิเคราะห์")
        return

    # --- Data Processing ---
    try:
        df = pd.read_csv(uploaded_file)
        # --- Feature Engineering: สร้างคอลัมน์ใหม่เพื่อการวิเคราะห์ ---
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

    # --- สร้าง Layout แบบ Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 ภาพรวม (Dashboard)",
        "🔗 วิเคราะห์ความสัมพันธ์ (Correlation)",
        "📈 การกระจายตัว (Distribution)",
        "🔬 สำรวจราย Window (Explorer)"
    ])

    # --- Tab 1: Dashboard ---
    with tab1:
        st.header("📈 ภาพรวมประสิทธิภาพ (Overall Performance)")
        
        # KPIs
        kpi_cols = st.columns(5)
        kpi_cols[0].metric("กำไรสุทธิรวม", f"{total_net_profit:,.2f}")
        kpi_cols[1].metric("Profit Factor", f"{profit_factor:.2f}", help="Gross Profit / Gross Loss. ค่าที่สูงกว่า 1.5 ถือว่าดี")
        kpi_cols[2].metric("Win Rate", f"{win_rate:.2f}%")
        kpi_cols[3].metric("Average Win", f"{avg_win:,.2f}")
        kpi_cols[4].metric("Average Loss", f"{avg_loss:,.2f}")

        st.markdown("---")
        st.subheader("💰 กำไรสุทธิ (Net Profit) vs. การเปลี่ยนแปลงราคา (%)")
        
        # กราฟหลัก (Dual-Axis)
        fig_main = go.Figure()
        fig_main.add_trace(go.Bar(
            x=df['timeline'], y=df['max_net'], name='Net Profit',
            marker_color=df['result_color'], yaxis='y1',
            hovertemplate='<b>Window %{customdata[0]}</b><br>Timeline: %{x}<br>Net Profit: %{y:,.2f}<extra></extra>',
            customdata=df[['window_number']]
        ))
        fig_main.add_trace(go.Scatter(
            x=df['timeline'], y=df['price_change_pct'], name='Price Change (%)',
            mode='lines+markers', marker_color='#1f77b4', line_width=3, yaxis='y2',
            hovertemplate='Price Change: %{y:.2f}%<extra></extra>'
        ))
        fig_main.update_layout(
            xaxis_title='Timeline Window',
            yaxis=dict(title=dict(text='Net Profit', font_color='#008000'), tickfont_color='#008000'),
            yaxis2=dict(title=dict(text='Price Change (%)', font_color='#1f77b4'), tickfont_color='#1f77b4', overlaying='y', side='right'),
            legend=dict(x=0, y=1.1, orientation='h'), hovermode='x unified', height=500
        )
        st.plotly_chart(fig_main, use_container_width=True)

    # --- Tab 2: Correlation Analysis ---
    with tab2:
        st.header("🔗 การวิเคราะห์ความสัมพันธ์ระหว่างตัวแปร")
        st.markdown("ส่วนนี้ช่วยให้เราเข้าใจว่าปัจจัยต่างๆ ส่งผลต่อกำไรอย่างไร")

        corr_cols = st.columns(2)
        with corr_cols[0]:
            st.subheader("กำไร vs. การเปลี่ยนแปลงราคา")
            fig_corr1 = px.scatter(
                df, x='price_change_pct', y='max_net', color='result',
                color_discrete_map={'Win': '#2ca02c', 'Loss': '#d62728'},
                trendline="ols",  # Ordinary Least Squares trendline
                trendline_color_override="gray",
                labels={'price_change_pct': 'Price Change (%)', 'max_net': 'Net Profit'},
                title="โมเดลทำกำไรได้ดีในสภาวะตลาดแบบใด?"
            )
            fig_corr1.add_hline(y=0, line_width=1, line_dash="dash", line_color="black")
            fig_corr1.add_vline(x=0, line_width=1, line_dash="dash", line_color="black")
            st.plotly_chart(fig_corr1, use_container_width=True)
            st.markdown("""
            *   **จุดสีเขียว (Win):** Window ที่ทำกำไร
            *   **จุดสีแดง (Loss):** Window ที่ขาดทุน
            *   **เส้นแนวโน้ม (สีเทา):** แสดงทิศทางความสัมพันธ์โดยรวม
            *   **หากเส้นเอียงขึ้น:** หมายความว่าโมเดลมักจะทำกำไรได้ดีเมื่อราคาสูงขึ้น (ตลาดกระทิง)
            *   **หากเส้นเอียงลง:** หมายความว่าโมเดลมักจะทำกำไรได้ดีเมื่อราคาลดลง (ตลาดหมี)
            """)

        with corr_cols[1]:
            st.subheader("กำไร vs. จำนวน Actions")
            fig_corr2 = px.scatter(
                df, x='action_count', y='max_net', color='result',
                color_discrete_map={'Win': '#2ca02c', 'Loss': '#d62728'},
                trendline="ols", trendline_color_override="gray",
                labels={'action_count': 'Number of Actions', 'max_net': 'Net Profit'},
                title="การซื้อขายบ่อยขึ้นส่งผลดีหรือไม่?"
            )
            fig_corr2.add_hline(y=0, line_width=1, line_dash="dash", line_color="black")
            st.plotly_chart(fig_corr2, use_container_width=True)
            st.markdown("""
            *   กราฟนี้แสดงว่าการมี Action (ซื้อ/ขาย) จำนวนมากสัมพันธ์กับกำไรที่สูงขึ้นหรือต่ำลงหรือไม่
            *   **หากเส้นเอียงขึ้น:** การเทรดบ่อยอาจส่งผลดี
            *   **หากเส้นแบนหรือเอียงลง:** การเทรดบ่อยอาจไม่ช่วยเพิ่มกำไร หรืออาจทำให้ขาดทุน (Over-trading)
            """)
        
        st.markdown("---")
        st.subheader("Correlation Heatmap")
        st.markdown("ภาพรวมความสัมพันธ์เชิงเส้นระหว่างตัวแปรทั้งหมด ค่าเข้าใกล้ 1 (สีน้ำเงินเข้ม) หมายถึงสัมพันธ์กันในทิศทางเดียวกัน, ค่าเข้าใกล้ -1 (สีแดงเข้ม) หมายถึงสัมพันธ์กันในทิศทางตรงกันข้าม")
        
        numeric_cols = df.select_dtypes(include=np.number).drop(columns=['window_number', 'start_index', 'end_index'])
        corr = numeric_cols.corr()
        fig_heatmap = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=corr.round(2).values,
            texttemplate="%{text}"
        ))
        fig_heatmap.update_layout(height=600, title="Heatmap ความสัมพันธ์ของตัวแปรเชิงตัวเลข")
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # --- Tab 3: Distribution Analysis ---
    with tab3:
        st.header("📈 การวิเคราะห์การกระจายตัวของข้อมูล")
        st.markdown("ทำความเข้าใจลักษณะและแนวโน้มของข้อมูลสำคัญ")

        dist_cols = st.columns(2)
        with dist_cols[0]:
            st.subheader("การกระจายตัวของกำไร/ขาดทุน (Net Profit)")
            fig_dist1 = px.histogram(
                df, x='max_net', nbins=50,
                color='result',
                color_discrete_map={'Win': '#2ca02c', 'Loss': '#d62728'},
                marginal="box", # เพิ่ม Box Plot ด้านบน
                title="Histogram & Box Plot ของ Net Profit"
            )
            fig_dist1.add_vline(x=0, line_width=2, line_dash="dash", line_color="black")
            st.plotly_chart(fig_dist1, use_container_width=True)
            st.markdown("กราฟนี้แสดงความถี่ของผลกำไร/ขาดทุนในแต่ละช่วง ช่วยให้เห็นว่าเรามีกำไร/ขาดทุนเล็กน้อยบ่อยครั้ง หรือมีกำไร/ขาดทุนก้อนใหญ่เป็นครั้งคราว")

        with dist_cols[1]:
            st.subheader("การกระจายตัวของจำนวน Actions")
            fig_dist2 = px.histogram(
                df, x='action_count', nbins=20,
                marginal="rug",
                title="Histogram ของ Action Count"
            )
            st.plotly_chart(fig_dist2, use_container_width=True)
            st.markdown("แสดงให้เห็นว่าโดยปกติแล้วโมเดลมีจำนวน Action ต่อ Window บ่อยที่สุดที่เท่าไหร่")

    # --- Tab 4: Window Explorer ---
    with tab4:
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
