import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Volatility Harvesting Simulator", layout="wide")

st.title("📈 Volatility Harvesting & Rebalancing Premium Simulator")
st.markdown("""
กราฟจำลองเปรียบเทียบระหว่าง **กระแสเงินสดรวมจากการ Rebalance อย่างต่อเนื่อง** เทียบกับเส้นสมการ **$fix\_c \cdot \ln(P_t/P_0)$**
ส่วนต่างที่เกิดขึ้น (ช่องว่างระหว่างเส้น) คือกำไรที่สกัดมาจากความผันผวนของตลาด (Volatility Premium)
""")

# --- Sidebar Inputs ---
st.sidebar.header("ตั้งค่าพารามิเตอร์")
fix_c = st.sidebar.number_input("Fixed Capital (fix_c) - มูลค่าพอร์ตคงที่", value=10000, step=1000)
initial_price = st.sidebar.number_input("Initial Price (P0) - ราคาเริ่มต้น", value=100.0, step=10.0)

st.sidebar.subheader("พฤติกรรมของราคา (Market Condition)")
volatility = st.sidebar.slider("ความผันผวน (Volatility / Sigma)", min_value=0.1, max_value=1.5, value=0.5, step=0.1)
drift = st.sidebar.slider("แนวโน้มราคา (Drift / Mu)", min_value=-0.5, max_value=0.5, value=0.0, step=0.05)
days = st.sidebar.slider("ระยะเวลาจำลอง (วัน)", min_value=100, max_value=1000, value=365, step=50)

# สร้างปุ่มเพื่อสุ่มราคาใหม่
if st.sidebar.button("🎲 สุ่มกราฟราคาใหม่ (Regenerate)"):
    st.rerun()

# --- Simulation Logic ---
# 1. จำลองเส้นทางราคาด้วย Geometric Brownian Motion (GBM)
dt = 1 / 252 # 1 วันทำการ (สมมติ 252 วันต่อปี)
np.random.seed() # ให้สุ่มใหม่ทุกครั้ง
Z = np.random.normal(0, 1, days)
price_paths = np.zeros(days)
price_paths[0] = initial_price

for t in range(1, days):
    # สมการ GBM: Pt = Pt-1 * exp((mu - 0.5*sigma^2)dt + sigma*sqrt(dt)*Z)
    price_paths[t] = price_paths[t-1] * np.exp((drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * Z[t])

# 2. คำนวณ Rebalancing
# ถ้ารักษามูลค่าไว้ที่ fix_c ตลอดเวลา กระแสเงินสดในแต่ละวันคือ fix_c * (Pt/Pt-1 - 1)
returns = np.diff(price_paths) / price_paths[:-1]
daily_cashflow = fix_c * returns

# กระแสเงินสดสะสม (Cumulative Cashflow จากการ Rebalance)
cumulative_cashflow = np.insert(np.cumsum(daily_cashflow), 0, 0)

# 3. คำนวณเส้น Benchmark ตามสมการ fix_c * ln(Pt/P0)
benchmark = fix_c * np.log(price_paths / initial_price)

# 4. คำนวณ Volatility Premium (On Top)
volatility_premium = cumulative_cashflow - benchmark

# สร้าง DataFrame เพื่อพล็อตกราฟ
df = pd.DataFrame({
    "Day": np.arange(days),
    "Price": price_paths,
    "Cumulative Cashflow (Rebalance)": cumulative_cashflow,
    "Benchmark: fix_c * ln(Pt/P0)": benchmark,
    "Volatility Premium (On Top)": volatility_premium
})

# --- Plotting with Plotly ---
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📊 ราคาสินทรัพย์ (Asset Price)")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df["Day"], y=df["Price"], mode='lines', name='Price (Pt)', line=dict(color='orange')))
    fig_price.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_price, use_container_width=True)
    
    st.info(f"""
    **สถิติเมื่อจบการจำลอง:**
    - ราคาเริ่มต้น ($P_0$): {initial_price:.2f}
    - ราคาจุดจบ ($P_t$): {price_paths[-1]:.2f}
    - เปลี่ยนแปลง: {((price_paths[-1] - initial_price)/initial_price)*100:.2f}%
    """)

with col2:
    st.subheader("💰 เปรียบเทียบ Cashflow & Benchmark")
    fig_cf = go.Figure()
    
    # พล็อตกระแสเงินสดสะสม
    fig_cf.add_trace(go.Scatter(x=df["Day"], y=df["Cumulative Cashflow (Rebalance)"], mode='lines', 
                                name='Total Cashflow (Real Rebalance)', line=dict(color='green', width=2.5)))
    
    # พล็อตสมการ Benchmark
    fig_cf.add_trace(go.Scatter(x=df["Day"], y=df["Benchmark: fix_c * ln(Pt/P0)"], mode='lines', 
                                name='Benchmark (fix_c * ln(Pt/P0))', line=dict(color='blue', dash='dash')))
    
    # เติมสีตรงช่องว่าง (Volatility Premium)
    fig_cf.add_trace(go.Scatter(x=df["Day"], y=df["Cumulative Cashflow (Rebalance)"], mode='lines', showlegend=False, line=dict(width=0)))
    fig_cf.add_trace(go.Scatter(x=df["Day"], y=df["Benchmark: fix_c * ln(Pt/P0)"], fill='tonexty', mode='lines', 
                                name='Volatility Premium (On Top)', line=dict(width=0), fillcolor='rgba(0, 255, 0, 0.2)'))

    fig_cf.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified")
    st.plotly_chart(fig_cf, use_container_width=True)

# สรุปตัวเลข
st.divider()
st.subheader("🎯 สรุปผลลัพธ์ (วันที่สิ้นสุด)")
m1, m2, m3 = st.columns(3)
m1.metric("1. Cumulative Cashflow (ระบบเทรดจริง)", f"{cumulative_cashflow[-1]:,.2f}")
m2.metric("2. Benchmark (กำไรจากทิศทางราคา)", f"{benchmark[-1]:,.2f}")
m3.metric("3. Volatility Premium (ส่วน On Top)", f"{volatility_premium[-1]:,.2f}", "+ Pure Alpha")
