import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as si

st.set_page_config(page_title="LEAPS Convexity vs Stock Rebalance", layout="wide")

st.title("🚀 Convexity Simulator: Stock Rebalance vs LEAPS Option")
st.markdown("""
กราฟจำลองเปรียบเทียบกลยุทธ์ตามที่คุณออกแบบ:
1. **Benchmark:** $fix\_c \cdot \ln(P_t / P_0)$
2. **Rebalance:** กระแสเงินสดสะสมจากการ Rebalance หุ้นรายวัน (Volatility Premium)
3. **LEAPS Convexity:** การใช้ Deep ITM Call Option แทนหุ้น (สะท้อนพลังของ Gamma และการหดตัวของ EV)
""")

# --- Black-Scholes Formula ---
def bs_call(S, K, T, r, sigma):
    # ป้องกัน T ติดลบหรือเป็น 0
    T = np.maximum(T, 1e-5)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    return call

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ ตั้งค่าพารามิเตอร์")
fix_c = st.sidebar.number_input("Fixed Capital (fix_c)", value=100000, step=10000)
P0 = st.sidebar.number_input("Initial Price (P0)", value=100.0, step=10.0)

st.sidebar.subheader("LEAPS Option Specs")
# กำหนดให้เป็น Deep ITM เพื่อใช้แทนหุ้น
strike = st.sidebar.number_input("Strike Price (K) - ควรต่ำกว่า P0", value=80.0, step=5.0)
r = st.sidebar.number_input("Risk-free Rate (r)", value=0.02, step=0.01)

st.sidebar.subheader("Market Dynamics")
sigma = st.sidebar.slider("Volatility (Sigma)", min_value=0.1, max_value=1.0, value=0.4, step=0.05)
drift = st.sidebar.slider("Drift (Mu)", min_value=-0.5, max_value=0.5, value=0.05, step=0.05)
days = st.sidebar.slider("Days to Expiry (T)", min_value=100, max_value=730, value=365, step=30)

if st.sidebar.button("🎲 สุ่มกราฟราคาใหม่"):
    st.rerun()

# --- Simulation Logic ---
dt = 1 / 252 # 1 วันทำการ
np.random.seed()
Z = np.random.normal(0, 1, days)
P_t = np.zeros(days)
P_t[0] = P0

# 1. สร้างเส้นทางราคา (GBM)
for t in range(1, days):
    P_t[t] = P_t[t-1] * np.exp((drift - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t])

# 2. คำนวณเส้นที่ 1: Benchmark (Log Return)
benchmark = fix_c * np.log(P_t / P0)

# 3. คำนวณเส้นที่ 2: Daily Rebalance (Stock)
returns = np.diff(P_t) / P_t[:-1]
daily_rebalance_pnl = fix_c * returns
rebalance_cum = np.insert(np.cumsum(daily_rebalance_pnl), 0, 0)

# 4. คำนวณเส้นที่ 3: LEAPS Option Convexity
# สมมติเราซื้อ Option ให้ครอบคลุมมูลค่าเทียบเท่าการถือหุ้น fix_c (เทียบเท่าจำนวนหุ้น = fix_c / P0)
qty = fix_c / P0 
T_array = np.linspace(days/365, 0.001, days) # เวลาค่อยๆ นับถอยหลังสู่ 0

# หาราคา Option แต่ละวัน
leaps_price = bs_call(P_t, strike, T_array, r, sigma)
leaps_price_0 = bs_call(P0, strike, days/365, r, sigma)

# PnL ของ LEAPS
leaps_pnl = qty * (leaps_price - leaps_price_0)

# แกะค่า EV (Extrinsic Value) ออกมาดู
intrinsic_value = np.maximum(P_t - strike, 0)
ev_array = leaps_price - intrinsic_value
total_ev_value = qty * ev_array

# สร้าง DataFrame
df = pd.DataFrame({
    "Day": np.arange(days),
    "Price": P_t,
    "Benchmark": benchmark,
    "Stock Rebalance": rebalance_cum,
    "LEAPS PnL": leaps_pnl,
    "LEAPS EV (Cost of Convexity)": total_ev_value
})

# --- Plotting ---
st.subheader("📊 เปรียบเทียบผลตอบแทน: Linear vs Convexity")
fig = go.Figure()

# 1. เส้น Benchmark
fig.add_trace(go.Scatter(x=df["Day"], y=df["Benchmark"], mode='lines', 
                         name='1. Benchmark: fix_c * ln(P_t/P_0)', line=dict(color='gray', dash='dash')))

# 2. เส้น Stock Rebalance
fig.add_trace(go.Scatter(x=df["Day"], y=df["Stock Rebalance"], mode='lines', 
                         name='2. Daily Rebalance (Stock)', line=dict(color='blue', width=2)))

# 3. เส้น LEAPS PnL (Convex Function)
fig.add_trace(go.Scatter(x=df["Day"], y=df["LEAPS PnL"], mode='lines', 
                         name='3. LEAPS Convex PnL', line=dict(color='magenta', width=2.5)))

# 4. เส้น EV (ไว้ดูล่างกราฟ)
fig.add_trace(go.Scatter(x=df["Day"], y=df["LEAPS EV (Cost of Convexity)"], mode='lines', 
                         name='Option EV (Theta Decay)', line=dict(color='orange', width=1, dash='dot')))

fig.update_layout(height=550, hovermode="x unified", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# --- Summary Metrics ---
st.divider()
st.subheader("🎯 เจาะลึกผลลัพธ์วันสุดท้าย (Expiry Day)")
col1, col2, col3 = st.columns(3)

col1.metric("Stock Rebalance PnL", f"{rebalance_cum[-1]:,.2f} ฿", 
            delta=f"On Top: {rebalance_cum[-1] - benchmark[-1]:,.2f}", delta_color="normal")

# ค้นหาว่า LEAPS ชนะ Stock หรือไม่
leaps_vs_stock = leaps_pnl[-1] - rebalance_cum[-1]
col2.metric("LEAPS PnL", f"{leaps_pnl[-1]:,.2f} ฿", 
            delta=f"vs Stock: {leaps_vs_stock:,.2f}", delta_color="normal" if leaps_vs_stock > 0 else "inverse")

col3.metric("Initial EV Paid (ต้นทุน Convexity)", f"{-total_ev_value[0]:,.2f} ฿", "ระเหยกลายเป็น 0 เมื่อหมดอายุ")
