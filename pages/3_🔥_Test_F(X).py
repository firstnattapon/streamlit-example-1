import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math

st.set_page_config(page_title="Market Crash: LEAPS vs Stock Rebalance", layout="wide")

st.title("📉 Market Crash Simulator: ทำไม LEAPS ถึงเป็นร่มชูชีพชั้นดี")
st.markdown("""
จำลองสภาวะ **"ตลาดพังทลาย" (Severe Drawdown)** เพื่อเปรียบเทียบ:
1. **Benchmark:** $fix\_c \cdot \ln(P_t / P_0)$
2. **Stock Rebalance:** การพยายามซื้อถัวเพื่อรักษาสมดุล $fix\_c$ (รับมีดตก)
3. **LEAPS Convexity:** การใช้ Option ที่ล็อกความเสี่ยงขาลงไว้แค่ค่า Premium
""")

# --- Custom Normal CDF (ไม่ใช้ scipy) ---
def norm_cdf(x):
    vectorized_erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + vectorized_erf(x / np.sqrt(2.0)))

# --- Black-Scholes Formula ---
def bs_call(S, K, T, r, sigma):
    T = np.maximum(T, 1e-5)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return (S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2))

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ ตั้งค่าพารามิเตอร์")
fix_c = st.sidebar.number_input("Fixed Capital (fix_c)", value=100000, step=10000)
P0 = st.sidebar.number_input("Initial Price (P0)", value=100.0, step=10.0)

st.sidebar.subheader("LEAPS Option Specs")
strike = st.sidebar.number_input("Strike Price (K)", value=80.0, step=5.0)
r = st.sidebar.number_input("Risk-free Rate (r)", value=0.02, step=0.01)

st.sidebar.subheader("Market Dynamics")
sigma = st.sidebar.slider("Volatility (Sigma)", min_value=0.1, max_value=1.0, value=0.4, step=0.05)
days = st.sidebar.slider("Days to Expiry (T)", min_value=100, max_value=730, value=365, step=30)

st.sidebar.markdown("---")
st.sidebar.subheader("🚨 Market Crash Event")
simulate_crash = st.sidebar.checkbox("เปิดโหมดจำลองวิกฤตตลาด", value=True)
crash_day = st.sidebar.slider("เกิดวิกฤตในวันที่", min_value=10, max_value=days-10, value=days//2)
crash_magnitude = st.sidebar.slider("ความรุนแรง (หุ้นร่วง %)", min_value=10, max_value=80, value=40, step=5)

if st.sidebar.button("🎲 สุ่มกราฟราคาใหม่"):
    st.rerun()

# --- Simulation Logic ---
dt = 1 / 252 
np.random.seed()
Z = np.random.normal(0, 1, days)
P_t = np.zeros(days)
P_t[0] = P0
drift = 0.05

# 1. สร้างเส้นทางราคา + แทรก Market Crash
for t in range(1, days):
    if simulate_crash and t == crash_day:
        # วันที่เกิดวิกฤต หุ้นร่วงหนักทันที
        P_t[t] = P_t[t-1] * (1 - (crash_magnitude / 100))
    else:
        # วันปกติ วิ่งตาม GBM
        P_t[t] = P_t[t-1] * np.exp((drift - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t])

# 2. Benchmark
benchmark = fix_c * np.log(P_t / P0)

# 3. Stock Rebalance
returns = np.diff(P_t) / P_t[:-1]
daily_rebalance_pnl = fix_c * returns
rebalance_cum = np.insert(np.cumsum(daily_rebalance_pnl), 0, 0)

# 4. LEAPS Convexity
qty = fix_c / P0 
T_array = np.linspace(days/365, 0.001, days)
leaps_price = bs_call(P_t, strike, T_array, r, sigma)
leaps_price_0 = bs_call(P0, strike, days/365, r, sigma)

# Max Loss ของ LEAPS คือเสียค่า Premium ทั้งหมดที่จ่ายไปตอนแรก (qty * leaps_price_0)
leaps_pnl = qty * (leaps_price - leaps_price_0)

df = pd.DataFrame({
    "Day": np.arange(days),
    "Price": P_t,
    "Benchmark": benchmark,
    "Stock Rebalance": rebalance_cum,
    "LEAPS PnL": leaps_pnl
})

# --- Plotting ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📉 ราคาสินทรัพย์ (Asset Price)")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df["Day"], y=df["Price"], mode='lines', name='Price', line=dict(color='black')))
    if simulate_crash:
        fig_price.add_vline(x=crash_day, line_width=2, line_dash="dash", line_color="red", annotation_text="💥 CRASH!")
    fig_price.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_price, use_container_width=True)

with col2:
    st.subheader("📊 เปรียบเทียบ PnL ขาลง (Downside Protection)")
    fig_pnl = go.Figure()

    fig_pnl.add_trace(go.Scatter(x=df["Day"], y=df["Benchmark"], mode='lines', 
                             name='1. Benchmark', line=dict(color='gray', dash='dash')))

    fig_pnl.add_trace(go.Scatter(x=df["Day"], y=df["Stock Rebalance"], mode='lines', 
                             name='2. Stock Rebalance (รับมีดตก)', line=dict(color='blue', width=2)))

    fig_pnl.add_trace(go.Scatter(x=df["Day"], y=df["LEAPS PnL"], mode='lines', 
                             name='3. LEAPS (ล็อกขาดทุนสูงสุด)', line=dict(color='magenta', width=3)))

    if simulate_crash:
        fig_pnl.add_vline(x=crash_day, line_width=2, line_dash="dash", line_color="red")
        
    # เพิ่มเส้นสมมติ Max Loss ของ LEAPS
    max_loss = -(qty * leaps_price_0)
    fig_pnl.add_hline(y=max_loss, line_width=1, line_dash="dot", line_color="magenta", annotation_text="Max Loss Capped")

    fig_pnl.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified")
    st.plotly_chart(fig_pnl, use_container_width=True)

# --- Summary Metrics ---
st.divider()
st.subheader("บทสรุปความเสียหายหลังผ่านวิกฤต (วันสุดท้าย)")
m1, m2, m3 = st.columns(3)

m1.metric("Stock Rebalance PnL", f"{rebalance_cum[-1]:,.2f} ฿", "ถูกลากลงจากการพยายามซื้อถัว (Buy the dip)")

leaps_advantage = leaps_pnl[-1] - rebalance_cum[-1]
m2.metric("LEAPS PnL", f"{leaps_pnl[-1]:,.2f} ฿", 
          delta=f"ขาดทุนน้อยกว่า Stock: +{leaps_advantage:,.2f} ฿", delta_color="normal")

m3.metric("LEAPS Max Risk (Premium Paid)", f"{-qty * leaps_price_0:,.2f} ฿", "นี่คือจุดที่แย่ที่สุดที่จะเกิดขึ้นได้")
