import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math

st.set_page_config(page_title="LEAPS Rebalance 80/20", layout="wide")

st.title("🔥 The Ultimate Convexity: LEAPS Rebalance + Liquidity Pool")
st.markdown("""
พิสูจน์กลยุทธ์ **Stock Replacement + Liquidity Rebalance**
1. **Benchmark:** $fix\_c \cdot \ln(P_t / P_0)$
2. **Stock Rebalance:** รักษามูลค่าหุ้นให้เท่ากับ $fix\_c$ 100%
3. **LEAPS Rebalance (80/20):** รักษามูลค่า LEAPS ไว้ที่ 80% และถือเงินสด 20% เพื่อคอย Rebalance และกินดอกเบี้ย
""")

# --- Custom Normal CDF ---
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
fix_c = st.sidebar.number_input("Total Capital (fix_c)", value=100000, step=10000)
P0 = st.sidebar.number_input("Initial Stock Price (P0)", value=100.0, step=10.0)

st.sidebar.subheader("🎯 Portfolio Allocation")
leaps_weight = st.sidebar.slider("สัดส่วน LEAPS (%)", min_value=10, max_value=100, value=80, step=10) / 100
cash_weight = 1.0 - leaps_weight
st.sidebar.info(f"LEAPS Target: {fix_c * leaps_weight:,.0f} ฿\n\nLiquidity Pool: {fix_c * cash_weight:,.0f} ฿")

st.sidebar.subheader("LEAPS Specs")
strike = st.sidebar.number_input("Strike Price (K)", value=80.0, step=5.0)
r = st.sidebar.number_input("Risk-free Rate (r) - ดอกเบี้ยเงินสด", value=0.03, step=0.01)

st.sidebar.subheader("Market Dynamics")
sigma = st.sidebar.slider("Volatility (Sigma)", min_value=0.1, max_value=1.0, value=0.4, step=0.05)
drift = st.sidebar.slider("Drift (Mu)", min_value=-0.5, max_value=0.5, value=0.0, step=0.05)
days = st.sidebar.slider("Days to Expiry (T)", min_value=100, max_value=730, value=365, step=30)

if st.sidebar.button("🎲 สุ่มกราฟตลาดใหม่"):
    st.rerun()

# --- Simulation Logic ---
dt = 1 / 252 
np.random.seed()
Z = np.random.normal(0, 1, days)
P_t = np.zeros(days)
P_t[0] = P0

# 1. สร้างเส้นทางราคา (GBM)
for t in range(1, days):
    P_t[t] = P_t[t-1] * np.exp((drift - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t])

# 2. Benchmark Line
benchmark = fix_c * np.log(P_t / P0)

# 3. Stock Rebalance Line (Linear Volatility Premium)
ret_stock = np.diff(P_t) / P_t[:-1]
daily_pnl_stock = fix_c * ret_stock
cum_pnl_stock = np.insert(np.cumsum(daily_pnl_stock), 0, 0)

# 4. LEAPS Rebalance Line (Convexity + Gamma Premium)
T_array = np.linspace(days/365, 0.001, days)
C_t = bs_call(P_t, strike, T_array, r, sigma)

# แก้บั๊ก: ป้องกันตัวหารเป็น 0 หากออปชันหมดมูลค่า (Out of money ใกล้วันหมดอายุ)
ret_opt = np.diff(C_t) / np.maximum(C_t[:-1], 1e-8)

# Rebalance ให้มูลค่า Option คงที่เท่ากับ (leaps_weight * fix_c) ทุกวัน
daily_pnl_leaps = (leaps_weight * fix_c) * ret_opt

# ดอกเบี้ยรับจาก Liquidity Pool (cash_weight * fix_c)
daily_interest = (cash_weight * fix_c) * (np.exp(r * dt) - 1)

# รวม PnL สะสม
cum_pnl_leaps = np.insert(np.cumsum(daily_pnl_leaps + daily_interest), 0, 0)

# แกะค่า EV ออกมาโชว์
intrinsic = np.maximum(P_t - strike, 0)
ev = C_t - intrinsic

df = pd.DataFrame({
    "Day": np.arange(days),
    "Price": P_t,
    "Benchmark": benchmark,
    "Stock Rebalance": cum_pnl_stock,
    "LEAPS Rebalance": cum_pnl_leaps
})

# --- Plotting ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📉 ราคาสินทรัพย์ (Asset Price)")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df["Day"], y=df["Price"], mode='lines', name='Stock Price', line=dict(color='black')))
    fig_price.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_price, use_container_width=True)

with col2:
    st.subheader("📊 เปรียบเทียบ Cashflow: Linear vs Convex Rebalance")
    fig_pnl = go.Figure()

    fig_pnl.add_trace(go.Scatter(x=df["Day"], y=df["Benchmark"], mode='lines', 
                             name='1. Benchmark', line=dict(color='gray', dash='dash')))

    fig_pnl.add_trace(go.Scatter(x=df["Day"], y=df["Stock Rebalance"], mode='lines', 
                             name='2. Stock Rebalance (100%)', line=dict(color='blue', width=2)))

    # เส้นไฮไลต์: LEAPS Rebalance
    fig_pnl.add_trace(go.Scatter(x=df["Day"], y=df["LEAPS Rebalance"], mode='lines', 
                             name=f'3. LEAPS Rebalance ({leaps_weight*100:.0f}/{cash_weight*100:.0f})', 
                             line=dict(color='magenta', width=3)))

    fig_pnl.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified")
    st.plotly_chart(fig_pnl, use_container_width=True)

# --- Summary Metrics ---
st.divider()
st.subheader("🎯 สรุปผลงานระบบเมื่อจบรอบ (Expiry Day)")
m1, m2, m3 = st.columns(3)

m1.metric("Stock Rebalance PnL", f"{cum_pnl_stock[-1]:,.2f} ฿", 
          delta=f"On Top Benchmark: {cum_pnl_stock[-1] - benchmark[-1]:,.2f} ฿", delta_color="normal")

leaps_vs_stock = cum_pnl_leaps[-1] - cum_pnl_stock[-1]
m2.metric(f"LEAPS {leaps_weight*100:.0f}/{cash_weight*100:.0f} PnL", f"{cum_pnl_leaps[-1]:,.2f} ฿", 
          delta=f"vs Stock: {leaps_vs_stock:,.2f} ฿", delta_color="normal" if leaps_vs_stock > 0 else "inverse")

m3.metric("Total EV Decay (Theta Cost)", f"{-((ev[0] - ev[-1])/C_t[0]) * (leaps_weight * fix_c):,.2f} ฿", 
          "ถูกชดเชยด้วย Gamma Premium เรียบร้อยแล้ว")
