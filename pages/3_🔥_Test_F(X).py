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
3. **LEAPS Rebalance (80/20):** รักษามูลค่า LEAPS ไว้ที่ 80% และถือเงินสด 20%
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

for t in range(1, days):
    P_t[t] = P_t[t-1] * np.exp((drift - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t])

benchmark = fix_c * np.log(P_t / P0)

ret_stock = np.diff(P_t) / P_t[:-1]
daily_pnl_stock = fix_c * ret_stock
cum_pnl_stock = np.insert(np.cumsum(daily_pnl_stock), 0, 0)

T_array = np.linspace(days/365, 0.001, days)
C_t = bs_call(P_t, strike, T_array, r, sigma)
ret_opt = np.diff(C_t) / np.maximum(C_t[:-1], 1e-8)

daily_pnl_leaps = (leaps_weight * fix_c) * ret_opt
daily_interest = (cash_weight * fix_c) * (np.exp(r * dt) - 1)
cum_pnl_leaps = np.insert(np.cumsum(daily_pnl_leaps + daily_interest), 0, 0)

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
    st.subheader("📉 ราคาสินทรัพย์เทียบกับเวลา")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df["Day"], y=df["Price"], mode='lines', name='Stock Price', line=dict(color='black')))
    fig_price.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_price, use_container_width=True)

with col2:
    st.subheader("📊 กระแสเงินสดสะสมเทียบกับเวลา")
    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(x=df["Day"], y=df["Benchmark"], mode='lines', name='1. Benchmark', line=dict(color='gray', dash='dash')))
    fig_pnl.add_trace(go.Scatter(x=df["Day"], y=df["Stock Rebalance"], mode='lines', name='2. Stock Rebalance', line=dict(color='blue', width=2)))
    fig_pnl.add_trace(go.Scatter(x=df["Day"], y=df["LEAPS Rebalance"], mode='lines', name='3. LEAPS Rebalance', line=dict(color='magenta', width=3)))
    fig_pnl.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified")
    st.plotly_chart(fig_pnl, use_container_width=True)

st.divider()

# แถวที่ 2: กราฟ X=Price, Y=Cashflow (Payoff Profile - Path Version)
st.subheader("📍 Trading Path: ราคา (X) vs กระแสเงินสดสะสม (Y)")
fig_payoff = go.Figure()

# เปลี่ยน mode เป็น 'lines' เพื่อลากเส้นต่อเนื่องตามลำดับเวลา
fig_payoff.add_trace(go.Scatter(x=df["Price"], y=df["Benchmark"], mode='lines', 
                                line=dict(color='gray', width=1.5, dash='dot'), name='1. Benchmark Path'))

fig_payoff.add_trace(go.Scatter(x=df["Price"], y=df["Stock Rebalance"], mode='lines', 
                                line=dict(color='blue', width=1.5), opacity=0.7, name='2. Stock Path'))

fig_payoff.add_trace(go.Scatter(x=df["Price"], y=df["LEAPS Rebalance"], mode='lines', 
                                line=dict(color='magenta', width=2), opacity=0.9, name='3. LEAPS Path'))

# เพิ่มเส้น Zero Line (ทุน = 0)
fig_payoff.add_hline(y=0, line_width=1, line_dash="solid", line_color="black", opacity=0.3)

fig_payoff.update_layout(
    xaxis_title="Stock Price (P_t)",
    yaxis_title="Cumulative PnL (Cashflow)",
    height=500,
    hovermode="closest",
    template="plotly_white"
)
st.plotly_chart(fig_payoff, use_container_width=True)

# --- Summary Metrics ---
st.divider()
st.subheader("🎯 สรุปผลงานระบบเมื่อจบรอบ (Expiry Day)")
m1, m2, m3 = st.columns(3)
m1.metric("Stock Rebalance PnL", f"{cum_pnl_stock[-1]:,.2f} ฿", delta=f"On Top Benchmark: {cum_pnl_stock[-1] - benchmark[-1]:,.2f} ฿", delta_color="normal")
leaps_vs_stock = cum_pnl_leaps[-1] - cum_pnl_stock[-1]
m2.metric(f"LEAPS {leaps_weight*100:.0f}/{cash_weight*100:.0f} PnL", f"{cum_pnl_leaps[-1]:,.2f} ฿", delta=f"vs Stock: {leaps_vs_stock:,.2f} ฿", delta_color="normal" if leaps_vs_stock > 0 else "inverse")
m3.metric("Total EV Decay (Theta Cost)", f"{-((ev[0] - ev[-1])/C_t[0]) * (leaps_weight * fix_c):,.2f} ฿")
