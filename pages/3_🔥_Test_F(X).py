import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math

st.set_page_config(page_title="Anti-Fragile: LEAPS + Put Hedge", layout="wide")

st.title("🛡️ Anti-Fragile Simulator: การ Scale Up อย่างไร้รอยขีดข่วน")
st.markdown("""
จำลองสภาวะ **วิกฤตตลาด (Market Crash)** เพื่อทดสอบกลยุทธ์:
1. **Stock Rebalance (100%):** พังพินาศจากการรับมีด
2. **LEAPS 80/20 (Unhedged):** ล็อกขาดทุนชนพื้น (Max Loss Capped)
3. **LEAPS 80/20 + Put Hedge:** เอา Volatility Premium ไปซื้อ OTM Puts ทุกเดือน **(ยิ่งตลาดยับ ยิ่งรวย!)**
""")

# --- Custom Normal CDF (No Scipy) ---
def norm_cdf(x):
    vectorized_erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + vectorized_erf(x / np.sqrt(2.0)))

# --- Black-Scholes Formulas ---
def bs_call(S, K, T, r, sigma):
    T = np.maximum(T, 1e-5)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return (S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2))

def bs_put(S, K, T, r, sigma):
    T = np.maximum(T, 1e-5)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    # Put formula: K*e^(-rT)*N(-d2) - S*N(-d1)
    # N(-x) = 1 - N(x)
    return (K * np.exp(-r * T) * (1.0 - norm_cdf(d2)) - S * (1.0 - norm_cdf(d1)))

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ ตั้งค่าพารามิเตอร์เงินทุน")
fix_c = st.sidebar.number_input("Total Capital (fix_c)", value=1000000, step=100000)
P0 = st.sidebar.number_input("Initial Stock Price", value=100.0, step=10.0)

st.sidebar.subheader("🛡️ Self-Funding Put Hedge")
put_budget_pct = st.sidebar.slider("งบซื้อ Put ต่อเดือน (% ของพอร์ต)", min_value=0.1, max_value=2.0, value=0.5, step=0.1) / 100
st.sidebar.info(f"จ่ายค่าประกัน (Put Premium): {fix_c * put_budget_pct:,.0f} ฿ / เดือน\n(หักจาก Liquidity Pool)")

st.sidebar.subheader("🚨 Market Crash Event")
crash_day = st.sidebar.slider("เกิดวิกฤตในวันที่", min_value=50, max_value=300, value=180)
crash_magnitude = st.sidebar.slider("หุ้นร่วงหนัก (%)", min_value=10, max_value=80, value=40, step=5)

st.sidebar.subheader("Market Specs")
sigma = st.sidebar.slider("Volatility (Sigma)", min_value=0.1, max_value=1.0, value=0.3, step=0.05)
r = 0.03 # Risk-free rate
days = 365
leaps_weight = 0.8
cash_weight = 0.2

if st.sidebar.button("🎲 รัน Simulator ใหม่"):
    st.rerun()

# --- Simulation Logic ---
dt = 1 / 252 
np.random.seed()
Z = np.random.normal(0, 1, days)
P_t = np.zeros(days)
P_t[0] = P0
drift = 0.05

# 1. Price Path + Crash
for t in range(1, days):
    if t == crash_day:
        P_t[t] = P_t[t-1] * (1 - (crash_magnitude / 100))
    else:
        P_t[t] = P_t[t-1] * np.exp((drift - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t])

# 2. Stock Rebalance
ret_stock = np.diff(P_t) / P_t[:-1]
daily_pnl_stock = fix_c * ret_stock
cum_pnl_stock = np.insert(np.cumsum(daily_pnl_stock), 0, 0)

# 3. LEAPS 80/20 (Unhedged)
T_array_leaps = np.linspace(days/365, 0.001, days)
strike_leaps = P0 * 0.8 # ITM LEAPS
C_t = bs_call(P_t, strike_leaps, T_array_leaps, r, sigma)
ret_opt = np.diff(C_t) / np.maximum(C_t[:-1], 1e-8)
daily_pnl_leaps = (leaps_weight * fix_c) * ret_opt
daily_interest = (cash_weight * fix_c) * (np.exp(r * dt) - 1)
cum_pnl_leaps = np.insert(np.cumsum(daily_pnl_leaps + daily_interest), 0, 0)

# 4. Self-Funding Put Hedge Logic
put_pnl_daily = np.zeros(days)
days_to_expiry_put = 0
current_put_qty = 0
current_put_K = 0

for t in range(days):
    if days_to_expiry_put <= 0 or t == 0:
        # Roll Put Option ทุกเดือน (21 วันทำการ)
        current_put_K = P_t[t] * 0.90 # ซื้อ Put 10% OTM
        days_to_expiry_put = 21 
        
        # คำนวณราคา Put ปัจจุบัน
        put_price = bs_put(P_t[t], current_put_K, days_to_expiry_put/252, r, sigma)
        
        # ใช้งบ (Budget) ที่กำหนดไปซื้อ Put
        budget = fix_c * put_budget_pct
        current_put_qty = budget / put_price if put_price > 0 else 0
        
        # บันทึกค่าใช้จ่ายลงใน PnL (จ่ายค่าเบี้ยประกัน)
        put_pnl_daily[t] = -budget 
    else:
        # คำนวณส่วนต่างราคา Put รายวัน
        put_price_prev = bs_put(P_t[t-1], current_put_K, (days_to_expiry_put+1)/252, r, sigma)
        put_price_curr = bs_put(P_t[t], current_put_K, days_to_expiry_put/252, r, sigma)
        
        # กำไร/ขาดทุนรายวันจาก Put
        put_pnl_daily[t] = current_put_qty * (put_price_curr - put_price_prev)
    
    days_to_expiry_put -= 1

cum_pnl_put = np.cumsum(put_pnl_daily)

# 5. Anti-Fragile Portfolio (LEAPS 80/20 + Put Hedge)
cum_pnl_antifragile = cum_pnl_leaps + cum_pnl_put

df = pd.DataFrame({
    "Day": np.arange(days),
    "Price": P_t,
    "Stock Rebalance": cum_pnl_stock,
    "LEAPS Unhedged": cum_pnl_leaps,
    "Anti-Fragile": cum_pnl_antifragile,
    "Put Hedge Value": cum_pnl_put
})

# --- Plotting ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📉 พฤติกรรมราคา (Market Crash)")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df["Day"], y=df["Price"], mode='lines', name='Stock Price', line=dict(color='black')))
    fig_price.add_vline(x=crash_day, line_width=2, line_dash="dash", line_color="red", annotation_text="💥 BLACK SWAN")
    fig_price.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_price, use_container_width=True)

with col2:
    st.subheader("📊 เปรียบเทียบ PnL: จากพังพินาศ สู่การทำกำไรขาลง")
    fig_pnl = go.Figure()

    fig_pnl.add_trace(go.Scatter(x=df["Day"], y=df["Stock Rebalance"], mode='lines', 
                             name='1. Stock Rebalance (รับมีดตก)', line=dict(color='blue', width=2, dash='dot')))

    fig_pnl.add_trace(go.Scatter(x=df["Day"], y=df["LEAPS Unhedged"], mode='lines', 
                             name='2. LEAPS 80/20 (Flatline พื้น)', line=dict(color='orange', width=2)))

    # เส้นไม้ตาย: Anti-Fragile
    fig_pnl.add_trace(go.Scatter(x=df["Day"], y=df["Anti-Fragile"], mode='lines', 
                             name='3. Anti-Fragile (LEAPS + Put)', line=dict(color='green', width=3.5)))

    fig_pnl.add_vline(x=crash_day, line_width=2, line_dash="dash", line_color="red")
    fig_pnl.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified")
    st.plotly_chart(fig_pnl, use_container_width=True)

# --- Summary Metrics ---
st.divider()
st.subheader("🎯 สรุปผลลัพธ์หลังวิกฤต (วันสุดท้าย)")
m1, m2, m3 = st.columns(3)

m1.metric("Stock Rebalance (ไม่ป้องกัน)", f"{cum_pnl_stock[-1]:,.2f} ฿", "ถูกลากลงเหว")
m2.metric("LEAPS 80/20 (ป้องกันระดับแรก)", f"{cum_pnl_leaps[-1]:,.2f} ฿", "หยุดเลือดไหล (Flatline)")

anti_fragile_edge = cum_pnl_antifragile[-1] - cum_pnl_leaps[-1]
m3.metric("Anti-Fragile (ป้องกันระดับสูงสุด)", f"{cum_pnl_antifragile[-1]:,.2f} ฿", 
          delta=f"กำไรจาก Put ล้วนๆ: +{anti_fragile_edge:,.2f} ฿", delta_color="normal")
