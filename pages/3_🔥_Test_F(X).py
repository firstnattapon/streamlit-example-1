import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Dragon Portfolio Simulator (Retail Edition)", layout="wide")

# --- ส่วนหัว ---
st.title("🐉 Dragon Portfolio Simulator (ฉบับรายย่อย)")
st.markdown("""
แบบจำลองพอร์ต **Dragon Portfolio** สูตรรายย่อย **20/20/20/20/20** โดยเน้นดูการทำงานของ **เงินสด (Cash)** และ **Long Volatility** ในช่วงวิกฤตที่จำลองขึ้น
""")

# --- Sidebar: ตั้งค่าพารามิเตอร์ ---
st.sidebar.header("⚙️ ตั้งค่าการจำลอง")
initial_investment = st.sidebar.number_input("เงินลงทุนเริ่มต้น (บาท)", value=1_000_000, step=100_000)
years = st.sidebar.slider("ระยะเวลาจำลอง (ปี)", 5, 20, 10)
volatility_impact = st.sidebar.slider("ความรุนแรงของวิกฤต (Crisis Severity)", 1, 10, 8, help="ยิ่งมาก หุ้นยิ่งตกหนัก และ Long Vol ยิ่งพุ่งแรง")

st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ สัดส่วนพอร์ต (Target Weights)")
w_equity = st.sidebar.slider("1. หุ้น (Equities)", 0, 100, 20) / 100
w_gold = st.sidebar.slider("2. ทองคำ (Gold)", 0, 100, 20) / 100
w_commodity = st.sidebar.slider("3. สินค้าโภคภัณฑ์ (Trend)", 0, 100, 20) / 100
w_longvol = st.sidebar.slider("4. Long Volatility (Options)", 0, 100, 20) / 100
w_cash = st.sidebar.slider("5. เงินสด (Cash/Bills)", 0, 100, 20) / 100

total_weight = w_equity + w_gold + w_commodity + w_longvol + w_cash
if abs(total_weight - 1.0) > 0.01:
    st.sidebar.warning(f"⚠️ ผลรวมสัดส่วนปัจจุบันคือ {total_weight*100:.0f}% (ควรเป็น 100%)")

# --- ฟังก์ชันสร้างข้อมูลจำลอง (Synthetic Data Generator) ---
def generate_market_data(years, crisis_severity):
    np.random.seed(42)
    days = years * 252
    dates = pd.date_range(start="2024-01-01", periods=days, freq="B")
    
    # 1. หุ้น: Random Walk with Drift (ขึ้นเรื่อยๆ แต่ผันผวน)
    equity_returns = np.random.normal(0.0004, 0.01, days) # Daily return ~0.04%
    
    # 2. ทองคำ: Uncorrelated, ป้องกันเงินเฟ้อ
    gold_returns = np.random.normal(0.0002, 0.008, days) + (np.random.normal(0, 0.005, days) * 0.3)
    
    # 3. สินค้าโภคภัณฑ์ (Trend): ผันผวนสูง
    comm_returns = np.random.normal(0.0002, 0.012, days)
    
    # 4. Long Volatility: Bleed (ขาดทุนวันละนิด) แต่ Spike ตอนวิกฤต
    # Cost of carry ~ -5% to -10% per year
    lvol_returns = np.random.normal(-0.0004, 0.015, days) 
    
    # 5. เงินสด: ได้ดอกเบี้ยคงที่ (สมมติ 3% ต่อปี)
    cash_rate = 0.03 / 252
    cash_returns = np.full(days, cash_rate)
    
    # --- สร้างวิกฤต (Crisis Event) ตรงกลางช่วงเวลา ---
    crisis_start = int(days * 0.5)
    crisis_duration = 60 # 2-3 เดือน
    
    for i in range(crisis_start, crisis_start + crisis_duration):
        # หุ้นตกหนัก
        equity_returns[i] -= (0.005 * crisis_severity) 
        # ทองคำผันผวนแต่ยืนได้
        gold_returns[i] += 0.001 
        # Commodity ร่วงตาม Demand หาย (หรือพุ่งถ้าเป็น Supply shock - ในที่นี้จำลอง Deflationary crash)
        comm_returns[i] -= 0.002
        # Long Vol พุ่งทะยาน (Convexity)
        lvol_returns[i] += (0.015 * crisis_severity) # พุ่งแรงมาก
        # เงินสดคงที่ (Safe Haven)
    
    # สร้าง DataFrame
    data = pd.DataFrame({
        'Equities': equity_returns,
        'Gold': gold_returns,
        'Commodities': comm_returns,
        'Long Volatility': lvol_returns,
        'Cash': cash_returns
    }, index=dates)
    
    # แปลงเป็น Cumulative Returns (ราคา)
    prices = (1 + data).cumprod()
    return prices, data

# --- ฟังก์ชัน Backtest แบบ Rebalance ---
def backtest_portfolio(prices_df, weights, initial_fund, rebalance_freq='M'):
    # weights: dict เช่น {'Equities': 0.2, ...}
    
    # Normalize weights ในกรณีที่ผู้ใช้ใส่มาไม่ครบ 100%
    w_sum = sum(weights.values())
    if w_sum == 0:
        return pd.DataFrame()
    norm_weights = {k: v/w_sum for k, v in weights.items()}
    
    portfolio_value = []
    
    # เริ่มต้น: แบ่งเงินตามสัดส่วน
    holdings = {k: initial_fund * w for k, w in norm_weights.items()}
    
    # แปลงความถี่ Rebalance
    rebalance_days = []
    if rebalance_freq == 'M': # Monthly
        rebalance_days = prices_df.resample('ME').last().index
    elif rebalance_freq == 'Q': # Quarterly
        rebalance_days = prices_df.resample('QE').last().index
    
    asset_names = list(weights.keys())
    # สร้าง Normalized Price (เริ่มที่ 1.0)
    norm_prices = prices_df / prices_df.iloc[0]
    
    # เก็บข้อมูลรายวัน
    portfolio_history = pd.DataFrame(index=prices_df.index, columns=['Total Value'] + asset_names)
    
    # ตัวแปรจำลองจำนวนหน่วย (Units)
    units = {k: holdings[k] / norm_prices.iloc[0][k] for k in asset_names}
    
    for i, date in enumerate(prices_df.index):
        # 1. คำนวณมูลค่าปัจจุบัน
        daily_val = 0
        asset_vals = {}
        for k in asset_names:
            val = units[k] * norm_prices.iloc[i][k]
            asset_vals[k] = val
            daily_val += val
            
        portfolio_history.iloc[i] = [daily_val] + [asset_vals[k] for k in asset_names]
        
        # 2. ตรวจสอบว่าต้อง Rebalance หรือไม่
        is_rebalance_day = date in rebalance_days
        
        if is_rebalance_day:
            # ขายตัวที่กำไร ซื้อตัวที่ขาดทุน ให้กลับมาเท่าเดิม
            for k in asset_names:
                target_val = daily_val * norm_weights[k]
                price_now = norm_prices.iloc[i][k]
                units[k] = target_val / price_now # ปรับจำนวนหน่วยใหม่
                
    return portfolio_history

# --- Main App Logic ---

# 1. Generate Data
prices, returns = generate_market_data(years, volatility_impact)

# 2. Run Backtest
target_weights = {
    'Equities': w_equity,
    'Gold': w_gold,
    'Commodities': w_commodity,
    'Long Volatility': w_longvol,
    'Cash': w_cash
}

# พอร์ต Dragon (Rebalance ทุกเดือน)
port_dragon = backtest_portfolio(prices, target_weights, initial_investment, 'M')

# พอร์ต Benchmark (ถือหุ้นล้วน 100%)
bench_weights = {'Equities': 1.0, 'Gold': 0, 'Commodities': 0, 'Long Volatility': 0, 'Cash': 0}
port_bench = backtest_portfolio(prices, bench_weights, initial_investment, 'N') # N = No Rebalance

# --- แสดงผล ---

if not port_dragon.empty and not port_bench.empty:
    # Metrics สำคัญ
    final_dragon = port_dragon['Total Value'].iloc[-1]
    final_bench = port_bench['Total Value'].iloc[-1]
    return_dragon = (final_dragon - initial_investment) / initial_investment * 100
    return_bench = (final_bench - initial_investment) / initial_investment * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("มูลค่าพอร์ตสุดท้าย (Dragon)", f"{final_dragon:,.0f} บาท", f"{return_dragon:+.2f}%")
    col2.metric("เทียบกับถือหุ้นล้วน (Benchmark)", f"{final_bench:,.0f} บาท", f"{return_bench:+.2f}%")
    
    # Calculate Drawdown for Dragon
    rolling_max = port_dragon['Total Value'].cummax()
    drawdown = (port_dragon['Total Value'] - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100
    col3.metric("Max Drawdown (Dragon)", f"{max_dd:.2f}%", "ความเสี่ยงต่ำกว่า")

    # Chart 1: เปรียบเทียบมูลค่าพอร์ต
    st.subheader("📈 ผลการดำเนินงานเทียบกับตลาด (Simulated)")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=port_dragon.index, y=port_dragon['Total Value'], mode='lines', name='Dragon Portfolio (20x5)', line=dict(color='#00CC96', width=3)))
    fig1.add_trace(go.Scatter(x=port_bench.index, y=port_bench['Total Value'], mode='lines', name='Stock Market (100% Equities)', line=dict(color='gray', dash='dot')))

    # Highlight Crisis Zone
    crisis_start_idx = int(len(port_dragon)*0.5)
    crisis_end_idx = crisis_start_idx + 60
    if crisis_end_idx < len(port_dragon):
        crisis_start_date = port_dragon.index[crisis_start_idx]
        crisis_end_date = port_dragon.index[crisis_end_idx]
        fig1.add_vrect(x0=crisis_start_date, x1=crisis_end_date, fillcolor="red", opacity=0.1, annotation_text="Market Crash Scenario", annotation_position="top left")

    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2: ดูไส้ในพอร์ต (Asset Allocation Area Chart)
    st.subheader("🎨 สัดส่วนสินทรัพย์ตามเวลา (The Anatomy of the Dragon)")
    st.write("สังเกตช่วงวิกฤต (แถบสีแดง): พื้นที่สีม่วง (Long Vol) จะขยายใหญ่ขึ้น และระบบจะขายมันออกเพื่อไปซื้อพื้นที่สีฟ้า (หุ้น) ที่หดเล็กลง")

    stack_data = port_dragon.drop(columns=['Total Value'])
    fig2 = px.area(stack_data, x=stack_data.index, y=stack_data.columns, 
                   color_discrete_map={
                       'Equities': '#636EFA', 
                       'Gold': '#EF553B', 
                       'Commodities': '#FFA15A', 
                       'Long Volatility': '#AB63FA', 
                       'Cash': '#00CC96'
                   })
    fig2.update_layout(yaxis_title="มูลค่าสินทรัพย์ (บาท)")
    st.plotly_chart(fig2, use_container_width=True)

    # Chart 3: ประสิทธิภาพของแต่ละสินทรัพย์ (Normalized Prices)
    st.subheader("📊 พฤติกรรมของสินทรัพย์แต่ละตัว")
    norm_prices = prices / prices.iloc[0]
    fig3 = px.line(norm_prices, x=norm_prices.index, y=norm_prices.columns)
    st.plotly_chart(fig3, use_container_width=True)

    # คำอธิบาย
    st.info("""
    **💡 สิ่งที่คุณจะเห็นจากแบบจำลองนี้:**
    1. **ช่วงปกติ:** เส้นสีม่วง (Long Vol) จะค่อยๆ ลดลง (Bleed) ดึงผลตอบแทนรวมลงเล็กน้อย นี่คือ "ค่าเบี้ยประกัน" ที่ต้องจ่าย
    2. **ช่วงวิกฤต (แถบแดง):** หุ้น (น้ำเงิน) และสินค้าโภคภัณฑ์ (ส้ม) ตกหนัก แต่ Long Vol (ม่วง) จะพุ่งสวนขึ้นมารุนแรงมาก
    3. **การทำงานของ Cash (เขียว):** เส้นสีเขียวจะค่อยๆ ขึ้นอย่างมั่นคง เป็นฐานให้พอร์ต และเป็นกระสุนดินดำ (Dry Powder) รอช้อนซื้อของถูก
    4. **Rebalancing Magic:** ถ้าคุณสังเกตกราฟพื้นที่ (Area Chart) หลังจบวิกฤต สัดส่วนหุ้นจะกลับมาเท่าเดิมได้เร็ว เพราะเราขายกำไรจาก Long Vol ไปซื้อหุ้นตอนราคาถูกที่สุด
    """)
else:
    st.error("เกิดข้อผิดพลาดในการคำนวณ กรุณาตรวจสอบสัดส่วนการลงทุน")
