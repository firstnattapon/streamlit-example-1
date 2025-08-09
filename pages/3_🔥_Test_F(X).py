# 10_🧠_Portfolio_Orchestrator.py
# ---------------------------------------------------------------
# Portfolio Risk-Budgeting layer (PhD-grade but practical):
# - Inverse-Vol weights (+ max weight cap)
# - Convert weights -> per-asset FIX budget (fix_i)
# - Compute per-asset SumUSD_min (action=1 daily) and Ref-Log
# - Aggregate to portfolio; add txn-cost & turnover penalty
# - DD governance (kill-switch), basic KPIs: MaxDD, Vol, CAGR
# ---------------------------------------------------------------

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import List, Dict, Tuple

st.set_page_config(page_title="Portfolio_Orchestrator", page_icon="🧠", layout="wide")

# ------------- Helpers -------------
@st.cache_data(ttl=3600)
def fetch_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    frames = []
    for t in tickers:
        try:
            h = yf.Ticker(t).history(start=start, end=end)[['Close']]
            if h.empty: continue
            if h.index.tz is None:
                h = h.tz_localize('UTC').tz_convert('Asia/Bangkok')
            else:
                h = h.tz_convert('Asia/Bangkok')
            frames.append(h.rename(columns={'Close': t}))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1).ffill().dropna(how='all')
    return df.dropna(axis=1)

def max_drawdown(equity: pd.Series) -> float:
    if equity.empty: return 0.0
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())

def annualized_vol(returns: pd.Series, freq: int = 252) -> float:
    if returns.std(skipna=True) == 0 or returns.dropna().empty:
        return 0.0
    return float(returns.std(ddof=0) * np.sqrt(freq))

def cagr(equity: pd.Series, freq: int = 252) -> float:
    if equity.empty: return 0.0
    n = len(equity)
    if equity.iloc[0] <= 0: return 0.0
    years = n / freq
    return float((equity.iloc[-1] / equity.iloc[0])**(1/years) - 1) if years > 0 else 0.0

def pnl_turnover(weights: pd.DataFrame) -> Tuple[pd.Series, float]:
    if weights.empty: return pd.Series(dtype=float), 0.0
    tw = weights.fillna(0.0)
    tw_lag = tw.shift(1).fillna(0.0)
    turnover_series = (tw - tw_lag).abs().sum(axis=1)
    return turnover_series, float(turnover_series.sum())

def inverse_vol_weights(prices: pd.DataFrame, lookback: int, max_w: float) -> pd.Series:
    # ใช้ข้อมูลล่าสุดเป็นตัวคำนวณน้ำหนักคงที่ (static) เพื่อความเรียบง่าย/เร็ว
    rets = prices.pct_change().dropna()
    if len(rets) < lookback: lookback = len(rets)
    if lookback <= 1:  # fallback: เท่าๆกัน
        w = pd.Series(1.0, index=prices.columns); w = w / w.sum()
        return w.clip(upper=max_w)

    vol = rets.tail(lookback).std()
    vol = vol.replace(0, np.nan)
    inv = 1.0 / vol
    inv = inv.replace([np.inf, -np.inf], np.nan).fillna(0)
    if inv.sum() == 0:
        w = pd.Series(1.0, index=prices.columns); w = w / w.sum()
    else:
        w = inv / inv.sum()
    # cap & renormalize
    w = w.clip(upper=max_w)
    s = w.sum()
    if s == 0:  # all capped to 0 somehow
        w = pd.Series(1.0, index=prices.columns); w = w / w.sum()
    else:
        w = w / s
    return w

# ------------- Core single-asset engine (action=1 baseline) -------------
def sumusd_min_and_refer(prices: pd.Series, fix: float) -> Tuple[pd.Series, pd.Series]:
    """
    ตรรกะเดียวกับแกนเดิม:
    action = 1 ทุกวัน -> rebalance เป็น amount = fix/price
    refer = -fix * ln(P0 / Pt)   (เส้นอ้างอิงแบบ Log) 
    """
    p = prices.to_numpy(dtype=float)
    n = len(p)
    amount = np.empty(n); buffer = np.zeros(n); cash = np.empty(n); sumusd = np.empty(n)
    amount[0] = fix / p[0]; cash[0] = fix; sumusd[0] = cash[0] + amount[0]*p[0]
    for i in range(1, n):
        # rebalance
        buffer[i] = amount[i-1]*p[i] - fix
        cash[i]   = cash[i-1] + buffer[i]
        amount[i] = fix / p[i]
        sumusd[i] = cash[i] + amount[i]*p[i]
    refer = -fix * np.log(p[0] / p)  # เส้นอ้างอิง log
    return pd.Series(sumusd, index=prices.index), pd.Series(refer, index=prices.index)

# ------------- Portfolio orchestration -------------
def build_portfolio(prices: pd.DataFrame,
                    total_budget: float,
                    lookback: int,
                    max_weight: float,
                    txn_cost_bps: float,
                    dd_kill: float) -> Dict[str, any]:
    tickers = list(prices.columns)
    # 1) คำนวณน้ำหนักแบบ inverse-vol (static ณ วันสุดท้าย)
    w = inverse_vol_weights(prices, lookback, max_weight)  # Series
    # 2) แปลงเป็น FIX_i
    fix_map = {t: float(w.get(t, 0.0) * total_budget) for t in tickers}

    # 3) รันสินทรัพย์ละ SumUSD_min + Refer แล้ว “รวม”
    sumusd_cols, refer_cols = [], []
    for t in tickers:
        s_sumusd, s_refer = sumusd_min_and_refer(prices[t].dropna(), fix_map[t])
        sumusd_cols.append(s_sumusd.rename(t))
        refer_cols.append(s_refer.rename(t))
    sumusd_df = pd.concat(sumusd_cols, axis=1).fillna(method='ffill').dropna()
    refer_df  = pd.concat(refer_cols,  axis=1).reindex(sumusd_df.index).fillna(method='ffill')

    # 4) คำนวณน้ำหนักรายวัน (ถือคงที่) เพื่อวัด turnover/txn-cost
    w_df = pd.DataFrame(index=sumusd_df.index, columns=tickers, data=[w.values]*len(sumusd_df))
    turnover_series, turnover_total = pnl_turnover(w_df)
    # txn cost ต่อวัน = sum(|Δw|) * bps * (มูลค่าพอร์ต ณ วันนั้น)
    port_sumusd = sumusd_df.sum(axis=1)
    daily_cost  = turnover_series * (txn_cost_bps/10000.0) * port_sumusd.shift(1).fillna(port_sumusd.iloc[0])
    daily_cost  = daily_cost.fillna(0.0)
    # พอร์ตหลังหักต้นทุน
    port_sumusd_netfee = port_sumusd - daily_cost.cumsum()

    # 5) Ref-Log รวมพอร์ต และ Net
    port_refer = refer_df.sum(axis=1)
    init_cap   = sumusd_df.iloc[0].sum()
    port_net   = port_sumusd_netfee - port_refer - init_cap

    # 6) DD governance: ถ้า DD < -dd_kill ให้ scale-down 50% จากวันถัดไป
    dd = max_drawdown(port_sumusd_netfee)
    if dd_kill > 0 and dd < -abs(dd_kill):
        # scale half จากวัน breach เป็นต้นไปอย่างง่าย
        crest = port_sumusd_netfee.cummax()
        breach = (port_sumusd_netfee / crest - 1.0) <= -abs(dd_kill)
        if breach.any():
            idx0 = breach.idxmax()  # วันแรกที่ breach
            scale = pd.Series(1.0, index=port_sumusd_netfee.index)
            scale.loc[idx0:] = 0.5
            port_sumusd_netfee = (port_sumusd_netfee / port_sumusd_netfee.iloc[0])**scale * port_sumusd_netfee.iloc[0]
            # recompute net (แบบง่าย)
            port_net = port_sumusd_netfee - port_refer - init_cap
            dd = max_drawdown(port_sumusd_netfee)

    # 7) KPI
    ret = port_sumusd_netfee.pct_change().fillna(0.0)
    kpis = dict(
        MaxDD = dd,
        Vol   = annualized_vol(ret),
        CAGR  = cagr(port_sumusd_netfee),
        Turnover = turnover_total,
        TxnCost_bps = txn_cost_bps
    )

    return dict(
        weights=w, fix_map=fix_map,
        equity=port_sumusd_netfee, refer=port_refer, net=port_net,
        kpis=kpis, daily_cost=daily_cost
    )

# ------------- UI -------------
st.title("🧠 Portfolio Orchestrator (Risk-Budget Layer)")
st.caption("เพิ่มชั้นบริหารพอร์ตบนกลไก SumUSD/Ref-Log เดิม โดยไม่แกะโค้ดเก่า")

# Inputs
colA, colB = st.columns(2)
tickers_str = colA.text_input("Tickers (คั่นด้วยคอมมา)", value="FFWM,NEGG,RIVN,APLS,NVTS,QXO,RXRX,AGL,FLNC,GERN,DYN,DJT")
start_date  = colA.date_input("Start", value=datetime(2024,1,1)).strftime("%Y-%m-%d")
end_date    = colA.date_input("End"  , value=datetime.now()).strftime("%Y-%m-%d")

total_budget = colB.number_input("Portfolio Budget (Σ fix_i)", value=1500.0*6, min_value=1000.0, step=500.0)
lookback     = colB.number_input("Lookback (days) for vol", min_value=10, value=60, step=5)
max_weight   = colB.slider("Max weight per asset", min_value=0.05, max_value=0.6, value=0.25, step=0.05)
txn_bps      = colB.number_input("Txn cost (bps, round trip)", min_value=0.0, value=10.0, step=1.0)
dd_kill      = colB.number_input("Kill-Switch DD threshold (e.g. 0.2 = 20%)", min_value=0.0, max_value=0.9, value=0.3, step=0.05)

tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

if st.button("🚀 Build Portfolio"):
    px = fetch_prices(tickers, start_date, end_date)
    if px.empty:
        st.error("ไม่พบราคา/ช่วงวันสั้นเกินไป")
    else:
        res = build_portfolio(px, total_budget, int(lookback), float(max_weight), float(txn_bps), float(dd_kill))
        w, fix_map, eq, ref, net, kpis, daily_cost = res['weights'], res['fix_map'], res['equity'], res['refer'], res['net'], res['kpis'], res['daily_cost']

        st.subheader("Weights & FIX budget")
        df_w = pd.DataFrame({'weight': w, 'fix': pd.Series(fix_map)}).sort_values('weight', ascending=False)
        st.dataframe(df_w.style.format({'weight': '{:.2%}', 'fix': '{:,.2f}'}), use_container_width=True)

        st.subheader("KPIs")
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("CAGR", f"{kpis['CAGR']:.2%}")
        k2.metric("Vol (ann.)", f"{kpis['Vol']:.2%}")
        k3.metric("MaxDD", f"{kpis['MaxDD']:.2%}")
        k4.metric("Turnover (Σ|Δw|)", f"{kpis['Turnover']:.2f}")

        st.subheader("Equity / Ref-Log / Net")
        chart_df = pd.DataFrame({'Equity': eq, 'RefLog(Σ)': ref, 'Net(Equity-RefLog-Init)': net})
        st.line_chart(chart_df)

        with st.expander("Daily Txn-cost (bps model)"):
            st.line_chart(daily_cost.rename("Daily TxnCost"))

        st.success("เสร็จแล้ว: ชั้นพอร์ตถูกคุมด้วย risk-budget + DD governance และคิดต้นทุนพื้นฐาน")
