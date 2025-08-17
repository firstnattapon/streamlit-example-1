# -*- coding: utf-8 -*-
"""
Cash_Balan Optimizer — Maximize Portfolio MIRR (3Y)
- อ่าน un15_fx_config.json
- ดึงราคาแบบรายเดือนย้อนหลัง ≤36 เดือน (จุดตัดร่วมทุก Ticker)
- จำลอง DCA รายเดือน: ลงเงิน b_i ต่อเดือน/ต่อ Ticker
- จัดสรรงบรวม (เท่ากับผลรวม Cash_Balan เดิม เว้นแต่ผู้ใช้ยกเลิก lock)
- ข้อจำกัด 1 ≤ b_i ≤ 3000
- ใช้ greedy บน "terminal wealth per $1" ต่อ Ticker
- คำนวณ MIRR ของพอร์ต (กระแสเงินสดรวม)
!! รวม hotfix กัน ValueError กรณี Series/DataFrame ambiguity
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import numpy_financial as npf
import yfinance as yf
from typing import Dict, List, Tuple

st.set_page_config(page_title="Cash_Balan Optimizer (MIRR 3Y)", page_icon="🚀", layout="wide")

CONFIG_FILE = "un15_fx_config.json"

# ------------------- Config I/O -------------------
@st.cache_data(show_spinner=False)
def load_config(filename: str = CONFIG_FILE) -> Tuple[Dict, Dict]:
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        st.error(f"ไม่พบไฟล์คอนฟิก: {filename}")
        return {}, {}
    except json.JSONDecodeError:
        st.error(f"รูปแบบ JSON ของ {filename} ไม่ถูกต้อง")
        return {}, {}

    default_cfg = data.get("__DEFAULT_CONFIG__", {})
    tickers = {k: v for k, v in data.items() if k != "__DEFAULT_CONFIG__"}

    # fill defaults + set Ticker key
    for t, cfg in tickers.items():
        for k, v in default_cfg.items():
            cfg.setdefault(k, v)
        cfg.setdefault("Ticker", t)
    return tickers, default_cfg

@st.cache_data(show_spinner=False)
def dump_json_patch(updated_cash: Dict[str, float]) -> str:
    patch = {t: {"Cash_Balan": float(v)} for t, v in updated_cash.items()}
    return json.dumps(patch, indent=2, ensure_ascii=False)

# ------------------- Price Data -------------------
@st.cache_data(show_spinner=False)
def fetch_monthly_adjclose(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    # add small padding days to ensure last month included
    df = yf.download(
        ticker,
        start=start - pd.DateOffset(days=10),
        end=end + pd.DateOffset(days=3),
        interval="1mo",
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        return pd.DataFrame(columns=["Date", "price"])
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    out = (
        df.reset_index()
          .rename(columns={price_col: "price"})
          .loc[:, ["Date", "price"]]
          .dropna()
          .sort_values("Date")
    )
    # normalize to month-end timestamps without tz
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)
    return out

def to_month_end_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    tmp = df.copy()
    tmp["Date"] = tmp["Date"].dt.to_period("M").dt.to_timestamp("M")
    tmp = tmp.groupby("Date", as_index=False)["price"].last()
    return tmp.set_index("Date")

def align_prices(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for t, df in raw.items():
        if df is None or df.empty:
            continue
        me = to_month_end_index(df).rename(columns={"price": t})
        frames.append(me)
    if not frames:
        return pd.DataFrame()
    wide = pd.concat(frames, axis=1).sort_index()
    return wide

def build_common_timeline(price_wide: pd.DataFrame, months_limit: int = 36) -> pd.DataFrame:
    """
    คืน DataFrame ที่เหลือเฉพาะ 'แถวที่ทุกคอลัมน์ไม่ใช่ NaN' (จุดตัดร่วม)
    และจำกัดไม่เกิน 36 เดือนล่าสุด
    """
    full = price_wide.dropna(how="any")
    if full.empty:
        return full
    if len(full) > months_limit:
        full = full.iloc[-months_limit:]
    return full

# ------------------- DCA + MIRR -------------------
def dca_terminal_value_per_dollar(prices) -> float:
    """
    Robust ต่อ Series/DataFrame:
    - ถ้าได้ DataFrame เข้ามา: ใช้คอลัมน์แรก
    - แปลงเป็นตัวเลข + dropna
    - terminal wealth ต่อ $1/เดือน = last_price * Σ(1/price_m)
    """
    import pandas as pd
    import numpy as np

    if isinstance(prices, pd.DataFrame):
        if prices.shape[1] == 0:
            return np.nan
        prices = prices.iloc[:, 0]

    prices = pd.to_numeric(pd.Series(prices), errors="coerce").dropna()
    if prices.empty:
        return np.nan

    inv_sum = (1.0 / prices).sum()
    last_p = float(prices.iloc[-1])
    return last_p * float(inv_sum)

def make_portfolio_cashflows(prices_df: pd.DataFrame, cash_per_ticker: Dict[str, float]) -> List[float]:
    """
    สมมติ DCA รายเดือน N งวด:
      เดือน 1..N: outflow = -Σ b_i
      งวดสุดท้าย: inflow = Σ_i (b_i * terminal_value_per_$1_i)
    """
    N = len(prices_df.index)
    if N == 0:
        return []
    total_b = sum(cash_per_ticker.values())
    outflows = [-total_b] * N

    terminal = 0.0
    for t, b in cash_per_ticker.items():
        if t not in prices_df.columns:
            continue
        col = prices_df[[t]].squeeze()
        tv = dca_terminal_value_per_dollar(col)
        if not np.isnan(tv):
            terminal += b * tv
    return outflows + [terminal]

def mirr_of_portfolio(cashflows: List[float], finance_rate: float, reinvest_rate: float) -> float:
    if not cashflows or len(cashflows) < 2:
        return np.nan
    try:
        return float(npf.mirr(cashflows, finance_rate, reinvest_rate))
    except Exception:
        return np.nan

# ------------------- Greedy Optimizer -------------------
def greedy_optimize(
    prices_df: pd.DataFrame,
    base_cash: Dict[str, float],
    lower: float = 1.0,
    upper: float = 3000.0,
) -> Dict[str, float]:
    """
    ล็อก 'งบรวม' = sum(base_cash) และบังคับ 1 ≤ b_i ≤ 3000
    ขั้นตอน:
      1) ให้ขั้นต่ำทุกตัว
      2) คำนวณคะแนน per $1: terminal wealth factor จาก DCA
      3) เติมงบส่วนที่เหลือให้ Ticker คะแนนสูงสุดก่อน (จนชนเพดาน)
    """
    tickers = [t for t in base_cash.keys() if t in prices_df.columns]
    if not tickers:
        return {}

    # ขั้นต่ำ
    alloc = {t: float(max(lower, 0.0)) for t in base_cash.keys()}  # รวม key เดิม แม้บางตัวไม่มีข้อมูล
    total_budget = float(sum(base_cash.values()))
    min_total = float(lower) * len(alloc)

    if total_budget <= 0:
        return {t: 0.0 for t in alloc.keys()}

    if total_budget < min_total:
        # scale ต่ำกว่าขั้นต่ำรวม: กระจายแบบสัดส่วน
        scale = total_budget / (min_total + 1e-12)
        return {t: float(lower) * float(scale) for t in alloc.keys()}

    # คะแนนต่อ $1
    scores: Dict[str, float] = {}
    for t in tickers:
        col = prices_df[[t]].squeeze()
        tv = dca_terminal_value_per_dollar(col)
        scores[t] = -np.inf if np.isnan(tv) else float(tv)

    # ให้ขั้นต่ำก่อน
    alloc = {t: float(lower) for t in alloc.keys()}
    remaining = total_budget - sum(alloc.values())
    if remaining <= 1e-12:
        return alloc

    # เรียงจากคะแนนสูง→ต่ำ (เฉพาะที่มีข้อมูล)
    order = sorted(tickers, key=lambda x: scores.get(x, -np.inf), reverse=True)

    # เติมงบตามลำดับคะแนน
    for t in order:
        if remaining <= 1e-12:
            break
        cap = float(upper) - alloc[t]
        if cap <= 0:
            continue
        add = min(cap, remaining)
        alloc[t] += add
        remaining -= add

    # เศษเล็ก ๆ ถ้ายังเหลือ ให้ปัดกระจาย
    if remaining > 1e-9:
        i = 0
        L = len(order)
        while remaining > 1e-9 and L > 0:
            t = order[i % L]
            cap = float(upper) - alloc[t]
            if cap > 0:
                add = min(cap, remaining)
                alloc[t] += add
                remaining -= add
            i += 1

    # clamp เผื่อเลขทศนิยม
    alloc = {t: float(max(lower, min(upper, v))) for t, v in alloc.items()}
    return alloc

# ------------------- UI -------------------
st.title("🚀 Cash_Balan Optimizer — Maximize 3-Year MIRR (Portfolio)")

tickers_cfg, default_cfg = load_config()
if not tickers_cfg:
    st.stop()

all_tickers = sorted(list(tickers_cfg.keys()))
base_cash_orig = {t: float(tickers_cfg[t].get("Cash_Balan", 1.0)) for t in all_tickers}
total_base_cash = float(sum(base_cash_orig.values()))

left, right = st.columns([1.3, 1.0], gap="large")

with left:
    st.subheader("1) Settings")
    today = pd.Timestamp.today(tz="Asia/Bangkok").tz_localize(None)
    three_years_ago = today - pd.DateOffset(years=3)

    # กำหนด start จาก max(filter_date, today-3y)
    cfg_dates = []
    for t, cfg in tickers_cfg.items():
        d = pd.to_datetime(cfg.get("filter_date", three_years_ago), errors="coerce")
        if d is not None and not pd.isna(d):
            try:
                d = d.tz_convert("Asia/Bangkok").tz_localize(None)
            except Exception:
                d = d.tz_localize(None) if getattr(d, "tzinfo", None) else d
            cfg_dates.append(d)
    default_start = max(three_years_ago, max(cfg_dates) if cfg_dates else three_years_ago)

    start_date = st.date_input("Start (≤ 3Y back, ใช้จุดตัดร่วม)", default_start.date())
    start_date = pd.Timestamp(start_date)
    end_date = st.date_input("End (วันนี้หรือล่าสุด)", today.date())
    end_date = pd.Timestamp(end_date)
    if end_date > today:
        end_date = today
    if start_date > end_date:
        st.warning("Start > End → ปรับ Start = End-36 เดือนอัตโนมัติ")
        start_date = end_date - pd.DateOffset(months=36)

    finance_rate = st.number_input("Finance rate (ต่อคาบเดือน)", min_value=-1.0, max_value=1.0, value=0.0, step=0.001)
    reinvest_rate = st.number_input("Reinvest rate (ต่อคาบเดือน)", min_value=-1.0, max_value=1.0, value=0.0, step=0.001)

    lock_total = st.checkbox("Lock: ใช้งบรวมเท่ากับผลรวม Cash_Balan เดิม", value=True)
    if lock_total:
        base_cash = dict(base_cash_orig)
    else:
        manual_total = st.number_input("งบรวมต่อเดือน (USD)", min_value=1.0, value=max(1.0, total_base_cash), step=1.0)
        s = sum(base_cash_orig.values())
        base_cash = {t: (v * manual_total / s if s > 0 else manual_total / max(1, len(base_cash_orig))) for t, v in base_cash_orig.items()}

# ดึงราคา
with st.spinner("ดาวน์โหลดราคา (เดือน)…"):
    raw_prices = {t: fetch_monthly_adjclose(t, start_date, end_date) for t in all_tickers}
price_wide = align_prices(raw_prices)
if price_wide.empty:
    st.error("ไม่พบราคาที่ใช้งานได้")
    st.stop()

prices = build_common_timeline(price_wide, months_limit=36)
if prices.empty:
    st.error("ไม่มีช่วงเวลาที่ข้อมูลครบทุก Ticker (จุดตัดร่วมเป็นค่าว่าง)")
    st.stop()

if len(prices) < 6:
    st.warning("จำนวนงวดจุดตัดร่วม < 6 เดือน — MIRR อาจไม่นิ่ง")

# คะแนนต่อ $1 (terminal wealth factor) สำหรับแนะนำการจัดสรร
scores = {t: dca_terminal_value_per_dollar(prices[[t]].squeeze()) for t in prices.columns}
scores_ser = pd.Series(scores, dtype=float).replace([np.inf, -np.inf], np.nan).sort_values(ascending=False)

with left:
    st.subheader("2) Optimize")
    c1, c2, _ = st.columns([1, 1, 1])
    with c1:
        lower = st.number_input("ขั้นต่ำต่อ Ticker", min_value=0.0, value=1.0, step=1.0)
    with c2:
        upper = st.number_input("ขั้นสูงสุดต่อ Ticker", min_value=1.0, value=3000.0, step=50.0)

    run = st.button("✅ Run Optimizer", type="primary", use_container_width=True)

    if run:
        alloc_opt = greedy_optimize(prices, base_cash, lower, upper)

        # MIRR ก่อน/หลัง
        cf_before = make_portfolio_cashflows(prices, base_cash)
        cf_after  = make_portfolio_cashflows(prices, alloc_opt)
        mirr_before = mirr_of_portfolio(cf_before, finance_rate, reinvest_rate)
        mirr_after  = mirr_of_portfolio(cf_after, finance_rate, reinvest_rate)

        st.success("Optimization เสร็จ")

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Portfolio MIRR (Before)", f"{(mirr_before*100) if not np.isnan(mirr_before) else 0:,.3f} % /period")
        with m2:
            st.metric("Portfolio MIRR (After)",  f"{(mirr_after*100) if not np.isnan(mirr_after) else 0:,.3f} % /period")

        df_view = pd.DataFrame({
            "Score per $1 (terminal wealth)": [scores.get(t, np.nan) for t in prices.columns],
            "Cash_Balan (Before)": [base_cash.get(t, np.nan) for t in prices.columns],
            "Cash_Balan (After)":  [alloc_opt.get(t, np.nan) for t in prices.columns],
        }, index=prices.columns).sort_values("Score per $1 (terminal wealth)", ascending=False)

        st.dataframe(
            df_view.style.format({
                "Score per $1 (terminal wealth)": "{:,.4f}",
                "Cash_Balan (Before)": "{:,.2f}",
                "Cash_Balan (After)": "{:,.2f}",
            }),
            use_container_width=True
        )

        # JSON Patch เฉพาะ Cash_Balan
        patch = dump_json_patch(alloc_opt)
        st.download_button(
            "⬇️ ดาวน์โหลด JSON patch (อัปเดตเฉพาะ Cash_Balan)",
            data=patch.encode("utf-8"),
            file_name="cash_balan_patch.json",
            mime="application/json",
            use_container_width=True
        )

with right:
    st.subheader("Reference / Scores")
    st.caption("คะแนนยิ่งสูง → ควรได้รับงบมากกว่า (ภายใต้กรอบ 1–3000)")
    st.table(scores_ser.to_frame("Score per $1").style.format("{:,.4f}"))

    st.caption("หมายเหตุ")
    st.markdown("""
- ใช้ DCA รายเดือนบนจุดตัดร่วม ≤ 36 เดือน เพื่อให้ MIRR รวมคำนวณบนจำนวนงวดเท่ากัน  
- MIRR ใช้กระแสเงินสดรวม: เดือน 1..N เป็นเงินออกเท่ากัน (ผลรวม Cash_Balan ของพอร์ต), งวดสุดท้ายเป็นเงินเข้า (มูลค่าปิดรวม)  
- ค่าธรรมเนียม/สเปรด/ภาษี = 0 (เพิ่มได้ในรุ่นถัดไป)
- Hotfix: ป้องกัน `ValueError: The truth value of a Series is ambiguous` โดยบังคับ Series 1D และไม่ใช้เช็ค boolean บน Series/DF ตรง ๆ
""")
