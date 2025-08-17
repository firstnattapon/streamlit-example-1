# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import json
import numpy_financial as npf
import yfinance as yf
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

st.set_page_config(page_title="Cash_Balan Optimizer (MIRR 3Y)", page_icon="🚀", layout="wide")

# ------------------- Config I/O -------------------
CONFIG_FILE = "un15_fx_config.json"

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

    default_config = data.get("__DEFAULT_CONFIG__", {})
    # ตีความ "แต่ละ key = ticker" แบบเดิม
    tickers = {k: v for k, v in data.items() if k != "__DEFAULT_CONFIG__"}
    # เติมค่า default ให้ครบคีย์
    for t, cfg in tickers.items():
        for k, v in default_config.items():
            tickers[t].setdefault(k, v)
        tickers[t].setdefault("Ticker", t)
    return tickers, default_config

@st.cache_data(show_spinner=False)
def dump_json_patch(updated_cash: Dict[str, float]) -> str:
    """
    สร้าง JSON patch minimal: { <TICKER>: { "Cash_Balan": <value> }, ... }
    (ผู้ใช้จะนำไป merge กับไฟล์เดิมเอง หรือใช้ logic app เดิมเขียนทับเฉพาะฟิลด์นี้)
    """
    patch = {t: {"Cash_Balan": float(v)} for t, v in updated_cash.items()}
    return json.dumps(patch, indent=2, ensure_ascii=False)

# ------------------- Price Data -------------------
@st.cache_data(show_spinner=False)
def fetch_monthly_adjclose(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end + pd.Timedelta(days=3), interval="1mo", auto_adjust=True, progress=False)
    if df.empty:
        return df
    df = df.rename_axis("Date").reset_index()
    # ใช้ Adj Close หากมี ไม่งั้น Close
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    df = df[["Date", price_col]].rename(columns={price_col: "price"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.dropna().sort_values("Date").reset_index(drop=True)
    return df

def month_end_series(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    # สร้างช่วง MonthEnd ที่เท่ากันสำหรับทุกตัว
    start_m = (start.to_period("M").to_timestamp("M"))
    end_m   = (end.to_period("M").to_timestamp("M"))
    idx = pd.date_range(start=start_m, end=end_m, freq="M")
    return idx

def align_all_prices(raw_prices: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    รับ dict: ticker -> df(Date, price)
    คืน: wide DataFrame (index = month-end union), และ index จุดตัดร่วม (intersection 3Y)
    """
    if not raw_prices:
        return pd.DataFrame(), pd.DatetimeIndex([])
    frames = []
    for t, df in raw_prices.items():
        if df.empty:
            continue
        tmp = df.copy()
        tmp["Date"] = tmp["Date"].dt.to_period("M").dt.to_timestamp("M")
        tmp = tmp.groupby("Date", as_index=False)["price"].last()
        tmp = tmp.set_index("Date").rename(columns={"price": t})
        frames.append(tmp)
    if not frames:
        return pd.DataFrame(), pd.DatetimeIndex([])
    wide = pd.concat(frames, axis=1).sort_index()
    # ลบท้ายที่มี NaN เป็นแถว ๆ ออกไปเมื่อจำเป็น
    return wide, wide.index

# ------------------- DCA + MIRR -------------------
def build_common_timeline(price_wide: pd.DataFrame, months_limit: int = 36) -> pd.DatetimeIndex:
    # ใช้ "จุดตัดร่วม" ของทุก Ticker เพื่อคงจำนวนงวดเท่ากัน
    full = price_wide.dropna(how="any")
    if full.empty:
        return pd.DatetimeIndex([])
    # ตัดให้เหลือไม่เกิน 36 เดือนล่าสุด
    if len(full.index) > months_limit:
        full = full.iloc[-months_limit:]
    return full.index

def dca_terminal_value_per_dollar(prices: pd.Series) -> float:
    """
    ราคาต่อเดือน (N งวด) → ลง $1 ทุกเดือนที่ราคา p_m
    หน่วยหุ้นรวม = Σ(1/p_m); มูลค่าปลายงวด = last_price * Σ(1/p_m)
    คืนค่า terminal wealth ต่อ $1/เดือน
    """
    if prices.isna().any() or prices.empty:
        return np.nan
    inv_sum = np.sum(1.0 / prices.values)
    last_p = float(prices.values[-1])
    return last_p * inv_sum

def mirr_of_portfolio(cashflows: List[float], finance_rate: float, reinvest_rate: float) -> float:
    try:
        return float(npf.mirr(cashflows, finance_rate, reinvest_rate))
    except Exception:
        return np.nan

def make_portfolio_cashflows(prices_wide: pd.DataFrame, cash_per_ticker: Dict[str, float]) -> List[float]:
    """
    สมมติ DCA รายเดือน: เดือน 1..N เป็น outflow = -Σ b_i (ทุกเดือน)
    งวดสุดท้าย inflow = Σ (b_i * terminal_value_per_1USD_i)
    """
    cols = list(cash_per_ticker.keys())
    sub = prices_wide[cols]
    # outflow รายเดือนเท่ากันทุกเดือน
    total_b = sum(cash_per_ticker.values())
    outflows = [-total_b] * len(sub.index)
    # มูลค่าปลายงวด
    terminal = 0.0
    for t in cols:
        tv = dca_terminal_value_per_dollar(sub[t].dropna())
        if np.isnan(tv):
            continue
        terminal += cash_per_ticker[t] * tv
    return outflows + [terminal]

# ------------------- Greedy Optimizer -------------------
def greedy_optimize(
    prices_wide: pd.DataFrame,
    base_cash: Dict[str, float],
    lower: float = 1.0,
    upper: float = 3000.0,
) -> Dict[str, float]:
    """
    คง "งบรวม" = sum(base_cash). บังคับ 1 ≤ b_i ≤ 3000
    แบ่งส่วนเกินแบบ greedy ไปยัง Ticker ที่มี terminal wealth/1USD สูงสุด
    """
    tickers = list(base_cash.keys())
    N = len(tickers)
    # ขั้นต่ำให้ทุกตัว
    alloc = {t: max(lower, min(upper, float(base_cash[t]))) for t in tickers}
    total_budget = sum(base_cash.values())
    min_total = lower * N
    if total_budget < min_total:
        # ถ้างบรวมน้อยกว่าขั้นต่ำรวม ปรับลงแบบสัดส่วน (แต่ไม่น้อยกว่า 0)
        scale = total_budget / (min_total + 1e-12)
        for t in tickers:
            alloc[t] = lower * scale
        return alloc

    # คำนวณคะแนนต่อ $1 (terminal wealth factor)
    scores = {}
    for t in tickers:
        tv = dca_terminal_value_per_dollar(prices_wide[t])
        scores[t] = -np.inf if np.isnan(tv) else float(tv)

    # ตั้งต้น: ให้ขั้นต่ำก่อน
    for t in tickers:
        alloc[t] = max(lower, min(upper, lower))

    remaining = total_budget - sum(alloc.values())
    if remaining <= 0:
        return alloc

    # เรียงจากคะแนนสูง→ต่ำ
    order = sorted(tickers, key=lambda x: scores.get(x, -np.inf), reverse=True)

    # ใส่ส่วนเพิ่มให้ตัวที่คะแนนสูงก่อนจนชนเพดานหรือหมดงบ
    for t in order:
        if remaining <= 0:
            break
        can_add = upper - alloc[t]
        if can_add <= 0:
            continue
        delta = min(can_add, remaining)
        alloc[t] += delta
        remaining -= delta

    # เผื่อมีเศษเล็ก ๆ เพราะปัดเลข
    if remaining > 1e-9:
        # กระจายเศษตามลำดับเดิม
        i = 0
        L = len(order)
        while remaining > 1e-9 and L > 0:
            t = order[i % L]
            can_add = upper - alloc[t]
            if can_add > 0:
                add = min(can_add, remaining)
                alloc[t] += add
                remaining -= add
            i += 1

    return alloc

# ------------------- UI -------------------
st.title("🚀 Cash_Balan Optimizer — Maximize 3-Year MIRR (Portfolio)")

tickers_cfg, default_cfg = load_config()
if not tickers_cfg:
    st.stop()

all_tickers = sorted(list(tickers_cfg.keys()))
left, right = st.columns([1.3, 1.0], gap="large")

with left:
    st.subheader("1) Settings")
    today = pd.Timestamp.today(tz="Asia/Bangkok").tz_localize(None)
    three_years_ago = today - pd.DateOffset(years=3)

    # ใช้ filter_date ล่าสุดใน config แต่จำกัดไม่เกิน 3 ปี
    cfg_dates = []
    for t, cfg in tickers_cfg.items():
        d = pd.to_datetime(cfg.get("filter_date", three_years_ago))
        try:
            d = d.tz_convert("Asia/Bangkok").tz_localize(None)
        except Exception:
            d = d.tz_localize(None) if getattr(d, "tzinfo", None) else d
        cfg_dates.append(d)
    max_filter = max(min(d, today), three_years_ago) if cfg_dates else three_years_ago

    start_date = st.date_input("Start (<= 3Y back, ใช้จุดตัดร่วม)", max_filter.date())
    start_date = pd.Timestamp(start_date)
    end_date = st.date_input("End (วันนี้หรือล่าสุด)", today.date())
    end_date = pd.Timestamp(end_date)
    end_date = min(end_date, today)

    finance_rate = st.number_input("Finance rate (ต่อคาบเดือน)", min_value=-1.0, max_value=1.0, value=0.0, step=0.001, help="อัตราดอกเบี้ยของเงินทุน (ต่อเดือน) สำหรับ MIRR")
    reinvest_rate = st.number_input("Reinvest rate (ต่อคาบเดือน)", min_value=-1.0, max_value=1.0, value=0.0, step=0.001, help="อัตราที่นำกระแสเงินสดบวกไปลงทุนซ้ำ (ต่อเดือน) สำหรับ MIRR")

    # งบรวมต่อเดือน = ผลรวม Cash_Balan เดิม
    base_cash = {t: float(tickers_cfg[t].get("Cash_Balan", 1.0)) for t in all_tickers}
    lock_total = st.checkbox("Lock: ใช้งบรวมเท่ากับผลรวม Cash_Balan เดิม", value=True)
    if not lock_total:
        manual_total = st.number_input("งบรวมต่อเดือน (USD)", min_value=1.0, value=float(sum(base_cash.values())), step=1.0)
        # scale base_cash ให้ผลรวม = manual_total (ยังคงใช้เป็น baseline)
        s = sum(base_cash.values())
        if s > 0:
            base_cash = {t: v * manual_total / s for t, v in base_cash.items()}

with st.spinner("ดาวน์โหลดราคา..."):
    raw_prices = {}
    for t in all_tickers:
        dfp = fetch_monthly_adjclose(t, start_date - pd.DateOffset(days=10), end_date)
        raw_prices[t] = dfp

prices_wide, union_idx = align_all_prices(raw_prices)
if prices_wide.empty:
    st.error("ไม่มีข้อมูลราคาที่นำมาคิดได้")
    st.stop()

timeline = build_common_timeline(prices_wide, months_limit=36)
if len(timeline) < 6:
    st.warning("จุดตัดร่วมของเดือนที่มีข้อมูลครบทุก Ticker น้อย (<6) อาจทำให้ MIRR ไม่นิ่ง")
prices = prices_wide.reindex(timeline).dropna(how="any")

# คะแนนต่อ 1 USD/เดือน (ใช้ในการจัดสรร)
scores = {t: dca_terminal_value_per_dollar(prices[t]) for t in prices.columns}
scores_ser = pd.Series(scores).sort_values(ascending=False)

with left:
    st.subheader("2) Optimize")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        lower = st.number_input("ขั้นต่ำต่อ Ticker", min_value=0.0, value=1.0, step=1.0)
    with c2:
        upper = st.number_input("ขั้นสูงสุดต่อ Ticker", min_value=1.0, value=3000.0, step=50.0)
    with c3:
        st.write("")

    if st.button("✅ Run Optimizer", type="primary", use_container_width=True):
        alloc_opt = greedy_optimize(prices, base_cash, lower, upper)
        # คำนวณ MIRR ก่อน/หลัง
        cf_before = make_portfolio_cashflows(prices, base_cash)
        cf_after  = make_portfolio_cashflows(prices, alloc_opt)
        mirr_before = mirr_of_portfolio(cf_before, finance_rate, reinvest_rate)
        mirr_after  = mirr_of_portfolio(cf_after, finance_rate, reinvest_rate)

        st.success("Optimization เสร็จ")
        st.metric("Portfolio MIRR (Before)", f"{mirr_before*100:,.3f} % /period")
        st.metric("Portfolio MIRR (After)",  f"{mirr_after*100:,.3f} % /period")

        # ตาราง Before/After + Score
        df_view = pd.DataFrame({
            "Score per $1 (terminal wealth)": [scores[t] for t in prices.columns],
            "Cash_Balan (Before)": [base_cash.get(t, np.nan) for t in prices.columns],
            "Cash_Balan (After)":  [alloc_opt.get(t, np.nan) for t in prices.columns],
        }, index=prices.columns).sort_values("Score per $1 (terminal wealth)", ascending=False)

        st.dataframe(df_view.style.format({
            "Score per $1 (terminal wealth)": "{:,.4f}",
            "Cash_Balan (Before)": "{:,.2f}",
            "Cash_Balan (After)": "{:,.2f}",
        }), use_container_width=True)

        # JSON Patch download
        patch = dump_json_patch(alloc_opt)
        st.download_button(
            "⬇️ ดาวน์โหลด JSON patch (อัปเดตเฉพาะ Cash_Balan)",
            data=patch.encode("utf-8"),
            file_name="cash_balan_patch.json",
            mime="application/json",
            use_container_width=True
        )

with right:
    st.subheader("Reference / Sanity Check")
    st.caption("คะแนนยิ่งสูง → ควรได้รับสัดส่วนเงินมากกว่า (ภายใต้กรอบ 1–3000)")
    st.table(scores_ser.to_frame("Score per $1").style.format("{:,.4f}"))

    st.caption("หมายเหตุ")
    st.markdown("""
- ใช้ DCA รายเดือนบนจุดตัดร่วม ≤ 36 เดือน เพื่อให้ MIRR รวมคำนวณบนจำนวนงวดเท่ากัน  
- MIRR คำนวณจากกระแสเงินสดรวม: เดือน 1..N เป็นเงินออกเท่ากัน (ผลรวม Cash_Balan ของพอร์ต), งวดสุดท้ายเป็นเงินเข้า (มูลค่าปิดรวม)  
- งบรวมถูกล็อกไว้เท่ากับผลรวม Cash_Balan ปัจจุบันของไฟล์ (ยกเลิกล็อกได้ใน UI)
- ไม่มีค่าคอมมิชชั่น/สเปรด/ภาษี (ควรบวกในรุ่นถัดไปถ้าต้องการความสมจริง)
""")
