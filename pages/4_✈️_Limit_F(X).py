
import streamlit as st
import pandas as pd
import numpy as np
import json
import numpy_financial as npf
import yfinance as yf
from typing import Dict, List, Tuple

st.set_page_config(page_title="Cash_Balan Optimizer (MIRR 3Y + Risk Parity)", page_icon="🚀", layout="wide")

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
    full = price_wide.dropna(how="any")
    if full.empty:
        return full
    if len(full) > months_limit:
        full = full.iloc[-months_limit:]
    return full

# ------------------- DCA + MIRR -------------------
def dca_terminal_value_per_dollar(prices) -> float:
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

# ------------------- Risk Parity / Risk Budgeting -------------------
def returns_and_cov(prices_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Monthly simple returns & covariance with small ridge for stability."""
    rets = prices_df.pct_change().dropna(how="any")
    if rets.empty:
        return rets, np.array([[]])
    cov = np.cov(rets.values, rowvar=False)
    # ridge for numerical stability
    ridge = 1e-8 * np.trace(cov) / cov.shape[0] if cov.shape[0] > 0 else 0.0
    cov = cov + ridge * np.eye(cov.shape[0])
    return rets, cov

def risk_budgeting_weights(cov: np.ndarray, budget: np.ndarray, max_iter: int = 10_000, tol: float = 1e-9) -> np.ndarray:
    """
    Risk Budgeting via fixed-point iteration:
        target RC_i = b_i * (w^T Σ w)
        update: w <- target / (Σ w), then renormalize sum(w)=1
    - เมื่อ Σ เป็นเส้นทแยงมุม: คำตอบ w_i ∝ sqrt(b_i)/σ_i
    """
    n = cov.shape[0]
    b = np.array(budget, dtype=float)
    b = np.maximum(b, 1e-16)
    b = b / b.sum()
    # init: inverse-vol
    iv = 1.0 / np.sqrt(np.maximum(np.diag(cov), 1e-16))
    w = iv / iv.sum()

    for _ in range(max_iter):
        m = cov @ w                    # marginal risk
        T = float(w @ m)               # total variance
        target = b * T                 # target risk contribution
        w_new = target / np.maximum(m, 1e-16)
        w_new = np.maximum(w_new, 1e-16)
        w_new = w_new / w_new.sum()
        if np.linalg.norm(w_new - w, 1) < tol:
            w = w_new
            break
        w = w_new
    return w

def risk_contributions(cov: np.ndarray, w: np.ndarray) -> np.ndarray:
    m = cov @ w
    rc = w * m
    return rc

def allocate_with_bounds_from_weights(
    weights: Dict[str, float],
    all_tickers: List[str],
    total_budget: float,
    lower: float,
    upper: float,
) -> Dict[str, float]:
    """
    เริ่มต้นให้ขั้นต่ำกับทุกตัว จากนั้นกระจาย 'ส่วนที่เหลือ' ตาม weights (เฉพาะตัวที่มี weight)
    แล้วเคารพเพดาน/พื้น ด้วยการรีดิสทริบิวต์ซ้ำจนเหลือเศษน้อยมากหรือทุกตัวชนเพดาน
    """
    n_all = len(all_tickers)
    alloc = {t: float(lower) for t in all_tickers}
    remaining = total_budget - lower * n_all
    remaining = float(max(0.0, remaining))

    active = [t for t, w in weights.items() if w > 0 and t in all_tickers]
    if not active or remaining <= 1e-12:
        return {t: float(max(lower, min(upper, v))) for t, v in alloc.items()}

    w_vec = np.array([weights[t] for t in active], dtype=float)
    w_vec = np.maximum(w_vec, 0.0)
    if w_vec.sum() <= 0:
        w_vec = np.ones_like(w_vec) / len(w_vec)
    else:
        w_vec = w_vec / w_vec.sum()

    add = remaining * w_vec
    for t, a in zip(active, add):
        alloc[t] += a

    # enforce caps with redistribution
    def excess(t): return max(0.0, alloc[t] - upper)
    def deficit(t): return max(0.0, lower - alloc[t])

    for _ in range(20):
        # clip to [lower, upper]
        clipped = False
        excess_amt = 0.0
        for t in all_tickers:
            if alloc[t] > upper:
                excess_amt += alloc[t] - upper
                alloc[t] = upper
                clipped = True
            elif alloc[t] < lower:
                # (ปกติจะไม่ต่ำกว่าเพราะเริ่มด้วย lower)
                pass
        if not clipped or excess_amt <= 1e-12:
            break
        # redistribute excess to non-capped actives by weight
        room = np.array([max(0.0, upper - alloc[t]) if t in active else 0.0 for t in all_tickers])
        if room.sum() <= 1e-12:
            break
        room_weights = np.array([weights.get(t, 0.0) if t in active else 0.0 for t in all_tickers])
        room_weights = room_weights * (room > 0)
        if room_weights.sum() <= 0:
            # fallback: proportional to room
            distrib = excess_amt * (room / room.sum())
        else:
            room_weights = room_weights / room_weights.sum()
            distrib = excess_amt * room_weights
        for i, t in enumerate(all_tickers):
            if room[i] > 0:
                add_i = min(room[i], distrib[i])
                alloc[t] += add_i

    # final clamp
    alloc = {t: float(max(lower, min(upper, v))) for t, v in alloc.items()}
    return alloc

# ------------------- Greedy (เดิม) -------------------
def greedy_optimize(prices_df: pd.DataFrame, base_cash: Dict[str, float], lower: float = 1.0, upper: float = 3000.0) -> Dict[str, float]:
    tickers = [t for t in base_cash.keys() if t in prices_df.columns]
    if not tickers:
        return {}
    alloc = {t: float(lower) for t in base_cash.keys()}
    total_budget = float(sum(base_cash.values()))
    min_total = float(lower) * len(alloc)
    if total_budget <= 0:
        return {t: 0.0 for t in alloc.keys()}
    if total_budget < min_total:
        scale = total_budget / (min_total + 1e-12)
        return {t: float(lower) * float(scale) for t in alloc.keys()}
    scores = {}
    for t in tickers:
        col = prices_df[[t]].squeeze()
        tv = dca_terminal_value_per_dollar(col)
        scores[t] = -np.inf if np.isnan(tv) else float(tv)
    remaining = total_budget - sum(alloc.values())
    order = sorted(tickers, key=lambda x: scores.get(x, -np.inf), reverse=True)
    for t in order:
        if remaining <= 1e-12:
            break
        cap = float(upper) - alloc[t]
        if cap <= 0:
            continue
        add = min(cap, remaining)
        alloc[t] += add
        remaining -= add
    # spread leftover if any
    if remaining > 1e-9:
        i = 0
        active = [t for t in order if alloc[t] < upper]
        L = len(active)
        while remaining > 1e-9 and L > 0:
            t = active[i % L]
            cap = float(upper) - alloc[t]
            if cap > 0:
                add = min(cap, remaining)
                alloc[t] += add
                remaining -= add
            i += 1
    alloc = {t: float(max(lower, min(upper, v))) for t, v in alloc.items()}
    return alloc

# ------------------- UI -------------------
st.title("🚀 Cash_Balan Optimizer — Maximize 3Y MIRR + Risk Parity")

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

    # default start = max(filter_date, today-3y)
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

    start_date = st.date_input("Start (≤ 3Y back, จุดตัดร่วม)", default_start.date())
    start_date = pd.Timestamp(start_date)
    end_date = st.date_input("End (วันนี้หรือล่าสุด)", today.date())
    end_date = pd.Timestamp(end_date)
    if end_date > today:
        end_date = today
    if start_date > end_date:
        st.warning("Start > End → ปรับ Start = End-36 เดือน")
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

# คะแนนต่อ $1 (terminal wealth factor)
scores = {t: dca_terminal_value_per_dollar(prices[[t]].squeeze()) for t in prices.columns}
scores_ser = pd.Series(scores, dtype=float).replace([np.inf, -np.inf], np.nan).sort_values(ascending=False)

with left:
    st.subheader("2) Allocation Method")
    method = st.selectbox(
        "เลือกวิธีจัดสรร",
        ["Greedy (Score per $1)", "Risk Parity (Equal RC)", "Risk Budgeting (RC ∝ Score^α)"],
        index=0
    )
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        lower = st.number_input("ขั้นต่ำต่อ Ticker", min_value=0.0, value=1.0, step=1.0)
    with c2:
        upper = st.number_input("ขั้นสูงสุดต่อ Ticker", min_value=1.0, value=3000.0, step=50.0)
    with c3:
        alpha = st.number_input("α (เฉพาะ Risk Budgeting)", min_value=0.0, value=1.0, step=0.25)

    run = st.button("✅ Run Optimizer", type="primary", use_container_width=True)

# ------------------- Run -------------------
if run:
    if method == "Greedy (Score per $1)":
        alloc_opt = greedy_optimize(prices, base_cash, lower, upper)

        # MIRR ก่อน/หลัง
        cf_before = make_portfolio_cashflows(prices, base_cash)
        cf_after  = make_portfolio_cashflows(prices, alloc_opt)
        mirr_before = mirr_of_portfolio(cf_before, finance_rate, reinvest_rate)
        mirr_after  = mirr_of_portfolio(cf_after, finance_rate, reinvest_rate)

        st.success("Optimization (Greedy) เสร็จ")
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
        st.dataframe(df_view.style.format({
            "Score per $1 (terminal wealth)": "{:,.4f}",
            "Cash_Balan (Before)": "{:,.2f}",
            "Cash_Balan (After)": "{:,.2f}",
        }), use_container_width=True)

        patch = dump_json_patch(alloc_opt)
        st.download_button("⬇️ ดาวน์โหลด JSON patch (Cash_Balan)", data=patch.encode("utf-8"),
                           file_name="cash_balan_patch.json", mime="application/json",
                           use_container_width=True)

    else:
        # ===== Risk Parity / Risk Budgeting =====
        rets, cov = returns_and_cov(prices)
        if rets.empty or cov.size == 0:
            st.error("คำนวณ covariance ไม่ได้ (ข้อมูลไม่พอ)")
            st.stop()

        act = list(prices.columns)
        n = len(act)

        if method == "Risk Parity (Equal RC)":
            budget = np.ones(n) / n
        else:
            # Risk Budgeting — งบความเสี่ยงตาม Score^α
            # ถ้า score บางตัว NaN/<=0 ให้แทนเป็นค่าน้อย ๆ เพื่อไม่ทิ้งตัวนั้น
            raw = np.array([scores.get(t, np.nan) for t in act], dtype=float)
            raw = np.where(np.isfinite(raw) & (raw > 0), raw, 1e-6)
            budget = raw ** float(alpha)
            budget = budget / budget.sum()

        w = risk_budgeting_weights(cov, budget)
        # ตรวจ RC%
        rc = risk_contributions(cov, w)
        rc_pct = rc / rc.sum() if rc.sum() > 0 else rc

        # แปลงเป็น Cash_Balan ภายใต้กรอบ 1–3000 และมีขั้นต่ำทุกตัว (รวมถึง non-active)
        total_budget = float(sum(base_cash.values()))
        weights_map = {t: float(w[i]) for i, t in enumerate(act)}
        alloc_opt = allocate_with_bounds_from_weights(weights_map, all_tickers, total_budget, lower, upper)

        # MIRR ก่อน/หลัง
        cf_before = make_portfolio_cashflows(prices, base_cash)
        cf_after  = make_portfolio_cashflows(prices, alloc_opt)
        mirr_before = mirr_of_portfolio(cf_before, finance_rate, reinvest_rate)
        mirr_after  = mirr_of_portfolio(cf_after, finance_rate, reinvest_rate)

        # แสดงผล
        title = "Risk Parity (Equal RC)" if method.startswith("Risk Parity") else f"Risk Budgeting (α={alpha})"
        st.success(f"Optimization — {title} เสร็จ")

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Portfolio MIRR (Before)", f"{(mirr_before*100) if not np.isnan(mirr_before) else 0:,.3f} % /period")
        with m2:
            st.metric("Portfolio MIRR (After)",  f"{(mirr_after*100) if not np.isnan(mirr_after) else 0:,.3f} % /period")

        df_alloc = pd.DataFrame({
            "Cash_Balan (Before)": [base_cash.get(t, np.nan) for t in prices.columns],
            "Cash_Balan (After)":  [alloc_opt.get(t, np.nan) for t in prices.columns],
            "Score per $1":        [scores.get(t, np.nan) for t in prices.columns],
            "Weight (w)":          [weights_map.get(t, np.nan) for t in prices.columns],
            "RC %":                [float(rc_pct[i]) if i < len(rc_pct) else np.nan for i, t in enumerate(prices.columns)],
        }, index=prices.columns).sort_values("RC %", ascending=False)

        st.dataframe(df_alloc.style.format({
            "Cash_Balan (Before)": "{:,.2f}",
            "Cash_Balan (After)": "{:,.2f}",
            "Score per $1": "{:,.4f}",
            "Weight (w)": "{:,.4%}",
            "RC %": "{:,.2%}",
        }), use_container_width=True)

        patch = dump_json_patch(alloc_opt)
        st.download_button("⬇️ ดาวน์โหลด JSON patch (Cash_Balan)", data=patch.encode("utf-8"),
                           file_name="cash_balan_patch.json", mime="application/json",
                           use_container_width=True)

with right:
    st.subheader("Reference / Scores")
    st.caption("คะแนนยิ่งสูง → ควรได้รับงบมากกว่า (ในโหมด Greedy) หรือเป็น 'งบความเสี่ยงเป้า' ในโหมด Risk Budgeting")
    st.table(scores_ser.to_frame("Score per $1").style.format("{:,.4f}"))

    st.subheader("Notes")
    st.markdown("""
- Risk Parity = Equal Risk Contribution (ERC): ให้แต่ละตัว **มีส่วนแบ่งความเสี่ยงเท่ากัน**  
- Risk Budgeting (RC ∝ Score^α): ตั้ง **งบความเสี่ยงเป้า** ตาม Score per $1 (ยกกำลัง α) แล้วแก้ w ให้ RC_i ตามสัดส่วนดังกล่าว  
- หลังได้ weight จะถูกแปลงเป็น Cash_Balan และเคารพกรอบ 1–3000 ด้วยการรีดิสทริบิวต์ส่วนเกิน/ขาดอัตโนมัติ  
- Covariance ใช้ผลตอบแทนรายเดือน (simple return) บนช่วงจุดตัดร่วม และมีใส่ ridge เล็กน้อยเพื่อความเสถียรเชิงตัวเลข
""")
