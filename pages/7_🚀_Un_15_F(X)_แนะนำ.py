"""Un_15 F(X) · แนะนำ — เครื่องยนต์เดียวกับ pages/7_🚀_Un_15_F(X).py บวกข้อแนะนำจากการรีวิวหลักการ

หน้าเดิม: หลาย chain แบบ constant value → ΣE (ส่วนเกินเหนือ log reference) กับพื้นเงินสดของ ΣA
หน้านี้เพิ่ม:
  1. A guard      ΣA / capital (เงินสดจริง) ข้าง ΣE / capital + เตือนตัว leveraged / inverse / กลับทิศ
  2. Correlation  ρ matrix · ρ เฉลี่ย · effective bets (Σλ)²/Σλ² · Diversification ratio ของพื้นเงินสด
  3. Rolling      พื้นเงินสดจากหลายวันเริ่ม (ค่ากลาง / p5) แทนวันเริ่มวันเดียว
  4. Sizing       งบรวมคงที่ + risk parity: FAVᵢ ∝ 1/σᵢ (หรือเท่ากัน / กำหนดเองแบบเดิม)
  5. Projection   E ต่อปีจาก σ ล่าสุด: Σ ½·FAVᵢ·σᵢ²  แทน "ค่าเฉลี่ยในอดีต × 252"
  6. Survival     Cash_Balan (หน้าเดิมหักล้างตัวเองใน net_pv) → เทียบกับเงินสำรองที่ต้องใช้จริง
  7. อื่น ๆ        Return on capital ไม่มี look-ahead · ราคาไม่ปรับปันผลเป็นค่าเริ่มต้น · DIFF threshold แบบ bot

ฟังก์ชันส่วนบน (ก่อน "Data + UI") เป็น pure: ไม่เรียก streamlit / yfinance — tests/test_un15_recommended.py อ่านด้วย ast
"""

import json

import numpy as np
import numpy_financial as npf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# --------------------------------
# Constants
# --------------------------------
TRADING_DAYS = 252
FIXED_ASSET_RESET_USD = 1500.0
LEGO_DIFF_PCT = 50.0 / 1500.0 * 100.0          # bot LEGO: DIFF 50 บน FIX_C 1,500 = 3.33%
INVERSE_CORR_THRESHOLD = -0.8
SIZING_RISK_PARITY = "Risk parity · งบรวมคงที่"
SIZING_EQUAL = "เท่ากัน · งบรวมคงที่"
SIZING_MANUAL = "กำหนดเองรายตัว (แบบหน้าเดิม)"

# ETF ที่ปรับพอร์ตเองทุกวัน (leveraged / inverse / VIX futures): E ที่ bot เก็บได้ถูกหักใน R จนเงินสดจริงไม่โต
LEVERAGED_OR_INVERSE = frozenset({
    "ZSL", "AGQ", "DUST", "NUGT", "JDST", "JNUG", "GLL", "UGL",
    "SQQQ", "TQQQ", "QID", "QLD", "PSQ", "SPXU", "SPXL", "SPXS", "UPRO", "SDS", "SSO", "SH", "SDOW", "UDOW",
    "SOXL", "SOXS", "TECL", "TECS", "FNGU", "FNGD", "LABU", "LABD", "TNA", "TZA", "FAS", "FAZ",
    "TBT", "TMF", "TMV", "TBF", "TTT", "YINN", "YANG", "ERX", "ERY", "GUSH", "DRIP",
    "UVXY", "VXX", "VIXY", "SVXY", "SVIX", "UVIX",
    "BOIL", "KOLD", "UCO", "SCO",
    "NVDL", "NVDS", "TSLL", "TSLQ", "TSLZ", "CONL", "MSTU", "MSTZ", "BITX", "SBIT", "ETHU",
})


# --------------------------------
# Core model (pure)
# --------------------------------
def build_actions(pred, n):
    """pred = 1 → act ทุกแท่ง · pred เป็น list (DNA action stream) → ต้องยาวเท่าราคา"""
    if np.isscalar(pred):
        return np.full(n, int(pred), dtype=np.int8)
    actions = np.asarray(pred, dtype=np.int8)
    if actions.ndim != 1 or len(actions) != n:
        raise ValueError("pred action stream must have the same length as prices")
    return actions


def constant_value_core(prices, actions, fixed_asset_value, diff=0.0):
    """re (เงินสดที่เกิดจริงต่อแท่ง) และ amount (จำนวนหุ้นหลังแท่ง) ของ chain เดียว

    act เมื่อ actions[i] = 1 และ |amount·P − FAV| > diff → re = amount·P − FAV แล้วรีเซ็ต amount = FAV / P
    diff = 0 และ actions = 1 ทุกแท่ง = รีบาลานซ์ทุกแท่งแบบหน้าเดิม
    """
    prices = np.asarray(prices, dtype=np.float64)
    n = len(prices)
    amount = np.empty(n, dtype=np.float64)
    re = np.zeros(n, dtype=np.float64)
    if n == 0:
        return re, amount
    amount[0] = fixed_asset_value / prices[0]
    for idx in range(1, n):
        gap = amount[idx - 1] * prices[idx] - fixed_asset_value
        if actions[idx] != 0 and abs(gap) > diff:
            re[idx] = gap
            amount[idx] = fixed_asset_value / prices[idx]
        else:
            amount[idx] = amount[idx - 1]
    return re, amount


def simulate_constant_value_path(prices, actions, fixed_asset_value, starting_cash, diff=0.0):
    """เหมือนหน้าเดิม + DIFF threshold · net_pv = E แบบ mark-to-market (≥ 0 จาก x − 1 ≥ ln x)"""
    prices = np.asarray(prices, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.int8)
    fixed_asset_value = float(fixed_asset_value)
    starting_cash = float(starting_cash)
    diff = float(diff)

    if prices.ndim != 1 or actions.ndim != 1 or len(prices) != len(actions):
        raise ValueError("prices and actions must be one-dimensional arrays of equal length")
    if len(prices) == 0:
        return pd.DataFrame()
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("all prices must be finite and greater than zero")
    if not np.isfinite(fixed_asset_value) or fixed_asset_value < 0:
        raise ValueError("Fixed_Asset_Value must be finite and non-negative")
    if not np.isfinite(starting_cash):
        raise ValueError("Cash_Balan must be finite")
    if not np.isfinite(diff) or diff < 0:
        raise ValueError("diff must be finite and non-negative")

    actions = (actions != 0).astype(np.int8)
    actions[0] = 0
    re, amount = constant_value_core(prices, actions, fixed_asset_value, diff)

    cash_balance = starting_cash + np.cumsum(re)
    portfolio_value = cash_balance + amount * prices
    reference_cash_flow = fixed_asset_value * np.log(prices / prices[0])
    reference_portfolio_value = starting_cash + fixed_asset_value + reference_cash_flow
    net_pv = portfolio_value - reference_portfolio_value

    net_pv[np.isclose(net_pv, 0.0, atol=1e-10, rtol=0.0)] = 0.0
    if np.min(net_pv) < -1e-7:
        raise ArithmeticError("log-reference invariant failed: net_pv must be >= 0")

    return pd.DataFrame({
        "re": re,
        "Cash_Balan": cash_balance,
        "Amount_Asset": amount,
        "pv": portfolio_value,
        "refer_cash_flow": reference_cash_flow,
        "refer_pv": reference_portfolio_value,
        "net_pv": net_pv,
    })


def aggregate_portfolio(results):
    """results = {ticker: DataFrame[re, net_pv]} → ตารางรวมแบบหน้าเดิม (re = flow → 0 · net_pv = state → ffill)"""
    all_re, all_net = [], []
    for ticker, result in results.items():
        if result is None or result.empty:
            continue
        all_re.append(result[["re"]].rename(columns={"re": f"{ticker}_re"}))
        all_net.append(result[["net_pv"]].rename(columns={"net_pv": f"{ticker}_net_pv"}))
    if not all_re:
        return pd.DataFrame()
    df_re = pd.concat(all_re, axis=1).sort_index().fillna(0.0)
    df_net = pd.concat(all_net, axis=1).sort_index().ffill().fillna(0.0)
    df_re["portfolio_cash_flow"] = df_re.sum(axis=1)
    df_re["cumulative_cash_flow"] = df_re["portfolio_cash_flow"].cumsum()
    df_net["portfolio_excess_profit"] = df_net.sum(axis=1)
    return pd.concat([df_re, df_net], axis=1)


def running_floor(cumulative_cash):
    """พื้นเงินสด = min(0, ค่าต่ำสุดสะสม) — ตัวเดียวกับ Cash Burn ของ LEGO"""
    cumulative_cash = np.asarray(cumulative_cash, dtype=np.float64)
    return np.minimum.accumulate(np.minimum(cumulative_cash, 0.0))


def capital_at_risk_path(sum_fixed_asset_value, cumulative_cash):
    """ทุนที่ใช้จริง ณ เวลา t = ΣFAV + เงินสำรองที่ใช้ไปแล้วถึง t (ไม่ใช้ค่าในอนาคต)"""
    return float(sum_fixed_asset_value) - running_floor(cumulative_cash)


def annualized_vol(prices, lookback=None):
    """σ ต่อปีจากผลตอบแทนรายแท่งแบบง่าย · lookback = จำนวนผลตอบแทนล่าสุด (None = ทั้งช่วง)"""
    prices = np.asarray(prices, dtype=np.float64)
    if len(prices) < 3:
        return float("nan")
    returns = prices[1:] / prices[:-1] - 1.0
    if lookback:
        returns = returns[-int(lookback):]
    if len(returns) < 2:
        return float("nan")
    return float(np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS))


def risk_parity_allocation(vols, budget):
    """FAVᵢ = budget · (1/σᵢ) / Σ(1/σⱼ) → FAVᵢ·σᵢ เท่ากันทุกตัว · ตัวที่ไม่มี σ ไม่ได้เงิน"""
    inverse = {t: 1.0 / v for t, v in vols.items() if v is not None and np.isfinite(v) and v > 0}
    total = sum(inverse.values())
    if total <= 0:
        return {}
    return {t: float(budget) * w / total for t, w in inverse.items()}


def equal_allocation(tickers, budget):
    tickers = list(tickers)
    if not tickers:
        return {}
    return {t: float(budget) / len(tickers) for t in tickers}


def correlation_summary(returns):
    """(ρ matrix, ρ เฉลี่ยทุกคู่, effective bets = (Σλ)²/Σλ²)"""
    corr = returns.corr()
    n = corr.shape[0]
    if n == 0:
        return corr, float("nan"), float("nan")
    if n == 1:
        return corr, float("nan"), 1.0
    values = np.nan_to_num(corr.to_numpy(dtype=np.float64), nan=0.0)
    np.fill_diagonal(values, 1.0)
    upper = values[np.triu_indices(n, 1)]
    eigen = np.clip(np.linalg.eigvalsh(values), 0.0, None)
    effective = float(eigen.sum() ** 2 / (eigen ** 2).sum()) if (eigen ** 2).sum() > 0 else float("nan")
    return corr, float(np.mean(upper)), effective


def dalio_risk_ratio(avg_rho, n):
    """σ พอร์ต / σ ตัวเดียว = √(ρ + (1 − ρ)/N) (ทุกตัวแกว่งเท่ากัน น้ำหนักเท่ากัน) · พื้นต่ำสุด √ρ"""
    if n < 1 or not np.isfinite(avg_rho):
        return float("nan")
    value = avg_rho + (1.0 - avg_rho) / n
    return float(np.sqrt(max(value, 0.0)))


def diversification_ratio(portfolio_floor, ticker_floors):
    """พื้นพอร์ต ÷ ผลรวมพื้นรายตัว · 1 = ไม่หักล้างเลย · 0 = หักล้างหมด · ไม่มีตัวไหนติดลบ → nan"""
    total = float(sum(ticker_floors))
    if total >= 0:
        return float("nan")
    return float(portfolio_floor) / total


def trap_flags(tickers, corr=None, threshold=INVERSE_CORR_THRESHOLD):
    """{ticker: [เหตุผล]} สำหรับตัวที่ E อาจไม่ใช่เงินจริง: leveraged/inverse ETF หรือ ρ กับอีกตัว ≤ threshold"""
    flags = {}
    for ticker in tickers:
        reasons = []
        if str(ticker).upper().split("-")[0] in LEVERAGED_OR_INVERSE:
            reasons.append("ETF leveraged/inverse: ปรับพอร์ตทุกวันแบบซื้อแพงขายถูก E ที่ได้ถูกหักใน R")
        if corr is not None and ticker in corr.index:
            for other in tickers:
                if other == ticker or other not in corr.columns:
                    continue
                rho = corr.loc[ticker, other]
                if np.isfinite(rho) and rho <= threshold:
                    reasons.append(f"ρ กับ {other} = {rho:.2f}: เดินกลับทิศ ทิศทางถูกลบทิ้ง E เพิ่มแต่เงินสดไม่โต")
        if reasons:
            flags[ticker] = reasons
    return flags


def projected_annual_excess(favs, vols):
    """E ต่อปีที่คาด ≈ Σ ½·FAVᵢ·σᵢ² (σ ต่อปี) — E โตตาม σ² จึงต้องใช้ σ ล่าสุด"""
    total = 0.0
    for ticker, fav in favs.items():
        vol = vols.get(ticker)
        if vol is None or not np.isfinite(vol):
            return float("nan")
        total += 0.5 * float(fav) * vol * vol
    return float(total)


def rolling_window_stats(price_frame, favs, horizon, step=5, diff_pct=0.0):
    """เริ่ม chain ใหม่ทุก ``step`` แท่ง ยาว ``horizon`` แท่ง (act ทุกแท่ง + DIFF) → พื้น / E / A ต่อหน้าต่าง

    price_frame: ราคาปิดที่เรียงวันตรงกัน (คอลัมน์ = ticker) · ใช้เฉพาะแถวที่ทุกตัวมีราคา
    """
    tickers = [t for t in price_frame.columns if float(favs.get(t, 0.0)) > 0]
    columns = ["start", "end", "floor", "sum_ticker_floors", "excess", "cash"]
    if not tickers:
        return pd.DataFrame(columns=columns)
    frame = price_frame[tickers].dropna()
    n = len(frame)
    if n < 2:
        return pd.DataFrame(columns=columns)
    horizon = int(min(max(int(horizon), 1), n - 1))
    step = max(int(step), 1)
    values = {t: frame[t].to_numpy(dtype=np.float64) for t in tickers}
    rows = []
    for start in range(0, n - horizon, step):
        total = np.zeros(horizon + 1)
        excess = 0.0
        floors_sum = 0.0
        for ticker in tickers:
            prices = values[ticker][start:start + horizon + 1]
            fav = float(favs[ticker])
            re, amount = constant_value_core(prices, np.ones(len(prices), dtype=np.int8), fav, diff_pct * fav)
            cum = np.cumsum(re)
            total += cum
            floors_sum += min(0.0, float(cum.min()))
            excess += float(cum[-1] + amount[-1] * prices[-1] - fav - fav * np.log(prices[-1] / prices[0]))
        rows.append({
            "start": frame.index[start], "end": frame.index[start + horizon],
            "floor": min(0.0, float(total.min())), "sum_ticker_floors": floors_sum,
            "excess": excess, "cash": float(total[-1]),
        })
    return pd.DataFrame(rows, columns=columns)


DETAIL_PERIOD_OFFSETS = {
    "1M": pd.DateOffset(months=1),
    "3M": pd.DateOffset(months=3),
    "6M": pd.DateOffset(months=6),
    "1Y": pd.DateOffset(years=1),
}


def build_detailed_simulation_frame(data, successful_tickers, detail_period="ALL"):
    """ตัดช่วงแล้ว rebase ให้เริ่มที่ 0 (เหมือนหน้าเดิม) · เพิ่มพื้นเงินสดรายตัวและของพอร์ต"""
    if data.empty:
        return data.copy()
    if detail_period == "ALL":
        frame = data.copy()
    else:
        if detail_period not in DETAIL_PERIOD_OFFSETS:
            raise ValueError(f"Unsupported detail_period: {detail_period}")
        cutoff = data.index.max() - DETAIL_PERIOD_OFFSETS[detail_period]
        frame = data.loc[data.index >= cutoff].copy()
    if frame.empty:
        return frame
    for ticker in successful_tickers:
        re_col, floor_col, net_col = f"{ticker}_re", f"{ticker}_Running_Cash_Floor", f"{ticker}_net_pv"
        if re_col in frame.columns:
            cum = frame[re_col].cumsum()
            cum = cum - float(cum.iloc[0])
            frame[re_col] = cum
            frame[floor_col] = running_floor(cum.to_numpy(dtype=np.float64))
        if net_col in frame.columns:
            rebased = (frame[net_col] - float(frame[net_col].iloc[0])).to_numpy(dtype=np.float64, copy=True)
            rebased[np.isclose(rebased, 0.0, atol=1e-10, rtol=0.0)] = 0.0
            frame[net_col] = rebased
    if "cumulative_cash_flow" in frame.columns:
        cum = frame["cumulative_cash_flow"] - float(frame["cumulative_cash_flow"].iloc[0])
        frame["cumulative_cash_flow"] = cum
        frame["Portfolio_Running_Cash_Floor"] = running_floor(cum.to_numpy(dtype=np.float64))
    if "portfolio_excess_profit" in frame.columns:
        rebased = (frame["portfolio_excess_profit"] - float(frame["portfolio_excess_profit"].iloc[0])).to_numpy(
            dtype=np.float64, copy=True)
        rebased[np.isclose(rebased, 0.0, atol=1e-10, rtol=0.0)] = 0.0
        frame["portfolio_excess_profit"] = rebased
    return frame


# --------------------------------
# Data + UI
# --------------------------------
def load_config(filename="un15_fx_config.json"):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        st.error(f"Error: Configuration file '{filename}' not found.")
        return {}, {}
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from '{filename}'. Please check its format.")
        return {}, {}
    fallback_default = {
        "Fixed_Asset_Value": 1500.0, "Cash_Balan": 650.0, "step": 0.01,
        "filter_date": "2024-01-01 12:00:00+07:00", "pred": 1,
    }
    default_config = data.pop("__DEFAULT_CONFIG__", fallback_default)
    return data, default_config


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_close(ticker, filter_date, auto_adjust):
    """ราคาปิดตั้งแต่ filter_date (เวลา Asia/Bangkok) · auto_adjust=False = ราคาจริง (ปรับ split ไม่ปรับปันผล)"""
    hist = yf.Ticker(ticker).history(period="max", auto_adjust=auto_adjust)
    if hist.empty:
        return pd.Series(dtype=np.float64)
    if hist.index.tz is None:
        hist.index = hist.index.tz_localize("UTC").tz_convert("Asia/Bangkok")
    else:
        hist.index = hist.index.tz_convert("Asia/Bangkok")
    start = pd.Timestamp(filter_date)
    start = start.tz_localize("Asia/Bangkok") if start.tzinfo is None else start.tz_convert("Asia/Bangkok")
    close = hist.loc[hist.index >= start, "Close"].astype(np.float64)
    return close[np.isfinite(close) & (close > 0)]


def _money(value):
    return "—" if value is None or not np.isfinite(value) else f"{value:,.2f}"


def _pct(value, digits=2):
    return "—" if value is None or not np.isfinite(value) else f"{value:,.{digits}f}%"


def _ratio(value):
    return "—" if value is None or not np.isfinite(value) else f"{value:.2f}"


def ticker_controls(full_config, default_config):
    """เพิ่ม / เลือก ticker (session state แยกจากหน้าเดิมด้วย prefix un15r_)"""
    custom = st.session_state.setdefault("un15r_custom_tickers", {})
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Add New Ticker")
        new_ticker = st.text_input("Ticker (e.g., AAPL):", key="un15r_new_ticker").strip().upper()
        if st.button("Add Ticker", key="un15r_add_ticker", width="stretch"):
            if not new_ticker:
                st.warning("Please enter a ticker symbol.")
            elif new_ticker in full_config or new_ticker in custom:
                st.warning(f"{new_ticker} already exists.")
            else:
                custom[new_ticker] = {"Ticker": new_ticker, **default_config}
                st.success(f"Added {new_ticker}!")
                st.rerun()
    with c2:
        st.subheader("Select Tickers to Analyze")
        options = list(full_config.keys()) + list(custom.keys())
        selected = st.multiselect("Select from available tickers:", options=options,
                                  default=list(full_config.keys()), key="un15r_selected")
    return {t: dict(full_config.get(t) or custom.get(t)) for t in selected}


def main():
    st.set_page_config(page_title="Un_15_F(X) แนะนำ", page_icon="🚀", layout="wide")
    st.title("🚀 Un_15 F(X) · แนะนำ")
    st.caption(
        "เครื่องยนต์เดิมของ Un_15 (constant value + log reference) + ข้อแนะนำ: A guard · Correlation · "
        "Rolling start · Risk parity · E projection จาก σ · Survival test · ไม่มี look-ahead"
    )

    full_config, default_config = load_config()
    if not (full_config or default_config):
        st.error("Could not load any configuration. Please check that 'un15_fx_config.json' exists.")
        return
    active = ticker_controls(full_config, default_config)
    if not active:
        st.warning("Please select at least one ticker to start the analysis.")
        return

    with st.expander("⚙️ ตั้งค่า (ข้อแนะนำ 3, 4, 7)", expanded=True):
        s1, s2, s3 = st.columns(3)
        sizing = s1.radio("แบ่งเงิน FAV", [SIZING_RISK_PARITY, SIZING_EQUAL, SIZING_MANUAL], key="un15r_sizing",
                          help="Risk parity: FAVᵢ ∝ 1/σᵢ ให้ทุกตัวรับความเสี่ยงเท่ากันภายในงบรวมเดิม")
        budget = float(s1.number_input("งบรวม ΣFAV (USD)", min_value=0.0, value=FIXED_ASSET_RESET_USD, step=100.0,
                                       key="un15r_budget", disabled=sizing == SIZING_MANUAL))
        lookback_label = s2.selectbox("σ lookback", ["60 แท่ง", "126 แท่ง", "252 แท่ง", "ทั้งช่วง"], key="un15r_lookback")
        lookback = None if lookback_label == "ทั้งช่วง" else int(lookback_label.split()[0])
        diff_pct = float(s2.number_input(
            "DIFF (% ของ FAV)", min_value=0.0, max_value=50.0, value=0.0, step=0.5, key="un15r_diff",
            help=f"0 = รีบาลานซ์ทุกแท่งแบบหน้าเดิม · bot LEGO ใช้ DIFF 50 บน 1,500 = {LEGO_DIFF_PCT:.2f}%")) / 100.0
        horizon = int(s3.selectbox("Rolling horizon (แท่ง)", [63, 126, 252], index=1, key="un15r_horizon"))
        roll_step = int(s3.number_input("Rolling step (แท่ง)", min_value=1, max_value=63, value=5, key="un15r_step"))
        auto_adjust = st.checkbox("ใช้ราคาปรับปันผล (auto_adjust) แบบหน้าเดิม", value=False, key="un15r_adjust",
                                  help="ปิด = ราคาจริงแบบที่ bot เห็น (ปรับ split แต่ไม่ปรับปันผล)")

    # ---- prices ----
    closes, failed = {}, []
    with st.spinner("Downloading prices..."):
        for ticker, cfg in active.items():
            try:
                series = fetch_close(cfg.get("Ticker", ticker), cfg.get("filter_date", default_config.get("filter_date")),
                                     auto_adjust)
            except Exception as exc:  # network / symbol errors → skip that ticker
                st.warning(f"{ticker}: download failed ({exc})")
                series = pd.Series(dtype=np.float64)
            if len(series) >= 3:
                closes[ticker] = series
            else:
                failed.append(ticker)
    if failed:
        st.warning("ไม่มีข้อมูลราคาพอ: " + ", ".join(failed))
    if not closes:
        st.error("Failed to generate data.")
        return

    vols = {t: annualized_vol(s.to_numpy(), lookback) for t, s in closes.items()}
    if sizing == SIZING_RISK_PARITY:
        favs = risk_parity_allocation(vols, budget)
    elif sizing == SIZING_EQUAL:
        favs = equal_allocation(closes.keys(), budget)
    else:
        favs = {}
        with st.expander("Per-Ticker Fixed_Asset_Value", expanded=True):
            cols = st.columns(min(3, len(closes)))
            for i, ticker in enumerate(closes):
                start = float(active[ticker].get("Fixed_Asset_Value", default_config.get("Fixed_Asset_Value",
                                                                                          FIXED_ASSET_RESET_USD)))
                start = min(max(start, 0.0), 5000.0)
                favs[ticker] = float(cols[i % len(cols)].slider(f"FAV — {ticker}", 0.0, 5000.0, start, 1.0,
                                                                 key=f"un15r_fav_{ticker}"))
    favs = {t: favs.get(t, 0.0) for t in closes}

    # ---- simulate ----
    results, sim_errors = {}, []
    for ticker, series in closes.items():
        cfg = active[ticker]
        try:
            prices = series.to_numpy(dtype=np.float64)
            sim = simulate_constant_value_path(prices, build_actions(cfg.get("pred", 1), len(prices)), favs[ticker],
                                               cfg.get("Cash_Balan", default_config.get("Cash_Balan", 0.0)),
                                               diff=diff_pct * favs[ticker])
            sim.index = series.index
            results[ticker] = sim[["re", "net_pv"]]
        except Exception as exc:
            sim_errors.append(f"{ticker}: {exc}")
    for message in sim_errors:
        st.warning(f"simulation skipped — {message}")
    data = aggregate_portfolio(results)
    if data.empty:
        st.error("Failed to generate data.")
        return
    tickers = [t for t in results if f"{t}_net_pv" in data.columns]
    sum_fav = float(sum(favs[t] for t in tickers))
    cash_budget = float(sum(float(active[t].get("Cash_Balan", default_config.get("Cash_Balan", 0.0))) for t in tickers))

    cum_cash = data["cumulative_cash_flow"].to_numpy(dtype=np.float64)
    excess = data["portfolio_excess_profit"].to_numpy(dtype=np.float64)
    floor_path = running_floor(cum_cash)
    max_buffer = abs(float(floor_path.min()))
    capital_path = capital_at_risk_path(sum_fav, cum_cash)
    capital = sum_fav + max_buffer
    ticker_floors = {t: float(running_floor(data[f"{t}_re"].cumsum().to_numpy()).min()) for t in tickers}
    div_ratio = diversification_ratio(float(floor_path.min()), ticker_floors.values())

    aligned = pd.DataFrame({t: closes[t] for t in tickers}).dropna()
    returns = aligned.pct_change().dropna()
    corr, avg_rho, eff_bets = correlation_summary(returns)
    flags = trap_flags(tickers, corr if len(tickers) > 1 else None)
    rolling = rolling_window_stats(aligned, favs, horizon, roll_step, diff_pct)

    final_e = float(excess[-1])
    final_a = float(cum_cash[-1])
    harvest = 100.0 * final_e / capital if capital > 0 else float("nan")
    cash_return = 100.0 * final_a / capital if capital > 0 else float("nan")
    survival = cash_budget - max_buffer

    # ---- 1. A guard ----
    if flags:
        st.error("⚠️ **E อาจไม่ใช่เงินจริง** — " + " · ".join(f"**{t}**: {'; '.join(r)}" for t, r in flags.items()))

    st.subheader("Key Performance Indicators")
    k = st.columns(4)
    k[0].metric("ΣE · Excess vs Log Ref", _money(final_e), help="ส่วนเกินเหนือ FAV·ln(P/P₀) — ไม่ใช่เงินสดที่ถอนได้")
    k[1].metric("ΣA · เงินสดจริง", _money(final_a), help="ผลรวมเงินสดที่เกิดจากการรีบาลานซ์ = R + E")
    k[2].metric("Max Cash Buffer Required", _money(max_buffer))
    k[3].metric("Capital at Risk", _money(capital), help="ΣFAV + เงินสำรองที่ต้องใช้จริง")
    k = st.columns(5)
    k[0].metric("Harvest Return (E/ทุน)", _pct(harvest))
    k[1].metric("Cash Return (A/ทุน)", _pct(cash_return), help="ถ้า Harvest สูงแต่ Cash ต่ำมาก = สัญญาณ E ปลอม")
    k[2].metric("Diversification ratio", _ratio(div_ratio),
                help="พื้นพอร์ต ÷ ผลรวมพื้นรายตัว · 1 = ไม่หักล้าง · ยิ่งต่ำยิ่งหักล้างกันมาก")
    floor_p5 = float(rolling["floor"].quantile(0.05)) if len(rolling) else float("nan")
    k[3].metric(f"Floor p5 (rolling {horizon})", _money(floor_p5),
                help=f"{len(rolling)} หน้าต่าง เริ่มทุก {roll_step} แท่ง · ค่ากลาง "
                     f"{_money(float(rolling['floor'].median()) if len(rolling) else float('nan'))}")
    k[4].metric("Survival (ΣCash_Balan − buffer)", _money(survival),
                delta="รอด" if survival >= 0 else "เงินสดหมดก่อน", delta_color="normal" if survival >= 0 else "inverse")

    # ---- per ticker ----
    st.subheader("รายตัว")
    rows = []
    for t in tickers:
        a_t = float(data[f"{t}_re"].sum())
        e_t = float(data[f"{t}_net_pv"].iloc[-1])
        others = [corr.loc[t, u] for u in tickers if u != t and t in corr.index and u in corr.columns]
        rows.append({
            "Ticker": t, "FAV": favs[t], "σ ต่อปี": vols.get(t), "FAV·σ": favs[t] * vols.get(t, np.nan),
            "ρ เฉลี่ยกับตัวอื่น": float(np.nanmean(others)) if others else np.nan,
            "E": e_t, "A": a_t, "R = A − E": a_t - e_t, "Floor": ticker_floors[t],
            "E / |Floor|": e_t / abs(ticker_floors[t]) if ticker_floors[t] < 0 else np.nan,
            "ธง": "⚠️" if t in flags else "",
        })
    table = pd.DataFrame(rows).set_index("Ticker")
    for col in table.columns:
        if col != "ธง":
            table[col] = [_money(float(v)) if v is not None else "—" for v in table[col]]
    st.dataframe(table, width="stretch")
    st.caption("E / |Floor| โตตาม σ: ตัวที่แกว่งแรงเก็บ E ต่อความเสี่ยงได้มากกว่า แต่ถ้าแกว่งเกิน ~2 เท่าของตัวอื่น "
               "จะทำให้พื้นของพอร์ตลึกขึ้น · R = A − E = FAV·ln(P_acted/P₀) คือส่วนของทิศทางราคา")

    # ---- 2. correlation ----
    st.subheader("Correlation (ข้อแนะนำ 2)")
    if len(tickers) < 2:
        st.info("ต้องมีอย่างน้อย 2 ตัวจึงจะวัด correlation ได้")
    else:
        c = st.columns(4)
        c[0].metric("ρ เฉลี่ยทุกคู่", _ratio(avg_rho))
        c[1].metric("Effective bets", f"{eff_bets:.1f} / {len(tickers)}", help="(Σλ)²/Σλ² จาก eigenvalue ของ ρ matrix")
        c[2].metric("σ พอร์ต / σ ตัวเดียว (Dalio)", _ratio(dalio_risk_ratio(avg_rho, len(tickers))),
                    help="√(ρ + (1 − ρ)/N) · ทุกตัวแกว่งเท่ากัน น้ำหนักเท่ากัน")
        c[3].metric("พื้นต่ำสุดที่เป็นไปได้ √ρ", _ratio(np.sqrt(max(avg_rho, 0.0))))
        fig = px.imshow(corr.round(2), text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                        aspect="auto", title=f"ρ ของผลตอบแทนรายแท่ง ({len(returns)} แท่งที่ทุกตัวมีราคา)")
        st.plotly_chart(fig, width="stretch")
        st.caption("E = Σ ½·FAVᵢ·σᵢ²·t ไม่มี ρ — ρ เปลี่ยนแค่ความลึกของพื้นเงินสด · ลด ρ ให้ผลมากกว่าเพิ่มจำนวนตัว")

    # ---- 3. rolling ----
    st.subheader(f"Rolling start (ข้อแนะนำ 3) · เริ่มใหม่ทุก {roll_step} แท่ง ยาว {horizon} แท่ง")
    if rolling.empty:
        st.info("ข้อมูลไม่พอสำหรับ rolling")
    else:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=rolling["start"], y=rolling["floor"], name="พื้นเงินสดของพอร์ต",
                             hovertemplate="เริ่ม %{x|%d %b %Y}<br>พื้น %{y:,.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=rolling["start"], y=rolling["excess"], name="E ของหน้าต่าง", mode="lines"))
        fig.add_hline(y=floor_p5, line_dash="dash", annotation_text="p5")
        if cash_budget > 0:
            fig.add_hline(y=-cash_budget, line_dash="dot", line_color="red", annotation_text="−ΣCash_Balan")
        fig.update_layout(height=420, hovermode="x unified", legend={"orientation": "h", "y": -0.2},
                          yaxis_title="USD", title="พื้นเงินสดขึ้นกับวันเริ่ม — ดูทั้งการกระจาย ไม่ใช่เส้นเดียว")
        st.plotly_chart(fig, width="stretch")
        breach = float((rolling["floor"] < -cash_budget).mean()) if cash_budget > 0 else float("nan")
        roll_div = (rolling["floor"] / rolling["sum_ticker_floors"]).where(rolling["sum_ticker_floors"] < 0)
        st.caption(
            f"พื้น: แย่สุด {_money(float(rolling['floor'].min()))} · p5 {_money(floor_p5)} · "
            f"ค่ากลาง {_money(float(rolling['floor'].median()))} · ดีสุด {_money(float(rolling['floor'].max()))} · "
            f"หน้าต่างที่พื้นลึกกว่า ΣCash_Balan: {_pct(100 * breach, 0)} · "
            f"Diversification ratio ค่ากลาง {_ratio(float(roll_div.median()))} · rolling ใช้ act ทุกแท่ง + DIFF "
            "(ไม่ใช้ DNA pred)")

    # ---- 5. projection ----
    st.subheader("E ต่อปีที่คาด (ข้อแนะนำ 5)")
    proj_rows = []
    for label, lb in (("σ 60 แท่ง", 60), ("σ 252 แท่ง", 252), ("σ ทั้งช่วง", None)):
        v = {t: annualized_vol(closes[t].to_numpy(), lb) for t in tickers}
        proj_rows.append({"ฐาน": label, "E ต่อปี ≈ Σ ½·FAV·σ²": projected_annual_excess({t: favs[t] for t in tickers}, v)})
    n_bars = max(len(data) - 1, 1)
    proj_rows.append({"ฐาน": "ค่าเฉลี่ยในอดีต × 252 (หน้าเดิม)", "E ต่อปี ≈ Σ ½·FAV·σ²": final_e / n_bars * TRADING_DAYS})
    proj = pd.DataFrame([{k: (_money(v) if isinstance(v, float) else v) for k, v in row.items()} for row in proj_rows])
    st.dataframe(proj.set_index("ฐาน"), width="stretch")
    st.caption("ทฤษฎีแบบต่อเนื่อง ใช้เทียบขนาด · E ∝ σ² — vol เปลี่ยน 2 เท่า E เปลี่ยน 4 เท่า")

    # ---- charts (no look-ahead) ----
    st.subheader("Performance Charts")
    g1, g2 = st.columns(2)
    df_all = pd.DataFrame({"ΣE · Excess": excess, "ΣA · Cash": cum_cash, "Running Cash Floor": floor_path},
                          index=data.index)
    line_colors = {"ΣE · Excess": "#16A34A", "ΣA · Cash": "#2563EB", "Running Cash Floor": "#DC2626",
                   "Harvest Return % (E)": "#16A34A", "Cash Return % (A)": "#2563EB"}
    g1.plotly_chart(px.line(df_all, title="ΣE vs ΣA vs Running Cash Floor", color_discrete_map=line_colors,
                            labels={"value": "USD", "variable": "", "index": ""}), width="stretch")
    with np.errstate(divide="ignore", invalid="ignore"):
        df_ret = pd.DataFrame({
            "Harvest Return % (E)": np.where(capital_path > 0, 100.0 * excess / capital_path, np.nan),
            "Cash Return % (A)": np.where(capital_path > 0, 100.0 * cum_cash / capital_path, np.nan),
        }, index=data.index)
    g2.plotly_chart(px.line(df_ret, title="Return on capital ณ เวลานั้น (ไม่ใช้ buffer ในอนาคต)",
                            color_discrete_map=line_colors, labels={"value": "%", "variable": "", "index": ""}),
                    width="stretch")

    # ---- original MIRR, labelled ----
    with st.expander("Scenario MIRR (สูตรหน้าเดิม · ใช้ E เป็นกระแสเงินสด)", expanded=False):
        st.caption("E คือส่วนเกินเหนือ log reference ไม่ใช่เงินที่ถอนได้ — ดู Cash Return (A/ทุน) ประกอบเสมอ")
        m1, m2, m3 = st.columns(3)
        years = int(m1.number_input("Projection years", 1, 10, 3, key="un15r_mirr_years"))
        days = int(m2.number_input("Trading days/year", 1, 366, TRADING_DAYS, key="un15r_mirr_days"))
        recovery = float(m3.slider("Capital recovered at exit (%)", 0.0, 100.0, 50.0, 1.0, key="un15r_mirr_exit"))
        annual = final_e / n_bars * days
        flows = [-capital] + [annual] * max(years - 1, 0) + [annual + capital * recovery / 100.0]
        mirr = npf.mirr(flows, 0.0, 0.0) if capital > 0 else float("nan")
        st.metric(f"Scenario MIRR ({years}-Year)", f"{mirr:.2%}" if np.isfinite(mirr) else "N/A")

    # ---- detail ----
    st.subheader("Detailed Simulation Data")
    v1, v2 = st.columns([2, 1])
    view = v1.selectbox("View", ["Portfolio Overview", "Cash Flow by Ticker", "Cash Floor by Ticker",
                                 "Excess Profit by Ticker"], key="un15r_view")
    period = v2.selectbox("Period", ["ALL", "1Y", "6M", "3M", "1M"], key="un15r_period")
    frame = build_detailed_simulation_frame(data, tickers, period)
    if frame.empty:
        st.info("No simulation data available for the selected period.")
    else:
        fig = go.Figure()
        if view == "Portfolio Overview":
            for col, name, dash, color in (("cumulative_cash_flow", "ΣA · Cumulative Cash", "solid", "#2563EB"),
                                           ("Portfolio_Running_Cash_Floor", "Running Cash Floor", "dash", "#DC2626"),
                                           ("portfolio_excess_profit", "ΣE · Excess", "dot", "#16A34A")):
                fig.add_trace(go.Scatter(x=frame.index, y=frame[col], name=name, line={"dash": dash, "color": color}))
        else:
            suffix = {"Cash Flow by Ticker": "_re", "Cash Floor by Ticker": "_Running_Cash_Floor",
                      "Excess Profit by Ticker": "_net_pv"}[view]
            for t in tickers:
                fig.add_trace(go.Scatter(x=frame.index, y=frame[f"{t}{suffix}"], name=t))
        fig.add_hline(y=0, line_width=1, line_dash="dot")
        fig.update_layout(height=520, hovermode="x unified", yaxis_title="USD", legend={"orientation": "h", "y": -0.15},
                          title=f"{view} · {period} (rebase = 0 ที่ต้นช่วง)")
        st.plotly_chart(fig, width="stretch")

    st.json({
        "reference_model": {"formula": "R_n = FAV × ln(P_n / P_0)", "invariant": "net_pv >= 0"},
        "sizing": sizing, "favs": {t: round(favs[t], 2) for t in tickers}, "diff_pct": round(diff_pct * 100, 4),
        "sum_fixed_asset_value": round(sum_fav, 2), "cash_budget_sum_Cash_Balan": round(cash_budget, 2),
        "max_cash_buffer_required": round(max_buffer, 2), "capital_at_risk": round(capital, 2),
        "excess_E": round(final_e, 2), "cash_A": round(final_a, 2),
        "diversification_ratio": None if not np.isfinite(div_ratio) else round(div_ratio, 4),
        "avg_rho": None if not np.isfinite(avg_rho) else round(avg_rho, 4),
        "effective_bets": None if not np.isfinite(eff_bets) else round(eff_bets, 2),
        "rolling": {"horizon": horizon, "step": roll_step, "windows": int(len(rolling)),
                    "floor_p5": None if not np.isfinite(floor_p5) else round(floor_p5, 2)},
        "flags": flags,
    }, expanded=False)


if __name__ == "__main__":
    main()
