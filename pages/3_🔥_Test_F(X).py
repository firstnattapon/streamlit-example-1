# app_advanced_fx.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import math, json, io
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt

# =========================
# 0) Streamlit Page Config
# =========================
st.set_page_config(page_title="Advanced ln(F) Engine (no Altair)", page_icon="🧮", layout="wide")

# =========================
# 1) Utilities
# =========================
@st.cache_data(show_spinner=False)
def get_prices(tickers: Iterable[str], start: str = "2024-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """ดึงราคาปิดจาก Yahoo Finance รวมเป็น DataFrame (คอลัมน์ต่อตัว)"""
    frames = []
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(start=start, end=end)
            if hist.empty or "Close" not in hist:
                continue
            frames.append(hist[["Close"]].rename(columns={"Close": t}))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1).ffill().dropna(how="all")
    return df

def ewma_vol(series: pd.Series, halflife: int = 20) -> float:
    """EWMA std ของ ln-return ณ จุดล่าสุด"""
    s = series.dropna()
    if s.size < 3: return float("nan")
    r = np.log(s).diff()
    v = r.ewm(halflife=halflife, adjust=False).std().iloc[-1]
    return float(v) if pd.notna(v) else float("nan")

def rolling_ln_sigma(series: pd.Series, window: int = 20) -> float:
    """rolling std ของ ln-return ณ จุดล่าสุด (ใช้กับ no-trade band)"""
    r = np.log(series).diff()
    sig = r.rolling(window=window, min_periods=max(3, window//2)).std().iloc[-1]
    return float(sig) if pd.notna(sig) else 0.0

def to_csv_download(df: pd.DataFrame, name: str) -> Tuple[bytes, str]:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8"), f"{name}.csv"

# =========================
# 2) Core Ideas / Math
# =========================
def b_scale_inverse_vol(prices_df: pd.DataFrame, halflife: int = 20, floor: float = 0.5, cap: float = 1.5) -> Dict[str, float]:
    """Risk-budgeted b จาก inverse-vol และ normalize เฉลี่ย ≈ 1 พร้อม clip [floor, cap]"""
    vols = {c: ewma_vol(prices_df[c], halflife=halflife) for c in prices_df.columns}
    inv = {c: (1.0/v) if (v and v == v and v > 0) else np.nan for c, v in vols.items()}
    s = pd.Series(inv).dropna()
    if s.empty:
        return {c: 1.0 for c in prices_df.columns}
    scaled = (s / s.mean()).clip(lower=floor, upper=cap)
    return {c: float(scaled.get(c, 1.0)) for c in prices_df.columns}

def regime_mult(prices_df: pd.DataFrame, short_hl: int = 10, long_hl: int = 60,
                low: float = 0.7, high: float = 1.3, mult_low: float = 1.2, mult_high: float = 0.8) -> Dict[str, float]:
    """ตัวคูณตาม vol regime: ratio = vol_short/vol_long"""
    out = {}
    for c in prices_df.columns:
        vs = ewma_vol(prices_df[c], halflife=short_hl)
        vl = ewma_vol(prices_df[c], halflife=long_hl)
        if not vs or not vl or vs != vs or vl != vl or vl <= 0:
            out[c] = 1.0
            continue
        ratio = vs / vl
        if ratio < low:   out[c] = mult_low     # ตลาดนิ่ง -> ขยาย
        elif ratio > high:out[c] = mult_high    # ผันผวน -> หด
        else:             out[c] = 1.0
    return out

def no_trade_band_clip(ln_value: float, sigma_now: float, k: float) -> float:
    """ถ้า |ln| < k*sigma -> 0 (ไม่ทำอะไร)"""
    return ln_value if abs(ln_value) >= k * sigma_now else 0.0

def cap_portfolio_cf(contrib: Dict[str, float], daily_abs_cap: float) -> Dict[str, float]:
    """จำกัดผลรวม |contribution| ต่อวันไม่เกิน daily_abs_cap โดยคงสัดส่วนเดิม"""
    total_abs = sum(abs(v) for v in contrib.values())
    if daily_abs_cap <= 0 or total_abs <= daily_abs_cap or total_abs == 0:
        return contrib
    scale = daily_abs_cap / total_abs
    return {k: v * scale for k, v in contrib.items()}

@dataclass
class AssetInput:
    ticker: str
    live_price: float
    ref_price: float
    fix_c: float
    b: float  # โหมด "scale": ใช้เป็นตัวคูณบน ln ; โหมด "offset": ใช้เป็น offset

def stress_surface(assets: List[AssetInput], shocks: Iterable[float], mode: str = "scale"
                   ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """คืน (contrib_by_asset, total_contrib) สำหรับ shock แต่ละค่า"""
    rows = []
    for s in shocks:
        row = {}
        total = 0.0
        for a in assets:
            tn = a.live_price * (1.0 + s)
            if tn <= 0 or a.ref_price <= 0:
                val = 0.0
            else:
                ln_part = math.log(tn / a.ref_price)
                if mode == "scale":
                    val = a.fix_c * (a.b * ln_part)
                else:
                    val = a.fix_c * ln_part + a.b
            row[a.ticker] = val
            total += val
        row["_TOTAL_"] = total
        rows.append((s, row))
    contrib = pd.DataFrame({s: r for s, r in rows}).T
    contrib.index.name = "shock"
    total = contrib[["_TOTAL_"]].copy()
    return contrib.drop(columns=["_TOTAL_"]), total

def compute_contrib(live: Dict[str, float], ref: Dict[str, float], fix_map: Dict[str, float],
                    b_param: Dict[str, float], mode: str = "scale") -> Dict[str, float]:
    """f = fix * (b*ln)  (scale)  หรือ  f = fix*ln + b  (offset)"""
    out = {}
    for t in live.keys():
        tn, t0 = float(live[t]), float(ref[t])
        if tn <= 0 or t0 <= 0:
            out[t] = 0.0
            continue
        ln_part = math.log(tn / t0)
        if mode == "scale":
            out[t] = fix_map[t] * (b_param.get(t, 1.0) * ln_part)
        else:
            out[t] = fix_map[t] * ln_part + b_param.get(t, 0.0)
    return out

# =========================
# 3) Sidebar Inputs (UI)
# =========================
st.sidebar.header("⚙️ การตั้งค่า/พารามิเตอร์")

default_assets = [
    {"ticker": "FFWM", "reference_price": 6.88, "fix_c": 1500, "live_price": None},
    {"ticker": "NEGG", "reference_price": 25.20, "fix_c": 1500, "live_price": None},
    {"ticker": "RIVN", "reference_price": 14.20, "fix_c": 1500, "live_price": None},
    {"ticker": "APLS", "reference_price": 39.61, "fix_c": 1500, "live_price": None},
    {"ticker": "NVTS", "reference_price": 8.25, "fix_c": 1500, "live_price": None},
]
default_text = json.dumps(default_assets, ensure_ascii=False, indent=2)

assets_json = st.sidebar.text_area(
    "รายการสินทรัพย์ (JSON: [{ticker, reference_price, fix_c, live_price?}])",
    value=default_text,
    height=220
)

start_date = st.sidebar.date_input("เริ่มดึงราคา (สำหรับคำนวณ vol/regime/band)", value=date(2024, 1, 1))
use_yahoo_live = st.sidebar.checkbox("ดึงราคาปัจจุบันจาก Yahoo อัตโนมัติ", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Risk-Budgeted b")
b_halflife = st.sidebar.number_input("EWMA halflife", min_value=5, max_value=180, value=20)
b_floor = st.sidebar.number_input("b floor", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
b_cap = st.sidebar.number_input("b cap", min_value=0.5, max_value=3.0, value=1.5, step=0.1)

st.sidebar.subheader("Regime Switch")
short_hl = st.sidebar.number_input("short halflife", 5, 120, 10)
long_hl  = st.sidebar.number_input("long halflife", 20, 240, 60)
low_thr  = st.sidebar.number_input("ratio low", 0.3, 1.0, 0.7, 0.05)
high_thr = st.sidebar.number_input("ratio high", 1.0, 2.0, 1.3, 0.05)
mult_low = st.sidebar.number_input("mult @low (นิ่ง)", 0.5, 2.0, 1.2, 0.05)
mult_high= st.sidebar.number_input("mult @high (ผันผวน)", 0.5, 2.0, 0.8, 0.05)

st.sidebar.subheader("No-Trade Band & CF Cap")
band_k   = st.sidebar.number_input("k (band = k·σ_ln)", 0.0, 5.0, 0.5, 0.05)
cf_cap   = st.sidebar.number_input("จำกัด |CF| รวมต่อวัน (0=ปิด)", 0.0, 1e9, 0.0, 100.0)

st.sidebar.subheader("โหมดคำนวณ")
mode = st.sidebar.selectbox("โหมด f", options=["scale", "offset"], index=0)

run_btn = st.sidebar.button("▶️ Run / คำนวณ", use_container_width=True)

# =========================
# 4) Main
# =========================
st.title("🧮 Advanced ln(F) Portfolio Engine (Matplotlib only)")
st.caption("แกน f = b · ln(tₙ/t₀) + Risk-budgeted b + Regime + No-trade band + CF cap + Stress test (ไม่มี Altair)")

def parse_assets_json(txt: str) -> List[Dict]:
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            st.error("⚠️ JSON ต้องเป็น list ของ objects")
            return []
    except Exception as e:
        st.error(f"⚠️ JSON ไม่ถูกต้อง: {e}")
        return []

assets = parse_assets_json(assets_json)
tickers = [a["ticker"] for a in assets if "ticker" in a and a.get("reference_price", 0) > 0]

if run_btn and not tickers:
    st.warning("ใส่สินทรัพย์อย่างน้อย 1 รายการ พร้อม reference_price > 0")

if run_btn and tickers:
    with st.spinner("กำลังคำนวณ..."):
        # 4.1 Maps เบื้องต้น
        ref_map  = {a["ticker"]: float(a["reference_price"]) for a in assets}
        fix_map  = {a["ticker"]: float(a.get("fix_c", 1500.0)) for a in assets}

        # 4.2 ราคาปัจจุบัน
        live_map: Dict[str, float] = {}
        prices_hist = get_prices(tickers, start=str(start_date))
        if use_yahoo_live:
            if prices_hist.empty:
                st.error("ไม่สามารถดึงราคาจาก Yahoo ได้")
                st.stop()
            last_row = prices_hist.iloc[-1]
            for t in tickers:
                live_map[t] = float(last_row.get(t, np.nan))
        else:
            for a in assets:
                v = a.get("live_price", None)
                if v is None:
                    st.error(f"โปรดระบุ live_price สำหรับ {a['ticker']} หรือเลือกใช้ Yahoo Live")
                    st.stop()
                live_map[a["ticker"]] = float(v)

        if prices_hist.empty:
            st.error("ราคาประวัติเปล่า ลองเลือกวันเริ่มต้นที่เก่ากว่านี้")
            st.stop()

        # 4.3 Risk-budgeted b และ Regime
        b_scale = b_scale_inverse_vol(prices_hist, halflife=b_halflife, floor=b_floor, cap=b_cap)
        regime  = regime_mult(prices_hist, short_hl=short_hl, long_hl=long_hl,
                              low=low_thr, high=high_thr, mult_low=mult_low, mult_high=mult_high)
        b_final = {t: b_scale.get(t, 1.0) * regime.get(t, 1.0) for t in tickers}

        # 4.4 คำนวณรายการรายสินทรัพย์
        records = []
        contrib_after_band: Dict[str, float] = {}
        contrib_raw = compute_contrib(live_map, ref_map, fix_map, b_final, mode=mode)

        for t in tickers:
            tn, t0 = live_map[t], ref_map[t]
            ln_now = math.log(tn / t0) if t0 > 0 and tn > 0 else 0.0
            sigma_now = rolling_ln_sigma(prices_hist[t]) if t in prices_hist else 0.0
            ln_kept = no_trade_band_clip(ln_now, sigma_now, band_k)

            if mode == "scale":
                contrib_after_band[t] = fix_map[t] * (b_final[t] * ln_kept)
            else:
                contrib_after_band[t] = fix_map[t] * ln_kept + b_final[t]

            records.append({
                "Ticker": t,
                "Ref(t0)": t0,
                "Live(tn)": tn,
                "ln(tn/t0)": ln_now,
                "σ_ln (roll)": sigma_now,
                "b_scale": b_scale.get(t, 1.0),
                "regime_mult": regime.get(t, 1.0),
                "b_final": b_final.get(t, 1.0),
                "fix_c": fix_map[t],
                "Contrib_raw": contrib_raw.get(t, 0.0),
                f"Contrib_after_band(k={band_k})": contrib_after_band[t],
            })

        # 4.5 CF cap ระดับพอร์ต
        contrib_capped = cap_portfolio_cf(contrib_after_band, cf_cap) if cf_cap > 0 else contrib_after_band
        total_raw   = sum(contrib_raw.values())
        total_after = sum(contrib_capped.values())

    # =========================
    # 5) Results
    # =========================
    st.subheader("📊 รายงานรายสินทรัพย์")
    df_res = pd.DataFrame.from_records(records)
    st.dataframe(df_res, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Σ Contribution (Raw)", f"{total_raw:,.2f}")
    with c2:
        st.metric("Σ Contribution (After CF Cap)", f"{total_after:,.2f}")

    data_bytes, fname = to_csv_download(df_res, "lnF_results")
    st.download_button("⬇️ ดาวน์โหลดผลลัพธ์ CSV", data=data_bytes, file_name=fname, mime="text/csv", use_container_width=True)

    # =========================
    # 6) Stress Engine (แก้บั๊ก slider)
    # =========================
    st.markdown("---")
    st.subheader("🧪 Stress Test (Shock Surface)")

    # ✅ st.slider ให้ค่าได้ 2 ตัว (range) เท่านั้น — แยก step ออกมาเป็น number_input
    shock_min, shock_max = st.slider(
        "ช่วง Shock (เช่น -40% ถึง +40%)",
        min_value=-0.9, max_value=0.9, value=(-0.4, 0.4)
    )
    shock_step = st.number_input(
        "ความละเอียด (step)",
        min_value=0.01, max_value=0.5, value=0.05, step=0.01
    )
    # guard เล็กน้อย
    if shock_max <= shock_min:
        st.warning("ช่วง shock ต้องมีค่าน้อย < ค่ามาก — จะใช้ค่าเริ่มต้นแทน")
        shock_min, shock_max = -0.4, 0.4
    if shock_step <= 0:
        st.warning("step ต้องมากกว่า 0 — จะใช้ 0.05 แทน")
        shock_step = 0.05

    shocks = np.round(np.arange(shock_min, shock_max + 1e-12, shock_step), 4)

    assets_for_stress = [
        AssetInput(ticker=t, live_price=live_map[t], ref_price=ref_map[t], fix_c=fix_map[t], b=b_final[t])
        for t in tickers
    ]
    contrib_surface, total_surface = stress_surface(assets_for_stress, shocks=shocks, mode=mode)

    st.write("**Total Contribution vs Shock**")
    fig = plt.figure()
    plt.plot(total_surface.index.values, total_surface["_TOTAL_"].values)
    plt.xlabel("Shock (fraction)")
    plt.ylabel("Total Contribution")
    plt.title("Total Contribution under Price Shocks")
    st.pyplot(fig, clear_figure=True, use_container_width=True)

    with st.expander("ดูตาราง Stress Surface (รายสินทรัพย์)"):
        st.dataframe(contrib_surface, use_container_width=True)
        data_bytes2, fname2 = to_csv_download(contrib_surface.reset_index(), "stress_surface")
        st.download_button("⬇️ ดาวน์โหลด Surface CSV", data=data_bytes2, file_name=fname2, mime="text/csv", use_container_width=True)

else:
    st.info("ตั้งค่าพารามิเตอร์ทางซ้าย แล้วกด ▶️ Run / คำนวณ เพื่อเริ่มต้น")
