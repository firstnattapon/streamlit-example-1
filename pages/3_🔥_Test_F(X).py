# -*- coding: utf-8 -*-
"""
Cash_Balan Optimizer — 3‑Year MIRR + Risk‑Parity (Score‑aware)
-----------------------------------------------------------------
Goal
  • Allocate Cash_Balan (budget $1–$3000) across tickers to maximize portfolio MIRR (3Y proxy)
  • Start from Score per $1, then apply Risk Parity where higher Score reduces risk (per user rule)
  • Keep the rest of the app unchanged; this can be a standalone Streamlit page

How it works (high level)
  1) Fetch 3Y monthly prices via yfinance (Adj Close)
  2) Compute per‑ticker 3Y MIRR (annualized) using monthly cash‑flows [-1, 0, …, FinalValueRatio]
  3) Compute volatility (monthly stdev of returns)
  4) Build Score per $1 (configurable formula; default = FV factor ‑ 1)
  5) Risk Parity weights: w ∝ 1 / (σ * (1 + γ * score_norm))  → higher Score → lower weight
  6) Allocate budget B using those weights; report table + portfolio MIRR proxy (weighted MIRR)

Notes
  • MIRR of a combined portfolio is not exactly a weighted average of component MIRRs; we use a clean proxy
  • You can tune the score → risk link via γ (gamma) and choose the score formula
  • Designed to not alter your existing outputs; treat this as a new page (e.g., pages/8_MIRR_RP_Optimizer.py)

Dependencies
  pip install streamlit yfinance numpy numpy_financial pandas plotly
"""

import math
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import numpy_financial as npf
import plotly.express as px

# -----------------------------
# Config & Defaults
# -----------------------------
DEFAULT_TICKERS: List[str] = [
    # Based on tickers appearing in prior runs/messages
    "NEGG", "NVTS", "QXO", "RIVN", "APLS", "FFWM", "CLSK", "RXRX", "FLNC",
    "IBRX", "DJT", "DYN", "GERN", "SG", "AGL"
]

CONFIG_FILE = "un15_fx_config.json"  # optional; if present, can seed tickers and sliders

@dataclass
class AppConfig:
    tickers: List[str]
    min_budget: int = 1
    max_budget: int = 3000
    default_budget: int = 650  # aligns with prior default Cash_Balan in your JSON
    finance_rate: float = 0.00  # MIRR finance rate
    reinvest_rate: float = 0.00  # MIRR reinvestment rate


def load_optional_config() -> AppConfig:
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # If your JSON has a __DEFAULT_CONFIG__ block and per‑ticker entries, prefer its keys
        tickers = []
        for k, v in raw.items():
            if k == "__DEFAULT_CONFIG__":
                continue
            if isinstance(v, dict) and v.get("Ticker"):
                tickers.append(v["Ticker"])
        if not tickers:
            tickers = DEFAULT_TICKERS
        default_budget = (
            raw.get("__DEFAULT_CONFIG__", {}).get("Cash_Balan", 650.0)
        )
        return AppConfig(tickers=list(dict.fromkeys(tickers)), default_budget=int(default_budget))
    except Exception:
        return AppConfig(tickers=DEFAULT_TICKERS)


# -----------------------------
# Data Layer
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_monthly_prices(tickers: List[str], years: int = 3) -> pd.DataFrame:
    """Download monthly adj close for past `years` (buffered to be safe)."""
    period = f"{years + 1}y"  # buffer to ensure ≥ 36 monthly points
    df = yf.download(
        tickers=tickers,
        period=period,
        interval="1mo",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    # Normalize to a wide DataFrame of Adj Close
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance returns (Ticker, Field)
        closes = {
            t: df[t]["Close"].rename(t) if "Close" in df[t].columns else df[t]["Adj Close"].rename(t)
            for t in tickers
            if t in df.columns.get_level_values(0)
        }
        out = pd.concat(closes.values(), axis=1)
    else:
        # Single ticker
        out = df.rename(columns={"Close": tickers[0], "Adj Close": tickers[0]})[[tickers[0]]]
    out = out.dropna(how="all")
    # Keep last 36 months if available
    return out.tail(36)


def monthly_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


# -----------------------------
# Metrics
# -----------------------------

def mirr_from_prices(prices: pd.Series, finance_rate: float, reinvest_rate: float) -> Optional[float]:
    """Annualized MIRR from monthly prices using simple two‑point cashflow: [-1, 0, …, FV].
    Returns annualized MIRR (per year). None if insufficient data."""
    prices = prices.dropna()
    if len(prices) < 2:
        return None
    ratio = float(prices.iloc[-1] / prices.iloc[0])
    n = len(prices) - 1  # periods (months between first and last)
    flows = np.zeros(n + 1)
    flows[0] = -1.0
    flows[-1] = ratio  # redeem at terminal value
    try:
        per_period_mirr = npf.mirr(flows, finance_rate / 12.0, reinvest_rate / 12.0)
        if per_period_mirr is None or np.isnan(per_period_mirr):
            return None
        annual = (1.0 + per_period_mirr) ** 12 - 1.0
        return float(annual)
    except Exception:
        return None


def score_per_dollar(
    prices: pd.Series,
    mirr_annual: Optional[float],
    method: str = "fv_gain",
) -> Optional[float]:
    """Compute Score per $1.
    • fv_gain: (Final/Start - 1)
    • mirr: annualized MIRR (as decimal)
    • hybrid: 0.5 * (Final/Start - 1) + 0.5 * MIRR
    """
    prices = prices.dropna()
    if len(prices) < 2:
        return None
    fv_gain = float(prices.iloc[-1] / prices.iloc[0] - 1.0)
    if method == "fv_gain":
        return fv_gain
    if method == "mirr" and mirr_annual is not None:
        return float(mirr_annual)
    if method == "hybrid" and mirr_annual is not None:
        return 0.5 * fv_gain + 0.5 * float(mirr_annual)
    # fallback
    return fv_gain


# -----------------------------
# Risk Parity (Score‑aware)
# -----------------------------

def rp_weights(
    vols: pd.Series,
    scores: pd.Series,
    gamma: float = 1.0,
    score_transform: str = "minmax",
) -> pd.Series:
    """Compute risk‑parity weights with score‑aware scaling.
    w_i ∝ 1 / (σ_i * (1 + γ * score_norm_i))
    Higher Score → larger denominator → smaller weight (reduce risk when Score is high).
    """
    vols = vols.copy()
    scores = scores.copy()

    # Normalize scores → [0,1]
    if score_transform == "minmax":
        s_min, s_max = np.nanmin(scores.values), np.nanmax(scores.values)
        if not np.isfinite(s_min) or not np.isfinite(s_max) or s_max <= s_min:
            s_norm = pd.Series(0.5, index=scores.index)  # flat
        else:
            s_norm = (scores - s_min) / (s_max - s_min)
    elif score_transform == "rank":
        s_norm = scores.rank(pct=True)
    else:
        s_norm = scores.copy()

    denom = vols.replace(0, np.nan) * (1.0 + gamma * s_norm)
    inv = 1.0 / denom
    inv = inv.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if inv.sum() <= 0:
        return pd.Series(0.0, index=vols.index)
    w = inv / inv.sum()
    return w


# -----------------------------
# Streamlit UI
# -----------------------------

def main():
    st.set_page_config(page_title="Cash_Balan Optimizer (MIRR + RP)", page_icon="📈", layout="wide")
    st.title("🚀 Cash_Balan Optimizer — 3Y MIRR × Risk Parity (Score‑aware)")
    st.caption("Goal: allocate $1–$3000 across tickers to maximize MIRR proxy; higher Score → lower risk weight.")

    cfg = load_optional_config()

    with st.sidebar:
        st.header("Controls")
        tickers_input = st.text_area(
            "Tickers (comma‑separated)",
            value=", ".join(cfg.tickers),
            help="Override the list here."
        )
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        budget = st.slider("Cash_Balan budget ($)", min_value=cfg.min_budget, max_value=cfg.max_budget, value=cfg.default_budget, step=1)
        st.divider()
        st.subheader("MIRR")
        finance_rate = st.number_input("Finance rate (annual, e.g. borrow cost)", value=cfg.finance_rate, step=0.01, format="%.4f")
        reinvest_rate = st.number_input("Reinvest rate (annual)", value=cfg.reinvest_rate, step=0.01, format="%.4f")
        st.subheader("Score per $1")
        score_method = st.selectbox("Score method", ["fv_gain", "mirr", "hybrid"], index=0, help="How to compute Score per $1")
        st.subheader("Risk Parity (Score‑aware)")
        gamma = st.slider("Gamma (Score → risk strength)", 0.0, 5.0, 1.0, 0.1)
        transform = st.selectbox("Score normalization", ["minmax", "rank"], index=0)
        st.caption("Higher gamma reduces weights for high‑Score names more aggressively.")

    if not tickers:
        st.warning("No tickers provided.")
        return

    with st.status("Fetching monthly prices…", expanded=False):
        px_monthly = fetch_monthly_prices(tickers)

    # Align per‑ticker series
    results: List[Dict] = []
    for t in tickers:
        s = px_monthly.get(t)
        if s is None:
            results.append({"Ticker": t})
            continue
        mirr_ann = mirr_from_prices(s, finance_rate, reinvest_rate)
        vol_m = monthly_returns(s).std() if s.dropna().size > 2 else None
        score = score_per_dollar(s, mirr_ann, method=score_method)
        fv_factor = float(s.dropna().iloc[-1] / s.dropna().iloc[0]) if s.dropna().size > 1 else None
        results.append({
            "Ticker": t,
            "MIRR_annual": mirr_ann,
            "Vol_monthly": vol_m,
            "Score_per_$1": score,
            "FV_factor": fv_factor,
            "Start": float(s.dropna().iloc[0]) if s.dropna().size else None,
            "End": float(s.dropna().iloc[-1]) if s.dropna().size else None,
            "Months": int(s.dropna().size)
        })

    df = pd.DataFrame(results).set_index("Ticker")

    # Build RP weights
    rp_w = rp_weights(
        vols=df["Vol_monthly"].astype(float),
        scores=df["Score_per_$1"].astype(float),
        gamma=gamma,
        score_transform=transform,
    )

    # Dollar allocation
    alloc = (rp_w * float(budget)).round(2)

    # Portfolio MIRR proxy (weighted by weights where MIRR available)
    mirr_series = df["MIRR_annual"].copy()
    # Only weight tickers with MIRR present:
    valid_mask = mirr_series.notna() & rp_w.notna()
    if valid_mask.any() and rp_w[valid_mask].sum() > 0:
        # re‑normalize weights to valid subset for proxy calc
        w_valid = rp_w[valid_mask] / rp_w[valid_mask].sum()
        port_mirr_proxy = float((w_valid * mirr_series[valid_mask]).sum())
    else:
        port_mirr_proxy = float("nan")

    # Final table
    out = df.copy()
    out["RP_weight"] = rp_w
    out["Cash_alloc_$"] = alloc
    out["Exp_MIRR_contrib"] = rp_w * df["MIRR_annual"].fillna(0.0)

    st.subheader("Optimization Output")
    st.dataframe(
        out[[
            "Months", "Start", "End", "FV_factor", "MIRR_annual", "Vol_monthly", "Score_per_$1", "RP_weight", "Cash_alloc_$", "Exp_MIRR_contrib"
        ]].sort_values("Cash_alloc_$", ascending=False),
        use_container_width=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Budget ($)", f"{budget:,.0f}")
    with c2:
        st.metric("Σ weights", f"{rp_w.sum():.3f}")
    with c3:
        st.metric("Portfolio MIRR (proxy, annual)", f"{port_mirr_proxy:.2%}" if math.isfinite(port_mirr_proxy) else "NA")

    st.divider()
    st.subheader("Allocation Chart")
    chart_df = out.reset_index()[["Ticker", "Cash_alloc_$"]]
    fig = px.bar(chart_df, x="Ticker", y="Cash_alloc_$", title="Cash_Balan Allocation ($)")
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download allocation CSV",
        data=out.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="cash_balan_allocation.csv",
        mime="text/csv",
    )

    st.caption(
        """
        Notes:
        • Score per $1: choose fv_gain (Final/Start − 1), mirr (annual), or hybrid.
        • Risk Parity uses monthly volatility; higher Score reduces weight via γ.
        • Portfolio MIRR shown is a proxy (weighted MIRR), not an exact portfolio MIRR.
        • Keep your original outputs/pages intact — use this as an add‑on optimizer page.
        """
    )


if __name__ == "__main__":
    main()
