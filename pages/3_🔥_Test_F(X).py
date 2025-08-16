# -*- coding: utf-8 -*-
"""
Exist_F(X) — Cash_Balan Optimizer for MIRR (Full App)

✅ Goal: For each selected Ticker, search Cash_Balan in [0, MAX] to maximize MIRR
   while keeping all other logic/outputs unchanged. This app preserves the
   original analysis view and adds a new optimization view.

Notes
- Uses coarse→fine search to keep runtime practical.
- Caches Yahoo Finance history for speed and to avoid repeated network calls.
- Lets you download an optimized config JSON and/or apply it in-session.

Author: (you)
"""

import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy_financial as npf
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# ------------------------- Streamlit App Setup -------------------------
st.set_page_config(page_title="Exist_F(X) — MIRR Optimizer", page_icon="☀", layout="wide")

# ------------------------- Data Classes -------------------------------
@dataclass
class TickerConfig:
    Ticker: str
    Fixed_Asset_Value: float = 1500.0
    Cash_Balan: float = 650.0
    step: float = 0.01
    filter_date: str = "2024-01-01 12:00:00+07:00"
    pred: int = 1

# ------------------------- Config Loader ------------------------------
def load_config(filename: str = "un15_fx_config.json") -> Tuple[Dict[str, dict], dict]:
    """Load unified config file with optional __DEFAULT_CONFIG__."""
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
        "Fixed_Asset_Value": 1500.0,
        "Cash_Balan": 650.0,
        "step": 0.01,
        "filter_date": "2024-01-01 12:00:00+07:00",
        "pred": 1,
    }
    default_config = data.pop("__DEFAULT_CONFIG__", fallback_default)
    ticker_configs = data
    return ticker_configs, default_config

# ------------------------- Yahoo Helpers ------------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_hist(ticker: str, start_iso: str) -> pd.DataFrame:
    """Fetch full history then filter to start_iso (Asia/Bangkok tz)."""
    df = yf.Ticker(ticker).history(period="max")
    if df.empty:
        return df
    try:
        df.index = df.index.tz_convert(tz="Asia/Bangkok")
    except Exception:
        # sometimes index is naive
        df.index = pd.to_datetime(df.index).tz_localize("Asia/Bangkok")
    df = df[df.index >= start_iso][["Close"]].copy()
    return df

@st.cache_data(show_spinner=False, ttl=60 * 10)
def get_last_price(ticker: str) -> Optional[float]:
    try:
        fi = yf.Ticker(ticker).fast_info
        p = fi.get("lastPrice")
        if p is None or (isinstance(p, float) and (math.isnan(p) or p <= 0)):
            raise ValueError("fast_info empty")
        return float(p)
    except Exception:
        # fallback to last close
        h = yf.Ticker(ticker).history(period="5d")
        if not h.empty:
            return float(h["Close"].iloc[-1])
    return None

# ------------------------- Core Model -------------------------------
def calculate_cash_balance_model(entry: float, step: float, Fixed_Asset_Value: float, Cash_Balan: float) -> pd.DataFrame:
    """Reproduce the original core model grid for a given entry & Cash_Balan."""
    if entry >= 10000 or entry <= 0:
        return pd.DataFrame()
    samples = np.arange(0, np.around(entry, 2) * 3 + step, step)

    df = pd.DataFrame()
    df["Asset_Price"] = np.around(samples, 2)
    df["Fixed_Asset_Value"] = Fixed_Asset_Value
    df["Amount_Asset"] = df["Fixed_Asset_Value"] / df["Asset_Price"]

    # top branch
    df_top = df[df.Asset_Price >= np.around(entry, 2)].copy()
    if not df_top.empty:
        df_top["Cash_Balan_top"] = (df_top["Amount_Asset"].shift(1) - df_top["Amount_Asset"]) * df_top["Asset_Price"]
        df_top.fillna(0, inplace=True)
        arr = df_top["Cash_Balan_top"].values
        xx = np.zeros(len(arr))
        y0 = Cash_Balan
        for i, v in enumerate(arr):
            y0 = y0 + v
            xx[i] = y0
        df_top["Cash_Balan"] = xx
        df_top = df_top.sort_values(by="Amount_Asset")[:-1]
    else:
        df_top = pd.DataFrame(columns=["Asset_Price", "Fixed_Asset_Value", "Amount_Asset", "Cash_Balan"])

    # down branch
    df_down = df[df.Asset_Price <= np.around(entry, 2)].copy()
    if not df_down.empty:
        df_down["Cash_Balan_down"] = (df_down["Amount_Asset"].shift(-1) - df_down["Amount_Asset"]) * df_down["Asset_Price"]
        df_down.fillna(0, inplace=True)
        df_down = df_down.sort_values(by="Asset_Price", ascending=False)
        arr = df_down["Cash_Balan_down"].values
        xxx = np.zeros(len(arr))
        y1 = Cash_Balan
        for i, v in enumerate(arr):
            y1 = y1 + v
            xxx[i] = y1
        df_down["Cash_Balan"] = xxx
    else:
        df_down = pd.DataFrame(columns=["Asset_Price", "Fixed_Asset_Value", "Amount_Asset", "Cash_Balan"])

    combined = pd.concat([df_top, df_down], axis=0, ignore_index=True)
    return combined[["Asset_Price", "Fixed_Asset_Value", "Amount_Asset", "Cash_Balan"]]


def simulate_series_for_config(cfg: TickerConfig, Cash_Balan_override: Optional[float] = None) -> Optional[pd.DataFrame]:
    """Simulate time series (like original delta6) for a *single* ticker.

    Returns columns: ['net_pv', 're'] indexed by date, or None on failure.
    """
    try:
        hist = get_hist(cfg.Ticker, cfg.filter_date)
        if hist.empty:
            return None
        entry = float(np.around(hist["Close"].iloc[0], 2))
        step = cfg.step
        fav = cfg.Fixed_Asset_Value
        cb = cfg.Cash_Balan if Cash_Balan_override is None else float(Cash_Balan_override)

        df_model = calculate_cash_balance_model(entry, step, fav, cb)
        if df_model.empty:
            return None

        ticker_data = hist.copy()
        ticker_data["Close"] = np.around(ticker_data["Close"].values, 2)
        ticker_data["pred"] = cfg.pred
        ticker_data["Fixed_Asset_Value"] = fav
        ticker_data["Amount_Asset"] = 0.0
        ticker_data["re"] = 0.0
        ticker_data["Cash_Balan"] = cb
        ticker_data.iloc[0, ticker_data.columns.get_loc("Amount_Asset")] = fav / ticker_data.iloc[0]["Close"]

        close_vals = ticker_data["Close"].values
        pred_vals = ticker_data["pred"].values
        amount_asset_vals = ticker_data["Amount_Asset"].values
        re_vals = ticker_data["re"].values
        cash_balan_sim_vals = ticker_data["Cash_Balan"].values

        for idx in range(1, len(amount_asset_vals)):
            if int(pred_vals[idx]) == 1:
                amount_asset_vals[idx] = fav / close_vals[idx]
                re_vals[idx] = (amount_asset_vals[idx - 1] * close_vals[idx]) - fav
            else:
                amount_asset_vals[idx] = amount_asset_vals[idx - 1]
                re_vals[idx] = 0.0
            cash_balan_sim_vals[idx] = cash_balan_sim_vals[idx - 1] + re_vals[idx]

        # Map refer_model from grid to price
        original_index = ticker_data.index
        merged = ticker_data.merge(
            df_model[["Asset_Price", "Cash_Balan"]].rename(columns={"Cash_Balan": "refer_model"}),
            left_on="Close",
            right_on="Asset_Price",
            how="left",
        ).drop("Asset_Price", axis=1)
        merged.set_index(original_index, inplace=True)
        merged["refer_model"].interpolate(method="linear", inplace=True)
        merged.fillna(method="bfill", inplace=True)
        merged.fillna(method="ffill", inplace=True)

        merged["pv"] = merged["Cash_Balan"] + (merged["Amount_Asset"] * merged["Close"])
        merged["refer_pv"] = merged["refer_model"] + fav
        merged["net_pv"] = merged["pv"] - merged["refer_pv"]
        return merged[["net_pv", "re"]]
    except Exception:
        return None

# ------------------------- KPI & MIRR (per ticker) --------------------
def compute_kpis_for_series(series_df: pd.DataFrame, fav: float) -> Dict[str, float]:
    """Compute metrics compatible with the original app for a single ticker."""
    if series_df is None or series_df.empty:
        return {
            "final_sum_delta": 0.0,
            "max_buffer_used": 0.0,
            "true_alpha": 0.0,
            "avg_cf": 0.0,
            "mirr": 0.0,
            "days": 0.0,
        }

    # Sum_Delta uses last net_pv (like original portfolio logic)
    final_sum_delta = float(series_df["net_pv"].iloc[-1])

    # Max buffer: cumsum(re) then take the most negative prefix (abs(min))
    cumsum_re = series_df["re"].cumsum()
    max_buffer_used = float(abs(min(0.0, cumsum_re.min())))

    # True Alpha = (cf / (fav + max_buffer)) * 100
    denom = fav + max_buffer_used
    true_alpha = 0.0 if denom == 0 else (final_sum_delta / denom) * 100.0

    days = float(len(series_df))
    avg_cf = 0.0 if days == 0 else (final_sum_delta / days)

    # MIRR over 3 years with your specified rules
    initial_investment = fav + max_buffer_used
    if initial_investment <= 0:
        mirr_val = 0.0
    else:
        annual_cash_flow = avg_cf * 252.0
        exit_multiple = initial_investment * 0.5
        cash_flows = [
            -initial_investment,
            annual_cash_flow,
            annual_cash_flow,
            annual_cash_flow + exit_multiple,
        ]
        mirr_val = float(npf.mirr(cash_flows, 0.0, 0.0))

    return {
        "final_sum_delta": final_sum_delta,
        "max_buffer_used": max_buffer_used,
        "true_alpha": true_alpha,
        "avg_cf": avg_cf,
        "mirr": mirr_val,
        "days": days,
    }

# ------------------------- Optimizer ----------------------------------
def coarse_to_fine_grid(max_cash: int, coarse: int, mid: int, fine: int) -> List[int]:
    """Make a 3-stage grid within [0, max_cash]."""
    coarse_points = list(range(0, max_cash + 1, max(1, coarse)))
    return sorted(set(coarse_points))


def refine_around(best_x: int, span: int, step: int, lo: int, hi: int) -> List[int]:
    a = max(lo, best_x - span)
    b = min(hi, best_x + span)
    return list(range(a, b + 1, max(1, step)))


def optimize_cash_balan_for_ticker(cfg: TickerConfig, max_cash: int = 3000, coarse: int = 150, mid: int = 25, fine: int = 5) -> Tuple[int, Dict[str, float], pd.DataFrame]:
    """Return (best_cash_balan, best_metrics, curve_df)."""
    # Stage 1: coarse scan
    candidates = coarse_to_fine_grid(max_cash, coarse, mid, fine)
    records: List[Tuple[int, float]] = []

    def eval_point(cb_val: int) -> float:
        series = simulate_series_for_config(cfg, Cash_Balan_override=cb_val)
        metrics = compute_kpis_for_series(series, cfg.Fixed_Asset_Value)
        return metrics["mirr"]

    best_cb = 0
    best_mirr = -1e9
    for cb in candidates:
        mirr_val = eval_point(cb)
        records.append((cb, mirr_val))
        if mirr_val > best_mirr:
            best_mirr = mirr_val
            best_cb = cb

    # Stage 2: mid refinement
    mid_span = max(2 * mid, 50)
    for cb in refine_around(best_cb, mid_span, mid, 0, max_cash):
        if cb in [r[0] for r in records]:
            continue
        mirr_val = eval_point(cb)
        records.append((cb, mirr_val))
        if mirr_val > best_mirr:
            best_mirr = mirr_val
            best_cb = cb

    # Stage 3: fine refinement
    fine_span = max(3 * fine, 15)
    for cb in refine_around(best_cb, fine_span, fine, 0, max_cash):
        if cb in [r[0] for r in records]:
            continue
        mirr_val = eval_point(cb)
        records.append((cb, mirr_val))
        if mirr_val > best_mirr:
            best_mirr = mirr_val
            best_cb = cb

    # Final metrics at best point
    best_series = simulate_series_for_config(cfg, Cash_Balan_override=best_cb)
    best_metrics = compute_kpis_for_series(best_series, cfg.Fixed_Asset_Value)

    curve_df = pd.DataFrame(records, columns=["Cash_Balan", "MIRR"]).sort_values("Cash_Balan")
    return int(best_cb), best_metrics, curve_df

# ------------------------- UI: Controls -------------------------------
full_config, DEFAULTS = load_config()
if not full_config and not DEFAULTS:
    st.stop()

if "custom_tickers" not in st.session_state:
    st.session_state.custom_tickers: Dict[str, dict] = {}

st.sidebar.header("Ticker Universe")
new_ticker = st.sidebar.text_input("Add ticker (e.g., AAPL)").upper().strip()
if st.sidebar.button("Add", use_container_width=True) and new_ticker:
    if new_ticker not in full_config and new_ticker not in st.session_state.custom_tickers:
        st.session_state.custom_tickers[new_ticker] = {"Ticker": new_ticker, **DEFAULTS}
        st.sidebar.success(f"Added {new_ticker}")
    else:
        st.sidebar.warning("Already exists.")

all_tickers = list(full_config.keys()) + list(st.session_state.custom_tickers.keys())
default_selection = [t for t in list(full_config.keys()) if t in all_tickers]
selected = st.sidebar.multiselect("Select tickers", options=all_tickers, default=default_selection)

st.sidebar.divider()
st.sidebar.header("Optimization Range")
max_cash = st.sidebar.number_input("Max Cash_Balan to search", min_value=0, max_value=10000, value=3000, step=100)
coarse = st.sidebar.number_input("Coarse step", min_value=1, max_value=1000, value=150, step=1)
mid = st.sidebar.number_input("Mid step", min_value=1, max_value=200, value=25, step=1)
fine = st.sidebar.number_input("Fine step", min_value=1, max_value=50, value=5, step=1)

# ------------------------- Main Layout -------------------------------
st.title("Exist_F(X)")
tab_analyze, tab_opt = st.tabs(["Analyze (Original)", "Optimize Cash_Balan for MIRR"])

# ------------------------- Tab 1: Original Analysis -------------------
with tab_analyze:
    st.subheader("Portfolio Simulation (unchanged logic)")

    # Build active configs
    active_configs: Dict[str, dict] = {}
    for t in selected:
        active_configs[t] = full_config.get(t, st.session_state.custom_tickers.get(t))

    if not active_configs:
        st.info("Select at least one ticker from the sidebar.")
    else:
        # --- Reuse the original aggregation pipeline ---
        def delta6(asset_config: dict) -> Optional[pd.DataFrame]:
            try:
                cfg = TickerConfig(
                    Ticker=asset_config["Ticker"].strip(),
                    Fixed_Asset_Value=float(asset_config.get("Fixed_Asset_Value", 1500.0)),
                    Cash_Balan=float(asset_config.get("Cash_Balan", 650.0)),
                    step=float(asset_config.get("step", 0.01)),
                    filter_date=str(asset_config.get("filter_date", DEFAULTS.get("filter_date", "2024-01-01 12:00:00+07:00"))),
                    pred=int(asset_config.get("pred", 1)),
                )
                return simulate_series_for_config(cfg)
            except Exception:
                return None

        def un_16(active_cfgs: Dict[str, dict]) -> pd.DataFrame:
            all_re = []
            all_net_pv = []
            for ticker_name, cfg in active_cfgs.items():
                df = delta6(cfg)
                if df is not None and not df.empty:
                    all_re.append(df[["re"]].rename(columns={"re": f"{ticker_name}_re"}))
                    all_net_pv.append(df[["net_pv"]].rename(columns={"net_pv": f"{ticker_name}_net_pv"}))
            if not all_re:
                return pd.DataFrame()
            df_re = pd.concat(all_re, axis=1)
            df_net = pd.concat(all_net_pv, axis=1)
            df_re.fillna(0, inplace=True)
            df_net.fillna(0, inplace=True)
            df_re["maxcash_dd"] = df_re.sum(axis=1).cumsum()
            df_net["cf"] = df_net.sum(axis=1)
            return pd.concat([df_re, df_net], axis=1)

        with st.spinner("Calculating…"):
            data = un_16(active_configs)

        if data.empty:
            st.error("No data. Some tickers may lack history in the chosen period.")
        else:
            df_new = data.copy()
            # roll_over = running min of cumulative cash usage
            roll_over = []
            vals = df_new.maxcash_dd.values
            for i in range(len(vals)):
                prev = vals[:i]
                roll_over.append(np.min(prev) if len(prev) > 0 else 0)
            cf_values = df_new.cf.values
            df_all = pd.DataFrame({"Sum_Delta": cf_values, "Max_Sum_Buffer": roll_over}, index=df_new.index)

            # True Alpha based on total capital at risk (len * 1500 + |min buffer|)
            num_sel = max(1, len(active_configs))
            initial_capital = num_sel * 1500.0
            max_buffer_used = abs(float(np.min(roll_over)))
            total_capital_at_risk = initial_capital + max_buffer_used
            if total_capital_at_risk == 0:
                total_capital_at_risk = 1
            true_alpha_series = (df_new.cf.values / total_capital_at_risk) * 100.0
            df_all_2 = pd.DataFrame({"True_Alpha": true_alpha_series}, index=df_new.index)

            # KPIs (portfolio-level)
            final_sum_delta = float(df_all.Sum_Delta.iloc[-1])
            final_max_buffer = float(df_all.Max_Sum_Buffer.iloc[-1])
            final_true_alpha = float(df_all_2.True_Alpha.iloc[-1])
            num_days = int(len(df_new))
            avg_cf = 0.0 if num_days == 0 else final_sum_delta / num_days
            avg_burn_cash = 0.0 if num_days == 0 else abs(final_max_buffer) / num_days

            # Portfolio MIRR with prior rules
            initial_investment = (num_sel * 1500.0) + abs(final_max_buffer)
            if initial_investment > 0:
                annual_cf = avg_cf * 252.0
                exit_mult = initial_investment * 0.5
                cash_flows = [-initial_investment, annual_cf, annual_cf, annual_cf + exit_mult]
                mirr_val = float(npf.mirr(cash_flows, 0.0, 0.0))
            else:
                mirr_val = 0.0

            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("Total Net Profit (cf)", f"{final_sum_delta:,.2f}")
            k2.metric("Max Cash Buffer Used", f"{final_max_buffer:,.2f}")
            k3.metric("True Alpha (%)", f"{final_true_alpha:,.2f}%")
            k4.metric("Avg. Daily Profit", f"{avg_cf:,.2f}")
            k5.metric("Avg. Daily Buffer Used", f"{avg_burn_cash:,.2f}")
            k6.metric("MIRR (3-Year)", f"{mirr_val:.2%}")

            st.divider()
            g1, g2 = st.columns(2)
            g1.plotly_chart(px.line(df_all.reset_index(drop=True), title="Cumulative Profit (Sum_Delta) vs. Max Buffer Used"), use_container_width=True)
            g2.plotly_chart(px.line(df_all_2, title="True Alpha (%)"), use_container_width=True)

            st.divider()
            st.subheader("Detailed Simulation Data")
            # cumulate re per-ticker for visual
            for t in selected:
                col = f"{t}_re"
                if col in df_new.columns:
                    df_new[col] = df_new[col].cumsum()
            st.plotly_chart(px.line(df_new, title="Portfolio Simulation Details"), use_container_width=True)

# ------------------------- Tab 2: Optimizer ---------------------------
with tab_opt:
    st.subheader("Per-Ticker Cash_Balan Optimizer for MIRR")
    st.caption("Searches Cash_Balan in [0, Max] using coarse→fine grid; keeps other logic intact.")

    if not selected:
        st.info("Select at least one ticker from the sidebar.")
        st.stop()

    # Build configs for selected tickers
    cfgs: Dict[str, TickerConfig] = {}
    for t in selected:
        base = full_config.get(t, st.session_state.custom_tickers.get(t))
        cfgs[t] = TickerConfig(
            Ticker=base["Ticker"].strip(),
            Fixed_Asset_Value=float(base.get("Fixed_Asset_Value", 1500.0)),
            Cash_Balan=float(base.get("Cash_Balan", 650.0)),
            step=float(base.get("step", 0.01)),
            filter_date=str(base.get("filter_date", DEFAULTS.get("filter_date", "2024-01-01 12:00:00+07:00"))),
            pred=int(base.get("pred", 1)),
        )

    run_btn = st.button("Run Optimization", type="primary")

    if run_btn:
        results_rows: List[dict] = []
        curves: Dict[str, pd.DataFrame] = {}

        prog = st.progress(0.0)
        for i, (t, cfg) in enumerate(cfgs.items(), start=1):
            best_cb, metrics, curve = optimize_cash_balan_for_ticker(
                cfg, max_cash=max_cash, coarse=coarse, mid=mid, fine=fine
            )
            curves[t] = curve
            results_rows.append(
                {
                    "Ticker": t,
                    "Best_Cash_Balan": best_cb,
                    "MIRR": metrics["mirr"],
                    "True_Alpha_%": metrics["true_alpha"],
                    "Final_Sum_Delta": metrics["final_sum_delta"],
                    "Max_Buffer_Used": metrics["max_buffer_used"],
                    "Days": metrics["days"],
                }
            )
            prog.progress(i / max(1, len(cfgs)))

        results_df = pd.DataFrame(results_rows).sort_values("MIRR", ascending=False)
        st.success("Done!")
        st.dataframe(results_df, use_container_width=True)

        # Plot MIRR curves for up to top 4 tickers
        st.divider()
        st.subheader("MIRR vs Cash_Balan (Top Tickers)")
        top_t = results_df.head(4)["Ticker"].tolist()
        for t in top_t:
            fig = px.line(curves[t], x="Cash_Balan", y="MIRR", title=f"{t}: MIRR vs Cash_Balan")
            st.plotly_chart(fig, use_container_width=True)

        # Create an optimized config you can download/apply
        st.divider()
        st.subheader("Apply / Export Optimized Config")

        # Start from original file so we preserve other keys exactly
        orig_file = "un15_fx_config.json"
        try:
            with open(orig_file, "r", encoding="utf-8") as f:
                original_payload = json.load(f)
        except Exception:
            # Fall back to reconstructed payload in-memory
            original_payload = {"__DEFAULT_CONFIG__": DEFAULTS}
            original_payload.update(full_config)
            for k, v in st.session_state.custom_tickers.items():
                original_payload[k] = v

        updated_payload = json.loads(json.dumps(original_payload))  # deep copy
        for _, row in results_df.iterrows():
            t = row["Ticker"]
            best_cb = int(row["Best_Cash_Balan"])
            # ensure key exists (handle custom tickers too)
            if t not in updated_payload:
                updated_payload[t] = {"Ticker": t, **DEFAULTS}
            updated_payload[t]["Cash_Balan"] = float(best_cb)

        st.json({k: updated_payload[k] for k in ["__DEFAULT_CONFIG__"] + top_t if k in updated_payload})
        st.download_button(
            label="Download optimized JSON",
            file_name="un15_fx_config.optimized.json",
            mime="application/json",
            data=json.dumps(updated_payload, indent=2, ensure_ascii=False),
            use_container_width=True,
        )

        if st.button("Apply to session and re-run"):
            # Update in-memory configs
            for t in results_df["Ticker"].tolist():
                best_cb = int(results_df.loc[results_df["Ticker"] == t, "Best_Cash_Balan"].values[0])
                if t in full_config:
                    full_config[t]["Cash_Balan"] = float(best_cb)
                elif t in st.session_state.custom_tickers:
                    st.session_state.custom_tickers[t]["Cash_Balan"] = float(best_cb)
            st.rerun()

    else:
        st.info("Set search range/steps on the left, then click **Run Optimization**.")
