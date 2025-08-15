import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import json
import plotly.express as px
import numpy_financial as npf

# ------------------- Load Config -------------------
def load_config(filename="un15_fx_config.json"):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        st.error(f"Error: Configuration file '{filename}' not found.")
        return {}, {}
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from '{filename}'. Please check its format.")
        return {}, {}

    fallback_default = {
        "Fixed_Asset_Value": 1500.0, "Cash_Balan": 650.0, "step": 0.01,
        "filter_date": "2024-01-01 12:00:00+07:00", "pred": 1
    }
    default_config = data.pop('__DEFAULT_CONFIG__', fallback_default)
    ticker_configs = data
    return ticker_configs, default_config

# ------------------- Static F-model (reference) -------------------
def calculate_cash_balance_model(entry, step, Fixed_Asset_Value, Cash_Balan):
    """
    Reference mapping (static-c) for Cash_Balan vs price using:
    F = c * ln(p/t) + b
    Where F is tracked by cumulative rebalancing cash so that
    pv = Cash_Balan + units*price and refer_pv = b + c (constant baseline).
    """
    if entry >= 10000 or entry <= 0:
        return pd.DataFrame()

    samples = np.arange(0, np.around(entry, 2) * 3 + step, step)
    df = pd.DataFrame()
    df['Asset_Price'] = np.around(samples, 2)
    df['Fixed_Asset_Value'] = Fixed_Asset_Value
    df['Amount_Asset'] = df['Fixed_Asset_Value'] / df['Asset_Price']

    # Price >= entry
    df_top = df[df.Asset_Price >= np.around(entry, 2)].copy()
    if not df_top.empty:
        df_top['Cash_Balan_top'] = (df_top['Amount_Asset'].shift(1) - df_top['Amount_Asset']) * df_top['Asset_Price']
        df_top.fillna(0, inplace=True)
        np_Cash_Balan_top = df_top['Cash_Balan_top'].values
        xx = np.zeros(len(np_Cash_Balan_top))
        y_0 = Cash_Balan
        for idx, v_0 in enumerate(np_Cash_Balan_top):
            y_0 = y_0 + v_0
            xx[idx] = y_0
        df_top['Cash_Balan'] = xx
        df_top = df_top.sort_values(by='Amount_Asset')[:-1]
    else:
        df_top = pd.DataFrame(columns=['Asset_Price', 'Fixed_Asset_Value', 'Amount_Asset', 'Cash_Balan'])

    # Price <= entry
    df_down = df[df.Asset_Price <= np.around(entry, 2)].copy()
    if not df_down.empty:
        df_down['Cash_Balan_down'] = (df_down['Amount_Asset'].shift(-1) - df_down['Amount_Asset']) * df_down['Asset_Price']
        df_down.fillna(0, inplace=True)
        df_down = df_down.sort_values(by='Asset_Price', ascending=False)
        np_Cash_Balan_down = df_down['Cash_Balan_down'].values
        xxx = np.zeros(len(np_Cash_Balan_down))
        y_1 = Cash_Balan
        for idx, v_1 in enumerate(np_Cash_Balan_down):
            y_1 = y_1 + v_1
            xxx[idx] = y_1
        df_down['Cash_Balan'] = xxx
    else:
        df_down = pd.DataFrame(columns=['Asset_Price', 'Fixed_Asset_Value', 'Amount_Asset', 'Cash_Balan'])

    combined_df = pd.concat([df_top, df_down], axis=0, ignore_index=True)
    return combined_df[['Asset_Price', 'Fixed_Asset_Value', 'Amount_Asset', 'Cash_Balan']]

# ------------------- Risk Parity Helpers -------------------
def _max_filter_date(active_configs: dict) -> pd.Timestamp:
    # ใช้ max ของ filter_date เพื่อให้ช่วงเวลาร่วมกันเท่ากัน
    tz = "Asia/Bangkok"
    dates = []
    for cfg in active_configs.values():
        try:
            dates.append(pd.Timestamp(cfg['filter_date']))
        except Exception:
            dates.append(pd.Timestamp("2024-01-01 12:00:00+07:00"))
    return max(dates) if dates else pd.Timestamp("2024-01-01 12:00:00+07:00")

def fetch_close_panel(active_configs: dict) -> pd.DataFrame:
    tickers = list(active_configs.keys())
    if not tickers:
        return pd.DataFrame()

    start_dt = _max_filter_date(active_configs).tz_convert("Asia/Bangkok") if pd.Timestamp.now().tz is not None else _max_filter_date(active_configs)
    # yfinance ใช้เวลา UTC; เราใช้ start ที่เป็น naive (UTC) โดยลบ offset 7 ชั่วโมงแบบหยาบ ๆ ได้
    # แต่เพื่อความเรียบง่าย ใช้ start.date() ก็พอ
    start = start_dt.tz_convert("UTC").date() if start_dt.tzinfo else start_dt.date()

    # ดึงข้อมูล close panel
    df = yf.download(tickers=tickers, start=str(start), auto_adjust=True, progress=False)['Close']
    if isinstance(df, pd.Series):
        df = df.to_frame()
        df.columns = [tickers[0]]
    # จัด index เป็น Asia/Bangkok
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("Asia/Bangkok")
    df = df.dropna(how='all')
    return df

def compute_rp_weights(close_panel: pd.DataFrame, window: int = 20,
                       min_w: float = 0.02, max_w: float = 0.5) -> pd.DataFrame:
    """
    Risk-parity weights by inverse rolling volatility of log returns.
    w_{i,t} ∝ 1 / σ_{i,t}, clipped to [min_w, max_w], renormalized to 1.
    """
    if close_panel.empty:
        return pd.DataFrame()

    # log returns
    logret = np.log(close_panel).diff()
    # rolling vol (population std dev)
    vol = logret.rolling(window=window, min_periods=max(5, window//2)).std()

    # guard: replace zero/NaN vol with median of row
    vol = vol.replace(0, np.nan)
    vol = vol.fillna(method='ffill').fillna(method='bfill')

    inv_vol = 1.0 / vol
    # initial weights (row-wise normalize)
    w = inv_vol.div(inv_vol.sum(axis=1), axis=0)

    # clip then renormalize
    w = w.clip(lower=min_w, upper=max_w)
    w = w.div(w.sum(axis=1), axis=0)
    return w

def capitals_from_weights(weights: pd.DataFrame, base_caps: dict) -> pd.DataFrame:
    """
    Convert weights to capital series: c_{i,t} = w_{i,t} * sum(base_caps)
    """
    if weights.empty:
        return pd.DataFrame()
    total_base = sum(float(v) for v in base_caps.values())
    c = weights * total_base
    return c

# ------------------- Dynamic-c Simulation -------------------
def simulate_with_dynamic_c(close_series: pd.Series,
                            c_series: pd.Series,
                            pred: int,
                            base_cash_balan: float,
                            base_c_fixed: float):
    """
    Simulate rebalancing with dynamic c_t, tracking re (cash flow),
    cash balance, units, pv; also compute reference pv from static model.
    """
    if close_series.empty or c_series.empty:
        return None

    # Align and forward fill c_series to close_series index
    c_series = c_series.reindex(close_series.index).ffill_
