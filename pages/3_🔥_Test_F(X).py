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
    c_series = c_series.reindex(close_series.index).ffill().bfill()

    n = len(close_series)
    prices = close_series.values.astype(float)
    c_t = c_series.values.astype(float)

    amount = np.zeros(n, dtype=float)
    re = np.zeros(n, dtype=float)
    cash_balan = np.zeros(n, dtype=float)

    # initialize at t0
    amount[0] = c_t[0] / prices[0]
    cash_balan[0] = base_cash_balan
    re[0] = 0.0

    for i in range(1, n):
        if pred == 1:
            # cash needed to move from current units to target c_t
            re[i] = (amount[i-1] * prices[i]) - c_t[i]
            amount[i] = c_t[i] / prices[i]
            cash_balan[i] = cash_balan[i-1] + re[i]
        else:
            amount[i] = amount[i-1]
            re[i] = 0.0
            cash_balan[i] = cash_balan[i-1]

    # Reference model (static c = base_c_fixed) for net_pv benchmark
    entry = float(prices[0])
    ref_map = calculate_cash_balance_model(entry=entry,
                                           step=0.01,
                                           Fixed_Asset_Value=base_c_fixed,
                                           Cash_Balan=base_cash_balan)
    ref = pd.DataFrame({"Close": np.round(prices, 2)}, index=close_series.index)
    if not ref_map.empty:
        ref = ref.merge(ref_map[['Asset_Price', 'Cash_Balan']].rename(columns={'Cash_Balan': 'refer_model'}),
                        left_on='Close', right_on='Asset_Price', how='left').drop('Asset_Price', axis=1)
        ref['refer_model'] = ref['refer_model'].interpolate(method='linear')
        ref['refer_model'] = ref['refer_model'].fillna(method='bfill').fillna(method='ffill')
    else:
        ref['refer_model'] = base_cash_balan

    pv = cash_balan + (amount * prices)
    refer_pv = ref['refer_model'].values + base_c_fixed
    net_pv = pv - refer_pv

    out = pd.DataFrame({
        "re": re,
        "net_pv": net_pv
    }, index=close_series.index)
    return out

def un_16_dynamic(active_configs: dict,
                  enable_dynamic_c: bool = True,
                  window: int = 20,
                  min_w: float = 0.02,
                  max_w: float = 0.50):
    """
    Aggregate simulation across tickers. When dynamic c is enabled, compute
    risk-parity weights and allocate capital per day, then simulate.
    """
    tickers = list(active_configs.keys())
    if not tickers:
        return pd.DataFrame()

    # 1) Price panel
    close_panel = fetch_close_panel(active_configs)
    if close_panel.empty:
        return pd.DataFrame()

    # จำกัดคอลัมน์เท่ากับ tickers ที่ขอจริง ๆ
    close_panel = close_panel[[c for c in tickers if c in close_panel.columns]]
    if close_panel.empty:
        return pd.DataFrame()

    # 2) Build base capital dict
    base_caps = {t: float(active_configs[t].get('Fixed_Asset_Value', 1500.0)) for t in close_panel.columns}
    base_cash = {t: float(active_configs[t].get('Cash_Balan', 650.0)) for t in close_panel.columns}
    pred_flag = {t: int(active_configs[t].get('pred', 1)) for t in close_panel.columns}

    # 3) Risk parity capitals over time
    if enable_dynamic_c:
        weights = compute_rp_weights(close_panel, window=window, min_w=min_w, max_w=max_w)
        capitals = capitals_from_weights(weights, base_caps)
    else:
        # static capitals (each column constant=its base)
        capitals = pd.DataFrame({t: base_caps[t] for t in close_panel.columns}, index=close_panel.index)

    # 4) Per-ticker simulation
    all_re = []
    all_net_pv = []
    for t in close_panel.columns:
        series = close_panel[t].dropna()
        if series.empty:
            continue
        c_series = capitals[t]
        res = simulate_with_dynamic_c(series, c_series,
                                      pred=pred_flag[t],
                                      base_cash_balan=base_cash[t],
                                      base_c_fixed=base_caps[t])
        if res is None or res.empty:
            continue
        all_re.append(res[['re']].rename(columns={'re': f"{t}_re"}))
        all_net_pv.append(res[['net_pv']].rename(columns={'net_pv': f"{t}_net_pv"}))

    if not all_re:
        return pd.DataFrame()

    df_re = pd.concat(all_re, axis=1).fillna(0.0)
    df_net = pd.concat(all_net_pv, axis=1).fillna(0.0)

    # Portfolio-level series
    df_re['maxcash_dd'] = df_re.sum(axis=1).cumsum()
    df_net['cf'] = df_net.sum(axis=1)

    final_df = pd.concat([df_re, df_net], axis=1)
    return final_df

# ------------------- STREAMLIT UI -------------------
st.set_page_config(page_title="Exist_F(X)", page_icon="☀", layout="wide")

full_config, DEFAULT_CONFIG = load_config()

if full_config or DEFAULT_CONFIG:
    if 'custom_tickers' not in st.session_state:
        st.session_state.custom_tickers = {}

    # Controls
    control_col1, control_col2 = st.columns([1, 2])
    with control_col1:
        st.subheader("Add New Ticker")
        new_ticker = st.text_input("Ticker (e.g., AAPL):", key="new_ticker_input").upper()
        if st.button("Add Ticker", key="add_ticker_button", use_container_width=True):
            if new_ticker and new_ticker not in full_config and new_ticker not in st.session_state.custom_tickers:
                st.session_state.custom_tickers[new_ticker] = {"Ticker": new_ticker, **DEFAULT_CONFIG}
                st.success(f"Added {new_ticker}!")
                st.rerun()
            elif new_ticker in full_config:
                st.warning(f"{new_ticker} already exists in config file.")
            elif new_ticker in st.session_state.custom_tickers:
                st.warning(f"{new_ticker} has already been added.")
            else:
                st.warning("Please enter a ticker symbol.")

    with control_col2:
        st.subheader("Select Tickers to Analyze")
        all_tickers = list(full_config.keys()) + list(st.session_state.custom_tickers.keys())
        default_selection = [t for t in list(full_config.keys()) if t in all_tickers]
        selected_tickers = st.multiselect(
            "Select from available tickers:",
            options=all_tickers,
            default=default_selection
        )

    # Risk Parity Controls
    with st.expander("⚙️ Risk Parity Settings", expanded=True):
        enable_dynamic_c = st.toggle("Enable Dynamic c (Risk Parity)", value=True)
        rp_col1, rp_col2, rp_col3 = st.columns(3)
        with rp_col1:
            rp_window = st.number_input("Volatility Window (days)", min_value=5, max_value=252, value=20, step=1)
        with rp_col2:
            rp_min_w = st.slider("Min Weight", min_value=0.0, max_value=0.5, value=0.02, step=0.01)
        with rp_col3:
            rp_max_w = st.slider("Max Weight", min_value=0.1, max_value=1.0, value=0.50, step=0.01)

    st.divider()

    # Active configs
    active_configs = {t: full_config.get(t, st.session_state.custom_tickers.get(t)) for t in selected_tickers}

    if not active_configs:
        st.warning("Please select at least one ticker to start the analysis.")
    else:
        with st.spinner('Calculating... Please wait.'):
            data = un_16_dynamic(active_configs,
                                 enable_dynamic_c=enable_dynamic_c,
                                 window=int(rp_window),
                                 min_w=float(rp_min_w),
                                 max_w=float(rp_max_w))

        if data.empty:
            st.error("Failed to generate data. This might happen if tickers have no historical data or another error occurred.")
        else:
            # KPI preparation
            df_new = data.copy()

            roll_over = []
            max_dd_values = df_new.maxcash_dd.values
            for i in range(len(max_dd_values)):
                roll = max_dd_values[:i]
                roll_min = np.min(roll) if len(roll) > 0 else 0
                roll_over.append(roll_min)

            cf_values = df_new.cf.values
            df_all = pd.DataFrame({'Sum_Delta': cf_values, 'Max_Sum_Buffer': roll_over}, index=df_new.index)

            # --- True Alpha (unchanged definition) ---
            num_selected_tickers = len(active_configs)
            # ใช้ทุนตั้งต้น = sum Fixed_Asset_Value ของ tickers ที่เลือก
            initial_capital = sum(float(cfg.get('Fixed_Asset_Value', 1500.0)) for cfg in active_configs.values())
            max_buffer_used = abs(np.min(roll_over))  # max draw on buffer
            total_capital_at_risk = initial_capital + max_buffer_used
            if total_capital_at_risk == 0:
                total_capital_at_risk = 1.0
            true_alpha_values = (df_new.cf.values / total_capital_at_risk) * 100.0
            df_all_2 = pd.DataFrame({'True_Alpha': true_alpha_values}, index=df_new.index)

            # KPIs
            st.subheader("Key Performance Indicators")
            final_sum_delta = df_all.Sum_Delta.iloc[-1]
            final_max_buffer = df_all.Max_Sum_Buffer.iloc[-1]
            final_true_alpha = df_all_2.True_Alpha.iloc[-1]
            num_days = len(df_new)
            avg_cf = final_sum_delta / num_days if num_days > 0 else 0
            avg_burn_cash = abs(final_max_buffer) / num_days if num_days > 0 else 0

            # MIRR (3Y) with your spec
            mirr_value = 0.0
            initial_investment = initial_capital + abs(final_max_buffer)
            if initial_investment > 0:
                annual_cash_flow = avg_cf * 252
                exit_multiple = initial_investment * 0.5
                cash_flows = [
                    -initial_investment,
                    annual_cash_flow,
                    annual_cash_flow,
                    annual_cash_flow + exit_multiple
                ]
                finance_rate = 0.0
                reinvest_rate = 0.0
                mirr_value = npf.mirr(cash_flows, finance_rate, reinvest_rate)

            kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
            kpi1.metric(label="Total Net Profit (cf)", value=f"{final_sum_delta:,.2f}")
            kpi2.metric(label="Max Cash Buffer Used", value=f"{final_max_buffer:,.2f}")
            kpi3.metric(label="True Alpha (%)", value=f"{final_true_alpha:,.2f}%")
            kpi4.metric(label="Avg. Daily Profit", value=f"{avg_cf:,.2f}")
            kpi5.metric(label="Avg. Daily Buffer Used", value=f"{avg_burn_cash:,.2f}")
            kpi6.metric(label="MIRR (3-Year)", value=f"{mirr_value:.2%}")

            st.divider()

            # Charts
            st.subheader("Performance Charts")
            graph_col1, graph_col2 = st.columns(2)
            graph_col1.plotly_chart(
                px.line(df_all.reset_index(drop=True), title="Cumulative Profit (Sum_Delta) vs. Max Buffer Used"),
                use_container_width=True
            )
            graph_col2.plotly_chart(
                px.line(df_all_2, title="True Alpha (%)"),
                use_container_width=True
            )

            st.divider()
            st.subheader("Detailed Simulation Data")
            # cum re per ticker
            for c in [c for c in df_new.columns if c.endswith('_re')]:
                df_new[c] = df_new[c].cumsum()
            st.plotly_chart(px.line(df_new, title="Portfolio Simulation Details"), use_container_width=True)
else:
    st.error("Could not load any configuration. Please check that 'un15_fx_config.json' exists and is correctly formatted.")
