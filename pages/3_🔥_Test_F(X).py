# pages/7_🚀_Un_15_F(X).py
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import json
import plotly.express as px
import numpy_financial as npf
from functools import lru_cache

# =================== CONFIG ===================
st.set_page_config(page_title="Exist_F(X) + Cash Optimizer", page_icon="☀", layout="wide")

# ------------------- ฟังก์ชันสำหรับโหลด Config -------------------
def load_config(filename="un15_fx_config.json"):
    """
    Loads configurations from a JSON file.
    It expects a special key '__DEFAULT_CONFIG__' for default values.
    Returns a tuple: (ticker_configs, default_config)
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        st.error(f"Error: Configuration file '{filename}' not found.")
        return {}, {}  # คืนค่า dict ว่าง
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from '{filename}'. Please check its format.")
        return {}, {}  # คืนค่า dict ว่าง

    fallback_default = {
        "Fixed_Asset_Value": 1500.0, "Cash_Balan": 650.0, "step": 0.01,
        "filter_date": "2024-01-01 12:00:00+07:00", "pred": 1
    }
    default_config = data.pop('__DEFAULT_CONFIG__', fallback_default)
    ticker_configs = data
    return ticker_configs, default_config

# =================== CORE MODEL ===================

def calculate_cash_balance_model(entry, step, Fixed_Asset_Value, Cash_Balan):
    """Calculates the core cash balance model DataFrame."""
    if entry >= 10000 or entry <= 0:
        return pd.DataFrame()

    samples = np.arange(0, np.around(entry, 2) * 3 + step, step)

    df = pd.DataFrame()
    df['Asset_Price'] = np.around(samples, 2)
    df['Fixed_Asset_Value'] = Fixed_Asset_Value
    df['Amount_Asset'] = df['Fixed_Asset_Value'] / df['Asset_Price']

    df_top = df[df.Asset_Price >= np.around(entry, 2)].copy()
    if not df_top.empty:
        df_top['Cash_Balan_top'] = (df_top['Amount_Asset'].shift(1) - df_top['Amount_Asset']) * df_top['Asset_Price']
        df_top.fillna(0, inplace=True)

        np_Cash_Balan_top = df_top['Cash_Balan_top'].values
        xx = np.zeros(len(np_Cash_Balan_top))
        y_0 = Cash_Balan
        for idx, v_0 in enumerate(np_Cash_Balan_top):
            z_0 = y_0 + v_0
            y_0 = z_0
            xx[idx] = y_0

        df_top['Cash_Balan'] = xx
        df_top = df_top.sort_values(by='Amount_Asset')[:-1]
    else:
        df_top = pd.DataFrame(columns=['Asset_Price', 'Fixed_Asset_Value', 'Amount_Asset', 'Cash_Balan'])

    df_down = df[df.Asset_Price <= np.around(entry, 2)].copy()
    if not df_down.empty:
        df_down['Cash_Balan_down'] = (df_down['Amount_Asset'].shift(-1) - df_down['Amount_Asset']) * df_down['Asset_Price']
        df_down.fillna(0, inplace=True)
        df_down = df_down.sort_values(by='Asset_Price', ascending=False)

        np_Cash_Balan_down = df_down['Cash_Balan_down'].values
        xxx = np.zeros(len(np_Cash_Balan_down))
        y_1 = Cash_Balan
        for idx, v_1 in enumerate(np_Cash_Balan_down):
            z_1 = y_1 + v_1
            y_1 = z_1
            xxx[idx] = y_1

        df_down['Cash_Balan'] = xxx
    else:
        df_down = pd.DataFrame(columns=['Asset_Price', 'Fixed_Asset_Value', 'Amount_Asset', 'Cash_Balan'])

    combined_df = pd.concat([df_top, df_down], axis=0, ignore_index=True)
    return combined_df[['Asset_Price', 'Fixed_Asset_Value', 'Amount_Asset', 'Cash_Balan']]

@lru_cache(maxsize=256)
def fetch_history_cached(ticker: str):
    """ดาวน์โหลดราคา (period='max') แบบ cache เพื่อลดการเรียกซ้ำ"""
    try:
        hist = yf.Ticker(ticker).history(period='max')
        return hist
    except Exception:
        return pd.DataFrame()

def delta6(asset_config):
    """Performs historical simulation based on asset configuration."""
    try:
        # history (cached)
        raw_hist = fetch_history_cached(asset_config['Ticker'])
        if raw_hist.empty:
            return None

        hist = raw_hist.copy()
        hist.index = hist.index.tz_localize(None) if hist.index.tz is None else hist.index
        hist.index = hist.index.tz_convert(tz='Asia/Bangkok') if hist.index.tz is not None else hist.index.tz_localize('Asia/Bangkok')

        # filter by date
        fdate = pd.Timestamp(asset_config['filter_date'])
        fdate = fdate.tz_convert('Asia/Bangkok') if fdate.tz is not None else fdate.tz_localize('Asia/Bangkok')
        hist = hist[hist.index >= fdate][['Close']]
        if hist.empty:
            return None

        entry = hist['Close'].iloc[0]
        df_model = calculate_cash_balance_model(entry, asset_config['step'], asset_config['Fixed_Asset_Value'], asset_config['Cash_Balan'])
        if df_model.empty:
            return None

        ticker_data = hist.copy()
        ticker_data['Close'] = np.around(ticker_data['Close'].values, 2)
        ticker_data['pred'] = asset_config['pred']
        ticker_data['Fixed_Asset_Value'] = asset_config['Fixed_Asset_Value']
        ticker_data['Amount_Asset'] = 0.0
        ticker_data['re'] = 0.0
        ticker_data['Cash_Balan'] = asset_config['Cash_Balan']
        ticker_data.iloc[0, ticker_data.columns.get_loc('Amount_Asset')] = (
            ticker_data['Fixed_Asset_Value'].iloc[0] / ticker_data['Close'].iloc[0]
        )

        close_vals = ticker_data['Close'].values
        pred_vals = ticker_data['pred'].values
        amount_asset_vals = ticker_data['Amount_Asset'].values
        re_vals = ticker_data['re'].values
        cash_balan_sim_vals = ticker_data['Cash_Balan'].values

        for idx in range(1, len(amount_asset_vals)):
            if pred_vals[idx] == 1:
                amount_asset_vals[idx] = asset_config['Fixed_Asset_Value'] / close_vals[idx]
                re_vals[idx] = (amount_asset_vals[idx-1] * close_vals[idx]) - asset_config['Fixed_Asset_Value']
            else:
                amount_asset_vals[idx] = amount_asset_vals[idx-1]
                re_vals[idx] = 0
            cash_balan_sim_vals[idx] = cash_balan_sim_vals[idx-1] + re_vals[idx]

        original_index = ticker_data.index
        ticker_data = ticker_data.merge(
            df_model[['Asset_Price', 'Cash_Balan']].rename(columns={'Cash_Balan': 'refer_model'}),
            left_on='Close', right_on='Asset_Price', how='left'
        ).drop('Asset_Price', axis=1)
        ticker_data.set_index(original_index, inplace=True)

        ticker_data['refer_model'].interpolate(method='linear', inplace=True)
        ticker_data.fillna(method='bfill', inplace=True)
        ticker_data.fillna(method='ffill', inplace=True)

        ticker_data['pv'] = ticker_data['Cash_Balan'] + (ticker_data['Amount_Asset'] * ticker_data['Close'])
        ticker_data['refer_pv'] = ticker_data['refer_model'] + asset_config['Fixed_Asset_Value']
        ticker_data['net_pv'] = ticker_data['pv'] - ticker_data['refer_pv']

        return ticker_data[['net_pv', 're']]

    except Exception:
        return None

def un_16(active_configs):
    """Aggregates results from multiple assets specified in active_configs."""
    all_re = []
    all_net_pv = []

    for ticker_name, config in active_configs.items():
        result_df = delta6(config)
        if result_df is not None and not result_df.empty:
            all_re.append(result_df[['re']].rename(columns={"re": f"{ticker_name}_re"}))
            all_net_pv.append(result_df[['net_pv']].rename(columns={"net_pv": f"{ticker_name}_net_pv"}))

    if not all_re:
        return pd.DataFrame()

    df_re = pd.concat(all_re, axis=1)
    df_net_pv = pd.concat(all_net_pv, axis=1)

    df_re.fillna(0, inplace=True)
    df_net_pv.fillna(0, inplace=True)

    df_re['maxcash_dd'] = df_re.sum(axis=1).cumsum()   # cash drawdown trace
    df_net_pv['cf'] = df_net_pv.sum(axis=1)            # cumulative net profit vs reference

    final_df = pd.concat([df_re, df_net_pv], axis=1)
    return final_df

# =================== HELPERS FOR OPTIMIZATION ===================

def compute_roll_min(series_values):
    roll_over = []
    for i in range(len(series_values)):
        roll = series_values[:i]
        roll_min = np.min(roll) if len(roll) > 0 else 0
        roll_over.append(roll_min)
    return np.array(roll_over)

def portfolio_metrics(df_new, selected_tickers, fixed_asset_sum, cash_per_ticker, safety_mult=1.0):
    """
    คืน KPI สำคัญสำหรับพอร์ต ณ cash_per_ticker (USD ต่อ 1 ticker)
    - MIRR (3Y)
    - Avg Daily Profit
    - Max buffer used (absolute)
    - Score per $1/day = avg_daily_profit / total_starting_cash
    - feasibility: total_starting_cash >= safety_mult * abs(max_buffer_used)
    """
    if df_new.empty:
        return None

    # roll min of cash dd
    max_dd_trace = df_new['maxcash_dd'].values
    roll_over = compute_roll_min(max_dd_trace)
    max_buffer_used = abs(np.min(roll_over))

    cf_values = df_new['cf'].values
    num_days = len(df_new)
    final_sum_delta = cf_values[-1] if num_days > 0 else 0.0
    avg_cf = final_sum_delta / num_days if num_days > 0 else 0.0

    num_tickers = len(selected_tickers)
    total_starting_cash = cash_per_ticker * num_tickers

    # Feasibility: ต้องมี cash >= buffer_used * safety
    feasible = total_starting_cash >= safety_mult * max_buffer_used

    # Initial investment = สินทรัพย์คงที่รวม + เงินสดตั้งต้นรวม (ไม่นับ buffer ซ้ำ)
    initial_investment = fixed_asset_sum + total_starting_cash

    # Annualize daily average to cash flow per year (252 trading days)
    annual_cash_flow = avg_cf * 252
    exit_multiple = initial_investment * 0.5

    try:
        mirr_value = npf.mirr(
            [-initial_investment, annual_cash_flow, annual_cash_flow, annual_cash_flow + exit_multiple],
            0.0, 0.0
        )
    except Exception:
        mirr_value = np.nan

    # True Alpha (คงสูตรเดิมของหน้าเดิม) = cf / (fixed_asset_sum + max_buffer_used)
    denom_alpha = fixed_asset_sum + max_buffer_used
    if denom_alpha == 0:
        denom_alpha = 1.0
    true_alpha_final = (final_sum_delta / denom_alpha) * 100.0

    score_per_usd_per_day = (avg_cf / total_starting_cash) if total_starting_cash > 0 else np.nan

    return {
        "feasible": feasible,
        "final_sum_delta": final_sum_delta,
        "max_buffer_used": max_buffer_used,
        "avg_daily_profit": avg_cf,
        "annual_cash_flow": annual_cash_flow,
        "MIRR_3Y": mirr_value,
        "True_Alpha_pct": true_alpha_final,
        "Score_per_$1_per_day": score_per_usd_per_day,
        "initial_investment": initial_investment,
        "total_starting_cash": total_starting_cash
    }

def simulate_with_cash(active_configs, cash_per_ticker_override):
    """ปรับ Cash_Balan ของทุก ticker แล้วรัน un_16"""
    mod_configs = {}
    for t, cfg in active_configs.items():
        newcfg = dict(cfg)
        newcfg['Cash_Balan'] = float(cash_per_ticker_override)
        mod_configs[t] = newcfg
    return un_16(mod_configs)

# =================== APP ===================

# 1) Load config
full_config, DEFAULT_CONFIG = load_config()

if not (full_config or DEFAULT_CONFIG):
    st.error("Could not load any configuration. Please check that 'un15_fx_config.json' exists and is correctly formatted.")
    st.stop()

# 2) Session state for runtime added tickers
if 'custom_tickers' not in st.session_state:
    st.session_state.custom_tickers = {}

# 3) Controls
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

st.divider()

# 4) Build active configs
active_configs = {ticker: full_config.get(ticker, st.session_state.custom_tickers.get(ticker)) for ticker in selected_tickers}
if not active_configs:
    st.warning("Please select at least one ticker to start the analysis.")
    st.stop()

# คำนวณมูลค่าทุนคงที่รวมจริง (ไม่ใช้ 1500 คงที่)
fixed_asset_sum = float(np.sum([cfg['Fixed_Asset_Value'] for cfg in active_configs.values()]))

# ============ Optimization Controls ============
with st.expander("🎯 Optimize Cash_Balan (per ticker) for Max MIRR (3-Year)", expanded=True):
    opt_enable = st.checkbox("Enable Optimization", value=True)
    cmin, cmax = st.columns(2)
    cash_min = cmin.number_input("Min Cash_Balan (per ticker, USD)", min_value=1.0, max_value=10000.0, value=1.0, step=1.0)
    cash_max = cmax.number_input("Max Cash_Balan (per ticker, USD)", min_value=cash_min, max_value=10000.0, value=3000.0, step=1.0)
    step = st.number_input("Grid Step (USD)", min_value=1.0, max_value=1000.0, value=50.0, step=1.0)
    safety_mult = st.number_input("Safety Buffer Multiplier (≥1.0)", min_value=1.0, max_value=2.0, value=1.0, step=0.1,
                                  help="ต้องมีเงินสดตั้งต้นรวม ≥ safety × max drawdown ที่ระบบใช้จริง")

    target_day = st.number_input("Target Avg Profit per Day (USD/day)", min_value=0.0, value=100.0, step=10.0)
    st.caption("Definition: Score per $1/day = (Average Daily Profit) / (Total Starting Cash_Balan across selected tickers)")

# ============ Run Optimization / Base Simulation ============
def run_once_and_metrics(cash_per_ticker):
    data = simulate_with_cash(active_configs, cash_per_ticker)
    if data.empty:
        return None, None
    metrics = portfolio_metrics(data, selected_tickers, fixed_asset_sum, cash_per_ticker, safety_mult=safety_mult)
    return data, metrics

best_choice = None
opt_table = None
chosen_cash = None
data_for_display = None
metrics_for_display = None

if opt_enable:
    candidates = np.arange(cash_min, cash_max + step, step, dtype=float)
    rows = []
    best_mirr = -np.inf
    for cval in candidates:
        data_c, m_c = run_once_and_metrics(cval)
        if (data_c is None) or (m_c is None):
            continue
        rows.append({
            "Cash_Balan_per_ticker": cval,
            "Feasible": m_c["feasible"],
            "MIRR_3Y": m_c["MIRR_3Y"],
            "Avg_Daily_Profit": m_c["avg_daily_profit"],
            "Score_per_$1_per_day": m_c["Score_per_$1_per_day"],
            "Max_Buffer_Used": m_c["max_buffer_used"],
            "True_Alpha_%": m_c["True_Alpha_pct"],
            "Initial_Investment": m_c["initial_investment"]
        })
        if m_c["feasible"] and np.isfinite(m_c["MIRR_3Y"]) and (m_c["MIRR_3Y"] > best_mirr):
            best_mirr = m_c["MIRR_3Y"]
            best_choice = (cval, data_c, m_c)

    if rows:
        opt_table = pd.DataFrame(rows).sort_values(by=["Feasible","MIRR_3Y"], ascending=[False, False]).reset_index(drop=True)

    if best_choice is not None:
        chosen_cash, data_for_display, metrics_for_display = best_choice
    else:
        st.warning("No feasible candidate under current safety constraint. Showing last computed candidate (if any).")
        if rows:
            # fallback: first row
            chosen_cash = float(opt_table.iloc[0]['Cash_Balan_per_ticker'])
            data_for_display, metrics_for_display = run_once_and_metrics(chosen_cash)
else:
    # ใช้ค่า Cash_Balan จาก config ของแต่ละ ticker (ไม่ optimize)
    # ใช้ค่าเฉลี่ย (แสดงผล)
    chosen_cash = float(np.mean([cfg['Cash_Balan'] for cfg in active_configs.values()]))
    data_for_display, metrics_for_display = run_once_and_metrics(chosen_cash)

if (data_for_display is None) or (metrics_for_display is None) or data_for_display.empty:
    st.error("Failed to generate data. This might happen if tickers have no historical data for the selected period or another error occurred.")
    st.stop()

# =================== KPI & Charts (เหมือนเดิม + KPI ใหม่) ===================
df_new = data_for_display.copy()

# เตรียม roll min และกราฟ set
roll_over = compute_roll_min(df_new['maxcash_dd'].values)
cf_values = df_new['cf'].values
df_all = pd.DataFrame({'Sum_Delta': cf_values, 'Max_Sum_Buffer': roll_over}, index=df_new.index)

# True Alpha (final point) ตามสูตรเดิมในหน้าเก่า
true_alpha_final = metrics_for_display["True_Alpha_pct"]

# KPI หลัก
st.subheader("Key Performance Indicators")

final_sum_delta = df_all['Sum_Delta'].iloc[-1]
final_max_buffer = df_all['Max_Sum_Buffer'].iloc[-1]
num_days = len(df_new)
avg_cf = final_sum_delta / num_days if num_days > 0 else 0.0

# MIRR และ Score
mirr_value = metrics_for_display["MIRR_3Y"]
score_per_1 = metrics_for_display["Score_per_$1_per_day"]
total_starting_cash = metrics_for_display["total_starting_cash"]

kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
kpi1.metric(label="Total Net Profit (cf)", value=f"{final_sum_delta:,.2f}")
kpi2.metric(label="Max Cash Buffer Used", value=f"{final_max_buffer:,.2f}")
kpi3.metric(label="True Alpha (%)", value=f"{true_alpha_final:,.2f}%")
kpi4.metric(label="Avg. Daily Profit", value=f"{avg_cf:,.2f}")
kpi5.metric(label="MIRR (3-Year)", value=f"{mirr_value:.2%}")
kpi6.metric(label="Score per $1/day", value=(f"{score_per_1:,.6f}" if np.isfinite(score_per_1) else "N/A"),
            help="AvgDailyProfit / (Cash_Balan_per_ticker × #tickers)")

st.caption(f"Chosen Cash_Balan per ticker = **{chosen_cash:,.2f} USD**  |  Total starting cash = {total_starting_cash:,.2f} USD")

# Target calculator
st.divider()
st.subheader("🎯 Target Cash Calculator")
if np.isfinite(score_per_1) and score_per_1 > 0:
    required_total_cash = target_day / score_per_1
    required_per_ticker = required_total_cash / max(1, len(selected_tickers))
    ok_safety = required_total_cash >= (metrics_for_display["max_buffer_used"] * safety_mult)
    cols = st.columns(3)
    cols[0].metric("Target (USD/day)", f"{target_day:,.2f}")
    cols[1].metric("Required Total Cash_Balan (USD)", f"{required_total_cash:,.2f}")
    cols[2].metric("Required Cash_Balan per ticker (USD)", f"{required_per_ticker:,.2f}")
    st.caption(f"Safety check: required_total_cash {'✓' if ok_safety else '✗'} ≥ safety × max_buffer_used "
               f"({required_total_cash:,.2f} {'>=' if ok_safety else '<'} {metrics_for_display['max_buffer_used']*safety_mult:,.2f})")
else:
    st.info("Score per $1/day ≤ 0 หรือคำนวณไม่ได้ จึงไม่สามารถประเมินเงินสดเพื่อให้ถึง target ได้")

st.divider()

# ตารางผลลัพธ์ optimization
if opt_enable and (opt_table is not None):
    st.subheader("Optimization Table (Cash_Balan per ticker)")
    st.dataframe(opt_table, use_container_width=True)

st.subheader("Performance Charts")
graph_col1, graph_col2 = st.columns(2)
graph_col1.plotly_chart(px.line(df_all.reset_index(drop=True), title="Cumulative Profit (Sum_Delta) vs. Max Buffer Used"),
                        use_container_width=True)
df_all_2 = pd.DataFrame({'True_Alpha': np.full(len(df_all), true_alpha_final)}, index=df_all.index)
graph_col2.plotly_chart(px.line(df_all_2, title="True Alpha (%)"), use_container_width=True)

st.divider()
st.subheader("Detailed Simulation Data")
for ticker in selected_tickers:
    col_name = f'{ticker}_re'
    if col_name in df_new.columns:
        df_new[col_name] = df_new[col_name].cumsum()
st.plotly_chart(px.line(df_new, title="Portfolio Simulation Details"), use_container_width=True)
