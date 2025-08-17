# pages/7_🚀_Un_15_F(X).py
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import json
import plotly.express as px
import numpy_financial as npf
from typing import Dict, Tuple, Optional, List

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
        return {}, {} # คืนค่า dict ว่าง
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from '{filename}'. Please check its format.")
        return {}, {} # คืนค่า dict ว่าง

    fallback_default = {
        "Fixed_Asset_Value": 1500.0, "Cash_Balan": 650.0, "step": 0.01,
        "filter_date": "2024-01-01 12:00:00+07:00", "pred": 1
    }
    default_config = data.pop('__DEFAULT_CONFIG__', fallback_default)
    ticker_configs = data
    return ticker_configs, default_config

# ------------------- ฟังก์ชันคำนวณหลัก -------------------
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

def delta_1(asset_config):
    """Calculates Production_Costs based on asset configuration."""
    try:
        ticker_data = yf.Ticker(asset_config['Ticker'])
        entry = ticker_data.fast_info['lastPrice']
        df_model = calculate_cash_balance_model(entry, asset_config['step'], asset_config['Fixed_Asset_Value'], asset_config['Cash_Balan'])
        if not df_model.empty:
            production_costs = df_model['Cash_Balan'].iloc[-1] - asset_config['Cash_Balan']
            return abs(production_costs)
    except Exception:
        return None

def delta6(asset_config):
    """Performs historical simulation based on asset configuration."""
    try:
        ticker_hist = yf.Ticker(asset_config['Ticker']).history(period='max')
        if ticker_hist.empty:
            return None
            
        ticker_hist.index = ticker_hist.index.tz_convert(tz='Asia/bangkok')
        ticker_hist = ticker_hist[ticker_hist.index >= asset_config['filter_date']][['Close']]
        
        if ticker_hist.empty:
            return None

        entry = ticker_hist['Close'].iloc[0]
        df_model = calculate_cash_balance_model(entry, asset_config['step'], asset_config['Fixed_Asset_Value'], asset_config['Cash_Balan'])
        
        if df_model.empty:
            return None

        ticker_data = ticker_hist.copy()
        ticker_data['Close'] = np.around(ticker_data['Close'].values, 2)
        ticker_data['pred'] = asset_config['pred']
        ticker_data['Fixed_Asset_Value'] = asset_config['Fixed_Asset_Value']
        ticker_data['Amount_Asset'] = 0.0
        ticker_data['re'] = 0.0
        ticker_data['Cash_Balan'] = asset_config['Cash_Balan']
        ticker_data.iloc[0, ticker_data.columns.get_loc('Amount_Asset')] = ticker_data['Fixed_Asset_Value'].iloc[0] / ticker_data['Close'].iloc[0]

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

def un_16(active_configs: Dict[str, Dict]):
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

    df_re['maxcash_dd'] = df_re.sum(axis=1).cumsum()
    df_net_pv['cf'] = df_net_pv.sum(axis=1)

    final_df = pd.concat([df_re, df_net_pv], axis=1)
    return final_df

# ------------------- Helpers สำหรับ Optimization (ไม่กระทบ output เดิม) -------------------
@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_hist(ticker: str, filter_date: str) -> Optional[pd.DataFrame]:
    try:
        hist = yf.Ticker(ticker).history(period="max")
        if hist.empty:
            return None
        hist = hist.tz_convert("Asia/Bangkok")
        hist = hist[hist.index >= filter_date][['Close']].copy()
        return None if hist.empty else hist
    except Exception:
        return None

def _simulate_with_cash(hist: pd.DataFrame, asset_config: Dict, cash_balan: float) -> Optional[pd.DataFrame]:
    """จำลองเหมือน delta6 แต่ฉีด Cash_Balan ได้ โดยไม่ดึงข้อมูลซ้ำ"""
    try:
        entry = hist['Close'].iloc[0]
        df_model = calculate_cash_balance_model(entry, asset_config['step'], asset_config['Fixed_Asset_Value'], cash_balan)
        if df_model.empty:
            return None

        ticker_data = hist.copy()
        ticker_data['Close'] = np.around(ticker_data['Close'].values, 2)
        ticker_data['pred'] = asset_config['pred']
        ticker_data['Fixed_Asset_Value'] = asset_config['Fixed_Asset_Value']
        ticker_data['Amount_Asset'] = 0.0
        ticker_data['re'] = 0.0
        ticker_data['Cash_Balan'] = cash_balan
        # init
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
                re_vals[idx] = 0.0
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

def _score_for_cash(hist: pd.DataFrame, asset_config: Dict, cash_balan: float) -> Optional[Tuple[float, float, int]]:
    """
    คำนวณ Score ต่อ $1/วัน และ avg_daily_profit สำหรับ Cash_Balan ที่กำหนด
    Score_i = (avg_daily_profit_i) / cash_balan
    """
    sim = _simulate_with_cash(hist, asset_config, cash_balan)
    if sim is None or sim.empty:
        return None
    days = len(sim)
    if days == 0:
        return None
    avg_daily_profit = float(sim['re'].sum() / days)
    score = avg_daily_profit / max(cash_balan, 1e-9)
    return score, avg_daily_profit, days

def optimize_score_for_ticker(hist: pd.DataFrame, asset_config: Dict, lo: int, hi: int, step: int) -> Optional[Dict]:
    """
    ลอง Cash_Balan เป็น grid search [lo..hi] step=step แล้วหา cash ที่ทำให้ Score/1$/วัน สูงสุด
    คืน dict: {best_cash, best_score, avg_daily_profit, days}
    """
    best = {"best_cash": None, "best_score": -np.inf, "avg_daily_profit": 0.0, "days": 0}
    for c in range(int(lo), int(hi) + 1, int(step)):
        res = _score_for_cash(hist, asset_config, float(c))
        if res is None:
            continue
        score, avgp, days = res
        if score > best["best_score"]:
            best.update({"best_cash": float(c), "best_score": float(score), "avg_daily_profit": float(avgp), "days": int(days)})
    return None if best["best_cash"] is None else best

# ------------------- ส่วนแสดงผล STREAMLIT -------------------
st.set_page_config(page_title="Exist_F(X)", page_icon="☀", layout="wide")

# 1. โหลด config
full_config, DEFAULT_CONFIG = load_config()

if full_config or DEFAULT_CONFIG:
    # 2. ตั้งค่า Session State
    if 'custom_tickers' not in st.session_state:
        st.session_state.custom_tickers = {}

    # 3. ส่วนควบคุมบนหน้าหลัก
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

    # 4. สร้าง Dict Config ของ Ticker ที่เลือก
    active_configs = {ticker: full_config.get(ticker, st.session_state.custom_tickers.get(ticker)) for ticker in selected_tickers}

    if not active_configs:
        st.warning("Please select at least one ticker to start the analysis.")
    else:
        # 5. รันการคำนวณ (เดิม)
        with st.spinner('Calculating... Please wait.'):
            data = un_16(active_configs)

        if data.empty:
            st.error("Failed to generate data. This might happen if tickers have no historical data for the selected period or another error occurred.")
        else:
            # 6. คำนวณค่าสำหรับแสดงผล (เดิม)
            df_new = data.copy()
            roll_over = []
            max_dd_values = df_new.maxcash_dd.values
            for i in range(len(max_dd_values)):
                roll = max_dd_values[:i]
                roll_min = np.min(roll) if len(roll) > 0 else 0
                roll_over.append(roll_min)
            
            cf_values = df_new.cf.values
            df_all = pd.DataFrame({'Sum_Delta': cf_values, 'Max_Sum_Buffer': roll_over}, index=df_new.index)
            
            # --- GOAL 1 MODIFICATION - True Alpha ใหม่ (คงเดิมจากเวอร์ชันก่อนหน้านี้) ---
            num_selected_tickers = len(selected_tickers)
            initial_capital = num_selected_tickers * 1500.0
            max_buffer_used = abs(np.min(roll_over))
            total_capital_at_risk = initial_capital + max_buffer_used
            if total_capital_at_risk == 0:
                total_capital_at_risk = 1
            true_alpha_values = (df_new.cf.values / total_capital_at_risk) * 100
            df_all_2 = pd.DataFrame({'True_Alpha': true_alpha_values}, index=df_new.index)
            # --- END ---

            # 7. แสดงผล KPI (เดิม)
            st.subheader("Key Performance Indicators")
            final_sum_delta = df_all.Sum_Delta.iloc[-1]
            final_max_buffer = df_all.Max_Sum_Buffer.iloc[-1]
            final_true_alpha = df_all_2.True_Alpha.iloc[-1]
            num_days = len(df_new)
            avg_cf = final_sum_delta / num_days if num_days > 0 else 0
            avg_burn_cash = abs(final_max_buffer) / num_days if num_days > 0 else 0

            # --- MIRR CALCULATION (เดิม) ---
            mirr_value = 0.0
            initial_investment = (num_selected_tickers * 1500) + abs(final_max_buffer)
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

            # 8. กราฟ (เดิม)
            st.subheader("Performance Charts")
            graph_col1, graph_col2 = st.columns(2)
            graph_col1.plotly_chart(px.line(df_all.reset_index(drop=True), title="Cumulative Profit (Sum_Delta) vs. Max Buffer Used"), use_container_width=True)
            graph_col2.plotly_chart(px.line(df_all_2, title="True Alpha (%)"), use_container_width=True)
            st.divider()

            # ------------------- ✅ OPTIMIZATION PANEL (ใหม่ตาม Goal 1) -------------------
            with st.expander("🔎 Optimization: Score per $1/day & Target Planner", expanded=True):
                left, mid, right = st.columns([1.1, 1, 1])
                with left:
                    target_daily = st.number_input("Target Profit (USD/day)", min_value=0.0, value=100.0, step=10.0, help="เช่น 100 = อยากทำกำไรเฉลี่ย 100$/วัน")
                with mid:
                    opt_min, opt_max = st.slider("Cash_Balan Range (USD)", 1, 3000, value=(1, 3000), step=1)
                with right:
                    opt_step = st.selectbox("Grid Step (USD)", options=[1, 5, 10, 25, 50, 100], index=3, help="ยิ่งละเอียด ยิ่งช้า")
                
                run_opt = st.button("Run Optimization", type="primary", use_container_width=True)

                if run_opt:
                    st.info("คำนวณ Score ต่อ $1/วัน แบบ grid search ต่อ Ticker โดยไม่เปลี่ยนผลลัพธ์ส่วนอื่นของหน้า")
                    results = []
                    # เตรียม historical data ล่วงหน้า (cache)
                    hist_cache: Dict[str, Optional[pd.DataFrame]] = {}
                    for tkr, cfg in active_configs.items():
                        hist_cache[tkr] = _fetch_hist(cfg['Ticker'].strip(), cfg['filter_date'])

                    for tkr, cfg in active_configs.items():
                        hist = hist_cache.get(tkr)
                        if hist is None:
                            continue
                        opt = optimize_score_for_ticker(hist, cfg, lo=opt_min, hi=opt_max, step=int(opt_step))
                        if opt is None:
                            continue
                        results.append({
                            "Ticker": tkr,
                            "Best_Cash_Balan": opt["best_cash"],
                            "Score_per_$1_per_day": opt["best_score"],
                            "Avg_Daily_Profit_at_Best": opt["avg_daily_profit"],
                            "Days": opt["days"]
                        })

                    if len(results) == 0:
                        st.warning("No optimization results (อาจเพราะข้อมูลไม่พอหรือช่วงเวลาที่เลือกบางตัวไม่มีข้อมูล)")
                    else:
                        df_opt = pd.DataFrame(results).sort_values(by="Score_per_$1_per_day", ascending=False).reset_index(drop=True)
                        st.dataframe(df_opt, use_container_width=True)
                        
                        # พอร์ต Score รวม
                        portfolio_score_per_$1 = float(df_opt["Score_per_$1_per_day"].sum())
                        st.markdown(f"**Portfolio Score per $1/day = {portfolio_score_per_$1:,.6f}**")
                        
                        # เงินรวมที่ต้องใช้เพื่อแตะ Target: Cash = Target / Score_per_$1
                        required_cash_total = (target_daily / portfolio_score_per_$1) if portfolio_score_per_$1 > 0 else np.nan
                        st.markdown(f"**Required Cash_Balan (est.) to reach {target_daily:,.2f} USD/day = {required_cash_total:,.2f} USD**")

                        # จัดสรรตามสัดส่วน Score (สมมติ linear scaling)
                        if portfolio_score_per_$1 > 0 and np.isfinite(required_cash_total):
                            df_opt["Weight"] = df_opt["Score_per_$1_per_day"] / portfolio_score_per_$1
                            df_opt["Proposed_Cash_Balan"] = df_opt["Weight"] * required_cash_total
                            df_opt["Expected_Daily_Profit"] = df_opt["Score_per_$1_per_day"] * df_opt["Proposed_Cash_Balan"]
                            st.markdown("**Proposed Allocation (by Score weight)**")
                            st.dataframe(
                                df_opt[["Ticker", "Score_per_$1_per_day", "Weight", "Proposed_Cash_Balan", "Expected_Daily_Profit"]],
                                use_container_width=True
                            )

                            # MIRR Proxy (3Y) จาก Target (สมมติทำได้จริงตามแผน)
                            # cash_flows = [-Initial, CF, CF, CF + ExitMultiple]
                            # Initial ≈ (1500 * N) + Required_Cash_Total  (proxy)
                            initial_proxy = (len(df_opt) * 1500.0) + required_cash_total
                            annual_cf_proxy = target_daily * 252.0
                            exit_multiple_proxy = initial_proxy * 0.5
                            mirr_proxy = 0.0
                            if initial_proxy > 0:
                                cash_flows_proxy = [-initial_proxy, annual_cf_proxy, annual_cf_proxy, annual_cf_proxy + exit_multiple_proxy]
                                mirr_proxy = float(npf.mirr(cash_flows_proxy, 0.0, 0.0))
                            st.metric(label="MIRR (3-Year) — Proxy from Target Plan", value=f"{mirr_proxy:.2%}")
                        else:
                            st.warning("ไม่สามารถคำนวณการจัดสรร/ MIRR proxy ได้เพราะ Portfolio Score เป็นศูนย์หรือไม่กำหนด")

            st.divider()
            # ------------------- END OPTIMIZATION PANEL -------------------

            st.subheader("Detailed Simulation Data")
            for ticker in selected_tickers:
                col_name = f'{ticker}_re'
                if col_name in df_new.columns:
                    df_new[col_name] = df_new[col_name].cumsum()
            st.plotly_chart(px.line(df_new, title="Portfolio Simulation Details"), use_container_width=True)

else:
    st.error("Could not load any configuration. Please check that 'un15_fx_config.json' exists and is correctly formatted.")
