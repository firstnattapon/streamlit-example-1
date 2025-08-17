import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy_financial as npf
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

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
    except Exception as e:
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
        ticker_data['Amount_Asset'].iloc[0] = ticker_data['Fixed_Asset_Value'].iloc[0] / ticker_data['Close'].iloc[0]

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
        ticker_data = ticker_data.merge(df_model[['Asset_Price', 'Cash_Balan']].rename(columns={'Cash_Balan': 'refer_model'}), 
                                        left_on='Close', right_on='Asset_Price', how='left').drop('Asset_Price', axis=1)
        ticker_data.set_index(original_index, inplace=True)

        ticker_data['refer_model'].interpolate(method='linear', inplace=True)
        ticker_data.fillna(method='bfill', inplace=True)
        ticker_data.fillna(method='ffill', inplace=True)

        ticker_data['pv'] = ticker_data['Cash_Balan'] + (ticker_data['Amount_Asset'] * ticker_data['Close'])
        ticker_data['refer_pv'] = ticker_data['refer_model'] + asset_config['Fixed_Asset_Value']
        ticker_data['net_pv'] = ticker_data['pv'] - ticker_data['refer_pv']
        
        return ticker_data[['net_pv', 're']]
        
    except Exception as e:
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

    df_re['maxcash_dd'] = df_re.sum(axis=1).cumsum()
    df_net_pv['cf'] = df_net_pv.sum(axis=1)

    final_df = pd.concat([df_re, df_net_pv], axis=1)
    return final_df

# ------------------- NEW: Cash_Balan Optimization Functions -------------------
def calculate_mirr_for_cash_balan(cash_balan_value, active_configs, num_selected_tickers):
    """Calculate MIRR for a specific Cash_Balan value."""
    try:
        # Update all configs with new Cash_Balan
        test_configs = {}
        for ticker, config in active_configs.items():
            test_config = config.copy()
            test_config['Cash_Balan'] = cash_balan_value
            test_configs[ticker] = test_config
        
        # Run simulation
        data = un_16(test_configs)
        if data.empty:
            return cash_balan_value, 0.0, 0.0, 0.0
        
        # Calculate metrics
        df_new = data.copy()
        roll_over = []
        max_dd_values = df_new.maxcash_dd.values
        for i in range(len(max_dd_values)):
            roll = max_dd_values[:i]
            roll_min = np.min(roll) if len(roll) > 0 else 0
            roll_over.append(roll_min)
        
        final_sum_delta = df_new.cf.iloc[-1]
        final_max_buffer = roll_over[-1] if roll_over else 0
        num_days = len(df_new)
        avg_cf = final_sum_delta / num_days if num_days > 0 else 0
        
        # MIRR calculation
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
        else:
            mirr_value = 0.0
        
        # Score per 1 USD calculation
        score_per_usd = (avg_cf * 252) / cash_balan_value if cash_balan_value > 0 else 0
        
        return cash_balan_value, mirr_value, score_per_usd, avg_cf * 252
    
    except Exception as e:
        return cash_balan_value, 0.0, 0.0, 0.0

def optimize_cash_balan(active_configs, selected_tickers, optimization_range=(1, 3000, 50)):
    """Optimize Cash_Balan to maximize MIRR."""
    start_val, end_val, step_val = optimization_range
    cash_balan_values = list(range(start_val, end_val + 1, step_val))
    num_selected_tickers = len(selected_tickers)
    
    optimization_results = []
    
    # Use progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, cash_val in enumerate(cash_balan_values):
        status_text.text(f'Optimizing Cash_Balan: {cash_val} USD ({i+1}/{len(cash_balan_values)})')
        progress_bar.progress((i + 1) / len(cash_balan_values))
        
        cash_val, mirr, score_per_usd, annual_profit = calculate_mirr_for_cash_balan(
            cash_val, active_configs, num_selected_tickers
        )
        
        optimization_results.append({
            'Cash_Balan': cash_val,
            'MIRR': mirr,
            'Score_per_USD': score_per_usd,
            'Annual_Profit': annual_profit
        })
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(optimization_results)

# ------------------- ส่วนแสดงผล STREAMLIT -------------------
st.set_page_config(page_title="Optimized F(X) System", page_icon="🚀", layout="wide")

# 1. โหลด config
full_config, DEFAULT_CONFIG = load_config()

if full_config or DEFAULT_CONFIG:
    # 2. ตั้งค่า Session State
    if 'custom_tickers' not in st.session_state:
        st.session_state.custom_tickers = {}
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = None
    if 'optimal_cash_balan' not in st.session_state:
        st.session_state.optimal_cash_balan = DEFAULT_CONFIG.get('Cash_Balan', 650.0)

    # 3. ส่วนควบคุมบนหน้าหลัก
    st.title("🚀 Optimized F(X) System with Cash_Balan Optimization")
    
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
        default_selection = [t for t in list(full_config.keys())[:5] if t in all_tickers]  # Limit default selection
        selected_tickers = st.multiselect(
            "Select from available tickers:",
            options=all_tickers,
            default=default_selection
        )

    # 4. Optimization Section
    st.divider()
    st.subheader("🎯 Cash_Balan Optimization")
    
    opt_col1, opt_col2, opt_col3 = st.columns([1, 1, 1])
    with opt_col1:
        min_cash = st.number_input("Min Cash_Balan (USD)", min_value=1, max_value=2999, value=100, step=50)
    with opt_col2:
        max_cash = st.number_input("Max Cash_Balan (USD)", min_value=2, max_value=3000, value=1500, step=50)
    with opt_col3:
        step_cash = st.number_input("Step Size (USD)", min_value=10, max_value=200, value=50, step=10)

    if st.button("🔍 Run Cash_Balan Optimization", use_container_width=True, type="primary"):
        if not selected_tickers:
            st.error("Please select at least one ticker to run optimization.")
        else:
            active_configs = {ticker: full_config.get(ticker, st.session_state.custom_tickers.get(ticker)) 
                            for ticker in selected_tickers}
            
            with st.spinner('🔍 Running Cash_Balan optimization... This may take a few minutes.'):
                st.session_state.optimization_results = optimize_cash_balan(
                    active_configs, 
                    selected_tickers, 
                    (min_cash, max_cash, step_cash)
                )
            
            if not st.session_state.optimization_results.empty:
                # Find optimal Cash_Balan
                best_result = st.session_state.optimization_results.loc[
                    st.session_state.optimization_results['MIRR'].idxmax()
                ]
                st.session_state.optimal_cash_balan = best_result['Cash_Balan']
                
                st.success(f"✅ Optimization Complete! Optimal Cash_Balan: {st.session_state.optimal_cash_balan} USD")
                st.success(f"📊 Max MIRR: {best_result['MIRR']:.2%}")
                st.success(f"🎯 Score per USD: {best_result['Score_per_USD']:.4f}")

    # 5. Display Optimization Results
    if st.session_state.optimization_results is not None and not st.session_state.optimization_results.empty:
        st.divider()
        st.subheader("📈 Optimization Results")
        
        # Create optimization charts
        opt_results = st.session_state.optimization_results
        
        # Charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            fig_mirr = px.line(opt_results, x='Cash_Balan', y='MIRR', 
                              title='MIRR vs Cash_Balan',
                              labels={'Cash_Balan': 'Cash Balance (USD)', 'MIRR': 'MIRR (%)'})
            fig_mirr.update_traces(line_color='green')
            st.plotly_chart(fig_mirr, use_container_width=True)
        
        with chart_col2:
            fig_score = px.line(opt_results, x='Cash_Balan', y='Score_per_USD',
                               title='Score per USD vs Cash_Balan',
                               labels={'Cash_Balan': 'Cash Balance (USD)', 'Score_per_USD': 'Score per USD'})
            fig_score.update_traces(line_color='blue')
            st.plotly_chart(fig_score, use_container_width=True)
        
        # Show top 10 results
        st.subheader("🏆 Top 10 Optimization Results")
        top_results = opt_results.nlargest(10, 'MIRR')[['Cash_Balan', 'MIRR', 'Score_per_USD', 'Annual_Profit']]
        top_results['MIRR'] = top_results['MIRR'].apply(lambda x: f"{x:.2%}")
        top_results['Score_per_USD'] = top_results['Score_per_USD'].apply(lambda x: f"{x:.4f}")
        top_results['Annual_Profit'] = top_results['Annual_Profit'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(top_results, use_container_width=True)

    st.divider()

    # 6. สร้าง Dict Config ของ Ticker ที่เลือก (ใช้ optimal Cash_Balan)
    if selected_tickers:
        active_configs = {}
        for ticker in selected_tickers:
            config = full_config.get(ticker, st.session_state.custom_tickers.get(ticker)).copy()
            config['Cash_Balan'] = st.session_state.optimal_cash_balan  # Use optimal value
            active_configs[ticker] = config

        # 7. รันการคำนวณด้วย Optimal Cash_Balan
        st.subheader(f"📊 Performance Analysis (Cash_Balan: {st.session_state.optimal_cash_balan} USD)")
        
        with st.spinner('Calculating with optimal Cash_Balan... Please wait.'):
            data = un_16(active_configs)

        if data.empty:
            st.error("Failed to generate data. This might happen if tickers have no historical data for the selected period or another error occurred.")
        else:
            # 8. คำนวณค่าสำหรับแสดงผล
            df_new = data.copy()
            roll_over = []
            max_dd_values = df_new.maxcash_dd.values
            for i in range(len(max_dd_values)):
                roll = max_dd_values[:i]
                roll_min = np.min(roll) if len(roll) > 0 else 0
                roll_over.append(roll_min)
            
            cf_values = df_new.cf.values
            df_all = pd.DataFrame({'Sum_Delta': cf_values, 'Max_Sum_Buffer': roll_over}, index=df_new.index)
            
            # Redefine True Alpha with optimal Cash_Balan
            num_selected_tickers = len(selected_tickers)
            initial_capital = num_selected_tickers * 1500.0
            max_buffer_used = abs(np.min(roll_over))
            total_capital_at_risk = initial_capital + max_buffer_used
            
            if total_capital_at_risk == 0:
                total_capital_at_risk = 1 

            true_alpha_values = (df_new.cf.values / total_capital_at_risk) * 100
            df_all_2 = pd.DataFrame({'True_Alpha': true_alpha_values}, index=df_new.index)

            # 9. แสดงผล KPI
            st.subheader("📈 Key Performance Indicators")
            final_sum_delta = df_all.Sum_Delta.iloc[-1]
            final_max_buffer = df_all.Max_Sum_Buffer.iloc[-1]
            final_true_alpha = df_all_2.True_Alpha.iloc[-1]
            num_days = len(df_new)
            avg_cf = final_sum_delta / num_days if num_days > 0 else 0
            avg_burn_cash = abs(final_max_buffer) / num_days if num_days > 0 else 0

            # MIRR CALCULATION
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
            
            # Score per USD calculation
            score_per_usd = (avg_cf * 252) / st.session_state.optimal_cash_balan if st.session_state.optimal_cash_balan > 0 else 0

            # KPI Display
            kpi1, kpi2, kpi3, kpi4, kpi5, kpi6, kpi7 = st.columns(7)
            kpi1.metric(label="Total Net Profit", value=f"${final_sum_delta:,.2f}")
            kpi2.metric(label="Max Cash Buffer Used", value=f"${final_max_buffer:,.2f}")
            kpi3.metric(label="True Alpha (%)", value=f"{final_true_alpha:,.2f}%")
            kpi4.metric(label="Avg. Daily Profit", value=f"${avg_cf:,.2f}")
            kpi5.metric(label="Avg. Daily Buffer Used", value=f"${avg_burn_cash:,.2f}")
            kpi6.metric(label="MIRR (3-Year)", value=f"{mirr_value:.2%}")
            kpi7.metric(label="Score per USD", value=f"{score_per_usd:.4f}")
            
            st.divider()

            # 10. แสดงผลกราฟ
            st.subheader("📊 Performance Charts")
            graph_col1, graph_col2 = st.columns(2)
            
            graph_col1.plotly_chart(px.line(df_all.reset_index(drop=True), title="Cumulative Profit vs. Max Buffer Used"), use_container_width=True)
            graph_col2.plotly_chart(px.line(df_all_2, title="True Alpha (%)"), use_container_width=True)
            
            st.divider()
            
            st.subheader("📋 Detailed Simulation Data")
            for ticker in selected_tickers:
                col_name = f'{ticker}_re'
                if col_name in df_new.columns:
                    df_new[col_name] = df_new[col_name].cumsum()
            
            st.plotly_chart(px.line(df_new, title="Portfolio Simulation Details"), use_container_width=True)
    else:
        st.warning("Please select at least one ticker to start the analysis.")

else:
    st.error("Could not load any configuration. Please check that 'un15_fx_config.json' exists and is correctly formatted.")
