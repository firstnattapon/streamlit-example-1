# un15_fx_config.json (No changes needed, remains the same as your provided file)

# streamlit_app.py
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import json
import plotly.express as px
import numpy_financial as npf
from datetime import datetime

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
    # Clean up ticker names that might have leading/trailing spaces
    cleaned_ticker_configs = {k.strip(): v for k, v in ticker_configs.items()}
    for ticker, config in cleaned_ticker_configs.items():
        config['Ticker'] = config['Ticker'].strip()
        
    return cleaned_ticker_configs, default_config

# ------------------- ฟังก์ชันคำนวณหลัก (Core Model) -------------------
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

# ------------------- ฟังก์ชันคำนวณสำหรับ Optimization Loop -------------------
def calculate_mirr_for_simulation(historical_data, asset_config, cash_balance_to_test):
    """
    A lean function to run a simulation on PRE-FETCHED data for a specific cash balance
    and return the calculated MIRR. Designed for speed inside an optimization loop.
    """
    if historical_data.empty:
        return -np.inf # Return a very small number if no data

    ticker_data = historical_data.copy()
    
    # Run the simulation logic
    ticker_data['re'] = 0.0
    cash_balan_sim_vals = np.zeros(len(ticker_data))
    cash_balan_sim_vals[0] = cash_balance_to_test
    amount_asset_vals = np.zeros(len(ticker_data))
    amount_asset_vals[0] = asset_config['Fixed_Asset_Value'] / ticker_data['Close'].iloc[0]

    re_vals = ticker_data['re'].values
    close_vals = ticker_data['Close'].values

    for idx in range(1, len(ticker_data)):
        # Simplified logic from delta6 for re-calculation
        prev_amount = amount_asset_vals[idx-1]
        current_amount = asset_config['Fixed_Asset_Value'] / close_vals[idx]
        amount_asset_vals[idx] = current_amount
        re_vals[idx] = (prev_amount * close_vals[idx]) - asset_config['Fixed_Asset_Value']
        cash_balan_sim_vals[idx] = cash_balan_sim_vals[idx-1] + re_vals[idx]

    ticker_data['Cash_Balan'] = cash_balan_sim_vals
    
    # Calculate metrics needed for MIRR
    final_sum_delta = ticker_data['Cash_Balan'].iloc[-1] - cash_balance_to_test
    num_days = len(ticker_data)
    avg_cf = final_sum_delta / num_days if num_days > 0 else 0
    
    # Calculate max cash buffer used
    roll_over = []
    max_dd_values = (ticker_data['Cash_Balan'] - cash_balance_to_test).cumsum()
    for i in range(len(max_dd_values)):
        roll = max_dd_values[:i]
        roll_min = np.min(roll) if len(roll) > 0 else 0
        roll_over.append(roll_min)
    final_max_buffer = np.min(roll_over) if roll_over else 0

    # Calculate MIRR
    initial_investment = asset_config['Fixed_Asset_Value'] + abs(final_max_buffer)
    if initial_investment <= 0:
        return -np.inf

    annual_cash_flow = avg_cf * 252
    exit_multiple = initial_investment * 0.5
    cash_flows = [
        -initial_investment,
        annual_cash_flow,
        annual_cash_flow,
        annual_cash_flow + exit_multiple
    ]
    
    try:
        mirr_value = npf.mirr(cash_flows, finance_rate=0.0, reinvest_rate=0.0)
        return mirr_value if np.isfinite(mirr_value) else -np.inf
    except Exception:
        return -np.inf

# ------------------- ฟังก์ชันจำลองและรวมผล (Portfolio Level) -------------------
def delta6(asset_config):
    """Performs historical simulation for a single asset with a given config."""
    try:
        ticker_hist = yf.Ticker(asset_config['Ticker']).history(period='max', auto_adjust=False, actions=False)
        if ticker_hist.empty: return None
            
        ticker_hist.index = ticker_hist.index.tz_convert(tz='Asia/bangkok')
        ticker_hist = ticker_hist[ticker_hist.index >= asset_config['filter_date']][['Close']]
        
        if ticker_hist.empty: return None

        entry = ticker_hist['Close'].iloc[0]
        df_model = calculate_cash_balance_model(entry, asset_config['step'], asset_config['Fixed_Asset_Value'], asset_config['Cash_Balan'])
        
        if df_model.empty: return None

        ticker_data = ticker_hist.copy()
        ticker_data['Close'] = np.around(ticker_data['Close'].values, 2)
        ticker_data['Fixed_Asset_Value'] = asset_config['Fixed_Asset_Value']
        ticker_data['Amount_Asset'] = 0.0
        ticker_data['re'] = 0.0
        ticker_data['Cash_Balan'] = asset_config['Cash_Balan']
        ticker_data['Amount_Asset'].iloc[0] = ticker_data['Fixed_Asset_Value'].iloc[0] / ticker_data['Close'].iloc[0]

        close_vals = ticker_data['Close'].values
        amount_asset_vals = ticker_data['Amount_Asset'].values
        re_vals = ticker_data['re'].values
        cash_balan_sim_vals = ticker_data['Cash_Balan'].values

        for idx in range(1, len(amount_asset_vals)):
            amount_asset_vals[idx] = asset_config['Fixed_Asset_Value'] / close_vals[idx]
            re_vals[idx] = (amount_asset_vals[idx-1] * close_vals[idx]) - asset_config['Fixed_Asset_Value']
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
        st.warning(f"Could not process {asset_config.get('Ticker', 'N/A')}: {e}")
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

# ------------------- ส่วนแสดงผล STREAMLIT -------------------
st.set_page_config(page_title="Exist_F(X) Optimizer", page_icon="☀", layout="wide")
st.title("Exist_F(X) - Cash Balance Optimizer")

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
        new_ticker = st.text_input("Ticker (e.g., AAPL):", key="new_ticker_input").upper().strip()
        if st.button("Add Ticker", key="add_ticker_button", use_container_width=True):
            if new_ticker and new_ticker not in full_config and new_ticker not in st.session_state.custom_tickers:
                st.session_state.custom_tickers[new_ticker] = {"Ticker": new_ticker, **DEFAULT_CONFIG}
                st.success(f"Added {new_ticker}!")
                st.rerun() 
            elif new_ticker in full_config or new_ticker in st.session_state.custom_tickers:
                st.warning(f"{new_ticker} already exists.")
            else:
                st.warning("Please enter a ticker symbol.")

    with control_col2:
        st.subheader("Select Tickers to Analyze")
        all_tickers = sorted(list(full_config.keys()) + list(st.session_state.custom_tickers.keys()))
        default_selection = [t for t in list(full_config.keys()) if t in all_tickers]
        selected_tickers = st.multiselect(
            "Select from available tickers:",
            options=all_tickers,
            default=default_selection
        )
    st.divider()
    
    # 4. สร้าง Dict Config ของ Ticker ที่เลือก
    active_configs = {ticker: full_config.get(ticker, st.session_state.custom_tickers.get(ticker)) for ticker in selected_tickers}
    
    if st.button("Run Optimization and Analysis", use_container_width=True, type="primary") and active_configs:
        
        # --- START: GOAL 1 - OPTIMIZATION PROCESS ---
        st.subheader("Optimization in Progress...")
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        optimized_configs = {}
        optimization_results = []
        
        total_tickers = len(active_configs)
        CASH_BALAN_RANGE = range(0, 3001, 50) # Range from 0 to 3000, with a step of 50 for speed

        for i, (ticker_name, config) in enumerate(active_configs.items()):
            status_text.info(f"({i+1}/{total_tickers}) Optimizing for **{ticker_name}**... Fetching data...")
            
            # Fetch data ONCE per ticker
            try:
                hist_data = yf.Ticker(config['Ticker']).history(period='max', auto_adjust=False, actions=False)
                hist_data.index = hist_data.index.tz_convert(tz='Asia/bangkok')
                hist_data = hist_data[hist_data.index >= config['filter_date']][['Close']]
                if hist_data.empty:
                    st.warning(f"No historical data for {ticker_name} in the selected date range. Skipping.")
                    continue
            except Exception as e:
                st.warning(f"Could not fetch data for {ticker_name}: {e}. Skipping.")
                continue

            best_mirr = -np.inf
            optimal_cash_balan = config['Cash_Balan'] # Default to config value

            # Loop through cash balance range to find the best MIRR
            for cash_val in CASH_BALAN_RANGE:
                status_text.info(f"({i+1}/{total_tickers}) Optimizing for **{ticker_name}**... Testing Cash Balance: ${cash_val:,.0f}")
                current_mirr = calculate_mirr_for_simulation(hist_data, config, cash_val)
                if current_mirr > best_mirr:
                    best_mirr = current_mirr
                    optimal_cash_balan = cash_val
            
            progress_bar.progress((i + 1) / total_tickers)

            # Store the result
            new_config = config.copy()
            new_config['Cash_Balan'] = optimal_cash_balan
            optimized_configs[ticker_name] = new_config
            optimization_results.append({
                "Ticker": ticker_name,
                "Optimal Cash Balance": f"${optimal_cash_balan:,.2f}",
                "Best MIRR": f"{best_mirr:.2%}" if best_mirr != -np.inf else "N/A"
            })

        status_text.success("Optimization Complete! Running final portfolio simulation.")
        
        # Display Optimization Results
        st.subheader("Optimization Results")
        st.table(pd.DataFrame(optimization_results).set_index("Ticker"))
        st.divider()
        # --- END: GOAL 1 ---

        # 5. รันการคำนวณสุดท้ายด้วย Config ที่ดีที่สุด
        with st.spinner('Calculating final portfolio...'):
            data = un_16(optimized_configs)

        if data.empty:
            st.error("Failed to generate final data after optimization.")
        else:
            # --- The rest of the display logic is UNCHANGED as per goal_2 ---
            df_new = data.copy()
            roll_over = []
            max_dd_values = df_new.maxcash_dd.values
            for i in range(len(max_dd_values)):
                roll = max_dd_values[:i]
                roll_min = np.min(roll) if len(roll) > 0 else 0
                roll_over.append(roll_min)
            
            cf_values = df_new.cf.values
            df_all = pd.DataFrame({'Sum_Delta': cf_values, 'Max_Sum_Buffer': roll_over}, index=df_new.index)
            
            num_selected_tickers = len(optimized_configs)
            # Use sum of Fixed_Asset_Value from the actual configs used
            initial_capital = sum(c['Fixed_Asset_Value'] for c in optimized_configs.values())
            max_buffer_used = abs(np.min(roll_over))
            total_capital_at_risk = initial_capital + max_buffer_used
            if total_capital_at_risk == 0: total_capital_at_risk = 1 

            true_alpha_values = (df_new.cf.values / total_capital_at_risk) * 100
            df_all_2 = pd.DataFrame({'True_Alpha': true_alpha_values}, index=df_new.index)

            st.subheader("Key Performance Indicators (Based on Optimized Values)")
            final_sum_delta = df_all.Sum_Delta.iloc[-1]
            final_max_buffer = df_all.Max_Sum_Buffer.iloc[-1]
            final_true_alpha = df_all_2.True_Alpha.iloc[-1]
            num_days = len(df_new)
            avg_cf = final_sum_delta / num_days if num_days > 0 else 0
            avg_burn_cash = abs(final_max_buffer) / num_days if num_days > 0 else 0

            # MIRR calculation for the entire portfolio
            initial_investment_portfolio = initial_capital + abs(final_max_buffer)
            mirr_value_portfolio = 0.0
            if initial_investment_portfolio > 0:
                annual_cash_flow = avg_cf * 252
                exit_multiple = initial_investment_portfolio * 0.5
                cash_flows = [-initial_investment_portfolio, annual_cash_flow, annual_cash_flow, annual_cash_flow + exit_multiple]
                mirr_value_portfolio = npf.mirr(cash_flows, 0.0, 0.0)

            kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
            kpi1.metric(label="Total Net Profit (cf)", value=f"${final_sum_delta:,.2f}")
            kpi2.metric(label="Max Cash Buffer Used", value=f"${final_max_buffer:,.2f}")
            kpi3.metric(label="True Alpha (%)", value=f"{final_true_alpha:,.2f}%")
            kpi4.metric(label="Avg. Daily Profit", value=f"${avg_cf:,.2f}")
            kpi5.metric(label="Avg. Daily Buffer Used", value=f"${avg_burn_cash:,.2f}")
            kpi6.metric(label="Portfolio MIRR (3-Yr)", value=f"{mirr_value_portfolio:.2%}")
            
            st.divider()

            st.subheader("Performance Charts")
            graph_col1, graph_col2 = st.columns(2)
            graph_col1.plotly_chart(px.line(df_all, title="Cumulative Profit (Sum_Delta) vs. Max Buffer Used"), use_container_width=True)
            graph_col2.plotly_chart(px.line(df_all_2, title="True Alpha (%)"), use_container_width=True)
            
            st.divider()
            
            st.subheader("Detailed Simulation Data")
            for ticker in optimized_configs.keys():
                col_name = f'{ticker}_re'
                if col_name in df_new.columns:
                    df_new[col_name] = df_new[col_name].cumsum()
            
            st.plotly_chart(px.line(df_new.drop(columns=['maxcash_dd', 'cf']), title="Portfolio Simulation Details"), use_container_width=True)

    elif not active_configs:
        st.warning("Please select at least one ticker and click the 'Run' button to start.")

else:
    st.error("Could not load any configuration. Please check that 'un15_fx_config.json' exists and is correctly formatted.")
