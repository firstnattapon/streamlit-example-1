import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import json
import plotly.express as px
import numpy_financial as npf

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

def calculate_risk_parity_weights(returns_data):
    """
    คำนวณ Risk Parity Weights สำหรับ Dynamic c
    Returns: array of weights summing to 1
    """
    if returns_data.empty:
        return np.array([1.0])  # Default weight if no data
    
    # คำนวณ volatility ของแต่ละ asset
    volatilities = returns_data.std()
    
    # Risk Parity: weight = 1/volatility
    inverse_vol = 1 / (volatilities + 1e-8)  # เพิ่ม small epsilon เพื่อป้องกันการหารด้วยศูนย์
    
    # Normalize weights ให้รวมเป็น 1
    weights = inverse_vol / inverse_vol.sum()
    
    return weights.values

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
        return pd.DataFrame(), np.array([1.0])
        
    df_re = pd.concat(all_re, axis=1)
    df_net_pv = pd.concat(all_net_pv, axis=1)

    df_re.fillna(0, inplace=True)
    df_net_pv.fillna(0, inplace=True)

    # คำนวณ Risk Parity Weights
    risk_parity_weights = calculate_risk_parity_weights(df_re)

    df_re['maxcash_dd'] = df_re.sum(axis=1).cumsum()
    df_net_pv['cf'] = df_net_pv.sum(axis=1)

    final_df = pd.concat([df_re, df_net_pv], axis=1)
    return final_df, risk_parity_weights

def calculate_logarithmic_true_alpha(performance, risk_measure, weights, baseline=0):
    """
    คำนวณ True Alpha ตามสูตร F = c ⋅ ln(p/t) + b
    
    Parameters:
    - performance (p): กำไรสะสม
    - risk_measure (t): ความเสี่ยง (ใช้ max drawdown)
    - weights (c): Dynamic c จาก risk parity
    - baseline (b): ค่าฐาน (default = 0)
    
    Returns:
    - True Alpha value
    """
    # ป้องกันการหารด้วยศูนย์และ log ของจำนวนลบ
    if risk_measure <= 0:
        risk_measure = 1e-8
    
    if performance <= 0:
        performance = 1e-8
    
    # Dynamic c = average of risk parity weights
    c = np.mean(weights)
    
    # F = c ⋅ ln(p/t) + b
    p_over_t = performance / abs(risk_measure)
    true_alpha = c * np.log(p_over_t) + baseline
    
    return true_alpha * 100  # แปลงเป็น %

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
        # 5. รันการคำนวณ
        with st.spinner('Calculating... Please wait.'):
            data, risk_parity_weights = un_16(active_configs)

        if data.empty:
            st.error("Failed to generate data. This might happen if tickers have no historical data for the selected period or another error occurred.")
        else:
            # 6. คำนวณค่าสำหรับแสดงผล
            df_new = data.copy()
            roll_over = []
            max_dd_values = df_new.maxcash_dd.values
            for i in range(len(max_dd_values)):
                roll = max_dd_values[:i]
                roll_min = np.min(roll) if len(roll) > 0 else 0
                roll_over.append(roll_min)
            
            cf_values = df_new.cf.values
            df_all = pd.DataFrame({'Sum_Delta': cf_values, 'Max_Sum_Buffer': roll_over}, index=df_new.index)
            
            # --- NEW: LOGARITHMIC TRUE ALPHA CALCULATION ---
            num_selected_tickers = len(selected_tickers)
            initial_capital = num_selected_tickers * 1500.0
            max_buffer_used = abs(np.min(roll_over))
            total_capital_at_risk = initial_capital + max_buffer_used
            
            if total_capital_at_risk == 0:
                total_capital_at_risk = 1 

            # คำนวณ True Alpha ใหม่ตามสูตร F = c ⋅ ln(p/t) + b
            true_alpha_values = []
            for i, cf_val in enumerate(cf_values):
                if i == 0:
                    true_alpha_values.append(0)
                else:
                    # p = cumulative profit, t = total capital at risk
                    true_alpha = calculate_logarithmic_true_alpha(
                        performance=cf_val + total_capital_at_risk,  # เพิ่ม capital เพื่อให้เป็นค่าบวก
                        risk_measure=total_capital_at_risk,
                        weights=risk_parity_weights,
                        baseline=0
                    )
                    true_alpha_values.append(true_alpha)
            
            df_all_2 = pd.DataFrame({'True_Alpha': true_alpha_values}, index=df_new.index)
            # --- END: NEW TRUE ALPHA CALCULATION ---

            # 7. แสดงผล KPI
            st.subheader("Key Performance Indicators")
            final_sum_delta = df_all.Sum_Delta.iloc[-1]
            final_max_buffer = df_all.Max_Sum_Buffer.iloc[-1]
            final_true_alpha = df_all_2.True_Alpha.iloc[-1]
            num_days = len(df_new)
            avg_cf = final_sum_delta / num_days if num_days > 0 else 0
            avg_burn_cash = abs(final_max_buffer) / num_days if num_days > 0 else 0

            # --- MIRR CALCULATION (UNCHANGED) ---
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
            # --- END MIRR CALCULATION ---

            # --- KPI Display ---
            kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
            kpi1.metric(label="Total Net Profit (cf)", value=f"{final_sum_delta:,.2f}")
            kpi2.metric(label="Max Cash Buffer Used", value=f"{final_max_buffer:,.2f}")
            kpi3.metric(label="True Alpha (%) [Log Formula]", value=f"{final_true_alpha:,.2f}%")
            kpi4.metric(label="Avg. Daily Profit", value=f"{avg_cf:,.2f}")
            kpi5.metric(label="Avg. Daily Buffer Used", value=f"{avg_burn_cash:,.2f}")
            kpi6.metric(label="MIRR (3-Year)", value=f"{mirr_value:.2%}")
            
            # แสดง Risk Parity Weights
            st.subheader("Risk Parity Analysis")
            weights_col1, weights_col2 = st.columns(2)
            with weights_col1:
                st.write("**Risk Parity Weights:**")
                for i, ticker in enumerate(selected_tickers):
                    if i < len(risk_parity_weights):
                        st.write(f"- {ticker}: {risk_parity_weights[i]:.4f}")
            
            with weights_col2:
                st.write("**Formula Components:**")
                st.write(f"- Dynamic c (avg weights): {np.mean(risk_parity_weights):.4f}")
                st.write(f"- Performance (p): {final_sum_delta + total_capital_at_risk:,.2f}")
                st.write(f"- Risk measure (t): {total_capital_at_risk:,.2f}")
                st.write(f"- Baseline (b): 0")
            
            st.divider()

            # 8. แสดงผลกราฟ
            st.subheader("Performance Charts")
            graph_col1, graph_col2 = st.columns(2)
            
            graph_col1.plotly_chart(px.line(df_all.reset_index(drop=True), title="Cumulative Profit (Sum_Delta) vs. Max Buffer Used"), use_container_width=True)
            graph_col2.plotly_chart(px.line(df_all_2, title="True Alpha (%) - Logarithmic Formula"), use_container_width=True)
            
            st.divider()
            
            st.subheader("Detailed Simulation Data")
            for ticker in selected_tickers:
                col_name = f'{ticker}_re'
                if col_name in df_new.columns:
                    df_new[col_name] = df_new[col_name].cumsum()
            
            st.plotly_chart(px.line(df_new, title="Portfolio Simulation Details"), use_container_width=True)

else:
    st.error("Could not load any configuration. Please check that 'un15_fx_config.json' exists and is correctly formatted.")
