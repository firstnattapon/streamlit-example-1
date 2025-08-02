import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import json
import plotly.express as px

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
    return ticker_configs, default_config

# ------------------- ฟังก์ชันคำนวณหลัก (Refactored) -------------------
# --- REFACTOR: Replaced loops with vectorized pandas operations (.cumsum) for clarity and performance.
def calculate_cash_balance_model(entry, step, Fixed_Asset_Value, Cash_Balan):
    """Calculates the core cash balance model DataFrame."""
    if entry >= 10000 or entry <= 0:
        return pd.DataFrame()

    samples = np.arange(0, np.around(entry, 2) * 3 + step, step)
    
    df = pd.DataFrame()
    df['Asset_Price'] = np.around(samples, 2)
    df['Fixed_Asset_Value'] = Fixed_Asset_Value
    df['Amount_Asset'] = df['Fixed_Asset_Value'] / df['Asset_Price']

    # --- Top part calculation (Price >= entry) ---
    df_top = df[df.Asset_Price >= np.around(entry, 2)].copy()
    if not df_top.empty:
        df_top['Cash_Balan_top'] = (df_top['Amount_Asset'].shift(1) - df_top['Amount_Asset']) * df_top['Asset_Price']
        df_top.fillna(0, inplace=True)
        df_top['Cash_Balan'] = Cash_Balan + df_top['Cash_Balan_top'].cumsum()
        df_top = df_top.sort_values(by='Amount_Asset')[:-1]
    else:
        df_top = pd.DataFrame(columns=['Asset_Price', 'Fixed_Asset_Value', 'Amount_Asset', 'Cash_Balan'])

    # --- Down part calculation (Price <= entry) ---
    df_down = df[df.Asset_Price <= np.around(entry, 2)].copy()
    if not df_down.empty:
        df_down = df_down.sort_values(by='Asset_Price', ascending=False)
        df_down['Cash_Balan_down'] = (df_down['Amount_Asset'].shift(-1) - df_down['Amount_Asset']) * df_down['Asset_Price']
        df_down.fillna(0, inplace=True)
        df_down['Cash_Balan'] = Cash_Balan + df_down['Cash_Balan_down'].cumsum()
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

# --- REFACTOR: Replaced main loop with vectorized operations (.where, .ffill, .shift, .cumsum)
def delta6(asset_config):
    """Performs historical simulation based on asset configuration."""
    try:
        ticker_hist = yf.Ticker(asset_config['Ticker']).history(period='max')
        if ticker_hist.empty: return None
            
        ticker_hist.index = ticker_hist.index.tz_convert(tz='Asia/bangkok')
        ticker_hist = ticker_hist[ticker_hist.index >= asset_config['filter_date']][['Close']]
        if ticker_hist.empty: return None

        entry = ticker_hist['Close'].iloc[0]
        df_model = calculate_cash_balance_model(entry, asset_config['step'], asset_config['Fixed_Asset_Value'], asset_config['Cash_Balan'])
        if df_model.empty: return None

        ticker_data = ticker_hist.copy()
        ticker_data['Close'] = np.around(ticker_data['Close'].values, 2)
        
        # Vectorized calculation replacing the loop
        ticker_data['pred'] = asset_config['pred']
        
        # Calculate Amount_Asset using forward-fill for days where pred != 1
        initial_amount_asset = asset_config['Fixed_Asset_Value'] / ticker_data['Close'].iloc[0]
        amount_asset_calc = np.where(ticker_data['pred'] == 1, asset_config['Fixed_Asset_Value'] / ticker_data['Close'], np.nan)
        ticker_data['Amount_Asset'] = pd.Series(amount_asset_calc, index=ticker_data.index).fillna(initial_amount_asset).ffill()

        # Calculate 're' (realized profit/loss)
        prev_amount_asset = ticker_data['Amount_Asset'].shift(1)
        ticker_data['re'] = np.where(
            ticker_data['pred'] == 1,
            (prev_amount_asset * ticker_data['Close']) - asset_config['Fixed_Asset_Value'],
            0
        )
        ticker_data.loc[ticker_data.index[0], 're'] = 0 # First day has no 're'

        # Calculate cumulative cash balance
        ticker_data['Cash_Balan'] = asset_config['Cash_Balan'] + ticker_data['re'].cumsum()
        
        # Merge with model reference data
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
        
    except Exception as e:
        # st.error(f"Error in delta6 for {asset_config.get('Ticker', 'N/A')}: {e}") # Optional: for debugging
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
    
    if not all_re: return pd.DataFrame()
        
    df_re = pd.concat(all_re, axis=1)
    df_net_pv = pd.concat(all_net_pv, axis=1)

    df_re.fillna(0, inplace=True)
    df_net_pv.fillna(0, inplace=True)

    df_re['maxcash_dd'] = df_re.sum(axis=1).cumsum()
    df_net_pv['cf'] = df_net_pv.sum(axis=1)

    final_df = pd.concat([df_re, df_net_pv], axis=1)
    return final_df

# --- NEW: Function to calculate MIRR based on Goal 1 requirements ---
def calculate_mirr(avg_daily_profit, num_selected_tickers, max_buffer_used):
    """Calculates the projected 3-year MIRR."""
    try:
        # Constants
        TRADING_DAYS_PER_YEAR = 252
        PROJECT_DURATION_YEARS = 3
        REINVEST_RATE = 0.0  # 0%
        FINANCE_RATE = 0.0   # Assume finance rate is same as reinvest rate
        EXIT_MULTIPLE = 0.5

        # Calculate cash flow components based on provided formulas
        initial_investment_per_ticker = 1500
        total_asset_investment = num_selected_tickers * initial_investment_per_ticker
        initial_investment = total_asset_investment + abs(max_buffer_used)

        if initial_investment == 0: return 0.0

        annual_cash_flow = avg_daily_profit * TRADING_DAYS_PER_YEAR
        exit_value = initial_investment * EXIT_MULTIPLE
        
        # Create cash flow series
        cash_flows = [-initial_investment]
        for year in range(PROJECT_DURATION_YEARS):
            if year < PROJECT_DURATION_YEARS - 1:
                cash_flows.append(annual_cash_flow)
            else: # Final year includes exit value
                cash_flows.append(annual_cash_flow + exit_value)
        
        mirr_value = np.mirr(cash_flows, finance_rate=FINANCE_RATE, reinvest_rate=REINVEST_RATE)
        return mirr_value
    except Exception:
        return np.nan # Return Not a Number if calculation fails

# ------------------- ส่วนแสดงผล STREAMLIT -------------------
st.set_page_config(page_title="Exist_F(X)", page_icon="☀", layout="wide")

# 1. โหลด config
full_config, DEFAULT_CONFIG = load_config()

if full_config or DEFAULT_CONFIG:
    # 2. ตั้งค่า Session State
    if 'custom_tickers' not in st.session_state:
        st.session_state.custom_tickers = {}

    # 3. ส่วนควบคุมบนหน้าหลัก
    st.title("F(X) Model Dashboard")
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
        default_selection = list(full_config.keys())
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
            data = un_16(active_configs)

        if data.empty:
            st.error("Failed to generate data. This might happen if tickers have no historical data for the selected period or another error occurred.")
        else:
            # 6. คำนวณค่าสำหรับแสดงผล
            df_new = data.copy()
            
            # --- REFACTOR: Replaced loop with .cummin() for performance and clarity
            roll_over = df_new.maxcash_dd.cummin().values
            
            cf_values = df_new.cf.values
            df_all = pd.DataFrame({'Sum_Delta': cf_values, 'Max_Sum_Buffer': roll_over}, index=df_new.index)
            
            min_sum_val = np.min(roll_over)
            min_sum = abs(min_sum_val) if min_sum_val != 0 else 1
            true_alpha_values = (df_new.cf.values / min_sum) * 100
            df_all_2 = pd.DataFrame({'True_Alpha': true_alpha_values}, index=df_new.index)

            # 7. แสดงผล KPI
            st.subheader("Key Performance Indicators")
            final_sum_delta = df_all.Sum_Delta.iloc[-1]
            max_cash_buffer_used = df_all.Max_Sum_Buffer.min() # Use the overall minimum
            final_true_alpha = df_all_2.True_Alpha.iloc[-1]
            num_days = len(df_new)
            avg_cf = final_sum_delta / num_days if num_days > 0 else 0
            avg_burn_cash = abs(max_cash_buffer_used) / num_days if num_days > 0 else 0
            
            # --- NEW: Calculate and display MIRR ---
            mirr_value = calculate_mirr(avg_cf, len(selected_tickers), max_cash_buffer_used)

            kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
            kpi1.metric(label="Total Net Profit (cf)", value=f"{final_sum_delta:,.2f}")
            kpi2.metric(label="Max Cash Buffer Used", value=f"{max_cash_buffer_used:,.2f}")
            kpi3.metric(label="True Alpha (%)", value=f"{final_true_alpha:,.2f}%")
            kpi4.metric(label="Avg. Daily Profit", value=f"{avg_cf:,.2f}")
            kpi5.metric(label="Avg. Daily Buffer Used", value=f"{avg_burn_cash:,.2f}")
            kpi6.metric(label="MIRR (3-Year Proj.)", value=f"{mirr_value:.2%}", help="Modified Internal Rate of Return based on a 3-year projection with 0% reinvestment rate and a 0.5x exit multiple on initial investment.")
            
            st.divider()

            # 8. แสดงผลกราฟ
            st.subheader("Performance Charts")
            graph_col1, graph_col2 = st.columns(2)
            
            # This graph correctly uses a numeric index for visualization of progress over time steps.
            graph_col1.plotly_chart(px.line(df_all.reset_index(drop=True), title="Cumulative Profit vs. Max Buffer Used (by Timestep)"), use_container_width=True)
            
            # This graph correctly uses the datetime index.
            graph_col2.plotly_chart(px.line(df_all_2, title="True Alpha (%) Over Time"), use_container_width=True)
            
            st.divider()
            
            st.subheader("Detailed Simulation Data")
            # Calculate cumulative profit for each individual ticker for plotting
            for ticker in selected_tickers:
                col_name = f'{ticker}_re'
                if col_name in df_new.columns:
                    df_new[f'{ticker}_cum_re'] = df_new[col_name].cumsum()

            # Select only the relevant columns for the final chart
            plot_cols = [f'{ticker}_cum_re' for ticker in selected_tickers] + ['maxcash_dd', 'cf']
            st.plotly_chart(px.line(df_new[plot_cols], title="Portfolio Simulation Details (Cumulative)"), use_container_width=True)

else:
    st.error("Could not load any configuration. Please check that 'un15_fx_config.json' exists and is correctly formatted.")
