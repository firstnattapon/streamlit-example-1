import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import json
from typing import Dict, Optional

# --- Configuration & Data Loading ---

def load_config(filename: str = "un15_fx_config.json") -> Dict:
    """Loads asset configurations from a JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error: Configuration file '{filename}' not found. Please create it.")
        return {}
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from '{filename}'. Please check its format.")
        return {}

# --- Core Calculation Functions (Refactored for Performance & Clarity) ---

def calculate_cash_balance_model(entry: float, config: Dict) -> pd.DataFrame:
    """
    Calculates the core cash balance model DataFrame based on an entry price.
    This version is refactored to use vectorized operations (cumsum) instead of loops.
    """
    Fixed_Asset_Value = config['Fixed_Asset_Value']
    Cash_Balan = config['Cash_Balan']
    step = config['step']

    if entry >= 10000 or entry <= 0:
        return pd.DataFrame()

    samples = np.arange(0, np.around(entry, 2) * 3 + step, step)
    
    df = pd.DataFrame({'Asset_Price': np.around(samples, 2)})
    df['Fixed_Asset_Value'] = Fixed_Asset_Value
    # np.divide with where clause avoids division by zero errors
    df['Amount_Asset'] = np.divide(df['Fixed_Asset_Value'], df['Asset_Price'], 
                                   out=np.zeros_like(df['Asset_Price']), where=df['Asset_Price']!=0)

    # --- Top part calculation (Vectorized) ---
    df_top = df[df.Asset_Price >= np.around(entry, 2)].copy()
    df_top['Cash_Balan_Change'] = (df_top['Amount_Asset'].shift(1) - df_top['Amount_Asset']) * df_top['Asset_Price']
    df_top.fillna(0, inplace=True)
    df_top['Cash_Balan'] = Cash_Balan + df_top['Cash_Balan_Change'].cumsum()
    df_top = df_top.sort_values(by='Amount_Asset')[:-1]

    # --- Down part calculation (Vectorized) ---
    df_down = df[df.Asset_Price <= np.around(entry, 2)].copy()
    df_down['Cash_Balan_Change'] = (df_down['Amount_Asset'].shift(-1) - df_down['Amount_Asset']) * df_down['Asset_Price']
    df_down.fillna(0, inplace=True)
    df_down = df_down.sort_values(by='Asset_Price', ascending=False)
    df_down['Cash_Balan'] = Cash_Balan + df_down['Cash_Balan_Change'].cumsum()

    return pd.concat([df_top, df_down], axis=0).drop(columns=['Cash_Balan_Change'])


def delta_1(config: Dict) -> Optional[float]:
    """Calculates Production_Costs based on asset configuration."""
    try:
        tickerData = yf.Ticker(config['Ticker'])
        entry_price = tickerData.fast_info['lastPrice']
        
        df_model = calculate_cash_balance_model(entry_price, config)

        if not df_model.empty:
            production_costs = df_model['Cash_Balan'].iloc[-1] - config['Cash_Balan']
            return abs(production_costs)
    except Exception as e:
        st.warning(f"Could not calculate delta_1 for {config.get('Ticker', 'N/A')}: {e}")
    return None


def delta6(config: Dict) -> Optional[pd.DataFrame]:
    """
    Performs a vectorized historical simulation based on asset configuration.
    This version fixes the order of operations for Amount_Asset calculation.
    """
    try:
        # 1. Load data and initial setup
        ticker_hist = yf.Ticker(config['Ticker']).history(period='max')
        if ticker_hist.empty:
            st.warning(f"yfinance returned no data for Ticker: {config['Ticker']}")
            return None
        
        ticker_hist.index = ticker_hist.index.tz_convert(tz='Asia/bangkok')
        sim_df = ticker_hist[ticker_hist.index >= config['filter_date']][['Close']].copy()
        
        if sim_df.empty:
            st.warning(f"No historical data found for {config['Ticker']} after {config['filter_date']}")
            return None

        entry_price = sim_df['Close'].iloc[0]
        df_model = calculate_cash_balance_model(entry_price, config)
        if df_model.empty: return None

        # 2. Vectorized Simulation (FIXED LOGIC)
        sim_df['Close'] = np.around(sim_df['Close'], 2)
        sim_df['pred'] = config['pred']
        
        # --- Start of Corrected Logic ---
        # Calculate Amount_Asset for days where a trade happens (pred=1)
        sim_df['Amount_Asset'] = np.where(
            sim_df['pred'] == 1,
            np.divide(config['Fixed_Asset_Value'], sim_df['Close'], where=sim_df['Close']!=0),
            np.nan  # Leave non-trade days as NaN for now
        )
        
        # CRITICAL FIX: Set the initial value for the very first day, regardless of 'pred'
        initial_close = sim_df['Close'].iloc[0]
        if initial_close > 0:
            sim_df['Amount_Asset'].iloc[0] = config['Fixed_Asset_Value'] / initial_close
        else:
            sim_df['Amount_Asset'].iloc[0] = 0
        
        # Now, forward-fill any remaining NaNs (which are the pred=0 days)
        sim_df['Amount_Asset'].ffill(inplace=True)
        # --- End of Corrected Logic ---

        # Calculate realized_gain_loss using shift() and np.where
        sim_df['realized_gain_loss'] = np.where(
            sim_df['pred'] == 1,
            (sim_df['Amount_Asset'].shift(1) * sim_df['Close']) - config['Fixed_Asset_Value'],
            0
        )
        sim_df['realized_gain_loss'].iloc[0] = 0 # First day has no gain/loss

        # Calculate Cash_Balan using cumsum()
        sim_df['Cash_Balan'] = config['Cash_Balan'] + sim_df['realized_gain_loss'].cumsum()

        # 3. Merge with reference model and calculate final metrics
        sim_df = pd.merge_asof(
            sim_df.sort_values('Close'),
            df_model[['Asset_Price', 'Cash_Balan']].rename(columns={'Cash_Balan': 'refer_model', 'Asset_Price': 'Close'}).sort_values('Close'),
            on='Close',
            direction='nearest'
        )
        sim_df.sort_index(inplace=True)
        
        sim_df['pv'] = sim_df['Cash_Balan'] + (sim_df['Amount_Asset'] * sim_df['Close'])
        sim_df['refer_pv'] = sim_df['refer_model'] + config['Fixed_Asset_Value']
        sim_df['net_pv'] = sim_df['pv'] - sim_df['refer_pv']
        
        return sim_df[['net_pv', 'pred', 'realized_gain_loss', 'Cash_Balan', 'Close']]
        
    except Exception as e:
        st.warning(f"Could not process delta6 for {config.get('Ticker', 'N/A')}: {e}")
    return None


def run_portfolio_simulation(active_configs: Dict[str, Dict]) -> pd.DataFrame:
    """Aggregates results from multiple assets specified in active_configs."""
    gains_df = pd.DataFrame()
    net_pv_df = pd.DataFrame()
    
    for ticker, config in active_configs.items():
        sim_result = delta6(config)
        if sim_result is not None and not sim_result.empty:
            gains_df = pd.concat([gains_df, sim_result[['realized_gain_loss']].rename(columns={"realized_gain_loss": f"{ticker}_re"})], axis=1)
            net_pv_df = pd.concat([net_pv_df, sim_result[['net_pv']].rename(columns={"net_pv": f"{ticker}_net_pv"})], axis=1)

    if gains_df.empty:
        return pd.DataFrame()
        
    # Combine results
    gains_df['maxcash_dd'] = gains_df.sum(axis=1, numeric_only=True).cumsum()
    net_pv_df['net_pv_sum'] = net_pv_df.sum(axis=1, numeric_only=True)
    
    return pd.concat([gains_df, net_pv_df], axis=1)


# --- Main Application & UI ---

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="Exist_F(X)", page_icon="☀", layout="wide")
    st.title("Exist F(X) - Portfolio Analysis")

    # 1. Load config and get user selection
    full_config = load_config()
    if not full_config:
        st.stop()

    all_tickers = list(full_config.keys())
    selected_tickers = st.multiselect(
        "Select Tickers to Analyze",
        options=all_tickers,
        default=all_tickers
    )

    active_configs = {ticker: full_config[ticker] for ticker in selected_tickers if ticker in full_config}

    if not active_configs:
        st.warning("Please select at least one ticker to start the analysis.")
        st.stop()

    # 2. Run calculation
    with st.spinner('Calculating portfolio simulation... This may take a moment.'):
        results_df = run_portfolio_simulation(active_configs)
    
    if results_df.empty:
        st.error("Failed to generate data for the selected tickers. Please check the warnings above or your configuration.")
        st.stop()

    # 3. Process final results for plotting
    try:
        for ticker in selected_tickers:
            col_name = f'{ticker}_re'
            if col_name in results_df.columns:
                results_df[col_name] = results_df[col_name].cumsum()

        cumulative_gains = results_df['maxcash_dd']
        running_max = cumulative_gains.cummax()
        roll_over = cumulative_gains - running_max
        
        min_sum_val = roll_over.min()
        min_sum_abs = abs(min_sum_val) if min_sum_val != 0 else 1.0

        df_all = pd.DataFrame({
            'Sum_Delta': results_df['net_pv_sum'],
            'Max_Sum_Buffer': roll_over
        })
        df_all_2 = pd.DataFrame({
            'True_Alpha': (results_df['net_pv_sum'] / min_sum_abs) * 100
        })

        # 4. Display results
        st.write('____')
        
        # Display metrics in columns for better layout
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric(
                label="Final Values (Sum_Delta, Max_Sum_Buffer)",
                value=f"{df_all.Sum_Delta.iloc[-1]:,.2f}",
                delta=f"{df_all.Max_Sum_Buffer.iloc[-1]:,.2f} (Buffer)",
                delta_color="inverse"
            )
        with m_col2:
            st.metric(label="Final True Alpha (%)", value=f"{df_all_2.True_Alpha.iloc[-1]:.2f}%")

        plot_col1, plot_col2 = st.columns(2)
        plot_col1.plotly_chart(px.line(df_all, title="Portfolio: Sum Delta vs Max Sum Buffer"), use_container_width=True)
        plot_col2.plotly_chart(px.line(df_all_2, title="Portfolio: True Alpha (%)"), use_container_width=True)
        
        st.write('____')
        st.plotly_chart(px.line(results_df, title="Detailed Portfolio Simulation"), use_container_width=True)

    except (IndexError, KeyError) as e:
        st.error(f"An error occurred during final processing. The calculated data might be incomplete. Error: {e}")
        st.dataframe(results_df) # Display the raw dataframe for debugging

if __name__ == "__main__":
    main()
