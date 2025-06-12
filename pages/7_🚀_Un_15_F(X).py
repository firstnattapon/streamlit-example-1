import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import json
import plotly.express as px
from typing import Dict, Any, Optional

# ------------------- CONFIG LOADER ------------------- #
def load_config(filename: str = "un15_fx_config.json") -> Dict[str, Any]:
    """Load asset configurations from a JSON file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error: Configuration file '{filename}' not found.")
        return {}
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from '{filename}'. Please check its format.")
        return {}

# ------------------- CORE MODEL ------------------- #
def calculate_cash_balance_model(
    entry: float, step: float, Fixed_Asset_Value: float, Cash_Balan: float
) -> pd.DataFrame:
    """Calculate the cash balance model DataFrame."""
    if entry >= 10000:
        return pd.DataFrame()
    samples = np.arange(0, np.around(entry, 2) * 3 + step, step)
    df = pd.DataFrame({'Asset_Price': np.around(samples, 2)})
    df['Fixed_Asset_Value'] = Fixed_Asset_Value
    df['Amount_Asset'] = Fixed_Asset_Value / df['Asset_Price']

    # Top
    df_top = df[df.Asset_Price >= np.around(entry, 2)].copy()
    df_top['Cash_Balan'] = (df_top['Amount_Asset'].shift(1) - df_top['Amount_Asset']) * df_top['Asset_Price']
    df_top['Cash_Balan'] = df_top['Cash_Balan'].fillna(0)
    df_top['Cash_Balan'] = df_top['Cash_Balan'].cumsum() + Cash_Balan
    df_top = df_top.sort_values(by='Amount_Asset')[:-1]

    # Down
    df_down = df[df.Asset_Price <= np.around(entry, 2)].copy()
    df_down['Cash_Balan'] = (df_down['Amount_Asset'].shift(-1) - df_down['Amount_Asset']) * df_down['Asset_Price']
    df_down['Cash_Balan'] = df_down['Cash_Balan'].fillna(0)
    df_down = df_down.sort_values(by='Asset_Price', ascending=False)
    df_down['Cash_Balan'] = df_down['Cash_Balan'].cumsum() + Cash_Balan

    # Combine
    combined_df = pd.concat([df_top, df_down], axis=0)
    return combined_df

# ------------------- DELTA FUNCTIONS ------------------- #
def delta_1(asset_config: Dict[str, Any]) -> Optional[float]:
    """Calculate Production_Costs based on asset configuration."""
    try:
        Ticker = asset_config['Ticker']
        Fixed_Asset_Value = asset_config['Fixed_Asset_Value']
        Cash_Balan = asset_config['Cash_Balan']
        step = asset_config['step']
        tickerData = yf.Ticker(Ticker)
        entry = tickerData.fast_info['lastPrice']
        df_model = calculate_cash_balance_model(entry, step, Fixed_Asset_Value, Cash_Balan)
        if not df_model.empty:
            Production_Costs = (df_model['Cash_Balan'].values[-1]) - Cash_Balan
            return abs(Production_Costs)
    except Exception:
        return None

def delta6(asset_config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Perform historical simulation based on asset configuration."""
    try:
        Ticker = asset_config['Ticker']
        pred = asset_config['pred']
        filter_date = asset_config['filter_date']
        Fixed_Asset_Value = asset_config['Fixed_Asset_Value']
        Cash_Balan = asset_config['Cash_Balan']
        step = asset_config['step']

        ticker_hist = yf.Ticker(Ticker).history(period='max')
        ticker_hist.index = ticker_hist.index.tz_convert('Asia/Bangkok')
        ticker_hist = ticker_hist[ticker_hist.index >= filter_date][['Close']]
        entry = ticker_hist.Close.iloc[0]

        df_model = calculate_cash_balance_model(entry, step, Fixed_Asset_Value, Cash_Balan)
        if df_model.empty:
            return None

        tickerData = ticker_hist.copy()
        tickerData['Close'] = np.around(tickerData['Close'].values, 2)
        tickerData['pred'] = pred
        tickerData['Fixed_Asset_Value'] = Fixed_Asset_Value
        tickerData['Amount_Asset'] = 0.0
        tickerData.iloc[0, tickerData.columns.get_loc('Amount_Asset')] = Fixed_Asset_Value / tickerData['Close'].iloc[0]
        tickerData['re'] = 0.0
        tickerData['Cash_Balan'] = Cash_Balan

        for idx in range(1, len(tickerData)):
            if tickerData['pred'].iloc[idx] == 1:
                tickerData.iloc[idx, tickerData.columns.get_loc('Amount_Asset')] = Fixed_Asset_Value / tickerData['Close'].iloc[idx]
                tickerData.iloc[idx, tickerData.columns.get_loc('re')] = (
                    tickerData['Amount_Asset'].iloc[idx-1] * tickerData['Close'].iloc[idx] - Fixed_Asset_Value
                )
            else:
                tickerData.iloc[idx, tickerData.columns.get_loc('Amount_Asset')] = tickerData['Amount_Asset'].iloc[idx-1]
                tickerData.iloc[idx, tickerData.columns.get_loc('re')] = 0
            tickerData.iloc[idx, tickerData.columns.get_loc('Cash_Balan')] = (
                tickerData['Cash_Balan'].iloc[idx-1] + tickerData['re'].iloc[idx]
            )

        price = tickerData['Close'].values
        refer_model = np.full_like(price, 0.0)
        for idx, x_3 in enumerate(price):
            try:
                refer_model[idx] = df_model[df_model['Asset_Price'] == x_3]['Cash_Balan'].values[0]
            except IndexError:
                refer_model[idx] = np.nan
        tickerData['refer_model'] = pd.Series(refer_model, index=tickerData.index)
        tickerData['refer_model'] = tickerData['refer_model'].interpolate(method='linear').bfill().ffill()

        tickerData['pv'] = tickerData['Cash_Balan'] + (tickerData['Amount_Asset'] * tickerData['Close'])
        tickerData['refer_pv'] = tickerData['refer_model'] + Fixed_Asset_Value
        tickerData['net_pv'] = tickerData['pv'] - tickerData['refer_pv']

        return tickerData[['net_pv', 'pred', 're', 'Cash_Balan', 'Close']]
    except Exception:
        return None

def un_16(active_configs: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Aggregate results from multiple assets specified in active_configs."""
    a_0 = pd.DataFrame()
    a_1 = pd.DataFrame()
    Max_Production = 0.0

    for ticker_name, config in active_configs.items():
        a_2 = delta6(config)
        if a_2 is not None:
            a_0 = pd.concat([a_0, a_2[['re']].rename(columns={"re": f"{ticker_name}_re"})], axis=1)
            a_1 = pd.concat([a_1, a_2[['net_pv']].rename(columns={"net_pv": f"{ticker_name}_net_pv"})], axis=1)
        prod_cost = delta_1(config)
        if prod_cost is not None:
            Max_Production += prod_cost

    if a_0.empty:
        return pd.DataFrame()
    net_dd = np.cumsum(a_0.sum(axis=1, numeric_only=True).values)
    a_0['maxcash_dd'] = net_dd
    a_1['cf'] = a_1.sum(axis=1, numeric_only=True)
    a_x = pd.concat([a_0, a_1], axis=1)
    return a_x

# ------------------- STREAMLIT UI ------------------- #
st.set_page_config(page_title="Exist_F(X)", page_icon="☀")

full_config = load_config()

if full_config:
    all_tickers = list(full_config.keys())
    selected_tickers = st.multiselect(
        "Select Tickers to Analyze",
        options=all_tickers,
        default=all_tickers
    )
    active_configs = {ticker: full_config[ticker] for ticker in selected_tickers if ticker in full_config}
    if not active_configs:
        st.warning("Please select at least one ticker to start the analysis.")
    else:
        with st.spinner('Calculating... Please wait.'):
            data = un_16(active_configs)
        if data.empty:
            st.error("Failed to generate data for the selected tickers. Please check logs or try again.")
        else:
            for i in selected_tickers:
                col_name = f'{i}_re'
                if col_name in data.columns:
                    data[col_name] = np.cumsum(data[col_name].values)
            df_new = data
            max_dd = df_new.maxcash_dd.values
            roll_over = np.array([np.min(max_dd[:i]) if i > 0 else 0 for i in range(len(max_dd))])
            min_sum = abs(np.min(roll_over)) if np.min(roll_over) != 0 else 1
            sum_val = (df_new.cf.values / min_sum) * 100
            cf = df_new.cf.values
            df_all = pd.DataFrame(list(zip(cf, roll_over)), columns=['Sum_Delta', 'Max_Sum_Buffer'])
            df_all_2 = pd.DataFrame(sum_val, columns=['True_Alpha'])

            st.write('____')
            st.write(f"({df_all.Sum_Delta.values[-1]:.2f}, {df_all.Max_Sum_Buffer.values[-1]:.2f}) , {df_all_2.True_Alpha.values[-1]:.2f}")

            col1, col2 = st.columns(2)
            col1.plotly_chart(px.line(df_all, title="Sum Delta vs Max Sum Buffer"))
            col2.plotly_chart(px.line(df_all_2, title="True Alpha (%)"))
            st.write('____')
            st.plotly_chart(px.line(df_new, title="Detailed Portfolio Simulation"))
