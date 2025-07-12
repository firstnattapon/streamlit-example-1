# import numpy as np
# import pandas as pd
# import yfinance as yf
# import streamlit as st
# import json
# import plotly.express as px

# st.set_page_config(layout="wide", page_title="Portfolio Backtesting Engine", page_icon="📈")

# # ------------------- UTILITY FUNCTIONS -------------------

# def load_config(filename="un15_fx_config.json"):
#     """
#     Loads asset configurations from a JSON file.
#     Includes error handling for file not found or malformed JSON.
#     """
#     try:
#         with open(filename, 'r') as f:
#             return json.load(f)
#     except FileNotFoundError:
#         st.error(f"Error: Configuration file '{filename}' not found. Please create it in the same directory.")
#         return {}
#     except json.JSONDecodeError:
#         st.error(f"Error: Could not decode JSON from '{filename}'. Please check its format.")
#         return {}

# # ------------------- CORE CALCULATION ENGINE -------------------

# def calculate_cash_balance_model(entry, step, Fixed_Asset_Value, Cash_Balan):
#     """
#     Creates a theoretical reference model for a rebalancing strategy.
#     It calculates the ideal cash balance at various price points if the asset's
#     value were always kept constant at Fixed_Asset_Value.
#     """
#     if entry <= 0 or step <= 0:
#         return pd.DataFrame()

#     # Create a grid of prices around the entry price
#     samples = np.arange(step, np.around(entry, 2) * 3 + step, step)
    
#     df = pd.DataFrame()
#     df['Asset_Price'] = np.around(samples, 2)
#     # Filter out zero or negative prices which are invalid
#     df = df[df['Asset_Price'] > 0]
#     if df.empty:
#         return pd.DataFrame()

#     df['Fixed_Asset_Value'] = Fixed_Asset_Value
#     df['Amount_Asset'] = df['Fixed_Asset_Value'] / df['Asset_Price']

#     # --- Top part (price goes up, we sell asset to generate cash) ---
#     df_top = df[df.Asset_Price >= np.around(entry, 2)].copy()
#     if not df_top.empty:
#         df_top['Cash_Balan_top'] = (df_top['Amount_Asset'].shift(1) - df_top['Amount_Asset']) * df_top['Asset_Price']
#         df_top.fillna(0, inplace=True)
        
#         # Calculate cumulative cash balance
#         cumulative_cash = np.cumsum(df_top['Cash_Balan_top'].values) + Cash_Balan
#         df_top['Cash_Balan'] = cumulative_cash
#         df_top = df_top.drop(columns=['Cash_Balan_top'])
#         df_top = df_top.sort_values(by='Amount_Asset')[:-1]
    
#     # --- Down part (price goes down, we spend cash to buy asset) ---
#     df_down = df[df.Asset_Price <= np.around(entry, 2)].copy()
#     if not df_down.empty:
#         df_down = df_down.sort_values(by='Asset_Price', ascending=False)
#         df_down['Cash_Balan_down'] = (df_down['Amount_Asset'].shift(-1) - df_down['Amount_Asset']) * df_down['Asset_Price']
#         df_down.fillna(0, inplace=True)
        
#         # Calculate cumulative cash balance
#         cumulative_cash = np.cumsum(df_down['Cash_Balan_down'].values) + Cash_Balan
#         df_down['Cash_Balan'] = cumulative_cash
#         df_down = df_down.drop(columns=['Cash_Balan_down'])

#     # --- Combine and return the complete reference model ---
#     combined_df = pd.concat([df_top, df_down], axis=0)
#     return combined_df

# # ------------------- SINGLE ASSET ANALYSIS FUNCTIONS -------------------

# def delta_1(asset_config):
#     """
#     Calculates the 'Production Cost' for a single asset.
#     This represents the maximum cash required to follow the strategy down to its lowest price.
#     """
#     try:
#         tickerData = yf.Ticker(asset_config['Ticker'])
#         entry = tickerData.fast_info.get('lastPrice')
#         if not entry:
#             st.warning(f"Could not get last price for {asset_config['Ticker']}. Skipping delta_1.")
#             return None

#         # Call the core model function
#         df_model = calculate_cash_balance_model(
#             entry, 
#             asset_config['step'], 
#             asset_config['Fixed_Asset_Value'], 
#             asset_config['Cash_Balan']
#         )

#         if not df_model.empty:
#             # Production cost is the difference between initial and minimum cash balance
#             production_costs = df_model['Cash_Balan'].min() - asset_config['Cash_Balan']
#             return abs(production_costs)
#         return None
#     except Exception as e:
#         # st.warning(f"Could not process delta_1 for {asset_config.get('Ticker', 'N/A')}: {e}")
#         return None

# def delta6(asset_config):
#     """
#     Performs a full historical backtest simulation for a single asset based on its configuration.
#     It compares the performance of a selective rebalancing strategy (based on 'pred')
#     against the ideal theoretical model.
#     """
#     try:
#         # 1. Load historical data
#         ticker_hist = yf.Ticker(asset_config['Ticker']).history(period='max')
#         if ticker_hist.empty:
#             st.warning(f"No historical data found for {asset_config['Ticker']}.")
#             return None
#         ticker_hist.index = ticker_hist.index.tz_convert(tz='Asia/bangkok')
#         ticker_hist = ticker_hist[ticker_hist.index >= asset_config['filter_date']][['Close']]
#         if ticker_hist.empty:
#             st.warning(f"No data for {asset_config['Ticker']} after filter date {asset_config['filter_date']}.")
#             return None

#         entry = ticker_hist.Close[0]

#         # 2. Create the theoretical reference model
#         df_model = calculate_cash_balance_model(
#             entry, 
#             asset_config['step'], 
#             asset_config['Fixed_Asset_Value'], 
#             asset_config['Cash_Balan']
#         )
#         if df_model.empty:
#             return None

#         # 3. Set up the simulation DataFrame
#         sim_df = ticker_hist.copy()
#         sim_df['Close'] = np.around(sim_df['Close'].values, 2)
#         sim_df['pred'] = asset_config['pred'] # This is the rebalancing signal (1=rebalance, 0=hold)
#         sim_df['Fixed_Asset_Value'] = asset_config['Fixed_Asset_Value']
#         sim_df['Amount_Asset'] = 0.0
#         sim_df['re'] = 0.0 # 're' stands for rebalancing cash flow
#         sim_df['Cash_Balan'] = float(asset_config['Cash_Balan'])
        
#         # Initialize first day
#         sim_df.iloc[0, sim_df.columns.get_loc('Amount_Asset')] = sim_df.iloc[0]['Fixed_Asset_Value'] / sim_df.iloc[0]['Close']
        
#         # 4. Run the day-by-day simulation loop
#         for i in range(1, len(sim_df)):
#             if sim_df.iloc[i]['pred'] == 1:  # Rebalance only if pred signal is 1
#                 current_amount = asset_config['Fixed_Asset_Value'] / sim_df.iloc[i]['Close']
#                 rebalance_cashflow = (sim_df.iloc[i-1]['Amount_Asset'] - current_amount) * sim_df.iloc[i]['Close']
#                 sim_df.iloc[i, sim_df.columns.get_loc('Amount_Asset')] = current_amount
#                 sim_df.iloc[i, sim_df.columns.get_loc('re')] = rebalance_cashflow
#             else:  # Hold if pred signal is 0
#                 sim_df.iloc[i, sim_df.columns.get_loc('Amount_Asset')] = sim_df.iloc[i-1]['Amount_Asset']
#                 sim_df.iloc[i, sim_df.columns.get_loc('re')] = 0.0
            
#             # Update cash balance
#             sim_df.iloc[i, sim_df.columns.get_loc('Cash_Balan')] = sim_df.iloc[i-1]['Cash_Balan'] + sim_df.iloc[i]['re']
        
#         # 5. Compare simulation to the theoretical model
#         sim_df['refer_model'] = sim_df['Close'].apply(lambda x: df_model.iloc[(df_model['Asset_Price']-x).abs().argsort()[:1]]['Cash_Balan'].values[0] if not df_model.empty else np.nan)
#         sim_df['refer_model'].interpolate(method='linear', inplace=True)
#         sim_df.fillna(method='bfill', inplace=True)
#         sim_df.fillna(method='ffill', inplace=True)

#         # 6. Calculate final performance metrics
#         sim_df['pv'] = sim_df['Cash_Balan'] + (sim_df['Amount_Asset'] * sim_df['Close']) # Simulated Portfolio Value
#         sim_df['refer_pv'] = sim_df['refer_model'] + asset_config['Fixed_Asset_Value'] # Reference Portfolio Value
#         sim_df['net_pv'] = sim_df['pv'] - sim_df['refer_pv'] # Net performance (Alpha)
        
#         return sim_df[['net_pv', 're']]
        
#     except Exception as e:
#         st.warning(f"Could not process simulation for {asset_config.get('Ticker', 'N/A')}: {e}")
#         return None

# # ------------------- PORTFOLIO AGGREGATION FUNCTION -------------------

# def un_16(active_configs):
#     """
#     Aggregates simulation results from multiple assets into a single portfolio view.
#     """
#     re_dfs = []
#     net_pv_dfs = []
    
#     # Run simulation for each selected ticker
#     for ticker_name, config in active_configs.items():
#         result_df = delta6(config)
#         if result_df is not None:
#             re_dfs.append(result_df[['re']].rename(columns={"re": f"{ticker_name}_re"}))
#             net_pv_dfs.append(result_df[['net_pv']].rename(columns={"net_pv": f"{ticker_name}_net_pv"}))
    
#     if not re_dfs:
#         return pd.DataFrame() # Return empty if no simulations were successful
        
#     # Combine results into a single DataFrame
#     portfolio_df = pd.concat(re_dfs + net_pv_dfs, axis=1)
    
#     # Calculate portfolio-level metrics
#     portfolio_df['portfolio_cash_flow'] = portfolio_df[[col for col in portfolio_df.columns if '_re' in col]].sum(axis=1)
#     portfolio_df['cash_drawdown'] = portfolio_df['portfolio_cash_flow'].cumsum()
#     portfolio_df['portfolio_net_pv'] = portfolio_df[[col for col in portfolio_df.columns if '_net_pv' in col]].sum(axis=1)

#     return portfolio_df

# # ------------------- STREAMLIT UI AND DISPLAY -------------------

# st.title("📈 Portfolio Rebalancing Strategy Backtester")

# # 1. Load all configurations from the JSON file
# full_config = load_config()

# if full_config:
#     # 2. Create a user-friendly multi-select widget for tickers
#     all_tickers = list(full_config.keys())
#     selected_tickers = st.multiselect(
#         "Select Tickers to Analyze (from un15_fx_config.json)",
#         options=all_tickers,
#         default=all_tickers  # Select all by default
#     )

#     # 3. Create a dictionary of only the configs for the selected tickers
#     active_configs = {ticker: full_config[ticker] for ticker in selected_tickers if ticker in full_config}

#     # 4. Run analysis only if tickers are selected
#     if not active_configs:
#         st.warning("Please select at least one ticker to start the analysis.")
#     else:
#         # 5. Run the main aggregation function with a loading spinner
#         with st.spinner('Running historical simulations... This may take a moment.'):
#             portfolio_data = un_16(active_configs)
        
#         if portfolio_data.empty:
#             st.error("Failed to generate data for the selected tickers. Check console warnings for details.")
#         else:
#             # 6. Calculate final metrics for charting
#             final_metrics = pd.DataFrame()
#             final_metrics['Sum_Alpha'] = portfolio_data['portfolio_net_pv']
            
#             # Calculate Maximum Drawdown (the lowest point of the cash balance)
#             cumulative_cash = portfolio_data['cash_drawdown']
#             running_max = cumulative_cash.cummax() # This is not needed for drawdown calc, but useful for other metrics
#             final_metrics['Max_Cash_Drawdown'] = cumulative_cash.cummin()

#             # Calculate Risk-Adjusted Return (Alpha per unit of max drawdown)
#             # Avoid division by zero if there's no drawdown
#             min_drawdown = abs(final_metrics['Max_Cash_Drawdown'].min())
#             if min_drawdown == 0:
#                 final_metrics['True_Alpha'] = 0
#             else:
#                 final_metrics['True_Alpha'] = (final_metrics['Sum_Alpha'] / min_drawdown) * 100
                
#             st.header("Portfolio Performance Summary")

#             # Display final KPI values
#             latest_alpha = final_metrics['Sum_Alpha'].iloc[-1]
#             max_drawdown = final_metrics['Max_Cash_Drawdown'].iloc[-1]
#             latest_true_alpha = final_metrics['True_Alpha'].iloc[-1]
            
#             kpi1, kpi2, kpi3 = st.columns(3)
#             kpi1.metric(label="Total Alpha (Outperformance)", value=f"${latest_alpha:,.2f}")
#             kpi2.metric(label="Maximum Cash Drawdown", value=f"${max_drawdown:,.2f}")
#             kpi3.metric(label="True Alpha (Risk-Adjusted Return)", value=f"{latest_true_alpha:.2f} %")
            
#             st.write("---")

#             # Display charts
#             col1, col2 = st.columns(2)
            
#             fig1 = px.line(final_metrics[['Sum_Alpha', 'Max_Cash_Drawdown']], title="<b>Portfolio Alpha vs. Max Cash Drawdown</b>")
#             fig1.update_layout(yaxis_title="Value ($)", legend_title="Metric")
#             col1.plotly_chart(fig1, use_container_width=True)

#             fig2 = px.line(final_metrics[['True_Alpha']], title="<b>Portfolio True Alpha (Risk-Adjusted Performance)</b>")
#             fig2.update_layout(yaxis_title="Percentage (%)", showlegend=False)
#             col2.plotly_chart(fig2, use_container_width=True)
            
#             st.write("---")
#             st.header("Detailed Simulation Data")
#             st.dataframe(portfolio_data)
