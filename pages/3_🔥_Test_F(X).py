import streamlit as st
import json
import pandas as pd

# --- Configuration Data ---
# Storing the provided JSON in a multiline string
config_json = """
{
  "average_cf_config": {
    "ticker": "FFWM",
    "field": 1,
    "channel_id": 2394198,
    "write_api_key": "OVZNYQBL57GJW5JF",
    "offset": 1700.00 ,
    "max_roll_over": 1169.00  , 
    "filter_date_cf": "2025-06-16 12:00:00+07:00"
  },
  "monitor_config": {
    "filter_date": "2025-06-09 12:00:00+07:00"
  }, 
  "assets": [
    { "ticker": "FFWM", "monitor_field": 2, "channel_id": 2385118, "write_api_key": "IPSG3MMMBJEB9DY8", "production_params": { "t0": 6.88, "fix": 1500.0 }},
    { "ticker": "NEGG", "monitor_field": 3, "channel_id": 2385118, "write_api_key": "IPSG3MMMBJEB9DY8", "production_params": { "t0": 25.2, "fix": 1500.0 }},
    { "ticker": "RIVN", "monitor_field": 4, "channel_id": 2385118, "write_api_key": "IPSG3MMMBJEB9DY8", "production_params": { "t0": 10.07, "fix": 1500.0 }},
    { "ticker": "APLS", "monitor_field": 5, "channel_id": 2385118, "write_api_key": "IPSG3MMMBJEB9DY8", "production_params": { "t0": 39.61, "fix": 1500.0 }},
    { "ticker": "NVTS", "monitor_field": 6, "channel_id": 2385118, "write_api_key": "IPSG3MMMBJEB9DY8", "production_params": { "t0": 3.05, "fix": 1500.0 }},
    { "ticker": "QXO", "monitor_field": 7, "channel_id": 2385118, "write_api_key": "IPSG3MMMBJEB9DY8", "production_params": { "t0": 19.0, "fix": 1500.0 }},
    { "ticker": "RXRX", "monitor_field": 8, "channel_id": 2385118, "write_api_key": "IPSG3MMMBJEB9DY8", "production_params": { "t0": 5.4, "fix": 1500.0 }},
    { "ticker": "AGL", "monitor_field": 1, "channel_id": 2385118, "write_api_key": "IPSG3MMMBJEB9DY8", "production_params": { "t0": 3.0, "fix": 1500.0 }},
    { "ticker": "FLNC", "monitor_field": 1, "channel_id": 2988808, "write_api_key": "QBIPN1AIZ88QQW5V", "production_params": { "t0": 7.0, "fix": 1500.0 }},
    { "ticker": "GERN", "monitor_field": 2, "channel_id": 2988808, "write_api_key": "QBIPN1AIZ88QQW5V", "production_params": { "t0": 1.85, "fix": 1500.0 }}
  ]
}
"""

def calculate_dashboard_metrics(config):
    """
    Calculates all necessary metrics based on the config and simulated data.
    Returns a dictionary of metrics and a DataFrame of asset details.
    """
    # --- Data Simulation ---
    # In a real application, you would fetch this from an API (e.g., yfinance).
    # For this example, we'll use a dictionary of mock "current prices".
    mock_current_prices = {
        "FFWM": 7.50, "NEGG": 23.10, "RIVN": 11.25, "APLS": 41.50,
        "NVTS": 3.00, "QXO": 20.15, "RXRX": 6.80, "AGL": 2.75,
        "FLNC": 8.10, "GERN": 2.05
    }

    asset_details = []
    total_net_usd = 0.0
    total_investment = 0.0

    for asset in config['assets']:
        ticker = asset['ticker']
        t0 = asset['production_params']['t0']
        fix_investment = asset['production_params']['fix']
        current_price = mock_current_prices.get(ticker, t0) # Default to t0 if not in mock data

        # A common P&L calculation: (Current Price - Initial Price) * Number of Shares
        # Number of Shares is derived from (Total Investment / Initial Price)
        # Simplified formula: P&L = (current_price - t0) * (fix_investment / t0)
        pnl = (current_price - t0) * (fix_investment / t0)
        
        total_net_usd += pnl
        total_investment += fix_investment

        asset_details.append({
            "Ticker": ticker,
            "Initial Price (t0)": t0,
            "Current Price": current_price,
            "Investment (fix)": fix_investment,
            "P&L (USD)": pnl
        })

    # Create a pandas DataFrame for detailed view
    df_assets = pd.DataFrame(asset_details)

    # Extract other key metrics from the config
    max_roll = config['average_cf_config']['max_roll_over']
    offset = config['average_cf_config']['offset']
    
    # Calculate the new metric as requested
    net_after_roll = total_net_usd - max_roll

    # Package all results neatly
    metrics = {
        "total_net_usd": total_net_usd,
        "total_investment": total_investment,
        "offset": offset,
        "max_roll": max_roll,
        "net_after_roll": net_after_roll
    }

    return metrics, df_assets


def main():
    """
    The main function to run the Streamlit app.
    """
    st.set_page_config(page_title="Financial Dashboard", layout="wide")
    st.title("📈 Financial Performance Dashboard")

    # Load configuration from the JSON string
    config = json.loads(config_json)

    # Calculate all metrics
    metrics, df_assets = calculate_dashboard_metrics(config)

    # --- Main Display Logic ---
    st.header("Overall Summary")
    
    # Create 5 columns for the main metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        label="Net P&L (USD)",
        value=f"${metrics['total_net_usd']:,.2f}",
        delta=f"{metrics['total_net_usd'] / metrics['total_investment'] * 100:.2f}%"
    )
    
    col2.metric(
        label="Total Investment (USD)",
        value=f"${metrics['total_investment']:,.2f}"
    )

    col3.metric(
        label="Config Offset (USD)",
        value=f"${metrics['offset']:,.2f}"
    )

    col4.metric(
        label="Max Roll Over (USD)",
        value=f"${metrics['max_roll']:,.2f}"
    )

    # --- THIS IS THE NEW METRIC YOU REQUESTED ---
    col5.metric(
        label="Net - Max Roll (USD)",
        value=f"${metrics['net_after_roll']:,.2f}",
        help="This is the total Net P&L after subtracting the Max Roll Over value."
    )

    st.divider()

    # --- Detailed Asset Breakdown ---
    st.header("Asset Breakdown")
    st.dataframe(df_assets.style.format({
        'Initial Price (t0)': '${:,.2f}',
        'Current Price': '${:,.2f}',
        'Investment (fix)': '${:,.2f}',
        'P&L (USD)': '${:,.2f}'
    }).apply(
        lambda x: ['background-color: #225522' if x['P&L (USD)'] > 0 else 'background-color: #552222' for i in x],
        axis=1
    ), use_container_width=True)


if __name__ == "__main__":
    main()
