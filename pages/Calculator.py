import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import thingspeak
import json
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# --- 1. CONFIGURATION & DATA STRUCTURES ---

# Define data classes to represent the structure of your JSON config.
# This provides type safety, autocompletion, and clearer code.

@dataclass
class ProductionParams:
    """Parameters for calculating production cost."""
    t0: float
    fix: float

@dataclass
class AssetConfig:
    """Configuration for a single monitored asset."""
    ticker: str
    monitor_field: int
    channel_id: int
    write_api_key: str
    production_params: ProductionParams

    # This special method allows creating an AssetConfig directly from a dictionary
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssetConfig":
        return cls(
            ticker=data['ticker'],
            monitor_field=data['monitor_field'],
            channel_id=data['channel_id'],
            write_api_key=data['write_api_key'],
            production_params=ProductionParams(**data['production_params'])
        )

@dataclass
class AverageCFConfig:
    """Configuration for the average CF calculation."""
    ticker: str
    field: int
    channel_id: int
    write_api_key: str
    offset: int
    filter_date: str

@dataclass
class MonitorConfig:
    """Global configuration for monitoring."""
    filter_date: str

@dataclass
class AppConfig:
    """Top-level container for all application configurations."""
    average_cf_config: AverageCFConfig
    monitor_config: MonitorConfig
    assets: List[AssetConfig]

# Use a constant for the filepath
CONFIG_FILEPATH = "calculator_config.json"

st.set_page_config(page_title="Calculator", page_icon="⌨️")

@st.cache_data(ttl=300) # Cache config for 5 minutes
def load_and_parse_config(filepath: str) -> Optional[AppConfig]:
    """
    Loads and parses the configuration from a JSON file into data class objects.
    """
    config_path = Path(filepath)
    if not config_path.is_file():
        st.error(f"Error: Configuration file not found at '{filepath}'")
        return None
    try:
        with config_path.open('r', encoding='utf-8') as f:
            config_data = json.load(f)

        # Parse the raw dictionary into our structured AppConfig object
        return AppConfig(
            average_cf_config=AverageCFConfig(**config_data['average_cf_config']),
            monitor_config=MonitorConfig(**config_data['monitor_config']),
            assets=[AssetConfig.from_dict(asset_data) for asset_data in config_data['assets']]
        )
    except (json.JSONDecodeError, KeyError) as e:
        st.error(f"Error processing config file '{filepath}'. Check for syntax errors or missing keys: {e}")
        return None

# --- 2. CORE LOGIC FUNCTIONS ---

@st.cache_data(ttl=600) # Cache a Ticker's history for 10 minutes
def get_ticker_history(ticker_symbol: str) -> pd.DataFrame:
    """Fetches and processes historical 'Close' price for a given ticker."""
    ticker = yf.Ticker(ticker_symbol)
    history = ticker.history(period='max')[['Close']]
    history.index = history.index.tz_convert(tz='Asia/Bangkok')
    return round(history, 3)

def calculate_average_cf(cf_config: AverageCFConfig) -> float:
    """Calculates the average CF based on configuration."""
    history = get_ticker_history(cf_config.ticker)
    filtered_data = history[history.index >= cf_config.filter_date]
    count_data = len(filtered_data)

    if count_data == 0:
        return 0.0

    try:
        client = thingspeak.Channel(id=cf_config.channel_id, api_key=cf_config.write_api_key)
        # Safer: Directly convert to int, avoiding eval()
        field_data = client.get_field_last(field=f"{cf_config.field}")
        value = int(json.loads(field_data)[f"field{cf_config.field}"])
        adjusted_value = value - cf_config.offset
        return adjusted_value / count_data
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        st.warning(f"Could not retrieve or parse ThingSpeak data for average_cf: {e}")
        return 0.0

@st.cache_data(ttl=60) # Cache calculation for 1 minute
def calculate_production_cost(ticker: str, params: ProductionParams) -> Optional[float]:
    """
    Calculates Production based on the formula: (fix * -1) * ln(t0 / current_price)
    """
    if params.t0 <= 0 or params.fix == 0:
        return 0.0

    try:
        ticker_info = yf.Ticker(ticker)
        current_price = ticker_info.fast_info.get('lastPrice')

        if current_price is None or current_price <= 0:
            st.warning(f"Invalid current price for {ticker}: {current_price}")
            return None

        return (params.fix * -1) * math.log(params.t0 / current_price)

    except Exception as e:
        st.warning(f"Could not calculate Production for {ticker}: {e}")
        return None

def get_monitor_data(asset: AssetConfig, filter_date: str) -> (pd.DataFrame, int):
    """Retrieves and prepares monitoring data for an asset."""
    history = get_ticker_history(asset.ticker)
    filtered_data = history[history.index >= filter_date].copy()

    fx_js = 0
    try:
        client = thingspeak.Channel(id=asset.channel_id, api_key=asset.write_api_key)
        field_data = client.get_field_last(field=f'{asset.monitor_field}')
        fx_js = int(json.loads(field_data)[f"field{asset.monitor_field}"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        st.info(f"Could not get f(x) for {asset.ticker}, defaulting to 0.")

    # Create the display DataFrame
    rng = np.random.default_rng(fx_js)
    future_index = pd.DataFrame(index=['+0', "+1", "+2", "+3", "+4"])
    combined_df = pd.concat([filtered_data, future_index]).fillna("")
    combined_df['action'] = rng.integers(2, size=len(combined_df))
    combined_df['index'] = ""

    if not filtered_data.empty:
        combined_df.loc[filtered_data.index, 'index'] = range(1, len(filtered_data) + 1)

    return combined_df.tail(7), fx_js

# --- 3. UI DISPLAY FUNCTIONS ---

def display_average_cf_section(cf_config: AverageCFConfig):
    """Displays the calculated average cash flow."""
    st.write('____')
    cf_day = calculate_average_cf(cf_config)
    st.write(f"average_cf_day: {cf_day:.2f} USD  |  average_cf_mo: {cf_day * 30:.2f} USD")
    st.write('____')

def display_asset_section(asset: AssetConfig, monitor_filter_date: str):
    """Displays the complete section for a single asset."""
    df_7, fx_js = get_monitor_data(asset, monitor_filter_date)

    prod_cost = calculate_production_cost(
        ticker=asset.ticker,
        params=asset.production_params
    )
    prod_cost_display = f"{prod_cost:.2f}" if prod_cost is not None else "N/A"

    st.write(f"**{asset.ticker}**")
    st.write(f"f(x): {fx_js} | Production: {prod_cost_display}")
    st.table(df_7)
    st.write("_____")


# --- 4. MAIN APPLICATION EXECUTION ---

def main():
    """Main function to run the Streamlit app."""
    config = load_and_parse_config(CONFIG_FILEPATH)
    if not config:
        st.stop()  # Stop execution if config fails to load

    if st.button("Rerun"):
        st.rerun()

    # Display Average CF section
    display_average_cf_section(config.average_cf_config)

    # Display each asset
    for asset in config.assets:
        display_asset_section(asset, config.monitor_config.filter_date)

    st.write("***ก่อนตลาดเปิดตรวจสอบ TB ล่าสุด > RE เมื่อตลอดเปิด")
    st.write("***RE > 60 USD")

if __name__ == "__main__":
    main()
