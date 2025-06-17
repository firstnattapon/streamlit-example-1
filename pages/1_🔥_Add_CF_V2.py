import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import thingspeak
import json
import streamlit.components.v1 as components
from typing import Dict, Any, Tuple, List

# --- Page Configuration ---
st.set_page_config(page_title="Add_CF_Calculator", page_icon="🚀")

# --- 1. CONFIGURATION & INITIALIZATION FUNCTIONS ---

@st.cache_data
def load_config(filename: str = "add_cf_config.json") -> Dict[str, Any]:
    """Loads the entire configuration from a JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.error(f"Error loading or parsing {filename}: {e}")
        st.stop()

@st.cache_resource
def initialize_thingspeak_clients(config: Dict[str, Any], stock_assets: List[Dict[str, Any]]) -> Tuple[thingspeak.Channel, Dict[str, thingspeak.Channel]]:
    """Initializes and returns the main and asset-specific ThingSpeak clients."""
    main_channel_config = config.get('thingspeak_channels', {}).get('main_output', {})
    
    try:
        client_main = thingspeak.Channel(
            main_channel_config['channel_id'],
            main_channel_config['write_api_key']
        )
        asset_clients = {}
        for asset in stock_assets: # Only initialize for stocks
            ticker = asset['ticker']
            channel_info = asset.get('holding_channel', {})
            if channel_info.get('channel_id') and channel_info.get('write_api_key'):
                asset_clients[ticker] = thingspeak.Channel(
                    channel_info['channel_id'],
                    channel_info['write_api_key']
                )
        st.success(f"Initialized main client and {len(asset_clients)} stock clients.")
        return client_main, asset_clients
    except (KeyError, Exception) as e:
        st.error(f"Failed to initialize ThingSpeak clients. Error: {e}")
        st.stop()

def fetch_initial_data(stock_assets: List[Dict[str, Any]], option_assets: List[Dict[str, Any]], asset_clients: Dict[str, thingspeak.Channel]) -> Dict[str, Dict[str, float]]:
    """Fetches last holding and last price for each STOCK and OPTION UNDERLYING asset."""
    initial_data = {}
    tickers_to_fetch = {asset['ticker'] for asset in stock_assets}
    
    # Also need prices for underlying stocks of options
    option_underlyings = {item.get('underlying_ticker') for item in option_assets if item.get('underlying_ticker')}
    tickers_to_fetch.update(option_underlyings)

    for ticker in tickers_to_fetch:
        initial_data[ticker] = {}
        # Fetch last price from yfinance
        try:
            last_price = yf.Ticker(ticker).fast_info['lastPrice']
            initial_data[ticker]['last_price'] = last_price
        except Exception:
            # Try to find a reference price if yfinance fails
            ref_price = 0.0
            for asset in stock_assets:
                if asset['ticker'] == ticker:
                    ref_price = asset.get('reference_price', 0.0)
                    break
            initial_data[ticker]['last_price'] = ref_price
            st.warning(f"Could not fetch price for {ticker}. Defaulting to {ref_price}.")
    
    # Fetch holdings only for stocks
    for asset in stock_assets:
        ticker = asset["ticker"]
        last_holding = 0.0
        if ticker in asset_clients:
            try:
                client = asset_clients[ticker]
                field = asset['holding_channel']['field']
                last_asset_json_string = client.get_field_last(field=field)
                if last_asset_json_string:
                    data_dict = json.loads(last_asset_json_string)
                    last_holding = float(data_dict[field])
            except Exception as e:
                 st.warning(f"Could not fetch last holding for {ticker}. Defaulting to 0. Error: {e}")
        initial_data[ticker]['last_holding'] = last_holding
            
    return initial_data

# --- 2. UI & DISPLAY FUNCTIONS ---

def render_ui_and_get_inputs(stock_assets: List[Dict[str, Any]], option_assets: List[Dict[str, Any]], initial_data: Dict[str, Dict[str, float]], product_cost_default: float) -> Dict[str, Any]:
    """Renders all Streamlit input widgets for STOCKS and returns their current values."""
    user_inputs = {}
    st.header("📈 Core Portfolio (Stocks)")
    
    # --- Price Inputs (for all relevant tickers) ---
    st.write("📊 Current Asset Prices")
    current_prices = {}
    
    all_tickers = {asset['ticker'] for asset in stock_assets}
    all_tickers.update({opt['underlying_ticker'] for opt in option_assets})

    for ticker in sorted(list(all_tickers)):
        price_value = initial_data.get(ticker, {}).get('last_price', 0.0)
        current_prices[ticker] = st.number_input(f"ราคา_{ticker}", value=price_value, key=f"price_{ticker}", format="%.2f")
    user_inputs['current_prices'] = current_prices

    st.divider()

    # --- Holding Inputs (for stocks only) ---
    st.write("📦 Stock Holdings")
    current_holdings = {}
    total_asset_value = 0.0
    for asset in stock_assets:
        ticker = asset["ticker"]
        holding_value = initial_data.get(ticker, {}).get('last_holding', 0.0)
        asset_holding = st.number_input(f"{ticker}_asset", value=holding_value, key=f"holding_{ticker}", format="%.2f")
        current_holdings[ticker] = asset_holding
        individual_asset_value = asset_holding * current_prices[ticker]
        st.write(f"มูลค่า {ticker}: **{individual_asset_value:,.2f}**")
        total_asset_value += individual_asset_value
    user_inputs['current_holdings'] = current_holdings
    user_inputs['total_stock_value'] = total_asset_value

    st.divider()
    
    # --- Final Calculation Inputs ---
    st.write("⚙️ Calculation Parameters")
    user_inputs['product_cost'] = st.number_input('Product_cost', value=product_cost_default, format="%.2f")
    user_inputs['portfolio_cash'] = st.number_input('Portfolio_cash', value=0.00, format="%.2f")
    
    return user_inputs

def display_results(stock_metrics: Dict[str, float], options_pl: float, config: Dict[str, Any]):
    """Displays all the calculated metrics in separate sections."""
    
    # --- Stock Portfolio Results ---
    st.divider()
    with st.expander("📈 Core Portfolio Results", expanded=True):
        st.metric('Current Stock Portfolio Value (Assets + Cash):', f"{stock_metrics['now_pv']:,.2f}")
        col1, col2 = st.columns(2)
        col1.metric('t_0 (Product of Reference Prices)', f"{stock_metrics['t_0']:,.2f}")
        col2.metric('t_n (Product of Live Prices)', f"{stock_metrics['t_n']:,.2f}")
        st.metric('Log PV (Calculated Stock Cost)', f"{stock_metrics['log_pv']:,.2f}")
        
        st.metric(label="💰 Net Cashflow (From Stocks Only)", value=f"{stock_metrics['net_cf']:,.2f}", 
                  help="This value is stable and only reflects the performance of the core stock portfolio and cash.")
        st.metric(label=f"💰 Baseline @ {config.get('cashflow_offset_comment', '')}", value=f"{stock_metrics['net_cf'] - config.get('cashflow_offset', 0.0):,.2f}")

    # --- Options Portfolio Results ---
    with st.expander("⌥ Options Portfolio Results", expanded=True):
        st.metric(label="💰 Options Unrealized P/L", value=f"{options_pl:,.2f}",
                  help="This is the (volatile) Profit/Loss from your options positions, based on intrinsic value.")

    # --- Grand Total ---
    st.divider()
    grand_total_value = stock_metrics['now_pv'] + options_pl
    st.metric("👑 Grand Total (Core PV + Options P/L)", f"{grand_total_value:,.2f}",
              help="This represents the total value of your core portfolio plus the current unrealized profit or loss from your options.")


def render_charts(config: Dict[str, Any]):
    """Renders all ThingSpeak charts using iframes."""
    st.write("📊 ThingSpeak Charts")
    main_channel_config = config.get('thingspeak_channels', {}).get('main_output', {})
    main_channel_id = main_channel_config.get('channel_id')
    main_fields_map = main_channel_config.get('fields', {})

    def create_chart_iframe(channel_id, field_name, chart_title):
        if channel_id and field_name:
            chart_number = field_name.replace('field', '')
            url = f'https://thingspeak.com/channels/{channel_id}/charts/{chart_number}?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&type=line&update=15'
            st.write(f"**{chart_title}**")
            components.iframe(url, width=800, height=200)
            st.divider()
        else:
            st.warning(f"Chart for '{chart_title}' cannot be displayed. Missing config for field '{field_name}'.")

    create_chart_iframe(main_channel_id, main_fields_map.get('net_cf'), 'Cashflow')
    create_chart_iframe(main_channel_id, main_fields_map.get('pure_alpha'), 'Pure_Alpha')
    create_chart_iframe(main_channel_id, main_fields_map.get('cost_minus_cf'), 'Product_cost - CF')
    create_chart_iframe(main_channel_id, main_fields_map.get('buffer'), 'Buffer')


# --- 3. CORE LOGIC & UPDATE FUNCTIONS ---

def perform_calculations(stock_assets, option_assets, user_inputs):
    """Performs all financial calculations for both stocks and options."""
    # --- Stock Calculations ---
    stock_metrics = {}
    total_stock_value = user_inputs['total_stock_value']
    portfolio_cash = user_inputs['portfolio_cash']
    current_prices = user_inputs['current_prices']

    stock_metrics['now_pv'] = total_stock_value + portfolio_cash
    
    reference_prices = [asset['reference_price'] for asset in stock_assets]
    live_prices = [current_prices[asset['ticker']] for asset in stock_assets]

    stock_metrics['t_0'] = np.prod(reference_prices) if reference_prices else 0
    stock_metrics['t_n'] = np.prod(live_prices) if live_prices else 0
    t_0, t_n = stock_metrics['t_0'], stock_metrics['t_n']
    
    stock_metrics['ln'] = -1500 * np.log(t_0 / t_n) if t_0 > 0 and t_n > 0 else 0
    number_of_assets = len(stock_assets)
    stock_metrics['log_pv'] = (number_of_assets * 1500) + stock_metrics['ln']
    stock_metrics['net_cf'] = stock_metrics['now_pv'] - stock_metrics['log_pv']

    # --- Options P/L Calculation ---
    total_options_pl = 0.0
    for option in option_assets:
        underlying_ticker = option.get("underlying_ticker")
        if not underlying_ticker:
            continue
            
        last_price = current_prices.get(underlying_ticker, 0.0)
        strike = option.get("strike", 0.0)
        contracts = option.get("contracts_or_shares", 0.0) # Using your 'contracts' field as number of shares
        premium = option.get("premium_paid_per_share", 0.0)

        # 1. Calculate Total Cost Basis
        total_cost_basis = contracts * premium
        
        # 2. Calculate Total Current Intrinsic Value
        # Assuming Call option for now. To handle Puts, you'd need a "type" field in the option config.
        intrinsic_value_per_share = max(0, last_price - strike)
        total_intrinsic_value = intrinsic_value_per_share * contracts
        
        # 3. Calculate Unrealized P/L for this option
        unrealized_pl = total_intrinsic_value - total_cost_basis
        total_options_pl += unrealized_pl
        
    return stock_metrics, total_options_pl

def handle_thingspeak_update(config, clients, stock_assets, metrics, user_inputs):
    """Handles updating ThingSpeak. IMPORTANT: Sends stock-only metrics to keep charts stable."""
    client_main, asset_clients = clients
    
    with st.expander("⚠️ Confirm to Add Cashflow and Update Holdings", expanded=False):
        if st.button("Confirm and Send All Data"):
            try:
                # We send the STABLE net_cf from stocks only.
                payload = {
                    'field1': metrics['net_cf'],
                    'field2': metrics['net_cf'] / user_inputs['product_cost'],
                    'field3': user_inputs['portfolio_cash'],
                    'field4': user_inputs['product_cost'] - metrics['net_cf']
                }
                client_main.update(payload)
                st.success("✅ Main Channel updated on Thingspeak (with stable stock data)!")
            except Exception as e:
                st.error(f"❌ Failed to update Main Channel: {e}")
            
            st.divider()
            # Update individual asset channels
            for ticker, client in asset_clients.items():
                try:
                    holding = user_inputs['current_holdings'][ticker]
                    channel_info = next((a['holding_channel'] for a in stock_assets if a['ticker'] == ticker), None)
                    if channel_info:
                        client.update({channel_info['field']: holding})
                        st.success(f"✅ Holding for {ticker} updated.")
                except Exception as e:
                    st.error(f"❌ Failed to update holding for {ticker}: {e}")

# --- 4. MAIN APPLICATION FLOW ---

def main():
    config = load_config()
    if not config: return

    # Separate assets into stocks and options once
    all_assets = config.get('assets', [])
    stock_assets = [item for item in all_assets if item.get('type', 'stock') == 'stock']
    option_assets = [item for item in all_assets if item.get('type') == 'option']

    clients = initialize_thingspeak_clients(config, stock_assets)
    
    initial_data = fetch_initial_data(stock_assets, option_assets, clients[1])
    
    user_inputs = render_ui_and_get_inputs(stock_assets, option_assets, initial_data, config.get('product_cost_default', 0.0))

    if st.button("Recalculate"):
        # This button just triggers a rerun, which is Streamlit's default behavior
        pass
        
    # --- Calculations ---
    stock_metrics, total_options_pl = perform_calculations(
        stock_assets,
        option_assets,
        user_inputs
    )
    
    # --- Display Results ---
    display_results(stock_metrics, total_options_pl, config)
    
    # --- Update Logic ---
    handle_thingspeak_update(config, clients, stock_assets, stock_metrics, user_inputs)
    
    # --- Chart Display ---
    render_charts(config)


if __name__ == "__main__":
    main()
