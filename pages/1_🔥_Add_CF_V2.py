import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import thingspeak
import json
import streamlit.components.v1 as components
from typing import Dict, Any, Tuple, List

# --- Page Configuration ---
st.set_page_config(page_title="Add_CF_V1_with_Options", page_icon="🚀")

# --- 1. CONFIGURATION & INITIALIZATION FUNCTIONS ---

@st.cache_data
def load_config(filename: str = "add_cf_config.json") -> Dict[str, Any]:
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.error(f"Error loading or parsing {filename}: {e}")
        st.stop()

@st.cache_resource
def initialize_thingspeak_clients(config: Dict[str, Any], stock_assets: List[Dict[str, Any]]) -> Tuple[thingspeak.Channel, Dict[str, thingspeak.Channel]]:
    main_channel_config = config.get('thingspeak_channels', {}).get('main_output', {})
    try:
        client_main = thingspeak.Channel(main_channel_config['channel_id'], main_channel_config['write_api_key'])
        asset_clients = {}
        for asset in stock_assets:
            ticker = asset['ticker']
            channel_info = asset.get('holding_channel', {})
            if channel_info.get('channel_id'):
                asset_clients[ticker] = thingspeak.Channel(channel_info['channel_id'], channel_info['write_api_key'])
        st.success(f"Initialized main client and {len(asset_clients)} stock clients.")
        return client_main, asset_clients
    except Exception as e:
        st.error(f"Failed to initialize ThingSpeak clients: {e}")
        st.stop()

def fetch_initial_data(stock_assets: List[Dict[str, Any]], option_assets: List[Dict[str, Any]], asset_clients: Dict[str, thingspeak.Channel]) -> Dict[str, Dict[str, Any]]:
    initial_data = {}
    tickers_to_fetch = {asset['ticker'].strip() for asset in stock_assets}
    tickers_to_fetch.update({opt.get('underlying_ticker').strip() for opt in option_assets if opt.get('underlying_ticker')})
    
    for ticker in tickers_to_fetch:
        initial_data[ticker] = {}
        try:
            last_price = yf.Ticker(ticker).fast_info['lastPrice']
            initial_data[ticker]['last_price'] = last_price
        except Exception:
            ref_price = next((a.get('reference_price', 0.0) for a in stock_assets if a['ticker'].strip() == ticker), 0.0)
            initial_data[ticker]['last_price'] = ref_price
            st.warning(f"Could not fetch price for {ticker}. Defaulting to {ref_price}.")
            
    for asset in stock_assets:
        ticker = asset["ticker"].strip()
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
                st.warning(f"Could not fetch holding for {ticker}. Defaulting to 0. Error: {e}")
        initial_data[ticker]['last_holding'] = last_holding
    return initial_data

# --- 2. UI & DISPLAY FUNCTIONS ---

def render_ui_and_get_inputs(stock_assets: List[Dict[str, Any]], option_assets: List[Dict[str, Any]], initial_data: Dict[str, Dict[str, Any]], product_cost_default: float) -> Dict[str, Any]:
    user_inputs = {}
    st.write("📊 Current Asset Prices")
    current_prices = {}
    all_tickers = {asset['ticker'].strip() for asset in stock_assets}
    all_tickers.update({opt['underlying_ticker'].strip() for opt in option_assets if opt.get('underlying_ticker')})
    
    for ticker in sorted(list(all_tickers)):
        label = f"ราคา_{ticker}"
        price_value = initial_data.get(ticker, {}).get('last_price', 0.0)
        current_prices[ticker] = st.number_input(label, value=price_value, key=f"price_{ticker}", format="%.2f")
    user_inputs['current_prices'] = current_prices
    
    st.divider()
    st.write("📦 Stock Holdings")
    current_holdings = {}
    total_stock_value = 0.0
    for asset in stock_assets:
        ticker = asset["ticker"].strip()
        holding_value = initial_data.get(ticker, {}).get('last_holding', 0.0)
        asset_holding = st.number_input(f"{ticker}_asset", value=holding_value, key=f"holding_{ticker}", format="%.2f")
        current_holdings[ticker] = asset_holding
        individual_asset_value = asset_holding * current_prices.get(ticker, 0.0)
        st.write(f"มูลค่า {ticker}: **{individual_asset_value:,.2f}**")
        total_stock_value += individual_asset_value
    user_inputs['current_holdings'] = current_holdings
    user_inputs['total_stock_value'] = total_stock_value
    
    st.divider()
    st.write("⚙️ Calculation Parameters")
    user_inputs['product_cost'] = st.number_input('Product_cost', value=product_cost_default, format="%.2f")
    user_inputs['portfolio_cash'] = st.number_input('Portfolio_cash', value=0.00, format="%.2f")
    return user_inputs

def display_results(metrics: Dict[str, float], options_pl: float, config: Dict[str, Any]):
    st.divider()
    with st.expander("📈 Results", expanded=True):
        st.metric(f"Current Total Value (Stocks + Cash + Options P/L: {options_pl:,.2f})", f"{metrics['now_pv']:,.2f}")
        col1, col2 = st.columns(2)
        col1.metric('t_0 (Product of Stock Reference Prices)', f"{metrics['t_0']:,.2f}")
        col2.metric('t_n (Product of Stock Live Prices)', f"{metrics['t_n']:,.2f}")
        st.metric('Fix Component (ln)', f"{metrics['ln']:,.2f}")
        st.metric( f"Log PV (Calculated: {metrics['log_pv'] - metrics['ln']  :,.2f}{metrics['ln']:,.2f} )" , f"{metrics['log_pv']:,.2f}")
        st.metric(label="💰 Net Cashflow (Combined)", value=f"{metrics['net_cf']:,.2f}")
        st.metric(label=f"💰 Baseline {metrics['log_pv'] -  metrics['ln']  :,.1f } - {config.get('product_cost_default', 0)}  = +2750 ", value=f"{metrics['net_cf'] - config.get('cashflow_offset', 0.0):,.2f}")
        st.metric(label=f"💰 Net  @ {config.get('cashflow_offset_comment', '')}",  value=   f"( { 1699.46 - ( metrics['net_cf'] - config.get('cashflow_offset', 0.0)):,.2f} )" )


def render_charts(config: Dict[str, Any]):
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
    create_chart_iframe(main_channel_id, main_fields_map.get('net_cf'), 'Cashflow')
    create_chart_iframe(main_channel_id, main_fields_map.get('pure_alpha'), 'Pure_Alpha')
    create_chart_iframe(main_channel_id, main_fields_map.get('cost_minus_cf'), 'Product_cost - CF')
    create_chart_iframe(main_channel_id, main_fields_map.get('buffer'), 'Buffer')

# --- 3. CORE LOGIC & UPDATE FUNCTIONS ---

def calculate_metrics(stock_assets: List[Dict[str, Any]], option_assets: List[Dict[str, Any]], user_inputs: Dict[str, Any]) -> Tuple[Dict[str, float], float]:
    metrics = {}
    total_stock_value = user_inputs['total_stock_value']
    portfolio_cash = user_inputs['portfolio_cash']
    current_prices = user_inputs['current_prices']
    total_options_pl = 0.0
    for option in option_assets:
        underlying_ticker = option.get("underlying_ticker")
        if not underlying_ticker: continue
        
        underlying_ticker = underlying_ticker.strip() # Sanitize ticker
        
        last_price = current_prices.get(underlying_ticker, 0.0)
        strike = option.get("strike", 0.0)
        contracts = option.get("contracts_or_shares", 0.0)
        premium = option.get("premium_paid_per_share", 0.0)
        total_cost_basis = contracts * premium
        intrinsic_value_per_share = max(0, last_price - strike)
        total_intrinsic_value = intrinsic_value_per_share * contracts
        unrealized_pl = total_intrinsic_value - total_cost_basis
        total_options_pl += unrealized_pl
        
    metrics['now_pv'] = total_stock_value + portfolio_cash + total_options_pl
    reference_prices = [asset['reference_price'] for asset in stock_assets]
    live_prices = [current_prices[asset['ticker'].strip()] for asset in stock_assets]
    metrics['t_0'] = np.prod(reference_prices) if reference_prices else 0
    metrics['t_n'] = np.prod(live_prices) if live_prices else 0
    t_0, t_n = metrics['t_0'], metrics['t_n']
    metrics['ln'] = -1500 * np.log(t_0 / t_n) if t_0 > 0 and t_n > 0 else 0
    number_of_assets = len(stock_assets)
    metrics['log_pv'] = (number_of_assets * 1500) + metrics['ln']
    metrics['net_cf'] = metrics['now_pv'] - metrics['log_pv']
    return metrics, total_options_pl

def handle_thingspeak_update(config: Dict[str, Any], clients: Tuple, stock_assets: List[Dict[str, Any]], metrics: Dict[str, float], user_inputs: Dict[str, Any]):
    client_main, asset_clients = clients
    with st.expander("⚠️ Confirm to Add Cashflow and Update Holdings", expanded=False):
        if st.button("Confirm and Send All Data"):
            diff =  metrics['net_cf'] - config.get('cashflow_offset', 0.0) 
            try:
                payload = {
                    'field1': diff ,
                    'field2': diff  / user_inputs['product_cost'],
                    'field3': user_inputs['portfolio_cash'],
                    'field4': user_inputs['product_cost'] - diff
                }
                client_main.update(payload)
                st.success("✅ Successfully updated Main Channel on Thingspeak!")
            except Exception as e:
                st.error(f"❌ Failed to update Main Channel on Thingspeak: {e}")
            st.divider()
            for asset in stock_assets:
                ticker = asset['ticker'].strip()
                if ticker in asset_clients:
                    try:
                        current_holding = user_inputs['current_holdings'][ticker]
                        field_to_update = asset['holding_channel']['field']
                        asset_clients[ticker].update({field_to_update: current_holding})
                        st.success(f"✅ Successfully updated holding for {ticker}.")
                    except Exception as e:
                        st.error(f"❌ Failed to update holding for {ticker}: {e}")

# --- 4. MAIN APPLICATION FLOW ---

def main():
    config = load_config()
    if not config: return
    all_assets = config.get('assets', [])
    stock_assets = [item for item in all_assets if item.get('type', 'stock') == 'stock']
    option_assets = [item for item in all_assets if item.get('type') == 'option']
    clients = initialize_thingspeak_clients(config, stock_assets)
    initial_data = fetch_initial_data(stock_assets, option_assets, clients[1])
    user_inputs = render_ui_and_get_inputs(stock_assets, option_assets, initial_data, config.get('product_cost_default', 0.0))
    if st.button("Recalculate"):
        pass 
    metrics, options_pl = calculate_metrics(stock_assets, option_assets, user_inputs)
    display_results(metrics, options_pl, config) 
    handle_thingspeak_update(config, clients, stock_assets, metrics, user_inputs)
    render_charts(config)

if __name__ == "__main__":
    main()
