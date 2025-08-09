# """
# Portfolio Management Tool (Updated)
# ===================================

# This module is an updated version of the original `Add_CF` page from the
# user's Streamlit project.  It preserves the original user‑interface and
# output but replaces the core calculation model with a new formulation
# requested by the user: for each asset, the adjustment is computed as

#     **F = b * ln(tn / t0)**

# where `b` is a configurable coefficient for the asset, `tn` is the live
# price (current asset value), and `t0` is the reference price.  This differs
# from the original implementation, which used a constant `fix_c` plus a
# logarithmic term.  The updated model treats the baseline as zero and
# accumulates all dynamic contributions (including any historical `b_offset`)
# to compute the portfolio’s logarithmic value.

# All functions unrelated to the core calculation (loading configuration,
# rendering the UI, displaying results, updating ThingSpeak, etc.) are
# retained with only minimal changes to variable names for clarity.  Users
# who previously supplied a `fix_c` value in the asset configuration can
# continue to do so; the value will be treated as the `b` coefficient.

# """

# import pandas as pd
# import numpy as np
# import yfinance as yf
# import streamlit as st
# import thingspeak
# import json
# import streamlit.components.v1 as components
# from typing import Dict, Any, Tuple, List

# # --- Page Configuration ---
# st.set_page_config(page_title="Add_CF_F_Update", page_icon="🧮", layout="centered")

# # Initialize 'portfolio_cash' in session_state if it doesn't exist.
# if 'portfolio_cash' not in st.session_state:
#     st.session_state.portfolio_cash = 0.00

# # --- 1. CONFIGURATION & INITIALIZATION FUNCTIONS ---

# @st.cache_data
# def load_config(filename: str = "add_cf_config.json") -> Dict[str, Any]:
#     """Loads and parses the JSON configuration file."""
#     try:
#         with open(filename, 'r', encoding='utf-8') as f:
#             return json.load(f)
#     except (FileNotFoundError, json.JSONDecodeError) as e:
#         st.error(f"Error loading or parsing {filename}: {e}")
#         st.stop()


# @st.cache_resource
# def initialize_thingspeak_clients(
#     config: Dict[str, Any],
#     stock_assets: List[Dict[str, Any]],
#     option_assets: List[Dict[str, Any]]
# ) -> Tuple[thingspeak.Channel, Dict[str, thingspeak.Channel]]:
#     """
#     Initializes ThingSpeak clients for the main channel and individual asset channels.

#     Args:
#         config: Parsed configuration dictionary.
#         stock_assets: List of stock asset definitions.
#         option_assets: List of option asset definitions.

#     Returns:
#         Tuple containing the main channel client and a mapping of ticker to asset channel.
#     """
#     main_channel_config = config.get('thingspeak_channels', {}).get('main_output', {})
#     try:
#         client_main = thingspeak.Channel(
#             main_channel_config['channel_id'], main_channel_config['write_api_key']
#         )
#         asset_clients: Dict[str, thingspeak.Channel] = {}
#         for asset in stock_assets:
#             ticker = asset['ticker']
#             channel_info = asset.get('holding_channel', {})
#             if channel_info.get('channel_id'):
#                 asset_clients[ticker] = thingspeak.Channel(
#                     channel_info['channel_id'], channel_info['write_api_key']
#                 )

#         num_asset_clients = len(asset_clients)
#         num_option_assets = len(option_assets)
#         st.success(
#             f"Initialized main client and {num_asset_clients} asset {num_option_assets} option holding clients."
#         )

#         return client_main, asset_clients
#     except Exception as e:
#         st.error(f"Failed to initialize ThingSpeak clients: {e}")
#         st.stop()


# def fetch_initial_data(
#     stock_assets: List[Dict[str, Any]],
#     option_assets: List[Dict[str, Any]],
#     asset_clients: Dict[str, thingspeak.Channel]
# ) -> Dict[str, Dict[str, Any]]:
#     """
#     Fetches initial prices from yfinance and last holdings from ThingSpeak.
#     """
#     initial_data: Dict[str, Dict[str, Any]] = {}
#     tickers_to_fetch = {asset['ticker'].strip() for asset in stock_assets}
#     tickers_to_fetch.update(
#         {opt.get('underlying_ticker').strip() for opt in option_assets if opt.get('underlying_ticker')}
#     )

#     for ticker in tickers_to_fetch:
#         initial_data[ticker] = {}
#         try:
#             last_price = yf.Ticker(ticker).fast_info['lastPrice']
#             initial_data[ticker]['last_price'] = last_price
#         except Exception:
#             ref_price = next(
#                 (a.get('reference_price', 0.0) for a in stock_assets if a['ticker'].strip() == ticker),
#                 0.0,
#             )
#             initial_data[ticker]['last_price'] = ref_price
#             st.warning(f"Could not fetch price for {ticker}. Defaulting to reference price {ref_price}.")

#     for asset in stock_assets:
#         ticker = asset['ticker'].strip()
#         last_holding = 0.0
#         if ticker in asset_clients:
#             try:
#                 client = asset_clients[ticker]
#                 field = asset['holding_channel']['field']
#                 last_asset_json_string = client.get_field_last(field=field)
#                 if last_asset_json_string:
#                     data_dict = json.loads(last_asset_json_string)
#                     last_holding = float(data_dict[field])
#             except Exception as e:
#                 st.warning(f"Could not fetch holding for {ticker}. Defaulting to 0. Error: {e}")
#         initial_data[ticker]['last_holding'] = last_holding
#     return initial_data


# # --- 2. UI & DISPLAY FUNCTIONS ---

# def render_ui_and_get_inputs(
#     stock_assets: List[Dict[str, Any]],
#     option_assets: List[Dict[str, Any]],
#     initial_data: Dict[str, Dict[str, Any]],
#     product_cost_default: float
# ) -> Dict[str, Any]:
#     """
#     Renders all UI components and collects user inputs into a dictionary.
#     The interface remains unchanged from the original version to minimise
#     disruption for existing users.
#     """
#     user_inputs: Dict[str, Any] = {}
#     st.write("📊 Current Asset Prices")
#     current_prices: Dict[str, float] = {}
#     all_tickers = {asset['ticker'].strip() for asset in stock_assets}
#     all_tickers.update(
#         {opt['underlying_ticker'].strip() for opt in option_assets if opt.get('underlying_ticker')}
#     )

#     for ticker in sorted(list(all_tickers)):
#         label = f"ราคา_{ticker}"
#         price_value = initial_data.get(ticker, {}).get('last_price', 0.0)
#         current_prices[ticker] = st.number_input(
#             label, value=price_value, key=f"price_{ticker}", format="%.2f"
#         )
#     user_inputs['current_prices'] = current_prices

#     st.divider()
#     st.write("📦 Stock Holdings")
#     current_holdings: Dict[str, float] = {}
#     total_stock_value = 0.0
#     for asset in stock_assets:
#         ticker = asset['ticker'].strip()
#         holding_value = initial_data.get(ticker, {}).get('last_holding', 0.0)
#         asset_holding = st.number_input(
#             f"{ticker}_asset", value=holding_value, key=f"holding_{ticker}", format="%.2f"
#         )
#         current_holdings[ticker] = asset_holding
#         individual_asset_value = asset_holding * current_prices.get(ticker, 0.0)
#         st.write(f"มูลค่า {ticker}: **{individual_asset_value:,.2f}**")
#         total_stock_value += individual_asset_value
#     user_inputs['current_holdings'] = current_holdings
#     user_inputs['total_stock_value'] = total_stock_value

#     st.divider()
#     st.write("⚙️ Calculation Parameters")
#     user_inputs['product_cost'] = st.number_input(
#         'Product_cost', value=product_cost_default, format="%.2f"
#     )
#     st.number_input('Portfolio_cash', key='portfolio_cash', format="%.2f")
#     user_inputs['portfolio_cash'] = st.session_state.portfolio_cash
#     return user_inputs


# # --- 3. DISPLAY & CHARTING FUNCTIONS ---

# def display_results(metrics: Dict[str, float], options_pl: float, total_option_cost: float, config: Dict[str, Any]):
#     """
#     Displays all calculated metrics, including a detailed breakdown of the
#     logarithmic contributions.  The UI replicates the original display
#     while showing the updated formula.
#     """
#     st.divider()
#     with st.expander("📈 Results", expanded=True):
#         metric_label = (
#             f"Current Total Value (Stocks + Cash + Current_Options P/L: {options_pl:,.2f}) "
#             f"| Max_Roll_Over: ({-total_option_cost:,.2f})"
#         )
#         st.metric(label=metric_label, value=f"{metrics['now_pv']:,.2f}")

#         col1, col2 = st.columns(2)
#         col1.metric('log_pv Baseline', f"{metrics.get('log_pv_baseline', 0.0):,.2f}")
#         col2.metric('log_pv Adjustment (ln_weighted)', f"{metrics.get('ln_weighted', 0.0):,.2f}")

#         # Breakdown of contributions
#         with st.expander("Show 'ln_weighted' Calculation Breakdown"):
#             st.write(
#                 "ค่า `ln_weighted` คือผลรวมของการเปลี่ยนแปลงทั้งหมด (sum of b_offset + sum of b * ln part)"
#             )
#             ln_breakdown_data = metrics.get('ln_breakdown', [])
#             total_dynamic_contribution = 0.0
#             for item in ln_breakdown_data:
#                 total_dynamic_contribution += item['total_contribution']
#                 if item['ref_price'] > 0:
#                     formula_string = (
#                         f"{item['ticker']:<6}: {item['total_contribution']:+9.4f} = ["
#                         f"{item['b_offset']:>7.2f} (offset) + {item['b_coefficient']} * ln( {item['live_price']:.2f} / {item['ref_price']:.2f} ) ]"
#                     )
#                 else:
#                     formula_string = (
#                         f"{item['ticker']:<6}: {item['total_contribution']:+9.4f} = ["
#                         f"{item['b_offset']:>7.2f} (offset) + 0.00 ] (ref_price is zero)"
#                     )
#                 st.code(formula_string, language='text')
#             st.code("-------------------------------------------------------------------------")
#             st.code(f"Total Sum (ln_weighted) = {total_dynamic_contribution:+51.4f}")


# def render_charts(config: Dict[str, Any]):
#     """
#     Renders ThingSpeak charts using iframe components in a specific order.
#     """
#     st.write("📊 ThingSpeak Charts")
#     main_channel_config = config.get('thingspeak_channels', {}).get('main_output', {})
#     main_channel_id = main_channel_config.get('channel_id')
#     main_fields_map = main_channel_config.get('fields', {})

#     def create_chart_iframe(channel_id, field_name, chart_title):
#         if channel_id and field_name:
#             chart_number = field_name.replace('field', '')
#             url = (
#                 f'https://thingspeak.com/channels/{channel_id}/charts/{chart_number}?bgcolor=%23ffffff'
#                 '&color=%23d62020&dynamic=true&results=60&type=line&update=15'
#             )
#             st.write(f"**{chart_title}**")
#             components.iframe(url, width=800, height=200)
#             st.divider()

#     create_chart_iframe(main_channel_id, main_fields_map.get('net_cf'), 'Cashflow')
#     create_chart_iframe(main_channel_id, main_fields_map.get('now_pv'), 'Current Total Value')
#     create_chart_iframe(main_channel_id, main_fields_map.get('pure_alpha'), 'Pure_Alpha')
#     create_chart_iframe(main_channel_id, main_fields_map.get('cost_minus_cf'), 'Product_cost - CF')
#     create_chart_iframe(main_channel_id, main_fields_map.get('buffer'), 'Buffer')


# # --- 4. CALCULATION FUNCTION ---

# def calculate_metrics(
#     stock_assets: List[Dict[str, Any]],
#     option_assets: List[Dict[str, Any]],
#     user_inputs: Dict[str, Any],
#     config: Dict[str, Any]
# ) -> Tuple[Dict[str, float], float, float]:
#     """
#     Calculates all core metrics using the new F = b * ln(live/ref) model.

#     The function computes the total present value of the portfolio, the
#     logarithmic baseline (which is zero in this model), the sum of
#     logarithmic adjustments (`ln_weighted`), and the net cashflow.  It
#     preserves the option P/L logic from the original implementation.

#     Returns:
#         metrics: Dictionary of computed metrics.
#         total_options_pl: Profit/loss from options positions.
#         total_option_cost: Total cost basis for options.
#     """
#     metrics: Dict[str, float] = {}
#     portfolio_cash = user_inputs['portfolio_cash']
#     current_prices = user_inputs['current_prices']
#     total_stock_value = user_inputs['total_stock_value']

#     # P/L calculation for options (unchanged)
#     total_options_pl = 0.0
#     total_option_cost = 0.0
#     for option in option_assets:
#         underlying_ticker = option.get("underlying_ticker", "").strip()
#         if not underlying_ticker:
#             continue
#         last_price = current_prices.get(underlying_ticker, 0.0)
#         strike = option.get("strike", 0.0)
#         contracts = option.get("contracts_or_shares", 0.0)
#         premium = option.get("premium_paid_per_share", 0.0)
#         total_cost_basis = contracts * premium
#         total_option_cost += total_cost_basis
#         intrinsic_value = max(0.0, last_price - strike) * contracts
#         total_options_pl += intrinsic_value - total_cost_basis

#     # Present value of portfolio
#     metrics['now_pv'] = total_stock_value + portfolio_cash + total_options_pl

#     # Logarithmic baseline is zero under the new model
#     log_pv_baseline = 0.0
#     ln_weighted = 0.0
#     total_b_offset = 0.0
#     ln_breakdown: List[Dict[str, Any]] = []

#     for asset in stock_assets:
#         ticker = asset['ticker'].strip()
#         # The coefficient `b` replaces the old `fix_c` constant.  For backward
#         # compatibility, we fall back to the `fix_c` value if provided.
#         b_coeff = asset.get('b_coefficient', asset.get('fix_c', 1500))
#         b_offset = asset.get('b_offset', 0.0)
#         ref_price = asset.get('reference_price', 0.0)
#         live_price = current_prices.get(ticker, 0.0)

#         total_b_offset += b_offset
#         ln_part_contribution = 0.0
#         if ref_price > 0 and live_price > 0:
#             ln_part_contribution = b_coeff * np.log(live_price / ref_price)
#         ln_weighted += ln_part_contribution

#         ln_breakdown.append(
#             {
#                 "ticker": ticker,
#                 "b_offset": b_offset,
#                 "b_coefficient": b_coeff,
#                 "live_price": live_price,
#                 "ref_price": ref_price,
#                 "total_contribution": b_offset + ln_part_contribution,
#             }
#         )

#     # Assemble metrics
#     metrics['log_pv_baseline'] = log_pv_baseline
#     metrics['ln_weighted'] = total_b_offset + ln_weighted
#     metrics['log_pv'] = log_pv_baseline + metrics['ln_weighted']
#     metrics['net_cf'] = metrics['now_pv'] - metrics['log_pv']
#     metrics['ln_breakdown'] = ln_breakdown

#     return metrics, total_options_pl, total_option_cost


# # --- 5. THINGSPEAK UPDATE FUNCTION ---

# def handle_thingspeak_update(
#     config: Dict[str, Any],
#     clients: Tuple,  # (client_main, asset_clients)
#     stock_assets: List[Dict[str, Any]],
#     metrics: Dict[str, float],
#     user_inputs: Dict[str, Any],
# ) -> None:
#     """
#     Handles the UI for confirming and sending data to ThingSpeak.  When the
#     user confirms, it pushes the updated net cashflow and other values
#     to the appropriate channels.
#     """
#     client_main, asset_clients = clients
#     with st.expander("⚠️ Confirm to Add Cashflow and Update Holdings", expanded=False):
#         if st.button("Confirm and Send All Data"):
#             diff = metrics['net_cf'] - config.get('cashflow_offset', 0.0)
#             try:
#                 fields_map = config.get('thingspeak_channels', {}).get('main_output', {}).get('fields', {})
#                 payload = {
#                     fields_map.get('net_cf', 'field1'): diff,
#                     fields_map.get('pure_alpha', 'field2'): diff / user_inputs['product_cost'] if user_inputs['product_cost'] != 0 else 0,
#                     fields_map.get('buffer', 'field3'): user_inputs['portfolio_cash'],
#                     fields_map.get('cost_minus_cf', 'field4'): user_inputs['product_cost'] - diff,
#                     fields_map.get('now_pv', 'field5'): metrics.get('now_pv', 0.0),
#                 }
#                 client_main.update(payload)
#                 st.success("✅ Successfully updated Main Channel on Thingspeak!")
#             except Exception as e:
#                 st.error(f"❌ Failed to update Main Channel on Thingspeak: {e}")
#             st.divider()
#             for asset in stock_assets:
#                 ticker = asset['ticker'].strip()
#                 if ticker in asset_clients:
#                     try:
#                         current_holding = user_inputs['current_holdings'][ticker]
#                         field_to_update = asset['holding_channel']['field']
#                         asset_clients[ticker].update({field_to_update: current_holding})
#                         st.success(f"✅ Successfully updated holding for {ticker}.")
#                     except Exception as e:
#                         st.error(f"❌ Failed to update holding for {ticker}: {e}")


# # --- main() FUNCTION ---

# def main() -> None:
#     """Main entry point for the Streamlit application."""
#     config = load_config()
#     if not config:
#         return
#     all_assets = config.get('assets', [])
#     stock_assets = [item for item in all_assets if item.get('type', 'stock') == 'stock']
#     option_assets = [item for item in all_assets if item.get('type') == 'option']
#     clients = initialize_thingspeak_clients(config, stock_assets, option_assets)
#     initial_data = fetch_initial_data(stock_assets, option_assets, clients[1])
#     user_inputs = render_ui_and_get_inputs(
#         stock_assets,
#         option_assets,
#         initial_data,
#         config.get('product_cost_default', 0.0),
#     )
#     if st.button("Recalculate"):
#         pass  # This button currently triggers no extra logic but retains the original placement.
#     # Perform calculations
#     metrics, options_pl, total_option_cost = calculate_metrics(
#         stock_assets, option_assets, user_inputs, config
#     )
#     # Compute dynamic cashflow offset (baseline is zero, so offset equals product cost)
#     log_pv_baseline = metrics.get('log_pv_baseline', 0.0)
#     product_cost = user_inputs.get('product_cost', 0.0)
#     dynamic_offset = product_cost - log_pv_baseline
#     config['cashflow_offset'] = dynamic_offset
#     # Display results and provide update controls
#     display_results(metrics, options_pl, total_option_cost, config)
#     handle_thingspeak_update(config, clients, stock_assets, metrics, user_inputs)
#     render_charts(config)


# if __name__ == "__main__":
#     main()
