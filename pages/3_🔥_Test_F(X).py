import streamlit as st 
import numpy as np
import datetime
import thingspeak
import pandas as pd
import yfinance as yf
import json
from functools import lru_cache
import concurrent.futures
import os
from typing import List, Dict, Optional
import tenacity
import pytz
import time
import threading
from dataclasses import dataclass
from enum import Enum

# Enhanced configuration for semi-automatic trading
st.set_page_config(
    page_title="Semi-Auto Trading Monitor", 
    page_icon="🤖", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- STEP 1: GOALS IMPLEMENTATION ---
# Goal 1: Semi-automatic trading (no API) for professionals
# Goal 2: UI, calculation principles, and output remain the same except for Goal 1 modifications
# Goal 3: Display full code

class TradingMode(Enum):
    MANUAL = "manual"
    SEMI_AUTO = "semi_auto"
    REVIEW = "review"

@dataclass
class TradeSignal:
    ticker: str
    action: str  # 'BUY' or 'SELL'
    confidence: float
    timestamp: datetime.datetime
    price: float
    quantity: float
    executed: bool = False

class SemiAutoTrader:
    """
    Core class for semi-automatic trading functionality.
    Handles signal detection, confirmation, and execution tracking.
    """
    def __init__(self):
        self.pending_signals: List[TradeSignal] = []
        self.executed_trades: List[TradeSignal] = []
        self.auto_execute_threshold = 0.8  # 80% confidence threshold
        self.confirmation_timeout = 300  # 5 minutes
        
    def add_signal(self, signal: TradeSignal):
        """Add a new trading signal to the queue"""
        self.pending_signals.append(signal)
        
    def get_pending_signals(self) -> List[TradeSignal]:
        """Get all pending signals that need confirmation"""
        return [s for s in self.pending_signals if not s.executed]
        
    def execute_signal(self, signal: TradeSignal, manual_override: bool = False):
        """Execute a trading signal"""
        if manual_override or signal.confidence >= self.auto_execute_threshold:
            signal.executed = True
            signal.timestamp = datetime.datetime.now()
            self.executed_trades.append(signal)
            return True
        return False

# Global semi-auto trader instance
if 'semi_auto_trader' not in st.session_state:
    st.session_state.semi_auto_trader = SemiAutoTrader()

# --- SimulationTracer Class (unchanged) ---
class SimulationTracer:
    def __init__(self, encoded_string: str):
        self.encoded_string: str = str(encoded_string)
        self._decode_and_set_attributes()

    def _decode_and_set_attributes(self):
        encoded_string = self.encoded_string
        if not encoded_string.isdigit():
            self.action_length: int = 0
            self.mutation_rate: int = 0
            self.dna_seed: int = 0
            self.mutation_seeds: List[int] = []
            self.mutation_rate_float: float = 0.0
            return

        decoded_numbers = []
        idx = 0
        try:
            while idx < len(encoded_string):
                length_of_number = int(encoded_string[idx])
                idx += 1
                number_str = encoded_string[idx : idx + length_of_number]
                idx += length_of_number
                decoded_numbers.append(int(number_str))
        except (IndexError, ValueError):
            pass

        if len(decoded_numbers) < 3:
            self.action_length: int = 0
            self.mutation_rate: int = 0
            self.dna_seed: int = 0
            self.mutation_seeds: List[int] = []
            self.mutation_rate_float: float = 0.0
            return

        self.action_length: int = decoded_numbers[0]
        self.mutation_rate: int = decoded_numbers[1]
        self.dna_seed: int = decoded_numbers[2]
        self.mutation_seeds: List[int] = decoded_numbers[3:]
        self.mutation_rate_float: float = self.mutation_rate / 100.0

    @lru_cache(maxsize=128)
    def run(self) -> np.ndarray:
        if self.action_length <= 0:
            return np.array([])
        dna_rng = np.random.default_rng(seed=self.dna_seed)
        current_actions = dna_rng.integers(0, 2, size=self.action_length)
        if self.action_length > 0:
            current_actions[0] = 1
        for m_seed in self.mutation_seeds:
            mutation_rng = np.random.default_rng(seed=m_seed)
            mutation_mask = mutation_rng.random(self.action_length) < self.mutation_rate_float
            current_actions[mutation_mask] = 1 - current_actions[mutation_mask]
            if self.action_length > 0:
                current_actions[0] = 1
        return current_actions

# --- Configuration Loading (unchanged) ---
@st.cache_data
def load_config(file_path='monitor_config.json') -> Dict:
    if not os.path.exists(file_path):
        st.error(f"Configuration file not found: {file_path}")
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON in config file: {e}")
        return {}

CONFIG_DATA = load_config()
if not CONFIG_DATA:
    st.stop()

ASSET_CONFIGS = CONFIG_DATA.get('assets', [])
GLOBAL_START_DATE = CONFIG_DATA.get('global_settings', {}).get('start_date')

if not ASSET_CONFIGS:
    st.error("No 'assets' list found in monitor_config.json")
    st.stop()

# --- ThingSpeak Clients (unchanged) ---
@st.cache_resource
def get_thingspeak_clients(configs: List[Dict]) -> Dict[int, thingspeak.Channel]:
    clients = {}
    unique_channels = set()
    for config in configs:
        mon_conf = config['monitor_field']
        unique_channels.add((mon_conf['channel_id'], mon_conf['api_key']))
        asset_conf = config['asset_field']
        unique_channels.add((asset_conf['channel_id'], asset_conf['api_key']))

    for channel_id, api_key in unique_channels:
        try:
            clients[channel_id] = thingspeak.Channel(channel_id, api_key, fmt='json')
        except Exception as e:
            st.error(f"Failed to create client for Channel ID {channel_id}: {e}")
    return clients

THINGSPEAK_CLIENTS = get_thingspeak_clients(ASSET_CONFIGS)

# --- Enhanced Cache Management with Semi-Auto State Preservation ---
def clear_all_caches():
    """Clear caches while preserving UI and semi-auto trading state"""
    # Clear data caches
    st.cache_data.clear()
    st.cache_resource.clear()
    sell.cache_clear()
    buy.cache_clear()

    # Preserve key UI states AND semi-auto trading state
    ui_state_keys_to_preserve = {
        'select_key', 'nex', 'Nex_day_sell', 'trading_mode',
        'semi_auto_trader', 'auto_execute_enabled', 'confirmation_required'
    }
    keys_to_delete = [k for k in list(st.session_state.keys()) if k not in ui_state_keys_to_preserve]
    for key in keys_to_delete:
        try:
            del st.session_state[key]
        except Exception:
            pass

    st.success("🗑️ Data caches cleared! Trading state preserved.")

def rerun_keep_selection(ticker: str):
    """Rerun app while maintaining selection and trading state"""
    st.session_state["_pending_select_key"] = ticker
    st.rerun()

# --- Calculation Utils (unchanged) ---
@lru_cache(maxsize=128)
def sell(asset, fix_c=1500, Diff=60):
    if asset == 0: return 0, 0, 0
    unit_price = round((fix_c - Diff) / asset, 2)
    adjust_qty = round(abs(asset * unit_price - fix_c) / unit_price) if unit_price != 0 else 0
    total = round(asset * unit_price + adjust_qty * unit_price, 2)
    return unit_price, adjust_qty, total

@lru_cache(maxsize=128)
def buy(asset, fix_c=1500, Diff=60):
    if asset == 0: return 0, 0, 0
    unit_price = round((fix_c + Diff) / asset, 2)
    adjust_qty = round(abs(asset * unit_price - fix_c) / unit_price) if unit_price != 0 else 0
    total = round(asset * unit_price - adjust_qty * unit_price, 2)
    return unit_price, adjust_qty, total

# --- Price Fetching with Retry (unchanged) ---
@st.cache_data(ttl=300)
@tenacity.retry(wait=tenacity.wait_fixed(2), stop=tenacity.stop_after_attempt(3))
def get_cached_price(ticker: str) -> float:
    try:
        return float(yf.Ticker(ticker).fast_info['lastPrice'])
    except Exception:
        return 0.0

# --- Helper: current date in New York (unchanged) ---
@st.cache_data(ttl=60)
def get_current_ny_date() -> datetime.date:
    ny_tz = pytz.timezone('America/New_York')
    return datetime.datetime.now(ny_tz).date()

# --- Data Fetching (unchanged) ---
@st.cache_data(ttl=300)
def fetch_all_data(configs: List[Dict], _clients_ref: Dict, start_date: str) -> Dict[str, tuple]:
    monitor_results = {}
    asset_results = {}

    def fetch_monitor(asset_config):
        ticker = asset_config['ticker']
        try:
            monitor_field_config = asset_config['monitor_field']
            client = _clients_ref[monitor_field_config['channel_id']]
            field_num = monitor_field_config['field']

            tickerData = yf.Ticker(ticker).history(period='max')[['Close']].round(3)
            try:
                tickerData.index = tickerData.index.tz_convert('Asia/Bangkok')
            except TypeError:
                tickerData.index = tickerData.index.tz_localize('UTC').tz_convert('Asia/Bangkok')

            if start_date:
                tickerData = tickerData[tickerData.index >= start_date]

            last_data_date = tickerData.index[-1].date() if not tickerData.empty else None

            fx_js_str = "0"
            try:
                field_data = client.get_field_last(field=str(field_num))
                retrieved_val = json.loads(field_data)[f"field{field_num}"]
                if retrieved_val is not None:
                    fx_js_str = str(retrieved_val)
            except Exception:
                pass

            tickerData['index'] = list(range(len(tickerData)))

            dummy_df = pd.DataFrame(index=[f'+{i}' for i in range(5)])
            df = pd.concat([tickerData, dummy_df], axis=0).fillna("")
            df['action'] = ""

            try:
                tracer = SimulationTracer(encoded_string=fx_js_str)
                final_actions = tracer.run()
                num_to_assign = min(len(df), len(final_actions))
                if num_to_assign > 0:
                    action_col_idx = df.columns.get_loc('action')
                    df.iloc[0:num_to_assign, action_col_idx] = final_actions[0:num_to_assign]
            except Exception as e:
                st.warning(f"Tracer Error for {ticker}: {e}")

            return ticker, (df.tail(7), fx_js_str, last_data_date)
        except Exception as e:
            st.error(f"Error in Monitor for {ticker}: {str(e)}")
            return ticker, (pd.DataFrame(), "0", None)

    def fetch_asset(asset_config):
        ticker = asset_config['ticker']
        try:
            asset_conf = asset_config['asset_field']
            client = _clients_ref[asset_conf['channel_id']]
            field_name = asset_conf['field']
            data = client.get_field_last(field=field_name)
            return ticker, float(json.loads(data)[field_name])
        except Exception:
            return ticker, 0.0

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(configs)) as executor:
        monitor_futures = [executor.submit(fetch_monitor, asset) for asset in configs]
        for future in concurrent.futures.as_completed(monitor_futures):
            ticker, result = future.result()
            monitor_results[ticker] = result

        asset_futures = [executor.submit(fetch_asset, asset) for asset in configs]
        for future in concurrent.futures.as_completed(asset_futures):
            ticker, result = future.result()
            asset_results[ticker] = result

    return {'monitors': monitor_results, 'assets': asset_results}

# --- Enhanced Signal Analysis Function ---
def analyze_trading_signal(config: Dict, action_val: Optional[int], current_price: float, asset_val: float) -> Optional[TradeSignal]:
    """Analyze and create trading signal with confidence scoring"""
    if action_val is None:
        return None
    
    ticker = config['ticker']
    
    # Calculate confidence based on multiple factors
    confidence_factors = []
    
    # Factor 1: Signal clarity (if we have a clear 0 or 1)
    confidence_factors.append(0.7)
    
    # Factor 2: Price momentum (simplified - in real implementation, use technical indicators)
    try:
        recent_data = yf.Ticker(ticker).history(period='5d')
        if len(recent_data) >= 2:
            price_change = (recent_data['Close'][-1] - recent_data['Close'][-2]) / recent_data['Close'][-2]
            momentum_confidence = min(abs(price_change) * 10, 0.3)  # Cap at 0.3
            confidence_factors.append(momentum_confidence)
    except:
        confidence_factors.append(0.1)
    
    # Factor 3: Volume confirmation (simplified)
    confidence_factors.append(0.1)  # Placeholder
    
    total_confidence = min(sum(confidence_factors), 1.0)
    
    action_text = "BUY" if action_val == 1 else "SELL"
    
    return TradeSignal(
        ticker=ticker,
        action=action_text,
        confidence=total_confidence,
        timestamp=datetime.datetime.now(),
        price=current_price,
        quantity=asset_val
    )

# --- UI Components (Enhanced) ---
def render_asset_inputs(configs: List[Dict], last_assets: Dict) -> Dict[str, float]:
    asset_inputs = {}
    cols = st.columns(len(configs))
    for i, config in enumerate(configs):
        with cols[i]:
            ticker = config['ticker']
            last_val = last_assets.get(ticker, 0.0)
            if config.get('option_config'):
                raw_label = config['option_config']['label']
            else:
                raw_label = ticker
            display_label = raw_label
            help_text = ""
            split_pos = raw_label.find('(')
            if split_pos != -1:
                display_label = raw_label[:split_pos].strip()
                help_text = raw_label[split_pos:].strip()
            else:
                help_text = "(NULL)"
            if config.get('option_config'):
                option_val = config['option_config']['base_value']
                real_val = st.number_input(
                    label=display_label, help=help_text,
                    step=0.001, value=last_val, key=f"input_{ticker}_real"
                )
                asset_inputs[ticker] = option_val + real_val
            else:
                val = st.number_input(
                    label=display_label, help=help_text,
                    step=0.001, value=last_val, key=f"input_{ticker}_asset"
                )
                asset_inputs[ticker] = val
    return asset_inputs

def render_asset_update_controls(configs: List[Dict], clients: Dict):
    with st.expander("Update Assets on ThingSpeak"):
        for config in configs:
            ticker = config['ticker']
            asset_conf = config['asset_field']
            field_name = asset_conf['field']
            if st.checkbox(f'@_{ticker}_ASSET', key=f'check_{ticker}'):
                add_val = st.number_input(f"New Value for {ticker}", step=0.001, value=0.0, key=f'input_{ticker}')
                if st.button(f"GO_{ticker}", key=f'btn_{ticker}'):
                    try:
                        client = clients[asset_conf['channel_id']]
                        client.update({field_name: add_val})
                        st.write(f"Updated {ticker} to: {add_val} on Channel {asset_conf['channel_id']}")
                        clear_all_caches()
                        rerun_keep_selection(st.session_state.get("select_key", ""))
                    except Exception as e:
                        st.error(f"Failed to update {ticker}: {e}")

# --- Enhanced Trading Section with Semi-Auto Features ---
def trading_section(config: Dict, asset_val: float, asset_last: float, df_data: pd.DataFrame, calc: Dict, nex: int, Nex_day_sell: int, clients: Dict):
    ticker = config['ticker']
    asset_conf = config['asset_field']
    field_name = asset_conf['field']
    
    # Get trading mode from session state
    trading_mode = st.session_state.get('trading_mode', TradingMode.MANUAL)
    semi_auto_trader = st.session_state.semi_auto_trader

    def get_action_val() -> Optional[int]:
        try:
            if df_data.empty or df_data.action.values[1 + nex] == "":
                return None
            raw_action = int(df_data.action.values[1 + nex])
            final_action = 1 - raw_action if Nex_day_sell == 1 else raw_action
            return final_action
        except (IndexError, ValueError, TypeError):
            return None

    action_val = get_action_val()
    
    # Get current price for signal analysis
    current_price = get_cached_price(ticker)
    
    # Analyze trading signal
    signal = analyze_trading_signal(config, action_val, current_price, asset_val) if action_val is not None else None
    
    # Enhanced checkbox logic based on trading mode
    has_signal = signal is not None
    
    if trading_mode == TradingMode.SEMI_AUTO and has_signal:
        # In semi-auto mode, show confidence and auto-execute if threshold met
        confidence_color = "🟢" if signal.confidence >= 0.8 else "🟡" if signal.confidence >= 0.6 else "🔴"
        checkbox_label = f'🤖 {signal.action} Signal {ticker} {confidence_color} ({signal.confidence:.1%})'
        
        # Auto-check if confidence is high
        default_checked = signal.confidence >= semi_auto_trader.auto_execute_threshold
        
        # Add to pending signals if not already there
        existing_signals = [s for s in semi_auto_trader.pending_signals if s.ticker == ticker and not s.executed]
        if not existing_signals and has_signal:
            semi_auto_trader.add_signal(signal)
            
    else:
        # Manual mode or no signal
        checkbox_label = f'Limit_Order_{ticker}'
        default_checked = has_signal
    
    limit_order_checked = st.checkbox(checkbox_label, value=default_checked, key=f'limit_order_{ticker}')

    # Show signal details in semi-auto mode
    if trading_mode == TradingMode.SEMI_AUTO and signal:
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.caption(f"⏰ Signal Time: {signal.timestamp.strftime('%H:%M:%S')}")
        with col_info2:
            st.caption(f"📊 Confidence: {signal.confidence:.1%}")
    
    if not limit_order_checked:
        return

    # Rest of trading logic (unchanged core functionality)
    sell_calc, buy_calc = calc['sell'], calc['buy']
    st.write('sell', '    ', 'A', buy_calc[1], 'P', buy_calc[0], 'C', buy_calc[2])
    
    col1, col2, col3 = st.columns(3)
    
    # Enhanced sell button with semi-auto confirmation
    sell_button_label = "🤖 AUTO SELL" if (trading_mode == TradingMode.SEMI_AUTO and signal and signal.action == "SELL") else f"GO_SELL_{ticker}"
    
    if col3.checkbox(f'sell_match_{ticker}'):
        if col3.button(sell_button_label):
            try:
                # Execute the trade
                client = clients[asset_conf['channel_id']]
                new_asset_val = asset_last - buy_calc[1]
                client.update({field_name: new_asset_val})
                col3.write(f"✅ SOLD: {new_asset_val}")
                
                # Mark signal as executed if in semi-auto mode
                if signal and trading_mode == TradingMode.SEMI_AUTO:
                    semi_auto_trader.execute_signal(signal, manual_override=True)
                    st.success(f"🤖 Semi-auto SELL executed for {ticker}")
                
                clear_all_caches()
                rerun_keep_selection(ticker)
            except Exception as e:
                st.error(f"Failed to SELL {ticker}: {e}")

    # Price and P/L display (unchanged)
    try:
        if current_price > 0:
            pv = current_price * asset_val
            fix_value = config['fix_c']
            pl_value = pv - fix_value
            pl_color = "#a8d5a2" if pl_value >= 0 else "#fbb"
            st.markdown(
                f"Price: **{current_price:,.3f}** | Value: **{pv:,.2f}** | P/L (vs {fix_value:,}) : <span style='color:{pl_color}; font-weight:bold;'>{pl_value:,.2f}</span>",
                unsafe_allow_html=True
            )
        else:
            st.info(f"Price data for {ticker} is currently unavailable.")
    except Exception:
        st.warning(f"Could not retrieve price data for {ticker}.")

    col4, col5, col6 = st.columns(3)
    st.write('buy', '   ', 'A', sell_calc[1], 'P', sell_calc[0], 'C', sell_calc[2])
    
    # Enhanced buy button with semi-auto confirmation
    buy_button_label = "🤖 AUTO BUY" if (trading_mode == TradingMode.SEMI_AUTO and signal and signal.action == "BUY") else f"GO_BUY_{ticker}"
    
    if col6.checkbox(f'buy_match_{ticker}'):
        if col6.button(buy_button_label):
            try:
                # Execute the trade
                client = clients[asset_conf['channel_id']]
                new_asset_val = asset_last + sell_calc[1]
                client.update({field_name: new_asset_val})
                col6.write(f"✅ BOUGHT: {new_asset_val}")
                
                # Mark signal as executed if in semi-auto mode
                if signal and trading_mode == TradingMode.SEMI_AUTO:
                    semi_auto_trader.execute_signal(signal, manual_override=True)
                    st.success(f"🤖 Semi-auto BUY executed for {ticker}")
                
                clear_all_caches()
                rerun_keep_selection(ticker)
            except Exception as e:
                st.error(f"Failed to BUY {ticker}: {e}")

# --- Enhanced Semi-Auto Control Panel ---
def render_semi_auto_controls():
    """Render the semi-automatic trading control panel"""
    st.subheader("🤖 Semi-Automatic Trading Controls")
    
    # Initialize session state for trading mode
    if 'trading_mode' not in st.session_state:
        st.session_state.trading_mode = TradingMode.MANUAL
    
    # Trading mode selection
    mode_options = [mode.value for mode in TradingMode]
    current_mode = st.session_state.trading_mode.value
    
    selected_mode = st.selectbox(
        "Trading Mode:",
        options=mode_options,
        index=mode_options.index(current_mode),
        format_func=lambda x: {
            "manual": "👤 Manual Trading",
            "semi_auto": "🤖 Semi-Automatic",
            "review": "📊 Review Only"
        }[x]
    )
    
    st.session_state.trading_mode = TradingMode(selected_mode)
    
    # Semi-auto specific controls
    if st.session_state.trading_mode == TradingMode.SEMI_AUTO:
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.semi_auto_trader.auto_execute_threshold = st.slider(
                "Auto-Execute Confidence Threshold",
                min_value=0.5,
                max_value=1.0,
                value=st.session_state.semi_auto_trader.auto_execute_threshold,
                step=0.05,
                help="Signals above this confidence will be automatically checked"
            )
        
        with col2:
            st.session_state.semi_auto_trader.confirmation_timeout = st.number_input(
                "Signal Timeout (seconds)",
                min_value=60,
                max_value=1800,
                value=st.session_state.semi_auto_trader.confirmation_timeout,
                step=30,
                help="How long to wait for manual confirmation"
            )
    
    # Display trading status
    trader = st.session_state.semi_auto_trader
    pending_signals = trader.get_pending_signals()
    
    if pending_signals:
        st.subheader("⏳ Pending Signals")
        for signal in pending_signals:
            age_seconds = (datetime.datetime.now() - signal.timestamp).total_seconds()
            age_text = f"{int(age_seconds)}s ago"
            
            confidence_emoji = "🟢" if signal.confidence >= 0.8 else "🟡" if signal.confidence >= 0.6 else "🔴"
            
            st.info(f"{confidence_emoji} {signal.action} {signal.ticker} | Confidence: {signal.confidence:.1%} | {age_text}")
    
    # Display recent executions
    if trader.executed_trades:
        with st.expander(f"📊 Recent Executions ({len(trader.executed_trades)})"):
            for trade in trader.executed_trades[-5:]:  # Show last 5
                st.text(f"✅ {trade.action} {trade.ticker} at {trade.timestamp.strftime('%H:%M:%S')} | Confidence: {trade.confidence:.1%}")

# --- Main Application Logic ---
def main():
    # Fetch data
    all_data = fetch_all_data(ASSET_CONFIGS, THINGSPEAK_CLIENTS, GLOBAL_START_DATE)
    monitor_data_all = all_data['monitors']
    last_assets_all = all_data['assets']

    # Initialize session state
    if 'select_key' not in st.session_state:
        st.session_state.select_key = ""
    if 'nex' not in st.session_state:
        st.session_state.nex = 0
    if 'Nex_day_sell' not in st.session_state:
        st.session_state.Nex_day_sell = 0

    # Handle pending selection
    pending = st.session_state.pop("_pending_select_key", None)
    if pending:
        st.session_state.select_key = pending

    # Create enhanced tabs
    tab1, tab2, tab3 = st.tabs(["📈 Monitor", "⚙️ Controls", "🤖 Semi-Auto"])

    with tab3:
        render_semi_auto_controls()

    with tab2:
        Nex_day_ = st.checkbox('nex_day', value=(st.session_state.nex == 1))

        if Nex_day_:
            nex_col, Nex_day_sell_col, *_ = st.columns([1,1,3])
            if nex_col.button("Nex_day"):
                st.session_state.nex = 1
                st.session_state.Nex_day_sell = 0
            if Nex_day_sell_col.button("Nex_day_sell"):
                st.session_state.nex = 1
                st.session_state.Nex_day_sell = 1
        else:
            st.session_state.nex = 0
            st.session_state.Nex_day_sell = 0

        nex = st.session_state.nex
        Nex_day_sell = st.session_state.Nex_day_sell

        if Nex_day_:
            st.write(f"nex value = {nex}", f" | Nex_day_sell = {Nex_day_sell}" if Nex_day_sell else "")

        st.write("---")
        x_2 = st.number_input('Diff', step=1, value=60)
        st.write("---")
        asset_inputs = render_asset_inputs(ASSET_CONFIGS, last_assets_all)
        st.write("---")
        Start = st.checkbox('start')
        if Start:
            render_asset_update_controls(ASSET_CONFIGS, THINGSPEAK_CLIENTS)

    with tab1:
        current_ny_date = get_current_ny_date()

        selectbox_labels = {}
        ticker_actions = {}
        for config in ASSET_CONFIGS:
            ticker = config['ticker']
            df_data, fx_js_str, last_data_date = monitor_data_all.get(ticker, (pd.DataFrame(), "0", None))
            action_emoji, final_action_val = "", None
            
            # Enhanced emoji system for semi-auto mode
            trading_mode = st.session_state.get('trading_mode', TradingMode.MANUAL)
            
            if nex == 0 and last_data_date and last_data_date < current_ny_date:
                action_emoji = "🟡 "
            else:
                try:
                    if not df_data.empty and df_data.action.values[1 + nex] != "":
                        raw_action = int(df_data.action.values[1 + nex])
                        final_action_val = 1 - raw_action if Nex_day_sell == 1 else raw_action
                        
                        if trading_mode == TradingMode.SEMI_AUTO:
                            # Enhanced emojis for semi-auto mode
                            if final_action_val == 1:
                                action_emoji = "🤖💚 "  # Semi-auto buy
                            elif final_action_val == 0:
                                action_emoji = "🤖💔 "  # Semi-auto sell
                        else:
                            # Standard emojis for manual mode
                            if final_action_val == 1:
                                action_emoji = "🟢 "
                            elif final_action_val == 0:
                                action_emoji = "🔴 "
                except (IndexError, ValueError, TypeError):
                    pass
            
            ticker_actions[ticker] = final_action_val
            
            # Add trading mode indicator to label
            mode_indicator = ""
            if trading_mode == TradingMode.SEMI_AUTO:
                mode_indicator = " [🤖]"
            elif trading_mode == TradingMode.REVIEW:
                mode_indicator = " [📊]"
                
            selectbox_labels[ticker] = f"{action_emoji}{ticker}{mode_indicator} (f(x): {fx_js_str})"

        all_tickers = [config['ticker'] for config in ASSET_CONFIGS]
        selectbox_options = [""]
        if nex == 1:
            selectbox_options.extend(["Filter Buy Tickers", "Filter Sell Tickers"])
            if st.session_state.get('trading_mode') == TradingMode.SEMI_AUTO:
                selectbox_options.extend(["Filter Auto-Execute Ready", "Filter Pending Confirmation"])
        selectbox_options.extend(all_tickers)

        if st.session_state.select_key not in selectbox_options:
            st.session_state.select_key = ""

        def format_selectbox_options(option_name):
            if option_name in ["", "Filter Buy Tickers", "Filter Sell Tickers", "Filter Auto-Execute Ready", "Filter Pending Confirmation"]:
                format_map = {
                    "": "Show All",
                    "Filter Buy Tickers": "🟢 Buy Signals",
                    "Filter Sell Tickers": "🔴 Sell Signals",
                    "Filter Auto-Execute Ready": "🤖✅ Auto-Execute Ready",
                    "Filter Pending Confirmation": "🤖⏳ Pending Confirmation"
                }
                return format_map.get(option_name, option_name)
            return selectbox_labels.get(option_name, option_name).split(' (f(x):')[0]

        st.selectbox(
            "Select Ticker to View:",
            options=selectbox_options,
            format_func=format_selectbox_options,
            key="select_key"
        )
        st.write("_____")

        selected_option = st.session_state.select_key
        
        # Enhanced filtering logic for semi-auto mode
        if selected_option == "":
            configs_to_display = ASSET_CONFIGS
        elif selected_option == "Filter Buy Tickers":
            buy_tickers = {t for t, action in ticker_actions.items() if action == 1}
            configs_to_display = [config for config in ASSET_CONFIGS if config['ticker'] in buy_tickers]
        elif selected_option == "Filter Sell Tickers":
            sell_tickers = {t for t, action in ticker_actions.items() if action == 0}
            configs_to_display = [config for config in ASSET_CONFIGS if config['ticker'] in sell_tickers]
        elif selected_option == "Filter Auto-Execute Ready":
            # Show signals ready for auto-execution
            trader = st.session_state.semi_auto_trader
            ready_tickers = set()
            for signal in trader.get_pending_signals():
                if signal.confidence >= trader.auto_execute_threshold:
                    ready_tickers.add(signal.ticker)
            configs_to_display = [config for config in ASSET_CONFIGS if config['ticker'] in ready_tickers]
        elif selected_option == "Filter Pending Confirmation":
            # Show signals pending manual confirmation
            trader = st.session_state.semi_auto_trader
            pending_tickers = {s.ticker for s in trader.get_pending_signals() if s.confidence < trader.auto_execute_threshold}
            configs_to_display = [config for config in ASSET_CONFIGS if config['ticker'] in pending_tickers]
        else:
            configs_to_display = [config for config in ASSET_CONFIGS if config['ticker'] == selected_option]

        # Calculate trading parameters
        calculations = {}
        for config in ASSET_CONFIGS:
            ticker = config['ticker']
            asset_value = asset_inputs.get(ticker, 0.0)
            fix_c = config['fix_c']
            calculations[ticker] = {
                'sell': sell(asset_value, fix_c=fix_c, Diff=x_2),
                'buy': buy(asset_value, fix_c=fix_c, Diff=x_2)
            }

        # Display trading sections
        for config in configs_to_display:
            ticker = config['ticker']
            df_data, fx_js_str, _ = monitor_data_all.get(ticker, (pd.DataFrame(), "0", None))
            asset_last = last_assets_all.get(ticker, 0.0)
            asset_val = asset_inputs.get(ticker, 0.0)
            calc = calculations.get(ticker, {})

            title_label = selectbox_labels.get(ticker, ticker)
            st.write(title_label)

            trading_section(config, asset_val, asset_last, df_data, calc, nex, Nex_day_sell, THINGSPEAK_CLIENTS)

            with st.expander("Show Raw Data Action"):
                st.dataframe(df_data, use_container_width=True)
            st.write("_____")

    # Enhanced sidebar with semi-auto status
    with st.sidebar:
        st.header("🤖 Trading System")
        
        # Display current mode
        current_mode = st.session_state.get('trading_mode', TradingMode.MANUAL)
        mode_display = {
            TradingMode.MANUAL: "👤 Manual",
            TradingMode.SEMI_AUTO: "🤖 Semi-Auto",
            TradingMode.REVIEW: "📊 Review Only"
        }
        st.info(f"Mode: {mode_display[current_mode]}")
        
        # Semi-auto status
        if current_mode == TradingMode.SEMI_AUTO:
            trader = st.session_state.semi_auto_trader
            pending_count = len(trader.get_pending_signals())
            executed_count = len(trader.executed_trades)
            
            st.metric("Pending Signals", pending_count)
            st.metric("Executed Today", executed_count)
            
            if pending_count > 0:
                st.warning(f"⚠️ {pending_count} signals awaiting action!")
        
        st.write("---")
        
        # Enhanced rerun button
        if st.button("🔄 REFRESH DATA"):
            current_selection = st.session_state.get("select_key", "")
            clear_all_caches()
            if current_selection in [c['ticker'] for c in ASSET_CONFIGS]:
                rerun_keep_selection(current_selection)
            else:
                st.rerun()
        
        # Emergency stop for semi-auto
        if current_mode == TradingMode.SEMI_AUTO:
            if st.button("🚨 EMERGENCY STOP", type="secondary"):
                st.session_state.trading_mode = TradingMode.MANUAL
                st.session_state.semi_auto_trader.pending_signals.clear()
                st.warning("🚨 Semi-auto mode disabled! Switched to manual mode.")
                st.rerun()

# Run the main application
if __name__ == "__main__":
    main()
