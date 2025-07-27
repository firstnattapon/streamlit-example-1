# main
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Tuple, Dict, Any

# ! NUMBA: Import Numba's Just-In-Time compiler for core acceleration
from numba import njit

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================
st.set_page_config(page_title="Hybrid_Multi_Mutation", page_icon="🧬", layout="wide")

class Strategy:
    """คลาสสำหรับเก็บชื่อกลยุทธ์ต่างๆ เพื่อให้เรียกใช้ง่ายและลดข้อผิดพลาด"""
    REBALANCE_DAILY = "Rebalance Daily"
    PERFECT_FORESIGHT = "Perfect Foresight (Max)"
    HYBRID_MULTI_MUTATION = "Hybrid (Multi-Mutation)"
    ORIGINAL_DNA = "Original DNA (Pre-Mutation)"

def load_config(filepath: str = "hybrid_seed_config.json") -> Dict[str, Any]:
    # Updated to match your JSON
    return {
        "assets": ["FFWM", "NEGG", "RIVN", "APLS", "NVTS", "QXO", "RXRX", "AGL", "FLNC", "GERN", "DYN"],
        "default_settings": {
            "selected_ticker": "FFWM", "start_date": "2024-01-01",
            "window_size": 30, "num_seeds": 1000, "max_workers": 1,
            "mutation_rate": 10.0, "num_mutations": 5
        },
        "manual_seed_by_asset": { ... }  # Your manual seeds, not used in this refactor but can be extended
    }

def initialize_session_state(config: Dict[str, Any]):
    defaults = config.get('default_settings', {})
    if 'test_ticker' not in st.session_state: st.session_state.test_ticker = defaults.get('selected_ticker', 'FFWM')
    if 'start_date' not in st.session_state:
        try: st.session_state.start_date = datetime.strptime(defaults.get('start_date', '2024-01-01'), '%Y-%m-%d').date()
        except ValueError: st.session_state.start_date = datetime(2024, 1, 1).date()
    if 'end_date' not in st.session_state: st.session_state.end_date = datetime.now().date()
    if 'window_size' not in st.session_state: st.session_state.window_size = defaults.get('window_size', 30)  
    if 'num_seeds' not in st.session_state: st.session_state.num_seeds = defaults.get('num_seeds', 1000)
    if 'max_workers' not in st.session_state: st.session_state.max_workers = defaults.get('max_workers', 8)
    if 'mutation_rate' not in st.session_state: st.session_state.mutation_rate = defaults.get('mutation_rate', 10.0)
    if 'num_mutations' not in st.session_state: st.session_state.num_mutations = defaults.get('num_mutations', 5)
    # New: For all_assets results
    if 'all_results' not in st.session_state: st.session_state.all_results = {}

# ==============================================================================
# 2. Core Calculation & Data Functions (Unchanged)
# ==============================================================================
# (get_ticker_data, _calculate_net_profit_numba, run_simulation remain the same as your original code)

# ==============================================================================
# 3. Strategy Action Generation (Unchanged)
# ==============================================================================
# (generate_actions_rebalance_daily, generate_actions_perfect_foresight, find_best_seed_for_window, find_best_mutation_for_sequence, generate_actions_hybrid_multi_mutation remain the same)

# ==============================================================================
# 4. Simulation Tracer Class (Unchanged)
# ==============================================================================
# (SimulationTracer class remains the same)

# ==============================================================================
# 5. UI Rendering Functions
# ==============================================================================
def display_comparison_charts(results: Dict[str, pd.DataFrame], chart_title: str = '📊 เปรียบเทียบกำไรสุทธิ (Net Profit)'):
    # (Unchanged)

def render_settings_tab():
    # (Unchanged, but note: test_ticker is now just for display, loop uses all assets)

def process_single_asset(ticker: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """New function: Process simulation for one asset"""
    try:
        start_date = str(st.session_state.start_date)
        end_date = str(st.session_state.end_date)
        ticker_data = get_ticker_data(ticker, start_date, end_date)
        if ticker_data.empty:
            st.warning(f"ไม่พบข้อมูลสำหรับ {ticker}")
            return None

        original_actions, final_actions, df_windows = generate_actions_hybrid_multi_mutation(
            ticker_data, st.session_state.window_size, st.session_state.num_seeds,
            st.session_state.max_workers, st.session_state.mutation_rate,
            st.session_state.num_mutations
        )

        prices = ticker_data['Close'].to_numpy()
        results = {
            Strategy.HYBRID_MULTI_MUTATION: run_simulation(prices.tolist(), final_actions.tolist()),
            Strategy.ORIGINAL_DNA: run_simulation(prices.tolist(), original_actions.tolist()),
            Strategy.REBALANCE_DAILY: run_simulation(prices.tolist(), generate_actions_rebalance_daily(len(prices)).tolist()),
            Strategy.PERFECT_FORESIGHT: run_simulation(prices.tolist(), generate_actions_perfect_foresight(prices.tolist()).tolist())
        }
        for name, df in results.items():
            if not df.empty: df.index = ticker_data.index[:len(df)]

        return {
            "simulation_results": results,
            "df_windows": df_windows,
            "ticker_data": ticker_data
        }
    except Exception as e:
        st.error(f"Error processing {ticker}: {e}")
        return None

def render_hybrid_multi_mutation_tab(config: Dict[str, Any]):
    # (Most unchanged, but button click now loops all assets)
    st.write("---")
    st.markdown(f"### 🧬 {Strategy.HYBRID_MULTI_MUTATION}")
    # (Info and expander unchanged)

    if st.button(f"🚀 Start Hybrid Multi-Mutation", type="primary"):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้อง")
            return

        assets = config["assets"]
        num_assets = len(assets)
        progress_bar = st.progress(0, text="Initializing...")
        st.session_state.all_results = {}  # Clear previous results

        with ThreadPoolExecutor(max_workers=st.session_state.max_workers) as executor:
            futures = {executor.submit(process_single_asset, ticker, config): ticker for ticker in assets}
            completed = 0
            for future in as_completed(futures):
                ticker = futures[future]
                result = future.result()
                if result:
                    st.session_state.all_results[ticker] = result
                completed += 1
                progress_bar.progress(completed / num_assets, text=f"Processed {completed}/{num_assets} assets")

        progress_bar.empty()
        st.success("การทดสอบเสร็จสมบูรณ์สำหรับทุก assets!")

    # Display results for all assets (in expanders to keep UI similar)
    if 'all_results' in st.session_state and st.session_state.all_results:
        for ticker, data in st.session_state.all_results.items():
            with st.expander(f"Results for {ticker}"):
                results = data["simulation_results"]
                df_windows = data["df_windows"]
                ticker_data = data["ticker_data"]

                chart_results = {k: v for k, v in results.items() if k != Strategy.ORIGINAL_DNA}
                display_comparison_charts(chart_results)

                st.divider()
                st.write("### 📈 สรุปผลลัพธ์")
                
                # (Summary metrics unchanged, but using data from this asset)

                if not df_windows.empty:
                    # (Window details, download, encode section unchanged, but scoped to this asset)

# (render_tracer_tab unchanged)

# ==============================================================================
# 6. Main Application
# ==============================================================================
def main():
    st.markdown("### 🧬 Hybrid Strategy Lab (Multi-Mutation)")
    st.caption("เครื่องมือทดลองและพัฒนากลยุทธ์ด้วย Numba-Accelerated Parallel Random Search")

    config = load_config()
    initialize_session_state(config)

    tab_list = ["⚙️ การตั้งค่า", f"🧬 {Strategy.HYBRID_MULTI_MUTATION}", "🔍 Tracer"]
    tabs = st.tabs(tab_list)

    with tabs[0]:
        render_settings_tab()
    with tabs[1]:
        render_hybrid_multi_mutation_tab(config)
    with tabs[2]:
        render_tracer_tab()

if __name__ == "__main__":
    main()
