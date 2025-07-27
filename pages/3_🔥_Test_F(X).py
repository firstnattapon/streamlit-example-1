# main.py (Full Refactored Code)

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Tuple, Dict, Any
from numba import njit
import json  # เพิ่มสำหรับ load JSON config
import os  # สำหรับ cpu count ถ้าต้องการ

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================
st.set_page_config(page_title="Hybrid_Multi_Mutation", page_icon="🧬", layout="wide")

class Strategy:
    REBALANCE_DAILY = "Rebalance Daily"
    PERFECT_FORESIGHT = "Perfect Foresight (Max)"
    HYBRID_MULTI_MUTATION = "Hybrid (Multi-Mutation)"
    ORIGINAL_DNA = "Original DNA (Pre-Mutation)"

def load_config(json_str: str = None) -> Dict[str, Any]:
    # Load from JSON string (หรือ file ถ้าต้องการ: with open(filepath) as f: return json.load(f))
    if json_str is None:
        # Default JSON จากที่คุณให้
        json_str = '''
        {
          "assets": ["FFWM", "NEGG", "RIVN", "APLS", "NVTS", "QXO", "RXRX", "AGL", "FLNC", "GERN", "DYN"],
          "default_settings": {
            "selected_ticker": "FFWM", "start_date": "2024-01-01",
            "window_size": 30, "num_seeds": 1000, "max_workers": 1,
            "mutation_rate": 10.0, "num_mutations": 5
          },
          "manual_seed_by_asset": {
            "FFWM": [{"seed": 9999, "size": 60, "tail": 30}],
            "NEGG": [{"seed": 9999, "size": 60, "tail": 30}],
            "APLS": [{"seed": 9999, "size": 60, "tail": 30}],
            "QXO": [{"seed": 9999, "size": 60, "tail": 30}],
            "RXRX": [{"seed": 9999, "size": 60, "tail": 30}],
            "AGL": [{"seed": 9999, "size": 60, "tail": 30}],
            "NVTS": [{"seed": 9999, "size": 60, "tail": 30}],
            "RIVN": [{"seed": 9999, "size": 60, "tail": 30}],
            "FLNC": [{"seed": 9999, "size": 60, "tail": 30}],
            "GERN": [{"seed": 9999, "size": 60, "tail": 30}],
            "DYN": [{"seed": 9999, "size": 60, "tail": 30}]
          }
        }
        '''
    return json.loads(json_str)

def initialize_session_state(config: Dict[str, Any]):
    defaults = config.get('default_settings', {})
    st.session_state.setdefault('test_ticker', defaults.get('selected_ticker', 'FFWM'))
    if 'start_date' not in st.session_state:
        try:
            st.session_state.start_date = datetime.strptime(defaults.get('start_date', '2024-01-01'), '%Y-%m-%d').date()
        except ValueError:
            st.session_state.start_date = datetime(2024, 1, 1).date()
    st.session_state.setdefault('end_date', datetime.now().date())
    st.session_state.setdefault('window_size', defaults.get('window_size', 30))
    st.session_state.setdefault('num_seeds', defaults.get('num_seeds', 1000))
    st.session_state.setdefault('max_workers', defaults.get('max_workers', 1))
    st.session_state.setdefault('mutation_rate', defaults.get('mutation_rate', 10.0))
    st.session_state.setdefault('num_mutations', defaults.get('num_mutations', 5))
    st.session_state.setdefault('batch_results', {})  # สำหรับเก็บ results ของ all assets

# ==============================================================================
# 2. Core Calculation & Data Functions (เหมือนเดิม แต่เพิ่ม caching สำหรับ batch)
# ==============================================================================
@st.cache_data(ttl=3600)
def get_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        data = yf.Ticker(ticker).history(start=start_date, end=end_date)[['Close']]
        if data.empty:
            return pd.DataFrame()
        if data.index.tz is None:
            data = data.tz_localize('UTC').tz_convert('Asia/Bangkok')
        else:
            data = data.tz_convert('Asia/Bangkok')
        return data
    except Exception as e:
        st.error(f"❌ ไม่สามารถดึงข้อมูล {ticker} ได้: {str(e)}")
        return pd.DataFrame()

# (ส่วนอื่นๆ ของ Core Functions เหมือนเดิม: _calculate_net_profit_numba, run_simulation, generate_actions_rebalance_daily, generate_actions_perfect_foresight, find_best_seed_for_window, find_best_mutation_for_sequence)

# ปรับ generate_actions_hybrid_multi_mutation เพื่อรับ manual_seed
def generate_actions_hybrid_multi_mutation(
    ticker_data: pd.DataFrame,
    window_size: int,
    num_seeds: int,
    max_workers: int,
    mutation_rate_pct: float,
    num_mutations: int,
    manual_seed: int = None  # เพิ่ม parameter สำหรับ manual seed
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    # (logic เดิม แต่ปรับ dna_seed ถ้ามี manual_seed)
    prices = ticker_data['Close'].to_numpy()
    n = len(prices)
    # ... (ส่วนอื่นเหมือนเดิม)
    for i, start_index in enumerate(range(0, n, window_size)):
        # ...
        dna_seed, current_best_net, current_best_actions = find_best_seed_for_window(prices_window, num_seeds, max_workers)
        if manual_seed is not None:
            dna_seed = manual_seed  # ใช้ manual ถ้ามี
        # ... (ส่วนอื่นเหมือนเดิม)
    return original_actions_full, final_actions, pd.DataFrame(window_details_list)

# ==============================================================================
# 3. Batch Simulation Function (ใหม่ สำหรับ goal_1)
# ==============================================================================
def run_batch_simulation(config: Dict[str, Any], assets: List[str]) -> Dict[str, Dict[str, Any]]:
    batch_results = {}
    total_assets = len(assets)
    progress_bar = st.progress(0, text="Initializing Batch Processing...")
    
    # Parallel data fetching เพื่อเร็วขึ้น
    with ThreadPoolExecutor(max_workers=st.session_state.max_workers) as executor:
        futures = {}
        for idx, ticker in enumerate(assets):
            futures[executor.submit(get_ticker_data, ticker, str(st.session_state.start_date), str(st.session_state.end_date))] = ticker
        
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                ticker_data = future.result()
                if ticker_data.empty:
                    continue
                
                # Get manual seed ถ้ามี (จาก config)
                manual_seeds = config.get('manual_seed_by_asset', {}).get(ticker, [])
                manual_seed = manual_seeds[0].get('seed') if manual_seeds else None
                
                original_actions, final_actions, df_windows = generate_actions_hybrid_multi_mutation(
                    ticker_data, st.session_state.window_size, st.session_state.num_seeds,
                    st.session_state.max_workers, st.session_state.mutation_rate,
                    st.session_state.num_mutations, manual_seed=manual_seed
                )
                
                prices = ticker_data['Close'].to_numpy().tolist()
                results = {
                    Strategy.HYBRID_MULTI_MUTATION: run_simulation(prices, final_actions.tolist()),
                    Strategy.ORIGINAL_DNA: run_simulation(prices, original_actions.tolist()),
                    Strategy.REBALANCE_DAILY: run_simulation(prices, generate_actions_rebalance_daily(len(prices)).tolist()),
                    Strategy.PERFECT_FORESIGHT: run_simulation(prices, generate_actions_perfect_foresight(prices).tolist())
                }
                for name, df in results.items():
                    if not df.empty:
                        df.index = ticker_data.index[:len(df)]
                
                batch_results[ticker] = {'results': results, 'df_windows': df_windows, 'ticker_data': ticker_data}
                
                progress_bar.progress((len(batch_results) / total_assets), text=f"Processed {ticker} ({len(batch_results)}/{total_assets})")
            except Exception as e:
                st.warning(f"Skipping {ticker} due to error: {e}")
    
    progress_bar.empty()
    return batch_results

# ==============================================================================
# 4. UI Rendering Functions (ปรับให้รองรับ batch)
# ==============================================================================
# (render_settings_tab, display_comparison_charts, render_tracer_tab เหมือนเดิม)

def render_hybrid_multi_mutation_tab(config: Dict[str, Any]):
    st.write("---")
    st.markdown(f"### 🧬 {Strategy.HYBRID_MULTI_MUTATION}")
    st.info("กลยุทธ์นี้ทำงานโดย: 1. ค้นหา 'DNA' ที่ดีที่สุดในแต่ละ Window 2. นำ DNA นั้นมาพยายาม 'กลายพันธุ์' (Mutate) ซ้ำๆ เพื่อหาผลลัพธ์ที่ดีกว่าเดิม")
    # (expander และ code ตัวอย่างเหมือนเดิม)

    if st.button(f"🚀 Start Hybrid Multi-Mutation", type="primary"):
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("❌ กรุณาตั้งค่าช่วงวันที่ให้ถูกต้อง")
            return
        assets = config['assets']  # Loop all assets สำหรับ goal_1
        with st.spinner("กำลังประมวลผลสำหรับ all assets..."):
            batch_results = run_batch_simulation(config, assets)
            st.session_state.batch_results = batch_results
            st.success("Batch processing เสร็จสมบูรณ์!")

    # Display results (รองรับ batch แต่ UI เหมือนเดิม โดยใช้ expander)
    if 'batch_results' in st.session_state:
        batch_results = st.session_state.batch_results
        st.write("### 📈 สรุปผลลัพธ์สำหรับ All Assets")
        summary_data = []
        for ticker, data in batch_results.items():
            results = data['results']
            total_perfect_net = results[Strategy.PERFECT_FORESIGHT]['net'].iloc[-1] if not results[Strategy.PERFECT_FORESIGHT].empty else 0.0
            total_hybrid_net = results[Strategy.HYBRID_MULTI_MUTATION]['net'].iloc[-1] if not results[Strategy.HYBRID_MULTI_MUTATION].empty else 0.0
            # (เพิ่ม metrics อื่นๆ)
            summary_data.append({'Ticker': ticker, 'Perfect Net': total_perfect_net, 'Hybrid Net': total_hybrid_net})
        st.dataframe(pd.DataFrame(summary_data))
        
        for ticker, data in batch_results.items():
            with st.expander(f"Details for {ticker}"):
                display_comparison_charts(data['results'])
                st.dataframe(data['df_windows'])
                # (download button เหมือนเดิม แต่ per ticker)

# ==============================================================================
# 5. Main Application (เหมือนเดิม)
# ==============================================================================
def main():
    config = load_config()  # Load full config
    initialize_session_state(config)
    # (tabs และ render functions เหมือนเดิม แต่ส่ง config ไปยัง render_hybrid_multi_mutation_tab(config))

if __name__ == "__main__":
    main()
