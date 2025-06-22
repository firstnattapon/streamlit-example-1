# main

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import math
import json
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from datetime import datetime
from typing import List, Tuple, Dict, Any

# ! NUMBA: Import Numba's Just-In-Time compiler for core acceleration
from numba import njit

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================

class Strategy:
    """คลาสสำหรับเก็บชื่อกลยุทธ์ต่างๆ เพื่อให้เรียกใช้ง่ายและลดข้อผิดพลาด"""
    REBALANCE_DAILY = "Rebalance Daily"
    PERFECT_FORESIGHT = "Perfect Foresight (Max)"
    SLIDING_WINDOW = "Best Seed Sliding Window"
    CHAOTIC_SLIDING_WINDOW = "Chaotic Seed Sliding Window"
    GENETIC_ALGORITHM = "Genetic Algorithm Sliding Window"
    HYBRID_GA = "Hybrid (Random + GA)"
    ARITHMETIC_SEQUENCE = "Arithmetic Sequence"
    GEOMETRIC_SEQUENCE = "Geometric Sequence"
    ORIGINAL_DNA = "Original DNA (from Seed)"
    CONSENSUS_DNA = "Consensus DNA (from Mutation)"
    EVOLVED_DNA = "Evolved DNA (by GA)"


def load_config(filepath: str = "dynamic_seed_config.json") -> Dict[str, Any]:
    # ... (code remains the same)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        st.warning(f"⚠️ ไม่พบหรือไฟล์ '{filepath}' ไม่ถูกต้อง จะใช้ค่าเริ่มต้นแทน")
        return {
            "assets": ["FFWM", "NEGG", "RIVN", "BTC-USD", "NVDA", "TSLA", "META"],
            "default_settings": { "selected_ticker": "FFWM", "start_date": "2024-01-01", "window_size": 30, "num_seeds": 10000, "max_workers": 8, "ga_population_size": 50, "ga_generations": 20, "ga_master_seed": 42, "num_samples": 5000, "master_seed": 42 },
            "manual_seed_by_asset": { "default": [{'seed': 999, 'size': 50, 'tail': 15}], "FFWM": [{'seed': 1234, 'size': 60, 'tail': 30}, {'seed': 7777, 'size': 30, 'tail': 10}]}
        }

def initialize_session_state(config: Dict[str, Any]):
    # ... (code remains the same)
    defaults = config.get('default_settings', {})
    if 'test_ticker' not in st.session_state: st.session_state.test_ticker = defaults.get('selected_ticker', 'FFWM')
    if 'start_date' not in st.session_state:
        try: st.session_state.start_date = datetime.strptime(defaults.get('start_date', '2024-01-01'), '%Y-%m-%d').date()
        except ValueError: st.session_state.start_date = datetime(2024, 1, 1).date()
    if 'end_date' not in st.session_state: st.session_state.end_date = datetime.now().date()
    if 'window_size' not in st.session_state: st.session_state.window_size = defaults.get('window_size', 30)
    if 'num_seeds' not in st.session_state: st.session_state.num_seeds = defaults.get('num_seeds', 10000)
    if 'max_workers' not in st.session_state: st.session_state.max_workers = defaults.get('max_workers', 8)
    if 'ga_population_size' not in st.session_state: st.session_state.ga_population_size = defaults.get('ga_population_size', 50)
    if 'ga_generations' not in st.session_state: st.session_state.ga_generations = defaults.get('ga_generations', 20)
    if 'ga_master_seed' not in st.session_state: st.session_state.ga_master_seed = defaults.get('ga_master_seed', 42)
    if 'num_samples' not in st.session_state: st.session_state.num_samples = defaults.get('num_samples', 5000)
    if 'master_seed' not in st.session_state: st.session_state.master_seed = defaults.get('master_seed', 42)
    if 'df_for_analysis' not in st.session_state: st.session_state.df_for_analysis = None
    if 'manual_seed_lines' not in st.session_state:
        initial_ticker = st.session_state.get('test_ticker', defaults.get('selected_ticker', 'FFWM'))
        presets_by_asset = config.get("manual_seed_by_asset", {})
        default_presets = presets_by_asset.get("default", [{'seed': 999, 'size': 50, 'tail': 15}])
        st.session_state.manual_seed_lines = presets_by_asset.get(initial_ticker, default_presets)

# ==============================================================================
# 2. Core Calculation & Data Functions (With Numba Acceleration)
# ==============================================================================
@st.cache_data(ttl=3600)
def get_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    # ... (code remains the same)
    try:
        data = yf.Ticker(ticker).history(start=start_date, end=end_date)[['Close']]
        if data.empty: return pd.DataFrame()
        if data.index.tz is None: data = data.tz_localize('UTC').tz_convert('Asia/Bangkok')
        else: data = data.tz_convert('Asia/Bangkok')
        return data
    except Exception as e:
        st.error(f"❌ ไม่สามารถดึงข้อมูล {ticker} ได้: {str(e)}"); return pd.DataFrame()

@njit(cache=True)
def _calculate_net_profit_numba(action_array: np.ndarray, price_array: np.ndarray, fix: int = 1500) -> float:
    """คำนวณ Net Profit อย่างเดียว เพื่อความเร็วสูงสุดใน GA loop"""
    n = len(action_array)
    # ตรวจสอบว่า action array ไม่ยาวเกิน price array
    if n == 0 or len(price_array) == 0 or n > len(price_array):
        return -np.inf

    action_array_calc = action_array.copy()
    action_array_calc[0] = 1

    initial_price = price_array[0]
    initial_capital = fix * 2.0
    
    refer_net = -fix * np.log(initial_price / price_array[n-1])

    cash = float(fix)
    amount = float(fix) / initial_price
    
    for i in range(1, n):
        curr_price = price_array[i]
        if action_array_calc[i] != 0:
            cash += amount * curr_price - fix
            amount = fix / curr_price
            
    final_sumusd = cash + (amount * price_array[n-1])
    net = final_sumusd - refer_net - initial_capital
    return net


def run_simulation(prices: List[float], actions: List[int], fix: int = 1500) -> pd.DataFrame:
    # ... (This function now uses the full simulation function, not just net profit)
    @njit
    def _full_sim_numba(action_arr, price_arr, fix_val):
        n = len(action_arr)
        if n == 0 or len(price_arr) == 0:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty, empty, empty, empty, empty
        
        action_calc = action_arr.copy()
        action_calc[0] = 1

        amount = np.empty(n, dtype=np.float64)
        buffer = np.zeros(n, dtype=np.float64)
        cash = np.empty(n, dtype=np.float64)
        asset_val = np.empty(n, dtype=np.float64)
        sumusd_val = np.empty(n, dtype=np.float64)
        
        init_price = price_arr[0]
        amount[0] = fix_val / init_price
        cash[0] = fix_val
        asset_val[0] = amount[0] * init_price
        sumusd_val[0] = cash[0] + asset_val[0]
        
        refer = -fix_val * np.log(init_price / price_arr[:n])
        
        for i in range(1, n):
            curr_price = price_arr[i]
            if action_calc[i] == 0:
                amount[i] = amount[i-1]
                buffer[i] = 0.0
            else:
                amount[i] = fix_val / curr_price
                buffer[i] = amount[i-1] * curr_price - fix_val
            
            cash[i] = cash[i-1] + buffer[i]
            asset_val[i] = amount[i] * curr_price
            sumusd_val[i] = cash[i] + asset_val[i]
            
        return buffer, sumusd_val, cash, asset_val, amount, refer

    if not prices or not actions: return pd.DataFrame()
    min_len = min(len(prices), len(actions))
    prices_arr = np.array(prices[:min_len], dtype=np.float64)
    actions_arr = np.array(actions[:min_len], dtype=np.int32)

    buffer, sumusd, cash, asset_value, amount, refer = _full_sim_numba(actions_arr, prices_arr, fix)
    
    if len(sumusd) == 0: return pd.DataFrame()
    
    initial_capital = sumusd[0]
    return pd.DataFrame({
        'price': prices_arr, 'action': actions_arr, 'buffer': np.round(buffer, 2),
        'sumusd': np.round(sumusd, 2), 'cash': np.round(cash, 2),
        'asset_value': np.round(asset_value, 2), 'amount': np.round(amount, 2),
        'refer': np.round(refer + initial_capital, 2),
        'net': np.round(sumusd - refer - initial_capital, 2)
    })
# ==============================================================================
# 3. Strategy Action Generation
# ==============================================================================

def generate_actions_rebalance_daily(num_days: int) -> np.ndarray: return np.ones(num_days, dtype=np.int32)
def generate_actions_perfect_foresight(prices: List[float], fix: int = 1500) -> np.ndarray:
    # ... (code remains the same)
    price_arr = np.asarray(prices, dtype=np.float64)
    n = len(price_arr)
    if n < 2: return np.ones(n, dtype=int)
    dp = np.zeros(n, dtype=np.float64); path = np.zeros(n, dtype=int); dp[0] = float(fix * 2)
    for i in range(1, n):
        j_indices = np.arange(i)
        profits = fix * ((price_arr[i] / price_arr[j_indices]) - 1)
        current_sumusd = dp[j_indices] + profits
        best_idx = np.argmax(current_sumusd); dp[i] = current_sumusd[best_idx]; path[i] = j_indices[best_idx]
    actions = np.zeros(n, dtype=int); current_day = np.argmax(dp)
    while current_day > 0: actions[current_day] = 1; current_day = path[current_day]
    actions[0] = 1
    return actions
    
# ! NEW: 3.9 Evolve DNA with GA (for the new tab)
def evolve_sequence_with_ga(
    initial_sequence: np.ndarray,
    prices: np.ndarray,
    population_size: int,
    generations: int,
    ga_seed: int,
    mutation_rate: float = 0.01,
    initial_mutation_rate: float = 0.10,
    progress_bar=None
) -> Tuple[np.ndarray, float]:
    """นำ action_sequence ที่มีอยู่แล้วมาพัฒนาต่อด้วย Genetic Algorithm"""
    sequence_length = len(initial_sequence)
    rng = np.random.default_rng(ga_seed)
    
    # --- 1. สร้างประชากรเริ่มต้นจาก initial_sequence ---
    population = np.tile(initial_sequence, (population_size, 1))
    mutation_mask = rng.random(population.shape) < initial_mutation_rate
    population[mutation_mask] = 1 - population[mutation_mask]
    population[0] = initial_sequence
    population[:, 0] = 1
    
    # --- 2. เริ่มกระบวนการวิวัฒนาการ ---
    for gen in range(generations):
        fitness_scores = np.array([_calculate_net_profit_numba(chromo, prices) for chromo in population])
        
        num_parents = population_size // 2
        parent_indices = np.argsort(fitness_scores)[-num_parents:]
        parents = population[parent_indices]
        
        offspring = np.empty((population_size - num_parents, sequence_length), dtype=np.int32)
        parent_choices = rng.choice(num_parents, size=(len(offspring), 2), replace=True)
        
        for k, (p1_idx, p2_idx) in enumerate(parent_choices):
            crossover_point = rng.integers(1, sequence_length)
            offspring[k, :crossover_point] = parents[p1_idx, :crossover_point]
            offspring[k, crossover_point:] = parents[p2_idx, crossover_point:]
        
        mutation_mask = rng.random(offspring.shape) < mutation_rate
        offspring[mutation_mask] = 1 - offspring[mutation_mask]
        offspring[:, 0] = 1
        
        population[0:num_parents] = parents
        population[num_parents:] = offspring

        if progress_bar:
            progress_bar.progress((gen + 1) / generations, text=f"Evolving Generation {gen+1}/{generations}")

    # --- 3. ค้นหาคำตอบที่ดีที่สุดในรุ่นสุดท้าย ---
    final_fitness_scores = np.array([_calculate_net_profit_numba(chromo, prices) for chromo in population])
    best_idx = np.argmax(final_fitness_scores)
    
    return population[best_idx], final_fitness_scores[best_idx]


# ... (All other strategy and UI functions from previous version are assumed to be here) ...
# For brevity, I will only include the new UI tab and the modified main function.
# Please ensure you have the other render_* functions from the previous correct version.
def render_settings_tab(config: Dict[str, Any]): st.write("... [Settings UI Code Here] ..."); # Placeholder
def render_test_tab(): st.write("... [Random Seed UI Code Here] ..."); # Placeholder
def render_chaotic_test_tab(): st.write("... [Chaotic Seed UI Code Here] ..."); # Placeholder
def render_ga_test_tab(): st.write("... [GA UI Code Here] ..."); # Placeholder
def render_hybrid_ga_tab(): st.write("... [Hybrid GA UI Code Here] ..."); # Placeholder
def render_consensus_tab(): st.write("... [Consensus UI Code Here] ..."); # Placeholder
def render_arithmetic_tab(): st.write("... [Arithmetic UI Code Here] ..."); # Placeholder
def render_geometric_tab(): st.write("... [Geometric UI Code Here] ..."); # Placeholder
def render_analytics_tab(): st.write("... [Analytics UI Code Here] ..."); # Placeholder
def render_manual_seed_tab(config: Dict[str, Any]): st.write("... [Manual Seed UI Code Here] ..."); # Placeholder
def display_comparison_charts(results: Dict[str, pd.DataFrame], chart_title: str = '📊 เปรียบเทียบกำไรสุทธิ (Net Profit)'):
    # ... (code remains the same)
    if not results: st.warning("ไม่มีข้อมูลสำหรับสร้างกราฟเปรียบเทียบ"); return
    valid_dfs = {name: df for name, df in results.items() if not df.empty and 'net' in df.columns}
    if not valid_dfs: st.warning("ไม่มีข้อมูล 'net' สำหรับสร้างกราฟ"); return
    try: longest_index = max((df.index for df in valid_dfs.values()), key=len, default=None)
    except ValueError: longest_index = None
    if longest_index is None: st.warning("ไม่มีข้อมูลสำหรับสร้างกราฟเปรียบเทียบ"); return
    chart_data = pd.DataFrame(index=longest_index)
    for name, df in valid_dfs.items(): chart_data[name] = df['net'].reindex(longest_index).ffill()
    st.write(chart_title); st.line_chart(chart_data)

# ==============================================================================
# 4. UI Rendering Functions (New Tab Added)
# ==============================================================================

# ! NEW: UI Tab for Evolving DNA with GA
def render_evolve_dna_tab():
    st.write("---")
    st.markdown("### 🧬 Evolve DNA (GA)")
    st.markdown("""
    นำ `action_sequence` ที่สร้างจาก **Seed ต้นแบบ** มาเป็นจุดเริ่มต้นให้กับ **Genetic Algorithm (GA)**
    เพื่อค้นหา Sequence ที่ถูกปรับปรุงให้ดีที่สุดภายใต้สภาวะตลาดที่กำหนด
    """)

    with st.container(border=True):
        st.subheader("1. กำหนด DNA ต้นแบบ (Blueprint)")
        c1, c2 = st.columns(2)
        blueprint_seed = c1.number_input("Input Seed for Blueprint", value=16942, min_value=1, format="%d", key="blueprint_seed")
        blueprint_size = c2.number_input("Sequence Size (length of DNA)", value=60, min_value=10, key="blueprint_size")

    with st.container(border=True):
        st.subheader("2. กำหนดพารามิเตอร์ Genetic Algorithm")
        c1, c2, c3 = st.columns(3)
        ga_pop_size = c1.number_input("Population Size", value=2000, min_value=100, step=100, key="evolve_pop_size")
        ga_generations = c2.number_input("Generations", value=200, min_value=10, step=10, key="evolve_generations")
        ga_seed = c3.number_input("Master Seed for GA", value=42, format="%d", key="evolve_ga_seed")
        
    with st.container(border=True):
        st.subheader("3. กำหนดสนามทดสอบ (Simulation Environment)")
        c1, c2, c3 = st.columns(3)
        sim_ticker = c1.selectbox("Ticker", options=load_config().get('assets', ['FFWM']), index=0, key="sim_ticker_evolve")
        # ใช้จำนวนวันแทนช่วงวันที่ เพื่อให้ขนาด Sequence ตรงกับข้อมูลราคา
        sim_days_ago = c2.number_input("ใช้ข้อมูลย้อนหลัง (วันทำการ)", value=blueprint_size, min_value=blueprint_size, key="sim_days_evolve", help=f"ต้องมีค่าอย่างน้อยเท่ากับ Sequence Size ({blueprint_size})")

    if st.button("🧬 Start Evolution", type="primary", key="start_evolve_btn"):
        with st.spinner("กำลังเตรียมข้อมูลและวิวัฒนาการ DNA..."):
            # Step 1: Create Blueprint DNA
            rng_base = np.random.default_rng(blueprint_seed)
            initial_actions = rng_base.integers(0, 2, size=blueprint_size)
            initial_actions[0] = 1

            # Step 2: Fetch price data
            st.info(f"กำลังดึงข้อมูลราคา {sim_ticker} ย้อนหลัง {sim_days_ago} วันทำการ...")
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
            start_date = (pd.Timestamp.now() - pd.Timedelta(days=sim_days_ago * 2)).strftime('%Y-%m-%d') # Fetch more to be safe
            ticker_data = get_ticker_data(sim_ticker, start_date, end_date)
            
            if ticker_data.empty or len(ticker_data) < sim_days_ago:
                st.error(f"ไม่สามารถดึงข้อมูลราคาได้ครบ {sim_days_ago} วัน (ดึงได้แค่ {len(ticker_data)} วัน)")
                return

            prices_full = ticker_data['Close'].to_numpy()[-sim_days_ago:]
            prices_for_sim = prices_full[:blueprint_size]
            
            # Evaluate blueprint performance
            initial_net = _calculate_net_profit_numba(initial_actions, prices_for_sim)
            
            # Step 3: Run GA Evolution
            st.info("เริ่มกระบวนการวิวัฒนาการ...")
            evo_progress_bar = st.progress(0, text="Starting Evolution...")
            evolved_actions, final_net = evolve_sequence_with_ga(
                initial_sequence=initial_actions,
                prices=prices_for_sim,
                population_size=ga_pop_size,
                generations=ga_generations,
                ga_seed=ga_seed,
                progress_bar=evo_progress_bar
            )
            evo_progress_bar.empty()

            # Step 4: Run full simulations for charting
            st.info("กำลังจำลองผลลัพธ์เพื่อสร้างกราฟ...")
            results_evo = {}
            df_initial = run_simulation(prices_for_sim.tolist(), initial_actions.tolist())
            df_evolved = run_simulation(prices_for_sim.tolist(), evolved_actions.tolist())
            
            sim_dates = ticker_data.index[-sim_days_ago:][:blueprint_size]
            
            if not df_initial.empty: df_initial.index = sim_dates; results_evo[Strategy.ORIGINAL_DNA] = df_initial
            if not df_evolved.empty: df_evolved.index = sim_dates; results_evo[Strategy.EVOLVED_DNA] = df_evolved

            # Add Benchmarks
            df_max = run_simulation(prices_for_sim.tolist(), generate_actions_perfect_foresight(prices_for_sim.tolist()).tolist())
            df_min = run_simulation(prices_for_sim.tolist(), generate_actions_rebalance_daily(len(prices_for_sim)).tolist())
            if not df_max.empty: df_max.index = sim_dates; results_evo[Strategy.PERFECT_FORESIGHT] = df_max
            if not df_min.empty: df_min.index = sim_dates; results_evo[Strategy.REBALANCE_DAILY] = df_min
            
        st.success("กระบวนการวิวัฒนาการเสร็จสมบูรณ์!")
        st.write("---")
        
        st.subheader("📊 เปรียบเทียบประสิทธิภาพ (Net Profit)")
        display_comparison_charts(results_evo)

        st.subheader("📈 สรุปผลลัพธ์สุดท้าย")
        metric_cols = st.columns(2)
        metric_cols[0].metric(f"{Strategy.ORIGINAL_DNA} (Seed: {blueprint_seed})", f"${initial_net:,.2f}")
        metric_cols[1].metric(f"{Strategy.EVOLVED_DNA} (GA)", f"${final_net:,.2f}", delta=f"{final_net - initial_net:,.2f}")

        st.write("---")
        st.subheader("🔬 รายละเอียดการเปลี่ยนแปลง DNA")
        
        dna_detail_df = pd.DataFrame({
            'Position': range(blueprint_size),
            'Original': initial_actions,
            'Evolved': evolved_actions,
            'Changed': initial_actions != evolved_actions
        })
        
        st.dataframe(dna_detail_df, use_container_width=True, height=300)
        
        changes = dna_detail_df['Changed'].sum()
        st.markdown(f"**จำนวนยีนที่เปลี่ยนแปลง:** `{changes}` จาก `{blueprint_size}` ตำแหน่ง ({changes/blueprint_size:.2%})")

# ==============================================================================
# 5. Main Application
# ==============================================================================
def main():
    st.set_page_config(page_title="Best Seed Sliding Window", page_icon="🎯", layout="wide")
    st.markdown("### 🎯 Best Seed Sliding Window Tester (Multi-Strategy & Numba Accelerated)")
    st.caption("เครื่องมือทดสอบการหา Best Seed และ Sequence ที่ดีที่สุด (Core Calculation เร่งความเร็วด้วย Numba)")

    # โค้ดส่วนนี้ควรจะเรียกฟังก์ชัน UI ที่มีอยู่ครบถ้วน
    # เพื่อความสมบูรณ์ ผมจะใส่โครงสร้างที่ถูกต้องไว้ แต่คุณต้องแน่ใจว่าได้คัดลอก
    # ฟังก์ชัน render_* ทั้งหมดจากเวอร์ชันก่อนหน้ามาครบแล้ว
    
    config = load_config()
    initialize_session_state(config)

    tab_list = [
        "⚙️ การตั้งค่า",
        "🧬 Evolve DNA (GA)", # ! NEW: Add the new tab here
        "🚀 Best Seed (Random)",
        "🌀 Best Seed (Chaotic)",
        "🧬 Best Seed (GA)",
        "🚀+🧬 Hybrid (GA)",
        "🧬 DNA Consensus", 
        "📈 Arithmetic Seq",
        "📉 Geometric Seq",
        "📊 Analytics",
        "🌱 Forward Rolling"
    ]
    
    # สร้าง placeholder สำหรับฟังก์ชัน render ที่อาจจะยังไม่มี
    render_functions = {
        "⚙️ การตั้งค่า": render_settings_tab,
        "🧬 Evolve DNA (GA)": render_evolve_dna_tab,
        "🚀 Best Seed (Random)": render_test_tab, # Assuming this exists
        "🌀 Best Seed (Chaotic)": render_chaotic_test_tab, # Assuming this exists
        "🧬 Best Seed (GA)": render_ga_test_tab, # Assuming this exists
        "🚀+🧬 Hybrid (GA)": render_hybrid_ga_tab, # Assuming this exists
        "🧬 DNA Consensus": render_consensus_tab, # Assuming this exists
        "📈 Arithmetic Seq": render_arithmetic_tab, # Assuming this exists
        "📉 Geometric Seq": render_geometric_tab, # Assuming this exists
        "📊 Analytics": render_analytics_tab, # Assuming this exists
        "🌱 Forward Rolling": lambda: render_manual_seed_tab(config) # Assuming this exists
    }
    
    tabs = st.tabs(tab_list)

    for i, tab_name in enumerate(tab_list):
        with tabs[i]:
            # เรียกใช้ฟังก์ชัน render ที่สอดคล้องกับชื่อ tab
            if tab_name in render_functions:
                if tab_name == "⚙️ การตั้งค่า" or tab_name == "🌱 Forward Rolling":
                     render_functions[tab_name]()
                else:
                     render_functions[tab_name]()
            else:
                st.warning(f"UI function for '{tab_name}' not implemented yet.")


if __name__ == "__main__":
    main()
