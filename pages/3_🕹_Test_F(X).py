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
    # ! NEW: Strategy names for the new tab
    ORIGINAL_DNA = "Original DNA (from Seed)"
    CONSENSUS_DNA = "Consensus DNA (from Mutation)"


def load_config(filepath: str = "dynamic_seed_config.json") -> Dict[str, Any]:
    """
    โหลดการตั้งค่าจากไฟล์ JSON
    หากไฟล์ไม่พบหรือมีข้อผิดพลาด จะคืนค่า default เพื่อให้โปรแกรมทำงานต่อได้
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        st.warning(f"⚠️ ไม่พบหรือไฟล์ '{filepath}' ไม่ถูกต้อง จะใช้ค่าเริ่มต้นแทน")
        return {
            "assets": ["FFWM", "NEGG", "RIVN", "BTC-USD", "NVDA", "TSLA", "META"],
            "default_settings": {
                "selected_ticker": "FFWM", "start_date": "2024-01-01", "window_size": 30,
                "num_seeds": 10000, "max_workers": 8,
                "ga_population_size": 50, "ga_generations": 20, "ga_master_seed": 42,
                "num_samples": 5000, "master_seed": 42
            },
            "manual_seed_by_asset": {
                "default": [{'seed': 999, 'size': 50, 'tail': 15}],
                "FFWM": [{'seed': 1234, 'size': 60, 'tail': 30}, {'seed': 7777, 'size': 30, 'tail': 10}]
            }
        }

def on_ticker_change_callback(config: Dict[str, Any]):
    """
    Callback ที่จะถูกเรียกเมื่อ Ticker ใน Tab Manual Seed เปลี่ยน
    เพื่อโหลดค่า Preset ของ Seed/Size/Tail ให้ตรงกับ Ticker ที่เลือก
    """
    selected_ticker = st.session_state.get("manual_ticker_key")
    if not selected_ticker:
        return
    presets_by_asset = config.get("manual_seed_by_asset", {})
    default_presets = presets_by_asset.get("default", [{'seed': 999, 'size': 50, 'tail': 15}])
    st.session_state.manual_seed_lines = presets_by_asset.get(selected_ticker, default_presets)

def initialize_session_state(config: Dict[str, Any]):
    """
    ตั้งค่าเริ่มต้นสำหรับ Streamlit session state โดยใช้ค่าจาก config
    เพื่อให้ค่าต่างๆ ยังคงอยู่เมื่อผู้ใช้เปลี่ยนหน้าหรือทำ Action อื่นๆ
    """
    defaults = config.get('default_settings', {})

    if 'test_ticker' not in st.session_state:
        st.session_state.test_ticker = defaults.get('selected_ticker', 'FFWM')
    if 'start_date' not in st.session_state:
        try:
            st.session_state.start_date = datetime.strptime(defaults.get('start_date', '2024-01-01'), '%Y-%m-%d').date()
        except ValueError:
            st.session_state.start_date = datetime(2024, 1, 1).date()
    if 'end_date' not in st.session_state:
        st.session_state.end_date = datetime.now().date()
    if 'window_size' not in st.session_state:
        st.session_state.window_size = defaults.get('window_size', 30)
    if 'num_seeds' not in st.session_state:
        st.session_state.num_seeds = defaults.get('num_seeds', 10000)
    if 'max_workers' not in st.session_state:
        st.session_state.max_workers = defaults.get('max_workers', 8)

    if 'ga_population_size' not in st.session_state:
        st.session_state.ga_population_size = defaults.get('ga_population_size', 50)
    if 'ga_generations' not in st.session_state:
        st.session_state.ga_generations = defaults.get('ga_generations', 20)
    if 'ga_master_seed' not in st.session_state:
        st.session_state.ga_master_seed = defaults.get('ga_master_seed', 42)

    if 'num_samples' not in st.session_state:
        st.session_state.num_samples = defaults.get('num_samples', 5000)
    if 'master_seed' not in st.session_state:
        st.session_state.master_seed = defaults.get('master_seed', 42)

    if 'df_for_analysis' not in st.session_state:
        st.session_state.df_for_analysis = None

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
    """
    ดึงข้อมูลราคาหุ้น/สินทรัพย์จาก Yahoo Finance และ Cache ผลลัพธ์ไว้ 1 ชั่วโมง
    """
    try:
        data = yf.Ticker(ticker).history(start=start_date, end=end_date)[['Close']]
        if data.empty: return pd.DataFrame()
        if data.index.tz is None:
            data = data.tz_localize('UTC').tz_convert('Asia/Bangkok')
        else:
            data = data.tz_convert('Asia/Bangkok')
        return data
    except Exception as e:
        st.error(f"❌ ไม่สามารถดึงข้อมูล {ticker} ได้: {str(e)}")
        return pd.DataFrame()

@njit(cache=True)
def _calculate_simulation_numba(action_array: np.ndarray, price_array: np.ndarray, fix: int = 1500) -> Tuple:
    """
    คำนวณผลลัพธ์การจำลองการเทรด (หัวใจของการคำนวณ)
    - ใช้ Numba @njit(cache=True) เพื่อคอมไพล์เป็น Machine Code ทำให้ทำงานเร็วมาก
    """
    n = len(action_array)
    if n == 0 or len(price_array) == 0:
        empty_arr = np.empty(0, dtype=np.float64)
        return (empty_arr, empty_arr, empty_arr, empty_arr, empty_arr, empty_arr)

    action_array_calc = action_array.copy()
    if n > 0: action_array_calc[0] = 1

    amount = np.empty(n, dtype=np.float64)
    buffer = np.zeros(n, dtype=np.float64)
    cash = np.empty(n, dtype=np.float64)
    asset_value = np.empty(n, dtype=np.float64)
    sumusd = np.empty(n, dtype=np.float64)

    initial_price = price_array[0]
    amount[0] = fix / initial_price
    cash[0] = fix
    asset_value[0] = amount[0] * initial_price
    sumusd[0] = cash[0] + asset_value[0]

    refer = -fix * np.log(initial_price / price_array)

    for i in range(1, n):
        curr_price = price_array[i]
        if action_array_calc[i] == 0:
            amount[i] = amount[i-1]
            buffer[i] = 0.0
        else:
            amount[i] = fix / curr_price
            buffer[i] = amount[i-1] * curr_price - fix

        cash[i] = cash[i-1] + buffer[i]
        asset_value[i] = amount[i] * curr_price
        sumusd[i] = cash[i] + asset_value[i]

    return buffer, sumusd, cash, asset_value, amount, refer

@lru_cache(maxsize=16384)
def calculate_optimized_cached(action_tuple: Tuple[int, ...], price_tuple: Tuple[float, ...], fix: int = 1500) -> Tuple:
    """
    Wrapper function สำหรับเรียกฟังก์ชัน Numba โดยใช้ Cache
    """
    action_array = np.asarray(action_tuple, dtype=np.int32)
    price_array = np.asarray(price_tuple, dtype=np.float64)
    return _calculate_simulation_numba(action_array, price_array, fix)

def run_simulation(prices: List[float], actions: List[int], fix: int = 1500) -> pd.DataFrame:
    """
    สร้าง DataFrame ผลลัพธ์จากการจำลองการเทรด
    """
    if not prices or not actions: return pd.DataFrame()
    min_len = min(len(prices), len(actions))
    prices = prices[:min_len]
    actions = actions[:min_len]
    
    buffer, sumusd, cash, asset_value, amount, refer = calculate_optimized_cached(tuple(actions), tuple(prices), fix)
    
    if len(sumusd) == 0: return pd.DataFrame()

    initial_capital = sumusd[0]
    return pd.DataFrame({
        'price': prices, 'action': actions, 'buffer': np.round(buffer, 2),
        'sumusd': np.round(sumusd, 2), 'cash': np.round(cash, 2),
        'asset_value': np.round(asset_value, 2), 'amount': np.round(amount, 2),
        'refer': np.round(refer + initial_capital, 2),
        'net': np.round(sumusd - refer - initial_capital, 2)
    })

@njit(cache=True)
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

# ==============================================================================
# 3. Strategy Action Generation
# ==============================================================================

# ... (All previous strategy functions remain the same) ...

# ! NEW: 3.8 DNA Consensus Strategy (Based on user request)
def find_consensus_dna(seed: int, sequence_size: int, population_size: int, mutation_rate: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ค้นหา "แก่นแท้ของ DNA" จาก Seed ที่กำหนด โดยใช้วิธี Mutation และ Consensus
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
        - base_sequence: ลำดับ Action ที่สร้างจาก Seed ดั้งเดิม
        - consensus_sequence: ลำดับ Action ที่ได้จากฉันทามติของประชากรที่กลายพันธุ์
        - confidence: ค่าความเชื่อมั่น (0-1) ของฉันทามติในแต่ละตำแหน่ง
    """
    rng = np.random.default_rng(seed)
    
    # 1. สร้าง DNA ต้นแบบ (Base DNA)
    base_sequence = rng.integers(0, 2, size=sequence_size, dtype=np.int32)
    base_sequence[0] = 1 # บังคับ Action แรก
    
    # 2. สร้างประชากรและทำให้กลายพันธุ์
    # สร้างประชากรโดยการคัดลอก base_sequence
    population = np.tile(base_sequence, (population_size, 1))
    
    # สร้าง Mask สำหรับการกลายพันธุ์
    mutation_mask = rng.random(population.shape) < mutation_rate
    
    # ทำการกลายพันธุ์ (flip 0 to 1 and 1 to 0)
    population[mutation_mask] = 1 - population[mutation_mask]
    
    # นำ DNA ต้นฉบับกลับเข้าไปในประชากร 1 ตัว เพื่อรับประกันว่ามันยังอยู่
    population[0] = base_sequence
    # บังคับ Action แรกของทุกคนในประชากรให้เป็น 1 เสมอ
    population[:, 0] = 1
    
    # 3. หาฉันทามติ (Find Consensus)
    # นับจำนวน '1' ในแต่ละคอลัมน์ (แต่ละตำแหน่งของยีน)
    consensus_counts_1 = np.sum(population, axis=0)
    
    # สร้าง Consensus DNA จากเสียงส่วนใหญ่
    consensus_sequence = (consensus_counts_1 > population_size / 2).astype(np.int32)
    
    # 4. คำนวณค่าความเชื่อมั่น
    # หาจำนวนเสียงข้างมากในแต่ละตำแหน่ง
    majority_votes = np.maximum(consensus_counts_1, population_size - consensus_counts_1)
    confidence = majority_votes / population_size
    
    return base_sequence, consensus_sequence, confidence

# ==============================================================================
# 4. UI Rendering Functions
# ==============================================================================

# ... (All previous UI render functions remain the same) ...

# ! NEW: UI Tab for DNA Consensus Finder
def render_consensus_tab():
    """แสดงผล UI สำหรับ Tab DNA Consensus Finder"""
    st.write("---")
    st.markdown(f"### 🧬 DNA Consensus Finder")
    st.markdown("""
    แนวคิดนี้คือการสกัด **"แก่นแท้ของ DNA"** จาก Seed ที่ให้ผลลัพธ์ดี
    1.  สร้าง Action Sequence (DNA) จาก Seed ที่กำหนด
    2.  นำ DNA นั้นไปสร้าง "ประชากร" แล้วทำให้เกิดการ **"กลายพันธุ์" (Mutation)**
    3.  ค้นหา **"ฉันทามติ" (Consensus)** หรือยีนที่แข็งแกร่งที่สุดจากประชากรทั้งหมด
    4.  เปรียบเทียบประสิทธิภาพระหว่าง **DNA ต้นฉบับ** และ **DNA จากฉันทามติ**
    """)

    with st.container(border=True):
        st.subheader("1. กำหนดพารามิเตอร์การค้นหา")
        c1, c2, c3, c4 = st.columns(4)
        dna_seed = c1.number_input("Input Seed", value=16942, min_value=1, format="%d")
        dna_size = c2.number_input("Sequence Size (Forward Rolling)", value=60, min_value=10)
        dna_pop_size = c3.number_input("Population Size", value=1001, min_value=101, step=100, help="ควรเป็นเลขคี่เพื่อหลีกเลี่ยงการโหวตที่เท่ากัน")
        dna_mutation_rate = c4.slider("Mutation Rate", min_value=0.0, max_value=0.5, value=0.05, step=0.01, help="โอกาสที่ยีนแต่ละตัวจะกลายพันธุ์")

    with st.container(border=True):
        st.subheader("2. กำหนดช่วงเวลาและ Ticker สำหรับจำลองผล")
        c1, c2, c3 = st.columns(3)
        sim_ticker = c1.selectbox("Ticker สำหรับจำลอง", options=load_config().get('assets', ['FFWM']), index=0, key="sim_ticker_consensus")
        sim_start_date = c2.date_input("วันที่เริ่มต้นจำลอง", value=st.session_state.start_date, key="sim_start_consensus")
        sim_end_date = c3.date_input("วันที่สิ้นสุดจำลอง", value=st.session_state.end_date, key="sim_end_consensus")
        
    if st.button("🔬 เริ่มการวิเคราะห์และเปรียบเทียบ DNA", type="primary", key="find_consensus_btn"):
        if sim_start_date >= sim_end_date:
            st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
            return
            
        with st.spinner("กำลังค้นหา Consensus DNA และจำลองการเทรด..."):
            # Step 1: Find the consensus DNA
            base_actions, consensus_actions, confidence = find_consensus_dna(
                dna_seed, dna_size, dna_pop_size, dna_mutation_rate
            )
            
            # Step 2: Fetch price data
            ticker_data = get_ticker_data(sim_ticker, str(sim_start_date), str(sim_end_date))
            if ticker_data.empty:
                st.error(f"ไม่พบข้อมูลสำหรับ {sim_ticker} ในช่วงวันที่ที่เลือก")
                return

            prices = ticker_data['Close'].to_list()
            
            # Step 3: Run simulations
            results_dna = {}
            df_base = run_simulation(prices, base_actions.tolist())
            df_consensus = run_simulation(prices, consensus_actions.tolist())
            
            if not df_base.empty:
                df_base.index = ticker_data.index[:len(df_base)]
                results_dna[Strategy.ORIGINAL_DNA] = df_base
            
            if not df_consensus.empty:
                df_consensus.index = ticker_data.index[:len(df_consensus)]
                results_dna[Strategy.CONSENSUS_DNA] = df_consensus
            
            # Add Benchmarks
            if not ticker_data.empty:
                df_max = run_simulation(prices, generate_actions_perfect_foresight(prices).tolist())
                df_min = run_simulation(prices, generate_actions_rebalance_daily(len(prices)).tolist())
                if not df_max.empty:
                    df_max.index = ticker_data.index[:len(df_max)]
                    results_dna[Strategy.PERFECT_FORESIGHT] = df_max
                if not df_min.empty:
                    df_min.index = ticker_data.index[:len(df_min)]
                    results_dna[Strategy.REBALANCE_DAILY] = df_min

        st.success("การวิเคราะห์เสร็จสมบูรณ์!")
        st.write("---")
        
        # Display results
        st.subheader("📊 เปรียบเทียบประสิทธิภาพ (Net Profit)")
        display_comparison_charts(results_dna)
        
        st.subheader("📈 สรุปผลลัพธ์สุดท้าย (Final Net Profit)")
        final_net_base = results_dna.get(Strategy.ORIGINAL_DNA, pd.DataFrame({'net': [0]}))['net'].iloc[-1]
        final_net_consensus = results_dna.get(Strategy.CONSENSUS_DNA, pd.DataFrame({'net': [0]}))['net'].iloc[-1]

        metric_cols = st.columns(2)
        metric_cols[0].metric(Strategy.ORIGINAL_DNA, f"${final_net_base:,.2f}")
        metric_cols[1].metric(Strategy.CONSENSUS_DNA, f"${final_net_consensus:,.2f}", delta=f"{final_net_consensus - final_net_base:,.2f}")

        st.write("---")
        st.subheader("🔬 รายละเอียด DNA")
        
        # Create a DataFrame for detailed comparison
        dna_detail_df = pd.DataFrame({
            'Position': range(dna_size),
            'Original': base_actions,
            'Consensus': consensus_actions,
            'Confidence': [f"{c:.1%}" for c in confidence],
            'Changed': base_actions != consensus_actions
        })
        
        st.dataframe(dna_detail_df, use_container_width=True)
        
        st.markdown(f"**จำนวนยีนที่เปลี่ยนแปลง:** `{dna_detail_df['Changed'].sum()}` จาก `{dna_size}` ตำแหน่ง")

# ==============================================================================
# 5. Main Application
# ==============================================================================
def main():
    st.set_page_config(page_title="Best Seed Sliding Window", page_icon="🎯", layout="wide")
    st.markdown("### 🎯 Best Seed Sliding Window Tester (Multi-Strategy & Numba Accelerated)")
    st.caption("เครื่องมือทดสอบการหา Best Seed และ Sequence ที่ดีที่สุด (Core Calculation เร่งความเร็วด้วย Numba)")

    config = load_config(); initialize_session_state(config)

    tab_list = [
        "⚙️ การตั้งค่า",
        "🚀 Best Seed (Random)",
        "🌀 Best Seed (Chaotic)",
        "🧬 Best Seed (Genetic Algo)",
        "🚀+🧬 Hybrid (Random + GA)",
        "🧬 DNA Consensus Finder", # ! NEW TAB
        "📈 Arithmetic Seq",
        "📉 Geometric Seq",
        "📊 Advanced Analytics",
        "🌱 Forward Rolling Comparator"
    ]
    tabs = st.tabs(tab_list)

    with tabs[0]: render_settings_tab(config)
    with tabs[1]: render_test_tab()
    with tabs[2]: render_chaotic_test_tab()
    with tabs[3]: render_ga_test_tab()
    with tabs[4]: render_hybrid_ga_tab()
    with tabs[5]: render_consensus_tab() # ! RENDER NEW TAB
    with tabs[6]: render_arithmetic_tab()
    with tabs[7]: render_geometric_tab()
    with tabs[8]: render_analytics_tab()
    with tabs[9]: render_manual_seed_tab(config)

    with st.expander("📖 คำอธิบายวิธีการทำงานและแนวคิด (v.Consensus)"):
        st.markdown("""
        ### หลักการทำงานของเวอร์ชันนี้:

        **เป้าหมาย: คง Logic เดิม, เพิ่มความเร็วสูงสุด, และเพิ่มโมเดลใหม่ๆ**

        1.  **🚀+🧬 โมเดลผสม (Hybrid Strategy)**:
            - **Phase 1 (Random Search)**: ค้นหา `action_sequence` ที่ดีในระดับหนึ่ง
            - **Phase 2 (GA Refinement)**: นำ `action_sequence` ที่ดีที่สุดมาเป็น **"ประชากรเริ่มต้น"** ของ Genetic Algorithm เพื่อให้ GA พัฒนาต่อยอด
        
        2.  **🧬 DNA Consensus Finder**:
            - เป็นการทดลองตามแนวคิด: "เราสามารถสกัดแก่นแท้ของกลยุทธ์จาก Seed ที่ดีได้หรือไม่?"
            - **ขั้นตอน:** สร้างประชากรจาก Seed ดั้งเดิม -> ทำให้เกิดการกลายพันธุ์ -> หาเสียงส่วนใหญ่ (Consensus) ในแต่ละตำแหน่ง -> สร้างเป็น Sequence ใหม่ -> เปรียบเทียบประสิทธิภาพ

        3.  **✨ โมเดลใหม่ (Sequence-based)**:
            - **📈 Arithmetic Sequence**: สร้าง Action จากสมการลำดับเลขคณิต `Action(t) = sigmoid(a1 + t * d)`
            - **📉 Geometric Sequence**: สร้าง Action จากสมการลำดับเรขาคณิต `Action(t) = sigmoid(a1 * r^t)`

        4.  **⚡ Core Acceleration**:
            - ฟังก์ชันที่ทำงานช้าที่สุดคือ `_calculate_simulation_numba` ซึ่งเป็น Loop คำนวณผลการเทรด
            - ฟังก์ชันนี้ถูกเร่งความเร็วด้วย **Numba (`@njit`)** ซึ่งจะแปลงโค้ด Python เป็น Machine Code ที่ทำงานเร็วเทียบเท่าภาษา C
        """)

if __name__ == "__main__":
    main()
