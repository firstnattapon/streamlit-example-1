import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import calendar
from datetime import datetime, date, timedelta
from typing import List, Tuple, Dict, Any, Optional
import json  # ! NEW: For Export Functionality
from pathlib import Path
import streamlit.components.v1 as components

# ! NUMBA: Import Numba's Just-In-Time compiler for core acceleration
from numba import njit

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================
st.set_page_config(page_title="Hybrid_Multi_Mutation_Tabs", page_icon="🧬", layout="wide")

class Strategy:
    """คลาสสำหรับเก็บชื่อกลยุทธ์ต่างๆ"""
    REBALANCE_DAILY = "Rebalance Daily"
    PERFECT_FORESIGHT = "Perfect Foresight (Max)"
    HYBRID_MULTI_MUTATION = "Hybrid (Multi-Mutation)"
    ORIGINAL_DNA = "Original DNA (Pre-Mutation)"
    SPECTRUM_BEST = "Spectrum-10 (Best of 10)"
    SPECTRUM_TUNED = "Spectrum (Tuned k + phase)"
    SPECTRUM_COMPILED = "Spectrum (Compiled Max-E)"
    SPECTRUM_FAST = "Spectrum (Fast k + phase)"
    PERFECT_WINDOW = "Perfect Window (ต่อกัน)"

# ! DNA Spectrum: 10 DNA (N=100) เรียงจากความถี่สั้น -> ความถี่ยาว ใช้แทนการสุ่ม num_seeds ใน Phase 1
# ทุก code เป็น stream format เดิม [len][rate][dna_seed][mutation_seeds...] ถอดด้วย SimulationTracer ได้ตรงตามบิตที่ระบุ
SPECTRUM_LENGTH = 100
DNA_SPECTRUM_PRESETS: Dict[str, List[Tuple[str, str]]] = {
    "Density · 5-Mutation @10%": [
        ("Shortest · 86/100", "310021034383314338137273717223"),
        ("Very Short · 75/100", "3100210383239523933314231763694"),
        ("Short · 64/100", "310021025332403434355336363126"),
        ("Mid-Short · 55/100", "310021033523645340934272903968"),
        ("Medium-High · 50/100", "310021032792943597359537383270"),
        ("Medium · 45/100", "310021034252903219354832053493"),
        ("Mid-Long · 35/100", "310021033333811262316534903164"),
        ("Long · 25/100", "310021037623642310923532793996"),
        ("Very Long · 20/100", "3100210367735743415368234163493"),
        ("Longest · 19/100", "3100210314632243647314733413408"),
    ],
    "Periodic · คาบ k เป๊ะ": [
        ("k=1 · 100/100", "310014662249828936126265908610845464227193735"),
        ("k=2 · 50/100", "310014619777534456167586663299563079266129653"),
        ("k=4 · 25/100", "3100136229050471706429512625001049825310647393"),
        ("k=5 · 20/100", "31001466524681221762004304522121064966753186"),
        ("k=10 · 10/100", "3100165307424851854459563846975224236629094"),
        ("k=20 · 5/100", "310018569334611592862085356595944635764643444"),
        ("k=25 · 4/100", "31001654309425539694183256894762901446352202"),
        ("k=50 · 2/100", "31001654309425539694183261507126200183531310"),
        ("[0,99] · 2/100", "31001861938565821735250155469095120886754620"),
        ("k=100 · 1/100", "3100155430943235599950615535466957056396248"),
    ],
}

# ! GOAL_1: Multi-Timeframe Support (30m, 1h, 4h, 1d) via Yahoo Finance
ENCODED_DNA_MAGIC = 0
ENCODED_DNA_VERSION = 1
TIMEFRAME_OPTIONS = {
    "1d": {"minutes": 1440, "yfinance_interval": "1d"},
    "4h": {"minutes": 240, "yfinance_interval": "1h"},   # 4h = resample จาก 1h
    "1h": {"minutes": 60, "yfinance_interval": "1h"},
    "30m": {"minutes": 30, "yfinance_interval": "30m"},
}

# ! GOAL_1: ขีดจำกัดการดึงข้อมูลย้อนหลังของ yfinance ต่อ interval (วัน) เพื่อ clamp start_date
YFINANCE_INTRADAY_MAX_LOOKBACK_DAYS = {
    "30m": 59,   # yfinance รองรับ 30m ย้อนหลัง ~60 วัน
    "1h": 729,   # 1h / 4h ย้อนหลัง ~730 วัน
    "4h": 729,
}

def get_timeframe_minutes(timeframe: str) -> int:
    if timeframe not in TIMEFRAME_OPTIONS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return int(TIMEFRAME_OPTIONS[timeframe]["minutes"])

def get_timeframe_label_from_minutes(minutes: Optional[int]) -> str:
    for label, meta in TIMEFRAME_OPTIONS.items():
        if meta["minutes"] == minutes:
            return label
    return "legacy"

def encode_number_stream(numbers: List[int]) -> str:
    # ! GOAL_1/goal_2: เลข <=9 หลัก ใช้ length-prefix 1 หลักแบบเดิมเป๊ะ (legacy identical)
    # เลข >=10 หลัก (เช่น Unix timestamp) ใช้ escape '0' + ความยาว 2 หลัก
    # ('0' เป็น prefix ที่ legacy ไม่เคยสร้าง เพราะทุกเลขมีอย่างน้อย 1 หลัก)
    parts = []
    for num in numbers:
        s = str(int(num))
        length = len(s)
        if length <= 9:
            parts.append(f"{length}{s}")
        else:
            parts.append(f"0{length:02d}{s}")
    return "".join(parts)

def to_unix_timestamp(value: Any) -> int:
    return int(pd.Timestamp(value).timestamp())

def format_timeline_value(value: Any, timeframe: str) -> str:
    fmt = "%Y-%m-%d" if timeframe == "1d" else "%Y-%m-%d %H:%M"
    return pd.Timestamp(value).strftime(fmt)

def load_config() -> Dict[str, Any]:
    return {
        "assets": ["RGTI", "QBTS", "USAR"],
        "default_settings": {
            "start_date": "2024-01-01",
            "window_size": 30, "num_seeds": 1000, "max_workers": 1,
            "mutation_rate": 10.0, "num_mutations": 5
        }
    }

def initialize_session_state(config: Dict[str, Any]):
    defaults = config.get('default_settings', {})
    asset_list = config.get('assets', [])
    
    # ! GOAL: Default Select ALL Tickers + Dynamic Tickers Support
    if 'custom_tickers' not in st.session_state:
        st.session_state.custom_tickers = [
            "AAPL", "RGTI", "QBTS", "USAR", "NVDA", "TSLA", "IONQ", "QUBT",
            "FFWM", "NEGG", "RIVN", "APLS", "QXO", "RXRX", "AGL", "NVTS",
            "FLNC", "GERN", "DYN", "DJT", "CLSK", "LUNR", "IBRX", "SG"
        ]
    if 'selected_tickers' not in st.session_state: 
        st.session_state.selected_tickers = list(asset_list)
    if 'explorer_tickers' not in st.session_state:
        st.session_state.explorer_tickers = ["AAPL", "RGTI"]
    if 'explorer_custom_data' not in st.session_state:
        st.session_state.explorer_custom_data = {}
    
    if 'start_date' not in st.session_state:
        try: st.session_state.start_date = datetime.strptime(defaults.get('start_date', '2024-01-01'), '%Y-%m-%d').date()
        except ValueError: st.session_state.start_date = datetime(2024, 1, 1).date()
    if 'end_date' not in st.session_state: st.session_state.end_date = datetime.now().date()
    
    # Simulation Params
    if 'window_size' not in st.session_state: st.session_state.window_size = defaults.get('window_size', 30)
    if 'num_seeds' not in st.session_state: st.session_state.num_seeds = defaults.get('num_seeds', 1000)
    if 'max_workers' not in st.session_state: st.session_state.max_workers = defaults.get('max_workers', 8)
    if 'mutation_rate' not in st.session_state: st.session_state.mutation_rate = defaults.get('mutation_rate', 10.0)
    if 'num_mutations' not in st.session_state: st.session_state.num_mutations = defaults.get('num_mutations', 5)
    if 'selected_timeframe' not in st.session_state: st.session_state.selected_timeframe = "1d"

    # ! GOAL Step 4: Global Encoding Settings
    if 'trace_target_window' not in st.session_state: st.session_state.trace_target_window = 1
    if 'trace_action_length' not in st.session_state: st.session_state.trace_action_length = 0 # 0 means Auto (Window Size)
    if 'trace_start_timestamp' not in st.session_state: st.session_state.trace_start_timestamp = 0 # 0 means Auto (Window Start)
    
    if 'batch_results' not in st.session_state: st.session_state.batch_results = {}
    if 'spectrum_results' not in st.session_state: st.session_state.spectrum_results = {}
    if 'classifier_result' not in st.session_state: st.session_state.classifier_result = None
    if 'spectrum_mode' not in st.session_state: st.session_state.spectrum_mode = "best10"
    if 'spectrum_preset' not in st.session_state: st.session_state.spectrum_preset = next(iter(DNA_SPECTRUM_PRESETS))

# ==============================================================================
# 2. Core Calculation & Data Functions
# ==============================================================================
def clamp_intraday_start_date(start_date: str, end_date: str, timeframe: str) -> Tuple[str, bool]:
    """
    ! GOAL_1: yfinance ดึงข้อมูล intraday ย้อนหลังได้จำกัด (30m ~60 วัน, 1h/4h ~730 วัน)
    ปรับ start_date ให้อยู่ในช่วงที่ดึงได้ เพื่อไม่ให้ได้ข้อมูลว่าง
    Returns: (effective_start_date, was_clamped)
    """
    max_lookback = YFINANCE_INTRADAY_MAX_LOOKBACK_DAYS.get(timeframe)
    if max_lookback is None:  # 1d ดึงย้อนหลังได้ไม่จำกัด
        return start_date, False
    end_ts = pd.Timestamp(end_date)
    earliest_allowed = (end_ts - pd.Timedelta(days=max_lookback)).normalize()
    requested_start = pd.Timestamp(start_date)
    if requested_start < earliest_allowed:
        return earliest_allowed.strftime('%Y-%m-%d'), True
    return start_date, False

@st.cache_data(ttl=3600)
def get_ticker_data(ticker: str, start_date: str, end_date: str, timeframe: str = "1d") -> pd.DataFrame:
    try:
        if timeframe not in TIMEFRAME_OPTIONS:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        interval = str(TIMEFRAME_OPTIONS[timeframe]["yfinance_interval"])
        effective_start, _ = clamp_intraday_start_date(start_date, end_date, timeframe)
        data = yf.Ticker(ticker).history(start=effective_start, end=end_date, interval=interval)[['Close']]
        if data.empty: return pd.DataFrame()
        if data.index.tz is None:
            data = data.tz_localize('UTC').tz_convert('Asia/Bangkok')
        else:
            data = data.tz_convert('Asia/Bangkok')
        if timeframe == "4h":
            data = data.resample("4h").last().dropna()
        return data
    except Exception as e:
        return pd.DataFrame()

@njit(cache=True)
def _calculate_net_profit_numba(action_array: np.ndarray, price_array: np.ndarray, fix: int = 1500) -> float:
    n = len(action_array)
    if n == 0 or len(price_array) == 0 or n > len(price_array): return -np.inf
    action_array_calc = action_array.copy(); action_array_calc[0] = 1
    initial_price = price_array[0]; initial_capital = fix * 2.0
    refer_net = -fix * np.log(initial_price / price_array[n-1])
    cash = float(fix); amount = float(fix) / initial_price
    for i in range(1, n):
        curr_price = price_array[i]
        if action_array_calc[i] != 0: cash += amount * curr_price - fix; amount = fix / curr_price
    final_sumusd = cash + (amount * price_array[n-1])
    net = final_sumusd - refer_net - initial_capital
    return net

def run_simulation(prices: List[float], actions: List[int], fix: int = 1500) -> pd.DataFrame:
    @njit
    def _full_sim_numba(action_arr, price_arr, fix_val):
        n = len(action_arr); empty = np.empty(0, dtype=np.float64)
        if n == 0 or len(price_arr) == 0: return empty, empty, empty, empty, empty, empty
        action_calc = action_arr.copy(); action_calc[0] = 1
        amount = np.empty(n, dtype=np.float64); buffer = np.zeros(n, dtype=np.float64)
        cash = np.empty(n, dtype=np.float64); asset_val = np.empty(n, dtype=np.float64)
        sumusd_val = np.empty(n, dtype=np.float64)
        init_price = price_arr[0]; amount[0] = fix_val / init_price; cash[0] = fix_val
        asset_val[0] = amount[0] * init_price; sumusd_val[0] = cash[0] + asset_val[0]
        refer = -fix_val * np.log(init_price / price_arr[:n])
        for i in range(1, n):
            curr_price = price_arr[i]
            if action_calc[i] == 0: amount[i] = amount[i-1]; buffer[i] = 0.0
            else: amount[i] = fix_val / curr_price; buffer[i] = amount[i-1] * curr_price - fix_val
            cash[i] = cash[i-1] + buffer[i]; asset_val[i] = amount[i] * curr_price; sumusd_val[i] = cash[i] + asset_val[i]
        return buffer, sumusd_val, cash, asset_val, amount, refer

    if not prices or not actions: return pd.DataFrame()
    min_len = min(len(prices), len(actions))
    prices_arr = np.array(prices[:min_len], dtype=np.float64); actions_arr = np.array(actions[:min_len], dtype=np.int32)
    buffer, sumusd, cash, asset_value, amount, refer = _full_sim_numba(actions_arr, prices_arr, fix)
    if len(sumusd) == 0: return pd.DataFrame()
    initial_capital = sumusd[0]
    return pd.DataFrame({
        'price': prices_arr, 'action': actions_arr, 'buffer': np.round(buffer, 2),
        'sumusd': np.round(sumusd, 2), 'cash': np.round(cash, 2), 'asset_value': np.round(asset_value, 2),
        'amount': np.round(amount, 2), 'refer': np.round(refer + initial_capital, 2),
        'net': np.round(sumusd - refer - initial_capital, 2)
    })

# ==============================================================================
# 3. Strategy Action Generation
# ==============================================================================
def generate_actions_rebalance_daily(num_days: int) -> np.ndarray:
    return np.ones(num_days, dtype=np.int32)

def generate_actions_perfect_foresight(prices: List[float], fix: int = 1500) -> np.ndarray:
    # หลักการ 1 ประโยค: คำตอบ optimal ซื้อ-ขายเฉพาะ "จุดกลับตัว" (local extrema) เท่านั้น
    # จึง pre-filter เหลือแค่จุดกลับตัวก่อนแล้วรัน DP เดิม → ได้ net เท่าเดิมเป๊ะ แต่โหนดน้อยลง เร็วขึ้น
    # พิสูจน์: แทรกโหนด b ระหว่าง a,c เปลี่ยนค่า = -(Pb/Pa-1)(Pc/Pb-1) → คุ้ม ⟺ b เป็นจุดกลับตัว
    # (บนช่วง monotone ผลคูณ>0 ตัดทิ้งได้เสมอ) ดังนั้น optimal ⊆ turning points ∪ {จุดเริ่ม}
    price_arr = np.asarray(prices, dtype=np.float64)
    n = len(price_arr)
    if n < 2:
        return np.ones(n, dtype=np.int32)
    # Pre-filter O(n) แบบ plateau-safe: ยุบ run ราคาเท่ากัน แล้วเก็บดัชนีแรกของ run ที่เป็น
    # จุดกลับทิศของลำดับราคาที่ต่างค่ากัน (strict > / < จะพังเมื่อราคาเท่ากัน จึงเทียบทิศล่าสุด)
    cand = [0]
    prev_price = price_arr[0]; prev_dir = 0; run_start = 0
    for i in range(1, n):
        p = price_arr[i]
        if p == prev_price: continue
        cur_dir = 1 if p > prev_price else -1
        if prev_dir != 0 and cur_dir != prev_dir and run_start != 0: cand.append(run_start)
        prev_dir = cur_dir; prev_price = p; run_start = i
    # DP โครงเดิม แต่รันบน "เฉพาะโหนด turning point" → เลือก subset ที่ optimal จริง (exact)
    cand_arr = np.asarray(cand, dtype=np.int64)
    cp = price_arr[cand_arr]; m = len(cand_arr)
    dp = np.full(m, -np.inf, dtype=np.float64)
    parent = np.zeros(m, dtype=np.int64)
    dp[0] = float(fix * 2.0)
    for k in range(1, m):
        profits = fix * ((cp[k] / cp[:k]) - 1.0)
        c = dp[:k] + profits
        b = int(np.argmax(c))
        dp[k] = c[b]; parent[k] = b
    final_scores = dp + fix * ((price_arr[-1] / cp) - 1.0)
    end_k = int(np.argmax(final_scores))
    actions = np.zeros(n, dtype=np.int32)
    while end_k > 0:
        actions[cand_arr[end_k]] = 1
        end_k = parent[end_k]
    actions[0] = 1
    return actions

def find_best_seed_for_window(prices_window: np.ndarray, num_seeds_to_try: int, max_workers: int) -> Tuple[int, float, np.ndarray]:
    window_len = len(prices_window)
    if window_len < 2: return 1, 0.0, np.ones(window_len, dtype=np.int32)

    def evaluate_seed_batch(seed_batch: np.ndarray) -> List[Tuple[int, float]]:
        results = []
        for seed in seed_batch:
            rng = np.random.default_rng(seed)
            actions = rng.integers(0, 2, size=window_len).astype(np.int32)
            actions[0] = 1
            net = _calculate_net_profit_numba(actions, prices_window)
            results.append((seed, net))
        return results

    best_seed, max_net = -1, -np.inf
    random_seeds = np.arange(num_seeds_to_try)
    batch_size = max(1, num_seeds_to_try // (max_workers * 4 if max_workers > 0 else 1))
    seed_batches = [random_seeds[j:j+batch_size] for j in range(0, len(random_seeds), batch_size)]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_seed_batch, batch) for batch in seed_batches]
        for future in as_completed(futures):
            for seed, final_net in future.result():
                if final_net > max_net: max_net, best_seed = final_net, seed
    if best_seed >= 0:
        rng_best = np.random.default_rng(best_seed)
        best_actions = rng_best.integers(0, 2, size=window_len).astype(np.int32)
    else:
        best_seed, best_actions, max_net = 1, np.ones(window_len, dtype=np.int32), 0.0
    best_actions[0] = 1
    return best_seed, max_net, best_actions

def find_best_mutation_for_sequence(original_actions: np.ndarray, prices_window: np.ndarray, num_mutation_seeds: int, mutation_rate: float, max_workers: int) -> Tuple[int, float, np.ndarray]:
    window_len = len(original_actions)
    if window_len < 2: return 1, -np.inf, original_actions

    def evaluate_mutation_seed_batch(seed_batch: np.ndarray) -> List[Tuple[int, float]]:
        results = []
        for seed in seed_batch:
            mutation_rng = np.random.default_rng(seed)
            mutated_actions = original_actions.copy()
            mutation_mask = mutation_rng.random(window_len) < mutation_rate
            mutated_actions[mutation_mask] = 1 - mutated_actions[mutation_mask]
            mutated_actions[0] = 1
            net = _calculate_net_profit_numba(mutated_actions.astype(np.int32), prices_window)
            results.append((seed, net))
        return results

    best_mutation_seed, max_mutated_net = -1, -np.inf
    mutation_seeds_to_try = np.arange(num_mutation_seeds)
    batch_size = max(1, num_mutation_seeds // (max_workers * 4 if max_workers > 0 else 1))
    seed_batches = [mutation_seeds_to_try[j:j+batch_size] for j in range(0, len(mutation_seeds_to_try), batch_size)]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_mutation_seed_batch, batch) for batch in seed_batches]
        for future in as_completed(futures):
            for seed, net in future.result():
                if net > max_mutated_net: max_mutated_net = net; best_mutation_seed = seed
    if best_mutation_seed >= 0:
        mutation_rng = np.random.default_rng(best_mutation_seed)
        final_mutated_actions = original_actions.copy()
        mutation_mask = mutation_rng.random(window_len) < mutation_rate
        final_mutated_actions[mutation_mask] = 1 - final_mutated_actions[mutation_mask]
        final_mutated_actions[0] = 1
    else:
        best_mutation_seed = -1; max_mutated_net = -np.inf; final_mutated_actions = original_actions.copy()
    return best_mutation_seed, max_mutated_net, final_mutated_actions

def generate_actions_hybrid_multi_mutation(
    ticker_data: pd.DataFrame, window_size: int, num_seeds: int, max_workers: int, 
    mutation_rate_pct: float, num_mutations: int, progress_bar=None, ticker_name:str="",
    timeframe: str = "1d"
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:

    prices = ticker_data['Close'].to_numpy()
    n = len(prices)
    final_actions = np.array([], dtype=np.int32)
    original_actions_full = np.array([], dtype=np.int32)
    window_details_list = []

    num_windows = (n + window_size - 1) // window_size
    mutation_rate = mutation_rate_pct / 100.0

    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        if len(prices_window) < 2: continue

        if progress_bar:
            progress_text = f"{ticker_name}: Window {i+1}/{num_windows}..."
            progress_bar.progress((i + 1) / num_windows, text=progress_text)

        dna_seed, current_best_net, current_best_actions = find_best_seed_for_window(prices_window, num_seeds, max_workers)
        original_actions_window = current_best_actions.copy()
        original_net_for_display = current_best_net
        successful_mutation_seeds = []

        for mutation_round in range(num_mutations):
            mutation_seed, mutated_net, mutated_actions = find_best_mutation_for_sequence(
                current_best_actions, prices_window, num_seeds, mutation_rate, max_workers
            )
            if mutated_net > current_best_net:
                current_best_net = mutated_net
                current_best_actions = mutated_actions
                successful_mutation_seeds.append(int(mutation_seed))

        final_actions = np.concatenate((final_actions, current_best_actions.astype(np.int32)))
        original_actions_full = np.concatenate((original_actions_full, original_actions_window.astype(np.int32)))

        start_date = ticker_data.index[start_index]; end_date = ticker_data.index[end_index-1]
        detail = {
            'window': i + 1,
            'timeframe': timeframe,
            'timeline': f"{format_timeline_value(start_date, timeframe)} to {format_timeline_value(end_date, timeframe)}",
            'start_timestamp': to_unix_timestamp(start_date),
            'end_timestamp': to_unix_timestamp(end_date),
            'dna_seed': dna_seed,
            'mutation_seeds': str(successful_mutation_seeds) if successful_mutation_seeds else "None",
            'improvements': len(successful_mutation_seeds),
            'original_net': round(original_net_for_display, 2),
            'final_net': round(current_best_net, 2)
        }
        window_details_list.append(detail)

    return original_actions_full, final_actions, pd.DataFrame(window_details_list)

# ==============================================================================
# 4. Simulation Tracer Class
# ==============================================================================
class SimulationTracer:
    def __init__(self, encoded_string: str):
        self.encoded_string: str = encoded_string
        self._decode_and_set_attributes()

    @staticmethod
    def decode_number_stream(encoded_string: str) -> List[int]:
        if not isinstance(encoded_string, str) or not encoded_string.isdigit():
            raise ValueError("Input ต้องเป็นสตริงที่ประกอบด้วยตัวเลขเท่านั้น")
        decoded_numbers = []
        idx = 0
        while idx < len(encoded_string):
            try:
                first = encoded_string[idx]; idx += 1
                if first == '0':
                    # escape: อ่านความยาว 2 หลักถัดไป (สำหรับเลข >=10 หลัก)
                    length_str = encoded_string[idx : idx + 2]; idx += 2
                    if len(length_str) != 2:
                        raise ValueError
                    length_of_number = int(length_str)
                    if length_of_number < 10:
                        raise ValueError
                else:
                    length_of_number = int(first)
                    if length_of_number <= 0:
                        raise ValueError
                number_str = encoded_string[idx : idx + length_of_number]; idx += length_of_number
                if len(number_str) != length_of_number:
                    raise ValueError
                decoded_numbers.append(int(number_str))
            except (IndexError, ValueError):
                raise ValueError(f"รูปแบบของสตริงไม่ถูกต้องที่ตำแหน่ง {idx}")
        return decoded_numbers

    def _decode_and_set_attributes(self):
        # ! GOAL_1: รองรับ timeline metadata (timeframe + start_timestamp) แบบ backward-compatible
        # ถ้าไม่มี magic prefix -> ถอดรหัสแบบ legacy เดิม (goal_2)
        decoded_numbers = self.decode_number_stream(self.encoded_string)
        self.raw_numbers: List[int] = decoded_numbers
        self.has_timeline_metadata: bool = False
        self.version: Optional[int] = None
        self.timeframe_minutes: Optional[int] = None
        self.start_timestamp: Optional[int] = None

        dna_numbers = decoded_numbers
        if decoded_numbers and decoded_numbers[0] == ENCODED_DNA_MAGIC:
            if len(decoded_numbers) < 7:
                raise ValueError("Timeline DNA string must include version, timeframe, timestamp, length, rate, and DNA seed")
            self.version = int(decoded_numbers[1])
            if self.version != ENCODED_DNA_VERSION:
                raise ValueError(f"Unsupported timeline DNA version: {self.version}")
            self.timeframe_minutes = int(decoded_numbers[2])
            if self.timeframe_minutes not in [meta["minutes"] for meta in TIMEFRAME_OPTIONS.values()]:
                raise ValueError(f"Unsupported timeframe minutes: {self.timeframe_minutes}")
            self.start_timestamp = int(decoded_numbers[3])
            if self.start_timestamp < 0:
                raise ValueError("Start timestamp must be greater than or equal to 0")
            self.has_timeline_metadata = True
            dna_numbers = decoded_numbers[4:]

        if len(dna_numbers) < 3:
            raise ValueError("ข้อมูลในสตริงไม่ครบถ้วน")
        self.action_length: int = int(dna_numbers[0])
        if self.action_length <= 0:
            raise ValueError("Action length must be greater than 0")
        self.mutation_rate: int = int(dna_numbers[1])
        self.dna_seed: int = int(dna_numbers[2])
        self.mutation_seeds: List[int] = [int(seed) for seed in dna_numbers[3:]]
        self.mutation_rate_float: float = self.mutation_rate / 100.0

    def run(self) -> np.ndarray:
        dna_rng = np.random.default_rng(seed=self.dna_seed)
        current_actions = dna_rng.integers(0, 2, size=self.action_length).astype(np.int32)
        current_actions[0] = 1
        for m_seed in self.mutation_seeds:
            mutation_rng = np.random.default_rng(seed=m_seed)
            mutation_mask = mutation_rng.random(self.action_length) < self.mutation_rate_float
            current_actions[mutation_mask] = 1 - current_actions[mutation_mask]
            current_actions[0] = 1
        return current_actions

    def __str__(self) -> str:
        timeline = ""
        if self.has_timeline_metadata:
            timeline = f", TF={get_timeframe_label_from_minutes(self.timeframe_minutes)}, StartTS={self.start_timestamp}"
        return (f"✅ Decoded: Len={self.action_length}, Rate={self.mutation_rate}%, DNA={self.dna_seed}, Mut={self.mutation_seeds}{timeline}")

    @staticmethod
    def encode(
        action_length: int,
        mutation_rate: int,
        dna_seed: int,
        mutation_seeds: List[int],
        timeframe_minutes: Optional[int] = None,
        start_timestamp: Optional[int] = None
    ) -> str:
        # ! GOAL_2: ไม่ส่ง timeline -> ได้ digit stream แบบเดิมเป๊ะทุกไบต์
        all_numbers = [int(action_length), int(mutation_rate), int(dna_seed)] + [int(seed) for seed in mutation_seeds]
        if timeframe_minutes is not None or start_timestamp is not None:
            if timeframe_minutes is None or start_timestamp is None:
                raise ValueError("timeframe_minutes and start_timestamp must be provided together")
            if int(timeframe_minutes) not in [meta["minutes"] for meta in TIMEFRAME_OPTIONS.values()]:
                raise ValueError(f"Unsupported timeframe minutes: {timeframe_minutes}")
            if int(start_timestamp) < 0:
                raise ValueError("start_timestamp must be greater than or equal to 0")
            all_numbers = [
                ENCODED_DNA_MAGIC,
                ENCODED_DNA_VERSION,
                int(timeframe_minutes),
                int(start_timestamp),
            ] + all_numbers
        return encode_number_stream(all_numbers)

# ==============================================================================
# 4b. DNA Spectrum (10 DNA แทนการสุ่ม num_seeds)
# ==============================================================================
def spectrum_actions(code: str, length: int) -> Tuple[SimulationTracer, np.ndarray]:
    """ถอด preset code ที่ความยาว length ด้วย SimulationTracer เดิม
    PCG64 สร้างบิตเรียงตามลำดับ จึงได้ prefix ของ 100 บิตเดิมเสมอเมื่อ length <= 100"""
    preset = SimulationTracer(code)
    tracer = SimulationTracer(SimulationTracer.encode(
        action_length=length, mutation_rate=preset.mutation_rate,
        dna_seed=preset.dna_seed, mutation_seeds=preset.mutation_seeds
    ))
    return tracer, tracer.run()

def find_best_spectrum_dna_for_window(
    prices_window: np.ndarray, spectrum_codes: List[Tuple[str, str]]
) -> Tuple[int, SimulationTracer, float, np.ndarray]:
    """Phase 1 แบบ Spectrum: ลอง DNA ทั้ง 10 ชุด แล้วเลือกตัวที่ net สูงสุด (เสมอกัน -> ความถี่สั้นกว่าชนะ)"""
    best = None
    for rank, (_, code) in enumerate(spectrum_codes, start=1):
        tracer, actions = spectrum_actions(code, len(prices_window))
        net = _calculate_net_profit_numba(actions, prices_window)
        if best is None or net > best[2]:
            best = (rank, tracer, net, actions)
    return best

# ---- 3 โหมดของแท็บ DNA Spectrum -----------------------------------------------
SPECTRUM_STRATEGY: Dict[str, str] = {
    "best10": Strategy.SPECTRUM_BEST, "tuned": Strategy.SPECTRUM_TUNED, "compiled": Strategy.SPECTRUM_COMPILED,
    "fast": Strategy.SPECTRUM_FAST,
}
SPECTRUM_MODES: Dict[str, str] = {
    "best10": "1) Best of 10 · เลือก 1 ใน 10 preset",
    "tuned": "2) จูนความถี่ k + จังหวะ (phase)",
    "compiled": "3) คอมไพล์ E สูงสุด (Perfect Foresight → DNA)",
    "fast": "4) จูน k + phase · Fast (Codebook)",
}
COMPILER_SEEDS = 100_000          # seed ที่คอมไพเลอร์ใช้ค้น (dna_seed และ mutation seeds)
COMPILER_RATES = [3, 5, 8, 12]    # mutation rate (%) ที่ลอง · ≥ 2 เพื่อให้ตีความตรงกันทุก decoder

def periodic_actions(length: int, k: int, phase: int) -> np.ndarray:
    """act ทุก k แท่ง เริ่มที่แท่ง phase (0 ≤ phase < k) · แท่งแรก act เสมอ"""
    actions = np.zeros(length, dtype=np.int32)
    actions[phase::k] = 1
    actions[0] = 1
    return actions

def find_best_period_for_window(prices_window: np.ndarray) -> Tuple[int, int, float, np.ndarray]:
    """โหมด 2: ลองทุก k = 1..L−1 และทุก phase แล้วเก็บคู่ที่ net สูงสุด (เสมอกัน -> k และ phase น้อยกว่า)"""
    length = len(prices_window)
    best = (1, 0, -np.inf, periodic_actions(length, 1, 0))
    for k in range(1, max(2, length)):
        for phase in range(k):
            actions = periodic_actions(length, k, phase)
            net = _calculate_net_profit_numba(actions, prices_window)
            if net > best[2]:
                best = (k, phase, net, actions)
    return best

@st.cache_resource(show_spinner="กำลังสร้างตาราง seed สำหรับคอมไพเลอร์ (ครั้งแรกครั้งเดียว)...")
def get_compiler_tables(length: int, n_seeds: int = COMPILER_SEEDS) -> Tuple[np.ndarray, np.ndarray]:
    """ตาราง PCG64 ของทุก seed ที่ความยาว length (ใช้ prefix กับ window ที่สั้นกว่าได้)
    base[s] = default_rng(s).integers(0, 2, length) · min_rate[s, i] = rate (%) ต่ำสุดที่บิต i ถูก flip
    (เงื่อนไขเดียวกับ decoder: default_rng(s).random(length) < rate / 100)"""
    base = np.empty((n_seeds, length), dtype=np.uint8)
    min_rate = np.empty((n_seeds, length), dtype=np.uint8)
    chunk = 5_000                                    # ทำทีละก้อน: หน่วยความจำชั่วคราวไม่โตตามจำนวน seed
    uniform = np.empty((min(chunk, n_seeds), length), dtype=np.float64)
    for start in range(0, n_seeds, chunk):
        stop = min(start + chunk, n_seeds)
        for seed in range(start, stop):
            base[seed] = np.random.default_rng(seed).integers(0, 2, size=length)
            uniform[seed - start] = np.random.default_rng(seed).random(length)
        u = uniform[:stop - start]
        rate = np.floor(u * 100).astype(np.int64) + 1
        rate = np.where(u >= rate / 100.0, rate + 1, rate)                     # กันปัดเศษ float
        rate = np.where(u < (rate - 1) / 100.0, rate - 1, rate)
        min_rate[start:stop] = np.clip(rate, 2, 255)
    return base, min_rate

def compile_actions_to_dna(target: np.ndarray, tables: Tuple[np.ndarray, np.ndarray]) -> Tuple[int, int, List[int]]:
    """โหมด 2/3: หา (rate, dna_seed, mutation_seeds) ที่ SimulationTracer ถอดออกมาได้ target พอดี
    เริ่มจาก dna_seed ที่ใกล้ target ที่สุด แล้วเพิ่ม mutation seed ทีละตัวที่แก้บิตผิดได้มากที่สุด"""
    base, min_rate = tables
    length = len(target)
    want = target.astype(np.uint8).copy(); want[0] = 1
    start = base[:, :length].copy(); start[:, 0] = 1
    candidates = np.argsort((start[:, 1:] != want[1:]).sum(axis=1), kind="stable")[:2]
    best = None
    for rate in COMPILER_RATES:
        flips = (min_rate[:, 1:length] <= rate).astype(np.float32)
        for dna_seed in candidates:
            bits = start[dna_seed].copy(); seeds: List[int] = []
            for _ in range(4 * length):
                wrong = bits[1:] != want[1:]
                if not wrong.any():
                    break
                gain = flips @ np.where(wrong, 1.0, -1.0).astype(np.float32)
                seed = int(np.argmax(gain))
                if gain[seed] <= 0:
                    break
                seeds.append(seed)
                bits[1:] = np.where(flips[seed] > 0, 1 - bits[1:], bits[1:])
            if (bits[1:] == want[1:]).all():
                size = len(SimulationTracer.encode(length, rate, int(dna_seed), seeds))
                if best is None or size < best[0]:
                    best = (size, rate, int(dna_seed), seeds)
    if best is None:
        raise ValueError("คอมไพล์ DNA ไม่สำเร็จ")
    _, rate, dna_seed, seeds = best
    decoded = SimulationTracer(SimulationTracer.encode(length, rate, dna_seed, seeds)).run()
    if not np.array_equal(decoded, want.astype(np.int32)):
        raise ValueError("DNA ที่คอมไพล์ถอดกลับไม่ตรง target")
    return rate, dna_seed, seeds

# ---- โหมด 4: จูน k + phase แบบเร็ว (numba grid + สมุด code + preset ร่วมแข่ง) ------------
CODEBOOK_PATH = Path(__file__).resolve().parent.parent / "periodic_codebook.json"

@njit(cache=True)
def _periodic_net_grid(prices: np.ndarray, fix: int = 1500) -> np.ndarray:
    """net ของทุก (k, phase) ในรอบเดียว · grid[k, phase] (ช่องที่ไม่ใช้ = -inf)
    สูตรและลำดับการคำนวณเดียวกับ _calculate_net_profit_numba(periodic_actions(L, k, phase))"""
    n = len(prices)
    size = max(2, n)
    grid = np.full((size, size), -np.inf)
    initial_price = prices[0]
    refer_net = -fix * np.log(initial_price / prices[n - 1])
    for k in range(1, size):
        for phase in range(k):
            cash = float(fix); amount = float(fix) / initial_price
            for i in range(1, n):
                if i >= phase and (i - phase) % k == 0:
                    cash += amount * prices[i] - fix; amount = fix / prices[i]
            grid[k, phase] = cash + (amount * prices[n - 1]) - refer_net - fix * 2.0
    return grid

def find_best_period_fast(prices_window: np.ndarray) -> Tuple[int, int, float, np.ndarray, np.ndarray]:
    """ผลเดียวกับ find_best_period_for_window (เสมอกัน -> k และ phase น้อยกว่า) แต่คำนวณทั้งตารางใน numba"""
    grid = _periodic_net_grid(prices_window)
    k, phase = (int(v) for v in np.unravel_index(int(np.argmax(grid)), grid.shape))
    actions = periodic_actions(len(prices_window), k, phase)
    return k, phase, float(_calculate_net_profit_numba(actions, prices_window)), actions, grid

def find_best_preset_any(prices_window: np.ndarray) -> Tuple[str, SimulationTracer, float, np.ndarray]:
    """preset ที่ net สูงสุดจากทุกชุด (Density + Periodic = 20 ตัว)"""
    best = None
    for preset_name, codes in DNA_SPECTRUM_PRESETS.items():
        for label, code in codes:
            tracer, actions = spectrum_actions(code, len(prices_window))
            net = float(_calculate_net_profit_numba(actions, prices_window))
            if best is None or net > best[2]:
                best = (f"{preset_name.split(' ·')[0]} · {label}", tracer, net, actions)
    return best

@st.cache_resource
def load_periodic_codebook() -> Tuple[int, Dict[Tuple[int, int], Tuple[int, int, List[int]]]]:
    """สมุด code สำเร็จรูป (สร้างด้วย scripts/build_periodic_codebook.py): (k, phase) -> (rate, dna_seed, mutation_seeds)
    ของ window ยาว length · ใช้กับ window ที่สั้นกว่าได้ (PCG64 ให้บิตเดิมเป็น prefix) · ไม่มีไฟล์ -> คอมไพล์แทน"""
    try:
        data = json.loads(CODEBOOK_PATH.read_text(encoding="utf-8"))
        book = {}
        for key, code in data["codes"].items():
            k, phase = (int(v) for v in key.split(":"))
            tracer = SimulationTracer(code)
            book[(k, phase)] = (tracer.mutation_rate, tracer.dna_seed, list(tracer.mutation_seeds))
        return int(data["length"]), book
    except Exception:
        return 0, {}

@st.cache_data(show_spinner=False)
def compile_periodic_cached(length: int, k: int, phase: int, table_length: int) -> Tuple[int, int, List[int]]:
    return compile_actions_to_dna(periodic_actions(length, k, phase), get_compiler_tables(table_length))

def periodic_dna(length: int, k: int, phase: int, table_length: int) -> Tuple[int, int, List[int], str]:
    """DNA ของ 'act ทุก k แท่ง เริ่มที่ phase' · หยิบจากสมุด code ถ้ามี (ตรวจถอดกลับทุกครั้ง) ไม่งั้นคอมไพล์ + cache"""
    book_length, book = load_periodic_codebook()
    if length <= book_length and (k, phase) in book:
        rate, dna_seed, seeds = book[(k, phase)]
        decoded = SimulationTracer(SimulationTracer.encode(length, rate, dna_seed, seeds)).run()
        if np.array_equal(decoded, periodic_actions(length, k, phase)):
            return rate, dna_seed, list(seeds), "codebook"
    rate, dna_seed, seeds = compile_periodic_cached(length, k, phase, max(length, table_length))
    return rate, dna_seed, list(seeds), "compiled"

def spectrum_curve(prices_window: np.ndarray, spectrum_codes: List[Tuple[str, str]]) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """กราฟ E-Spectrum ของ 1 window: net ที่ดีที่สุดของทุก k (เลือก phase แล้ว), จุดของ 10 preset และเพดาน max"""
    length = len(prices_window)
    curve = []
    for k in range(1, max(2, length)):
        nets = [_calculate_net_profit_numba(periodic_actions(length, k, ph), prices_window) for ph in range(k)]
        curve.append({"k": float(k), "net": float(max(nets))})
    points = []
    for rank, (label, code) in enumerate(spectrum_codes, start=1):
        actions = spectrum_actions(code, length)[1]
        points.append({"k": length / max(1, int(actions.sum())), "net": float(_calculate_net_profit_numba(actions, prices_window)),
                       "preset": f"R{rank} {label}"})
    max_net = float(_calculate_net_profit_numba(generate_actions_perfect_foresight(prices_window.tolist()).astype(np.int32), prices_window))
    return pd.DataFrame(curve), pd.DataFrame(points), max_net

def generate_actions_spectrum(
    ticker_data: pd.DataFrame, window_size: int, spectrum_codes: List[Tuple[str, str]],
    mode: str = "best10", progress_bar=None, ticker_name: str = "", timeframe: str = "1d"
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """ทุก Window เลือก actions ตามโหมด แล้วจบ (ไม่กลายพันธุ์ต่อ)
    best10   = DNA ที่ net สูงสุด 1 ใน 10 preset (preset ผ่านการปรับปรุงมาแล้ว)
    tuned    = act ทุก k แท่ง ที่ phase ที่ดีที่สุด แล้วคอมไพล์เป็น DNA
    compiled = actions ที่ net สูงสุดจริง (Perfect Foresight) แล้วคอมไพล์เป็น DNA
    fast     = เหมือน tuned แต่ค้นหาด้วย numba + สมุด code + ให้ 20 preset ร่วมแข่ง (ไม่น้อยกว่า tuned)
    max_net / efficiency = เพดานของ window และสัดส่วนที่ได้
    คืน (actions ของ DNA, ตาราง window, actions ของ Perfect Window ทุก window ต่อกัน)"""
    prices = ticker_data['Close'].to_numpy()
    n = len(prices)
    final_actions = np.array([], dtype=np.int32)
    ceiling_actions = np.array([], dtype=np.int32)
    window_details_list = []
    num_windows = (n + window_size - 1) // window_size
    tables = get_compiler_tables(window_size) if mode in ("tuned", "compiled") else None

    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        if len(prices_window) < 2: continue

        if progress_bar:
            progress_bar.progress((i + 1) / num_windows, text=f"{ticker_name}: Window {i+1}/{num_windows}...")

        max_actions = generate_actions_perfect_foresight(prices_window.tolist()).astype(np.int32)
        max_net = float(_calculate_net_profit_numba(max_actions, prices_window))
        detail_mode: Dict[str, Any] = {}
        if mode == "tuned":
            k, phase, net, actions = find_best_period_for_window(prices_window)
            rate, dna_seed, seeds = compile_actions_to_dna(actions, tables)
            detail_mode = {'spectrum_label': f"k={k} · phase {phase}", 'k': k, 'phase': phase}
        elif mode == "compiled":
            actions, net = max_actions, max_net
            rate, dna_seed, seeds = compile_actions_to_dna(actions, tables)
            detail_mode = {'spectrum_label': f"Max-E · act {int(actions.sum())} ครั้ง"}
        elif mode == "fast":
            # โหมด 4: k + phase ที่ดีที่สุด (เท่าโหมด 2) แข่งกับ 20 preset -> ไม่มีทางได้น้อยกว่าโหมด 2
            k, phase, net, actions, grid = find_best_period_fast(prices_window)
            preset_label, tracer, preset_net, preset_actions = find_best_preset_any(prices_window)
            if preset_net > net:
                net, actions = preset_net, preset_actions
                rate, dna_seed, seeds = tracer.mutation_rate, tracer.dna_seed, list(tracer.mutation_seeds)
                detail_mode = {'spectrum_label': f"preset · {preset_label}", 'source': 'preset', 'phase_dependence': np.nan}
            else:
                rate, dna_seed, seeds, source = periodic_dna(len(prices_window), k, phase, window_size)
                dependence = float(grid[k, :k].mean() / net) if net > 0 else np.nan
                detail_mode = {'spectrum_label': f"k={k} · phase {phase}", 'k': k, 'phase': phase,
                               'source': source, 'phase_dependence': round(dependence, 3)}
        else:
            rank, tracer, net, actions = find_best_spectrum_dna_for_window(prices_window, spectrum_codes)
            rate, dna_seed, seeds = tracer.mutation_rate, tracer.dna_seed, list(tracer.mutation_seeds)
            detail_mode = {'spectrum_rank': rank, 'spectrum_label': spectrum_codes[rank - 1][0]}
        final_actions = np.concatenate((final_actions, actions.astype(np.int32)))
        ceiling_actions = np.concatenate((ceiling_actions, max_actions))

        start_date = ticker_data.index[start_index]; end_date = ticker_data.index[end_index-1]
        window_details_list.append({
            'window': i + 1,
            'timeframe': timeframe,
            'timeline': f"{format_timeline_value(start_date, timeframe)} to {format_timeline_value(end_date, timeframe)}",
            'start_timestamp': to_unix_timestamp(start_date),
            'end_timestamp': to_unix_timestamp(end_date),
            **detail_mode,
            'acts': int(actions.sum()),
            'dna_seed': int(dna_seed),
            'mutation_seeds': str([int(s) for s in seeds]) if seeds else "None",
            'mutation_rate': int(rate),
            'net': round(float(net), 2),
            'max_net': round(max_net, 2),
            'efficiency': round(float(net) / max_net, 3) if max_net > 0 else np.nan,
        })

    return final_actions, pd.DataFrame(window_details_list), ceiling_actions

def perfect_window_actions(prices: np.ndarray, window_size: int) -> Tuple[np.ndarray, List[float]]:
    """Perfect Foresight ภายในแต่ละ Window ต่อกันจนจบ + net สูงสุดของแต่ละ Window
    แบ่ง Window แบบเดียวกับ generator อื่น (ข้าม Window ที่สั้นกว่า 2 แท่ง)"""
    actions, max_nets = [], []
    for start in range(0, len(prices), window_size):
        prices_window = prices[start:start + window_size]
        if len(prices_window) < 2: continue
        best = generate_actions_perfect_foresight(prices_window.tolist()).astype(np.int32)
        actions.append(best)
        max_nets.append(float(_calculate_net_profit_numba(best, prices_window)))
    return (np.concatenate(actions) if actions else np.array([], dtype=np.int32)), max_nets

def build_spectrum_table(spectrum_codes: List[Tuple[str, str]], window_size: int) -> pd.DataFrame:
    rows = []
    for rank, (label, code) in enumerate(spectrum_codes, start=1):
        tracer, actions = spectrum_actions(code, SPECTRUM_LENGTH)
        rows.append({
            'rank': rank,
            'label': label,
            'dna_code': code,
            'rate_%': tracer.mutation_rate,
            'dna_seed': tracer.dna_seed,
            'mutation_seeds': str(tracer.mutation_seeds),
            f'active_{SPECTRUM_LENGTH}': int(actions.sum()),
            f'active_first_{window_size}': int(actions[:window_size].sum()),
            'decoded_100bit_dna': ''.join(map(str, actions)),
        })
    return pd.DataFrame(rows)

# ==============================================================================
# 4c. Spectrum Classifier: ดึงราคา -> วัดพฤติกรรมราคา -> หาย่านความถี่ DNA ที่ดีที่สุด
# ==============================================================================
# yfinance ดึงแท่ง intraday ย้อนหลังได้จำกัด (15m/30m ~60 วัน, 1h ~730 วัน)
CLASSIFIER_INTERVALS: Dict[str, Dict[str, Any]] = {
    "15m": {"yf": "15m", "max_days": 59},
    "30m": {"yf": "30m", "max_days": 59},
    "1h": {"yf": "1h", "max_days": 729},
    "1d": {"yf": "1d", "max_days": 3650},
}
CLASSIFIER_FIX_C = 1500.0
CLASSIFIER_VR_LAGS = [2, 4, 8, 16, 26, 52, 100]

@st.cache_data(ttl=900)
def get_classifier_prices(ticker: str, interval: str, days: int) -> pd.DataFrame:
    """ราคาปิดย้อนหลัง days วัน (เฉพาะชั่วโมงปกติ) index เป็นเวลาตลาดของหุ้นนั้น"""
    try:
        end = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)
        start = end - pd.Timedelta(days=int(days) + 1)
        data = yf.Ticker(ticker).history(
            start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'),
            interval=CLASSIFIER_INTERVALS[interval]["yf"]
        )[['Close']].dropna()
        return data
    except Exception:
        return pd.DataFrame()

def gated_excess(prices: np.ndarray, gates: np.ndarray, fix_c: float = CLASSIFIER_FIX_C) -> float:
    """E สุดท้ายของ gated ledger (สูตรเดียวกับ lego_dash_core.gated_rebalancing_cashflow_from_prices)
    gates[i] = gate ของรอบ i+1 (รอบ 0 = anchor ที่ prices[0]) · E = Fix_c × Σ (x − 1 − ln x)
    โดย x = ราคาตอน act ÷ ราคาตอน act ครั้งก่อน · ช่วง Gate 0 หลัง act สุดท้ายไม่ถูกนับ"""
    acts = np.flatnonzero(gates) + 1
    pa = prices[np.concatenate(([0], acts))]
    x = pa[1:] / pa[:-1]
    return float(fix_c * np.sum(x - 1.0 - np.log(x)))

def cycled_gates(bits: np.ndarray, rounds: int, phase: int = 0) -> np.ndarray:
    """วน DNA 100 บิตให้ยาวเท่าจำนวนรอบ (เหมือน build_gate_actions) · phase = เลื่อนจุดเริ่ม · รอบแรก act เสมอ"""
    gates = np.resize(np.roll(bits, -phase), rounds).astype(np.int8)
    if rounds > 0:
        gates[0] = 1
    return gates

def phase_averaged_excess(prices: np.ndarray, bits: np.ndarray) -> float:
    """เฉลี่ย E จากการเลื่อนจุดเริ่ม DNA ครบ 100 ตำแหน่ง -> ตัดโชคเรื่องจังหวะ gate ตรงกับแท่งใหญ่"""
    rounds = len(prices) - 1
    return float(np.mean([gated_excess(prices, cycled_gates(bits, rounds, ph)) for ph in range(len(bits))]))

def variance_ratio(log_returns: np.ndarray, q: int) -> float:
    """VR(q) > 1 = ราคาไหลต่อ (โมเมนตัม) · < 1 = ดึงกลับ · ≈ 1 = สุ่มเดิน"""
    sums = np.convolve(log_returns, np.ones(q), mode='valid')
    return float(sums.var(ddof=1) / (q * log_returns.var(ddof=1)))

def classify_spectrum(closes: pd.Series, intraday: bool) -> Dict[str, Any]:
    """วิเคราะห์ทั้ง 2 ชุด DNA บนราคาจริง แล้วตรวจซ้ำหลายแบบ (เต็ม / ตัด gap ข้ามคืน / ครึ่งแรก / ครึ่งหลัง)"""
    prices = closes.to_numpy(dtype=np.float64)
    log_r = np.diff(np.log(prices))
    n = len(log_r)

    # จุดเริ่มวันใหม่ (สำหรับ gap ข้ามคืน และแบ่งครึ่งตรงรอยต่อวัน)
    if intraday:
        session = np.array([ts.date() for ts in closes.index])
        new_day = session[1:] != session[:-1]
        day_starts = np.flatnonzero(new_day) + 1
        split = int(day_starts[np.argmin(np.abs(day_starts - len(prices) / 2))]) if len(day_starts) else len(prices) // 2
    else:
        new_day = np.zeros(n, dtype=bool)
        split = len(prices) // 2

    checks: Dict[str, np.ndarray] = {"เต็มช่วง": prices}
    if intraday and new_day.any():
        checks["ตัด gap ข้ามคืน"] = prices[0] * np.exp(np.concatenate(([0.0], np.cumsum(np.where(new_day, 0.0, log_r)))))
    checks["ครึ่งแรก"] = prices[:split]
    checks["ครึ่งหลัง"] = prices[split:]

    vr_rows = []
    for q in CLASSIFIER_VR_LAGS:
        if q * 4 > n:
            continue
        vr = variance_ratio(log_r, q)
        se = float(np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * n)))
        vr_rows.append({"q (แท่ง)": q, "VR": round(vr, 2), "±2SE": round(2 * se, 2),
                        "พฤติกรรม": "ไหลต่อ (โมเมนตัม)" if vr > 1.1 else ("ดึงกลับ" if vr < 0.9 else "ใกล้สุ่มเดิน"),
                        "มีนัยสำคัญ": "ใช่" if abs(vr - 1) > 2 * se else "ไม่"})

    presets = {}
    for preset_name, codes in DNA_SPECTRUM_PRESETS.items():
        rows = []
        for rank, (label, code) in enumerate(codes, start=1):
            bits = spectrum_actions(code, SPECTRUM_LENGTH)[1]
            row = {"rank": rank, "label": label, "dna_code": code,
                   "gates": int(cycled_gates(bits, n).sum()),
                   "E (phase 0)": gated_excess(prices, cycled_gates(bits, n))}
            for check_name, series in checks.items():
                base = gated_excess(series, np.ones(len(series) - 1, dtype=np.int8))
                row[check_name] = phase_averaged_excess(series, bits) / base if base > 0 else np.nan
            row["E เฉลี่ย"] = phase_averaged_excess(prices, bits)
            rows.append(row)
        df = pd.DataFrame(rows)
        ratio_cols = list(checks.keys())
        df["ต่ำสุดทุกการตรวจ"] = df[ratio_cols].min(axis=1)
        presets[preset_name] = {
            "table": df,
            "best": int(df["เต็มช่วง"].idxmax()),
            "robust": int(df["ต่ำสุดทุกการตรวจ"].idxmax()),
        }

    baseline = gated_excess(prices, np.ones(n, dtype=np.int8))
    return {"prices": prices, "n_rounds": n, "checks": list(checks.keys()), "vr": pd.DataFrame(vr_rows),
            "presets": presets, "baseline": baseline,
            "sessions": int(new_day.sum()) + 1 if intraday else len(prices)}

# ==============================================================================
# 5. UI Rendering & Logic Functions
# ==============================================================================

# ! GOAL Step 3: Optimization & Refactoring
# Helper Function เพื่อใช้ซ้ำทั้งใน UI รายตัว และใน Batch Export
def generate_encoded_dna_for_ticker(
    df_windows: pd.DataFrame, 
    data_len: int, 
    target_win_num: int, 
    global_act_len: int, 
    mutation_rate: float,
    window_size: int,
    timeframe: str,
    start_timestamp: int,
    with_timeline: bool = True
) -> Tuple[str, int, int, int]:
    """
    Generate Encoded String logic extracted for reuse.
    with_timeline=False -> DNA ล้วน [len][rate][dna_seed][mutation_seeds] ที่ dna_engine ของบอทถอดได้
    Returns: (Encoded String, Actual Window Used, Final Action Length, Start Timestamp)
    """
    max_win = len(df_windows)
    use_win_num = target_win_num if target_win_num <= max_win else max_win

    start_idx = (use_win_num - 1) * window_size
    remaining = data_len - start_idx
    real_len = min(window_size, remaining)

    final_act_len = global_act_len if global_act_len > 0 else real_len

    row = df_windows.iloc[use_win_num - 1]
    # ผล Spectrum เก็บ rate ของ DNA แชมป์ไว้ต่อ window -> ใช้ค่านั้นแทน rate ใน Settings
    if 'mutation_rate' in row.index:
        mutation_rate = row['mutation_rate']
    dna_seed = int(row['dna_seed'])
    mut_seeds_str = row['mutation_seeds']
    final_start_timestamp = int(start_timestamp) if int(start_timestamp) > 0 else int(row.get('start_timestamp', 0))

    mut_seeds = []
    if mut_seeds_str not in ["None", "[]"]:
        clean = mut_seeds_str.strip('[]')
        if clean: mut_seeds = [int(s.strip()) for s in clean.split(',')]

    encoded_string = SimulationTracer.encode(
        action_length=int(final_act_len),
        mutation_rate=int(mutation_rate),
        dna_seed=dna_seed,
        mutation_seeds=mut_seeds,
        timeframe_minutes=get_timeframe_minutes(timeframe) if with_timeline else None,
        start_timestamp=final_start_timestamp if with_timeline else None
    )

    return encoded_string, use_win_num, int(final_act_len), final_start_timestamp

def bot_rate_warning(mutation_rate: float) -> Optional[str]:
    """dna_engine ของบอทอ่าน rate เป็นจำนวนเต็ม และตีความ 1 = 100% (≤ 1 คือความน่าจะเป็น)
    ส่วนหน้านี้อ่าน 1 = 1% -> rate 1 หรือมีทศนิยม จะถอดไม่ตรงกับที่จำลอง"""
    rate = float(mutation_rate)
    if rate != int(rate):
        return f"Mutation Rate {rate:g}% มีทศนิยม: DNA เก็บได้แค่จำนวนเต็ม ({int(rate)}%) บอทจะถอดไม่ตรงกับที่จำลอง · ตั้งเป็นจำนวนเต็ม ≥ 2"
    if int(rate) == 1:
        return "Mutation Rate 1%: dna_engine ของบอทตีความ rate 1 เป็น 100% จะถอดไม่ตรงกับที่จำลอง · ตั้ง ≥ 2"
    return None


def display_comparison_charts(results: Dict[str, pd.DataFrame], chart_title: str = '📊 เปรียบเทียบกำไรสุทธิ (Net Profit)'):
    valid_dfs = {name: df for name, df in results.items() if not df.empty and 'net' in df.columns}
    if not valid_dfs: st.warning("ไม่มีข้อมูลสำหรับสร้างกราฟ"); return
    try: longest_index = max((df.index for df in valid_dfs.values()), key=len, default=None)
    except ValueError: longest_index = None
    if longest_index is None: return
    chart_data = pd.DataFrame(index=longest_index)
    for name, df in valid_dfs.items(): chart_data[name] = df['net'].reindex(longest_index).ffill()
    st.write(chart_title); st.line_chart(chart_data)

def render_settings_tab():
    st.write("⚙️ **การตั้งค่าพารามิเตอร์**")
    config = load_config()
    asset_list = config.get('assets', [])

    # Dynamic Tickers Support in Settings
    all_ticker_options = list(dict.fromkeys(
        list(asset_list)
        + st.session_state.get("custom_tickers", [])
        + st.session_state.get("selected_tickers", [])
    ))
    dc1, dc2 = st.columns([3, 2])
    with dc2:
        new_tickers_raw = st.text_input(
            "➕ เพิ่ม Tickers แบบ Dynamic (คั่นด้วยเว้นวรรคหรือลูกน้ำ)",
            value="",
            placeholder="เช่น AAPL, NVDA, TSLA, MSFT",
            key="settings_dynamic_ticker_input",
            help="พิมพ์ชื่อหุ้นที่ต้องการเพิ่มแบบ Dynamic แล้วกด Enter ระบบจะเพิ่มเข้าในรายการให้ทันที"
        )
        if new_tickers_raw.strip():
            parsed_new = [
                t.strip().upper()
                for t in new_tickers_raw.replace(",", " ").replace(";", " ").split()
                if t.strip()
            ]
            added_any = False
            for sym in parsed_new:
                if sym not in all_ticker_options:
                    all_ticker_options.append(sym)
                if sym not in st.session_state.custom_tickers:
                    st.session_state.custom_tickers.append(sym)
                if sym not in st.session_state.selected_tickers:
                    st.session_state.selected_tickers.append(sym)
                    added_any = True
            if added_any:
                st.toast(f"เพิ่ม Tickers: {', '.join(parsed_new)}")
    with dc1:
        valid_defaults = [t for t in st.session_state.selected_tickers if t in all_ticker_options]
        st.session_state.selected_tickers = st.multiselect(
            "เลือก Tickers ที่ต้องการทดสอบ (Dynamic Tickers · เลือกได้หลายตัว)",
            options=all_ticker_options,
            default=valid_defaults
        )

    timeframe_labels = list(TIMEFRAME_OPTIONS.keys())
    st.session_state.selected_timeframe = st.selectbox(
        "Timeframe",
        options=timeframe_labels,
        index=timeframe_labels.index(st.session_state.selected_timeframe) if st.session_state.selected_timeframe in timeframe_labels else 0,
        help="เลือกใช้แท่ง 1d, 4h, 1h หรือ 30m สำหรับการจำลองและการเข้ารหัส timeline"
    )

    c1, c2 = st.columns(2)
    st.session_state.window_size = c1.number_input("ขนาด Window (แท่ง)", min_value=2, value=st.session_state.window_size, help="จำนวนแท่งราคาต่อ 1 Window ตาม Timeframe ที่เลือก")
    st.session_state.num_seeds = c2.number_input("จำนวน Seeds (DNA)", min_value=100, value=st.session_state.num_seeds, format="%d")

    c1, c2 = st.columns(2)
    st.session_state.start_date = c1.date_input("วันที่เริ่มต้น", value=st.session_state.start_date)
    st.session_state.end_date = c2.date_input("วันที่สิ้นสุด", value=st.session_state.end_date)

    st.divider()
    st.subheader("Parameters for Hybrid Strategy")
    c1, c2, c3 = st.columns(3)
    st.session_state.max_workers = c1.number_input("Max Workers", min_value=1, max_value=16, value=st.session_state.max_workers)
    st.session_state.mutation_rate = c2.slider("Mutation Rate (%)", 0.0, 50.0, st.session_state.mutation_rate, 0.5)
    st.session_state.num_mutations = c3.number_input("Mutation Rounds", min_value=0, max_value=10, value=st.session_state.num_mutations)

    # ! GOAL: Moved Inputs Here
    st.divider()
    with st.expander("🔌 **Global Encoding & Tracing Settings** (Advanced)", expanded=True):
        st.info("ตั้งค่า Default สำหรับการ Generate Encoded String ในหน้าผลลัพธ์")
        gc1, gc2 = st.columns(2)
        st.session_state.trace_target_window = gc1.number_input(
            "Target Window # (สำหรับ Encode)", 
            min_value=1, 
            value=st.session_state.trace_target_window,
            help="เลือก Window ลำดับที่ต้องการจะนำมาสร้าง Encoded String โดยอัตโนมัติในหน้าผลลัพธ์"
        )
        st.session_state.trace_action_length = gc2.number_input(
            "Action Length (0 = Auto/Window Size)", 
            min_value=0, 
            value=st.session_state.trace_action_length,
            help="กำหนดความยาว Action Sequence สำหรับ Encode (ใส่ 0 เพื่อใช้ความยาวจริงของ Window นั้นๆ)"
        )
        st.session_state.trace_start_timestamp = st.number_input(
            "Unix Start Timestamp (0 = Auto/Selected Window Start)",
            min_value=0,
            value=st.session_state.trace_start_timestamp,
            format="%d",
            help="Cloud decode ใช้ Unix timestamp นี้ร่วมกับ timeframe เพื่อ map เวลาปัจจุบันไปยัง index ของ DNA action"
        )

def execute_batch_processing(spectrum_codes: Optional[List[Tuple[str, str]]] = None, spectrum_mode: str = "best10"):
    # spectrum_codes=None -> Hybrid เดิม (สุ่ม num_seeds) / ส่งมา -> DNA Spectrum (เก็บผลแยกใน spectrum_results)
    results_key = 'spectrum_results' if spectrum_codes else 'batch_results'
    tickers = st.session_state.selected_tickers
    if not tickers:
        st.error("กรุณาเลือก Ticker อย่างน้อย 1 ตัว"); return

    start_str = str(st.session_state.start_date)
    end_str = str(st.session_state.end_date)
    selected_timeframe = st.session_state.selected_timeframe

    # ! GOAL_1: แจ้งเตือนเมื่อ start_date ถูก clamp เพราะข้อจำกัด intraday ของ yfinance
    effective_start, was_clamped = clamp_intraday_start_date(start_str, end_str, selected_timeframe)
    if was_clamped:
        st.warning(
            f"⚠️ Timeframe {selected_timeframe}: yfinance ดึงข้อมูลย้อนหลังได้จำกัด "
            f"จึงปรับวันที่เริ่มต้นจาก {start_str} เป็น {effective_start} โดยอัตโนมัติ"
        )

    st.session_state[results_key] = {}

    overall_progress = st.progress(0, text="Starting Batch Process...")
    total_tickers = len(tickers)

    for idx, ticker in enumerate(tickers):
        ticker_data = get_ticker_data(ticker, start_str, end_str, selected_timeframe)
        if ticker_data.empty:
            st.warning(f"Skipping {ticker}: No Data Found.")
            continue

        prices = ticker_data['Close'].to_numpy()
        if spectrum_codes:
            # DNA Spectrum: เลือก actions ต่อ Window ตามโหมด ไม่มีขั้นกลายพันธุ์
            final_actions, df_windows, ceiling_actions = generate_actions_spectrum(
                ticker_data, st.session_state.window_size, spectrum_codes, mode=spectrum_mode,
                progress_bar=overall_progress, ticker_name=ticker, timeframe=selected_timeframe
            )
            # เพดาน = Perfect Window ทุก window ต่อกันจนจบ (กติกาเดียวกับ DNA: act ที่ต้นทุก window)
            sim_results = {
                SPECTRUM_STRATEGY[spectrum_mode]: run_simulation(prices.tolist(), final_actions.tolist()),
                Strategy.PERFECT_WINDOW: run_simulation(prices.tolist(), ceiling_actions.tolist()),
            }
        else:
            original_actions, final_actions, df_windows = generate_actions_hybrid_multi_mutation(
                ticker_data, st.session_state.window_size, st.session_state.num_seeds,
                st.session_state.max_workers, st.session_state.mutation_rate,
                st.session_state.num_mutations,
                progress_bar=overall_progress,
                ticker_name=ticker,
                timeframe=selected_timeframe
            )
            # เพดาน = Perfect Window ทุก window ต่อกันจนจบ (กติกาเดียวกับ Hybrid: act ที่ต้นทุก window)
            ceiling_actions, window_max = perfect_window_actions(prices, st.session_state.window_size)
            df_windows = df_windows.assign(max_net=np.round(window_max, 2))
            df_windows['efficiency'] = np.where(df_windows['max_net'] > 0, (df_windows['final_net'] / df_windows['max_net']).round(3), np.nan)
            sim_results = {
                Strategy.HYBRID_MULTI_MUTATION: run_simulation(prices.tolist(), final_actions.tolist()),
                Strategy.ORIGINAL_DNA: run_simulation(prices.tolist(), original_actions.tolist()),
                Strategy.PERFECT_WINDOW: run_simulation(prices.tolist(), ceiling_actions.tolist()),
            }
        sim_results[Strategy.REBALANCE_DAILY] = run_simulation(prices.tolist(), generate_actions_rebalance_daily(len(prices)).tolist())
        for name, df in sim_results.items():
            if not df.empty: df.index = ticker_data.index[:len(df)]
            
        st.session_state[results_key][ticker] = {
            "sim_results": sim_results,
            "df_windows": df_windows,
            "data_len": len(ticker_data),
            "timeframe": selected_timeframe,
            "prices": prices,
            "window_size": int(st.session_state.window_size),
            "spectrum_mode": spectrum_mode if spectrum_codes else None,
        }
        overall_progress.progress((idx + 1) / total_tickers, text=f"Completed {ticker} ({idx+1}/{total_tickers})")

    overall_progress.empty()
    st.success(f"✅ Processed {len(st.session_state[results_key])} tickers successfully!")

def render_single_ticker_result(
    ticker: str, result_data: Dict[str, Any], key_prefix: str = "",
    hybrid: Tuple[str, str] = (Strategy.HYBRID_MULTI_MUTATION, "Hybrid Strategy"),
    original: Optional[Tuple[str, str]] = (Strategy.ORIGINAL_DNA, "Original DNA"),
    ceiling: Tuple[str, str] = (Strategy.PERFECT_FORESIGHT, "Perfect Foresight"),
    show_metrics: bool = True
):
    # hybrid / original / ceiling = (ชื่อ strategy ใน sim_results, ป้าย metric) · original=None -> ไม่แสดง
    sim_results = result_data["sim_results"]
    df_windows = result_data["df_windows"]
    data_len = result_data["data_len"]
    timeframe = result_data.get("timeframe", st.session_state.selected_timeframe)

    chart_results = {k: v for k, v in sim_results.items() if original is None or k != original[0]}
    display_comparison_charts(chart_results, chart_title=f'📊 {ticker} - Net Profit Comparison')

    def get_final_net(strategy_name):
        df = sim_results.get(strategy_name)
        return df['net'].iloc[-1] if df is not None and not df.empty else 0.0

    if show_metrics:
        if original is None:
            c1, c2, c4 = st.columns(3)
        else:
            c1, c2, c3, c4 = st.columns(4)
        c1.metric(ceiling[1], f"${get_final_net(ceiling[0]):,.0f}")
        c2.metric(hybrid[1], f"${get_final_net(hybrid[0]):,.0f}", delta_color="normal")
        if original is not None:
            c3.metric(original[1], f"${get_final_net(original[0]):,.0f}")
        c4.metric("Rebalance Daily", f"${get_final_net(Strategy.REBALANCE_DAILY):,.0f}")
    
    st.divider()
    st.write(f"📝 **Detailed Window Results ({ticker})**")
    with st.expander("Dataframe_Results"):
        st.dataframe(df_windows, use_container_width=True)
    
    # ! GOAL: Use Global Settings for Encoding (Refactored to use Helper)
    st.markdown(f"#### 🎁 Generate Encoded String for **{ticker}**")
    
    target_win_num = st.session_state.trace_target_window
    global_act_len = st.session_state.trace_action_length
    window_size = st.session_state.window_size
    trace_start_timestamp = st.session_state.trace_start_timestamp

    # Calculate display parameters first for UI (logic duplication minimalized for display only)
    max_win = len(df_windows)
    use_win_num_disp = target_win_num if target_win_num <= max_win else max_win
    auto_start_ts_disp = int(df_windows.iloc[use_win_num_disp - 1].get('start_timestamp', 0)) if max_win > 0 else 0
    start_ts_disp = trace_start_timestamp if trace_start_timestamp > 0 else auto_start_ts_disp
    rate_disp = st.session_state.mutation_rate
    if max_win > 0 and 'mutation_rate' in df_windows.columns:
        rate_disp = df_windows.iloc[use_win_num_disp - 1]['mutation_rate']

    c_enc_1, c_enc_2 = st.columns([3, 1])
    with c_enc_1:
        st.info(f"Using Global Settings: **Window {use_win_num_disp}**, **Len {global_act_len if global_act_len>0 else 'Auto'}**, **TF {timeframe}**, **StartTS {start_ts_disp}** (Rate: {rate_disp}%)")
        if target_win_num > max_win:
             st.caption(f"⚠️ Window {target_win_num} เกินจำนวนที่มี (Max {max_win}). ใช้ Window {max_win} แทน")

    with c_enc_2:
        clicked = st.button(f"Encode ({ticker})", key=f"btn_enc_{key_prefix}{ticker}", use_container_width=True)
    if clicked:
        try:
            # ! REFACTORED: Use helper function
            args = (df_windows, data_len, target_win_num, global_act_len,
                    st.session_state.mutation_rate, window_size, timeframe, trace_start_timestamp)
            dna_code, final_win, final_len, _ = generate_encoded_dna_for_ticker(*args, with_timeline=False)
            encoded, _, _, final_start_ts = generate_encoded_dna_for_ticker(*args)

            st.success(f"🧬 DNA สำหรับบอท ({ticker} Win {final_win}, Len {final_len}) · ใส่ใน dna_engine ได้เลย:")
            st.code(dna_code, language='text')
            warning = bot_rate_warning(rate_disp)
            if warning:
                st.warning(f"⚠️ {warning}")
            st.caption(f"Timeline string (รูปแบบเดิม มี TF {timeframe} + StartTS {final_start_ts}) · ใช้กับ Tracer ในหน้านี้ "
                       "· dna_engine ของบอทยังถอดรูปแบบนี้ไม่ได้")
            st.code(encoded, language='text')
        except Exception as e:
            st.error(f"Encoding Error: {e}")

def render_methodology_expander():
    st.markdown(f"### 🧬 {Strategy.HYBRID_MULTI_MUTATION}")
    st.info("กลยุทธ์นี้ทำงานโดย: 1. ค้นหา 'DNA' ที่ดีที่สุดในแต่ละ Window 2. นำ DNA นั้นมาพยายาม 'กลายพันธุ์' (Mutate) ซ้ำๆ เพื่อหาผลลัพธ์ที่ดีกว่าเดิม")

    with st.expander("📖 คำอธิบายวิธีการทำงานและแนวคิด (Multi-Mutation)"):
        st.markdown(
            """
            แนวคิด **Hybrid (Multi-Mutation)** ได้รับแรงบันดาลใจจากกระบวนการ **วิวัฒนาการและการคัดเลือกสายพันธุ์ (Evolution & Selective Breeding)** โดยมีเป้าหมายเพื่อ "พัฒนา" รูปแบบการซื้อขาย (Actions) ที่ดีที่สุดให้ดียิ่งขึ้นไปอีกแบบซ้ำๆ ภายในแต่ละ Window

            แทนที่จะเปรียบเทียบระหว่าง "DNA ดั้งเดิม" กับ "การกลายพันธุ์แค่ครั้งเดียว" กลยุทธ์นี้จะนำผู้ชนะ (Champion) มาผ่านกระบวนการกลายพันธุ์ซ้ำๆ หลายรอบ เพื่อค้นหาการปรับปรุงที่ดีขึ้นเรื่อยๆ

            ---

            #### 🧬 กระบวนการทำงานในแต่ละ Window:

            1.  **เฟส 1: ค้นหา "แชมป์เปี้ยนตั้งต้น" (Initial Champion)**
                * โปรแกรมจะทำการสุ่ม Actions หรือ "DNA" ขึ้นมาตามจำนวน `num_seeds` ที่กำหนด
                * DNA ที่สร้างกำไร (Net Profit) ได้สูงสุด จะถูกคัดเลือกให้เป็น **"แชมป์เปี้ยนตัวแรก"**
                * `DNA_Original = argmax_{s in S_dna} [ Profit(Generate_DNA(s)) ]`

            2.  **เฟส 2: กระบวนการ "กลายพันธุ์ต่อเนื่อง" (Iterative Mutation)**
                * โปรแกรมจะเริ่มลูปการกลายพันธุ์ตามจำนวนรอบ (`num_mutations`) ที่กำหนด
                * **ในแต่ละรอบ:**
                    * **สร้างผู้ท้าชิง:** นำ Actions ของ **"แชมป์เปี้ยนปัจจุบัน"** มาเป็นต้นแบบ แล้วค้นหารูปแบบการกลายพันธุ์ (Mutation Pattern) ที่ดีที่สุดเพื่อสร้าง "ผู้ท้าชิง" (Challenger)
                    * `Challenger = argmax_{s_m in S_mutation} [ Profit(Mutate(Current_Champion, s_m)) ]`
                    * **คัดเลือกผู้ที่แข็งแกร่งที่สุด (Survival of the Fittest):** เปรียบเทียบกำไรระหว่าง "ผู้ท้าชิง" กับ "แชมป์เปี้ยนปัจจุบัน"
                        * **ถ้าผู้ท้าชิงชนะ:** ผู้ท้าชิงจะกลายเป็น **"แชมป์เปี้ยนคนใหม่"** และจะถูกนำไปใช้เป็นต้นแบบในรอบการกลายพันธุ์ถัดไป
                        * **ถ้าแชมป์เปี้ยนปัจจุบันชนะ:** แชมป์เปี้ยนจะยังคงตำแหน่งเดิม และถูกนำไปใช้เป็นต้นแบบในรอบถัดไป
            
            3.  **เฟส 3: ผลลัพธ์สุดท้าย**
                * หลังจากผ่านกระบวนการกลายพันธุ์ครบทุกรอบแล้ว **"แชมป์เปี้ยนตัวสุดท้าย"** ที่รอดมาได้ คือ Actions ที่จะถูกนำไปใช้สำหรับ Window นั้นจริงๆ

            ---
            
            #### ตัวอย่าง: (สมมติ `num_mutations = 2`)

            1.  **ค้นหา DNA ดั้งเดิม:** พบว่า Seed `5784` ให้กำไรดีที่สุด `Net Profit = $1,200`
                * **แชมป์เปี้ยนปัจจุบัน:** Actions จาก Seed `5784` (Profit: $1,200)

            2.  **Mutation รอบที่ 1:**
                * นำ Actions ของแชมป์เปี้ยน (Seed `5784`) ไปค้นหารูปแบบกลายพันธุ์ที่ดีที่สุด
                * พบว่า Mutation Seed `8871` สามารถพัฒนากำไรเป็น `$1,550` ได้
                * เนื่องจาก `$1,550 > $1,200` → ผู้ท้าชิงชนะ!
                * **แชมป์เปี้ยนคนใหม่:** Actions ที่กลายพันธุ์จาก Seed `8871` (Profit: $1,550)

            3.  **Mutation รอบที่ 2:**
                * นำ Actions ของแชมป์เปี้ยนคนใหม่ (ที่มาจาก Mutation Seed `8871`) ไปค้นหารูปแบบกลายพันธุ์ที่ดีที่สุดอีกครั้ง
                * พบว่า Mutation Seed `10524` สามารถพัฒนากำไรต่อได้เป็น `$1,620`
                * เนื่องจาก `$1,620 > $1,550` → ผู้ท้าชิงชนะอีกครั้ง!
                * **แชมป์เปี้ยนคนใหม่:** Actions ที่กลายพันธุ์จาก Seed `10524` (Profit: $1,620)

            4.  **จบกระบวนการ:** Actions สุดท้ายสำหรับ Window นี้คือ Actions ที่ให้กำไร `$1,620` ซึ่งเป็นผลลัพธ์จากการพัฒนาต่อยอดมา 2 รอบ

            แนวคิด **Hybrid (Multi-Mutation)** ได้รับแรงบันดาลใจจากกระบวนการ **วิวัฒนาการและการคัดเลือกสายพันธุ์ (Evolution & Selective Breeding)** โดยมีเป้าหมายเพื่อ "พัฒนา" รูปแบบการซื้อขาย (Actions) ที่ดีที่สุดให้ดียิ่งขึ้นไปอีกแบบซ้ำๆ ภายในแต่ละ Window

            ---

            ### 🔬 เจาะลึก Logic: หัวใจของกระบวนการกลายพันธุ์ (Mutation)
    
            กระบวนการกลายพันธุ์คือการนำรูปแบบการซื้อขาย (Actions) ของ **"แชมป์เปี้ยนปัจจุบัน"** มาทำการ **"ปรับปรุงเล็กน้อยอย่างสุ่ม"** เพื่อมองหาโอกาสที่จะพัฒนามันให้ดียิ่งขึ้นไปอีก เปรียบเสมือนการคัดเลือกสายพันธุ์เพื่อหาลักษณะเด่นที่ดีกว่าเดิม
    
            หัวใจสำคัญของกระบวนการนี้เกิดขึ้นภายในฟังก์ชัน `find_best_mutation_for_sequence` ซึ่งจะสร้าง "ผู้ท้าชิง" (Challenger) ขึ้นมาหลายพันราย โดยแต่ละรายจะถูกสร้างผ่าน 3 ขั้นตอนหลักดังนี้:
    
            ---
    
            #### ขั้นตอนที่ 1: 📜 สร้าง "แผนผังการกลายพันธุ์" (Mutation Blueprint)
    
            ในขั้นตอนนี้ โปรแกรมจะใช้ `mutation_seed` ที่ไม่ซ้ำกันเพื่อสร้าง "พิมพ์เขียว" ที่กำหนดว่ายีน (Action) ในวันไหนควรจะเปลี่ยนแปลง
    
            ```python
            # สร้างอาเรย์ของเลขสุ่ม (0.0 - 1.0) ตาม seed ที่กำหนด
            # แล้วเปรียบเทียบกับ mutation_rate เพื่อสร้าง "แผนผัง"
            mutation_mask = mutation_rng.random(window_len) < mutation_rate
            ```
    
            * **`mutation_rng.random(window_len)`**: สร้างชุดตัวเลขสุ่มขึ้นมา 1 ตัวต่อ 1 วันใน Window การใช้ `seed` ที่ต่างกันจะให้ชุดตัวเลขสุ่มที่ต่างกัน
            * **`< mutation_rate`**: นำตัวเลขสุ่มแต่ละตัวมาเทียบกับอัตราการกลายพันธุ์ (เช่น 5% หรือ 0.05)
                * ถ้าน้อยกว่า ➡️ `True` (ตำแหน่งนี้จะเกิดการเปลี่ยนแปลง)
                * ถ้ามากกว่า ➡️ `False` (ตำแหน่งนี้จะคงเดิม)
            * **ผลลัพธ์**: คือ "แผนผัง" ที่เป็น `True` / `False` ซึ่งเป็นเหมือนพิมพ์เขียวสำหรับการเปลี่ยนแปลงในขั้นตอนต่อไป
    
            **ตัวอย่าง:**
            * `original_actions`: `[1, 0, 1, 1]`
            * `mutation_rate`: 50% (0.5)
            * `เลขสุ่มที่สร้างได้`: `[0.23, 0.81, 0.99, 0.45]`
            * **แผนผัง (`mutation_mask`)**: `[True, False, False, True]`
    
            ---
    
            #### ขั้นตอนที่ 2: 🧬 ดำเนินการเปลี่ยนแปลงตามแผนผัง (Flipping the Genes)
    
            โปรแกรมจะนำ Actions ของแชมป์เปี้ยนมาคัดลอก แล้ว "พลิกค่า" เฉพาะในตำแหน่งที่แผนผังเป็น `True`
    
            ```python
            # คัดลอก Actions เดิมมา
            mutated_actions = original_actions.copy()
            
            # ใช้แผนผัง (mask) เพื่อเลือกตำแหน่งที่จะ "พลิกค่า"
            mutated_actions[mutation_mask] = 1 - mutated_actions[mutation_mask]
            ```
    
            * `1 - action` เป็นเทคนิคที่รวดเร็วในการพลิกค่า:
                * ถ้า Action เดิมเป็น `1` (ซื้อ) ➡️ `1 - 1` จะได้ `0` (ถือ)
                * ถ้า Action เดิมเป็น `0` (ถือ) ➡️ `1 - 0` จะได้ `1` (ซื้อ)
    
            **ตัวอย่าง (ต่อ):**
            * **ต้นฉบับ**: `[1, 0, 1, 1]`
            * **แผนผัง**: `[T, F, F, T]` (เปลี่ยนแปลงตำแหน่งที่ 0 และ 3)
            * **ตำแหน่ง 0**: `1` พลิกเป็น `0`
            * **ตำแหน่ง 3**: `1` พลิกเป็น `0`
            * **ผลลัพธ์หลังการพลิกยีน**: `[0, 0, 1, 0]`
    
            ---
    
            #### ขั้นตอนที่ 3: 🛡️ บังคับใช้กฎเหล็ก (The First-Day Rule)
    
            เพื่อรับประกันว่าทุกการจำลองจะเริ่มต้นด้วยการซื้อเสมอ โปรแกรมจะบังคับให้ Action ของวันแรกสุดเป็น `1` เสมอ ไม่ว่าการกลายพันธุ์จะให้ผลเป็นอย่างไรก็ตาม
    
            ```python
            # ไม่ว่าผลจะเป็นอย่างไร วันแรกต้องเป็น 1 เสมอ
            mutated_actions[0] = 1
            ```
    
            **ตัวอย่าง (สุดท้าย):**
            * **ผลลัพธ์จากการพลิกยีน**: `[0, 0, 1, 0]`
            * **บังคับกฎข้อแรก**: `[1, 0, 1, 0]`
    
            > ✨ **ผลลัพธ์สุดท้าย** คือ Actions ของ "ผู้ท้าชิง" หนึ่งราย ที่พร้อมจะถูกนำไปประเมินผลกำไรเพื่อท้าชิงตำแหน่งแชมป์เปี้ยนต่อไป กระบวนการทั้งหมดนี้จะเกิดขึ้นซ้ำๆ หลายพันครั้งเพื่อค้นหารูปแบบการกลายพันธุ์ที่ดีที่สุดเพียงหนึ่งเดียวในแต่ละรอบ
            """
        )
        
        code = """ ตัวอย่าง code
        import numpy as np
        dna_rng = np.random.default_rng(seed=239)
        current_actions = dna_rng.integers(0, 2, size=30)
        default_actions = current_actions.copy() 
        
        mutation_seeds = [30]
        #รอบที่ for loop
        m_seed = 30
        mutation_rng = np.random.default_rng(seed=30)
        mutation_mask = mutation_rng.random(30) < 0.10 # Mutation Rate 10(%)
        [0.72..., 0.39..., 0.03..., 0.58..., 0.41..., ...]
        [False False  True False False False False False False False False False
        False False False False False False False False False False  True False
        False False False False False False]
        
        current_actions[mutation_mask] = 1 - current_actions[mutation_mask] # Flipping the Genes
        current_actions[0] = 1
        default_actions[0] = 1
        
        print( "mutation_mask" , mutation_mask)
        print( "default_actions" , default_actions)
        print( "current_actions" , current_actions)
        """
        st.code(code, language="python")

def render_export_section(
    batch_results: Dict[str, Any], key: str, file_name: str,
    extra_metadata: Optional[Dict[str, Any]] = None
):
    st.write("---")
    exp_c1, exp_c2 = st.columns([3, 1])
    with exp_c1:
        st.markdown("#### 💾 Export Encoded Strings")
        st.caption(f"Settings: Window {st.session_state.trace_target_window}, Len {st.session_state.trace_action_length}, TF {st.session_state.selected_timeframe}, StartTS {st.session_state.trace_start_timestamp}, Rate {st.session_state.mutation_rate}%")

    with exp_c2:
        # Prepare data for JSON export
        export_payload = {
            "metadata": {
                "trace_target_window": st.session_state.trace_target_window,
                "trace_action_length": st.session_state.trace_action_length,
                "timeframe": st.session_state.selected_timeframe,
                "timeframe_minutes": get_timeframe_minutes(st.session_state.selected_timeframe),
                "trace_start_timestamp": st.session_state.trace_start_timestamp,
                "mutation_rate_percent": st.session_state.mutation_rate,
                "exported_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "dna_codes_format": "dna_codes = DNA ล้วน [len][rate][dna_seed][mutation_seeds] ถอดด้วย dna_engine.decode_dna ได้ · tickers = timeline string รูปแบบเดิม",
                **(extra_metadata or {})
            },
            "tickers": {},
            "dna_codes": {},
            "ticker_metadata": {}
        }

        # Loop through all results and generate strings
        for ticker, data in batch_results.items():
            try:
                timeframe = data.get("timeframe", st.session_state.selected_timeframe)
                encoded, final_win, final_len, final_start_ts = generate_encoded_dna_for_ticker(
                    data["df_windows"],
                    data["data_len"],
                    st.session_state.trace_target_window,
                    st.session_state.trace_action_length,
                    st.session_state.mutation_rate,
                    st.session_state.window_size,
                    timeframe,
                    st.session_state.trace_start_timestamp
                )
                export_payload["tickers"][ticker] = encoded
                export_payload["dna_codes"][ticker] = generate_encoded_dna_for_ticker(
                    data["df_windows"], data["data_len"], st.session_state.trace_target_window,
                    st.session_state.trace_action_length, st.session_state.mutation_rate,
                    st.session_state.window_size, timeframe, st.session_state.trace_start_timestamp,
                    with_timeline=False
                )[0]
                ticker_meta = {
                    "window": final_win,
                    "action_length": final_len,
                    "timeframe": timeframe,
                    "timeframe_minutes": get_timeframe_minutes(timeframe),
                    "start_timestamp": final_start_ts,
                }
                # ผล Spectrum: rate จริงของ string และ rank ที่ชนะใน window นั้น
                row = data["df_windows"].iloc[final_win - 1]
                if 'spectrum_label' in row.index:
                    ticker_meta["mutation_rate_percent"] = int(row['mutation_rate'])
                    ticker_meta["spectrum_label"] = row['spectrum_label']
                    if 'spectrum_rank' in row.index and pd.notna(row['spectrum_rank']):
                        ticker_meta["spectrum_rank"] = int(row['spectrum_rank'])
                    if 'efficiency' in row.index and pd.notna(row['efficiency']):
                        ticker_meta["efficiency"] = float(row['efficiency'])
                export_payload["ticker_metadata"][ticker] = ticker_meta
            except Exception as e:
                export_payload["tickers"][ticker] = f"Error: {str(e)}"

        json_str = json.dumps(export_payload, indent=4, ensure_ascii=False)
        st.download_button(
            label="💾 Download JSON (All Tickers)",
            data=json_str,
            file_name=file_name,
            mime="application/json",
            use_container_width=True,
            key=key
        )

def render_chained_vs_perfect_window(
    data: Dict[str, Any], strategy: str, who: str, net_col: str,
    extra_cols: Optional[Dict[str, str]] = None, secondary: Optional[Tuple[str, str]] = None
):
    """Perfect Window vs กลยุทธ์ ทีละ Window แล้วต่อกันจนจบ (ทั้งคู่ act ที่แท่งแรกของทุก Window)
    strategy = ชื่อใน sim_results · net_col = คอลัมน์ net ของกลยุทธ์ใน df_windows · extra_cols = คอลัมน์อื่นที่โชว์ {เดิม: ป้าย}
    secondary = (ชื่อใน sim_results, ป้าย) ของอีกเส้นที่สรุปไว้ใต้ตัวเลข"""
    sims, df_windows = data["sim_results"], data["df_windows"]
    def chained(name: str) -> float:
        df = sims.get(name)
        return float(df['net'].iloc[-1]) if df is not None and not df.empty else 0.0
    pw_total, main_total = chained(Strategy.PERFECT_WINDOW), chained(strategy)
    pct = (lambda v: f"{v / pw_total:.1%}" if pw_total > 0 else "–")

    st.write(f"🔗 **ต่อกันจนจบ: Perfect Window vs {who}** (ทุก Window เล่นกติกาเดียวกัน: act ที่แท่งแรกของ Window)")
    m1, m2, m3 = st.columns(3)
    m1.metric("Perfect Window · ต่อกันจนจบ", f"${pw_total:,.2f}")
    m2.metric(f"{who} · ต่อกันจนจบ", f"${main_total:,.2f}")
    m3.metric(f"{who} ได้กี่ % ของเพดาน", pct(main_total))
    notes = [f"เส้น {strategy}"]
    if secondary is not None:
        notes.append(f"{secondary[1]} ต่อกัน \\${chained(secondary[0]):,.2f} ({pct(chained(secondary[0]))})")
    notes.append(f"Rebalance Daily \\${chained(Strategy.REBALANCE_DAILY):,.2f} ({pct(chained(Strategy.REBALANCE_DAILY))})")
    st.caption(" · ".join(notes))

    extra_cols = extra_cols or {}
    cols = ['window', 'timeline'] + list(extra_cols) + ['max_net', net_col]
    per_window = df_windows[cols].rename(columns={**extra_cols, 'max_net': 'Perfect Window', net_col: f'{who} net'})
    main_col = f'{who} net'
    per_window['% ของเพดาน'] = np.where(per_window['Perfect Window'] > 0, per_window[main_col] / per_window['Perfect Window'], np.nan)
    total_pw, total_main = per_window['Perfect Window'].sum(), per_window[main_col].sum()
    total_row = {c: '' for c in per_window.columns}
    total_row.update({'window': 'รวม', 'Perfect Window': total_pw, main_col: total_main,
                      '% ของเพดาน': total_main / total_pw if total_pw > 0 else np.nan})
    money_cols = ['Perfect Window', main_col] + [label for col, label in extra_cols.items() if col.endswith('_net')]
    for c in money_cols:
        if c not in ('Perfect Window', main_col):
            total_row[c] = per_window[c].sum()
    per_window['window'] = per_window['window'].astype(str)
    per_window = pd.concat([per_window, pd.DataFrame([total_row])], ignore_index=True)
    with st.expander(f"📋 ราย Window: Perfect Window vs {who}", expanded=True):
        st.dataframe(per_window.style.format({**{c: '${:,.2f}' for c in money_cols}, '% ของเพดาน': '{:.1%}'}, na_rep='–'),
                     width="stretch", hide_index=True)
        st.caption("ตารางคิดทีละ Window (เริ่มใหม่ทุก Window) · ตัวเลขด้านบนเป็นแบบต่อกันจนจบเป็นเส้นเดียว "
                   "จึงต่างกันเล็กน้อยตรงรอยต่อระหว่าง Window")

def render_simulation_tabs():
    st.write("---")
    render_methodology_expander()
    
    st.divider()
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f"### 🚀 Batch Runner")
        st.caption("เลือก Tickers ในหน้า Settings แล้วกดปุ่มด้านขวาเพื่อเริ่มคำนวณ")
    with c2:
        if st.button("🚀 Start One-Click Loop All", type="primary", use_container_width=True):
            execute_batch_processing()
            
    # ! GOAL Step 1: Export JSON Feature
    if st.session_state.batch_results:
        render_export_section(st.session_state.batch_results, key="export_hybrid", file_name="encoded_strings.json")

        tickers = list(st.session_state.batch_results.keys())
        st.write(f"✅ Results available for: {', '.join(tickers)}")
        tabs = st.tabs([f"📈 {t}" for t in tickers])
        for tab, ticker in zip(tabs, tickers):
            with tab:
                data = st.session_state.batch_results[ticker]
                if Strategy.PERFECT_WINDOW in data["sim_results"]:
                    render_chained_vs_perfect_window(
                        data, Strategy.HYBRID_MULTI_MUTATION, "Hybrid", "final_net",
                        extra_cols={'original_net': 'Original DNA net'},
                        secondary=(Strategy.ORIGINAL_DNA, "Original DNA (ก่อนกลายพันธุ์)")
                    )
                render_single_ticker_result(
                    ticker, data,
                    ceiling=(Strategy.PERFECT_WINDOW, "Perfect Window (ต่อกัน)"),
                    show_metrics=Strategy.PERFECT_WINDOW not in data["sim_results"]
                )
    else:
        st.info("ยังไม่มีผลลัพธ์ กรุณากดปุ่ม Start")

SPECTRUM_MODE_HELP: Dict[str, str] = {
    "best10": "เลือก DNA ที่ net สูงสุด 1 ใน 10 preset ของชุดที่เลือก (preset ผ่านการปรับปรุงมาแล้ว ไม่กลายพันธุ์ต่อ)",
    "tuned": "ลองทุกความถี่ k = 1 … ความยาว Window และทุกจังหวะ (phase) แล้วคอมไพล์ pattern ที่ net สูงสุดเป็น DNA",
    "compiled": "คำนวณจุด act ที่ให้ net สูงสุดจริงของ Window (Perfect Foresight) แล้วคอมไพล์เป็น DNA = 100% ของเพดาน",
    "fast": "เหมือนโหมด 2 (ลองทุก k และ phase) แต่เร็วกว่า: ค้นหาด้วย numba รอบเดียว · หยิบ DNA จากสมุด code สำเร็จรูป "
            "(Window ≤ 30 แท่ง ไม่ต้องคอมไพล์) · ให้ 20 preset ร่วมแข่ง -> ได้ไม่น้อยกว่าโหมด 2 ทุก Window "
            "· phase_dependence = เฉลี่ยทุก phase ÷ ค่าที่เลือก (ต่ำ = ได้เพราะจังหวะดี)",
}

def window_options_table(prices_window: np.ndarray, spectrum_codes: List[Tuple[str, str]], row: pd.Series) -> Tuple[pd.DataFrame, float]:
    """ทุกตัวเลือกของ 1 window: 10 preset, ความถี่ที่จูนแล้ว และเพดาน พร้อม % ของเพดาน
    chosen = ตัวที่โหมดที่รันเลือก (best10 = rank ที่ชนะ, tuned/fast = ตัวจูน, fast ที่ preset ชนะ = แถว preset, compiled = เพดาน)"""
    length = len(prices_window)
    max_actions = generate_actions_perfect_foresight(prices_window.tolist()).astype(np.int32)
    max_net = float(_calculate_net_profit_numba(max_actions, prices_window))
    chosen_rank = int(row['spectrum_rank']) if 'spectrum_rank' in row.index and pd.notna(row.get('spectrum_rank')) else None
    preset_pick = 'source' in row.index and row.get('source') == 'preset'
    tuned_mode = chosen_rank is None and not preset_pick and 'k' in row.index and pd.notna(row.get('k'))
    rows = []
    for rank, (label, code) in enumerate(spectrum_codes, start=1):
        actions = spectrum_actions(code, length)[1]
        rows.append({"ตัวเลือก": f"R{rank} · {label.split(' ·')[0]}", "net": float(_calculate_net_profit_numba(actions, prices_window)),
                     "act": int(actions.sum()), "กลุ่ม": "10 preset", "chosen": rank == chosen_rank})
    k, phase, net, actions = find_best_period_for_window(prices_window)
    rows.append({"ตัวเลือก": f"จูนแล้ว · act ทุก {k} แท่ง", "net": float(net), "act": int(actions.sum()),
                 "กลุ่ม": "ความถี่ที่จูนแล้ว", "chosen": tuned_mode})
    if preset_pick:
        rows.append({"ตัวเลือก": f"โหมดนี้ · {row['spectrum_label']}", "net": float(row['net']), "act": int(row['acts']),
                     "กลุ่ม": "โหมดนี้เลือก", "chosen": True})
    rows.append({"ตัวเลือก": "เพดาน · Perfect Window", "net": max_net, "act": int(max_actions.sum()),
                 "กลุ่ม": "เพดาน", "chosen": chosen_rank is None and not tuned_mode and not preset_pick})
    df = pd.DataFrame(rows)
    df["pct"] = df["net"] / max_net if max_net > 0 else 0.0
    df.loc[df["chosen"], "กลุ่ม"] = "โหมดนี้เลือก"
    df["ป้าย"] = [f"{p:.0%} · ${n:,.0f} · act {a} ครั้ง" for p, n, a in zip(df["pct"], df["net"], df["act"])]
    return df, max_net

def render_spectrum_curve_chart(prices_window: np.ndarray, spectrum_codes: List[Tuple[str, str]], row: pd.Series):
    """กราฟแท่งของ 1 window: แต่ละตัวเลือกได้กี่ % ของเพดาน (แท่งยาว = ดี)"""
    import altair as alt
    df, max_net = window_options_table(prices_window, spectrum_codes, row)
    colors = alt.Scale(domain=["10 preset", "ความถี่ที่จูนแล้ว", "เพดาน", "โหมดนี้เลือก"],
                       range=["#94a3b8", "#38bdf8", "#a78bfa", "#10b981"])
    base = alt.Chart(df).encode(
        y=alt.Y("ตัวเลือก:N", sort=list(df["ตัวเลือก"]), title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("pct:Q", title="ได้กี่ % ของเพดาน (แท่งยาว = ดี · 100% = เพดาน)",
                axis=alt.Axis(format="%", values=[0, 0.2, 0.4, 0.6, 0.8, 1.0]), scale=alt.Scale(domain=[0, 1.45], nice=False)),
    )
    bars = base.mark_bar(cornerRadiusEnd=3).encode(
        color=alt.Color("กลุ่ม:N", scale=colors, legend=alt.Legend(title=None, orient="bottom")),
        tooltip=["ตัวเลือก", alt.Tooltip("pct:Q", format=".1%", title="% ของเพดาน"),
                 alt.Tooltip("net:Q", format=",.2f", title="net ($)"), alt.Tooltip("act:Q", title="act (ครั้ง)")],
    )
    labels = base.mark_text(align="left", dx=5, fontSize=12).encode(text="ป้าย:N")
    full = alt.Chart(pd.DataFrame({"pct": [1.0]})).mark_rule(color="#a78bfa", strokeDash=[4, 4]).encode(x="pct:Q")
    st.altair_chart((bars + labels + full).properties(height=34 * len(df)), width="stretch")
    st.caption(f"R1 = เปิดถี่สุด → R10 = เปิดห่างสุด · เพดาน = จุด act ที่ดีที่สุดของ window นี้ (\\${max_net:,.2f}) "
               "· ป้าย = % ของเพดาน · net · จำนวนครั้งที่ act · แท่งสีเขียว = ตัวที่โหมดนี้เลือก")

    with st.expander("🔬 ขั้นสูง: ลองทุกความถี่ (act ทุก k แท่ง)"):
        curve, _, _ = spectrum_curve(prices_window, spectrum_codes)
        curve["pct"] = curve["net"] / max_net if max_net > 0 else 0.0
        best = curve.loc[curve["pct"].idxmax()]
        line = alt.Chart(curve).mark_line(point=True, color="#38bdf8").encode(
            x=alt.X("k:Q", title="act ทุก k แท่ง  (ซ้าย = ถี่ · ขวา = ห่าง)"),
            y=alt.Y("pct:Q", title="% ของเพดาน", axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("k:Q", title="act ทุก k แท่ง"), alt.Tooltip("pct:Q", format=".1%", title="% ของเพดาน")])
        top = alt.Chart(pd.DataFrame([best])).mark_point(size=160, filled=True, color="#10b981").encode(x="k:Q", y="pct:Q")
        st.altair_chart((line + top).properties(height=240), width="stretch")
        st.caption(f"ความถี่ที่ดีที่สุดของ window นี้: act ทุก {int(best['k'])} แท่ง ได้ {best['pct']:.0%} ของเพดาน (เลือกจังหวะเริ่มที่ดีที่สุดแล้ว)")

def render_spectrum_tab():
    st.markdown("### 🌈 DNA Spectrum")
    st.info(
        f"ไม่สุ่ม {st.session_state.num_seeds:,} Seeds · ทุก Window เลือก actions ตาม**โหมด** แล้วจบ (ไม่กลายพันธุ์ต่อ) "
        "· ทุกโหมดได้ DNA code รูปแบบเดิมที่ Tracer / bot ถอดได้ · Mutation Rounds ใน Settings ใช้กับแท็บ Hybrid เท่านั้น"
    )
    mode = st.radio("โหมด", list(SPECTRUM_MODES.keys()), format_func=lambda m: SPECTRUM_MODES[m], horizontal=True, key="spectrum_mode")
    st.caption(SPECTRUM_MODE_HELP[mode])

    preset_names = list(DNA_SPECTRUM_PRESETS.keys())
    st.session_state.spectrum_preset = st.selectbox(
        "ชุด DNA Spectrum" + (" (ใช้เลือก DNA)" if mode == "best10" else " (ใช้เป็นจุดเทียบบนกราฟ)"),
        options=preset_names,
        index=preset_names.index(st.session_state.spectrum_preset) if st.session_state.spectrum_preset in preset_names else 0,
        help="Density = จำนวน gate ลดหลั่น (rate 10%, 5 mutations) · Periodic = 1 ทุก k แท่งเป๊ะ"
    )
    spectrum_codes = DNA_SPECTRUM_PRESETS[st.session_state.spectrum_preset]
    window_size = int(st.session_state.window_size)

    with st.expander(f"🧬 10 DNA ({SPECTRUM_LENGTH} บิต) · เรียงความถี่สั้น → ยาว", expanded=False):
        st.dataframe(build_spectrum_table(spectrum_codes, window_size), width="stretch", hide_index=True)
        st.caption(f"Window ยาว {window_size} แท่ง ใช้ {min(window_size, SPECTRUM_LENGTH)} บิตแรกของแต่ละ DNA (คอลัมน์ active_first_{window_size})")
    if mode == "best10" and window_size > SPECTRUM_LENGTH:
        st.warning(f"⚠️ Window {window_size} แท่ง ยาวกว่า {SPECTRUM_LENGTH}: บิตหลังตำแหน่ง {SPECTRUM_LENGTH} เป็นค่าสุ่มต่อจาก seed ไม่อยู่ในสเปกตรัม")

    c1, c2 = st.columns([3, 1])
    with c1:
        st.caption("เลือก Tickers / Timeframe / Window ในหน้า Settings แล้วกดปุ่มด้านขวา · โหมด 2–3 (และโหมด 4 ที่ Window ยาวกว่า 30 แท่ง) ครั้งแรกต้องสร้างตาราง seed (~3 วินาที)")
    with c2:
        if st.button("🌈 Run Spectrum", type="primary", width="stretch", key="btn_run_spectrum"):
            execute_batch_processing(spectrum_codes=spectrum_codes, spectrum_mode=mode)

    results = st.session_state.spectrum_results
    if not results:
        st.info("ยังไม่มีผลลัพธ์ กรุณากดปุ่ม Run Spectrum")
        return

    run_mode = next(iter(results.values())).get("spectrum_mode") or "best10"
    if run_mode != mode:
        st.caption(f"ผลด้านล่างมาจากโหมด {SPECTRUM_MODES[run_mode]} · กด Run Spectrum เพื่อรันโหมดที่เลือกใหม่")
    render_export_section(
        results, key="export_spectrum", file_name="encoded_strings_spectrum.json",
        extra_metadata={"source": "dna_spectrum", "spectrum_mode": run_mode, "spectrum_preset": st.session_state.spectrum_preset}
    )
    tickers = list(results.keys())
    st.write(f"✅ Results available for: {', '.join(tickers)}")
    tabs = st.tabs([f"🌈 {t}" for t in tickers])
    for tab, ticker in zip(tabs, tickers):
        with tab:
            data = results[ticker]
            df_windows = data["df_windows"]
            if df_windows.empty:
                st.warning("ไม่มี Window ที่คำนวณได้"); continue
            strategy = SPECTRUM_STRATEGY[run_mode]
            render_chained_vs_perfect_window(data, strategy, "DNA", "net", extra_cols={'spectrum_label': 'DNA'})
            if 'source' in df_windows.columns:
                where = df_windows['source'].value_counts()
                names = {"codebook": "สมุด code", "compiled": "คอมไพล์", "preset": "preset ชนะ"}
                dep = df_windows['phase_dependence'].median() if 'phase_dependence' in df_windows.columns else np.nan
                st.caption("ที่มาของ DNA: " + " · ".join(f"{names.get(s, s)} {n} Window" for s, n in where.items())
                           + (f" · phase_dependence มัธยฐาน {dep:.2f}" if pd.notna(dep) else ""))
            if 'spectrum_rank' in df_windows.columns:
                wins = df_windows.groupby(['spectrum_rank', 'spectrum_label']).size().reset_index(name='windows_won')
                st.write("🏆 **DNA ที่ชนะในแต่ละ Window (นับตาม rank)**")
                st.dataframe(wins, width="stretch", hide_index=True)

            st.write("📊 **เทียบทีละ Window: แต่ละตัวเลือกได้กี่ % ของเพดาน**")
            win_no = st.selectbox("Window", df_windows['window'].tolist(), key=f"spectrum_curve_win_{ticker}",
                                  format_func=lambda w, d=df_windows: f"Window {w} · {d.loc[d['window'] == w, 'timeline'].iloc[0]}")
            row = df_windows.loc[df_windows['window'] == win_no].iloc[0]
            ws = data.get("window_size", window_size)
            prices_all = data.get("prices")
            if prices_all is not None:
                render_spectrum_curve_chart(prices_all[(win_no - 1) * ws: win_no * ws], spectrum_codes, row)

            render_single_ticker_result(
                ticker, data, key_prefix="spectrum_",
                hybrid=(strategy, strategy),
                original=None,
                ceiling=(Strategy.PERFECT_WINDOW, "Perfect Window (ต่อกัน)"),
                show_metrics=False
            )

def render_classifier_tab():
    st.markdown("### 🔬 Spectrum Classifier")
    st.info(
        "ใส่หุ้น เลือก Timeframe กับจำนวนวัน แล้วกดวิเคราะห์ · เครื่องมือจะดึงราคา วัดว่าราคา **ไหลต่อ / ดึงกลับ / สุ่มเดิน** "
        "แล้วรัน DNA Spectrum ทั้ง 20 ชุด (Density + Periodic) ผ่าน gated ledger เพื่อหาย่านความถี่ที่เก็บ Eₙ ได้ดีที่สุด"
    )
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    ticker = c1.text_input("หุ้น (Ticker)", value="QBTS", key="clf_ticker").strip().upper()
    interval = c2.selectbox("Timeframe", list(CLASSIFIER_INTERVALS.keys()), index=0, key="clf_interval")
    max_days = CLASSIFIER_INTERVALS[interval]["max_days"]
    days = c3.number_input("ย้อนหลัง (วัน)", min_value=5, max_value=max_days, value=min(30, max_days), key="clf_days",
                           help=f"yfinance ดึง {interval} ย้อนหลังได้ไม่เกิน {max_days} วัน")
    c4.write(""); c4.write("")
    if c4.button("🔬 วิเคราะห์", type="primary", width="stretch", key="clf_run"):
        if not ticker:
            st.error("กรุณาใส่ Ticker")
        else:
            with st.spinner(f"กำลังดึง {ticker} {interval} ย้อนหลัง {days} วัน..."):
                data = get_classifier_prices(ticker, interval, int(days))
            if len(data) < 30:
                st.session_state.classifier_result = None
                st.error(f"ดึงข้อมูล {ticker} ไม่ได้ หรือข้อมูลน้อยเกินไป ({len(data)} แท่ง)")
            else:
                result = classify_spectrum(data['Close'], intraday=interval != "1d")
                result.update({"ticker": ticker, "interval": interval, "days": int(days), "closes": data['Close']})
                st.session_state.classifier_result = result

    result = st.session_state.get("classifier_result")
    if not result:
        st.caption("ยังไม่มีผลลัพธ์ · ตัวอย่าง: QBTS · 15m · 30 วัน")
        return

    closes = result["closes"]
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ข้อมูล", f"{len(closes):,} แท่ง", f"{result['sessions']} วันทำการ", delta_color="off")
    m2.metric("ราคา", f"${closes.iloc[-1]:,.2f}", f"{closes.iloc[-1] / closes.iloc[0] - 1:+.1%} จาก ${closes.iloc[0]:,.2f}")
    m3.metric("Eₙ ถ้า act ทุกแท่ง", f"${result['baseline']:,.2f}", "เกณฑ์ (ratio = 1.00)", delta_color="off")
    periodic_name, density_name = list(DNA_SPECTRUM_PRESETS.keys())[1], list(DNA_SPECTRUM_PRESETS.keys())[0]
    best_row = result["presets"][periodic_name]["table"].iloc[result["presets"][periodic_name]["best"]]
    m4.metric("ย่านที่ดีที่สุด (Periodic)", f"Rank {best_row['rank']} · {best_row['label'].split(' ·')[0]}",
              f"{best_row['เต็มช่วง']:.2f}× ของ act ทุกแท่ง")
    if result["n_rounds"] < 200:
        st.warning("⚠️ ข้อมูลน้อยกว่า 200 แท่ง: DNA 100 บิตวนได้ไม่ถึง 2 รอบ ย่านความถี่ยาว (k ≥ 50) จะไม่น่าเชื่อถือ")
    st.line_chart(closes.rename(f"{result['ticker']} {result['interval']}"), height=220)

    st.markdown("#### 1) พฤติกรรมราคา (Variance Ratio)")
    st.caption("VR(q) เทียบความแปรปรวนของผลตอบแทน q แท่งกับ q × 1 แท่ง · > 1 ราคาไหลต่อ → รอนานขึ้นได้เปรียบ · < 1 ราคาดึงกลับ → act ถี่ได้เปรียบ · ≈ 1 สุ่มเดิน → ความถี่ไม่มีผล")
    st.dataframe(result["vr"], width="stretch", hide_index=True)

    st.markdown("#### 2) ย่านความถี่ DNA ที่ดีที่สุด")
    st.caption(
        "ตัวเลข = Eₙ ÷ Eₙ ของการ act ทุกแท่ง ในข้อมูลชุดเดียวกัน (> 1 = ดีกว่า) · เฉลี่ยจากการเลื่อนจุดเริ่ม DNA ครบ 100 ตำแหน่ง "
        "เพื่อตัดโชคเรื่องจังหวะ · 'ต่ำสุดทุกการตรวจ' = ค่าที่แย่ที่สุดในทุกแบบ ยิ่งสูงยิ่งเชื่อถือได้"
    )
    for preset_name in [periodic_name, density_name]:
        info = result["presets"][preset_name]
        df = info["table"]
        best, robust = df.iloc[info["best"]], df.iloc[info["robust"]]
        st.markdown(f"**{preset_name}** · ดีที่สุดเต็มช่วง: **Rank {best['rank']} ({best['label']})** "
                    f"{best['เต็มช่วง']:.2f}× · เชื่อถือได้ที่สุด: **Rank {robust['rank']} ({robust['label']})** "
                    f"ต่ำสุด {robust['ต่ำสุดทุกการตรวจ']:.2f}×")
        show_cols = ["rank", "label", "gates"] + result["checks"] + ["ต่ำสุดทุกการตรวจ", "E เฉลี่ย", "E (phase 0)"]
        styled = (df[show_cols].style
                  .format({c: "{:.2f}" for c in result["checks"] + ["ต่ำสุดทุกการตรวจ"]} | {"E เฉลี่ย": "${:.2f}", "E (phase 0)": "${:.2f}"})
                  .highlight_max(subset=["เต็มช่วง", "ต่ำสุดทุกการตรวจ"], color="rgba(16,185,129,0.35)"))
        st.dataframe(styled, width="stretch", hide_index=True)
        picks = [("ดีที่สุดเต็มช่วง", best)] + ([("เชื่อถือได้ที่สุด", robust)] if robust['rank'] != best['rank'] else [])
        st.code("\n".join(f"{tag} · Rank {row['rank']} ({row['label']}): {row['dna_code']}" for tag, row in picks), language="text")

    st.caption(
        f"ข้อมูลจาก yfinance (เฉพาะชั่วโมงปกติ) · Fix_c = ${CLASSIFIER_FIX_C:,.0f} · E (phase 0) = รัน DNA จากบิตแรกตามปกติ "
        "· ผลนี้มาจากข้อมูลย้อนหลัง ไม่รับประกันอนาคต และข้อมูลไม่กี่สัปดาห์มีความคลาดเคลื่อนสูง"
    )

# ! GOAL: Restore Full Tracer (Decode + Encode)
def render_tracer_tab():
    st.markdown("### 🔍 Action Sequence Tracer & Encoder")
    st.info("เครื่องมือนี้ใช้สำหรับ 1. **ถอดรหัส (Decode)** String เพื่อจำลองผลลัพธ์ และ 2. **เข้ารหัส (Encode)** พารามิเตอร์เพื่อสร้าง String")

    st.markdown("---")
    st.markdown("#### 1. ถอดรหัส (Decode) String")

    encoded_string = st.text_input(
        "ป้อน Encoded String ที่นี่:",
        "26021034252903219354832053493",
        help="สตริงที่เข้ารหัสพารามิเตอร์ต่างๆ เช่น action_length, mutation_rate, dna_seed, และ mutation_seeds",
        key="decoder_input"
    )

    if st.button("Trace & Simulate", type="primary", key="tracer_button"):
        if not encoded_string:
            st.warning("กรุณาป้อน Encoded String")
        else:
            with st.spinner(f"กำลังถอดรหัสและจำลองสำหรับ: {encoded_string[:20]}..."):
                try:
                    tracer = SimulationTracer(encoded_string=encoded_string)
                    st.success("ถอดรหัสสำเร็จ!")
                    st.code(str(tracer), language='bash')
                    final_actions = tracer.run()
                    st.write("---")
                    st.markdown("#### 🎉 ผลลัพธ์ Action Sequence สุดท้าย:")
                    st.dataframe(pd.DataFrame(final_actions, columns=['Action']), use_container_width=True)
                    st.write("Raw Array:")
                    st.code(str(final_actions))
                except ValueError as e:
                    st.error(f"❌ เกิดข้อผิดพลาดในการประมวลผล: {e}")

    st.divider()

    st.markdown("#### 2. เข้ารหัส (Encode) พารามิเตอร์")
    st.write("ป้อนพารามิเตอร์เพื่อสร้าง Encoded String สำหรับการทดลองซ้ำ")

    col1, col2 = st.columns(2)
    with col1:
        action_length_input = st.number_input("Action Length", min_value=1, value=60, key="enc_len", help="ความยาวของ action sequence")
        dna_seed_input = st.number_input("DNA Seed", min_value=0, value=900, format="%d", key="enc_dna", help="Seed สำหรับสร้าง DNA ดั้งเดิม")
    with col2:
        mutation_rate_input = st.number_input("Mutation Rate (%)", min_value=0, value=10, key="enc_rate", help="อัตราการกลายพันธุ์เป็นเปอร์เซ็นต์ (เช่น 5 สำหรับ 5%)")
        mutation_seeds_str = st.text_input(
            "Mutation Seeds (คั่นด้วยจุลภาค ,)",
            "899, 530, 35, 814, 646",
            key="enc_seeds_str",
            help="ชุดของ Seed สำหรับการกลายพันธุ์แต่ละรอบ คั่นด้วยเครื่องหมายจุลภาค"
        )

    meta_col1, meta_col2 = st.columns(2)
    with meta_col1:
        manual_timeframe_input = st.selectbox(
            "Encode Timeframe",
            options=list(TIMEFRAME_OPTIONS.keys()),
            index=list(TIMEFRAME_OPTIONS.keys()).index(st.session_state.selected_timeframe) if st.session_state.selected_timeframe in TIMEFRAME_OPTIONS else 0,
            key="enc_timeframe",
            help="ใช้เฉพาะเมื่อ Unix Start Timestamp มากกว่า 0 (ใส่ timeline metadata)"
        )
    with meta_col2:
        manual_start_timestamp_input = st.number_input(
            "Unix Start Timestamp (0 = legacy/ไม่มี timeline metadata)",
            min_value=0,
            value=0,
            format="%d",
            key="enc_start_ts"
        )

    if st.button("Encode Parameters", key="encoder_button"):
        try:
            if mutation_seeds_str.strip():
                mutation_seeds_list = [int(s.strip()) for s in mutation_seeds_str.split(',')]
            else:
                mutation_seeds_list = []

            generated_string = SimulationTracer.encode(
                action_length=int(action_length_input),
                mutation_rate=int(mutation_rate_input),
                dna_seed=int(dna_seed_input),
                mutation_seeds=mutation_seeds_list,
                timeframe_minutes=get_timeframe_minutes(manual_timeframe_input) if int(manual_start_timestamp_input) > 0 else None,
                start_timestamp=int(manual_start_timestamp_input) if int(manual_start_timestamp_input) > 0 else None
            )

            st.success("เข้ารหัสสำเร็จ! สามารถคัดลอก String ด้านล่างไปใช้ได้")
            st.code(generated_string, language='text')

        except (ValueError, TypeError) as e:
            st.error(f"❌ เกิดข้อผิดพลาด: กรุณาตรวจสอบว่า Mutation Seeds เป็นตัวเลขที่คั่นด้วยจุลภาคเท่านั้น ({e})")

# ==============================================================================
# 5b. Tab 6: DNA Spectrum Explorer (100% Template HTML + Dynamic Tickers)
# ==============================================================================
DNA_SPECTRUM_EXPLORER_TEMPLATE_HTML: str = '<!DOCTYPE html>\n<html lang="th" class="dark">\n<head>\n  <meta charset="UTF-8">\n  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0">\n  <title>DNA Spectrum Lab — __TITLE_TICKERS__ (__TF_SHORT__ × __WIN_MODE_TITLE__)</title>\n  <!-- Google Fonts: Inter, Noto Sans Thai, JetBrains Mono -->\n  <link rel="preconnect" href="https://fonts.googleapis.com">\n  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Noto+Sans+Thai:wght@400;500;600;700&display=swap" rel="stylesheet">\n  <style>*,:after,:before{--tw-border-spacing-x:0;--tw-border-spacing-y:0;--tw-translate-x:0;--tw-translate-y:0;--tw-rotate:0;--tw-skew-x:0;--tw-skew-y:0;--tw-scale-x:1;--tw-scale-y:1;--tw-pan-x: ;--tw-pan-y: ;--tw-pinch-zoom: ;--tw-scroll-snap-strictness:proximity;--tw-gradient-from-position: ;--tw-gradient-via-position: ;--tw-gradient-to-position: ;--tw-ordinal: ;--tw-slashed-zero: ;--tw-numeric-figure: ;--tw-numeric-spacing: ;--tw-numeric-fraction: ;--tw-ring-inset: ;--tw-ring-offset-width:0px;--tw-ring-offset-color:#fff;--tw-ring-color:rgba(59,130,246,.5);--tw-ring-offset-shadow:0 0 #0000;--tw-ring-shadow:0 0 #0000;--tw-shadow:0 0 #0000;--tw-shadow-colored:0 0 #0000;--tw-blur: ;--tw-brightness: ;--tw-contrast: ;--tw-grayscale: ;--tw-hue-rotate: ;--tw-invert: ;--tw-saturate: ;--tw-sepia: ;--tw-drop-shadow: ;--tw-backdrop-blur: ;--tw-backdrop-brightness: ;--tw-backdrop-contrast: ;--tw-backdrop-grayscale: ;--tw-backdrop-hue-rotate: ;--tw-backdrop-invert: ;--tw-backdrop-opacity: ;--tw-backdrop-saturate: ;--tw-backdrop-sepia: ;--tw-contain-size: ;--tw-contain-layout: ;--tw-contain-paint: ;--tw-contain-style: }::backdrop{--tw-border-spacing-x:0;--tw-border-spacing-y:0;--tw-translate-x:0;--tw-translate-y:0;--tw-rotate:0;--tw-skew-x:0;--tw-skew-y:0;--tw-scale-x:1;--tw-scale-y:1;--tw-pan-x: ;--tw-pan-y: ;--tw-pinch-zoom: ;--tw-scroll-snap-strictness:proximity;--tw-gradient-from-position: ;--tw-gradient-via-position: ;--tw-gradient-to-position: ;--tw-ordinal: ;--tw-slashed-zero: ;--tw-numeric-figure: ;--tw-numeric-spacing: ;--tw-numeric-fraction: ;--tw-ring-inset: ;--tw-ring-offset-width:0px;--tw-ring-offset-color:#fff;--tw-ring-color:rgba(59,130,246,.5);--tw-ring-offset-shadow:0 0 #0000;--tw-ring-shadow:0 0 #0000;--tw-shadow:0 0 #0000;--tw-shadow-colored:0 0 #0000;--tw-blur: ;--tw-brightness: ;--tw-contrast: ;--tw-grayscale: ;--tw-hue-rotate: ;--tw-invert: ;--tw-saturate: ;--tw-sepia: ;--tw-drop-shadow: ;--tw-backdrop-blur: ;--tw-backdrop-brightness: ;--tw-backdrop-contrast: ;--tw-backdrop-grayscale: ;--tw-backdrop-hue-rotate: ;--tw-backdrop-invert: ;--tw-backdrop-opacity: ;--tw-backdrop-saturate: ;--tw-backdrop-sepia: ;--tw-contain-size: ;--tw-contain-layout: ;--tw-contain-paint: ;--tw-contain-style: }/*! tailwindcss v3.4.19 | MIT License | https://tailwindcss.com*/*,:after,:before{box-sizing:border-box;border:0 solid #e5e7eb}:after,:before{--tw-content:""}:host,html{line-height:1.5;-webkit-text-size-adjust:100%;-moz-tab-size:4;-o-tab-size:4;tab-size:4;font-family:Inter,Noto Sans Thai,-apple-system,BlinkMacSystemFont,sans-serif;font-feature-settings:normal;font-variation-settings:normal;-webkit-tap-highlight-color:transparent}body{margin:0;line-height:inherit}hr{height:0;color:inherit;border-top-width:1px}abbr:where([title]){-webkit-text-decoration:underline dotted;text-decoration:underline dotted}h1,h2,h3,h4,h5,h6{font-size:inherit;font-weight:inherit}a{color:inherit;text-decoration:inherit}b,strong{font-weight:bolder}code,kbd,pre,samp{font-family:JetBrains Mono,monospace;font-feature-settings:normal;font-variation-settings:normal;font-size:1em}small{font-size:80%}sub,sup{font-size:75%;line-height:0;position:relative;vertical-align:baseline}sub{bottom:-.25em}sup{top:-.5em}table{text-indent:0;border-color:inherit;border-collapse:collapse}button,input,optgroup,select,textarea{font-family:inherit;font-feature-settings:inherit;font-variation-settings:inherit;font-size:100%;font-weight:inherit;line-height:inherit;letter-spacing:inherit;color:inherit;margin:0;padding:0}button,select{text-transform:none}button,input:where([type=button]),input:where([type=reset]),input:where([type=submit]){-webkit-appearance:button;background-color:transparent;background-image:none}:-moz-focusring{outline:auto}:-moz-ui-invalid{box-shadow:none}progress{vertical-align:baseline}::-webkit-inner-spin-button,::-webkit-outer-spin-button{height:auto}[type=search]{-webkit-appearance:textfield;outline-offset:-2px}::-webkit-search-decoration{-webkit-appearance:none}::-webkit-file-upload-button{-webkit-appearance:button;font:inherit}summary{display:list-item}blockquote,dd,dl,figure,h1,h2,h3,h4,h5,h6,hr,p,pre{margin:0}fieldset{margin:0}fieldset,legend{padding:0}menu,ol,ul{list-style:none;margin:0;padding:0}dialog{padding:0}textarea{resize:vertical}input::-moz-placeholder,textarea::-moz-placeholder{opacity:1;color:#9ca3af}input::placeholder,textarea::placeholder{opacity:1;color:#9ca3af}[role=button],button{cursor:pointer}:disabled{cursor:default}audio,canvas,embed,iframe,img,object,svg,video{display:block;vertical-align:middle}img,video{max-width:100%;height:auto}[hidden]:where(:not([hidden=until-found])){display:none}.container{width:100%}@media (min-width:640px){.container{max-width:640px}}@media (min-width:768px){.container{max-width:768px}}@media (min-width:1024px){.container{max-width:1024px}}@media (min-width:1280px){.container{max-width:1280px}}@media (min-width:1536px){.container{max-width:1536px}}.pointer-events-none{pointer-events:none}.visible{visibility:visible}.invisible{visibility:hidden}.fixed{position:fixed}.absolute{position:absolute}.relative{position:relative}.sticky{position:sticky}.inset-y-0{top:0;bottom:0}.inset-y-1{top:.25rem;bottom:.25rem}.bottom-0{bottom:0}.left-0{left:0}.left-1\\/2{left:50%}.left-8{left:2rem}.right-1{right:.25rem}.top-0{top:0}.top-1\\/4{top:25%}.top-2\\/4{top:50%}.top-3\\/4{top:75%}.z-10{z-index:10}.z-20{z-index:20}.z-30{z-index:30}.z-40{z-index:40}.z-50{z-index:50}.mx-auto{margin-left:auto;margin-right:auto}.-mt-5{margin-top:-1.25rem}.mb-2{margin-bottom:.5rem}.mb-3{margin-bottom:.75rem}.ml-1{margin-left:.25rem}.ml-2{margin-left:.5rem}.mt-0\\.5{margin-top:.125rem}.mt-1{margin-top:.25rem}.mt-3{margin-top:.75rem}.block{display:block}.inline-block{display:inline-block}.inline{display:inline}.flex{display:flex}.inline-flex{display:inline-flex}.grid{display:grid}.hidden{display:none}.h-0\\.5{height:.125rem}.h-1\\.5{height:.375rem}.h-2{height:.5rem}.h-2\\.5{height:.625rem}.h-3{height:.75rem}.h-3\\.5{height:.875rem}.h-4{height:1rem}.h-44{height:11rem}.h-6{height:1.5rem}.h-64{height:16rem}.h-8{height:2rem}.h-9{height:2.25rem}.h-full{height:100%}.min-h-screen{min-height:100vh}.w-1{width:.25rem}.w-14{width:3.5rem}.w-16{width:4rem}.w-2{width:.5rem}.w-2\\.5{width:.625rem}.w-3{width:.75rem}.w-3\\.5{width:.875rem}.w-36{width:9rem}.w-4{width:1rem}.w-5{width:1.25rem}.w-9{width:2.25rem}.w-full{width:100%}.min-w-\\[520px\\]{min-width:520px}.max-w-7xl{max-width:80rem}.flex-1{flex:1 1 0%}.shrink-0{flex-shrink:0}.-translate-x-1\\/2{--tw-translate-x:-50%}.-translate-x-1\\/2,.-translate-y-1\\/2{transform:translate(var(--tw-translate-x),var(--tw-translate-y)) rotate(var(--tw-rotate)) skewX(var(--tw-skew-x)) skewY(var(--tw-skew-y)) scaleX(var(--tw-scale-x)) scaleY(var(--tw-scale-y))}.-translate-y-1\\/2{--tw-translate-y:-50%}.translate-y-1\\/2{--tw-translate-y:50%}.transform,.translate-y-1\\/2{transform:translate(var(--tw-translate-x),var(--tw-translate-y)) rotate(var(--tw-rotate)) skewX(var(--tw-skew-x)) skewY(var(--tw-skew-y)) scaleX(var(--tw-scale-x)) scaleY(var(--tw-scale-y))}.cursor-default{cursor:default}.cursor-pointer{cursor:pointer}.resize{resize:both}.grid-cols-1{grid-template-columns:repeat(1,minmax(0,1fr))}.grid-cols-10{grid-template-columns:repeat(10,minmax(0,1fr))}.grid-cols-11{grid-template-columns:repeat(11,minmax(0,1fr))}.grid-cols-2{grid-template-columns:repeat(2,minmax(0,1fr))}.flex-col{flex-direction:column}.flex-wrap{flex-wrap:wrap}.items-center{align-items:center}.items-baseline{align-items:baseline}.justify-end{justify-content:flex-end}.justify-center{justify-content:center}.justify-between{justify-content:space-between}.gap-0\\.5{gap:.125rem}.gap-1{gap:.25rem}.gap-1\\.5{gap:.375rem}.gap-2{gap:.5rem}.gap-2\\.5{gap:.625rem}.gap-3{gap:.75rem}.gap-4{gap:1rem}.gap-x-3{-moz-column-gap:.75rem;column-gap:.75rem}.gap-x-4{-moz-column-gap:1rem;column-gap:1rem}.gap-y-1\\.5{row-gap:.375rem}.gap-y-2{row-gap:.5rem}.space-y-2>:not([hidden])~:not([hidden]){--tw-space-y-reverse:0;margin-top:calc(.5rem*(1 - var(--tw-space-y-reverse)));margin-bottom:calc(.5rem*var(--tw-space-y-reverse))}.space-y-2\\.5>:not([hidden])~:not([hidden]){--tw-space-y-reverse:0;margin-top:calc(.625rem*(1 - var(--tw-space-y-reverse)));margin-bottom:calc(.625rem*var(--tw-space-y-reverse))}.space-y-3>:not([hidden])~:not([hidden]){--tw-space-y-reverse:0;margin-top:calc(.75rem*(1 - var(--tw-space-y-reverse)));margin-bottom:calc(.75rem*var(--tw-space-y-reverse))}.space-y-4>:not([hidden])~:not([hidden]){--tw-space-y-reverse:0;margin-top:calc(1rem*(1 - var(--tw-space-y-reverse)));margin-bottom:calc(1rem*var(--tw-space-y-reverse))}.self-start{align-self:flex-start}.overflow-hidden{overflow:hidden}.overflow-visible{overflow:visible}.overflow-x-auto{overflow-x:auto}.truncate{overflow:hidden;text-overflow:ellipsis}.truncate,.whitespace-nowrap{white-space:nowrap}.rounded{border-radius:.25rem}.rounded-2xl{border-radius:1rem}.rounded-full{border-radius:9999px}.rounded-lg{border-radius:.5rem}.rounded-md{border-radius:.375rem}.rounded-sm{border-radius:.125rem}.rounded-xl{border-radius:.75rem}.rounded-r{border-top-right-radius:.25rem;border-bottom-right-radius:.25rem}.rounded-r-md{border-top-right-radius:.375rem;border-bottom-right-radius:.375rem}.border{border-width:1px}.border-2{border-width:2px}.border-x{border-left-width:1px;border-right-width:1px}.border-b{border-bottom-width:1px}.border-t{border-top-width:1px}.border-t-2{border-top-width:2px}.border-dashed{border-style:dashed}.border-amber-400{--tw-border-opacity:1;border-color:rgb(251 191 36/var(--tw-border-opacity,1))}.border-amber-400\\/80{border-color:rgba(251,191,36,.8)}.border-amber-500\\/30{border-color:rgba(245,158,11,.3)}.border-cyan-300{--tw-border-opacity:1;border-color:rgb(103 232 249/var(--tw-border-opacity,1))}.border-cyan-400{--tw-border-opacity:1;border-color:rgb(34 211 238/var(--tw-border-opacity,1))}.border-cyan-400\\/40{border-color:rgba(34,211,238,.4)}.border-cyan-400\\/50{border-color:rgba(34,211,238,.5)}.border-cyan-400\\/60{border-color:rgba(34,211,238,.6)}.border-cyan-400\\/70{border-color:rgba(34,211,238,.7)}.border-cyan-500\\/20{border-color:rgba(6,182,212,.2)}.border-cyan-500\\/30{border-color:rgba(6,182,212,.3)}.border-cyan-500\\/40{border-color:rgba(6,182,212,.4)}.border-emerald-400\\/50{border-color:rgba(52,211,153,.5)}.border-emerald-400\\/70{border-color:rgba(52,211,153,.7)}.border-emerald-500\\/30{border-color:rgba(16,185,129,.3)}.border-emerald-500\\/50{border-color:rgba(16,185,129,.5)}.border-emerald-500\\/60{border-color:rgba(16,185,129,.6)}.border-rose-500\\/30{border-color:rgba(244,63,94,.3)}.border-rose-500\\/50{border-color:rgba(244,63,94,.5)}.border-rose-500\\/60{border-color:rgba(244,63,94,.6)}.border-slate-600\\/50{border-color:rgba(71,85,105,.5)}.border-slate-900{--tw-border-opacity:1;border-color:rgb(15 23 42/var(--tw-border-opacity,1))}.border-slate-950{--tw-border-opacity:1;border-color:rgb(2 6 23/var(--tw-border-opacity,1))}.border-transparent{border-color:transparent}.border-tv-border{--tw-border-opacity:1;border-color:rgb(36 43 61/var(--tw-border-opacity,1))}.border-tv-border\\/70{border-color:rgba(36,43,61,.7)}.border-tv-borderLight{--tw-border-opacity:1;border-color:rgb(51 60 84/var(--tw-border-opacity,1))}.border-white{--tw-border-opacity:1;border-color:rgb(255 255 255/var(--tw-border-opacity,1))}.bg-amber-400{--tw-bg-opacity:1;background-color:rgb(251 191 36/var(--tw-bg-opacity,1))}.bg-amber-500\\/15{background-color:rgba(245,158,11,.15)}.bg-amber-500\\/20{background-color:rgba(245,158,11,.2)}.bg-cyan-300{--tw-bg-opacity:1;background-color:rgb(103 232 249/var(--tw-bg-opacity,1))}.bg-cyan-400{--tw-bg-opacity:1;background-color:rgb(34 211 238/var(--tw-bg-opacity,1))}.bg-cyan-400\\/20{background-color:rgba(34,211,238,.2)}.bg-cyan-400\\/25{background-color:rgba(34,211,238,.25)}.bg-cyan-400\\/40{background-color:rgba(34,211,238,.4)}.bg-cyan-500\\/10{background-color:rgba(6,182,212,.1)}.bg-cyan-500\\/15{background-color:rgba(6,182,212,.15)}.bg-cyan-500\\/20{background-color:rgba(6,182,212,.2)}.bg-cyan-500\\/25{background-color:rgba(6,182,212,.25)}.bg-emerald-400{--tw-bg-opacity:1;background-color:rgb(52 211 153/var(--tw-bg-opacity,1))}.bg-emerald-400\\/20{background-color:rgba(52,211,153,.2)}.bg-emerald-400\\/45{background-color:rgba(52,211,153,.45)}.bg-emerald-400\\/75{background-color:rgba(52,211,153,.75)}.bg-emerald-400\\/80{background-color:rgba(52,211,153,.8)}.bg-emerald-500\\/15{background-color:rgba(16,185,129,.15)}.bg-emerald-500\\/20{background-color:rgba(16,185,129,.2)}.bg-emerald-500\\/35{background-color:rgba(16,185,129,.35)}.bg-rose-500{--tw-bg-opacity:1;background-color:rgb(244 63 94/var(--tw-bg-opacity,1))}.bg-rose-500\\/15{background-color:rgba(244,63,94,.15)}.bg-rose-500\\/20{background-color:rgba(244,63,94,.2)}.bg-slate-300\\/80{background-color:rgba(203,213,225,.8)}.bg-slate-400\\/75{background-color:rgba(148,163,184,.75)}.bg-slate-500\\/45{background-color:rgba(100,116,139,.45)}.bg-slate-700\\/40{background-color:rgba(51,65,85,.4)}.bg-slate-950\\/95{background-color:rgba(2,6,23,.95)}.bg-tv-bg{--tw-bg-opacity:1;background-color:rgb(11 14 20/var(--tw-bg-opacity,1))}.bg-tv-bg\\/60{background-color:rgba(11,14,20,.6)}.bg-tv-bg\\/90{background-color:rgba(11,14,20,.9)}.bg-tv-card{--tw-bg-opacity:1;background-color:rgb(26 31 46/var(--tw-bg-opacity,1))}.bg-tv-panel{--tw-bg-opacity:1;background-color:rgb(19 23 34/var(--tw-bg-opacity,1))}.bg-tv-panel\\/30{background-color:rgba(19,23,34,.3)}.bg-tv-panel\\/50{background-color:rgba(19,23,34,.5)}.bg-tv-panel\\/60{background-color:rgba(19,23,34,.6)}.bg-white{--tw-bg-opacity:1;background-color:rgb(255 255 255/var(--tw-bg-opacity,1))}.bg-white\\/85{background-color:hsla(0,0%,100%,.85)}.bg-gradient-to-br{background-image:linear-gradient(to bottom right,var(--tw-gradient-stops))}.from-cyan-400{--tw-gradient-from:#22d3ee var(--tw-gradient-from-position);--tw-gradient-to:rgba(34,211,238,0) var(--tw-gradient-to-position);--tw-gradient-stops:var(--tw-gradient-from),var(--tw-gradient-to)}.to-emerald-500{--tw-gradient-to:#10b981 var(--tw-gradient-to-position)}.p-0\\.5{padding:.125rem}.p-1{padding:.25rem}.p-2{padding:.5rem}.p-2\\.5{padding:.625rem}.p-3{padding:.75rem}.p-3\\.5{padding:.875rem}.p-4{padding:1rem}.px-1\\.5{padding-left:.375rem;padding-right:.375rem}.px-2{padding-left:.5rem;padding-right:.5rem}.px-2\\.5{padding-left:.625rem;padding-right:.625rem}.px-3{padding-left:.75rem;padding-right:.75rem}.px-3\\.5{padding-left:.875rem;padding-right:.875rem}.py-0\\.5{padding-top:.125rem;padding-bottom:.125rem}.py-1{padding-top:.25rem;padding-bottom:.25rem}.py-1\\.5{padding-top:.375rem;padding-bottom:.375rem}.py-2{padding-top:.5rem;padding-bottom:.5rem}.py-2\\.5{padding-top:.625rem;padding-bottom:.625rem}.py-3{padding-top:.75rem;padding-bottom:.75rem}.pb-0\\.5{padding-bottom:.125rem}.pb-1{padding-bottom:.25rem}.pb-10{padding-bottom:2.5rem}.pb-2{padding-bottom:.5rem}.pb-3{padding-bottom:.75rem}.pl-8{padding-left:2rem}.pr-1{padding-right:.25rem}.pt-1{padding-top:.25rem}.pt-2{padding-top:.5rem}.pt-4{padding-top:1rem}.pt-8{padding-top:2rem}.text-left{text-align:left}.font-mono{font-family:JetBrains Mono,monospace}.text-\\[10px\\]{font-size:10px}.text-\\[11px\\]{font-size:11px}.text-\\[8px\\]{font-size:8px}.text-\\[9px\\]{font-size:9px}.text-lg{font-size:1.125rem;line-height:1.75rem}.text-sm{font-size:.875rem;line-height:1.25rem}.text-xl{font-size:1.25rem;line-height:1.75rem}.text-xs{font-size:.75rem;line-height:1rem}.font-bold{font-weight:700}.font-medium{font-weight:500}.font-normal{font-weight:400}.font-semibold{font-weight:600}.leading-none{line-height:1}.leading-tight{line-height:1.25}.tracking-tight{letter-spacing:-.025em}.text-amber-300{--tw-text-opacity:1;color:rgb(252 211 77/var(--tw-text-opacity,1))}.text-amber-400{--tw-text-opacity:1;color:rgb(251 191 36/var(--tw-text-opacity,1))}.text-cyan-200{--tw-text-opacity:1;color:rgb(165 243 252/var(--tw-text-opacity,1))}.text-cyan-300{--tw-text-opacity:1;color:rgb(103 232 249/var(--tw-text-opacity,1))}.text-cyan-400{--tw-text-opacity:1;color:rgb(34 211 238/var(--tw-text-opacity,1))}.text-emerald-300{--tw-text-opacity:1;color:rgb(110 231 183/var(--tw-text-opacity,1))}.text-emerald-400{--tw-text-opacity:1;color:rgb(52 211 153/var(--tw-text-opacity,1))}.text-rose-300{--tw-text-opacity:1;color:rgb(253 164 175/var(--tw-text-opacity,1))}.text-slate-100{--tw-text-opacity:1;color:rgb(241 245 249/var(--tw-text-opacity,1))}.text-slate-200{--tw-text-opacity:1;color:rgb(226 232 240/var(--tw-text-opacity,1))}.text-slate-300{--tw-text-opacity:1;color:rgb(203 213 225/var(--tw-text-opacity,1))}.text-slate-700{--tw-text-opacity:1;color:rgb(51 65 85/var(--tw-text-opacity,1))}.text-slate-800{--tw-text-opacity:1;color:rgb(30 41 59/var(--tw-text-opacity,1))}.text-slate-950{--tw-text-opacity:1;color:rgb(2 6 23/var(--tw-text-opacity,1))}.text-tv-dim{--tw-text-opacity:1;color:rgb(100 116 139/var(--tw-text-opacity,1))}.text-tv-muted{--tw-text-opacity:1;color:rgb(148 163 184/var(--tw-text-opacity,1))}.text-white{--tw-text-opacity:1;color:rgb(255 255 255/var(--tw-text-opacity,1))}.antialiased{-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}.opacity-60{opacity:.6}.opacity-75{opacity:.75}.opacity-80{opacity:.8}.shadow-lg{--tw-shadow:0 10px 15px -3px rgba(0,0,0,.1),0 4px 6px -4px rgba(0,0,0,.1);--tw-shadow-colored:0 10px 15px -3px var(--tw-shadow-color),0 4px 6px -4px var(--tw-shadow-color)}.shadow-lg,.shadow-md{box-shadow:var(--tw-ring-offset-shadow,0 0 #0000),var(--tw-ring-shadow,0 0 #0000),var(--tw-shadow)}.shadow-md{--tw-shadow:0 4px 6px -1px rgba(0,0,0,.1),0 2px 4px -2px rgba(0,0,0,.1);--tw-shadow-colored:0 4px 6px -1px var(--tw-shadow-color),0 2px 4px -2px var(--tw-shadow-color)}.shadow-sm{--tw-shadow:0 1px 2px 0 rgba(0,0,0,.05);--tw-shadow-colored:0 1px 2px 0 var(--tw-shadow-color);box-shadow:var(--tw-ring-offset-shadow,0 0 #0000),var(--tw-ring-shadow,0 0 #0000),var(--tw-shadow)}.shadow-amber-400\\/50{--tw-shadow-color:rgba(251,191,36,.5);--tw-shadow:var(--tw-shadow-colored)}.shadow-amber-500\\/20{--tw-shadow-color:rgba(245,158,11,.2);--tw-shadow:var(--tw-shadow-colored)}.shadow-cyan-400\\/50{--tw-shadow-color:rgba(34,211,238,.5);--tw-shadow:var(--tw-shadow-colored)}.shadow-cyan-500\\/20{--tw-shadow-color:rgba(6,182,212,.2);--tw-shadow:var(--tw-shadow-colored)}.shadow-white{--tw-shadow-color:#fff;--tw-shadow:var(--tw-shadow-colored)}.outline{outline-style:solid}.ring{--tw-ring-offset-shadow:var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);--tw-ring-shadow:var(--tw-ring-inset) 0 0 0 calc(3px + var(--tw-ring-offset-width)) var(--tw-ring-color)}.ring,.ring-1{box-shadow:var(--tw-ring-offset-shadow),var(--tw-ring-shadow),var(--tw-shadow,0 0 #0000)}.ring-1{--tw-ring-offset-shadow:var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);--tw-ring-shadow:var(--tw-ring-inset) 0 0 0 calc(1px + var(--tw-ring-offset-width)) var(--tw-ring-color)}.ring-2{--tw-ring-offset-shadow:var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);--tw-ring-shadow:var(--tw-ring-inset) 0 0 0 calc(2px + var(--tw-ring-offset-width)) var(--tw-ring-color);box-shadow:var(--tw-ring-offset-shadow),var(--tw-ring-shadow),var(--tw-shadow,0 0 #0000)}.ring-amber-400{--tw-ring-opacity:1;--tw-ring-color:rgb(251 191 36/var(--tw-ring-opacity,1))}.ring-cyan-300{--tw-ring-opacity:1;--tw-ring-color:rgb(103 232 249/var(--tw-ring-opacity,1))}.ring-cyan-400\\/50{--tw-ring-color:rgba(34,211,238,.5)}.drop-shadow-sm{--tw-drop-shadow:drop-shadow(0 1px 1px rgba(0,0,0,.05))}.drop-shadow-sm,.filter{filter:var(--tw-blur) var(--tw-brightness) var(--tw-contrast) var(--tw-grayscale) var(--tw-hue-rotate) var(--tw-invert) var(--tw-saturate) var(--tw-sepia) var(--tw-drop-shadow)}.backdrop-blur-md{--tw-backdrop-blur:blur(12px);-webkit-backdrop-filter:var(--tw-backdrop-blur) var(--tw-backdrop-brightness) var(--tw-backdrop-contrast) var(--tw-backdrop-grayscale) var(--tw-backdrop-hue-rotate) var(--tw-backdrop-invert) var(--tw-backdrop-opacity) var(--tw-backdrop-saturate) var(--tw-backdrop-sepia);backdrop-filter:var(--tw-backdrop-blur) var(--tw-backdrop-brightness) var(--tw-backdrop-contrast) var(--tw-backdrop-grayscale) var(--tw-backdrop-hue-rotate) var(--tw-backdrop-invert) var(--tw-backdrop-opacity) var(--tw-backdrop-saturate) var(--tw-backdrop-sepia)}.transition{transition-property:color,background-color,border-color,text-decoration-color,fill,stroke,opacity,box-shadow,transform,filter,-webkit-backdrop-filter;transition-property:color,background-color,border-color,text-decoration-color,fill,stroke,opacity,box-shadow,transform,filter,backdrop-filter;transition-property:color,background-color,border-color,text-decoration-color,fill,stroke,opacity,box-shadow,transform,filter,backdrop-filter,-webkit-backdrop-filter;transition-timing-function:cubic-bezier(.4,0,.2,1);transition-duration:.15s}.hover\\:border-emerald-500\\/50:hover{border-color:rgba(16,185,129,.5)}.hover\\:border-tv-borderLight:hover{--tw-border-opacity:1;border-color:rgb(51 60 84/var(--tw-border-opacity,1))}.hover\\:bg-amber-500\\/25:hover{background-color:rgba(245,158,11,.25)}.hover\\:bg-emerald-500\\/25:hover{background-color:rgba(16,185,129,.25)}.hover\\:bg-tv-card:hover{--tw-bg-opacity:1;background-color:rgb(26 31 46/var(--tw-bg-opacity,1))}.hover\\:bg-tv-hover:hover{--tw-bg-opacity:1;background-color:rgb(34 40 59/var(--tw-bg-opacity,1))}.hover\\:text-cyan-300:hover{--tw-text-opacity:1;color:rgb(103 232 249/var(--tw-text-opacity,1))}.hover\\:text-white:hover{--tw-text-opacity:1;color:rgb(255 255 255/var(--tw-text-opacity,1))}.hover\\:brightness-125:hover{--tw-brightness:brightness(1.25);filter:var(--tw-blur) var(--tw-brightness) var(--tw-contrast) var(--tw-grayscale) var(--tw-hue-rotate) var(--tw-invert) var(--tw-saturate) var(--tw-sepia) var(--tw-drop-shadow)}.group:hover .group-hover\\:text-emerald-300{--tw-text-opacity:1;color:rgb(110 231 183/var(--tw-text-opacity,1))}@media (min-width:640px){.sm\\:left-10{left:2.5rem}.sm\\:ml-auto{margin-left:auto}.sm\\:inline-block{display:inline-block}.sm\\:inline{display:inline}.sm\\:h-2\\.5{height:.625rem}.sm\\:h-3{height:.75rem}.sm\\:h-4{height:1rem}.sm\\:h-48{height:12rem}.sm\\:h-72{height:18rem}.sm\\:w-2\\.5{width:.625rem}.sm\\:w-3{width:.75rem}.sm\\:w-4{width:1rem}.sm\\:w-6{width:1.5rem}.sm\\:grid-cols-3{grid-template-columns:repeat(3,minmax(0,1fr))}.sm\\:gap-2{gap:.5rem}.sm\\:gap-3\\.5{gap:.875rem}.sm\\:p-4{padding:1rem}.sm\\:p-5{padding:1.25rem}.sm\\:px-3{padding-left:.75rem;padding-right:.75rem}.sm\\:px-4{padding-left:1rem;padding-right:1rem}.sm\\:px-6{padding-left:1.5rem;padding-right:1.5rem}.sm\\:pl-10{padding-left:2.5rem}.sm\\:text-2xl{font-size:1.5rem;line-height:2rem}.sm\\:text-\\[11px\\]{font-size:11px}.sm\\:text-base{font-size:1rem;line-height:1.5rem}.sm\\:text-sm{font-size:.875rem;line-height:1.25rem}.sm\\:text-xs{font-size:.75rem;line-height:1rem}}@media (min-width:768px){.md\\:inline-flex{display:inline-flex}.md\\:hidden{display:none}.md\\:grid-cols-5{grid-template-columns:repeat(5,minmax(0,1fr))}.md\\:flex-row{flex-direction:row}.md\\:items-center{align-items:center}.md\\:justify-end{justify-content:flex-end}.md\\:justify-between{justify-content:space-between}.md\\:pb-0{padding-bottom:0}}@media (min-width:1024px){.lg\\:col-span-5{grid-column:span 5/span 5}.lg\\:col-span-7{grid-column:span 7/span 7}.lg\\:grid-cols-12{grid-template-columns:repeat(12,minmax(0,1fr))}.lg\\:grid-cols-4{grid-template-columns:repeat(4,minmax(0,1fr))}.lg\\:flex-row{flex-direction:row}.lg\\:items-center{align-items:center}.lg\\:self-auto{align-self:auto}}</style>\n  <style>\n    body {\n      background-color: #0B0E14;\n      color: #F1F5F9;\n      -webkit-font-smoothing: antialiased;\n      -webkit-tap-highlight-color: transparent;\n      overflow-x: hidden;\n    }\n\n    /* Custom subtle scrollbars */\n    ::-webkit-scrollbar {\n      width: 6px;\n      height: 6px;\n    }\n    ::-webkit-scrollbar-track {\n      background: #0B0E14;\n    }\n    ::-webkit-scrollbar-thumb {\n      background: #242B3D;\n      border-radius: 9999px;\n    }\n    ::-webkit-scrollbar-thumb:hover {\n      background: #333C54;\n    }\n\n    /* Smooth transitions for interactive chart elements */\n    .k-col-btn {\n      transition: background-color 0.16s ease, border-color 0.16s ease, transform 0.12s ease;\n    }\n    .k-col-btn:hover {\n      background-color: rgba(34, 211, 238, 0.05);\n    }\n    .k-col-btn.active-k {\n      background: linear-gradient(180deg, rgba(34, 211, 238, 0.12) 0%, rgba(34, 211, 238, 0.03) 100%);\n      border-color: rgba(34, 211, 238, 0.5);\n    }\n\n    .phase-dot {\n      transition: transform 0.15s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.15s ease;\n    }\n    .phase-dot:hover {\n      transform: translate(-50%, 50%) scale(1.35);\n      z-index: 30;\n    }\n    .phase-dot.selected-dot {\n      transform: translate(-50%, 50%) scale(1.3);\n      z-index: 25;\n    }\n\n    .phase-card {\n      transition: all 0.15s ease;\n    }\n    .phase-card:hover {\n      transform: translateY(-1px);\n    }\n  </style>\n</head>\n<body class="min-h-screen pb-10">\n\n  <!-- Top Navigation & Scenario Bar -->\n  <header class="sticky top-0 z-40 bg-tv-bg/90 backdrop-blur-md border-b border-tv-border">\n    <div class="max-w-7xl mx-auto px-3 sm:px-6 py-3 flex flex-col md:flex-row md:items-center md:justify-between gap-3">\n      <!-- Brand -->\n      <div class="flex items-center justify-between">\n        <div class="flex items-center gap-3">\n          <div class="w-9 h-9 rounded-xl bg-gradient-to-br from-cyan-400 to-emerald-500 flex items-center justify-center text-slate-950 font-mono font-bold text-sm shadow-lg shadow-cyan-500/20">\n            k·φ\n          </div>\n          <div>\n            <div class="flex items-center gap-2">\n              <h1 class="text-sm sm:text-base font-bold text-white tracking-tight">\n                DNA Spectrum Explorer\n              </h1>\n              <span class="hidden sm:inline-block text-[11px] font-mono px-2 py-0.5 rounded-md bg-cyan-500/10 text-cyan-300 border border-cyan-500/20">\n                actions[phase::k] = 1\n              </span>\n            </div>\n            <p class="text-[11px] sm:text-xs text-tv-muted">\n              วิเคราะห์สเปกตรัมความถี่ <span class="text-cyan-300 font-mono">k</span> และจังหวะ <span class="text-emerald-300 font-mono">phase</span> เทียบเพดาน Perfect 100%\n            </p>\n          </div>\n        </div>\n\n        <!-- Unit Switcher (Mobile Top Right) -->\n        <div class="flex md:hidden bg-tv-panel p-0.5 rounded-lg border border-tv-border">\n          <button onclick="setUnit(\'pct\')" id="btn-unit-pct-m" class="px-2.5 py-1 rounded-md text-xs font-mono font-semibold bg-cyan-400 text-slate-950 transition">\n            %\n          </button>\n          <button onclick="setUnit(\'usd\')" id="btn-unit-usd-m" class="px-2.5 py-1 rounded-md text-xs font-mono font-semibold text-tv-muted hover:text-white transition">\n            $E\n          </button>\n        </div>\n      </div>\n\n      <!-- Market Scenarios & Desktop Unit Toggle -->\n      <div class="flex items-center justify-between md:justify-end gap-2 overflow-x-auto pb-0.5 md:pb-0">\n        <div class="inline-flex bg-tv-panel p-1 rounded-xl border border-tv-border shrink-0">\n          __TICKER_BUTTONS__</div>\n\n        <!-- Desktop Unit Switcher -->\n        <div class="hidden md:inline-flex bg-tv-panel p-1 rounded-xl border border-tv-border shrink-0">\n          <button onclick="setUnit(\'pct\')" id="btn-unit-pct" class="px-3 py-1.5 rounded-lg text-xs font-mono font-semibold bg-cyan-400 text-slate-950 transition">\n            % เพดาน\n          </button>\n          <button onclick="setUnit(\'usd\')" id="btn-unit-usd" class="px-3 py-1.5 rounded-lg text-xs font-mono font-semibold text-tv-muted hover:text-white transition">\n            $ ค่า E\n          </button>\n        </div>\n      </div>\n    </div>\n  </header>\n\n  <!-- Main Content Container -->\n  <main class="max-w-7xl mx-auto px-3 sm:px-6 pt-4 space-y-4">\n    <section class="bg-tv-panel border border-tv-border rounded-2xl p-3 sm:p-4 space-y-2">\n      <div class="flex flex-wrap items-center justify-between gap-2">\n        <h2 class="text-sm font-bold text-white">__WIN_SECTION_HEADING__</h2>\n        <span id="win-summary" class="text-[11px] font-mono text-cyan-300"></span>\n      </div>\n      <div id="win-chips" class="flex gap-2 overflow-x-auto pb-1"></div>\n    </section>\n\n    <!-- 1. Top Telemetry KPI Strip (4 Cards) -->\n    <section class="grid grid-cols-2 lg:grid-cols-4 gap-2.5 sm:gap-3.5">\n      <!-- KPI 1: 100% Perfect Foresight Ceiling -->\n      <div class="bg-tv-panel border border-tv-border rounded-2xl p-3.5 sm:p-4 relative overflow-hidden">\n        <div class="absolute top-0 left-0 bottom-0 w-1 bg-amber-400"></div>\n        <div class="flex items-center justify-between text-[11px] font-medium text-tv-muted">\n          <span>เพดาน Perfect</span>\n          <span class="font-mono text-[10px] px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-300 border border-amber-500/30">\n            100% MAX\n          </span>\n        </div>\n        <div id="kpi-ceiling-val" class="font-mono text-xl sm:text-2xl font-bold text-amber-400 mt-1">\n          $450\n        </div>\n        <div id="kpi-ceiling-sub" class="text-[11px] text-tv-dim mt-0.5 truncate">\n          เคาะจุดกลับตัวเป๊ะ 14 ครั้ง\n        </div>\n      </div>\n\n      <!-- KPI 2: Global Champion (Best k, phase) -->\n      <div onclick="selectChampion()" class="bg-tv-panel border border-tv-border hover:border-emerald-500/50 rounded-2xl p-3.5 sm:p-4 relative overflow-hidden cursor-pointer transition group">\n        <div class="absolute top-0 left-0 bottom-0 w-1 bg-emerald-400"></div>\n        <div class="flex items-center justify-between text-[11px] font-medium text-tv-muted">\n          <span class="group-hover:text-emerald-300 transition">★ แชมเปียนสูงสุด</span>\n          <span id="kpi-champ-tag" class="font-mono text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">\n            k=3 · ph=1\n          </span>\n        </div>\n        <div id="kpi-champ-val" class="font-mono text-xl sm:text-2xl font-bold text-emerald-400 mt-1">\n          89.4%\n        </div>\n        <div id="kpi-champ-sub" class="text-[11px] text-tv-dim mt-0.5 truncate">\n          แตะเพื่อเลือกแชมเปียนทันที\n        </div>\n      </div>\n\n      <!-- KPI 3: Currently Selected (k, phase) -->\n      <div class="bg-tv-panel border border-tv-border rounded-2xl p-3.5 sm:p-4 relative overflow-hidden">\n        <div class="absolute top-0 left-0 bottom-0 w-1 bg-cyan-400"></div>\n        <div class="flex items-center justify-between text-[11px] font-medium text-tv-muted">\n          <span>พิกัดที่เลือกดู</span>\n          <span id="kpi-sel-tag" class="font-mono text-[10px] px-1.5 py-0.5 rounded bg-cyan-500/15 text-cyan-300 border border-cyan-500/30">\n            k=3 · ph=1\n          </span>\n        </div>\n        <div id="kpi-sel-val" class="font-mono text-xl sm:text-2xl font-bold text-cyan-300 mt-1">\n          89.4%\n        </div>\n        <div id="kpi-sel-sub" class="text-[11px] text-tv-dim mt-0.5 truncate">\n          เคาะ 10/30 แท่ง\n        </div>\n      </div>\n\n      <!-- KPI 4: Phase Stability (Robustness) of Selected k -->\n      <div class="bg-tv-panel border border-tv-border rounded-2xl p-3.5 sm:p-4 relative overflow-hidden">\n        <div id="kpi-stab-bar" class="absolute top-0 left-0 bottom-0 w-1 bg-emerald-400"></div>\n        <div class="flex items-center justify-between text-[11px] font-medium text-tv-muted">\n          <span>ความเสถียรของ k</span>\n          <span id="kpi-stab-badge" class="font-mono text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">\n            เสถียรสูง\n          </span>\n        </div>\n        <div id="kpi-stab-val" class="font-mono text-xl sm:text-2xl font-bold text-white mt-1">\n          0.86\n        </div>\n        <div id="kpi-stab-sub" class="text-[11px] text-tv-dim mt-0.5 truncate">\n          เฉลี่ยทุกเฟส ÷ เฟสสูงสุด\n        </div>\n      </div>\n    </section>\n\n    <!-- 2. HERO SECTION: Redesigned Frequency Spectrum Chart (k + phase vs 100% Ceiling) -->\n    <section class="bg-tv-panel border border-tv-border rounded-2xl p-4 sm:p-5 space-y-4">\n      <!-- Chart Header & Mode Switcher -->\n      <div class="flex flex-col lg:flex-row lg:items-center justify-between gap-3 pb-3 border-b border-tv-border">\n        <div>\n          <div class="flex items-center gap-2">\n            <h2 class="text-sm sm:text-base font-bold text-white">\n              กราฟสเปกตรัมความถี่ (k + phase เทียบเพดาน 100%)\n            </h2>\n          </div>\n          <p class="text-xs text-tv-muted mt-0.5">\n            แต่ละคอลัมน์มีความกว้างเท่ากัน แสดง <strong class="text-slate-200">แถบช่วงกำไรต่ำสุด–สูงสุด (Min–Max)</strong>, <strong class="text-slate-200">จุดของแต่ละ phase</strong> และ <strong class="text-slate-200">ขีดค่าเฉลี่ย</strong> เทียบเส้นเพดาน 100%\n          </p>\n        </div>\n\n        <!-- 3 Chart View Modes -->\n        <div class="flex items-center gap-2 self-start lg:self-auto">\n          <div class="inline-flex bg-tv-bg p-1 rounded-xl border border-tv-border">\n            <button onclick="setChartMode(\'range\')" id="mode-range" class="chart-mode-btn px-3 py-1.5 rounded-lg text-xs font-medium bg-tv-card text-cyan-300 border border-cyan-500/30 shadow-sm transition">\n              📊 สเปกตรัมช่วง + จุดเฟส\n            </button>\n            <button onclick="setChartMode(\'Rank\')" id="mode-Rank" class="chart-mode-btn px-3 py-1.5 rounded-lg text-xs font-medium text-tv-muted hover:text-white transition">\n              ☰ แท่งแนวนอน\n            </button>\n            <button onclick="setChartMode(\'matrix\')" id="mode-matrix" class="chart-mode-btn px-3 py-1.5 rounded-lg text-xs font-medium text-tv-muted hover:text-white transition">\n              🗺️ ตาราง 2D\n            </button>\n          </div>\n\n          <button onclick="selectChampion()" class="px-3 py-2 rounded-xl bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-300 border border-emerald-500/30 text-xs font-semibold transition whitespace-nowrap">\n            ⚡ จูนอัตโนมัติ\n          </button>\n        </div>\n      </div>\n\n      <div id="k-zoom" class="flex flex-wrap items-center gap-2 text-xs px-3 py-2 rounded-xl bg-tv-bg border border-tv-border">\n        <span class="text-tv-muted font-medium">ช่วง k ที่แสดง</span>\n        <input id="kz-from" type="number" min="1" value="1" onchange="zoomFromInputs()" class="w-16 px-2 py-1 rounded-lg bg-tv-panel border border-tv-border font-mono text-[11px] text-white">\n        <span class="text-tv-dim">–</span>\n        <input id="kz-to" type="number" min="1" value="1" onchange="zoomFromInputs()" class="w-16 px-2 py-1 rounded-lg bg-tv-panel border border-tv-border font-mono text-[11px] text-white">\n        <button data-kz="10" onclick="zoomPreset(10)" class="kz-btn">1–10</button>\n        <button data-kz="30" onclick="zoomPreset(30)" class="kz-btn">1–30</button>\n        <button data-kz="100" onclick="zoomPreset(100)" class="kz-btn">1–100</button>\n        <button data-kz="all" onclick="zoomPreset(\'all\')" class="kz-btn">ทั้งหมด (1–N)</button>\n        <button onclick="zoomPreset(\'champ\')" class="px-2.5 py-1 rounded-lg font-mono text-[11px] font-semibold bg-amber-500/15 text-amber-300 border border-amber-500/30 hover:bg-amber-500/25">★ ±15</button>\n        <span id="kz-info" class="font-mono text-[11px] text-cyan-300 sm:ml-auto"></span>\n      </div>\n\n      <!-- CHART VIEW 1: Floating Range & Phase Dot Spectrum (Default) -->\n      <div id="view-range" class="space-y-3">\n        <!-- Legend & 100% Ceiling Callout -->\n        <div class="flex flex-wrap items-center justify-between gap-2 px-3 py-2 rounded-xl bg-tv-bg border border-tv-border text-xs">\n          <div class="flex flex-wrap items-center gap-x-4 gap-y-1.5">\n            <span class="inline-flex items-center gap-1.5 text-amber-300 font-medium">\n              <span class="w-4 border-t-2 border-dashed border-amber-400 inline-block"></span>\n              เพดาน 100% Perfect\n            </span>\n            <span class="inline-flex items-center gap-1.5 text-tv-muted">\n              <span class="w-2.5 h-3.5 rounded-sm bg-cyan-400/25 border border-cyan-400/50 inline-block"></span>\n              แถบช่วง Min–Max ของเฟส\n            </span>\n            <span class="inline-flex items-center gap-1.5 text-tv-muted">\n              <span class="w-2.5 h-2.5 rounded-full bg-emerald-400 inline-block"></span>\n              จุดแต่ละ Phase\n            </span>\n            <span class="inline-flex items-center gap-1.5 text-tv-muted">\n              <span class="w-3.5 h-0.5 bg-white inline-block"></span>\n              ขีดค่าเฉลี่ยทุกเฟส\n            </span>\n          </div>\n          <div id="range-ceiling-label" class="font-mono text-amber-300 font-semibold">\n            100% = $450.00\n          </div>\n        </div>\n\n        <!-- Equal-Width Interactive Spectrum Columns (k = 1..10) -->\n        <div class="relative bg-tv-bg border border-tv-border rounded-2xl pt-8 pb-3 px-2 sm:px-4">\n          <!-- Y-Axis Reference Lines (100%, 75%, 50%, 25%, 0%) -->\n          <div class="relative h-64 sm:h-72 pl-8 sm:pl-10 pr-1">\n            <!-- 100% Perfect Ceiling Line -->\n            <div class="absolute left-8 sm:left-10 right-1 top-0 border-t-2 border-dashed border-amber-400/80 z-10 pointer-events-none flex justify-end">\n              <span class=" -mt-5 px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 font-mono text-[10px] font-bold border border-amber-500/30">\n                👑 เพดาน 100%\n              </span>\n            </div>\n            <div id="y-lbl-100" class="absolute left-0 top-0 -translate-y-1/2 font-mono text-[10px] text-amber-400 font-semibold">100%</div>\n\n            <!-- 75% Line -->\n            <div class="absolute left-8 sm:left-10 right-1 top-1/4 border-t border-tv-border/70 pointer-events-none"></div>\n            <div id="y-lbl-75" class="absolute left-0 top-1/4 -translate-y-1/2 font-mono text-[10px] text-tv-dim">75%</div>\n\n            <!-- 50% Line -->\n            <div class="absolute left-8 sm:left-10 right-1 top-2/4 border-t border-tv-border/70 pointer-events-none"></div>\n            <div id="y-lbl-50" class="absolute left-0 top-2/4 -translate-y-1/2 font-mono text-[10px] text-tv-dim">50%</div>\n\n            <!-- 25% Line -->\n            <div class="absolute left-8 sm:left-10 right-1 top-3/4 border-t border-tv-border/70 pointer-events-none"></div>\n            <div id="y-lbl-25" class="absolute left-0 top-3/4 -translate-y-1/2 font-mono text-[10px] text-tv-dim">25%</div>\n\n            <!-- 0% Baseline -->\n            <div class="absolute left-8 sm:left-10 right-1 bottom-0 border-t border-tv-borderLight pointer-events-none"></div>\n            <div class="absolute left-0 bottom-0 translate-y-1/2 font-mono text-[10px] text-tv-dim">0%</div>\n\n            <!-- 10 Equal-Width Columns Grid -->\n            <div id="range-columns-grid" class="relative w-full h-full grid grid-cols-10 gap-1 sm:gap-2 z-20">\n              <!-- Populated by JS -->\n            </div>\n          </div>\n\n          <!-- X-Axis Column Footer Labels (k=1 .. k=10) -->\n          <div id="range-x-labels" class="grid grid-cols-10 gap-1 sm:gap-2 pl-8 sm:pl-10 pr-1 mt-3 pt-2 border-t border-tv-border">\n            <!-- Populated by JS -->\n          </div>\n          <div id="range-compact-note" class="pl-8 sm:pl-10 pt-1 text-[11px] text-tv-dim"></div>\n        </div>\n      </div>\n\n      <!-- CHART VIEW 2: Horizontal Spectrum Ranking (Mobile-Friendly Horizontal Bars) -->\n      <div id="view-Rank" class="hidden space-y-2.5 bg-tv-bg border border-tv-border rounded-2xl p-3.5 sm:p-4">\n        <div class="flex items-center justify-between text-xs text-tv-muted pb-2 border-b border-tv-border">\n          <span>ความถี่ k (แถบสว่าง = ค่าเฉลี่ยทุกเฟส · แถบเต็ม = เฟสที่ดีที่สุด)</span>\n          <span class="font-mono text-amber-400">ขวาสุด = 100% เพดาน Perfect</span>\n        </div>\n        <div id="horizontal-rank-list" class="space-y-2">\n          <!-- Populated by JS -->\n        </div>\n      </div>\n\n      <!-- CHART VIEW 3: Clean 2D Heatmap Matrix (k × phase) -->\n      <div id="view-matrix" class="hidden space-y-2.5 bg-tv-bg border border-tv-border rounded-2xl p-3 sm:p-4">\n        <div class="flex items-center justify-between text-xs text-tv-muted pb-1">\n          <span id="mx-range-label">แนวนอน: ความถี่ k · แนวตั้ง: จังหวะเริ่ม phase</span>\n          <span class="font-mono text-amber-300">★ = แชมเปียนสูงสุด</span>\n        </div>\n        <div class="overflow-x-auto">\n          <div id="heatmap-grid" class="relative">\n            <!-- Populated by JS -->\n          </div>\n        </div>\n      </div>\n\n      <!-- 3. Interactive Phase Breakdown Deck (For Currently Selected k) -->\n      <div class="bg-tv-bg border border-tv-border rounded-2xl p-3.5 sm:p-4 space-y-3">\n        <div class="flex flex-wrap items-center justify-between gap-2">\n          <div class="flex items-center gap-2">\n            <span class="w-2 h-2 rounded-full bg-cyan-400"></span>\n            <h3 class="text-xs sm:text-sm font-bold text-white">\n              เจาะลึกทุกจังหวะเริ่ม (Phase Breakdown) ของความถี่ <span id="deck-k-title" class="text-cyan-400 font-mono">k = 3</span>\n            </h3>\n          </div>\n          <span id="deck-k-summary" class="text-[11px] font-mono text-tv-muted">\n            แตะเลือกการ์ดเพื่อเปลี่ยน phase (0 ถึง 2)\n          </span>\n        </div>\n\n        <!-- Touch-Friendly Phase Cards Grid -->\n        <div id="phase-cards-grid" class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2.5">\n          <!-- Populated by JS -->\n        </div>\n      </div>\n    </section>\n\n    <!-- 4. Bottom Verification Section: Price Action SVG + 30-Bit Sequence & Code -->\n    <section class="grid grid-cols-1 lg:grid-cols-12 gap-4">\n      <!-- Left (7 cols): Price Chart with Rebalance Nodes vs Perfect Turning Points -->\n      <div class="lg:col-span-7 bg-tv-panel border border-tv-border rounded-2xl p-4 sm:p-5 flex flex-col justify-between space-y-3">\n        <div class="flex flex-wrap items-center justify-between gap-2">\n          <div>\n            <h3 class="text-sm font-bold text-white">\n              จุดเคาะบนกราฟราคาจริง __TF_LABEL__ (<span id="n-bars">–</span> แท่ง)\n            </h3>\n            <p class="text-xs text-tv-muted">\n              เปรียบเทียบจุดเคาะตามรอบ <code id="chart-slice-label" class="text-cyan-300 font-mono">actions[1::3] = 1</code> กับจุดกลับตัว Perfect\n            </p>\n          </div>\n          <div class="flex flex-wrap items-center gap-3 text-[11px] text-tv-muted">\n            <span class="inline-flex items-center gap-1.5">\n              <span class="w-2.5 h-2.5 rounded-full bg-rose-500 inline-block"></span>\n              t=0 (บังคับเคาะ)\n            </span>\n            <span class="inline-flex items-center gap-1.5">\n              <span class="w-2.5 h-2.5 rounded-full bg-emerald-400 inline-block"></span>\n              เคาะตามรอบ (phase::k)\n            </span>\n            <span class="inline-flex items-center gap-1.5">\n              <span class="w-2.5 h-2.5 rounded-full border-2 border-amber-400 inline-block"></span>\n              จุดกลับตัว Perfect\n            </span>\n          </div>\n        </div>\n\n        <div class="bg-tv-bg border border-tv-border rounded-xl p-3">\n          <svg id="price-svg" viewBox="0 0 640 200" class="w-full h-44 sm:h-48 overflow-visible"></svg>\n        </div>\n\n        <div id="price-insight-bar" class="text-xs text-tv-muted bg-tv-bg/60 border border-tv-border rounded-xl px-3.5 py-2.5 flex items-center justify-between">\n          <span>จังหวะการเคาะตรงกับจุดกลับตัวสำคัญ ช่วยเก็บกระแสเงินสดส่วนเกินได้ต่อเนื่อง</span>\n          <span id="price-insight-acts" class="font-mono text-cyan-300 font-semibold shrink-0 ml-2">10 ครั้ง</span>\n        </div>\n      </div>\n\n      <!-- Right (5 cols): 30-Bit Sequence Grid & O(1) Protocol Encoder -->\n      <div class="lg:col-span-5 bg-tv-panel border border-tv-border rounded-2xl p-4 sm:p-5 flex flex-col justify-between space-y-4">\n        <div>\n          <div class="flex items-center justify-between mb-2">\n            <h3 class="text-sm font-bold text-white">\n              ลำดับบิต Action <span id="n-bits"></span>\n            </h3>\n            <span id="bit-count-badge" class="font-mono text-xs px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">\n              Active 33%\n            </span>\n          </div>\n          <p class="text-xs text-tv-muted mb-3">\n            สีชมพู = แท่งแรก <code class="font-mono text-rose-300">actions[0]=1</code> · สีเขียว = เคาะรีบาลานซ์ตามคาบ <code class="font-mono text-emerald-300">phase::k</code>\n          </p>\n\n          <!-- Clean 10x3 Bit Grid -->\n          <div id="bit-sequence-grid" style="grid-template-columns:repeat(26,minmax(0,1fr))" class="grid gap-0.5 bg-tv-bg border border-tv-border rounded-xl p-2">\n            <!-- Populated by JS -->\n          </div>\n        </div>\n\n        <!-- Fast Code & Protocol Copy Box -->\n        <div class="bg-tv-bg border border-tv-border rounded-xl p-3.5 space-y-2.5">\n          <div class="flex items-center justify-between text-xs">\n            <span class="text-tv-muted font-medium">Python Generator:</span>\n            <button onclick="copyText(\'py-code-text\', this)" class="text-cyan-400 hover:text-cyan-300 font-mono text-[11px] font-semibold">\n              คัดลอกโค้ด\n            </button>\n          </div>\n          <div id="py-code-text" class="font-mono text-xs text-slate-200 bg-tv-panel px-3 py-2 rounded-lg border border-tv-border overflow-x-auto whitespace-nowrap">\n            actions = np.zeros(30, dtype=np.int32); actions[1::3] = 1; actions[0] = 1\n          </div>\n\n          <div class="flex items-center justify-between pt-1">\n            <div class="text-xs">\n              <span class="text-tv-muted">รหัสย่อ O(1) (N แท่ง): </span>\n              <code id="protocol-code-text" class="font-mono font-bold text-amber-300 ml-1">P:1000:3:1</code>\n            </div>\n            <button onclick="copyText(\'protocol-code-text\', this)" class="px-2.5 py-1 rounded-lg bg-tv-card hover:bg-tv-hover text-xs font-medium text-white border border-tv-borderLight transition">\n              คัดลอกรหัสย่อ\n            </button>\n          </div>\n        </div>\n      </div>\n    </section>\n\n    <section class="bg-tv-panel border border-tv-border rounded-2xl p-4 sm:p-5 space-y-3">\n      <div class="flex flex-wrap items-center justify-between gap-2">\n        <div>\n          <h3 class="text-sm font-bold text-white">ความสม่ำเสมอข้าม Window (ค่าเฉลี่ยทุกเฟสของแต่ละ k)</h3>\n          <p class="text-xs text-tv-muted">แถว = window · คอลัมน์ = k (ตามช่วง k ที่เลือกด้านบน) · กรอบทอง = แชมเปียนของ window · ช่องว่าง = window นั้นสั้นกว่า k · แตะเพื่อดูรายละเอียด</p>\n        </div>\n        <span id="cons-best" class="font-mono text-xs text-amber-300"></span>\n      </div>\n      <div class="overflow-x-auto"><div id="cons-grid" class="relative"></div></div>\n    </section>\n  </main>\n\n  <script>\nconst MAX_K_CAP = __MAX_K_CAP_JSON__;\nconst META = __META_JSON__;\nconst DATA = __DATA_JSON__;\n// =========================================================================\n// Config & State  (DATA = { SYM: [ {a,b,days:[[idx,\'MM-DD\']],c:[closes]} ... ] })\n// MAX_K_CAP (injected by build.py): null = every possible k per window\n// (k = 1..N, phase = 0..k-1, so max phase = N-1); a number caps k at min(cap, N).\n// =========================================================================\nconst FIX_CAP = 1500;\nconst K_CAP = (typeof MAX_K_CAP === \'number\' && MAX_K_CAP > 0) ? MAX_K_CAP : null;\nconst COMPACT_COLS = 40;            // above this many k columns the range chart goes compact\nconst TICKERS = Object.keys(DATA);\nconst MT = (typeof META === \'object\' && META) ? META : { tf: \'M15\', tfLabel: \'15 นาที\', pending: [] };\nconst TH_MON = [\'ม.ค.\',\'ก.พ.\',\'มี.ค.\',\'เม.ย.\',\'พ.ค.\',\'มิ.ย.\',\'ก.ค.\',\'ส.ค.\',\'ก.ย.\',\'ต.ค.\',\'พ.ย.\',\'ธ.ค.\'];\n\nlet state = { ticker: TICKERS[0], unit: \'pct\', chartMode: \'range\', wi: 0, selK: 3, selPh: 1, models: {},\n              kz: { from: 1, to: Infinity }, colorMode: \'rank\', topPct: 100 };\n\nfunction fmtDate(iso) {\n  const [, m, d] = iso.split(\'-\');\n  return `${parseInt(d, 10)} ${TH_MON[parseInt(m, 10) - 1]}`;\n}\n// Calendar label of a half-month window: "1–14 ต.ค." or "15–31 ต.ค."\nfunction halfLabel(win) {\n  const [, m, d1] = win.wa.split(\'-\'), d2 = win.wb.split(\'-\')[2];\n  return `${parseInt(d1, 10)}–${parseInt(d2, 10)} ${TH_MON[parseInt(m, 10) - 1]}`;\n}\nconst kMaxOf = (N) => K_CAP ? Math.min(K_CAP, N) : N;\n\n// =========================================================================\n// Core maths (same as template)\n// =========================================================================\nfunction calcNetProfit(actions, prices) {\n  const p0 = prices[0], pEnd = prices[prices.length - 1];\n  const referNet = -FIX_CAP * Math.log(p0 / pEnd);\n  let cash = FIX_CAP, amount = FIX_CAP / p0;\n  for (let i = 1; i < prices.length; i++) {\n    if (actions[i] === 1) {\n      cash += amount * prices[i] - FIX_CAP;\n      amount = FIX_CAP / prices[i];\n    }\n  }\n  return (cash + amount * pEnd) - referNet - (FIX_CAP * 2.0);\n}\n\n// Same arithmetic as calcNetProfit(actions[ph::k]=1, actions[0]=1) in the same order,\n// but it only visits the rebalance bars, so one (k, ph) costs O(N/k) instead of O(N).\nfunction calcNetPeriodic(prices, k, ph) {\n  const N = prices.length, p0 = prices[0], pEnd = prices[N - 1];\n  const referNet = -FIX_CAP * Math.log(p0 / pEnd);\n  let cash = FIX_CAP, amount = FIX_CAP / p0;\n  for (let i = ph === 0 ? k : ph; i < N; i += k) {\n    cash += amount * prices[i] - FIX_CAP;\n    amount = FIX_CAP / prices[i];\n  }\n  return (cash + amount * pEnd) - referNet - (FIX_CAP * 2.0);\n}\n\nfunction makeActions(N, k, ph) {\n  const actions = new Array(N).fill(0);\n  for (let i = ph; i < N; i += k) actions[i] = 1;\n  actions[0] = 1;\n  return actions;\n}\n\nfunction actsCount(N, k, ph) {\n  const start = ph === 0 ? k : ph;\n  return 1 + (start < N ? Math.floor((N - 1 - start) / k) + 1 : 0);\n}\n\nfunction computePerfectForesight(prices) {\n  const n = prices.length;\n  const dp = new Array(n).fill(-Infinity);\n  const parent = new Array(n).fill(0);\n  dp[0] = FIX_CAP * 2.0;\n  for (let i = 1; i < n; i++) {\n    for (let j = 0; j < i; j++) {\n      const score = dp[j] + FIX_CAP * (prices[i] / prices[j] - 1.0);\n      if (score > dp[i]) { dp[i] = score; parent[i] = j; }\n    }\n  }\n  let bestEnd = 0, bestScore = -Infinity;\n  for (let j = 0; j < n; j++) {\n    const total = dp[j] + FIX_CAP * (prices[n - 1] / prices[j] - 1.0);\n    if (total > bestScore) { bestScore = total; bestEnd = j; }\n  }\n  const acts = new Array(n).fill(0);\n  let cur = bestEnd;\n  while (cur > 0) { acts[cur] = 1; cur = parent[cur]; }\n  acts[0] = 1;\n  return acts;\n}\n\nconst clamp01 = (x) => Math.max(0, Math.min(100, x));\n\nfunction buildModel(win) {\n  const prices = win.c;\n  const N = prices.length;\n  const K = kMaxOf(N);\n  const m = { win, prices, N, K, combos: K * (K + 1) / 2, grid: {}, kStats: {}, champ: { k: 1, ph: 0, net: -Infinity, pct: 0 } };\n  m.perfectActs = computePerfectForesight(prices);\n  m.perfectNet = Math.max(1, calcNetProfit(m.perfectActs, prices));\n\n  for (let k = 1; k <= K; k++) {\n    const net = new Float64Array(k);\n    let minNet = Infinity, maxNet = -Infinity, sumNet = 0, bestPh = 0;\n    for (let ph = 0; ph < k; ph++) {\n      const v = calcNetPeriodic(prices, k, ph);\n      net[ph] = v;\n      sumNet += v;\n      if (v < minNet) minNet = v;\n      if (v > maxNet) { maxNet = v; bestPh = ph; }\n      if (v > m.champ.net) m.champ = { k, ph, net: v, pct: clamp01((v / m.perfectNet) * 100) };\n    }\n    m.grid[k] = net;\n    const meanNet = sumNet / k;\n    m.kStats[k] = {\n      minNet, maxNet, meanNet,\n      minPct: clamp01(minNet / m.perfectNet * 100),\n      maxPct: clamp01(maxNet / m.perfectNet * 100),\n      meanPct: clamp01(meanNet / m.perfectNet * 100),\n      bestPh,\n      stability: maxNet > 0 ? Math.max(0, Math.min(1, meanNet / maxNet)) : 0\n    };\n  }\n  return m;\n}\n\n// One (k, ph) cell; actions are built only when someone asks for them.\nfunction cell(m, k, ph) {\n  const net = m.grid[k][ph];\n  return { net, pct: clamp01((net / m.perfectNet) * 100), acts: actsCount(m.N, k, ph),\n           get actions() { return makeActions(m.N, k, ph); } };\n}\n\nfunction M() { return state.models[state.ticker][state.wi]; }\n\nfunction formatMetric(net, pct, decimals = 0) {\n  if (state.unit === \'pct\') return `${pct.toFixed(decimals)}%`;\n  return `$${Math.round(net)}`;\n}\n\n// =========================================================================\n// k zoom (which k columns the spectrum / ranking / matrix / consistency show)\n// =========================================================================\nfunction kRange(K) {\n  const lo = Math.max(1, Math.min(state.kz.from, K));\n  const hi = Math.max(lo, Math.min(state.kz.to, K));\n  return [lo, hi];\n}\n\nfunction ensureKVisible(k, K) {\n  const [lo, hi] = kRange(K);\n  if (k >= lo && k <= hi) return;\n  const w = hi - lo;\n  const nlo = Math.max(1, Math.min(K - w, k - Math.floor(w / 2)));\n  state.kz = { from: nlo, to: nlo + w };\n}\n\nfunction setZoom(from, to) {\n  const m = M();\n  from = Math.max(1, Math.floor(from) || 1);\n  to = to === Infinity ? Infinity : Math.max(from, Math.floor(to) || from);\n  state.kz = { from, to };\n  const [lo, hi] = kRange(m.K);\n  if (state.selK < lo || state.selK > hi) {\n    let bk = lo;\n    for (let k = lo; k <= hi; k++) if (m.kStats[k].maxNet > m.kStats[bk].maxNet) bk = k;\n    state.selK = bk; state.selPh = m.kStats[bk].bestPh;\n  }\n  renderAll();\n}\nfunction zoomPreset(p) {\n  if (p === \'all\') return setZoom(1, Infinity);\n  if (p === \'champ\') { const c = M().champ.k; return setZoom(Math.max(1, c - 15), c + 15); }\n  setZoom(1, p);\n}\nfunction zoomFromInputs() {\n  const f = parseInt(document.getElementById(\'kz-from\').value, 10);\n  const t = parseInt(document.getElementById(\'kz-to\').value, 10);\n  setZoom(f || 1, t || Infinity);\n}\n\nfunction renderZoomBar() {\n  const m = M();\n  const [lo, hi] = kRange(m.K);\n  const f = document.getElementById(\'kz-from\'), t = document.getElementById(\'kz-to\');\n  if (!f) return;\n  f.max = m.K; t.max = m.K;\n  f.value = lo; t.value = hi;\n  document.getElementById(\'kz-info\').textContent =\n    `window นี้ ${m.N} แท่ง → k สูงสุด ${m.K} · phase สูงสุด ${m.K - 1} · ${m.combos.toLocaleString()} คู่ (k, ph)`;\n  document.querySelectorAll(\'[data-kz]\').forEach(b => {\n    const p = b.dataset.kz;\n    const on = (p === \'all\' && lo === 1 && hi === m.K) || (String(hi) === p && lo === 1 && hi !== m.K);\n    b.className = `kz-btn px-2.5 py-1 rounded-lg font-mono text-[11px] font-semibold transition ${on ? \'bg-cyan-400 text-slate-950\' : \'bg-tv-bg text-tv-muted border border-tv-border hover:text-white\'}`;\n  });\n}\n\n// =========================================================================\n// Shared helpers for big grids\n// =========================================================================\nlet tipEl = null;\nfunction tip(e, text) {\n  if (!tipEl) {\n    tipEl = document.createElement(\'div\');\n    tipEl.className = \'fixed z-50 pointer-events-none px-2 py-1 rounded-md bg-slate-950/95 border border-tv-borderLight font-mono text-[11px] text-slate-100 shadow-lg whitespace-nowrap\';\n    document.body.appendChild(tipEl);\n  }\n  if (!text) { tipEl.style.display = \'none\'; return; }\n  tipEl.textContent = text;\n  tipEl.style.display = \'block\';\n  const x = Math.min(window.innerWidth - tipEl.offsetWidth - 8, e.clientX + 12);\n  tipEl.style.left = `${Math.max(4, x)}px`;\n  tipEl.style.top = `${e.clientY + 14}px`;\n}\n\nfunction niceStep(n, maxLabels) {\n  const raw = Math.max(1, Math.ceil(n / maxLabels));\n  for (const s of [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000]) if (s >= raw) return s;\n  return raw;\n}\n\nfunction rrect(ctx, x, y, w, h, r) {\n  r = Math.min(r, w / 2, h / 2);\n  ctx.beginPath();\n  if (ctx.roundRect && r > 0.5) ctx.roundRect(x, y, w, h, r); else ctx.rect(x, y, w, h);\n}\n\n// Canvas heatmap: spec = { host, cols, rows, labelW, headH, minCw, rowH(nRows, cw), colLabel(c), rowLabel(r),\n//   rowLabelStyle(r), cell(c, r) -> null | {bg, text, color, ring, outline, bold}, marks: [{c, r, color}],\n//   tipText(c, r), onClick(c, r) }\n// Big grids (> 60 columns or > 40 rows) are drawn "tight": no gaps, fractional cell size that fills the\n// whole width, so every k gets the same share of pixels and clusters keep their true shape.\nfunction drawHeatCanvas(spec) {\n  const host = spec.host;\n  host.innerHTML = \'\';\n  const avail = Math.max(320, host.clientWidth || host.parentElement.clientWidth || 900);\n  const tight = spec.cols > 60 || spec.rows > 40;\n  const gap = tight ? 0 : 2, rgap = tight ? 0 : 2;\n  const cw = tight ? Math.max(spec.minCw, (avail - spec.labelW) / spec.cols)\n                   : Math.max(spec.minCw, Math.floor((avail - spec.labelW) / spec.cols) - gap);\n  const rh = spec.rowH(spec.rows, cw);\n  const cx = (c) => spec.labelW + (tight ? Math.round(c * cw) : c * (cw + gap));\n  const cwOf = (c) => tight ? Math.max(1, Math.round((c + 1) * cw) - Math.round(c * cw)) : cw;\n  const ry = (r) => spec.headH + (tight ? Math.round(r * rh) : r * (rh + rgap));\n  const rhOf = (r) => tight ? Math.max(1, Math.round((r + 1) * rh) - Math.round(r * rh)) : rh;\n  const W = Math.ceil(cx(spec.cols)), H = Math.ceil(ry(spec.rows));\n  const dpr = window.devicePixelRatio || 1;\n  const cv = document.createElement(\'canvas\');\n  cv.width = Math.ceil(W * dpr); cv.height = Math.ceil(H * dpr);\n  cv.style.width = `${W}px`; cv.style.height = `${H}px`;\n  cv.style.cursor = \'crosshair\';\n  host.appendChild(cv);\n  const ctx = cv.getContext(\'2d\');\n  ctx.scale(dpr, dpr);\n  ctx.imageSmoothingEnabled = false;\n  ctx.textAlign = \'center\'; ctx.textBaseline = \'middle\';\n  const font = (px, w) => `${w || 600} ${px}px "JetBrains Mono", monospace`;\n  const big = cw >= 10 && rh >= 10;\n\n  // axis labels (thinned); highlighted ones always draw and push neighbours away\n  const cStep = niceStep(spec.cols, Math.max(4, Math.floor((W - spec.labelW) / 34)));\n  const colPx = 32 / (cw + gap), rowPx = 14 / (rh + rgap);\n  const alwaysC = [], alwaysR = [];\n  for (let c = 0; c < spec.cols; c++) if (spec.colLabel(c).always) alwaysC.push(c);\n  if (spec.rowLabelStyle) for (let r = 0; r < spec.rows; r++) if (spec.rowLabelStyle(r).always) alwaysR.push(r);\n  const crowded = (i, list, px) => list.some(j => j !== i && Math.abs(j - i) < px);\n  for (let c = 0; c < spec.cols; c++) {\n    const lab = spec.colLabel(c);\n    if (!lab.always && ((c % cStep !== 0 && cw < 30) || crowded(c, alwaysC, colPx))) continue;\n    const x = cx(c), w = cwOf(c);\n    if (w >= 22) { ctx.fillStyle = lab.bg || \'#131722\'; rrect(ctx, x, 0, w, spec.headH - 4, 4); ctx.fill(); }\n    ctx.fillStyle = lab.color || \'#CBD5E1\'; ctx.font = font(w >= 30 ? 12 : 10, 700);\n    ctx.fillText(lab.text, x + w / 2, (spec.headH - 4) / 2);\n  }\n  const rStep = niceStep(spec.rows, Math.max(4, Math.floor((H - spec.headH) / 16)));\n  for (let r = 0; r < spec.rows; r++) {\n    const y = ry(r), h = rhOf(r);\n    const st = spec.rowLabelStyle ? spec.rowLabelStyle(r) : {};\n    if (st.always || ((h >= 14 || r % rStep === 0) && !crowded(r, alwaysR, rowPx))) {\n      if (h >= 14) { ctx.fillStyle = st.bg || \'#131722\'; rrect(ctx, 0, y, spec.labelW - 4, h, 4); ctx.fill(); }\n      ctx.fillStyle = st.color || \'#94A3B8\'; ctx.font = font(h >= 14 ? 11 : 9, st.bold ? 700 : 600);\n      ctx.fillText(spec.rowLabel(r), (spec.labelW - 4) / 2, y + Math.max(h, 10) / 2);\n    }\n    for (let c = 0; c < spec.cols; c++) {\n      const x = cx(c), w = cwOf(c);\n      const v = spec.cell(c, r);\n      if (!v) {\n        if (big) { ctx.fillStyle = \'rgba(19,23,34,0.30)\'; rrect(ctx, x, y, w, h, 4); ctx.fill(); }\n        continue;\n      }\n      ctx.fillStyle = v.bg;\n      if (big) { rrect(ctx, x, y, w, h, 4); ctx.fill(); } else ctx.fillRect(x, y, w, h);\n      if (big && (v.ring || v.outline)) {\n        ctx.strokeStyle = v.ring ? \'#FBBF24\' : \'#67E8F9\'; ctx.lineWidth = 2;\n        rrect(ctx, x + 1, y + 1, w - 2, h - 2, 4); ctx.stroke();\n      }\n      if (v.text && w >= 34 && h >= 18) {\n        ctx.fillStyle = v.color || \'#FFFFFF\'; ctx.font = font(11, v.bold ? 700 : 600);\n        ctx.fillText(v.text, x + w / 2, y + h / 2);\n      }\n    }\n  }\n  // small cells: rings would be invisible, so mark with a crosshair + circle instead\n  if (!big) {\n    const marks = (spec.marks || []).filter(mk => mk.c >= 0 && mk.c < spec.cols && mk.r >= 0 && mk.r < spec.rows);\n    for (const mk of marks) {\n      const x = cx(mk.c) + cwOf(mk.c) / 2, y = ry(mk.r) + rhOf(mk.r) / 2;\n      if (mk.cross !== false) {\n      ctx.save();\n      ctx.strokeStyle = mk.color; ctx.globalAlpha = 0.35; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);\n      ctx.beginPath(); ctx.moveTo(x, spec.headH); ctx.lineTo(x, H); ctx.moveTo(spec.labelW, y); ctx.lineTo(W, y); ctx.stroke();\n      ctx.restore();\n      }\n      const rad = Math.max(6, Math.min(cw, rh) * 1.6);   // dark halo first so the ring shows on yellow cells too\n      ctx.strokeStyle = \'#0B0E14\'; ctx.lineWidth = 4.5;\n      ctx.beginPath(); ctx.arc(x, y, rad, 0, Math.PI * 2); ctx.stroke();\n      ctx.strokeStyle = mk.color; ctx.lineWidth = 2;\n      ctx.beginPath(); ctx.arc(x, y, rad, 0, Math.PI * 2); ctx.stroke();\n    }\n  }\n  const hit = (e) => {\n    const b = cv.getBoundingClientRect();\n    const x = e.clientX - b.left - spec.labelW, y = e.clientY - b.top - spec.headH;\n    if (x < 0 || y < 0) return null;\n    const c = Math.floor(x / (cw + gap)), r = Math.floor(y / (rh + rgap));\n    if (c >= spec.cols || r >= spec.rows || !spec.cell(c, r)) return null;\n    return [c, r];\n  };\n  cv.onmousemove = (e) => { const h = hit(e); tip(e, h ? spec.tipText(h[0], h[1]) : \'\'); };\n  cv.onmouseleave = (e) => tip(e, \'\');\n  cv.onclick = (e) => { const h = hit(e); if (h) { tip(e, \'\'); spec.onClick(h[0], h[1]); } };\n  return { cw, rh, gap };\n}\n\n// =========================================================================\n// Colour scale for the heatmaps. \'rank\' colours by percentile inside the visible\n// grid (every colour band holds the same number of cells, so clusters pop even when\n// most values sit in a narrow band); \'rel\' stretches min..max; \'fixed\' = the\n// original 4 bands. topPct dims everything outside the best X% of cells.\n// =========================================================================\nconst PAL = [[0, [15, 23, 42]], [0.35, [30, 58, 138]], [0.6, [8, 145, 178]], [0.78, [16, 185, 129]], [0.92, [250, 204, 21]], [1, [255, 247, 214]]];\nconst DIM = \'rgba(30, 41, 59, 0.30)\';\nfunction palette(t) {\n  t = Math.max(0, Math.min(1, t));\n  let i = 1; while (i < PAL.length - 1 && t > PAL[i][0]) i++;\n  const [t0, c0] = PAL[i - 1], [t1, c1] = PAL[i], f = (t - t0) / (t1 - t0 || 1);\n  return `rgb(${c0.map((v, j) => Math.round(v + (c1[j] - v) * f)).join(\',\')})`;\n}\nfunction makeScale(values) {\n  const sorted = Float64Array.from(values).sort();\n  const n = sorted.length;\n  const q = (p) => sorted[Math.min(n - 1, Math.max(0, Math.round(p * (n - 1))))];\n  const min = sorted[0], max = sorted[n - 1];\n  const thr = state.topPct >= 100 ? -Infinity : q(1 - state.topPct / 100);\n  const rank = (v) => { let lo = 0, hi = n; while (lo < hi) { const mid = (lo + hi) >> 1; if (sorted[mid] <= v) lo = mid + 1; else hi = mid; } return n > 1 ? (lo - 1) / (n - 1) : 1; };\n  const tOf = (v) => state.colorMode === \'rank\' ? rank(v) : (max > min ? (v - min) / (max - min) : 1);\n  return {\n    min, max, med: q(0.5), thr, n,\n    color: (v) => v < thr ? DIM : (state.colorMode === \'fixed\' ? cellBg(v) : palette(tOf(v))),\n    text: (v) => (v >= thr && state.colorMode !== \'fixed\' && tOf(v) > 0.86) ? \'#0B0E14\' : \'#FFFFFF\',\n    topOf: (v) => Math.max(0, 1 - rank(v)) * 100,\n  };\n}\nfunction setColorMode(m) { state.colorMode = m; renderHeatmapMatrix(); renderConsistency(); }\nfunction setTopPct(p) { state.topPct = p; renderHeatmapMatrix(); renderConsistency(); }\n\nfunction renderColorBar(host, id, sc, what) {\n  const wrap = host.parentElement;\n  let bar = document.getElementById(id);\n  if (!bar) {\n    bar = document.createElement(\'div\');\n    bar.id = id;\n    bar.className = \'flex flex-wrap items-center gap-x-3 gap-y-2 pb-2 text-[11px] text-tv-muted\';\n    wrap.parentElement.insertBefore(bar, wrap);\n  }\n  const btn = (on, label, call) =>\n    `<button type="button" onclick="${call}" class="px-2.5 py-1 rounded-lg font-mono text-[11px] font-semibold transition ${on ? \'bg-cyan-400 text-slate-950\' : \'bg-tv-bg text-tv-muted border border-tv-border hover:text-white\'}">${label}</button>`;\n  const grad = `linear-gradient(90deg, ${PAL.map(([t, c]) => `rgb(${c.join(\',\')}) ${t * 100}%`).join(\', \')})`;\n  const legend = state.colorMode === \'fixed\'\n    ? [\'rgba(244,63,94,0.18)\', \'rgba(34,211,238,0.12)\', \'rgba(16,185,129,0.24)\', \'rgba(16,185,129,0.45)\']\n        .map((c, i) => `<span class="inline-flex items-center gap-1"><span class="w-3 h-3 rounded-sm inline-block" style="background:${c}"></span>${[\'<30%\', \'30–55%\', \'55–80%\', \'≥80%\'][i]}</span>`).join(\' \')\n    : `<span class="font-mono">${sc.min.toFixed(0)}%</span><span class="inline-block w-36 h-2.5 rounded-full" style="background:${grad}"></span><span class="font-mono">${sc.max.toFixed(0)}%</span>` +\n      `<span class="text-tv-dim">${state.colorMode === \'rank\' ? `· สีแบ่งตามอันดับ มัธยฐาน ${sc.med.toFixed(1)}%` : \'· สียืดตามต่ำสุด–สูงสุด\'}</span>`;\n  const tops = [100, 25, 10, 5, 1];\n  bar.innerHTML =\n    `<span class="font-medium">สี${what}:</span>` +\n    btn(state.colorMode === \'rank\', \'อันดับ (percentile)\', "setColorMode(\'rank\')") +\n    btn(state.colorMode === \'rel\', \'ต่ำสุด–สูงสุด\', "setColorMode(\'rel\')") +\n    btn(state.colorMode === \'fixed\', \'4 ระดับเดิม\', "setColorMode(\'fixed\')") +\n    `<span class="font-medium ml-1">เน้นกลุ่มบนสุด:</span>` +\n    tops.map(p => btn(state.topPct === p, p === 100 ? \'ทั้งหมด\' : `Top ${p}%`, `setTopPct(${p})`)).join(\'\') +\n    `<span class="inline-flex items-center gap-2 sm:ml-auto">${legend}${state.topPct < 100 ? `<span class="text-amber-300 font-mono">· แสดงเฉพาะ ≥ ${sc.thr.toFixed(1)}%</span>` : \'\'}</span>`;\n}\n\n// =========================================================================\n// KPI strip\n// =========================================================================\nfunction renderKPIs() {\n  const m = M();\n  const cur = cell(m, state.selK, state.selPh);\n  const ks = m.kStats[state.selK];\n  const perfActsCount = m.perfectActs.reduce((a, b) => a + b, 0);\n\n  document.getElementById(\'kpi-ceiling-val\').textContent =\n    state.unit === \'pct\' ? `100% ($${Math.round(m.perfectNet)})` : `$${m.perfectNet.toFixed(1)}`;\n  document.getElementById(\'kpi-ceiling-sub\').textContent =\n    `${state.ticker} · จุดกลับตัวเป๊ะ ${perfActsCount} ครั้ง จาก ${m.N} แท่ง`;\n\n  document.getElementById(\'kpi-champ-tag\').textContent = `k=${m.champ.k} · ph=${m.champ.ph}`;\n  document.getElementById(\'kpi-champ-val\').textContent = formatMetric(m.champ.net, m.champ.pct, 1);\n  document.getElementById(\'kpi-champ-sub\').textContent =\n    (state.unit === \'pct\' ? `กำไร E = $${m.champ.net.toFixed(1)}` : `คิดเป็น ${m.champ.pct.toFixed(1)}% ของเพดาน`) +\n    ` · จาก ${m.combos.toLocaleString()} คู่`;\n\n  document.getElementById(\'kpi-sel-tag\').textContent = `k=${state.selK} · ph=${state.selPh}`;\n  document.getElementById(\'kpi-sel-val\').textContent = formatMetric(cur.net, cur.pct, 1);\n  document.getElementById(\'kpi-sel-sub\').textContent =\n    `เคาะ ${cur.acts}/${m.N} แท่ง · ห่างเพดาน -$${Math.max(0, m.perfectNet - cur.net).toFixed(0)}`;\n\n  const stab = ks.stability;\n  const badge = document.getElementById(\'kpi-stab-badge\');\n  const bar = document.getElementById(\'kpi-stab-bar\');\n  document.getElementById(\'kpi-stab-val\').textContent = stab.toFixed(2);\n  if (stab >= 0.78) {\n    badge.textContent = \'เสถียรสูง\';\n    badge.className = \'font-mono text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30\';\n    bar.className = \'absolute top-0 left-0 bottom-0 w-1 bg-emerald-400\';\n  } else if (stab >= 0.50) {\n    badge.textContent = \'ปานกลาง\';\n    badge.className = \'font-mono text-[10px] px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-300 border border-amber-500/30\';\n    bar.className = \'absolute top-0 left-0 bottom-0 w-1 bg-amber-400\';\n  } else {\n    badge.textContent = \'พึ่งเฟสสูง\';\n    badge.className = \'font-mono text-[10px] px-1.5 py-0.5 rounded bg-rose-500/15 text-rose-300 border border-rose-500/30\';\n    bar.className = \'absolute top-0 left-0 bottom-0 w-1 bg-rose-500\';\n  }\n  document.getElementById(\'kpi-stab-sub\').textContent =\n    `เฉลี่ยทุกเฟสของ k=${state.selK} ได้ ${ks.meanPct.toFixed(0)}%`;\n}\n\n// =========================================================================\n// Window bar\n// =========================================================================\nfunction renderWindowBar() {\n  const wins = state.models[state.ticker];\n  const bar = document.getElementById(\'win-chips\');\n  bar.innerHTML = \'\';\n  wins.forEach((m, i) => {\n    const b = document.createElement(\'button\');\n    b.type = \'button\';\n    b.onclick = () => setWindow(i);\n    const active = i === state.wi;\n    b.className = `shrink-0 text-left px-2.5 py-1.5 rounded-lg border text-[11px] font-mono transition ${\n      active ? \'bg-cyan-400 text-slate-950 border-cyan-300 font-bold\' : \'bg-tv-bg text-tv-muted border-tv-border hover:text-white hover:border-tv-borderLight\'\n    }`;\n    b.title = `W${i + 1}: ${halfLabel(m.win)} · เทรดจริง ${fmtDate(m.win.a)} – ${fmtDate(m.win.b)}`;\n    b.innerHTML = `<div>W${i + 1} · ${halfLabel(m.win)}${m.win.partial ? \' ⏳\' : \'\'}</div>\n      <div class="text-[10px] ${active ? \'text-slate-800\' : \'text-tv-dim\'}">${m.N} แท่ง · ★k${m.champ.k}/ph${m.champ.ph}</div>`;\n    bar.appendChild(b);\n  });\n  // today\'s half-month with no bar yet: shown greyed out so the strip always reaches today\n  (MT.pending || []).forEach((w, j) => {\n    const d = document.createElement(\'div\');\n    d.className = \'shrink-0 text-left px-2.5 py-1.5 rounded-lg border border-dashed border-tv-borderLight text-[11px] font-mono text-tv-dim cursor-default\';\n    d.title = \'window ปัจจุบัน ยังไม่มีแท่งราคา\';\n    d.innerHTML = `<div>W${wins.length + j + 1} · ${halfLabel(w)} ⏳</div><div class="text-[10px]">ยังไม่มีแท่ง · วันนี้ ${fmtDate(MT.asof)}</div>`;\n    bar.appendChild(d);\n  });\n  const m = M();\n  const partial = (m.win.partial ? \' (window ยังไม่ครบ)\' : \'\') + (m.win.headPartial ? \' (ข้อมูลเริ่มกลาง window)\' : \'\');\n  document.getElementById(\'win-summary\').textContent =\n    `${state.ticker} · W${state.wi + 1}/${wins.length} · ${halfLabel(m.win)} (เทรด ${fmtDate(m.win.a)} – ${fmtDate(m.win.b)}) · ${m.N} แท่ง ${MT.tfLabel}${partial}${MT.lastBar ? ` · ข้อมูลถึง ${fmtDate(MT.lastBar)}` : \'\'} · k 1–${m.K} × ph 0–${m.K - 1}`;\n  const active = bar.children[state.wi];\n  if (active) bar.scrollLeft = Math.max(0, active.offsetLeft - bar.clientWidth / 2 + active.offsetWidth / 2);\n}\n\n// =========================================================================\n// Primary chart: range + phase dots\n// =========================================================================\nfunction renderRangeSpectrum() {\n  const m = M();\n  const [lo, hi] = kRange(m.K);\n  const n = hi - lo + 1;\n  const compact = n > COMPACT_COLS;\n  document.getElementById(\'range-ceiling-label\').textContent = `เพดาน 100% = $${m.perfectNet.toFixed(2)}`;\n  document.getElementById(\'y-lbl-100\').textContent = state.unit === \'pct\' ? \'100%\' : `$${Math.round(m.perfectNet)}`;\n  document.getElementById(\'y-lbl-75\').textContent = state.unit === \'pct\' ? \'75%\' : `$${Math.round(m.perfectNet * 0.75)}`;\n  document.getElementById(\'y-lbl-50\').textContent = state.unit === \'pct\' ? \'50%\' : `$${Math.round(m.perfectNet * 0.50)}`;\n  document.getElementById(\'y-lbl-25\').textContent = state.unit === \'pct\' ? \'25%\' : `$${Math.round(m.perfectNet * 0.25)}`;\n\n  const colsGrid = document.getElementById(\'range-columns-grid\');\n  const xLabelsGrid = document.getElementById(\'range-x-labels\');\n  colsGrid.innerHTML = \'\';\n  xLabelsGrid.innerHTML = \'\';\n  const tpl = `repeat(${n}, minmax(0, 1fr))`;\n  colsGrid.style.gridTemplateColumns = tpl;\n  xLabelsGrid.style.gridTemplateColumns = tpl;\n  colsGrid.style.columnGap = compact ? \'0px\' : (n > 20 ? \'2px\' : \'\');\n  xLabelsGrid.style.columnGap = colsGrid.style.columnGap;\n  const xStep = niceStep(n, 14);\n\n  for (let k = lo; k <= hi; k++) {\n    const ks = m.kStats[k];\n    const isSelectedK = (k === state.selK);\n    const isChampK = (k === m.champ.k);\n\n    const col = document.createElement(\'div\');\n    col.className = compact\n      ? `k-col-btn relative h-full cursor-pointer flex flex-col items-center justify-end ${isSelectedK ? \'active-k border-x\' : \'\'}`\n      : `k-col-btn relative h-full rounded-xl border cursor-pointer flex flex-col items-center justify-end ${isSelectedK ? \'active-k\' : \'border-transparent\'}`;\n    col.onclick = () => selectKColumn(k);\n    if (compact) col.title = `k=${k} · สูงสุด ${ks.maxPct.toFixed(1)}% (ph=${ks.bestPh}) · เฉลี่ย ${ks.meanPct.toFixed(1)}% · ต่ำสุด ${ks.minPct.toFixed(1)}%`;\n\n    if (!compact || isChampK || isSelectedK) {\n      const topBadge = document.createElement(\'div\');\n      topBadge.style.bottom = `calc(${Math.min(92, Math.max(6, ks.maxPct))}% + 10px)`;\n      topBadge.className = `absolute left-1/2 -translate-x-1/2 px-1.5 py-0.5 rounded font-mono text-[10px] sm:text-[11px] font-bold whitespace-nowrap pointer-events-none transition z-30 ${\n        isChampK ? \'bg-amber-400 text-slate-950 shadow-md shadow-amber-500/20\'\n          : (isSelectedK ? \'bg-cyan-400/20 text-cyan-200 border border-cyan-400/40\' : \'text-slate-300\')}`;\n      topBadge.textContent = (isChampK ? \'★ \' : \'\') + (compact ? `k${k} ` : \'\') + formatMetric(ks.maxNet, ks.maxPct, 0);\n      col.appendChild(topBadge);\n    }\n\n    const capsule = document.createElement(\'div\');\n    capsule.style.bottom = `${ks.minPct}%`;\n    capsule.style.height = `${Math.max(compact ? 1 : 2.5, ks.maxPct - ks.minPct)}%`;\n    if (compact) capsule.style.width = \'max(2px, 70%)\';\n    capsule.className = `absolute left-1/2 -translate-x-1/2 ${compact ? \'rounded-sm\' : \'w-3 sm:w-4 rounded-full\'} transition ${\n      isSelectedK ? \'bg-cyan-400/40 border border-cyan-400/70\'\n        : (isChampK ? \'bg-emerald-400/45 border border-emerald-400/70\' : (compact ? \'bg-slate-500/45\' : \'bg-slate-700/40 border border-slate-600/50\'))}`;\n    col.appendChild(capsule);\n\n    const meanTick = document.createElement(\'div\');\n    meanTick.style.bottom = `${ks.meanPct}%`;\n    if (compact) meanTick.style.width = \'100%\';\n    meanTick.className = `absolute left-1/2 -translate-x-1/2 ${compact ? \'\' : \'w-5 sm:w-6\'} h-0.5 rounded-full z-10 pointer-events-none ${isSelectedK ? \'bg-white shadow-sm shadow-white\' : \'bg-slate-300/80\'}`;\n    meanTick.title = `ค่าเฉลี่ยทุกเฟสของ k=${k}: ${ks.meanPct.toFixed(1)}%`;\n    col.appendChild(meanTick);\n\n    if (!compact || isSelectedK || isChampK) {\n      const phs = (compact && !isSelectedK) ? [m.champ.ph] : [...Array(k).keys()];\n      for (const ph of phs) {\n        const item = cell(m, k, ph);\n        const isThisSelected = (k === state.selK && ph === state.selPh);\n        const isThisChamp = (k === m.champ.k && ph === m.champ.ph);\n        const dot = document.createElement(\'button\');\n        dot.type = \'button\';\n        dot.style.bottom = `${item.pct}%`;\n        const jitter = (!compact && k > 1) ? ((ph % 2 === 0 ? -1 : 1) * (ph > 0 ? 2 : 0)) : 0;\n        dot.style.left = `calc(50% + ${jitter}px)`;\n\n        let dotStyle = \'w-2.5 h-2.5 sm:w-3 sm:h-3 bg-emerald-400/80 border border-slate-950\';\n        if (isThisChamp) dotStyle = \'w-3.5 h-3.5 sm:w-4 sm:h-4 bg-amber-400 border-2 border-white shadow-lg shadow-amber-400/50\';\n        else if (isThisSelected) dotStyle = \'w-3.5 h-3.5 sm:w-4 sm:h-4 bg-cyan-300 border-2 border-white shadow-lg shadow-cyan-400/50\';\n        else if (compact) dotStyle = \'w-2 h-2 bg-emerald-400/80 border border-slate-950\';\n        else if (!isSelectedK) dotStyle = \'w-2 h-2 sm:w-2.5 sm:h-2.5 bg-slate-400/75 border border-slate-900\';\n\n        dot.className = `phase-dot absolute -translate-x-1/2 translate-y-1/2 rounded-full ${dotStyle} ${isThisSelected ? \'selected-dot\' : \'\'} ${isThisChamp || isThisSelected ? \'z-20\' : \'\'}`;\n        dot.title = `k=${k}, phase=${ph} → ${item.pct.toFixed(1)}% ($${item.net.toFixed(2)})`;\n        dot.onclick = (e) => { e.stopPropagation(); selectCoordinate(k, ph); };\n        col.appendChild(dot);\n      }\n    }\n    colsGrid.appendChild(col);\n\n    const xBtn = document.createElement(\'button\');\n    xBtn.type = \'button\';\n    xBtn.onclick = () => selectKColumn(k);\n    if (compact) {\n      const show = isSelectedK || isChampK || (k - lo) % xStep === 0;\n      xBtn.className = `py-1 font-mono text-[10px] font-semibold flex justify-center overflow-visible whitespace-nowrap ${\n        isSelectedK ? \'text-cyan-300 font-bold\' : (isChampK ? \'text-amber-300 font-bold\' : \'text-tv-muted hover:text-white\')}`;\n      xBtn.textContent = show ? (isChampK ? `★${k}` : `${k}`) : \'\';\n    } else {\n      xBtn.className = `py-1.5 rounded-lg font-mono text-xs font-semibold transition flex flex-col items-center ${\n        isSelectedK ? \'bg-cyan-400 text-slate-950 font-bold shadow-sm\'\n          : (isChampK ? \'bg-amber-500/15 text-amber-300 border border-amber-500/30\' : \'text-tv-muted hover:text-white hover:bg-tv-card\')}`;\n      xBtn.innerHTML = n <= 12\n        ? `<span>k=${k}</span><span class="text-[9px] font-normal opacity-80 hidden sm:inline">${k} เฟส</span>`\n        : `<span>${n <= 20 ? \'k\' : \'\'}${k}</span>`;\n    }\n    xLabelsGrid.appendChild(xBtn);\n  }\n  const note = document.getElementById(\'range-compact-note\');\n  if (note) note.textContent = compact\n    ? `โหมดย่อ ${n} คอลัมน์ (k ${lo}–${hi}): แสดงจุดทุกเฟสเฉพาะ k ที่เลือก · ชี้คอลัมน์เพื่อดูค่า · ซูมช่วง k ≤ ${COMPACT_COLS} เพื่อเห็นจุดทุกคอลัมน์`\n    : \'\';\n}\n\n// =========================================================================\n// View 2: horizontal ranking\n// =========================================================================\nfunction renderHorizontalRanking() {\n  const m = M();\n  const [lo, hi] = kRange(m.K);\n  const list = document.getElementById(\'horizontal-rank-list\');\n  list.innerHTML = \'\';\n  list.style.maxHeight = (hi - lo + 1) > 14 ? \'32rem\' : \'\';\n  list.style.overflowY = (hi - lo + 1) > 14 ? \'auto\' : \'\';\n  list.style.paddingRight = (hi - lo + 1) > 14 ? \'4px\' : \'\';\n  let selRow = null;\n  for (let k = lo; k <= hi; k++) {\n    const ks = m.kStats[k];\n    const isSelK = (k === state.selK);\n    const isChampK = (k === m.champ.k);\n    const row = document.createElement(\'div\');\n    row.onclick = () => selectKColumn(k);\n    row.className = `p-2.5 rounded-xl border cursor-pointer transition flex items-center gap-3 ${\n      isSelK ? \'bg-tv-card border-cyan-400/50\' : \'bg-tv-panel/60 border-tv-border hover:border-tv-borderLight\'}`;\n    row.innerHTML = `\n      <div class="w-16 shrink-0 font-mono text-xs font-bold ${isSelK ? \'text-cyan-300\' : \'text-white\'}">\n        k = ${k} ${isChampK ? \'<span class="text-amber-400">★</span>\' : \'\'}\n      </div>\n      <div class="flex-1">\n        <div class="relative h-6 bg-tv-bg rounded-lg overflow-hidden border border-tv-border">\n          <div style="width: ${ks.maxPct}%" class="absolute inset-y-0 left-0 ${isChampK ? \'bg-emerald-500/35\' : \'bg-cyan-500/25\'} rounded-r-md"></div>\n          <div style="width: ${ks.meanPct}%" class="absolute inset-y-1 left-0 ${isChampK ? \'bg-emerald-400\' : \'bg-cyan-400\'} rounded-r"></div>\n          <div class="relative z-10 h-full px-2.5 flex items-center justify-between font-mono text-[11px]">\n            <span class="text-slate-950 font-bold drop-shadow-sm bg-white/85 px-1.5 rounded text-[10px]">เฉลี่ย ${ks.meanPct.toFixed(0)}%</span>\n            <span class="text-slate-200 font-semibold">สูงสุด ${formatMetric(ks.maxNet, ks.maxPct, 1)} (ph=${ks.bestPh})</span>\n          </div>\n        </div>\n      </div>`;\n    if (isSelK) selRow = row;\n    list.appendChild(row);\n  }\n  if (selRow && list.style.overflowY === \'auto\' && !list.closest(\'.hidden\'))\n    list.scrollTop = Math.max(0, selRow.offsetTop - list.offsetTop - list.clientHeight / 2);\n}\n\n// =========================================================================\n// View 3: 2D matrix (canvas so k up to N still renders fast)\n// =========================================================================\nfunction cellBg(pct) {\n  if (pct >= 80) return \'rgba(16, 185, 129, 0.45)\';\n  if (pct >= 55) return \'rgba(16, 185, 129, 0.24)\';\n  if (pct < 30) return \'rgba(244, 63, 94, 0.18)\';\n  return \'rgba(34, 211, 238, 0.12)\';\n}\n\nfunction renderHeatmapMatrix() {\n  const m = M();\n  const host = document.getElementById(\'heatmap-grid\');\n  if (host.closest(\'.hidden\')) return;          // drawn when the view is opened\n  const [lo, hi] = kRange(m.K);\n  const lab = document.getElementById(\'mx-range-label\');\n  if (lab) lab.textContent = `แนวนอน: ความถี่ k (${lo}–${hi}) · แนวตั้ง: จังหวะเริ่ม phase (0–${hi - 1}) · ชี้ช่องเพื่อดูค่า · คลิกเพื่อเลือก`;\n  const vals = [];\n  for (let k = lo; k <= hi; k++) { const g = m.grid[k]; for (let ph = 0; ph < k; ph++) vals.push(clamp01(g[ph] / m.perfectNet * 100)); }\n  const sc = makeScale(vals);\n  renderColorBar(host, \'mx-color-bar\', sc, \'ช่อง k×phase\');\n  const cols = hi - lo + 1, rows = hi;\n  drawHeatCanvas({\n    host, cols, rows, labelW: 48, headH: 32, minCw: 2,\n    rowH: (r, cw) => r <= 16 ? 32 : Math.max(2, Math.min(32, cw, 900 / r)),\n    colLabel: (c) => {\n      const k = lo + c;\n      return k === state.selK ? { text: `k${k}`, bg: \'rgba(34,211,238,0.20)\', color: \'#67E8F9\', always: true }\n        : k === m.champ.k ? { text: `★${k}`, color: \'#FCD34D\', always: true } : { text: `k${k}` };\n    },\n    rowLabel: (r) => `ph ${r}`,\n    rowLabelStyle: (r) => r === state.selPh ? { bg: \'rgba(34,211,238,0.20)\', color: \'#67E8F9\', always: true, bold: true }\n      : r === m.champ.ph ? { color: \'#FCD34D\', always: true, bold: true } : {},\n    cell: (c, r) => {\n      const k = lo + c;\n      if (r >= k) return null;\n      const it = cell(m, k, r);\n      const isSel = k === state.selK && r === state.selPh;\n      const isChamp = k === m.champ.k && r === m.champ.ph;\n      return { bg: sc.color(it.pct), color: sc.text(it.pct), ring: isChamp, outline: isSel && !isChamp, bold: isChamp,\n               text: (isChamp ? \'★\' : \'\') + formatMetric(it.net, it.pct, 0) };\n    },\n    marks: [{ c: m.champ.k - lo, r: m.champ.ph, color: \'#FBBF24\' },\n            ...((state.selK !== m.champ.k || state.selPh !== m.champ.ph) ? [{ c: state.selK - lo, r: state.selPh, color: \'#67E8F9\' }] : [])],\n    tipText: (c, r) => { const k = lo + c, it = cell(m, k, r); return `k=${k}, ph=${r} → ${it.pct.toFixed(1)}% ($${it.net.toFixed(2)}) · top ${sc.topOf(it.pct).toFixed(1)}% · เคาะ ${it.acts}`; },\n    onClick: (c, r) => selectCoordinate(lo + c, r),\n  });\n}\n\n// =========================================================================\n// Phase deck\n// =========================================================================\nfunction renderPhaseDeck() {\n  const m = M();\n  const k = state.selK;\n  const ks = m.kStats[k];\n  document.getElementById(\'deck-k-title\').textContent = `k = ${k}`;\n  document.getElementById(\'deck-k-summary\').textContent =\n    `มี ${k} จังหวะ (phase 0 ถึง ${k - 1}) · เฉลี่ย ${ks.meanPct.toFixed(1)}% · สูงสุด ${ks.maxPct.toFixed(1)}%`;\n  const deck = document.getElementById(\'phase-cards-grid\');\n  deck.innerHTML = \'\';\n  const big = k > 15;\n  deck.style.maxHeight = big ? \'26rem\' : \'\';\n  deck.style.overflowY = big ? \'auto\' : \'\';\n  deck.style.paddingRight = big ? \'4px\' : \'\';\n  let selCard = null;\n  for (let ph = 0; ph < k; ph++) {\n    const item = cell(m, k, ph);\n    const isSel = (ph === state.selPh);\n    const isBestOfK = (ph === ks.bestPh);\n    const isGlobalChamp = (k === m.champ.k && ph === m.champ.ph);\n    const card = document.createElement(\'button\');\n    card.type = \'button\';\n    card.onclick = () => selectCoordinate(k, ph);\n    card.className = `phase-card text-left p-3 rounded-xl border flex flex-col justify-between gap-2 cursor-pointer ${\n      isSel ? \'bg-tv-card border-cyan-400 ring-1 ring-cyan-400/50 shadow-md\' : \'bg-tv-panel border-tv-border hover:border-tv-borderLight\'}`;\n    let badgeHTML;\n    if (isGlobalChamp) badgeHTML = `<span class="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded bg-amber-400 text-slate-950">★ แชมเปียน</span>`;\n    else if (isBestOfK) badgeHTML = `<span class="text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">ดีสุดของ k${k}</span>`;\n    else badgeHTML = `<span class="text-[10px] font-mono text-tv-dim">เคาะ ${item.acts} ครั้ง</span>`;\n    card.innerHTML = `\n      <div class="flex items-center justify-between w-full">\n        <span class="font-mono text-xs font-bold ${isSel ? \'text-cyan-300\' : \'text-slate-200\'}">phase ${ph}</span>\n        ${badgeHTML}\n      </div>\n      <div class="flex items-baseline justify-between w-full">\n        <span class="font-mono text-lg font-bold ${isGlobalChamp ? \'text-amber-300\' : (isSel ? \'text-white\' : \'text-slate-200\')}">${item.pct.toFixed(1)}%</span>\n        <span class="font-mono text-xs text-tv-muted">$${item.net.toFixed(0)}</span>\n      </div>\n      <div class="w-full h-1.5 bg-tv-bg rounded-full overflow-hidden">\n        <div style="width: ${item.pct}%" class="h-full rounded-full ${isGlobalChamp ? \'bg-amber-400\' : (isSel ? \'bg-cyan-400\' : \'bg-emerald-400/75\')}"></div>\n      </div>`;\n    if (isSel) selCard = card;\n    deck.appendChild(card);\n  }\n  if (big && selCard) deck.scrollTop = Math.max(0, selCard.offsetTop - deck.offsetTop - deck.clientHeight / 2);\n}\n\n// =========================================================================\n// Price chart + bit sequence\n// =========================================================================\nfunction renderPriceAndBits() {\n  const m = M();\n  const N = m.N;\n  const cur = cell(m, state.selK, state.selPh);\n  const actions = cur.actions;\n  document.getElementById(\'chart-slice-label\').textContent = `actions[${state.selPh}::${state.selK}] = 1`;\n  document.getElementById(\'price-insight-acts\').textContent = `เคาะ ${cur.acts}/${N} แท่ง`;\n  document.getElementById(\'n-bars\').textContent = N;\n  document.getElementById(\'n-bits\').textContent = `${N} แท่ง (#0 – #${N - 1})`;\n\n  const svg = document.getElementById(\'price-svg\');\n  const W = 640, H = 200, padL = 42, padR = 14, padT = 16, padB = 26;\n  const pw = W - padL - padR, ph = H - padT - padB;\n  const minP = Math.min(...m.prices) * 0.99;\n  const maxP = Math.max(...m.prices) * 1.01;\n  const xOf = (i) => padL + (i / (N - 1)) * pw;\n  const yOf = (p) => padT + (1 - (p - minP) / (maxP - minP)) * ph;\n  const dec = maxP < 20 ? 2 : 1;\n\n  let out = \'\';\n  for (let g = 0; g <= 2; g++) {\n    const y = padT + (g / 2) * ph;\n    const val = (maxP - (g / 2) * (maxP - minP)).toFixed(dec);\n    out += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="#242B3D" stroke-width="1"/>`;\n    out += `<text x="${padL - 6}" y="${y + 4}" text-anchor="end" fill="#64748B" font-size="10" font-family="JetBrains Mono">$${val}</text>`;\n  }\n  const days = m.win.days;\n  const step = Math.max(1, Math.ceil(days.length / 6));\n  days.forEach(([idx, label], di) => {\n    const x = xOf(idx).toFixed(1);\n    out += `<line x1="${x}" y1="${padT}" x2="${x}" y2="${H - padB}" stroke="#1A1F2E" stroke-width="1"/>`;\n    if (di % step === 0) out += `<text x="${x}" y="${H - 8}" fill="#64748B" font-size="9" font-family="JetBrains Mono">${label}</text>`;\n  });\n  const big = N > 120;\n  if (!big) {\n    for (let i = 0; i < N; i++) if (actions[i] === 1)\n      out += `<line x1="${xOf(i).toFixed(1)}" y1="${padT}" x2="${xOf(i).toFixed(1)}" y2="${H - padB}" stroke="rgba(16,185,129,0.14)" stroke-width="1"/>`;\n  }\n  out += `<polyline fill="none" stroke="#64748B" stroke-width="1.4" stroke-linejoin="round" points="${m.prices.map((p, i) => `${xOf(i).toFixed(1)},${yOf(p).toFixed(1)}`).join(\' \')}"/>`;\n  for (let i = 0; i < N; i++) if (m.perfectActs[i] === 1)\n    out += `<circle cx="${xOf(i).toFixed(1)}" cy="${yOf(m.prices[i]).toFixed(1)}" r="5.5" fill="none" stroke="#F59E0B" stroke-width="1.5" opacity="0.9"/>`;\n  const r = k => (k <= 2 ? 1.6 : 2.6);\n  for (let i = 0; i < N; i++) {\n    if (actions[i] === 1) {\n      const color = i === 0 ? \'#F43F5E\' : \'#10B981\';\n      out += `<circle cx="${xOf(i).toFixed(1)}" cy="${yOf(m.prices[i]).toFixed(1)}" r="${i === 0 ? 4.5 : r(state.selK)}" fill="${color}" stroke="#0B0E14" stroke-width="0.8"/>`;\n    }\n  }\n  svg.innerHTML = out;\n\n  const bitGrid = document.getElementById(\'bit-sequence-grid\');\n  bitGrid.innerHTML = \'\';\n  actions.forEach((bit, idx) => {\n    const c = document.createElement(\'div\');\n    let style = \'bg-tv-panel/50 border-tv-border text-tv-dim opacity-60\';\n    if (idx === 0) style = \'bg-rose-500/20 border-rose-500/60 text-rose-300 font-bold\';\n    else if (bit === 1) style = \'bg-emerald-500/20 border-emerald-500/60 text-emerald-300 font-bold\';\n    c.className = `h-4 rounded-sm border font-mono text-[8px] leading-none flex items-center justify-center ${style}`;\n    c.title = `#${idx}`;\n    c.textContent = bit;\n    bitGrid.appendChild(c);\n  });\n  const activePct = Math.round((cur.acts / N) * 100);\n  document.getElementById(\'bit-count-badge\').textContent = `เคาะ ${cur.acts}/${N} (${activePct}%)`;\n  document.getElementById(\'py-code-text\').textContent =\n    `actions = np.zeros(${N}, dtype=np.int32); actions[${state.selPh}::${state.selK}] = 1; actions[0] = 1`;\n  document.getElementById(\'protocol-code-text\').textContent = `P:${N}:${state.selK}:${state.selPh}`;\n}\n\n// =========================================================================\n// Cross-window consistency heatmap (windows x k, value = mean over phases)\n// k columns go up to the largest K of the ticker\'s windows; a window shorter\n// than k leaves that cell empty. The "best k" only counts k that every window has.\n// =========================================================================\nfunction renderConsistency() {\n  const wins = state.models[state.ticker];\n  const nW = wins.length;\n  const Kall = Math.max(...wins.map(w => w.K));\n  // the still-running window (few bars so far) must not shrink the k range used for "best k"\n  const done = wins.filter(w => !w.win.partial);\n  const Kcommon = Math.min(...(done.length ? done : wins).map(w => w.K));\n  const sums = new Float64Array(Kall + 1), sumsNet = new Float64Array(Kall + 1), cnt = new Int32Array(Kall + 1), wins1 = new Int32Array(Kall + 1);\n  wins.forEach(m => {\n    wins1[m.champ.k]++;\n    for (let k = 1; k <= m.K; k++) { sums[k] += m.kStats[k].meanPct; sumsNet[k] += m.kStats[k].meanNet; cnt[k]++; }\n  });\n  const avgOf = (k) => sums[k] / cnt[k];\n  let bestK = 1;\n  for (let k = 2; k <= Kcommon; k++) if (avgOf(k) > avgOf(bestK)) bestK = k;\n\n  const [lo, hi] = kRange(Kall);\n  const cols = hi - lo + 1;\n  const AVG = nW, CH = nW + 1;\n  const vals = [];\n  wins.forEach(m => { for (let k = lo; k <= Math.min(hi, m.K); k++) vals.push(m.kStats[k].meanPct); });\n  const sc = makeScale(vals);\n  const host = document.getElementById(\'cons-grid\');\n  renderColorBar(host, \'cons-color-bar\', sc, \'ค่าเฉลี่ยทุกเฟส\');\n  drawHeatCanvas({\n    host, cols, rows: nW + 2, labelW: 118, headH: 32, minCw: 3,\n    rowH: () => 32,\n    colLabel: (c) => {\n      const k = lo + c;\n      return k === state.selK ? { text: `k${k}`, bg: \'rgba(34,211,238,0.20)\', color: \'#67E8F9\', always: true }\n        : k === bestK ? { text: `★${k}`, color: \'#FCD34D\', always: true } : { text: `k${k}` };\n    },\n    rowLabel: (r) => r < nW ? `W${r + 1} ${halfLabel(wins[r].win)}` : (r === AVG ? \'เฉลี่ยทุก window\' : \'แชมเปียน (นับ)\'),\n    rowLabelStyle: (r) => r === state.wi ? { bg: \'rgba(34,211,238,0.20)\', color: \'#67E8F9\', bold: true }\n      : r === AVG ? { color: \'#FCD34D\', bold: true } : {},\n    cell: (c, r) => {\n      const k = lo + c;\n      if (r < nW) {\n        const m = wins[r];\n        if (k > m.K) return null;\n        const ks = m.kStats[k];\n        const isChampK = k === m.champ.k;\n        return { bg: sc.color(ks.meanPct), color: sc.text(ks.meanPct), ring: isChampK, outline: r === state.wi && k === state.selK, bold: isChampK,\n                 text: (isChampK ? \'★\' : \'\') + formatMetric(ks.meanNet, ks.meanPct, 0) };\n      }\n      if (!cnt[k]) return null;\n      if (r === AVG) {\n        const avg = sums[k] / cnt[k];\n        return { bg: sc.color(avg), color: sc.text(avg), ring: k === bestK, bold: true,\n                 text: state.unit === \'pct\' ? `${avg.toFixed(1)}%` : `$${Math.round(sumsNet[k] / cnt[k])}` };\n      }\n      return { bg: wins1[k] ? \'rgba(16,185,129,0.30)\' : \'#131722\', color: wins1[k] ? \'#6EE7B7\' : \'#64748B\', bold: !!wins1[k],\n               text: `${wins1[k]}/${nW}` };\n    },\n    tipText: (c, r) => {\n      const k = lo + c;\n      if (r < nW) { const ks = wins[r].kStats[k]; return `W${r + 1} k=${k}: เฉลี่ย ${ks.meanPct.toFixed(1)}% (top ${sc.topOf(ks.meanPct).toFixed(1)}%) / สูงสุด ${ks.maxPct.toFixed(1)}% (ph=${ks.bestPh})`; }\n      if (r === AVG) return `k=${k}: เฉลี่ย ${(sums[k] / cnt[k]).toFixed(1)}% จาก ${cnt[k]}/${nW} window`;\n      return `k=${k}: เป็นแชมเปียน ${wins1[k]}/${nW} window`;\n    },\n    marks: wins.map((m, r) => ({ c: m.champ.k - lo, r, color: \'#FBBF24\', cross: false })),\n    onClick: (c, r) => {\n      if (r >= nW) return;\n      const k = lo + c;\n      state.wi = r; state.selK = k; state.selPh = wins[r].kStats[k].bestPh;\n      renderAll(); window.scrollTo({ top: 0, behavior: \'smooth\' });\n    },\n  });\n\n  const champKs = [...wins1.keys()].filter(k => wins1[k] > 0).sort((a, b) => wins1[b] - wins1[a]).slice(0, 3)\n    .map(k => `k${k}×${wins1[k]}`).join(\', \');\n  document.getElementById(\'cons-best\').textContent =\n    `k ที่เฉลี่ยดีที่สุดตลอด ${nW} window (k 1–${Kcommon} ที่ทุก window ที่จบแล้วมี): k=${bestK} (${avgOf(bestK).toFixed(1)}% ของเพดาน) · แชมเปียนบ่อย: ${champKs}`;\n}\n\n// =========================================================================\n// Controllers\n// =========================================================================\nfunction renderAll() {\n  const m = M();\n  if (state.selK > m.K) { state.selK = m.champ.k; state.selPh = m.champ.ph; }\n  if (state.selPh >= state.selK) state.selPh = m.kStats[state.selK].bestPh;\n  ensureKVisible(state.selK, m.K);\n  renderZoomBar();\n  renderWindowBar();\n  renderKPIs();\n  renderRangeSpectrum();\n  renderHorizontalRanking();\n  renderHeatmapMatrix();\n  renderPhaseDeck();\n  renderPriceAndBits();\n  renderConsistency();\n}\n\nfunction selectCoordinate(k, ph) { state.selK = k; state.selPh = ph; renderAll(); }\nfunction selectKColumn(k) { state.selK = k; state.selPh = M().kStats[k].bestPh; renderAll(); }\nfunction selectChampion() { const c = M().champ; state.selK = c.k; state.selPh = c.ph; renderAll(); }\nfunction setWindow(i) { state.wi = i; selectChampion(); }\n\nfunction setTicker(sym) {\n  state.ticker = sym;\n  document.querySelectorAll(\'.mkt-btn\').forEach(b => {\n    b.className = \'mkt-btn px-3 sm:px-4 py-1.5 rounded-lg text-xs font-mono font-semibold text-tv-muted hover:text-white transition whitespace-nowrap\';\n  });\n  const btn = document.getElementById(`tk-${sym}`);\n  if (btn) btn.className = \'mkt-btn px-3 sm:px-4 py-1.5 rounded-lg text-xs font-mono font-semibold bg-tv-card text-white shadow-sm border border-tv-borderLight transition whitespace-nowrap\';\n  state.wi = state.models[sym].length - 1;\n  selectChampion();\n}\n\nfunction setUnit(unit) {\n  state.unit = unit;\n  const activeCls = \'px-3 py-1.5 rounded-lg text-xs font-mono font-semibold bg-cyan-400 text-slate-950 transition\';\n  const inactiveCls = \'px-3 py-1.5 rounded-lg text-xs font-mono font-semibold text-tv-muted hover:text-white transition\';\n  document.getElementById(\'btn-unit-pct\').className = unit === \'pct\' ? activeCls : inactiveCls;\n  document.getElementById(\'btn-unit-usd\').className = unit === \'usd\' ? activeCls : inactiveCls;\n  const activeM = \'px-2.5 py-1 rounded-md text-xs font-mono font-semibold bg-cyan-400 text-slate-950 transition\';\n  const inactiveM = \'px-2.5 py-1 rounded-md text-xs font-mono font-semibold text-tv-muted hover:text-white transition\';\n  document.getElementById(\'btn-unit-pct-m\').className = unit === \'pct\' ? activeM : inactiveM;\n  document.getElementById(\'btn-unit-usd-m\').className = unit === \'usd\' ? activeM : inactiveM;\n  renderAll();\n}\n\nfunction setChartMode(mode) {\n  state.chartMode = mode;\n  [\'range\', \'Rank\', \'matrix\'].forEach(x => {\n    const viewEl = document.getElementById(`view-${x}`);\n    const btnEl = document.getElementById(`mode-${x}`);\n    if (x === mode) {\n      viewEl.classList.remove(\'hidden\');\n      btnEl.className = \'chart-mode-btn px-3 py-1.5 rounded-lg text-xs font-medium bg-tv-card text-cyan-300 border border-cyan-500/30 shadow-sm transition\';\n    } else {\n      viewEl.classList.add(\'hidden\');\n      btnEl.className = \'chart-mode-btn px-3 py-1.5 rounded-lg text-xs font-medium text-tv-muted hover:text-white transition\';\n    }\n  });\n  if (mode === \'matrix\') renderHeatmapMatrix();\n  if (mode === \'Rank\') renderHorizontalRanking();\n}\n\nfunction copyText(elementId, btn) {\n  const text = document.getElementById(elementId).textContent.trim();\n  const temp = document.createElement(\'textarea\');\n  temp.value = text;\n  document.body.appendChild(temp);\n  temp.select();\n  document.execCommand(\'copy\');\n  document.body.removeChild(temp);\n  const original = btn.textContent;\n  btn.textContent = \'✓ คัดลอกแล้ว\';\n  setTimeout(() => { btn.textContent = original; }, 1200);\n}\n\nlet resizeT = null;\nwindow.addEventListener(\'resize\', () => {\n  clearTimeout(resizeT);\n  resizeT = setTimeout(() => { if (Object.keys(state.models).length) { renderHeatmapMatrix(); renderConsistency(); } }, 150);\n});\n\nwindow.onload = function () {\n  TICKERS.forEach(sym => { state.models[sym] = (DATA[sym] || []).map(buildModel); });\n  setTicker(TICKERS[0]);\n};\n\n</script>\n</body>\n</html>\n'

DNA_SPECTRUM_EXPLORER_DEFAULT_META_JSON: str = '{"tf": "D", "tfLabel": "รายวัน", "asof": "2026-11-09", "lastBar": "2026-11-06", "pending": []}'

DNA_SPECTRUM_EXPLORER_DEFAULT_DATA_JSON: str = '{"AAPL":[{"wa":"2025-11-15","wb":"2025-11-30","a":"2025-11-17","b":"2025-11-28","days":[[0,"11/17"],[1,"11/18"],[2,"11/19"],[3,"11/20"],[4,"11/21"],[5,"11/24"],[6,"11/25"],[7,"11/26"],[8,"11/28"]],"c":[223.22,216.81,220.44,207.8,207.06,195.38,200.76,201.77,208.62],"partial":false,"headPartial":false},{"wa":"2025-12-01","wb":"2025-12-14","a":"2025-12-01","b":"2025-12-12","days":[[0,"12/01"],[1,"12/02"],[2,"12/03"],[3,"12/04"],[4,"12/05"],[5,"12/08"],[6,"12/09"],[7,"12/10"],[8,"12/11"],[9,"12/12"]],"c":[205.99,208.04,206.55,202.74,203.47,197.08,195.33,198.73,199.02,196.98],"partial":false,"headPartial":false},{"wa":"2025-12-15","wb":"2025-12-31","a":"2025-12-15","b":"2025-12-31","days":[[0,"12/15"],[1,"12/16"],[2,"12/17"],[3,"12/18"],[4,"12/19"],[5,"12/22"],[6,"12/23"],[7,"12/24"],[8,"12/26"],[9,"12/29"],[10,"12/30"],[11,"12/31"]],"c":[207.76,208.06,205.01,205.83,203.14,201.18,199.42,209.52,209.63,210.55,214.1,224.9],"partial":false,"headPartial":false},{"wa":"2026-01-01","wb":"2026-01-14","a":"2026-01-02","b":"2026-01-14","days":[[0,"01/02"],[1,"01/05"],[2,"01/06"],[3,"01/07"],[4,"01/08"],[5,"01/09"],[6,"01/12"],[7,"01/13"],[8,"01/14"]],"c":[223.64,220.15,233.77,225.24,223.18,226.9,239.85,234.15,219.94],"partial":false,"headPartial":false},{"wa":"2026-01-15","wb":"2026-01-31","a":"2026-01-15","b":"2026-01-30","days":[[0,"01/15"],[1,"01/16"],[2,"01/20"],[3,"01/21"],[4,"01/22"],[5,"01/23"],[6,"01/26"],[7,"01/27"],[8,"01/28"],[9,"01/29"],[10,"01/30"]],"c":[223.58,220.66,218.53,221.05,222.28,223.9,221.49,228.63,237.23,237.42,234.74],"partial":false,"headPartial":false},{"wa":"2026-02-01","wb":"2026-02-14","a":"2026-02-02","b":"2026-02-13","days":[[0,"02/02"],[1,"02/03"],[2,"02/04"],[3,"02/05"],[4,"02/06"],[5,"02/09"],[6,"02/10"],[7,"02/11"],[8,"02/12"],[9,"02/13"]],"c":[239.04,241.89,235.6,232.89,239.0,238.43,236.47,237.72,237.52,237.51],"partial":false,"headPartial":false},{"wa":"2026-02-15","wb":"2026-02-28","a":"2026-02-17","b":"2026-02-27","days":[[0,"02/17"],[1,"02/18"],[2,"02/19"],[3,"02/20"],[4,"02/23"],[5,"02/24"],[6,"02/25"],[7,"02/26"],[8,"02/27"]],"c":[226.04,235.93,237.11,231.91,237.59,239.42,239.53,245.8,259.7],"partial":false,"headPartial":false},{"wa":"2026-03-01","wb":"2026-03-14","a":"2026-03-02","b":"2026-03-13","days":[[0,"03/02"],[1,"03/03"],[2,"03/04"],[3,"03/05"],[4,"03/06"],[5,"03/09"],[6,"03/10"],[7,"03/11"],[8,"03/12"],[9,"03/13"]],"c":[263.29,273.73,288.49,281.13,277.45,282.83,270.76,268.95,277.64,284.88],"partial":false,"headPartial":false},{"wa":"2026-03-15","wb":"2026-03-31","a":"2026-03-16","b":"2026-03-31","days":[[0,"03/16"],[1,"03/17"],[2,"03/18"],[3,"03/19"],[4,"03/20"],[5,"03/23"],[6,"03/24"],[7,"03/25"],[8,"03/26"],[9,"03/27"],[10,"03/30"],[11,"03/31"]],"c":[289.01,275.32,291.5,295.51,301.18,304.9,291.82,281.35,272.8,285.88,279.94,278.24],"partial":false,"headPartial":false},{"wa":"2026-04-01","wb":"2026-04-14","a":"2026-04-01","b":"2026-04-14","days":[[0,"04/01"],[1,"04/02"],[2,"04/06"],[3,"04/07"],[4,"04/08"],[5,"04/09"],[6,"04/10"],[7,"04/13"],[8,"04/14"]],"c":[276.47,280.42,283.93,285.83,276.81,255.09,256.15,257.87,265.8],"partial":false,"headPartial":false},{"wa":"2026-04-15","wb":"2026-04-30","a":"2026-04-15","b":"2026-04-30","days":[[0,"04/15"],[1,"04/16"],[2,"04/17"],[3,"04/20"],[4,"04/21"],[5,"04/22"],[6,"04/23"],[7,"04/24"],[8,"04/27"],[9,"04/28"],[10,"04/29"],[11,"04/30"]],"c":[265.12,261.07,263.75,253.5,246.9,240.68,242.42,255.09,259.96,267.38,268.97,270.85],"partial":false,"headPartial":false},{"wa":"2026-05-01","wb":"2026-05-14","a":"2026-05-01","b":"2026-05-14","days":[[0,"05/01"],[1,"05/04"],[2,"05/05"],[3,"05/06"],[4,"05/07"],[5,"05/08"],[6,"05/11"],[7,"05/12"],[8,"05/13"],[9,"05/14"]],"c":[267.8,265.92,249.77,242.26,234.46,239.37,243.34,250.28,260.02,258.04],"partial":false,"headPartial":false},{"wa":"2026-05-15","wb":"2026-05-31","a":"2026-05-15","b":"2026-05-29","days":[[0,"05/15"],[1,"05/18"],[2,"05/19"],[3,"05/20"],[4,"05/21"],[5,"05/22"],[6,"05/26"],[7,"05/27"],[8,"05/28"],[9,"05/29"]],"c":[264.85,277.79,277.73,285.28,284.1,272.65,273.47,267.85,268.58,265.47],"partial":false,"headPartial":false},{"wa":"2026-06-01","wb":"2026-06-14","a":"2026-06-01","b":"2026-06-12","days":[[0,"06/01"],[1,"06/02"],[2,"06/03"],[3,"06/04"],[4,"06/05"],[5,"06/08"],[6,"06/09"],[7,"06/10"],[8,"06/11"],[9,"06/12"]],"c":[267.59,253.29,266.49,275.55,273.13,266.93,266.95,271.46,274.33,277.09],"partial":false,"headPartial":false},{"wa":"2026-06-15","wb":"2026-06-30","a":"2026-06-15","b":"2026-06-30","days":[[0,"06/15"],[1,"06/16"],[2,"06/17"],[3,"06/18"],[4,"06/22"],[5,"06/23"],[6,"06/24"],[7,"06/25"],[8,"06/26"],[9,"06/29"],[10,"06/30"]],"c":[263.45,256.0,264.6,266.19,256.06,250.5,241.43,235.27,242.22,240.87,247.69],"partial":false,"headPartial":false},{"wa":"2026-07-01","wb":"2026-07-14","a":"2026-07-01","b":"2026-07-14","days":[[0,"07/01"],[1,"07/02"],[2,"07/06"],[3,"07/07"],[4,"07/08"],[5,"07/09"],[6,"07/10"],[7,"07/13"],[8,"07/14"]],"c":[248.86,239.4,235.47,249.14,244.3,237.8,245.26,245.82,245.78],"partial":false,"headPartial":false},{"wa":"2026-07-15","wb":"2026-07-31","a":"2026-07-15","b":"2026-07-31","days":[[0,"07/15"],[1,"07/16"],[2,"07/17"],[3,"07/20"],[4,"07/21"],[5,"07/22"],[6,"07/23"],[7,"07/24"],[8,"07/27"],[9,"07/28"],[10,"07/29"],[11,"07/30"],[12,"07/31"]],"c":[254.95,243.53,246.11,239.51,230.95,226.16,232.8,231.96,221.11,214.86,223.22,229.05,222.67],"partial":false,"headPartial":false},{"wa":"2026-08-01","wb":"2026-08-14","a":"2026-08-03","b":"2026-08-14","days":[[0,"08/03"],[1,"08/04"],[2,"08/05"],[3,"08/06"],[4,"08/07"],[5,"08/10"],[6,"08/11"],[7,"08/12"],[8,"08/13"],[9,"08/14"]],"c":[236.45,234.24,227.1,226.3,231.72,241.25,247.6,244.89,241.38,232.4],"partial":false,"headPartial":false},{"wa":"2026-08-15","wb":"2026-08-31","a":"2026-08-17","b":"2026-08-31","days":[[0,"08/17"],[1,"08/18"],[2,"08/19"],[3,"08/20"],[4,"08/21"],[5,"08/24"],[6,"08/25"],[7,"08/26"],[8,"08/27"],[9,"08/28"],[10,"08/31"]],"c":[231.96,228.43,229.86,222.41,219.47,214.53,207.93,201.63,196.38,190.44,193.21],"partial":false,"headPartial":false},{"wa":"2026-09-01","wb":"2026-09-14","a":"2026-09-01","b":"2026-09-14","days":[[0,"09/01"],[1,"09/02"],[2,"09/03"],[3,"09/04"],[4,"09/08"],[5,"09/09"],[6,"09/10"],[7,"09/11"],[8,"09/14"]],"c":[193.7,187.16,189.62,191.01,198.14,197.16,196.1,192.64,193.26],"partial":false,"headPartial":false},{"wa":"2026-09-15","wb":"2026-09-30","a":"2026-09-15","b":"2026-09-30","days":[[0,"09/15"],[1,"09/16"],[2,"09/17"],[3,"09/18"],[4,"09/21"],[5,"09/22"],[6,"09/23"],[7,"09/24"],[8,"09/25"],[9,"09/28"],[10,"09/29"],[11,"09/30"]],"c":[195.7,196.57,195.29,198.11,198.57,199.83,195.06,196.09,191.09,186.97,187.74,197.85],"partial":false,"headPartial":false},{"wa":"2026-10-01","wb":"2026-10-14","a":"2026-10-01","b":"2026-10-14","days":[[0,"10/01"],[1,"10/02"],[2,"10/05"],[3,"10/06"],[4,"10/07"],[5,"10/08"],[6,"10/09"],[7,"10/12"],[8,"10/13"],[9,"10/14"]],"c":[202.89,202.92,201.99,207.09,214.42,221.11,220.06,221.12,219.25,217.65],"partial":false,"headPartial":false},{"wa":"2026-10-15","wb":"2026-10-31","a":"2026-10-15","b":"2026-10-30","days":[[0,"10/15"],[1,"10/16"],[2,"10/19"],[3,"10/20"],[4,"10/21"],[5,"10/22"],[6,"10/23"],[7,"10/26"],[8,"10/27"],[9,"10/28"],[10,"10/29"],[11,"10/30"]],"c":[212.9,211.54,199.89,193.76,195.87,191.23,193.87,187.46,185.69,187.57,182.59,179.26],"partial":false,"headPartial":false},{"wa":"2026-11-01","wb":"2026-11-14","a":"2026-11-02","b":"2026-11-06","days":[[0,"11/02"],[1,"11/03"],[2,"11/04"],[3,"11/05"],[4,"11/06"]],"c":[190.6,187.27,183.9,181.2,181.35],"partial":true,"headPartial":false}],"RGTI":[{"wa":"2025-11-15","wb":"2025-11-30","a":"2025-11-17","b":"2025-11-28","days":[[0,"11/17"],[1,"11/18"],[2,"11/19"],[3,"11/20"],[4,"11/21"],[5,"11/24"],[6,"11/25"],[7,"11/26"],[8,"11/28"]],"c":[11.99,12.29,12.54,12.41,12.3,11.85,11.4,11.19,11.39],"partial":false,"headPartial":false},{"wa":"2025-12-01","wb":"2025-12-14","a":"2025-12-01","b":"2025-12-12","days":[[0,"12/01"],[1,"12/02"],[2,"12/03"],[3,"12/04"],[4,"12/05"],[5,"12/08"],[6,"12/09"],[7,"12/10"],[8,"12/11"],[9,"12/12"]],"c":[11.43,11.66,11.95,11.9,12.53,12.37,12.17,12.03,12.38,12.34],"partial":false,"headPartial":false},{"wa":"2025-12-15","wb":"2025-12-31","a":"2025-12-15","b":"2025-12-31","days":[[0,"12/15"],[1,"12/16"],[2,"12/17"],[3,"12/18"],[4,"12/19"],[5,"12/22"],[6,"12/23"],[7,"12/24"],[8,"12/26"],[9,"12/29"],[10,"12/30"],[11,"12/31"]],"c":[12.11,12.24,12.6,12.56,12.48,12.83,13.15,13.47,13.2,13.3,13.42,12.81],"partial":false,"headPartial":false},{"wa":"2026-01-01","wb":"2026-01-14","a":"2026-01-02","b":"2026-01-14","days":[[0,"01/02"],[1,"01/05"],[2,"01/06"],[3,"01/07"],[4,"01/08"],[5,"01/09"],[6,"01/12"],[7,"01/13"],[8,"01/14"]],"c":[13.2,13.23,13.12,13.88,13.93,13.67,13.65,13.85,14.13],"partial":false,"headPartial":false},{"wa":"2026-01-15","wb":"2026-01-31","a":"2026-01-15","b":"2026-01-30","days":[[0,"01/15"],[1,"01/16"],[2,"01/20"],[3,"01/21"],[4,"01/22"],[5,"01/23"],[6,"01/26"],[7,"01/27"],[8,"01/28"],[9,"01/29"],[10,"01/30"]],"c":[14.11,13.8,13.56,13.27,12.92,12.91,12.75,12.77,12.5,12.22,12.16],"partial":false,"headPartial":false},{"wa":"2026-02-01","wb":"2026-02-14","a":"2026-02-02","b":"2026-02-13","days":[[0,"02/02"],[1,"02/03"],[2,"02/04"],[3,"02/05"],[4,"02/06"],[5,"02/09"],[6,"02/10"],[7,"02/11"],[8,"02/12"],[9,"02/13"]],"c":[12.05,12.41,12.12,12.19,12.17,11.77,11.98,12.04,11.39,11.57],"partial":false,"headPartial":false},{"wa":"2026-02-15","wb":"2026-02-28","a":"2026-02-17","b":"2026-02-27","days":[[0,"02/17"],[1,"02/18"],[2,"02/19"],[3,"02/20"],[4,"02/23"],[5,"02/24"],[6,"02/25"],[7,"02/26"],[8,"02/27"]],"c":[11.84,11.6,12.1,12.15,11.79,11.96,11.85,11.45,10.82],"partial":false,"headPartial":false},{"wa":"2026-03-01","wb":"2026-03-14","a":"2026-03-02","b":"2026-03-13","days":[[0,"03/02"],[1,"03/03"],[2,"03/04"],[3,"03/05"],[4,"03/06"],[5,"03/09"],[6,"03/10"],[7,"03/11"],[8,"03/12"],[9,"03/13"]],"c":[10.89,10.69,10.86,10.97,10.82,10.68,11.16,11.64,11.87,11.76],"partial":false,"headPartial":false},{"wa":"2026-03-15","wb":"2026-03-31","a":"2026-03-16","b":"2026-03-31","days":[[0,"03/16"],[1,"03/17"],[2,"03/18"],[3,"03/19"],[4,"03/20"],[5,"03/23"],[6,"03/24"],[7,"03/25"],[8,"03/26"],[9,"03/27"],[10,"03/30"],[11,"03/31"]],"c":[11.47,11.35,11.83,11.97,11.88,11.51,11.56,11.3,11.31,11.52,11.83,11.88],"partial":false,"headPartial":false},{"wa":"2026-04-01","wb":"2026-04-14","a":"2026-04-01","b":"2026-04-14","days":[[0,"04/01"],[1,"04/02"],[2,"04/06"],[3,"04/07"],[4,"04/08"],[5,"04/09"],[6,"04/10"],[7,"04/13"],[8,"04/14"]],"c":[11.77,11.84,12.39,12.5,12.06,12.53,12.62,13.73,13.57],"partial":false,"headPartial":false},{"wa":"2026-04-15","wb":"2026-04-30","a":"2026-04-15","b":"2026-04-30","days":[[0,"04/15"],[1,"04/16"],[2,"04/17"],[3,"04/20"],[4,"04/21"],[5,"04/22"],[6,"04/23"],[7,"04/24"],[8,"04/27"],[9,"04/28"],[10,"04/29"],[11,"04/30"]],"c":[13.54,13.43,13.56,13.36,12.86,12.23,12.19,12.04,11.86,12.5,12.75,13.1],"partial":false,"headPartial":false},{"wa":"2026-05-01","wb":"2026-05-14","a":"2026-05-01","b":"2026-05-14","days":[[0,"05/01"],[1,"05/04"],[2,"05/05"],[3,"05/06"],[4,"05/07"],[5,"05/08"],[6,"05/11"],[7,"05/12"],[8,"05/13"],[9,"05/14"]],"c":[12.94,13.04,12.97,12.9,13.14,13.35,13.45,13.5,13.0,12.76],"partial":false,"headPartial":false},{"wa":"2026-05-15","wb":"2026-05-31","a":"2026-05-15","b":"2026-05-29","days":[[0,"05/15"],[1,"05/18"],[2,"05/19"],[3,"05/20"],[4,"05/21"],[5,"05/22"],[6,"05/26"],[7,"05/27"],[8,"05/28"],[9,"05/29"]],"c":[12.88,12.76,12.51,12.63,12.33,12.12,12.39,12.28,12.72,12.89],"partial":false,"headPartial":false},{"wa":"2026-06-01","wb":"2026-06-14","a":"2026-06-01","b":"2026-06-12","days":[[0,"06/01"],[1,"06/02"],[2,"06/03"],[3,"06/04"],[4,"06/05"],[5,"06/08"],[6,"06/09"],[7,"06/10"],[8,"06/11"],[9,"06/12"]],"c":[13.16,13.54,12.79,12.77,12.35,12.2,12.17,12.56,12.54,13.06],"partial":false,"headPartial":false},{"wa":"2026-06-15","wb":"2026-06-30","a":"2026-06-15","b":"2026-06-30","days":[[0,"06/15"],[1,"06/16"],[2,"06/17"],[3,"06/18"],[4,"06/22"],[5,"06/23"],[6,"06/24"],[7,"06/25"],[8,"06/26"],[9,"06/29"],[10,"06/30"]],"c":[12.89,12.87,12.9,13.21,13.19,12.99,12.88,12.42,12.82,13.06,13.2],"partial":false,"headPartial":false},{"wa":"2026-07-01","wb":"2026-07-14","a":"2026-07-01","b":"2026-07-14","days":[[0,"07/01"],[1,"07/02"],[2,"07/06"],[3,"07/07"],[4,"07/08"],[5,"07/09"],[6,"07/10"],[7,"07/13"],[8,"07/14"]],"c":[13.29,13.09,12.96,13.22,12.96,12.7,13.21,13.73,13.83],"partial":false,"headPartial":false},{"wa":"2026-07-15","wb":"2026-07-31","a":"2026-07-15","b":"2026-07-31","days":[[0,"07/15"],[1,"07/16"],[2,"07/17"],[3,"07/20"],[4,"07/21"],[5,"07/22"],[6,"07/23"],[7,"07/24"],[8,"07/27"],[9,"07/28"],[10,"07/29"],[11,"07/30"],[12,"07/31"]],"c":[14.02,14.08,14.59,14.43,14.46,14.64,15.17,15.07,15.16,14.97,14.73,14.48,14.51],"partial":false,"headPartial":false},{"wa":"2026-08-01","wb":"2026-08-14","a":"2026-08-03","b":"2026-08-14","days":[[0,"08/03"],[1,"08/04"],[2,"08/05"],[3,"08/06"],[4,"08/07"],[5,"08/10"],[6,"08/11"],[7,"08/12"],[8,"08/13"],[9,"08/14"]],"c":[14.84,14.58,14.36,14.21,14.34,14.59,14.13,14.26,13.9,14.15],"partial":false,"headPartial":false},{"wa":"2026-08-15","wb":"2026-08-31","a":"2026-08-17","b":"2026-08-31","days":[[0,"08/17"],[1,"08/18"],[2,"08/19"],[3,"08/20"],[4,"08/21"],[5,"08/24"],[6,"08/25"],[7,"08/26"],[8,"08/27"],[9,"08/28"],[10,"08/31"]],"c":[13.81,13.43,13.19,13.17,12.84,12.91,12.96,12.98,13.07,13.36,13.11],"partial":false,"headPartial":false},{"wa":"2026-09-01","wb":"2026-09-14","a":"2026-09-01","b":"2026-09-14","days":[[0,"09/01"],[1,"09/02"],[2,"09/03"],[3,"09/04"],[4,"09/08"],[5,"09/09"],[6,"09/10"],[7,"09/11"],[8,"09/14"]],"c":[13.11,13.1,12.43,12.55,12.31,12.08,11.95,11.98,12.08],"partial":false,"headPartial":false},{"wa":"2026-09-15","wb":"2026-09-30","a":"2026-09-15","b":"2026-09-30","days":[[0,"09/15"],[1,"09/16"],[2,"09/17"],[3,"09/18"],[4,"09/21"],[5,"09/22"],[6,"09/23"],[7,"09/24"],[8,"09/25"],[9,"09/28"],[10,"09/29"],[11,"09/30"]],"c":[12.22,12.01,12.22,12.35,12.23,12.22,12.47,12.6,12.82,12.86,13.23,13.96],"partial":false,"headPartial":false},{"wa":"2026-10-01","wb":"2026-10-14","a":"2026-10-01","b":"2026-10-14","days":[[0,"10/01"],[1,"10/02"],[2,"10/05"],[3,"10/06"],[4,"10/07"],[5,"10/08"],[6,"10/09"],[7,"10/12"],[8,"10/13"],[9,"10/14"]],"c":[14.12,13.56,14.14,13.94,13.46,12.66,12.79,13.14,13.47,13.34],"partial":false,"headPartial":false},{"wa":"2026-10-15","wb":"2026-10-31","a":"2026-10-15","b":"2026-10-30","days":[[0,"10/15"],[1,"10/16"],[2,"10/19"],[3,"10/20"],[4,"10/21"],[5,"10/22"],[6,"10/23"],[7,"10/26"],[8,"10/27"],[9,"10/28"],[10,"10/29"],[11,"10/30"]],"c":[13.83,13.8,13.6,13.62,13.69,13.71,13.85,13.09,13.6,13.41,13.63,13.35],"partial":false,"headPartial":false},{"wa":"2026-11-01","wb":"2026-11-14","a":"2026-11-02","b":"2026-11-06","days":[[0,"11/02"],[1,"11/03"],[2,"11/04"],[3,"11/05"],[4,"11/06"]],"c":[13.41,13.35,12.82,12.95,13.15],"partial":true,"headPartial":false}]}'

EXPLORER_TIMEFRAMES: Dict[str, Dict[str, Any]] = {
    "1d": {"tf": "D", "tfLabel": "รายวัน", "tfShort": "1D", "yf": "1d", "max_days": 3650, "default_days": 365, "min_bars": 5},
    "15m": {"tf": "M15", "tfLabel": "15 นาที", "tfShort": "15m", "yf": "15m", "max_days": 59, "default_days": 59, "min_bars": 20},
    "30m": {"tf": "M30", "tfLabel": "30 นาที", "tfShort": "30m", "yf": "30m", "max_days": 59, "default_days": 59, "min_bars": 15},
    "1h": {"tf": "H1", "tfLabel": "1 ชั่วโมง", "tfShort": "1H", "yf": "1h", "max_days": 729, "default_days": 365, "min_bars": 15},
}


@st.cache_resource
def get_default_explorer_payload() -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    """โหลดข้อมูลตั้งต้น (AAPL, RGTI) จาก template.html ต้นฉบับ 100%"""
    return (
        json.loads(DNA_SPECTRUM_EXPLORER_DEFAULT_META_JSON),
        json.loads(DNA_SPECTRUM_EXPLORER_DEFAULT_DATA_JSON),
    )


def parse_dynamic_tickers(raw_text: str) -> List[str]:
    """แยกชื่อหุ้นจากข้อความที่คั่นด้วยเว้นวรรค ลูกน้ำ หรือเซมิโคลอน"""
    if not raw_text:
        return []
    tokens = raw_text.replace(",", " ").replace(";", " ").replace("·", " ").split()
    seen: List[str] = []
    for tok in tokens:
        sym = tok.strip().upper()
        if sym and sym not in seen:
            seen.append(sym)
    return seen


def align_to_half_month_boundary(d: date) -> date:
    """เลื่อนวันเริ่มต้นไปยังต้นครึ่งเดือนถัดไป (วันที่ 1 หรือ 15) เพื่อให้ Window แรกครบสมบูรณ์"""
    if d.day == 1 or d.day == 15:
        return d
    if d.day < 15:
        return d.replace(day=15)
    if d.month == 12:
        return date(d.year + 1, 1, 1)
    return date(d.year, d.month + 1, 1)


def half_month_bounds(d: date) -> Tuple[str, str]:
    """คืนค่า (wa, wb) ของครึ่งเดือนปฏิทิน: วันที่ 1–14 หรือ 15–สิ้นเดือน"""
    if d.day <= 14:
        return f"{d.year:04d}-{d.month:02d}-01", f"{d.year:04d}-{d.month:02d}-14"
    last_day = calendar.monthrange(d.year, d.month)[1]
    return f"{d.year:04d}-{d.month:02d}-15", f"{d.year:04d}-{d.month:02d}-{last_day:02d}"


def build_explorer_windows_from_series(
    closes: pd.Series,
    window_mode: str = "half_month",
    fixed_window_size: int = 30,
    asof_date: Optional[date] = None,
    min_bars_finished: int = 5,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """แปลง Series ราคาปิดให้เป็นโครงสร้าง DATA[ticker] ของ DNA Spectrum Explorer
    [{"wa","wb","a","b","days":[[idx,"MM/DD"],...],"c":[...],"partial":bool,"headPartial":bool}, ...]"""
    clean = closes.dropna()
    if len(clean) < 2:
        return [], None

    idx = clean.index
    if getattr(idx, "tz", None) is not None:
        ny_dates = [pd.Timestamp(ts).tz_convert("America/New_York").date() for ts in idx]
    else:
        ny_dates = [pd.Timestamp(ts).date() for ts in idx]

    prices_arr = [round(float(v), 4 if abs(float(v)) < 1.0 else 2) for v in clean.to_numpy()]
    last_bar_str = ny_dates[-1].strftime("%Y-%m-%d")
    if asof_date is None:
        asof_date = max(ny_dates[-1], datetime.now().date())
    asof_str = asof_date.strftime("%Y-%m-%d")

    windows: List[Dict[str, Any]] = []

    if window_mode == "fixed_bars":
        ws = max(2, int(fixed_window_size))
        for start_i in range(0, len(prices_arr), ws):
            end_i = min(start_i + ws, len(prices_arr))
            c_slice = prices_arr[start_i:end_i]
            d_slice = ny_dates[start_i:end_i]
            if len(c_slice) < 2:
                continue
            days_meta: List[List[Any]] = []
            prev_d = None
            for local_i, d_val in enumerate(d_slice):
                if d_val != prev_d:
                    days_meta.append([local_i, d_val.strftime("%m/%d")])
                    prev_d = d_val
            wa_str = d_slice[0].strftime("%Y-%m-%d")
            wb_str = d_slice[-1].strftime("%Y-%m-%d")
            windows.append({
                "wa": wa_str,
                "wb": wb_str,
                "a": wa_str,
                "b": wb_str,
                "days": days_meta,
                "c": c_slice,
                "partial": bool(end_i == len(prices_arr) and len(c_slice) < ws),
                "headPartial": False,
            })
        return windows, last_bar_str

    # Default: half_month mode (วันที่ 1–14 และ 15–สิ้นเดือน ตาม template.html 100%)
    grouped: Dict[Tuple[str, str], Tuple[List[date], List[float]]] = {}
    for d_val, p_val in zip(ny_dates, prices_arr):
        bounds = half_month_bounds(d_val)
        if bounds not in grouped:
            grouped[bounds] = ([], [])
        grouped[bounds][0].append(d_val)
        grouped[bounds][1].append(p_val)

    all_candidate_windows: List[Dict[str, Any]] = []
    for (wa_str, wb_str), (d_list, c_list) in grouped.items():
        if len(c_list) < 2:
            continue
        days_meta = []
        prev_d = None
        for local_i, d_val in enumerate(d_list):
            if d_val != prev_d:
                days_meta.append([local_i, d_val.strftime("%m/%d")])
                prev_d = d_val
        is_partial = bool(wb_str >= asof_str)
        win_obj = {
            "wa": wa_str,
            "wb": wb_str,
            "a": d_list[0].strftime("%Y-%m-%d"),
            "b": d_list[-1].strftime("%Y-%m-%d"),
            "days": days_meta,
            "c": c_list,
            "partial": is_partial,
            "headPartial": False,
        }
        all_candidate_windows.append(win_obj)
        if is_partial or len(c_list) >= min_bars_finished:
            windows.append(win_obj)

    if not windows and all_candidate_windows:
        windows = all_candidate_windows

    return windows, last_bar_str


@st.cache_data(ttl=900, show_spinner=False)
def fetch_explorer_ticker_windows(
    ticker: str,
    timeframe: str = "1d",
    lookback_days: int = 365,
    window_mode: str = "half_month",
    fixed_window_size: int = 30,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """ดึงข้อมูลราคาจาก yfinance แล้วตัดแบ่งเป็น Windows สำหรับ DNA Spectrum Explorer"""
    tf_meta = EXPLORER_TIMEFRAMES.get(timeframe, EXPLORER_TIMEFRAMES["1d"])
    max_days = int(tf_meta["max_days"])
    eff_days = min(max(10, int(lookback_days)), max_days)
    try:
        now_ny = pd.Timestamp.now(tz="America/New_York")
        end_date = now_ny.date() + timedelta(days=1)
        raw_start = now_ny.date() - timedelta(days=eff_days)
        start_date = align_to_half_month_boundary(raw_start) if window_mode == "half_month" and eff_days >= 35 else raw_start
        hist = yf.Ticker(ticker).history(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval=tf_meta["yf"],
        )
        if hist.empty or "Close" not in hist.columns:
            return [], None
        return build_explorer_windows_from_series(
            hist["Close"],
            window_mode=window_mode,
            fixed_window_size=fixed_window_size,
            asof_date=now_ny.date(),
            min_bars_finished=int(tf_meta["min_bars"]),
        )
    except Exception:
        return [], None


def render_ticker_buttons_html(tickers: List[str]) -> str:
    """สร้างปุ่มเลือก Ticker บนแถบ Header ของ DNA Spectrum Explorer ให้ตรงกับ template.html 100%"""
    buttons: List[str] = []
    for idx, sym in enumerate(tickers):
        if idx == 0:
            cls = (
                "mkt-btn px-3 sm:px-4 py-1.5 rounded-lg text-xs font-mono font-semibold "
                "bg-tv-card text-white shadow-sm border border-tv-borderLight transition whitespace-nowrap"
            )
        else:
            cls = (
                "mkt-btn px-3 sm:px-4 py-1.5 rounded-lg text-xs font-mono font-semibold "
                "text-tv-muted hover:text-white transition whitespace-nowrap"
            )
        buttons.append(f'<button onclick="setTicker(\'{sym}\')" id="tk-{sym}" class="{cls}">{sym}</button>')
    return "".join(buttons)


def build_dna_spectrum_explorer_html(
    data_map: Dict[str, List[Dict[str, Any]]],
    meta_dict: Dict[str, Any],
    max_k_cap: Optional[int] = None,
    window_mode: str = "half_month",
    fixed_window_size: int = 30,
    exact_template_json: bool = False,
) -> str:
    """ประกอบไฟล์ HTML ของ DNA Spectrum Explorer จาก template.html ให้เหมือนต้นฉบับ 100%"""
    tickers = list(data_map.keys())
    if not tickers:
        default_meta, default_data = get_default_explorer_payload()
        data_map = default_data
        meta_dict = default_meta
        tickers = list(data_map.keys())
        exact_template_json = True

    tf_code = str(meta_dict.get("tf", "D"))
    tf_label = str(meta_dict.get("tfLabel", "รายวัน"))
    tf_short_map = {"D": "1D", "M15": "15m", "M30": "30m", "H1": "1H", "H4": "4H"}
    tf_short = tf_short_map.get(tf_code, tf_code)

    if window_mode == "fixed_bars":
        win_mode_title = f"windows {fixed_window_size} แท่ง"
        win_section_heading = (
            f"Windows ขนาดคงที่ ({fixed_window_size} แท่งต่อ window) · "
            f"ข้อมูล{tf_label} ({len(tickers)} Tickers: {' · '.join(tickers)})"
        )
    else:
        win_mode_title = "windows ครึ่งเดือน"
        if tf_code == "D":
            win_section_heading = (
                "Windows ครึ่งเดือน (วันที่ 1–14 · วันที่ 15–สิ้นเดือน) · "
                "ข้อมูลรายวันจาก Webull ย้อนหลังประมาณ 1 ปี (ราว 9–13 แท่งต่อ window)"
            )
        else:
            win_section_heading = (
                f"Windows ครึ่งเดือน (วันที่ 1–14 · วันที่ 15–สิ้นเดือน) · "
                f"ข้อมูลแท่ง {tf_label} (ตัดแบ่งตามครึ่งเดือนปฏิทิน)"
            )

    max_k_json = "null" if (max_k_cap is None or int(max_k_cap) <= 0) else str(int(max_k_cap))
    if exact_template_json:
        meta_json = DNA_SPECTRUM_EXPLORER_DEFAULT_META_JSON
        data_json = DNA_SPECTRUM_EXPLORER_DEFAULT_DATA_JSON
    else:
        meta_json = json.dumps(meta_dict, ensure_ascii=False)
        data_json = json.dumps(data_map, ensure_ascii=False, separators=(",", ":"))

    html_out = DNA_SPECTRUM_EXPLORER_TEMPLATE_HTML
    html_out = html_out.replace("__TITLE_TICKERS__", " · ".join(tickers), 1)
    html_out = html_out.replace("__TF_SHORT__", tf_short, 1)
    html_out = html_out.replace("__WIN_MODE_TITLE__", win_mode_title, 1)
    html_out = html_out.replace("__TICKER_BUTTONS__", render_ticker_buttons_html(tickers), 1)
    html_out = html_out.replace("__WIN_SECTION_HEADING__", win_section_heading, 1)
    html_out = html_out.replace("__TF_LABEL__", tf_label, 1)
    html_out = html_out.replace("__MAX_K_CAP_JSON__", max_k_json, 1)
    html_out = html_out.replace("__META_JSON__", meta_json, 1)
    html_out = html_out.replace("__DATA_JSON__", data_json, 1)
    return html_out


def resolve_explorer_data_for_tickers(
    tickers: List[str],
    timeframe: str = "1d",
    lookback_days: int = 365,
    window_mode: str = "half_month",
    fixed_window_size: int = 30,
    prefer_template_default: bool = True,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any], bool, List[str]]:
    """เตรียม DATA และ META สำหรับรายการ Dynamic Tickers ที่ผู้ใช้เลือก
    - หากเลือกใช้ข้อมูล Template สำหรับ AAPL/RGTI (โหมด 1d + half_month) จะใช้ข้อมูลจาก template.html เป๊ะ 100%
    - สำหรับหุ้นตัวอื่น หรือเมื่อกดดึงข้อมูลสด จะดึงจาก yfinance (และหากออฟไลน์จะ fallback จากผลจำลองใน session_state)"""
    default_meta, default_data = get_default_explorer_payload()
    tf_meta = EXPLORER_TIMEFRAMES.get(timeframe, EXPLORER_TIMEFRAMES["1d"])

    if (
        prefer_template_default
        and timeframe == "1d"
        and window_mode == "half_month"
        and tickers == list(default_data.keys())
    ):
        return default_data, default_meta, True, []

    data_map: Dict[str, List[Dict[str, Any]]] = {}
    missing_tickers: List[str] = []
    last_bars: List[str] = []

    for sym in tickers:
        if (
            prefer_template_default
            and timeframe == "1d"
            and window_mode == "half_month"
            and sym in default_data
        ):
            data_map[sym] = default_data[sym]
            if default_data[sym]:
                last_bars.append(default_data[sym][-1]["b"])
            continue

        wins, last_bar = fetch_explorer_ticker_windows(
            sym,
            timeframe=timeframe,
            lookback_days=lookback_days,
            window_mode=window_mode,
            fixed_window_size=fixed_window_size,
        )
        if not wins:
            for res_key in ("batch_results", "spectrum_results"):
                cached_res = st.session_state.get(res_key, {}).get(sym)
                if cached_res and "sim_results" in cached_res:
                    first_df = next((df for df in cached_res["sim_results"].values() if not df.empty and "price" in df.columns), None)
                    if first_df is not None:
                        wins, last_bar = build_explorer_windows_from_series(
                            first_df["price"],
                            window_mode=window_mode,
                            fixed_window_size=fixed_window_size,
                            min_bars_finished=2,
                        )
                        if wins:
                            break
        if not wins and sym in default_data:
            wins = default_data[sym]
            last_bar = wins[-1]["b"] if wins else None

        if wins:
            data_map[sym] = wins
            if last_bar:
                last_bars.append(last_bar)
        else:
            missing_tickers.append(sym)

    asof_str = datetime.now().strftime("%Y-%m-%d")
    if prefer_template_default and set(data_map.keys()).issubset(set(default_data.keys())):
        asof_str = str(default_meta.get("asof", asof_str))

    meta_dict = {
        "tf": tf_meta["tf"],
        "tfLabel": tf_meta["tfLabel"],
        "asof": asof_str,
        "lastBar": max(last_bars) if last_bars else default_meta.get("lastBar", asof_str),
        "pending": [],
    }
    return data_map, meta_dict, False, missing_tickers


def render_dna_spectrum_explorer_tab():
    """แท็บที่ 6: DNA Spectrum Explorer (เหมือน template.html 100% พร้อมรองรับ Dynamic Tickers)"""
    default_meta, default_data = get_default_explorer_payload()
    config = load_config()
    base_assets = config.get("assets", ["RGTI", "QBTS", "USAR"])

    all_known_tickers = list(dict.fromkeys(
        list(default_data.keys())
        + list(base_assets)
        + st.session_state.get("custom_tickers", [])
        + st.session_state.get("selected_tickers", [])
        + st.session_state.get("explorer_tickers", [])
    ))

    with st.expander("🎛️ **ตั้งค่า Dynamic Tickers & พารามิเตอร์ของ DNA Spectrum Explorer**", expanded=True):
        r1c1, r1c2, r1c3 = st.columns([3, 2, 1.5])
        with r1c2:
            custom_input = st.text_input(
                "➕ พิมพ์เพิ่ม Tickers แบบ Dynamic (คั่นด้วยเว้นวรรคหรือลูกน้ำ)",
                value="",
                placeholder="เช่น NVDA TSLA QBTS USAR IONQ",
                key="explorer_dynamic_input",
                help="พิมพ์ชื่อหุ้นใดก็ได้ (คั่นด้วยเว้นวรรคหรือ ,) แล้วกด Enter เพื่อเพิ่มเข้าสู่ DNA Spectrum Explorer ทันที",
            )
            if custom_input.strip():
                new_syms = parse_dynamic_tickers(custom_input)
                added = []
                for s in new_syms:
                    if s not in all_known_tickers:
                        all_known_tickers.append(s)
                    if s not in st.session_state.custom_tickers:
                        st.session_state.custom_tickers.append(s)
                    if s not in st.session_state.explorer_tickers:
                        st.session_state.explorer_tickers.append(s)
                        added.append(s)
                if added:
                    st.toast(f"เพิ่ม Dynamic Tickers: {', '.join(added)}")

        with r1c3:
            st.write("")
            btn_col1, btn_col2 = st.columns(2)
            if btn_col1.button("📌 ใช้จาก Settings", use_container_width=True, help="ดึงรายชื่อ Tickers ที่เลือกไว้ในแท็บ Settings มาแสดง"):
                if st.session_state.get("selected_tickers"):
                    st.session_state.explorer_tickers = list(st.session_state.selected_tickers)
                    st.rerun()
            if btn_col2.button("📄 ค่าตั้งต้น", use_container_width=True, help="รีเซ็ตกลับเป็น AAPL · RGTI ตาม template.html ต้นฉบับ"):
                st.session_state.explorer_tickers = list(default_data.keys())
                st.rerun()

        with r1c1:
            valid_current = [t for t in st.session_state.explorer_tickers if t in all_known_tickers]
            if not valid_current:
                valid_current = list(default_data.keys())
            selected_explorer_tickers = st.multiselect(
                "เลือก Tickers ที่ต้องการวิเคราะห์ (Dynamic Tickers · สลับดูแต่ละตัวได้บนแถบปุ่มใน Dashboard)",
                options=all_known_tickers,
                default=valid_current,
                key="explorer_tickers_multiselect",
            )
            st.session_state.explorer_tickers = selected_explorer_tickers

        r2c1, r2c2, r2c3, r2c4 = st.columns([1.4, 1.8, 1.2, 1.6])
        with r2c1:
            tf_choice = st.selectbox(
                "Timeframe",
                options=list(EXPLORER_TIMEFRAMES.keys()),
                index=0,
                format_func=lambda k: f"{k} ({EXPLORER_TIMEFRAMES[k]['tfLabel']})",
                key="explorer_tf",
            )
        with r2c2:
            win_mode = st.selectbox(
                "การแบ่ง Window",
                options=["half_month", "fixed_bars"],
                index=0,
                format_func=lambda m: (
                    "ครึ่งเดือน (วันที่ 1–14 · 15–สิ้นเดือน ตาม Template 100%)"
                    if m == "half_month"
                    else f"ตามจำนวนแท่งคงที่ ({st.session_state.window_size} แท่ง/Window)"
                ),
                key="explorer_win_mode",
            )
        with r2c3:
            max_days_allowed = int(EXPLORER_TIMEFRAMES[tf_choice]["max_days"])
            default_days_val = int(EXPLORER_TIMEFRAMES[tf_choice]["default_days"])
            lookback_days = st.number_input(
                "ย้อนหลัง (วัน)",
                min_value=15,
                max_value=max_days_allowed,
                value=min(default_days_val, max_days_allowed),
                step=15,
                key="explorer_lookback_days",
                help=f"จำนวนวันย้อนหลังสำหรับดึงข้อมูลจาก yfinance (สูงสุด {max_days_allowed} วันสำหรับ {tf_choice})",
            )
        with r2c4:
            data_source_mode = st.selectbox(
                "แหล่งข้อมูล AAPL / RGTI",
                options=["template_first", "live_yfinance"],
                index=0,
                format_func=lambda s: (
                    "Template ต้นฉบับ 100% + ดึงสดหุ้นอื่น"
                    if s == "template_first"
                    else "ดึงข้อมูลสดจาก yfinance ทุกตัว"
                ),
                key="explorer_source_mode",
            )

    active_tickers = list(st.session_state.explorer_tickers)
    if not active_tickers:
        st.warning("กรุณาเลือก Ticker อย่างน้อย 1 ตัว (ระบบแสดงผล AAPL · RGTI เป็นค่าเริ่มต้น)")
        active_tickers = list(default_data.keys())

    with st.spinner(f"กำลังเตรียมข้อมูล DNA Spectrum Explorer สำหรับ {', '.join(active_tickers)}..."):
        data_map, meta_dict, is_exact_template, missing = resolve_explorer_data_for_tickers(
            tickers=active_tickers,
            timeframe=tf_choice,
            lookback_days=int(lookback_days),
            window_mode=win_mode,
            fixed_window_size=int(st.session_state.window_size),
            prefer_template_default=(data_source_mode == "template_first"),
        )

    if missing:
        st.warning(f"⚠️ ไม่พบข้อมูลราคาสำหรับ: {', '.join(missing)} (ตรวจสอบชื่อ Ticker หรือการเชื่อมต่ออินเทอร์เน็ต)")

    if not data_map:
        st.info("ใช้ข้อมูลตั้งต้นจาก Template (AAPL · RGTI)")
        data_map, meta_dict = default_data, default_meta
        is_exact_template = True

    rendered_html = build_dna_spectrum_explorer_html(
        data_map=data_map,
        meta_dict=meta_dict,
        max_k_cap=None,
        window_mode=win_mode,
        fixed_window_size=int(st.session_state.window_size),
        exact_template_json=is_exact_template,
    )

    dl_col1, dl_col2 = st.columns([4, 1])
    with dl_col1:
        loaded_syms = list(data_map.keys())
        win_counts = ", ".join(f"{s} ({len(data_map[s])}W)" for s in loaded_syms)
        st.caption(f"✅ พร้อมแสดงผล: **{win_counts}** · คลิกปุ่มชื่อหุ้นที่มุมขวาบนของ Dashboard เพื่อสลับ Ticker ได้ทันที")
    with dl_col2:
        file_tag_syms = "_".join(loaded_syms[:5])
        st.download_button(
            label="💾 ดาวน์โหลด HTML",
            data=rendered_html,
            file_name=f"dna_spectrum_{file_tag_syms}_{tf_choice}.html",
            mime="text/html",
            use_container_width=True,
            key="btn_download_explorer_html",
        )

    components.html(rendered_html, height=1950, scrolling=True)


# ==============================================================================
# 6. Main Application
# ==============================================================================
def main():
    config = load_config()
    initialize_session_state(config)

    tab_list = [
        "⚙️ Settings",
        "🧬 Simulation Results",
        "🌈 DNA Spectrum",
        "🔬 Spectrum Classifier",
        "🔍 Tracer",
        "🧪 DNA Spectrum Explorer",
    ]
    tabs = st.tabs(tab_list)

    with tabs[0]:
        render_settings_tab()
    with tabs[1]:
        render_simulation_tabs()
    with tabs[2]:
        render_spectrum_tab()
    with tabs[3]:
        render_classifier_tab()
    with tabs[4]:
        render_tracer_tab()
    with tabs[5]:
        render_dna_spectrum_explorer_tab()

if __name__ == "__main__":
    main()
