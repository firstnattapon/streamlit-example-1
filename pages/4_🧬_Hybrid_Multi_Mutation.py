import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
import json  # ! NEW: For Export Functionality

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
    
    # ! GOAL: Default Select ALL Tickers
    if 'selected_tickers' not in st.session_state: 
        st.session_state.selected_tickers = asset_list
    
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

def generate_actions_spectrum(
    ticker_data: pd.DataFrame, window_size: int, spectrum_codes: List[Tuple[str, str]],
    progress_bar=None, ticker_name: str = "", timeframe: str = "1d"
) -> Tuple[np.ndarray, pd.DataFrame]:
    """ทุก Window: เลือก DNA ที่ดีที่สุด 1 ใน 10 แล้วจบ
    ไม่กลายพันธุ์ต่อ เพราะ 10 DNA ผ่านการปรับปรุง (mutation seeds) มาแล้ว"""
    prices = ticker_data['Close'].to_numpy()
    n = len(prices)
    final_actions = np.array([], dtype=np.int32)
    window_details_list = []
    num_windows = (n + window_size - 1) // window_size

    for i, start_index in enumerate(range(0, n, window_size)):
        end_index = min(start_index + window_size, n)
        prices_window = prices[start_index:end_index]
        if len(prices_window) < 2: continue

        if progress_bar:
            progress_bar.progress((i + 1) / num_windows, text=f"{ticker_name}: Window {i+1}/{num_windows}...")

        rank, tracer, net, actions = find_best_spectrum_dna_for_window(prices_window, spectrum_codes)
        final_actions = np.concatenate((final_actions, actions.astype(np.int32)))

        start_date = ticker_data.index[start_index]; end_date = ticker_data.index[end_index-1]
        window_details_list.append({
            'window': i + 1,
            'timeframe': timeframe,
            'timeline': f"{format_timeline_value(start_date, timeframe)} to {format_timeline_value(end_date, timeframe)}",
            'start_timestamp': to_unix_timestamp(start_date),
            'end_timestamp': to_unix_timestamp(end_date),
            'spectrum_rank': rank,
            'spectrum_label': spectrum_codes[rank - 1][0],
            'dna_seed': tracer.dna_seed,
            'mutation_seeds': str(list(tracer.mutation_seeds)) if tracer.mutation_seeds else "None",
            'mutation_rate': tracer.mutation_rate,
            'net': round(net, 2),
        })

    return final_actions, pd.DataFrame(window_details_list)

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
    start_timestamp: int
) -> Tuple[str, int, int, int]:
    """
    Generate Encoded String logic extracted for reuse.
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
        timeframe_minutes=get_timeframe_minutes(timeframe),
        start_timestamp=final_start_timestamp
    )

    return encoded_string, use_win_num, int(final_act_len), final_start_timestamp


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

    # Default Select ALL Tickers
    st.session_state.selected_tickers = st.multiselect(
        "เลือก Tickers ที่ต้องการทดสอบ (เลือกได้หลายตัว)", 
        options=asset_list, 
        default=st.session_state.selected_tickers
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

def execute_batch_processing(spectrum_codes: Optional[List[Tuple[str, str]]] = None):
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
            # DNA Spectrum: เลือก 1 ใน 10 ต่อ Window ไม่มีขั้นกลายพันธุ์
            final_actions, df_windows = generate_actions_spectrum(
                ticker_data, st.session_state.window_size, spectrum_codes,
                progress_bar=overall_progress, ticker_name=ticker, timeframe=selected_timeframe
            )
            sim_results = {Strategy.SPECTRUM_BEST: run_simulation(prices.tolist(), final_actions.tolist())}
        else:
            original_actions, final_actions, df_windows = generate_actions_hybrid_multi_mutation(
                ticker_data, st.session_state.window_size, st.session_state.num_seeds,
                st.session_state.max_workers, st.session_state.mutation_rate,
                st.session_state.num_mutations,
                progress_bar=overall_progress,
                ticker_name=ticker,
                timeframe=selected_timeframe
            )
            sim_results = {
                Strategy.HYBRID_MULTI_MUTATION: run_simulation(prices.tolist(), final_actions.tolist()),
                Strategy.ORIGINAL_DNA: run_simulation(prices.tolist(), original_actions.tolist()),
            }
        sim_results.update({
            Strategy.REBALANCE_DAILY: run_simulation(prices.tolist(), generate_actions_rebalance_daily(len(prices)).tolist()),
            Strategy.PERFECT_FORESIGHT: run_simulation(prices.tolist(), generate_actions_perfect_foresight(prices.tolist()).tolist())
        })
        for name, df in sim_results.items():
            if not df.empty: df.index = ticker_data.index[:len(df)]
            
        st.session_state[results_key][ticker] = {
            "sim_results": sim_results,
            "df_windows": df_windows,
            "data_len": len(ticker_data),
            "timeframe": selected_timeframe
        }
        overall_progress.progress((idx + 1) / total_tickers, text=f"Completed {ticker} ({idx+1}/{total_tickers})")

    overall_progress.empty()
    st.success(f"✅ Processed {len(st.session_state[results_key])} tickers successfully!")

def render_single_ticker_result(
    ticker: str, result_data: Dict[str, Any], key_prefix: str = "",
    hybrid: Tuple[str, str] = (Strategy.HYBRID_MULTI_MUTATION, "Hybrid Strategy"),
    original: Optional[Tuple[str, str]] = (Strategy.ORIGINAL_DNA, "Original DNA")
):
    # hybrid / original = (ชื่อ strategy ใน sim_results, ป้าย metric) · original=None -> ไม่แสดง
    sim_results = result_data["sim_results"]
    df_windows = result_data["df_windows"]
    data_len = result_data["data_len"]
    timeframe = result_data.get("timeframe", st.session_state.selected_timeframe)

    chart_results = {k: v for k, v in sim_results.items() if original is None or k != original[0]}
    display_comparison_charts(chart_results, chart_title=f'📊 {ticker} - Net Profit Comparison')

    if original is None:
        c1, c2, c4 = st.columns(3)
    else:
        c1, c2, c3, c4 = st.columns(4)
    def get_final_net(strategy_name):
        df = sim_results.get(strategy_name)
        return df['net'].iloc[-1] if df is not None and not df.empty else 0.0

    c1.metric("Perfect Foresight", f"${get_final_net(Strategy.PERFECT_FORESIGHT):,.0f}")
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
        if st.button(f"Encode ({ticker})", key=f"btn_enc_{key_prefix}{ticker}", use_container_width=True):
            try:
                # ! REFACTORED: Use helper function
                encoded, final_win, final_len, final_start_ts = generate_encoded_dna_for_ticker(
                    df_windows, data_len, target_win_num, global_act_len,
                    st.session_state.mutation_rate, window_size, timeframe, trace_start_timestamp
                )

                st.success(f"Encoded String ({ticker} Win {final_win}, TF {timeframe}, StartTS {final_start_ts}):")
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
                **(extra_metadata or {})
            },
            "tickers": {},
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
                ticker_meta = {
                    "window": final_win,
                    "action_length": final_len,
                    "timeframe": timeframe,
                    "timeframe_minutes": get_timeframe_minutes(timeframe),
                    "start_timestamp": final_start_ts,
                }
                # ผล Spectrum: rate จริงของ string และ rank ที่ชนะใน window นั้น
                row = data["df_windows"].iloc[final_win - 1]
                if 'spectrum_rank' in row.index:
                    ticker_meta["mutation_rate_percent"] = int(row['mutation_rate'])
                    ticker_meta["spectrum_rank"] = int(row['spectrum_rank'])
                    ticker_meta["spectrum_label"] = row['spectrum_label']
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
                render_single_ticker_result(ticker, st.session_state.batch_results[ticker])
    else:
        st.info("ยังไม่มีผลลัพธ์ กรุณากดปุ่ม Start")

def render_spectrum_tab():
    st.markdown("### 🌈 DNA Spectrum (10 DNA แทนการสุ่ม Seeds)")
    st.info(
        f"ทุก Window: ไม่สุ่ม {st.session_state.num_seeds:,} Seeds แต่**เลือก DNA ที่ net สูงสุด 1 ใน 10 ชุด** "
        "เรียงจากความถี่สั้น (เปิด gate ถี่) ไปความถี่ยาว (เปิด gate ห่าง) แล้วจบ · **ไม่กลายพันธุ์ต่อ** "
        "เพราะ 10 DNA ผ่านการปรับปรุง (mutation seeds) มาแล้ว · Mutation Rounds ใน Settings ใช้กับแท็บ Hybrid เท่านั้น"
    )

    preset_names = list(DNA_SPECTRUM_PRESETS.keys())
    st.session_state.spectrum_preset = st.selectbox(
        "ชุด DNA Spectrum",
        options=preset_names,
        index=preset_names.index(st.session_state.spectrum_preset) if st.session_state.spectrum_preset in preset_names else 0,
        help="Density = จำนวน gate ลดหลั่น (rate 10%, 5 mutations) · Periodic = 1 ทุก k แท่งเป๊ะ"
    )
    spectrum_codes = DNA_SPECTRUM_PRESETS[st.session_state.spectrum_preset]
    window_size = int(st.session_state.window_size)

    with st.expander(f"🧬 10 DNA ({SPECTRUM_LENGTH} บิต) · เรียงความถี่สั้น → ยาว", expanded=True):
        st.dataframe(build_spectrum_table(spectrum_codes, window_size), width="stretch", hide_index=True)
        st.caption(f"Window ยาว {window_size} แท่ง ใช้ {min(window_size, SPECTRUM_LENGTH)} บิตแรกของแต่ละ DNA (คอลัมน์ active_first_{window_size})")
    if window_size > SPECTRUM_LENGTH:
        st.warning(f"⚠️ Window {window_size} แท่ง ยาวกว่า {SPECTRUM_LENGTH}: บิตหลังตำแหน่ง {SPECTRUM_LENGTH} เป็นค่าสุ่มต่อจาก seed ไม่อยู่ในสเปกตรัม")

    c1, c2 = st.columns([3, 1])
    with c1:
        st.caption("เลือก Tickers / Timeframe / Window ในหน้า Settings แล้วกดปุ่มด้านขวา")
    with c2:
        if st.button("🌈 Run Spectrum (10 DNA)", type="primary", width="stretch", key="btn_run_spectrum"):
            execute_batch_processing(spectrum_codes=spectrum_codes)

    results = st.session_state.spectrum_results
    if not results:
        st.info("ยังไม่มีผลลัพธ์ กรุณากดปุ่ม Run Spectrum")
        return

    render_export_section(
        results, key="export_spectrum", file_name="encoded_strings_spectrum.json",
        extra_metadata={"source": "dna_spectrum", "spectrum_preset": st.session_state.spectrum_preset}
    )
    tickers = list(results.keys())
    st.write(f"✅ Results available for: {', '.join(tickers)}")
    tabs = st.tabs([f"🌈 {t}" for t in tickers])
    for tab, ticker in zip(tabs, tickers):
        with tab:
            df_windows = results[ticker]["df_windows"]
            if 'spectrum_rank' in df_windows.columns:
                wins = df_windows.groupby(['spectrum_rank', 'spectrum_label']).size().reset_index(name='windows_won')
                st.write("🏆 **DNA ที่ชนะในแต่ละ Window (นับตาม rank)**")
                st.dataframe(wins, width="stretch", hide_index=True)
            render_single_ticker_result(
                ticker, results[ticker], key_prefix="spectrum_",
                hybrid=(Strategy.SPECTRUM_BEST, "Spectrum (Best of 10)"),
                original=None
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
# 6. Main Application
# ==============================================================================
def main():
    config = load_config()
    initialize_session_state(config)

    tab_list = ["⚙️ Settings", "🧬 Simulation Results", "🌈 DNA Spectrum (10)", "🔬 Spectrum Classifier", "🔍 Tracer"]
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

if __name__ == "__main__":
    main()
