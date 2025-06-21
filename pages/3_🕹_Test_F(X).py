ด้านล่างคือ “โค้ดเต็ม” (single-file) ที่ประกอบด้วยส่วนสำคัญทั้งหมดสำหรับกลยุทธ์ใหม่ ​PCG64 Seed Search Optimizer — เขียนให้ copy ได้ครั้งเดียวโดยไม่ขาดตอน พร้อมคอมเมนต์อธิบายครบถ้วน

````python
"""
pcg64_seed_optimizer.py
----------------------------------------------------------
โปรเจ็กต์สาธิตการสร้าง Action-Sequence ด้วยการค้นหา
Random-Seed (PCG64) แบบรวดเร็ว/เสถียร บนโครงสร้าง
Sliding-Window   ใช้ได้ทั้งรันเดี่ยว ๆ หรือ import เป็นโมดูล
----------------------------------------------------------
ความต้องการ (pip install ...)
    numpy pandas numba streamlit (optional – แสดง progress bar)
----------------------------------------------------------
วิธีใช้ย่อ:
    from pcg64_seed_optimizer import (
        Strategy, generate_actions_sliding_window_rng
    )

    # สมมติ ticker_df คือ DataFrame มีคอลัมน์ 'Close' แล้ว
    actions, detail_df = generate_actions_sliding_window_rng(
        ticker_df,
        window_size = 128,
        iterations  = 50_000,
        base_seed   = 2025,
    )
----------------------------------------------------------
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Tuple, List

import numpy as np
import pandas as pd

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False

from numba import njit, prange


# =========================================================
# 1) ENUM กลยุทธ์ทั้งหมด
# =========================================================
class Strategy:
    REBALANCE_DAILY      = "Rebalance Daily"
    PERFECT_FORESIGHT    = "Perfect Foresight (Max)"
    SLIDING_WINDOW       = "Best Seed Sliding Window"
    RNG_PCG64_OPTIMIZED  = "PCG64 Seed Search"          # <-- NEW


# =========================================================
# 2) NUMBA: ฟังก์ชันประเมินผลลัพธ์เร็ว ๆ
# =========================================================
@njit(cache=True, fastmath=True)
def _score_actions_numba(actions: np.ndarray,
                         prices: np.ndarray,
                         fix: int = 1500) -> float:
    """
    คำนวณ Net-Profit เร็ว ๆ (return เป็น float ตัวเดียว)
    actions[i] == 1 ⇒ rebalance ณ วันนั้น (เปลี่ยนจากหุ้นเป็นเงินสดแล้วซื้อใหม่ทันที)
    """
    n = prices.size
    if n == 0:
        return -1.0e18

    cash   = fix         # เงินสด ณ ต้นงวด
    amount = fix / prices[0]  # หน่วยหุ้นที่ถืออยู่

    for i in range(1, n):
        if actions[i]:
            cash   = fix
            amount = fix / prices[i]

    final_val = cash + amount * prices[-1]      # มูลค่าพอร์ตสุดท้าย
    return final_val - fix                      # Net-Profit (กำไรสุทธิ)


# =========================================================
# 3) อัลกอริทึมค้นหา Seed ที่ดีที่สุด
# =========================================================
def find_best_seed_pcg64(
    prices_window : np.ndarray,
    base_seed     : int,
    iterations    : int = 50_000,
    top_k         : int = 32,
    neighbourhood : int = 1000
) -> Tuple[int, float, np.ndarray]:
    """
    กลไก 2-Phase
        Phase-1  Global search  : ยิงสุ่ม seeds (PCG64) จำนวน iterations
        Phase-2  Local refine   : ค้นรอบ ๆ seed เด่น ๆ เพื่อ fine-tune
    Return  (best_seed, best_net, best_actions)
    ------------------------------------------------------
    Tips: เพิ่ม iterations หรือ neighbourhood เพื่อความแม่น (แต่ช้าลง)
    """
    win_len = prices_window.size
    if win_len < 2:
        return (1, 0.0, np.ones(win_len, dtype=np.int32))

    # แปลงเป็น dtype/structure ที่ numba ถนัด
    prices_np = prices_window.astype(np.float64)

    # ---------- Phase-1 : global random search -----------------
    rng_global = np.random.Generator(np.random.PCG64(base_seed))

    # best_pool เก็บ top-k (net, seed)  เรียงสุ่มน้อย→มาก
    best_pool: List[Tuple[float, int]] = []
    best_net, best_seed = -np.inf, -1
    best_actions        = None

    for _ in range(iterations):
        seed = int(rng_global.integers(0, 2**63 - 1, dtype=np.int64))
        rng  = np.random.Generator(np.random.PCG64(seed))

        actions = rng.integers(0, 2, size=win_len, dtype=np.int8)
        actions[0] = 1            # rebalance วันแรกเสมอ

        net = _score_actions_numba(actions, prices_np, 1500)

        # จัดอันดับ
        if len(best_pool) < top_k:
            best_pool.append((net, seed))
            best_pool.sort(key=lambda x: x[0])
        elif net > best_pool[0][0]:
            best_pool[0] = (net, seed)
            best_pool.sort(key=lambda x: x[0])

        if net > best_net:
            best_net, best_seed = net, seed
            best_actions = actions.copy()

    # ---------- Phase-2 : local refinement ---------------------
    rng_local = np.random.Generator(np.random.PCG64(best_seed ^ 0xDEADBEEF))

    # ใช้ครึ่งหลังของ best_pool (ตัวที่คะแนนสูง) เป็นศูนย์กลาง
    for net0, seed0 in best_pool[-top_k // 2:]:
        for _ in range(neighbourhood):
            neighbour_seed = (seed0 +
                              int(rng_local.integers(-1_000_000, 1_000_000))
                              ) & ((1 << 63) - 1)
            rngN = np.random.Generator(np.random.PCG64(neighbour_seed))
            actionsN = rngN.integers(0, 2, size=win_len, dtype=np.int8)
            actionsN[0] = 1

            netN = _score_actions_numba(actionsN, prices_np, 1500)
            if netN > best_net:
                best_net, best_seed, best_actions = netN, neighbour_seed, actionsN.copy()

    return int(best_seed), float(round(best_net, 2)), best_actions.astype(np.int32)


# =========================================================
# 4) Sliding-Window Wrapper (ใช้งานจริง)
# =========================================================
def generate_actions_sliding_window_rng(
    ticker_data : pd.DataFrame,
    window_size : int = 128,
    iterations  : int = 50_000,
    base_seed   : int = 2025
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    เก็บผลลัพธ์ Action-Sequence ครบทุก window รวมเป็น array เดียว
    พร้อมตารางรายละเอียดของแต่ละ window (DataFrame)
    ----------------------------------------------------------
    Parameters
    ----------
    ticker_data : DataFrame | ต้องมีคอลัมน์ 'Close', index เป็น Datetime
    window_size : int       | ความยาวแต่ละช่วง (days)
    iterations  : int       | จำนวน sample seeds ต่อ window
    base_seed   : int       | ซีดหลัก (re-producible)
    """
    prices = ticker_data["Close"].to_numpy(dtype=np.float64)
    n      = prices.size

    final_actions: List[int] = []
    window_rows  : List[dict] = []
    n_win = (n + window_size - 1) // window_size

    progress_bar = None
    if _HAS_STREAMLIT:
        progress_bar = st.progress(0.0, text=f"Searching Seeds ({Strategy.RNG_PCG64_OPTIMIZED})…")
    else:
        print(f"\n>> Searching Seeds ({Strategy.RNG_PCG64_OPTIMIZED})…")

    for i, start in enumerate(range(0, n, window_size)):
        end = min(start + window_size, n)
        pw  = prices[start:end]

        win_seed = base_seed + i * 12_345   # ทำให้ reproducible แต่ต่างกันได้

        best_seed, best_net, best_act = find_best_seed_pcg64(
            pw,
            base_seed   = win_seed,
            iterations  = iterations,
        )

        final_actions.extend(best_act.tolist())

        window_rows.append({
            "window_number" : i + 1,
            "timeline"      : f"{ticker_data.index[start].date()} → {ticker_data.index[end-1].date()}",
            "best_seed"     : best_seed,
            "max_net"       : best_net,
            "action_count"  : int(best_act.sum()),
            "window_size"   : len(best_act),
        })

        pct = (i + 1) / n_win
        if progress_bar is not None:
            progress_bar.progress(pct, text=f"Window {i+1}/{n_win}")
        else:
            sys.stdout.write(f"\r   processed {i+1}/{n_win} windows ...")
            sys.stdout.flush()

    if progress_bar is None:
        print("\n>> done.")

    actions_arr = np.asarray(final_actions, dtype=np.int32)
    detail_df   = pd.DataFrame(window_rows)

    return actions_arr, detail_df


# =========================================================
# 5) main demo (รันไฟล์ตรง ๆ)
# =========================================================
def _demo():
    """
    สาธิตแบบ command-line
    อ่านไฟล์ CSV (OHLC) แล้วทดลองรันกลยุทธ์ใหม่
    """
    import argparse

    p = argparse.ArgumentParser(description="PCG64 Seed-Search Optimizer demo")
    p.add_argument("--csv", required=True,
                   help="path to csv having at least Date,Close columns")
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--iter",   type=int, default=50_000)
    p.add_argument("--seed",   type=int, default=2025)
    args = p.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["Date"], index_col="Date")
    print(f"Loaded {len(df):,} rows.")

    acts, detail = generate_actions_sliding_window_rng(
        df,
        window_size = args.window,
        iterations  = args.iter,
        base_seed   = args.seed,
    )

    print("\n=== Summary ===")
    print(detail.to_string(index=False))
    print(f"\nTotal actions = {acts.sum():,} within {len(acts)} days")

    # Example: คำนวณ Net-Profit รวมเทียบ buy&hold
    cash0   = 1500
    prices  = df["Close"].to_numpy(dtype=np.float64)
    net_new = _score_actions_numba(acts.astype(np.int8), prices, cash0)
    net_bh  = prices[-1] / prices[0] * cash0 - cash0
    print(f"\nSeed-Search Net: {net_new:,.2f}  | Buy&Hold: {net_bh:,.2f}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=NumbaWarning)  # mute Numba cache msg
    try:
        from numba.core.errors import NumbaWarning
    except Exception:
        pass
    _demo()
````



📝 จุดสังเกต

1. ไฟล์เดียวจบ: import ได้หรือรันเดโม (`python pcg64_seed_optimizer.py --csv price.csv`) ก็ได้
2. ตัด dependency ให้เหลือแค่ `numpy, pandas, numba`, ส่วน `streamlit` แค่ optional
3. ค่า default:
   • `window_size = 128`,  
   • `iterations  = 50_000`,  
   • `base_seed   = 2025`  
   สามารถปรับตามทรัพยากร/ความเร็ว  
4. `_score_actions_numba` เร็วมากเพราะ numba + fastmath
5. deterministic: ใช้ seed เดิม → ได้ผลเดิมทุกครั้ง

คัดลอกบล็อกโค้ดข้างบนครั้งเดียวก็พร้อมใช้งานครับ 🚀
