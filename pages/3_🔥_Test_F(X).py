"""
fx_core.py — โมดูลแกนกลางสำหรับโปรเจกต์ลงทุน (20 ไฟล์)

เป้าหมาย:
- รวม logic สูตร log: f = b*ln(tn/t0) และรูปแบบเดิม (offset + coef*ln) ไว้ที่จุดเดียว
- ยกเครื่องฟังก์ชันจำลอง/อ้างอิงที่ถูกใช้ซ้ำในหลายไฟล์ ให้คงผลลัพธ์เดิม (backward compatible)
- ลดความซ้ำซ้อน, เพิ่มเสถียรภาพเชิงตัวเลข และให้ทางเลือกสลับโหมดได้ผ่าน config

การใช้งานย่อ:
from fx_core import log_form, cash_grid, simulate_rebalance, aggregate_net

f = log_form(tn=live, t0=ref, coef=fix_c, offset=b_offset, mode="add")  # โหมดเดิม (ค่าเริ่มต้น)
f_new = log_form(tn=live, t0=ref, coef=b, offset=0.0, mode="mul")        # โหมดใหม่ตาม goal: f=b*ln(tn/t0)

หมายเหตุ: โค้ดนี้ไม่ผูกกับ UI/ThingSpeak และไม่มี side-effect
"""
from __future__ import annotations
from typing import Iterable, Literal, Tuple, Union, Dict
import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, pd.Series, Iterable[float], float]

# -----------------------------------------------------------------------------
# 1) Core log form
# -----------------------------------------------------------------------------
def _safe_ratio(tn: ArrayLike, t0: ArrayLike, eps: float = 1e-12) -> np.ndarray:
    """ป้องกันหารศูนย์/ลอคค่าติดลบ: บีบค่าต่ำสุดเป็น eps"""
    tn_arr = np.asarray(tn, dtype=np.float64)
    t0_arr = np.asarray(t0, dtype=np.float64)
    tn_arr = np.maximum(tn_arr, eps)
    t0_arr = np.maximum(t0_arr, eps)
    return tn_arr / t0_arr

def log_form(
    tn: ArrayLike,
    t0: ArrayLike,
    *,
    coef: float,
    offset: float = 0.0,
    mode: Literal["add", "mul"] = "add",
    clip: Tuple[float, float] | None = None,
) -> np.ndarray:
    """คำนวณ f ตามสูตร log แบบยืดหยุ่น

    โหมด "add" (ค่าเริ่มต้น, backward-compatible):
        f = offset + coef * ln(tn / t0)
        - ตรงกับสูตรที่ใช้ใน Add_CF / Benchmark เดิม (fix_c กับ b_offset)

    โหมด "mul" (ตาม goal ใหม่):
        f = coef * ln(tn / t0)
        - ใช้เมื่ออยาก pure ตามแกน f = b*ln(tn/t0)

    Parameters
    ----------
    tn : ราคาปัจจุบัน (live)
    t0 : ราคาอ้างอิง (reference)
    coef : ค่าสัมประสิทธิ์ของพจน์ log (เช่น fix_c หรือ b)
    offset : ค่าคงที่ (b_offset) — ใช้เฉพาะ mode="add"
    mode : "add" หรือ "mul"
    clip : (min, max) เพื่อตัดค่า f ให้อยู่ในกรอบ หากต้องการ
    """
    ratio = _safe_ratio(tn, t0)
    f = coef * np.log(ratio)
    if mode == "add":
        f = offset + f
    if clip is not None:
        f = np.clip(f, clip[0], clip[1])
    return f

# -----------------------------------------------------------------------------
# 2) Cash grid อ้างอิง (ยุบรวม logic จาก CF_Graph / Un_15 ฯลฯ)
# -----------------------------------------------------------------------------
def cash_grid(
    *,
    entry: float,
    step: float,
    fixed_asset_value: float,
    cash_start: float,
    max_mult: float = 3.0,
) -> pd.DataFrame:
    """สร้างกริดราคา + cash balance อ้างอิง (บน-ล่าง) แบบเดิม แต่ robust ขึ้น

    Returns: DataFrame[['Asset_Price','Fixed_Asset_Value','Amount_Asset','Cash_Balan','net_pv']]
    """
    if not (step > 0 and entry > 0 and fixed_asset_value > 0):
        return pd.DataFrame(columns=["Asset_Price","Fixed_Asset_Value","Amount_Asset","Cash_Balan","net_pv"])    

    samples = np.arange(0, np.around(entry, 2) * max_mult + step, step)
    df = pd.DataFrame({"Asset_Price": np.around(samples, 2)})
    df = df[df.Asset_Price > 0].copy()
    df["Fixed_Asset_Value"] = float(fixed_asset_value)
    df["Amount_Asset"] = df["Fixed_Asset_Value"] / df["Asset_Price"]

    # ส่วนบน (ราคา >= entry)
    top = df[df.Asset_Price >= np.around(entry, 2)].copy()
    if not top.empty:
        delta = (top["Amount_Asset"].shift(1) - top["Amount_Asset"]) * top["Asset_Price"]
        delta = delta.fillna(0.0).to_numpy()
        cash = np.zeros_like(delta)
        acc = cash_start
        for i, v in enumerate(delta):
            acc += float(v)
            cash[i] = acc
        top["Cash_Balan"] = cash
        top = top.sort_values("Amount_Asset").iloc[:-1] if len(top) > 1 else top
    else:
        top = pd.DataFrame(columns=["Asset_Price","Fixed_Asset_Value","Amount_Asset","Cash_Balan"])

    # ส่วนล่าง (ราคา <= entry)
    down = df[df.Asset_Price <= np.around(entry, 2)].copy()
    if not down.empty:
        down = down.sort_values("Asset_Price", ascending=False)
        delta = (down["Amount_Asset"].shift(-1) - down["Amount_Asset"]) * down["Asset_Price"]
        delta = delta.fillna(0.0).to_numpy()
        cash = np.zeros_like(delta)
        acc = cash_start
        for i, v in enumerate(delta):
            acc += float(v)
            cash[i] = acc
        down["Cash_Balan"] = cash
    else:
        down = pd.DataFrame(columns=["Asset_Price","Fixed_Asset_Value","Amount_Asset","Cash_Balan"])

    out = pd.concat([top, down], axis=0, ignore_index=True)
    out["net_pv"] = out["Fixed_Asset_Value"] + out["Cash_Balan"]
    return out[["Asset_Price","Fixed_Asset_Value","Amount_Asset","Cash_Balan","net_pv"]]

# -----------------------------------------------------------------------------
# 3) ตัวจำลอง Rebalance แบบ stateful (โครงเดียวกับของเดิม แต่ไม่พึ่ง Numba)
# -----------------------------------------------------------------------------
def simulate_rebalance(
    prices: ArrayLike,
    actions: ArrayLike,
    *,
    fix: float = 1500.0,
) -> pd.DataFrame:
    """จำลองกระบวนการ Rebalance/ถือ (actions: 1=rebalance, 0=hold)

    คืนค่า DataFrame คอลัมน์: price, action, buffer, cash, asset_value, sumusd, refer, net
    - refer = เส้นอ้างอิงแบบ log เดิม: -fix*ln(p0 / p)
    - net   = sumusd - refer - sumusd[0]
    """
    price = np.asarray(prices, dtype=np.float64)
    act = np.asarray(actions, dtype=np.int32)
    n = min(len(price), len(act))
    if n == 0:
        return pd.DataFrame()

    price = price[:n]
    act = act[:n]
    if n > 0:
        act[0] = 1  # บังคับวันแรกต้อง rebalance (เหมือนเดิม)

    amount = np.empty(n, dtype=np.float64)
    buffer = np.zeros(n, dtype=np.float64)
    cash = np.empty(n, dtype=np.float64)
    asset = np.empty(n, dtype=np.float64)
    sumusd = np.empty(n, dtype=np.float64)

    p0 = float(price[0])
    amount[0] = fix / p0
    cash[0] = fix
    asset[0] = amount[0] * p0
    sumusd[0] = cash[0] + asset[0]

    refer = -fix * np.log(_safe_ratio(p0, price))  # เส้นอ้างอิงเดิม

    for i in range(1, n):
        p = float(price[i])
        if act[i] == 0:
            amount[i] = amount[i - 1]
            buffer[i] = 0.0
        else:
            amount[i] = fix / p
            buffer[i] = amount[i - 1] * p - fix
        cash[i] = cash[i - 1] + buffer[i]
        asset[i] = amount[i] * p
        sumusd[i] = cash[i] + asset[i]

    initial = float(sumusd[0])
    df = pd.DataFrame(
        {
            "price": price,
            "action": act,
            "buffer": np.round(buffer, 2),
            "cash": np.round(cash, 2),
            "asset_value": np.round(asset, 2),
            "sumusd": np.round(sumusd, 2),
            "refer": np.round(refer + initial, 2),
            "net": np.round(sumusd - refer - initial, 2),
        }
    )
    return df

# -----------------------------------------------------------------------------
# 4) ตัวช่วยรวมผลหลายสินทรัพย์
# -----------------------------------------------------------------------------
def aggregate_net(dfs: Dict[str, pd.DataFrame], col: str = "net") -> pd.DataFrame:
    """รวมผลคอลัมน์ "net" ของหลายสินทรัพย์ให้เป็นตารางเดียวแบบ align index
    ใช้กับ Benchmark/Un_15 ได้ทันที
    """
    if not dfs:
        return pd.DataFrame()
    # เลือก index ยาวสุดเป็นแกน
    base_index = max((d.index for d in dfs.values() if not d.empty), key=lambda x: len(x), default=None)
    if base_index is None:
        return pd.DataFrame()
    out = pd.DataFrame(index=base_index)
    for name, d in dfs.items():
        if d.empty or col not in d.columns:
            continue
        out[name] = d[col].reindex(base_index).ffill()
    return out

# -----------------------------------------------------------------------------
# 5) ยูทิลิตี้เล็กน้อย
# -----------------------------------------------------------------------------
def ensure_series(x: ArrayLike, index=None) -> pd.Series:
    arr = np.asarray(x)
    if index is None:
        return pd.Series(arr)
    return pd.Series(arr, index=index)

__all__ = [
    "log_form",
    "cash_grid",
    "simulate_rebalance",
    "aggregate_net",
    "ensure_series",
]
