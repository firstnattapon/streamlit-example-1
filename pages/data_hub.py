# helpers/data_hub.py
# --------------------------------------------------------
# ขา Data-Layer ความเร็วสูง  (ThingSpeak + Yahoo + Cache)
# --------------------------------------------------------
import asyncio, datetime as dt, json
from typing import Dict, List, Tuple

import httpx                      # ต้อง  pip install httpx
import numpy as np
import pandas as pd
import streamlit as st
import thingspeak
import yfinance as yf


# ==============  MEM (singleton ultra-fast in-memory cache)  ==============
class _MemBox(dict):
    def ttl_get(self, key, ttl):
        v, ts = self.get(key, (None, None))
        if v is not None and (dt.datetime.now() - ts).total_seconds() < ttl:
            return v
        return None

    def ttl_set(self, key, obj):                  # บันทึกค่าพร้อมเวลา
        self[key] = (obj, dt.datetime.now())


MEM = _MemBox()                                   # export ไว้ให้หน้า UI flush ได้

# --------------------------------------------------------------------------
#                              YFINANCE  (Batch)
# --------------------------------------------------------------------------
async def _dl_yf_async(tickers: List[str], period="max") -> pd.DataFrame:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: yf.download(
            tickers=" ".join(tickers),
            period=period,
            group_by="column",
            threads=True,
            progress=False,
        ),
    )


@st.cache_data(ttl=300, max_entries=1, show_spinner=False)
def get_histories(tickers: Tuple[str]) -> Dict[str, pd.DataFrame]:
    """
    ดึง historical ราคาหุ้นทุกตัวแบบ batch แล้วแตกคืนเป็น Dict
    """
    raw = asyncio.run(_dl_yf_async(list(tickers), "max")).round(3)
    out: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = raw.xs(t, level=1, axis=1) if len(tickers) > 1 else raw
            out[t] = df[["Close"]].copy()
        except Exception:
            out[t] = pd.DataFrame()
    return out


# --------------------------------------------------------------------------
#                       ThingSpeak – client & batch getter
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_ts_clients(channel_cfg: Tuple[Tuple[int, str]]):
    """คืน dict[channel_id] = thingspeak.Channel(...)"""
    return {
        cid: thingspeak.Channel(cid, apikey, fmt="json")
        for cid, apikey in channel_cfg
    }


async def _get_field_last_async(client: thingspeak.Channel, field_no: int):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: client.get_field_last(field=str(field_no))
    )


async def batch_get_fields(tasks: List[Tuple[thingspeak.Channel, int]]):
    return await asyncio.gather(
        *[_get_field_last_async(c, f) for c, f in tasks], return_exceptions=True
    )


# --------------------------------------------------------------------------
#                               PUBLIC API
# --------------------------------------------------------------------------
def fetch_monitor_and_asset(
    asset_cfg: List[dict], start_date: str | None
) -> Dict[str, dict]:
    """
    ดึงข้อมูลทั้งหมด (history + f(x) + current asset) ครั้งเดียวแบบ async/batch
    คืน  Dict[ticker] = {history, fx_str, asset_val}
    """
    tickers = tuple(a["ticker"] for a in asset_cfg)
    histories = get_histories(tickers)

    # ----------  สร้าง ThingSpeak client รวมครั้งเดียว ----------
    chan_tuple = {
        (a["monitor_field"]["channel_id"], a["monitor_field"]["api_key"])
        for a in asset_cfg
    } | {
        (a["asset_field"]["channel_id"], a["asset_field"]["api_key"])
        for a in asset_cfg
    }
    ts_clients = get_ts_clients(tuple(chan_tuple))

    # ----------  เตรียม batch task ----------
    m_tasks, a_tasks = [], []
    for a in asset_cfg:
        m_conf, a_conf = a["monitor_field"], a["asset_field"]
        m_tasks.append((ts_clients[m_conf["channel_id"]], m_conf["field"]))
        a_tasks.append((ts_clients[a_conf["channel_id"]], a_conf["field"]))

    fx_raw_list, asset_raw_list = asyncio.run(
        asyncio.gather(batch_get_fields(m_tasks), batch_get_fields(a_tasks))
    )

    # ----------  ประกอบผล ----------
    results: Dict[str, dict] = {}
    for idx, cfg in enumerate(asset_cfg):
        tkr = cfg["ticker"]

        # f(x)
        fx_str = "0"
        raw_fx = fx_raw_list[idx]
        if not isinstance(raw_fx, Exception):
            try:
                js = json.loads(raw_fx)
                fx_str = str(js.get(f"field{cfg['monitor_field']['field']}", "0"))
            except Exception:
                pass

        # current asset
        asset_val = 0.0
        raw_asset = asset_raw_list[idx]
        if not isinstance(raw_asset, Exception):
            try:
                js = json.loads(raw_asset)
                asset_val = float(js[cfg["asset_field"]["field"]])
            except Exception:
                pass

        hist = histories[tkr]
        if start_date:
            hist = hist[hist.index >= start_date]

        results[tkr] = {
            "history": hist,
            "fx_str": fx_str,
            "asset_val": asset_val,
        }

    # เก็บ client ไว้ใน MEM เพื่อให้หน้า UI update asset ได้
    MEM.ttl_set("ts_clients", ts_clients)
    return results
