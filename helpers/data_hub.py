# helpers/data_hub.py
import asyncio, datetime as dt, json
from typing import Dict, List, Tuple

import numpy as np            # (ยังเผื่อใช้)
import pandas as pd
import streamlit as st
import thingspeak
import yfinance as yf


# ----------------  MEM cache  ----------------
class _MemBox(dict):
    def ttl_get(self, k, ttl):
        v, ts = self.get(k, (None, None))
        if v is not None and (dt.datetime.now() - ts).total_seconds() < ttl:
            return v
        return None
    def ttl_set(self, k, v): self[k] = (v, dt.datetime.now())
MEM = _MemBox()

# ---------------- YFinance (batch) ------------
async def _dl_yf_async(tickers: List[str], period="max") -> pd.DataFrame:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: yf.download(
            tickers=" ".join(tickers),
            period=period, group_by="column",
            threads=True, progress=False
        )
    )

@st.cache_data(ttl=300, max_entries=1, show_spinner=False)
def get_histories(tickers: Tuple[str]) -> Dict[str, pd.DataFrame]:
    raw = asyncio.run(_dl_yf_async(list(tickers), "max")).round(3)
    out = {}
    for t in tickers:
        try:
            df = raw.xs(t, level=1, axis=1) if len(tickers) > 1 else raw
            out[t] = df[["Close"]].copy()
        except Exception:
            out[t] = pd.DataFrame()
    return out

# ---------------- ThingSpeak ------------------
@st.cache_resource(show_spinner=False)
def get_ts_clients(ch_tuple: Tuple[Tuple[int, str]]):
    return {cid: thingspeak.Channel(cid, apikey, fmt="json")
            for cid, apikey in ch_tuple}

async def _get_field_last_async(client: thingspeak.Channel, field_no: int):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: client.get_field_last(field=str(field_no))
    )

async def _batch_get_fields(tasks):
    return await asyncio.gather(
        *[_get_field_last_async(c, f) for c, f in tasks],
        return_exceptions=True
    )

# ---------------- PUBLIC ----------------------
def fetch_monitor_and_asset(asset_cfg: List[dict], start_date: str | None):
    tickers = tuple(a["ticker"] for a in asset_cfg)
    histories = get_histories(tickers)

    ch_tuple = {(a["monitor_field"]["channel_id"], a["monitor_field"]["api_key"])
                for a in asset_cfg} | \
               {(a["asset_field"]["channel_id"],   a["asset_field"]["api_key"])
                for a in asset_cfg}
    ts_clients = get_ts_clients(tuple(ch_tuple))

    m_tasks, a_tasks = [], []
    for a in asset_cfg:
        m_conf, a_conf = a["monitor_field"], a["asset_field"]
        m_tasks.append((ts_clients[m_conf["channel_id"]], m_conf["field"]))
        a_tasks.append((ts_clients[a_conf["channel_id"]], a_conf["field"]))

    # ---- FIX : รวบรวมใน async-func แล้วค่อย run ----
    async def _collector():
        return await asyncio.gather(
            _batch_get_fields(m_tasks), _batch_get_fields(a_tasks)
        )

    fx_raw_list, asset_raw_list = asyncio.run(_collector())

    results: Dict[str, dict] = {}
    for idx, cfg in enumerate(asset_cfg):
        tk = cfg["ticker"]

        fx_str = "0"
        raw_fx = fx_raw_list[idx]
        if not isinstance(raw_fx, Exception):
            try:
                js = json.loads(raw_fx)
                fx_str = str(js.get(f"field{cfg['monitor_field']['field']}", "0"))
            except Exception:
                pass

        asset_val = 0.0
        raw_asset = asset_raw_list[idx]
        if not isinstance(raw_asset, Exception):
            try:
                js = json.loads(raw_asset)
                asset_val = float(js[cfg["asset_field"]["field"]])
            except Exception:
                pass

        hist = histories[tk]
        if start_date:
            hist = hist[hist.index >= start_date]

        results[tk] = {
            "history": hist,
            "fx_str":  fx_str,
            "asset_val": asset_val,
        }

    MEM.ttl_set("ts_clients", ts_clients)
    return results
