# helpers/data_hub.py

import streamlit as st
import thingspeak
import yfinance as yf
import json
import concurrent.futures
import datetime
from threading import Lock
from typing import Dict, Any, List

# --------------------------------------------------------------------
#               A Simple In-Memory Time-To-Live (TTL) Cache
# --------------------------------------------------------------------
class SimpleTTLCache:
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
        self._lock = Lock()

    def ttl_get(self, key: str, ttl_seconds: int) -> Any:
        with self._lock:
            if key not in self._cache:
                return None
            if (datetime.datetime.now() - self._timestamps[key]).total_seconds() > ttl_seconds:
                self.delete(key)
                return None
            return self._cache.get(key)

    def set(self, key: str, value: Any):
        with self._lock:
            self._cache[key] = value
            self._timestamps[key] = datetime.datetime.now()

    def get(self, key: str) -> Any:
         with self._lock:
            return self._cache.get(key)

    def delete(self, key: str):
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._timestamps[key]

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()

# Global instance of our custom cache, can be imported elsewhere
MEM = SimpleTTLCache()

# --------------------------------------------------------------------
#                       ThingSpeak Client Management
# --------------------------------------------------------------------
def get_ts_clients(configs: List[Dict]) -> Dict[int, thingspeak.Channel]:
    cached_clients = MEM.ttl_get("ts_clients", 3600) # Cache clients for 1 hour
    if cached_clients is not None:
        return cached_clients

    clients = {}
    unique_channels = set()
    for config in configs:
        for key in ['monitor_field', 'asset_field']:
            if key in config:
                conf = config[key]
                unique_channels.add((conf['channel_id'], conf['api_key']))

    for channel_id, api_key in unique_channels:
        try:
            clients[channel_id] = thingspeak.Channel(channel_id, api_key, fmt='json')
        except Exception as e:
            st.warning(f"[DataHub] Failed to create client for Channel ID {channel_id}: {e}")

    MEM.set("ts_clients", clients)
    return clients

# --------------------------------------------------------------------
#                Concurrent Data Fetching using Threads
# --------------------------------------------------------------------
def _fetch_worker(asset_config: Dict, ts_clients: Dict, start_date: str) -> Dict[str, Any]:
    """
    (Internal Worker) Fetches all data for a single asset.
    - yfinance history
    - ThingSpeak f(x) string
    - ThingSpeak asset value
    """
    ticker = asset_config['ticker']
    result = {
        'ticker': ticker,
        'history': None,
        'fx_str': "0",
        'asset_val': 0.0
    }

    # 1. Fetch yfinance history
    try:
        history_df = yf.Ticker(ticker).history(period='max', auto_adjust=True)[['Close']].round(3)
        history_df.index = history_df.index.tz_convert(tz='Asia/Bangkok')
        if start_date:
            history_df = history_df[history_df.index >= start_date]
        result['history'] = history_df.tail(7)
    except Exception as e:
        st.warning(f"[DataHub: yfinance] Error for {ticker}: {e}")

    # 2. Fetch f(x) string from ThingSpeak
    try:
        monitor_conf = asset_config.get('monitor_field', {})
        client = ts_clients.get(monitor_conf.get('channel_id'))
        if client:
            field_num = monitor_conf['field']
            field_data = client.get_field_last(field=str(field_num))
            retrieved_val = json.loads(field_data).get(f"field{field_num}")
            if retrieved_val is not None:
                result['fx_str'] = str(retrieved_val)
    except Exception:
        pass # Fail silently, fx_str will default to "0"

    # 3. Fetch asset value from ThingSpeak
    try:
        asset_conf = asset_config.get('asset_field', {})
        client = ts_clients.get(asset_conf.get('channel_id'))
        if client:
            field_name = asset_conf['field']
            data = client.get_field_last(field=field_name)
            result['asset_val'] = float(json.loads(data)[field_name])
    except Exception:
        pass # Fail silently, asset_val will default to 0.0

    return result


def fetch_monitor_and_asset(configs: List[Dict], start_date: str) -> Dict[str, Dict]:
    """
    The main function of this module.
    Fetches all required data for all assets concurrently using ThreadPoolExecutor.
    The entire result bundle is cached for 60 seconds.
    """
    # Check cache first for the entire data bundle
    cached_bundle = MEM.ttl_get("data_bundle", 60)
    if cached_bundle:
        return cached_bundle

    ts_clients = get_ts_clients(configs)
    results_bundle = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(configs) or 1) as executor:
        future_to_ticker = {
            executor.submit(_fetch_worker, asset_cfg, ts_clients, start_date): asset_cfg['ticker']
            for asset_cfg in configs
        }
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                data = future.result()
                results_bundle[ticker] = data
            except Exception as e:
                st.error(f"[DataHub] Critical error in fetch worker for {ticker}: {e}")
                results_bundle[ticker] = {'ticker': ticker, 'history': None, 'fx_str': "0", 'asset_val': 0.0}

    # Cache the freshly fetched bundle
    MEM.set("data_bundle", results_bundle)
    return results_bundle
