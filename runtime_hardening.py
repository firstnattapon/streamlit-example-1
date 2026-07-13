from __future__ import annotations

import concurrent.futures
import datetime as dt
import faulthandler
import functools
import importlib
import logging
import numbers
import os
import threading
import time
from typing import Any, Hashable

_LOCK = threading.Lock()
_INSTALLED = False
_NEGATIVE_CACHE: dict[Hashable, float] = {}
_NEGATIVE_LOCK = threading.Lock()


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        import pandas as pd
        result = pd.isna(value)
        try:
            return bool(result)
        except (TypeError, ValueError):
            return False
    except Exception:
        return False


def _timestamp_type() -> type:
    try:
        import pandas as pd
        return pd.Timestamp
    except Exception:
        return dt.datetime


def _family(value: Any) -> str:
    if _missing(value):
        return "missing"
    if isinstance(value, (_timestamp_type(), dt.datetime, dt.date)):
        return "datetime"
    if isinstance(value, (str, bytes, bytearray)):
        return "text"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, numbers.Number):
        return "number"
    return type(value).__name__


def _mixed(values: Any) -> bool:
    families = {_family(value) for value in values}
    families.discard("missing")
    return len(families) > 1


def _text(value: Any) -> str:
    if _missing(value):
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, bytearray):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, (_timestamp_type(), dt.datetime, dt.date)):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def normalize_dataframe_for_streamlit(data: Any) -> Any:
    """Return an Arrow-safe display copy without changing source data."""
    try:
        import pandas as pd
    except Exception:
        return data
    if not isinstance(data, pd.DataFrame):
        return data

    output = data.copy()
    if isinstance(output.index, pd.MultiIndex):
        if any(_mixed(level) for level in output.index.levels):
            output.index = pd.MultiIndex.from_tuples(
                [tuple(_text(part) for part in item) for item in output.index],
                names=output.index.names,
            )
    elif output.index.dtype == object and _mixed(output.index):
        output.index = pd.Index(
            [_text(value) for value in output.index],
            name=output.index.name,
            dtype="object",
        )

    for column in output.columns:
        series = output[column]
        if series.dtype == object and _mixed(series.array):
            output[column] = series.map(_text)
    return output


def _install_streamlit() -> None:
    try:
        import streamlit as st
    except Exception:
        return

    current = getattr(st, "dataframe", None)
    if current is not None and not getattr(current, "_runtime_hardened", False):
        original = current

        @functools.wraps(original)
        def safe_dataframe(data: Any = None, *args: Any, **kwargs: Any) -> Any:
            return original(normalize_dataframe_for_streamlit(data), *args, **kwargs)

        safe_dataframe._runtime_hardened = True  # type: ignore[attr-defined]
        st.dataframe = safe_dataframe

    try:
        from streamlit.delta_generator import DeltaGenerator
        current_method = DeltaGenerator.dataframe
        if not getattr(current_method, "_runtime_hardened", False):
            original_method = current_method

            @functools.wraps(original_method)
            def safe_method(self: Any, data: Any = None, *args: Any, **kwargs: Any) -> Any:
                return original_method(self, normalize_dataframe_for_streamlit(data), *args, **kwargs)

            safe_method._runtime_hardened = True  # type: ignore[attr-defined]
            DeltaGenerator.dataframe = safe_method
    except Exception:
        pass

    if hasattr(st, "iframe"):
        try:
            components = importlib.import_module("streamlit.components.v1")
            current_iframe = getattr(components, "iframe", None)
            if current_iframe is not None and not getattr(current_iframe, "_runtime_hardened", False):
                def safe_iframe(
                    src: str,
                    width: int | None = None,
                    height: int | None = None,
                    scrolling: bool | int = False,
                    **kwargs: Any,
                ) -> Any:
                    supported = {k: v for k, v in kwargs.items() if k == "tab_index"}
                    return st.iframe(
                        src,
                        width=width,
                        height=height,
                        scrolling=bool(scrolling),
                        **supported,
                    )

                safe_iframe._runtime_hardened = True  # type: ignore[attr-defined]
                components.iframe = safe_iframe
        except Exception:
            pass


def _install_thread_cap() -> None:
    try:
        import concurrent.futures.thread as thread_module
        original = thread_module.ThreadPoolExecutor
        if getattr(original, "_runtime_hardened", False):
            return
        cap = _env_int("STREAMLIT_MAX_THREAD_WORKERS", 2)

        class CappedThreadPoolExecutor(original):
            _runtime_hardened = True

            def __init__(self, max_workers: int | None = None, *args: Any, **kwargs: Any) -> None:
                max_workers = cap if max_workers is None else min(max(1, int(max_workers)), cap)
                super().__init__(max_workers=max_workers, *args, **kwargs)

        thread_module.ThreadPoolExecutor = CappedThreadPoolExecutor
        concurrent.futures.ThreadPoolExecutor = CappedThreadPoolExecutor
    except Exception:
        pass


def _empty_history() -> Any:
    try:
        import pandas as pd
        return pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"],
            index=pd.DatetimeIndex([], tz="UTC", name="Date"),
        )
    except Exception:
        return None


def _install_yfinance() -> None:
    try:
        import pandas as pd
        import yfinance as yf
    except Exception:
        return

    ticker_class = getattr(yf, "Ticker", None)
    history = getattr(ticker_class, "history", None)
    if history is not None and not getattr(history, "_runtime_hardened", False):
        original = history
        ttl = _env_int("YFINANCE_NEGATIVE_CACHE_TTL_SECONDS", 21600)

        @functools.wraps(original)
        def safe_history(self: Any, *args: Any, **kwargs: Any) -> Any:
            symbol = str(getattr(self, "ticker", "")).upper()
            key = (symbol, repr(args), repr(sorted(kwargs.items())))
            now = time.monotonic()
            with _NEGATIVE_LOCK:
                if _NEGATIVE_CACHE.get(key, 0.0) > now:
                    return _empty_history()
            try:
                result = original(self, *args, **kwargs)
            except Exception as exc:
                message = str(exc).lower()
                markers = ("possibly delisted", "no price data found", "no data found", "symbol may be delisted")
                if not any(marker in message for marker in markers):
                    raise
                result = _empty_history()
            if isinstance(result, pd.DataFrame) and result.empty:
                with _NEGATIVE_LOCK:
                    _NEGATIVE_CACHE[key] = now + ttl
            return result

        safe_history._runtime_hardened = True  # type: ignore[attr-defined]
        ticker_class.history = safe_history

    class NoDataFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            message = record.getMessage().lower()
            return not any(text in message for text in ("possibly delisted", "no price data found", "symbol may be delisted"))

    for name in ("yfinance", "yfinance.scrapers.history"):
        logger = logging.getLogger(name)
        if not any(isinstance(item, NoDataFilter) for item in logger.filters):
            logger.addFilter(NoDataFilter())


def install_runtime_hardening() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    with _LOCK:
        if _INSTALLED:
            return
        try:
            if not faulthandler.is_enabled():
                faulthandler.enable(all_threads=True)
        except Exception:
            pass
        _install_streamlit()
        _install_yfinance()
        _install_thread_cap()
        _INSTALLED = True
