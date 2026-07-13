"""Automatically loaded process guards for Streamlit Community Cloud."""

import os
import sys

os.environ.setdefault("PYTHONFAULTHANDLER", "1")
os.environ.setdefault("STREAMLIT_MAX_THREAD_WORKERS", "2")
os.environ.setdefault("YFINANCE_NEGATIVE_CACHE_TTL_SECONDS", "21600")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

try:
    from runtime_hardening import install_runtime_hardening

    install_runtime_hardening()
except Exception as exc:
    print(f"[sitecustomize] runtime hardening skipped: {exc}", file=sys.stderr)
