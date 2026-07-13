from __future__ import annotations

import concurrent.futures
import datetime as dt
import faulthandler
import functools
import importlib
import logging
import numbers
import os
import sys
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

