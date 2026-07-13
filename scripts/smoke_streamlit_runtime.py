"""Offline Arrow-safety smoke test."""

import argparse
import sys

import pandas as pd

from runtime_hardening import install_runtime_hardening, normalize_dataframe_for_streamlit


def run_cycle() -> None:
    frame = pd.DataFrame(
        {"index": [1, 0], "Close": ["", 10.25], "action": ["", 1]},
        index=["+1", pd.Timestamp("2026-07-13 09:30:00", tz="Asia/Bangkok")],
    )
    frame.index.name = "↓ index"
    safe = normalize_dataframe_for_streamlit(frame)
    assert all(isinstance(value, str) for value in safe.index)
    try:
        import pyarrow as pa
    except ImportError:
        return
    pa.Table.from_pandas(safe, preserve_index=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=20)
    args = parser.parse_args()
    install_runtime_hardening()
    for _ in range(max(1, args.cycles)):
        run_cycle()
    print(f"PASS: {max(1, args.cycles)} Arrow-safe cycles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
