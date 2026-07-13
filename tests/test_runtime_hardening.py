from __future__ import annotations

import concurrent.futures
import datetime as dt
import os
import unittest

import pandas as pd

from runtime_hardening import install_runtime_hardening, normalize_dataframe_for_streamlit


class DataFrameNormalizationTests(unittest.TestCase):
    def test_mixed_index_becomes_text(self) -> None:
        frame = pd.DataFrame(
            {"Close": [None, 10.5]},
            index=["+1", pd.Timestamp("2026-07-13 12:00:00", tz="Asia/Bangkok")],
        )
        frame.index.name = "↓ index"
        result = normalize_dataframe_for_streamlit(frame)
        self.assertEqual(result.index.name, "↓ index")
        self.assertTrue(all(isinstance(value, str) for value in result.index))
        self.assertIsInstance(frame.index[1], pd.Timestamp)

    def test_datetime_index_keeps_dtype(self) -> None:
        frame = pd.DataFrame(
            {"Close": [10.0, 11.0]},
            index=pd.date_range("2026-07-12", periods=2, tz="Asia/Bangkok"),
        )
        self.assertIsInstance(normalize_dataframe_for_streamlit(frame).index, pd.DatetimeIndex)

    def test_mixed_object_column_becomes_text(self) -> None:
        frame = pd.DataFrame({"label": ["future", dt.datetime(2026, 7, 13)]})
        self.assertEqual(
            normalize_dataframe_for_streamlit(frame)["label"].tolist(),
            ["future", "2026-07-13T00:00:00"],
        )


class RuntimeInstallationTests(unittest.TestCase):
    def test_thread_pool_is_capped(self) -> None:
        os.environ["STREAMLIT_MAX_THREAD_WORKERS"] = "2"
        install_runtime_hardening()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            self.assertLessEqual(executor._max_workers, 2)


if __name__ == "__main__":
    unittest.main()
