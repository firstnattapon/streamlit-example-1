"""pages/7_🚀_Un_15_F(X)_แนะนำ.py: pure functions are read from the page via ``ast`` (no streamlit / yfinance needed).

Checks:
- the engine is the original Un_15 engine (same numbers when DIFF = 0, with pred = 1 or a DNA action stream)
- the review findings the page is built on: Cash_Balan cancels out of net_pv, an inverse pair inflates E while the
  cash A stays near zero (and gets flagged), rolling floors equal re-based cumulative cash, no look-ahead
- the helper formulas: risk parity, effective bets, Dalio's √(ρ + (1 − ρ)/N), E ≈ ½·FAV·σ²
- a Streamlit AppTest run of the whole page with a fake yfinance (skipped when streamlit is not installed)
"""

from __future__ import annotations

import ast
import os
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PAGE = next((ROOT / "pages").glob("7_*Un_15_F(X)_แนะนำ.py"))
ORIGINAL = next(p for p in (ROOT / "pages").glob("7_*Un_15_F(X).py"))

PURE_NAMES = {
    "TRADING_DAYS", "INVERSE_CORR_THRESHOLD", "LEVERAGED_OR_INVERSE", "DETAIL_PERIOD_OFFSETS",
    "build_actions", "constant_value_core", "simulate_constant_value_path", "aggregate_portfolio", "running_floor",
    "capital_at_risk_path", "annualized_vol", "risk_parity_allocation", "equal_allocation", "correlation_summary",
    "dalio_risk_ratio", "diversification_ratio", "trap_flags", "projected_annual_excess", "rolling_window_stats",
    "build_detailed_simulation_frame",
}


def load_pure(path: Path, names: set[str]) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            body.append(node)
    found = {n.name if isinstance(n, ast.FunctionDef) else n.targets[0].id for n in body}
    missing = names - found
    if missing:
        raise AssertionError(f"{path.name}: missing {sorted(missing)}")
    namespace = {"np": np, "pd": pd}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


NEW = load_pure(PAGE, PURE_NAMES)
OLD = load_pure(ORIGINAL, {"simulate_constant_value_path"})


def gbm(n: int, sigma: float, seed: int, p0: float = 30.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = rng.normal(-0.5 * sigma ** 2 / 252, sigma / np.sqrt(252), n - 1)
    return p0 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))


class EngineMatchesOriginal(unittest.TestCase):
    def test_every_bar_equals_original(self):
        prices = gbm(400, 0.6, 1)
        ones = np.ones(len(prices), dtype=np.int8)
        new = NEW["simulate_constant_value_path"](prices, ones, 1500.0, 650.0)
        old = OLD["simulate_constant_value_path"](prices, ones, 1500.0, 650.0)
        pd.testing.assert_frame_equal(new, old, rtol=1e-12, atol=1e-9)

    def test_dna_stream_equals_original(self):
        prices = gbm(300, 0.8, 2)
        actions = np.random.default_rng(3).integers(0, 2, len(prices)).astype(np.int8)
        new = NEW["simulate_constant_value_path"](prices, actions, 1200.0, 0.0)
        old = OLD["simulate_constant_value_path"](prices, actions, 1200.0, 0.0)
        pd.testing.assert_frame_equal(new, old, rtol=1e-12, atol=1e-9)

    def test_diff_threshold_keeps_invariant_and_trades_less(self):
        prices = gbm(500, 0.7, 4)
        ones = np.ones(len(prices), dtype=np.int8)
        every = NEW["simulate_constant_value_path"](prices, ones, 1500.0, 0.0)
        gated = NEW["simulate_constant_value_path"](prices, ones, 1500.0, 0.0, diff=50.0)
        self.assertGreaterEqual(float(gated["net_pv"].min()), 0.0)
        self.assertLess(int((gated["re"] != 0).sum()), int((every["re"] != 0).sum()))
        self.assertTrue(np.all(np.abs(gated["re"][gated["re"] != 0]) > 50.0))

    def test_build_actions_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            NEW["build_actions"]([1, 0, 1], 4)
        self.assertEqual(NEW["build_actions"](1, 3).tolist(), [1, 1, 1])


class ReviewFindings(unittest.TestCase):
    def test_cash_balan_cancels_out_of_net_pv(self):
        prices = gbm(250, 0.5, 5)
        ones = np.ones(len(prices), dtype=np.int8)
        a = NEW["simulate_constant_value_path"](prices, ones, 1500.0, 0.0)["net_pv"]
        b = NEW["simulate_constant_value_path"](prices, ones, 1500.0, 650.0)["net_pv"]
        np.testing.assert_allclose(a, b, atol=1e-9)

    def test_inverse_pair_inflates_e_but_not_cash_and_is_flagged(self):
        silver = gbm(504, 0.6, 6)
        r = silver[1:] / silver[:-1] - 1
        inverse = 50.0 * np.concatenate([[1.0], np.cumprod(1 - 2 * r)])   # −2x reset every bar
        ones = np.ones(len(silver), dtype=np.int8)
        sim = NEW["simulate_constant_value_path"]
        alone = sim(silver, ones, 1500.0, 0.0)
        leg_s = sim(silver, ones, 1000.0, 0.0)
        leg_z = sim(inverse, ones, 500.0, 0.0)
        e_pair = leg_s["net_pv"].iloc[-1] + leg_z["net_pv"].iloc[-1]
        a_pair = leg_s["re"].sum() + leg_z["re"].sum()
        self.assertGreater(e_pair, 1.8 * alone["net_pv"].iloc[-1])           # E ≈ ×2
        self.assertLess(abs(a_pair), 1e-6 * 1500)                              # A = F·(3w − 2)·Σr = 0 at w = 2/3
        returns = pd.DataFrame({"SLV": silver, "INV": inverse}).pct_change().dropna()
        corr, _, _ = NEW["correlation_summary"](returns)
        flags = NEW["trap_flags"](["SLV", "INV"], corr)
        self.assertIn("INV", flags)
        self.assertIn("ZSL", NEW["trap_flags"](["SLV", "ZSL"]))

    def test_rolling_floor_equals_rebased_cumulative_cash(self):
        idx = pd.date_range("2024-01-01", periods=260, freq="B")
        frame = pd.DataFrame({"A": gbm(260, 0.5, 7), "B": gbm(260, 0.7, 8, 80.0)}, index=idx)
        favs = {"A": 900.0, "B": 600.0}
        rolling = NEW["rolling_window_stats"](frame, favs, horizon=60, step=10)
        ones = np.ones(260, dtype=np.int8)
        cum = sum(np.cumsum(NEW["constant_value_core"](frame[t].to_numpy(), ones, favs[t])[0]) for t in favs)
        for _, row in rolling.iterrows():
            s = frame.index.get_loc(row["start"])
            window = cum[s:s + 61] - cum[s]
            self.assertAlmostEqual(row["floor"], min(0.0, window.min()), places=8)
            self.assertAlmostEqual(row["cash"], window[-1], places=8)
        self.assertEqual(len(rolling), len(range(0, 260 - 60, 10)))
        self.assertTrue((rolling["excess"] >= -1e-9).all())

    def test_capital_path_has_no_lookahead(self):
        cum = np.array([0.0, -10.0, 5.0, -30.0, -20.0, 40.0])
        path = NEW["capital_at_risk_path"](1500.0, cum)
        np.testing.assert_allclose(path, [1500, 1510, 1510, 1530, 1530, 1530])
        for t in range(len(cum)):
            self.assertEqual(NEW["capital_at_risk_path"](1500.0, cum[:t + 1])[-1], path[t])


class Formulas(unittest.TestCase):
    def test_risk_parity_equalises_fav_times_sigma(self):
        favs = NEW["risk_parity_allocation"]({"SLV": 0.39, "PLTR": 0.68, "X": float("nan")}, 1500.0)
        self.assertNotIn("X", favs)
        self.assertAlmostEqual(sum(favs.values()), 1500.0)
        self.assertAlmostEqual(favs["SLV"] * 0.39, favs["PLTR"] * 0.68)
        self.assertAlmostEqual(favs["SLV"] / 1500.0, 0.68 / (0.39 + 0.68))      # w_SLV = σ_P/(σ_S + σ_P) = 0.64

    def test_effective_bets_and_average_rho(self):
        rng = np.random.default_rng(9)
        independent = pd.DataFrame(rng.normal(size=(20000, 4)))
        _, avg, eff = NEW["correlation_summary"](independent)
        self.assertAlmostEqual(avg, 0.0, delta=0.03)
        self.assertAlmostEqual(eff, 4.0, delta=0.05)
        base = rng.normal(size=500)
        _, avg, eff = NEW["correlation_summary"](pd.DataFrame({"a": base, "b": 2 * base, "c": base + 1}))
        self.assertAlmostEqual(avg, 1.0)
        self.assertAlmostEqual(eff, 1.0)

    def test_dalio_ratio_matches_holy_grail_chart(self):
        f = NEW["dalio_risk_ratio"]
        self.assertAlmostEqual(f(0.0, 20), np.sqrt(0.05))          # 10% → 2.2%
        self.assertAlmostEqual(f(0.6, 20), np.sqrt(0.62))          # 10% → 7.9%
        self.assertAlmostEqual(f(0.2, 1), 1.0)

    def test_diversification_ratio(self):
        f = NEW["diversification_ratio"]
        self.assertAlmostEqual(f(-100.0, [-60.0, -40.0]), 1.0)
        self.assertAlmostEqual(f(0.0, [-60.0, -40.0]), 0.0)
        self.assertTrue(np.isnan(f(0.0, [0.0, 0.0])))

    def test_projected_excess_is_half_f_sigma_squared(self):
        self.assertAlmostEqual(NEW["projected_annual_excess"]({"a": 1500.0}, {"a": 0.5}), 187.5)
        sims = [NEW["simulate_constant_value_path"](gbm(253, 0.5, s), np.ones(253, dtype=np.int8), 1500.0, 0.0)
                ["net_pv"].iloc[-1] for s in range(300)]
        self.assertAlmostEqual(float(np.mean(sims)), 187.5, delta=0.1 * 187.5)

    def test_annualized_vol(self):
        prices = gbm(2000, 0.4, 10)
        self.assertAlmostEqual(NEW["annualized_vol"](prices), 0.4, delta=0.03)
        self.assertTrue(np.isnan(NEW["annualized_vol"]([1.0, 2.0])))


def _fake_yfinance() -> types.ModuleType:
    module = types.ModuleType("yfinance")

    class Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, period="max", auto_adjust=True):
            idx = pd.date_range("2023-01-02", periods=700, freq="B", tz="America/New_York")
            seed = sum(map(ord, self.symbol))
            return pd.DataFrame({"Close": gbm(len(idx), 0.6, seed, 20.0 + seed % 50)}, index=idx)

    module.Ticker = Ticker
    return module


try:
    from streamlit.testing.v1 import AppTest
except Exception:  # streamlit not installed
    AppTest = None


@unittest.skipUnless(AppTest is not None, "streamlit not installed")
class PageSmoke(unittest.TestCase):
    def setUp(self):
        self._cwd = os.getcwd()
        self._yf = sys.modules.get("yfinance")
        os.chdir(ROOT)
        sys.modules["yfinance"] = _fake_yfinance()

    def tearDown(self):
        os.chdir(self._cwd)
        if self._yf is None:
            sys.modules.pop("yfinance", None)
        else:
            sys.modules["yfinance"] = self._yf

    def test_page_runs_with_two_tickers(self):
        at = AppTest.from_file(str(PAGE), default_timeout=120)
        base = {"Fixed_Asset_Value": 1500.0, "Cash_Balan": 650.0, "filter_date": "2024-06-01 12:00:00+07:00", "pred": 1}
        at.session_state["un15r_custom_tickers"] = {t: {"Ticker": t, **base} for t in ("SLV", "PLTR")}
        at.run()
        self.assertFalse(at.exception, at.exception)
        at.multiselect(key="un15r_selected").set_value(["SLV", "PLTR"]).run()
        self.assertFalse(at.exception, at.exception)
        labels = [m.label for m in at.metric]
        for label in ("ΣE · Excess vs Log Ref", "ΣA · เงินสดจริง", "Diversification ratio", "Effective bets"):
            self.assertIn(label, labels)
        for sizing in ("เท่ากัน · งบรวมคงที่", "กำหนดเองรายตัว (แบบหน้าเดิม)"):
            at.radio(key="un15r_sizing").set_value(sizing).run()
            self.assertFalse(at.exception, at.exception)


if __name__ == "__main__":
    unittest.main()
