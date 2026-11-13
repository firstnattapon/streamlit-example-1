"""Tab 9 (Now Panel): the panel is injected into the DNA Spectrum Explorer template at build time.

Only string constants and the pure ``inject_fx_now_panel`` function are read from the page via
``ast`` (no streamlit, numba or yfinance needed). The tests make sure that every anchor is found
exactly once, that the injection only adds lines, and that the numbers given in the spec are the
ones that ended up in the JavaScript.

The cost table ("E หลังหักต้นทุนเทรด") is checked two ways: its markup and defaults as strings, and its
arithmetic by running the panel's own JavaScript in Node against an order-by-order simulation written
independently here (skipped when Node is not installed).
"""

from __future__ import annotations

import ast
import difflib
import json
import math
import re
import shutil
import subprocess
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

PAGE = next((Path(__file__).resolve().parent.parent / "pages").glob("4_*Hybrid_Multi_Mutation.py"))
REQUIRED_CSS_CLASSES = [
    "fx-card", "fx-verdict", "fx-badge-trend", "fx-badge-revert", "fx-badge-flat", "fx-pair",
    "fx-grid", "fx-cell", "fx-cell-dim", "fx-cell-sig", "fx-rowlabel", "fx-howto",
]
REQUIRED_IDS = [
    "fx-now-section", "fx-verdict", "fx-regime-badge", "fx-verdict-text", "fx-apply-btn",
    "fx-price-svg", "fx-ratio-grid", "fx-howto",
    "fx-cost", "fx-cost-pct", "fx-cost-fee", "fx-cost-table", "fx-cost-note",
]
FIX_CAP = 1500.0
COST_LADDER_M15 = [("15m", 1), ("30m", 2), ("1h", 4), ("4h", 16), ("1d", 26), ("1w", 130)]


def simulate_orders(prices: List[float], k: int, ph: int) -> tuple[float, float, int]:
    """One (k, ph) run, order by order, the way the page's calcNetProfit trades: returns (E, traded $, orders)."""
    n = len(prices)
    actions = [0] * n
    for i in range(ph, n, k):
        actions[i] = 1
    actions[0] = actions[-1] = 1
    cash, shares, traded, orders = FIX_CAP, FIX_CAP / prices[0], 0.0, 0
    for i in range(1, n):
        if not actions[i]:
            continue
        order_value = shares * prices[i] - FIX_CAP  # > 0 sell, < 0 buy, 0 = price unchanged, nothing to send
        cash += order_value
        shares = FIX_CAP / prices[i]
        if order_value != 0:
            traded += abs(order_value)
            orders += 1
    reference = FIX_CAP * math.log(prices[-1] / prices[0])
    return cash + shares * prices[-1] - reference - 2 * FIX_CAP, traded, orders


def phase_average(prices: List[float], k: int) -> tuple[float, float, float]:
    runs = [simulate_orders(prices, k, ph) for ph in range(k)]
    return tuple(sum(r[j] for r in runs) / k for j in range(3))  # type: ignore[return-value]


def load_page_pieces() -> dict[str, Any]:
    source = PAGE.read_text(encoding="utf-8")
    namespace: dict[str, Any] = {"List": List, "Dict": Dict, "Any": Any, "Optional": Optional}
    for node in ast.parse(source).body:
        name = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        elif isinstance(node, ast.FunctionDef):
            name = node.name
        if name == "DNA_SPECTRUM_EXPLORER_TEMPLATE_HTML" or (name or "").startswith("FX_NOW_") or name == "inject_fx_now_panel":
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(PAGE), "exec"), namespace)
    namespace["_source"] = source
    return namespace


class FxNowPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ns = load_page_pieces()
        cls.template: str = cls.ns["DNA_SPECTRUM_EXPLORER_TEMPLATE_HTML"]
        cls.main_tag: str = cls.ns["FX_NOW_MAIN_ANCHOR"]
        cls.inject = staticmethod(cls.ns["inject_fx_now_panel"])
        cls.patched: str = cls.inject(cls.template)

    def test_template_anchors_exist_once(self) -> None:
        self.assertEqual(self.template.count(self.main_tag), 1)
        self.assertEqual(self.template.count("</head>"), 1)
        self.assertEqual(len(re.findall(r"^const COMPACT_COLS\b", self.template, flags=re.M)), 1)
        self.assertEqual(len(re.findall(r"^function renderAll\(\) \{", self.template, flags=re.M)), 1)

    def test_panel_pieces_are_added_once(self) -> None:
        for element_id in REQUIRED_IDS:
            self.assertEqual(self.patched.count(f'id="{element_id}"'), 1, element_id)
        self.assertEqual(self.patched.count('id="fx-now-styles"'), 1)
        for needle in (
            "const FX_LADDER", "function fxRatioRow", "function computeFxVerdict", "function renderFxNow",
            "const FX_COST_LADDER", "function fxCostStats", "function fxCostRows", "function renderFxCost",
        ):
            self.assertEqual(self.patched.count(needle), 1, needle)

    def test_section_first_in_main_and_style_last_in_head(self) -> None:
        after_main = self.patched[self.patched.index(self.main_tag) + len(self.main_tag):].lstrip()
        self.assertTrue(after_main.startswith('<section id="fx-now-section"'))
        head = self.patched[: self.patched.index("</head>")].rstrip()
        self.assertTrue(head.endswith("</style>"))
        self.assertTrue(head[head.rindex("<style"):].startswith('<style id="fx-now-styles">'))

    def test_render_all_calls_panel_after_equity_curve(self) -> None:
        found = re.search(r"^function renderAll\(\) \{\n.*?^\}\n", self.patched, flags=re.M | re.S)
        self.assertIsNotNone(found)
        self.assertRegex(found.group(0), r"renderEquityCurve\(\);\s*renderFxNow\(\);\s*\}\n$")

    def test_injection_only_adds_lines(self) -> None:
        ops = difflib.SequenceMatcher(None, self.template.splitlines(), self.patched.splitlines(), autojunk=False).get_opcodes()
        self.assertEqual({tag for tag, *_ in ops} - {"equal", "insert"}, set())

    def test_injection_is_idempotent(self) -> None:
        self.assertEqual(self.inject(self.patched), self.patched)

    def test_missing_anchor_raises(self) -> None:
        broken = {
            "main": self.template.replace(self.main_tag, "<main>"),
            "COMPACT_COLS": self.template.replace("const COMPACT_COLS", "const COMPACT_COLS_X"),
            "renderAll": self.template.replace("function renderAll() {", "function renderEverything() {"),
            "renderEquityCurve": re.sub(
                r"(function renderAll\(\) \{.*?)renderEquityCurve\(\);(.*?^\}\n)", r"\1\2", self.template, count=1, flags=re.M | re.S
            ),
        }
        for label, html in broken.items():
            with self.subTest(anchor=label), self.assertRaises(ValueError):
                self.inject(html)

    def test_js_constants_match_spec(self) -> None:
        js = self.ns["FX_NOW_JS_CONSTS"]
        self.assertRegex(js, r"M15:\s*\[1, 2, 4, 8, 13, 26\]")
        self.assertRegex(js, r"D:\s*\[1, 2, 3, 5\]")
        self.assertRegex(js, r"FX_LOOKBACK\s*=\s*\{\s*M15:\s*3,\s*D:\s*0\s*\}")
        self.assertRegex(js, r"FX_Z_THR\s*=\s*2\.5;")
        self.assertRegex(js, r"FX_ELBOW\s*=\s*0\.9;")

    def test_cost_ladder_matches_spec(self) -> None:
        js = self.ns["FX_NOW_JS_CONSTS"]
        self.assertRegex(
            js, r"M15:\s*\[\['15m', 1\], \['30m', 2\], \['1h', 4\], \['4h', 16\], \['1d', 26\], \['1w', 130\]\]"
        )
        self.assertRegex(
            js, r"D:\s*\[\['15m', null\], \['30m', null\], \['1h', null\], \['4h', null\], \['1d', 1\], \['1w', 5\]\]"
        )

    def test_cost_table_renders_with_the_panel(self) -> None:
        found = re.search(r"^function renderFxNow\(\) \{\n.*?^\}\n", self.patched, flags=re.M | re.S)
        self.assertIsNotNone(found)
        self.assertIn("renderFxCost(v);", found.group(0))
        for field in ("fx-cost-pct", "fx-cost-fee"):
            self.assertRegex(self.patched, rf'<input id="{field}"[^>]*oninput="renderFxNow\(\)"')

    def test_cost_defaults_are_injected_safely(self) -> None:
        def values(html: str) -> tuple[str, str]:
            pct = re.search(r'<input id="fx-cost-pct"[^>]*\bvalue="([^"]*)"', html)
            fee = re.search(r'<input id="fx-cost-fee"[^>]*\bvalue="([^"]*)"', html)
            assert pct and fee
            return pct.group(1), fee.group(1)

        self.assertNotIn("__FX_COST_", self.patched)
        self.assertEqual(values(self.patched), ("0", "0"))
        self.assertEqual(values(self.inject(self.template, cost_pct=0.1, cost_fee=1)), ("0.1", "1"))
        self.assertEqual(values(self.inject(self.template, cost_pct=0.12345678, cost_fee=0.05)), ("0.1235", "0.05"))
        for bad in (-1, float("nan"), float("inf"), "abc", None, '1"><script>'):
            with self.subTest(bad=bad):
                self.assertEqual(values(self.inject(self.template, cost_pct=bad, cost_fee=bad)), ("0", "0"))

    @unittest.skipUnless(shutil.which("node"), "needs Node to run the panel's JavaScript")
    def test_cost_numbers_match_order_by_order_simulation(self) -> None:
        def js_block(pattern: str) -> str:
            found = re.search(pattern, self.patched, flags=re.M | re.S)
            self.assertIsNotNone(found, pattern)
            return found.group(0)

        # deterministic prices: a drifting zig-zag with flat stretches (no order when the price did not move)
        seed, prices = 12345, [20.0]
        for i in range(1, 61):
            seed = (seed * 1103515245 + 12345) % 2**31
            step = 0.0 if i % 7 == 0 else ((seed / 2**31) - 0.48) * 0.9
            prices.append(round(max(1.0, prices[-1] + step), 2))
        windows = [prices[:37], prices[20:], prices[5:9]]  # 37, 41 and 4 bars: k = 130 is longer than all of them
        pct, fee = 0.1, 0.25

        harness = "\n".join([
            "const FIX_CAP = 1500;",
            js_block(r"^function calcNetPeriodic\(prices, k, ph\) \{\n.*?^\}\n"),
            js_block(r"^const FX_COST_LADDER = \{\n.*?^\};\n"),
            js_block(r"^function fxCostStats\(prices, k\) \{\n.*?^\}\n"),
            js_block(r"^function fxCostRows\(wins, tf, pct, fee\) \{\n.*?^\}\n"),
            "const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));",
            "const wins = input.windows.map(prices => ({ prices }));",
            "console.log(JSON.stringify({",
            "  m15: fxCostRows(wins, 'M15', input.pct, input.fee),",
            "  d: fxCostRows(wins, 'D', input.pct, input.fee)",
            "}));",
        ])
        done = subprocess.run(
            ["node", "-e", harness],
            input=json.dumps({"windows": windows, "pct": pct, "fee": fee}),
            capture_output=True, text=True, timeout=60, check=False,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        got = json.loads(done.stdout)

        self.assertEqual([(r["label"], r["k"]) for r in got["m15"]], COST_LADDER_M15)
        for row in got["m15"]:
            with self.subTest(freq=row["label"]):
                per_window = [phase_average(w, row["k"]) for w in windows]
                e, traded, orders = (sum(x[j] for x in per_window) for j in range(3))
                cost = traded * pct / 100 + orders * fee
                self.assertTrue(row["ok"])
                self.assertAlmostEqual(row["e"], e, places=8)
                self.assertAlmostEqual(row["turn"], traded, places=8)
                self.assertAlmostEqual(row["ord"], orders, places=10)
                self.assertAlmostEqual(row["cost"], cost, places=8)
                self.assertAlmostEqual(row["net"], e - cost, places=8)
                self.assertAlmostEqual(row["bePct"], e / traded * 100, places=8)
                self.assertAlmostEqual(row["beFee"], e / orders, places=8)

        # fewer rebalances -> fewer orders and less traded value, whatever the prices do
        orders = [r["ord"] for r in got["m15"]]
        traded = [r["turn"] for r in got["m15"]]
        self.assertEqual(orders, sorted(orders, reverse=True))
        self.assertEqual(traded, sorted(traded, reverse=True))

        # daily bars cannot show intraday frequencies; 1d and 1w map to 1 and 5 bars
        self.assertEqual([(r["label"], r["k"], r["ok"]) for r in got["d"]],
                         [("15m", None, False), ("30m", None, False), ("1h", None, False), ("4h", None, False),
                          ("1d", 1, True), ("1w", 5, True)])
        self.assertAlmostEqual(got["d"][4]["e"], got["m15"][0]["e"], places=10)

    def test_css_uses_only_fx_classes(self) -> None:
        css = self.ns["FX_NOW_STYLES_HTML"]
        selectors = css
        while re.search(r"\{[^{}]*\}", selectors):  # strip declaration blocks, innermost first
            selectors = re.sub(r"\{[^{}]*\}", " ", selectors)
        classes = set(re.findall(r"\.([A-Za-z_][\w-]*)", selectors))
        self.assertTrue(classes)
        self.assertEqual({c for c in classes if not c.startswith("fx-")}, set())
        self.assertEqual(set(REQUIRED_CSS_CLASSES) - classes, set())

    def test_tab_9_is_registered(self) -> None:
        source = self.ns["_source"]
        main_fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "main")
        text = ast.get_source_segment(source, main_fn) or ""
        self.assertEqual(text.count("with tabs["), 9)
        self.assertIn("with tabs[8]:\n        render_fx_now_tab()", text)


if __name__ == "__main__":
    unittest.main()
