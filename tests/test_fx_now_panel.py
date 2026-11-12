"""Tab 9 (Now Panel): the panel is injected into the DNA Spectrum Explorer template at build time.

Only string constants and the pure ``inject_fx_now_panel`` function are read from the page via
``ast`` (no streamlit, numba or yfinance needed). The tests make sure that every anchor is found
exactly once, that the injection only adds lines, and that the numbers given in the spec are the
ones that ended up in the JavaScript.
"""

from __future__ import annotations

import ast
import difflib
import re
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
]


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
        for needle in ("const FX_LADDER", "function fxRatioRow", "function computeFxVerdict", "function renderFxNow"):
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
