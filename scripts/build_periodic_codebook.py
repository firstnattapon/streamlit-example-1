"""Build periodic_codebook.json for DNA Spectrum mode 4.

The codebook maps every "act every k bars, starting at bar `phase`" pattern
(k = 1 .. length-1, 0 <= phase < k) to a stream DNA code. Mode 4 looks codes
up here instead of compiling them at run time. A code built for `length` bars
also decodes correctly for any shorter window, because PCG64 yields the same
leading bits.

Run from the repository root:

    python scripts/build_periodic_codebook.py            # length 30
    python scripts/build_periodic_codebook.py --length 30

Every code is compiled by the page's own compiler and checked by decoding it
back before it is written.
"""

import argparse
import importlib.util
import json
import logging
import os
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "pages" / "4_🧬_Hybrid_Multi_Mutation.py"
OUTPUT = ROOT / "periodic_codebook.json"


def load_page():
    # Keep numba's on-disk cache out of pages/__pycache__ so it cannot clash
    # with the cache Streamlit writes when it runs the page itself.
    os.environ.setdefault("NUMBA_CACHE_DIR", tempfile.mkdtemp(prefix="numba-codebook-"))
    spec = importlib.util.spec_from_file_location("hybrid_page_codebook", PAGE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=30)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)

    page = load_page()
    tables = page.get_compiler_tables(args.length)
    codes = {}
    for k in range(1, args.length):
        for phase in range(k):
            target = page.periodic_actions(args.length, k, phase)
            rate, dna_seed, seeds = page.compile_actions_to_dna(target, tables)
            code = page.SimulationTracer.encode(args.length, rate, dna_seed, seeds)
            if not (page.SimulationTracer(code).run() == target).all():
                raise SystemExit(f"code for k={k} phase={phase} does not decode back")
            codes[f"{k}:{phase}"] = code

    payload = {
        "length": args.length,
        "format": "key = 'k:phase' (act every k bars from bar phase) · value = DNA [len][rate][dna_seed][mutation_seeds]",
        "codes": codes,
    }
    OUTPUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(codes)} codes for length {args.length} to {OUTPUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
