"""periodic_codebook.json: every code decodes to its periodic pattern.

Uses only numpy and the same decoding steps as SimulationTracer.run and the
bot's dna_engine.decode_dna: length-prefixed number stream
[len][rate][dna_seed][mutation_seeds...], base bits from PCG64(dna_seed),
then one flip mask per mutation seed (random < rate/100), first bit kept at 1.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

CODEBOOK = Path(__file__).resolve().parent.parent / "periodic_codebook.json"


def decode_numbers(text: str) -> list[int]:
    numbers, i = [], 0
    while i < len(text):
        width = int(text[i])
        numbers.append(int(text[i + 1:i + 1 + width]))
        i += 1 + width
    return numbers


def decode(code: str, length: int | None = None) -> np.ndarray:
    size, rate, seed, *mutation_seeds = decode_numbers(code)
    size = size if length is None else length
    bits = np.random.default_rng(seed).integers(0, 2, size=size).astype(np.int32)
    bits[0] = 1
    for mutation_seed in mutation_seeds:
        mask = np.random.default_rng(mutation_seed).random(size) < rate / 100.0
        bits[mask] = 1 - bits[mask]
        bits[0] = 1
    return bits


def periodic(length: int, k: int, phase: int) -> np.ndarray:
    bits = np.zeros(length, dtype=np.int32)
    bits[phase::k] = 1
    bits[0] = 1
    return bits


class PeriodicCodebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        data = json.loads(CODEBOOK.read_text(encoding="utf-8"))
        cls.length = int(data["length"])
        cls.codes = {tuple(int(v) for v in key.split(":")): code for key, code in data["codes"].items()}

    def test_every_k_and_phase_is_present(self) -> None:
        expected = {(k, phase) for k in range(1, self.length) for phase in range(k)}
        self.assertEqual(set(self.codes), expected)

    def test_codes_decode_to_their_pattern(self) -> None:
        for (k, phase), code in self.codes.items():
            self.assertTrue(np.array_equal(decode(code), periodic(self.length, k, phase)), (k, phase))

    def test_codes_work_for_shorter_windows(self) -> None:
        for (k, phase), code in self.codes.items():
            for length in range(2, self.length):
                self.assertTrue(np.array_equal(decode(code, length), periodic(length, k, phase)), (k, phase, length))

    def test_rates_are_read_the_same_by_every_decoder(self) -> None:
        # dna_engine reads rate <= 1 as a probability; the page reads it as percent.
        for code in self.codes.values():
            self.assertGreaterEqual(decode_numbers(code)[1], 2)


if __name__ == "__main__":
    unittest.main()
