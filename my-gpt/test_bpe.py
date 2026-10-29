from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bpe import BPETokenizer, pre_tokenize, train_bpe

TEXT = "ROMEO.\nIs the day so young?\n\nBENVOLIO.\nBut new struck nine.\n\nROMEO.\nAy me, sad hours seem long.\n" * 20


class BPETest(unittest.TestCase):
    def test_pre_tokenize_keeps_every_character(self) -> None:
        self.assertEqual("".join(pre_tokenize(TEXT)), TEXT)
        self.assertIn(" young", pre_tokenize(TEXT))

    def test_round_trip(self) -> None:
        tok = train_bpe(TEXT, 80)
        self.assertEqual(tok.decode(tok.encode(TEXT)), TEXT)
        unseen = "ROMEO, sad nights seem young?"
        self.assertEqual(tok.decode(tok.encode(unseen)), unseen)
        with self.assertRaises(ValueError):
            tok.encode("zebra")

    def test_merges_shorten_the_text(self) -> None:
        tok = train_bpe(TEXT, 80)
        self.assertEqual(tok.vocab_size, 80)
        self.assertLess(len(tok.encode(TEXT)), len(TEXT) / 2)
        self.assertIn("ROMEO", tok.symbols)

    def test_first_merge_is_the_most_frequent_pair(self) -> None:
        text = "ab ab ab cd"
        tok = train_bpe(text, len(set(text)) + 1)
        self.assertEqual(tok.merges, [("a", "b")])

    def test_truncation_is_a_prefix(self) -> None:
        tok = train_bpe(TEXT, 80)
        small = tok.truncated(50)
        self.assertEqual(small.merges, tok.merges[: 50 - len(tok.base)])
        self.assertEqual(small.decode(small.encode(TEXT)), TEXT)
        self.assertGreater(len(small.encode(TEXT)), len(tok.encode(TEXT)))

    def test_deterministic(self) -> None:
        self.assertEqual(train_bpe(TEXT, 80).merges, train_bpe(TEXT, 80).merges)

    def test_save_load(self) -> None:
        tok = train_bpe(TEXT, 70)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bpe.json"
            tok.save(path)
            again = BPETokenizer.load(path)
        self.assertEqual(again.merges, tok.merges)
        self.assertEqual(again.encode(TEXT), tok.encode(TEXT))


if __name__ == "__main__":
    unittest.main()
