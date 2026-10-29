"""A from-scratch byte-pair-encoding (BPE) tokenizer over characters.

Training follows GPT-2's recipe, minus the byte layer:

1. Split the text into pre-tokens with a regex (words with their leading space,
   numbers, punctuation runs, whitespace), so merges never cross word boundaries.
2. Start with one symbol per character.
3. Repeatedly merge the most frequent adjacent pair of symbols into a new symbol.

Pair counts are updated incrementally, touching only the words that contain the
merged pair, so thousands of merges train in seconds instead of hours.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# GPT-2's pre-tokenizer, with ASCII letter classes because Python's re has no \p{L}.
PRE_TOKEN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
)


def pre_tokenize(text: str) -> list[str]:
    pieces = PRE_TOKEN.findall(text)
    if "".join(pieces) != text:
        raise ValueError("pre-tokenizer dropped characters")
    return pieces


@dataclass
class BPETokenizer:
    base: list[str]
    merges: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.symbols = list(self.base) + [a + b for a, b in self.merges]
        self.stoi = {s: i for i, s in enumerate(self.symbols)}
        if len(self.stoi) != len(self.symbols):
            raise ValueError("duplicate symbols")
        self.rank = {pair: i for i, pair in enumerate(self.merges)}
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.symbols)

    def truncated(self, vocab_size: int) -> BPETokenizer:
        """The tokenizer after its first ``vocab_size - len(base)`` merges."""

        n = vocab_size - len(self.base)
        if not 0 <= n <= len(self.merges):
            raise ValueError(f"vocab_size {vocab_size} outside [{len(self.base)}, {self.vocab_size}]")
        return BPETokenizer(self.base, self.merges[:n])

    def _encode_piece(self, piece: str) -> list[int]:
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        parts = list(piece)
        while len(parts) > 1:
            ranked = [(self.rank.get((parts[i], parts[i + 1])), i) for i in range(len(parts) - 1)]
            ranked = [(r, i) for r, i in ranked if r is not None]
            if not ranked:
                break
            best = min(r for r, _ in ranked)
            pair = self.merges[best]
            merged, i = [], 0
            while i < len(parts):
                if i + 1 < len(parts) and (parts[i], parts[i + 1]) == pair:
                    merged.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        unknown = [p for p in parts if p not in self.stoi]
        if unknown:
            raise ValueError(f"characters outside the base vocabulary: {unknown}")
        ids = [self.stoi[p] for p in parts]
        self._cache[piece] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in pre_tokenize(text):
            ids.extend(self._encode_piece(piece))
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.symbols[i] for i in ids)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"base": self.base, "merges": self.merges}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> BPETokenizer:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(raw["base"], [tuple(m) for m in raw["merges"]])


def train_bpe(text: str, vocab_size: int, base: list[str] | None = None) -> BPETokenizer:
    """Learn merges until the vocabulary holds ``vocab_size`` symbols.

    Ties in pair frequency are broken by the pair itself, so training is deterministic.
    """

    base = sorted(set(text)) if base is None else list(base)
    word_counts = Counter(pre_tokenize(text))
    words = [list(w) for w in word_counts]
    freqs = list(word_counts.values())

    pair_counts: Counter[tuple[str, str]] = Counter()
    where: defaultdict[tuple[str, str], set[int]] = defaultdict(set)
    for wi, (word, freq) in enumerate(zip(words, freqs)):
        for pair in zip(word, word[1:]):
            pair_counts[pair] += freq
            where[pair].add(wi)

    merges: list[tuple[str, str]] = []
    while len(base) + len(merges) < vocab_size:
        if not pair_counts:
            break
        pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        if pair_counts[pair] < 2:
            break
        merges.append(pair)
        joined = pair[0] + pair[1]
        for wi in list(where[pair]):
            word, freq = words[wi], freqs[wi]
            for old in zip(word, word[1:]):
                pair_counts[old] -= freq
                if pair_counts[old] <= 0:
                    del pair_counts[old]
                where[old].discard(wi)
            new, i = [], 0
            while i < len(word):
                if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
                    new.append(joined)
                    i += 2
                else:
                    new.append(word[i])
                    i += 1
            words[wi] = new
            for p in zip(new, new[1:]):
                pair_counts[p] += freq
                where[p].add(wi)
        where.pop(pair, None)
    return BPETokenizer(base, merges)
