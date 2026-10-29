"""Stage 60: a from-scratch BPE tokenizer for Complete Shakespeare.

Every model so far reads one character at a time. Production GPTs read tokens:
sub-word pieces learned by byte-pair encoding. This stage trains a BPE tokenizer
(bpe.py) on the Complete Shakespeare training split only, then measures on the
validation split, at several vocabulary sizes:

- compression: characters per token (how much shorter the sequence becomes),
- exactness: decode(encode(val)) must equal val,
- rarity: how many tokens appear fewer than 10 times in the training text, since a
  4.8M-character corpus may not teach rare tokens well.

No model is trained here; Stage 61 trains the GPT on these tokens.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from bpe import BPETokenizer, train_bpe
from data_utils import CharacterData


def _load(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_STAGE_41 = _load("stage_41_for_stage_60", "41_final_compute_extension.py")

EXPERIMENT_STAGE = 60
CHECKPOINTS = Path(__file__).with_name("checkpoints")
TOKENIZER_PATH = CHECKPOINTS / "stage_60_bpe_4096.json"
REPORT_PATH = CHECKPOINTS / "stage_60_report.json"
MAX_VOCAB = 4096
VOCAB_SIZES = (256, 512, 1024, 2048, 4096)
RARE_BELOW = 10


def split_text(data: CharacterData) -> tuple[str, str]:
    decode = data.vocabulary.decode
    return decode(data.train_data.tolist()), decode(data.val_data.tolist())


def measure(tok: BPETokenizer, train: str, val: str) -> dict[str, Any]:
    val_ids = tok.encode(val)
    if tok.decode(val_ids) != val:
        raise AssertionError(f"vocab {tok.vocab_size}: validation round trip failed")
    train_counts = Counter(tok.encode(train))
    return {
        "vocab_size": tok.vocab_size,
        "val_tokens": len(val_ids),
        "chars_per_token": len(val) / len(val_ids),
        "train_tokens": sum(train_counts.values()),
        "unused_in_train": tok.vocab_size - len(train_counts),
        "rare_in_train": sum(1 for i in range(tok.vocab_size) if train_counts[i] < RARE_BELOW),
        "longest": sorted(tok.symbols, key=len, reverse=True)[:5],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    data = CharacterData.from_file(_STAGE_41.DEFAULT_COMPLETE_DATA_PATH)
    train, val = split_text(data)
    print(f"Stage {EXPERIMENT_STAGE}: BPE on Complete Shakespeare | train {len(train):,} chars | val {len(val):,} chars")

    started = time.perf_counter()
    full = train_bpe(train, MAX_VOCAB, base=list(data.vocabulary.chars))
    seconds = time.perf_counter() - started
    print(f"Trained {len(full.merges):,} merges over {len(full.base)} base characters in {seconds:.1f} s")
    CHECKPOINTS.mkdir(exist_ok=True)
    full.save(TOKENIZER_PATH)

    print("\nFirst 20 merges:")
    print("  " + " | ".join(repr(a + b) for a, b in full.merges[:20]))

    rows = [measure(full.truncated(len(full.base)), train, val)]
    rows += [measure(full.truncated(v), train, val) for v in VOCAB_SIZES]
    print(f"\nValidation split ({len(val):,} characters), round trip exact at every size:")
    print(f"{'vocab':>6s} {'val tokens':>11s} {'chars/token':>12s} {'unused':>7s} {f'<{RARE_BELOW} in train':>12s}  longest tokens")
    for r in rows:
        print(
            f"{r['vocab_size']:6d} {r['val_tokens']:11,d} {r['chars_per_token']:12.3f} {r['unused_in_train']:7d} "
            f"{r['rare_in_train']:12d}  {', '.join(repr(s) for s in r['longest'][:3])}"
        )

    sample = val[:200]
    tok = full.truncated(1024)
    pieces = [tok.symbols[i] for i in tok.encode(sample)]
    print("\nThe first 200 validation characters at vocab 1024, tokens separated by |:")
    print("|".join(pieces))

    report = {
        "stage": EXPERIMENT_STAGE,
        "train_characters": len(train),
        "val_characters": len(val),
        "base_characters": len(full.base),
        "training_seconds": seconds,
        "tokenizer": str(TOKENIZER_PATH.name),
        "rows": rows,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nTokenizer: my-gpt/checkpoints/{TOKENIZER_PATH.name}")
    print(f"Report: my-gpt/checkpoints/{REPORT_PATH.name}")
    return report


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()


if __name__ == "__main__":
    run(parse_args())
