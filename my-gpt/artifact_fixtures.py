"""Small, real checkpoints that stand in for ignored historical artifacts.

Stages 46-50 authenticate a handful of large checkpoint files that are
intentionally kept out of Git. Ordinary test runs must not depend on those
files, so these helpers run each stage's own writer with a reduced config to
produce tiny but structurally complete endpoints. The tests that use them
re-pin the stage constants to the values of the fixture they just produced, so
the authentication path itself is still exercised.

The canonical artifacts are authenticated separately by the artifact-gated
tests, which skip when the files are absent. Nothing here changes a recorded
hash or a published result.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from data_utils import CharacterData, CharacterVocabulary
from training import TrainingResult


FIXTURE_WIDTH = 16
FIXTURE_BATCH_SIZE = 2
FIXTURE_BLOCK_SIZE = 24
FIXTURE_MAX_ITERS = 3


def character_data() -> CharacterData:
    """A tiny corpus shaped like the real one: 90/10 split, five characters."""

    vocabulary = CharacterVocabulary.from_text("abcde")
    tokens = torch.arange(2400) % vocabulary.size
    return CharacterData(vocabulary, tokens[:1800], tokens[1800:], 2400)


def small_config(stage: Any, dropout: float) -> Any:
    """The stage's locked config reduced to a fixture-sized run.

    The learning rate and schedule stay canonical, so the checkpoint validators
    still check the real AdamW settings and the real schedule positions.
    """

    return replace(
        stage.canonical_config(dropout),
        batch_size=FIXTURE_BATCH_SIZE,
        block_size=FIXTURE_BLOCK_SIZE,
        n_embd=FIXTURE_WIDTH,
        max_iters=FIXTURE_MAX_ITERS,
        eval_interval=1,
        eval_iters=1,
    )


def build_branch(
    stage: Any,
    directory: Path,
    key: str,
    dropout: float,
) -> tuple[CharacterData, Any, Any, TrainingResult]:
    """Run one fixture branch through the stage's own writer.

    The branch keeps the stage's canonical artifact names, so a loader's name
    check and path lookup run unchanged; only the directory and the size of the
    run differ. Returns the data, config, and spec the stage used, plus its
    report.
    """

    data = character_data()
    config = small_config(stage, dropout)
    spec = replace(
        next(
            candidate
            for candidate in stage.branch_specs(directory)
            if candidate.key == key
        ),
        width=FIXTURE_WIDTH,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        report = stage.run_branch(spec, data, config, torch.device("cpu"))
    return data, config, spec, report
