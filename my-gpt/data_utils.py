from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "input.txt"


@dataclass(frozen=True, slots=True)
class CharacterVocabulary:
    chars: tuple[str, ...]
    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> CharacterVocabulary:
        chars = tuple(sorted(set(text)))

        if not chars:
            raise ValueError("Cannot build a vocabulary from empty text")

        stoi = {char: index for index, char in enumerate(chars)}
        itos = {index: char for char, index in stoi.items()}
        return cls(chars=chars, stoi=stoi, itos=itos)

    @property
    def size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[char] for char in text]

    def decode(self, token_ids: Iterable[int]) -> str:
        return "".join(self.itos[int(token_id)] for token_id in token_ids)


@dataclass(frozen=True, slots=True)
class CharacterData:
    vocabulary: CharacterVocabulary
    train_data: torch.Tensor
    val_data: torch.Tensor
    num_characters: int

    @classmethod
    def from_file(
        cls,
        path: str | Path = DEFAULT_DATA_PATH,
        train_fraction: float = 0.9,
    ) -> CharacterData:
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")

        text = Path(path).read_text(encoding="utf-8")
        vocabulary = CharacterVocabulary.from_text(text)
        data = torch.tensor(vocabulary.encode(text), dtype=torch.long)
        split_index = int(train_fraction * len(data))

        return cls(
            vocabulary=vocabulary,
            train_data=data[:split_index],
            val_data=data[split_index:],
            num_characters=len(text),
        )

    def get_batch(
        self,
        split: str,
        *,
        batch_size: int,
        block_size: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if split == "train":
            source = self.train_data
        elif split == "val":
            source = self.val_data
        else:
            raise ValueError(
                f"split must be 'train' or 'val', got {split!r}"
            )

        if len(source) <= block_size:
            raise ValueError(
                f"The {split} split has {len(source)} tokens, but block_size "
                f"is {block_size}"
            )

        starts = torch.randint(
            0,
            len(source) - block_size,
            (batch_size,),
            generator=generator,
        )
        offsets = torch.arange(block_size)
        indices = starts[:, None] + offsets[None, :]

        inputs = source[indices]
        targets = source[indices + 1]
        return inputs.to(device), targets.to(device)
