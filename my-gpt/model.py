from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingLanguageModel(nn.Module):
    """A token-local language model with learned token/position embeddings."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        *,
        use_position_embeddings: bool = True,
    ) -> None:
        super().__init__()

        if vocab_size <= 0 or block_size <= 0 or n_embd <= 0:
            raise ValueError(
                "vocab_size, block_size, and n_embd must all be positive"
            )

        self.block_size = block_size
        self.n_embd = n_embd
        self.use_position_embeddings = use_position_embeddings

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = (
            nn.Embedding(block_size, n_embd)
            if use_position_embeddings
            else None
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if idx.ndim != 2:
            raise ValueError(
                f"idx must have shape (B, T), got {tuple(idx.shape)}"
            )

        batch_size, sequence_length = idx.shape

        if sequence_length == 0:
            raise ValueError("idx must contain at least one token")

        if sequence_length > self.block_size:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds block_size "
                f"{self.block_size}"
            )

        if targets is not None and targets.shape != idx.shape:
            raise ValueError(
                "targets must have the same (B, T) shape as idx; "
                f"got {tuple(targets.shape)} and {tuple(idx.shape)}"
            )

        # (B, T) -> (B, T, C)
        token_embeddings = self.token_embedding_table(idx)
        x = token_embeddings

        if self.position_embedding_table is not None:
            # (T) -> (T, C), then broadcast over B.
            positions = torch.arange(sequence_length, device=idx.device)
            position_embeddings = self.position_embedding_table(positions)
            x = token_embeddings + position_embeddings

        # (B, T, C) -> (B, T, V)
        logits = self.lm_head(x)
        loss = None

        if targets is not None:
            _, _, vocab_size = logits.shape
            loss = F.cross_entropy(
                logits.reshape(batch_size * sequence_length, vocab_size),
                targets.reshape(batch_size * sequence_length),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        if idx.ndim != 2 or idx.shape[1] == 0:
            raise ValueError("idx must have non-empty shape (B, T)")

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        for _ in range(max_new_tokens):
            # Position embeddings only exist for 0 ... block_size - 1.
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            final_logits = logits[:, -1, :]
            probabilities = F.softmax(final_logits, dim=-1)
            idx_next = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
