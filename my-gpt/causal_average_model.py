from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalAverageLanguageModel(nn.Module):
    """A character model with fixed, uniform causal context mixing.

    Unlike ``EmbeddingLanguageModel``, each position receives the average of
    its own representation and all representations to its left. The mixing
    weights are fixed; queries, keys, and values are deliberately left for the
    next experiment.
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
    ) -> None:
        super().__init__()

        if vocab_size <= 0 or block_size <= 0 or n_embd <= 0:
            raise ValueError(
                "vocab_size, block_size, and n_embd must all be positive"
            )

        self.block_size = block_size
        self.n_embd = n_embd

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # A buffer follows the model between devices and is saved in its state
        # dict, but is not optimized. Row t permits communication from source
        # positions 0 through t and blocks all future source positions.
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size)),
        )

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
        positions = torch.arange(sequence_length, device=idx.device)
        position_embeddings = self.position_embedding_table(positions)
        x = token_embeddings + position_embeddings

        # (T, T): each row becomes a uniform distribution over the current
        # position and its prefix. The matrix contains no learned values.
        weights = self.tril[:sequence_length, :sequence_length]
        weights = weights / weights.sum(dim=1, keepdim=True)

        # (T, T) @ (B, T, C) -> (B, T, C). Matrix multiplication combines
        # the sequence dimension while independently preserving B and C.
        x_context = weights @ x

        # (B, T, C) -> (B, T, V)
        logits = self.lm_head(x_context)
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
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            final_logits = logits[:, -1, :]
            probabilities = F.softmax(final_logits, dim=-1)
            idx_next = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
