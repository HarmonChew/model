from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """All tunable values for the Stage 1 experiment."""

    batch_size: int = 32
    block_size: int = 8
    n_embd: int = 32
    learning_rate: float = 1e-2
    max_iters: int = 3_000
    eval_interval: int = 300
    eval_iters: int = 100
    seed: int = 1_337

    def __post_init__(self) -> None:
        positive_values = {
            "batch_size": self.batch_size,
            "block_size": self.block_size,
            "n_embd": self.n_embd,
            "learning_rate": self.learning_rate,
            "eval_interval": self.eval_interval,
            "eval_iters": self.eval_iters,
        }

        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        if self.max_iters < 0:
            raise ValueError(
                f"max_iters must be non-negative, got {self.max_iters}"
            )
