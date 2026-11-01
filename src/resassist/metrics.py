from __future__ import annotations

import random


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(values: list[float], iters: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval of the mean, so small differences are not oversold."""
    if not values:
        return float("nan"), float("nan")
    rng, n = random.Random(seed), len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def recall_at_k(ranked: list[int], gold: int, k: int) -> float:
    return 1.0 if gold in ranked[:k] else 0.0


def reciprocal_rank(ranked: list[int], gold: int, k: int = 10) -> float:
    for i, d in enumerate(ranked[:k], start=1):
        if d == gold:
            return 1.0 / i
    return 0.0
