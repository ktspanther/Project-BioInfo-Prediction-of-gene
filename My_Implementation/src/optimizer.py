"""
Shared result type and factory for all optimizers.

All three optimizers (PSO, DE, random) return the same OptimizerResult,
so the pipeline code doesn't need to care which one is used.
This also lets us swap them for ablation experiments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import numpy as np


@dataclass
class OptimizerResult:
    best_x: np.ndarray
    best_score: float
    history: list[float] = field(default_factory=list)
    n_evaluations: int = 0
    name: str = ""


OptimizerFn = Callable[..., OptimizerResult]


def get_optimizer(name: str) -> OptimizerFn:
    """Return the optimizer function by name: 'pso', 'de', or 'random'."""
    name = name.lower().strip()
    if name == "pso":
        from pso import pso_maximize_unified
        return pso_maximize_unified
    elif name == "de":
        from optimizer_de import de_maximize
        return de_maximize
    elif name == "random":
        from optimizer_random import random_maximize
        return random_maximize
    else:
        raise ValueError(f"Unknown optimizer '{name}'. Choose from: pso, de, random.")
