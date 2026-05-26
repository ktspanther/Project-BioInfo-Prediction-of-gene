"""
Shared optimizer interface.

Both PSO and Differential Evolution return the same `OptimizerResult` type,
so the rest of the pipeline doesn't care which one is used. A small factory
function `get_optimizer(name)` returns the function to call.

This lets us answer the question:
    "Is PSO essential to PatternChrome's success, or does any reasonable
     gradient-free optimizer work equally well?"

by swapping `name="pso"` for `name="de"` while keeping everything else
identical (same data, same XGBoost settings, same backward elimination,
same hyperparameter tuning).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import numpy as np


@dataclass
class OptimizerResult:
    """Unified result type for any optimizer wrapper."""
    best_x: np.ndarray
    best_score: float
    history: list[float] = field(default_factory=list)
    n_evaluations: int = 0          # how many times `fn` was called
    name: str = ""                  # which optimizer produced this result


# Type alias: every optimizer in this codebase has this signature.
OptimizerFn = Callable[..., OptimizerResult]


def get_optimizer(name: str) -> OptimizerFn:
    """Factory: return the optimizer function by name.

    Supported names:
        "pso" — Particle Swarm Optimization (the paper's choice)
        "de"  — Differential Evolution     (our alternative)
        "random" — uniform random search   (sanity-check baseline)
    """
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
        raise ValueError(
            f"Unknown optimizer '{name}'. "
            f"Choose from: pso, de, random."
        )
