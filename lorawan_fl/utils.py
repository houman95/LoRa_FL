"""Small deterministic-run and fairness utilities."""

from __future__ import annotations

import random
from typing import List

import numpy as np
import torch

def set_all_seeds(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def gini(x: List[int]) -> float:
    x = np.asarray(x, dtype=float)
    if np.all(x == 0): return 0.0
    x = np.sort(x); n = len(x)
    return (2 * np.sum(np.arange(1, n + 1) * x) / (n * np.sum(x))) - (n + 1) / n

# ---------------------------------------------------------------------------
# Phase 1 — Downlink
