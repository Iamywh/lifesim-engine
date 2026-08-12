from __future__ import annotations

from typing import NewType
import random

Seed = NewType("Seed", int)


def create_rng(seed: int | Seed) -> random.Random:
    return random.Random(int(seed))
