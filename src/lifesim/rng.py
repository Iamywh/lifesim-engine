from __future__ import annotations

import random
from typing import NewType

Seed = NewType("Seed", int)


def create_rng(seed: int | Seed) -> random.Random:
    return random.Random(int(seed))
