from __future__ import annotations

import hashlib
import random
from typing import NewType

Seed = NewType("Seed", int)


def create_rng(seed: int | Seed) -> random.Random:
    return random.Random(int(seed))


def derive_stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")
