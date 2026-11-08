"""Deterministic episode specs. Seed ranges never overlap across splits."""
import random

from .players import TRAIN_TYPES

# Frozen so test episodes stay identical across experiments (paired comparisons rely on it).
# Extra held-out types (e.g. 套话型) are evaluated separately via `ptypes=`.
TEST_TYPES = ["急躁型", "探索型", "闲聊型", "卡关型", "沉默型", "捣乱型"]

SINGLE = ["cipher_lock", "lever_order", "clock_time"]
RANGES = {"train": (0, 900_000), "val": (900_000, 950_000), "test": (1_000_000, 1_050_000)}


def spec(split: str, i: int, ptypes=None, memory: str = "full") -> dict:
    lo, hi = RANGES[split]
    rng = random.Random(f"{split}:{i}")
    seed = lo + (i * 7919) % (hi - lo)
    types = list(ptypes or (TRAIN_TYPES if split != "test" else TEST_TYPES))
    ptype = types[i % len(types)]
    extra = {"memory": "progressive"} if memory == "progressive" else {}
    if rng.random() < 0.25:
        return {"layout": "chain", "kinds": [rng.choice(SINGLE), "cipher_lock"], "seed": seed, "ptype": ptype,
                "max_turns": 24 if extra else 14, **extra}
    return {"layout": "single", "kinds": [SINGLE[(i // len(types)) % 3]], "seed": seed, "ptype": ptype,
            "max_turns": 16 if extra else 8, **extra}


def specs(split: str, n: int, ptypes=None, memory: str = "full") -> list:
    return [spec(split, i, ptypes, memory) for i in range(n)]
