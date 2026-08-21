#!/usr/bin/env python3
"""Example: turn an entropy_wall seed into an actual stream of random numbers.

entropy_wall.py only gives you seed *material* — raw, uniformly-whitened
bytes. It's not a random number generator by itself. This script shows two
common ways to turn that seed into something you can actually draw numbers
from:

1. `random.Random(seed)` — stdlib, no dependencies, good enough for
   simulations/games/anything non-security-sensitive. NOT cryptographically
   secure and NOT suitable for anything like generating passwords, tokens,
   or keys.
2. `numpy.random.Generator(numpy.random.PCG64(seed))` — same idea, if you're
   already using numpy and want vectorized draws (arrays of random numbers
   at once) or numpy's distributions (normal, binomial, etc).

Both are *deterministic*: the same seed always reproduces the same sequence
of numbers. That's normally what you want (reproducible experiments), but
it means the randomness quality is entirely inherited from how good the
seed was — see the "Important caveat" in the main README.

Usage
-----
    # From a hex string (what entropy_wall.py prints to stdout)
    python examples/use_seed.py e730999249e0bda3...

    # From a raw seed file (what entropy_wall.py --out writes)
    python examples/use_seed.py seed.bin

    # Pipe it straight through
    python entropy_wall.py image photo.jpg 2>/dev/null | python examples/use_seed.py -
"""

from __future__ import annotations

import os
import random
import sys


def load_seed(arg: str) -> bytes:
    """Accept a hex string, a path to a raw seed file, or '-' for stdin hex."""
    if arg == "-":
        arg = sys.stdin.read().strip()
    if os.path.isfile(arg):
        with open(arg, "rb") as f:
            return f.read()
    try:
        return bytes.fromhex(arg)
    except ValueError as e:
        raise SystemExit(
            f"'{arg}' isn't a valid hex seed and isn't a file that exists."
        ) from e


def demo_stdlib(seed: bytes) -> None:
    print("-- stdlib random.Random --")
    rng = random.Random(seed)  # accepts bytes directly
    print("5 random floats :", [round(rng.random(), 4) for _ in range(5)])
    print("dice roll (1-6) :", rng.randint(1, 6))
    print("shuffled deck   :", rng.sample(range(1, 14), 5), "(5 cards, ranks only)")


def demo_numpy(seed: bytes) -> None:
    try:
        import numpy as np
    except ImportError:
        print("\n(numpy not installed — skipping numpy example; `pip install numpy`)")
        return

    print("\n-- numpy Generator(PCG64) --")
    # PCG64 wants an integer seed; converting the seed bytes to a big int
    # keeps all of the entropy_wall output in play (not just the first 8
    # bytes the way int.from_bytes(seed[:8]) would).
    seed_int = int.from_bytes(seed, "big")
    rng = np.random.default_rng(seed_int)  # default_rng uses PCG64 internally
    print("5 random floats :", np.round(rng.random(5), 4))
    print("normal(0,1) x3  :", np.round(rng.normal(size=3), 4))
    print("random ints 0-99:", rng.integers(0, 100, size=5))


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)
    seed = load_seed(sys.argv[1])
    print(f"loaded {len(seed)}-byte seed: {seed.hex()}\n")
    demo_stdlib(seed)
    demo_numpy(seed)


if __name__ == "__main__":
    main()
