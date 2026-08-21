"""Shared "absorb a stream of frames into a pool" helper.

Used by both the CLI (entropy_wall.py) and the GUI (gui.py) so the actual
absorb/derive logic only lives in one place.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Optional, Tuple

import numpy as np

from .capture import extract_noise
from .pool import EntropyPool


def run_pool(
    frames: Iterable[np.ndarray], *, noise_extract: bool = False, tag: str = "frame"
) -> Tuple[EntropyPool, bytes]:
    """Absorb a stream of frames into a fresh EntropyPool.

    Returns (pool, sample) where `sample` is the bytes of the last
    processed frame (or noise-extracted payload), kept around only for the
    Shannon-entropy diagnostic.
    """
    pool = EntropyPool()
    previous: Optional[np.ndarray] = None
    sample = b""
    count = 0
    for i, frame in enumerate(frames):
        payload = extract_noise(frame, previous) if noise_extract else frame
        previous = frame
        data = payload.tobytes()
        pool.absorb(data, tag=f"{tag}:{i}")
        sample = data
        count += 1
    if count == 0:
        raise RuntimeError("No frames were captured — nothing to hash.")
    return pool, sample


def stream_pool(
    frames: Iterable[np.ndarray], *, noise_extract: bool = False, tag: str = "frame"
) -> Iterator[Tuple[EntropyPool, np.ndarray]]:
    """Absorb frames into one growing EntropyPool, yielding as each arrives.

    Unlike `run_pool`, this is for indefinite/interactive capture (the CLI's
    `--stream` mode, the GUI's Stream mode): it never returns on its own —
    the caller drives it (for a `frames` iterator that ends on Ctrl+C, a
    `--seconds`/stop-event cutoff, etc.) and decides when to `pool.derive()`
    and emit a seed, e.g. on an interval. Yields `(pool, frame)` after every
    absorbed frame so callers can also use `frame` for a live preview.
    """
    pool = EntropyPool()
    previous: Optional[np.ndarray] = None
    for i, frame in enumerate(frames):
        payload = extract_noise(frame, previous) if noise_extract else frame
        previous = frame
        pool.absorb(payload.tobytes(), tag=f"{tag}:{i}")
        yield pool, frame
