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
