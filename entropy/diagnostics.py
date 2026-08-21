"""Small sanity-check helpers — not part of the actual hashing pipeline."""

from __future__ import annotations

import numpy as np


def shannon_entropy_bits_per_byte(data: bytes) -> float:
    """Estimate how 'random-looking' raw bytes are, on a 0.0-8.0 scale.

    This is a rough diagnostic, not a security guarantee: a photo of pure
    static will score near 8.0, a photo of a blank white wall will score
    near 0.0, and most ordinary photos/video land somewhere in the middle
    because pixel values are spatially correlated. Use it to sanity-check
    that your input source actually has some texture/noise/motion in it
    before trusting it as an entropy source.
    """
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = counts[counts > 0] / len(data)
    return float(-(probs * np.log2(probs)).sum())
