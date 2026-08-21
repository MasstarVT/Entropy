"""entropy_wall — turn photos, videos, or a webcam feed into a random seed.

Inspired by Cloudflare's "Wall of Entropy" (LavaRand): point a camera at
something chaotic, hash the pixel data, and use the hash as unpredictable
seed material for a CSPRNG. This package does the same thing on a single
machine: absorb frame bytes into a running hash, then whiten/expand the
result with HKDF so the output is a clean, uniformly-distributed seed.
"""

from .pool import EntropyPool
from .diagnostics import shannon_entropy_bits_per_byte
from .pipeline import run_pool, stream_pool

__all__ = ["EntropyPool", "shannon_entropy_bits_per_byte", "run_pool", "stream_pool"]
