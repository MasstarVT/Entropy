"""Core entropy accumulator.

Two steps, same shape as Cloudflare's LavaRand pipeline:

1. Absorb  — every chunk of frame bytes gets folded into a running SHA3-512
   hash, domain-separated with a tag/counter and mixed with a high-resolution
   timestamp. For live capture, frame-arrival jitter is itself a small
   physical entropy source, so mixing it in doesn't hurt.
2. Derive  — the accumulated hash state is never handed out directly. It's
   run through HKDF (RFC 5869: HMAC-based Extract-and-Expand) to whiten it
   and expand it to however many bytes you asked for. This is standard
   practice any time you turn "some unpredictable-ish bytes" into "a seed
   you actually trust."

IMPORTANT CAVEAT: HKDF will make the *output* of this pool look statistically
uniform even if the *input* had almost no real randomness in it (e.g. a
photo of a blank wall, or a static/frozen video). A hash that "looks random"
is not the same thing as an unpredictable secret — unpredictability has to
come from the source. Use `shannon_entropy_bits_per_byte` on your raw frames
as a sanity check, and prefer live, changing, physically noisy input (a
webcam pointed at something with motion/sensor noise) over a single static
photo if you actually need this for security-sensitive randomness. For real
security-critical randomness, use your OS's CSPRNG (`os.urandom` /
`secrets`) directly — this tool is for learning/experimenting with the
LavaRand idea, not a replacement for it.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time


class EntropyPool:
    def __init__(self, salt: bytes | None = None):
        self._hash = hashlib.sha3_512()
        self._absorbed = 0
        # Local secret salt mixed into the HKDF extract step. This means the
        # derived output can't be reproduced by someone who only has the
        # same input video/photo (e.g. downloaded off the wire) — they'd
        # also need this salt, which never leaves the machine that ran it
        # unless you choose to save/share it.
        self._salt = salt if salt is not None else os.urandom(32)

    @property
    def salt(self) -> bytes:
        return self._salt

    @property
    def frames_absorbed(self) -> int:
        return self._absorbed

    def absorb(self, data: bytes, *, tag: str = "frame") -> None:
        """Fold one chunk of bytes (e.g. one frame) into the pool."""
        meta = f"{tag}|{self._absorbed}|{time.perf_counter_ns()}".encode()
        self._hash.update(len(meta).to_bytes(4, "big"))
        self._hash.update(meta)
        self._hash.update(len(data).to_bytes(8, "big"))
        self._hash.update(data)
        self._absorbed += 1

    def state(self) -> bytes:
        """Raw accumulated hash digest. Prefer `derive()` for actual output."""
        return self._hash.digest()

    def derive(self, length: int, info: bytes = b"entropy-wall-v1") -> bytes:
        """HKDF-Extract-and-Expand the pool's state into `length` bytes."""
        if length < 0:
            raise ValueError(f"length must be non-negative, got {length}")
        # HKDF-Expand (RFC 5869 section 2.3) caps output at 255 * HashLen: the
        # expand loop below numbers each block with a single byte, so it
        # can't run past counter 255.
        max_length = 255 * hashlib.sha256().digest_size
        if length > max_length:
            raise ValueError(
                f"length={length} exceeds HKDF's maximum output for SHA-256 "
                f"({max_length} bytes)"
            )
        prk = hmac.new(self._salt, self.state(), hashlib.sha256).digest()
        out = b""
        prev = b""
        counter = 1
        while len(out) < length:
            prev = hmac.new(prk, prev + info + bytes([counter]), hashlib.sha256).digest()
            out += prev
            counter += 1
        return out[:length]
