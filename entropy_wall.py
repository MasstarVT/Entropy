#!/usr/bin/env python3
"""entropy_wall — a tiny LavaRand-style entropy tool.

Point it at a photo, a video file, or your webcam, and it hashes the pixel
data into a random seed, the same basic idea as Cloudflare's Wall of Entropy
(they point cameras at lava lamps; this points a camera at whatever you've
got).

Examples
--------
    python entropy_wall.py image photo.jpg
    python entropy_wall.py video clip.mp4 --sample-every 3 --length 64
    python entropy_wall.py webcam --seconds 5
    python entropy_wall.py webcam --stream --interval 2

See entropy/pool.py for how absorb+derive works, and the security caveat
about "looks random" vs. "is unpredictable."
"""

from __future__ import annotations

import argparse
import sys
import time

from entropy import EntropyPool, run_pool, shannon_entropy_bits_per_byte, stream_pool
from entropy.capture import frames_from_image, frames_from_video, frames_from_webcam


def _emit(pool: EntropyPool, sample: bytes, args: argparse.Namespace) -> None:
    seed = pool.derive(args.length)
    print(f"frames absorbed : {pool.frames_absorbed}", file=sys.stderr)
    print(
        f"source entropy  : ~{shannon_entropy_bits_per_byte(sample):.2f} bits/byte "
        "(rough diagnostic, see entropy/diagnostics.py)",
        file=sys.stderr,
    )
    print(f"seed ({args.length} bytes, hex):", file=sys.stderr)
    print(seed.hex())
    if args.out:
        with open(args.out, "wb") as f:
            f.write(seed)
        print(f"also wrote raw bytes to {args.out}", file=sys.stderr)


def cmd_image(args: argparse.Namespace) -> None:
    frames = frames_from_image(args.path)
    pool, sample = run_pool(frames, noise_extract=args.noise_extract, tag="image")
    _emit(pool, sample, args)


def cmd_video(args: argparse.Namespace) -> None:
    frames = frames_from_video(args.path, sample_every=args.sample_every)
    pool, sample = run_pool(frames, noise_extract=args.noise_extract, tag="video")
    _emit(pool, sample, args)


def cmd_webcam(args: argparse.Namespace) -> None:
    if not args.stream:
        frames = frames_from_webcam(device=args.device, seconds=args.seconds)
        pool, sample = run_pool(frames, noise_extract=args.noise_extract, tag="webcam")
        _emit(pool, sample, args)
        return

    # Continuous mode: keep one pool growing forever (each new frame adds
    # more entropy on top of everything absorbed so far — a ratchet, like
    # Cloudflare's actual wall which never resets), and print a fresh
    # derived seed every --interval seconds until Ctrl+C.
    print(
        f"streaming from device {args.device}, emitting a seed every "
        f"{args.interval}s (Ctrl+C to stop)...",
        file=sys.stderr,
    )
    last_emit = time.monotonic()
    try:
        frames = frames_from_webcam(device=args.device, seconds=None)
        for pool, _frame in stream_pool(frames, noise_extract=args.noise_extract, tag="webcam"):
            now = time.monotonic()
            if now - last_emit >= args.interval:
                seed = pool.derive(args.length)
                print(f"[{pool.frames_absorbed} frames] {seed.hex()}")
                last_emit = now
    except KeyboardInterrupt:
        print("\nstopped.", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="entropy_wall",
        description="Hash a photo/video/webcam feed into a random seed, LavaRand-style.",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-l", "--length", type=int, default=32, help="bytes of seed to derive (default: 32)"
    )
    common.add_argument("-o", "--out", help="also write the raw seed bytes to this file")
    common.add_argument(
        "--noise-extract",
        action="store_true",
        help="hash frame-to-frame differences / pixel LSBs instead of raw "
        "frames, to weight the source toward actual sensor noise",
    )

    sub = p.add_subparsers(dest="command", required=True)

    p_img = sub.add_parser("image", parents=[common], help="hash a single photo")
    p_img.add_argument("path")
    p_img.set_defaults(func=cmd_image)

    p_vid = sub.add_parser("video", parents=[common], help="hash frames from a video file")
    p_vid.add_argument("path")
    p_vid.add_argument(
        "--sample-every", type=int, default=1, help="use every Nth frame (default: 1, all frames)"
    )
    p_vid.set_defaults(func=cmd_video)

    p_cam = sub.add_parser("webcam", parents=[common], help="hash frames from a live webcam")
    p_cam.add_argument("--device", type=int, default=0, help="webcam device index (default: 0)")
    p_cam.add_argument(
        "--seconds", type=float, default=5.0, help="how long to capture (default: 5s)"
    )
    p_cam.add_argument(
        "--stream",
        action="store_true",
        help="run indefinitely, printing a new derived seed every --interval seconds",
    )
    p_cam.add_argument(
        "--interval", type=float, default=2.0, help="seconds between emitted seeds in --stream mode"
    )
    p_cam.set_defaults(func=cmd_webcam)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
