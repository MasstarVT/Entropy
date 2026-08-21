"""Turns a photo, a video file, or a live webcam feed into a stream of
frames (numpy arrays), plus an optional "noise extraction" step that
isolates the parts of a frame most likely to carry real physical randomness
(sensor noise / motion) instead of the deterministic scene content.
"""

from __future__ import annotations

import time
from typing import Iterator, Optional

import cv2
import numpy as np


def frames_from_image(path: str) -> Iterator[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    yield img


def frames_from_video(path: str, sample_every: int = 1) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % max(1, sample_every) == 0:
                yield frame
            idx += 1
    finally:
        cap.release()


def frames_from_webcam(
    device: int = 0,
    seconds: Optional[float] = 5.0,
    max_frames: Optional[int] = None,
) -> Iterator[np.ndarray]:
    """Yield frames from a live webcam.

    `seconds=None` with a `max_frames` cap (or Ctrl+C) runs indefinitely —
    used for the continuous --stream mode.
    """
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam device {device}. Is it in use by another "
            "app, or is the device index wrong?"
        )
    start = time.monotonic()
    count = 0
    try:
        while seconds is None or (time.monotonic() - start) < seconds:
            ok, frame = cap.read()
            if not ok:
                continue
            yield frame
            count += 1
            if max_frames is not None and count >= max_frames:
                break
    finally:
        cap.release()


def extract_noise(frame: np.ndarray, previous: Optional[np.ndarray]) -> np.ndarray:
    """Isolate the part of a frame most likely to hold real randomness.

    - If a previous frame is available (video/webcam), return the
      frame-to-frame difference: static scene content cancels out, leaving
      mostly sensor noise and motion.
    - Otherwise (a single still photo), return just the least-significant
      bit of each channel: the low bits of a photo are dominated by sensor
      noise and JPEG quantization artifacts rather than the actual image
      content.
    """
    if previous is not None and previous.shape == frame.shape:
        return cv2.absdiff(frame, previous)
    return (frame & 0x01) * 255
