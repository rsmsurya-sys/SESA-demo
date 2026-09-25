"""Extract per-frame mean RGB from a fingertip camera video.

Frames are never stacked in RAM: only three running lists of scalar means
are kept (~900 * 3 * 8 bytes). Center 50% of each frame is the ROI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class VideoExtractionError(ValueError):
    """Raised when a video cannot be opened or yields too few usable frames."""


def extract_rgb_ppg(
    video_path: str | Path,
    roi_fraction: float = 0.5,
    expected_fps: float = 30.0,
    min_seconds: float = 8.0,
    max_frames: Optional[int] = None,
) -> tuple[np.ndarray, float, dict]:
    """Return RGB mean traces, sampling rate, and diagnostics.

    Parameters
    ----------
    video_path
        MP4/AVI (or any OpenCV-readable) path.
    roi_fraction
        Fraction of width/height kept around the image center (0.5 = center 50%).
    expected_fps
        Fallback if the container reports 0 FPS.
    min_seconds
        Reject clips shorter than this (HR FFT needs several beats).
    max_frames
        Optional cap (e.g. 900 for a 30 s @ 30 FPS window).

    Returns
    -------
    rgb : (N, 3) float64 array in OpenCV RGB order after conversion from BGR.
    fs : sampling rate in Hz (FPS).
    info : dict with n_frames, roi, mean luminance, warnings.
    """
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoExtractionError(f"OpenCV could not open: {path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not np.isfinite(fps) or fps < 5.0 or fps > 120.0:
            fps = float(expected_fps)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        means: list[tuple[float, float, float]] = []
        dark_frames = 0
        empty_frames = 0
        n_read = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n_read += 1
            if frame is None or frame.size == 0:
                empty_frames += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            if h < 8 or w < 8:
                empty_frames += 1
                continue

            frac = float(np.clip(roi_fraction, 0.1, 1.0))
            rh, rw = int(h * frac), int(w * frac)
            y0 = (h - rh) // 2
            x0 = (w - rw) // 2
            roi = rgb[y0 : y0 + rh, x0 : x0 + rw]
            # Mean intensity per channel (R, G, B).
            r, g, b = roi.reshape(-1, 3).mean(axis=0)
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            if luminance < 12.0:
                dark_frames += 1
            means.append((float(r), float(g), float(b)))

            if max_frames is not None and len(means) >= max_frames:
                break
    finally:
        cap.release()

    if len(means) < int(min_seconds * fps):
        raise VideoExtractionError(
            f"Need at least {min_seconds:.1f}s of video at {fps:.1f} FPS "
            f"({int(min_seconds * fps)} frames); got {len(means)} usable frames "
            f"(read {n_read}). Check that the finger covers the lens."
        )

    rgb_arr = np.asarray(means, dtype=np.float64)
    warnings: list[str] = []
    if dark_frames > 0.3 * len(means):
        warnings.append(
            "Many dark frames — enable the torch and cover the rear camera fully."
        )
    if empty_frames:
        warnings.append(f"Skipped {empty_frames} empty frames.")

    info = {
        "n_frames": int(rgb_arr.shape[0]),
        "fps": float(fps),
        "width": width,
        "height": height,
        "roi_fraction": frac,
        "mean_luminance": float(
            (0.2126 * rgb_arr[:, 0] + 0.7152 * rgb_arr[:, 1] + 0.0722 * rgb_arr[:, 2]).mean()
        ),
        "dark_frame_fraction": float(dark_frames / max(len(means), 1)),
        "warnings": warnings,
        "path": str(path),
    }
    return rgb_arr, float(fps), info
