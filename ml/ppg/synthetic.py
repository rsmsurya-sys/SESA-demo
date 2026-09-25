"""Synthetic fingertip PPG traces (and optional tiny MP4) for demos and tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def synthetic_rgb_ppg(
    duration_s: float = 30.0,
    fs: float = 30.0,
    hr_bpm: float = 72.0,
    rr_bpm: float = 16.0,
    spo2_target: float = 98.0,
    noise: float = 0.015,
    seed: int = 0,
) -> np.ndarray:
    """Build (N, 3) RGB means that mimic a covered-lens + torch capture.

    SpO2 is encoded via the red/green AC/DC ratio so 110 − 25R is near `spo2_target`.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    t = np.arange(n, dtype=np.float64) / fs
    hr_hz = hr_bpm / 60.0
    rr_hz = rr_bpm / 60.0

    # Two harmonics ≈ photoplethysmogram morphology.
    pulse = np.sin(2 * np.pi * hr_hz * t) + 0.25 * np.sin(4 * np.pi * hr_hz * t)
    resp = np.sin(2 * np.pi * rr_hz * t)

    # Target RoR R_target = (110 - SpO2) / 25
    r_target = (110.0 - spo2_target) / 25.0
    dc_r, dc_g, dc_b = 140.0, 110.0, 55.0
    ac_g = 6.0
    # R = (ACr/DCr) / (ACg/DCg)  => ACr = R * DCr * ACg / DCg
    ac_r = r_target * dc_r * ac_g / dc_g
    ac_b = 0.45 * ac_g

    red = dc_r + ac_r * pulse + 2.2 * resp
    green = dc_g + ac_g * pulse + 2.5 * resp
    blue = dc_b + ac_b * pulse + 1.0 * resp
    rgb = np.column_stack([red, green, blue])
    rgb += rng.normal(0.0, noise * 40.0, size=rgb.shape)
    return np.clip(rgb, 1.0, 255.0)


def write_synthetic_video(
    path: str | Path,
    rgb: np.ndarray | None = None,
    fs: float = 30.0,
    width: int = 160,
    height: int = 120,
) -> Path:
    """Write a small MP4 whose ROI mean tracks `rgb` (does not store 640x480 in RAM)."""
    import cv2

    if rgb is None:
        rgb = synthetic_rgb_ppg(fs=fs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fs, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    try:
        for r, g, b in rgb:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = (int(np.clip(b, 0, 255)), int(np.clip(g, 0, 255)), int(np.clip(r, 0, 255)))
            writer.write(frame)
    finally:
        writer.release()
    return path
