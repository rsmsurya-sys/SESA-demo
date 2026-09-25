"""Respiratory rate from PPG baseline modulation (0.1–0.5 Hz)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import find_peaks, get_window

from ml.ppg.heart_rate import _parabolic_peak
from ml.ppg.preprocess import preprocess_respiratory


@dataclass
class RespiratoryEstimate:
    breaths_per_min: float
    dominant_hz: Optional[float]
    component: np.ndarray
    method_used: str


def estimate_respiratory_rate(
    green_raw: np.ndarray,
    fs: float,
    rr_lo: float = 12.0,
    rr_hi: float = 30.0,
) -> RespiratoryEstimate:
    """FFT (primary) + time-domain peaks (backup) on the respiratory PPG band.

    Search internally 0.1–0.5 Hz (6–30 br/min) then clip to the UI range 12–30.
    """
    resp = preprocess_respiratory(green_raw, fs)
    x = resp - np.mean(resp)
    n = x.size
    fmin, fmax = 0.1, 0.5

    bpm_fft: Optional[float] = None
    hz: Optional[float] = None
    if n >= int(fs * 10):
        window = get_window("hann", n, fftbins=True)
        mag = np.abs(np.fft.rfft(x * window))
        freq = np.fft.rfftfreq(n, d=1.0 / fs)
        band = (freq >= fmin) & (freq <= fmax)
        mag_b = mag.copy()
        mag_b[~band] = 0.0
        if np.any(band) and mag_b.max() > 0:
            k = int(np.argmax(mag_b))
            hz = _parabolic_peak(freq, mag, k)
            if fmin <= hz <= fmax:
                bpm_fft = float(hz * 60.0)

    min_dist = max(1, int(fs * 60.0 / 40.0))
    peaks, _ = find_peaks(x, distance=min_dist)
    bpm_peaks: Optional[float] = None
    if len(peaks) >= 3:
        ibi = np.diff(peaks) / fs
        ibi = ibi[(ibi >= 2.0) & (ibi <= 10.0)]  # 6–30 br/min
        if ibi.size >= 2:
            bpm_peaks = float(60.0 / np.mean(ibi))

    method = "none"
    bpm = float("nan")
    if bpm_fft is not None and bpm_peaks is not None and abs(bpm_fft - bpm_peaks) <= 6.0:
        bpm, method = 0.5 * (bpm_fft + bpm_peaks), "fft+peaks"
    elif bpm_fft is not None:
        bpm, method = bpm_fft, "fft"
    elif bpm_peaks is not None:
        bpm, method = bpm_peaks, "peaks"

    if np.isfinite(bpm):
        bpm = float(np.clip(bpm, rr_lo, rr_hi))

    return RespiratoryEstimate(
        breaths_per_min=bpm,
        dominant_hz=hz,
        component=resp,
        method_used=method,
    )
