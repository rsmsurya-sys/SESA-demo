"""Wavelet denoise, baseline removal, detrend, and 0-1 scaling."""

from __future__ import annotations

import numpy as np
import pywt
from scipy.signal import butter, detrend, sosfiltfilt


def _safe_wavedec_level(n: int, wavelet: str, requested: int) -> int:
    """Cap DWT depth so the signal is long enough for this wavelet."""
    w = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(n, w.dec_len)
    return int(max(1, min(requested, max_level)))


def wavelet_denoise(
    x: np.ndarray,
    wavelet: str = "db4",
    level: int = 4,
) -> np.ndarray:
    """Soft-threshold DWT coefficients (VisuShrink) along the last axis.

    Typical fingertip PPG: db4, level 4 on ~900 samples.
    """
    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    n = x.shape[0]
    lev = _safe_wavedec_level(n, wavelet, level)
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        coeffs = pywt.wavedec(x[:, c], wavelet, level=lev, mode="periodization")
        detail = coeffs[-1]
        sigma = np.median(np.abs(detail)) / 0.6745
        if not np.isfinite(sigma) or sigma < 1e-12:
            out[:, c] = x[:, c]
            continue
        uthresh = sigma * np.sqrt(2.0 * np.log(n))
        rec = [coeffs[0]]
        rec.extend(pywt.threshold(d, uthresh, mode="soft") for d in coeffs[1:])
        y = pywt.waverec(rec, wavelet, mode="periodization")
        out[:, c] = y[:n]
    return out[:, 0] if squeeze else out


def butter_filter(
    x: np.ndarray,
    fs: float,
    cutoff: float | tuple[float, float],
    btype: str,
    order: int = 3,
) -> np.ndarray:
    """Zero-phase Butterworth via SOS (stable on short PPG windows)."""
    x = np.asarray(x, dtype=np.float64)
    nyq = 0.5 * fs
    if isinstance(cutoff, tuple):
        lo, hi = cutoff
        wn = (max(lo, 1e-4) / nyq, min(hi, nyq * 0.99) / nyq)
        if wn[0] >= wn[1]:
            return x.copy()
    else:
        wn = float(cutoff) / nyq
        if wn <= 0.0 or wn >= 0.99:
            return x.copy()
    sos = butter(order, wn, btype=btype, output="sos")
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    # filtfilt needs enough samples (padlen ~ 3 * (2*order))
    if x.shape[0] < 24:
        return x[:, 0] if squeeze else x
    y = np.column_stack([sosfiltfilt(sos, x[:, i]) for i in range(x.shape[1])])
    return y[:, 0] if squeeze else y


def highpass(x: np.ndarray, fs: float, cutoff: float = 0.5, order: int = 3) -> np.ndarray:
    """Remove baseline wander below `cutoff` Hz (HR path)."""
    return butter_filter(x, fs, cutoff, btype="highpass", order=order)


def bandpass(
    x: np.ndarray,
    fs: float,
    low: float,
    high: float,
    order: int = 3,
) -> np.ndarray:
    return butter_filter(x, fs, (low, high), btype="bandpass", order=order)


def lowpass(x: np.ndarray, fs: float, cutoff: float, order: int = 3) -> np.ndarray:
    return butter_filter(x, fs, cutoff, btype="lowpass", order=order)


def minmax_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale each channel independently to [0, 1]."""
    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    lo = x.min(axis=0)
    hi = x.max(axis=0)
    y = (x - lo) / np.maximum(hi - lo, eps)
    return y[:, 0] if squeeze else y


def linear_detrend(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return detrend(x, type="linear")
    return np.column_stack([detrend(x[:, i], type="linear") for i in range(x.shape[1])])


def preprocess_hr_spo2(rgb: np.ndarray, fs: float) -> np.ndarray:
    """Clean RGB traces for pulsatile analysis (HR + AC of SpO2).

    Pipeline: wavelet → high-pass 0.5 Hz → linear detrend → min-max.
    Respiratory energy below 0.5 Hz is intentionally removed here.
    """
    y = wavelet_denoise(rgb, wavelet="db4", level=4)
    y = highpass(y, fs, cutoff=0.5)
    y = linear_detrend(y)
    y = minmax_normalize(y)
    return y


def preprocess_respiratory(green: np.ndarray, fs: float) -> np.ndarray:
    """Keep 0.1–0.5 Hz (6–30 breaths/min) on the green channel.

    Must run on a path that is *not* high-passed at 0.5 Hz.
    """
    y = wavelet_denoise(np.asarray(green, dtype=np.float64), wavelet="db4", level=4)
    y = linear_detrend(y)
    y = bandpass(y, fs, low=0.1, high=0.5, order=2)
    return y
