"""Heart-rate from green-channel PPG: peaks, FFT, optional 1D-CNN+LSTM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import find_peaks, get_window, welch

from ml.ppg.preprocess import bandpass


@dataclass
class HeartRateEstimate:
    bpm: float
    peak_bpm: Optional[float]
    fft_bpm: Optional[float]
    dl_bpm: Optional[float]
    ibi_seconds: np.ndarray
    peak_indices: np.ndarray
    dominant_hz: Optional[float]
    method_used: str


def _parabolic_peak(freq: np.ndarray, mag: np.ndarray, k: int) -> float:
    """Sub-bin frequency estimate around bin k."""
    if k <= 0 or k >= len(mag) - 1:
        return float(freq[k])
    y0, y1, y2 = mag[k - 1], mag[k], mag[k + 1]
    denom = y0 - 2 * y1 + y2
    if abs(denom) < 1e-18:
        return float(freq[k])
    delta = 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    df = freq[1] - freq[0]
    return float(freq[k] + delta * df)


def hr_from_peaks(
    green: np.ndarray,
    fs: float,
    bpm_min: float = 48.0,
    bpm_max: float = 180.0,
) -> tuple[Optional[float], np.ndarray, np.ndarray]:
    """IBI heart rate from green PPG. `green` should already be pulsatile-cleaned."""
    x = np.asarray(green, dtype=np.float64)
    x = x - np.mean(x)
    min_dist = max(1, int(fs * 60.0 / bpm_max))
    prominence = 0.15 * (np.max(x) - np.min(x) + 1e-9)
    peaks, _ = find_peaks(x, distance=min_dist, prominence=prominence)
    if len(peaks) < 3:
        # Relax prominence once for weak perfusion.
        peaks, _ = find_peaks(x, distance=min_dist, prominence=prominence * 0.4)
    if len(peaks) < 3:
        return None, np.array([]), peaks.astype(int)

    ibi = np.diff(peaks.astype(np.float64)) / fs
    lo, hi = 60.0 / bpm_max, 60.0 / bpm_min
    ibi = ibi[(ibi >= lo) & (ibi <= hi)]
    if ibi.size < 2:
        return None, ibi, peaks.astype(int)
    # Reject IBI outliers (motion extra/missed beats).
    med = np.median(ibi)
    ibi = ibi[np.abs(ibi - med) <= 0.35 * med]
    if ibi.size < 2:
        return None, ibi, peaks.astype(int)
    bpm = float(60.0 / np.mean(ibi))
    if not (bpm_min <= bpm <= bpm_max):
        return None, ibi, peaks.astype(int)
    return bpm, ibi, peaks.astype(int)


def hr_from_fft(
    green: np.ndarray,
    fs: float,
    fmin: float = 0.8,
    fmax: float = 3.0,
) -> tuple[Optional[float], Optional[float], np.ndarray, np.ndarray]:
    """Dominant frequency in 0.8–3.0 Hz (48–180 BPM). Returns bpm, hz, freqs, psd."""
    x = np.asarray(green, dtype=np.float64)
    x = bandpass(x, fs, low=fmin, high=min(fmax, 0.45 * fs), order=2)
    x = x - np.mean(x)
    n = x.size
    if n < int(fs * 5):
        return None, None, np.array([]), np.array([])

    window = get_window("hann", n, fftbins=True)
    spec = np.fft.rfft(x * window)
    mag = np.abs(spec)
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    band = (freq >= fmin) & (freq <= fmax)
    if not np.any(band):
        return None, None, freq, mag
    mag_b = mag.copy()
    mag_b[~band] = 0.0
    k = int(np.argmax(mag_b))
    hz = _parabolic_peak(freq, mag, k)
    if hz < fmin or hz > fmax:
        return None, None, freq, mag
    return float(hz * 60.0), float(hz), freq, mag


def hr_from_welch(green: np.ndarray, fs: float, fmin: float = 0.8, fmax: float = 3.0) -> Optional[float]:
    """Backup PSD peak (Welch) when a single FFT is peaky from window length."""
    x = np.asarray(green, dtype=np.float64)
    nperseg = min(x.size, max(64, int(fs * 8)))
    f, pxx = welch(x, fs=fs, nperseg=nperseg, detrend="constant")
    band = (f >= fmin) & (f <= fmax)
    if not np.any(band):
        return None
    hz = float(f[band][np.argmax(pxx[band])])
    return hz * 60.0


def try_dl_hr(
    green: np.ndarray,
    fs: float,
    model_path: Optional[str | Path] = None,
) -> Optional[float]:
    """Load optional 1D-CNN+LSTM weights. Returns None if torch/weights missing."""
    if model_path is None:
        return None
    path = Path(model_path)
    if not path.is_file():
        return None
    try:
        import torch
        from ml.ppg.dl_hr import CNNLSTMHeartRate, resample_to_n
    except Exception:
        return None
    try:
        model = CNNLSTMHeartRate()
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        x = resample_to_n(np.asarray(green, dtype=np.float32), n=900)
        with torch.no_grad():
            y = model(torch.from_numpy(x)[None, None, :])
        bpm = float(y.squeeze().cpu())
        if 48.0 <= bpm <= 180.0:
            return bpm
    except Exception:
        return None
    return None


def estimate_heart_rate(
    green_clean: np.ndarray,
    fs: float,
    dl_model_path: Optional[str | Path] = None,
    bpm_lo: float = 60.0,
    bpm_hi: float = 180.0,
) -> HeartRateEstimate:
    """Fuse peak and FFT estimates; optionally a third DL vote."""
    peak_bpm, ibi, peaks = hr_from_peaks(green_clean, fs, bpm_min=48.0, bpm_max=180.0)
    fft_bpm, hz, _, _ = hr_from_fft(green_clean, fs)
    if fft_bpm is None:
        w = hr_from_welch(green_clean, fs)
        if w is not None:
            fft_bpm, hz = w, w / 60.0
    dl_bpm = try_dl_hr(green_clean, fs, dl_model_path)

    candidates: list[tuple[str, float]] = []
    if peak_bpm is not None:
        candidates.append(("peaks", peak_bpm))
    if fft_bpm is not None:
        candidates.append(("fft", fft_bpm))
    if dl_bpm is not None:
        candidates.append(("dl", dl_bpm))

    method = "none"
    bpm = float("nan")
    if len(candidates) == 1:
        method, bpm = candidates[0]
    elif len(candidates) >= 2:
        peak_v = peak_bpm
        fft_v = fft_bpm
        if peak_v is not None and fft_v is not None and abs(peak_v - fft_v) <= 8.0:
            bpm = 0.5 * (peak_v + fft_v)
            method = "peak+fft"
        elif fft_v is not None and (peak_v is None or len(ibi) < 4):
            bpm, method = fft_v, "fft"
        elif peak_v is not None:
            bpm, method = peak_v, "peaks"
        if dl_bpm is not None and np.isfinite(bpm) and abs(dl_bpm - bpm) <= 12.0:
            bpm = (bpm + dl_bpm) / 2.0
            method = method + "+dl"

    if np.isfinite(bpm):
        bpm = float(np.clip(bpm, bpm_lo, bpm_hi))

    return HeartRateEstimate(
        bpm=bpm,
        peak_bpm=peak_bpm,
        fft_bpm=fft_bpm,
        dl_bpm=dl_bpm,
        ibi_seconds=ibi,
        peak_indices=peaks,
        dominant_hz=hz,
        method_used=method,
    )
