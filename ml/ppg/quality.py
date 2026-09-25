"""Signal quality for fingertip PPG (SNR, peak regularity, optional motion)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import get_window

from ml.ppg.heart_rate import HeartRateEstimate


@dataclass
class QualityReport:
    score: float
    snr_db: float
    peak_regularity: float
    motion_score: float
    perfusion_index: float
    notes: list[str] = field(default_factory=list)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def snr_db_around_hr(green: np.ndarray, fs: float, hr_hz: Optional[float]) -> float:
    """Power in a ±0.2 Hz band around HR vs. the rest of 0.5–4 Hz."""
    x = np.asarray(green, dtype=np.float64)
    x = x - np.mean(x)
    n = x.size
    if n < 16 or hr_hz is None or not np.isfinite(hr_hz):
        return -10.0
    window = get_window("hann", n, fftbins=True)
    mag = np.abs(np.fft.rfft(x * window)) ** 2
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    signal_band = np.abs(freq - hr_hz) <= 0.2
    noise_band = (freq >= 0.5) & (freq <= 4.0) & ~signal_band
    p_s = float(mag[signal_band].sum())
    p_n = float(mag[noise_band].mean() * max(signal_band.sum(), 1))
    if p_n <= 1e-18:
        return 30.0
    return float(10.0 * np.log10(p_s / p_n + 1e-12))


def peak_regularity_score(ibi: np.ndarray) -> float:
    """1 = metronomic beats; 0 = chaotic / too few IBIs."""
    if ibi is None or ibi.size < 3:
        return 0.0
    cv = float(np.std(ibi) / (np.mean(ibi) + 1e-9))
    return float(np.clip(np.exp(-4.0 * cv), 0.0, 1.0))


def motion_quality(motion: Optional[np.ndarray], n_ppg: int) -> tuple[float, str | None]:
    """Return (score, note). `motion` is |a| or a 3-axis array aligned to PPG."""
    if motion is None:
        return 1.0, None
    m = np.asarray(motion, dtype=np.float64)
    if m.ndim == 2 and m.shape[1] == 3:
        mag = np.linalg.norm(m, axis=1)
    else:
        mag = m.reshape(-1)
    if mag.size != n_ppg:
        # Resample loosely to PPG length.
        t_old = np.linspace(0, 1, mag.size)
        t_new = np.linspace(0, 1, n_ppg)
        mag = np.interp(t_new, t_old, mag)
    mag = mag - np.median(mag)
    energy = float(np.sqrt(np.mean(mag * mag)))
    # Heuristic: phone-still energy is small; walking is large. Scale empirically.
    score = float(np.clip(1.0 - energy / 2.5, 0.0, 1.0))
    note = None
    if score < 0.5:
        note = "High motion energy — hold still with the finger covering the lens."
    return score, note


def perfusion_index(ac: float, dc: float) -> float:
    if abs(dc) < 1e-9:
        return 0.0
    return float(np.clip(abs(ac / dc), 0.0, 0.2))


def overall_quality(
    green_clean: np.ndarray,
    fs: float,
    hr: HeartRateEstimate,
    ac_green: float,
    dc_green: float,
    motion: Optional[np.ndarray] = None,
    extra_notes: Optional[list[str]] = None,
) -> QualityReport:
    notes = list(extra_notes or [])
    hz = hr.dominant_hz
    if hz is None and np.isfinite(hr.bpm):
        hz = hr.bpm / 60.0
    snr = snr_db_around_hr(green_clean, fs, hz)
    snr_s = _sigmoid((snr - 4.0) / 4.0)
    reg = peak_regularity_score(hr.ibi_seconds)
    mot, mot_note = motion_quality(motion, len(green_clean))
    if mot_note:
        notes.append(mot_note)
    pi = perfusion_index(ac_green, dc_green)
    pi_s = float(np.clip(pi / 0.02, 0.0, 1.0))  # PI ~2% is healthy fingertip

    if not np.isfinite(hr.bpm):
        notes.append("Heart-rate estimator failed — quality forced down.")
        score = 0.15 * mot
    else:
        score = 0.40 * snr_s + 0.30 * reg + 0.15 * mot + 0.15 * pi_s

    if snr < 2.0:
        notes.append("Low SNR in the heart-rate band.")
    if reg < 0.4 and hr.ibi_seconds.size:
        notes.append("Irregular peak intervals (motion or arrhythmia / missed beats).")
    if pi < 0.005:
        notes.append("Low perfusion index — press gently, use the torch, warm the finger.")

    score = float(np.clip(score, 0.0, 1.0))
    return QualityReport(
        score=score,
        snr_db=snr,
        peak_regularity=reg,
        motion_score=mot,
        perfusion_index=pi,
        notes=notes,
    )
