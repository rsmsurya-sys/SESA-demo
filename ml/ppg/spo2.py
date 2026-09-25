"""Smartphone SpO2 proxy via red/green ratio-of-ratios.

Phone cameras have no IR (~940 nm) photodiode. Classical pulse-ox uses red vs
infrared. This module substitutes **green (~530 nm)** for IR as specified for
the prototype. The 110 − 25R curve is empirical and **not** FDA-calibrated.
Treat values as research-grade; quality should drop when perfusion is poor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ml.ppg.preprocess import bandpass, lowpass, wavelet_denoise


@dataclass
class SpO2Estimate:
    spo2: float
    r_ratio: float
    ac_red: float
    dc_red: float
    ac_green: float
    dc_green: float
    formula: str


def _ac_rms(pulsatile: np.ndarray) -> float:
    x = np.asarray(pulsatile, dtype=np.float64)
    x = x - np.mean(x)
    return float(np.sqrt(np.mean(x * x) + 1e-18))


def ratio_of_ratios(
    rgb_raw: np.ndarray,
    fs: float,
    ac_band: tuple[float, float] = (0.7, 4.0),
) -> tuple[float, float, float, float, float]:
    """R = (AC_red/DC_red) / (AC_green/DC_green).

    DC: low-pass (~0.2 Hz) of wavelet-denoised intensity (illumination / blood volume).
    AC: RMS of band-passed pulsatile component (0.7–4 Hz).
    """
    rgb = np.asarray(rgb_raw, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] < 2:
        raise ValueError("rgb_raw must have shape (N, 3) with R and G columns")

    den = wavelet_denoise(rgb, wavelet="db4", level=4)
    red, green = den[:, 0], den[:, 1]

    dc_r = float(np.mean(lowpass(red, fs, cutoff=0.2, order=2)))
    dc_g = float(np.mean(lowpass(green, fs, cutoff=0.2, order=2)))
    # Fallback if the low-pass averaged near zero after filtering transients.
    if abs(dc_r) < 1e-3:
        dc_r = float(np.mean(red))
    if abs(dc_g) < 1e-3:
        dc_g = float(np.mean(green))
    if abs(dc_r) < 1e-6 or abs(dc_g) < 1e-6:
        raise ValueError("DC component too small — illumination/finger placement failed")

    ac_r = _ac_rms(bandpass(red, fs, ac_band[0], ac_band[1], order=2))
    ac_g = _ac_rms(bandpass(green, fs, ac_band[0], ac_band[1], order=2))
    if ac_g < 1e-9:
        raise ValueError("Green AC too small — no pulsatile PPG")

    r_ratio = (ac_r / abs(dc_r)) / (ac_g / abs(dc_g))
    return float(r_ratio), ac_r, dc_r, ac_g, dc_g


def spo2_from_ror(r_ratio: float) -> float:
    """Empirical calibration used in many smartphone PPG papers / this spec."""
    return float(110.0 - 25.0 * r_ratio)


def try_vit_spo2(_rgb_raw: np.ndarray, model_path: Optional[str] = None) -> Optional[float]:
    """Placeholder for a frame-stack ViT. Returns None unless weights exist."""
    if not model_path:
        return None
    # A real ViT would sample 32 ROI frames; not shipped in the prototype.
    return None


def estimate_spo2(
    rgb_raw: np.ndarray,
    fs: float,
    spo2_lo: float = 90.0,
    spo2_hi: float = 100.0,
    vit_model_path: Optional[str] = None,
) -> SpO2Estimate:
    r_ratio, ac_r, dc_r, ac_g, dc_g = ratio_of_ratios(rgb_raw, fs)
    vit = try_vit_spo2(rgb_raw, vit_model_path)
    if vit is not None:
        val, formula = vit, "vit_rgb_ratios"
    else:
        val, formula = spo2_from_ror(r_ratio), "110 - 25 * RoR (R vs G proxy for IR)"

    clipped = float(np.clip(val, spo2_lo, spo2_hi))
    return SpO2Estimate(
        spo2=clipped,
        r_ratio=r_ratio,
        ac_red=ac_r,
        dc_red=dc_r,
        ac_green=ac_g,
        dc_green=dc_g,
        formula=formula,
    )
