"""End-to-end fingertip PPG pipeline: video or (N, 3) RGB traces → vitals."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np

from ml.ppg.heart_rate import HeartRateEstimate, estimate_heart_rate, hr_from_fft
from ml.ppg.preprocess import preprocess_hr_spo2
from ml.ppg.quality import QualityReport, overall_quality
from ml.ppg.respiratory import RespiratoryEstimate, estimate_respiratory_rate
from ml.ppg.spo2 import SpO2Estimate, estimate_spo2
from ml.ppg.video import extract_rgb_ppg


@dataclass
class VitalsResult:
    heart_rate_bpm: float
    spo2_pct: float
    respiratory_rate: float
    quality: QualityReport
    hr: HeartRateEstimate
    spo2: SpO2Estimate
    respiratory: RespiratoryEstimate
    hr_ci: tuple[float, float]
    spo2_ci: tuple[float, float]
    rr_ci: tuple[float, float]
    fs: float
    rgb_raw: np.ndarray
    rgb_clean: np.ndarray
    elapsed_s: float
    video_info: dict = field(default_factory=dict)
    disclaimer: str = (
        "Research prototype for emergency triage assistance. Not a medical device "
        "and not a diagnostic measurement. MAE targets vs. clinical sensors require "
        "per-device calibration and are not guaranteed by this code."
    )

    def as_dict(self) -> dict:
        return {
            "heart_rate_bpm": self.heart_rate_bpm,
            "spo2_pct": self.spo2_pct,
            "respiratory_rate": self.respiratory_rate,
            "quality": self.quality.score,
            "snr_db": self.quality.snr_db,
            "hr_method": self.hr.method_used,
            "rr_method": self.respiratory.method_used,
            "hr_ci": self.hr_ci,
            "spo2_ci": self.spo2_ci,
            "rr_ci": self.rr_ci,
            "r_ratio": self.spo2.r_ratio,
            "fs": self.fs,
            "n_samples": int(self.rgb_raw.shape[0]),
            "elapsed_s": self.elapsed_s,
            "notes": self.quality.notes,
            "disclaimer": self.disclaimer,
        }


def _ibi_confidence_interval(ibi: np.ndarray, bpm_lo: float, bpm_hi: float) -> tuple[float, float]:
    if ibi is None or ibi.size < 3:
        return (float("nan"), float("nan"))
    bpm_series = 60.0 / ibi
    se = float(np.std(bpm_series, ddof=1) / np.sqrt(len(bpm_series)))
    mu = float(np.mean(bpm_series))
    lo = float(np.clip(mu - 1.96 * se, bpm_lo, bpm_hi))
    hi = float(np.clip(mu + 1.96 * se, bpm_lo, bpm_hi))
    return lo, hi


class PPGPipeline:
    """Configure once, run on many videos.

    Parameters
    ----------
    roi_fraction
        Center crop used as the fingertip ROI (0.5 = center 50%).
    max_frames
        Default 900 = 30 s at 30 FPS.
    dl_hr_model_path
        Optional PyTorch state_dict for the 1D-CNN+LSTM HR head.
    """

    def __init__(
        self,
        roi_fraction: float = 0.5,
        max_frames: int = 900,
        expected_fps: float = 30.0,
        dl_hr_model_path: Optional[str | Path] = None,
        vit_spo2_model_path: Optional[str | Path] = None,
    ) -> None:
        self.roi_fraction = roi_fraction
        self.max_frames = max_frames
        self.expected_fps = expected_fps
        self.dl_hr_model_path = dl_hr_model_path
        self.vit_spo2_model_path = (
            str(vit_spo2_model_path) if vit_spo2_model_path else None
        )

    def from_rgb(
        self,
        rgb: np.ndarray,
        fs: float,
        motion: Optional[np.ndarray] = None,
        video_info: Optional[dict] = None,
    ) -> VitalsResult:
        t0 = perf_counter()
        rgb = np.asarray(rgb, dtype=np.float64)
        if rgb.ndim != 2 or rgb.shape[1] != 3:
            raise ValueError(f"Expected RGB array of shape (N, 3); got {rgb.shape}")
        if rgb.shape[0] < int(fs * 8):
            raise ValueError(
                f"Need >= 8 seconds of PPG (got {rgb.shape[0]} samples at {fs} Hz)"
            )
        if not np.isfinite(rgb).all():
            raise ValueError("RGB traces contain NaN/Inf")

        rgb_clean = preprocess_hr_spo2(rgb, fs)
        green_clean = rgb_clean[:, 1]
        green_raw = rgb[:, 1]

        hr = estimate_heart_rate(green_clean, fs, dl_model_path=self.dl_hr_model_path)
        spo2 = estimate_spo2(
            rgb, fs, vit_model_path=self.vit_spo2_model_path
        )
        rr = estimate_respiratory_rate(green_raw, fs)

        notes = list((video_info or {}).get("warnings") or [])
        quality = overall_quality(
            green_clean,
            fs,
            hr,
            ac_green=spo2.ac_green,
            dc_green=spo2.dc_green,
            motion=motion,
            extra_notes=notes,
        )

        hr_ci = _ibi_confidence_interval(hr.ibi_seconds, 60.0, 180.0)
        if not np.isfinite(hr_ci[0]) and np.isfinite(hr.bpm):
            # FFT-only: use 0.05 Hz spectral resolution as a rough CI.
            _, _, freq, mag = hr_from_fft(green_clean, fs)
            df = float(freq[1] - freq[0]) * 60.0 if freq.size > 1 else 3.0
            hr_ci = (
                float(np.clip(hr.bpm - 1.96 * df, 60.0, 180.0)),
                float(np.clip(hr.bpm + 1.96 * df, 60.0, 180.0)),
            )

        # SpO2 CI is wide on purpose: uncalibrated R/G proxy.
        spo2_ci = (
            float(np.clip(spo2.spo2 - 2.0, 90.0, 100.0)),
            float(np.clip(spo2.spo2 + 2.0, 90.0, 100.0)),
        )
        rr_ci = (
            float(np.clip(rr.breaths_per_min - 2.0, 12.0, 30.0)),
            float(np.clip(rr.breaths_per_min + 2.0, 12.0, 30.0)),
        )

        elapsed = perf_counter() - t0
        if elapsed > 5.0:
            quality.notes.append(
                f"Processing took {elapsed:.2f}s (target < 5s). Consider shorter ROI or fewer frames."
            )

        return VitalsResult(
            heart_rate_bpm=float(hr.bpm) if np.isfinite(hr.bpm) else float("nan"),
            spo2_pct=float(spo2.spo2),
            respiratory_rate=float(rr.breaths_per_min) if np.isfinite(rr.breaths_per_min) else float("nan"),
            quality=quality,
            hr=hr,
            spo2=spo2,
            respiratory=rr,
            hr_ci=hr_ci,
            spo2_ci=spo2_ci,
            rr_ci=rr_ci,
            fs=float(fs),
            rgb_raw=rgb,
            rgb_clean=rgb_clean,
            elapsed_s=float(elapsed),
            video_info=video_info or {},
        )

    def from_video(
        self,
        video_path: str | Path,
        motion: Optional[np.ndarray] = None,
    ) -> VitalsResult:
        rgb, fs, info = extract_rgb_ppg(
            video_path,
            roi_fraction=self.roi_fraction,
            expected_fps=self.expected_fps,
            max_frames=self.max_frames,
        )
        return self.from_rgb(rgb, fs, motion=motion, video_info=info)


def estimate_from_video(
    video_path: str | Path,
    **kwargs,
) -> VitalsResult:
    """Convenience wrapper used in example usage."""
    motion = kwargs.pop("motion", None)
    pipe = PPGPipeline(**kwargs)
    return pipe.from_video(video_path, motion=motion)


def estimate_from_rgb(
    rgb: np.ndarray,
    fs: float = 30.0,
    **kwargs,
) -> VitalsResult:
    motion = kwargs.pop("motion", None)
    pipe = PPGPipeline(**kwargs)
    return pipe.from_rgb(rgb, fs, motion=motion)
