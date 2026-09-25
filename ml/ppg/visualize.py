"""Matplotlib (always) and optional Plotly figures for a PPG session."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ml.ppg.pipeline import VitalsResult


def plot_session(
    result: VitalsResult,
    out_path: str | Path | None = None,
    show: bool = False,
) -> Optional[Path]:
    """Raw RGB, cleaned green, HR spectrum, respiratory component, vitals text."""
    import matplotlib.pyplot as plt

    fs = result.fs
    n = result.rgb_raw.shape[0]
    t = np.arange(n) / fs
    green_c = result.rgb_clean[:, 1]
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    window = np.hanning(n)
    mag = np.abs(np.fft.rfft((green_c - green_c.mean()) * window))

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), constrained_layout=True)
    ax = axes[0]
    ax.plot(t, result.rgb_raw[:, 0], color="#c0392b", label="R", lw=1)
    ax.plot(t, result.rgb_raw[:, 1], color="#1e8449", label="G", lw=1)
    ax.plot(t, result.rgb_raw[:, 2], color="#2471a3", label="B", lw=1)
    ax.set_title("Raw mean-ROI PPG (R, G, B)")
    ax.set_ylabel("Intensity")
    ax.legend(loc="upper right", ncol=3)
    ax.set_xlim(t[0], t[-1])

    ax = axes[1]
    ax.plot(t, green_c, color="#1e8449", lw=1, label="Green cleaned")
    peaks = result.hr.peak_indices
    if peaks.size:
        ax.plot(t[peaks], green_c[peaks], "k.", ms=6, label="Peaks")
    ax.set_title("Filtered green PPG (HR path)")
    ax.set_ylabel("Normalized")
    ax.legend(loc="upper right")
    ax.set_xlim(t[0], t[-1])

    ax = axes[2]
    ax.plot(freq, mag, color="#4a4a4a", lw=1)
    if result.hr.dominant_hz is not None:
        ax.axvline(result.hr.dominant_hz, color="#c0392b", ls="--", label="HR peak")
        ax.axvspan(0.8, 3.0, color="#c0392b", alpha=0.08, label="HR band 0.8–3 Hz")
    ax.set_xlim(0, 4.0)
    ax.set_title("FFT magnitude (cleaned green)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|X|")
    ax.legend(loc="upper right")

    ax = axes[3]
    tr = np.arange(result.respiratory.component.size) / fs
    ax.plot(tr, result.respiratory.component, color="#6c3483", lw=1)
    ax.set_title("Respiratory component (0.1–0.5 Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("a.u.")
    ax.set_xlim(tr[0], tr[-1])

    hr = result.heart_rate_bpm
    spo2 = result.spo2_pct
    rr = result.respiratory_rate
    q = result.quality.score
    fig.suptitle(
        f"HR {hr:.1f} BPM  |  SpO2 {spo2:.1f}%  |  RR {rr:.1f} /min  |  quality {q:.2f}\n"
        f"CI HR [{result.hr_ci[0]:.1f}, {result.hr_ci[1]:.1f}]  "
        f"method={result.hr.method_used}  (not a medical device)",
        fontsize=11,
    )

    saved: Optional[Path] = None
    if out_path is not None:
        saved = Path(out_path)
        saved.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saved, dpi=120)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return saved


def try_plotly_html(result: VitalsResult, out_html: str | Path) -> Optional[Path]:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return None

    fs = result.fs
    t = np.arange(result.rgb_raw.shape[0]) / fs
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Raw RGB", "Clean green"))
    fig.add_trace(go.Scatter(x=t, y=result.rgb_raw[:, 1], name="G raw"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=result.rgb_clean[:, 1], name="G clean"), row=2, col=1)
    fig.update_layout(
        title=f"HR {result.heart_rate_bpm:.1f} | SpO2 {result.spo2_pct:.1f} | RR {result.respiratory_rate:.1f}",
        template="plotly_white",
        height=520,
    )
    path = Path(out_html)
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path
