"""Example usage of the fingertip PPG pipeline.

Run from the repository root after installing ml/ppg/requirements.txt:

    pip install -r "ml/ppg/requirements.txt"
    python ml/ppg/example_usage.py
    python -m ml.ppg --demo --plot artifacts/ppg_demo.png
    python -m ml.ppg path/to/finger_30s.mp4 --plot artifacts/ppg.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.ppg.pipeline import estimate_from_rgb, estimate_from_video
from ml.ppg.synthetic import synthetic_rgb_ppg, write_synthetic_video
from ml.ppg.visualize import plot_session


def main() -> None:
    # --- 1) Traces only (fast, no video decode) --------------------------------
    rgb = synthetic_rgb_ppg(
        duration_s=30.0,
        fs=30.0,
        hr_bpm=72.0,
        rr_bpm=16.0,
        spo2_target=98.0,
        seed=7,
    )
    assert rgb.shape == (900, 3)

    result = estimate_from_rgb(rgb, fs=30.0)
    print("From synthetic RGB traces:")
    print(json.dumps(result.as_dict(), indent=2))

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    plot_session(result, out_path=artifacts / "ppg_traces.png", show=False)

    # --- 2) Tiny MP4 whose pixel means match the traces -------------------------
    video_path = artifacts / "synthetic_ppg.mp4"
    write_synthetic_video(video_path, rgb=rgb, fs=30.0)
    from_video = estimate_from_video(video_path)
    print("\nFrom synthetic MP4:")
    print(
        json.dumps(
            {
                "heart_rate_bpm": from_video.heart_rate_bpm,
                "spo2_pct": from_video.spo2_pct,
                "respiratory_rate": from_video.respiratory_rate,
                "quality": from_video.quality.score,
                "elapsed_s": from_video.elapsed_s,
            },
            indent=2,
        )
    )
    print("\n" + result.disclaimer)


if __name__ == "__main__":
    main()
