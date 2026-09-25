"""CLI: python -m ml.ppg VIDEO.mp4   or   python -m ml.ppg --demo"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fingertip PPG vitals from a 30 s rear-camera video (research prototype)."
    )
    parser.add_argument("video", nargs="?", help="Path to MP4/AVI (finger on lens + torch)")
    parser.add_argument("--demo", action="store_true", help="Run on synthetic 30 s traces")
    parser.add_argument("--plot", default=None, help="Save matplotlib PNG")
    parser.add_argument("--html", default=None, help="Save Plotly HTML if plotly is installed")
    parser.add_argument("--dl-hr", default=None, help="Optional CNN-LSTM state_dict path")
    parser.add_argument("--motion", default=None, help="Optional .npy motion magnitude (N,) or (N,3)")
    args = parser.parse_args(argv)

    # Allow running from repo root without installing the package.
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from ml.ppg.pipeline import PPGPipeline
    from ml.ppg.synthetic import synthetic_rgb_ppg
    from ml.ppg.visualize import plot_session, try_plotly_html

    motion = None
    if args.motion:
        import numpy as np

        motion = np.load(args.motion)

    pipe = PPGPipeline(dl_hr_model_path=args.dl_hr)

    if args.demo:
        rgb = synthetic_rgb_ppg(duration_s=30.0, fs=30.0, hr_bpm=72.0, rr_bpm=16.0, spo2_target=98.0)
        result = pipe.from_rgb(rgb, fs=30.0, motion=motion)
    elif args.video:
        result = pipe.from_video(args.video, motion=motion)
    else:
        parser.print_help()
        return 2

    print(json.dumps(result.as_dict(), indent=2))
    print(result.disclaimer)

    if args.plot:
        out = plot_session(result, out_path=args.plot, show=False)
        print(f"Wrote {out}")
    if args.html:
        html = try_plotly_html(result, args.html)
        print(f"Wrote {html}" if html else "plotly not installed; skipped HTML")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
