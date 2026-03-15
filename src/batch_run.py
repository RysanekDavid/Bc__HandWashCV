"""
Batch processing: run baseline motion detection across all clips
and produce a single aggregated CSV.

Usage:
    python src/batch_run.py                        # all clips, default params
    python src/batch_run.py --limit 20             # first 20 clips
    python src/batch_run.py --motion-thresh 3500   # override threshold
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import DATA_CLIPS_DIR, DEFAULT_ROI_PATH, OUTPUTS_DIR, DetectionParams
from roi_select import load_roi
from baseline_motion import detect_wash_events


def run_batch(
    clips: list[Path],
    roi: dict[str, int],
    params: DetectionParams,
    overlay_dir: Path | None = None,
) -> pd.DataFrame:
    """Process a list of clips and return a combined events DataFrame."""
    all_events: list[pd.DataFrame] = []

    for clip in tqdm(clips, desc="Processing clips"):
        overlay_path = None
        if overlay_dir is not None:
            overlay_dir.mkdir(parents=True, exist_ok=True)
            overlay_path = str(overlay_dir / f"overlay_{clip.stem}.mp4")

        df = detect_wash_events(
            video_path=str(clip),
            roi=roi,
            params=params,
            show_preview=False,
            overlay_path=overlay_path,
        )
        if not df.empty:
            all_events.append(df)

    if all_events:
        return pd.concat(all_events, ignore_index=True)
    return pd.DataFrame(columns=["video", "start_sec", "end_sec", "duration_sec"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch hand-wash detection.")
    parser.add_argument("--roi", default=str(DEFAULT_ROI_PATH), help="ROI JSON path.")
    parser.add_argument("--out-csv", default=str(OUTPUTS_DIR / "all_events.csv"),
                        help="Output CSV path.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N clips (for testing).")
    parser.add_argument("--with-overlays", action="store_true",
                        help="Also produce overlay videos (slower).")
    parser.add_argument("--motion-thresh", type=int, default=None)
    parser.add_argument("--wash-on", type=float, default=None)
    parser.add_argument("--wash-off", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    roi = load_roi(Path(args.roi))

    clips = sorted(DATA_CLIPS_DIR.glob("clip_*.mp4"))
    if not clips:
        print(f"ERROR: No clips found in {DATA_CLIPS_DIR}")
        return

    if args.limit:
        clips = clips[: args.limit]

    print(f"Clips to process: {len(clips)}")

    params = DetectionParams()
    if args.motion_thresh is not None:
        params.motion_thresh = args.motion_thresh
    if args.wash_on is not None:
        params.wash_sec_on = args.wash_on
    if args.wash_off is not None:
        params.wash_sec_off = args.wash_off

    overlay_dir = OUTPUTS_DIR / "overlays" if args.with_overlays else None

    combined = run_batch(clips, roi, params, overlay_dir)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False, encoding="utf-8")

    print(f"\nTotal events: {len(combined)}")
    print(f"CSV saved   : {out_csv}")

    if not combined.empty:
        print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
