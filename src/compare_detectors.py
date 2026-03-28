"""
Full-video detector comparison for the hand-wash detection thesis.

Runs all available detectors (heuristic + YOLO models) on a given video,
evaluates each against ground truth, and produces a formatted comparison
table with per-detector breakdown.

Results are saved to outputs/evaluation/detector_comparison.json for reproducibility.

Usage:
    python src/compare_detectors.py <video> --gt <gt_json>

Example:
    python src/compare_detectors.py data_clips/2026-02-06/20260127_234623_tp00003.mp4 \\
        --gt outputs/ground_truth/full_video_gt.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import OUTPUTS_DIR, TRAINING_DIR, EVAL_DIR, DETECTIONS_DIR, DEFAULT_ROI_PATH, DetectionParams, PROJECT_ROOT
from evaluate_full import match_events


# ── Detector registry ────────────────────────────────────────

DETECTORS = [
    {
        "name": "Soap Trigger (heuristic)",
        "key": "soap_trigger",
        "type": "soap_trigger",
    },
    {
        "name": "YOLOv8n-cls",
        "key": "yolov8n",
        "type": "yolo_cls",
        "model": str(TRAINING_DIR / "yolov8n_run" / "weights" / "best.pt"),
    },
    {
        "name": "YOLO11n-cls",
        "key": "yolo11n",
        "type": "yolo_cls",
        "model": str(TRAINING_DIR / "yolo11n_run" / "weights" / "best.pt"),
    },
    {
        "name": "YOLO26n-cls",
        "key": "yolo26n",
        "type": "yolo_cls",
        "model": str(TRAINING_DIR / "yolo26n_run" / "weights" / "best.pt"),
    },
]


# ── Runner functions ─────────────────────────────────────────

def run_soap_trigger(video_path: str, roi: dict, soap_zones: list,
                     sink_zones: list) -> list[dict]:
    """Run the heuristic soap-trigger detector."""
    from soap_trigger_detector import detect_wash_events as soap_detect

    params = DetectionParams()
    df = soap_detect(
        video_path, roi, soap_zones, params,
        show_preview=False, sink_zones=sink_zones,
    )
    if hasattr(df, "to_dict"):
        return df.to_dict("records")
    return list(df)


def run_yolo_cls(video_path: str, roi: dict, sink_zones: list,
                 model_path: str) -> list[dict]:
    """Run a YOLO classification detector."""
    from yolo_cls_detector import detect_wash_events as yolo_detect, YoloCLSParams

    params = YoloCLSParams()
    df = yolo_detect(
        video_path=video_path,
        roi=roi,
        sink_zones=sink_zones,
        model_path=model_path,
        params=params,
        show_preview=False,
    )
    if hasattr(df, "to_dict"):
        return df.to_dict("records")
    return list(df)


# ── Formatting helpers ───────────────────────────────────────

def fmt_time(sec: float) -> str:
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:02d}:{s:04.1f}"


def print_table(results: list[dict], gt_count: int) -> None:
    """Print a formatted comparison table."""
    header = (
        f"{'Detector':<28s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'Spl':>4s} "
        f"{'Prec':>7s} {'Recall':>7s} {'F1':>7s} {'IoU':>7s} {'Time':>8s}"
    )
    print(f"\n{'=' * len(header)}")
    print(f"  DETECTOR COMPARISON  (GT events: {gt_count})")
    print(f"{'=' * len(header)}")
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['name']:<28s} {r['tp']:>4d} {r['fp']:>4d} {r['fn']:>4d} {r['splits']:>4d} "
            f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f} "
            f"{r['mean_iou']:>7.4f} {r['runtime_sec']:>7.1f}s"
        )

    print("-" * len(header))

    best = max(results, key=lambda r: r["f1"])
    print(f"\n  Best F1: {best['name']} ({best['f1']:.4f})")
    print(f"  (Spl = split detections: one GT event fragmented into multiple dets, not counted as FP)")


def print_detail(name: str, det_events: list, gt_events: list,
                 match_result: dict) -> None:
    """Print per-event detail for one detector."""
    print(f"\n--- {name} ({len(det_events)} detections) ---")

    if match_result["matches"]:
        print(f"  MATCHED (TP={match_result['tp']}, splits={match_result['splits']}):")
        for m in match_result["matches"]:
            g = gt_events[m["gt_idx"]]
            d = det_events[m["det_idx"]]
            tag = " [split]" if m.get("split") else ""
            print(f"    Det {fmt_time(d['start_sec'])}-{fmt_time(d['end_sec'])} "
                  f"<-> GT {fmt_time(g['start_sec'])}-{fmt_time(g['end_sec'])}  "
                  f"IoU={m['iou']:.2f}{tag}")

    if match_result["unmatched_det"]:
        print(f"  FALSE POSITIVES ({match_result['fp']}):")
        for di in match_result["unmatched_det"]:
            d = det_events[di]
            dur = d["end_sec"] - d["start_sec"]
            station = d.get("station", "?")
            print(f"    {fmt_time(d['start_sec'])} -> {fmt_time(d['end_sec'])}  "
                  f"({dur:.0f}s)  [S{station}]")

    if match_result["unmatched_gt"]:
        print(f"  FALSE NEGATIVES ({match_result['fn']}):")
        for gi in match_result["unmatched_gt"]:
            g = gt_events[gi]
            dur = g["end_sec"] - g["start_sec"]
            print(f"    GT {fmt_time(g['start_sec'])} -> {fmt_time(g['end_sec'])}  ({dur:.0f}s)")


# ── Main ─────────────────────────────────────────────────────

def load_roi_data(roi_path: str) -> tuple[dict, list, list]:
    """Load and split ROI JSON into roi, soap_zones, sink_zones."""
    roi_data = json.load(open(roi_path, encoding="utf-8"))
    soap_zones = roi_data.pop("soap_zones", None)
    if soap_zones is None:
        single = roi_data.pop("soap_zone", None)
        soap_zones = [single] if single else []
    else:
        roi_data.pop("soap_zone", None)
    sink_zones = roi_data.pop("sink_zones", None) or []
    return roi_data, soap_zones, sink_zones


def run_comparison(video_path: str, gt_path: str, roi_path: str,
                   iou_thresh: float = 0.3,
                   detectors: list[dict] | None = None,
                   verbose: bool = True) -> list[dict]:
    """Run all detectors and evaluate against GT. Returns list of result dicts."""
    gt_data = json.load(open(gt_path, encoding="utf-8"))
    gt_events = gt_data["events"]
    print(f"Video:     {Path(video_path).name}")
    print(f"GT:        {Path(gt_path).name} ({len(gt_events)} events)")
    print(f"IoU thresh: {iou_thresh}")

    roi, soap_zones, sink_zones = load_roi_data(roi_path)

    if not sink_zones:
        print("ERROR: ROI JSON missing 'sink_zones'. Required for YOLO detectors.")
        sys.exit(1)

    detector_list = detectors or DETECTORS
    all_results = []

    for i, det_cfg in enumerate(detector_list):
        name = det_cfg["name"]
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(detector_list)}] Running: {name}")
        print(f"{'='*60}")

        t0 = time.time()

        if det_cfg["type"] == "soap_trigger":
            det_events = run_soap_trigger(video_path, roi, soap_zones, sink_zones)
        elif det_cfg["type"] == "yolo_cls":
            model_path = det_cfg["model"]
            if not Path(model_path).exists():
                print(f"  SKIP: model not found at {model_path}")
                continue
            det_events = run_yolo_cls(video_path, roi, sink_zones, model_path)
        else:
            print(f"  SKIP: unknown detector type '{det_cfg['type']}'")
            continue

        runtime = time.time() - t0
        print(f"  Detected {len(det_events)} events in {runtime:.1f}s")

        # Save per-detector CSV
        det_dir = DETECTIONS_DIR
        det_dir.mkdir(parents=True, exist_ok=True)
        video_stem = Path(video_path).stem
        csv_path = det_dir / f"{det_cfg['key']}_{video_stem}.csv"
        pd.DataFrame(det_events).to_csv(str(csv_path), index=False, encoding="utf-8")
        print(f"  Events saved: {csv_path}")

        # Evaluate
        match_result = match_events(gt_events, det_events, iou_thresh=iou_thresh)

        result = {
            "name": name,
            "key": det_cfg["key"],
            "det_count": len(det_events),
            "tp": match_result["tp"],
            "fp": match_result["fp"],
            "fn": match_result["fn"],
            "splits": match_result["splits"],
            "precision": match_result["precision"],
            "recall": match_result["recall"],
            "f1": match_result["f1"],
            "mean_iou": match_result["mean_iou"],
            "runtime_sec": round(runtime, 1),
            "events": det_events,
        }
        all_results.append(result)

        if verbose:
            print_detail(name, det_events, gt_events, match_result)

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare all detectors on a full video against GT."
    )
    parser.add_argument("video", help="Path to input video.")
    parser.add_argument("--gt", required=True, help="Path to ground-truth JSON.")
    parser.add_argument("--roi", default=str(DEFAULT_ROI_PATH), help="ROI JSON path.")
    parser.add_argument("--iou", type=float, default=0.3, help="IoU threshold (default 0.3).")
    parser.add_argument("--quiet", action="store_true", help="Skip per-event detail output.")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Run only these detectors by key (e.g. --only yolo26n soap_trigger).")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    gt_path = Path(args.gt)
    if not gt_path.exists():
        print(f"ERROR: GT not found: {gt_path}")
        sys.exit(1)

    # Filter detectors if --only specified
    detectors = DETECTORS
    if args.only:
        detectors = [d for d in DETECTORS if d["key"] in args.only]
        if not detectors:
            print(f"ERROR: No matching detectors for --only {args.only}")
            print(f"Available keys: {[d['key'] for d in DETECTORS]}")
            sys.exit(1)

    # Run comparison
    gt_data = json.load(open(args.gt, encoding="utf-8"))
    gt_count = len(gt_data["events"])

    results = run_comparison(
        str(video_path), str(gt_path), args.roi,
        iou_thresh=args.iou,
        detectors=detectors,
        verbose=not args.quiet,
    )

    # Print summary table
    print_table(results, gt_count)

    # Save results (strip per-event lists from JSON summary — those are in CSVs)
    summary_results = [{k: v for k, v in r.items() if k != "events"} for r in results]
    output = {
        "timestamp": datetime.now().isoformat(),
        "video": str(video_path),
        "gt_path": str(gt_path),
        "gt_count": gt_count,
        "iou_threshold": args.iou,
        "results": summary_results,
    }

    out_path = EVAL_DIR / "detector_comparison.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
