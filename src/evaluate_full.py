"""
Evaluate detector against full-video ground truth.

1. Runs the detector on the full raw video.
2. Loads GT from outputs/full_video_gt.json.
3. Computes TP, FP, FN, Precision, Recall, F1, IoU.

Usage:
    python src/evaluate_full.py data_clips/2026-02-06/20260127_193759_tp00002.mp4
    python src/evaluate_full.py --gt outputs/full_video_gt.json <video>
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import OUTPUTS_DIR, DEFAULT_ROI_PATH, DetectionParams
from roi_select import load_roi
from soap_trigger_detector import detect_wash_events


GT_PATH = OUTPUTS_DIR / "full_video_gt.json"
IOU_THRESHOLD = 0.1


def match_events(gt_events, det_events, iou_thresh=IOU_THRESHOLD):
    """Greedy IoU matching between GT and detected events."""
    matched_gt = set()
    matched_det = set()
    matches = []

    # Sort by IoU descending for greedy matching
    pairs = []
    for di, det in enumerate(det_events):
        for gi, gt in enumerate(gt_events):
            inter_s = max(det["start_sec"], gt["start_sec"])
            inter_e = min(det["end_sec"], gt["end_sec"])
            inter = max(0, inter_e - inter_s)
            union = (det["end_sec"] - det["start_sec"]) + (gt["end_sec"] - gt["start_sec"]) - inter
            iou = inter / union if union > 0 else 0
            if iou > iou_thresh:
                pairs.append((iou, di, gi))

    pairs.sort(reverse=True)

    for iou, di, gi in pairs:
        if di not in matched_det and gi not in matched_gt:
            matched_det.add(di)
            matched_gt.add(gi)
            matches.append({"det_idx": di, "gt_idx": gi, "iou": iou})

    tp = len(matches)
    fp = len(det_events) - tp
    fn = len(gt_events) - tp

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        "mean_iou": sum(m["iou"] for m in matches) / len(matches) if matches else 0,
        "matches": matches,
        "unmatched_det": [i for i in range(len(det_events)) if i not in matched_det],
        "unmatched_gt": [i for i in range(len(gt_events)) if i not in matched_gt],
    }


def fmt(sec):
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:02d}:{s:04.1f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to full video")
    parser.add_argument("--gt", default=str(GT_PATH), help="GT JSON path")
    parser.add_argument("--roi", default=str(DEFAULT_ROI_PATH))
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD)
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    # Load GT
    gt_path = Path(args.gt)
    if not gt_path.exists():
        print(f"ERROR: GT not found: {gt_path}")
        print(f"Run annotate_full.py first to create ground truth.")
        sys.exit(1)

    gt_data = json.load(open(gt_path, encoding="utf-8"))
    gt_events = gt_data.get("events", [])
    print(f"GT events: {len(gt_events)}")

    # Load ROI
    roi_data = json.load(open(args.roi, encoding="utf-8"))
    roi = roi_data
    soap_zones = roi_data.get("soap_zones", [])
    sink_zones = roi_data.get("sink_zones", [])

    # Run detector
    print(f"Running detector on {video_path.name}...")
    params = DetectionParams()
    det_events = detect_wash_events(
        str(video_path), roi, soap_zones, params,
        sink_zones=sink_zones if sink_zones else None,
    )
    print(f"Detected events: {len(det_events)}")

    # Convert det_events format if needed
    det_list = []
    for ev in det_events:
        det_list.append({
            "start_sec": ev["start_sec"],
            "end_sec": ev["end_sec"],
            "station": ev.get("station", "?"),
        })

    # Match
    results = match_events(gt_events, det_list, iou_thresh=args.iou)

    # Print results
    print(f"\n{'='*60}")
    print(f"  FULL VIDEO EVALUATION")
    print(f"{'='*60}")
    print(f"  GT events:       {len(gt_events)}")
    print(f"  Detected events: {len(det_list)}")
    print(f"  TP: {results['tp']}  FP: {results['fp']}  FN: {results['fn']}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-Score:  {results['f1']:.4f}")
    print(f"  Mean IoU:  {results['mean_iou']:.4f}")
    print(f"{'='*60}")

    # Matched pairs
    if results["matches"]:
        print(f"\n  MATCHED (TP):")
        for m in results["matches"]:
            g = gt_events[m["gt_idx"]]
            d = det_list[m["det_idx"]]
            print(f"    Det {fmt(d['start_sec'])}-{fmt(d['end_sec'])} "
                  f"<-> GT {fmt(g['start_sec'])}-{fmt(g['end_sec'])}  "
                  f"IoU={m['iou']:.2f}")

    # False Positives
    if results["unmatched_det"]:
        print(f"\n  FALSE POSITIVES ({results['fp']}):")
        for di in results["unmatched_det"]:
            d = det_list[di]
            dur = d["end_sec"] - d["start_sec"]
            print(f"    Det: {fmt(d['start_sec'])} -> {fmt(d['end_sec'])}  "
                  f"({dur:.0f}s)  [Station {d['station']}]")

    # False Negatives
    if results["unmatched_gt"]:
        print(f"\n  FALSE NEGATIVES ({results['fn']}):")
        for gi in results["unmatched_gt"]:
            g = gt_events[gi]
            dur = g["end_sec"] - g["start_sec"]
            print(f"    GT: {fmt(g['start_sec'])} -> {fmt(g['end_sec'])}  ({dur:.0f}s)")

    # Save summary
    summary = {
        "video": str(video_path),
        "gt_count": len(gt_events),
        "det_count": len(det_list),
        **{k: v for k, v in results.items() if k not in ("matches", "unmatched_det", "unmatched_gt")},
    }
    summary_path = OUTPUTS_DIR / "eval_full_video_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
