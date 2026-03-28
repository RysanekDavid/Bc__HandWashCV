"""
Evaluate detector against full-video ground truth.

1. Runs the detector on the full raw video.
2. Loads GT from outputs/ground_truth/full_video_gt.json.
3. Computes TP, FP, FN, Precision, Recall, F1, IoU.

Usage:
    python src/evaluate_full.py data_clips/2026-02-06/20260127_193759_tp00002.mp4
    python src/evaluate_full.py --gt outputs/ground_truth/full_video_gt.json <video>
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import OUTPUTS_DIR, GT_DIR, EVAL_DIR, DEFAULT_ROI_PATH, DetectionParams
from roi_select import load_roi
from soap_trigger_detector import detect_wash_events


GT_PATH = GT_DIR / "full_video_gt.json"
IOU_THRESHOLD = 0.3


def match_events(gt_events, det_events, iou_thresh=IOU_THRESHOLD):
    """Greedy IoU matching between GT and detected events.

    Returns strict 1:1 metrics (tp = unique GT events matched) and also
    tracks split detections (extra dets falling inside an already-matched
    GT window) separately so they don't inflate TP or FP.
    """
    matched_gt = set()
    matched_det = set()
    matches = []

    # Build all pairs with IoU
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

    # Greedy: best IoU first — strict 1:1 matching
    pairs.sort(reverse=True)

    for iou, di, gi in pairs:
        if di not in matched_det and gi not in matched_gt:
            matched_det.add(di)
            matched_gt.add(gi)
            matches.append({"det_idx": di, "gt_idx": gi, "iou": iou})

    # Split detections: extra dets whose midpoint falls inside any GT window.
    # These are NOT counted as TP or FP — they are fragments of already-detected events.
    split_dets = []
    for di, det in enumerate(det_events):
        if di in matched_det:
            continue
        det_mid = (det["start_sec"] + det["end_sec"]) / 2
        for gi, gt in enumerate(gt_events):
            if gt["start_sec"] <= det_mid <= gt["end_sec"]:
                matched_det.add(di)
                split_dets.append({"det_idx": di, "gt_idx": gi})
                matches.append({"det_idx": di, "gt_idx": gi, "iou": 0.0, "split": True})
                break

    # Strict metrics: 1 TP per unique GT event matched
    tp = len(matched_gt)
    fp = len(det_events) - len(matched_det)
    fn = len(gt_events) - len(matched_gt)
    n_splits = len(split_dets)

    iou_matches = [m for m in matches if not m.get("split")]
    mean_iou = (sum(m["iou"] for m in iou_matches) / len(iou_matches)) if iou_matches else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "splits": n_splits,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        "mean_iou": mean_iou,
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

    # Load ROI (must pop extra keys, same as evaluate.py)
    roi_data = json.load(open(args.roi, encoding="utf-8"))
    soap_zones = roi_data.pop("soap_zones", None)
    if soap_zones is None:
        single = roi_data.pop("soap_zone", None)
        if single is not None:
            soap_zones = [single]
    else:
        roi_data.pop("soap_zone", None)
    sink_zones = roi_data.pop("sink_zones", None)
    roi = roi_data

    # Run detector
    print(f"Running detector on {video_path.name}...")
    params = DetectionParams()

    df = detect_wash_events(
        str(video_path), roi, soap_zones or [], params,
        show_preview=False,
        sink_zones=sink_zones,
    )

    # Convert DataFrame to list of dicts
    if hasattr(df, 'to_dict'):
        det_list = df.to_dict('records')
    else:
        det_list = list(df)

    print(f"Detected events: {len(det_list)}")

    # Match
    results = match_events(gt_events, det_list, iou_thresh=args.iou)

    # Print results
    print(f"\n{'='*60}")
    print(f"  FULL VIDEO EVALUATION")
    print(f"{'='*60}")
    print(f"  GT events:       {len(gt_events)}")
    print(f"  Detected events: {len(det_list)}")
    print(f"  TP: {results['tp']}  FP: {results['fp']}  FN: {results['fn']}  Splits: {results['splits']}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-Score:  {results['f1']:.4f}")
    print(f"  Mean IoU:  {results['mean_iou']:.4f}")
    print(f"{'='*60}")

    # Matched pairs
    if results["matches"]:
        print(f"\n  MATCHED (TP={results['tp']}, splits={results['splits']}):")
        for m in results["matches"]:
            g = gt_events[m["gt_idx"]]
            d = det_list[m["det_idx"]]
            tag = " [split]" if m.get("split") else ""
            print(f"    Det {fmt(d['start_sec'])}-{fmt(d['end_sec'])} "
                  f"<-> GT {fmt(g['start_sec'])}-{fmt(g['end_sec'])}  "
                  f"IoU={m['iou']:.2f}{tag}")

    # False Positives
    if results["unmatched_det"]:
        print(f"\n  FALSE POSITIVES ({results['fp']}):")
        for di in results["unmatched_det"]:
            d = det_list[di]
            dur = d["end_sec"] - d["start_sec"]
            station = d.get("station", "?")
            print(f"    Det: {fmt(d['start_sec'])} -> {fmt(d['end_sec'])}  "
                  f"({dur:.0f}s)  [Station {station}]")

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
    summary_path = EVAL_DIR / "eval_full_video_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
