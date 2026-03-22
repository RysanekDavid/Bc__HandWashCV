"""
Evaluation script for hand-wash detection.
Compares Ground Truth (annotations.json) vs. detector predictions.

Supports multiple detectors:
  - baseline  : MOG2 background subtraction (baseline_motion.py)
  - mediapipe : MediaPipe Hands + motion    (mediapipe_detector.py)

Metrics calculated:
  - Precision, Recall, F1-Score (Event-level)
  - Mean IoU (Intersection over Union) of matched events

Results are printed and saved to outputs/eval_<detector>.csv.
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from config import LABELED_DIR, DEFAULT_ROI_PATH, OUTPUTS_DIR, DetectionParams
from roi_select import load_roi


def _get_detector_function(detector_name: str):
    """Import and return the detect_wash_events function for the given detector."""
    if detector_name == "baseline":
        from baseline_motion import detect_wash_events
        return detect_wash_events
    elif detector_name == "mediapipe":
        from mediapipe_detector import detect_wash_events
        return detect_wash_events
    elif detector_name == "soap_trigger":
        from soap_trigger_detector import detect_wash_events
        return detect_wash_events
    else:
        raise ValueError(f"Unknown detector: {detector_name}. Use 'baseline', 'mediapipe', or 'soap_trigger'.")

def calculate_iou(start1, end1, start2, end2):
    """Calculate temporal Intersection over Union for two segments."""
    inter_start = max(start1, start2)
    inter_end = min(end1, end2)
    intersection = max(0, inter_end - inter_start)
    
    union = (end1 - start1) + (end2 - start2) - intersection
    return intersection / union if union > 0 else 0

def evaluate_performance(iou_threshold=0.1, detector_name="baseline", params=None):
    detect_wash_events = _get_detector_function(detector_name)
    roi_data = load_roi(DEFAULT_ROI_PATH)
    if params is None:
        params = DetectionParams()

    # Separate soap zones if present (for soap_trigger detector)
    soap_zones = roi_data.pop("soap_zones", None)
    if soap_zones is None:
        single = roi_data.pop("soap_zone", None)
        if single is not None:
            soap_zones = [single]
    else:
        roi_data.pop("soap_zone", None)

    # Sink zones for per-station tracking
    sink_zones = roi_data.pop("sink_zones", None)

    roi = roi_data
    
    # Load Ground Truth
    annotations_path = OUTPUTS_DIR / "annotations.json"
    if not annotations_path.exists():
        print(f"ERROR: Annotations not found at {annotations_path}")
        return
        
    with open(annotations_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
        
    clips = sorted(LABELED_DIR.glob("clip_*.mp4"))
    if not clips:
        print(f"ERROR: No clips found in {LABELED_DIR}")
        return

    print(f"Evaluating {len(clips)} labeled clips...")
    
    tp, fp, fn = 0, 0, 0
    iou_scores = []
    skipped = 0
    
    results_list = []

    for clip_path in tqdm(clips, desc="Running detector"):
        name = clip_path.name
        clip_meta = gt_data.get(name, {})

        # Skip excluded clips (e.g. cleaning staff anomalies)
        if clip_meta.get("exclude", False):
            skipped += 1
            continue

        gt_events = clip_meta.get("events", [])
        
        # Run detection
        extra_kwargs = {}
        if detector_name == "soap_trigger":
            if soap_zones is None:
                print("ERROR: soap_trigger requires 'soap_zones' in roi.json.")
                print("Run: python src/roi_select.py --soap-zone")
                return
            extra_kwargs["soap_zones"] = soap_zones
            if sink_zones is not None:
                extra_kwargs["sink_zones"] = sink_zones

        pred_df = detect_wash_events(
            video_path=str(clip_path),
            roi=roi,
            params=params,
            show_preview=False,
            **extra_kwargs,
        )
        pred_events = pred_df.to_dict('records') if not pred_df.empty else []
        
        matched_gt = set()
        matched_pred = set()
        
        # 1. Check for TPs and save IoUs
        for i, gt in enumerate(gt_events):
            best_iou = 0
            best_pred_idx = -1
            
            for j, pred in enumerate(pred_events):
                if j in matched_pred: continue
                
                iou = calculate_iou(
                    gt["start_sec"], gt["end_sec"],
                    pred["start_sec"], pred["end_sec"]
                )
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = j
            
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(i)
                matched_pred.add(best_pred_idx)
                iou_scores.append(best_iou)
            else:
                fn += 1
                
        # 2. Remaining predictions are FPs
        for j in range(len(pred_events)):
            if j not in matched_pred:
                fp += 1
                
        results_list.append({
            "video": name,
            "gt_count": len(gt_events),
            "pred_count": len(pred_events),
            "match": len(matched_gt)
        })

    # Summary Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    m_iou = np.mean(iou_scores) if iou_scores else 0
    
    print("\n" + "="*40)
    print(f" EVALUATION SUMMARY ({detector_name})")
    print("="*40)
    print(f"Total Clips processed : {len(clips) - skipped}")
    if skipped:
        print(f"Clips excluded        : {skipped}")
    print(f"Ground Truth Events   : {tp + fn}")
    print(f"Detected Events       : {tp + fp}")
    print("-"*40)
    print(f"True Positives (TP)   : {tp}")
    print(f"False Positives (FP)  : {fp}")
    print(f"False Negatives (FN)  : {fn}")
    print("-"*40)
    print(f"Precision            : {precision:.4f}")
    print(f"Recall               : {recall:.4f}")
    print(f"F1-Score             : {f1:.4f}")
    print(f"Mean IoU (matched)    : {m_iou:.4f}")
    print("="*40)

    # Save per-clip results to CSV
    df_results = pd.DataFrame(results_list)
    csv_path = OUTPUTS_DIR / f"eval_{detector_name}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\nPer-clip results saved: {csv_path}")

    # Save summary metrics to JSON for notebook consumption
    summary = {
        "detector": detector_name,
        "clips_evaluated": len(clips) - skipped,
        "clips_excluded": skipped,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "mean_iou": round(m_iou, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }
    summary_path = OUTPUTS_DIR / f"eval_{detector_name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved        : {summary_path}")

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate hand-wash detection.")
    parser.add_argument("--detector", choices=["baseline", "mediapipe", "soap_trigger"], default="baseline",
                        help="Which detector to evaluate (default: baseline).")
    parser.add_argument("--iou-threshold", type=float, default=0.1,
                        help="Minimum IoU to count as a match (default: 0.1).")
    parser.add_argument("--soap-min-contact", type=float, default=None)
    parser.add_argument("--soap-confirm-window", type=float, default=None)
    parser.add_argument("--soap-sink-min-y", type=float, default=None)
    parser.add_argument("--soap-ignore-top", type=float, default=None)
    parser.add_argument("--soap-min-duration", type=float, default=None)
    parser.add_argument("--soap-min-sink-time", type=float, default=None)
    args = parser.parse_args()

    params = DetectionParams()
    if args.soap_min_contact is not None:
        params.soap_trigger_min_contact_sec = args.soap_min_contact
    if args.soap_confirm_window is not None:
        params.soap_post_trigger_confirm_sec = args.soap_confirm_window
    if args.soap_sink_min_y is not None:
        params.soap_sink_min_y_ratio = args.soap_sink_min_y
    if args.soap_ignore_top is not None:
        params.soap_motion_ignore_top_ratio = args.soap_ignore_top
    if args.soap_min_duration is not None:
        params.soap_min_event_duration_sec = args.soap_min_duration
    if args.soap_min_sink_time is not None:
        params.soap_min_sink_time_sec = args.soap_min_sink_time

    evaluate_performance(iou_threshold=args.iou_threshold, detector_name=args.detector, params=params)
