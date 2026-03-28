"""
Automated Grid Search for Soap Trigger detector parameters.

Usage:
    python src/tune_params.py

Searches over key parameters and reports the best combination
by F1-Score. Results are saved to outputs/grid_search_results.csv.
"""

import sys
import json
import itertools
import pandas as pd
from pathlib import Path
from datetime import datetime

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DetectionParams, OUTPUTS_DIR, EVAL_DIR
from evaluate import evaluate_performance


# ── Search Space ─────────────────────────────────────────────
# Keep the grid small to finish in reasonable time.
# Each combination runs the full 128-clip evaluation (~15s/clip × 128 ≈ 30 min).
# With 12 combos → ~6 hours. Reduce if needed.

GRID = {
    "hand_detection_grace_sec": [0.5, 1.0, 1.5],
    "merge_gap_sec":            [2.0, 3.0, 5.0],
    "soap_min_event_duration_sec": [2.0, 3.0],
}

# Fixed params (not searched)
FIXED = {
    "motion_thresh": 2500,
    "wash_sec_on": 5.0,
    "wash_sec_off": 2.0,
    "soap_trigger_min_contact_sec": 0.0,
    "soap_min_sink_time_sec": 0.0,
}


def run_grid_search():
    """Run all parameter combinations and return results DataFrame."""
    keys = list(GRID.keys())
    values = list(GRID.values())
    combos = list(itertools.product(*values))

    print(f"Grid Search: {len(combos)} combinations to evaluate")
    print(f"Parameters: {keys}")
    print(f"Estimated time: ~{len(combos) * 30 // 60} hours\n")

    results = []

    for i, combo in enumerate(combos):
        # Build params
        params = DetectionParams()
        param_dict = dict(zip(keys, combo))

        # Apply fixed params
        for k, v in FIXED.items():
            setattr(params, k, v)

        # Apply search params
        for k, v in param_dict.items():
            setattr(params, k, v)

        print(f"[{i+1}/{len(combos)}] {param_dict}")

        try:
            summary = evaluate_performance(
                iou_threshold=0.1,
                detector_name="soap_trigger",
                params=params,
            )
            result = {**param_dict, **summary}
            results.append(result)

            print(f"  → P={summary['precision']:.3f}  R={summary['recall']:.3f}  "
                  f"F1={summary['f1_score']:.3f}  IoU={summary['mean_iou']:.3f}  "
                  f"TP={summary['tp']} FP={summary['fp']} FN={summary['fn']}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    df = pd.DataFrame(results)

    if df.empty:
        print("No results!")
        return df

    # Sort by F1, then by Recall (tiebreaker)
    df = df.sort_values(["f1_score", "recall"], ascending=False)

    # Save
    out_path = EVAL_DIR / "grid_search_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to: {out_path}")
    print(f"{'='*60}\n")

    # Top 5
    print("Top 5 configurations by F1-Score:\n")
    display_cols = keys + ["precision", "recall", "f1_score", "mean_iou", "tp", "fp", "fn"]
    print(df[display_cols].head(5).to_string(index=False))

    # Best
    best = df.iloc[0]
    print(f"\n★ Best: F1={best['f1_score']:.4f}  P={best['precision']:.4f}  R={best['recall']:.4f}")
    print(f"  Params: {dict(zip(keys, [best[k] for k in keys]))}")

    return df


if __name__ == "__main__":
    run_grid_search()
