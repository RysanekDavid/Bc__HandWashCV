"""
Generates YOLO classification dataset using Pseudo-Labeling.
Uses the heuristic detector (soap_trigger_detector) to auto-label an unannotated video.

Usage:
    python src/pseudo_label_dataset.py <video>
    python src/pseudo_label_dataset.py data_clips/2026-02-06/20260127_234623_tp00003.mp4
"""

import argparse
import cv2
import json
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUTS_DIR, DEFAULT_ROI_PATH, DetectionParams, PROJECT_ROOT
from soap_trigger_detector import detect_wash_events

# Parameters
FPS_EXTRACT_WASHING = 2.0
FPS_EXTRACT_NOT_WASHING = 0.2
PAD_X, PAD_Y = 150, 100

def get_padded_crop(frame, zone):
    h, w = frame.shape[:2]
    zx, zy, zw, zh = zone["x"], zone["y"], zone["w"], zone["h"]
    x1 = max(0, zx - PAD_X)
    y1 = max(0, zy - PAD_Y)
    x2 = min(w, zx + zw + PAD_X)
    y2 = min(h, zy + zh + PAD_Y)
    return frame[y1:y2, x1:x2]

def main():
    parser = argparse.ArgumentParser(description="Generate YOLO dataset via pseudo-labeling.")
    parser.add_argument("video", help="Path to input video.")
    parser.add_argument("--roi", default=str(DEFAULT_ROI_PATH), help="ROI JSON path.")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "datasets" / "yolo_cls"),
                        help="Output dataset directory.")
    args = parser.parse_args()

    video_path = Path(args.video)
    roi_path = Path(args.roi)
    out_dir = Path(args.out_dir)

    print(f"Running Teacher model (Heuristic) on {video_path.name}...")
    roi_data = json.load(open(roi_path, encoding="utf-8"))
    soap_zones = roi_data.pop("soap_zones", None)
    roi_data.pop("soap_zone", None)
    sink_zones = roi_data.pop("sink_zones", None)
    roi = roi_data

    params = DetectionParams(
        soap_post_trigger_confirm_sec=0.0,
        post_trigger_min_sink_sec=0.0,
        wash_sec_off=3.0,
    )

    df_events = detect_wash_events(
        str(video_path), roi, soap_zones or [], params,
        show_preview=False, sink_zones=sink_zones
    )

    events = df_events.to_dict('records')
    print(f"Teacher detected {len(events)} events! Extracting frames...")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    frame_interval_wash = int(fps / FPS_EXTRACT_WASHING)
    frame_interval_not_wash = int(fps / FPS_EXTRACT_NOT_WASHING)
    
    frame_idx = 0
    saved_wash = 0
    saved_not_wash = 0
    
    while True:
        ok, frame = cap.read()
        if not ok: break
            
        current_sec = frame_idx / fps
        
        # Find active events at this exact second
        active_stations = set()
        for ev in events:
            if ev["start_sec"] <= current_sec <= ev["end_sec"]:
                active_stations.add(ev["station_id"])
                
        is_any_wash = len(active_stations) > 0
        interval = frame_interval_wash if is_any_wash else frame_interval_not_wash
        
        if frame_idx % interval == 0:
            split = "val" if current_sec > (48 * 60 * 0.8) else "train"
            
            for i, zone in enumerate(sink_zones):
                label = "washing" if i in active_stations else "not_washing"
                crop = get_padded_crop(frame, zone)
                
                label_dir = out_dir / split / label
                label_dir.mkdir(parents=True, exist_ok=True)
                stem = video_path.stem.split("_")[-1]  # e.g. tp00003
                out_file = label_dir / f"{stem}_st{i}_{frame_idx:07d}.jpg"
                cv2.imwrite(str(out_file), crop)
                
                if label == "washing": saved_wash += 1
                else: saved_not_wash += 1
                
        frame_idx += 1
        if frame_idx % (30 * 60) == 0:
            print(f"Processed {current_sec/60:.1f} mins. Wash: {saved_wash}, Not wash: {saved_not_wash}")

    cap.release()
    print(f"\nDone Pseudo-Labeling! Saved {saved_wash} washing and {saved_not_wash} not_washing crops from {video_path.name}.")

if __name__ == "__main__":
    main()
