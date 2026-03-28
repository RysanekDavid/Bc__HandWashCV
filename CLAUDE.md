# Hand-Wash Detection CV Project (Bachelor's Thesis)

## Stack
- Python 3.13, venv in `.venv/`
- OpenCV, MediaPipe, Ultralytics (YOLO), PyTorch
- NumPy, Pandas, Matplotlib, tqdm
- FFmpeg for video processing

## Commands
| Task | Command |
|------|---------|
| Activate venv | `.venv\Scripts\activate` (Windows) |
| Install deps | `pip install -r requirements.txt` |
| Run YOLO detector | `python src/yolo_cls_detector.py <video> --model <weights>` |
| Run soap trigger detector | `python src/soap_trigger_detector.py <video> outputs/roi.json` |
| Evaluate (full video) | `python src/evaluate_full.py <video> --gt <gt_json>` |
| Annotate full video | `python src/annotate_full.py <video>` |
| Generate YOLO dataset | `python src/generate_yolo_dataset.py <video> <gt_json>` |
| Train YOLO | `python src/train_yolo.py --model models/yolo26n-cls.pt --epochs 50` |
| Compare all detectors | `python src/compare_detectors.py <video> --gt <gt_json>` |
| Compute statistics | `python src/compute_statistics.py <detection_csv> --gt <gt_json>` |
| Person tracking | `python src/person_tracker.py <video> --wash-csv <csv>` |

## Project Structure
```
CV_Bc_project/
├── src/                          # All Python scripts
│   ├── config.py                 # Central config (paths, DetectionParams)
│   ├── soap_trigger_detector.py  # Best heuristic detector
│   ├── yolo_cls_detector.py      # YOLO classification detector
│   ├── compare_detectors.py      # Multi-detector comparison
│   ├── compute_statistics.py     # Wash statistics (WHO, duration, gaps)
│   ├── person_tracker.py         # Person tracking + compliance
│   ├── evaluate_full.py          # Full-video evaluation (strict 1:1 IoU)
│   ├── annotate_full.py          # Interactive GT annotation tool
│   ├── generate_yolo_dataset.py  # Extract labeled sink-zone crops
│   ├── train_yolo.py             # YOLO training wrapper
│   └── (legacy: evaluate.py, baseline_motion.py, mediapipe_detector.py, ...)
├── models/                       # All model weights
│   ├── hand_landmarker.task      # MediaPipe hand detection
│   ├── yolov8n.pt                # Person detection (tracking)
│   └── yolo*-cls.pt              # Classification base models
├── outputs/
│   ├── roi.json                  # ROI + soap/sink/exit zones
│   ├── ground_truth/             # GT annotation files
│   ├── evaluation/               # Eval results, comparison JSONs, statistics
│   ├── detections/               # Per-detector event CSVs
│   ├── charts/                   # Visualization PNGs
│   ├── training/                 # YOLO training runs + weights
│   └── debug/                    # Debug clips, event previews
├── notebooks/                    # Jupyter analysis notebooks
├── docs/                         # Thesis docs (goals, handoff)
├── data_clips/                   # Raw video recordings
└── datasets/                     # YOLO training dataset
```

## Key Data
- Videos: `data_clips/2026-02-06/` (48-min recordings, 256 MB each)
- tp00002: annotated (29 events), GT in `outputs/ground_truth/full_video_gt_tp00002.json`
- tp00003: annotated (37 events), GT in `outputs/ground_truth/full_video_gt.json`
- YOLO dataset: `datasets/yolo_cls/{train,val}/{washing,not_washing}/`
- Trained weights: `outputs/training/{yolov8n,yolo11n,yolo26n}_run/weights/best.pt`

## Conventions
- All scripts run from project root with `python src/<script>.py`
- ROI JSON pops `soap_zones`/`sink_zones` keys to get plain ROI dict
- GT JSON format: `{video_path, video_name, last_position_sec, events: [{start_sec, end_sec}]}`
- Detection output: DataFrame with columns `video, start_sec, end_sec, duration_sec, station`

## Do NOT
- Delete or overwrite `full_video_gt_tp00002.json` (backup of tp00002 annotations)
- Run long video processing without `--no-preview` in headless contexts
- Assume `full_video_gt.json` is for tp00002 (it was overwritten with tp00003)
