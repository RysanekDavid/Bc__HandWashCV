"""
Train YOLO classification model for the hand-wash detection project.
"""

from pathlib import Path
from ultralytics import YOLO
import argparse

from config import PROJECT_ROOT, MODELS_DIR, TRAINING_DIR

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO cls model.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=224, help="Image size")
    parser.add_argument("--model", type=str, default=str(MODELS_DIR / "yolo26n-cls.pt"), help="Base model to use")
    parser.add_argument("--name", type=str, default="yolo26n_run", help="Run name")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Define dataset path
    data_dir = PROJECT_ROOT / "datasets" / "yolo_cls"
    
    # Initialize the model
    print(f"Loading {args.model}...")
    model = YOLO(args.model)
    
    # Train the model
    print(f"Starting training on {data_dir}")
    model.train(
        data=str(data_dir),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(TRAINING_DIR),
        name=args.name,
        exist_ok=True, # Overwrite if run exists
    )
    
if __name__ == "__main__":
    main()
