"""Train a YOLO stick detector on the dataset made by prepare_dataset.py.

Usage:
    python train.py                       # defaults: yolo11n, 100 epochs, 960 px
    python train.py --epochs 150 --imgsz 1280 --device 0

A stick is thin and often small in a wide camera view, so the default image size is
960 (larger than the usual 640). Use --imgsz 640 if training is too slow or memory runs out.

Output: runs/stick/weights/best.pt  (also copied to stick_best.pt next to this script)
"""

import argparse
import shutil
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception as exc:
    sys.exit(f"ultralytics is not installed ({exc}). Run: pip install ultralytics")

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "dataset" / "data.yaml"))
    ap.add_argument("--model", default="yolo11n.pt", help="start from this pretrained model")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = pick automatically")
    ap.add_argument("--device", default=None, help='"cpu" or 0 for the first GPU (default: auto)')
    ap.add_argument("--patience", type=int, default=30, help="stop early if no improvement")
    args = ap.parse_args()

    if not Path(args.data).exists():
        sys.exit(f"{args.data} not found. Run prepare_dataset.py first.")

    model = YOLO(args.model)
    kw = dict(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
              patience=args.patience, project=str(HERE / "runs"), name="stick", exist_ok=True,
              # augmentation that suits a fixed CCTV view with a thin object
              fliplr=0.5, flipud=0.0, degrees=10.0, scale=0.4, hsv_v=0.4, mosaic=1.0)
    if args.device is not None:
        kw["device"] = args.device
    model.train(**kw)

    best = HERE / "runs" / "stick" / "weights" / "best.pt"
    if best.exists():
        shutil.copy2(best, HERE / "stick_best.pt")
        metrics = YOLO(str(best)).val(data=args.data, imgsz=args.imgsz, verbose=False)
        print(f"\nSaved {HERE / 'stick_best.pt'}")
        print(f"Validation  precision {metrics.box.mp:.2f}  recall {metrics.box.mr:.2f}  "
              f"mAP50 {metrics.box.map50:.2f}")
        print("Next: python test_video.py <your video>")
    else:
        print("Training finished but best.pt was not found; check the runs/stick folder.")


if __name__ == "__main__":
    main()
