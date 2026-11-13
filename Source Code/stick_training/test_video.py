"""Run the trained stick detector on a video and save a copy with the boxes drawn.

Usage:
    python test_video.py my_video.mp4
    python test_video.py my_video.mp4 --model stick_best.pt --conf 0.35 --imgsz 960 --show

Output: stick_<video name>.mp4 and a summary of how many frames had a stick.
Use a video that was NOT used for training, or the result will look better than it is.
"""

import argparse
import sys
from pathlib import Path

import cv2

try:
    from ultralytics import YOLO
except Exception as exc:
    sys.exit(f"ultralytics is not installed ({exc}). Run: pip install ultralytics")

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default=str(HERE / "stick_best.pt"))
    ap.add_argument("--conf", type=float, default=0.35, help="raise it for fewer false boxes")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default=None)
    ap.add_argument("--show", action="store_true", help="live window (q to stop)")
    args = ap.parse_args()

    if not Path(args.model).exists():
        sys.exit(f"{args.model} not found. Run train.py first.")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit("Could not open the video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = HERE / f"stick_{Path(args.video).stem}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    model = YOLO(args.model)
    frames = with_stick = boxes_total = 0
    confs = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        kw = dict(conf=args.conf, imgsz=args.imgsz, verbose=False)
        if args.device is not None:
            kw["device"] = args.device
        res = model.predict(frame, **kw)[0]
        n = 0
        if res.boxes is not None:
            for c, xyxy in zip(res.boxes.conf.cpu().tolist(), res.boxes.xyxy.cpu().tolist()):
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 215, 255), 2)
                cv2.putText(frame, f"stick {c:.2f}", (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 215, 255), 1, cv2.LINE_AA)
                confs.append(c)
                n += 1
        frames += 1
        with_stick += 1 if n else 0
        boxes_total += n
        writer.write(frame)
        if args.show:
            cv2.imshow("stick detector (q to stop)", cv2.resize(frame, (w // 2, h // 2)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"\nFrames: {frames}   with a stick: {with_stick} ({100 * with_stick / max(1, frames):.0f}%)   "
          f"boxes: {boxes_total}")
    if confs:
        print(f"Confidence: mean {sum(confs) / len(confs):.2f}, min {min(confs):.2f}")
    print(f"Saved {out_path}  <- watch it and check the box stays on the stick")


if __name__ == "__main__":
    main()
