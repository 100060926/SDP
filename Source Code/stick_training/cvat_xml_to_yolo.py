"""Convert a CVAT "CVAT for video 1.1" (or "CVAT for images 1.1") XML export + your own video
into a YOLO training folder.  Works on the free CVAT plan (no images needed from CVAT).

1. In CVAT: Export task dataset -> "CVAT for video 1.1" -> Save images OFF -> download, unzip.
   (Use "CVAT for images 1.1" if the task was made from pictures; then give a folder of those pictures.)
2. Run:
       python cvat_xml_to_yolo.py my_video.mp4 cvat_video1/annotations.xml --out video1_full

Boxes drawn as Shape or Track both work; frames where a track is marked Outside get no box.
Frames in the task that have no box are written with an empty label (= "no stick here").
Check the pictures in <out>/check: the yellow box must sit on the stick.
Use --offset 1 / -1 if every box is one frame early or late.
"""
import argparse, re, sys, shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="the video file (or a folder of pictures for an images task)")
    ap.add_argument("xml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--checks", type=int, default=12)
    ap.add_argument("--no-negatives", action="store_true", help="only write frames that have a box")
    args = ap.parse_args()

    root = ET.parse(args.xml).getroot()
    task = root.find("meta/task")
    if task is None:
        task = root.find("meta/job")
    def meta(tag, default=None):
        if task is None: return default
        el = task.find(tag)
        return el.text.strip() if el is not None and el.text else default
    if task is not None and task.find("original_size") is not None:
        ow = float(task.find("original_size/width").text); oh = float(task.find("original_size/height").text)
    else:
        ow = oh = None
    labels = [l.find("name").text for l in root.findall("meta/task/labels/label")] or ["stick"]
    cls = {n: i for i, n in enumerate(labels)}

    import math
    def env(b):
        x1,y1,x2,y2=(float(b.get(k)) for k in("xtl","ytl","xbr","ybr"))
        r=math.radians(float(b.get("rotation","0") or 0))
        if r==0: return x1,y1,x2,y2
        cx,cy=(x1+x2)/2,(y1+y2)/2
        xs=[];ys=[]
        for px,py in((x1,y1),(x2,y1),(x2,y2),(x1,y2)):
            dx,dy=px-cx,py-cy
            xs.append(cx+dx*math.cos(r)-dy*math.sin(r)); ys.append(cy+dx*math.sin(r)+dy*math.cos(r))
        return min(xs),min(ys),max(xs),max(ys)
    # boxes[frame] = list of (label, xtl, ytl, xbr, ybr)
    boxes = {}
    for tr in root.findall("track"):
        lab = tr.get("label")
        for b in tr.findall("box"):
            if b.get("outside") == "1": continue
            boxes.setdefault(int(b.get("frame")), []).append(
                (lab,)+env(b))
    for im in root.findall("image"):
        for b in im.findall("box"):
            boxes.setdefault(int(im.get("id")), []).append(
                (b.get("label"),)+env(b))
    if not boxes:
        sys.exit("No boxes found in the XML. The boxes were not saved in CVAT, or this is the wrong task.")

    start = int(meta("start_frame", 0) or 0)
    stop = meta("stop_frame")
    m = re.search(r"step=(\d+)", meta("frame_filter", "") or "")
    step = int(m.group(1)) if m else 1
    frames_in_task = None
    if stop is not None:
        frames_in_task = list(range(start, int(stop) + 1, step))
    wanted = sorted(set(boxes) if (args.no_negatives or not frames_in_task)
                    else set(boxes) | set(frames_in_task))
    # CVAT may number frames 0..N-1 within the task instead of by original video frame
    if step > 1 and max(boxes) < (int(stop) - start) // step + 1 and max(boxes) * step + start <= int(stop):
        print(f"Note: frame numbers look task-relative; converting with step={step}, start={start}")
        boxes = {f * step + start: v for f, v in boxes.items()}
        wanted = sorted(set(boxes) | (set() if args.no_negatives else set(frames_in_task or [])))
    wanted = [w + args.offset for w in wanted]
    boxes = {f + args.offset: v for f, v in boxes.items()}

    out = Path(args.out)
    for sub in ("images/Train", "labels/Train", "check"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    (out / "data.yaml").write_text("names:\n" + "".join(f"  {i}: {n}\n" for n, i in cls.items()))

    src = Path(args.source)
    def frames():
        if src.is_dir():
            files = sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            for i in wanted:
                if 0 <= i < len(files): yield i, cv2.imread(str(files[i]))
        else:
            cap = cv2.VideoCapture(str(src))
            if not cap.isOpened(): sys.exit("Could not open the video.")
            n = 0; need = set(wanted); last = max(wanted)
            while n <= last:
                ok, f = cap.read()
                if not ok: break
                if n in need: yield n, f
                n += 1
            cap.release()

    saved = checks = nbox = 0
    for n, frame in frames():
        if frame is None: continue
        h, w = frame.shape[:2]
        sx, sy = (w / ow, h / oh) if ow else (1.0, 1.0)   # boxes are in the task's original size
        name = f"frame_{n:06d}"
        cv2.imwrite(str(out / "images/Train" / f"{name}.jpg"), frame)
        lines, prev = [], frame.copy()
        for lab, x1, y1, x2, y2 in boxes.get(n, []):
            x1, x2, y1, y2 = x1 * sx, x2 * sx, y1 * sy, y2 * sy
            lines.append(f"{cls.get(lab, 0)} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
            cv2.rectangle(prev, (int(x1), int(y1)), (int(x2), int(y2)), (0, 215, 255), 3)
        (out / "labels/Train" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        nbox += len(lines); saved += 1
        if lines and checks < args.checks:
            cv2.imwrite(str(out / "check" / f"{name}.jpg"), prev); checks += 1
    print(f"Wrote {saved} frames and {nbox} boxes to {out}")
    print(f"Now open {out / 'check'} and confirm the yellow boxes sit on the stick.")

if __name__ == "__main__":
    main()
