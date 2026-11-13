"""Merge one or more CVAT exports into a single YOLO training dataset.

CVAT export used:  "Ultralytics YOLO Detection 1.0"  with  "Save images"  ON.

Usage (unzip each export into its own folder first):
    python prepare_dataset.py cvat_video1 cvat_video2 cvat_video3
    python prepare_dataset.py exports/* --out dataset

What it does:
  * finds every image and its label file (.txt) in each export
  * images with no label file are kept as "background" (no stick)
  * splits into train / val IN BLOCKS of consecutive frames, so that near-identical
    neighbouring frames never end up one in train and one in val (that would make
    the validation score look much better than it really is)
  * writes dataset/images/{train,val}, dataset/labels/{train,val}, dataset/data.yaml
"""

import argparse
import os
import re
import shutil
import stat
import sys
import time
from pathlib import Path

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def find_label(img: Path, root: Path):
    """CVAT puts labels at labels/<subset>/<name>.txt for images/<subset>/<name>.jpg."""
    rel = img.relative_to(root)
    parts = list(rel.parts)
    cands = []
    for i, p in enumerate(parts[:-1]):
        if p.lower() == "images":
            q = parts.copy()
            q[i] = "labels"
            cands.append(root.joinpath(*q).with_suffix(".txt"))
    cands.append(img.with_suffix(".txt"))
    for c in cands:
        if c.exists():
            return c
    return None


def read_names(root: Path):
    f = root / "data.yaml"
    if not f.exists():
        return None
    names = yaml.safe_load(f.read_text(encoding="utf-8")).get("names")
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return names


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def _force_remove(func, path, exc_info):
    """Clear the read-only flag and retry (Windows)."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def safe_rmtree(path: Path, retries: int = 5):
    """rmtree that tolerates read-only files and brief locks (OneDrive, antivirus, Explorer)."""
    for _ in range(retries):
        try:
            shutil.rmtree(path, onexc=_force_remove)
            return
        except PermissionError:
            time.sleep(1.5)
        if not path.exists():
            return
    sys.exit(f"Could not delete {path}. Close anything using it (Explorer, image viewer, a running "
             f"training job), pause OneDrive sync, or run with --out <new_folder>.")


def main():
    ap =argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exports", nargs="+", help="unzipped CVAT export folders")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--block", type=int, default=10, help="consecutive frames per block (default 10)")
    ap.add_argument("--val-every", type=int, default=5, help="every Nth block goes to validation (default 5 = 20%%)")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        safe_rmtree(out)
    for sub in ("train", "val"):
        (out / "images" / sub).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sub).mkdir(parents=True, exist_ok=True)

    names = None
    counts = {"train": [0, 0], "val": [0, 0]}  # [images, boxes]
    background = {"train": 0, "val": 0}

    for src_i, src in enumerate(args.exports):
        root = Path(src)
        if not root.is_dir():
            sys.exit(f"{src} is not a folder. Unzip each CVAT export first.")
        n = read_names(root)
        if n:
            if names and n != names:
                sys.exit(f"Label names differ between exports: {names} vs {n}")
            names = n
        imgs = sorted([p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT], key=natural_key)
        if not imgs:
            sys.exit(f"No images found in {src}. Did you turn on 'Save images' when exporting?")
        tag = re.sub(r"\W+", "_", root.name) or f"src{src_i}"
        for k, img in enumerate(imgs):
            block = k // args.block
            sub = "val" if (block % args.val_every == args.val_every - 1) else "train"
            stem = f"{tag}_{k:05d}"
            shutil.copy2(img, out / "images" / sub / f"{stem}{img.suffix.lower()}")
            lab = find_label(img, root)
            dst = out / "labels" / sub / f"{stem}.txt"
            text = lab.read_text(encoding="utf-8").strip() if lab else ""
            dst.write_text(text + ("\n" if text else ""), encoding="utf-8")
            nbox = len([l for l in text.splitlines() if l.strip()])
            counts[sub][0] += 1
            counts[sub][1] += nbox
            if nbox == 0:
                background[sub] += 1

    names = names or ["stick"]
    (out / "data.yaml").write_text(
        yaml.safe_dump({"path": str(out.resolve()), "train": "images/train", "val": "images/val",
                        "names": {i: n for i, n in enumerate(names)}}, sort_keys=False),
        encoding="utf-8")

    print(f"Classes: {names}")
    for sub in ("train", "val"):
        print(f"{sub:5}: {counts[sub][0]:5d} images, {counts[sub][1]:5d} boxes, "
              f"{background[sub]} images without boxes")
    print(f"Wrote {out / 'data.yaml'}")
    if counts["val"][1] == 0:
        print("WARNING: the validation set has no boxes. Label more frames or lower --val-every.")
    if counts["train"][1] < 50:
        print("WARNING: fewer than 50 training boxes. Label more frames for a usable model.")


if __name__ == "__main__":
    main()
