import argparse
import cgi
import html
import json
import math
import re
import threading
import time
import traceback
import urllib.parse
import uuid
from collections import OrderedDict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    from ultralytics import YOLO

    ULTRALYTICS_IMPORT_ERROR = ""
except Exception as import_exc:
    YOLO = None
    ULTRALYTICS_IMPORT_ERROR = f"{type(import_exc).__name__}: {import_exc}"

APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parent
UPLOAD_DIR = APP_ROOT / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_FINETUNED_MODEL_PATH = REPO_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"

STREAMS = {}
STREAMS_LOCK = threading.Lock()
ANALYZERS = {}
ANALYZER_LOCK = threading.Lock()

CAMERA_PRESETS = OrderedDict(
    [
        ("Camera 1", "rtsp://admin:Xray%4012345%3B@10.10.184.220:554/h264Preview_01_sub"),
        ("Camera 2", "rtsp://admin:Xray%4012345%3B@10.10.185.38:554/h264Preview_01_sub"),
        ("Camera 3", "rtsp://admin:Xray%4012345%3B@10.10.187.125:554/h264Preview_01_sub"),
    ]
)

DEFAULT_CAMERA_URL = CAMERA_PRESETS["Camera 3"]
DEFAULT_THRESHOLD = 0.6 # lowered from 0.50
PERSON_MODEL_NAME = "yolo11n.pt"
DEFAULT_VIRTUAL_LINE = (0.22, 0.307, 0.366, 0.256)
LINE_CROSSING_DEADBAND_PX = 12
TRACK_CLEANUP_FRAMES = 90
TRACK_CROSSING_COOLDOWN_FRAMES = 12
TRACK_CROSSING_CONFIRMATION_FRAMES = 10
TRACK_CROSSING_MIN_DISTANCE_RATIO = 0.05
TRACK_SIDE_REARM_DISTANCE_RATIO = 0.08
TRACK_CROSSING_PENDING_TIMEOUT_FRAMES = 5
TRACK_CROSSING_APPROACH_FRAMES = 2
TRACK_CROSSING_LOST_CONFIRMATION_FRAMES = 10
PERSON_CROSSING_POINT_VERTICAL_RATIO = 0.85
TRACK_CROSSING_ENDPOINT_EXTENSION_RATIO = 0.15

SEGMENT_LEN = 1
SELECTED_FRAMES_PER_SEGMENT = 1
INFERENCE_EVERY_SAMPLED_FRAMES = 1
APP_DEFAULT_FRAME_STRIDE = 1

APP_CONFIDENCE_THRESHOLD = 0.6  # lowered from 0.25
APP_IOU_THRESHOLD = 0.45
APP_IMAGE_SIZE = 640
APP_INSTRUCT_PATH = ""

# ---------------------------------------------------------------------------
# Temporal smoothing parameters
# ---------------------------------------------------------------------------
ALERT_COOLDOWN_FRAMES = 30      # stay in danger for at least this many sampled frames after last detection
CONFIRMATION_WINDOW = 2         # need at least this many detections in recent history to trigger alert
CONFIRMATION_HISTORY = 4        # size of the sliding window used for confirmation counting
SCORE_EMA_ALPHA = 0.40          # exponential moving average weight (higher = more reactive)

PREVIEW_WIDTH = 960
PREVIEW_HEIGHT = 540

GUN_KEYWORDS = {
    "gun",
    "pistol",
    "handgun",
    "firearm",
    "rifle",
    "shotgun",
    "revolver",
    "weapon",
}
KNIFE_KEYWORDS = {"knife", "blade", "dagger", "machete", "cutter", "boxcutter", "sword"}


def _discover_model_choices():
    candidates = []

    # 1. Explicit finetuned filenames — highest priority
    explicit_finetuned = [
        REPO_ROOT / "yolo_finetuned.pt",
        REPO_ROOT / "Yolo_finetuned.pt",
        APP_ROOT / "yolo_finetuned.pt",
        APP_ROOT / "Yolo_finetuned.pt",
    ]
    for finetuned_path in explicit_finetuned:
        if finetuned_path.exists() and finetuned_path.is_file():
            candidates.append(str(finetuned_path.resolve()))

    # 2. Training run checkpoints — second priority
    tuned_best = REPO_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
    if tuned_best.exists() and tuned_best.is_file():
        candidates.append(str(tuned_best.resolve()))
    for best_ckpt in sorted(
        (REPO_ROOT / "runs" / "detect").glob("*/weights/best.pt"), reverse=True
    ):
        if best_ckpt.is_file():
            candidates.append(str(best_ckpt.resolve()))
    # Also search APP_ROOT runs folder (script and runs in same directory)
    for best_ckpt in sorted(
        (APP_ROOT / "runs" / "detect").glob("*/weights/best.pt"), reverse=True
    ):
        if best_ckpt.is_file():
            candidates.append(str(best_ckpt.resolve()))

    # 3. Only scan for generic *.pt files if NO finetuned/run model was found.
    #    This prevents stray pretrained weights (yolo12m.pt, yolov8n.pt, etc.)
    #    sitting in the working directory from jumping ahead of the correct model.
    if not candidates:
        print(
            "[model-discovery] No finetuned/run weights found. "
            "Scanning for any *.pt files as fallback.",
            flush=True,
        )
        for root in [APP_ROOT, REPO_ROOT]:
            for pattern in ["yolo12*.pt", "yolov12*.pt", "*knife*gun*.pt", "*.pt"]:
                for path in sorted(root.glob(pattern)):
                    if path.is_file():
                        candidates.append(str(path.resolve()))

    # Deduplicate while preserving order
    deduped = []
    seen = set()
    for item in candidates:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    # Always keep yolo12n.pt as a last-resort fallback (will be downloaded if needed)
    if not any("yolo12n.pt" in p.lower() for p in deduped):
        deduped.append("yolo12n.pt")

    return deduped


MODEL_CHOICES = _discover_model_choices()
if DEFAULT_FINETUNED_MODEL_PATH.exists() and DEFAULT_FINETUNED_MODEL_PATH.is_file():
    DEFAULT_MODEL_NAME = str(DEFAULT_FINETUNED_MODEL_PATH.resolve())
    if DEFAULT_MODEL_NAME not in MODEL_CHOICES:
        MODEL_CHOICES = [DEFAULT_MODEL_NAME] + MODEL_CHOICES
else:
    DEFAULT_MODEL_NAME = MODEL_CHOICES[0] if MODEL_CHOICES else "yolo12n.pt"

MODEL_PRESETS = OrderedDict(
    [
        ("model1_finetuned", {"label": "model 1 finetuned (CLI)", "target": DEFAULT_MODEL_NAME}),
    ]
)
DEFAULT_MODEL_PRESET_KEY = "model1_finetuned"


def _resolve_input_path(raw_path):
    candidate = Path(str(raw_path or "").strip()).expanduser()
    if candidate.is_absolute():
        return candidate
    # Try APP_ROOT first — most intuitive when running from the script directory
    app_relative = (APP_ROOT / candidate).resolve()
    if app_relative.exists():
        return app_relative
    # Then try REPO_ROOT (one level up)
    repo_relative = (REPO_ROOT / candidate).resolve()
    if repo_relative.exists():
        return repo_relative
    # Return APP_ROOT-based path even if not found so error messages are clear
    return app_relative


def _resolve_model_name(model_name):
    candidate_name = str(model_name or "").strip()
    if not candidate_name:
        return DEFAULT_MODEL_NAME
    path = _resolve_input_path(candidate_name)
    if path.exists() and path.is_file():
        return str(path)
    # Warn loudly instead of silently falling back to a wrong model
    if candidate_name != DEFAULT_MODEL_NAME:
        print(f"[WARNING] Model path not found: '{candidate_name}'", flush=True)
        print(f"[WARNING] Tried resolved path: {path}", flush=True)
        print(f"[WARNING] Falling back to DEFAULT_MODEL_NAME: {DEFAULT_MODEL_NAME}", flush=True)
    return DEFAULT_MODEL_NAME


def _resolve_model_selection(raw_model):
    key = str(raw_model or "").strip()
    preset = MODEL_PRESETS.get(key)
    if preset is not None:
        return _resolve_model_name(preset["target"])
    return _resolve_model_name(key)


def _is_valid_model_selection(model_name):
    candidate = str(model_name or "").strip()
    if not candidate:
        return False
    if candidate in MODEL_CHOICES:
        return True
    path = _resolve_input_path(candidate)
    if path.exists() and path.is_file() and path.suffix.lower() in {".pt", ".onnx", ".engine"}:
        return True
    return bool(candidate.lower().endswith((".pt", ".onnx", ".engine")))


def _normalize_label(text):
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


# Simplified direct lookup for the two known classes
CLASS_TO_WEAPON_TYPE = {
    "gun": "gun",
    "knife": "knife",
}


def map_to_weapon_type(label_name):
    normalized = _normalize_label(label_name)
    if not normalized:
        return None
    # Direct lookup first
    direct = CLASS_TO_WEAPON_TYPE.get(normalized)
    if direct is not None:
        return direct
    # Fallback fuzzy match
    for key in GUN_KEYWORDS:
        if key in normalized:
            return "gun"
    for key in KNIFE_KEYWORDS:
        if key in normalized:
            return "knife"
    return None


def stable_output_label(pred):
    return "abnormal" if pred == 1 else "normal"


def _compute_iou(box_a, box_b):
    x0 = max(box_a[0], box_b[0])
    y0 = max(box_a[1], box_b[1])
    x1 = min(box_a[2], box_b[2])
    y1 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x1 - x0)
    inter_h = max(0.0, y1 - y0)
    inter_area = inter_w * inter_h
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _dedupe_overlapping_weapon_detections(detections, iou_threshold=0.60):
    if len(detections) <= 1:
        return detections
    sorted_dets = sorted(detections, key=lambda item: -item["confidence"])
    keep = []
    for det in sorted_dets:
        bbox = det["bbox"]
        if any(_compute_iou(bbox, kept["bbox"]) >= iou_threshold for kept in keep):
            continue
        keep.append(det)
    return keep


def format_status_label(pred, anomaly_type, is_cooldown=False):
    if pred == 1:
        return f"DANGER DETECTED ({anomaly_type.upper()})"
    return "SCENE IS NORMAL"


def sample_recent_uniform(frames_bgr, k=8):
    if not frames_bgr:
        return []
    n = len(frames_bgr)
    if n <= k:
        idx = list(range(n))
    else:
        idx = np.linspace(0, n - 1, k, dtype=np.int32).tolist()
    return [frames_bgr[i] for i in idx]


def encode_preview(
    frame_bgr,
    status_text,
    score,
    threshold,
    is_danger=False,
    is_cooldown=False,
    cooldown_remaining=0,
    show_status_banner=True,
):
    preview = cv2.resize(frame_bgr, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
    danger = bool(is_danger)
    if not danger:
        overlay_color = (24, 28, 36)
        text_color = (70, 240, 120)
    elif is_cooldown:
        # Keep cooldown carry-over in the same red danger palette.
        overlay_color = (20, 20, 95)
        text_color = (70, 70, 255)
    else:
        overlay_color = (20, 20, 95)
        text_color = (70, 70, 255)
    if show_status_banner:
        cv2.rectangle(preview, (0, 0), (PREVIEW_WIDTH, 110), overlay_color, -1)
        cv2.putText(preview, str(status_text or "Monitoring stream"), (24, 54), cv2.FONT_HERSHEY_DUPLEX, 1.0, text_color, 2, cv2.LINE_AA)
        if is_cooldown and cooldown_remaining > 0:
            cd_text = f"Cooldown: {cooldown_remaining} frames remaining"
            cv2.putText(preview, cd_text, (24, 90), cv2.FONT_HERSHEY_DUPLEX, 0.6, text_color, 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("Could not encode preview frame.")
    return encoded.tobytes()


def build_analysis_mosaic(frames_bgr, title="LAST ANALYZED SEGMENT"):
    if not frames_bgr:
        return None
    n = len(frames_bgr)
    cols = 3 if n > 3 else n
    cols = max(1, cols)
    rows = int(math.ceil(n / cols))
    tile_w, tile_h = 320, 180
    header_h = 56
    canvas = np.zeros((header_h + rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    canvas[:] = (12, 20, 30)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], header_h), (25, 35, 50), -1)
    cv2.putText(canvas, title, (18, 38), cv2.FONT_HERSHEY_DUPLEX, 0.85, (220, 240, 255), 2, cv2.LINE_AA)
    for idx, frame in enumerate(frames_bgr):
        r = idx // cols
        c = idx % cols
        x0 = c * tile_w
        y0 = header_h + r * tile_h
        tile = cv2.resize(frame, (tile_w, tile_h))
        cv2.rectangle(tile, (0, 0), (tile_w - 1, tile_h - 1), (90, 110, 130), 1)
        cv2.putText(tile, f"F{idx + 1}", (12, 28), cv2.FONT_HERSHEY_DUPLEX, 0.8, (230, 245, 255), 2, cv2.LINE_AA)
        canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
    ok, encoded = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None
    return encoded.tobytes()


class YOLOWeaponAnalyzer:
    def __init__(self, model_name, conf_threshold=0.15, iou_threshold=0.45, img_size=640):
        self.model_name = _resolve_model_name(model_name)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.img_size = int(img_size)
        self.device_arg = 0 if (torch is not None and torch.cuda.is_available()) else "cpu"
        self.model = None
        self.lock = threading.Lock()
        self.class_filter_ids = None
        self.label_lookup = {}
        self.supported_weapon_types = set()

    def _ensure_model(self):
        if YOLO is None:
            raise RuntimeError(
                "ultralytics is not installed. Install with `pip install ultralytics`.\n"
                f"Import error: {ULTRALYTICS_IMPORT_ERROR}"
            )
        with self.lock:
            if self.model is not None:
                return
            self.model = YOLO(self.model_name)
            names = getattr(self.model, "names", {})
            if isinstance(names, dict):
                self.label_lookup = {int(k): str(v) for k, v in names.items()}
            elif isinstance(names, list):
                self.label_lookup = {i: str(v) for i, v in enumerate(names)}
            else:
                self.label_lookup = {}
            class_ids = []
            supported_types = set()
            for cls_id, name in self.label_lookup.items():
                weapon_type = map_to_weapon_type(name)
                if weapon_type is not None:
                    class_ids.append(cls_id)
                    supported_types.add(weapon_type)
            self.class_filter_ids = sorted(set(class_ids)) if class_ids else None
            self.supported_weapon_types = supported_types
            print(
                f"[yolo] model={self.model_name} supports_weapon_types={sorted(self.supported_weapon_types)} "
                f"class_filter_ids={self.class_filter_ids or []}",
                flush=True,
            )
            if not self.supported_weapon_types:
                print(
                    f"[WARNING] Model has {len(self.label_lookup)} classes but NONE match gun/knife.",
                    flush=True,
                )
                print(f"[WARNING] Model classes: {list(self.label_lookup.values())}", flush=True)
                print("[WARNING] You may have loaded the wrong weights file!", flush=True)
            else:
                print(
                    f"[OK] Weapon classes confirmed: {sorted(self.supported_weapon_types)} "
                    f"-> class IDs {self.class_filter_ids}",
                    flush=True,
                )

    @staticmethod
    def _draw_detections(frame_bgr, detections):
        out = frame_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            if det["weapon_type"] == "gun":
                color = (48, 60, 255)
            elif det["weapon_type"] == "knife":
                color = (0, 160, 255)
            else:
                color = (95, 230, 120)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                out,
                f"{det['class_name']} {det['confidence']:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.62,
                color,
                2,
                cv2.LINE_AA,
            )
        return out

    def infer_binary(self, frames_bgr, show_boxes=False):
        self._ensure_model()
        if not frames_bgr:
            return {
                "label": 0,
                "anomaly_type": "normal",
                "output_label": "normal",
                "short_description": "No frames available for inference.",
                "reason_code": "no_frames",
                "score": 0.0,
                "all_detection_count": 0,
                "weapon_detection_count": 0,
                "raw_text": "",
                "analysis_mosaic": None,
                "last_annotated_frame": None,
            }

        all_detections = []
        weapon_detections = []
        annotated_frames = []
        for frame_idx, frame in enumerate(frames_bgr, start=1):
            result = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                device=self.device_arg,
                verbose=False,
                classes=self.class_filter_ids if self.class_filter_ids else None,
            )[0]
            frame_dets = []
            boxes = getattr(result, "boxes", None)
            if boxes is not None:
                for box in boxes:
                    try:
                        cls_id = int(box.cls[0].item())
                    except Exception:
                        cls_id = int(box.cls[0])
                    try:
                        conf = float(box.conf[0].item())
                    except Exception:
                        conf = float(box.conf[0])
                    label_name = self.label_lookup.get(cls_id, str(cls_id))
                    weapon_type = map_to_weapon_type(label_name)
                    xyxy = box.xyxy[0].tolist()
                    det_item = {
                        "frame_index": frame_idx,
                        "class_id": cls_id,
                        "class_name": label_name,
                        "weapon_type": weapon_type or "",
                        "confidence": conf,
                        "bbox": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
                    }
                    frame_dets.append(det_item)
                    if weapon_type is not None:
                        weapon_detections.append(det_item)
            all_detections.extend(frame_dets)
            weapon_frame_dets = [d for d in frame_dets if d["weapon_type"] in ("gun", "knife")]
            if show_boxes:
                annotated_frames.append(self._draw_detections(frame, weapon_frame_dets))
            else:
                annotated_frames.append(frame.copy())

        weapon_detections = _dedupe_overlapping_weapon_detections(weapon_detections, iou_threshold=0.60)

        types = {det["weapon_type"] for det in weapon_detections}
        if "gun" in types and "knife" in types:
            anomaly_type = "gun,knife"
        elif "gun" in types:
            anomaly_type = "gun"
        elif "knife" in types:
            anomaly_type = "knife"
        else:
            anomaly_type = "normal"
        pred = 1 if anomaly_type != "normal" else 0
        score = max((float(det["confidence"]) for det in weapon_detections), default=0.0)

        object_counts = {}
        for det in all_detections:
            name = det["class_name"]
            object_counts[name] = object_counts.get(name, 0) + 1
        top_objects = sorted(object_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:12]

        if pred == 1:
            gun_count = sum(1 for det in weapon_detections if det["weapon_type"] == "gun")
            knife_count = sum(1 for det in weapon_detections if det["weapon_type"] == "knife")
            short_description = (
                f"Detected objects={len(all_detections)}; weapon hits: gun={gun_count}, "
                f"knife={knife_count}, max_weapon_conf={score:.2f}."
            )
            reason_code = "visible_weapon"
        else:
            short_description = f"Detected objects={len(all_detections)}; no gun/knife detected."
            reason_code = "no_visible_weapon"

        raw_payload = {
            "frames_analyzed": len(frames_bgr),
            "model": self.model_name,
            "detect_all_objects": True,
            "all_object_counts": object_counts,
            "top_objects": [{"class_name": name, "count": count} for name, count in top_objects],
            "all_detections": all_detections[:200],
            "weapon_detections": weapon_detections[:80],
        }
        return {
            "label": pred,
            "anomaly_type": anomaly_type,
            "output_label": stable_output_label(pred),
            "short_description": short_description,
            "reason_code": reason_code,
            "score": score,
            "all_detection_count": len(all_detections),
            "weapon_detection_count": len(weapon_detections),
            "raw_text": json.dumps(raw_payload, ensure_ascii=True),
            "analysis_mosaic": None,
            # Return the last raw frame so stream_worker can use it as the live preview
            "last_annotated_frame": annotated_frames[-1] if annotated_frames else None,
        }


def _line_signed_distance(point, line):
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    line_length = math.hypot(dx, dy)
    if line_length == 0:
        return 0.0
    return (dx * (point[1] - y1) - dy * (point[0] - x1)) / line_length


def _crosses_virtual_line_segment(previous_point, current_point, line):
    previous_distance = _line_signed_distance(previous_point, line)
    current_distance = _line_signed_distance(current_point, line)
    if previous_distance == current_distance:
        return False
    fraction = previous_distance / (previous_distance - current_distance)
    if fraction < 0.0 or fraction > 1.0:
        return False
    crossing_x = previous_point[0] + fraction * (current_point[0] - previous_point[0])
    crossing_y = previous_point[1] + fraction * (current_point[1] - previous_point[1])
    line_x1, line_y1, line_x2, line_y2 = line
    line_dx = line_x2 - line_x1
    line_dy = line_y2 - line_y1
    line_length_squared = line_dx * line_dx + line_dy * line_dy
    if line_length_squared == 0:
        return False
    projection = (
        (crossing_x - line_x1) * line_dx + (crossing_y - line_y1) * line_dy
    ) / line_length_squared
    return 0.0 <= projection <= 1.0


def _point_near_virtual_line_segment(
    point, line, max_distance, endpoint_extension=0.0
):
    line_x1, line_y1, line_x2, line_y2 = line
    line_dx = line_x2 - line_x1
    line_dy = line_y2 - line_y1
    line_length_squared = line_dx * line_dx + line_dy * line_dy
    if line_length_squared == 0:
        return False
    projection = (
        (point[0] - line_x1) * line_dx + (point[1] - line_y1) * line_dy
    ) / line_length_squared
    return (
        -endpoint_extension <= projection <= 1.0 + endpoint_extension
        and abs(_line_signed_distance(point, line)) <= max_distance
    )


def update_person_flow(tracks, detections, frame_index, line, inside_side):
    entries = 0
    exits = 0
    event = ""
    inside_sign = 1 if inside_side == "below" else -1
    line_length = math.hypot(line[2] - line[0], line[3] - line[1])
    minimum_crossing_distance = max(
        LINE_CROSSING_DEADBAND_PX * 2,
        line_length * TRACK_CROSSING_MIN_DISTANCE_RATIO,
    )
    rearm_distance = max(
        LINE_CROSSING_DEADBAND_PX * 2,
        line_length * TRACK_SIDE_REARM_DISTANCE_RATIO,
    )
    near_crossing_distance = max(
        LINE_CROSSING_DEADBAND_PX * 2,
        line_length * 0.08,
    )

    for detection in detections:
        track_id = detection["track_id"]
        if track_id is None:
            continue
        x1, y1, x2, y2 = detection["bbox"]
        point = (
            (x1 + x2) / 2.0,
            y1 + (y2 - y1) * PERSON_CROSSING_POINT_VERTICAL_RATIO,
        )
        distance = _line_signed_distance(point, line)
        side = (
            1
            if distance > LINE_CROSSING_DEADBAND_PX
            else -1
            if distance < -LINE_CROSSING_DEADBAND_PX
            else 0
        )
        previous = tracks.get(track_id)
        if previous is None:
            previous = {
                "last_crossing": -TRACK_CLEANUP_FRAMES,
                "side": side,
                "point": point,
                "last_side": side,
                "candidate_side": 0,
                "candidate_count": 0,
                "candidate_last_seen": frame_index,
                "candidate_max_distance": 0.0,
                "rearmed_since_crossing": False,
                "distance": distance,
                "approach_frames": 0,
                "approached_line": False,
            }
        else:
            previous_distance = previous.get("distance", distance)
            stable_side = previous["side"]
            if side and stable_side and side == stable_side:
                if abs(distance) < abs(previous_distance) - 1:
                    previous["approach_frames"] += 1
                elif abs(distance) > abs(previous_distance) + 3:
                    previous["approach_frames"] = 0
                    if abs(distance) > near_crossing_distance + LINE_CROSSING_DEADBAND_PX:
                        previous["approached_line"] = False
                if (
                    previous["approach_frames"] >= TRACK_CROSSING_APPROACH_FRAMES
                    and abs(distance) <= near_crossing_distance
                ):
                    previous["approached_line"] = True
            elif side and stable_side and side != stable_side:
                previous["approach_frames"] = 0
                previous["approached_line"] = False

        if previous is not None and side:
            stable_side = previous["side"]
            candidate_side = previous.get("candidate_side", 0)
            if stable_side == 0:
                if candidate_side == side:
                    previous["candidate_count"] += 1
                    previous["candidate_last_seen"] = frame_index
                    previous["candidate_max_distance"] = max(
                        previous["candidate_max_distance"], abs(distance)
                    )
                elif _point_near_virtual_line_segment(
                    previous["point"],
                    line,
                    near_crossing_distance,
                    TRACK_CROSSING_ENDPOINT_EXTENSION_RATIO,
                ):
                    previous["candidate_side"] = side
                    previous["candidate_count"] = 1
                    previous["candidate_last_seen"] = frame_index
                    previous["candidate_max_distance"] = abs(distance)
            elif side == stable_side:
                previous["candidate_side"] = 0
                previous["candidate_count"] = 0
                previous["candidate_max_distance"] = 0.0
            else:
                if candidate_side != side:
                    crossed_line = _crosses_virtual_line_segment(
                        previous["point"], point, line
                    )
                    departed_deadband = (
                        previous.get("last_side", stable_side) == 0
                        and _point_near_virtual_line_segment(
                            previous["point"], line, LINE_CROSSING_DEADBAND_PX
                        )
                    )
                    if crossed_line or departed_deadband or candidate_side:
                        previous["candidate_side"] = side
                        previous["candidate_count"] = 1
                        previous["candidate_last_seen"] = frame_index
                        previous["candidate_max_distance"] = abs(distance)
                else:
                    previous["candidate_count"] += 1
                    previous["candidate_last_seen"] = frame_index
                    previous["candidate_max_distance"] = max(
                        previous["candidate_max_distance"], abs(distance)
                    )

                if (
                    previous["candidate_side"] == side
                    and previous["last_crossing"] >= 0
                    and not previous["rearmed_since_crossing"]
                    and frame_index - previous["last_crossing"]
                    <= TRACK_CROSSING_COOLDOWN_FRAMES
                    and abs(distance) >= rearm_distance
                    and previous["candidate_count"]
                    < TRACK_CROSSING_CONFIRMATION_FRAMES
                ):
                    previous["side"] = side
                    previous["candidate_side"] = 0
                    previous["candidate_count"] = 0
                    previous["candidate_max_distance"] = 0.0
                    previous["rearmed_since_crossing"] = True
                elif (
                    previous["candidate_side"] == side
                    and previous["candidate_count"]
                    >= TRACK_CROSSING_CONFIRMATION_FRAMES
                    and abs(distance) >= minimum_crossing_distance
                    and frame_index - previous["last_crossing"]
                    >= TRACK_CROSSING_COOLDOWN_FRAMES
                ):
                    if side == inside_sign:
                        entries += 1
                        event = "entry"
                    else:
                        exits += 1
                        event = "exit"
                    previous["side"] = side
                    previous["last_crossing"] = frame_index
                    previous["candidate_side"] = 0
                    previous["candidate_count"] = 0
                    previous["candidate_max_distance"] = 0.0
                    previous["rearmed_since_crossing"] = False
                    previous["approached_line"] = False

            if (
                previous["side"] == 0
                and previous["candidate_side"] == side
                and previous["candidate_count"] >= TRACK_CROSSING_CONFIRMATION_FRAMES
                and abs(distance) >= minimum_crossing_distance
            ):
                if side == inside_sign:
                    entries += 1
                    event = "entry"
                else:
                    exits += 1
                    event = "exit"
                previous["side"] = side
                previous["last_crossing"] = frame_index
                previous["candidate_side"] = 0
                previous["candidate_count"] = 0
                previous["candidate_max_distance"] = 0.0
                previous["rearmed_since_crossing"] = False
                previous["approached_line"] = False

        previous["last_seen"] = frame_index
        previous["last_side"] = side
        previous["distance"] = distance
        if side == 0 and previous["candidate_side"]:
            previous.setdefault("candidate_last_seen", frame_index)
        if side:
            previous["point"] = point
        tracks[track_id] = previous

    for track in tracks.values():
        if (
            track["candidate_side"]
            and frame_index - track["candidate_last_seen"]
            >= TRACK_CROSSING_PENDING_TIMEOUT_FRAMES
            and track["side"] != 0
            and track["candidate_count"] >= 3
            and track["candidate_max_distance"] >= minimum_crossing_distance
        ):
            if track["candidate_side"] == inside_sign:
                entries += 1
                event = "entry"
            else:
                exits += 1
                event = "exit"
            track["side"] = track["candidate_side"]
            track["last_crossing"] = frame_index
            track["candidate_side"] = 0
            track["candidate_count"] = 0
            track["candidate_max_distance"] = 0.0
            track["rearmed_since_crossing"] = False

        if (
            not track.get("crossing_lost_confirmed", False)
            and frame_index - track["last_seen"]
            >= TRACK_CROSSING_LOST_CONFIRMATION_FRAMES
            and track["side"] != 0
            and track.get("last_side") == track["side"]
            and track.get("approached_line", False)
            and abs(track.get("distance", 0.0)) > LINE_CROSSING_DEADBAND_PX
            and abs(track.get("distance", float("inf"))) <= near_crossing_distance
            and _point_near_virtual_line_segment(
                track["point"], line, near_crossing_distance
            )
        ):
            if track["side"] == inside_sign:
                exits += 1
                event = "exit"
            else:
                entries += 1
                event = "entry"
            track["side"] = -track["side"]
            track["last_crossing"] = frame_index
            track["crossing_lost_confirmed"] = True
            track["approached_line"] = False
            track["candidate_side"] = 0
            track["candidate_count"] = 0

    for track_id in [
        track_id
        for track_id, track in tracks.items()
        if frame_index - track["last_seen"] > TRACK_CLEANUP_FRAMES
    ]:
        del tracks[track_id]
    return entries, exits, event


class YOLOPersonFlowTracker:
    def __init__(self, model_name):
        if YOLO is None:
            raise RuntimeError(
                "ultralytics is not installed. Install with `pip install ultralytics`.\n"
                f"Import error: {ULTRALYTICS_IMPORT_ERROR}"
            )
        self.model = YOLO(model_name)
        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            name_items = names.items()
        elif isinstance(names, list):
            name_items = enumerate(names)
        else:
            name_items = []
        person_ids = [
            int(class_id)
            for class_id, name in name_items
            if _normalize_label(name) == "person"
        ]
        if not person_ids:
            raise RuntimeError(
                f"The person detection model '{model_name}' has no 'person' class."
            )
        self.person_class_id = person_ids[0]
        self.device_arg = 0 if (torch is not None and torch.cuda.is_available()) else "cpu"

    def track(self, frame):
        result = self.model.track(
            source=frame,
            persist=True,
            classes=[self.person_class_id],
            conf=0.25,
            iou=APP_IOU_THRESHOLD,
            imgsz=APP_IMAGE_SIZE,
            device=self.device_arg,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]
        detections = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections
        for box in boxes:
            coordinates = box.xyxy[0].tolist()
            confidence = float(box.conf[0].item())
            track_id = None
            box_id = getattr(box, "id", None)
            if box_id is not None:
                track_id = int(box_id[0].item())
            detections.append(
                {
                    "bbox": [float(value) for value in coordinates],
                    "confidence": confidence,
                    "track_id": track_id,
                }
            )
        return detections


def get_analyzer(model_name):
    key = (
        str(_resolve_model_name(model_name)),
        float(APP_CONFIDENCE_THRESHOLD),
        float(APP_IOU_THRESHOLD),
        int(APP_IMAGE_SIZE),
    )
    with ANALYZER_LOCK:
        analyzer = ANALYZERS.get(key)
        if analyzer is None:
            analyzer = YOLOWeaponAnalyzer(
                model_name=key[0],
                conf_threshold=key[1],
                iou_threshold=key[2],
                img_size=key[3],
            )
            ANALYZERS[key] = analyzer
        return analyzer


def set_stream_state(stream_id, **updates):
    with STREAMS_LOCK:
        if stream_id in STREAMS:
            STREAMS[stream_id].update(updates)


def stream_worker(
    stream_id,
    camera_url,
    threshold,
    frame_stride,
    model_name,
    is_uploaded_source=False,
    realtime_playback=False,
    show_boxes=False,
    virtual_line=DEFAULT_VIRTUAL_LINE,
    inside_side="below",
):
    analyzer = get_analyzer(model_name=model_name)
    cap = cv2.VideoCapture(camera_url)
    if not cap.isOpened():
        set_stream_state(stream_id, status="error", error="Could not open source stream.")
        return
    try:
        person_tracker = YOLOPersonFlowTracker(PERSON_MODEL_NAME)
    except Exception as exc:
        cap.release()
        set_stream_state(stream_id, status="error", error=f"Could not initialize person tracking: {type(exc).__name__}: {exc}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    line_ratio = tuple(float(value) for value in virtual_line)
    person_tracks = {}
    person_detections = []
    entry_count = 0
    exit_count = 0
    sampled_count = 0
    score_history = deque(maxlen=60)
    latency_history_ms = deque(maxlen=40)

    # ------------------------------------------------------------------
    # Temporal smoothing state
    # ------------------------------------------------------------------
    alert_cooldown_remaining = 0                            # frames left in cooldown
    detection_history = deque(maxlen=CONFIRMATION_HISTORY)  # sliding window of 1/0 hits
    smoothed_score = 0.0                                    # EMA smoothed confidence
    last_alert_anomaly_type = "normal"                      # remembered weapon type during cooldown
    last_alert_description = ""                             # remembered description during cooldown
    last_annotated_frame = None                             # most recent frame with drawn boxes

    infer_state = {"job_seq": 0, "done_seq": 0, "frames": None, "result": None, "error": ""}
    infer_lock = threading.Lock()
    infer_stop = threading.Event()

    def queue_infer_job(frames_bgr):
        with infer_lock:
            infer_state["frames"] = frames_bgr
            infer_state["job_seq"] += 1

    def infer_loop():
        last_seq = 0
        while not infer_stop.is_set():
            frames_job = None
            seq = 0
            with infer_lock:
                if infer_state["job_seq"] > last_seq and infer_state["frames"] is not None:
                    seq = int(infer_state["job_seq"])
                    frames_job = list(infer_state["frames"])
            if frames_job is None:
                time.sleep(0.02)
                continue
            try:
                t0 = time.perf_counter()
                result = analyzer.infer_binary(frames_job, show_boxes=show_boxes)
                result["infer_ms"] = (time.perf_counter() - t0) * 1000.0
            except Exception as exc:
                with infer_lock:
                    infer_state["error"] = f"{type(exc).__name__}: {exc}"
                time.sleep(0.05)
                continue
            with infer_lock:
                infer_state["done_seq"] = seq
                infer_state["result"] = result
            last_seq = seq

    infer_thread = threading.Thread(target=infer_loop, daemon=True)
    infer_thread.start()

    frame_index = 0
    playback_start = time.perf_counter()
    last_applied_done_seq = 0
    total_object_hits = 0
    total_weapon_hits = 0
    total_confirmed_alert_frames = 0
    best_weapon_score = 0.0
    best_weapon_snapshot = {
        "anomaly_type": "normal",
        "short_description": "",
        "raw_detections": "",
        "score": 0.0,
    }

    set_stream_state(stream_id, status="running", status_label="Monitoring stream (waiting for first inference)", status_note="Weapon detection is running.", fps=float(fps))

    try:
        while True:
            with STREAMS_LOCK:
                state = STREAMS.get(stream_id)
                if state is None or state.get("stop_requested"):
                    break
                updated_inside_side = state.get("inside_side", inside_side)
            inside_side = updated_inside_side
            ok, frame = cap.read()
            if not ok:
                if is_uploaded_source:
                    break
                set_stream_state(stream_id, status="error", error="Stream disconnected.")
                break
            if frame_index % max(1, int(frame_stride)) == 0:
                sampled_count += 1
                queue_infer_job([frame.copy()])
                person_detections = person_tracker.track(frame)
                height, width = frame.shape[:2]
                line = (
                    line_ratio[0] * width,
                    line_ratio[1] * height,
                    line_ratio[2] * width,
                    line_ratio[3] * height,
                )
                new_entries, new_exits, _ = update_person_flow(
                    person_tracks, person_detections, frame_index, line, inside_side
                )
                entry_count += new_entries
                exit_count += new_exits

            updated = {}
            with infer_lock:
                done_seq = int(infer_state["done_seq"])
                infer_result = dict(infer_state["result"]) if infer_state["result"] is not None else None
                infer_error = str(infer_state.get("error", "")).strip()

            if infer_error:
                set_stream_state(stream_id, status="error", error=infer_error)
                break

            if infer_result is not None and done_seq > last_applied_done_seq:
                infer_ms = float(infer_result.get("infer_ms", 0.0))
                if infer_ms > 0:
                    latency_history_ms.append(infer_ms)
                avg_infer_ms = float(np.mean(latency_history_ms)) if latency_history_ms else 0.0
                segments_per_sec = (1000.0 / avg_infer_ms) if avg_infer_ms > 0 else 0.0

                pred = int(infer_result.get("label", 0))
                anomaly_type = str(infer_result.get("anomaly_type", "normal"))
                raw_score = float(infer_result.get("score", 0.0))
                all_det_count = int(infer_result.get("all_detection_count", 0))
                weapon_det_count = int(infer_result.get("weapon_detection_count", 0))
                total_object_hits += max(0, all_det_count)
                total_weapon_hits += max(0, weapon_det_count)

                # Keep annotated frame for live preview
                if infer_result.get("last_annotated_frame") is not None:
                    last_annotated_frame = infer_result["last_annotated_frame"]

                # ------------------------------------------------------
                # 1. Score Memory — Exponential Moving Average
                # ------------------------------------------------------
                if raw_score > 0:
                    smoothed_score = SCORE_EMA_ALPHA * raw_score + (1.0 - SCORE_EMA_ALPHA) * smoothed_score
                else:
                    # Decay smoothed score when nothing detected
                    smoothed_score = smoothed_score * (1.0 - SCORE_EMA_ALPHA * 0.5)

                score_history.append(smoothed_score)

                # ------------------------------------------------------
                # 2. Confirmation Window
                # A frame is considered a "raw detection" only if
                # YOLO predicted a weapon AND raw score >= threshold.
                # ------------------------------------------------------
                frame_detected = bool(pred == 1 and raw_score >= threshold)
                detection_history.append(1 if frame_detected else 0)
                recent_detections = sum(detection_history)
                confirmed_new_alert = recent_detections >= CONFIRMATION_WINDOW

                # ------------------------------------------------------
                # 3. Alert Cooldown
                # ------------------------------------------------------
                if confirmed_new_alert:
                    # Fresh or renewed detection — reset cooldown
                    alert_cooldown_remaining = ALERT_COOLDOWN_FRAMES
                    last_alert_anomaly_type = anomaly_type if anomaly_type != "normal" else last_alert_anomaly_type
                    last_alert_description = str(infer_result.get("short_description", ""))
                elif alert_cooldown_remaining > 0:
                    # No new detection but still within cooldown window
                    alert_cooldown_remaining -= 1

                in_alert = confirmed_new_alert or alert_cooldown_remaining > 0
                is_cooldown_carry = in_alert and not confirmed_new_alert

                # ------------------------------------------------------
                # Final decision
                # ------------------------------------------------------
                if in_alert:
                    final_pred = 1
                    final_type = anomaly_type if anomaly_type != "normal" else last_alert_anomaly_type
                    if confirmed_new_alert:
                        reason_code = "visible_weapon"
                        short_description = last_alert_description
                    else:
                        reason_code = "cooldown_carry"
                        frames_since = ALERT_COOLDOWN_FRAMES - alert_cooldown_remaining
                        short_description = (
                            f"Weapon last confirmed {frames_since} frame(s) ago "
                            f"({last_alert_anomaly_type}). "
                            f"Holding alert for {alert_cooldown_remaining} more frame(s). "
                            f"Smoothed confidence: {smoothed_score:.3f}."
                        )
                else:
                    final_pred = 0
                    final_type = "normal"
                    smoothed_score = 0.0  # hard reset when cooldown fully expires
                    if frame_detected and not confirmed_new_alert:
                        reason_code = "pending_confirmation"
                        short_description = (
                            f"Weapon candidate detected ({anomaly_type}, conf={raw_score:.2f}) "
                            f"but needs {CONFIRMATION_WINDOW - recent_detections} more "
                            f"hit(s) in window of {CONFIRMATION_HISTORY} to confirm."
                        )
                    elif pred == 1 and raw_score < threshold:
                        reason_code = "low_confidence_candidate"
                        short_description = (
                            f"Low-confidence weapon candidate ({anomaly_type}, conf={raw_score:.2f}) "
                            "not promoted to alert."
                        )
                    else:
                        reason_code = "no_visible_weapon"
                        short_description = str(infer_result.get("short_description", ""))

                if final_pred == 1:
                    total_confirmed_alert_frames += 1
                    if raw_score >= best_weapon_score:
                        best_weapon_score = raw_score
                        best_weapon_snapshot = {
                            "anomaly_type": final_type,
                            "short_description": short_description,
                            "raw_detections": str(infer_result.get("raw_text", "")),
                            "score": raw_score,
                        }

                updated.update(
                    {
                        "current_score": smoothed_score,
                        "raw_score": raw_score,
                        "max_recent_score": float(max(score_history)) if score_history else smoothed_score,
                        "score_points": list(score_history),
                        "status_label": format_status_label(final_pred, final_type, is_cooldown=is_cooldown_carry),
                        "status_note": f"Frame-by-frame detection updated on sampled frame #{sampled_count}.",
                        "is_danger": bool(final_pred == 1),
                        "is_cooldown": bool(is_cooldown_carry),
                        "cooldown_remaining": int(alert_cooldown_remaining),
                        "recent_detections": int(recent_detections),
                        "confirmation_window": int(CONFIRMATION_HISTORY),
                        "processed_samples": sampled_count,
                        "raw_detections": str(infer_result.get("raw_text", "")),
                        "short_description": short_description,
                        "anomaly_type": final_type,
                        "output_label": stable_output_label(final_pred),
                        "reason_code": reason_code,
                        "last_infer_ms": infer_ms,
                        "avg_infer_ms": avg_infer_ms,
                        "segments_per_sec": segments_per_sec,
                        "segment_len": SEGMENT_LEN,
                        "selected_frames_per_segment": SELECTED_FRAMES_PER_SEGMENT,
                        "inference_every_sampled_frames": INFERENCE_EVERY_SAMPLED_FRAMES,
                    }
                )


            # Use annotated frame for preview if available, otherwise raw frame
            display_frame = (last_annotated_frame if last_annotated_frame is not None else frame).copy()
            height, width = display_frame.shape[:2]
            line = (
                int(line_ratio[0] * width),
                int(line_ratio[1] * height),
                int(line_ratio[2] * width),
                int(line_ratio[3] * height),
            )
            cv2.line(display_frame, line[:2], line[2:], (0, 220, 255), 3, cv2.LINE_AA)
            cv2.circle(display_frame, line[:2], 7, (0, 220, 255), -1, cv2.LINE_AA)
            cv2.circle(display_frame, line[2:], 7, (0, 220, 255), -1, cv2.LINE_AA)
            for detection in person_detections:
                x1, y1, x2, y2 = [int(value) for value in detection["bbox"]]
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (60, 235, 130), 2)
            cv2.putText(
                display_frame,
                f"Entered: {entry_count}  Exited: {exit_count}",
                (20, max(30, height - 20)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            current_score = float(updated.get("current_score", state.get("current_score", 0.0)))
            current_is_danger = bool(updated.get("is_danger", state.get("is_danger", False)))
            current_is_cooldown = bool(updated.get("is_cooldown", state.get("is_cooldown", False)))
            current_cooldown_remaining = int(updated.get("cooldown_remaining", state.get("cooldown_remaining", 0)))
            status_text = updated.get("status_label", state.get("status_label", "Waiting for inference"))

            preview = encode_preview(
                display_frame,
                status_text,
                current_score,
                threshold,
                is_danger=current_is_danger,
                is_cooldown=current_is_cooldown,
                cooldown_remaining=current_cooldown_remaining,
                show_status_banner=bool(state.get("show_status_banner", True)),
            )
            updated.update({
                "processed_samples": sampled_count,
                "latest_frame": preview,
                "last_frame_index": frame_index,
                "last_frame_time_sec": float(frame_index / fps),
                "entry_count": entry_count,
                "exit_count": exit_count,
                "net_flow": entry_count - exit_count,
                "people_visible": len(person_detections),
            })
            set_stream_state(stream_id, **updated)
            frame_index += 1
            if realtime_playback:
                expected_elapsed = frame_index / fps
                actual_elapsed = time.perf_counter() - playback_start
                if expected_elapsed > actual_elapsed:
                    time.sleep(min(expected_elapsed - actual_elapsed, 0.2))
    except Exception as exc:
        tb = traceback.format_exc(limit=20)
        set_stream_state(stream_id, status="error", error=f"{type(exc).__name__}: {exc}\n{tb}")
    finally:
        infer_stop.set()
        infer_thread.join(timeout=1.0)
        cap.release()
        with STREAMS_LOCK:
            state = STREAMS.get(stream_id)
            if state is not None and state.get("status") != "error":
                state["status"] = "stopped" if state.get("stop_requested") else "completed"
                if not state.get("stop_requested"):
                    if state.get("is_uploaded_source"):
                        state["status_label"] = "Uploaded video completed"
                        state["status_note"] = (
                            "Video summary: "
                            f"total_objects={total_object_hits}, total_weapon_detections={total_weapon_hits}, "
                            f"confirmed_alert_frames={total_confirmed_alert_frames}; "
                            f"people entries={entry_count}, exits={exit_count}."
                        )
                        if total_confirmed_alert_frames > 0:
                            state["output_label"] = "abnormal"
                            state["anomaly_type"] = best_weapon_snapshot.get("anomaly_type", "gun")
                            state["short_description"] = (
                                "Video-level result: weapon detected. "
                                f"confirmed_alert_frames={total_confirmed_alert_frames}, "
                                f"best_weapon_conf={best_weapon_snapshot.get('score', 0.0):.2f}."
                            )
                            state["reason_code"] = "visible_weapon"
                            state["raw_detections"] = best_weapon_snapshot.get("raw_detections", state.get("raw_detections", ""))
                            state["current_score"] = max(float(state.get("current_score", 0.0)), float(best_weapon_snapshot.get("score", 0.0)))
                            state["max_recent_score"] = max(float(state.get("max_recent_score", 0.0)), float(best_weapon_snapshot.get("score", 0.0)))
                            state["is_danger"] = True
                    else:
                        state["status_label"] = "Stream completed"


def lab_flow_worker(
   stream_id,
   camera_url,
   frame_stride,
   is_uploaded_source,
   realtime_playback,
   virtual_line,
   lab_side,
):
   cap = cv2.VideoCapture(camera_url)
   if not cap.isOpened():
       set_stream_state(stream_id, status="error", error="Could not open source stream.")
       return

   try:
       tracker = YOLOPersonFlowTracker(PERSON_MODEL_NAME)
       fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
       line_ratio = tuple(float(value) for value in virtual_line)
       inside_sign = 1 if lab_side == "left" else -1
       entries = 0
       exits = 0
       frame_index = 0
       sampled_count = 0
       tracks = {}
       latency_history_ms = deque(maxlen=40)
       playback_start = time.perf_counter()
       current_detections = []

       set_stream_state(
           stream_id,
           status="running",
           status_label="Lab flow: waiting for people",
           status_note=(
               f"Tracking people with {PERSON_MODEL_NAME}. The virtual line marks where "
               "people cross; the selected lab side determines entry direction."
           ),
           fps=float(fps),
       )

       while True:
           with STREAMS_LOCK:
               state = STREAMS.get(stream_id)
               if state is None or state.get("stop_requested"):
                   break

           ok, frame = cap.read()
           if not ok:
               if is_uploaded_source:
                   break
               set_stream_state(stream_id, status="error", error="Stream disconnected.")
               break

           infer_ms = 0.0
           event = ""
           if frame_index % max(1, int(frame_stride)) == 0:
               sampled_count += 1
               infer_started = time.perf_counter()
               current_detections = tracker.track(frame)
               infer_ms = (time.perf_counter() - infer_started) * 1000.0
               latency_history_ms.append(infer_ms)
               height, width = frame.shape[:2]
               line = (
                   line_ratio[0] * width,
                   line_ratio[1] * height,
                   line_ratio[2] * width,
                   line_ratio[3] * height,
               )
               new_entries, new_exits, event = update_person_flow(
                   tracks,
                   current_detections,
                   frame_index,
                   line,
                   "below" if inside_sign == 1 else "above",
               )
               entries += new_entries
               exits += new_exits

           height, width = frame.shape[:2]
           line = (
               int(line_ratio[0] * width),
               int(line_ratio[1] * height),
               int(line_ratio[2] * width),
               int(line_ratio[3] * height),
           )
           preview = frame.copy()
           cv2.line(preview, line[:2], line[2:], (0, 220, 255), 3, cv2.LINE_AA)
           cv2.circle(preview, line[:2], 7, (0, 220, 255), -1, cv2.LINE_AA)
           cv2.circle(preview, line[2:], 7, (0, 220, 255), -1, cv2.LINE_AA)
           for detection in current_detections:
               x1, y1, x2, y2 = [int(value) for value in detection["bbox"]]
               cv2.rectangle(preview, (x1, y1), (x2, y2), (60, 235, 130), 2)
               track_id = detection["track_id"]
               label = f"Person {track_id}" if track_id is not None else "Person"
               cv2.putText(
                   preview,
                   label,
                   (x1, max(20, y1 - 8)),
                   cv2.FONT_HERSHEY_DUPLEX,
                   0.62,
                   (60, 235, 130),
                   2,
                   cv2.LINE_AA,
               )

           visible_people = len(current_detections)
           status_text = f"VIRTUAL LINE  IN {entries}  OUT {exits}"
           cv2.putText(
               preview,
               f"People in view: {visible_people}",
               (20, max(30, height - 20)),
               cv2.FONT_HERSHEY_DUPLEX,
               0.7,
               (255, 255, 255),
               2,
               cv2.LINE_AA,
           )
           encoded_preview = encode_preview(preview, status_text, 0.0, 0.0)
           avg_infer_ms = float(np.mean(latency_history_ms)) if latency_history_ms else 0.0
           set_stream_state(
               stream_id,
               status="running",
               status_label=f"Lab flow: {entries} entered / {exits} exited",
               status_note=(
                   f"Last event: {event or 'none'}; virtual line is yellow; detected people are green. "
                   f"Net flow since monitoring started: {entries - exits}."
               ),
               output_label="lab_flow",
               anomaly_type="people",
               reason_code="virtual_line_crossing" if event else "monitoring",
               short_description=(
                   f"Entries: {entries}; exits: {exits}; people currently in view: {visible_people}."
               ),
               entry_count=entries,
               exit_count=exits,
               net_flow=entries - exits,
               people_visible=visible_people,
               processed_samples=sampled_count,
               last_frame_index=frame_index,
               last_frame_time_sec=float(frame_index / fps),
               last_infer_ms=infer_ms,
               avg_infer_ms=avg_infer_ms,
               latest_frame=encoded_preview,
           )

           frame_index += 1
           if realtime_playback:
               expected_elapsed = frame_index / fps
               actual_elapsed = time.perf_counter() - playback_start
               if expected_elapsed > actual_elapsed:
                   time.sleep(min(expected_elapsed - actual_elapsed, 0.2))
   except Exception as exc:
       tb = traceback.format_exc(limit=20)
       set_stream_state(stream_id, status="error", error=f"{type(exc).__name__}: {exc}\n{tb}")
   finally:
       cap.release()
       with STREAMS_LOCK:
           state = STREAMS.get(stream_id)
           if state is not None and state.get("status") != "error":
               state["status"] = "stopped" if state.get("stop_requested") else "completed"
               if not state.get("stop_requested"):
                   state["status_label"] = "Lab flow video completed" if is_uploaded_source else "Lab flow stream completed"
                   if is_uploaded_source:
                       state["status_note"] = (
                           f"Video summary: {state.get('entry_count', 0)} entries, "
                           f"{state.get('exit_count', 0)} exits."
                       )


PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SDP Safety Monitor</title>
  <style>
    :root {{
      --bg-1: #08131f; --bg-2: #10283c; --panel: rgba(7, 16, 27, 0.76);
      --line: rgba(190, 217, 239, 0.16); --text: #edf6ff; --muted: #99aec2;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family:"Trebuchet MS","Segoe UI",sans-serif; color:var(--text); background: linear-gradient(140deg, var(--bg-1), var(--bg-2) 52%, #06111a); }}
    .wrap {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
    .grid {{ display:grid; grid-template-columns: 1.1fr 0.9fr; gap: 20px; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 22px; padding: 18px; }}
    label {{ display:block; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; color:var(--muted); margin-bottom:8px; }}
    input, select, button {{ width:100%; border-radius:10px; border:1px solid var(--line); background:rgba(13,26,42,0.86); color:var(--text); padding:11px 12px; font-weight:600; }}
    button {{ cursor:pointer; border:none; background:linear-gradient(90deg,#2ba6f1,#2fd3c1); color:#031321; margin-top:10px; }}
    button.stop {{ background: linear-gradient(90deg,#ff4d57,#ff7a59); color:#fff; }}
    .alarm {{ border-radius:18px; border:1px solid var(--line); padding:14px; background:rgba(18,33,27,0.45); }}
    .alarm.danger {{ background:rgba(76,15,23,0.66); border-color:rgba(255,98,118,0.6); }}
    .alarm.cooldown {{ background:rgba(76,15,23,0.66); border-color:rgba(255,98,118,0.6); }}
    .video-head {{ display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:10px; }}
    .video-shell {{ border-radius:16px; overflow:hidden; background:#000; }}
    .video-frame {{ display:block; width:100%; aspect-ratio:16 / 9; height:auto; border-radius:16px; border:1px solid var(--line); object-fit:fill; background:#000; }}
    button.fs-btn {{ width:auto; margin-top:0; padding:8px 12px; border-radius:9px; background:linear-gradient(90deg,#88b4d4,#5bc9d8); color:#031322; font-size:0.85rem; }}
    .video-shell:fullscreen {{ width:100vw; height:100vh; border-radius:0; display:flex; align-items:center; justify-content:center; }}
    .video-shell:-webkit-full-screen {{ width:100vw; height:100vh; border-radius:0; display:flex; align-items:center; justify-content:center; }}
    .video-shell:fullscreen .video-frame {{ width:100%; height:100%; min-height:0; border:0; border-radius:0; object-fit:contain; }}
    .video-shell:-webkit-full-screen .video-frame {{ width:100%; height:100%; min-height:0; border:0; border-radius:0; object-fit:contain; }}
    .metrics {{ display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:10px; margin-top:12px; }}
    .metric {{ border:1px solid var(--line); border-radius:12px; padding:10px; }}
    .metric.highlight {{ border-color: rgba(255,180,60,0.5); background: rgba(255,180,60,0.07); }}
    .k {{ color:var(--muted); font-size:0.72rem; text-transform:uppercase; }}
    .v {{ font-size:1.2rem; font-weight:800; margin-top:4px; }}
    .help {{ color:var(--muted); white-space:pre-wrap; word-break:break-word; }}
    .virtual-line-config {{ display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:10px; margin:12px 0; }}
    .virtual-line-config p {{ grid-column:1 / -1; margin:0; }}
    .virtual-line-config label {{ margin:0; }}
    .virtual-line-config input, .virtual-line-config select {{ margin-top:6px; }}
    .two-up {{ display:grid; grid-template-columns: 1fr 1fr; gap:18px; }}
    .subhead {{ margin: 0 0 10px 0; color:var(--muted); font-size:0.9rem; text-transform:uppercase; letter-spacing:0.06em; }}
    .cooldown-bar-wrap {{ margin-top:10px; background: rgba(255,255,255,0.07); border-radius:8px; height:8px; overflow:hidden; }}
    .cooldown-bar {{ height:100%; border-radius:8px; background: linear-gradient(90deg,#ff4d57,#ff7a59); transition: width 0.25s; }}
    @media (max-width: 980px) {{ .grid {{ grid-template-columns: 1fr; }} .two-up {{ grid-template-columns:1fr; }} .metrics {{ grid-template-columns: repeat(2,1fr); }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="grid">
      <section class="card">
        <h2>SDP Safety Monitor</h2>
        <form id="stream-form">
          <div><label for="camera_preset">Camera</label><select id="camera_preset">{camera_preset_options}</select></div>
          <div><label for="model_name">Weapon model</label><select id="model_name" name="model_name">{model_options}</select></div>
          <div><label for="frame_stride">Sampling Stride</label><input id="frame_stride" name="frame_stride" type="number" min="1" max="16" step="1" value="{default_frame_stride}"></div>
          <div><label><input id="show_boxes" name="show_boxes" type="checkbox" value="1"> Show boxes and classes</label></div>
          <div id="virtual-line-config" class="virtual-line-config">
            <p class="help">The entry/exit counting line is fixed across the sliding-door threshold.</p>
            <label>Lab is
              <select id="inside-side" name="inside_side"><option value="above">Above the line</option><option value="below" selected>Below the line</option></select>
            </label>
          </div>
          <button type="submit">Start Monitor</button><button id="stop-btn" type="button" class="stop">Stop Stream</button>
        </form>
        <form id="upload-form" action="/upload" method="post" enctype="multipart/form-data">
          <div><label for="video_file">Upload Video</label><input id="video_file" name="video" type="file" accept="video/*" required></div>
          <input id="upload_model_name" name="model_name" type="hidden" value="{default_model_key}">
          <input id="upload_frame_stride" name="frame_stride" type="hidden" value="{default_frame_stride}">
          <input id="upload_show_boxes" name="show_boxes" type="hidden" value="0">
          <input id="upload_inside_side" name="inside_side" type="hidden" value="below">
          <button type="submit">Analyze Uploaded Video</button>
        </form>
        <p id="form-message" class="help">{message}</p>
      </section>
      <section class="card">
        <div id="alarm-card" class="alarm">
          <small>Live status</small><h2 id="status-label">Waiting for stream</h2>
          <p id="status-detail" class="help">No active camera session.</p>
          <p id="output-label" class="help"></p>
          <p id="anomaly-type" class="help"></p>
          <p id="reason-code" class="help"></p>
          <p id="short-description" class="help"></p>
          <div id="cooldown-bar-wrap" class="cooldown-bar-wrap" style="display:none">
            <div id="cooldown-bar" class="cooldown-bar" style="width:0%"></div>
          </div>
        </div>
      </section>
    </div>
    <section class="card" style="margin-top:18px;">
      <div class="two-up">
        <div>
          <div class="video-head">
            <div class="subhead">Live preview</div>
            <button id="status-banner-btn" type="button" class="fs-btn" style="display:none" aria-pressed="true">Hide Status Banner</button>
            <button id="fullscreen-btn" type="button" class="fs-btn" style="display:none">Full Screen</button>
          </div>
          <div id="live-frame-shell" class="video-shell">
            <img id="live-frame" class="video-frame" alt="Live preview">
          </div>
        </div>
      </div>
      <div id="weapon-metrics" class="metrics">
        <div class="metric"><div class="k">Smoothed confidence</div><div id="current-score" class="v">0.000</div></div>
        <div class="metric"><div class="k">Raw confidence</div><div id="raw-score" class="v">0.000</div></div>
        <div class="metric"><div class="k">Peak confidence</div><div id="recent-max" class="v">0.000</div></div>
        <div class="metric"><div class="k">Frame</div><div id="frame-index" class="v">0</div></div>
        <div class="metric"><div class="k">Processed samples</div><div id="processed-samples" class="v">0</div></div>
        <div class="metric"><div class="k">Last infer (ms)</div><div id="last-infer-ms" class="v">0</div></div>
        <div class="metric"><div class="k">Avg infer (ms)</div><div id="avg-infer-ms" class="v">0</div></div>
        <div class="metric"><div class="k">Segments / sec</div><div id="segments-per-sec" class="v">0.00</div></div>
        <div class="metric"><div class="k">Reason code</div><div id="reason-code-metric" class="v">-</div></div>
        <div class="metric highlight"><div class="k">Cooldown frames left</div><div id="cooldown-remaining" class="v">0</div></div>
        <div class="metric highlight"><div class="k">Hits / window</div><div id="recent-hits" class="v">0 / {confirmation_history}</div></div>
        <div class="metric highlight"><div class="k">EMA alpha</div><div id="ema-alpha" class="v">{ema_alpha}</div></div>
        <div class="metric highlight"><div class="k">Entries</div><div id="entry-count" class="v">0</div></div>
        <div class="metric highlight"><div class="k">Exits</div><div id="exit-count" class="v">0</div></div>
        <div class="metric"><div class="k">Net flow since start</div><div id="net-flow" class="v">0</div></div>
        <div class="metric"><div class="k">People in view</div><div id="people-visible" class="v">0</div></div>
      </div>
    </section>
  </div>
  <script>
    (() => {{
      let streamId = null; let pollTimer = null;
      const form = document.getElementById("stream-form");
      const uploadForm = document.getElementById("upload-form");
      const stopBtn = document.getElementById("stop-btn");
      const formMessage = document.getElementById("form-message");
      const cameraPreset = document.getElementById("camera_preset");
      const cameraPresetMap = {camera_preset_json};
      const modelNameInput = document.getElementById("model_name");
      const frameStrideInput = document.getElementById("frame_stride");
      const showBoxesCheckbox = document.getElementById("show_boxes");
      const uploadModelName = document.getElementById("upload_model_name");
      const uploadFrameStride = document.getElementById("upload_frame_stride");
      const uploadShowBoxes = document.getElementById("upload_show_boxes");
      const insideSide = document.getElementById("inside-side");
      const uploadInsideSide = document.getElementById("upload_inside_side");
      const alarmCard = document.getElementById("alarm-card");
      const statusLabel = document.getElementById("status-label");
      const statusDetail = document.getElementById("status-detail");
      const outputLabel = document.getElementById("output-label");
      const anomalyType = document.getElementById("anomaly-type");
      const reasonCode = document.getElementById("reason-code");
      const reasonCodeMetric = document.getElementById("reason-code-metric");
      const shortDescription = document.getElementById("short-description");
      const liveFrame = document.getElementById("live-frame");
      const liveFrameShell = document.getElementById("live-frame-shell");
      const statusBannerBtn = document.getElementById("status-banner-btn");
      const fullscreenBtn = document.getElementById("fullscreen-btn");
      const currentScore = document.getElementById("current-score");
      const rawScore = document.getElementById("raw-score");
      const recentMax = document.getElementById("recent-max");
      const frameIndex = document.getElementById("frame-index");
      const processedSamples = document.getElementById("processed-samples");
      const lastInferMs = document.getElementById("last-infer-ms");
      const avgInferMs = document.getElementById("avg-infer-ms");
      const segmentsPerSec = document.getElementById("segments-per-sec");
      const cooldownRemaining = document.getElementById("cooldown-remaining");
      const recentHits = document.getElementById("recent-hits");
      const cooldownBarWrap = document.getElementById("cooldown-bar-wrap");
      const cooldownBar = document.getElementById("cooldown-bar");
      const maxCooldown = {alert_cooldown_frames};
      const confirmationHistory = {confirmation_history};
      function getFullscreenElement() {{
        return document.fullscreenElement || document.webkitFullscreenElement || document.msFullscreenElement || null;
      }}

      async function exitFullscreenSafe() {{
        if (document.exitFullscreen) {{
          await document.exitFullscreen();
          return;
        }}
        if (document.webkitExitFullscreen) {{
          document.webkitExitFullscreen();
          return;
        }}
        if (document.msExitFullscreen) {{
          document.msExitFullscreen();
        }}
      }}

      function updateFullscreenButtonLabel() {{
        const activeEl = getFullscreenElement();
        const isFullscreen = activeEl === liveFrameShell || activeEl === liveFrame;
        fullscreenBtn.textContent = isFullscreen ? "Exit Full Screen" : "Full Screen";
      }}

      function setFullscreenVisibility(active) {{
        fullscreenBtn.style.display = active ? "inline-block" : "none";
        if (!active && getFullscreenElement()) {{
          exitFullscreenSafe().catch(() => {{}});
        }}
      }}

      async function toggleFullscreen() {{
        const activeEl = getFullscreenElement();
        if (activeEl) {{
          await exitFullscreenSafe();
          updateFullscreenButtonLabel();
          return;
        }}
        const target = liveFrameShell || liveFrame;
        if (!target) return;
        if (target.requestFullscreen) {{
          await target.requestFullscreen();
        }} else if (target.webkitRequestFullscreen) {{
          target.webkitRequestFullscreen();
        }} else if (target.msRequestFullscreen) {{
          target.msRequestFullscreen();
        }}
        updateFullscreenButtonLabel();
      }}

      async function pollState() {{
        if (!streamId) return;
        try {{
          const response = await fetch(`/api/stream_state?id=${{encodeURIComponent(streamId)}}`, {{ cache: "no-store" }});
          if (!response.ok) throw new Error("Could not fetch stream state.");
          const data = await response.json();
          statusBannerBtn.textContent = data.show_status_banner === false ? "Show Status Banner" : "Hide Status Banner";
          statusBannerBtn.setAttribute("aria-pressed", String(data.show_status_banner !== false));

          statusLabel.textContent = data.status_label || data.status || "Waiting for stream";
          statusDetail.textContent = data.error || data.status_note || "Monitoring stream.";
          outputLabel.textContent = data.output_label ? `Output label: ${{data.output_label}}` : "";
          anomalyType.textContent = data.anomaly_type ? `Anomaly type: ${{data.anomaly_type}}` : "";
          reasonCode.textContent = data.reason_code ? `Reason code: ${{data.reason_code}}` : "";
          reasonCodeMetric.textContent = data.reason_code || "-";
          shortDescription.textContent = data.short_description ? `${{data.short_description}}` : "";

          currentScore.textContent = Number(data.current_score || 0).toFixed(3);
          rawScore.textContent = Number(data.raw_score || 0).toFixed(3);
          recentMax.textContent = Number(data.max_recent_score || 0).toFixed(3);
          frameIndex.textContent = String(data.last_frame_index || 0);
          processedSamples.textContent = String(data.processed_samples || 0);
          lastInferMs.textContent = Number(data.last_infer_ms || 0).toFixed(1);
          avgInferMs.textContent = Number(data.avg_infer_ms || 0).toFixed(1);
          segmentsPerSec.textContent = Number(data.segments_per_sec || 0).toFixed(2);
          document.getElementById("entry-count").textContent = String(data.entry_count || 0);
          document.getElementById("exit-count").textContent = String(data.exit_count || 0);
          document.getElementById("net-flow").textContent = String(data.net_flow || 0);
          document.getElementById("people-visible").textContent = String(data.people_visible || 0);

          // Cooldown metrics
          const cdLeft = data.cooldown_remaining || 0;
          const recentDets = data.recent_detections || 0;
          const confWin = data.confirmation_window || confirmationHistory;
          cooldownRemaining.textContent = String(cdLeft);
          recentHits.textContent = `${{recentDets}} / ${{confWin}}`;

          // Cooldown progress bar
          if (cdLeft > 0) {{
            cooldownBarWrap.style.display = "block";
            cooldownBar.style.width = `${{Math.round((cdLeft / maxCooldown) * 100)}}%`;
          }} else {{
            cooldownBarWrap.style.display = "none";
            cooldownBar.style.width = "0%";
          }}

          // Alarm card styling
          alarmCard.classList.remove("danger", "cooldown");
          if (data.is_danger) {{
            if (data.is_cooldown) {{
              alarmCard.classList.add("cooldown");
            }} else {{
              alarmCard.classList.add("danger");
            }}
          }}

          // Live preview only
          liveFrame.src = `/api/live_frame?id=${{encodeURIComponent(streamId)}}&t=${{Date.now()}}`;

          if (data.status === "error" || data.status === "stopped" || data.status === "completed") {{
            statusBannerBtn.style.display = "none";
            setFullscreenVisibility(false);
            window.clearTimeout(pollTimer); pollTimer = null; return;
          }}
          statusBannerBtn.style.display = "inline-block";
          setFullscreenVisibility(true);
        }} catch (error) {{
          formMessage.textContent = error.message;
        }}
        pollTimer = window.setTimeout(pollState, 250);
      }}

      form.addEventListener("submit", async (event) => {{
        event.preventDefault();
        if (pollTimer) {{ window.clearTimeout(pollTimer); pollTimer = null; }}
        const formData = new FormData(form);
        const selectedCamera = cameraPreset.value;
        const selectedUrl = cameraPresetMap[selectedCamera] || "";
        if (!selectedUrl) {{ formMessage.textContent = "Invalid camera selection."; return; }}
        formData.set("camera_url", selectedUrl);
        const response = await fetch("/start_stream", {{ method: "POST", body: new URLSearchParams(formData) }});
        const data = await response.json();
        if (!response.ok) {{ formMessage.textContent = data.error || "Could not start stream."; return; }}
        streamId = data.stream_id;
        setFullscreenVisibility(true);
        formMessage.textContent = "Stream started.";
        pollState();
      }});
      stopBtn.addEventListener("click", async () => {{
        if (!streamId) return;
        await fetch(`/stop_stream?id=${{encodeURIComponent(streamId)}}`, {{ method: "POST" }});
        setFullscreenVisibility(false);
        formMessage.textContent = "Stop request sent.";
      }});

      statusBannerBtn.addEventListener("click", async () => {{
        if (!streamId) return;
        try {{
          const response = await fetch(`/toggle_status_banner?id=${{encodeURIComponent(streamId)}}`, {{ method: "POST" }});
          const data = await response.json();
          if (!response.ok) throw new Error(data.error || "Could not toggle status banner.");
          statusBannerBtn.textContent = data.show_status_banner ? "Hide Status Banner" : "Show Status Banner";
          statusBannerBtn.setAttribute("aria-pressed", String(data.show_status_banner));
        }} catch (error) {{
          formMessage.textContent = error.message;
        }}
      }});

      fullscreenBtn.addEventListener("click", async () => {{
        try {{
          await toggleFullscreen();
        }} catch (error) {{
          formMessage.textContent = `Fullscreen unavailable: ${{error.message || error}}`;
        }}
      }});
      document.addEventListener("fullscreenchange", updateFullscreenButtonLabel);
      document.addEventListener("webkitfullscreenchange", updateFullscreenButtonLabel);
      document.addEventListener("MSFullscreenChange", updateFullscreenButtonLabel);

      function syncUploadFields() {{
        uploadModelName.value = modelNameInput.value;
        uploadFrameStride.value = frameStrideInput.value;
        uploadShowBoxes.value = showBoxesCheckbox.checked ? "1" : "0";
        uploadInsideSide.value = insideSide.value;
      }}
      modelNameInput.addEventListener("change", syncUploadFields);
      frameStrideInput.addEventListener("change", syncUploadFields);
      showBoxesCheckbox.addEventListener("change", syncUploadFields);
      insideSide.addEventListener("change", syncUploadFields);
      uploadForm.addEventListener("submit", syncUploadFields);
      syncUploadFields();

      const urlParams = new URLSearchParams(window.location.search);
      const existingStreamId = urlParams.get("stream_id");
      if (existingStreamId) {{
        streamId = existingStreamId;
        setFullscreenVisibility(true);
        formMessage.textContent = urlParams.get("uploaded") === "1" ? "Uploaded video accepted. Processing started." : "Resuming stream status.";
        pollState();
      }}
    }})();
  </script>
</body>
</html>
"""


def _parse_virtual_line_config(value_getter):
    inside_side = str(value_getter("inside_side", "below") or "below").strip().lower()
    if inside_side not in {"above", "below"}:
        raise ValueError("inside_side must be 'above' or 'below'.")
    return DEFAULT_VIRTUAL_LINE, inside_side


class AppHandler(BaseHTTPRequestHandler):
    server_version = "SDPSafetyMonitor/1.0"

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self._render_page()
            return
        if parsed.path == "/api/stream_state":
            stream_id = urllib.parse.parse_qs(parsed.query).get("id", [""])[0]
            with STREAMS_LOCK:
                payload = STREAMS.get(stream_id)
            if payload is None:
                self._send_json({"error": "Stream not found."}, status=404)
                return
            safe_payload = {k: v for k, v in payload.items() if k not in {"latest_frame", "analysis_frame", "thread"}}
            self._send_json(safe_payload)
            return
        if parsed.path == "/api/live_frame":
            self._serve_binary_frame(parsed.query, "latest_frame")
            return
        self.send_error(404, "Not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path == "/start_stream":
                self._handle_start_stream()
                return
            if parsed.path == "/upload":
                self._handle_upload()
                return
            if parsed.path == "/stop_stream":
                self._handle_stop_stream(parsed.query)
                return
            if parsed.path == "/toggle_status_banner":
                self._handle_toggle_status_banner(parsed.query)
                return
            self.send_error(404, "Not found")
        except Exception as exc:
            if parsed.path in {"/start_stream", "/stop_stream", "/toggle_status_banner"}:
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)
            else:
                self._render_page(f"Upload failed: {type(exc).__name__}: {exc}")

    def _serve_binary_frame(self, query, key):
        stream_id = urllib.parse.parse_qs(query).get("id", [""])[0]
        with STREAMS_LOCK:
            payload = STREAMS.get(stream_id)
        if payload is None or not payload.get(key):
            self.send_error(404, "Frame not found")
            return
        frame = payload[key]
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(frame)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(frame)

    def _handle_start_stream(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        form = urllib.parse.parse_qs(raw)
        camera_url = urllib.parse.unquote(form.get("camera_url", [DEFAULT_CAMERA_URL])[0].strip())
        try:
            virtual_line, inside_side = _parse_virtual_line_config(
                lambda name, default: form.get(name, [default])[0]
            )
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        # Always use the CLI-validated DEFAULT_MODEL_NAME — ignore any model value
        # submitted by the form to prevent stray .pt files from being used
        model_name = DEFAULT_MODEL_NAME
        threshold = float(DEFAULT_THRESHOLD)
        frame_stride = int(form.get("frame_stride", [str(APP_DEFAULT_FRAME_STRIDE)])[0] or str(APP_DEFAULT_FRAME_STRIDE))
        show_boxes = bool(int(form.get("show_boxes", ["0"])[0] or "0"))
        if not camera_url:
            self._send_json({"error": "Camera URL is required."}, status=400)
            return
        if not _is_valid_model_selection(model_name):
            self._send_json({"error": "Invalid model selection."}, status=400)
            return
        if frame_stride < 1:
            self._send_json({"error": "frame_stride must be >= 1."}, status=400)
            return
        if threshold < 0.05 or threshold > 0.99:
            self._send_json({"error": "threshold must be between 0.05 and 0.99."}, status=400)
            return
        stream_id = self._create_stream_state(
            camera_url,
            threshold,
            frame_stride,
            model_name,
            status_label="Connecting to camera",
            status_note="SDP safety monitoring is starting.",
            show_boxes=show_boxes,
            virtual_line=virtual_line,
            inside_side=inside_side,
        )
        self._send_json({"stream_id": stream_id, "status": "starting"})

    def _handle_upload(self):
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", "")})
        # Always use the CLI-validated DEFAULT_MODEL_NAME for uploads too
        model_name = DEFAULT_MODEL_NAME
        if not _is_valid_model_selection(model_name):
            self._render_page("Invalid model selection for upload.")
            return
        if "video" not in form or not getattr(form["video"], "filename", ""):
            self._render_page("No video file was uploaded.")
            return
        threshold = float(DEFAULT_THRESHOLD)
        try:
            virtual_line, inside_side = _parse_virtual_line_config(
                lambda name, default: form.getvalue(name, default)
            )
        except ValueError as exc:
            self._render_page(f"Invalid virtual line configuration: {exc}")
            return
        stride_value = form.getvalue("frame_stride", str(APP_DEFAULT_FRAME_STRIDE))
        show_boxes = bool(int(form.getvalue("show_boxes", "0") or "0"))
        try:
            frame_stride = int(stride_value or str(APP_DEFAULT_FRAME_STRIDE))
        except ValueError:
            frame_stride = APP_DEFAULT_FRAME_STRIDE
        frame_stride = max(1, min(16, frame_stride))
        upload = form["video"]
        original_name = Path(upload.filename).name
        safe_name = f"{uuid.uuid4().hex}_{original_name}"
        video_path = UPLOAD_DIR / safe_name
        with open(video_path, "wb") as handle:
            handle.write(upload.file.read())
        stream_id = self._create_stream_state(
            str(video_path), threshold, frame_stride, model_name,
            status_label=f"Analyzing uploaded video: {original_name}",
            status_note="Uploaded video source is running.",
            is_uploaded_source=True,
            realtime_playback=True,
            show_boxes=show_boxes,
            virtual_line=virtual_line,
            inside_side=inside_side,
        )
        location = f"/?stream_id={urllib.parse.quote(stream_id)}&uploaded=1"
        self.send_response(303)
        self.send_header("Location", location)
        self.end_headers()

    def _create_stream_state(
        self,
        camera_url,
        threshold,
        frame_stride,
        model_name,
        status_label="Connecting to camera",
        status_note="SDP safety monitoring is idle.",
        is_uploaded_source=False,
        realtime_playback=False,
        show_boxes=False,
        virtual_line=DEFAULT_VIRTUAL_LINE,
        inside_side="below",
    ):
        stream_id = uuid.uuid4().hex
        state = {
            "stream_id": stream_id,
            "status": "starting",
            "status_label": status_label,
            "status_note": status_note,
            "camera_url": camera_url,
            "model_name": model_name,
            "virtual_line": list(virtual_line),
            "inside_side": inside_side,
            "threshold": threshold,
            "frame_stride": frame_stride,
            "show_boxes": bool(show_boxes),
            "current_score": 0.0,
            "raw_score": 0.0,
            "max_recent_score": 0.0,
            "processed_samples": 0,
            "entry_count": 0,
            "exit_count": 0,
            "net_flow": 0,
            "people_visible": 0,
            "last_frame_index": 0,
            "is_danger": False,
            "is_cooldown": False,
            "cooldown_remaining": 0,
            "recent_detections": 0,
            "confirmation_window": CONFIRMATION_HISTORY,
            "raw_detections": "",
            "short_description": "",
            "anomaly_type": "normal",
            "output_label": "normal",
            "reason_code": "no_visible_weapon",
            "last_infer_ms": 0.0,
            "avg_infer_ms": 0.0,
            "segments_per_sec": 0.0,
            "segment_len": SEGMENT_LEN,
            "selected_frames_per_segment": SELECTED_FRAMES_PER_SEGMENT,
            "inference_every_sampled_frames": INFERENCE_EVERY_SAMPLED_FRAMES,
            "latest_frame": None,
            "stop_requested": False,
            "is_uploaded_source": bool(is_uploaded_source),
            "realtime_playback": bool(realtime_playback),
            "show_boxes": bool(show_boxes),
            "show_status_banner": True,
        }
        worker = threading.Thread(
            target=stream_worker,
            args=(
                stream_id,
                camera_url,
                threshold,
                frame_stride,
                model_name,
                bool(is_uploaded_source),
                bool(realtime_playback),
                bool(show_boxes),
                virtual_line,
                inside_side,
            ),
            daemon=True,
        )
        state["thread"] = worker
        with STREAMS_LOCK:
            STREAMS[stream_id] = state
        worker.start()
        return stream_id

    def _handle_stop_stream(self, query):
        stream_id = urllib.parse.parse_qs(query).get("id", [""])[0]
        with STREAMS_LOCK:
            payload = STREAMS.get(stream_id)
            if payload is None:
                self._send_json({"error": "Stream not found."}, status=404)
                return
            payload["stop_requested"] = True
            payload["status"] = "stopped"
            payload["status_label"] = "Stream stopped"
            payload["status_note"] = "Stream was stopped by user request."
        self._send_json({"ok": True})

    def _handle_toggle_status_banner(self, query):
        stream_id = urllib.parse.parse_qs(query).get("id", [""])[0]
        with STREAMS_LOCK:
            payload = STREAMS.get(stream_id)
            if payload is None:
                self._send_json({"error": "Stream not found."}, status=404)
                return
            payload["show_status_banner"] = not bool(payload.get("show_status_banner", True))
            show_status_banner = payload["show_status_banner"]
        self._send_json({"ok": True, "show_status_banner": show_status_banner})

    def _render_page(self, message="SDP safety monitoring is ready."):
        camera_options = []
        for camera_name, _ in CAMERA_PRESETS.items():
            selected = " selected" if camera_name == "Camera 3" else ""
            camera_options.append(f'<option value="{html.escape(camera_name)}"{selected}>{html.escape(camera_name)}</option>')
        model_options = []
        for model_key, preset in MODEL_PRESETS.items():
            selected = " selected" if model_key == DEFAULT_MODEL_PRESET_KEY else ""
            label = str(preset.get("label", model_key))
            model_options.append(f'<option value="{html.escape(model_key)}"{selected}>{html.escape(label)}</option>')
        page = PAGE_TEMPLATE.format(
            camera_preset_options="".join(camera_options),
            model_options="".join(model_options),
            camera_preset_json=json.dumps(dict(CAMERA_PRESETS)),
            default_model_key=html.escape(DEFAULT_MODEL_PRESET_KEY),
            default_frame_stride=int(APP_DEFAULT_FRAME_STRIDE),
            message=html.escape(message),
            alert_cooldown_frames=int(ALERT_COOLDOWN_FRAMES),
            confirmation_history=int(CONFIRMATION_HISTORY),
            ema_alpha=float(SCORE_EMA_ALPHA),
            person_model_name=html.escape(PERSON_MODEL_NAME),
        )
        self._send_html(page)

    def _send_html(self, body, status=200):
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        return


def parse_cli_args():
    parser = argparse.ArgumentParser(description="SDP safety monitoring real-time stream app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8013, type=int)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--person-model-name", default=PERSON_MODEL_NAME,
                        help="YOLO model with a person class for lab entry/exit monitoring")
    parser.add_argument("--cache-dir", default=str(Path("cache")))
    parser.add_argument("--instruct-path", default="")
    parser.add_argument("--segment-len", default=SEGMENT_LEN, type=int)
    parser.add_argument("--selected-frames", default=SELECTED_FRAMES_PER_SEGMENT, type=int)
    parser.add_argument("--step", default=INFERENCE_EVERY_SAMPLED_FRAMES, type=int)
    parser.add_argument("--frame-stride", default=APP_DEFAULT_FRAME_STRIDE, type=int)
    parser.add_argument("--conf-thres", default=APP_CONFIDENCE_THRESHOLD, type=float)
    parser.add_argument("--iou-thres", default=APP_IOU_THRESHOLD, type=float)
    parser.add_argument("--imgsz", default=APP_IMAGE_SIZE, type=int)
    parser.add_argument("--alert-thres", default=DEFAULT_THRESHOLD, type=float,
                        help="Minimum weapon confidence to trigger alert (default 0.25)")
    parser.add_argument("--cooldown-frames", default=ALERT_COOLDOWN_FRAMES, type=int,
                        help="Frames to hold danger state after last detection (default 30)")
    parser.add_argument("--confirm-window", default=CONFIRMATION_WINDOW, type=int,
                        help="Min detections in recent history to trigger alert (default 2)")
    parser.add_argument("--confirm-history", default=CONFIRMATION_HISTORY, type=int,
                        help="Size of confirmation sliding window (default 4)")
    parser.add_argument("--ema-alpha", default=SCORE_EMA_ALPHA, type=float,
                        help="EMA smoothing factor for confidence score (default 0.40)")
    return parser.parse_args()


def main(
    host="127.0.0.1",
    port=8013,
    model_name=DEFAULT_MODEL_NAME,
    cache_dir="cache",
    instruct_path="",
    segment_len=SEGMENT_LEN,
    selected_frames=SELECTED_FRAMES_PER_SEGMENT,
    step=INFERENCE_EVERY_SAMPLED_FRAMES,
    frame_stride=APP_DEFAULT_FRAME_STRIDE,
    conf_thres=APP_CONFIDENCE_THRESHOLD,
    iou_thres=APP_IOU_THRESHOLD,
    imgsz=APP_IMAGE_SIZE,
    alert_thres=DEFAULT_THRESHOLD,
    cooldown_frames=ALERT_COOLDOWN_FRAMES,
    confirm_window=CONFIRMATION_WINDOW,
    confirm_history=CONFIRMATION_HISTORY,
    ema_alpha=SCORE_EMA_ALPHA,
    person_model_name=PERSON_MODEL_NAME,
):
    global SEGMENT_LEN, SELECTED_FRAMES_PER_SEGMENT, INFERENCE_EVERY_SAMPLED_FRAMES
    global APP_DEFAULT_FRAME_STRIDE, APP_CONFIDENCE_THRESHOLD, APP_IOU_THRESHOLD, APP_IMAGE_SIZE, APP_INSTRUCT_PATH
    global DEFAULT_MODEL_NAME, MODEL_CHOICES
    global PERSON_MODEL_NAME
    global DEFAULT_THRESHOLD, ALERT_COOLDOWN_FRAMES, CONFIRMATION_WINDOW, CONFIRMATION_HISTORY, SCORE_EMA_ALPHA

    _ = cache_dir
    APP_INSTRUCT_PATH = str(instruct_path or "")
    SEGMENT_LEN = 1
    SELECTED_FRAMES_PER_SEGMENT = 1
    INFERENCE_EVERY_SAMPLED_FRAMES = 1
    APP_DEFAULT_FRAME_STRIDE = max(1, min(16, int(frame_stride)))
    APP_CONFIDENCE_THRESHOLD = max(0.01, min(0.99, float(conf_thres)))
    APP_IOU_THRESHOLD = max(0.05, min(0.99, float(iou_thres)))
    APP_IMAGE_SIZE = max(160, int(imgsz))
    DEFAULT_THRESHOLD = max(0.05, min(0.99, float(alert_thres)))
    ALERT_COOLDOWN_FRAMES = max(1, int(cooldown_frames))
    CONFIRMATION_WINDOW = max(1, int(confirm_window))
    CONFIRMATION_HISTORY = max(CONFIRMATION_WINDOW, int(confirm_history))
    SCORE_EMA_ALPHA = max(0.05, min(0.99, float(ema_alpha)))

    # --- Validate and lock in the CLI-supplied model before anything else ---
    # Resolve the path explicitly so we catch problems before the server starts
    raw_model_path = Path(str(model_name).strip())
    if raw_model_path.is_absolute():
        resolved_model_path = raw_model_path
    else:
        resolved_model_path = _resolve_input_path(str(model_name))

    if not resolved_model_path.exists() or not resolved_model_path.is_file():
        print(f"[FATAL] Model file not found: {resolved_model_path}", flush=True)
        print(f"[FATAL] Raw input was:         {model_name}", flush=True)
        print(f"[FATAL] APP_ROOT={APP_ROOT}", flush=True)
        print(f"[FATAL] REPO_ROOT={REPO_ROOT}", flush=True)
        print("[FATAL] Pass an absolute path, e.g.:", flush=True)
        print(f'[FATAL]   --model-name "C:\\path\\to\\best.pt"', flush=True)
        raise FileNotFoundError(f"Model not found: {resolved_model_path}")

    model_name = str(resolved_model_path)
    PERSON_MODEL_NAME = str(person_model_name).strip() or "yolo11n.pt"
    person_model_path = _resolve_input_path(PERSON_MODEL_NAME)
    if person_model_path.exists() and person_model_path.is_file():
        PERSON_MODEL_NAME = str(person_model_path)

    # Force DEFAULT_MODEL_NAME and MODEL_PRESETS to the validated CLI model
    # so that the UI form and stream_worker always use the same weights
    DEFAULT_MODEL_NAME = model_name
    MODEL_PRESETS["model1_finetuned"]["target"] = model_name
    if model_name not in MODEL_CHOICES:
        MODEL_CHOICES = [model_name] + MODEL_CHOICES

    print(f"[OK] Model validated: {model_name}", flush=True)
    print(f"SDP safety monitor running at http://{host}:{port}")
    print(f"Model: {model_name}")
    print(f"Person flow model: {PERSON_MODEL_NAME}")
    print(f"conf_thres={APP_CONFIDENCE_THRESHOLD:.2f}, iou_thres={APP_IOU_THRESHOLD:.2f}, imgsz={APP_IMAGE_SIZE}")
    print(f"alert_thres={DEFAULT_THRESHOLD:.2f}")
    print(f"Temporal smoothing: cooldown={ALERT_COOLDOWN_FRAMES} frames, "
          f"confirm_window={CONFIRMATION_WINDOW}/{CONFIRMATION_HISTORY}, "
          f"ema_alpha={SCORE_EMA_ALPHA:.2f}")
    print("Press Ctrl+C to stop.")

    try:
        warmup = get_analyzer(model_name)
        warmup._ensure_model()
    except Exception as exc:
        print(f"[warning] model warm-up failed: {type(exc).__name__}: {exc}")

    server = ThreadingHTTPServer((host, int(port)), AppHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    cli_args = parse_cli_args()
    main(
        host=cli_args.host,
        port=cli_args.port,
        model_name=cli_args.model_name,
        person_model_name=cli_args.person_model_name,
        cache_dir=cli_args.cache_dir,
        instruct_path=cli_args.instruct_path,
        segment_len=cli_args.segment_len,
        selected_frames=cli_args.selected_frames,
        step=cli_args.step,
        frame_stride=cli_args.frame_stride,
        conf_thres=cli_args.conf_thres,
        iou_thres=cli_args.iou_thres,
        imgsz=cli_args.imgsz,
        alert_thres=cli_args.alert_thres,
        cooldown_frames=cli_args.cooldown_frames,
        confirm_window=cli_args.confirm_window,
        confirm_history=cli_args.confirm_history,
        ema_alpha=cli_args.ema_alpha,
    )
