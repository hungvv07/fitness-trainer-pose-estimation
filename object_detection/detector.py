"""MediaPipe Tasks object detector (EfficientDet, COCO classes)."""

import os
import threading
import urllib.request
from dataclasses import dataclass

import cv2

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

MODELS = {
    "lite0": {
        "name": "EfficientDet-Lite0 (nhanh)",
        "file": "efficientdet_lite0.tflite",
        "url": "https://storage.googleapis.com/mediapipe-models/object_detector/"
               "efficientdet_lite0/float16/1/efficientdet_lite0.tflite",
    },
    "lite2": {
        "name": "EfficientDet-Lite2 (chính xác hơn)",
        "file": "efficientdet_lite2.tflite",
        "url": "https://storage.googleapis.com/mediapipe-models/object_detector/"
               "efficientdet_lite2/float16/1/efficientdet_lite2.tflite",
    },
}

DEFAULT_MODEL = "lite0"

# The detector is created with a low threshold; the user-facing confidence
# slider filters on top of it, so moving the slider never reloads the model.
BASE_SCORE_THRESHOLD = 0.15

_download_lock = threading.Lock()


@dataclass
class Detection:
    label: str
    score: float
    bbox: tuple  # (x1, y1, x2, y2) in pixels


def model_path(model_key):
    return os.path.join(MODEL_DIR, MODELS[model_key]["file"])


def ensure_model(model_key):
    """Download the model file on first use. Returns the local path."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}")
    path = model_path(model_key)
    with _download_lock:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
        os.makedirs(MODEL_DIR, exist_ok=True)
        tmp_path = path + ".part"
        try:
            urllib.request.urlretrieve(MODELS[model_key]["url"], tmp_path)
            os.replace(tmp_path, path)
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise RuntimeError(
                f"Không tải được model {MODELS[model_key]['file']} ({e}). "
                f"Tải thủ công từ {MODELS[model_key]['url']} và đặt vào thư mục {MODEL_DIR}"
            ) from e
    return path


class ObjectDetector:
    """Thin wrapper around mediapipe.tasks ObjectDetector (IMAGE mode)."""

    def __init__(self, model_key=DEFAULT_MODEL, max_results=100):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        self._mp = mp
        self.model_key = model_key
        options = vision.ObjectDetectorOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=ensure_model(model_key)),
            running_mode=vision.RunningMode.IMAGE,
            max_results=max_results,
            score_threshold=BASE_SCORE_THRESHOLD,
        )
        self._detector = vision.ObjectDetector.create_from_options(options)

    def detect(self, frame_bgr):
        """Run detection on a BGR frame. Returns a list of Detection."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(image)
        h, w = frame_bgr.shape[:2]
        detections = []
        for det in result.detections:
            if not det.categories:
                continue
            cat = det.categories[0]
            box = det.bounding_box
            x1 = max(0, int(box.origin_x))
            y1 = max(0, int(box.origin_y))
            x2 = min(w - 1, int(box.origin_x + box.width))
            y2 = min(h - 1, int(box.origin_y + box.height))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(Detection(cat.category_name, float(cat.score), (x1, y1, x2, y2)))
        return detections

    def close(self):
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass
            self._detector = None
