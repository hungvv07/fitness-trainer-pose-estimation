"""Background worker that reads a camera / video / image, runs detection and
keeps the latest annotated JPEG plus counting stats.

The web page pulls single JPEG frames and polls a status endpoint, so the page
also works with the single-threaded Flask server used by app.py (an MJPEG
stream would block every other request).
"""

import logging
import os
import threading
import time

import cv2

from .counter import ObjectCounter
from .detector import MODELS, ObjectDetector, model_path

logger = logging.getLogger(__name__)

MAX_FRAME_WIDTH = 1280
JPEG_QUALITY = 80


def _resize(frame):
    h, w = frame.shape[:2]
    if w <= MAX_FRAME_WIDTH:
        return frame
    scale = MAX_FRAME_WIDTH / float(w)
    return cv2.resize(frame, (MAX_FRAME_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)


class CountingSession:
    def __init__(self, open_camera, release_camera):
        """
        Args:
            open_camera: callable returning an opened cv2.VideoCapture (or None).
            release_camera: callable releasing that camera.
        """
        self._open_camera = open_camera
        self._release_camera = release_camera
        self.counter = ObjectCounter()
        self._detector = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._jpeg = None
        self._frame_seq = 0
        self._config_version = 0
        self._status = self._idle_status()

    @staticmethod
    def _idle_status():
        return {
            "state": "idle",  # idle | loading_model | running | finished | error
            "source": None,
            "source_name": None,
            "error": None,
            "fps": 0.0,
            "inference_ms": 0.0,
            "frame_size": None,
            "progress": None,
        }

    # ----------------------------------------------------------- public API
    def start(self, source, path=None, source_name=None):
        if source not in ("camera", "video", "image"):
            raise ValueError("source must be camera, video or image")
        if source in ("video", "image") and (not path or not os.path.exists(path)):
            raise ValueError("Không tìm thấy file đã tải lên")
        self.stop()
        config = self.counter.config
        self.counter = ObjectCounter(config, single_image=(source == "image"))
        with self._lock:
            self._jpeg = None
            self._status = self._idle_status()
            self._status.update(state="loading_model", source=source, source_name=source_name)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(source, path), daemon=True)
        self._thread.start()

    def stop(self):
        thread = self._thread
        if thread and thread.is_alive():
            self._stop.set()
            thread.join(timeout=5)
        self._thread = None
        with self._lock:
            if self._status["state"] in ("loading_model", "running"):
                self._status["state"] = "idle"

    def stop_if_camera(self):
        if self._status.get("source") == "camera" and self._thread and self._thread.is_alive():
            self.stop()

    def reset_counts(self):
        self.counter.reset()
        self._config_version += 1

    def update_config(self, update):
        cfg = self.counter.update_config(update)
        self._config_version += 1
        return cfg

    def status(self):
        with self._lock:
            status = dict(self._status)
            status["frame_seq"] = self._frame_seq
        model_key = self.counter.config["model"]
        status["model"] = model_key
        status["model_name"] = MODELS[model_key]["name"]
        status["model_downloaded"] = os.path.exists(model_path(model_key))
        status.update(self.counter.stats())
        return status

    def latest_frame(self):
        with self._lock:
            return self._frame_seq, self._jpeg

    # --------------------------------------------------------------- worker
    def _set_status(self, **kwargs):
        with self._lock:
            self._status.update(kwargs)

    def _publish(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return
        with self._lock:
            self._jpeg = buf.tobytes()
            self._frame_seq += 1
            self._status["frame_size"] = [frame.shape[1], frame.shape[0]]

    def _ensure_detector(self):
        model_key = self.counter.config["model"]
        if self._detector is not None and self._detector.model_key == model_key:
            return self._detector
        self._set_status(state="loading_model")
        if self._detector is not None:
            self._detector.close()
            self._detector = None
        logger.info("Loading object detection model %s", model_key)
        self._detector = ObjectDetector(model_key)
        return self._detector

    def _process(self, frame, detector, info_lines):
        t0 = time.time()
        detections = detector.detect(frame)
        inference_ms = (time.time() - t0) * 1000.0
        h, w = frame.shape[:2]
        self.counter.process(detections, w, h)
        out = frame.copy()
        self.counter.draw(out, info_lines)
        return out, detections, inference_ms

    def _run(self, source, path):
        cap = None
        try:
            detector = self._ensure_detector()
            if source == "image":
                self._run_image(path)
                return
            if source == "camera":
                cap = self._open_camera()
                if cap is None or not cap.isOpened():
                    raise RuntimeError("Không mở được camera. Kiểm tra thiết bị hoặc quyền truy cập.")
            else:
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    raise RuntimeError("Không đọc được file video")

            video_fps = cap.get(cv2.CAP_PROP_FPS) if source == "video" else 0
            if not video_fps or video_fps != video_fps or video_fps > 120:
                video_fps = 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if source == "video" else 0
            frame_index = 0
            fail_count = 0
            fps_ema = 0.0
            self._set_status(state="running")

            while not self._stop.is_set():
                loop_start = time.time()
                ok, frame = cap.read()
                if not ok or frame is None:
                    if source == "video":
                        self._set_status(state="finished", progress=100.0)
                        logger.info("Object counting video finished")
                        return
                    fail_count += 1
                    if fail_count > 30:
                        raise RuntimeError("Mất tín hiệu camera")
                    time.sleep(0.05)
                    continue
                fail_count = 0
                frame_index += 1

                if self.counter.config["model"] != detector.model_key:
                    detector = self._ensure_detector()
                    self._set_status(state="running")

                frame = _resize(frame)
                info = [f"FPS {fps_ema:.1f}"]
                out, _dets, inference_ms = self._process(frame, detector, info)
                self._publish(out)

                elapsed = time.time() - loop_start
                fps_ema = (1.0 / elapsed) if fps_ema == 0 else 0.9 * fps_ema + 0.1 * (1.0 / max(elapsed, 1e-3))
                progress = (100.0 * frame_index / total_frames) if total_frames > 0 else None
                self._set_status(fps=round(fps_ema, 1), inference_ms=round(inference_ms, 1),
                                 progress=None if progress is None else round(min(progress, 100.0), 1))

                if source == "video":
                    # Play at the video's own speed when processing is faster.
                    remaining = (1.0 / video_fps) - (time.time() - loop_start)
                    if remaining > 0:
                        time.sleep(remaining)
        except Exception as e:
            logger.exception("Object counting error")
            self._set_status(state="error", error=str(e))
        finally:
            if cap is not None:
                if source == "camera":
                    self._release_camera()
                else:
                    cap.release()

    def _run_image(self, path):
        image = cv2.imread(path)
        if image is None:
            raise RuntimeError("Không đọc được file ảnh")
        image = _resize(image)
        detector = self._ensure_detector()
        t0 = time.time()
        detections = detector.detect(image)
        inference_ms = (time.time() - t0) * 1000.0
        self._set_status(state="running", inference_ms=round(inference_ms, 1), fps=0.0)
        rendered_version = None
        # Stay alive so toggles / slider changes re-render the same image.
        while not self._stop.is_set():
            if self.counter.config["model"] != detector.model_key:
                detector = self._ensure_detector()
                t0 = time.time()
                detections = detector.detect(image)
                inference_ms = (time.time() - t0) * 1000.0
                self._set_status(state="running", inference_ms=round(inference_ms, 1))
                rendered_version = None
            if rendered_version != self._config_version:
                rendered_version = self._config_version
                h, w = image.shape[:2]
                self.counter.reset()
                self.counter.process(detections, w, h)
                out = image.copy()
                self.counter.draw(out)
                self._publish(out)
            time.sleep(0.1)
