"""Counting logic: per-frame counts, unique objects and line crossings.

Kept independent from the camera/video loop so it can be unit tested with
synthetic detections.
"""

import copy
import threading
from collections import defaultdict

import cv2

from .detector import DEFAULT_MODEL, MODELS
from .labels import EVERYDAY, GROUPS, PERSON, VEHICLE, group_of, label_vi
from .tracker import Tracker, iou, match_key

DEFAULT_CONFIG = {
    "groups": {PERSON: True, VEHICLE: True, EVERYDAY: False},
    "line_enabled": True,
    "line_orientation": "horizontal",  # horizontal | vertical
    "line_position": 0.5,               # 0..1 of frame height (or width)
    "confidence": 0.4,
    "show_trails": True,
    "model": DEFAULT_MODEL,
}

# Dead band around the counting line (fraction of the frame) to avoid
# counting jitter of an object standing right on the line.
LINE_DEAD_BAND = 0.015

# BGR colours for the overlay
GROUP_COLORS = {
    PERSON: (80, 200, 60),
    VEHICLE: (0, 165, 255),
    EVERYDAY: (230, 140, 40),
}
LINE_COLOR = (0, 230, 255)


def _empty_label_stats():
    return {"current": 0, "unique": 0, "cross_fwd": 0, "cross_bwd": 0}


def sanitize_config(current, update):
    """Merge a partial config update into the current config, validating values."""
    cfg = copy.deepcopy(current)
    if not isinstance(update, dict):
        return cfg
    groups = update.get("groups")
    if isinstance(groups, dict):
        for g in GROUPS:
            if g in groups:
                cfg["groups"][g] = bool(groups[g])
    for key in ("line_enabled", "show_trails"):
        if key in update:
            cfg[key] = bool(update[key])
    if update.get("line_orientation") in ("horizontal", "vertical"):
        cfg["line_orientation"] = update["line_orientation"]
    for key, lo, hi in (("line_position", 0.05, 0.95), ("confidence", 0.15, 0.95)):
        if key in update:
            try:
                cfg[key] = min(hi, max(lo, float(update[key])))
            except (TypeError, ValueError):
                pass
    if update.get("model") in MODELS:
        cfg["model"] = update["model"]
    return cfg


def suppress_duplicates(detections, iou_threshold=0.7):
    """Cross-label NMS within a match key (e.g. car + truck on the same vehicle)."""
    kept = []
    for det in sorted(detections, key=lambda d: d.score, reverse=True):
        key = match_key(det.label)
        if any(match_key(k.label) == key and iou(k.bbox, det.bbox) > iou_threshold for k in kept):
            continue
        kept.append(det)
    return kept


class ObjectCounter:
    def __init__(self, config=None, single_image=False):
        self.lock = threading.RLock()
        self.config = sanitize_config(DEFAULT_CONFIG, config or {})
        self.single_image = single_image
        self.tracker = Tracker(min_hits=1 if single_image else 3)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        with self.lock:
            self.tracker.reset()
            self.label_stats = defaultdict(_empty_label_stats)
            self.visible = []
            self.frame_size = (0, 0)

    def update_config(self, update):
        with self.lock:
            old = self.config
            self.config = sanitize_config(old, update)
            line_changed = (old["line_orientation"] != self.config["line_orientation"]
                            or old["line_position"] != self.config["line_position"]
                            or old["line_enabled"] != self.config["line_enabled"])
            if line_changed:
                # Re-anchor every track on the new line so moving the line
                # does not create fake crossings.
                for track in self.tracker.tracks:
                    track.anchor_side = 0
            return copy.deepcopy(self.config)

    # -------------------------------------------------------------- per frame
    def filter_detections(self, detections):
        cfg = self.config
        kept = []
        for det in detections:
            group = group_of(det.label)
            if group is None or not cfg["groups"].get(group, False):
                continue
            if det.score < cfg["confidence"]:
                continue
            kept.append(det)
        return suppress_duplicates(kept)

    def _side(self, point, width, height):
        cfg = self.config
        if cfg["line_orientation"] == "horizontal":
            value, size = point[1], height
        else:
            value, size = point[0], width
        offset = value - cfg["line_position"] * size
        if abs(offset) <= LINE_DEAD_BAND * size:
            return 0
        return 1 if offset > 0 else -1

    def process(self, detections, width, height):
        """Feed raw detections of one frame. Returns the visible confirmed tracks."""
        with self.lock:
            self.frame_size = (width, height)
            dets = self.filter_detections(detections)
            visible, newly_confirmed = self.tracker.update(dets)

            for track in newly_confirmed:
                self.label_stats[track.label]["unique"] += 1

            if self.config["line_enabled"] and not self.single_image:
                for track in self.tracker.tracks:
                    if track.misses:
                        continue
                    side = self._side(track.center, width, height)
                    if side == 0:
                        continue
                    if track.anchor_side == 0:
                        track.anchor_side = side
                        continue
                    if track.confirmed and side != track.anchor_side:
                        # -1 -> +1 is "forward" (top to bottom / left to right)
                        direction = "cross_fwd" if track.anchor_side < 0 else "cross_bwd"
                        self.label_stats[track.label][direction] += 1
                        track.anchor_side = side

            for stats in self.label_stats.values():
                stats["current"] = 0
            for track in visible:
                self.label_stats[track.label]["current"] += 1
            self.visible = visible
            return visible

    # ------------------------------------------------------------------ stats
    def stats(self):
        with self.lock:
            groups = {g: {**_empty_label_stats(), "labels": {}} for g in GROUPS}
            for label, s in self.label_stats.items():
                group = group_of(label)
                if group is None:
                    continue
                agg = groups[group]
                for k in ("current", "unique", "cross_fwd", "cross_bwd"):
                    agg[k] += s[k]
                agg["labels"][label] = {**s, "label_vi": label_vi(label)}
            cfg = self.config
            if cfg["line_orientation"] == "horizontal":
                directions = {"fwd": "Từ trên xuống", "bwd": "Từ dưới lên"}
            else:
                directions = {"fwd": "Trái sang phải", "bwd": "Phải sang trái"}
            return {
                "groups": groups,
                "directions": directions,
                "config": copy.deepcopy(cfg),
            }

    # --------------------------------------------------------------- drawing
    def draw(self, frame, info_lines=()):
        with self.lock:
            cfg = self.config
            h, w = frame.shape[:2]
            scale = max(0.5, min(1.2, w / 1280.0))
            thickness = max(1, int(round(2 * scale)))

            if cfg["line_enabled"] and not self.single_image:
                self._draw_line(frame, w, h, scale, thickness)

            for track in self.visible:
                color = GROUP_COLORS.get(track.group, (200, 200, 200))
                x1, y1, x2, y2 = track.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                text = f"{track.label} #{track.id} {track.score:.2f}"
                self._label(frame, text, (x1, y1), color, 0.5 * scale)
                if cfg["show_trails"] and len(track.history) > 1:
                    pts = [(int(x), int(y)) for x, y in track.history]
                    for i in range(1, len(pts)):
                        cv2.line(frame, pts[i - 1], pts[i], color, thickness)
                    cv2.circle(frame, pts[-1], thickness + 2, color, -1)

            self._draw_hud(frame, scale, info_lines)

    def _draw_line(self, frame, w, h, scale, thickness):
        cfg = self.config
        totals = {"cross_fwd": 0, "cross_bwd": 0}
        for label, s in self.label_stats.items():
            group = group_of(label)
            if group and cfg["groups"].get(group):
                totals["cross_fwd"] += s["cross_fwd"]
                totals["cross_bwd"] += s["cross_bwd"]
        if cfg["line_orientation"] == "horizontal":
            y = int(cfg["line_position"] * h)
            cv2.line(frame, (0, y), (w, y), LINE_COLOR, thickness + 1)
            text = f"DOWN {totals['cross_fwd']}  |  UP {totals['cross_bwd']}"
            self._label(frame, text, (10, y - 4), LINE_COLOR, 0.6 * scale, text_color=(0, 0, 0))
        else:
            x = int(cfg["line_position"] * w)
            cv2.line(frame, (x, 0), (x, h), LINE_COLOR, thickness + 1)
            text = f"RIGHT {totals['cross_fwd']}  |  LEFT {totals['cross_bwd']}"
            self._label(frame, text, (x + 6, h - 12), LINE_COLOR, 0.6 * scale, text_color=(0, 0, 0))

    def _draw_hud(self, frame, scale, info_lines):
        cfg = self.config
        lines = []
        names = {PERSON: "People", VEHICLE: "Vehicles", EVERYDAY: "Objects"}
        per_group = {g: [0, 0] for g in GROUPS}
        for label, s in self.label_stats.items():
            group = group_of(label)
            if group:
                per_group[group][0] += s["current"]
                per_group[group][1] += s["unique"]
        for g in GROUPS:
            if cfg["groups"].get(g):
                cur, uniq = per_group[g]
                lines.append((f"{names[g]}: {cur} in frame / {uniq} seen", GROUP_COLORS[g]))
        for text in info_lines:
            lines.append((text, (220, 220, 220)))
        if not lines:
            return
        font_scale = 0.55 * scale
        line_h = int(26 * scale)
        width = int(max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0] for t, _ in lines)) + 20
        height = line_h * len(lines) + 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (8 + width, 8 + height), (20, 20, 30), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        for i, (text, color) in enumerate(lines):
            cv2.putText(frame, text, (18, 8 + line_h * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, 1, cv2.LINE_AA)

    @staticmethod
    def _label(frame, text, origin, bg_color, font_scale, text_color=(255, 255, 255)):
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        x, y = int(origin[0]), int(origin[1])
        y = max(th + base + 2, y)
        cv2.rectangle(frame, (x, y - th - base - 4), (x + tw + 6, y), bg_color, -1)
        cv2.putText(frame, text, (x + 3, y - base - 1), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, text_color, 1, cv2.LINE_AA)
