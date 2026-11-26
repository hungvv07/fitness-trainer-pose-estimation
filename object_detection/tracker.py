"""Lightweight multi-object tracker (greedy IoU + centroid matching).

Good enough to give each detected object a stable ID across frames, so we can
count unique objects and detect when an object crosses a counting line.
No extra dependencies (no SORT/DeepSORT, no scipy).
"""

import math
from collections import Counter, deque
from dataclasses import dataclass, field

from .labels import EVERYDAY, group_of


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def centroid(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def match_key(label):
    """Objects are only matched to tracks with the same key.

    People and vehicles match within their group (the detector often flips
    between car/truck for the same vehicle); everyday objects must keep the
    exact label (a cup should not turn into a bottle).
    """
    group = group_of(label)
    if group is None or group == EVERYDAY:
        return label
    return group


@dataclass
class Track:
    id: int
    key: str
    bbox: tuple
    score: float
    labels: Counter = field(default_factory=Counter)
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    history: deque = field(default_factory=lambda: deque(maxlen=40))
    # Counting-line state, managed by the counter.
    anchor_side: int = 0

    @property
    def label(self):
        return self.labels.most_common(1)[0][0]

    @property
    def group(self):
        return group_of(self.label)

    @property
    def center(self):
        return centroid(self.bbox)


class Tracker:
    def __init__(self, min_hits=3, max_misses=12, iou_threshold=0.2, max_center_dist=0.6):
        """
        Args:
            min_hits: frames an object must be seen before it is confirmed
                (filters one-frame false positives).
            max_misses: frames a track survives without a matching detection.
            iou_threshold: minimum IoU to match a detection to a track.
            max_center_dist: fallback match when IoU is low (fast motion): the
                centroid distance divided by the track box diagonal.
        """
        self.min_hits = min_hits
        self.max_misses = max_misses
        self.iou_threshold = iou_threshold
        self.max_center_dist = max_center_dist
        self.tracks = []
        self._next_id = 1

    def reset(self):
        self.tracks = []
        self._next_id = 1

    def update(self, detections):
        """Update tracks with this frame's detections.

        Returns (visible, newly_confirmed): tracks matched in this frame that
        are confirmed, and tracks that became confirmed in this frame.
        """
        candidates = []
        for ti, track in enumerate(self.tracks):
            tx1, ty1, tx2, ty2 = track.bbox
            diag = math.hypot(tx2 - tx1, ty2 - ty1) or 1.0
            tcx, tcy = track.center
            for di, det in enumerate(detections):
                if match_key(det.label) != track.key:
                    continue
                overlap = iou(track.bbox, det.bbox)
                dcx, dcy = centroid(det.bbox)
                dist = math.hypot(dcx - tcx, dcy - tcy) / diag
                if overlap >= self.iou_threshold or dist <= self.max_center_dist:
                    cost = (1.0 - overlap) + 0.5 * min(dist, 2.0)
                    candidates.append((cost, ti, di))

        candidates.sort()
        used_tracks, used_dets = set(), set()
        matched = []
        for _cost, ti, di in candidates:
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            matched.append((self.tracks[ti], detections[di]))

        newly_confirmed = []
        for track, det in matched:
            track.bbox = det.bbox
            track.score = det.score
            track.labels[det.label] += 1
            track.hits += 1
            track.misses = 0
            track.history.append(track.center)
            if not track.confirmed and track.hits >= self.min_hits:
                track.confirmed = True
                newly_confirmed.append(track)

        for ti, track in enumerate(self.tracks):
            if ti not in used_tracks:
                track.misses += 1

        for di, det in enumerate(detections):
            if di in used_dets:
                continue
            track = Track(id=self._next_id, key=match_key(det.label), bbox=det.bbox, score=det.score)
            track.labels[det.label] += 1
            track.history.append(track.center)
            self._next_id += 1
            if self.min_hits <= 1:
                track.confirmed = True
                newly_confirmed.append(track)
            self.tracks.append(track)

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        visible = [t for t in self.tracks if t.misses == 0 and t.confirmed]
        return visible, newly_confirmed
