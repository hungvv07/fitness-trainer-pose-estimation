"""Unit tests for object counting (tracker, unique counts, line crossing).

Uses synthetic detections, so no camera or model download is needed:
    python -m unittest test_object_counting
"""

import unittest

from object_detection.counter import ObjectCounter
from object_detection.detector import Detection

W, H = 640, 480


def box(cx, cy, w=60, h=120):
    return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))


def walk(counter, label, xs_ys, score=0.9):
    for cx, cy in xs_ys:
        counter.process([Detection(label, score, box(cx, cy))], W, H)


class ObjectCounterTest(unittest.TestCase):
    def test_person_walking_down_crosses_line_once(self):
        counter = ObjectCounter({"line_orientation": "horizontal", "line_position": 0.5})
        walk(counter, "person", [(320, y) for y in range(100, 400, 15)])
        person = counter.stats()["groups"]["person"]
        self.assertEqual(person["unique"], 1)
        self.assertEqual(person["cross_fwd"], 1)
        self.assertEqual(person["cross_bwd"], 0)
        self.assertEqual(person["current"], 1)

    def test_vehicle_moving_left_on_vertical_line(self):
        counter = ObjectCounter({"line_orientation": "vertical", "line_position": 0.5})
        walk(counter, "car", [(x, 240) for x in range(560, 80, -20)])
        vehicle = counter.stats()["groups"]["vehicle"]
        self.assertEqual(vehicle["unique"], 1)
        self.assertEqual(vehicle["cross_bwd"], 1)
        self.assertEqual(vehicle["cross_fwd"], 0)
        self.assertEqual(vehicle["labels"]["car"]["label_vi"], "Ô tô")

    def test_jitter_on_line_is_not_counted(self):
        counter = ObjectCounter({"line_position": 0.5})
        walk(counter, "person", [(320, 240 + (3 if i % 2 else -3)) for i in range(30)])
        self.assertEqual(counter.stats()["groups"]["person"]["cross_fwd"], 0)
        self.assertEqual(counter.stats()["groups"]["person"]["cross_bwd"], 0)

    def test_short_false_positive_is_not_counted(self):
        counter = ObjectCounter()
        walk(counter, "person", [(320, 200)])  # seen in a single frame only
        for _ in range(20):
            counter.process([], W, H)
        self.assertEqual(counter.stats()["groups"]["person"]["unique"], 0)

    def test_two_people_counted_separately(self):
        counter = ObjectCounter()
        for i in range(10):
            counter.process([
                Detection("person", 0.9, box(100 + i * 2, 200)),
                Detection("person", 0.8, box(500 - i * 2, 200)),
            ], W, H)
        person = counter.stats()["groups"]["person"]
        self.assertEqual(person["unique"], 2)
        self.assertEqual(person["current"], 2)

    def test_disabled_group_and_low_confidence_are_ignored(self):
        counter = ObjectCounter({"groups": {"vehicle": False}, "confidence": 0.5})
        for _ in range(5):
            counter.process([
                Detection("car", 0.9, box(100, 200)),
                Detection("person", 0.3, box(400, 200)),
                Detection("dog", 0.9, box(300, 300)),
            ], W, H)
        groups = counter.stats()["groups"]
        self.assertEqual(groups["vehicle"]["unique"], 0)
        self.assertEqual(groups["person"]["unique"], 0)

    def test_car_truck_flicker_is_one_vehicle(self):
        counter = ObjectCounter()
        for i in range(10):
            label = "car" if i % 2 else "truck"
            dets = [Detection(label, 0.9, box(200 + i * 5, 200))]
            if i == 4:  # both labels on the same box in one frame
                dets.append(Detection("car", 0.7, box(200 + i * 5, 200)))
            counter.process(dets, W, H)
        self.assertEqual(counter.stats()["groups"]["vehicle"]["unique"], 1)

    def test_everyday_objects_counted_by_label(self):
        counter = ObjectCounter({"groups": {"everyday": True}})
        for _ in range(5):
            counter.process([
                Detection("cup", 0.8, box(100, 100, 40, 40)),
                Detection("cup", 0.8, box(300, 100, 40, 40)),
                Detection("cell phone", 0.7, box(500, 300, 40, 80)),
            ], W, H)
        everyday = counter.stats()["groups"]["everyday"]
        self.assertEqual(everyday["current"], 3)
        self.assertEqual(everyday["labels"]["cup"]["current"], 2)
        self.assertEqual(everyday["labels"]["cell phone"]["unique"], 1)

    def test_single_image_counts_immediately(self):
        counter = ObjectCounter({"groups": {"everyday": True}}, single_image=True)
        counter.process([Detection("book", 0.6, box(100, 100)), Detection("person", 0.9, box(400, 200))], W, H)
        groups = counter.stats()["groups"]
        self.assertEqual(groups["everyday"]["current"], 1)
        self.assertEqual(groups["person"]["current"], 1)

    def test_moving_line_does_not_create_fake_crossing(self):
        counter = ObjectCounter({"line_position": 0.8})
        walk(counter, "person", [(320, 200)] * 5)
        counter.update_config({"line_position": 0.2})
        walk(counter, "person", [(320, 200)] * 5)
        person = counter.stats()["groups"]["person"]
        self.assertEqual(person["cross_fwd"] + person["cross_bwd"], 0)

    def test_draw_does_not_crash(self):
        import numpy as np
        counter = ObjectCounter({"groups": {"everyday": True}})
        walk(counter, "person", [(320, y) for y in range(100, 300, 20)])
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        counter.draw(frame, ["FPS 10.0"])
        self.assertGreater(int(frame.sum()), 0)


if __name__ == "__main__":
    unittest.main()
