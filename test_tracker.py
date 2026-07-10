import unittest
import numpy as np

from tracker import VehicleTracker


class TestVehicleTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = VehicleTracker()

    def test_tracker_initialization(self):
        self.assertIsNotNone(self.tracker)

    def test_empty_detections(self):

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        tracks = self.tracker.update([], frame)

        self.assertEqual(tracks, [])

    def test_single_detection(self):

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        detections = [
            [100, 100, 200, 200, 0.95]
        ]

        tracks = self.tracker.update(detections, frame)

        self.assertIsInstance(tracks, list)

    def test_multiple_detections(self):

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        detections = [
            [100, 100, 200, 200, 0.95],
            [300, 200, 420, 350, 0.88],
            [500, 300, 650, 450, 0.91]
        ]

        tracks = self.tracker.update(detections, frame)

        self.assertIsInstance(tracks, list)


if __name__ == "__main__":
    unittest.main()
