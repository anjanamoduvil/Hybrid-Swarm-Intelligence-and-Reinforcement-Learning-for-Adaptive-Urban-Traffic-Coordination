#!/usr/bin/env python3
"""
Traffic Monitoring & Adaptive Signal System - Member 1: Test Suite

This test suite verifies:
1. Frame Count Assertion: Validates that the video pipeline processes every frame of the input source.
2. Detection Shape/Format Assertion: Validates the data structure of output detections.

We programmatically generate a synthetic video to make this test completely self-contained,
speedy, and runnable in any environment without external video files.
We also use unittest.mock to test YOLOv8 detection parsing logic deterministically.
"""

import os
import unittest
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import yaml

from detector import TrafficDetector


class TestTrafficDetector(unittest.TestCase):
    """
    Unit and integration test cases for Member 1's TrafficDetector.
    """

    @classmethod
    def setUpClass(cls):
        """
        Creates a temporary config file and a synthetic test video programmatically.
        """
        cls.test_config_path = "test_config.yaml"
        cls.test_video_path = "test_synthetic.mp4"
        cls.num_frames = 15
        cls.width = 640
        cls.height = 480
        cls.fps = 10

        # Create temporary YAML config
        config_data = {
            "model_path": "yolov8n.pt",
            "video_path": cls.test_video_path,
            "conf_threshold": 0.25,
            "resize_dims": [800, 600]
        }
        with open(cls.test_config_path, "w") as f:
            yaml.dump(config_data, f)

        # Create synthetic video using OpenCV VideoWriter
        # We draw basic moving rectangles to simulate motion
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(cls.test_video_path, fourcc, cls.fps, (cls.width, cls.height))

        for idx in range(cls.num_frames):
            # Create a blank black frame
            frame = np.zeros((cls.height, cls.width, 3), dtype=np.uint8)
            # Draw a moving square (simulating a vehicle)
            x_pos = 50 + idx * 25
            y_pos = 200
            cv2.rectangle(frame, (x_pos, y_pos), (x_pos + 80, y_pos + 60), (0, 255, 0), -1)
            out.write(frame)
        out.release()

    @classmethod
    def tearDownClass(cls):
        """
        Cleans up the temporary files after all tests run.
        """
        if os.path.exists(cls.test_config_path):
            os.remove(cls.test_config_path)
        if os.path.exists(cls.test_video_path):
            os.remove(cls.test_video_path)

    def test_configuration_loading(self):
        """
        Test if configuration parameters are loaded correctly from config.yaml.
        """
        detector = TrafficDetector(config_path=self.test_config_path)
        self.assertEqual(detector.model_path, "yolov8n.pt")
        self.assertEqual(detector.video_path, self.test_video_path)
        self.assertEqual(detector.conf_threshold, 0.25)
        self.assertEqual(detector.resize_dims, [800, 600])

    def test_video_pipeline_frame_count(self):
        """
        Asserts that the number of frames processed matches the source video frame count.
        """
        detector = TrafficDetector(config_path=self.test_config_path)
        
        # Mock the YOLO model to avoid slow weight loading and downloads during integration check
        detector.model = MagicMock()
        
        # Mock model return value to return empty detections for mock frames
        mock_result = MagicMock()
        mock_result.boxes = None
        detector.model.return_value = [mock_result]

        processed_frames = 0
        for frame, detections in detector.process_video():
            processed_frames += 1
            # Verify frame was successfully resized to the dimensions from config.yaml
            self.assertEqual(frame.shape[1], 800)  # Width
            self.assertEqual(frame.shape[0], 600)  # Height
            # Verify detections list is returned (even if empty in mock)
            self.assertIsInstance(detections, list)

        # Assert Frame Count
        self.assertEqual(processed_frames, self.num_frames, 
                         f"Expected to process {self.num_frames} frames, but got {processed_frames}.")

    def test_detection_shape_and_parsing(self):
        """
        Verifies that YOLOv8 predictions are correctly parsed into the expected output shape/format:
        (x1, y1, x2, y2, confidence, class_id, class_name)
        """
        detector = TrafficDetector(config_path=self.test_config_path)
        
        # Set up a mock YOLOv8 result object with dummy box detections
        mock_model = MagicMock()
        detector.model = mock_model

        mock_boxes = MagicMock()
        # Mock 2 detections: class 2 (Car) and class 7 (Truck)
        mock_boxes.xyxy.cpu().numpy.return_value = np.array([
            [100.1, 150.2, 250.3, 300.4],
            [400.5, 200.6, 600.7, 450.8]
        ], dtype=np.float32)
        mock_boxes.conf.cpu().numpy.return_value = np.array([0.88, 0.92], dtype=np.float32)
        mock_boxes.cls.cpu().numpy.return_value = np.array([2.0, 7.0], dtype=np.float32)

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]

        # Process a blank frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.run_inference(dummy_frame)

        # Assert correct count of parsed motorized vehicle classes
        self.assertEqual(len(detections), 2)

        # Verify shapes and formats for each detection
        first_det = detections[0]
        self.assertIn("bbox", first_det)
        self.assertIn("confidence", first_det)
        self.assertIn("class_id", first_det)
        self.assertIn("class_name", first_det)

        # 1. BBox contains 4 items and are integers
        self.assertEqual(len(first_det["bbox"]), 4)
        self.assertEqual(first_det["bbox"], [100, 150, 250, 300]) # Cast to int check

        # 2. Confidence is float
        self.assertIsInstance(first_det["confidence"], float)
        self.assertAlmostEqual(first_det["confidence"], 0.88, places=4)

        # 3. Class ID and Name match mapped COCO classes
        self.assertEqual(first_det["class_id"], 2)
        self.assertEqual(first_det["class_name"], "Car")

        # Check second detection (Truck)
        second_det = detections[1]
        self.assertEqual(second_det["bbox"], [400, 200, 600, 450])
        self.assertEqual(second_det["class_id"], 7)
        self.assertEqual(second_det["class_name"], "Truck")

    def test_invalid_class_filtering(self):
        """
        Verifies that non-vehicle classes (e.g. Dog, Bird) are correctly filtered out.
        """
        detector = TrafficDetector(config_path=self.test_config_path)
        detector.model = MagicMock()

        mock_boxes = MagicMock()
        # Mock 2 detections: class 16 (Dog - should be filtered) and class 5 (Bus - should be kept)
        mock_boxes.xyxy.cpu().numpy.return_value = np.array([
            [10.0, 10.0, 50.0, 100.0],
            [100.0, 100.0, 300.0, 200.0]
        ], dtype=np.float32)
        mock_boxes.conf.cpu().numpy.return_value = np.array([0.90, 0.85], dtype=np.float32)
        mock_boxes.cls.cpu().numpy.return_value = np.array([16.0, 5.0], dtype=np.float32)

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        detector.model.return_value = [mock_result]

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.run_inference(dummy_frame)

        # Assert only the Bus (class 5) was retained
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["class_name"], "Bus")
        self.assertEqual(detections[0]["class_id"], 5)


if __name__ == "__main__":
    unittest.main()
