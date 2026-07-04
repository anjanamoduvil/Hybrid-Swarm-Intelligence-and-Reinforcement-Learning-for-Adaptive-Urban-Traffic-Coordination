#!/usr/bin/env python3
"""
Traffic Monitoring & Adaptive Signal System - Member 1: Video Pipeline & YOLO Setup

This module contains the core TrafficDetector class that:
1. Loads configuration from config.yaml or overrides via CLI.
2. Manages video ingestion and resizing.
3. Conducts high-performance YOLOv8 inference on each frame.
4. Renders a premium OpenCV Heads-Up Display (HUD) with anti-aliased detections,
   confidence scores, class colors, dynamic frame numbers, and FPS tracking.
"""

import os
import time
import argparse
import yaml
import cv2
import numpy as np

# Try to import YOLO, print helpful error if not installed
try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "The 'ultralytics' library is required. Please install it using 'pip install ultralytics'"
    )


class TrafficDetector:
    """
    Manages the video pipeline and YOLOv8 inference for traffic coordination.
    """

    # Target class IDs from COCO dataset representing road vehicles
    # Mapped class 0 to Pedestrian to detect walking people, alongside motorized road vehicles.
    VEHICLE_CLASSES = {
        0: "Pedestrian",
        1: "Bicycle",
        2: "Car",
        3: "Motorcycle",
        5: "Bus",
        7: "Truck"
    }

    # Highly aesthetic and professional BGR color palettes (Muted Enterprise Slate Theme)
    PALETTE = {
        "Pedestrian": (220, 220, 220), # Muted Off-White / Silver
        "Bicycle": (240, 200, 100),    # Cool Ice-Blue
        "Car": (230, 160, 50),         # Professional Cobalt Blue
        "Motorcycle": (180, 220, 80),  # Muted Teal / Sage Green
        "Bus": (180, 130, 110),        # Corporate Slate Blue
        "Truck": (80, 130, 220),       # Muted Autumn Gold / Amber
        "Default": (180, 180, 180)     # Clean Silver Grey
    }

    def __init__(self, config_path="config.yaml"):
        """
        Initializes the detector using declarative configuration parameters.
        """
        self.config_path = config_path
        self.model_path = "yolov8n.pt"
        self.video_path = "traffic_sample.mp4"
        self.conf_threshold = 0.25
        self.resize_dims = None  # None means use original video dimensions
        
        # Load configuration file
        self.load_config()
        self.model = None

    def load_config(self):
        """
        Safely loads parameters from config.yaml if it exists.
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    if config:
                        self.model_path = config.get("model_path", self.model_path)
                        self.video_path = config.get("video_path", self.video_path)
                        self.conf_threshold = float(config.get("conf_threshold", self.conf_threshold))
                        self.resize_dims = config.get("resize_dims", self.resize_dims)
                        print(f"[Detector] Configuration loaded successfully from {self.config_path}")
            except Exception as e:
                print(f"[Detector] Warning: Could not parse config file ({e}). Using default values.")
        else:
            print(f"[Detector] Config file '{self.config_path}' not found. Using default parameters.")

    def load_model(self):
        """
        Initializes the YOLOv8 model weights.
        """
        print(f"[Detector] Initializing YOLOv8 model from '{self.model_path}'...")
        try:
            self.model = YOLO(self.model_path)
            print("[Detector] YOLOv8 model loaded successfully.")
        except Exception as e:
            print(f"[Detector] Critical: Failed to load YOLO model: {e}")
            raise e

    def run_inference(self, frame):
        """
        Performs YOLOv8 inference on a single frame, filtering by vehicle classes and conf_threshold.
        Includes overlapping/rider filtering logic to separate true walking pedestrians from motorcycle riders.
        """
        if self.model is None:
            self.load_model()

        # Run inference using Ultralytics API
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        
        raw_vehicles = []
        raw_persons = []
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)

            for bbox, conf, cls in zip(boxes, confs, clss):
                if cls in self.VEHICLE_CLASSES:
                    x1, y1, x2, y2 = map(int, bbox)
                    class_name = self.VEHICLE_CLASSES[cls]
                    det = {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(conf),
                        "class_id": int(cls),
                        "class_name": class_name
                    }
                    if cls == 0:
                        raw_persons.append(det)
                    else:
                        raw_vehicles.append(det)

        # Helper to compute intersection fraction: Area(Intersection) / Area(Person Box)
        def box_intersection_fraction(box_p, box_v):
            px1, py1, px2, py2 = box_p
            vx1, vy1, vx2, vy2 = box_v
            ix1 = max(px1, vx1)
            iy1 = max(py1, vy1)
            ix2 = min(px2, vx2)
            iy2 = min(py2, vy2)
            if ix1 < ix2 and iy1 < iy2:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                person_area = (px2 - px1) * (py2 - py1)
                if person_area > 0:
                    return inter_area / person_area
            return 0.0

        # Reclassify and filter riders:
        # 1. If a person overlaps with a vehicle, they are a rider (we ignore the duplicate box).
        # 2. If a person does NOT overlap with any vehicle, they are a true Pedestrian.
        true_pedestrians = []
        for p_det in raw_persons:
            has_overlap = False
            for v_det in raw_vehicles:
                if box_intersection_fraction(p_det["bbox"], v_det["bbox"]) > 0.15:
                    has_overlap = True
                    break
            if not has_overlap:
                true_pedestrians.append(p_det)

        return raw_vehicles + true_pedestrians

    def draw_hud(self, frame, detections, frame_num, total_frames, fps):
        """
        Renders a premium Heads-Up Display (HUD) on the given frame.
        Draws semi-transparent dashboard, frame stats, dynamic vehicle count and elegant annotations.
        """
        h, w = frame.shape[:2]
        
        # 1. Draw a semi-transparent dark HUD bar at the top
        hud_height = 55
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_height), (15, 15, 20), -1)
        
        # Blend the overlay to achieve a premium glassmorphic/HUD look
        alpha = 0.75
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # 2. Draw HUD Metadata Text
        # Frame counter & percentage
        progress_text = f"Frame: {frame_num}"
        if total_frames > 0:
            pct = int((frame_num / total_frames) * 100)
            progress_text += f"/{total_frames} ({pct}%)"
            
        # Draw text with high-quality fonts and premium white & accent colors
        cv2.putText(frame, progress_text, (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
        
        # Draw dynamic vehicle statistics
        car_count = sum(1 for d in detections if d["class_name"] == "Car")
        truck_count = sum(1 for d in detections if d["class_name"] == "Truck")
        bus_count = sum(1 for d in detections if d["class_name"] == "Bus")
        # Combine Motorcycle and Bicycle classes as "Bikes" for urban tallying
        bike_count = sum(1 for d in detections if d["class_name"] in {"Motorcycle", "Bicycle"})
        ped_count = sum(1 for d in detections if d["class_name"] == "Pedestrian")
        total_vehicles = len(detections) - ped_count

        stats_text = f"Vehicles: {total_vehicles} (Cars: {car_count} | Trucks/Buses: {truck_count + bus_count} | Bikes: {bike_count}) | Peds: {ped_count}"
        cv2.putText(frame, stats_text, (w // 2 - 250, 32), cv2.FONT_HERSHEY_DUPLEX, 0.48, (100, 255, 255), 1, cv2.LINE_AA)

        # Draw processing performance (FPS)
        fps_text = f"Processing Speed: {fps:.1f} FPS"
        cv2.putText(frame, fps_text, (w - 180, 32), cv2.FONT_HERSHEY_DUPLEX, 0.48, (100, 255, 100), 1, cv2.LINE_AA)

        # 3. Draw Bounding Boxes and Label tags
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            class_name = det["class_name"]
            
            # Fetch vehicle-specific neon color
            color = self.PALETTE.get(class_name, self.PALETTE["Default"])

            # Draw smooth double-box or clean rounded border bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            
            # Add stylish corner ticks
            tick_len = min(15, int((x2 - x1) * 0.15), int((y2 - y1) * 0.15))
            # Top-Left corner
            cv2.line(frame, (x1, y1), (x1 + tick_len, y1), color, 4, cv2.LINE_AA)
            cv2.line(frame, (x1, y1), (x1, y1 + tick_len), color, 4, cv2.LINE_AA)
            # Top-Right corner
            cv2.line(frame, (x2, y1), (x2 - tick_len, y1), color, 4, cv2.LINE_AA)
            cv2.line(frame, (x2, y1), (x2, y1 + tick_len), color, 4, cv2.LINE_AA)
            # Bottom-Left corner
            cv2.line(frame, (x1, y2), (x1 + tick_len, y2), color, 4, cv2.LINE_AA)
            cv2.line(frame, (x1, y2), (x1, y2 - tick_len), color, 4, cv2.LINE_AA)
            # Bottom-Right corner
            cv2.line(frame, (x2, y2), (x2 - tick_len, y2), color, 4, cv2.LINE_AA)
            cv2.line(frame, (x2, y2), (x2, y2 - tick_len), color, 4, cv2.LINE_AA)

            # Draw premium filled tag for class label
            label = f"{class_name} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            
            # Clamp label positioning to stay inside the frame boundaries
            tag_y1 = max(0, y1 - th - 8)
            tag_y2 = y1 if y1 - th - 8 >= 0 else y1 + th + 12
            
            cv2.rectangle(frame, (x1, tag_y1), (x1 + tw + 10, tag_y2), color, -1)
            # High-contrast text label (dark grey/black on vivid background)
            cv2.putText(frame, label, (x1 + 5, tag_y1 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)

        return frame

    def process_video(self):
        """
        A generator that opens the video stream, processes frames in a loop,
        resizes them, runs detection, computes stats, and yields:
            (annotated_frame, list_of_detections)
            
        This is perfect for Member 2 and Member 3 to stream data in real-time.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[Detector] Error: Could not open video file at '{self.video_path}'")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_num = 0

        # Load YOLO model before entering loop to prevent first-frame lag
        if self.model is None:
            self.load_model()

        prev_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_num += 1

                # Extract and resize frame if requested in config
                if self.resize_dims is not None:
                    # resize_dims is [width, height]
                    frame = cv2.resize(frame, (self.resize_dims[0], self.resize_dims[1]))

                # Calculate current execution speed (FPS)
                current_time = time.time()
                time_diff = current_time - prev_time
                fps = 1.0 / time_diff if time_diff > 0 else 30.0
                prev_time = current_time

                # Core algorithms slice: Run YOLOv8 detection per frame
                detections = self.run_inference(frame)

                # Draw OpenCV display elements
                annotated_frame = frame.copy()
                annotated_frame = self.draw_hud(annotated_frame, detections, frame_num, total_frames, fps)

                yield annotated_frame, detections
        finally:
            cap.release()
            print("[Detector] Video stream released.")


def main():
    """
    Main driver function for testing the detector standalone.
    """
    parser = argparse.ArgumentParser(description="Traffic Monitoring - Member 1 YOLOv8 Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config yaml file")
    parser.add_argument("--video", help="Path to input video (overrides config)")
    parser.add_argument("--model", help="Path to YOLO model weights (overrides config)")
    parser.add_argument("--conf", type=float, help="Confidence threshold (overrides config)")
    args = parser.parse_args()

    # Initialize detector
    detector = TrafficDetector(config_path=args.config)

    # CLI Overrides
    if args.video:
        detector.video_path = args.video
    if args.model:
        detector.model_path = args.model
    if args.conf is not None:
        detector.conf_threshold = args.conf

    print(f"============================================================")
    print(f"  Traffic Coordination Prototype - Member 1: Standalone Running")
    print(f"  - Model: {detector.model_path}")
    print(f"  - Input Video: {detector.video_path}")
    print(f"  - Conf Threshold: {detector.conf_threshold}")
    print(f"  - Resize: {detector.resize_dims}")
    print(f"============================================================")

    # Start video processing loop
    frame_count = 0
    start_time = time.time()

    # Display loop window setup
    window_name = "Traffic Coordination HUD - Member 1: Detector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        for annotated_frame, detections in detector.process_video():
            frame_count += 1
            
            # Show output frame in window
            cv2.imshow(window_name, annotated_frame)
            
            # Print stats to terminal periodically
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames... Current vehicle count: {len(detections)}")

            # Key press actions: 'q' to quit, spacebar to pause
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[Detector] Quit requested by user.")
                break
            elif key == ord(' '):
                # Pause functionality
                while True:
                    key_paused = cv2.waitKey(30) & 0xFF
                    if key_paused == ord(' '):
                        break
                    elif key_paused == ord('q'):
                        key = ord('q')
                        break
                if key == ord('q'):
                    break
    except KeyboardInterrupt:
        print("[Detector] Interrupted by keyboard.")
    finally:
        cv2.destroyAllWindows()
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"--- Standalone Execution Summary ---")
        print(f"Total Frames Processed: {frame_count}")
        print(f"Total Execution Time:  {elapsed:.2f} seconds")
        print(f"Average Pipeline Speed: {avg_fps:.2f} FPS")
        print(f"------------------------------------")


if __name__ == "__main__":
    main()
