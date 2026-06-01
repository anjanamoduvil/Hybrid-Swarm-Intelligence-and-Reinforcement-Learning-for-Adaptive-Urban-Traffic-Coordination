# Member 1 Slice: Traffic Detector Pipeline

This directory contains the first complete vertical slice of the **Traffic Monitoring & Adaptive Signal System**: a high-fidelity video processing and YOLOv8 vehicle detection pipeline. 

It is designed to seamlessly ingest traffic intersection video feeds, run real-time inference to detect and classify vehicles, and render a premium anti-aliased Heads-Up Display (HUD) overlay. This slice acts as the foundational data source for downstream components (Member 2's tracking algorithms and Member 3/4's adaptive signal coordinators).

---

## 📂 Slice Components

- **`detector.py`**: The core application module containing the `TrafficDetector` class, visual HUD renderer, frame loading pipeline, and standalone executable CLI.
- **`config.yaml`**: Declarative system parameters file containing paths, thresholds, and frame resizing coordinates.
- **`test_detector.py`**: Automated test suite containing frame count validations, shape assertions, and an in-memory synthetic video generator for offline testing.

---

## 🛠️ Requirements & Installation

This slice requires Python 3.8+ and standard scientific/computer vision libraries:

```bash
pip install ultralytics opencv-python pyyaml numpy
```

---

## ⚙️ Configuration Parameters (`config.yaml`)

Parameters are managed declaratively in `config.yaml`. Other members can tweak these settings to automatically align the detection window, scale inputs, or filter detections:

```yaml
# Path to YOLOv8 model weights (e.g. local or auto-downloaded 'yolov8n.pt')
model_path: "yolov8n.pt"

# Path to the source traffic intersection video
video_path: "traffic_sample.mp4"

# Minimum confidence threshold for vehicle filtering
conf_threshold: 0.25

# Resizing dimensions [width, height] for uniform frame sizes (improves speed & consistency)
resize_dims: [800, 600]
```

---

## 🚀 Standalone Execution

You can run `detector.py` directly to see the YOLOv8 model perform inferences in real time with the custom premium HUD display.

```bash
python detector.py
```

### Command-Line Overrides
You can override `config.yaml` parameters directly from the terminal:
```bash
python detector.py --video path/to/another_traffic.mp4 --conf 0.35 --model yolov8s.pt
```

### Playback Controls
When the visual display window is active:
- **`Spacebar`**: Pause/Resume the video stream.
- **`q`**: Gracefully quit and output processing stats to the terminal.

---

## 🔗 Integration API (For Team Members 2, 3, & 4)

This slice provides a generator function (`process_video`) designed to stream detection lists and annotated frames to your tracking or coordinator loops frame-by-frame.

### Integration Code Example
```python
from detector import TrafficDetector

# 1. Initialize detector (loads config.yaml parameters)
detector = TrafficDetector(config_path="config.yaml")

# 2. Iterate through the generator
# Each step returns the beautifully annotated frame and list of parsed vehicle detections
for frame, detections in detector.process_video():
    # Process detections (Member 2 Tracker Integration)
    for det in detections:
        bbox = det["bbox"]          # [x1, y1, x2, y2] - Perfect for tracker input!
        confidence = det["confidence"] # float (0.0 to 1.0)
        class_name = det["class_name"] # "Car", "Motorcycle", "Bus", "Truck"
        
        # Feed detections to SORT/DeepSORT tracker...
        # tracker.update(bbox, confidence)
    
    # Calculate density & decide signals (Member 3 & 4 Adaptive Coordination)
    vehicle_count = len(detections)
    
    # Draw your tracker ID/coordinates or coordination overlays directly on the frame!
```

---

## 🎨 Heads-Up Display (HUD) Specifications

The OpenCV rendering is optimized for a clean, modern aesthetic:
- **Header HUD Bar**: A semi-transparent dark banner (`alpha = 0.75`) providing clear, flick-free readouts of:
  - **Frames**: Current position vs total frames.
  - **Live Vehicle Density**: Real-time breakdown of cars, trucks/buses, and motorcycles.
  - **Processing Speed**: Execution performance in FPS.
- **Vibrant Detections**: Anti-aliased double-border bounding boxes with custom class palettes:
  - 🚙 **Car**: Neon Cyan `BGR(255, 235, 100)`
  - 🏍️ **Motorcycle**: Neon Green `BGR(100, 255, 100)`
  - 🚌 **Bus**: Indigo/Magenta `BGR(210, 100, 255)`
  - 🚛 **Truck**: Orange/Amber `BGR(50, 130, 255)`
- **Class Label Tag**: Premium solid tags positioned above/below the bounding box displaying class name and confidence score with high-contrast text.

---

## 🧪 Running Unit Tests

The test suite programmatically creates a synthetic 15-frame `.mp4` video with moving geometric objects and utilizes mocking to ensure test results are fast, offline-friendly, and repeatable.

Run the test suite:
```bash
python -m unittest test_detector.py -v
```

Tests verified:
- **`test_configuration_loading`**: Asserts YAML configuration properties are parsed and type-cast correctly.
- **`test_video_pipeline_frame_count`**: Asserts that every single frame in the source video is extracted and processed.
- **`test_detection_shape_and_parsing`**: Asserts that YOLOv8 outputs are properly transformed into the standardized `(bbox, confidence, class_id, class_name)` schema.
- **`test_invalid_class_filtering`**: Asserts non-vehicle targets (e.g., pedestrians) are ignored.

---

---

# Member 3 Slice: Density Estimation & Congestion Alerts

This slice consumes the `vehicle_count` produced by Member 1's detector and adds the **density intelligence layer**: classifying traffic into congestion bands, triggering alerts, logging history to CSV, and rendering annotated overlays on the live video frames.

---

## 📂 Slice Components

- **`density.py`**: Core algorithm — density score, band classification, alert logic, CSV logging.
- **`alerts.py`**: OpenCV display — density bar, band label, red alert banner.
- **`config.py`**: Configuration parameters (thresholds, CSV path).
- **`test_density.py`**: Unit tests (pytest) — 20 tests.

---

## 🛠️ Requirements & Installation

```bash
pip install pytest opencv-python numpy
```
Standard library dependencies: `csv`, `os`, `datetime`.

---

## ⚙️ Configuration Parameters (`config.py`)

```python
THRESHOLDS = {
    "low":  5,
    "med":  10,
    "high": 20,
}
CSV_LOG_PATH = "density_log.csv"
```

---

## 🧮 Algorithms

### Density Score
```
density = min(vehicle_count / HIGH_THRESHOLD, 1.0)
```
Produces a value in **[0.0, 1.0]**, capped at 1.0 for extreme counts.

### Band Classification
| Condition | Band |
|-----------|------|
| count ≤ 5 | LOW |
| count ≤ 10 | MED |
| count > 10 | HIGH |

### Alert Trigger
An alert fires when `vehicle_count > THRESHOLDS["high"]` (default 20).

### CSV Log
Each frame with a logged event appends one row:
```
timestamp, frame, vehicle_count, band
```

---

## 🎨 OpenCV Display Modules (`alerts.py`)

| Function | What it draws |
|----------|--------------|
| `draw_band_label(frame, count)` | Colour-coded `DENSITY: LOW/MED/HIGH` text (top-left) |
| `draw_density_bar(frame, count, density)` | Vertical fill bar on the right edge |
| `draw_alert_banner(frame, count)` | Red semi-transparent banner (top) when alert is active |
| `annotate_frame(frame, count, density)` | Convenience wrapper — calls all three above |

---

## 🔗 Integration with `main.py`

```python
from density import compute_density, classify_density, log_count_to_csv
from alerts import annotate_frame

# Inside the frame loop (after Member 1's detector gives vehicle_count):
density = compute_density(vehicle_count)
frame   = annotate_frame(frame, vehicle_count, density)
log_count_to_csv(frame_number, vehicle_count, classify_density(vehicle_count))
```

---

## 🧪 Running Unit Tests

```bash
python -m pytest test_density.py -v
```

All 20 tests should pass with no display window required.
