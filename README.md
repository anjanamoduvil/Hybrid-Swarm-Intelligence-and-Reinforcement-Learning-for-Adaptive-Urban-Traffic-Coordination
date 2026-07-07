# Hybrid Swarm Intelligence & Reinforcement Learning Traffic Coordination System

An advanced, premium traffic monitoring and adaptive signal coordination prototype. The system ingests a traffic video stream, runs real-time object detection and multi-object tracking (SORT), partitions vehicles into separate lanes, and coordinates traffic light phase timings dynamically using a **Particle Swarm Optimization (PSO)** swarm-intelligence solver.

---

## 🚀 System Architecture & Integration Overview

The system integrates four distinct modules into a unified frame processing pipeline served via a **FastAPI** web application with a real-time glassmorphic dashboard interface:

```mermaid
graph TD
    A[Video Stream Ingestion] --> B[Member 1: YOLOv8 Detector Pipeline]
    B --> C[Member 2: SORT Tracker Kalman Filter]
    C --> D[ROI Partitioning: Lane 1 & Lane 2]
    D --> E[Member 3: Density & Congestion Analyzer]
    E --> E2[Member 3: Multi-Intersection Grid + Propagation]
    E2 --> F[Member 4: Coordinated PSO Signal Controller]
    F --> G[Unified Web Interface & HUD Rendering]
```

---

## 👥 Team Member Roles & Contributions

### 🔹 Member 1 (You): Traffic Ingestion, Object Detection & Traffic Prediction Module
* **Files**: [detector.py](file:///c:/Users/VICTUS/TEST/detector.py), [config.yaml](file:///c:/Users/VICTUS/TEST/config.yaml), [prediction.py](file:///c:/Users/VICTUS/TEST/prediction.py), [test_prediction.py](file:///c:/Users/VICTUS/TEST/test_prediction.py)
* **Responsibilities**:
  * Configured declarative configuration management loading parameters for video sources, model weights, confidence thresholds, and canvas resolutions.
  * Programmed high-speed, real-time vehicle detection and class filtering using **YOLOv8** weights.
  * Classified entities into specific traffic categories (`Car`, `Truck`, `Bus`, `Motorcycle`, `Bicycle`, `Pedestrian`).
  * Implemented an overlapping/rider filter to prevent duplicate boxes when a pedestrian is riding a motorcycle.
  * Engineered a custom OpenCV Heads-Up Display (HUD) overlay at the top of the video feed, rendering neon-colored bounding boxes, class labels, confidences, current frames, and processing speed (FPS).
  * **Week 3 Extension — Traffic Prediction Module**:
    * Programmed a **Moving-Average Smoother** to filter out vehicle count noise for reliable congestion metrics.
    * Engineered a **Weighted Linear Regression** forecaster (using `scikit-learn`'s `LinearRegression` with linearly decaying weights that prioritize recent frames) to predict future congestion levels and queue lengths $n$ steps ahead.
    * Coded a dynamic trend classifier flagging traffic density changes as `RISING`, `FALLING`, or `STABLE` based on regression slopes.
    * Programmed a prediction confidence metric using the weighted $R^2$ coefficient of determination.
    * Developed the glassmorphic **Traffic Forecasting & Predictions** dashboard card in `index.html` with real-time UI updates, color-coded trend badges, and confidence progress bars.
    * Built an automatic **Dynamic Layout Adaptation** system that checks the active ROIs in `config.yaml` to hide/show Lane 2 cards, metrics, alerts, and lights automatically.
  * **Week 4 Extension — Dynamic Traffic Graph & Graph Intelligence Module**:
    * Programmed a **NetworkX**-based `DynamicTrafficGraph` that continuously ingests live traffic density, queue lengths, speeds, and signal phases per intersection to dynamically build a connected graph representation of the traffic network.
    * Engineered a PyTorch Geometric **Graph Convolutional Network (GCN)** (`TrafficGCN`) that consumes the real-time graph state to model network-wide congestion propagation and anticipate gridlock bottlenecks.
    * Developed an online-learning `GraphIntelligenceModule` that performs forward passes predicting congestion 1–3 steps ahead for all nodes simultaneously, outputting network predictions and critical node importance scores.
    * Embedded a real-time Network Graph visualizer inside the frontend Glassmorphic dashboard alongside a new **Graph Analytics & Criticality** panel that explicitly outputs the GCN predictions and node importance metrics.
* **Key Packages Used**:
  * `ultralytics`: Used to instantiate and run YOLOv8 object detection models.
  * `opencv-python` (cv2): Used to capture video, resize frames, crop regions, and draw anti-aliased HUD elements.
  * `pyyaml`: Used to safely load configuration variables from YAML files.

---

### 🔹 Member 2: Multi-Object SORT Tracker
* **Files**: [tracker.py](file:///c:/Users/VICTUS/TEST/tracker.py)
* **Responsibilities**:
  * Replaced naive tracking with a robust **SORT (Simple Online and Realtime Tracking)** implementation from scratch.
  * Designed a Constant Velocity state-space **Kalman Filter** to estimate bounding box center coordinates $(u, v)$, box area $(s)$, aspect ratio $(r)$, and their respective velocities ($\dot{u}, \dot{v}, \dot{s}$).
  * Implemented data association of new detections to active tracks using an Intersection-over-Union (IoU) cost matrix matched via the Hungarian Algorithm.
  * Tracked frame-to-frame vehicle velocities to classify objects as queued (stationary) when velocity falls below `queue_speed_threshold` ($1.5$ px/frame).
* **Key Packages Used**:
  * `numpy`: Used for matrix math operations (dot products, inverses, transpose) for the Kalman Filter updates.
  * `scipy` (`scipy.optimize.linear_sum_assignment`): Used to execute the Hungarian algorithm for optimal bipartite matching of tracks to detections.

---

### 🔹 Member 3: Density Estimation, Congestion Alerts & Multi-Intersection Coordination
* **Files**: [density.py](file:///c:/Users/VICTUS/TEST/density.py), [alerts.py](file:///c:/Users/VICTUS/TEST/alerts.py), [intersection_sim.py](file:///c:/Users/VICTUS/TEST/intersection_sim.py), [test_intersection_sim.py](file:///c:/Users/VICTUS/TEST/test_intersection_sim.py)
* **Responsibilities**:
  * Created a normalized traffic density estimator (`[0.0 - 1.0]`) mapping vehicle counts relative to lane saturation levels.
  * Classified lane congestion levels into `LOW`, `MED`, and `HIGH` bands dynamically.
  * Implemented event-driven safety alarms showing a flashing congestion alert warning banner at the top of the HUD and UI when a lane exceeds thresholds.
  * Developed a dual-lane logging module to write timestamps, vehicle counts, and congestion levels into a persistent CSV database.
  * **Week 3 Extension — Multi-Intersection Coordination**:
    * Designed an `IntersectionGrid` class simulating **3–4 connected intersections**, each modelled as an independent `Intersection` node with its own vehicle count, density band, and history.
    * Implemented **congestion propagation**: a `HIGH`-band intersection spills a configurable fraction (`PROPAGATION_RATE`) of its vehicle load onto its neighbouring nodes each tick, modelling how gridlock at one junction spreads to adjacent junctions.
    * Built a **coordinated green-wave allocator** that ranks intersections by descending congestion load each tick and assigns green-light durations (bounded by `MIN_GREEN`/`MAX_GREEN`) with a priority head-start bonus for the most congested nodes first.
    * Integrated Member 1's `predict()` forecasting function per intersection — each node writes its own `lane1_count`-formatted CSV log so the existing prediction module can be called without any modification, with a local trend-based fallback when insufficient history exists.
    * Designed the integration point for Member 2's RL agent (`RLAgent.step(state)`), allowing the green-wave allocator to apply a learned adjustment on top of the swarm-priority base allocation.
    * Wrote a 29-test `pytest` suite (`test_intersection_sim.py`) covering grid initialization, propagation accuracy, green-wave ordering, forecast integration (including a real end-to-end call into `prediction.py`), and CSV logging — all passing against the live Week 2/3 codebase.
* **Key Packages Used**:
  * `csv` & `os`: Used to check, create, and append telemetry logging records (per-intersection and grid-summary logs).
  * `datetime`: Used to timestamp logged congestion occurrences.
  * `pytest`: Used to validate grid behaviour and the real integration with `prediction.predict()`.

---

### 🔹 Member 4: Coordinated Signals & Benchmarking Telemetry
* **Files**: [traffic_signal.py](file:///c:/Users/VICTUS/TEST/traffic_signal.py), [compare.py](file:///c:/Users/VICTUS/TEST/compare.py)
* **Responsibilities**:
  * Engineered a coordinated dual-lane traffic light state machine transitioning sequentially through:
    $$\text{LANE1\_GREEN} \rightarrow \text{LANE1\_YELLOW} \rightarrow \text{LANE2\_GREEN} \rightarrow \text{LANE2\_YELLOW}$$
  * Designed a **Particle Swarm Optimization (PSO)** coordinator. The swarm models green light discharge rates against red light arrival accumulations to solve for the optimal phase duration $g \in [10\text{s}, 60\text{s}]$ that minimizes total wait times and queue lengths.
  * Logged performance benchmarks comparing the PSO swarm decisions against a standard fixed-time 30-second green baseline.
  * Designed overlay HUD panels rendering coordinated traffic light signals, lane statistics, and a swarm solver search convergence chart on the video feed.
* **Key Packages Used**:
  * `numpy`: Used to initialize particle coordinates, calculate velocities, boundaries, and swarm fitness.

---

## 🛠️ Combined Web Interface (`app.py`)
The unified FastAPI web application orchestrates the entire pipeline:
* **FastAPI & Uvicorn**: Serves the REST API for real-time telemetry metrics (`/api/metrics`) and streams the processed frames as an MJPEG multipart response (`/video_feed`).
* **Jinja2**: Renders the frontend interface ([index.html](file:///c:/Users/VICTUS/TEST/templates/index.html)).
* **Interactive Frontend**: Houses a side-by-side dual-lane statistic monitor, a graphic traffic signal map syncing with the FSM state, and an SVG line graph charting the real-time cost convergence of the PSO swarm.

---

## ⚙️ How to Configure & Run

### 1. Requirements
Install the dependencies:
```bash
pip install ultralytics opencv-python pyyaml numpy scipy fastapi uvicorn jinja2 pytest
```

### 2. Standalone Testing
Verify individual modules using pytest:
```bash
python -m unittest test_detector.py -v
python -m pytest test_density.py test_traffic_signal.py test_prediction.py test_evaluator.py test_intersection_sim.py -v
```

### 3. Launching the Web Server
Run the FastAPI application locally:
```bash
python app.py
```
Open your browser and navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**
