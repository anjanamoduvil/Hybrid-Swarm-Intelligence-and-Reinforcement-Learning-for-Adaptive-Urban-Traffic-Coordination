#!/usr/bin/env python3
"""
app.py — Web Application Interface
Traffic Monitoring & Adaptive Signal System

Integrates all 4 member modules into a single FastAPI server with dual-lane coordinated signaling:
  Member 1 — YOLOv8 detection pipeline  (detector.py)
  Member 2 — SORT tracker                (tracker.py)
  Member 3 — Density & congestion alerts  (density.py / alerts.py)
  Member 4 — Coordinated PSO signaling   (traffic_signal.py / compare.py)
"""

import time
import cv2
import numpy as np
import yaml
from collections import defaultdict
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Member 1
from detector import TrafficDetector

# Member 2
from tracker import VehicleTracker

# Member 3
from density import compute_density, classify_density, should_trigger_alert, log_dual_counts_to_csv
from alerts import draw_lane_rois

# Member 4
from traffic_signal import CoordinatedSignalController
from compare import EfficiencyTracker, draw_traffic_light_hud

# Member 1 Traffic Prediction Extension
from prediction import predict
import config as _cfg

app = FastAPI(title="Traffic Monitoring & Adaptive Signal System")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ── Load Configuration ────────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    config_data = yaml.safe_load(f)

roi_lane1 = config_data.get("roi_lane1")
roi_lane2 = config_data.get("roi_lane2")
num_lanes = 2 if (roi_lane2 and len(roi_lane2) == 4) else 1
queue_speed_threshold = config_data.get("queue_speed_threshold", 1.5)

# ── Module initialisation ────────────────────────────────────────────────────
detector = TrafficDetector(config_path="config.yaml")
vehicle_tracker = VehicleTracker(queue_speed_threshold=queue_speed_threshold)
signal_ctrl = CoordinatedSignalController(config_path="config.yaml")
efficiency = EfficiencyTracker()

# ── Shared metrics state (updated by generate_frames, read by /api/metrics) ──
global_metrics = {
    "frame_num": 0,
    "fps": 0.0,
    "model_path": detector.model_path,
    "conf_threshold": detector.conf_threshold,
    "video_path": detector.video_path,
    "num_lanes": num_lanes,
    
    # Global Classification Counts
    "total_vehicles": 0,
    "cars": 0,
    "trucks_buses": 0,
    "bikes": 0,
    "pedestrians": 0,
    
    # Lane 1 (Main)
    "l1_count": 0,
    "l1_avg_count": 0.0,
    "l1_queue_len": 0,
    "l1_wait_time": 0.0,
    "l1_density_band": "LOW",
    "l1_density_score": 0.0,
    "l1_alert_active": False,
    
    # Lane 2 (Cross)
    "l2_count": 0,
    "l2_avg_count": 0.0,
    "l2_queue_len": 0,
    "l2_wait_time": 0.0,
    "l2_density_band": "LOW",
    "l2_density_score": 0.0,
    "l2_alert_active": False,

    # Coordinated Signaling & Swarm Info
    "signal_state": "LANE1_GREEN",
    "signal_time_left": 0.0,
    "saved_time_total": 0.0,
    
    # Swarm Decision Telemetry
    "pso_best_duration": 15.0,
    "pso_cost_history": [0.0],
    "pso_cost": 0.0,
    "pso_active_lane": 1,

    # Member 1 Forecast Telemetry
    "l1_pred_congestion": 0.0,
    "l1_pred_queue": 0,
    "l1_pred_trend": "STABLE",
    "l1_pred_confidence": 1.0,
    "l2_pred_congestion": 0.0,
    "l2_pred_queue": 0,
    "l2_pred_trend": "STABLE",
    "l2_pred_confidence": 1.0,
}


def is_in_roi(box, roi):
    """Checks if a bounding box center is inside the ROI rectangle."""
    if not roi:
        return False
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    rx1, ry1, rx2, ry2 = roi
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def generate_frames():
    """
    Core frame loop — chains detection, SORT tracking, queue/wait telemetry,
    and PSO signal coordination, yielding MJPEG frames.
    """
    global global_metrics

    if detector.model is None:
        detector.load_model()

    # Telemetry accumulators
    l1_count_sum = 0
    l2_count_sum = 0
    track_wait_times = defaultdict(float)
    last_frame_time = time.time()

    while True:
        for frame, detections in detector.process_video():
            current_time = time.time()
            dt = current_time - last_frame_time
            last_frame_time = current_time

            # 1. Run Tracker
            bboxes = [d["bbox"] for d in detections if d["class_name"] != "Pedestrian"]
            tracks = vehicle_tracker.update(bboxes)

            # Calculate Global Classification Counts
            cars = sum(1 for d in detections if d["class_name"] == "Car")
            trucks_buses = sum(1 for d in detections if d["class_name"] in {"Truck", "Bus"})
            bikes = sum(1 for d in detections if d["class_name"] in {"Motorcycle", "Bicycle"})
            pedestrians = sum(1 for d in detections if d["class_name"] == "Pedestrian")
            total_vehicles = cars + trucks_buses + bikes

            # 2. Partition Tracks by Lane ROIs
            l1_tracks = []
            l2_tracks = []
            for t in tracks:
                if is_in_roi(t["box"], roi_lane1):
                    l1_tracks.append(t)
                elif is_in_roi(t["box"], roi_lane2):
                    l2_tracks.append(t)

            # 3. Calculate metrics per lane
            l1_count = len(l1_tracks)
            l2_count = len(l2_tracks)

            l1_count_sum += l1_count
            l2_count_sum += l2_count
            global_metrics["frame_num"] += 1
            frame_num = global_metrics["frame_num"]

            l1_avg = l1_count_sum / frame_num
            l2_avg = l2_count_sum / frame_num

            # Queues
            l1_queue = sum(1 for t in l1_tracks if t["is_queued"])
            l2_queue = sum(1 for t in l2_tracks if t["is_queued"])

            # Waiting times
            for t in l1_tracks:
                if t["is_queued"]:
                    track_wait_times[t["id"]] += dt
            for t in l2_tracks:
                if t["is_queued"]:
                    track_wait_times[t["id"]] += dt

            # Clean up old tracks from waiting dict to prevent memory leak
            active_ids = {t["id"] for t in tracks}
            for tid in list(track_wait_times.keys()):
                if tid not in active_ids:
                    del track_wait_times[tid]

            l1_waits = [track_wait_times[t["id"]] for t in l1_tracks if t["is_queued"]]
            l2_waits = [track_wait_times[t["id"]] for t in l2_tracks if t["is_queued"]]
            l1_wait_time = np.mean(l1_waits) if l1_waits else 0.0
            l2_wait_time = np.mean(l2_waits) if l2_waits else 0.0

            # Density
            l1_density = compute_density(l1_count)
            l2_density = compute_density(l2_count)
            l1_band = classify_density(l1_count)
            l2_band = classify_density(l2_count)
            l1_alert = should_trigger_alert(l1_count)
            l2_alert = should_trigger_alert(l2_count)

            # Log to CSV
            log_dual_counts_to_csv(frame_num, l1_count, l1_band, l2_count, l2_band)

            # Member 1 Traffic Forecast Update
            if frame_num % 30 == 0 or frame_num == 1:
                try:
                    l1_pred_c, l1_pred_q, l1_pred_t, l1_pred_conf = predict(_cfg.CSV_LOG_PATH, n_steps=3, lane=1)
                    if num_lanes == 2:
                        l2_pred_c, l2_pred_q, l2_pred_t, l2_pred_conf = predict(_cfg.CSV_LOG_PATH, n_steps=3, lane=2)
                    else:
                        l2_pred_c, l2_pred_q, l2_pred_t, l2_pred_conf = 0.0, 0, "STABLE", 1.0

                    global_metrics.update({
                        "l1_pred_congestion": l1_pred_c,
                        "l1_pred_queue": l1_pred_q,
                        "l1_pred_trend": l1_pred_t,
                        "l1_pred_confidence": l1_pred_conf,
                        "l2_pred_congestion": l2_pred_c,
                        "l2_pred_queue": l2_pred_q,
                        "l2_pred_trend": l2_pred_t,
                        "l2_pred_confidence": l2_pred_conf,
                    })
                except Exception as e:
                    print(f"[Prediction Engine] Error: {e}")

            # 4. Update Signaling State Machine
            state, time_left, cycle_done = signal_ctrl.update_state_machine(
                l1_q=l1_queue,
                l2_q=l2_queue,
                l1_wait=l1_wait_time,
                l2_wait=l2_wait_time
            )

            if cycle_done:
                efficiency.register_completed_cycle(
                    adaptive_duration=signal_ctrl.last_pso_results["best_duration"],
                    vehicle_count=l1_queue + l2_queue
                )

            # 5. Overlays
            # Draw lane boundaries (ROIs)
            frame = draw_lane_rois(frame, roi_lane1, roi_lane2, l1_count, l2_count)

            # Draw traffic light housing and dashboards
            l1_metrics = {"count": l1_count, "avg_count": l1_avg, "queue_len": l1_queue, "wait_time": l1_wait_time}
            l2_metrics = {"count": l2_count, "avg_count": l2_avg, "queue_len": l2_queue, "wait_time": l2_wait_time}
            frame = draw_traffic_light_hud(
                frame=frame,
                state=state,
                time_left=time_left,
                tracker=efficiency,
                last_pso_results=signal_ctrl.last_pso_results,
                l1_metrics=l1_metrics,
                l2_metrics=l2_metrics
            )

            # 6. Sync Global Metrics for REST API
            global_metrics.update({
                # Lane 1
                "l1_count": l1_count,
                "l1_avg_count": round(l1_avg, 2),
                "l1_queue_len": l1_queue,
                "l1_wait_time": round(l1_wait_time, 2),
                "l1_density_band": l1_band,
                "l1_density_score": round(l1_density, 3),
                "l1_alert_active": l1_alert,
                # Lane 2
                "l2_count": l2_count,
                "l2_avg_count": round(l2_avg, 2),
                "l2_queue_len": l2_queue,
                "l2_wait_time": round(l2_wait_time, 2),
                "l2_density_band": l2_band,
                "l2_density_score": round(l2_density, 3),
                "l2_alert_active": l2_alert,
                # Global Classification Counts
                "total_vehicles": total_vehicles,
                "cars": cars,
                "trucks_buses": trucks_buses,
                "bikes": bikes,
                "pedestrians": pedestrians,
                # Signal
                "signal_state": state,
                "signal_time_left": round(time_left, 1),
                "saved_time_total": round(efficiency.accumulated_saved_time, 1),
                # PSO decision
                "pso_best_duration": signal_ctrl.last_pso_results["best_duration"],
                "pso_cost_history": signal_ctrl.last_pso_results["cost_history"],
                "pso_cost": signal_ctrl.last_pso_results["cost"],
                "pso_active_lane": signal_ctrl.last_pso_results["active_lane"],
            })

            # ── Encode & yield MJPEG frame ────────────────────────────────────
            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

            time.sleep(0.01)

        # Reset loop parameters for continuous video replay
        global_metrics["frame_num"] = 0
        l1_count_sum = 0
        l2_count_sum = 0
        track_wait_times.clear()
        print("[Web Server] Video completed, restarting stream loop...")


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, private",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/metrics")
def get_metrics():
    global_metrics["model_path"] = detector.model_path
    global_metrics["conf_threshold"] = detector.conf_threshold
    global_metrics["video_path"] = detector.video_path
    return JSONResponse(content=global_metrics)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
