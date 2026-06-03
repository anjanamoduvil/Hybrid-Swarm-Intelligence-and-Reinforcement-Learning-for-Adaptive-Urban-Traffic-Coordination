#!/usr/bin/env python3
"""
app.py — Web Application Interface
Traffic Monitoring & Adaptive Signal System

Integrates all 4 member modules into a single FastAPI server:
  Member 1 — YOLOv8 detection pipeline  (detector.py)
  Member 2 — Vehicle tracker             (tracker.py)
  Member 3 — Density estimation & alerts (density.py / alerts.py)
  Member 4 — Adaptive signal controller  (signal.py / compare.py)
"""

import time
import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Member 1
from detector import TrafficDetector

# Member 2
from tracker import VehicleTracker

# Member 3
from density import compute_density, classify_density, should_trigger_alert, log_count_to_csv
from alerts import annotate_frame as m3_annotate_frame

# Member 4
from signal import AdaptiveSignalController
from compare import EfficiencyTracker, draw_traffic_light_hud

app = FastAPI(title="Traffic Monitoring & Adaptive Signal System")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ── Module initialisation ────────────────────────────────────────────────────
detector       = TrafficDetector(config_path="config.yaml")
vehicle_tracker = VehicleTracker()
signal_ctrl    = AdaptiveSignalController()
efficiency     = EfficiencyTracker()

# ── Shared metrics state (updated by generate_frames, read by /api/metrics) ──
global_metrics = {
    # Member 1
    "frame_num":      0,
    "total_frames":   0,
    "fps":            0.0,
    "total_vehicles": 0,
    "cars":           0,
    "trucks_buses":   0,
    "bikes":          0,
    "pedestrians":    0,
    "model_path":     detector.model_path,
    "conf_threshold": detector.conf_threshold,
    "video_path":     detector.video_path,
    # Member 3
    "density_band":   "LOW",
    "density_score":  0.0,
    "alert_active":   False,
    # Member 4
    "signal_state":       "GREEN",
    "signal_time_left":   0.0,
    "adaptive_duration":  10,
    "saved_time_total":   0.0,
}


def generate_frames():
    """
    Core frame loop — chains all 4 member modules together and yields
    MJPEG-encoded frames for the browser video stream.
    """
    global global_metrics

    if detector.model is None:
        detector.load_model()

    while True:
        for frame, detections in detector.process_video():

            # ── Member 1: counts from detections ────────────────────────────
            cars        = sum(1 for d in detections if d["class_name"] == "Car")
            trucks_buses = sum(1 for d in detections if d["class_name"] in {"Truck", "Bus"})
            bikes       = sum(1 for d in detections if d["class_name"] in {"Motorcycle", "Bicycle"})
            pedestrians = sum(1 for d in detections if d["class_name"] == "Pedestrian")
            vehicle_count = cars + trucks_buses + bikes

            global_metrics["frame_num"]     += 1
            global_metrics["cars"]           = cars
            global_metrics["trucks_buses"]   = trucks_buses
            global_metrics["bikes"]          = bikes
            global_metrics["pedestrians"]    = pedestrians
            global_metrics["total_vehicles"] = vehicle_count

            # ── Member 2: tracker update ─────────────────────────────────────
            bboxes = [d["bbox"] for d in detections]
            vehicle_tracker.update(bboxes)

            # ── Member 3: density + alert overlays ───────────────────────────
            density = compute_density(vehicle_count)
            band    = classify_density(vehicle_count)
            alert   = should_trigger_alert(vehicle_count)
            log_count_to_csv(global_metrics["frame_num"], vehicle_count, band)

            frame = m3_annotate_frame(frame, vehicle_count, density)

            global_metrics["density_band"]  = band
            global_metrics["density_score"] = round(density, 3)
            global_metrics["alert_active"]  = alert

            # ── Member 4: signal state machine + comparison overlay ──────────
            state, time_left, cycle_done = signal_ctrl.update_state_machine(vehicle_count)

            if cycle_done:
                adaptive_dur = signal_ctrl.calculate_adaptive_duration(vehicle_count)
                efficiency.register_completed_cycle(adaptive_dur, vehicle_count)

            frame = draw_traffic_light_hud(frame, state, time_left, efficiency)

            global_metrics["signal_state"]      = state
            global_metrics["signal_time_left"]  = round(time_left, 1)
            global_metrics["adaptive_duration"] = signal_ctrl.calculate_adaptive_duration(vehicle_count)
            global_metrics["saved_time_total"]  = round(efficiency.accumulated_saved_time, 1)

            # ── Encode & yield MJPEG frame ────────────────────────────────────
            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

            time.sleep(0.01)

        global_metrics["frame_num"] = 0
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
            "Pragma":        "no-cache",
            "Expires":       "0",
        },
    )


@app.get("/api/metrics")
def get_metrics():
    global_metrics["model_path"]     = detector.model_path
    global_metrics["conf_threshold"] = detector.conf_threshold
    global_metrics["video_path"]     = detector.video_path
    return JSONResponse(content=global_metrics)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
