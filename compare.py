"""
compare.py — Performance Benchmarking, Logs, and OpenCV Overlay Module
Member 4: Adaptive Signal & Fixed vs. Adaptive Comparison
"""

import os
import csv
import cv2
import time
import numpy as np
from datetime import datetime
from config import FIXED_BASELINE, CYCLE_LOG_PATH

class EfficiencyTracker:
    """
    Evaluates algorithmic effectiveness by benchmarking adaptive window runtimes
    directly against an unchanging fixed baseline controller.
    """
    def __init__(self):
        self.total_adaptive_time = 0.0
        self.total_fixed_time = 0.0
        self.accumulated_saved_time = 0.0

    def register_completed_cycle(self, adaptive_duration: float, vehicle_count: int):
        """
        Logs timing discrepancies per cycle and saves performance metrics to a CSV file.
        """
        # Baseline cycle green time is FIXED_BASELINE (30s)
        time_saved = FIXED_BASELINE - adaptive_duration
        self.total_adaptive_time += adaptive_duration
        self.total_fixed_time += FIXED_BASELINE
        self.accumulated_saved_time += time_saved

        file_exists = os.path.isfile(CYCLE_LOG_PATH)
        with open(CYCLE_LOG_PATH, mode="a", newline="") as csvfile:
            fields = ["timestamp", "cycle_id", "vehicles_observed", "adaptive_duration", "fixed_duration", "time_saved"]
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "cycle_id": int(time.time()),
                "vehicles_observed": vehicle_count,
                "adaptive_duration": round(adaptive_duration, 2),
                "fixed_duration": FIXED_BASELINE,
                "time_saved": round(time_saved, 2)
            })


def draw_traffic_light_hud(frame: np.ndarray, state: str, time_left: float, tracker: EfficiencyTracker, last_pso_results: dict = None, l1_metrics: dict = None, l2_metrics: dict = None) -> np.ndarray:
    """
    Renders side-by-side coordinated traffic lights and a telemetry/swarm analytics dashboard.
    """
    h, w = frame.shape[:2]
    
    # ── 1. Render Traffic Light Housing Background (Top Right Corner) ────────
    # Expanded box to house two coordinated traffic lights side-by-side
    overlay = frame.copy()
    box_x1, box_y1 = w - 210, 65
    box_x2, box_y2 = w - 20, 270
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (20, 20, 25), -1)
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (60, 60, 65), 2, cv2.LINE_AA)
    
    # Coordinate calculations for dual traffic lights
    radius = 16
    cx_l1 = box_x1 + 45
    cx_l2 = box_x2 - 45
    r_cy, y_cy, g_cy = 110, 165, 220

    # Draw titles
    cv2.putText(overlay, "L1:MAIN", (cx_l1 - 32, box_y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(overlay, "L2:CROSS", (cx_l2 - 35, box_y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    # Inactive structural backings
    for cx in (cx_l1, cx_l2):
        cv2.circle(overlay, (cx, r_cy), radius, (15, 15, 30), -1)
        cv2.circle(overlay, (cx, y_cy), radius, (15, 30, 30), -1)
        cv2.circle(overlay, (cx, g_cy), radius, (15, 30, 15), -1)

    # Coordinated state mapping
    # LANE1_GREEN -> L1 Green, L2 Red
    # LANE1_YELLOW -> L1 Yellow, L2 Red
    # LANE2_GREEN -> L1 Red, L2 Green
    # LANE2_YELLOW -> L1 Red, L2 Yellow
    if state == "LANE1_GREEN":
        cv2.circle(overlay, (cx_l1, g_cy), radius, (0, 255, 0), -1)       # L1 Green
        cv2.circle(overlay, (cx_l2, r_cy), radius, (0, 0, 255), -1)       # L2 Red
    elif state == "LANE1_YELLOW":
        cv2.circle(overlay, (cx_l1, y_cy), radius, (0, 215, 255), -1)     # L1 Yellow
        cv2.circle(overlay, (cx_l2, r_cy), radius, (0, 0, 255), -1)       # L2 Red
    elif state == "LANE2_GREEN":
        cv2.circle(overlay, (cx_l1, r_cy), radius, (0, 0, 255), -1)       # L1 Red
        cv2.circle(overlay, (cx_l2, g_cy), radius, (0, 255, 0), -1)       # L2 Green
    elif state == "LANE2_YELLOW":
        cv2.circle(overlay, (cx_l1, r_cy), radius, (0, 0, 255), -1)       # L1 Red
        cv2.circle(overlay, (cx_l2, y_cy), radius, (0, 215, 255), -1)     # L2 Yellow

    # Smooth alpha blend for glassmorphic signal look
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Print remaining state timer directly under housing box
    timer_str = f"PHASE TIMER: {time_left:.1f}s"
    t_w, _ = cv2.getTextSize(timer_str, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)[0]
    cv2.putText(frame, timer_str, ((box_x1 + box_x2) // 2 - t_w // 2, box_y2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

    # ── 2. Render Comparative Analytics Side-By-Side Dashboard (Bottom Left) ──
    panel_w = 400
    panel_h = 175
    panel_y = h - panel_h - 20
    
    cv2.rectangle(frame, (20, panel_y), (20 + panel_w, h - 20), (12, 14, 20), -1)
    cv2.rectangle(frame, (20, panel_y), (20 + panel_w, h - 20), (70, 70, 75), 1, cv2.LINE_AA)

    # Dashboard header
    cv2.putText(frame, "COORDINATED TRAFFIC DASHBOARD", (32, panel_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 255), 1, cv2.LINE_AA)

    # Render Lane 1 & 2 Metrics side-by-side
    if l1_metrics and l2_metrics:
        # Lane 1 Column (x=30)
        cv2.putText(frame, "L1 (MAIN):", (32, panel_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 235, 100), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Count: {l1_metrics['count']} (Avg: {l1_metrics['avg_count']:.1f})", (32, panel_y + 57), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Queue: {l1_metrics['queue_len']} vehicles", (32, panel_y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Wait:  {l1_metrics['wait_time']:.1f}s", (32, panel_y + 87), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

        # Lane 2 Column (x=210)
        cv2.putText(frame, "L2 (CROSS):", (210, panel_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 100, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Count: {l2_metrics['count']} (Avg: {l2_metrics['avg_count']:.1f})", (210, panel_y + 57), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Queue: {l2_metrics['queue_len']} vehicles", (210, panel_y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Wait:  {l2_metrics['wait_time']:.1f}s", (210, panel_y + 87), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    # Render Swarm Optimization Solver Metrics
    if last_pso_results:
        cv2.putText(frame, "SWARM OPTIMIZER (PSO) STATUS:", (32, panel_y + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 255, 100), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Decision: Lane {last_pso_results['active_lane']} Green = {last_pso_results['best_duration']}s", (32, panel_y + 127), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)
        
        # Draw dynamic cost reduction progress
        history_str = " -> ".join([str(int(c)) for c in last_pso_results['cost_history'][:5]]) + "..."
        cv2.putText(frame, f"Swarm Cost: {last_pso_results['cost']:.1f} (history: {history_str})", (32, panel_y + 142), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 170), 1, cv2.LINE_AA)

    # Cumulative Time Saved vs Baseline (fixed 30s green intervals)
    cv2.putText(frame, f"PSO Swarm Time Saved: {tracker.accumulated_saved_time:.1f}s", (32, panel_y + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 100), 1, cv2.LINE_AA)

    return frame