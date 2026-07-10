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


def draw_traffic_light_hud(frame: np.ndarray, state: str, time_left: float, tracker: EfficiencyTracker, last_pso_results: dict = None, l1_metrics: dict = None, l2_metrics: dict = None, latest_explanation: dict = None) -> np.ndarray:
    """
    Renders side-by-side coordinated traffic lights and an updated telemetry/swarm analytics dashboard.
    Now includes a dedicated sub-panel for the Member 4 Live Explainable AI Decision Engine.
    """
    h, w = frame.shape[:2]
    
    # ── 1. Render Traffic Light Housing Background (Top Right Corner) ────────
    overlay = frame.copy()
    box_x1, box_y1 = w - 210, 65
    box_x2, box_y2 = w - 20, 270
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (20, 20, 25), -1)
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (60, 60, 65), 2, cv2.LINE_AA)
    
    radius = 16
    cx_l1 = box_x1 + 45
    cx_l2 = box_x2 - 45
    r_cy, y_cy, g_cy = 110, 165, 220

    cv2.putText(overlay, "L1:MAIN", (cx_l1 - 32, box_y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(overlay, "L2:CROSS", (cx_l2 - 35, box_y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    for cx in (cx_l1, cx_l2):
        cv2.circle(overlay, (cx, r_cy), radius, (15, 15, 30), -1)
        cv2.circle(overlay, (cx, y_cy), radius, (15, 30, 30), -1)
        cv2.circle(overlay, (cx, g_cy), radius, (15, 30, 15), -1)

    if state == "LANE1_GREEN":
        cv2.circle(overlay, (cx_l1, g_cy), radius, (0, 255, 0), -1)       
        cv2.circle(overlay, (cx_l2, r_cy), radius, (0, 0, 255), -1)       
    elif state == "LANE1_YELLOW":
        cv2.circle(overlay, (cx_l1, y_cy), radius, (0, 215, 255), -1)     
        cv2.circle(overlay, (cx_l2, r_cy), radius, (0, 0, 255), -1)       
    elif state == "LANE2_GREEN":
        cv2.circle(overlay, (cx_l1, r_cy), radius, (0, 0, 255), -1)       
        cv2.circle(overlay, (cx_l2, g_cy), radius, (0, 255, 0), -1)       
    elif state == "LANE2_YELLOW":
        cv2.circle(overlay, (cx_l1, r_cy), radius, (0, 0, 255), -1)       
        cv2.circle(overlay, (cx_l2, y_cy), radius, (0, 215, 255), -1)     

    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    timer_str = f"PHASE TIMER: {time_left:.1f}s"
    t_w, _ = cv2.getTextSize(timer_str, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)[0]
    cv2.putText(frame, timer_str, ((box_x1 + box_x2) // 2 - t_w // 2, box_y2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

    # ── 2. Render Comparative Analytics Side-By-Side Dashboard (Bottom Left) ──
    # Height adjusted (panel_h from 175 to 220) to tightly secure room for live explanation text
    panel_w = 420
    panel_h = 220
    panel_y = h - panel_h - 20
    
    cv2.rectangle(frame, (20, panel_y), (20 + panel_w, h - 20), (12, 14, 20), -1)
    cv2.rectangle(frame, (20, panel_y), (20 + panel_w, h - 20), (70, 70, 75), 1, cv2.LINE_AA)

    cv2.putText(frame, "COORDINATED TRAFFIC DASHBOARD", (32, panel_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 255), 1, cv2.LINE_AA)

    if l1_metrics and l2_metrics:
        cv2.putText(frame, "L1 (MAIN):", (32, panel_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 235, 100), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Count: {l1_metrics['count']}", (32, panel_y + 57), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Queue: {l1_metrics['queue_len']} veh", (32, panel_y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Wait:  {l1_metrics['wait_time']:.1f}s", (32, panel_y + 87), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.putText(frame, "L2 (CROSS):", (220, panel_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 100, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Count: {l2_metrics['count']}", (220, panel_y + 57), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Queue: {l2_metrics['queue_len']} veh", (220, panel_y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Wait:  {l2_metrics['wait_time']:.1f}s", (220, panel_y + 87), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    if last_pso_results:
        cv2.putText(frame, "SWARM OPTIMIZER (PSO) STATUS:", (32, panel_y + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 255, 100), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Decision: Lane {last_pso_results['active_lane']} Green = {last_pso_results['best_duration']}s", (32, panel_y + 127), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)
        
        history_str = " -> ".join([str(int(c)) for c in last_pso_results['cost_history'][:4]]) + "..."
        cv2.putText(frame, f"Swarm Cost: {last_pso_results['cost']:.1f} (history: {history_str})", (32, panel_y + 142), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 170), 1, cv2.LINE_AA)

    cv2.putText(frame, f"PSO Swarm Time Saved: {tracker.accumulated_saved_time:.1f}s", (32, panel_y + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 100), 1, cv2.LINE_AA)

    # ── 3. Render Task 6 Live XAI Explanation String ─────────────────────────
    cv2.line(frame, (25, panel_y + 172), (20 + panel_w - 25, panel_y + 172), (50, 50, 55), 1, cv2.LINE_AA)
    cv2.putText(frame, "LIVE DECISION EXPLANATION (XAI):", (32, panel_y + 187), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 150, 50), 1, cv2.LINE_AA)
    
    if latest_explanation and "explanation" in latest_explanation:
        explanation_string = latest_explanation["explanation"]
    else:
        explanation_string = "Awaiting cycle completion to gather baseline data logs..."

    # Simple text-wrapping mechanism to fit long explanations cleanly into two lines
    if len(explanation_string) > 55:
        split_idx = explanation_string.rfind(' ', 0, 55)
        if split_idx == -1: split_idx = 55
        line1 = explanation_string[:split_idx]
        line2 = explanation_string[split_idx:].strip()
        cv2.putText(frame, line1, (32, panel_y + 201), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, line2, (32, panel_y + 213), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (240, 240, 240), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, explanation_string, (32, panel_y + 201), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (240, 240, 240), 1, cv2.LINE_AA)

    return frame