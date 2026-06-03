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


def draw_traffic_light_hud(frame: np.ndarray, state: str, time_left: float, tracker: EfficiencyTracker) -> np.ndarray:
    """
    Renders high-visibility traffic light signals and telemetry benchmarking stats 
    on the video stream using semi-transparent professional overlays.
    """
    h, w = frame.shape[:2]
    
    # ── 1. Render Traffic Light Housing Background (Top Right Corner) ────────
    overlay = frame.copy()
    box_x1, box_y1 = w - 120, 70
    box_x2, box_y2 = w - 30, 290
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (20, 20, 25), -1)
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (50, 50, 55), 2, cv2.LINE_AA)
    
    # Define vertical signal positioning
    cx = (box_x1 + box_x2) // 2
    r_cy, y_cy, g_cy = 110, 180, 250
    radius = 20

    # Draw inactive structural dark backing circles
    cv2.circle(overlay, (cx, r_cy), radius, (10, 10, 30), -1)
    cv2.circle(overlay, (cx, y_cy), radius, (10, 30, 30), -1)
    cv2.circle(overlay, (cx, g_cy), radius, (10, 30, 10), -1)

    # Glow effect illumination matching current status
    if state == "RED":
        cv2.circle(overlay, (cx, r_cy), radius, (0, 0, 255), -1)        # Vivid Neon Red
    elif state == "YELLOW":
        cv2.circle(overlay, (cx, y_cy), radius, (0, 215, 255), -1)      # Vivid Neon Yellow
    elif state == "GREEN":
        cv2.circle(overlay, (cx, g_cy), radius, (0, 255, 0), -1)        # Vivid Neon Green

    # Smooth alpha blend for glassmorphic signal look
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Print remaining state timer directly under housing box
    timer_str = f"{time_left:.1f}s"
    t_w, _ = cv2.getTextSize(timer_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    cv2.putText(frame, timer_str, (cx - t_w // 2, box_y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # ── 2. Render Comparative Analytics Side-By-Side Dashboard (Bottom Left) 
    panel_y = h - 115
    cv2.rectangle(frame, (20, panel_y), (380, h - 20), (15, 15, 18), -1)
    cv2.rectangle(frame, (20, panel_y), (380, h - 20), (70, 70, 75), 1, cv2.LINE_AA)

    cv2.putText(frame, "ADAPTIVE VS FIXED BENCHMARK", (30, panel_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Fixed Baseline Standard: {FIXED_BASELINE}s always", (30, panel_y + 47), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Cumulative Saved Time:  {tracker.accumulated_saved_time:.1f}s", (30, panel_y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1, cv2.LINE_AA)

    return frame