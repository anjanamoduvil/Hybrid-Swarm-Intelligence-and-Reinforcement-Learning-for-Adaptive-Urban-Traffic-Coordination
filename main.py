#!/usr/bin/env python3
"""
main.py — Full Integration Entry Point
Traffic Monitoring & Adaptive Signal System

Chains all 4 member modules in a single frame loop:
  Member 1 — detector.py  : YOLOv8 detection + HUD
  Member 2 — tracker.py   : vehicle ID tracking
  Member 3 — density.py / alerts.py : density estimation + congestion overlays
  Member 4 — signal.py / compare.py : adaptive signal state machine + benchmark overlay
"""

import cv2
import argparse

from detector import TrafficDetector
from tracker import VehicleTracker
from density import compute_density, classify_density, should_trigger_alert, log_count_to_csv
from alerts import annotate_frame as m3_annotate_frame
from signal import AdaptiveSignalController
from compare import EfficiencyTracker, draw_traffic_light_hud


def main():
    parser = argparse.ArgumentParser(description="Traffic Monitoring — Full Integration")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--video",  help="Override video path")
    parser.add_argument("--conf",   type=float, help="Override confidence threshold")
    args = parser.parse_args()

    # ── Initialise all modules ───────────────────────────────────────────────
    detector        = TrafficDetector(config_path=args.config)
    vehicle_tracker = VehicleTracker()
    signal_ctrl     = AdaptiveSignalController()
    efficiency      = EfficiencyTracker()

    if args.video:
        detector.video_path = args.video
    if args.conf:
        detector.conf_threshold = args.conf

    print("=" * 60)
    print("  Traffic Monitoring & Adaptive Signal System")
    print(f"  Video  : {detector.video_path}")
    print(f"  Model  : {detector.model_path}")
    print(f"  Conf   : {detector.conf_threshold}")
    print("=" * 60)

    window = "Traffic System — Full Integration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    frame_num = 0

    try:
        for frame, detections in detector.process_video():
            frame_num += 1

            # ── Member 2: tracker ────────────────────────────────────────────
            bboxes = [d["bbox"] for d in detections]
            vehicle_tracker.update(bboxes)

            # ── Member 3: density + alert overlays ───────────────────────────
            vehicle_count = sum(
                1 for d in detections if d["class_name"] != "Pedestrian"
            )
            density = compute_density(vehicle_count)
            band    = classify_density(vehicle_count)
            log_count_to_csv(frame_num, vehicle_count, band)
            frame = m3_annotate_frame(frame, vehicle_count, density)

            # ── Member 4: signal state machine + comparison overlay ──────────
            state, time_left, cycle_done = signal_ctrl.update_state_machine(vehicle_count)
            if cycle_done:
                adaptive_dur = signal_ctrl.calculate_adaptive_duration(vehicle_count)
                efficiency.register_completed_cycle(adaptive_dur, vehicle_count)
            frame = draw_traffic_light_hud(frame, state, time_left, efficiency)

            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[main] Quit requested.")
                break
            elif key == ord(" "):
                while True:
                    k = cv2.waitKey(30) & 0xFF
                    if k in (ord(" "), ord("q")):
                        if k == ord("q"):
                            key = ord("q")
                        break
                if key == ord("q"):
                    break

    except KeyboardInterrupt:
        print("[main] Interrupted.")
    finally:
        cv2.destroyAllWindows()
        print(f"[main] Processed {frame_num} frames.")
        print(f"[main] Total time saved vs fixed baseline: {efficiency.accumulated_saved_time:.1f}s")


if __name__ == "__main__":
    main()
