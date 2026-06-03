"""
density.py — Density Estimation & Congestion Classification
Member 3: Urban Traffic Monitoring & Adaptive Signal System
"""

import csv
import os
from datetime import datetime
import config as _cfg


def compute_density(vehicle_count: int) -> float:
    """
    Compute a normalized density score (0.0 – 1.0) from raw vehicle count.
    Uses the HIGH threshold as the saturation point.
    """
    high = _cfg.THRESHOLDS["high"]
    return min(vehicle_count / high, 1.0)


def classify_density(vehicle_count: int) -> str:
    """
    Classify traffic density into LOW / MED / HIGH bands.

    Returns:
        str: 'LOW', 'MED', or 'HIGH'
    """
    if vehicle_count <= _cfg.THRESHOLDS["low"]:
        return "LOW"
    elif vehicle_count <= _cfg.THRESHOLDS["med"]:
        return "MED"
    else:
        return "HIGH"


def should_trigger_alert(vehicle_count: int) -> bool:
    """
    Return True when vehicle count exceeds the HIGH threshold.
    """
    return vehicle_count > _cfg.THRESHOLDS["high"]


def log_count_to_csv(frame_number: int, vehicle_count: int, band: str) -> None:
    """
    Append one row (timestamp, frame, count, band) to the CSV log.
    Creates the file with a header row if it does not exist yet.
    Reads CSV_LOG_PATH from config at call-time so tests can override it.
    """
    path = _cfg.CSV_LOG_PATH
    file_exists = os.path.isfile(path)

    with open(path, mode="a", newline="") as csvfile:
        fieldnames = ["timestamp", "frame", "vehicle_count", "band"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "frame": frame_number,
                "vehicle_count": vehicle_count,
                "band": band,
            }
        )


def log_dual_counts_to_csv(frame_number: int, l1_count: int, l1_band: str, l2_count: int, l2_band: str) -> None:
    """
    Append one row (timestamp, frame, lane1_count, lane1_band, lane2_count, lane2_band)
    to the CSV log. Creates the file with a header row if it does not exist yet.
    """
    path = _cfg.CSV_LOG_PATH
    file_exists = os.path.isfile(path)

    with open(path, mode="a", newline="") as csvfile:
        fieldnames = [
            "timestamp", 
            "frame", 
            "lane1_count", 
            "lane1_band", 
            "lane2_count", 
            "lane2_band"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "frame": frame_number,
                "lane1_count": l1_count,
                "lane1_band": l1_band,
                "lane2_count": l2_count,
                "lane2_band": l2_band
            }
        )
