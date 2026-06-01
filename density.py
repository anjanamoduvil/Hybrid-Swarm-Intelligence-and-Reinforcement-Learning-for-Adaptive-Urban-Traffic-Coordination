"""
density.py — Density Estimation & Congestion Classification
Member 3: Traffic Monitoring & Adaptive Signal System
"""

import csv
import os
from datetime import datetime
from config import THRESHOLDS, CSV_LOG_PATH


def compute_density(vehicle_count: int) -> float:
    """
    Compute a normalized density score (0.0 – 1.0) from raw vehicle count.
    Uses the HIGH threshold as the saturation point.
    """
    high = THRESHOLDS["high"]
    return min(vehicle_count / high, 1.0)


def classify_density(vehicle_count: int) -> str:
    """
    Classify traffic density into LOW / MED / HIGH bands.

    Returns:
        str: 'LOW', 'MED', or 'HIGH'
    """
    if vehicle_count <= THRESHOLDS["low"]:
        return "LOW"
    elif vehicle_count <= THRESHOLDS["med"]:
        return "MED"
    else:
        return "HIGH"


def should_trigger_alert(vehicle_count: int) -> bool:
    """
    Return True when vehicle count exceeds the HIGH threshold.
    """
    return vehicle_count > THRESHOLDS["high"]


def log_count_to_csv(frame_number: int, vehicle_count: int, band: str) -> None:
    """
    Append one row (timestamp, frame, count, band) to the CSV log.
    Creates the file with a header row if it does not exist yet.
    """
    file_exists = os.path.isfile(CSV_LOG_PATH)

    with open(CSV_LOG_PATH, mode="a", newline="") as csvfile:
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
