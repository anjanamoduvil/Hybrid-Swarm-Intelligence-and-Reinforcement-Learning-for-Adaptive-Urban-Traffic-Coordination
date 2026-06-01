"""
alerts.py — OpenCV Display Module for Density & Congestion Alerts
Member 3: Traffic Monitoring & Adaptive Signal System
"""

import cv2
import numpy as np
from config import THRESHOLDS


# Colour map for each density band  (BGR)
BAND_COLOURS = {
    "LOW":  (0, 200, 0),    # green
    "MED":  (0, 165, 255),  # orange
    "HIGH": (0, 0, 220),    # red
}


def draw_density_bar(frame: np.ndarray, vehicle_count: int, density: float) -> np.ndarray:
    """
    Draw a colour-coded vertical density bar on the right edge of the frame.

    The bar fills proportionally to `density` (0.0 – 1.0) and changes
    colour according to the current band.

    Args:
        frame:         BGR frame to draw on (modified in place).
        vehicle_count: Raw vehicle count used to determine the band.
        density:       Normalised density value in [0, 1].

    Returns:
        The annotated frame.
    """
    h, w = frame.shape[:2]

    from density import classify_density
    band = classify_density(vehicle_count)
    colour = BAND_COLOURS[band]

    bar_x = w - 40          # left edge of the bar
    bar_w = 25              # bar width in pixels
    bar_max_h = int(h * 0.6)  # maximum bar height (60 % of frame height)
    bar_top_y = int(h * 0.15)  # top anchor

    # Background (empty bar)
    cv2.rectangle(
        frame,
        (bar_x, bar_top_y),
        (bar_x + bar_w, bar_top_y + bar_max_h),
        (60, 60, 60),
        thickness=-1,
    )

    # Filled portion (grows upward from the bottom of the bar area)
    fill_h = int(bar_max_h * density)
    fill_top = bar_top_y + bar_max_h - fill_h

    if fill_h > 0:
        cv2.rectangle(
            frame,
            (bar_x, fill_top),
            (bar_x + bar_w, bar_top_y + bar_max_h),
            colour,
            thickness=-1,
        )

    # Bar border
    cv2.rectangle(
        frame,
        (bar_x, bar_top_y),
        (bar_x + bar_w, bar_top_y + bar_max_h),
        (200, 200, 200),
        thickness=1,
    )

    # "DENSITY" label above the bar
    cv2.putText(
        frame,
        "DENSITY",
        (bar_x - 10, bar_top_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    return frame


def draw_band_label(frame: np.ndarray, vehicle_count: int) -> np.ndarray:
    """
    Draw the density band label (LOW / MED / HIGH) in the top-left corner.

    Args:
        frame:         BGR frame to draw on (modified in place).
        vehicle_count: Raw vehicle count.

    Returns:
        The annotated frame.
    """
    from density import classify_density
    band = classify_density(vehicle_count)
    colour = BAND_COLOURS[band]

    label = f"DENSITY: {band}"
    cv2.putText(
        frame,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        colour,
        2,
        cv2.LINE_AA,
    )
    return frame


def draw_alert_banner(frame: np.ndarray, vehicle_count: int) -> np.ndarray:
    """
    Draw a red warning banner across the top of the frame when vehicle
    count exceeds the HIGH threshold.

    Args:
        frame:         BGR frame to draw on (modified in place).
        vehicle_count: Raw vehicle count.

    Returns:
        The annotated frame (banner added only when alert is active).
    """
    from density import should_trigger_alert
    if not should_trigger_alert(vehicle_count):
        return frame

    h, w = frame.shape[:2]
    banner_h = 40

    # Semi-transparent red overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 180), thickness=-1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Warning text
    text = f"!! HIGH CONGESTION ALERT — {vehicle_count} VEHICLES DETECTED !!"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(
        frame,
        text,
        (text_x, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return frame


def annotate_frame(frame: np.ndarray, vehicle_count: int, density: float) -> np.ndarray:
    """
    Convenience function: apply all three Member-3 overlays in one call.

    Call order matters — alert banner goes last so it is always on top.

    Args:
        frame:         BGR frame.
        vehicle_count: Raw vehicle count from the tracker.
        density:       Normalised density value (0.0 – 1.0).

    Returns:
        Fully annotated frame.
    """
    frame = draw_band_label(frame, vehicle_count)
    frame = draw_density_bar(frame, vehicle_count, density)
    frame = draw_alert_banner(frame, vehicle_count)
    return frame
