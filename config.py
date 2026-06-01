"""
config.py — Configuration Parameters (Member 3 block)
Traffic Monitoring & Adaptive Signal System
"""

# ── Density band thresholds (vehicle count) ─────────────────────────────────
THRESHOLDS = {
    "low":  5,   # <= 5  vehicles → LOW
    "med":  10,  # <= 10 vehicles → MED
    "high": 20,  # >  20 vehicles → HIGH  (also triggers alert)
}

# ── CSV log output path ──────────────────────────────────────────────────────
CSV_LOG_PATH = "density_log.csv"
