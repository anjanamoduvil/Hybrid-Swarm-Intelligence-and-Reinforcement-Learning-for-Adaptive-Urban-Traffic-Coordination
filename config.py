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
<<<<<<< HEAD
=======

"""
config.py — Configuration Parameters (Shared Ecosystem)
Traffic Monitoring & Adaptive Signal System
"""

# ── Density band thresholds (Member 3 block) ───────────────────────────────
THRESHOLDS = {
    "low":  5,   # <= 5  vehicles → LOW
    "med":  10,  # <= 10 vehicles → MED
    "high": 20,  # >  20 vehicles → HIGH  (also triggers alert)
}

# ── CSV log output path ──────────────────────────────────────────────────────
CSV_LOG_PATH = "density_log.csv"


# ── Adaptive Signal Configuration (Member 4 block) ──────────────────────────
MIN_GREEN = 10       # Baseline minimum floor duration for green signal (seconds)
MAX_GREEN = 60       # Maximum saturation ceiling duration for green signal (seconds)
YELLOW_DUR = 3       # Fixed transition window state duration (seconds)
FIXED_BASELINE = 30  # Standard non-adaptive control benchmark baseline (seconds)

# ── Performance Metrics Log Path ─────────────────────────────────────────────
CYCLE_LOG_PATH = "signal_cycle_log.csv"
>>>>>>> origin/master
