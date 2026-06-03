"""
config.py — Shared Configuration Parameters
Traffic Monitoring & Adaptive Signal System
"""

# ── Member 3: Density band thresholds (vehicle count) ───────────────────────
THRESHOLDS = {
    "low":  5,   # <= 5  vehicles → LOW
    "med":  10,  # <= 10 vehicles → MED
    "high": 20,  # >  20 vehicles → HIGH  (also triggers alert)
}

# ── Member 3: CSV log output path ────────────────────────────────────────────
CSV_LOG_PATH = "density_log.csv"


# ── Member 4: Adaptive Signal Configuration ──────────────────────────────────
MIN_GREEN     = 10   # Minimum green duration (seconds)
MAX_GREEN     = 60   # Maximum green duration (seconds)
YELLOW_DUR    = 3    # Fixed yellow transition duration (seconds)
FIXED_BASELINE = 30  # Fixed-time baseline for comparison (seconds)

# ── Member 4: Performance metrics log path ───────────────────────────────────
CYCLE_LOG_PATH = "signal_cycle_log.csv"
