"""
config.py — Shared Configuration Parameters
Traffic Monitoring & Adaptive Signal System
"""

# ── Member 3: Density band thresholds (vehicle count) ───────────────────────
THRESHOLDS = {
    "low":  3,   # <= 3  vehicles → LOW
    "med":  6,   # <= 6  vehicles → MED
    "high": 10,  # >  10 vehicles → HIGH  (also triggers alert)
}

# ── Member 3: CSV log output path ────────────────────────────────────────────
CSV_LOG_PATH = "density_log.csv"

# ── Member 3 (Week 3): Multi-Intersection Coordination ───────────────────────
N_INTERSECTIONS  = 4
PROPAGATION_RATE = 0.3
MULTI_LOG_PATH   = "intersection_grid_log.csv"

# ── Member 1 & 2 (Week 3): Prediction & RL Configuration ─────────────────────
PREDICTION_WINDOW = 5   # moving-average window for M1
RL_ALPHA = 0.1          # learning rate for M2
RL_GAMMA = 0.9          # discount factor for M2

# ── Member 4: Adaptive Signal Configuration ──────────────────────────────────
MIN_GREEN      = 10   # Minimum green duration (seconds)
MAX_GREEN      = 60   # Maximum green duration (seconds)
YELLOW_DUR     = 3    # Fixed yellow transition duration (seconds)
FIXED_BASELINE = 30   # Fixed-time baseline for comparison (seconds)

# ── Member 4: Performance metrics log path ───────────────────────────────────
CYCLE_LOG_PATH = "signal_cycle_log.csv"
