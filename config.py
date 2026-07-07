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

# ── Member 4: Adaptive Signal Configuration ──────────────────────────────────
MIN_GREEN      = 10   # Minimum green duration (seconds)
MAX_GREEN      = 60   # Maximum green duration (seconds)
YELLOW_DUR     = 3    # Fixed yellow transition duration (seconds)
FIXED_BASELINE = 30   # Fixed-time baseline for comparison (seconds)

# ── Member 4: Performance metrics log path ───────────────────────────────────
CYCLE_LOG_PATH = "signal_cycle_log.csv"

# ── Member 3 (Week 4): Federated Learning Prototype ──────────────────────────
FED_ROUNDS          = 8      # Default number of federated training rounds
FED_CONVERGENCE_TOL = 1e-3   # Max change in global weights to declare convergence
FED_MIN_LOCAL_POINTS = 2     # Minimum local history points needed to fit a local model

# ── Member 3 (Week 4): Digital Twin Network Simulation ───────────────────────
TWIN_LOG_PATH        = "digital_twin_log.csv"
TWIN_DEFAULT_HORIZON = 5     # Default number of ticks to simulate ahead
RECOVERY_MAX_TICKS   = 20    # Max ticks to search for congestion recovery
DISTURBANCE_DEFAULT_VEHICLES = 25  # Default surge size used in resilience scenarios
DISCHARGE_RATE_PER_SEC = 0.5  # Vehicles cleared per second of green time (saturation flow)

