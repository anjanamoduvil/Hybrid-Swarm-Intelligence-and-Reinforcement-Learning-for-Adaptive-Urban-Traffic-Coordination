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
MAX_LANE_CAPACITY = 40  # physical cap on vehicles queued at one intersection;
                        # bounds congestion propagation so it saturates
                        # instead of compounding indefinitely between HIGH nodes

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

# ── Member 3 (Week 4): Federated Learning Prototype ──────────────────────────
FED_ROUNDS          = 8      # Default number of federated training rounds
FED_CONVERGENCE_TOL = 1e-3   # Max change in global weights to declare convergence
FED_MIN_LOCAL_POINTS = 2     # Minimum local history points needed to fit a local model
FED_PERSONALIZATION_ALPHA = 0.7  # Blend weight for personalized federated learning:
                                  # personalized = alpha*local + (1-alpha)*global.
                                  # Fixes poor global-model fit under heterogeneous nodes.

# ── Member 3 (Week 4): Digital Twin Network Simulation ───────────────────────
TWIN_LOG_PATH        = "digital_twin_log.csv"
TWIN_DEFAULT_HORIZON = 5     # Default number of ticks to simulate ahead
RECOVERY_MAX_TICKS   = 20    # Max ticks to search for congestion recovery
DISTURBANCE_DEFAULT_VEHICLES = 25  # Default surge size used in resilience scenarios

# Bonus vehicles/sec cleared on top of IntersectionGrid's own built-in ~20%
# passive-departure baseline (see intersection_sim.tick()). Kept small since
# it only needs to differentiate strategies, not replace the baseline flow.
DISCHARGE_RATE_PER_SEC = 0.05
