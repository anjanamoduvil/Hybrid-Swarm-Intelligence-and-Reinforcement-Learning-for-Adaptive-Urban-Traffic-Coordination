"""
test_traffic_signal.py — Unit Testing Suite for Coordinated PSO Signals
Run with: python -m pytest test_traffic_signal.py -v
"""

import os
import sys
import pytest
import time
import numpy as np

# Ensure current folder paths take priority lookup for local imports
sys.path.insert(0, os.path.dirname(__file__))

from traffic_signal import ParticleSwarmOptimizer, CoordinatedSignalController

class TestCoordinatedSignalMechanics:

    def test_pso_boundaries(self):
        """Ensures that the PSO solver output is strictly bounded between MIN_GREEN (10s) and MAX_GREEN (60s)."""
        pso = ParticleSwarmOptimizer()
        
        # Test extreme low traffic
        g_opt, _ = pso.optimize(q_active=0, q_inactive=0, wait_active=0, wait_inactive=0)
        assert 10.0 <= g_opt <= 60.0

        # Test extreme high traffic
        g_opt, _ = pso.optimize(q_active=100, q_inactive=100, wait_active=500, wait_inactive=500)
        assert 10.0 <= g_opt <= 60.0

    def test_pso_cost_minimization(self):
        """Ensures that the cost history generally decreases or stays flat over iterations."""
        pso = ParticleSwarmOptimizer(num_particles=10, max_iter=5)
        _, history = pso.optimize(q_active=10, q_inactive=5, wait_active=30, wait_inactive=20)
        
        assert len(history) == 6
        assert history[-1] <= history[0]

    def test_state_transition_sequence(self):
        """Validates that phase shifts cleanly along the sequential 4-state pipeline."""
        controller = CoordinatedSignalController()
        assert controller.current_state == "LANE1_GREEN"
        
        # Force expiration of Lane 1 Green
        controller.current_duration = 5.0
        controller.timer_start = time.time() - 6.0
        state, _, completed = controller.update_state_machine(l1_q=5, l2_q=2, l1_wait=10.0, l2_wait=5.0)
        assert state == "LANE1_YELLOW"
        assert completed is False

        # Force expiration of Lane 1 Yellow to enter Lane 2 Green
        controller.timer_start = time.time() - 4.0
        state, _, completed = controller.update_state_machine(l1_q=5, l2_q=2, l1_wait=10.0, l2_wait=5.0)
        assert state == "LANE2_GREEN"
        assert completed is False
        # Verify PSO ran
        assert controller.last_pso_results["active_lane"] == 2

        # Force expiration of Lane 2 Green
        controller.current_duration = 5.0
        controller.timer_start = time.time() - 6.0
        state, _, completed = controller.update_state_machine(l1_q=5, l2_q=2, l1_wait=10.0, l2_wait=5.0)
        assert state == "LANE2_YELLOW"
        assert completed is False

        # Force expiration of Lane 2 Yellow to loop back to Lane 1 Green
        controller.timer_start = time.time() - 4.0
        state, _, completed = controller.update_state_machine(l1_q=5, l2_q=2, l1_wait=10.0, l2_wait=5.0)
        assert state == "LANE1_GREEN"
        assert completed is True
        assert controller.last_pso_results["active_lane"] == 1