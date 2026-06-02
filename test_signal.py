"""
test_signal.py — Unit Testing Suite for Member 4 Core Slice
Run with: python -m pytest test_signal.py -v
"""

import os
import sys
import pytest
import time

# Ensure current folder paths take priority lookup for local imports
sys.path.insert(0, os.path.dirname(__file__))

from signal import AdaptiveSignalController

class TestAdaptiveSignalMechanics:

    def test_timing_formula_minimum_floor(self):
        """Ensures that empty or low traffic lanes match the 10-second floor duration."""
        controller = AdaptiveSignalController()
        assert controller.calculate_adaptive_duration(0) == 10
        assert controller.calculate_adaptive_duration(3) == 16

    def test_timing_formula_maximum_ceiling(self):
        """Ensures high vehicle numbers safely saturate exactly at the 60s ceiling limit."""
        controller = AdaptiveSignalController()
        assert controller.calculate_adaptive_duration(25) == 60
        assert controller.calculate_adaptive_duration(150) == 60

    def test_state_transition_sequence(self):
        """Validates phase shifts cleanly along the sequential G -> Y -> R pipeline."""
        controller = AdaptiveSignalController()
        assert controller.current_state == "GREEN"
        
        # Artificially force timer expiration past max limits to trigger active state change
        controller.timer_start = time.time() - 65
        state, _, completed = controller.update_state_machine(5)
        
        assert state == "YELLOW"
        assert completed is False