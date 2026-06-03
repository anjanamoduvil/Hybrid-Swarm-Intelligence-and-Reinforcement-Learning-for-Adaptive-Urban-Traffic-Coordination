"""
signal.py — Signal State Machine & Adaptive Timing Logic Engine
Member 4: Adaptive Signal & Fixed vs. Adaptive Comparison
"""

import time
from config import MIN_GREEN, MAX_GREEN, YELLOW_DUR

class AdaptiveSignalController:
    """
    Manages a single-lane traffic light state machine running a 3-state pipeline:
    GREEN -> YELLOW -> RED. Computes vehicle-responsive dynamic windows.
    """
    def __init__(self):
        self.current_state = "GREEN"  # Core operational nodes: GREEN, YELLOW, RED
        self.timer_start = time.time()
        self.current_duration = MIN_GREEN
        self.cycle_count = 0

    def calculate_adaptive_duration(self, vehicle_count: int) -> int:
        """
        Computes dynamic window duration via deterministic linear expansion.
        Formula: 10 + (count * 2), bounded tightly within [10s, 60s].
        """
        calculated_time = MIN_GREEN + (vehicle_count * 2)
        return max(MIN_GREEN, min(calculated_time, MAX_GREEN))

    def update_state_machine(self, vehicle_count: int) -> tuple:
        """
        Executes cyclic state transitions based on time elapsed.
        
        Returns:
            tuple: (current_state: str, time_remaining: float, cycle_completed: bool)
        """
        elapsed = time.time() - self.timer_start
        cycle_completed = False

        if self.current_state == "GREEN":
            # Real-time tracking update of adaptive duration based on current count
            self.current_duration = self.calculate_adaptive_duration(vehicle_count)
            if elapsed >= self.current_duration:
                self.current_state = "YELLOW"
                self.timer_start = time.time()
                self.current_duration = YELLOW_DUR
                
        elif self.current_state == "YELLOW":
            if elapsed >= YELLOW_DUR:
                self.current_state = "RED"
                self.timer_start = time.time()
                # Use a standard static red duration for this simulation lane
                self.current_duration = 10 
                
        elif self.current_state == "RED":
            if elapsed >= 10:
                self.current_state = "GREEN"
                self.timer_start = time.time()
                self.current_duration = self.calculate_adaptive_duration(vehicle_count)
                self.cycle_count += 1
                cycle_completed = True  # Full cycle finished when returning to green

        time_remaining = max(0.0, self.current_duration - elapsed)
        return self.current_state, time_remaining, cycle_completed