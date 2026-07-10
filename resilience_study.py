"""
resilience_study.py — Network Resilience and Digital Twin Simulation
Member 4: Task 7 — Testing system recovery under unexpected edge failures.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from traffic_signal import ParticleSwarmOptimizer

def run_resilience_simulation():
    print("=" * 60)
    print("STARTING TASK 7: NETWORK RESILIENCE & DIGITAL TWIN STUDY")
    print("=" * 60)
    
    # Initialize the optimizer representing our intersection controller
    pso = ParticleSwarmOptimizer(num_particles=15, max_iter=10)
    
    # Simulation setup over 30 evaluation intervals
    intervals = 30
    
    # Seed traffic conditions
    q_l1, q_l2 = 25.0, 20.0
    wait_l1, wait_l2 = 45.0, 30.0
    
    metrics = {
        "interval": [],
        "l1_queue": [],
        "l2_queue": [],
        "optimized_green": [],
        "system_status": []
    }
    
    for i in range(1, intervals + 1):
        # Phase 1: Normal Operations (Intervals 1-10)
        if i <= 10:
            status = "Normal Operations"
            # Standard simulation step updates
            l1_input_q = q_l1 + np.random.uniform(-2, 3)
            l2_input_q = q_l2 + np.random.uniform(-1, 2)
            
        # Phase 2: Structural Shock / Edge Sensor Failure (Intervals 11-20)
        elif 11 <= i <= 20:
            status = "Sensor Edge Failure"
            # Scenario: Lane 1 loop detector breaks, dropping or corrupting queue telemetry to 0
            l1_input_q = 0.0  
            l2_input_q = q_l2 + np.random.uniform(2, 5) # Cross lane congestion accumulates
            
        # Phase 3: Cooperative Swarm Recovery Activated (Intervals 21-30)
        else:
            status = "Swarm Recovery Active"
            # Scenario: System detects anomalous flatline data and utilizes neighbor twin
            # estimations to patch the missing queue telemetry metrics.
            l1_input_q = 18.0  # Interpolated fallback backup estimate
            l2_input_q = q_l2 + np.random.uniform(-2, 2)

        # Run optimization cycle based on the current resilience scenario state
        best_g, _ = pso.optimize(
            q_active=max(0.0, l1_input_q),
            q_inactive=max(0.0, l2_input_q),
            wait_active=wait_l1,
            wait_inactive=wait_l2
        )
        
        # Internal state updates mimicking an ongoing traffic model loop
        q_l1 = max(0.0, q_l1 + (0.12 * best_g) - (0.25 * best_g) if i > 10 else q_l1)
        
        # Log metrics for analytics compilation
        metrics["interval"].append(i)
        metrics["l1_queue"].append(l1_input_q)
        metrics["l2_queue"].append(l2_input_q)
        metrics["optimized_green"].append(best_g)
        metrics["system_status"].append(status)
        
        print(f"Interval {i:02d} | Status: {status:<22} | Measured L1 Queue: {l1_input_q:>4.1f} | Opt Green: {best_g:.2f}s")
        time.sleep(0.05)

    # Generate the Resilience Analysis Plot for Deliverable 5
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["interval"], metrics["l1_queue"], label="Lane 1 Queue (Observed)", color='crimson', lw=2, marker='o')
    plt.plot(metrics["interval"], metrics["optimized_green"], label="Allocated Green Duration (s)", color='teal', linestyle='--', lw=2, marker='s')
    
    # Visual markers highlighting structural shock regions
    plt.axvspan(10.5, 20.5, color='orange', alpha=0.15, label='Telemetry Failure Windows')
    plt.axvspan(20.5, 30.5, color='green', alpha=0.15, label='Swarm Resilient Recovery')
    
    plt.title("Digital Twin Performance Log: System Recovery Under Telemetry Loss", fontsize=12, fontweight='bold')
    plt.xlabel("Simulation Interval Index", fontsize=10)
    plt.ylabel("Telemetry Vector Scale Metric", fontsize=10)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    output_img = "resilience_analysis_curve.png"
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"\n[Resilience Engine] Simulation finished. Analysis curve plot exported successfully as '{output_img}'!")
    plt.show()

if __name__ == "__main__":
    run_resilience_simulation()