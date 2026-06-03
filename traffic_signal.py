import time
import yaml
import numpy as np
import os

class ParticleSwarmOptimizer:
    """
    Solves for the optimal green light duration using a Particle Swarm Optimization (PSO) algorithm.
    It balances discharging queue lengths and accumulating waiting times.
    """
    def __init__(self, num_particles=15, max_iter=10, w_queue=1.5, w_wait=1.0):
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w_queue = w_queue
        self.w_wait = w_wait
        
        # Simulation model parameters
        self.discharge_rate = 0.25  # vehicles discharged per second of green
        self.arrival_rate = 0.12    # vehicles arriving per second of red
        self.min_green = 10
        self.max_green = 60

    def cost_function(self, g, q_active, q_inactive, wait_active, wait_inactive):
        """
        Calculates the fitness cost of a proposed green light duration 'g'.
        Goal: Minimize total queue size and waiting times.
        """
        # Active lane discharges
        q_active_end = max(0.0, q_active - self.discharge_rate * g)
        wait_active_end = max(0.0, wait_active - g)

        # Inactive lane accumulates queue and wait time
        q_inactive_end = q_inactive + self.arrival_rate * g
        wait_inactive_end = wait_inactive + g

        # Total Cost
        queue_penalty = self.w_queue * (q_active_end + q_inactive_end)
        wait_penalty = self.w_wait * (wait_active_end + wait_inactive_end)
        
        return queue_penalty + wait_penalty

    def optimize(self, q_active, q_inactive, wait_active, wait_inactive):
        """
        Runs the PSO algorithm to search for the g value that minimizes the cost.
        Returns:
            best_g (float): The optimal green light duration.
            fitness_history (list): Best cost found at each iteration (useful for UI plots).
        """
        # Particles represent proposed green durations in [min_green, max_green]
        particles = np.random.uniform(self.min_green, self.max_green, self.num_particles)
        velocities = np.random.uniform(-5.0, 5.0, self.num_particles)
        
        # Local & global bests
        p_best = np.copy(particles)
        p_best_cost = np.array([
            self.cost_function(p, q_active, q_inactive, wait_active, wait_inactive)
            for p in particles
        ])
        
        g_best_idx = np.argmin(p_best_cost)
        g_best = p_best[g_best_idx]
        g_best_cost = p_best_cost[g_best_idx]

        fitness_history = [float(g_best_cost)]

        # Swarm hyper-parameters
        omega = 0.5   # inertia weight
        c1 = 1.5      # cognitive (personal best) weight
        c2 = 1.5      # social (swarm best) weight

        for _ in range(self.max_iter):
            for i in range(self.num_particles):
                # Update velocity
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (omega * velocities[i] +
                                 c1 * r1 * (p_best[i] - particles[i]) +
                                 c2 * r2 * (g_best - particles[i]))
                
                # Update position
                particles[i] += velocities[i]
                
                # Clamp boundaries
                particles[i] = max(self.min_green, min(particles[i], self.max_green))
                
                # Evaluate fitness
                cost = self.cost_function(particles[i], q_active, q_inactive, wait_active, wait_inactive)
                
                # Update personal best
                if cost < p_best_cost[i]:
                    p_best[i] = particles[i]
                    p_best_cost[i] = cost
                    
            # Update global best
            best_idx = np.argmin(p_best_cost)
            if p_best_cost[best_idx] < g_best_cost:
                g_best = p_best[best_idx]
                g_best_cost = p_best_cost[best_idx]
                
            fitness_history.append(float(g_best_cost))

        return float(g_best), fitness_history


class CoordinatedSignalController:
    """
    Coordinated Traffic Signal controller for a two-lane intersection.
    Transitions through LANE1_GREEN -> LANE1_YELLOW -> LANE2_GREEN -> LANE2_YELLOW.
    Integrates PSO to optimize green durations.
    """
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.current_state = "LANE1_GREEN"
        self.timer_start = time.time()
        self.current_duration = 15.0
        self.cycle_count = 0
        
        # Load swarm settings
        self.pso_particles = 15
        self.pso_iterations = 10
        self.w_queue = 1.5
        self.w_wait = 1.0
        self.load_config()

        self.pso = ParticleSwarmOptimizer(
            num_particles=self.pso_particles,
            max_iter=self.pso_iterations,
            w_queue=self.w_queue,
            w_wait=self.w_wait
        )
        
        # Solver tracking metadata for API visualization
        self.last_pso_results = {
            "best_duration": 15.0,
            "cost_history": [0.0],
            "cost": 0.0,
            "active_lane": 1,
            "timestamp": time.time()
        }

    def load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    cfg = yaml.safe_load(f)
                    if cfg:
                        self.pso_particles = cfg.get("pso_particles", self.pso_particles)
                        self.pso_iterations = cfg.get("pso_iterations", self.pso_iterations)
                        self.w_queue = cfg.get("pso_weight_queue", self.w_queue)
                        self.w_wait = cfg.get("pso_weight_wait", self.w_wait)
            except Exception as e:
                print(f"[Signal] Warning loading config: {e}")

    def update_state_machine(self, l1_q, l2_q, l1_wait, l2_wait):
        """
        Updates the state machine. Triggers PSO optimization when starting a new green phase.
        Returns:
            state (str): Current active phase.
            time_left (float): Countdown timer.
            cycle_completed (bool): True if a full two-lane cycle completed.
        """
        elapsed = time.time() - self.timer_start
        cycle_completed = False
        yellow_dur = 3.0

        if self.current_state == "LANE1_GREEN":
            if elapsed >= self.current_duration:
                self.current_state = "LANE1_YELLOW"
                self.timer_start = time.time()
                self.current_duration = yellow_dur
                
        elif self.current_state == "LANE1_YELLOW":
            if elapsed >= yellow_dur:
                self.current_state = "LANE2_GREEN"
                self.timer_start = time.time()
                # Run PSO optimization for Lane 2 Green
                opt_dur, history = self.pso.optimize(
                    q_active=l2_q,
                    q_inactive=l1_q,
                    wait_active=l2_wait,
                    wait_inactive=l1_wait
                )
                self.current_duration = opt_dur
                self.last_pso_results = {
                    "best_duration": round(opt_dur, 2),
                    "cost_history": [round(x, 2) for x in history],
                    "cost": round(history[-1], 2),
                    "active_lane": 2,
                    "timestamp": time.time()
                }

        elif self.current_state == "LANE2_GREEN":
            if elapsed >= self.current_duration:
                self.current_state = "LANE2_YELLOW"
                self.timer_start = time.time()
                self.current_duration = yellow_dur

        elif self.current_state == "LANE2_YELLOW":
            if elapsed >= yellow_dur:
                self.current_state = "LANE1_GREEN"
                self.timer_start = time.time()
                # Run PSO optimization for Lane 1 Green
                opt_dur, history = self.pso.optimize(
                    q_active=l1_q,
                    q_inactive=l2_q,
                    wait_active=l1_wait,
                    wait_inactive=l2_wait
                )
                self.current_duration = opt_dur
                self.last_pso_results = {
                    "best_duration": round(opt_dur, 2),
                    "cost_history": [round(x, 2) for x in history],
                    "cost": round(history[-1], 2),
                    "active_lane": 1,
                    "timestamp": time.time()
                }
                self.cycle_count += 1
                cycle_completed = True

        time_remaining = max(0.0, self.current_duration - elapsed)
        return self.current_state, time_remaining, cycle_completed