"""
digital_twin.py — Digital Twin Network Simulation
Member 3 (Week 4): Hybrid Swarm Intelligence & RL for Adaptive Urban Traffic Coordination

Builds a virtual replica ("twin") of a live IntersectionGrid that:
    - replicates the current traffic state,
    - simulates future traffic evolution over N ticks,
    - evaluates alternative signal-timing strategies before they are
      applied to the live network,
    - predicts congestion recovery time after a simulated disturbance.

The twin is deliberately a *separate* IntersectionGrid instance (never the
live one) so experiments run inside it can never affect real traffic.

Compatibility note (updated for Member 1's Week 4 graph integration):
IntersectionGrid now optionally attaches a DynamicTrafficGraph +
GraphIntelligenceModule to itself and, on every tick(), retrains the GCN
and overwrites the shared "static/traffic_graph.png" dashboard asset. A
Digital Twin must not inherit that side effect — hypothetical/simulated
future ticks should never retrain the live model or overwrite the live
dashboard's graph image with imagined data. sync() explicitly detaches
those two attributes from the twin's mirror grid so tick()'s Week-4
Member-1 branch is skipped entirely inside the twin (the `if
self.traffic_graph is not None:` guard in intersection_sim.py becomes a
no-op). This keeps the twin fast and side-effect-free without touching
intersection_sim.py itself.
"""

from __future__ import annotations

import copy

import config as _cfg
from intersection_sim import IntersectionGrid


# ── Alternative signal strategies ────────────────────────────────────────────
# Each strategy is a function: IntersectionGrid -> {node_id: green_time_seconds}

def strategy_green_wave(grid: IntersectionGrid) -> dict:
    """Current cooperative strategy: RL + priority-order green-wave (Week 3)."""
    return grid.allocate_green_times()


def strategy_fixed_baseline(grid: IntersectionGrid) -> dict:
    """Naive baseline: every intersection gets the same fixed green time."""
    return {node_id: float(_cfg.FIXED_BASELINE) for node_id in grid.nodes}


def strategy_density_only(grid: IntersectionGrid) -> dict:
    """Green time scaled only by density — no RL adjustment, no priority bonus."""
    allocations = {}
    for node_id, node in grid.nodes.items():
        green = _cfg.MIN_GREEN + (_cfg.MAX_GREEN - _cfg.MIN_GREEN) * node.density
        allocations[node_id] = round(max(_cfg.MIN_GREEN, min(_cfg.MAX_GREEN, green)), 1)
    return allocations


STRATEGIES = {
    "green_wave": strategy_green_wave,
    "fixed_baseline": strategy_fixed_baseline,
    "density_only": strategy_density_only,
}


class DigitalTwin:
    """
    A synchronized virtual copy of an IntersectionGrid used to run
    what-if simulations without touching the live network.
    """

    def __init__(self, source_grid: IntersectionGrid, log_path: str = None):
        self.source_grid = source_grid
        self.log_path = log_path if log_path is not None else _cfg.TWIN_LOG_PATH
        self.mirror: IntersectionGrid = None
        self.sync()

    # ── Replication ──────────────────────────────────────────────────────

    def sync(self) -> None:
        """
        Replicate the current live network state (topology, vehicle
        counts, and history) into the twin. Call this whenever the twin
        should be reset to "now" before running a new what-if scenario.
        """
        self.mirror = IntersectionGrid(
            n=self.source_grid.n,
            topology=copy.deepcopy(self.source_grid.topology),
            log_path=self.log_path,
        )
        for node_id, node in self.source_grid.nodes.items():
            self.mirror.nodes[node_id].vehicle_count = node.vehicle_count
            self.mirror.nodes[node_id].history = list(node.history)
        self.mirror.tick_count = self.source_grid.tick_count

        # Detach Member 1's Week 4 graph-intelligence integration from the
        # twin — see module docstring. Harmless no-op if those attributes
        # don't exist / are already None (e.g. torch/torch_geometric not
        # installed).
        self.mirror.traffic_graph = None
        self.mirror.graph_intelligence = None

    # ── Discharge (service-rate) bonus for strategy comparison ───────────

    def _apply_strategy_discharge(self, green_times: dict) -> None:
        """
        IntersectionGrid.tick() already models a flat ~20% passive
        departure baseline every tick, but that baseline is identical
        regardless of signal strategy — so strategies alone can never be
        told apart on vehicle_count evolution. This adds a small *extra*
        clearance on top of that baseline, proportional to each node's
        allocated green_time, so better-timed strategies visibly clear
        more congestion than worse-timed ones (Task 5's "evaluate
        alternative signal strategies").
        """
        rate = _cfg.DISCHARGE_RATE_PER_SEC
        for node_id, green_time in green_times.items():
            node = self.mirror.nodes[node_id]
            bonus_cleared = green_time * rate
            node.vehicle_count = max(0, int(round(node.vehicle_count - bonus_cleared)))

    def _step_with_strategy(self, strategy_fn) -> dict:
        """
        Advance the twin by one tick using a *substituted* strategy
        instead of whatever grid.tick() would normally allocate,
        mirroring tick()'s own step order (propagate -> passive discharge
        -> record -> green-time allocation) so results stay comparable to
        the live network, then layers on the strategy-proportional bonus
        discharge described above.

        Returns:
            dict: {tick, green_times, avg_vehicle_count, max_vehicle_count, propagation}
        """
        propagation = self.mirror.propagate()

        # Same passive-departure baseline as intersection_sim.tick().
        for node in self.mirror.nodes.values():
            node.vehicle_count = max(0, int(node.vehicle_count * 0.80))

        for node in self.mirror.nodes.values():
            node.record()

        green_times = strategy_fn(self.mirror)
        self._apply_strategy_discharge(green_times)
        self.mirror.tick_count += 1

        counts_now = [n.vehicle_count for n in self.mirror.nodes.values()]
        return {
            "tick": self.mirror.tick_count,
            "green_times": green_times,
            "avg_vehicle_count": round(sum(counts_now) / len(counts_now), 2),
            "max_vehicle_count": max(counts_now),
            "propagation": propagation,
        }

    # ── Future evolution ─────────────────────────────────────────────────

    def simulate_future(self, n_ticks: int = None, external_counts: list = None) -> list:
        """
        Roll the twin forward n_ticks using the live network's own
        strategy (green-wave), to answer "what will traffic look like
        soon if nothing changes?" Delegates directly to
        IntersectionGrid.tick() (now that graph-intelligence side effects
        are detached, see sync()), so this stays exactly in sync with
        whatever Member 1/2/4 change about the base tick() logic.

        Args:
            n_ticks: How many ticks to simulate ahead (default: config.TWIN_DEFAULT_HORIZON).
            external_counts: Optional list of {node_id: count} dicts, one
                per tick, representing expected/observed new arrivals.

        Returns:
            list of per-tick result dicts, as returned by IntersectionGrid.tick().
        """
        n_ticks = n_ticks if n_ticks is not None else _cfg.TWIN_DEFAULT_HORIZON
        trajectory = []
        for t in range(n_ticks):
            counts = external_counts[t] if external_counts and t < len(external_counts) else None
            trajectory.append(self.mirror.tick(counts))
        return trajectory

    # ── Alternative-strategy evaluation ──────────────────────────────────

    def run_scenario(self, strategy_name: str, n_ticks: int = None, external_counts: list = None) -> list:
        """
        Reset the twin to the current live state, then simulate n_ticks
        forward using a *named* alternative signal strategy instead of
        whatever the live grid would normally do.

        Returns:
            list of per-tick dicts: {tick, green_times, avg_vehicle_count,
            max_vehicle_count, propagation}.
        """
        if strategy_name not in STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy_name}'. Options: {list(STRATEGIES)}")

        n_ticks = n_ticks if n_ticks is not None else _cfg.TWIN_DEFAULT_HORIZON
        strategy_fn = STRATEGIES[strategy_name]

        self.sync()
        trajectory = []

        for t in range(n_ticks):
            counts = external_counts[t] if external_counts and t < len(external_counts) else None
            if counts:
                self.mirror.set_counts(counts)
            trajectory.append(self._step_with_strategy(strategy_fn))

        return trajectory

    def compare_strategies(self, strategy_names: list = None, n_ticks: int = None,
                            external_counts: list = None) -> dict:
        """
        Run several alternative strategies from the *same* current state
        and compare their outcomes, so the best one can be chosen before
        touching the live network.

        Returns:
            dict: {strategy_name: {trajectory, avg_vehicle_count_overall,
            final_avg_vehicle_count, peak_vehicle_count}}
        """
        strategy_names = strategy_names if strategy_names is not None else list(STRATEGIES)
        results = {}
        for name in strategy_names:
            trajectory = self.run_scenario(name, n_ticks=n_ticks, external_counts=external_counts)
            avg_overall = sum(t["avg_vehicle_count"] for t in trajectory) / len(trajectory)
            results[name] = {
                "trajectory": trajectory,
                "avg_vehicle_count_overall": round(avg_overall, 2),
                "final_avg_vehicle_count": trajectory[-1]["avg_vehicle_count"],
                "peak_vehicle_count": max(t["max_vehicle_count"] for t in trajectory),
            }
        return results

    # ── Resilience: disturbance + recovery time ──────────────────────────

    def apply_disturbance(self, node_id: int, extra_vehicles: int = None) -> None:
        """
        Simulate a sudden traffic surge (or comparable shock) at one
        intersection inside the twin only.
        """
        extra_vehicles = extra_vehicles if extra_vehicles is not None else _cfg.DISTURBANCE_DEFAULT_VEHICLES
        self.mirror.nodes[node_id].vehicle_count += extra_vehicles

    def predict_recovery_time(self, node_id: int, strategy_name: str = "green_wave",
                               max_ticks: int = None) -> dict:
        """
        After a disturbance has been applied (see apply_disturbance),
        keep ticking the twin forward under the given strategy until the
        disturbed node drops out of the HIGH congestion band, or max_ticks
        is reached.

        Returns:
            dict: {
                "recovered": bool,
                "ticks_to_recover": int or None,
                "band_trajectory": [band_after_each_tick, ...],
            }
        """
        max_ticks = max_ticks if max_ticks is not None else _cfg.RECOVERY_MAX_TICKS
        strategy_fn = STRATEGIES[strategy_name]

        band_trajectory = []
        for t in range(1, max_ticks + 1):
            self._step_with_strategy(strategy_fn)
            band = self.mirror.nodes[node_id].band
            band_trajectory.append(band)

            if band != "HIGH":
                return {"recovered": True, "ticks_to_recover": t, "band_trajectory": band_trajectory}

        return {"recovered": False, "ticks_to_recover": None, "band_trajectory": band_trajectory}

    def run_resilience_scenario(self, node_id: int, extra_vehicles: int = None,
                                 strategy_name: str = "green_wave", max_ticks: int = None) -> dict:
        """
        Convenience wrapper: reset the twin to "now", apply a disturbance
        at node_id, then measure how long recovery takes under a given
        strategy. Used by Member 4's Network Resilience Analysis (Task 7).
        """
        self.sync()
        self.apply_disturbance(node_id, extra_vehicles)
        return self.predict_recovery_time(node_id, strategy_name=strategy_name, max_ticks=max_ticks)


if __name__ == "__main__":
    # Quick manual smoke test.
    live = IntersectionGrid(n=4)
    live.tick({0: 12, 1: 3, 2: 5, 3: 2})
    live.tick({0: 14, 1: 6})
    live.tick({0: 9, 1: 8, 2: 7})

    twin = DigitalTwin(live)

    print("=== Future evolution (green-wave, 3 ticks) ===")
    for step in twin.simulate_future(n_ticks=3):
        print(step["tick"], step["bands"], step["green_times"])

    print("\n=== Strategy comparison (5 ticks) ===")
    comparison = twin.compare_strategies(n_ticks=5)
    for name, result in comparison.items():
        print(f"{name:15s} avg={result['avg_vehicle_count_overall']:.2f} "
              f"peak={result['peak_vehicle_count']}")

    print("\n=== Resilience: surge at node 0 ===")
    recovery = twin.run_resilience_scenario(node_id=0, extra_vehicles=30)
    print(recovery)
