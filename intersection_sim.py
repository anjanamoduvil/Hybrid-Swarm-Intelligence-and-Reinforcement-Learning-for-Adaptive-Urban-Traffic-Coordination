"""
intersection_sim.py — Multi-Intersection Coordination
Member 3: Hybrid Swarm Intelligence & RL for Adaptive Urban Traffic Coordination

Simulates 3–4 connected intersections, propagates congestion between
neighbouring intersections, and produces a coordinated green-wave
allocation order. Integrates with:

    Member 1 (prediction.py) — predict(csv_path, n_steps, lane) ->
        (congestion, queue_length, trend, confidence)   [positional tuple]
    Member 2 (rl_agent.py)   — RLAgent.step(state) -> green_time_delta

Each intersection writes its own per-node history to MULTI_LOG_PATH using
a 'lane1_count' column so it can be fed directly into Member 1's predict()
without any changes to prediction.py. Both integrations are optional at
import-time: if either module is not yet available, a local fallback is
used so this file can be developed and tested independently.
"""

import csv
import os
from datetime import datetime

import config as _cfg
from density import compute_density, classify_density

# ── Optional integration with Member 1 / Member 2 ────────────────────────────
try:
    from prediction import predict as _predict
except ImportError:
    _predict = None

try:
    from rl_agent import RLAgent
except ImportError:
    RLAgent = None

# ── Week 4 Member 1 Integrations ──────────────────────────────────────────────
try:
    from traffic_graph import DynamicTrafficGraph
    from graph_intelligence import GraphIntelligenceModule
except ImportError:
    DynamicTrafficGraph = None
    GraphIntelligenceModule = None


class Intersection:
    """
    A single intersection node in the coordination grid.

    Attributes:
        node_id:        Integer index of this intersection within the grid.
        vehicle_count:   Current raw vehicle count waiting at this intersection.
        neighbours:      List of node_ids this intersection can propagate to.
        history:         List of past vehicle_count samples, oldest first.
    """

    def __init__(self, node_id: int, vehicle_count: int = 0):
        self.node_id = node_id
        self.vehicle_count = vehicle_count
        self.neighbours = []
        self.history = []

    def record(self) -> None:
        """Append the current vehicle_count to this intersection's history."""
        self.history.append(self.vehicle_count)

    @property
    def density(self) -> float:
        """Normalised density score (0.0 – 1.0) for this intersection."""
        return compute_density(self.vehicle_count)

    @property
    def band(self) -> str:
        """Density band classification: LOW / MED / HIGH."""
        return classify_density(self.vehicle_count)


class IntersectionGrid:
    """
    Coordinates a grid of 3–4 connected intersections, propagating
    congestion between neighbours and producing a green-wave priority
    order each simulation tick.

    Args:
        n: Number of intersections to simulate (3 or 4). Defaults to
           config.N_INTERSECTIONS.
        topology: Optional adjacency dict {node_id: [neighbour_ids]}.
                  If omitted, intersections are connected in a simple
                  line: 0 - 1 - 2 - 3 ...
        log_path: Path used for this grid's per-node CSV logs, one file
                  per node (e.g. "intersection_grid_log_node0.csv").
                  Defaults to config.MULTI_LOG_PATH as a prefix.
    """

    def __init__(self, n: int = None, topology: dict = None, log_path: str = None):
        self.n = n if n is not None else _cfg.N_INTERSECTIONS
        if self.n < 3 or self.n > 4:
            raise ValueError("IntersectionGrid supports 3 or 4 intersections")

        self.nodes = {i: Intersection(i) for i in range(self.n)}
        self.topology = topology if topology is not None else self._default_topology()
        for node_id, neighbour_ids in self.topology.items():
            self.nodes[node_id].neighbours = neighbour_ids

        self.log_path_prefix = log_path if log_path is not None else _cfg.MULTI_LOG_PATH
        self.tick_count = 0

        # Week 4 Member 1 integrations
        if DynamicTrafficGraph is not None:
            self.traffic_graph = DynamicTrafficGraph(num_nodes=self.n, topology=self.topology)
            self.graph_intelligence = GraphIntelligenceModule(num_nodes=self.n)
        else:
            self.traffic_graph = None
            self.graph_intelligence = None

    def _default_topology(self) -> dict:
        """Build a simple line topology: each node connects to its immediate neighbour(s)."""
        topo = {}
        for i in range(self.n):
            neighbours = []
            if i > 0:
                neighbours.append(i - 1)
            if i < self.n - 1:
                neighbours.append(i + 1)
            topo[i] = neighbours
        return topo

    def _node_log_path(self, node_id: int) -> str:
        """Per-node CSV log path, derived from the grid's log_path_prefix."""
        root, ext = os.path.splitext(self.log_path_prefix)
        return f"{root}_node{node_id}{ext}"

    # ── Vehicle count input ──────────────────────────────────────────────

    def set_counts(self, counts: dict) -> None:
        """
        Set raw vehicle counts for one or more intersections.

        Args:
            counts: {node_id: vehicle_count}
        """
        for node_id, count in counts.items():
            if node_id not in self.nodes:
                raise KeyError(f"Unknown intersection node_id: {node_id}")
            self.nodes[node_id].vehicle_count = max(0, count)

    # ── Congestion propagation ───────────────────────────────────────────

    def propagate(self) -> dict:
        """
        Propagate congestion overflow from HIGH-band intersections to their
        neighbours. A HIGH intersection spills PROPAGATION_RATE of its
        vehicle count to each neighbour, split evenly.

        Returns:
            dict: {node_id: delta_applied} — additional vehicles each node
                  received this tick (for logging/inspection).
        """
        rate = _cfg.PROPAGATION_RATE
        deltas = {i: 0.0 for i in self.nodes}

        for node_id, node in self.nodes.items():
            if node.band != "HIGH" or not node.neighbours:
                continue

            overflow = node.vehicle_count * rate
            share = overflow / len(node.neighbours)
            for neighbour_id in node.neighbours:
                deltas[neighbour_id] += share

        for node_id, delta in deltas.items():
            if delta > 0:
                self.nodes[node_id].vehicle_count += int(round(delta))

        return {node_id: round(delta, 2) for node_id, delta in deltas.items()}

    # ── Coordinated green-wave allocation ────────────────────────────────

    def green_wave_order(self) -> list:
        """
        Rank intersections by descending congestion load to decide which
        should receive priority (longer/earlier) green allocation.

        Returns:
            list of node_ids ordered from highest to lowest priority.
        """
        return sorted(
            self.nodes.keys(),
            key=lambda nid: self.nodes[nid].vehicle_count,
            reverse=True,
        )

    def allocate_green_times(self) -> dict:
        """
        Produce a coordinated green-time allocation for every intersection
        this tick, combining:
          - Member 2's RL agent (if available) to adjust green duration
          - Green-wave priority order as a tie-breaker / head-start bonus

        Returns:
            dict: {node_id: green_time_seconds}
        """
        priority_order = self.green_wave_order()
        allocations = {}

        for rank, node_id in enumerate(priority_order):
            node = self.nodes[node_id]
            base_green = _cfg.MIN_GREEN + (
                (_cfg.MAX_GREEN - _cfg.MIN_GREEN) * node.density
            )

            # Member 2 RL adjustment (optional — falls back to 0 if not available)
            if RLAgent is not None:
                agent = RLAgent()
                state = (node.band, len(node.history))
                base_green += agent.step(state)

            # Priority bonus: earlier rank gets a small head start in
            # the coordinated green-wave sequence
            priority_bonus = max(0, (len(priority_order) - rank - 1)) * 1.0
            green_time = base_green + priority_bonus

            green_time = max(_cfg.MIN_GREEN, min(_cfg.MAX_GREEN, green_time))
            allocations[node_id] = round(green_time, 1)

        return allocations

    # ── Prediction integration ───────────────────────────────────────────

    def forecast(self, node_id: int, n_steps: int = 3) -> dict:
        """
        Use Member 1's prediction module to forecast near-future congestion
        for a given intersection, falling back to a simple trend estimate
        from local history if prediction.py is not available or the node's
        log file does not have enough rows yet.

        Calls predict() exactly the way app.py does — as a positional
        tuple (congestion, queue_length, trend, confidence) — and wraps it
        into a dict for convenience within this module.

        Args:
            node_id: Intersection to forecast for.
            n_steps: How many ticks ahead to forecast.

        Returns:
            dict: {congestion, queue_length, trend, confidence}
        """
        node = self.nodes[node_id]
        node_log_path = self._node_log_path(node_id)

        if _predict is not None and os.path.isfile(node_log_path):
            congestion, queue_length, trend, confidence = _predict(
                node_log_path, n_steps=n_steps, lane=1
            )
            return {
                "congestion": congestion,
                "queue_length": queue_length,
                "trend": trend,
                "confidence": confidence,
            }

        # Local fallback: simple trend based on last two history samples
        history = node.history
        if len(history) < 2:
            trend = "STABLE"
        elif history[-1] > history[-2]:
            trend = "RISING"
        elif history[-1] < history[-2]:
            trend = "FALLING"
        else:
            trend = "STABLE"

        return {
            "congestion": node.density,
            "queue_length": node.vehicle_count,
            "trend": trend,
            "confidence": 0.5,  # lower confidence since this is a fallback
        }

    # ── Simulation tick ───────────────────────────────────────────────────

    def tick(self, counts: dict = None) -> dict:
        """
        Advance the simulation by one tick:
          1. Apply new vehicle counts (if provided)
          2. Propagate congestion between neighbours
          3. Record history for every node + append to its CSV log
          4. Compute the coordinated green-time allocation

        Args:
            counts: Optional new raw vehicle counts to apply before this tick.

        Returns:
            dict summary: {
                'tick': int,
                'propagation': {node_id: delta},
                'green_times': {node_id: seconds},
                'priority_order': [node_id, ...],
                'bands': {node_id: band},
            }
        """
        if counts:
            self.set_counts(counts)

        propagation = self.propagate()

        # Simulate vehicles departing (clear simulated queues so they don't grow infinitely)
        for node_id, node in self.nodes.items():
            if not counts or node_id not in counts:
                # Remove ~20% of vehicles per tick to simulate green light flow
                node.vehicle_count = max(0, int(node.vehicle_count * 0.80))

        for node_id, node in self.nodes.items():
            node.record()
            self._log_node(node_id, node)

        green_times = self.allocate_green_times()
        priority_order = self.green_wave_order()
        bands = {nid: node.band for nid, node in self.nodes.items()}

        self.tick_count += 1
        
        # ── Week 4 Member 1: Update Traffic Graph and run Intelligence ───────
        graph_preds = {}
        node_importance = []
        if self.traffic_graph is not None:
            for node_id, node in self.nodes.items():
                self.traffic_graph.update_node(
                    node_id=node_id,
                    density=node.density,
                    queue_length=node.vehicle_count * 0.8,
                    waiting_time=node.vehicle_count * 5.0,
                    average_speed=10.0 if node.band == "HIGH" else 30.0,
                    signal_phase=1.0 if node_id in green_times else 0.0
                )
                for neighbor_id in node.neighbours:
                    flow = (node.vehicle_count * _cfg.PROPAGATION_RATE) if node.band == "HIGH" else 0.0
                    self.traffic_graph.update_edge(
                        u=node_id, v=neighbor_id,
                        flow=flow,
                        travel_time=10.0,
                        propagation_rate=_cfg.PROPAGATION_RATE,
                        capacity=100.0
                    )
            
            try:
                self.traffic_graph.save_visualization("static/traffic_graph.png")
            except Exception as e:
                print(f"[Graph] Visual save failed: {e}")
                
            graph_snapshot = self.traffic_graph.get_snapshot()
            if self.graph_intelligence:
                target_congestion = [n.density for nid, n in self.nodes.items()]
                self.graph_intelligence.train_step(graph_snapshot, target_congestion)
                preds = self.graph_intelligence.predict_congestion(graph_snapshot)
                graph_preds = {nid: float(preds[nid]) for nid in self.nodes.keys()}
                node_importance = self.graph_intelligence.get_node_importance(graph_snapshot).tolist()

        return {
            "tick": self.tick_count,
            "propagation": propagation,
            "green_times": green_times,
            "priority_order": priority_order,
            "bands": bands,
            "graph_predictions": graph_preds,
            "node_importance": node_importance
        }

    def _log_node(self, node_id: int, node: "Intersection") -> None:
        """
        Append one row to this node's per-intersection CSV log, using a
        'lane1_count' column so Member 1's predict() can read it directly
        without modification.
        """
        path = self._node_log_path(node_id)
        file_exists = os.path.isfile(path)

        with open(path, mode="a", newline="") as csvfile:
            fieldnames = ["timestamp", "tick", "lane1_count", "band"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "tick": self.tick_count,
                    "lane1_count": node.vehicle_count,
                    "band": node.band,
                }
            )


def log_grid_summary_to_csv(tick_result: dict, grid: "IntersectionGrid", path: str = None) -> None:
    """
    Append one summary row per intersection for this tick to a combined
    grid-level CSV log (separate from the per-node logs used for forecasting).
    Creates the file with a header row if it does not exist yet.

    Args:
        tick_result: The dict returned by IntersectionGrid.tick().
        grid:        The IntersectionGrid instance (for vehicle counts).
        path:        Optional override of the log path (defaults to
                     config.MULTI_LOG_PATH so tests can override it).
    """
    path = path if path is not None else _cfg.MULTI_LOG_PATH
    file_exists = os.path.isfile(path)

    with open(path, mode="a", newline="") as csvfile:
        fieldnames = [
            "timestamp",
            "tick",
            "node_id",
            "vehicle_count",
            "band",
            "green_time",
            "priority_rank",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        priority_order = tick_result["priority_order"]
        timestamp = datetime.now().isoformat(timespec="seconds")

        for node_id, node in grid.nodes.items():
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "tick": tick_result["tick"],
                    "node_id": node_id,
                    "vehicle_count": node.vehicle_count,
                    "band": tick_result["bands"][node_id],
                    "green_time": tick_result["green_times"][node_id],
                    "priority_rank": priority_order.index(node_id),
                }
            )


if __name__ == "__main__":
    # Quick manual smoke test: simulate 5 ticks on a 4-intersection grid
    grid = IntersectionGrid(n=4)
    sample_counts = [
        {0: 12, 1: 3, 2: 5, 3: 2},
        {0: 14, 1: 6},
        {0: 9, 1: 8, 2: 7},
        {2: 13, 3: 4},
        {0: 4, 1: 4, 2: 4, 3: 4},
    ]

    for counts in sample_counts:
        result = grid.tick(counts)
        log_grid_summary_to_csv(result, grid)
        print(
            f"Tick {result['tick']}: bands={result['bands']} "
            f"green_times={result['green_times']} "
            f"priority={result['priority_order']}"
        )

    print("\nForecast for node 0:", grid.forecast(0))
