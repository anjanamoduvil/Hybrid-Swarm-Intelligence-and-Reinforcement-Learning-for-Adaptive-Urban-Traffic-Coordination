"""
federated_learning.py — Federated Learning Prototype
Member 3 (Week 4): Hybrid Swarm Intelligence & RL for Adaptive Urban Traffic Coordination

Simulates every intersection in an IntersectionGrid training its own local
traffic model on its own history, sharing ONLY model parameters (never raw
traffic data) with a central aggregator, and receiving an updated global
model back each round.

Integrates with Member 3's own Week 3 module:
    intersection_sim.IntersectionGrid — supplies per-node vehicle-count
    history (node.history) that each LocalClient trains on.

Local model: a simple 1-D weighted linear model (count vs. time index),
mirroring the lightweight, dependency-light style already used in
prediction.py. Only two numbers per client are ever transmitted each
round — coef and intercept — which keeps the federated protocol easy to
reason about and easy to measure communication overhead for.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

import config as _cfg

# Each transmitted message is a Python float64 pair (coef, intercept).
BYTES_PER_FLOAT = 8
PARAMS_PER_MESSAGE = 2  # coef + intercept


class LocalClient:
    """
    One intersection's local federated-learning participant.

    Trains a local linear model on its own vehicle-count history and
    exposes only the fitted parameters (coef, intercept, n_samples) —
    never the underlying history — to the federated server.
    """

    def __init__(self, node_id: int, history: list):
        self.node_id = node_id
        self.history = list(history)
        self.model = LinearRegression()
        self.n_samples = 0
        self._fitted = False

    def train_local(self) -> dict:
        """
        Fit the local model on this client's own history (count vs. time
        index). Falls back to a flat model when there isn't enough local
        data yet, so clients with sparse history can still participate.

        Returns:
            dict: {"coef", "intercept", "n_samples"} — the only
            information ever leaves this client.
        """
        y = np.asarray(self.history, dtype=float)
        n = len(y)
        self.n_samples = n

        if n < _cfg.FED_MIN_LOCAL_POINTS:
            coef = 0.0
            intercept = float(y[-1]) if n else 0.0
            self.model.coef_ = np.array([coef])
            self.model.intercept_ = intercept
            self._fitted = True
            return self.get_weights()

        X = np.arange(n).reshape(-1, 1)
        self.model.fit(X, y)
        self._fitted = True
        return self.get_weights()

    def get_weights(self) -> dict:
        """Return this client's current local model parameters only."""
        return {
            "coef": float(self.model.coef_[0]),
            "intercept": float(self.model.intercept_),
            "n_samples": self.n_samples,
        }

    def set_weights(self, weights: dict) -> None:
        """Load a (global) parameter set into this client's local model."""
        self.model.coef_ = np.array([weights["coef"]])
        self.model.intercept_ = weights["intercept"]
        self._fitted = True

    def evaluate(self, weights: dict = None) -> float:
        """
        Score a set of weights (local or global) against this client's own
        history using R^2. Defaults to the client's current local weights.

        Returns:
            float: R^2 in [0.0, 1.0] (clamped; 1.0 for degenerate/flat
            histories with zero variance).
        """
        y = np.asarray(self.history, dtype=float)
        n = len(y)
        if n < 2:
            return 1.0

        if weights is not None:
            coef, intercept = weights["coef"], weights["intercept"]
        elif self._fitted:
            coef, intercept = float(self.model.coef_[0]), float(self.model.intercept_)
        else:
            return 0.0

        X = np.arange(n, dtype=float)
        preds = coef * X + intercept
        ss_res = float(np.sum((y - preds) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))

        if ss_tot == 0.0:
            return 1.0
        r2 = 1.0 - ss_res / ss_tot
        return float(max(0.0, min(1.0, r2)))


class FederatedServer:
    """
    Central aggregator. Performs FedAvg — a sample-weighted average of
    every client's local parameters — and tracks aggregation history so
    convergence can be measured.
    """

    def __init__(self):
        self.global_weights = {"coef": 0.0, "intercept": 0.0}
        self.round_history: list[dict] = []

    def aggregate(self, client_weights_list: list) -> dict:
        """
        FedAvg: weight each client's (coef, intercept) by how many local
        samples it was trained on, so intersections with richer history
        influence the global model more.
        """
        total_samples = sum(w["n_samples"] for w in client_weights_list) or 1

        coef = sum(w["coef"] * w["n_samples"] for w in client_weights_list) / total_samples
        intercept = sum(w["intercept"] * w["n_samples"] for w in client_weights_list) / total_samples

        self.global_weights = {"coef": coef, "intercept": intercept}
        self.round_history.append(dict(self.global_weights))
        return self.global_weights

    def has_converged(self, tol: float = None) -> bool:
        """True once consecutive rounds' global weights stop moving by more than tol."""
        tol = tol if tol is not None else _cfg.FED_CONVERGENCE_TOL
        if len(self.round_history) < 2:
            return False
        prev, cur = self.round_history[-2], self.round_history[-1]
        delta = abs(cur["coef"] - prev["coef"]) + abs(cur["intercept"] - prev["intercept"])
        return delta < tol


def communication_cost_bytes(n_clients: int, n_rounds: int) -> int:
    """
    Communication overhead for n_rounds of federated training: each round
    every client uploads one (coef, intercept) pair and downloads one
    aggregated (coef, intercept) pair back.

    Returns:
        int: total bytes transmitted across the whole run (both directions).
    """
    per_round_per_client = 2 * PARAMS_PER_MESSAGE * BYTES_PER_FLOAT  # upload + download
    return n_clients * n_rounds * per_round_per_client


def simulate_federated_learning(node_histories: dict, rounds: int = None, tol: float = None) -> dict:
    """
    Run a full federated-learning simulation across a set of intersections.

    Args:
        node_histories: {node_id: [vehicle_count, ...]} — typically taken
            directly from IntersectionGrid.nodes[node_id].history, i.e.
            each intersection's own local data. Raw histories never leave
            this function; only derived (coef, intercept) pairs are
            "transmitted" between LocalClient and FederatedServer.
        rounds: Number of federated rounds to run (default: config.FED_ROUNDS).
        tol: Convergence tolerance (default: config.FED_CONVERGENCE_TOL).

    Returns:
        dict: {
            "rounds_run": int,
            "converged_round": int or None,
            "local_accuracy": {node_id: R^2 of that node's own local model},
            "global_accuracy_by_round": [avg R^2 of global model across all
                clients, one entry per round],
            "global_weights_final": {"coef", "intercept"},
            "communication_overhead_bytes": int,
            "communication_overhead_per_round_bytes": int,
        }
    """
    rounds = rounds if rounds is not None else _cfg.FED_ROUNDS
    tol = tol if tol is not None else _cfg.FED_CONVERGENCE_TOL

    clients = [LocalClient(nid, hist) for nid, hist in node_histories.items()]
    server = FederatedServer()

    global_accuracy_by_round = []
    converged_round = None

    for round_idx in range(1, rounds + 1):
        # 1. Each intersection trains its own local model on its own data.
        client_weights = [client.train_local() for client in clients]

        # 2. Only parameters are sent to the server for aggregation.
        global_weights = server.aggregate(client_weights)

        # 3. The updated global model is sent back to every intersection.
        for client in clients:
            client.set_weights(global_weights)

        # 4. Evaluate the redistributed global model against each client's
        #    own (still-local) data.
        round_scores = [client.evaluate(global_weights) for client in clients]
        global_accuracy_by_round.append(round(float(np.mean(round_scores)), 4))

        if converged_round is None and server.has_converged(tol):
            converged_round = round_idx

    local_accuracy = {}
    for client in clients:
        client.train_local()  # re-fit purely local model for reporting
        local_accuracy[client.node_id] = round(client.evaluate(), 4)
        client.set_weights(server.global_weights)  # leave client holding global model

    per_round_bytes = communication_cost_bytes(len(clients), 1)
    total_bytes = communication_cost_bytes(len(clients), rounds)

    return {
        "rounds_run": rounds,
        "converged_round": converged_round,
        "local_accuracy": local_accuracy,
        "global_accuracy_by_round": global_accuracy_by_round,
        "global_weights_final": dict(server.global_weights),
        "communication_overhead_bytes": total_bytes,
        "communication_overhead_per_round_bytes": per_round_bytes,
    }


if __name__ == "__main__":
    # Quick manual smoke test using synthetic per-intersection histories.
    sample_histories = {
        0: [2, 3, 4, 6, 7, 9, 10, 12],
        1: [5, 5, 6, 5, 6, 6, 7, 6],
        2: [1, 1, 2, 2, 3, 3, 4, 4],
        3: [10, 9, 8, 7, 6, 5, 4, 3],
    }

    report = simulate_federated_learning(sample_histories)

    print("Rounds run:               ", report["rounds_run"])
    print("Converged at round:       ", report["converged_round"])
    print("Local accuracy per node:  ", report["local_accuracy"])
    print("Global accuracy by round: ", report["global_accuracy_by_round"])
    print("Final global weights:     ", report["global_weights_final"])
    print("Total comm overhead (B):  ", report["communication_overhead_bytes"])
    print("Per-round comm (B):       ", report["communication_overhead_per_round_bytes"])
