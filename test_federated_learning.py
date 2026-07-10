"""
test_federated_learning.py — Unit Tests for Member 3 (Federated Learning Prototype)
Run with:  python -m pytest test_federated_learning.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from federated_learning import (
    LocalClient,
    FederatedServer,
    communication_cost_bytes,
    simulate_federated_learning,
)
import config as _cfg


# ────────────────────────────────────────────────────────────────────────────
# LocalClient tests
# ────────────────────────────────────────────────────────────────────────────

class TestLocalClient:

    def test_train_local_returns_expected_keys(self):
        client = LocalClient(0, [2, 4, 6, 8])
        weights = client.train_local()
        assert set(weights.keys()) == {"coef", "intercept", "n_samples"}

    def test_train_local_fits_rising_trend(self):
        client = LocalClient(0, [1, 2, 3, 4, 5])
        weights = client.train_local()
        assert weights["coef"] > 0

    def test_train_local_fits_falling_trend(self):
        client = LocalClient(0, [10, 8, 6, 4, 2])
        weights = client.train_local()
        assert weights["coef"] < 0

    def test_insufficient_history_falls_back_to_flat_model(self):
        client = LocalClient(0, [5])
        weights = client.train_local()
        assert weights["coef"] == 0.0
        assert weights["intercept"] == 5.0

    def test_empty_history_does_not_crash(self):
        client = LocalClient(0, [])
        weights = client.train_local()
        assert weights["n_samples"] == 0

    def test_get_set_weights_roundtrip(self):
        client = LocalClient(0, [1, 2, 3, 4])
        client.train_local()
        weights = client.get_weights()
        client2 = LocalClient(1, [9, 9, 9])
        client2.set_weights(weights)
        assert client2.get_weights()["coef"] == pytest.approx(weights["coef"])
        assert client2.get_weights()["intercept"] == pytest.approx(weights["intercept"])

    def test_evaluate_perfect_fit_is_high_r2(self):
        client = LocalClient(0, [1, 2, 3, 4, 5])
        client.train_local()
        assert client.evaluate() > 0.99

    def test_evaluate_flat_history_returns_one(self):
        client = LocalClient(0, [4, 4, 4, 4])
        client.train_local()
        assert client.evaluate() == 1.0

    def test_evaluate_with_explicit_weights(self):
        client = LocalClient(0, [1, 2, 3, 4])
        score = client.evaluate({"coef": 1.0, "intercept": 1.0})
        assert 0.0 <= score <= 1.0

    def test_evaluate_unfitted_client_without_weights_returns_zero(self):
        client = LocalClient(0, [1, 2, 3])
        assert client.evaluate() == 0.0


# ────────────────────────────────────────────────────────────────────────────
# FederatedServer tests
# ────────────────────────────────────────────────────────────────────────────

class TestFederatedServer:

    def test_aggregate_returns_weighted_average(self):
        server = FederatedServer()
        weights = [
            {"coef": 1.0, "intercept": 0.0, "n_samples": 10},
            {"coef": 3.0, "intercept": 4.0, "n_samples": 10},
        ]
        result = server.aggregate(weights)
        assert result["coef"] == pytest.approx(2.0)
        assert result["intercept"] == pytest.approx(2.0)

    def test_aggregate_weights_by_sample_count(self):
        server = FederatedServer()
        weights = [
            {"coef": 0.0, "intercept": 0.0, "n_samples": 90},
            {"coef": 10.0, "intercept": 0.0, "n_samples": 10},
        ]
        result = server.aggregate(weights)
        assert result["coef"] == pytest.approx(1.0)

    def test_aggregate_handles_zero_samples_without_crashing(self):
        server = FederatedServer()
        weights = [{"coef": 5.0, "intercept": 5.0, "n_samples": 0}]
        result = server.aggregate(weights)
        # A client with zero samples carries zero weight in FedAvg, so it
        # contributes nothing to the weighted sum; the important
        # guarantee here is that division-by-zero never happens.
        assert result["coef"] == pytest.approx(0.0)

    def test_has_converged_false_before_two_rounds(self):
        server = FederatedServer()
        server.aggregate([{"coef": 1.0, "intercept": 1.0, "n_samples": 5}])
        assert server.has_converged() is False

    def test_has_converged_true_when_stable(self):
        server = FederatedServer()
        stable = [{"coef": 2.0, "intercept": 2.0, "n_samples": 5}]
        server.aggregate(stable)
        server.aggregate(stable)
        assert server.has_converged() is True

    def test_has_converged_false_when_still_moving(self):
        server = FederatedServer()
        server.aggregate([{"coef": 1.0, "intercept": 1.0, "n_samples": 5}])
        server.aggregate([{"coef": 100.0, "intercept": 100.0, "n_samples": 5}])
        assert server.has_converged() is False


# ────────────────────────────────────────────────────────────────────────────
# Communication overhead tests
# ────────────────────────────────────────────────────────────────────────────

class TestCommunicationCost:

    def test_cost_scales_with_clients_and_rounds(self):
        cost_1x1 = communication_cost_bytes(1, 1)
        cost_2x1 = communication_cost_bytes(2, 1)
        cost_1x2 = communication_cost_bytes(1, 2)
        assert cost_2x1 == cost_1x1 * 2
        assert cost_1x2 == cost_1x1 * 2

    def test_cost_is_positive_integer(self):
        cost = communication_cost_bytes(4, 8)
        assert isinstance(cost, int)
        assert cost > 0


# ────────────────────────────────────────────────────────────────────────────
# Full simulation integration tests
# ────────────────────────────────────────────────────────────────────────────

class TestSimulateFederatedLearning:

    HISTORIES = {
        0: [2, 3, 4, 6, 7, 9, 10, 12],
        1: [5, 5, 6, 5, 6, 6, 7, 6],
        2: [1, 1, 2, 2, 3, 3, 4, 4],
        3: [10, 9, 8, 7, 6, 5, 4, 3],
    }

    def test_returns_expected_keys(self):
        report = simulate_federated_learning(self.HISTORIES, rounds=5)
        assert set(report.keys()) == {
            "rounds_run", "converged_round", "local_accuracy",
            "global_accuracy_by_round", "global_weights_final",
            "communication_overhead_bytes", "communication_overhead_per_round_bytes",
        }

    def test_rounds_run_matches_argument(self):
        report = simulate_federated_learning(self.HISTORIES, rounds=6)
        assert report["rounds_run"] == 6
        assert len(report["global_accuracy_by_round"]) == 6

    def test_local_accuracy_covers_every_node(self):
        report = simulate_federated_learning(self.HISTORIES, rounds=4)
        assert set(report["local_accuracy"].keys()) == set(self.HISTORIES.keys())

    def test_local_accuracy_in_valid_range(self):
        report = simulate_federated_learning(self.HISTORIES, rounds=4)
        for score in report["local_accuracy"].values():
            assert 0.0 <= score <= 1.0

    def test_global_accuracy_in_valid_range(self):
        report = simulate_federated_learning(self.HISTORIES, rounds=4)
        for score in report["global_accuracy_by_round"]:
            assert 0.0 <= score <= 1.0

    def test_default_rounds_uses_config(self):
        report = simulate_federated_learning(self.HISTORIES)
        assert report["rounds_run"] == _cfg.FED_ROUNDS

    def test_identical_clients_converge_quickly(self):
        flat_histories = {0: [5, 5, 5, 5], 1: [5, 5, 5, 5], 2: [5, 5, 5, 5]}
        report = simulate_federated_learning(flat_histories, rounds=5)
        assert report["converged_round"] is not None
        assert report["converged_round"] <= 2

    def test_communication_overhead_scales_with_rounds_and_clients(self):
        report = simulate_federated_learning(self.HISTORIES, rounds=8)
        expected = report["communication_overhead_per_round_bytes"] * 8
        assert report["communication_overhead_bytes"] == expected

    def test_single_client_still_runs(self):
        report = simulate_federated_learning({0: [1, 2, 3, 4, 5]}, rounds=3)
        assert report["rounds_run"] == 3
        assert 0 in report["local_accuracy"]

    def test_empty_history_client_does_not_crash(self):
        report = simulate_federated_learning({0: [], 1: [1, 2, 3, 4]}, rounds=3)
        assert report["rounds_run"] == 3
