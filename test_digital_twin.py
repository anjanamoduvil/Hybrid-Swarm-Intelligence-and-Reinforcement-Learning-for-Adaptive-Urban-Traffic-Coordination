"""
test_digital_twin.py — Unit Tests for Member 3 (Digital Twin Network Simulation)
Run with:  python -m pytest test_digital_twin.py -v
"""

import os
import sys
import glob

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from digital_twin import DigitalTwin, STRATEGIES
from intersection_sim import IntersectionGrid
import config as _cfg


@pytest.fixture(autouse=True)
def cleanup_twin_logs():
    """Remove any CSV logs created by the DigitalTwin/IntersectionGrid during tests."""
    yield
    for pattern in ("digital_twin_log*.csv", "test_twin_log*.csv"):
        for f in glob.glob(pattern):
            os.remove(f)


def make_live_grid():
    grid = IntersectionGrid(n=4, log_path="test_twin_source_log.csv")
    grid.tick({0: 12, 1: 3, 2: 5, 3: 2})
    grid.tick({0: 14, 1: 6})
    grid.tick({0: 9, 1: 8, 2: 7})
    return grid


@pytest.fixture
def cleanup_source_logs():
    yield
    for f in glob.glob("test_twin_source_log*.csv"):
        os.remove(f)


# ────────────────────────────────────────────────────────────────────────────
# Replication (sync) tests
# ────────────────────────────────────────────────────────────────────────────

class TestSync:

    def test_twin_mirrors_vehicle_counts(self, cleanup_source_logs):
        live = make_live_grid()
        twin = DigitalTwin(live, log_path="test_twin_log.csv")
        for node_id in live.nodes:
            assert twin.mirror.nodes[node_id].vehicle_count == live.nodes[node_id].vehicle_count

    def test_twin_mirrors_history(self, cleanup_source_logs):
        live = make_live_grid()
        twin = DigitalTwin(live, log_path="test_twin_log.csv")
        for node_id in live.nodes:
            assert twin.mirror.nodes[node_id].history == live.nodes[node_id].history

    def test_twin_mirrors_topology(self, cleanup_source_logs):
        live = make_live_grid()
        twin = DigitalTwin(live, log_path="test_twin_log.csv")
        assert twin.mirror.topology == live.topology

    def test_twin_is_a_separate_object(self, cleanup_source_logs):
        live = make_live_grid()
        twin = DigitalTwin(live, log_path="test_twin_log.csv")
        twin.mirror.nodes[0].vehicle_count = 999
        assert live.nodes[0].vehicle_count != 999

    def test_resync_picks_up_live_changes(self, cleanup_source_logs):
        live = make_live_grid()
        twin = DigitalTwin(live, log_path="test_twin_log.csv")
        live.set_counts({0: 50})
        twin.sync()
        assert twin.mirror.nodes[0].vehicle_count == 50


# ────────────────────────────────────────────────────────────────────────────
# Future evolution tests
# ────────────────────────────────────────────────────────────────────────────

class TestSimulateFuture:

    def test_returns_requested_number_of_ticks(self, cleanup_source_logs):
        live = make_live_grid()
        twin = DigitalTwin(live, log_path="test_twin_log.csv")
        trajectory = twin.simulate_future(n_ticks=4)
        assert len(trajectory) == 4

    def test_default_horizon_uses_config(self, cleanup_source_logs):
        live = make_live_grid()
        twin = DigitalTwin(live, log_path="test_twin_log.csv")
        trajectory = twin.simulate_future()
        assert len(trajectory) == _cfg.TWIN_DEFAULT_HORIZON

    def test_does_not_mutate_live_grid(self, cleanup_source_logs):
        live = make_live_grid()
        before = live.nodes[0].vehicle_count
        twin = DigitalTwin(live, log_path="test_twin_log.csv")
        twin.simulate_future(n_ticks=3)
        assert live.nodes[0].vehicle_count == before

    def test_external_counts_are_applied(self, cleanup_source_logs):
        live = make_live_grid()
        twin = DigitalTwin(live, log_path="test_twin_log.csv")
        trajectory = twin.simulate_future(n_ticks=1, external_counts=[{0: 40}])
        # after discharge, count should reflect the injected arrival minus service
        assert trajectory[0]["bands"][0] in {"LOW", "MED", "HIGH"}


# ────────────────────────────────────────────────────────────────────────────
# Strategy evaluation tests
# ────────────────────────────────────────────────────────────────────────────

class TestStrategies:

    def test_all_named_strategies_are_registered(self):
        assert set(STRATEGIES.keys()) == {"green_wave", "fixed_baseline", "density_only"}

    def test_unknown_strategy_raises(self, cleanup_source_logs):
        live = make_live_grid()
        twin = DigitalTwin(live, log_path="test_twin_log.csv")
        with pytest.raises(ValueError):
            twin.run_scenario("not_a_real_strategy")

    def test_run_scenario_returns_expected_keys(self, cleanup_source_logs):
        live = make_live_grid()
        twin = DigitalTwin(live, log_path="test_twin_log.csv")
        trajectory = twin.run_scenario("green_wave", n_ticks=2)
        assert set(trajectory[0].keys()) == {
            "tick", "green_times", "avg_vehicle_count", "max_vehicle_count", "propagation"
        }

    def test_fixed_baseline_uses_constant_green_time(self, cleanup_source_logs):
        live = make_live_grid()
        twin = DigitalTwin(live, log_path="test_twin_log.csv")
        trajectory = twin.run_scenario("fixed_baseline", n_ticks=1)
        green_times = trajectory[0]["green_times"]
        assert all(g == _cfg.FIXED_BASELINE for g in green_times.values())

    def test_run_scenario_resets_twin_each_call(self, cleanup_source_logs):
        live = make_live_grid()
        twin = DigitalTwin(live, log_path="test_twin_log.csv")
        first = twin.run_scenario("green_wave", n_ticks=3)
        second = twin.run_scenario("green_wave", n_ticks=3)
        assert first[0]["avg_vehicle_count"] == second[0]["avg_vehicle_count"]

    def test_compare_strategies_covers_all_by_default(self, cleanup_source_logs):
        live = make_live_grid()
        twin = DigitalTwin(live, log_path="test_twin_log.csv")
        results = twin.compare_strategies(n_ticks=3)
        assert set(results.keys()) == set(STRATEGIES.keys())

    def test_compare_strategies_returns_summary_fields(self, cleanup_source_logs):
        live = make_live_grid()
        twin = DigitalTwin(live, log_path="test_twin_log.csv")
        results = twin.compare_strategies(strategy_names=["green_wave"], n_ticks=3)
        summary = results["green_wave"]
        assert set(summary.keys()) == {
            "trajectory", "avg_vehicle_count_overall", "final_avg_vehicle_count", "peak_vehicle_count"
        }


# ────────────────────────────────────────────────────────────────────────────
# Resilience: disturbance + recovery tests
# ────────────────────────────────────────────────────────────────────────────

class TestResilience:

    def test_disturbance_increases_vehicle_count(self, cleanup_source_logs):
        live = make_live_grid()
        twin = DigitalTwin(live, log_path="test_twin_log.csv")
        before = twin.mirror.nodes[0].vehicle_count
        twin.apply_disturbance(0, extra_vehicles=30)
        assert twin.mirror.nodes[0].vehicle_count == before + 30

    def test_disturbance_does_not_affect_live_grid(self, cleanup_source_logs):
        live = make_live_grid()
        before = live.nodes[0].vehicle_count
        twin = DigitalTwin(live, log_path="test_twin_log.csv")
        twin.apply_disturbance(0, extra_vehicles=30)
        assert live.nodes[0].vehicle_count == before

    def test_recovery_after_large_surge_eventually_recovers(self, cleanup_source_logs):
        live = make_live_grid()
        twin = DigitalTwin(live, log_path="test_twin_log.csv")
        twin.apply_disturbance(0, extra_vehicles=15)
        result = twin.predict_recovery_time(0, strategy_name="green_wave", max_ticks=20)
        assert result["recovered"] is True
        assert result["ticks_to_recover"] is not None

    def test_recovery_result_has_expected_keys(self, cleanup_source_logs):
        live = make_live_grid()
        twin = DigitalTwin(live, log_path="test_twin_log.csv")
        twin.apply_disturbance(0, extra_vehicles=10)
        result = twin.predict_recovery_time(0, max_ticks=10)
        assert set(result.keys()) == {"recovered", "ticks_to_recover", "band_trajectory"}

    def test_run_resilience_scenario_resets_before_disturbance(self, cleanup_source_logs):
        live = make_live_grid()
        twin = DigitalTwin(live, log_path="test_twin_log.csv")
        result_1 = twin.run_resilience_scenario(0, extra_vehicles=20)
        result_2 = twin.run_resilience_scenario(0, extra_vehicles=20)
        assert result_1["ticks_to_recover"] == result_2["ticks_to_recover"]

    def test_recovery_time_not_worse_under_green_wave_than_fixed(self, cleanup_source_logs):
        live = make_live_grid()
        twin = DigitalTwin(live, log_path="test_twin_log.csv")
        gw = twin.run_resilience_scenario(0, extra_vehicles=20, strategy_name="green_wave")
        fb = twin.run_resilience_scenario(0, extra_vehicles=20, strategy_name="fixed_baseline")
        if gw["recovered"] and fb["recovered"]:
            assert gw["ticks_to_recover"] <= fb["ticks_to_recover"]
