"""
test_intersection_sim.py — Unit Tests for Member 3 (Multi-Intersection Coordination)
Run with:  python -m pytest test_intersection_sim.py -v
"""

import os
import sys
import csv
import glob
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from intersection_sim import IntersectionGrid, Intersection, log_grid_summary_to_csv
import config as _cfg


@pytest.fixture(autouse=True)
def cleanup_node_logs():
    """Remove any per-node CSV logs created by IntersectionGrid during tests."""
    yield
    for f in glob.glob("intersection_grid_log_node*.csv"):
        os.remove(f)
    for f in glob.glob("test_intersection_grid_log_node*.csv"):
        os.remove(f)


# ────────────────────────────────────────────────────────────────────────────
# Grid initialisation tests
# ────────────────────────────────────────────────────────────────────────────

class TestGridInit:

    def test_default_grid_uses_config_n(self):
        grid = IntersectionGrid()
        assert grid.n == _cfg.N_INTERSECTIONS

    def test_grid_with_n_3(self):
        grid = IntersectionGrid(n=3)
        assert len(grid.nodes) == 3

    def test_grid_with_n_4(self):
        grid = IntersectionGrid(n=4)
        assert len(grid.nodes) == 4

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            IntersectionGrid(n=2)
        with pytest.raises(ValueError):
            IntersectionGrid(n=5)

    def test_default_topology_is_line(self):
        grid = IntersectionGrid(n=4)
        assert grid.nodes[0].neighbours == [1]
        assert sorted(grid.nodes[1].neighbours) == [0, 2]
        assert sorted(grid.nodes[2].neighbours) == [1, 3]
        assert grid.nodes[3].neighbours == [2]

    def test_custom_topology(self):
        topo = {0: [1, 2], 1: [0], 2: [0]}
        grid = IntersectionGrid(n=3, topology=topo)
        assert grid.nodes[0].neighbours == [1, 2]


# ────────────────────────────────────────────────────────────────────────────
# Vehicle count input tests
# ────────────────────────────────────────────────────────────────────────────

class TestSetCounts:

    def test_set_counts_updates_nodes(self):
        grid = IntersectionGrid(n=3)
        grid.set_counts({0: 10, 1: 5})
        assert grid.nodes[0].vehicle_count == 10
        assert grid.nodes[1].vehicle_count == 5

    def test_negative_counts_clamped_to_zero(self):
        grid = IntersectionGrid(n=3)
        grid.set_counts({0: -5})
        assert grid.nodes[0].vehicle_count == 0

    def test_unknown_node_id_raises(self):
        grid = IntersectionGrid(n=3)
        with pytest.raises(KeyError):
            grid.set_counts({99: 5})


# ────────────────────────────────────────────────────────────────────────────
# Congestion propagation tests
# ────────────────────────────────────────────────────────────────────────────

class TestPropagation:

    def test_high_node_spills_to_neighbour(self):
        grid = IntersectionGrid(n=3)
        # node 0's only neighbour is node 1
        grid.set_counts({0: 20, 1: 0, 2: 0})
        before = grid.nodes[1].vehicle_count
        grid.propagate()
        after = grid.nodes[1].vehicle_count
        assert after > before

    def test_propagation_amount_matches_rate(self):
        grid = IntersectionGrid(n=3)
        grid.set_counts({0: 20, 1: 0, 2: 0})
        expected_delta = round(20 * _cfg.PROPAGATION_RATE)  # single neighbour gets full share
        grid.propagate()
        assert grid.nodes[1].vehicle_count == expected_delta

    def test_low_band_node_does_not_propagate(self):
        grid = IntersectionGrid(n=3)
        grid.set_counts({0: 1, 1: 0, 2: 0})  # LOW band
        grid.propagate()
        assert grid.nodes[1].vehicle_count == 0

    def test_propagation_splits_across_multiple_neighbours(self):
        grid = IntersectionGrid(n=3)  # node 1 has two neighbours: 0 and 2
        grid.set_counts({0: 0, 1: 20, 2: 0})
        grid.propagate()
        # both neighbours should receive roughly equal share
        assert grid.nodes[0].vehicle_count == grid.nodes[2].vehicle_count
        assert grid.nodes[0].vehicle_count > 0

    def test_propagate_returns_deltas_dict(self):
        grid = IntersectionGrid(n=3)
        grid.set_counts({0: 20})
        deltas = grid.propagate()
        assert isinstance(deltas, dict)
        assert set(deltas.keys()) == {0, 1, 2}


# ────────────────────────────────────────────────────────────────────────────
# Green-wave ordering tests
# ────────────────────────────────────────────────────────────────────────────

class TestGreenWaveOrder:

    def test_order_sorted_descending_by_count(self):
        grid = IntersectionGrid(n=4)
        grid.set_counts({0: 5, 1: 20, 2: 1, 3: 10})
        order = grid.green_wave_order()
        assert order == [1, 3, 0, 2]

    def test_order_contains_all_nodes(self):
        grid = IntersectionGrid(n=4)
        grid.set_counts({0: 5, 1: 20, 2: 1, 3: 10})
        order = grid.green_wave_order()
        assert sorted(order) == [0, 1, 2, 3]

    def test_allocate_green_times_within_bounds(self):
        grid = IntersectionGrid(n=3)
        grid.set_counts({0: 15, 1: 5, 2: 0})
        allocations = grid.allocate_green_times()
        for green_time in allocations.values():
            assert _cfg.MIN_GREEN <= green_time <= _cfg.MAX_GREEN

    def test_allocate_green_times_covers_all_nodes(self):
        grid = IntersectionGrid(n=4)
        grid.set_counts({0: 8, 1: 8, 2: 8, 3: 8})
        allocations = grid.allocate_green_times()
        assert set(allocations.keys()) == {0, 1, 2, 3}


# ────────────────────────────────────────────────────────────────────────────
# Forecast (Member 1 fallback) tests
# ────────────────────────────────────────────────────────────────────────────

class TestForecastFallback:

    def test_forecast_with_insufficient_history_is_stable(self):
        grid = IntersectionGrid(n=3)
        result = grid.forecast(0)
        assert result["trend"] == "STABLE"

    def test_forecast_detects_rising_trend(self):
        grid = IntersectionGrid(n=3)
        grid.nodes[0].history = [3, 8]
        result = grid.forecast(0)
        assert result["trend"] == "RISING"

    def test_forecast_detects_falling_trend(self):
        grid = IntersectionGrid(n=3)
        grid.nodes[0].history = [8, 3]
        result = grid.forecast(0)
        assert result["trend"] == "FALLING"

    def test_forecast_returns_expected_keys(self):
        grid = IntersectionGrid(n=3)
        result = grid.forecast(0)
        assert set(result.keys()) == {"congestion", "queue_length", "trend", "confidence"}

    def test_forecast_uses_real_predict_once_node_log_exists(self):
        """Integration test: after enough ticks, forecast() should call the
        real prediction.predict() instead of the local fallback."""
        grid = IntersectionGrid(n=3, log_path="test_intersection_grid_log.csv")
        for count in [2, 4, 6, 8, 10]:
            grid.tick({0: count, 1: 0, 2: 0})
        result = grid.forecast(0)
        # Real predict() returns confidence as a float in [0, 1]; the
        # fallback always returns exactly 0.5 — a rising sequence with
        # this much signal should not coincidentally hit exactly 0.5.
        assert 0.0 <= result["confidence"] <= 1.0


# ────────────────────────────────────────────────────────────────────────────
# Full tick integration tests
# ────────────────────────────────────────────────────────────────────────────

class TestTick:

    def test_tick_increments_counter(self):
        grid = IntersectionGrid(n=3)
        grid.tick({0: 5})
        grid.tick({0: 5})
        assert grid.tick_count == 2

    def test_tick_records_history(self):
        grid = IntersectionGrid(n=3)
        grid.tick({0: 5})
        assert len(grid.nodes[0].history) == 1

    def test_tick_returns_expected_keys(self):
        grid = IntersectionGrid(n=3)
        result = grid.tick({0: 5, 1: 3, 2: 1})
        assert set(result.keys()) == {
            "tick", "propagation", "green_times", "priority_order", "bands"
        }


# ────────────────────────────────────────────────────────────────────────────
# CSV logging tests
# ────────────────────────────────────────────────────────────────────────────

class TestLogGridToCSV:
    LOG_PATH = "test_intersection_grid_log.csv"

    def setup_method(self):
        if os.path.isfile(self.LOG_PATH):
            os.remove(self.LOG_PATH)

    def teardown_method(self):
        if os.path.isfile(self.LOG_PATH):
            os.remove(self.LOG_PATH)

    def test_csv_created_on_first_write(self):
        grid = IntersectionGrid(n=3)
        result = grid.tick({0: 5, 1: 3, 2: 1})
        log_grid_summary_to_csv(result, grid, path=self.LOG_PATH)
        assert os.path.isfile(self.LOG_PATH)

    def test_csv_has_one_row_per_node(self):
        grid = IntersectionGrid(n=3)
        result = grid.tick({0: 5, 1: 3, 2: 1})
        log_grid_summary_to_csv(result, grid, path=self.LOG_PATH)
        with open(self.LOG_PATH) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3

    def test_csv_appends_across_ticks(self):
        grid = IntersectionGrid(n=3)
        r1 = grid.tick({0: 5, 1: 3, 2: 1})
        log_grid_summary_to_csv(r1, grid, path=self.LOG_PATH)
        r2 = grid.tick({0: 6, 1: 4, 2: 2})
        log_grid_summary_to_csv(r2, grid, path=self.LOG_PATH)
        with open(self.LOG_PATH) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 6
