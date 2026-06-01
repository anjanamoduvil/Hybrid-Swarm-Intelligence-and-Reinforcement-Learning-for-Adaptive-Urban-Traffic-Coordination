"""
test_density.py — Unit Tests for Member 3 (Density Estimation & Alerts)
Run with:  python -m pytest test_density.py -v
"""

import os
import csv
import pytest
import numpy as np

# ── Make sure config and density are importable without installing ───────────
import sys
sys.path.insert(0, os.path.dirname(__file__))

from density import classify_density, should_trigger_alert, compute_density, log_count_to_csv
from config import THRESHOLDS


# ────────────────────────────────────────────────────────────────────────────
# Band classification tests
# ────────────────────────────────────────────────────────────────────────────

class TestClassifyDensity:

    def test_zero_vehicles_is_low(self):
        assert classify_density(0) == "LOW"

    def test_at_low_threshold_is_low(self):
        assert classify_density(THRESHOLDS["low"]) == "LOW"

    def test_just_above_low_is_med(self):
        assert classify_density(THRESHOLDS["low"] + 1) == "MED"

    def test_at_med_threshold_is_med(self):
        assert classify_density(THRESHOLDS["med"]) == "MED"

    def test_just_above_med_is_high(self):
        assert classify_density(THRESHOLDS["med"] + 1) == "HIGH"

    def test_well_above_high_threshold_is_high(self):
        assert classify_density(100) == "HIGH"

    def test_returns_string(self):
        assert isinstance(classify_density(5), str)

    def test_only_valid_bands_returned(self):
        valid = {"LOW", "MED", "HIGH"}
        for count in range(0, 50):
            assert classify_density(count) in valid


# ────────────────────────────────────────────────────────────────────────────
# Alert trigger tests
# ────────────────────────────────────────────────────────────────────────────

class TestAlertTrigger:

    def test_no_alert_below_high(self):
        assert should_trigger_alert(THRESHOLDS["high"]) is False

    def test_no_alert_at_high(self):
        # strictly greater than high threshold triggers alert
        assert should_trigger_alert(THRESHOLDS["high"]) is False

    def test_alert_triggered_above_high(self):
        assert should_trigger_alert(THRESHOLDS["high"] + 1) is True

    def test_alert_triggered_for_large_count(self):
        assert should_trigger_alert(999) is True

    def test_returns_bool(self):
        assert isinstance(should_trigger_alert(5), bool)


# ────────────────────────────────────────────────────────────────────────────
# Density score tests
# ────────────────────────────────────────────────────────────────────────────

class TestComputeDensity:

    def test_zero_count_gives_zero_density(self):
        assert compute_density(0) == pytest.approx(0.0)

    def test_at_high_threshold_gives_one(self):
        assert compute_density(THRESHOLDS["high"]) == pytest.approx(1.0)

    def test_above_high_capped_at_one(self):
        assert compute_density(THRESHOLDS["high"] * 10) == pytest.approx(1.0)

    def test_density_between_zero_and_one(self):
        for count in range(0, 30):
            d = compute_density(count)
            assert 0.0 <= d <= 1.0


# ────────────────────────────────────────────────────────────────────────────
# CSV logging tests
# ────────────────────────────────────────────────────────────────────────────

class TestLogCountToCSV:
    LOG_PATH = "test_density_log.csv"

    def setup_method(self):
        """Remove any leftover test CSV before each test."""
        if os.path.isfile(self.LOG_PATH):
            os.remove(self.LOG_PATH)
        # Override config path for tests
        import config as cfg
        cfg.CSV_LOG_PATH = self.LOG_PATH

    def teardown_method(self):
        if os.path.isfile(self.LOG_PATH):
            os.remove(self.LOG_PATH)

    def test_csv_created_on_first_write(self):
        log_count_to_csv(1, 7, "MED")
        assert os.path.isfile(self.LOG_PATH)

    def test_csv_has_header_row(self):
        log_count_to_csv(1, 7, "MED")
        with open(self.LOG_PATH) as f:
            header = f.readline().strip()
        assert "vehicle_count" in header
        assert "band" in header

    def test_csv_row_values_correct(self):
        log_count_to_csv(42, 15, "HIGH")
        with open(self.LOG_PATH) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["frame"] == "42"
        assert rows[0]["vehicle_count"] == "15"
        assert rows[0]["band"] == "HIGH"

    def test_csv_appends_multiple_rows(self):
        log_count_to_csv(1, 3, "LOW")
        log_count_to_csv(2, 8, "MED")
        log_count_to_csv(3, 25, "HIGH")
        with open(self.LOG_PATH) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3


# ────────────────────────────────────────────────────────────────────────────
# OpenCV overlay smoke tests (no display needed)
# ────────────────────────────────────────────────────────────────────────────

class TestOpenCVOverlays:

    def _blank_frame(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_draw_band_label_returns_frame(self):
        from alerts import draw_band_label
        frame = self._blank_frame()
        result = draw_band_label(frame, 7)
        assert result.shape == (480, 640, 3)

    def test_draw_density_bar_returns_frame(self):
        from alerts import draw_density_bar
        frame = self._blank_frame()
        result = draw_density_bar(frame, 7, 0.35)
        assert result.shape == (480, 640, 3)

    def test_draw_alert_banner_inactive_no_change(self):
        from alerts import draw_alert_banner
        frame = self._blank_frame()
        before = frame.copy()
        result = draw_alert_banner(frame, THRESHOLDS["high"])   # exactly at threshold → no alert
        # Frame should be unchanged (no banner drawn)
        assert np.array_equal(result, before)

    def test_draw_alert_banner_active_modifies_frame(self):
        from alerts import draw_alert_banner
        frame = self._blank_frame()
        result = draw_alert_banner(frame, THRESHOLDS["high"] + 1)
        # Banner modifies the top rows
        assert not np.array_equal(result[:40], np.zeros((40, 640, 3), dtype=np.uint8))

    def test_annotate_frame_all_overlays(self):
        from alerts import annotate_frame
        frame = self._blank_frame()
        result = annotate_frame(frame, 25, 1.0)
        assert result.shape == (480, 640, 3)
