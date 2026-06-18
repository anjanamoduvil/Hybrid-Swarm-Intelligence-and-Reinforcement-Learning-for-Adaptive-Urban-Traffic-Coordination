"""
test_prediction.py — Unit Testing Suite for Traffic Prediction Module
Run with: python -m pytest test_prediction.py -v
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure current folder paths take priority lookup for local imports
sys.path.insert(0, os.path.dirname(__file__))

from prediction import compute_moving_average, predict

def test_moving_average_known_values():
    """Test compute_moving_average with known input/output pairs."""
    # Input sequence
    series = pd.Series([2.0, 4.0, 6.0, 8.0])
    
    # window = 2
    # Step 0: [2.0] -> mean = 2.0
    # Step 1: [2.0, 4.0] -> mean = 3.0
    # Step 2: [4.0, 6.0] -> mean = 5.0
    # Step 3: [6.0, 8.0] -> mean = 7.0
    smoothed = compute_moving_average(series, window=2)
    
    expected = [2.0, 3.0, 5.0, 7.0]
    assert list(smoothed) == expected

def test_regression_synthetic_sequences(tmp_path):
    """Test linear regression forecasting on synthetic growing and declining sequences."""
    # 1. Growing sequence (RISING)
    growing_csv = tmp_path / "growing.csv"
    growing_data = pd.DataFrame({
        "lane1_count": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })
    growing_data.to_csv(growing_csv, index=False)
    
    # Predict 3 steps ahead
    congestion, queue_len, trend, confidence = predict(str(growing_csv), n_steps=3, window_size=10, lane=1)
    
    # Last count was 10. Growing sequence should predict > 10
    assert congestion > 10.0
    # Estimated queue length for 10 is 8. For predicted value it should be > 8
    assert queue_len > 8
    assert trend == "RISING"
    # Growing sequence is perfectly linear, so R^2 confidence should be very high (close to 1.0)
    assert confidence > 0.9

    # 2. Declining sequence (FALLING)
    declining_csv = tmp_path / "declining.csv"
    declining_data = pd.DataFrame({
        "lane1_count": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    })
    declining_data.to_csv(declining_csv, index=False)
    
    # Predict 3 steps ahead
    congestion, queue_len, trend, confidence = predict(str(declining_csv), n_steps=3, window_size=10, lane=1)
    
    # Last count was 1. Declining sequence should predict < 1.0 (clamped to >= 0)
    assert congestion < 1.0
    assert queue_len < 1
    assert trend == "FALLING"
    assert confidence > 0.9

def test_trend_flag_transitions(tmp_path):
    """Test trend flag transitions across band changes (flat/stable)."""
    # Flat sequence (STABLE)
    stable_csv = tmp_path / "stable.csv"
    stable_data = pd.DataFrame({
        "lane1_count": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    })
    stable_data.to_csv(stable_csv, index=False)
    
    congestion, queue_len, trend, confidence = predict(str(stable_csv), n_steps=3, window_size=10, lane=1)
    
    assert congestion == 5.0
    assert queue_len == 4  # round(5 * 0.8) = 4
    assert trend == "STABLE"
    # Variance is 0, confidence should be 1.0
    assert confidence == 1.0

def test_robustness_missing_and_empty():
    """Verify predictability robustness on missing files, empty files, or tiny datasets."""
    # 1. Missing file
    congestion, queue_len, trend, confidence = predict("nonexistent_file.csv", n_steps=3)
    assert congestion == 0.0
    assert queue_len == 0
    assert trend == "STABLE"
    assert confidence == 1.0

    # 2. Tiny dataset (1 row)
    # We can test how it handles a single row. Write a temp csv or mock pd.read_csv.
    # We'll write a single row CSV file to check.
    tiny_csv = "tiny_test.csv"
    try:
        df = pd.DataFrame({"vehicle_count": [7]})
        df.to_csv(tiny_csv, index=False)
        congestion, queue_len, trend, confidence = predict(tiny_csv, n_steps=3)
        assert congestion == 7.0
        assert queue_len == 6  # round(7 * 0.8) = 6
        assert trend == "STABLE"
        assert confidence == 1.0
    finally:
        if os.path.exists(tiny_csv):
            os.remove(tiny_csv)
