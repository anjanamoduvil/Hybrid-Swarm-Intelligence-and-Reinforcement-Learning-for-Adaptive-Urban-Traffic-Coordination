import pytest
from evaluator import calculate_congestion_reduction

def test_congestion_reduction_positive():
    """Test reduction calculation when adaptive performance is better (lower waiting time)."""
    # Fixed baseline = 30 seconds, Adaptive = 15 seconds -> should be 50% reduction
    fixed = 30.0
    adaptive = 15.0
    expected = 50.0
    assert calculate_congestion_reduction(fixed, adaptive) == expected

def test_congestion_reduction_no_change():
    """Test calculation when there is no change between baseline and adaptive."""
    fixed = 30.0
    adaptive = 30.0
    expected = 0.0
    assert calculate_congestion_reduction(fixed, adaptive) == expected

def test_congestion_reduction_zero_division():
    """Test that a fixed baseline of 0 safely returns 0.0 instead of crashing."""
    fixed = 0.0
    adaptive = 10.0
    expected = 0.0
    assert calculate_congestion_reduction(fixed, adaptive) == expected