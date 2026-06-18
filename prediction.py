"""
prediction.py — Traffic Prediction Module
Member 1: Traffic Prediction Module
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def compute_moving_average(series: pd.Series, window: int) -> pd.Series:
    """
    Computes moving-average smoother for congestion level (vehicle counts).
    If window size is larger than the series, adjusts window size.
    """
    if len(series) == 0:
        return series
    w = max(1, min(window, len(series)))
    return series.rolling(window=w, min_periods=1).mean()

def predict(csv_path: str, n_steps: int = 3, window_size: int = 20, ma_window: int = 5, lane: int = 1) -> tuple:
    """
    Reads density_log.csv and predicts:
      - congestion level (smoothed count at n_steps ahead)
      - queue length (estimated future queue length at n_steps ahead)
      - trend (RISING / STABLE / FALLING)
      - confidence (0.0 to 1.0 based on linear regression R^2)
      
    Returns:
        tuple: (congestion, queue_length, trend, confidence)
    """
    # 1. Check if CSV exists and is not empty
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return (0.0, 0, "STABLE", 1.0)

    try:
        # Load CSV using pandas
        # Since log can grow, only keep the tail for calculations to optimize performance
        df = pd.read_csv(csv_path)
    except Exception:
        return (0.0, 0, "STABLE", 1.0)

    if df.empty:
        return (0.0, 0, "STABLE", 1.0)

    # 2. Determine target columns
    col_name = None
    if f"lane{lane}_count" in df.columns:
        col_name = f"lane{lane}_count"
    elif "vehicle_count" in df.columns:
        col_name = "vehicle_count"
    else:
        # Fallback to any column ending in count
        count_cols = [c for c in df.columns if "count" in c]
        if count_cols:
            col_name = count_cols[0]

    if not col_name or len(df) == 0:
        return (0.0, 0, "STABLE", 1.0)

    # Use the target series
    series = df[col_name].astype(float)
    
    # 3. Fit window
    W = min(window_size, len(series))
    if W < 2:
        val = float(series.iloc[-1])
        q_val = int(round(val * 0.8))
        return (val, q_val, "STABLE", 1.0)

    # Get recent W steps for fitting
    recent_series = series.iloc[-W:].reset_index(drop=True)
    
    # Compute moving average of recent series for smoothed congestion prediction
    # Use ma_window (e.g. 5) to keep it responsive to local trends
    smoothed = compute_moving_average(recent_series, window=ma_window)
    
    # Estimate queue lengths from counts (queue length = round(count * 0.8))
    # This represents discharging queues / waiting counts under standard traffic conditions
    queue_lengths = recent_series.apply(lambda x: round(x * 0.8))

    # 4. Fit Linear Regression for Queue Length Forecasting (Weighted)
    X = np.arange(W).reshape(-1, 1)
    
    # Generate linear weights: oldest data has weight 0.2, newest has weight 1.0
    weights = np.linspace(0.2, 1.0, W)
    
    # Fit regression for queue lengths
    model_q = LinearRegression()
    model_q.fit(X, queue_lengths.values, sample_weight=weights)
    
    future_x = np.array([[W - 1 + n_steps]])
    pred_queue_len = max(0.0, float(model_q.predict(future_x)[0]))
    
    # Calculate confidence based on R^2 of queue length regression (weighted)
    # If the queue has zero variance, the trend is flat and perfectly predictable
    if np.var(queue_lengths.values) == 0:
        confidence = 1.0
    else:
        r2 = model_q.score(X, queue_lengths.values, sample_weight=weights)
        confidence = float(max(0.0, min(1.0, r2)))

    # 5. Fit Linear Regression on raw counts to predict future congestion and trend (Weighted)
    model_c = LinearRegression()
    model_c.fit(X, recent_series.values, sample_weight=weights)
    
    pred_congestion = max(0.0, float(model_c.predict(future_x)[0]))
    
    # Trend classification based on moving average slope
    slope = float(model_c.coef_[0])
    
    # Define thresholds for trend band transitions
    # A slope greater than 0.05 per observation indicates RISING
    # A slope less than -0.05 indicates FALLING, else STABLE
    if slope > 0.05:
        trend = "RISING"
    elif slope < -0.05:
        trend = "FALLING"
    else:
        trend = "STABLE"

    return (
        round(pred_congestion, 2),
        int(round(pred_queue_len)),
        trend,
        round(confidence, 2)
    )
