"""Evaluation metrics for the retail sales forecasting project.

All functions take array-likes of equal length and return a float.
WMAE follows the Kaggle Walmart competition: holiday weeks count 5x.
"""

import numpy as np

def _prepare(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)    
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if np.isnan(y_pred).any():
        raise ValueError("Predictions contain NaN values; fill them before scoring.")   
    return y_true, y_pred

def mae(y_true, y_pred):
    """Mean Absolute Error."""
    y_true, y_pred = _prepare(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    y_true, y_pred = _prepare(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def wmae(y_true, y_pred, is_holiday, holiday_weight=5):
    """Weighted MAE: rows flagged as holiday weeks get `holiday_weight`, others 1."""
    y_true, y_pred = _prepare(y_true, y_pred)
    w = np.where(np.asarray(is_holiday) == 1, holiday_weight, 1.0)
    return float(np.sum(w * np.abs(y_true - y_pred)) / np.sum(w)) 

def evaluate(y_true, y_pred, is_holiday):
    """Return all three project metrics as a dict."""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "WMAE": wmae(y_true, y_pred, is_holiday),
    }