from __future__ import annotations
import pandas as pd
from .Kalman_filter import run_kalman_filter


def compute_ols_spread(y: pd.Series, x: pd.Series, beta: float, alpha: float = 0.0) -> pd.Series:
    return y - (beta * x + alpha)


def compute_kalman_spread(
    y: pd.Series,
    x: pd.Series,
    delta: float = 1e-4,
    observation_var: float = 1e-3,
) -> pd.Series:
    kf_res = run_kalman_filter(y, x, delta=delta, observation_var=observation_var)
    return kf_res["spread"]
