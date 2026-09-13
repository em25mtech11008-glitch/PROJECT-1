from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def calculate_ou_half_life(spread: pd.Series) -> float:
    """
    Fits continuous-time Ornstein-Uhlenbeck mean-reversion:
        d(spread_t) = lambda * (mu - spread_{t-1}) * dt + sigma * dW_t
    Discrete regression form:
        delta_spread_t = alpha + lambda_discrete * spread_{t-1} + eps_t
    Half-life = -ln(2) / lambda_discrete.
    Returns np.inf if lambda_discrete >= 0 (no mean-reversion).
    """
    clean_spread = spread.dropna()
    if len(clean_spread) < 20:
        return np.inf

    spread_lag = clean_spread.shift(1).dropna()
    spread_diff = clean_spread.diff().dropna()
    spread_lag = spread_lag.loc[spread_diff.index]

    X = add_constant(spread_lag.values)
    model = OLS(spread_diff.values, X).fit()
    lambda_param = model.params[1]

    if lambda_param >= 0:
        return np.inf

    half_life = -np.log(2) / lambda_param
    return float(half_life)


def filter_by_half_life(
    spread: pd.Series,
    min_half_life: float = 1.0,
    max_half_life: float = 60.0,
) -> bool:
    """
    Checks if spread half-life lies within the tradeable execution window.
    - Too fast (< 1 day): likely high-frequency microstructural noise.
    - Too slow (> 60 days): capital tied up for too long, weak cointegration force.
    """
    hl = calculate_ou_half_life(spread)
    return bool(min_half_life <= hl <= max_half_life)
