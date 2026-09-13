from __future__ import annotations
import numpy as np
import pandas as pd


def filter_circuit_breaker_spikes(
    prices: pd.DataFrame,
    max_daily_return_threshold: float = 0.25,
) -> pd.DataFrame:
    """
    NSE equities generally have circuit filter limits (5%, 10%, 20%).
    A daily price jump > 25% often indicates bad tick/split artifact rather
    than a standard trading move. Replaces such spikes with previous day's close.
    """
    clean_df = prices.copy()
    daily_returns = clean_df.pct_change()

    for col in clean_df.columns:
        anomaly_mask = daily_returns[col].abs() > max_daily_return_threshold
        if anomaly_mask.any():
            clean_df.loc[anomaly_mask, col] = np.nan
            clean_df[col] = clean_df[col].ffill()

    return clean_df


def winsorize_returns(
    returns: pd.DataFrame,
    lower_quantile: float = 0.005,
    upper_quantile: float = 0.995,
) -> pd.DataFrame:
    """
    Caps extreme return tails at specified quantiles to stabilize covariance estimation.
    """
    winsorized = returns.copy()
    for col in winsorized.columns:
        q_low = winsorized[col].quantile(lower_quantile)
        q_high = winsorized[col].quantile(upper_quantile)
        winsorized[col] = winsorized[col].clip(lower=q_low, upper=q_high)
    return winsorized
