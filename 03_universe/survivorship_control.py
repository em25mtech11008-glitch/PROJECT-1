from __future__ import annotations
import pandas as pd


def get_point_in_time_universe(
    prices: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    min_history_days: int = 125,
) -> list[str]:
    """
    Returns only stocks that were actively trading with sufficient history
    at a specific point-in-time date.
    """
    as_of_ts = pd.to_datetime(as_of_date)
    sub = prices.loc[:as_of_ts]
    if len(sub) < min_history_days:
        return []

    valid = []
    for col in sub.columns:
        valid_history = sub[col].dropna()
        if len(valid_history) >= min_history_days and valid_history.index[-1] == sub.index[-1]:
            valid.append(col)

    return valid


def check_survivorship_bias(
    active_tickers: list[str],
    historical_universe: list[str],
) -> dict[str, float]:
    """
    Measures the turnover / survival fraction of the active list against
    the historical baseline universe.
    """
    set_active = set(active_tickers)
    set_hist = set(historical_universe)
    overlap = len(set_active.intersection(set_hist))
    survival_rate = overlap / len(set_hist) if set_hist else 1.0

    return {
        "survival_rate": survival_rate,
        "active_count": len(set_active),
        "historical_count": len(set_hist),
    }
